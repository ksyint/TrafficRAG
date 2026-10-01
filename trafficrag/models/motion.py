"""Raw video decoding, VideoMAE motion modeling, and CUDA optimization."""

import torch
import av
import numpy as np
import torch.nn.functional as F
import json
from pathlib import Path
from PIL import Image
from dataclasses import dataclass, asdict
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch.utils.data import Dataset, DataLoader


def execution_device(value='cuda'):
    device = torch.device(value)
    if device.type != 'cuda':
        raise ValueError('TrafficRAG encoders and model runners require cuda or cuda:N.')
    return device


class VideoSource:
    def __init__(self, path):
        self.path = str(Path(path).resolve())
        with av.open(self.path) as container:
            stream = container.streams.video[0]
            self.fps = float(stream.average_rate or stream.base_rate or 0)
            self.origin = float((stream.start_time or 0) * stream.time_base)
            self.duration = (
                float(stream.duration * stream.time_base)
                if stream.duration is not None
                else float(container.duration / av.time_base) if container.duration else 0
            )
        if self.duration <= 0 or self.fps <= 0:
            raise ValueError(f'Video must expose duration and frame-rate metadata: {self.path}')

    def read(self, start=0.0, end=None, frames=30, resize=None):
        end = self.duration if end is None else min(float(end), self.duration)
        start = max(0.0, float(start))
        if not 0 <= start < end or frames < 2:
            raise ValueError('Video crop requires start < end and at least two sampled frames.')
        targets = np.linspace(start, max(start, end - 1 / self.fps), frames)
        images, stamps = [], []
        last = None
        with av.open(self.path) as container:
            container.seek(int((start + self.origin) * av.time_base), backward=True)
            for frame in container.decode(video=0):
                if frame.time is None:
                    raise ValueError('Video frames require presentation timestamps.')
                stamp = float(frame.time) - self.origin
                if stamp < start:
                    continue
                if stamp >= end:
                    break
                array = frame.to_ndarray(format='rgb24')
                if resize:
                    array = np.asarray(
                        Image.fromarray(array).resize((resize, resize), Image.Resampling.BILINEAR)
                    )
                last = (array, stamp)
                while len(images) < frames and stamp >= targets[len(images)] - 1e-8:
                    images.append(array)
                    stamps.append(stamp - start)
                if len(images) == frames:
                    break
        if last is None:
            raise ValueError(f'No frames decoded for crop [{start}, {end}).')
        while len(images) < frames:
            images.append(last[0])
            stamps.append(last[1] - start)
        return np.stack(images), np.asarray(stamps)


def motion_pixels(frames):
    """RGB uint8 T,H,W,C -> ImageNet-normalized float T,C,H,W."""
    pixels = torch.as_tensor(frames.copy()).permute(0, 3, 1, 2).float() / 255.0
    mean = pixels.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
    std = pixels.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
    return (pixels - mean) / std


class TubeletProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv3d(3, 384, kernel_size=(2, 16, 16), stride=(2, 16, 16))

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class VideoAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(384, 1152, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(384))
        self.v_bias = nn.Parameter(torch.zeros(384))
        self.proj = nn.Linear(384, 384)

    def forward(self, x):
        bias = torch.cat((self.q_bias, torch.zeros_like(self.q_bias), self.v_bias))
        qkv = F.linear(x, self.qkv.weight, bias).reshape(x.shape[0], x.shape[1], 3, 6, 64)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        output = F.scaled_dot_product_attention(q, k, v)
        return self.proj(output.transpose(1, 2).reshape(x.shape))


class VideoMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(384, 1536)
        self.fc2 = nn.Linear(1536, 384)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class VideoBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(384, eps=1e-6)
        self.attn = VideoAttention()
        self.norm2 = nn.LayerNorm(384, eps=1e-6)
        self.mlp = VideoMLP()

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


def sinusoidal_positions(length, dim=384):
    angle = torch.arange(length).float()[:, None] / (10000 ** (2 * (torch.arange(dim) // 2) / dim))
    output = torch.empty_like(angle)
    output[:, 0::2], output[:, 1::2] = angle[:, 0::2].sin(), angle[:, 1::2].cos()
    return output[None]


class VideoMAEEncoder(nn.Module):
    @dataclass
    class Config:
        repository: str = 'OpenGVLab/VideoMAE2'
        filename: str = 'distill/vit_s_k710_dl_from_giant.pth'
        revision: str = 'main'
        weights: str = ''
        cache_dir: str = ''
        offline: bool = False
        frames: int = 30
        image_size: int = 350
        gradient_checkpointing: bool = True

        def __post_init__(self):
            if self.frames < 2 or self.frames % 2 or self.image_size < 16:
                raise ValueError('VideoMAE needs an even number of frames and image_size >= 16.')

    def __init__(self, config=None, initialize=True):
        super().__init__()
        self.cfg = config or self.Config()
        self.feature_dim = 384
        self.patch_embed = TubeletProjection()
        self.blocks = nn.ModuleList([VideoBlock() for _ in range(12)])
        self.fc_norm = nn.LayerNorm(384, eps=1e-6)
        count = self.cfg.frames // 2 * (self.cfg.image_size // 16) ** 2
        self.register_buffer('pos_embed', sinusoidal_positions(count), persistent=False)
        if initialize:
            self.load_released_weights()

    def load_released_weights(self):
        if self.cfg.weights:
            path = Path(self.cfg.weights)
            if path.is_dir():
                path = path / self.cfg.filename
        else:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(
                self.cfg.repository,
                self.cfg.filename,
                revision=self.cfg.revision,
                cache_dir=self.cfg.cache_dir or None,
                local_files_only=self.cfg.offline,
            )
        state = torch.load(path, map_location='cpu', weights_only=True)
        state = state.get('model', state.get('module', state.get('state_dict', state)))
        cleaned = {}
        for name, tensor in state.items():
            for prefix in ('module.', 'backbone.', 'encoder.'):
                name = name.removeprefix(prefix)
            if name.startswith('head.') or name == 'pos_embed':
                continue
            cleaned[name] = tensor
        # No arbitrary missing-key fallback: every learned ViT-S parameter must load.
        self.load_state_dict(cleaned, strict=True)

    def forward(self, pixels):
        # Batch,T,C,H,W normalized with the published ImageNet mean/std.
        if pixels.shape[1:] != (self.cfg.frames, 3, self.cfg.image_size, self.cfg.image_size):
            raise ValueError('Video tensor does not match the trained frame count and resolution.')
        tokens = self.patch_embed(pixels.transpose(1, 2))
        tokens = tokens + self.pos_embed.to(tokens.dtype)
        for block in self.blocks:
            tokens = (
                checkpoint(block, tokens, use_reentrant=False)
                if self.training and self.cfg.gradient_checkpointing
                else block(tokens)
            )
        return self.fc_norm(tokens.mean(1))


class BinaryMotionClassifier(nn.Module):
    """One independent model per traffic domain, optionally including an encoder."""

    def __init__(self, feature_dim, encoder=None):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(feature_dim, 1)

    def forward(self, data):
        features = self.encoder(data) if self.encoder is not None else data
        return self.head(features).squeeze(-1)


class AnnotatedSegments(Dataset):
    def __init__(self, path, encoder_config, domain):
        self.root = Path(path).resolve().parent
        self.rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        self.rows = [row for row in self.rows if row.get('domain', domain) == domain]
        self.cfg = encoder_config
        if not self.rows:
            raise ValueError('No annotated segments for the selected violation domain.')
        for row in self.rows:
            if row['label'] not in (0, 1):
                raise ValueError('Each motion segment needs binary label 0 or 1.')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = Path(row['video'])
        video = VideoSource(path if path.is_absolute() else self.root / path)
        frames, _ = video.read(row.get('start', 0), row.get('end'), self.cfg.frames, self.cfg.image_size)
        return motion_pixels(frames), torch.tensor(float(row['label']))


class MotionSystem:
    @dataclass
    class Config:
        seed: int = 42
        epochs: int = 50
        batch_size: int = 1
        lr: float = 1e-6
        weight_decay: float = 0.01

        def __post_init__(self):
            if self.epochs < 1 or self.batch_size < 1 or self.lr <= 0 or self.weight_decay < 0:
                raise ValueError(
                    'Motion training requires positive epochs, batch size, learning rate and nonnegative decay.'
                )

    def __init__(self, config, domain='red-light', encoder=None, device='cuda'):
        self.cfg, self.domain = config, domain
        self.encoder_cfg = encoder or VideoMAEEncoder.Config()
        self.device = execution_device(device)
        self.model = None
        self.optimizer = None

    def configure(self):
        encoder = VideoMAEEncoder(self.encoder_cfg)
        self.feature_dim = encoder.feature_dim
        self.model = BinaryMotionClassifier(self.feature_dim, encoder).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )

    def training_step(self, batch):
        pixels, labels = batch
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = self.model(pixels.to(self.device))
            return F.binary_cross_entropy_with_logits(logits.float(), labels.to(self.device))

    def fit(self, data_path):
        torch.manual_seed(self.cfg.seed)
        dataset = AnnotatedSegments(data_path, self.encoder_cfg, self.domain)
        self.configure()
        loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True)
        history = []
        for epoch in range(self.cfg.epochs):
            self.model.train()
            losses = []
            for batch in loader:
                loss = self.training_step(batch)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                losses.append(loss.item())
            history.append({'epoch': epoch, 'binary_cross_entropy': float(np.mean(losses))})
            print(json.dumps(history[-1]))
        return history

    def save(self, output, history):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                'model': self.model.state_dict(),
                'feature_dim': self.feature_dim,
                'encoder_config': asdict(self.encoder_cfg),
                'domain': self.domain,
                'config': asdict(self.cfg),
            },
            output / 'last.pt',
        )
        (output / 'metrics.json').write_text(json.dumps(history, indent=2) + '\n')

    @classmethod
    def restore(cls, path, device='cuda'):
        saved = torch.load(path, map_location='cpu', weights_only=True)
        cfg = VideoMAEEncoder.Config(**saved['encoder_config'])
        system = cls(cls.Config(**saved['config']), saved['domain'], cfg, device)
        encoder = VideoMAEEncoder(cfg, initialize=False)
        system.model = BinaryMotionClassifier(384, encoder).to(system.device).eval()
        system.model.load_state_dict(saved['model'], strict=True)
        return system

    def score_video(self, video, segments):
        values = []
        with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            for segment in segments:
                frames, _ = video.read(
                    segment.start, segment.end, self.encoder_cfg.frames, self.encoder_cfg.image_size
                )
                logits = self.model(motion_pixels(frames)[None].to(self.device))
                values.append(logits.float().sigmoid().item())
        return values
