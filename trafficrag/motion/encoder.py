"""Released VideoMAE V2 distilled ViT-S/16, retaining checkpoint parameter names."""
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


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
            path = hf_hub_download(self.cfg.repository, self.cfg.filename,
                                   revision=self.cfg.revision, cache_dir=self.cfg.cache_dir or None,
                                   local_files_only=self.cfg.offline)
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
            tokens = checkpoint(block, tokens, use_reentrant=False) if self.training and self.cfg.gradient_checkpointing else block(tokens)
        return self.fc_norm(tokens.mean(1))
