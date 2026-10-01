from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from trafficrag.runtime import execution_device
from .classifier import BinaryMotionClassifier
from .encoder import VideoMAEEncoder
from .video_dataset import AnnotatedSegments


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
                raise ValueError('Motion training requires positive epochs, batch size, learning rate and nonnegative decay.')

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
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr,
                                          weight_decay=self.cfg.weight_decay)

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
        torch.save({'model': self.model.state_dict(), 'feature_dim': self.feature_dim,
                    'encoder_config': asdict(self.encoder_cfg), 'domain': self.domain,
                    'config': asdict(self.cfg)}, output / 'last.pt')
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
        from trafficrag.media import motion_pixels
        values = []
        with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            for segment in segments:
                frames, _ = video.read(segment.start, segment.end, self.encoder_cfg.frames, self.encoder_cfg.image_size)
                logits = self.model(motion_pixels(frames)[None].to(self.device))
                values.append(logits.float().sigmoid().item())
        return values
