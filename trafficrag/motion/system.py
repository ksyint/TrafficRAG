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
from .data import segment_dataset


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
        self.encoder_name, self.device = encoder, execution_device(device)
        self.model = None
        self.optimizer = None

    def configure(self, dataset):
        encoder = VideoMAEEncoder(self.encoder_name) if self.encoder_name else None
        self.feature_dim = encoder.feature_dim if encoder else dataset.tensors[0].shape[-1]
        self.model = BinaryMotionClassifier(self.feature_dim, encoder).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr,
                                          weight_decay=self.cfg.weight_decay)

    def training_step(self, batch):
        features, labels = batch
        return F.binary_cross_entropy_with_logits(self.model(features.to(self.device)), labels.to(self.device))

    def fit(self, data_path=None):
        torch.manual_seed(self.cfg.seed)
        dataset = segment_dataset(data_path, pixels=bool(self.encoder_name))
        self.configure(dataset)
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
        torch.save({'model': self.model.state_dict(), 'feature_dim': self.feature_dim, 'encoder': self.encoder_name,
                    'domain': self.domain, 'config': asdict(self.cfg)}, output / 'last.pt')
        (output / 'metrics.json').write_text(json.dumps(history, indent=2) + '\n')

    @staticmethod
    def score(checkpoint_path, features, device='cuda'):
        device = execution_device(device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        encoder = VideoMAEEncoder(checkpoint['encoder']) if checkpoint['encoder'] else None
        model = BinaryMotionClassifier(checkpoint['feature_dim'], encoder).to(device).eval()
        model.load_state_dict(checkpoint['model'])
        with torch.no_grad():
            return torch.cat([model(chunk.to(device)).sigmoid().cpu() for chunk in features.split(8)]).numpy()
