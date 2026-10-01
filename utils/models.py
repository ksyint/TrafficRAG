import numpy as np
import torch
from torch import nn


class BinaryMotionClassifier(nn.Module):
    """One independent model per traffic domain, optionally including an encoder."""
    def __init__(self, feature_dim, encoder=None):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(feature_dim, 1)

    def forward(self, data):
        features = self.encoder(data) if self.encoder is not None else data
        return self.head(features).squeeze(-1)


class VideoMAEEncoder(nn.Module):
    """Adapter for a compatible Hugging Face VideoMAE checkpoint."""
    def __init__(self, checkpoint):
        super().__init__()
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(checkpoint)
        self.feature_dim = self.model.config.hidden_size

    def forward(self, pixels):
        # B,T,C,H,W; normalization and resize must match the chosen checkpoint.
        return self.model(pixel_values=pixels).last_hidden_state.mean(1)


class ModernBERTEncoder:
    def __init__(self, checkpoint, device='cpu'):
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModel.from_pretrained(checkpoint).to(device).eval().requires_grad_(False)
        self.device = device

    def __call__(self, captions):
        batch = self.tokenizer(captions, padding=True, truncation=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            hidden = self.model(**batch).last_hidden_state
            mask = batch['attention_mask'][..., None]
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        return pooled.cpu().numpy()
