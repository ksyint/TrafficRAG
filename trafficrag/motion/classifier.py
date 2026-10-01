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
