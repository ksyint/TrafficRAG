from torch import nn


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
