import torch

from trafficrag.runtime import execution_device


class ModernBERTEncoder:
    def __init__(self, checkpoint, device='cuda'):
        device = execution_device(device)
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
