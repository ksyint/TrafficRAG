import torch
from trafficrag.runtime import execution_device


class ModernBERTEncoder:
    def __init__(self, checkpoint='answerdotai/ModernBERT-base', device='cuda',
                 cache_dir=None, offline=False, revision='main'):
        device = execution_device(device)
        from transformers import AutoModel, AutoTokenizer
        options = dict(cache_dir=cache_dir, local_files_only=offline, revision=revision)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, **options)
        self.model = AutoModel.from_pretrained(checkpoint, attn_implementation='sdpa', **options).to(device).eval().requires_grad_(False)
        self.device = device

    def __call__(self, captions):
        batch = self.tokenizer(captions, padding=True, truncation=True, max_length=8192,
                               return_tensors='pt').to(self.device)
        with torch.inference_mode():
            hidden = self.model(**batch).last_hidden_state
            mask = batch['attention_mask'][..., None]
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        return pooled.float().cpu().numpy()
