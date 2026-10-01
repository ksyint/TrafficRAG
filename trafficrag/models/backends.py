"""Pretrained video-language stages and recorded evidence replay."""

import importlib
import torch
import numpy as np
import json
from trafficrag.models.motion import execution_device
from dataclasses import dataclass
from pathlib import Path
from trafficrag.models.embeddings import TextEncoder, TextEncoderConfig
from trafficrag.models.prompts import DOMAIN_RULES, caption_prompt, grounding_prompt
from trafficrag.models.responses import parse_grounding
from trafficrag.data.decoding import timestamp_metadata

_BACKENDS = {}


def register_backend(name):
    def register(factory):
        if name in _BACKENDS:
            raise ValueError(f'Backend already registered: {name}')
        _BACKENDS[name] = factory
        return factory

    return register


def create_backend(name, query):
    if ':' in name:
        module, symbol = name.split(':', 1)
        factory = getattr(importlib.import_module(module), symbol)
    else:
        if name not in _BACKENDS:
            raise ValueError(f'Unknown backend {name}; registered: {sorted(_BACKENDS)}')
        factory = _BACKENDS[name]
    backend = factory(query)
    if not all(callable(getattr(backend, method, None)) for method in ('caption', 'embed', 'ground')):
        raise TypeError('A grounding backend must implement caption, embed, and ground.')
    return backend


class ModernBERTEncoder(TextEncoder):
    def __init__(
        self,
        checkpoint='answerdotai/ModernBERT-base',
        device='cuda',
        cache_dir=None,
        offline=False,
        revision='main',
    ):
        config = TextEncoderConfig(checkpoint, revision, cache_dir, offline)
        super().__init__(config, device)


@register_backend("recorded")
class RecordedBackend:
    """Replay externally computed caption/embedding/grounding outputs exactly.

    Interval matching is strict. No nearest-caption substitution is performed.
    """

    def __init__(self, records):
        self.captions = records['captions']
        self.refinements = records['refinements']

    @staticmethod
    def matches(row, interval):
        return abs(row['start'] - interval.start) < 1e-5 and abs(row['end'] - interval.end) < 1e-5

    def caption(self, interval):
        rows = [row for row in self.captions if self.matches(row, interval)]
        if len(rows) != 1:
            raise ValueError(f'Exactly one caption record is required for candidate {interval}.')
        return rows[0]['caption']

    def embed(self, text):
        rows = [row for row in self.captions if row['caption'] == text]
        if not rows:
            raise ValueError('Missing caption embedding.')
        vectors = np.asarray([row['embedding'] for row in rows])
        if not np.allclose(vectors, vectors[:1]):
            raise ValueError('Repeated caption has inconsistent embeddings.')
        return vectors[0]

    def ground(self, crop, candidate):
        rows = [row for row in self.refinements if self.matches(row, crop)]
        if len(rows) != 1:
            raise ValueError(f'Exactly one refinement record is required for crop {crop}.')
        return rows[0]['local_interval']


class VideoLanguageSystem:
    @dataclass
    class Config:
        vlm: str = 'Qwen/Qwen3-VL-8B-Instruct'
        text_encoder: str = 'answerdotai/ModernBERT-base'
        cache_dir: str = ''
        offline: bool = False
        revision: str = 'main'
        frames: int = 32
        image_size: int = 448
        max_new_tokens: int = 256

        def __post_init__(self):
            if self.frames < 2 or self.frames % 2 or self.image_size < 32 or self.max_new_tokens < 1:
                raise ValueError(
                    'VLM sampling needs even frames, image_size >= 32 and a positive output length.'
                )

    def __init__(self, config=None, device='cuda'):
        self.cfg = config or self.Config()
        self.device = execution_device(device)
        self.model = self.processor = self.text_encoder = None

    def configure(self):
        from transformers import AutoProcessor
        from transformers import Qwen3VLForConditionalGeneration

        options = dict(
            cache_dir=self.cfg.cache_dir or None,
            local_files_only=self.cfg.offline,
            revision=self.cfg.revision,
        )
        self.processor = AutoProcessor.from_pretrained(self.cfg.vlm, **options)
        self.model = (
            Qwen3VLForConditionalGeneration.from_pretrained(
                self.cfg.vlm,
                dtype=torch.bfloat16,
                attn_implementation='sdpa',
                device_map={'': str(self.device)},
                **options,
            )
            .eval()
            .requires_grad_(False)
        )

    def generate(self, video, interval, prompt):
        if self.model is None:
            self.configure()
        frames, seconds = video.read(interval.start, interval.end, self.cfg.frames, self.cfg.image_size)
        messages = [
            {
                'role': 'user',
                'content': [
                    {'type': 'video', 'video': 'sampled_crop'},
                    {'type': 'text', 'text': prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Preserve actual crop-relative presentation times through the model's
        # native video timestamp tokens. Millisecond indices avoid frame-rate rounding.
        metadata = timestamp_metadata(seconds, interval.end - interval.start)
        batch = self.processor(
            text=[text],
            videos=[frames],
            video_metadata=[metadata],
            do_sample_frames=False,
            return_tensors='pt',
        ).to(self.device)
        with torch.inference_mode():
            result = self.model.generate(**batch, max_new_tokens=self.cfg.max_new_tokens, do_sample=False)
        generated = result[:, batch['input_ids'].shape[1] :]
        return self.processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

    def caption(self, video, interval, domain):
        return self.generate(video, interval, caption_prompt(domain))

    def embed(self, captions):
        if self.text_encoder is None:
            self.text_encoder = ModernBERTEncoder(
                self.cfg.text_encoder,
                self.device,
                cache_dir=self.cfg.cache_dir or None,
                offline=self.cfg.offline,
                revision=self.cfg.revision,
            )
        return self.text_encoder(captions)

    def ground(self, video, interval, candidate, domain):
        duration = interval.end - interval.start
        prompt = grounding_prompt(domain, duration, candidate.caption, candidate.neighbors)
        response = parse_grounding(self.generate(video, interval, prompt), duration)
        return response.interval

    def for_video(self, video, domain):
        if domain not in DOMAIN_RULES:
            raise ValueError(f'Unknown violation domain: {domain}')
        return VideoBackend(self, video, domain)


class VideoBackend:
    def __init__(self, system, video, domain):
        self.system, self.video, self.domain = system, video, domain

    def caption(self, interval):
        return self.system.caption(self.video, interval, self.domain)

    def embed(self, caption):
        return self.system.embed([caption])[0]

    def ground(self, interval, candidate):
        return self.system.ground(self.video, interval, candidate, self.domain)


def load_query(path):
    return json.loads(Path(path).read_text())
