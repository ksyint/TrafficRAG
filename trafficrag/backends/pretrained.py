"""One reusable Qwen3-VL model for KB captions, query captions, and grounding."""
from dataclasses import dataclass
import json
import re

import numpy as np
import torch

from trafficrag.runtime import execution_device
from trafficrag.media import VideoSource
from .modernbert import ModernBERTEncoder


DOMAIN_RULES = {
    'red-light': 'The ego vehicle crosses the stop line while the relevant traffic signal is red.',
    'blind-spot-left': 'A pedestrian or road user approaches on a collision trajectory from the left side.',
    'blind-spot-right': 'A pedestrian or road user approaches on a collision trajectory from the right side.',
}


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
                raise ValueError('VLM sampling needs even frames, image_size >= 32 and a positive output length.')

    def __init__(self, config=None, device='cuda'):
        self.cfg = config or self.Config()
        self.device = execution_device(device)
        self.model = self.processor = self.text_encoder = None

    def configure(self):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        options = dict(cache_dir=self.cfg.cache_dir or None, local_files_only=self.cfg.offline,
                       revision=self.cfg.revision)
        self.processor = AutoProcessor.from_pretrained(self.cfg.vlm, **options)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.cfg.vlm, dtype=torch.bfloat16, attn_implementation='sdpa',
            device_map={'': str(self.device)}, **options).eval().requires_grad_(False)

    def generate(self, video, interval, prompt):
        if self.model is None:
            self.configure()
        frames, seconds = video.read(interval.start, interval.end, self.cfg.frames, self.cfg.image_size)
        messages = [{'role': 'user', 'content': [{'type': 'video', 'video': 'sampled_crop'},
                                               {'type': 'text', 'text': prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Preserve actual crop-relative presentation times through the model's
        # native video timestamp tokens. Millisecond indices avoid frame-rate rounding.
        metadata = {'fps': 1000.0, 'total_num_frames': max(1, int(np.ceil((interval.end - interval.start) * 1000))),
                    'frames_indices': np.round(seconds * 1000).astype(int).tolist()}
        batch = self.processor(text=[text], videos=[frames], video_metadata=[metadata],
                               do_sample_frames=False, return_tensors='pt').to(self.device)
        with torch.inference_mode():
            result = self.model.generate(**batch, max_new_tokens=self.cfg.max_new_tokens, do_sample=False)
        generated = result[:, batch['input_ids'].shape[1]:]
        return self.processor.batch_decode(generated, skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)[0].strip()

    def caption(self, video, interval, domain):
        rule = DOMAIN_RULES[domain]
        prompt = ('Describe the visible driving event in this video. State the ego motion, relevant traffic signals, '
                  'stop-line crossing, neighboring road users, their relative directions and possible collision course. '
                  f'Focus on evidence for this event criterion: {rule} '
                  'Describe the observations without inventing unseen actions. Return a concise English paragraph.')
        return self.generate(video, interval, prompt)

    def embed(self, captions):
        if self.text_encoder is None:
            self.text_encoder = ModernBERTEncoder(self.cfg.text_encoder, self.device,
                                                  cache_dir=self.cfg.cache_dir or None,
                                                  offline=self.cfg.offline, revision=self.cfg.revision)
        return self.text_encoder(captions)

    def ground(self, video, interval, candidate, domain):
        duration = interval.end - interval.start
        prompt = (f'Localize this traffic event: {DOMAIN_RULES[domain]} '
                  f'This crop lasts {duration:.3f} seconds; timestamp zero is its first instant. '
                  f'Candidate observations: {candidate.caption}\n'
                  'Inspect the full crop and return only JSON {"start": number, "end": number} in seconds '
                  'relative to this crop. Start at the first visible violation instant and end when it stops. '
                  f'Require 0 <= start < end <= {duration:.3f}. If the criterion is not met, return {{"violation": false}}.')
        response = self.generate(video, interval, prompt)
        match = re.search(r'\{[^{}]*\}', response)
        if match is None:
            raise ValueError(f'Qwen grounding did not return a JSON object: {response}')
        result = json.loads(match.group())
        if result.get('violation') is False:
            return None
        return float(result['start']), float(result['end'])

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
