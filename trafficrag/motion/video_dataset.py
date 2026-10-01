import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from trafficrag.media import VideoSource, motion_pixels


class AnnotatedSegments(Dataset):
    def __init__(self, path, encoder_config, domain):
        self.root = Path(path).resolve().parent
        self.rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        self.rows = [row for row in self.rows if row.get('domain', domain) == domain]
        self.cfg = encoder_config
        if not self.rows:
            raise ValueError('No annotated segments for the selected violation domain.')
        for row in self.rows:
            if row['label'] not in (0, 1):
                raise ValueError('Each motion segment needs binary label 0 or 1.')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = Path(row['video'])
        video = VideoSource(path if path.is_absolute() else self.root / path)
        frames, _ = video.read(row.get('start', 0), row.get('end'), self.cfg.frames, self.cfg.image_size)
        return motion_pixels(frames), torch.tensor(float(row['label']))
