"""Timestamp-aware PyAV decoding for proposal and VLM crops."""
from pathlib import Path

import av
import numpy as np
from PIL import Image


class VideoSource:
    def __init__(self, path):
        self.path = str(Path(path).resolve())
        with av.open(self.path) as container:
            stream = container.streams.video[0]
            self.fps = float(stream.average_rate or stream.base_rate or 0)
            self.origin = float((stream.start_time or 0) * stream.time_base)
            self.duration = (float(stream.duration * stream.time_base) if stream.duration is not None
                             else float(container.duration / av.time_base) if container.duration else 0)
        if self.duration <= 0 or self.fps <= 0:
            raise ValueError(f'Video must expose duration and frame-rate metadata: {self.path}')

    def read(self, start=0.0, end=None, frames=30, resize=None):
        end = self.duration if end is None else min(float(end), self.duration)
        start = max(0.0, float(start))
        if not 0 <= start < end or frames < 2:
            raise ValueError('Video crop requires start < end and at least two sampled frames.')
        targets = np.linspace(start, max(start, end - 1 / self.fps), frames)
        images, stamps = [], []
        last = None
        with av.open(self.path) as container:
            container.seek(int((start + self.origin) * av.time_base), backward=True)
            for frame in container.decode(video=0):
                if frame.time is None:
                    raise ValueError('Video frames require presentation timestamps.')
                stamp = float(frame.time) - self.origin
                if stamp < start:
                    continue
                if stamp >= end:
                    break
                array = frame.to_ndarray(format='rgb24')
                if resize:
                    array = np.asarray(Image.fromarray(array).resize((resize, resize), Image.Resampling.BILINEAR))
                last = (array, stamp)
                while len(images) < frames and stamp >= targets[len(images)] - 1e-8:
                    images.append(array)
                    stamps.append(stamp - start)
                if len(images) == frames:
                    break
        if last is None:
            raise ValueError(f'No frames decoded for crop [{start}, {end}).')
        while len(images) < frames:
            images.append(last[0])
            stamps.append(last[1] - start)
        return np.stack(images), np.asarray(stamps)
