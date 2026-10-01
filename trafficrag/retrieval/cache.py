"""Content-addressed ModernBERT embeddings for repeated memory construction."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np


class TextEmbeddingCache:
    def __init__(self, directory, encoder_identity):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.identity = dict(encoder_identity)
        self.hits = 0
        self.misses = 0

    def key(self, caption):
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError("Embedding captions must be nonempty text")
        payload = json.dumps(
            {"encoder": self.identity, "caption": caption}, sort_keys=True, ensure_ascii=False
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def path(self, caption):
        key = self.key(caption)
        return self.directory / key[:2] / f"{key}.npz"

    def read(self, caption):
        path = self.path(caption)
        if not path.exists():
            self.misses += 1
            return None
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            value = archive["embedding"]
        if metadata != {"encoder": self.identity, "caption": caption}:
            raise ValueError("Cached text embedding differs from its encoder identity")
        if value.ndim != 1 or not np.isfinite(value).all() or np.linalg.norm(value) == 0:
            raise ValueError("Cached embedding must be a finite nonzero vector")
        self.hits += 1
        return value

    def write(self, caption, embedding):
        values = np.asarray(embedding, dtype=np.float32)
        if values.ndim != 1 or not np.isfinite(values).all() or np.linalg.norm(values) == 0:
            raise ValueError("Cannot cache a zero or nonfinite text embedding")
        path = self.path(caption)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                np.savez(
                    stream,
                    embedding=values,
                    metadata=np.asarray(
                        json.dumps(
                            {"encoder": self.identity, "caption": caption},
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                    ),
                )
                stream.flush()
                os.fsync(stream.fileno())
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        temporary.replace(path)

    def encode(self, captions, encoder, batch_size=16):
        if batch_size < 1 or not captions:
            raise ValueError("Text encoding requires captions and a positive batch size")
        unique = list(dict.fromkeys(captions))
        values = {caption: self.read(caption) for caption in unique}
        missing = [caption for caption in unique if values[caption] is None]
        for start in range(0, len(missing), batch_size):
            selected = missing[start : start + batch_size]
            encoded = np.asarray(encoder(selected), dtype=np.float32)
            if encoded.ndim != 2 or len(encoded) != len(selected):
                raise ValueError("Text encoder returned an invalid batch shape")
            for caption, embedding in zip(selected, encoded):
                self.write(caption, embedding)
                values[caption] = embedding
        dimensions = {len(value) for value in values.values()}
        if len(dimensions) != 1:
            raise ValueError("Cached embeddings use different feature dimensions")
        return np.stack([values[caption] for caption in captions])

    def summary(self):
        return {
            "encoder": self.identity,
            "hits": self.hits,
            "misses": self.misses,
            "directory": str(self.directory.resolve()),
        }
