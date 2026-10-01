"""Chunked exact cosine retrieval on CUDA with stable tie ordering."""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class SearchResult:
    indices: np.ndarray
    similarities: np.ndarray

    def __post_init__(self):
        if self.indices.shape != self.similarities.shape or self.indices.ndim != 2:
            raise ValueError("Retrieval result arrays must be aligned matrices")

    def one(self):
        if len(self.indices) != 1:
            raise ValueError("Single-query retrieval expected exactly one result row")
        return self.indices[0], self.similarities[0]


class CudaCosineIndex:
    def __init__(self, vectors, device="cuda", chunk_size=16384):
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("Semantic retrieval requires a CUDA device")
        values = np.asarray(vectors)
        if values.ndim != 2 or min(values.shape) < 1 or not np.isfinite(values).all():
            raise ValueError("Index vectors must be a finite nonempty matrix")
        if chunk_size < 1:
            raise ValueError("Retrieval chunk size must be positive")
        norms = np.linalg.norm(values, axis=1)
        if np.any(norms == 0):
            raise ValueError("Index vectors must have nonzero norms")
        self.count, self.dimension = values.shape
        self.chunk_size = int(chunk_size)
        self.vectors = np.ascontiguousarray(values / norms[:, None], dtype=np.float32)
        self.resident = None

    def preload(self):
        self.resident = torch.as_tensor(self.vectors, device=self.device)
        return self

    def release(self):
        self.resident = None

    def describe(self):
        return {
            "entries": self.count,
            "dimension": self.dimension,
            "dtype": "float32",
            "similarity": "cosine",
            "device": str(self.device),
            "chunk_size": self.chunk_size,
            "resident": self.resident is not None,
            "host_bytes": self.vectors.nbytes,
        }

    @torch.no_grad()
    def search(self, queries, k=10, exclusions=None):
        values = torch.as_tensor(queries, dtype=torch.float32, device=self.device)
        if values.ndim == 1:
            values = values[None]
        if values.ndim != 2 or values.shape[1] != self.dimension or len(values) < 1:
            raise ValueError("Query embedding width differs from the index")
        if not torch.isfinite(values).all() or (values.norm(dim=1) == 0).any():
            raise ValueError("Query embeddings must be finite with nonzero norms")
        if k < 1:
            raise ValueError("Retrieval neighbor count must be positive")
        queries = torch.nn.functional.normalize(values, dim=-1)
        excluded = [set() for _ in range(len(values))]
        if exclusions is not None:
            if len(exclusions) != len(values):
                raise ValueError("Each query must have one exclusion collection")
            excluded = [set(map(int, indices)) for indices in exclusions]
            if any(
                any(index < 0 or index >= self.count for index in indices) for indices in excluded
            ):
                raise ValueError("Excluded index falls outside the knowledge base")
        available = min(self.count - len(indices) for indices in excluded)
        count = min(k, available)
        if count < 1:
            raise ValueError("No knowledge-base entries remain after exclusions")
        best_scores = torch.empty((len(values), 0), device=self.device)
        best_indices = torch.empty((len(values), 0), device=self.device, dtype=torch.long)
        for start in range(0, self.count, self.chunk_size):
            end = min(start + self.chunk_size, self.count)
            block = (
                self.resident[start:end]
                if self.resident is not None
                else torch.as_tensor(self.vectors[start:end], device=self.device)
            )
            scores = queries @ block.T
            positions = torch.arange(start, end, device=self.device).expand(len(values), -1)
            for row, indices in enumerate(excluded):
                local = [index - start for index in indices if start <= index < end]
                if local:
                    scores[row, local] = -torch.inf
            scores = torch.cat((best_scores, scores), dim=1)
            positions = torch.cat((best_indices, positions), dim=1)
            index_order = torch.argsort(positions, dim=1, stable=True)
            scores = scores.gather(1, index_order)
            positions = positions.gather(1, index_order)
            order = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :count]
            best_scores = scores.gather(1, order)
            best_indices = positions.gather(1, order)
        if not torch.isfinite(best_scores).all():
            raise ValueError("Retrieval could not fill the requested neighbor set")
        return SearchResult(best_indices.cpu().numpy(), best_scores.cpu().numpy())

    def search_batches(self, queries, k=10, batch_size=64, exclusions=None):
        queries = np.asarray(queries)
        if queries.ndim != 2 or batch_size < 1:
            raise ValueError("Batched search requires a query matrix and positive batch size")
        if exclusions is not None:
            if len(exclusions) != len(queries):
                raise ValueError("Each query requires an exclusion collection")
            available = min(self.count - len(set(values)) for values in exclusions)
            k = min(k, available)
            if k < 1:
                raise ValueError("No knowledge-base entries remain for at least one query")
        results = []
        for start in range(0, len(queries), batch_size):
            selected = None if exclusions is None else exclusions[start : start + batch_size]
            results.append(self.search(queries[start : start + batch_size], k, selected))
        if not results:
            raise ValueError("Batched retrieval requires at least one query")
        return SearchResult(
            np.concatenate([result.indices for result in results]),
            np.concatenate([result.similarities for result in results]),
        )
