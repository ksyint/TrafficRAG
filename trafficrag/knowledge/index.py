import numpy as np


class KnowledgeBase:
    def __init__(self, embeddings, labels, captions=None, ids=None):
        vectors = np.asarray(embeddings, dtype=np.float64)
        raw_labels = np.asarray(labels)
        if not np.isin(raw_labels, [0, 1]).all():
            raise ValueError('Knowledge base labels must be binary integers.')
        self.labels = raw_labels.astype(np.int64)
        if vectors.ndim != 2 or len(vectors) == 0 or not np.isfinite(vectors).all():
            raise ValueError('KB embeddings must be a finite nonempty N,D matrix.')
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0) or self.labels.shape != (len(vectors),) or not np.isin(self.labels, [0, 1]).all():
            raise ValueError('KB embeddings need nonzero norms and one binary label each.')
        self.embeddings = vectors / norms
        self.captions = captions if captions is not None else [''] * len(vectors)
        self.ids = ids if ids is not None else [str(i) for i in range(len(vectors))]
        if len(self.captions) != len(vectors) or len(self.ids) != len(vectors):
            raise ValueError('KB metadata length mismatch.')

    def retrieve(self, embedding, k=10):
        query = np.asarray(embedding, dtype=np.float64)
        if query.shape != (self.embeddings.shape[1],) or not np.isfinite(query).all() or np.linalg.norm(query) == 0:
            raise ValueError('Query embedding shape/value mismatch.')
        if k < 1:
            raise ValueError('K must be positive.')
        similarities = self.embeddings @ (query / np.linalg.norm(query))
        order = np.argsort(-similarities, kind='stable')[:min(k, len(similarities))]
        return order, similarities[order], float(self.labels[order].mean())
