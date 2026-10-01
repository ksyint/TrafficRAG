import numpy as np

from .index import KnowledgeBase


def load_kb(path):
    with np.load(path, allow_pickle=False) as data:
        return KnowledgeBase(data['embeddings'], data['labels'], data['captions'].tolist(), data['ids'].tolist())
