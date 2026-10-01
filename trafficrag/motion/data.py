import numpy as np
import torch
from torch.utils.data import TensorDataset


def segment_dataset(path=None, pixels=False):
    if path:
        with np.load(path, allow_pickle=False) as data:
            features = torch.from_numpy(data['pixels' if pixels else 'features']).float()
            labels = torch.from_numpy(data['labels']).float()
    else:
        if pixels:
            raise ValueError('A pretrained encoder requires real pixel data.')
        features = torch.randn(128, 16)
        labels = (features[:, 0] + 0.5 * features[:, 1] > 0).float()
    if labels.shape != (len(features),) or not torch.isin(labels, torch.tensor([0.0, 1.0])).all():
        raise ValueError('One binary label is required per training segment.')
    return TensorDataset(features, labels)
