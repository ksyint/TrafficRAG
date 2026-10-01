import torch


def execution_device(value='cuda'):
    device = torch.device(value)
    if device.type != 'cuda':
        raise ValueError('TrafficRAG encoders and model runners require cuda or cuda:N.')
    return device
