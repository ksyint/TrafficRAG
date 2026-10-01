import torch


def motion_pixels(frames):
    """RGB uint8 T,H,W,C -> ImageNet-normalized float T,C,H,W."""
    pixels = torch.as_tensor(frames.copy()).permute(0, 3, 1, 2).float() / 255.0
    mean = pixels.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
    std = pixels.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
    return (pixels - mean) / std
