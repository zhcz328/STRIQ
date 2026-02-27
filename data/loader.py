"""
Data augmentation pipeline and DataLoader construction.
"""

from typing import Optional

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms as T


def get_transform(
    img_size: int = 256,
    augment: bool = False,
    rotation_range: int = 20,
    translate_range: float = 0.2,
    scale_range: tuple = (0.8, 1.2),
) -> T.Compose:
    """Construct torchvision transform pipeline.

    Args:
        img_size: target spatial resolution.
        augment: whether to apply stochastic augmentations.
        rotation_range: max rotation in degrees (±).
        translate_range: max translation fraction (±).
        scale_range: min/max scaling factors.
    """
    ops = []
    if augment:
        ops.extend([
            T.RandomAffine(
                degrees=rotation_range,
                translate=(translate_range, translate_range),
                scale=scale_range,
            ),
            T.ColorJitter(brightness=0.3, contrast=0.3),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        ])
    ops.extend([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])
    return T.Compose(ops)


class GaussianNoise:
    """Additive Gaussian noise augmentation (Sec. 3.1)."""

    def __init__(self, mean: float = 0.0, std: float = 0.02):
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(tensor) * self.std + self.mean
        return tensor + noise


def build_dataloader(
    dataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 8,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """Standard DataLoader with paper-specified defaults."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def build_joint_dataloader(
    datasets: list,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 8,
) -> DataLoader:
    """Concatenate multiple datasets (e.g., US4QA + CAMUS) for joint training."""
    concat = ConcatDataset(datasets)
    return build_dataloader(
        concat,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
