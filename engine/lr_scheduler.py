"""
Learning rate scheduling utilities.

Supports cosine annealing with warm-up, step decay, and constant baselines.
"""

from typing import Optional

import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    StepLR,
    _LRScheduler,
)


class WarmupCosineScheduler(_LRScheduler):
    """Cosine annealing with linear warm-up phase."""

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int = 10,
        total_epochs: int = 500,
        eta_min: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warm-up
            alpha = self.last_epoch / max(self.warmup_epochs, 1)
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine decay
            import math
            progress = (self.last_epoch - self.warmup_epochs) / max(
                self.total_epochs - self.warmup_epochs, 1
            )
            return [
                self.eta_min + (base_lr - self.eta_min) *
                0.5 * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]


def build_scheduler(
    optimizer: optim.Optimizer,
    cfg: Optional[dict] = None,
) -> Optional[_LRScheduler]:
    """Construct scheduler from configuration dictionary.

    Args:
        cfg: dict with 'type' key.  Supported: 'cosine', 'step', None.
    """
    if cfg is None:
        return None

    stype = cfg.get("type", "cosine")
    if stype == "cosine":
        return WarmupCosineScheduler(
            optimizer,
            warmup_epochs=cfg.get("warmup_epochs", 10),
            total_epochs=cfg.get("total_epochs", 500),
            eta_min=cfg.get("eta_min", 1e-6),
        )
    elif stype == "step":
        return StepLR(
            optimizer,
            step_size=cfg.get("step_size", 100),
            gamma=cfg.get("gamma", 0.5),
        )
    else:
        return None
