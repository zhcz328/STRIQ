import logging
import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    save_dir: str,
    tag: str = "latest",
    extra: Optional[dict] = None,
) -> str:
    """Persist training state to disk.

    Args:
        extra: additional metadata (e.g., norm_stats, task_vectors).
    Returns:
        Path to saved checkpoint.
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"striq_{tag}.pth")
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if extra is not None:
        state.update(extra)
    torch.save(state, path)
    logger.info("Checkpoint saved: %s (epoch %d)", path, epoch)
    return path


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cuda",
    strict: bool = True,
) -> Tuple[int, dict]:
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"], strict=strict)
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    epoch = state.get("epoch", 0)
    extra = {k: v for k, v in state.items()
             if k not in ("model_state_dict", "optimizer_state_dict", "epoch")}
    logger.info("Loaded checkpoint from %s (epoch %d)", checkpoint_path, epoch)
    return epoch, extra


def find_latest_checkpoint(save_dir: str) -> Optional[str]:
    """Locate the most recent checkpoint in a directory."""
    if not os.path.isdir(save_dir):
        return None
    candidates = [
        f for f in os.listdir(save_dir)
        if f.startswith("striq_") and f.endswith(".pth")
    ]
    if not candidates:
        return None
    # Prefer 'best', then 'final', then 'latest'
    for tag in ["best", "final", "latest"]:
        name = f"striq_{tag}.pth"
        if name in candidates:
            return os.path.join(save_dir, name)
    return os.path.join(save_dir, sorted(candidates)[-1])
