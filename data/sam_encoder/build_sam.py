import logging
from typing import Optional

import torch
import torch.nn as nn

from .modeling.image_encoder import SAMUSImageEncoder

logger = logging.getLogger(__name__)


def build_sam_encoder(
    checkpoint_path: Optional[str] = None,
    encoder_input_size: int = 256,
    device: str = "cuda",
) -> nn.Module:
    """Instantiate and optionally load SAMUS image encoder weights.

    Args:
        checkpoint_path: path to SAMUS .pth checkpoint.  If None, returns
            a randomly initialised encoder (useful for unit tests).
        encoder_input_size: spatial resolution expected by SAMUS (256).
        device: target device identifier.

    Returns:
        Frozen nn.Module producing [B, 256, 16, 16] feature embeddings.
    """
    encoder = SAMUSImageEncoder(
        img_size=encoder_input_size,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        out_chans=256,
    )

    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location="cpu")
        # SAMUS checkpoints store image encoder under 'image_encoder' key
        if "image_encoder" in state:
            state = state["image_encoder"]
        elif "model" in state:
            state = {
                k.replace("image_encoder.", ""): v
                for k, v in state["model"].items()
                if k.startswith("image_encoder.")
            }
        missing, unexpected = encoder.load_state_dict(state, strict=False)
        if missing:
            logger.warning("Missing keys in SAMUS encoder: %s", missing[:5])
        if unexpected:
            logger.warning("Unexpected keys in SAMUS encoder: %s", unexpected[:5])
        logger.info(
            "Loaded SAMUS encoder from %s (%d params)",
            checkpoint_path, sum(p.numel() for p in encoder.parameters()),
        )

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    return encoder.to(device)
