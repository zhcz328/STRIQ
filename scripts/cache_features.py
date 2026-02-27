"""
Cache SAMUS Feature Embeddings for Anchor Selection
"""

import argparse
import logging
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.sam_encoder import build_sam_encoder
from data.datasets import US4QADataset
from data.anchor_utils import extract_embeddings

logger = logging.getLogger("striq.cache")


def main():
    parser = argparse.ArgumentParser(description="Cache SAMUS Feature Embeddings")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--sam_ckpt", type=str, required=True)
    parser.add_argument("--output", type=str, default="./cache/embeddings.npy")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    # Build encoder
    encoder = build_sam_encoder(
        checkpoint_path=args.sam_ckpt,
        device=device,
    )

    # Build dataset
    dataset = US4QADataset(
        root=args.data_root,
        img_size=args.img_size,
        augment=False,
    )
    logger.info("Dataset: %d images", len(dataset))

    # Extract embeddings
    embeddings, plane_ids, filenames = extract_embeddings(
        encoder=encoder,
        dataset=dataset,
        batch_size=args.batch_size,
        device=device,
    )

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.save(args.output, {
        "embeddings": embeddings,
        "plane_ids": plane_ids,
        "filenames": filenames,
    })
    logger.info(
        "Cached %d embeddings (dim=%d) → %s",
        embeddings.shape[0], embeddings.shape[1], args.output,
    )


if __name__ == "__main__":
    main()
