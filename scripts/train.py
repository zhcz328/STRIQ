import argparse
import logging
import os
import sys
import random

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.builder import build_striq
from data.datasets import US4QADataset, US4QAPairDataset, CAMUSDataset, CAMUSPairDataset
from data.loader import build_dataloader, build_joint_dataloader, get_transform
from data.anchor_utils import run_anchor_selection
from data.sam_encoder import build_sam_encoder
from engine.trainer import STRIQTrainer
from engine.evaluator import compute_normalisation_stats
from utils.checkpointer import save_checkpoint

logger = logging.getLogger("striq.train")


def set_seed(seed: int = 42) -> None:
    """Ensure deterministic initialisation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="STRIQ Training")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--dataset", type=str, default="configs/datasets/us4qa_fetal.yaml")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--sam_ckpt", type=str, default=None,
                        help="Path to SAMUS checkpoint for anchor selection")
    parser.add_argument("--output_dir", type=str, default="./output/train")
    args = parser.parse_args()

    # Configuration
    cfg = load_config(args.config)
    data_cfg = load_config(args.dataset)
    set_seed(cfg.get("seed", 42))

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.output_dir, "train.log")),
        ],
    )

    # ---- Anchor Selection ----
    logger.info("Phase 1: Variance-spectrum anchor selection")
    full_dataset = US4QADataset(
        root=data_cfg["dataset"]["root"],
        img_size=cfg["system"]["img_size"],
        augment=False,
    )

    if args.sam_ckpt:
        sam_encoder = build_sam_encoder(
            checkpoint_path=args.sam_ckpt, device=device
        )
    else:
        sam_encoder = build_sam_encoder(checkpoint_path=None, device=device)
        logger.warning("No SAMUS checkpoint provided; using random encoder")

    anchor_dict = run_anchor_selection(
        encoder=sam_encoder,
        dataset=full_dataset,
        k1=cfg["anchor"]["k1"],
        device=device,
        cache_path=os.path.join(args.output_dir, "anchor_embeddings.npy"),
    )

    # ---- Construct Datasets ----
    logger.info("Phase 2: Constructing training and evaluation datasets")
    anchor_files = []
    for plane_anchors in anchor_dict.values():
        anchor_files.extend(plane_anchors)

    anchor_dataset = US4QADataset(
        root=data_cfg["dataset"]["root"],
        file_list=anchor_files,
        img_size=cfg["system"]["img_size"],
        augment=False,
    )

    train_dataset = US4QADataset(
        root=data_cfg["dataset"]["root"],
        split="train",
        img_size=cfg["system"]["img_size"],
        augment=True,
    )

    pair_dataset = US4QAPairDataset(
        anchor_dataset=anchor_dataset,
        train_dataset=train_dataset,
        pairs_per_epoch=len(train_dataset),
    )

    train_loader = build_dataloader(
        pair_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["system"]["num_workers"],
    )

    # ---- Build Model ----
    num_planes = len(data_cfg["dataset"]["planes"])
    model_cfg = {
        "num_planes": num_planes,
        "pretrained": cfg["backbone"]["pretrained"],
        "frozen_backbone": cfg["backbone"]["frozen"],
        "num_levels": cfg["lra"]["num_levels"],
        "transform_mode": cfg["lra"]["transform_mode"],
        "oks_rank": cfg["oks"]["rank"],
        "phi": cfg["oks"]["activation_threshold"],
        "tau": cfg["oks"]["activation_threshold"],
        "scale_factor": cfg["oks"]["scale_factor"],
    }
    model = build_striq(model_cfg)
    logger.info("Model: %d trainable params",
                sum(p.numel() for p in model.parameters() if p.requires_grad))

    # ---- Training ----
    trainer = STRIQTrainer(
        model=model,
        train_loader=train_loader,
        lr=cfg["training"]["lr"],
        epochs=cfg["training"]["epochs"],
        lambda_orth=cfg["oks"]["orthogonality_weight"],
        w_sim=cfg["loss"]["w_sim"],
        w_ncc=cfg["loss"]["w_ncc"],
        w_smooth=cfg["loss"]["w_smooth"],
        device=device,
        checkpoint_dir=os.path.join(args.output_dir, "checkpoints"),
    )

    logger.info("Phase 3: Training STRIQ")
    trainer.train()
    logger.info("Training complete")


if __name__ == "__main__":
    main()
