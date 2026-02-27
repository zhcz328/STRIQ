"""
STRIQ Evaluation Script

Usage:
    python scripts/test.py --config configs/base_config.yaml \
        --dataset configs/datasets/us4qa_fetal.yaml \
        --checkpoint output/train/checkpoints/striq_best.pth \
        --gpu 0
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.builder import build_striq
from data.datasets import US4QADataset, CAMUSDataset
from data.loader import build_dataloader
from engine.evaluator import QualityScoreEvaluator
from utils.metrics import compute_all_metrics
from utils.checkpointer import load_checkpoint
from utils.visualization import plot_score_distribution

logger = logging.getLogger("striq.test")


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="STRIQ Evaluation")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--dataset", type=str, default="configs/datasets/us4qa_fetal.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="./output/eval")
    parser.add_argument("--norm_stats", type=str, default=None,
                        help="Path to JSON file with normalisation statistics")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = load_config(args.dataset)
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # ---- Build Model and Load Checkpoint ----
    num_planes = len(data_cfg["dataset"]["planes"])
    model = build_striq({
        "num_planes": num_planes,
        "pretrained": cfg["backbone"]["pretrained"],
        "frozen_backbone": cfg["backbone"]["frozen"],
        "num_levels": cfg["lra"]["num_levels"],
        "transform_mode": cfg["lra"]["transform_mode"],
        "oks_rank": cfg["oks"]["rank"],
        "phi": cfg["oks"]["activation_threshold"],
        "tau": cfg["oks"]["activation_threshold"],
        "scale_factor": cfg["oks"]["scale_factor"],
    })
    load_checkpoint(model, args.checkpoint, device=device)
    model.eval()

    # ---- Load Normalisation Statistics ----
    norm_stats = None
    if args.norm_stats and os.path.exists(args.norm_stats):
        with open(args.norm_stats, "r") as f:
            norm_stats = json.load(f)
        logger.info("Loaded normalisation stats from %s", args.norm_stats)

    # ---- Prepare Test Dataset (D_C) and Anchor Images (D_A) ----
    dataset_name = data_cfg["dataset"]["name"]
    if dataset_name == "us4qa":
        test_dataset = US4QADataset(
            root=data_cfg["dataset"]["root"],
            split="test",
            img_size=cfg["system"]["img_size"],
            augment=False,
        )
        anchor_dataset = US4QADataset(
            root=data_cfg["dataset"]["root"],
            split="anchor",
            img_size=cfg["system"]["img_size"],
            augment=False,
        )
    elif dataset_name == "camus":
        test_dataset = CAMUSDataset(
            root=data_cfg["dataset"]["root"],
            split="test",
            img_size=cfg["system"]["img_size"],
            augment=False,
        )
        anchor_dataset = CAMUSDataset(
            root=data_cfg["dataset"]["root"],
            split="anchor",
            img_size=cfg["system"]["img_size"],
            augment=False,
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    test_loader = build_dataloader(
        test_dataset, batch_size=1, shuffle=False, num_workers=4
    )

    # Load anchor images into a single tensor
    anchor_loader = build_dataloader(anchor_dataset, batch_size=len(anchor_dataset), shuffle=False)
    anchor_batch = next(iter(anchor_loader))
    anchor_images = anchor_batch["image"]
    anchor_plane_ids = anchor_batch["plane_id"].tolist()

    # ---- Evaluate ----
    logger.info("Evaluating on %d test images with %d anchors",
                len(test_dataset), anchor_images.size(0))

    evaluator = QualityScoreEvaluator(
        model=model,
        anchor_images=anchor_images,
        anchor_plane_ids=anchor_plane_ids,
        w_sim=cfg["loss"]["w_sim"],
        w_ncc=cfg["loss"]["w_ncc"],
        w_smooth=cfg["loss"]["w_smooth"],
        acceptance_threshold=cfg["inference"]["acceptance_threshold"],
        device=device,
        norm_stats=norm_stats,
    )

    t0 = time.time()
    results = evaluator.score_batch(test_loader)
    elapsed = time.time() - t0
    n_images = len(results["predicted"])
    fps = n_images / elapsed if elapsed > 0 else 0

    # ---- Compute Metrics ----
    metrics = compute_all_metrics(
        results["predicted"],
        results["ground_truth"],
        threshold=cfg["inference"]["acceptance_threshold"],
    )

    logger.info("=== Evaluation Results ===")
    logger.info("  SRCC:  %.4f", metrics["srcc"])
    logger.info("  PLCC:  %.4f", metrics["plcc"])
    logger.info("  F1:    %.4f", metrics["f1"])
    logger.info("  Speed: %.1f FPS (%.2f ms/frame)", fps, 1000 / max(fps, 1))

    # ---- Save Results ----
    result_path = os.path.join(args.output_dir, "eval_results.json")
    with open(result_path, "w") as f:
        json.dump({
            "metrics": metrics,
            "n_images": n_images,
            "fps": fps,
            "ms_per_frame": 1000 / max(fps, 1),
        }, f, indent=2)
    logger.info("Results saved to %s", result_path)

    # ---- Visualisation ----
    plot_score_distribution(
        scores={"all": results["predicted"]},
        gt_scores={"all": results["ground_truth"]},
        save_path=os.path.join(args.output_dir, "score_distribution.pdf"),
    )


if __name__ == "__main__":
    main()
