"""
Visualization Utilities

Provides plotting functions for:
  - Quality score distributions per plane
  - Score response under progressive deformations
  - OKS subspace embeddings via t-SNE
  - Training loss curves
"""

import os
from typing import Dict, List, Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

PLANE_COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
PLANE_LABELS_US4QA = ["Abdomen", "4CH", "Kidney", "Face"]
PLANE_LABELS_CAMUS = ["A2C", "A4C"]


def plot_score_distribution(
    scores: Dict[str, np.ndarray],
    gt_scores: Dict[str, np.ndarray],
    save_path: str = "./figs/score_distribution.pdf",
) -> None:
    """Scatter plot of predicted vs. ground-truth quality scores per plane."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, len(scores), figsize=(4 * len(scores), 4))
    if len(scores) == 1:
        axes = [axes]

    for ax, (plane, pred) in zip(axes, scores.items()):
        gt = gt_scores[plane]
        ax.scatter(gt, pred, alpha=0.5, s=10, c=PLANE_COLORS[0])
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.5)
        ax.set_xlabel("Ground Truth")
        ax.set_ylabel("Predicted Q")
        ax.set_title(plane)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_deformation_response(
    severity_levels: List[str],
    scores: List[float],
    plane_name: str = "4CH",
    save_path: str = "./figs/deformation_response.pdf",
) -> None:
    """Bar chart of quality scores under increasing deformation severity."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3))
    x = np.arange(len(severity_levels))
    bars = ax.bar(x, scores, color=PLANE_COLORS[1], alpha=0.8)

    for bar, s in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{s:.3f}",
            ha="center", fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(severity_levels, fontsize=8)
    ax.set_ylabel("Quality Score Q")
    ax.set_title(f"Deformation Response — {plane_name}")
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_tsne_subspaces(
    embeddings: np.ndarray,
    plane_ids: np.ndarray,
    rank_label: str = "r=16",
    plane_names: Optional[List[str]] = None,
    save_path: str = "./figs/tsne_subspace.pdf",
) -> None:
    """t-SNE visualisation of OKS subspace embeddings."""
    from sklearn.manifold import TSNE

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    proj = tsne.fit_transform(embeddings)

    unique_ids = sorted(set(plane_ids))
    if plane_names is None:
        plane_names = [f"Plane {i}" for i in unique_ids]

    fig, ax = plt.subplots(figsize=(5, 5))
    for i, pid in enumerate(unique_ids):
        mask = plane_ids == pid
        ax.scatter(
            proj[mask, 0], proj[mask, 1],
            c=PLANE_COLORS[i % len(PLANE_COLORS)],
            label=plane_names[i],
            s=8, alpha=0.6,
        )

    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(f"OKS Subspace Embeddings ({rank_label})")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: str = "./figs/training_curves.pdf",
) -> None:
    """Plot training loss curves over epochs."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, values in history.items():
        ax.plot(values, label=key, linewidth=1.0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
