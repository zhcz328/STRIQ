"""
Quality Score Evaluator
An image is clinically acceptable when Q(I_C) > tau (default tau = 0.5).
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from core.builder import STRIQ
from losses.registration_loss import (
    SemanticAlignmentLoss,
    NormalizedCrossCorrelationLoss,
    SmoothnessRegularizer,
    RegistrationLoss,
)

logger = logging.getLogger(__name__)


class QualityScoreEvaluator:
    """Computes STRIQ quality scores for a set of query images."""

    def __init__(
        self,
        model: STRIQ,
        anchor_images: torch.Tensor,
        anchor_plane_ids: List[int],
        w_sim: float = 1.0,
        w_ncc: float = 1.0,
        w_smooth: float = 1.0,
        acceptance_threshold: float = 0.5,
        device: str = "cuda",
        norm_stats: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        """
        Args:
            model: trained STRIQ network.
            anchor_images: [k_1, 3, H, W] pre-loaded reference images.
            anchor_plane_ids: plane id for each anchor.
            norm_stats: per-loss min/max for phi normalisation.  If None,
                        raw scores are returned without normalisation.
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.anchor_images = anchor_images.to(device)
        self.anchor_plane_ids = anchor_plane_ids
        self.weights = {"sim": w_sim, "ncc": w_ncc, "smooth": w_smooth}
        self.threshold = acceptance_threshold
        self.norm_stats = norm_stats or {}

        # Instantiate per-term loss modules
        self._sim_fn = SemanticAlignmentLoss()
        self._ncc_fn = NormalizedCrossCorrelationLoss()
        self._smooth_fn = SmoothnessRegularizer()

        # Pre-extract and cache anchor features (Suppl. A.1: ~5.6 ms/frame)
        with torch.no_grad():
            self._anchor_feats = [
                self.model.extract_features(self.anchor_images[i : i + 1])
                for i in range(self.anchor_images.size(0))
            ]

    @torch.no_grad()
    def score_single(self, query_img: torch.Tensor) -> Dict[str, float]:
        """Compute Q(I_C) for a single query image.

        Returns dict with 'quality_score', 'acceptable', and per-loss terms.
        """
        query_img = query_img.to(self.device)
        if query_img.dim() == 3:
            query_img = query_img.unsqueeze(0)

        k1 = self.anchor_images.size(0)
        loss_accum = {"sim": 0.0, "ncc": 0.0, "smooth": 0.0}

        query_feats = self.model.extract_features(query_img)

        for j in range(k1):
            anchor_feats = self._anchor_feats[j]

            # Cascaded LRA alignment
            aligned_q, aligned_a, thetas = self.model.lra(
                query_feats, anchor_feats
            )

            # Per-term loss evaluation
            l_sim = self._sim_fn(aligned_q, aligned_a).item()
            l_ncc = self._ncc_fn(aligned_q, aligned_a).item()
            l_smooth = self._smooth_fn(thetas).item()

            loss_accum["sim"] += l_sim
            loss_accum["ncc"] += l_ncc
            loss_accum["smooth"] += l_smooth

        # Average over k_1 anchors
        for key in loss_accum:
            loss_accum[key] /= k1

        # Min-max normalisation phi(.) using pre-computed D_B statistics
        normalised = {}
        for key in loss_accum:
            if key in self.norm_stats:
                lo, hi = self.norm_stats[key]
                normalised[key] = (loss_accum[key] - lo) / max(hi - lo, 1e-8)
            else:
                normalised[key] = loss_accum[key]

        # Weighted combination (Eq. 6)
        weighted_sum = sum(
            self.weights[m] * normalised[m] for m in normalised
        )
        quality_score = 1.0 - weighted_sum

        return {
            "quality_score": float(np.clip(quality_score, 0.0, 1.0)),
            "acceptable": quality_score > self.threshold,
            "loss_sim": loss_accum["sim"],
            "loss_ncc": loss_accum["ncc"],
            "loss_smooth": loss_accum["smooth"],
        }

    @torch.no_grad()
    def score_batch(self, dataloader: DataLoader) -> Dict[str, object]:
        """Score all images in a DataLoader.

        Returns:
            Dictionary with arrays: 'predicted', 'ground_truth', 'filenames'.
        """
        predicted, ground_truth, filenames = [], [], []

        for batch in dataloader:
            for i in range(batch["image"].size(0)):
                result = self.score_single(batch["image"][i])
                predicted.append(result["quality_score"])
                ground_truth.append(batch["quality_score"][i].item())
                filenames.append(batch["filename"][i])

        return {
            "predicted": np.array(predicted),
            "ground_truth": np.array(ground_truth),
            "filenames": filenames,
        }


def compute_quality_scores(
    model: STRIQ,
    dataloader: DataLoader,
    anchor_images: Optional[torch.Tensor] = None,
    anchor_plane_ids: Optional[List[int]] = None,
    device: str = "cuda",
) -> Dict[str, np.ndarray]:
    """Convenience wrapper for batch quality score computation."""
    if anchor_images is None:
        batch = next(iter(dataloader))
        anchor_images = batch["image"][:20]
        anchor_plane_ids = batch.get("plane_id", [0] * 20)
        if isinstance(anchor_plane_ids, torch.Tensor):
            anchor_plane_ids = anchor_plane_ids.tolist()

    evaluator = QualityScoreEvaluator(
        model=model,
        anchor_images=anchor_images,
        anchor_plane_ids=anchor_plane_ids,
        device=device,
    )
    return evaluator.score_batch(dataloader)


def compute_normalisation_stats(
    model: STRIQ,
    train_loader: DataLoader,
    anchor_images: torch.Tensor,
    device: str = "cuda",
) -> Dict[str, Tuple[float, float]]:
    """Compute per-plane min-max statistics over D_B for phi normalisation.

    These are fixed at inference to prevent distribution shift (Suppl. A.1).
    """
    model.eval()
    accum = {"sim": [], "ncc": [], "smooth": []}

    evaluator = QualityScoreEvaluator(
        model=model,
        anchor_images=anchor_images,
        anchor_plane_ids=[0] * anchor_images.size(0),
        device=device,
    )

    for batch in train_loader:
        for i in range(batch["image"].size(0)):
            result = evaluator.score_single(batch["image"][i])
            accum["sim"].append(result["loss_sim"])
            accum["ncc"].append(result["loss_ncc"])
            accum["smooth"].append(result["loss_smooth"])

    norm = {}
    for key in accum:
        arr = np.array(accum[key])
        norm[key] = (float(arr.min()), float(arr.max()))
        logger.info(
            "Norm stats [%s]: min=%.4f, max=%.4f", key, norm[key][0], norm[key][1]
        )

    return norm
