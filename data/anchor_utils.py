"""
Variance-Spectrum Reference Anchor Selection — Sec. 2.1

For each anatomical plane c ∈ C, selects k_1 reference images from the
full corpus by minimising intra-class feature embedding variance:

  D_A^{(c)} = arg min_{A ⊂ X_c, |A|=k_1}  Σ_{x∈A} σ²_{F_pre}(x)

where σ²_{F_pre}(x) quantifies the embedding variance of x within plane c.
This isolates high-confidence, morphologically consistent anchors that serve
as device- and operator-invariant reference standards.

The complete reference library is  D_A = ∪_{c∈C} D_A^{(c)}.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


@torch.no_grad()
def extract_embeddings(
    encoder: nn.Module,
    dataset: Dataset,
    batch_size: int = 32,
    device: str = "cuda",
) -> Tuple[np.ndarray, List[int], List[str]]:
    """Extract global-average-pooled embeddings from F_pre.

    Args:
        encoder: frozen SAMUS / SAM image encoder (F_pre).
        dataset: image dataset returning dict with 'image' and 'plane_id'.
        batch_size: inference batch size.
        device: compute device.

    Returns:
        embeddings: [N, D] float32 array.
        plane_ids:  per-sample plane identifiers.
        filenames:  per-sample file names for provenance.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    embeddings_list = []
    plane_ids = []
    filenames = []

    gap = nn.AdaptiveAvgPool2d(1)

    for batch in loader:
        imgs = batch["image"].to(device)
        feats = encoder(imgs)                   # [B, C, H, W]
        pooled = gap(feats).flatten(1)           # [B, C]
        embeddings_list.append(pooled.cpu().numpy())
        plane_ids.extend(batch["plane_id"].tolist())
        filenames.extend(batch["filename"])

    return np.concatenate(embeddings_list, axis=0), plane_ids, filenames


def compute_intraclass_variance(
    embeddings: np.ndarray,
    plane_ids: List[int],
) -> Dict[int, np.ndarray]:
    """Compute per-sample embedding variance within each plane.

    For each sample x in plane c, variance is defined as the squared
    L2 distance to the plane centroid:
        σ²(x) = ‖F_pre(x) − μ_c‖²

    Returns:
        Dictionary mapping plane_id → [N_c] variance array.
    """
    unique_planes = sorted(set(plane_ids))
    emb = embeddings
    ids = np.array(plane_ids)

    variance_map: Dict[int, np.ndarray] = {}
    for c in unique_planes:
        mask = ids == c
        plane_emb = emb[mask]
        centroid = plane_emb.mean(axis=0, keepdims=True)
        dists = np.sum((plane_emb - centroid) ** 2, axis=1)
        variance_map[c] = dists

    return variance_map


def select_anchors(
    embeddings: np.ndarray,
    plane_ids: List[int],
    filenames: List[str],
    k1: int = 20,
) -> Dict[int, List[str]]:
    """Select k_1 anchors per plane via variance-spectrum criterion.

    Returns:
        Dictionary mapping plane_id → list of selected file paths (anchors).
    """
    variance_map = compute_intraclass_variance(embeddings, plane_ids)
    ids = np.array(plane_ids)
    fnames = np.array(filenames)

    anchors: Dict[int, List[str]] = {}
    for c, variances in variance_map.items():
        mask = ids == c
        c_fnames = fnames[mask]
        # Sort by ascending variance; take k_1 lowest-variance samples
        order = np.argsort(variances)
        selected = order[:k1]
        anchors[c] = c_fnames[selected].tolist()
        logger.info(
            "Plane %d: selected %d anchors (var range [%.4f, %.4f])",
            c, len(selected), variances[selected[0]], variances[selected[-1]],
        )

    return anchors


def run_anchor_selection(
    encoder: nn.Module,
    dataset: Dataset,
    k1: int = 20,
    batch_size: int = 32,
    device: str = "cuda",
    cache_path: Optional[str] = None,
) -> Dict[int, List[str]]:
    """End-to-end anchor selection pipeline.

    Optionally caches embeddings to disk for reproducibility and speed.
    """
    if cache_path and Path(cache_path).exists():
        logger.info("Loading cached embeddings from %s", cache_path)
        data = np.load(cache_path, allow_pickle=True).item()
        return select_anchors(
            data["embeddings"], data["plane_ids"], data["filenames"], k1
        )

    embeddings, plane_ids, filenames = extract_embeddings(
        encoder, dataset, batch_size=batch_size, device=device
    )

    if cache_path:
        np.save(cache_path, {
            "embeddings": embeddings,
            "plane_ids": plane_ids,
            "filenames": filenames,
        })
        logger.info("Cached embeddings to %s", cache_path)

    return select_anchors(embeddings, plane_ids, filenames, k1)
