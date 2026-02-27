import numpy as np
from scipy import stats


def compute_srcc(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
) -> float:
    if len(predicted) < 3:
        return 0.0
    mask = ground_truth >= 0  # exclude missing annotations
    if mask.sum() < 3:
        return 0.0
    corr, _ = stats.spearmanr(predicted[mask], ground_truth[mask])
    return float(corr) if not np.isnan(corr) else 0.0


def compute_plcc(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
) -> float:
    if len(predicted) < 3:
        return 0.0
    mask = ground_truth >= 0
    if mask.sum() < 3:
        return 0.0
    corr, _ = stats.pearsonr(predicted[mask], ground_truth[mask])
    return float(corr) if not np.isnan(corr) else 0.0


def compute_f1_at_threshold(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    threshold: float = 0.5,
    gt_threshold: float = 0.5,
) -> float:
    """Binary accept/reject F1 score at a given quality threshold.
    """
    pred_labels = (predicted >= threshold).astype(int)
    gt_labels = (ground_truth >= gt_threshold).astype(int)
    mask = ground_truth >= 0
    pred_labels = pred_labels[mask]
    gt_labels = gt_labels[mask]

    tp = ((pred_labels == 1) & (gt_labels == 1)).sum()
    fp = ((pred_labels == 1) & (gt_labels == 0)).sum()
    fn = ((pred_labels == 0) & (gt_labels == 1)).sum()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return float(f1)


def compute_all_metrics(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Compute SRCC, PLCC, and F1 in a single call."""
    return {
        "srcc": compute_srcc(predicted, ground_truth),
        "plcc": compute_plcc(predicted, ground_truth),
        "f1": compute_f1_at_threshold(predicted, ground_truth, threshold),
    }
