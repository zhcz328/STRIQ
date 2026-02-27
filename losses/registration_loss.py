"""
Registration Consistency Losses
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticAlignmentLoss(nn.Module):
    """L_sim: cosine similarity on unit-hypersphere S^{d-1}.

    Maximises the mean pixel-wise cosine similarity between ℓ₂-normalised
    aligned features across all hierarchical levels.
    """

    def forward(
        self,
        aligned_a: List[torch.Tensor],
        aligned_b: List[torch.Tensor],
    ) -> torch.Tensor:
        loss = torch.tensor(0.0, device=aligned_a[0].device)
        for fa, fb in zip(aligned_a, aligned_b):
            # ℓ₂ normalisation along channel axis
            fa_hat = F.normalize(fa, p=2, dim=1)
            fb_hat = F.normalize(fb, p=2, dim=1)
            # Pixel-wise cosine similarity
            cos_sim = (fa_hat * fb_hat).sum(dim=1)   # [B, H, W]
            loss = loss - cos_sim.mean()
        return loss / len(aligned_a)


class NormalizedCrossCorrelationLoss(nn.Module):
    """L_NCC: normalised cross-correlation in feature space.

    Inherently invariant to affine intensity transformations prevalent
    in clinical ultrasound acquisitions.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def _ncc(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute NCC between two feature maps [B, C, H, W]."""
        # Flatten spatial dimensions
        B, C = x.shape[:2]
        x_flat = x.view(B, C, -1)  # [B, C, N]
        y_flat = y.view(B, C, -1)

        # Mean subtraction
        x_mean = x_flat.mean(dim=-1, keepdim=True)
        y_mean = y_flat.mean(dim=-1, keepdim=True)
        x_c = x_flat - x_mean
        y_c = y_flat - y_mean

        # Covariance and standard deviations
        cov = (x_c * y_c).sum(dim=-1)
        std_x = torch.sqrt((x_c ** 2).sum(dim=-1) + self.eps)
        std_y = torch.sqrt((y_c ** 2).sum(dim=-1) + self.eps)

        ncc = cov / (std_x * std_y)  # [B, C]
        return ncc.mean()

    def forward(
        self,
        aligned_a: List[torch.Tensor],
        aligned_b: List[torch.Tensor],
    ) -> torch.Tensor:
        loss = torch.tensor(0.0, device=aligned_a[0].device)
        for fa, fb in zip(aligned_a, aligned_b):
            loss = loss - self._ncc(fa, fb)
        return loss / len(aligned_a)


class SmoothnessRegularizer(nn.Module):
    """L_smooth: squared Frobenius norm of displacement Jacobian (Sec. 2.2).

    Penalises ∫_Ω |J_{Ω_l}(p)|²_F dp to ensure the displacement field
    resides in the Sobolev space W^{1,2}(Ω), promoting well-conditioned
    affine transitions across cascaded levels.
    """

    def forward(self, thetas: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            thetas: list of [B, 2, 3] affine matrices at each level.
        Returns:
            Scalar smoothness penalty.
        """
        loss = torch.tensor(0.0, device=thetas[0].device)
        identity = torch.eye(2, device=thetas[0].device).unsqueeze(0)  # [1,2,2]

        for theta in thetas:
            # Extract the 2×2 linear part (displacement Jacobian)
            linear = theta[:, :, :2]   # [B, 2, 2]
            # Deviation from identity represents displacement gradient
            displacement = linear - identity
            # Squared Frobenius norm
            frob_sq = (displacement ** 2).sum(dim=(1, 2))  # [B]
            loss = loss + frob_sq.mean()

        return loss / len(thetas)


class RegistrationLoss(nn.Module):
    """Combined registration objective.
    Supports configurable per-term weights for ablation experiments.
    """

    def __init__(
        self,
        w_sim: float = 1.0,
        w_ncc: float = 1.0,
        w_smooth: float = 1.0,
    ):
        super().__init__()
        self.w_sim = w_sim
        self.w_ncc = w_ncc
        self.w_smooth = w_smooth

        self.sim_loss = SemanticAlignmentLoss()
        self.ncc_loss = NormalizedCrossCorrelationLoss()
        self.smooth_loss = SmoothnessRegularizer()

    def forward(
        self,
        aligned_a: List[torch.Tensor],
        aligned_b: List[torch.Tensor],
        thetas: List[torch.Tensor],
    ) -> dict:
        l_sim = self.sim_loss(aligned_a, aligned_b) if self.w_sim > 0 else 0.0
        l_ncc = self.ncc_loss(aligned_a, aligned_b) if self.w_ncc > 0 else 0.0
        l_smooth = self.smooth_loss(thetas) if self.w_smooth > 0 else 0.0

        total = (self.w_sim * l_sim
                 + self.w_ncc * l_ncc
                 + self.w_smooth * l_smooth)

        return {
            "loss_reg": total,
            "loss_sim": l_sim if isinstance(l_sim, torch.Tensor) else torch.tensor(0.0),
            "loss_ncc": l_ncc if isinstance(l_ncc, torch.Tensor) else torch.tensor(0.0),
            "loss_smooth": l_smooth if isinstance(l_smooth, torch.Tensor) else torch.tensor(0.0),
        }
