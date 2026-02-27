"""
Affine transformation parameterisation for LRA.
"""

import math
from typing import Dict, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Mode-specific matrix constructors
# ---------------------------------------------------------------------------

def _build_affine(params: torch.Tensor) -> torch.Tensor:
    """Full affine: 6 free parameters → [B,2,3]."""
    return params.view(-1, 2, 3)


def _build_translation(params: torch.Tensor) -> torch.Tensor:
    """Pure translation: tx, ty."""
    B = params.size(0)
    theta = torch.zeros(B, 2, 3, device=params.device, dtype=params.dtype)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    theta[:, 0, 2] = params[:, 0]
    theta[:, 1, 2] = params[:, 1]
    return theta


def _build_rotation(params: torch.Tensor) -> torch.Tensor:
    """Pure rotation by angle omega."""
    omega = params[:, 0]
    cos_w, sin_w = torch.cos(omega), torch.sin(omega)
    B = params.size(0)
    theta = torch.zeros(B, 2, 3, device=params.device, dtype=params.dtype)
    theta[:, 0, 0] = cos_w
    theta[:, 0, 1] = -sin_w
    theta[:, 1, 0] = sin_w
    theta[:, 1, 1] = cos_w
    return theta


def _build_scale(params: torch.Tensor) -> torch.Tensor:
    """Anisotropic scaling: sx, sy."""
    B = params.size(0)
    theta = torch.zeros(B, 2, 3, device=params.device, dtype=params.dtype)
    theta[:, 0, 0] = params[:, 0]
    theta[:, 1, 1] = params[:, 1]
    return theta


def _build_shear(params: torch.Tensor) -> torch.Tensor:
    """Shear: epsilon_x, epsilon_y."""
    B = params.size(0)
    theta = torch.zeros(B, 2, 3, device=params.device, dtype=params.dtype)
    theta[:, 0, 0] = 1.0
    theta[:, 0, 1] = params[:, 0]
    theta[:, 1, 0] = params[:, 1]
    theta[:, 1, 1] = 1.0
    return theta


def _build_rotation_scale(params: torch.Tensor) -> torch.Tensor:
    """Rotation + anisotropic scale: sx, sy, omega."""
    sx, sy, omega = params[:, 0], params[:, 1], params[:, 2]
    cos_w, sin_w = torch.cos(omega), torch.sin(omega)
    B = params.size(0)
    theta = torch.zeros(B, 2, 3, device=params.device, dtype=params.dtype)
    theta[:, 0, 0] = sx * cos_w
    theta[:, 0, 1] = -sin_w
    theta[:, 1, 0] = sin_w
    theta[:, 1, 1] = sy * cos_w
    return theta


def _build_translation_scale(params: torch.Tensor) -> torch.Tensor:
    """Translation + anisotropic scale: sx, sy, tx, ty."""
    B = params.size(0)
    theta = torch.zeros(B, 2, 3, device=params.device, dtype=params.dtype)
    theta[:, 0, 0] = params[:, 0]
    theta[:, 1, 1] = params[:, 1]
    theta[:, 0, 2] = params[:, 2]
    theta[:, 1, 2] = params[:, 3]
    return theta


def _build_rotation_translation(params: torch.Tensor) -> torch.Tensor:
    """Rotation + translation: omega, tx, ty."""
    omega, tx, ty = params[:, 0], params[:, 1], params[:, 2]
    cos_w, sin_w = torch.cos(omega), torch.sin(omega)
    B = params.size(0)
    theta = torch.zeros(B, 2, 3, device=params.device, dtype=params.dtype)
    theta[:, 0, 0] = cos_w
    theta[:, 0, 1] = -sin_w
    theta[:, 0, 2] = tx
    theta[:, 1, 0] = sin_w
    theta[:, 1, 1] = cos_w
    theta[:, 1, 2] = ty
    return theta


# ---------------------------------------------------------------------------
# Registry of supported modes (Supplementary Table 3)
# ---------------------------------------------------------------------------

SUPPORTED_MODES: Dict[str, Dict] = {
    "affine":                {"builder": _build_affine,                "n_params": 6},
    "translation":           {"builder": _build_translation,           "n_params": 2},
    "rotation":              {"builder": _build_rotation,              "n_params": 1},
    "scale":                 {"builder": _build_scale,                 "n_params": 2},
    "shear":                 {"builder": _build_shear,                 "n_params": 2},
    "rotation_scale":        {"builder": _build_rotation_scale,        "n_params": 3},
    "translation_scale":     {"builder": _build_translation_scale,     "n_params": 4},
    "rotation_translation":  {"builder": _build_rotation_translation,  "n_params": 3},
}


class AffineTransformer(nn.Module):
    """Construct a 2×3 affine matrix from a raw parameter vector.
    """

    def __init__(self, mode: str = "affine"):
        super().__init__()
        if mode not in SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported transform mode '{mode}'. "
                f"Choose from {list(SUPPORTED_MODES.keys())}."
            )
        self.mode = mode
        self._builder: Callable = SUPPORTED_MODES[mode]["builder"]
        self.n_params: int = SUPPORTED_MODES[mode]["n_params"]

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """
        Args:
            params: [B, n_params] raw predictions from localization network
        Returns:
            theta: [B, 2, 3] affine matrix
        """
        return self._builder(params)

    @staticmethod
    def grid_sample(features: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """ Bilinear grid sampling
        """
        grid = F.affine_grid(
            theta, features.size(), align_corners=False
        )
        return F.grid_sample(
            features, grid, mode="bilinear",
            padding_mode="border", align_corners=False
        )
