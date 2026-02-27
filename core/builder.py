"""
STRIQ Network Builder — Sec. 2 (Fig. 2)

Assembles the complete STRIQ pipeline:
  1. Siamese ResNet-18 backbone N_ω (frozen ImageNet weights).
  2. Latent Registration Aligner (LRA) for hierarchical feature alignment.
  3. Orthogonal Knowledge Subspace (OKS) for plane-discriminative subspace
     decomposition and Synergy Expert assembly.

The builder returns two callables:
  - ``encode_pair``: extracts and aligns features for a (I_A, I_B) pair.
  - ``model``: the full nn.Module supporting both training and inference.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.models as tvm

from .registration import LatentRegistrationAligner
from .subspace import OrthogonalKnowledgeSubspace, SynergyExpertAssembly


# -----------------------------------------------------------------------
# Hierarchical feature extractor from ResNet-18 (frozen)
# -----------------------------------------------------------------------

class HierarchicalEncoder(nn.Module):
    """Extract multi-scale features from ResNet-18 {R_i}^3_{i=1}.

    Outputs features at three cascaded residual block levels:
      - f_1: after layer1 (64-ch,  H/4 × W/4)
      - f_2: after layer2 (128-ch, H/8 × W/8)
      - f_3: after layer3 (256-ch, H/16 × W/16)
    """

    CHANNEL_DIMS = [64, 128, 256]

    def __init__(self, pretrained: bool = True, frozen: bool = True):
        super().__init__()
        backbone = tvm.resnet18(
            weights=tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1   # 64-ch
        self.layer2 = backbone.layer2   # 128-ch
        self.layer3 = backbone.layer3   # 256-ch

        if frozen:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        return [f1, f2, f3]


# -----------------------------------------------------------------------
# Complete STRIQ model
# -----------------------------------------------------------------------

class STRIQ(nn.Module):
    """Subspace-Guided Semantic and Topological Invariant Registration
    for Annotation-Free Ultrasound Plane Quality Control.

    Forward pass during training:
        Given (I_A, I_B), extracts hierarchical features via the Siamese
        encoder, aligns them with LRA, and applies OKS for subspace-enhanced
        representations used to compute L_total.

    Inference:
        Computes quality score Q(I_C) via Eq. 6.
    """

    def __init__(
        self,
        num_planes: int = 4,
        pretrained: bool = True,
        frozen_backbone: bool = True,
        # LRA hyperparameters
        num_levels: int = 3,
        transform_mode: str = "affine",
        loc_hidden_dim: int = 256,
        # OKS hyperparameters
        oks_rank: int = 16,
        phi: float = 0.1,
        tau: float = 0.1,
        scale_factor: float = 1.0,
    ):
        super().__init__()

        # Siamese encoder (weight-shared)
        self.encoder = HierarchicalEncoder(
            pretrained=pretrained, frozen=frozen_backbone
        )
        channel_dims = HierarchicalEncoder.CHANNEL_DIMS

        # Latent Registration Aligner
        self.lra = LatentRegistrationAligner(
            channel_dims=channel_dims,
            num_levels=num_levels,
            hidden_dim=loc_hidden_dim,
            transform_mode=transform_mode,
        )

        # Feature dimensionality for OKS (use deepest level)
        feat_d = channel_dims[-1]

        # Orthogonal Knowledge Subspace
        self.oks = OrthogonalKnowledgeSubspace(
            d=feat_d,
            num_planes=num_planes,
            r=oks_rank,
            phi=phi,
            tau=tau,
        )
        self.oks.scale_factor = scale_factor

        # Synergy Expert wrapper
        self.synergy = SynergyExpertAssembly(self.oks)

        # Adaptive pooling for OKS input
        self.gap = nn.AdaptiveAvgPool2d(1)

    def extract_features(
        self, img: torch.Tensor
    ) -> List[torch.Tensor]:
        """Hierarchical feature extraction from a single image."""
        return self.encoder(img)

    def forward_pair(
        self,
        img_a: torch.Tensor,
        img_b: torch.Tensor,
        use_oks: bool = True,
    ) -> Dict[str, object]:
        """Forward pass for a training pair (I_A, I_B).

        Returns a dictionary containing:
          - 'aligned_a', 'aligned_b': warped features at each level.
          - 'thetas': affine matrices at each level.
          - 'feat_a_oks', 'feat_b_oks': OKS-enhanced pooled features.
        """
        feats_a = self.extract_features(img_a)
        feats_b = self.extract_features(img_b)

        aligned_a, aligned_b, thetas = self.lra(feats_a, feats_b)

        result = {
            "aligned_a": aligned_a,
            "aligned_b": aligned_b,
            "thetas": thetas,
            "raw_a": feats_a,
            "raw_b": feats_b,
        }

        if use_oks:
            # Pool deepest aligned features for OKS
            pool_a = self.gap(aligned_a[-1]).flatten(1)   # [B, d]
            pool_b = self.gap(aligned_b[-1]).flatten(1)

            # Frozen backbone output (W_0 x)
            W0_a = self.gap(feats_a[-1]).flatten(1)
            W0_b = self.gap(feats_b[-1]).flatten(1)

            feat_a_oks = self.synergy(pool_a, W0_a)
            feat_b_oks = self.synergy(pool_b, W0_b)
            result["feat_a_oks"] = feat_a_oks
            result["feat_b_oks"] = feat_b_oks

        return result

    def forward(
        self,
        img_a: torch.Tensor,
        img_b: torch.Tensor,
        use_oks: bool = True,
    ) -> Dict[str, object]:
        return self.forward_pair(img_a, img_b, use_oks=use_oks)


def build_striq(cfg: dict) -> STRIQ:
    """Construct STRIQ from a flat configuration dictionary."""
    return STRIQ(
        num_planes=cfg.get("num_planes", 4),
        pretrained=cfg.get("pretrained", True),
        frozen_backbone=cfg.get("frozen_backbone", True),
        num_levels=cfg.get("num_levels", 3),
        transform_mode=cfg.get("transform_mode", "affine"),
        loc_hidden_dim=cfg.get("loc_hidden_dim", 256),
        oks_rank=cfg.get("oks_rank", 16),
        phi=cfg.get("phi", 0.1),
        tau=cfg.get("tau", 0.1),
        scale_factor=cfg.get("scale_factor", 1.0),
    )
