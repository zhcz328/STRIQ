"""
Latent Registration Aligner (LRA) — Sec. 2.2

Performs cascaded affine transformations in a hierarchical feature space.
At each resolution level l ∈ {1,2,3} a localisation network L^(l)_loc
predicts an affine matrix θ_l ∈ SE(2) from concatenated source features
of both branches (Eq. 1).  Progressive alignment resolves multi-scale
quality degradation ranging from global probe misalignments to local
tissue distortions.
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from .transformer import AffineTransformer, SUPPORTED_MODES


class LocalisationNetwork(nn.Module):
    """Channel-adaptive localisation sub-network predicting affine params.

    Architecture: AdaptiveAvgPool → FC → ReLU → FC → param_vector.
    The localisation network receives the concatenation [f^{A,s}_l ; f^{B,s}_l]
    along the channel axis and regresses the transformation parameters.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 256,
        n_params: int = 6,
    ):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, n_params),
        )
        # Initialise output bias to identity-like defaults
        self.fc[-1].weight.data.zero_()
        self.fc[-1].bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 2C, H, W] concatenation of source feature pairs.
        Returns:
            params: [B, n_params] raw affine parameters.
        """
        x = self.pool(x).flatten(1)
        return self.fc(x)


class LatentRegistrationAligner(nn.Module):
    """Cascaded multi-level latent-space affine alignment.

    For each hierarchical feature level l the module:
      1. Concatenates source features [f^{A,s}_l, f^{B,s}_l].
      2. Predicts θ_l via the localisation network (Eq. 1).
      3. Warps source features through bilinear grid sampling:
         f^{*,t}_l = G(f^{*}_l, θ_l).
      4. Propagates the warped output as source for the next level:
         f^{*,s}_{l+1} = f^{*,t}_l   (cascaded conditioning).

    The displacement fields {θ_l} are collected for the smoothness
    regulariser L_smooth.
    """

    def __init__(
        self,
        channel_dims: List[int],
        num_levels: int = 3,
        hidden_dim: int = 256,
        transform_mode: str = "affine",
    ):
        super().__init__()
        assert num_levels == len(channel_dims), (
            "channel_dims must match num_levels"
        )
        self.num_levels = num_levels
        self.transformer = AffineTransformer(mode=transform_mode)
        n_params = self.transformer.n_params

        self.loc_nets = nn.ModuleList([
            LocalisationNetwork(
                in_channels=2 * channel_dims[l],
                hidden_dim=hidden_dim,
                n_params=n_params,
            )
            for l in range(num_levels)
        ])

    def forward(
        self,
        feats_a: List[torch.Tensor],
        feats_b: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            feats_a: list of [B, C_l, H_l, W_l] features from branch A.
            feats_b: list of [B, C_l, H_l, W_l] features from branch B.
        Returns:
            aligned_a: warped features of branch A at each level.
            aligned_b: warped features of branch B at each level.
            thetas:    affine matrices at each level (for L_smooth).
        """
        aligned_a: List[torch.Tensor] = []
        aligned_b: List[torch.Tensor] = []
        thetas: List[torch.Tensor] = []

        # Source features initialised as the raw encoder outputs (Sec. 2.2)
        src_a = feats_a[0]
        src_b = feats_b[0]

        for l in range(self.num_levels):
            # Concatenate along channel axis
            concat = torch.cat([src_a, src_b], dim=1)

            # Predict affine parameters via localisation network
            params = self.loc_nets[l](concat)
            theta = self.transformer(params)         # [B, 2, 3]
            thetas.append(theta)

            # Warp both branches with the predicted transformation
            warped_a = AffineTransformer.grid_sample(feats_a[l], theta)
            warped_b = AffineTransformer.grid_sample(feats_b[l], theta)
            aligned_a.append(warped_a)
            aligned_b.append(warped_b)

            # Cascaded conditioning: warped output serves as next source
            if l + 1 < self.num_levels:
                src_a = warped_a
                src_b = warped_b
                # Spatially adapt source to next level resolution if needed
                if src_a.shape[2:] != feats_a[l + 1].shape[2:]:
                    src_a = nn.functional.interpolate(
                        src_a,
                        size=feats_a[l + 1].shape[2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    src_b = nn.functional.interpolate(
                        src_b,
                        size=feats_b[l + 1].shape[2:],
                        mode="bilinear",
                        align_corners=False,
                    )

        return aligned_a, aligned_b, thetas
