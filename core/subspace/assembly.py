"""
Synergy Expert Assembly
"""

from typing import Optional

import torch
import torch.nn as nn

from .oks_layer import OrthogonalKnowledgeSubspace


class SynergyExpertAssembly(nn.Module):
    """Thin wrapper around OKS forward for pipeline integration.

    Given a batch of feature vectors and the frozen backbone output,
    produces the Synergy Expert enhanced representation.
    """

    def __init__(self, oks: OrthogonalKnowledgeSubspace):
        super().__init__()
        self.oks = oks

    def forward(
        self,
        x: torch.Tensor,
        W0_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:         [B, d] pooled feature from backbone.
            W0_output: [B, d] backbone output (W_0 x, frozen).
        Returns:
            [B, d] synergy-enhanced features.
        """
        return self.oks(x, W0_output)

    @property
    def num_planes(self) -> int:
        return self.oks.num_planes

    @property
    def rank(self) -> int:
        return self.oks.r

    def get_expert_param_count(self) -> int:
        """Total trainable parameters in all experts and the general expert."""
        total = 0
        for expert in self.oks.experts.values():
            total += expert.A.numel() + expert.B.numel()
        total += (self.oks.general_expert.A.numel()
                  + self.oks.general_expert.B.numel())
        return total
