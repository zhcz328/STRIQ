"""
Orthogonality Loss L_orth
"""

import torch
import torch.nn as nn

from core.subspace.oks_layer import OrthogonalKnowledgeSubspace


class OrthogonalityLoss(nn.Module):
    """Pairwise inter-plane orthogonality penalty."""

    def forward(self, oks: OrthogonalKnowledgeSubspace) -> torch.Tensor:
        """Compute L_orth from the current expert weight matrices.

        Args:
            oks: the OKS module containing plane-specific experts.
        Returns:
            Scalar orthogonality loss.
        """
        plane_ids = sorted(int(k) for k in oks.experts.keys())
        n = len(plane_ids)
        if n < 2:
            return torch.tensor(0.0, device=next(oks.parameters()).device)

        loss = torch.tensor(0.0, device=next(oks.parameters()).device)
        count = 0

        for i in range(n):
            A_i = oks.experts[str(plane_ids[i])].A   # [r, d]
            for j in range(i + 1, n):
                A_j = oks.experts[str(plane_ids[j])].A   # [r, d]
                # Cross-Gram matrix: A_i · A_j^T  → [r, r]
                gram = A_i @ A_j.t()
                # L1 norm penalty (sparser than Frobenius)
                loss = loss + gram.abs().sum()
                count += 1

        # Normalise by number of pairs: |C|(|C|-1)/2 → symmetric pairs
        if count > 0:
            loss = loss / count

        return loss
