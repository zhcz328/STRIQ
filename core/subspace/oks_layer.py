"""
Orthogonal Knowledge Subspace
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PlaneExpert(nn.Module):
    """Low-rank expert E_c = B_c A_c for a single anatomical plane.

    Parameters:
        d: feature dimensionality (backbone output channels).
        r: low-rank dimension (default 16 per Table 4).
    """

    def __init__(self, d: int, r: int = 16):
        super().__init__()
        self.d = d
        self.r = r
        # Down-projection A_c ∈ R^{r×d}
        self.A = nn.Parameter(torch.randn(r, d) * 0.01)
        # Up-projection B_c ∈ R^{d×r}
        self.B = nn.Parameter(torch.randn(d, r) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute low-rank residual: B_c A_c x.

        Args:
            x: [B, d] input feature vector.
        Returns:
            [B, d] low-rank projection.
        """
        # z_c = A_c x  →  [B, r]
        z = x @ self.A.t()
        # B_c z_c      →  [B, d]
        return z @ self.B.t()

    def activation(self, x: torch.Tensor) -> torch.Tensor:
        """Low-rank activation z_c = A_c x used for subspace selection."""
        return x @ self.A.t()


class OrthogonalKnowledgeSubspace(nn.Module):
    """Complete OKS module managing plane-specific and general experts.
    Adaptive subspace selection → Synergy Expert assembly.
    """

    def __init__(
        self,
        d: int,
        num_planes: int,
        r: int = 16,
        phi: float = 0.1,
        tau: float = 0.1,
    ):
        super().__init__()
        self.d = d
        self.num_planes = num_planes
        self.r = r
        self.phi = phi            # activation threshold (Sec. 3.1)
        self.tau = tau            # general expert activation threshold

        # Plane-specific experts {E_c}_{c=1}^{|C|}
        self.experts = nn.ModuleDict({
            str(c): PlaneExpert(d, r) for c in range(num_planes)
        })
        # General (shared) expert E_g
        self.general_expert = PlaneExpert(d, r)

        # Frozen backbone weight W_0 is applied externally; alpha here
        self.scale_factor = 1.0

        # Task vector storage (populated during sequential plane training)
        self._task_vectors: Dict[int, Dict[str, torch.Tensor]] = {}
        self._pre_weights: Dict[int, Dict[str, torch.Tensor]] = {}
        self._conflict_masks: Dict[int, Dict[str, torch.Tensor]] = {}
        self._union_mask: Optional[Dict[str, torch.Tensor]] = None

    # ------------------------------------------------------------------
    # Task vector & conflict mask utilities (Eq. 4)
    # ------------------------------------------------------------------

    def snapshot_pre_weights(self, plane_id: int) -> None:
        """Store initial expert weights before training on plane c."""
        expert = self.experts[str(plane_id)]
        self._pre_weights[plane_id] = {
            "A": expert.A.data.clone(),
            "B": expert.B.data.clone(),
        }

    def record_task_vector(self, plane_id: int) -> None:
        """Compute T_c = W^{ft}_c − W^{pre}_c after fine-tuning on plane c."""
        expert = self.experts[str(plane_id)]
        pre = self._pre_weights[plane_id]
        self._task_vectors[plane_id] = {
            "A": expert.A.data - pre["A"],
            "B": expert.B.data - pre["B"],
        }

    def compute_conflict_mask(self, plane_id: int) -> None:
        """Binary mask via quantile thresholding"""
        tv = self._task_vectors[plane_id]
        q_frac = (self.num_planes - 1) / self.num_planes
        mask = {}
        for key in ("A", "B"):
            abs_tv = tv[key].abs()
            threshold = torch.quantile(abs_tv.float(), q_frac)
            mask[key] = (abs_tv > threshold).float()
        self._conflict_masks[plane_id] = mask

    def compute_union_mask(self) -> None:
        """Union mask M_uni = ∪_{c∈C} M_c partitioning conflict-prone regions."""
        union = {"A": torch.zeros_like(self.general_expert.A.data),
                 "B": torch.zeros_like(self.general_expert.B.data)}
        for cid in self._conflict_masks:
            for key in ("A", "B"):
                union[key] = torch.clamp(
                    union[key] + self._conflict_masks[cid][key], max=1.0
                )
        self._union_mask = union

    # ------------------------------------------------------------------
    # Orthogonal gradient projection for shared expert (Sec. 2.3)
    # ------------------------------------------------------------------

    def build_prior_knowledge_space(self, exclude_plane: int) -> torch.Tensor:
        """Stack normalised task vectors of previously seen planes → K.

        K ∈ R^{(|C|-1) × (r·d)} obtained by flattening and row-normalising
        each T_c for c ≠ exclude_plane.
        """
        rows = []
        for cid, tv in self._task_vectors.items():
            if cid == exclude_plane:
                continue
            vec = torch.cat([tv["A"].flatten(), tv["B"].flatten()])
            vec = F.normalize(vec.unsqueeze(0), dim=1)
            rows.append(vec)
        if len(rows) == 0:
            return None
        return torch.cat(rows, dim=0)  # [(|C|-1), r*d]

    def orthogonal_project(
        self, grad: torch.Tensor, K: torch.Tensor
    ) -> torch.Tensor:
        """Project gradient onto orthogonal complement of K.

        g_orth = g - K^T K g   (Sec. 2.3)
        """
        proj = K.t() @ (K @ grad.unsqueeze(-1))  # [r*d, 1]
        return grad - proj.squeeze(-1)

    def apply_projected_update(
        self,
        plane_id: int,
        lr: float,
    ) -> None:
        """Update E_g with selective orthogonal projection.

        For conflict-prone parameters (M_uni=1): use projected gradient.
        For conflict-free parameters (M_uni=0): use raw gradient.
        """
        if self._union_mask is None:
            return
        K = self.build_prior_knowledge_space(exclude_plane=plane_id)

        for key, param in [("A", self.general_expert.A),
                           ("B", self.general_expert.B)]:
            if param.grad is None:
                continue
            g = param.grad.data.clone()
            mask = self._union_mask[key].to(g.device)

            if K is not None:
                g_flat = g.flatten()
                g_orth_flat = self.orthogonal_project(g_flat, K)
                g_orth = g_orth_flat.view_as(g)
            else:
                g_orth = g

            # Selective update: M_uni ⊙ g_orth + (1 − M_uni) ⊙ g
            blended = mask * g_orth + (1.0 - mask) * g
            param.data -= lr * blended

    # ------------------------------------------------------------------
    # Adaptive subspace selection (inference, Sec. 2.3)
    # ------------------------------------------------------------------

    def select_bases(
        self, x: torch.Tensor, expert: PlaneExpert, threshold: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retain basis vectors whose activation exceeds threshold.
        """
        z = expert.activation(x)                    # [B, r]
        z_mean = z.abs().mean(dim=0)                # [r]
        active = z_mean > threshold
        kappa = max(1, self.r // self.num_planes)

        if active.sum() > kappa:
            _, topk_idx = torch.topk(z_mean, kappa)
            mask = torch.zeros_like(z_mean, dtype=torch.bool)
            mask[topk_idx] = True
        else:
            mask = active
            if mask.sum() == 0:
                # Fallback: retain the single strongest basis
                mask[z_mean.argmax()] = True

        A_sel = expert.A[mask]   # [k, d]
        B_sel = expert.B[:, mask]  # [d, k]
        return A_sel, B_sel

    def forward(
        self,
        x: torch.Tensor,
        W0_output: torch.Tensor,
    ) -> torch.Tensor:
        """Synergy Expert inference.
        Args:
            x:         [B, d] input features.
            W0_output: [B, d] frozen backbone output W_0 x.
        Returns:
            [B, d] enhanced feature vector.
        """
        A_parts: List[torch.Tensor] = []
        B_parts: List[torch.Tensor] = []

        # Plane-specific expert selection
        for cid in range(self.num_planes):
            A_sel, B_sel = self.select_bases(
                x, self.experts[str(cid)], self.phi
            )
            A_parts.append(A_sel)
            B_parts.append(B_sel)

        # General expert selection
        A_g, B_g = self.select_bases(x, self.general_expert, self.tau)
        A_parts.append(A_g)
        B_parts.append(B_g)

        # Concatenate along rank axis → Synergy Expert E
        A_E = torch.cat(A_parts, dim=0)   # [K_total, d]
        B_E = torch.cat(B_parts, dim=1)   # [d, K_total]

        # Low-rank residual
        residual = (x @ A_E.t()) @ B_E.t()   # [B, d]
        return W0_output + (self.scale_factor / self.r) * residual
