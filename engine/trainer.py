"""
STRIQ Training Engine — Sec. 2.4, Sec. 3.1

Implements the two-phase training protocol:
  Phase 1: Sequential plane-specific expert training with task-vector
            recording and conflict mask computation.
  Phase 2: General expert fine-tuning with orthogonal gradient projection
            to suppress negative knowledge transfer.

Optimiser: Adam (lr=1e-4, 500 epochs, batch_size=64).
Total loss: L_total = L_reg + λ L_orth  (Sec. 2.4).
"""

import logging
import time
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from core.builder import STRIQ
from losses import RegistrationLoss, OrthogonalityLoss
from .lr_scheduler import build_scheduler

logger = logging.getLogger(__name__)


class STRIQTrainer:
    """Orchestrates STRIQ training across anatomical planes."""

    def __init__(
        self,
        model: STRIQ,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        lr: float = 1e-4,
        epochs: int = 500,
        lambda_orth: float = 0.5,
        w_sim: float = 1.0,
        w_ncc: float = 1.0,
        w_smooth: float = 1.0,
        device: str = "cuda",
        checkpoint_dir: str = "./checkpoints",
        scheduler_cfg: Optional[dict] = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.epochs = epochs
        self.lambda_orth = lambda_orth
        self.checkpoint_dir = checkpoint_dir

        # Collect trainable parameters (LRA localisers + OKS experts)
        trainable = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = Adam(trainable, lr=lr)
        self.scheduler = build_scheduler(self.optimizer, scheduler_cfg)

        # Loss functions
        self.reg_loss = RegistrationLoss(w_sim=w_sim, w_ncc=w_ncc, w_smooth=w_smooth)
        self.orth_loss = OrthogonalityLoss()

        self.train_loader = train_loader
        self.val_loader = val_loader

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """Execute a single training epoch."""
        self.model.train()
        meter = {k: 0.0 for k in
                 ["loss_total", "loss_reg", "loss_sim", "loss_ncc",
                  "loss_smooth", "loss_orth"]}
        n_batches = 0

        for batch in self.train_loader:
            img_a = batch["img_a"].to(self.device)
            img_b = batch["img_b"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            out = self.model(img_a, img_b, use_oks=True)

            # Registration loss
            reg_dict = self.reg_loss(
                out["aligned_a"], out["aligned_b"], out["thetas"]
            )
            # Orthogonality loss
            l_orth = self.orth_loss(self.model.oks)

            # Total objective (Sec. 2.4)
            loss_total = reg_dict["loss_reg"] + self.lambda_orth * l_orth

            loss_total.backward()
            self.optimizer.step()

            # Accumulate metrics
            meter["loss_total"] += loss_total.item()
            meter["loss_reg"] += reg_dict["loss_reg"].item()
            meter["loss_sim"] += reg_dict["loss_sim"].item()
            meter["loss_ncc"] += reg_dict["loss_ncc"].item()
            meter["loss_smooth"] += reg_dict["loss_smooth"].item()
            meter["loss_orth"] += l_orth.item()
            n_batches += 1

        for k in meter:
            meter[k] /= max(n_batches, 1)

        if self.scheduler is not None:
            self.scheduler.step()

        return meter

    def train(self) -> None:
        """Full training loop across all epochs."""
        logger.info("Starting STRIQ training for %d epochs", self.epochs)
        best_metric = -float("inf")

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            metrics = self.train_one_epoch(epoch)
            elapsed = time.time() - t0

            logger.info(
                "[Epoch %03d/%03d] loss=%.4f (sim=%.4f ncc=%.4f smooth=%.4f "
                "orth=%.4f) | %.1fs",
                epoch, self.epochs, metrics["loss_total"],
                metrics["loss_sim"], metrics["loss_ncc"],
                metrics["loss_smooth"], metrics["loss_orth"], elapsed,
            )

            # Periodic validation
            if self.val_loader is not None and epoch % 10 == 0:
                val_metric = self._validate()
                if val_metric > best_metric:
                    best_metric = val_metric
                    self._save_checkpoint(epoch, tag="best")
                    logger.info("  → New best SRCC: %.4f", val_metric)

        self._save_checkpoint(self.epochs, tag="final")

    @torch.no_grad()
    def _validate(self) -> float:
        """Compute mean quality-score correlation on validation set."""
        from utils.metrics import compute_srcc
        from engine.evaluator import compute_quality_scores

        self.model.eval()
        # Delegate to evaluator logic
        results = compute_quality_scores(
            self.model, self.val_loader, device=self.device
        )
        srcc = compute_srcc(results["predicted"], results["ground_truth"])
        self.model.train()
        return srcc

    def _save_checkpoint(self, epoch: int, tag: str = "latest") -> None:
        """Persist model state and optimiser to disk."""
        import os
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.checkpoint_dir, f"striq_{tag}.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)
        logger.info("Checkpoint saved → %s", path)

    # ------------------------------------------------------------------
    # Sequential plane training with task-vector protocol (Sec. 2.3)
    # ------------------------------------------------------------------

    def sequential_plane_training(
        self,
        plane_loaders: Dict[int, DataLoader],
        epochs_per_plane: int = 100,
    ) -> None:
        """Train each plane expert sequentially, recording task vectors.

        Protocol:
          1. For each plane c:
             a. Snapshot pre-training weights.
             b. Train E_c for epochs_per_plane.
             c. Record task vector T_c.
             d. Compute binary conflict mask M_c (Eq. 4).
          2. Compute union mask M_uni.
          3. Fine-tune general expert E_g with orthogonal projection.
        """
        oks = self.model.oks
        plane_ids = sorted(plane_loaders.keys())

        for c in plane_ids:
            logger.info("=== Sequential training: Plane %d ===", c)
            oks.snapshot_pre_weights(c)

            # Freeze all experts except E_c
            for pid in plane_ids:
                expert = oks.experts[str(pid)]
                for p in expert.parameters():
                    p.requires_grad = (pid == c)

            loader = plane_loaders[c]
            opt_c = Adam(oks.experts[str(c)].parameters(), lr=1e-4)

            for ep in range(1, epochs_per_plane + 1):
                for batch in loader:
                    img_a = batch["img_a"].to(self.device)
                    img_b = batch["img_b"].to(self.device)
                    opt_c.zero_grad()

                    out = self.model(img_a, img_b, use_oks=True)
                    reg_dict = self.reg_loss(
                        out["aligned_a"], out["aligned_b"], out["thetas"]
                    )
                    reg_dict["loss_reg"].backward()
                    opt_c.step()

            oks.record_task_vector(c)
            oks.compute_conflict_mask(c)
            logger.info("  Task vector and conflict mask recorded for plane %d", c)

        # Union mask and general expert fine-tuning
        oks.compute_union_mask()
        logger.info("Union mask computed; starting general expert fine-tuning")

        # Unfreeze general expert
        for p in oks.general_expert.parameters():
            p.requires_grad = True

        opt_g = Adam(oks.general_expert.parameters(), lr=1e-4)

        for c in plane_ids:
            loader = plane_loaders[c]
            for ep in range(1, epochs_per_plane // 2 + 1):
                for batch in loader:
                    img_a = batch["img_a"].to(self.device)
                    img_b = batch["img_b"].to(self.device)
                    opt_g.zero_grad()

                    out = self.model(img_a, img_b, use_oks=True)
                    reg_dict = self.reg_loss(
                        out["aligned_a"], out["aligned_b"], out["thetas"]
                    )
                    reg_dict["loss_reg"].backward()

                    # Apply orthogonal gradient projection before step
                    oks.apply_projected_update(c, lr=1e-4)
                    opt_g.step()

        logger.info("Sequential plane training complete")
