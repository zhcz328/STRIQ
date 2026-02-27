"""
US4QA Dataset — Fetal Ultrasound Quality Assessment.

30,757 images across four standard planes: Abdomen, 4CH, Kidney, Face.
324 examinations; ground-truth quality scores averaged from six
sonographers on [0, 1].

Directory structure expected:
    US4QA/
    ├── Abdomen/
    │   ├── exam_001/
    │   │   ├── img_0001.png
    │   │   └── ...
    │   └── ...
    ├── 4CH/
    ├── Kidney/
    ├── Face/
    └── annotations/
        └── quality_scores.csv   # columns: filename, plane, score
"""

import os
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

from ..loader import get_transform

PLANE_NAMES = ["Abdomen", "4CH", "Kidney", "Face"]
PLANE_TO_ID = {name: idx for idx, name in enumerate(PLANE_NAMES)}


class US4QADataset(Dataset):
    """PyTorch Dataset for US4QA fetal ultrasound images.

    Each sample returns (image, plane_id, quality_score, filename).
    When used as reference set D_A, quality_score may be absent (set to -1).
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        file_list: Optional[List[str]] = None,
        img_size: int = 256,
        augment: bool = False,
        score_file: Optional[str] = None,
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.transform = get_transform(img_size=img_size, augment=augment)

        # Load quality score annotations
        self.scores: Dict[str, float] = {}
        if score_file is None:
            score_file = self.root / "annotations" / "quality_scores.csv"
        if Path(score_file).exists():
            with open(score_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.scores[row["filename"]] = float(row["score"])

        # Collect samples
        self.samples: List[Tuple[str, int, float]] = []
        if file_list is not None:
            for fp in file_list:
                plane = self._infer_plane(fp)
                score = self.scores.get(Path(fp).name, -1.0)
                self.samples.append((fp, PLANE_TO_ID[plane], score))
        else:
            for plane in PLANE_NAMES:
                plane_dir = self.root / plane
                if not plane_dir.exists():
                    continue
                for img_path in sorted(plane_dir.rglob("*.png")):
                    score = self.scores.get(img_path.name, -1.0)
                    self.samples.append(
                        (str(img_path), PLANE_TO_ID[plane], score)
                    )

    @staticmethod
    def _infer_plane(filepath: str) -> str:
        """Infer anatomical plane from directory hierarchy."""
        parts = Path(filepath).parts
        for p in parts:
            if p in PLANE_NAMES:
                return p
        return PLANE_NAMES[0]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        filepath, plane_id, score = self.samples[idx]
        img = Image.open(filepath).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return {
            "image": img,
            "plane_id": plane_id,
            "quality_score": torch.tensor(score, dtype=torch.float32),
            "filename": os.path.basename(filepath),
        }


class US4QAPairDataset(Dataset):
    """Generates (I_A, I_B) pairs for registration-based training.

    Pairs reference images from D_A with training images from D_B,
    ensuring plane consistency (same anatomical plane c).
    """

    def __init__(
        self,
        anchor_dataset: US4QADataset,
        train_dataset: US4QADataset,
        pairs_per_epoch: int = 10000,
    ):
        super().__init__()
        self.anchor = anchor_dataset
        self.train = train_dataset
        self.pairs_per_epoch = pairs_per_epoch

        # Index samples by plane for efficient pairing
        self._anchor_by_plane: Dict[int, List[int]] = {}
        for i, (_, pid, _) in enumerate(self.anchor.samples):
            self._anchor_by_plane.setdefault(pid, []).append(i)

        self._train_by_plane: Dict[int, List[int]] = {}
        for i, (_, pid, _) in enumerate(self.train.samples):
            self._train_by_plane.setdefault(pid, []).append(i)

    def __len__(self) -> int:
        return self.pairs_per_epoch

    def __getitem__(self, idx: int) -> Dict[str, object]:
        rng = np.random.RandomState(idx)
        plane_id = rng.choice(list(self._train_by_plane.keys()))

        a_idx = rng.choice(self._anchor_by_plane[plane_id])
        b_idx = rng.choice(self._train_by_plane[plane_id])

        sample_a = self.anchor[a_idx]
        sample_b = self.train[b_idx]

        return {
            "img_a": sample_a["image"],
            "img_b": sample_b["image"],
            "plane_id": plane_id,
            "score_b": sample_b["quality_score"],
        }
