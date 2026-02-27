"""
CAMUS Dataset — Cardiac Ultrasound (Public).

Apical 2-chamber (A2C) and 4-chamber (A4C) echocardiographic views.
Ordinal quality grades (Good / Medium / Poor) mapped to ranked sequence
for SRCC and PLCC evaluation.

Reference:
    Leclerc et al., "Deep Learning for Segmentation Using an Open
    Large-Scale Dataset in 2D Echocardiography", IEEE TMI 2019.

Directory structure expected:
    CAMUS/
    ├── patient0001/
    │   ├── patient0001_2CH_ED.mhd
    │   ├── patient0001_4CH_ED.mhd
    │   └── ...
    └── quality_grades.csv   # columns: patient_id, view, grade
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

CAMUS_PLANES = ["A2C", "A4C"]
PLANE_TO_ID = {name: idx for idx, name in enumerate(CAMUS_PLANES)}
GRADE_MAP = {"Good": 1.0, "Medium": 0.5, "Poor": 0.0}


class CAMUSDataset(Dataset):
    """PyTorch Dataset for CAMUS cardiac ultrasound images.

    Handles both .png-converted frames and raw .mhd volumetric data
    (only ED/ES key-frames are loaded when using .mhd).
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        file_list: Optional[List[str]] = None,
        img_size: int = 256,
        augment: bool = False,
        grade_file: Optional[str] = None,
        plane_id_offset: int = 0,
    ):
        super().__init__()
        self.root = Path(root)
        self.img_size = img_size
        self.transform = get_transform(img_size=img_size, augment=augment)
        self.plane_id_offset = plane_id_offset

        # Load ordinal quality grades
        self.grades: Dict[str, float] = {}
        if grade_file is None:
            grade_file = self.root / "quality_grades.csv"
        if Path(grade_file).exists():
            with open(grade_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = f"{row['patient_id']}_{row['view']}"
                    self.grades[key] = GRADE_MAP.get(row["grade"], 0.5)

        # Collect samples
        self.samples: List[Tuple[str, int, float]] = []
        if file_list is not None:
            for fp in file_list:
                view = self._infer_view(fp)
                pid = self._extract_patient_id(fp)
                score = self.grades.get(f"{pid}_{view}", -1.0)
                self.samples.append(
                    (fp, PLANE_TO_ID[view] + self.plane_id_offset, score)
                )
        else:
            for patient_dir in sorted(self.root.iterdir()):
                if not patient_dir.is_dir():
                    continue
                pid = patient_dir.name
                for view in CAMUS_PLANES:
                    # Prefer .png; fall back to searching for matching files
                    pattern = f"*{view.replace('A', '')}*CH*.png"
                    matches = list(patient_dir.glob(pattern))
                    if not matches:
                        matches = list(patient_dir.glob(f"*{view}*.png"))
                    for img_path in sorted(matches):
                        score = self.grades.get(f"{pid}_{view}", -1.0)
                        self.samples.append(
                            (str(img_path),
                             PLANE_TO_ID[view] + self.plane_id_offset,
                             score)
                        )

    @staticmethod
    def _infer_view(filepath: str) -> str:
        name = Path(filepath).stem.upper()
        if "2CH" in name or "A2C" in name:
            return "A2C"
        return "A4C"

    @staticmethod
    def _extract_patient_id(filepath: str) -> str:
        parts = Path(filepath).parts
        for p in parts:
            if p.startswith("patient"):
                return p
        return Path(filepath).stem

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


class CAMUSPairDataset(Dataset):
    """Generates plane-consistent (I_A, I_B) pairs for CAMUS."""

    def __init__(
        self,
        anchor_dataset: CAMUSDataset,
        train_dataset: CAMUSDataset,
        pairs_per_epoch: int = 5000,
    ):
        super().__init__()
        self.anchor = anchor_dataset
        self.train = train_dataset
        self.pairs_per_epoch = pairs_per_epoch

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
