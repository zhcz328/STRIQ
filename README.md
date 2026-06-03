# Subspace-Guided Semantic and Topological Invariant Registration for Annotation-Free Ultrasound Plane Quality Control

## Abstract

Reliable quality control (QC) of ultrasound images is essential for both real-time acquisition guidance and retrospective clinical audit, yet existing approaches rely heavily on per-plane annotations or employ pseudo-labeling prone to systematic bias under spatial deformations inherent in clinical acquisition. We present **STRIQ**, a registration-driven framework that recasts annotation-free US plane quality control as a subspace-guided consistency measurement problem. Specifically, STRIQ introduces a **Latent Registration Aligner (LRA)** to establish hierarchical feature space correspondences between query images and variance-driven anchors, which are autonomously distilled from unlabeled data via a variance spectrum criterion to serve as structurally stable prototypes. To further disambiguate anatomical planes and mitigate negative knowledge transfer, we propose an **Orthogonal Knowledge Subspace (OKS)** module. The OKS decomposes plane-specific representations into mutually orthogonal subspaces, enabling fine-grained expert collaboration while preventing inter-plane interference, ensuring that the quality metric is grounded in principled subspace proximity. Extensive experiments on the US4QA and public CAMUS datasets demonstrate that STRIQ achieves state-of-the-art correlation with clinical quality scores, establishing a new paradigm for annotation-free, real-time reliable ultrasound quality control.

![STRIQ Framework](./figs/STRIQ.png)

🎉 Accepted to MICCAI 2026. This repository provides the official PyTorch implementation of Subspace-Guided Semantic and Topological Invariant Registration for Annotation-Free Ultrasound Plane Quality Control.

If you find this repository useful, please consider giving it a ⭐!

## 🔨 PostScript

😄 This project is the PyTorch implementation of STRIQ.

😆 Our experimental platform is configured with one *RTX 4090* GPU (24 GB VRAM).

🚀 STRIQ achieves **~5.6 ms/frame (~180 FPS)** inference speed with pre-cached anchor features.

## 💻 Installation

1. Clone this repository.

   ```bash
   git clone https://******.git
   cd STRIQ
   ```

2. Create conda environment.

   ```bash
   conda env create -f environment.yml
   conda activate striq
   ```

3. Alternatively, install dependencies via pip.

   ```bash
   pip install torch torchvision numpy scipy scikit-learn matplotlib pillow pyyaml tqdm opencv-python einops timm
   ```

### US4QA (Fetal Ultrasound)

- **30,757 images** across four standard planes: Abdomen, 4CH, Kidney, Face
- **324 examinations** with ground-truth quality scores averaged from six sonographers on [0, 1]
- Organize the dataset directory as:

  ```
  data/US4QA/
  ├── Abdomen/
  ├── 4CH/
  ├── Kidney/
  ├── Face/
  └── annotations/
      └── quality_scores.csv
  ```

### CAMUS (Cardiac Ultrasound)

- Download from: [CAMUS Challenge](http://camus.creatis.insa-lyon.fr/challenge/)
- Apical 2-chamber (A2C) and 4-chamber (A4C) views with ordinal grades (Good / Medium / Poor)
- Convert to `.png` format and organize as:

  ```
  data/CAMUS/
  ├── patient0001/
  │   ├── patient0001_2CH_ED.png
  │   ├── patient0001_4CH_ED.png
  │   └── ...
  └── quality_grades.csv
  ```

## 🔧 Pre-trained Encoder

STRIQ uses a SAMUS-adapted ViT-B encoder (`F_pre`) for variance-spectrum anchor selection. Download the checkpoint:

- **SAM ViT-B**: [sam_vit_b_01ec64.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
- **SAMUS**: [Pre-trained weights](https://github.com/xianlin7/SAMUS)

Place checkpoints under `./checkpoints/`.

## 🐾 STRIQ Training

### Step 1: Cache Anchor Embeddings

Pre-compute SAMUS feature embeddings for the entire dataset to accelerate anchor selection:

```bash
python scripts/cache_features.py \
    --data_root ./data/US4QA \
    --sam_ckpt ./checkpoints/samus_vit_b.pth \
    --output ./cache/us4qa_embeddings.npy \
    --gpu 0
```

### Step 2: Train STRIQ

```bash
python scripts/train.py \
    --config configs/base_config.yaml \
    --dataset configs/datasets/us4qa_fetal.yaml \
    --sam_ckpt ./checkpoints/samus_vit_b.pth \
    --gpu 0 \
    --output_dir ./output/train
```

> **Default hyperparameters**: Adam (lr=1e-4), 500 epochs, batch size 64, OKS rank r=16, λ=0.5, φ=τ=0.1.

### Step 3: Evaluate

```bash
python scripts/test.py \
    --config configs/base_config.yaml \
    --dataset configs/datasets/us4qa_fetal.yaml \
    --checkpoint ./output/train/checkpoints/striq_best.pth \
    --gpu 0 \
    --output_dir ./output/eval
```

## 🔬 Reproduce Paper Results

### Table 1: Comparison with State-of-the-Art

```bash
bash scripts/experiments/run_us4qa.sh 0    # GPU ID 
```

### Table 2: Ablation Analysis

```bash
bash scripts/experiments/run_ablation.sh 0
```

## ⚙️ Key Hyperparameters

| Parameter | Symbol | Default | 
|-----------|--------|---------|
| OKS rank | r | 16 | 
| Orthogonality weight | λ | 0.5 | 
| Activation threshold | φ = τ | 0.1 | 
| Anchors per plane | k₁ | 20 | 
| Acceptance threshold | τ_QC | 0.5 |
| Loss weights | w_sim : w_NCC : w_smooth | 1 : 1 : 1 | 
| Cascaded LRA levels | L | 3 | 
| Transform mode | — | affine |

## 📈 Experimental Results

### US4QA (SRCC)

| Plane | BRISQUE | MSSIM | ARNIQA | SFD-IQA | Weight Score | CRL-UIQA | **STRIQ** |
|-------|---------|-------|--------|---------|--------------|-----------|-----------|
| Abdomen | 0.389 | 0.217 | 0.563 | 0.589 | 0.738 | 0.801 | **0.847** |
| 4CH | 0.407 | 0.405 | 0.494 | 0.477 | 0.679 | 0.825 | **0.883** |
| Kidney | 0.449 | 0.401 | 0.523 | 0.549 | 0.724 | 0.731 | **0.819** |
| Face | 0.488 | 0.469 | 0.571 | 0.533 | 0.692 | 0.759 | **0.831** |
| **Average** | 0.433 | 0.373 | 0.538 | 0.537 | 0.708 | 0.779 | **0.845** |

### CAMUS (SRCC)

| Plane | BRISQUE | MSSIM | ARNIQA | SFD-IQA | CRL-UIQA | **STRIQ** |
|-------|---------|-------|--------|---------|-----------|-----------|
| A2C | 0.341 | 0.298 | 0.512 | 0.527 | 0.683 | **0.762** |
| A4C | 0.367 | 0.312 | 0.538 | 0.541 | 0.701 | **0.779** |
| **Average** | 0.354 | 0.305 | 0.525 | 0.534 | 0.692 | **0.771** |

[//]: # (## 📂 Project Structure)

[//]: # ()
[//]: # (```📊 Datasets)

[//]: # (STRIQ/)

[//]: # (├── configs/                     # Configuration files)

[//]: # (│   ├── datasets/)

[//]: # (│   │   ├── us4qa_fetal.yaml     )

[//]: # (│   │   └── camus_cardiac.yaml   )

[//]: # (│   ├── models/)

[//]: # (│   │   ├── striq_resnet18.yaml  # Default model hyperparameters)

[//]: # (│   │   └── ablation_configs.yaml# Ablation variants)

[//]: # (│   └── base_config.yaml         # Global defaults)

[//]: # (├── core/                        # Core algorithmic modules)

[//]: # (│   ├── registration/)

[//]: # (│   │   ├── aligner.py           # LRA cascaded affine alignment)

[//]: # (│   │   └── transformer.py       # 8 affine transformation modes)

[//]: # (│   ├── subspace/)

[//]: # (│   │   ├── oks_layer.py         # OKS decomposition & conflict masks &#40;Eq. 4&#41;)

[//]: # (│   │   └── assembly.py          # Synergy Expert assembly &#40;Eq. 5&#41;)

[//]: # (│   └── builder.py               # Full STRIQ network construction)

[//]: # (├── data/)

[//]: # (│   ├── sam_encoder/             # SAMUS-based encoder F_pre for anchors)

[//]: # (│   │   ├── modeling/)

[//]: # (│   │   │   ├── image_encoder.py # ViT-B with position/feature adapters)

[//]: # (│   │   │   └── common.py        # LayerNorm2d, MLPBlock)

[//]: # (│   │   └── build_sam.py         # Encoder loading utility)

[//]: # (│   ├── datasets/)

[//]: # (│   │   ├── us4qa.py             # US4QA fetal dataset & pair generation)

[//]: # (│   │   └── camus.py             # CAMUS cardiac dataset)

[//]: # (│   ├── loader.py                # Augmentation & DataLoader factories)

[//]: # (│   └── anchor_utils.py          # Variance-spectrum anchor selection)

[//]: # (├── engine/)

[//]: # (│   ├── trainer.py               # Training loop with OKS gradient projection)

[//]: # (│   ├── evaluator.py             # Quality score Q&#40;I_C&#41; inference )

[//]: # (│   └── lr_scheduler.py          # Cosine annealing with warm-up)

[//]: # (├── losses/)

[//]: # (│   ├── registration_loss.py     )

[//]: # (│   └── orthogonality_loss.py   )

[//]: # (├── scripts/)

[//]: # (│   ├── experiments/)

[//]: # (│   │   ├── run_us4qa.sh        )

[//]: # (│   │   └── run_ablation.sh    )

[//]: # (│   ├── train.py)

[//]: # (│   ├── test.py)

[//]: # (│   └── cache_features.py        # Pre-compute SAMUS embeddings)

[//]: # (├── utils/)

[//]: # (│   ├── metrics.py               # SRCC, PLCC, F1 computation)

[//]: # (│   ├── visualization.py         # Score distributions, t-SNE, deformation plots)

[//]: # (│   └── checkpointer.py          # Checkpoint save/load/resume)

[//]: # (├── main.py                      # Unified CLI entry point)

[//]: # (├── environment.yml)

[//]: # (└── README.md)

[//]: # (```)

### 
