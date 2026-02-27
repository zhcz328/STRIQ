#!/bin/bash
set -e

GPU=${1:-0}
OUTPUT_DIR="./output/us4qa"
CONFIG="configs/base_config.yaml"
DATASET="configs/datasets/us4qa_fetal.yaml"
SAM_CKPT="./checkpoints/samus_vit_b.pth"

echo "======================================================"
echo " STRIQ — US4QA Fetal Ultrasound Evaluation Pipeline"
echo "======================================================"

# Step 1: Cache SAMUS embeddings for anchor selection
echo "[Step 1/4] Caching SAMUS feature embeddings..."
python scripts/cache_features.py \
    --data_root ./data/US4QA \
    --sam_ckpt ${SAM_CKPT} \
    --output ${OUTPUT_DIR}/cache/embeddings.npy \
    --gpu ${GPU}

# Step 2: Train STRIQ with 5-fold cross-validation
echo "[Step 2/4] Training STRIQ (5-fold CV, 500 epochs)..."
for FOLD in 1 2 3 4 5; do
    echo "  → Fold ${FOLD}/5"
    python scripts/train.py \
        --config ${CONFIG} \
        --dataset ${DATASET} \
        --sam_ckpt ${SAM_CKPT} \
        --gpu ${GPU} \
        --output_dir ${OUTPUT_DIR}/fold_${FOLD}
done

# Step 3: Evaluate each fold
echo "[Step 3/4] Evaluating on D_C test set..."
for FOLD in 1 2 3 4 5; do
    echo "  → Evaluating fold ${FOLD}/5"
    python scripts/test.py \
        --config ${CONFIG} \
        --dataset ${DATASET} \
        --checkpoint ${OUTPUT_DIR}/fold_${FOLD}/checkpoints/striq_best.pth \
        --gpu ${GPU} \
        --output_dir ${OUTPUT_DIR}/eval_fold_${FOLD}
done

# Step 4: Aggregate results across folds
echo "[Step 4/4] Aggregating cross-validation results..."
python -c "
import json, glob, numpy as np
results = []
for f in sorted(glob.glob('${OUTPUT_DIR}/eval_fold_*/eval_results.json')):
    with open(f) as fp:
        results.append(json.load(fp)['metrics'])
for key in ['srcc', 'plcc', 'f1']:
    vals = [r[key] for r in results]
    print(f'  {key.upper()}: {np.mean(vals):.4f} ± {np.std(vals):.4f}')
"

echo "======================================================"
echo " Evaluation complete. Results in ${OUTPUT_DIR}/"
echo "======================================================"
