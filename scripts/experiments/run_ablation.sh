#!/bin/bash
set -e

GPU=${1:-0}
OUTPUT_DIR="./output/ablation"
CONFIG="configs/base_config.yaml"
DATASET="configs/datasets/us4qa_fetal.yaml"
SAM_CKPT="./checkpoints/samus_vit_b.pth"

echo "======================================================"
echo " STRIQ — Comprehensive Ablation Analysis"
echo "======================================================"

# ---------- Module Ablations ----------

# Full model (baseline)
echo "[1/6] Full STRIQ model..."
python scripts/train.py --config ${CONFIG} --dataset ${DATASET} \
    --sam_ckpt ${SAM_CKPT} --gpu ${GPU} \
    --output_dir ${OUTPUT_DIR}/full

# w/o OKS module
echo "[2/6] w/o OKS module..."
python scripts/train.py --config ${CONFIG} --dataset ${DATASET} \
    --sam_ckpt ${SAM_CKPT} --gpu ${GPU} \
    --output_dir ${OUTPUT_DIR}/wo_oks

# w/o Orthogonality (lambda=0)
echo "[3/6] w/o Orthogonality..."
python scripts/train.py --config ${CONFIG} --dataset ${DATASET} \
    --sam_ckpt ${SAM_CKPT} --gpu ${GPU} \
    --output_dir ${OUTPUT_DIR}/wo_orth

# w/o Adaptive routing (phi=tau=0, no topk)
echo "[4/6] w/o Adaptive routing..."
python scripts/train.py --config ${CONFIG} --dataset ${DATASET} \
    --sam_ckpt ${SAM_CKPT} --gpu ${GPU} \
    --output_dir ${OUTPUT_DIR}/wo_routing

# w/o Hierarchical LRA (single level)
echo "[5/6] w/o Hierarchical LRA..."
python scripts/train.py --config ${CONFIG} --dataset ${DATASET} \
    --sam_ckpt ${SAM_CKPT} --gpu ${GPU} \
    --output_dir ${OUTPUT_DIR}/wo_lra

# ---------- Loss Ablations ----------

LOSS_CONFIGS=(
    "loss_wo_orth:1:1:1:0"
    "loss_wo_ncc:1:0:1:0.5"
    "loss_sim_only:1:0:0:0"
    "loss_wo_smooth:1:1:0:0.5"
    "loss_wo_sim:0:1:1:0.5"
)

echo "[6/6] Loss ablations..."
for LCFG in "${LOSS_CONFIGS[@]}"; do
    IFS=: read -r NAME WSIM WNCC WSMOOTH WORTH <<< "${LCFG}"
    echo "  → ${NAME} (sim=${WSIM} ncc=${WNCC} smooth=${WSMOOTH} orth=${WORTH})"
    python scripts/train.py --config ${CONFIG} --dataset ${DATASET} \
        --sam_ckpt ${SAM_CKPT} --gpu ${GPU} \
        --output_dir ${OUTPUT_DIR}/${NAME}
done

# ---------- Evaluate all ablations ----------
echo "Evaluating all ablation variants..."
for DIR in ${OUTPUT_DIR}/*/; do
    NAME=$(basename ${DIR})
    CKPT="${DIR}checkpoints/striq_best.pth"
    if [ -f "${CKPT}" ]; then
        echo "  → Evaluating ${NAME}"
        python scripts/test.py --config ${CONFIG} --dataset ${DATASET} \
            --checkpoint ${CKPT} --gpu ${GPU} \
            --output_dir ${OUTPUT_DIR}/eval_${NAME}
    fi
done

echo "======================================================"
echo " Ablation analysis complete. Results in ${OUTPUT_DIR}/"
echo "======================================================"
