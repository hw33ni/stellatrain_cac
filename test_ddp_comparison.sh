#!/bin/bash
set -e

# This script runs INSIDE the Docker container for PyTorch DDP
# It expects ENV_... variables to be set by the sourced set_env.sh
# It also expects DDP_NODE_RANK, DDP_NUM_NODES, DDP_GPUS_PER_NODE as arguments.

# DDP specific parameters passed to this script
ARG_DDP_NODE_RANK="$1"         # 0 or 1 (node rank for DDP)
ARG_DDP_TOTAL_NODES="$2"     # e.g., 2
ARG_DDP_GPUS_PER_NODE="$3"   # e.g., 2

# These come from the sourced set_env.sh
# For DDP, MASTER_ADDR is always the IP of the node with rank 0
DDP_PY_MASTER_ADDR="${ENV_MASTER_IP}"
DDP_PY_MASTER_PORT="${ENV_MASTER_BROKER_PORT}" # Re-using Stella's broker port for DDP store

# Fixed paths and default training params for DDP
PROJECT_CONTAINER_DIR="/home/stellatrain/explore-dp"
PYTHON_DDP_SCRIPT_NAME="ddp_comparison_train.py"
DATA_CONTAINER_DIR="/home/data"
MODEL_NAME="resnet50"
BATCH_SIZE_PER_GPU="64"
EPOCHS="3"
LR="0.1"
BACKEND="gloo"
DATALOADERS="4"
WARMUP="20"

if [ -z "$ARG_DDP_NODE_RANK" ] || [ -z "$ARG_DDP_TOTAL_NODES" ] || [ -z "$ARG_DDP_GPUS_PER_NODE" ]; then
  echo "Usage: $0 <ddp_node_rank> <ddp_total_nodes> <ddp_gpus_per_node>"
  exit 1
fi

echo "--- DDP Launch (Node Rank ${ARG_DDP_NODE_RANK}) ---"
echo "  DDP Master IP (from env): ${DDP_PY_MASTER_ADDR}"
echo "  DDP Master Port (from env): ${DDP_PY_MASTER_PORT}"
echo "  This Node's DDP Rank: ${ARG_DDP_NODE_RANK}"
echo "  Total DDP Nodes: ${ARG_DDP_TOTAL_NODES}"
echo "  GPUs per Node for DDP: ${ARG_DDP_GPUS_PER_NODE}"
echo "-----------------------------------------------------------------"

python3 "${PROJECT_CONTAINER_DIR}/${PYTHON_DDP_SCRIPT_NAME}" \
    --master-addr "${DDP_PY_MASTER_ADDR}" \
    --master-port "${DDP_PY_MASTER_PORT}" \
    --node-rank "${ARG_DDP_NODE_RANK}" \
    --num-nodes "${ARG_DDP_TOTAL_NODES}" \
    --num-gpus-per-node "${ARG_DDP_GPUS_PER_NODE}" \
    --model-name "${MODEL_NAME}" \
    --imagenet-root "${DATA_CONTAINER_DIR}" \
    --batch-size "${BATCH_SIZE_PER_GPU}" \
    --num-epochs "${EPOCHS}" \
    --lr "${LR}" \
    --backend "${BACKEND}" \
    --num-workers "${DATALOADERS}" \
    --warmup-iters "${WARMUP}"

echo "DDP Python script finished for Node Rank ${ARG_DDP_NODE_RANK}."