#!/bin/bash
set -e

# This script runs INSIDE the Docker container for PyTorch DDP
# It sources /etc/set_env.sh for common IP/Port info.
# It takes DDP_NODE_RANK, DDP_TOTAL_NODES, DDP_GPUS_PER_NODE as arguments.

ARG_DDP_NODE_RANK="$1"
ARG_DDP_TOTAL_NODES="$2"
ARG_DDP_GPUS_PER_NODE="$3"

if [ -z "$ARG_DDP_NODE_RANK" ] || [ -z "$ARG_DDP_TOTAL_NODES" ] || [ -z "$ARG_DDP_GPUS_PER_NODE" ]; then
  echo "Usage (inside container): $0 <this_ddp_node_rank> <total_ddp_nodes> <gpus_on_this_node>"
  exit 1
fi

if [ -f set_env.sh ]; then
    source set_env.sh
else
    echo "ERROR: set_env.sh not found!"
    exit 1
fi

DDP_PY_MASTER_ADDR="${ENV_MASTER_IP}"
DDP_PY_MASTER_PORT="${ENV_MASTER_BROKER_PORT}" # Using Stella's broker port for DDP store

DETECTED_IFACE=$(ip -o -4 route show to default | awk '{print $5}' | head -n1)
if [ -z "$DETECTED_IFACE" ]; then
    DETECTED_IFACE=$(ip -o -4 addr show | awk '!/^[0-9]*: ?lo|docker|veth|br-/ {print $2}' | head -n1)
fi
if [ -n "$DETECTED_IFACE" ]; then
    export NCCL_SOCKET_IFNAME="$DETECTED_IFACE"
    export GLOO_SOCKET_IFNAME="$DETECTED_IFACE"
else
    export NCCL_SOCKET_IFNAME="eth0" # Fallback
    export GLOO_SOCKET_IFNAME="eth0" # Fallback
fi
export NCCL_DEBUG="${NCCL_DEBUG_LEVEL:-WARN}"
echo "Using NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"

PROJECT_CONTAINER_DIR="/home/stellatrain/explore-dp"
PYTHON_DDP_SCRIPT_NAME="ddp_comparison_train.py"
DATA_CONTAINER_DIR="/home/data"
MODEL_NAME="resnet50" BATCH_SIZE_PER_GPU="64" EPOCHS="3" LR="0.1" TARGET_BACKEND="nccl" DATALOADERS="4" WARMUP="20"

echo "--- DDP Launch (Node Rank ${ARG_DDP_NODE_RANK}, Backend: ${TARGET_BACKEND}) ---"
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
    --backend "${TARGET_BACKEND}" \
    --num-workers "${DATALOADERS}" \
    --warmup-iters "${WARMUP}"
echo "DDP (NCCL) Python script finished for Node Rank ${ARG_DDP_NODE_RANK}."