#!/bin/bash
set -e

# Initialize variables to allow them to be overridden by parameters
MASTER_ADDR_PARAM=""
PUBLIC_IP_PARAM=""
WORLD_SIZE_PARAM=""
NUM_GPUS_PARAM=""
RANK_PARAM=""
MASTER_PORT_PARAM=""         # For the broker
FASTERDP_PULL_PORT_PARAM=""  # For this node's data socket

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --master-ip-address) MASTER_ADDR_PARAM="$2"; shift ;;
        --my-ip-address) PUBLIC_IP_PARAM="$2"; shift ;;
        --world-size) WORLD_SIZE_PARAM="$2"; shift ;;
        --num-gpus) NUM_GPUS_PARAM="$2"; shift ;;
        --rank) RANK_PARAM="$2"; shift ;;
        --master-port) MASTER_PORT_PARAM="$2"; shift ;;             # New
        --fasterdp-pull-port) FASTERDP_PULL_PORT_PARAM="$2"; shift ;; # New
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Use parameters if provided, otherwise, you might have defaults or error out
export MASTER_ADDR="${MASTER_ADDR_PARAM}"
export PUBLIC_IP="${PUBLIC_IP_PARAM}"
export WORLD_SIZE="${WORLD_SIZE_PARAM}"
export NUM_GPUS="${NUM_GPUS_PARAM}"
export RANK="${RANK_PARAM}"
export MASTER_PORT="${MASTER_PORT_PARAM}"             # Use passed param
export FASTERDP_PULL_PORT="${FASTERDP_PULL_PORT_PARAM}" # Use passed param

# Validate required variables (now including the new ones)
if [ -z "$MASTER_ADDR" ]  || \
   [ -z "$PUBLIC_IP" ] || \
   [ -z "$WORLD_SIZE" ] || \
   [ -z "$NUM_GPUS" ] || \
   [ -z "$RANK" ] || \
   [ -z "$MASTER_PORT" ] || \
   [ -z "$FASTERDP_PULL_PORT" ]; then
    echo "Error: --master-ip-address, --my-ip-address, --world-size, --num-gpus, --rank, --master-port, and --fasterdp-pull-port are required"
    exit 1
fi

# Determine CUDA_VISIBLE_DEVICES based on NUM_GPUS
if [ -n "$NUM_GPUS" ] && [ "$NUM_GPUS" -gt 0 ]; then
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $(($NUM_GPUS-1)))
else
    echo "Warning: NUM_GPUS not set or invalid, CUDA_VISIBLE_DEVICES might not be set correctly."
    # Potentially unset it or set to empty if no GPUs desired, or handle error
fi


# --- Your original script logic from here ---
# Assuming PROJECT_CONTAINER_DIR and DATA_CONTAINER_DIR are set if this runs inside Docker
# Or that the paths are relative to where this script is executed if run directly on host.
# For Docker, these might be fixed paths like /app or /workspace.
PROJECT_DIR_INSIDE_CONTAINER_OR_RELATIVE="." # Adjust if backend/test is not in current dir
DATA_ROOT_INSIDE_CONTAINER="/home/data" # Example

# This pushd might need adjustment depending on where this script is located
# relative to 'backend/test' when it's executed.
# If this script is at the root of your project:
pushd "${PROJECT_DIR_INSIDE_CONTAINER_OR_RELATIVE}/backend/test"

./cleanup.sh || true
sleep 1

# Variables are already exported

export TEST_MODEL=resnet152
# CUDA_VISIBLE_DEVICES is set above

export GLOBAL_MISC_COMMAND="--dataset=imagenet100 --num-epochs=100 --imagenet-root=${DATA_ROOT_INSIDE_CONTAINER}"
export CMDLINE="python test_end_to_end.py --model=$TEST_MODEL $GLOBAL_MISC_COMMAND"

echo "--- Configuration ---"
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT} (Broker)"
echo "PUBLIC_IP: ${PUBLIC_IP}"
echo "FASTERDP_PULL_PORT: ${FASTERDP_PULL_PORT} (This Node's Data Port)"
echo "WORLD_SIZE: ${WORLD_SIZE}"
echo "RANK: ${RANK}"
echo "NUM_GPUS: ${NUM_GPUS}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "---------------------"
echo "Executing: $CMDLINE"
$CMDLINE

popd


i mean, i want to run docker same command, only differences is run script command like this
node 0:
./~~~.sh 0(node) 2(world) 2(gpus)
./~~~.sh 0 2 2

please start with this script, and just add set_env.sh
#!/bin/bash
set -e

# Initialize variables to allow them to be overridden by parameters
MASTER_ADDR_PARAM=""
PUBLIC_IP_PARAM=""
WORLD_SIZE_PARAM=""
NUM_GPUS_PARAM=""
RANK_PARAM=""
MASTER_PORT_PARAM=""         # For the broker
FASTERDP_PULL_PORT_PARAM=""  # For this node's data socket

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --master-ip-address) MASTER_ADDR_PARAM="$2"; shift ;;
        --my-ip-address) PUBLIC_IP_PARAM="$2"; shift ;;
        --world-size) WORLD_SIZE_PARAM="$2"; shift ;;
        --num-gpus) NUM_GPUS_PARAM="$2"; shift ;;
        --rank) RANK_PARAM="$2"; shift ;;
        --master-port) MASTER_PORT_PARAM="$2"; shift ;;             # New
        --fasterdp-pull-port) FASTERDP_PULL_PORT_PARAM="$2"; shift ;; # New
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Use parameters if provided, otherwise, you might have defaults or error out
export MASTER_ADDR="${MASTER_ADDR_PARAM}"
export PUBLIC_IP="${PUBLIC_IP_PARAM}"
export WORLD_SIZE="${WORLD_SIZE_PARAM}"
export NUM_GPUS="${NUM_GPUS_PARAM}"
export RANK="${RANK_PARAM}"
export MASTER_PORT="${MASTER_PORT_PARAM}"             # Use passed param
export FASTERDP_PULL_PORT="${FASTERDP_PULL_PORT_PARAM}" # Use passed param

# Validate required variables (now including the new ones)
if [ -z "$MASTER_ADDR" ]  || \
   [ -z "$PUBLIC_IP" ] || \
   [ -z "$WORLD_SIZE" ] || \
   [ -z "$NUM_GPUS" ] || \
   [ -z "$RANK" ] || \
   [ -z "$MASTER_PORT" ] || \
   [ -z "$FASTERDP_PULL_PORT" ]; then
    echo "Error: --master-ip-address, --my-ip-address, --world-size, --num-gpus, --rank, --master-port, and --fasterdp-pull-port are required"
    exit 1
fi

# Determine CUDA_VISIBLE_DEVICES based on NUM_GPUS
if [ -n "$NUM_GPUS" ] && [ "$NUM_GPUS" -gt 0 ]; then
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $(($NUM_GPUS-1)))
else
    echo "Warning: NUM_GPUS not set or invalid, CUDA_VISIBLE_DEVICES might not be set correctly."
    # Potentially unset it or set to empty if no GPUs desired, or handle error
fi


# --- Your original script logic from here ---
# Assuming PROJECT_CONTAINER_DIR and DATA_CONTAINER_DIR are set if this runs inside Docker
# Or that the paths are relative to where this script is executed if run directly on host.
# For Docker, these might be fixed paths like /app or /workspace.
PROJECT_DIR_INSIDE_CONTAINER_OR_RELATIVE="." # Adjust if backend/test is not in current dir
DATA_ROOT_INSIDE_CONTAINER="/home/data" # Example

# This pushd might need adjustment depending on where this script is located
# relative to 'backend/test' when it's executed.
# If this script is at the root of your project:
pushd "${PROJECT_DIR_INSIDE_CONTAINER_OR_RELATIVE}/backend/test"

./cleanup.sh || true
sleep 1

# Variables are already exported

export TEST_MODEL=resnet152
# CUDA_VISIBLE_DEVICES is set above

export GLOBAL_MISC_COMMAND="--dataset=imagenet100 --num-epochs=100 --imagenet-root=${DATA_ROOT_INSIDE_CONTAINER}"
export CMDLINE="python test_end_to_end.py --model=$TEST_MODEL $GLOBAL_MISC_COMMAND"

echo "--- Configuration ---"
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT} (Broker)"
echo "PUBLIC_IP: ${PUBLIC_IP}"
echo "FASTERDP_PULL_PORT: ${FASTERDP_PULL_PORT} (This Node's Data Port)"
echo "WORLD_SIZE: ${WORLD_SIZE}"
echo "RANK: ${RANK}"
echo "NUM_GPUS: ${NUM_GPUS}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "---------------------"
echo "Executing: $CMDLINE"
$CMDLINE

popd