#!/bin/bash
set -euo pipefail

# Enable Lmod without sourcing user rc files
if [ -f /etc/profile.d/modules.sh ]; then
  . /etc/profile.d/modules.sh
elif [ -f /etc/profile.d/lmod.sh ]; then
  . /etc/profile.d/lmod.sh
fi


source "$VENV_DIR/bin/activate"

VENV_ROOT_LIB_PATH="$VENV_DIR/lib"
VENV_ROOT_LIB64_PATH="$VENV_DIR/lib64"

# Check lib64 first (common on HPC)
if [ -d "$VENV_ROOT_LIB64_PATH" ]; then
    export LD_LIBRARY_PATH="$VENV_ROOT_LIB64_PATH:$LD_LIBRARY_PATH"
    echo "Using lib64 path for dynamic linking."
elif [ -d "$VENV_ROOT_LIB_PATH" ]; then
    export LD_LIBRARY_PATH="$VENV_ROOT_LIB_PATH:$LD_LIBRARY_PATH"
    echo "Using lib path for dynamic linking."
fi

cd "$PROJECT_DIR"

echo "[${HOSTNAME%%.*}] python: $(which python)"
ldd "$(which python)" | grep -E "libpython|not found" || true
python - <<PY
import torch, socket
print("torch", torch.__version__, "cuda?", torch.cuda.is_available(),
      "ngpus", torch.cuda.device_count(), "host", socket.gethostname())
PY

exec stdbuf -oL -eL torchrun \
  --nnodes=${SLURM_NNODES} \
  --nproc_per_node=${SLURM_GPUS_ON_NODE} \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  --rdzv_id=${SLURM_JOB_ID} \
  --rdzv-conf="timeout=600" \
  main.py --mode=${TRAIN_MODE:-jvp}