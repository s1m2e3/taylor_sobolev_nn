#!/bin/bash
#SBATCH --job-name=my_fsdp_job
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1            # one torchrun launcher per node
#SBATCH --gpus-per-node=2              # GPUs per node
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --account=info621
#SBATCH --partition=gpu_standard
#SBATCH --array=1-33%1
#SBATCH -o slurm-%A_%a.out
#SBATCH -e slurm-%A_%a.err

set -euo pipefail

module purge
module load python/3.11.4

# Basic env
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PROJECT_DIR=/home/u3/ralphdsouza/myproject
export VENV_DIR="/home/u3/ralphdsouza/venv"
cd "$PROJECT_DIR"


# Use the wrapper script to set the TRAIN_MODE
export TRAIN_MODE=$(python -u "$PROJECT_DIR/run_sweep.py" $SLURM_ARRAY_TASK_ID)
echo "Current Task ID: $SLURM_ARRAY_TASK_ID, Running Mode: $TRAIN_MODE"


# Shared TF/Keras cache; keep TF from pre-allocating GPUs
export KERAS_HOME="$PROJECT_DIR/inputs"
export TF_FORCE_GPU_ALLOW_GROWTH=1
mkdir -p "$KERAS_HOME"

# Rendezvous
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29400

# NCCL / DDP debug
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_ASYNC_ERROR_HANDLING=1
export GLOO_SOCKET_IFNAME=$(ip route | grep default | awk '{print $5}')


echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT NODES=$SLURM_NNODES"
echo "Starting ${SLURM_NNODES}-node, ${SLURM_GPUS_ON_NODE:-2}-GPU-per-node job..."

# Launch
srun --mpi=pmix --label --kill-on-bad-exit=1 \
     --output=node_%t.out --error=node_%t.err \
     --export=ALL,MASTER_ADDR,MASTER_PORT,SLURM_NNODES,SLURM_GPUS_ON_NODE,PROJECT_DIR,VENV_DIR \
     bash "$PROJECT_DIR/launch_worker.sh"