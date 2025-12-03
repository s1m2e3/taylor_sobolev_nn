from import_large_model import import_large_model 
import torch
import tensorflow as tf
import argparse
from training_loop import train 
from testing_loop import test
from model import model
from torchvision import transforms

# The FSDP imports
import os
import torch.distributed as dist
from torch.distributed import init_process_group
from torch.distributed.fsdp.wrap import wrap

# FSDP saving imports
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, CPUOffload
from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType


# ARGUMENT PARSING
parser = argparse.ArgumentParser(description='FSDP Training Mode Selector')
parser.add_argument('--mode', type=str, default='data', choices=['data', 'teacher', 'jvp'],
                    help='Training mode: "data", "teacher", or "jvp".')
args = parser.parse_args()

# --- Set boolean flags ---
if args.mode == 'data':
    only_data = True
    teacher_only_data = False
    teacher_jvp = False
elif args.mode == 'teacher':
    only_data = False
    teacher_only_data = True
    teacher_jvp = False
elif args.mode == 'jvp':
    only_data = False
    teacher_only_data = False 
    teacher_jvp = True


# Distributed Setup
def setup_distributed():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

setup_distributed()
rank = dist.get_rank()
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))

# Data Loading
if rank != 0:
    dist.barrier() 

print(f"[Rank {rank}] Loading dataset...")
cifar = tf.keras.datasets.cifar10    
(x_train_full, y_train_full), (x_test, y_test) = cifar.load_data()

if rank == 0:
    print("[Rank 0] Dataset loaded/cached successfully.")
    dist.barrier()

x_valid, x_train = x_train_full[:5000], x_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                         std=(0.2470, 0.2435, 0.2616)),
])

# Model Loading
big_model = FSDP(import_large_model().to(device), use_orig_params=True)
big_model.eval()
for p in big_model.parameters():
    p.requires_grad_(False)

small_model = FSDP(model(in_channels=3, pixel_size=32, num_classes=10).to(device),use_orig_params=True)


#   EPOCHS & PATHS 
# Get Task ID 
task_id_str = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
try:
    task_id_int = int(task_id_str)
except ValueError:
    task_id_int = 0

#  Determine Epoch Count
# Runs 31, 32, 33 get 1000 epochs. All others get 100.
if task_id_int >= 31:
    NUM_EPOCHS = 1000
else:
    NUM_EPOCHS = 100

# 2. Determine Checkpoint Directory
mode = os.environ.get("TRAIN_MODE", "unknown")

if task_id_str == "31":
    checkpoint_dir = "checkpoints/1k_epoch_teacher"
elif task_id_str == "32":
    checkpoint_dir = "checkpoints/1k_epoch_jvp"
elif task_id_str == "33":
    checkpoint_dir = "checkpoints/1k_epoch_data"
else:
    # Standard naming for runs 1-30
    checkpoint_dir = f"checkpoints/{mode}_run_{task_id_str}"

# Run Training 
if rank == 0:
    print(f"--- Starting training in mode: '{args.mode}' for {NUM_EPOCHS} epochs ---")

    # Clean up old log file to prevent appending
    log_file = f"logs/{mode}/metrics_run_{task_id_str}.jsonl"
    if os.path.exists(log_file):
        try:
            os.remove(log_file)
            print(f"[Rank 0] Deleted old log file: {log_file}")
        except:
            pass

train(
    big_model, small_model, x_train, y_train, x_valid, y_valid, 
    preprocess=preprocess, device=device,
    only_data=only_data, 
    teacher_only_data=teacher_only_data, 
    teacher_jvp=teacher_jvp,
    epochs=NUM_EPOCHS 
)

# Save Model 
if dist.get_rank() == 0:
    print("Training complete. Saving model on rank 0...")

save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

os.makedirs(checkpoint_dir, exist_ok=True)
save_path = os.path.join(checkpoint_dir, "small_model_final.pth")

with FSDP.state_dict_type(small_model, StateDictType.FULL_STATE_DICT, save_policy):
    cpu_state_dict = small_model.state_dict()

if dist.get_rank() == 0:
    torch.save(cpu_state_dict, save_path) 
    print(f"Model saved successfully to {save_path}")

dist.destroy_process_group()
print(f"Job finished on rank {rank}.")