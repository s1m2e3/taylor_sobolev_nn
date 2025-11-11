from import_large_model import import_large_model 
import torch
import tensorflow as tf

# from training_loop import train  
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


# setting up for one process per GPU (i.e pytorch distributed run)
def setup_distributed():
    # Initialise distributed communication group using NCCL backend to exchange tensor parameters
    dist.init_process_group(backend="nccl")
    # tells cuda which GPU to use
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


# calling the funciton to initialise for PyTorch’s distributed process group for multi-process multi-GPU training
setup_distributed()
# picks the GPU device that this particular process should use
rank = dist.get_rank()
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))


# Added barrier to prevent data loading race condition ---
# Only Rank 0 downloads, others wait for it to finish.
if rank != 0:
    dist.barrier() 

print(f"[Rank {rank}] Loading dataset...")
cifar = tf.keras.datasets.cifar10    
(x_train_full, y_train_full), (x_test, y_test) = cifar.load_data()

# Rank 0 signals it's done, releasing other ranks to load from cache
if rank == 0:
    print("[Rank 0] Dataset loaded/cached successfully.")
    dist.barrier()
# --- End of FIX 2 ---


x_valid, x_train = x_train_full[:5000], x_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                         std=(0.2470, 0.2435, 0.2616)),
])

# --- optimized big_model loading ---
big_model = FSDP(import_large_model().to(device), use_orig_params=True)
big_model.eval()
for p in big_model.parameters():
    p.requires_grad_(False)

# --- small_model loading ---
small_model = FSDP(model(in_channels=3, pixel_size=32, num_classes=10).to(device),use_orig_params=True)

# --- Run Training ---
print(f"[Rank {rank}] Starting training...")
train(big_model,small_model,x_train,y_train,x_valid,y_valid, preprocess=preprocess,device = device )

# --- Save Model ---
if dist.get_rank() == 0:
    print("Training complete. Saving model on rank 0...")

# 1. Set up the save policy
save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

# 2. Get the full, unsharded model state dictionary
with FSDP.state_dict_type(small_model, StateDictType.FULL_STATE_DICT, save_policy):
    cpu_state_dict = small_model.state_dict()

# 3. Only rank 0 saves it to disk
if dist.get_rank() == 0:
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(cpu_state_dict, "checkpoints/small_model_final.pth")
    print("Model saved successfully to checkpoints/small_model_final.pth")



# test(big_model,small_model,x_test,y_test,device= device)

# Clean up distributed processes
dist.destroy_process_group()
print(f"Job finished on rank {rank}.")