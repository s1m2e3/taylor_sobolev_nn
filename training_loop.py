import torch 
from taylor_sobolev_utils import estimate_gradient, get_jacobian # <-- MODIFIED: We are using this
from PIL import Image
from torch.utils.data import Dataset, DataLoader # <-- MODIFIED
from torch.utils.data.distributed import DistributedSampler # <-- MODIFIED: Import Sampler
import torch.distributed as dist # <-- MODIFIED: Import distributed
import json, os

class CifarDataset(Dataset):
    def __init__(self, data, targets, transform):
        self.data = data
        self.targets = torch.tensor(targets, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)
        img = self.transform(img)
        return img, target

def train(big_model,small_model,x_train,y_train,x_valid,y_valid, preprocess, batch_size=50,epochs=100,lr=1e-3, weight_decay=5e-4, entropy_weight=0.25, jvp_weight = 0.1, distil_weight = 1.0, T = 1.0, num_samples_random=4,device = None):
    
    rank = dist.get_rank() # <-- MODIFIED: Get the rank of this process
    world_size = dist.get_world_size() # <-- MODIFIED: Get total number of processes

    # --- DATALOADER UPDATED ---
    train_dataset = CifarDataset(x_train, y_train, transform=preprocess)
    # Create the DistributedSampler
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) # <-- MODIFIED
    # shuffle=False on loader because the sampler handles shuffling
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, sampler=train_sampler # <-- MODIFIED
    )

    valid_dataset = CifarDataset(x_valid, y_valid, transform=preprocess)
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False) # <-- MODIFIED
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, sampler=valid_sampler # <-- MODIFIED
    )
    
    optimizer = torch.optim.SGD(small_model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    criterion_ce = torch.nn.CrossEntropyLoss()
    mse_point = torch.nn.MSELoss()
    mse_grad = torch.nn.MSELoss()
    kd = torch.nn.KLDivLoss(reduction="batchmean")


    def log_epoch_jsonl(epoch, training_loss, validation_loss, train_avg_entropy):
        # --- LOGGING UPDATED ---
        if rank == 0: # <-- MODIFIED: Only log from the main process
            path="logs/metrics.jsonl"
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            rec = {
                "epoch": {
                    "id": int(epoch),
                    "training_loss": float(training_loss),
                    "validation_loss": float(validation_loss),
                    "train_avg_entropy": float(train_avg_entropy)
                }
            }
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    for epoch in range(int(epochs)):
        # --- SAMPLER EPOCH UPDATE ---
        train_sampler.set_epoch(epoch) # <-- MODIFIED: Ensure proper shuffling
        
        sum_loss = 0.0
        num_batches = 0 
        entropy_sum = 0.0
        
        for i, (inputs, labels_full) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels_full = labels_full.to(device)
            
            optimizer.zero_grad()           
            small_model.train()
            inputs_nchw = inputs 
            outputs_ce = small_model(inputs_nchw)
            big_model.eval()

            probs = torch.softmax(outputs_ce, dim=1)
            entropy_loss = (probs * torch.log(probs.clamp(min=1e-8))).sum(dim=1).mean()

            ce_loss = criterion_ce(outputs_ce, labels_full.squeeze())
            
            if epoch > 10 and i % 2 == 0:
                with torch.no_grad():
                    outputs_big_model = big_model(inputs_nchw)
                
                # --- JVP LOGIC MODIFIED to use your FSDP-aware util file ---
                jvp_small = []
                jvp_large = []
                
                for _ in range(num_samples_random):
                    v = torch.randn_like(inputs,dtype=torch.float32,)
                    
                    was_training = small_model.training
                    small_model.eval()
                    # We need torch.enable_grad() for the student JVP
                    # to ensure gradients flow for the optimizer step.
                    with torch.enable_grad():
                         # <-- MODIFIED: Using FSDP-aware function
                         _,output_gradients = estimate_gradient(small_model,inputs_nchw,v) 
                    small_model.train(was_training)
                    jvp_small.append(output_gradients)
                    
                    with torch.no_grad():
                        # <-- MODIFIED: Using FSDP-aware function
                        _,output_gradients_big_model = estimate_gradient(big_model,inputs_nchw,v)
                        jvp_large.append(output_gradients_big_model)
                        
                jvp_small = torch.stack(jvp_small)
                jvp_large = torch.stack(jvp_large)
                # --- END JVP MODIFICATION ---

                loss = (
                    ce_loss
                    + distil_weight * (T * T) * kd(
                        torch.log_softmax(outputs_ce / T, dim=1),
                        torch.softmax(outputs_big_model / T, dim=1)
                    )
                    + jvp_weight * mse_grad(jvp_small, jvp_large)
                    + entropy_weight * entropy_loss
                )

                if i % 2 == 0 and rank == 0: # <-- MODIFIED: Only print from rank 0
                    with torch.no_grad():
                        preds = outputs_ce.argmax(dim=1)
                        y_flat = labels_full.view(-1)
                        correct = (preds == y_flat).sum().item()
                        n = y_flat.numel()
                        batch_acc = correct / n
                    print(
                        f"Combined Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], "
                        f"Loss: {ce_loss.item():.4f},Acc: {correct}/{n} ({batch_acc:.3f})"
                    )

            else:
                loss = ce_loss + entropy_weight * entropy_loss
                if (i + 1) % 10 == 0 and rank == 0: # <-- MODIFIED: Only print from rank 0
                    with torch.no_grad():
                        preds = outputs_ce.argmax(dim=1)
                        y_flat = labels_full.view(-1)
                        correct = (preds == y_flat).sum().item()
                        n = y_flat.numel()
                        batch_acc = correct / n
                    print(f"Combined Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {ce_loss.item():.4f},Acc: {correct}/{n} ({batch_acc:.3f})")
            
            loss.backward()
            optimizer.step()

            sum_loss += float(loss.item())
            entropy_sum += float(entropy_loss.item())
            num_batches += 1

        
        # --- AGGREGATE TRAINING LOSS & ENTROPY ---
        sum_loss_tensor = torch.tensor(sum_loss).to(device)
        entropy_sum_tensor = torch.tensor(entropy_sum).to(device)
        num_batches_tensor = torch.tensor(num_batches).to(device)
        
        dist.all_reduce(sum_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(entropy_sum_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_batches_tensor, op=dist.ReduceOp.SUM)
        
        # Get the global average
        training_loss = sum_loss_tensor.item() / max(1, num_batches_tensor.item())
        avg_train_entropy = entropy_sum_tensor.item() / max(1, num_batches_tensor.item())

        # --- VALIDATION PHASE ---
        small_model.eval()
        total_valid_loss = 0.0
        num_valid_batches = 0
        with torch.no_grad():
            for inputs, labels_full in valid_loader:
                inputs = inputs.to(device)
                labels_full = labels_full.to(device)
                inputs_nchw = inputs
                outputs = small_model(inputs_nchw)
                
                valid_loss = criterion_ce(outputs, labels_full.squeeze())
                total_valid_loss += valid_loss.item()
                num_valid_batches += 1
        
        # --- AGGREGATE VALIDATION LOSS ---
        total_valid_loss_tensor = torch.tensor(total_valid_loss).to(device)
        num_valid_batches_tensor = torch.tensor(num_valid_batches).to(device)

        dist.all_reduce(total_valid_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_valid_batches_tensor, op=dist.ReduceOp.SUM)

        # Get the global average
        avg_valid_loss = total_valid_loss_tensor.item() / max(1, num_valid_batches_tensor.item())

        # --- LOGGING / PRINTING UPDATED ---
        if rank == 0: # <-- MODIFIED: Only log and print from rank 0
            log_epoch_jsonl(epoch + 1, training_loss, avg_valid_loss, avg_train_entropy)
            print(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {avg_valid_loss:.4f}, Average Entropy: {float(avg_train_entropy)}")

    return small_model