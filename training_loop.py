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

def train(big_model,small_model,x_train,y_train,x_valid,y_valid, preprocess, only_data=False,teacher_only_data=True, teacher_jvp=False, batch_size=50,epochs=100,lr=1e-3, weight_decay=5e-4, entropy_weight=0.5, jvp_weight = 0.1, distil_weight = 1.0, T = 1.0, num_samples_random=4,device = None):
    
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


    def log_epoch_jsonl(epoch, training_loss,train_accuracy, validation_loss, validation_accuracy, train_avg_entropy, valid_avg_entropy):
        
        if rank == 0: # <-- MODIFIED: Only log from the main process
            mode = os.environ.get("TRAIN_MODE", "unknown") # <-- NEW: Read the current mode ('data')
            task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0") # <-- NEW: Read the unique run number (1-10)
            
            # Path now includes the mode and array task ID
            log_dir = f"logs/{mode}" # E.g., logs/data
            os.makedirs(log_dir or ".", exist_ok=True) # <-- MODIFIED: Create mode-specific directory
            path=f"{log_dir}/metrics_run_{task_id}.jsonl" # <-- MODIFIED: Unique file name (E.g., metrics_run_1.jsonl)
            
            rec = {
                "epoch": {
                    "id": int(epoch),
                    "training_loss": float(training_loss),
                    "training_accuracy": float(train_accuracy),
                    "validation_loss": float(validation_loss),
                    "validation_accuracy": float(validation_accuracy),
                    "train_avg_entropy": float(train_avg_entropy),
                    "valid_avg_entropy":float(valid_avg_entropy)
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

        # for train accuracy
        sum_train_batch_acc = 0.0
        
        for i, (inputs, labels_full) in enumerate(train_loader):
            correct = 0
            n = 0
            batch_acc = 0.0
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

            with torch.no_grad():
                preds = outputs_ce.argmax(dim=1)
                labels_squeezed = labels_full.squeeze()
                correct = (preds == labels_squeezed).sum().item()
                n = labels_squeezed.numel()
                # Calculate accuracy as a percentage for this batch
                batch_acc = (correct / max(1, n)) * 100 
                sum_train_batch_acc += batch_acc
            

            if not only_data:
                # batch accuracy and adding it to sum
                if epoch > 10 and i % 2 == 0:
                    
                    # --- FIX: Calculate outputs_big_model ONCE outside the checks ---
                    with torch.no_grad():
                        outputs_big_model = big_model(inputs_nchw)
                    # --- END FIX ---

                    if teacher_only_data:
                        # 'outputs_big_model' is already calculated
                        loss = (
                            ce_loss
                            + distil_weight * (T * T) * kd(
                                torch.log_softmax(outputs_ce / T, dim=1),
                                torch.softmax(outputs_big_model / T, dim=1)
                            )
                            + entropy_weight * entropy_loss
                        )
                    if teacher_jvp:
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

                        # This loss calculation now correctly uses outputs_big_model
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
                            pass
                        print(
                            f"Combined Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], "
                            f"Loss: {ce_loss.item():.4f},Acc: {correct}/{n} ({batch_acc/100:.3f})" # divide by 100 for non-percent
                        )

                else:
                    loss = ce_loss + entropy_weight * entropy_loss
                    if (i + 1) % 10 == 0 and rank == 0: # <-- MODIFIED: Only print from rank 0
                        with torch.no_grad():
                            pass
                        print(f"Combined Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {ce_loss.item():.4f},Acc: {correct}/{n} ({batch_acc/100:.3f})") # divide by 100 for non-percent
            else:
                loss = ce_loss + entropy_weight * entropy_loss
                if (i + 1) % 10 == 0 and rank == 0: # <-- MODIFIED: Only print from rank 0
                    with torch.no_grad():
                        pass
                    print(f"Combined Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {ce_loss.item():.4f},Acc: {correct}/{n} ({batch_acc/100:.3f})") # divide by 100 for non-percent
                
            loss.backward()
            optimizer.step()

            sum_loss += float(ce_loss.item())
            entropy_sum += float(entropy_loss.item())
            num_batches += 1

        
        # --- AGGREGATE TRAINING LOSS & ENTROPY ---
        sum_loss_tensor = torch.tensor(sum_loss).to(device)
        entropy_sum_tensor = torch.tensor(entropy_sum).to(device)
        num_batches_tensor = torch.tensor(num_batches).to(device)

        sum_train_batch_acc_tensor = torch.tensor(sum_train_batch_acc).to(device)
        
        dist.all_reduce(sum_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(entropy_sum_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_batches_tensor, op=dist.ReduceOp.SUM)

        # Aggregate batch acc sum
        dist.all_reduce(sum_train_batch_acc_tensor, op=dist.ReduceOp.SUM)
        
        # Get the global average
        training_loss = sum_loss_tensor.item() / max(1, num_batches_tensor.item())
        avg_train_entropy = entropy_sum_tensor.item() / max(1, num_batches_tensor.item())
        # average of batch accuracies
        avg_train_acc = sum_train_batch_acc_tensor.item() / max(1, num_batches_tensor.item())


        # --- VALIDATION PHASE ---
        small_model.eval()
        total_valid_loss = 0.0
        num_valid_batches = 0
        total_valid_entropy =0.0
        # validation accuracy 
        sum_valid_batch_acc = 0.0
        with torch.no_grad():
            for inputs, labels_full in valid_loader:
                inputs = inputs.to(device)
                labels_full = labels_full.to(device)
                inputs_nchw = inputs
                outputs = small_model(inputs_nchw)
                
                valid_loss = criterion_ce(outputs, labels_full.squeeze())
                total_valid_loss += valid_loss.item()
                # Calculate Batch Entropy
                probs = torch.softmax(outputs, dim=1)
                entropy_val_batch = (probs * torch.log(probs.clamp(min=1e-8))).sum(dim=1).mean()
                total_valid_entropy += entropy_val_batch.item()

                num_valid_batches += 1

                #  batch accuracy and add to sum
                preds = outputs.argmax(dim=1)
                labels_squeezed = labels_full.squeeze()
                correct = (preds == labels_squeezed).sum().item()
                n = labels_squeezed.numel()
                # Calculate accuracy as a percentage for this batch
                batch_acc = (correct / max(1, n)) * 100
                sum_valid_batch_acc += batch_acc
        
        # --- AGGREGATE VALIDATION LOSS 
        total_valid_loss_tensor = torch.tensor(total_valid_loss).to(device)
        total_valid_entropy_tensor = torch.tensor(total_valid_entropy).to(device)
        num_valid_batches_tensor = torch.tensor(num_valid_batches).to(device)
        # tensor for batch acc sum 
        sum_valid_batch_acc_tensor = torch.tensor(sum_valid_batch_acc).to(device)

        dist.all_reduce(total_valid_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_valid_batches_tensor, op=dist.ReduceOp.SUM)
        # Aggregate batch acc sum 
        dist.all_reduce(sum_valid_batch_acc_tensor, op=dist.ReduceOp.SUM)

        # Get the global average
        avg_valid_loss = total_valid_loss_tensor.item() / max(1, num_valid_batches_tensor.item())
        avg_valid_entropy = total_valid_entropy_tensor.item() / max(1, num_valid_batches_tensor.item())
        # Calculate average of batch accuracies 
        avg_valid_acc = sum_valid_batch_acc_tensor.item() / max(1, num_valid_batches_tensor.item())


        # --- LOGGING / PRINTING UPDATED ---
        if rank == 0: # <-- MODIFIED: Only log and print from rank 0
            log_epoch_jsonl(epoch + 1, training_loss, avg_train_acc, avg_valid_loss, avg_valid_acc, avg_train_entropy, avg_valid_entropy)
            
            print(
                f"Epoch [{epoch+1}/{epochs}], "
                f"Train Acc: {avg_train_acc:.2f}%, "
                f"Validation Loss: {avg_valid_loss:.4f}, "
                f"Validation Acc: {avg_valid_acc:.2f}%, "
                f"Average Entropy: {float(avg_train_entropy)}, "
                f"Val Entropy: {float(avg_valid_entropy):.4f}"
            )
            
    return small_model