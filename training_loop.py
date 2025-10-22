import torch 
from taylor_sobolev_utils import estimate_gradient
from PIL import Image
from torch.utils.data import Dataset
class CifarDataset(Dataset):
    def __init__(self, data, targets, transform):
        self.data = data
        self.targets = torch.tensor(targets, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        # The transforms expect a PIL Image, so we convert the numpy array
        img = Image.fromarray(img)
        img = self.transform(img)

        return img, target

def train(big_model,small_model,x_train,y_train,x_valid,y_valid, preprocess, batch_size=16,epochs=10,lr=1e-3, jvp_weight = 0.1, distil_weight = 0.5, T = 2.0):
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Create DataLoader instances
    train_dataset = CifarDataset(x_train, y_train, transform=preprocess)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = CifarDataset(x_valid, y_valid, transform=preprocess)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(small_model.parameters(), lr=lr)
    criterion_ce = torch.nn.CrossEntropyLoss()
    mse_point = torch.nn.MSELoss()
    mse_grad = torch.nn.MSELoss()
    kd = torch.nn.KLDivLoss(reduction="batchmean")

    for epoch in range(int(epochs)):        
        small_model.train()  # Set the combined model to training mode
        
        for i, (inputs, labels_full) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels_full = labels_full.to(device)
            # Zero the parameter gradients
            optimizer.zero_grad()           
            inputs_nchw = inputs # Data is now in correct N,C,H,W format from DataLoader
            outputs_ce = small_model(inputs_nchw)
            # Temporarily set models to eval mode for jvp, which requires a pure function
            small_model.eval()
            outputs = small_model(inputs_nchw)
            big_model.eval()
            v = torch.randn_like(inputs,dtype=torch.float32,)
            _,output_gradients = estimate_gradient(small_model,inputs_nchw,v) # Corrected displacement
            with torch.no_grad():
                outputs_big_model = big_model(inputs_nchw)
                _,output_gradients_big_model = estimate_gradient(big_model,inputs_nchw,v) # Corrected displacement
            # Return small_model to training mode for loss.backward()
            small_model.train()
            ce_loss = criterion_ce(outputs, labels_full.squeeze())
            loss = ce_loss+distil_weight*(T*T)*kd(torch.log_softmax(outputs / T, dim=1),torch.softmax(outputs_big_model / T, dim=1))\
                +jvp_weight*mse_grad(output_gradients,output_gradients_big_model)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Display loss at each step (or every few steps)
            if (i + 1) % 10 == 0: # Print every 100 mini-batches
                print(f"Combined Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {ce_loss.item():.4f}")

        # Validation phase
        small_model.eval()  # Set the small model to evaluation mode
        total_valid_loss = 0.0
        with torch.no_grad(): # No need to track gradients for validation
            for inputs, labels_full in valid_loader:
                # Ensure inputs are on the correct device if not already handled by DataLoader
                inputs = inputs.to(device)
                labels_full = labels_full.to(device)
                inputs_nchw = inputs # Data is now in correct N,C,H,W format
                outputs = small_model(inputs_nchw)
                
                # Calculate validation loss
                valid_loss = criterion_ce(outputs, labels_full.squeeze())
                total_valid_loss += valid_loss.item()
        
        avg_valid_loss = total_valid_loss / len(valid_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {avg_valid_loss:.4f}")

    return small_model
