import torch 
from taylor_sobolev_utils import estimate_gradient

def train(big_model,small_model,x_train_tensor,y_train,x_valid_tensor,y_valid,batch_size,epochs,lr):
    
    # Create DataLoader instances
    train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = torch.utils.data.TensorDataset(x_valid_tensor, y_valid)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(small_model.parameters(), lr=lr)
    criterion_ce = torch.nn.CrossEntropyLoss()
    mse_point = torch.nn.MSELoss()
    mse_grad = torch.nn.MSELoss()
    for epoch in range(epochs):
        
        small_model.train()  # Set the combined model to training mode
        
        for i, (inputs, labels_full) in enumerate(train_loader):
            # Zero the parameter gradients
            optimizer.zero_grad()           
            outputs = small_model(inputs)
            v = torch.randn_like(inputs,dtype=torch.float32,)
            _,output_gradients = estimate_gradient(small_model,inputs,inputs+v)
            with torch.no_grad():
                outputs_big_model = big_model(inputs)
                _,output_gradients_big_model = estimate_gradient(big_model,inputs,inputs+v)
            loss = criterion_ce(outputs, labels_full)+mse_point(outputs,outputs_big_model)+mse_grad(output_gradients,output_gradients_big_model)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Display loss at each step (or every few steps)
            if (i + 1) % 100 == 0: # Print every 100 mini-batches
                print(f"Combined Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # Validation phase
        small_model.eval()  # Set the small model to evaluation mode
        total_valid_loss = 0.0
        with torch.no_grad(): # No need to track gradients for validation
            for inputs, labels_full in valid_loader:
                # Ensure inputs are on the correct device if not already handled by DataLoader
                # (assuming DataLoader handles device placement or inputs are already on device)
                
                outputs = small_model(inputs)
                
                # Generate a random displacement for gradient estimation
                # This 'v' should ideally be consistent for a given input across models for a fair comparison
                # but for validation, regenerating per batch is acceptable.
                v = torch.randn_like(inputs, dtype=torch.float32)
                
                # Estimate gradients for the small model
                _, output_gradients = estimate_gradient(small_model, inputs, inputs + v)
                
                # Estimate outputs and gradients for the big model (no_grad already active)
                outputs_big_model = big_model(inputs)
                _, output_gradients_big_model = estimate_gradient(big_model, inputs, inputs + v)
                
                # Calculate validation loss
                valid_loss = criterion_ce(outputs, labels_full) + mse_point(outputs, outputs_big_model) + mse_grad(output_gradients, output_gradients_big_model)
                total_valid_loss += valid_loss.item()
        
        avg_valid_loss = total_valid_loss / len(valid_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {avg_valid_loss:.4f}")

    return small_model
