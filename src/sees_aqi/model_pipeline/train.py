import os
import torch
import torch.optim as optim

# 1. Import the production-ready FNO engine from the official library
from neuralop.models import FNO

# 2. Import your custom data loading and physics equation blocks
from load_data import load_processed_data
from physics import compute_comprehensive_physics_loss

def train_pipeline(epochs=30, lr=0.002):
    # Setup Lightning AI GPU acceleration if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training cluster initialized on hardware accelerator: {device}")
    
    # Load your multi-channel spatial environmental tensor
    # Returns Shape: [Batch=1, Channels=9, Height, Width, Time=4]
    inputs = load_processed_data(data_dir="data/final_maps", num_time_steps=4)
    inputs = inputs.to(device)
    
    # Initialize the official library FNO model
    # n_modes maps to your 3D dimensions: (Height modes, Width modes, Time modes)
    model = FNO(
        n_modes=(4, 4, 2), 
        hidden_channels=20, # Matches our previous 'width' representation
        in_channels=9,      # Your 9 environmental raster channels
        out_channels=1      # 1 target channel (Algae concentration output)
    ).to(device)
    
    # Initialize optimization parameter weights
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    print("\nStarting Physics-Informed Neural Operator Training Loop (Powered by neuralop)...")
    model.train()
    
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        
        # Pass the 9 environmental layers through the library's FNO network
        # Output shape: [Batch=1, Channels=1, Height, Width, Time=4]
        predictions = model(inputs)
        
        # Enforce physical constraints across all layers via your custom physics engine
        physics_loss = compute_comprehensive_physics_loss(predictions, inputs)
        
        # Backpropagation step
        physics_loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Log performance status metrics
        if epoch == 1 or epoch % 5 == 0:
            print(f"Epoch [{epoch:02d}/{epochs}] | "
                  f"Total Physics Residual Loss: {physics_loss.item():.6f} | "
                  f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")
            
    # Save optimized network parameters to disk
    os.makedirs("models", exist_ok=True)
    save_path = "models/fno_algae_physics_model.pt"
    torch.save(model.state_dict(), save_path)
    print(f"\nOptimization complete! Model weights cleanly stored to: {save_path}")

if __name__ == "__main__":
    train_pipeline(epochs=30, lr=0.002)