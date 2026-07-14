import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

# 1. Import the production-ready FNO engine from the official library
from neuralop.models import FNO

# 2. Import your custom data loading and physics equation blocks
from load_data import load_processed_data
from physics import calculate_physics_loss

def train_pipeline(epochs=500, lr=0.002):
    # Setup Lightning AI GPU acceleration if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training cluster initialized on hardware accelerator: {device}")
    
    # Load your multi-channel spatial environmental tensor
    # Returns Shape: [Batch=1, Channels=9, Height, Width, Time=4]
    inputs = load_processed_data(data_dir="data/final_maps")
    inputs = inputs.to(device)
    
    # Initialize the official library FNO model
    model = FNO(
        n_modes=(16, 16),    
        hidden_channels=24,  # Boosted capacity for sharper spatial boundaries
        in_channels=10,      # Your 10 clean raster maps
        out_channels=1       # Tomorrow's forecasted toxin concentration map
    ).to(device)
    
    # Initialize optimization parameter weights
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    print("\nStarting Physics-Informed Neural Operator Training Loop (Powered by neuralop)...")
    model.train()
    
    loss_history = []
    print("Starting Training Loop...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        raw_predictions = model(inputs)
        predictions = torch.nn.functional.softplus(raw_predictions)
        
        # Calculate your physics loss
        loss = calculate_physics_loss(predictions, inputs) 
        
        loss.backward()
        optimizer.step()
        
        # 2. Record the loss value
        loss_item = loss.item()
        loss_history.append(loss_item)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {loss_item:.6f}")

    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Physics Residual Loss', color='crimson')
    plt.yscale('log') # Log scale helps see small changes clearly
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Log Scale)')
    plt.title('PINO Training Convergence')
    plt.grid(True, which="both", ls="--")
    plt.legend()

    plt.savefig('training_loss_plot.png', dpi=300, bbox_inches='tight')
    print("✨ Loss graph successfully saved as 'training_loss_plot.png'!")

    # Save optimized network parameters to disk
    os.makedirs("models", exist_ok=True)
    save_path = "models/fno_algae_physics_model.pt"
    torch.save(model.state_dict(), save_path)
    print(f"\nOptimization complete! Model weights cleanly stored to: {save_path}")

if __name__ == "__main__":
    train_pipeline(epochs=300, lr=0.002)