import os
import torch
import torch.optim as optim
from neuralop.models import FNO

# Import your modular project functions
from load_data import load_processed_data
from physics import compute_advection_loss

# --- Configuration Constants ---
PROCESSED_DATA_DIR = "data/processed_data"
MODEL_SAVE_PATH = "model_pipeline/fno_advection_full_res.pth"

EPOCHS = 100
LEARNING_RATE = 0.001

def main():
    # Detect GPU hardware (Lightning AI A100 instance)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using execution device: {device}")

    # 1. Load Data
    inputs = load_processed_data(PROCESSED_DATA_DIR).to(device)
    
    # Isolate wind velocity tracks to pass cleanly to the physics engine
    u_tensor = inputs[:, 1:2, ...]
    v_tensor = inputs[:, 2:3, ...]

    # 2. Model Initialization (3D FNO for Space-Time tracking)
    print("Spawning 3D Fourier Neural Operator structure...")
    model = FNO(
        n_modes=(16, 16, 4), 
        in_channels=3, 
        out_channels=1, 
        hidden_channels=64
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Automatic Mixed Precision scaling to protect memory bandwidth
    scaler = torch.cuda.amp.GradScaler()

    # 3. Core Training Loop
    print(f"Beginning training loop ({EPOCHS} Epochs)...")
    model.train()
    
    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()
        
        # Forward pass in 16-bit float mode
        with torch.cuda.amp.autocast():
            predictions = model(inputs)
            loss = compute_advection_loss(predictions, u_tensor, v_tensor)
            
        # Backward pass using scaled gradients
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch:03d}/{EPOCHS}] ── Physical Residual Loss: {loss.item():.7f}")

    # 4. Save Model Weights
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Training finalized. State weights saved to: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()