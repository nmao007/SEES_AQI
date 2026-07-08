import torch
import torch.optim as optim
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import sees_aqi.data_processing.process_function_library as process
from neuralop.models import FNO

# 1. Configuration & File Paths
TIF_PATH = "data/test_data/test_raw_algae_map/lake_algae_2025.tif"
SPEED_CSV = "data/test_data/test_csvs/Wind Speed At 2Meters (Western Basin Area) - POWER_Regional_Daily_20250628_20260628 copy.csv"
DIR_CSV = "data/test_data/test_csvs/Wind Direction At 2Meters (Western Basin Area) - Sheet1.csv"

STRIDE = 50       # Aggressive downsampling for CPU speed
DX = 10.0 * STRIDE # 500 meters per pixel spatial step
DT = 3600.0       # 1 hour time step

def compute_advection_loss(pred, u, v):
    """Calculates the physical advection residual deviation from 0."""
    # Time derivative (Forward difference)
    dC_dt = (pred[..., 1:] - pred[..., :-1]) / DT
    
    # Spatial derivatives (Central difference)
    dC_dx = (pred[:, :, :, 2:, :-1] - pred[:, :, :, :-2, :-1]) / (2 * DX)
    dC_dy = (pred[:, :, 2:, :, :-1] - pred[:, :, :-2, :, :-1]) / (2 * DX)
    
    # Align dimensions to the interior cropped region
    u_int = u[:, :, 1:-1, 1:-1, :-1]
    v_int = v[:, :, 1:-1, 1:-1, :-1]
    
    residual = dC_dt[:, :, 1:-1, 1:-1, :] + (u_int * dC_dx[:, :, 1:-1, :]) + (v_int * dC_dy[:, :, :, 1:-1])
    return torch.mean(residual ** 2)

def run_cpu_training():
    # 2. Load and downsample data arrays
    print("Loading and downsampling geographical inputs...")
    with rasterio.open(TIF_PATH) as src:
        algae = src.read(1).astype(np.float32)[::STRIDE, ::STRIDE]
        
    u_wind, v_wind = process.map_wind_speed_direction(TIF_PATH, SPEED_CSV, DIR_CSV, target_doy=179)
    u_wind = u_wind[::STRIDE, ::STRIDE]
    v_wind = v_wind[::STRIDE, ::STRIDE]
    
    # 3. Format into a 5D Tensor: [Batch=1, Channels=3, Height, Width, Time=4]
    frame = np.stack([algae, u_wind, v_wind], axis=0)
    inputs = torch.from_numpy(frame).float().unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 1, 4)
    
    # Separate channels for the physics engine
    u_tensor = inputs[:, 1:2, ...]
    v_tensor = inputs[:, 2:3, ...]

    # 4. Initialize FNO Model & Optimizer
    model = FNO(n_modes=(16, 16, 4), in_channels=3, out_channels=1, hidden_channels=32)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    # 5. Quick CPU Training Loop
    print("\nStarting quick baseline physics training...")
    for epoch in range(1, 51):
        model.train()
        optimizer.zero_grad()
        
        predictions = model(inputs)
        loss = compute_advection_loss(predictions, u_tensor, v_tensor)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch}/50] ── Physics Residual Loss: {loss.item():.6f}")
            
    # 6. Plot the newly trained physical baseline result
    print("\nTraining complete! Generating plot...")
    final_output = predictions[0, 0, :, :, -1].detach().numpy()
    
    plt.figure(figsize=(8, 6))
    plt.imshow(final_output, cmap='viridis')
    plt.colorbar(label='Aerosol Plume Concentration')
    plt.title("FNO Output After 50 Epochs of Advection Optimization")
    plt.show()

if __name__ == "__main__":
    run_cpu_training()