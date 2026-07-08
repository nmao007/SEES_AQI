import torch
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import sees_aqi.data_processing.process_function_library as process
from neuralop.models import FNO

# 1. Define your file paths (make sure names match your folders exactly)
TIF_PATH = "data/test_data/test_raw_algae_map/lake_algae_2025.tif"
SPEED_CSV = "data/test_data/test_csvs/Wind Speed At 2Meters (Western Basin Area) - POWER_Regional_Daily_20250628_20260628 copy.csv"    
DIR_CSV = "data/test_data/test_csvs/Wind Direction At 2Meters (Western Basin Area) - Sheet1.csv"  

def run_local_fno_test():
    # 2. Read your true 10m algae imagery array and downsample it
    print("Loading algae map image...")
    with rasterio.open(TIF_PATH) as src:
        # The [::10, ::10] takes every 10th pixel, making the grid 100x smaller
        algae_array = src.read(1).astype(np.float32)[::10, ::10]
    
    # 3. Generate wind maps and downsample them to match
    print("Calculating wind velocity vector maps...")
    u_wind, v_wind = process.map_wind_speed_direction(
        tif_path=TIF_PATH,
        speed_csv_path=SPEED_CSV,
        dir_csv_path=DIR_CSV,
        target_year=2025,
        target_doy=179
    )
    u_wind = u_wind[::10, ::10]
    v_wind = v_wind[::10, ::10]
    
    # 4. Stack variables into a 3-channel frame shape: (3, Height, Width)
    frame = np.stack([algae_array, u_wind, v_wind], axis=0)
    
    # 5. Convert to a PyTorch tensor and force it to 32-bit float
    input_tensor = torch.from_numpy(frame).float() # ◄── ADD .float() HERE
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(-1) 
    input_tensor = input_tensor.repeat(1, 1, 1, 1, 4)            
    
    # 6. Initialize your FNO tuned down to 3 input channels
    print("Spawning 3D Fourier Neural Operator network...")
    model = FNO(
        n_modes=(16, 16, 4), 
        in_channels=3, 
        out_channels=1, 
        hidden_channels=64
    )
    model.eval() # Set model to evaluation/inference mode
    
    # 7. Execute the forward pass
    print("Passing data tensor through the operator...")
    with torch.no_grad():
        prediction_tensor = model(input_tensor)
        
    # 8. Squeeze the final output time-step back into a 2D matrix
    output_matrix = prediction_tensor[0, 0, :, :, -1].numpy()
    print(f"Success! Output matrix generated with shape: {output_matrix.shape}")
    
    # 9. Plot the mock prediction output map
    plt.figure(figsize=(8, 6))
    plt.imshow(output_matrix, cmap='viridis')
    plt.colorbar(label='Aerosol Concentration Index')
    plt.title("FNO Initial Untrained Prediction Output (Day 179)")
    plt.show()

if __name__ == "__main__":
    run_local_fno_test()