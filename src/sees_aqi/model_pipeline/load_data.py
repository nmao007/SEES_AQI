import os
import rasterio
import numpy as np
import torch

def load_processed_data(data_dir):
    """
    Loads separate processed raster layers from disk and aggregates them.
    
    Args:
        data_dir (str): Path to the folder containing processed tifs.
    Returns:
        inputs (Tensor): Shapes [1, 3, Height, Width, 4] containing
                         [Algae, U_wind, V_wind] duplicated across 4 time steps.
    """
    print(f"Loading processed 10m layers from: {data_dir}")
    
    # Define file paths
    algae_path = os.path.join(data_dir, "processed_algae.tif")
    u_wind_path = os.path.join(data_dir, "processed_u_wind.tif")
    v_wind_path = os.path.join(data_dir, "processed_v_wind.tif")
    
    # Read the individual raster channels
    with rasterio.open(algae_path) as src:
        algae = src.read(1).astype(np.float32)
    with rasterio.open(u_wind_path) as src:
        u_wind = src.read(1).astype(np.float32)
    with rasterio.open(v_wind_path) as src:
        v_wind = src.read(1).astype(np.float32)
        
    # Stack channels along the first axis: shape becomes [3, Height, Width]
    frame = np.stack([algae, u_wind, v_wind], axis=0)
    
    # Convert to torch tensor, add Batch dim, and project across Time dimension (T=4)
    # Target shape: [Batch=1, Channels=3, Height, Width, Time=4]
    inputs = torch.from_numpy(frame).float().unsqueeze(0).repeat(1, 1, 1, 1, 4)
    return inputs