import os
import rasterio
import numpy as np
import torch

def load_processed_data(data_dir="data/final_maps"):
    """
    Loads 10 processed raster maps, automatically downsamples them 4x
    to prevent VRAM crashes, and formats them for the 2D FNO.
    """
    layer_files = [
        "processed_2m_u_wind.tif", "processed_2m_v_wind.tif",
        "processed_10m_u_wind.tif", "processed_10m_v_wind.tif",
        "processed_elevation.tif", "processed_humidity.tif",
        "processed_precipitation.tif", "processed_temperature.tif",
        "processed_uv_index.tif",
        "lake_algae_2025_full.tif"  
    ]
    
    loaded_channels = []
    for filename in layer_files:
        full_path = os.path.join(data_dir, filename)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Missing required pipeline layer: '{full_path}'")
            
        with rasterio.open(full_path) as src:
            channel_data = src.read(1).astype(np.float32)
            channel_data = np.nan_to_num(channel_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Downsample 4x by taking every 4th pixel
            # Drops grid from (4809, 6406) down to (1203, 1602)
            channel_data = channel_data[::4, ::4]
            
            loaded_channels.append(channel_data)
            
    # Stack channels to [10, H, W] and add Batch dimension -> [1, 10, H, W]
    tensor_grid = torch.from_numpy(np.stack(loaded_channels, axis=0)).float().unsqueeze(0)
    return tensor_grid