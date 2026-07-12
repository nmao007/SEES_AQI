import os
import rasterio
import numpy as np
import torch

def load_processed_data(data_dir="data/processed_maps", num_time_steps=4):
    """
    Loads 10 processed raster maps (including satellite biomass) 
    and stacks them into an FNO space-time tensor.
    """
    layer_files = [
        "processed_2m_u_wind.tif", "processed_2m_v_wind.tif",
        "processed_10m_u_wind.tif", "processed_10m_v_wind.tif",
        "processed_elevation.tif", "processed_humidity.tif",
        "processed_precipitation.tif", "processed_temperature.tif",
        "processed_uv_index.tif",
        "processed_algae_biomass.tif"  # <-- Added as Channel Index 9
    ]
    
    loaded_channels = []
    for filename in layer_files:
        full_path = os.path.join(data_dir, filename)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Missing required pipeline layer: '{full_path}'")
            
        with rasterio.open(full_path) as src:
            channel_data = src.read(1).astype(np.float32)
            channel_data = np.nan_to_num(channel_data, nan=0.0, posinf=0.0, neginf=0.0)
            loaded_channels.append(channel_data)
            
    # Stack to [10, H, W], add Batch -> [1, 10, H, W], repeat across Time -> [1, 10, H, W, T]
    tensor_grid = torch.from_numpy(np.stack(loaded_channels, axis=0)).float().unsqueeze(0)
    return tensor_grid.unsqueeze(-1).repeat(1, 1, 1, 1, num_time_steps)