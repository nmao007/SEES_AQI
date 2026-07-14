import torch

def calculate_physics_loss(predictions, inputs, dx=40.0, dt=86400.0, D=0.05):
    """
    Computes the Physics-Informed loss over the ENTIRE grid (both land and water)
    using the exact TIFF loading order.
    
    inputs shape: [batch_size, 11, height, width]
    predictions shape: [batch_size, 1, height, width]
    """
    dy = dx  # Assume square grid pixels (40m x 40m)
    
    # 1. CORRECT CHANNEL EXTRACTIONS (Matches your layer_files + target_today)
    c_today   = inputs[:, 0:1, :, :]   # Ch 0: target_today (t0 concentration)
    algae     = inputs[:, 1:2, :, :]   # Ch 1: lake_algae_2025_final (Land = -2)
    u_wind_2m = inputs[:, 2:3, :, :]   # Ch 2: processed_2m_u_wind
    v_wind_2m = inputs[:, 3:4, :, :]   # Ch 3: processed_2m_v_wind
    u_wind_10 = inputs[:, 4:5, :, :]   # Ch 4: processed_10m_u_wind
    v_wind_10 = inputs[:, 5:6, :, :]   # Ch 5: processed_10m_v_wind
    # Channels 6, 7 & 8 are extracted but unused in this version
    elevation = inputs[:, 6:7, :, :]   # Ch 6: processed_elevation
    humidity  = inputs[:, 7:8, :, :]   # Ch 7: processed_humidity
    precip    = inputs[:, 8:9, :, :]   # Ch 8: processed_precipitation
    temp      = inputs[:, 9:10, :, :]  # Ch 9: processed_temperature
    uv        = inputs[:, 10:11, :, :] # Ch 10: processed_uv_index

    # Predicted Concentration at t1 (tomorrow)
    c_tomorrow = predictions  # Shape [batch_size, 1, height, width]
    
    # 2. Force Land (-2) to 0 so biological emissions are strictly 0 on land
    algae_physics = torch.clamp(algae, min=0.0)
    
    # 3. Calculate 2m Wind Speed Magnitude on the fly for Wave-Shearing Emissions
    wind_speed_2m = torch.sqrt(u_wind_2m**2 + v_wind_2m**2 + 1e-8)
    
    # 4. EMISSIONS (S(x,y) is 0 on land because algae_physics is clamped to 0)
    thermal_switch = torch.sigmoid(temp - 20.0)
    uv_trigger = torch.relu(uv)
    
    emissions = 0.005 * wind_speed_2m * algae_physics * thermal_switch * uv_trigger
    
    # 5. SPATIAL DERIVATIVES (Central Differences)
    # Pad boundaries to preserve spatial dimensions
    c_padded = torch.nn.functional.pad(c_tomorrow, (1, 1, 1, 1), mode='replicate')
    
    dC_dx = (c_padded[:, :, 1:-1, 2:] - c_padded[:, :, 1:-1, :-2]) / (2.0 * dx)
    dC_dy = (c_padded[:, :, 2:, 1:-1] - c_padded[:, :, :-2, 1:-1]) / (2.0 * dy)
    
    d2C_dx2 = (c_padded[:, :, 1:-1, 2:] - 2.0 * c_tomorrow + c_padded[:, :, 1:-1, :-2]) / (dx ** 2)
    d2C_dy2 = (c_padded[:, :, 2:, 1:-1] - 2.0 * c_tomorrow + c_padded[:, :, :-2, 1:-1]) / (dy ** 2)
    
    # 6. TIME DERIVATIVE (Temporal evolution over 24 hours)
    dC_dt = (c_tomorrow - c_today) / dt
    
    # 7. ADVECTION-DIFFUSION-REACTION RESIDUAL
    # Evaluated over the entire map. Even on land where emissions=0, the model must
    # satisfy: dC/dt + u*dC/dx + v*dC/dy - D*Laplacian(C) = 0.
    # This evaluates how the 10m winds advected/dispersed the 24-hour-prior plume across the land.
    residual = (
        dC_dt 
        + (u_wind_10 * dC_dx + v_wind_10 * dC_dy) # Wind transport (10m)
        - D * (d2C_dx2 + d2C_dy2)                 # Turbulent diffusion
        - emissions                               # Toxin source generation
    )
    
    # 8. GLOBAL MEAN SQUARED ERROR LOSS (Evaluated over all pixels)
    pde_loss = torch.mean(residual ** 2)

    # 9. PHYSICAL SAFEGUARD: Non-Negativity Penalty
    # Prevents the model from cheating the PDE with negative concentration values
    non_neg_penalty = torch.mean(torch.relu(-c_tomorrow) ** 2)
    
    total_loss = pde_loss + (10.0 * non_neg_penalty)
    
    return total_loss