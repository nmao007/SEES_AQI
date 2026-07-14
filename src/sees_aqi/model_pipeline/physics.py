import torch

def calculate_physics_loss(predictions, inputs, dx=40.0, dt=86400.0, D=0.05):
    """
    Computes the Physics-Informed loss over the ENTIRE grid (both land and water),
    allowing the model to learn how wind transports the plume over populated land.
    
    inputs shape: [batch_size, 9, height, width]
    predictions shape: [batch_size, 1, height, width]
    """
    # 1. CORRECT CHANNEL EXTRACTIONS
    c_today   = inputs[:, 0:1, :, :]  # Channel 0: Toxin concentration at t0
    u_wind_10 = inputs[:, 1:2, :, :]  # Channel 1: 10m Wind U (for Advection)
    v_wind_10 = inputs[:, 2:3, :, :]  # Channel 2: 10m Wind V (for Advection)
    temp      = inputs[:, 3:4, :, :]  # Channel 3: Temperature
    uv        = inputs[:, 4:5, :, :]  # Channel 4: UV Index
    # Channels 5 & 6 (Precip & Humidity) are extracted but unused in this version
    precip    = inputs[:, 5:6, :, :]  
    humidity  = inputs[:, 6:7, :, :]  
    wind_2m   = inputs[:, 7:8, :, :]  # Channel 7: 2m Surface Wind (for Emissions)
    algae     = inputs[:, 8:9, :, :]  # Channel 8: Sentinel-2 Algae (Land = -2)
    
    # 2. Predicted Concentration at t1
    c_tomorrow = predictions  # Shape [batch_size, 1, height, width]
    
    # 3. Force Land (-2) to 0 so biological emissions are strictly 0 on land
    algae_physics = torch.clamp(algae, min=0.0)
    
    # 4. EMISSIONS (Calculated using 2m wind speed)
    thermal_switch = torch.sigmoid(temp - 20.0)
    uv_trigger = torch.relu(uv)
    
    # S(x, y) is automatically 0 on land because algae_physics is 0 on land
    S = 0.005 * wind_2m * algae_physics * thermal_switch * uv_trigger
    
    # 5. SPATIAL DERIVATIVES (Central Differences)
    # Pad to maintain spatial resolution at the grid boundaries
    c_padded = torch.nn.functional.pad(c_tomorrow, (1, 1, 1, 1), mode='replicate')
    
    # Row axis = y, Col axis = x
    dC_dx = (c_padded[:, :, 1:-1, 2:] - c_padded[:, :, 1:-1, :-2]) / (2.0 * dx)
    dC_dy = (c_padded[:, :, 2:, 1:-1] - c_padded[:, :, :-2, 1:-1]) / (2.0 * dx)
    
    d2C_dx2 = (c_padded[:, :, 1:-1, 2:] - 2.0 * c_tomorrow + c_padded[:, :, 1:-1, :-2]) / (dx ** 2)
    d2C_dy2 = (c_padded[:, :, 2:, 1:-1] - 2.0 * c_tomorrow + c_padded[:, :, :-2, 1:-1]) / (dx ** 2)
    
    # 6. TIME DERIVATIVE
    dC_dt = (c_tomorrow - c_today) / dt
    
    # 7. ADVECTION-DIFFUSION-REACTION RESIDUAL
    # Evaluated everywhere. S will be 0 on land, leaving a pure transport equation.
    residual = (
        dC_dt 
        + (u_wind_10 * dC_dx + v_wind_10 * dC_dy) # Advected by high-altitude 10m wind
        - D * (d2C_dx2 + d2C_dy2)                 # Dispersed by turbulent diffusion
        - S                                       # Generated only over the lake surface
    )
    
    # 8. GLOBAL MEAN SQUARED ERROR LOSS
    # No water mask multiplication! We average the physical error over the entire 2D space.
    loss = torch.mean(residual ** 2)
    
    return loss