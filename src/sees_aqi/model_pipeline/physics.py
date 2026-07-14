import torch

def calculate_physics_loss(predictions, inputs, dx=40.0, dt=86400.0, D=0.05):
    """
    Computes the Physics-Informed loss starting from a zero-concentration initial state.
    """
    dy = dx  
    
    # 1. CLAMP ANOMALIES: Neutralize any -9999.0 NoData pixels from the TIFFs
    inputs = torch.nan_to_num(inputs, nan=0.0, posinf=0.0, neginf=0.0)
    inputs = torch.clamp(inputs, min=-1000.0, max=1000.0)
    
    # ====================================================================
    # 2. EXACT 10-CHANNEL EXTRACTIONS (Mapped exactly to your list)
    # ====================================================================
    algae     = inputs[:, 0:1, :, :]   # Ch 0: lake_algae_2025_final
    u_wind_2m = inputs[:, 1:2, :, :]   # Ch 1: processed_2m_u_wind
    v_wind_2m = inputs[:, 2:3, :, :]   # Ch 2: processed_2m_v_wind
    u_wind_10 = inputs[:, 3:4, :, :]   # Ch 3: processed_10m_u_wind
    v_wind_10 = inputs[:, 4:5, :, :]   # Ch 4: processed_10m_v_wind
    elevation = inputs[:, 5:6, :, :]   # Ch 5: processed_elevation
    humidity  = inputs[:, 6:7, :, :]   # Ch 6: processed_humidity
    precip    = inputs[:, 7:8, :, :]   # Ch 7: processed_precipitation
    temp      = inputs[:, 8:9, :, :]   # Ch 8: processed_temperature
    uv        = inputs[:, 9:10, :, :]  # Ch 9: processed_uv_index

    # 3. CONCENTRATION STATES
    c_tomorrow = predictions 
    
    # INITIAL CONDITION: Lake is completely clear at t=0
    c_today_physics = torch.zeros_like(c_tomorrow)
    
    # 4. FORCE LAND TO 0 FOR EMISSION MATH
    algae_physics = torch.clamp(algae, min=0.0)
    
    # 5. WIND SPEED & EMISSIONS
    wind_speed_2m = torch.sqrt(u_wind_2m**2 + v_wind_2m**2 + 1e-8)
    thermal_switch = torch.sigmoid(temp - 20.0)
    uv_trigger = torch.relu(uv)
    
    emissions = 0.005 * wind_speed_2m * algae_physics * thermal_switch * uv_trigger
    
    # 6. SPATIAL DERIVATIVES (Central Differences)
    c_padded = torch.nn.functional.pad(c_tomorrow, (1, 1, 1, 1), mode='replicate')
    
    dC_dx = (c_padded[:, :, 1:-1, 2:] - c_padded[:, :, 1:-1, :-2]) / (2.0 * dx)
    dC_dy = (c_padded[:, :, 2:, 1:-1] - c_padded[:, :, :-2, 1:-1]) / (2.0 * dy)
    
    d2C_dx2 = (c_padded[:, :, 1:-1, 2:] - 2.0 * c_tomorrow + c_padded[:, :, 1:-1, :-2]) / (dx ** 2)
    d2C_dy2 = (c_padded[:, :, 2:, 1:-1] - 2.0 * c_tomorrow + c_padded[:, :, :-2, 1:-1]) / (dy ** 2)
    
    # 6. TIME DERIVATIVE 
    dC_dt = (c_tomorrow - c_today_physics) / dt

    # 7. ADVECTION-DIFFUSION-REACTION RESIDUAL (BALANCED)
    residual = (
        dC_dt 
        + (u_wind_10 * dC_dx + v_wind_10 * dC_dy) 
        - D * (d2C_dx2 + d2C_dy2)                 
        - emissions                               
    )
    
    # 8. GLOBAL MEAN SQUARED ERROR LOSS
    # (You can leave your scaling factor at 1000.0 if you added it previously)
    SCALING_FACTOR = 1000.0
    pde_loss = torch.mean((residual * SCALING_FACTOR) ** 2)

    negative_penalty = torch.mean(torch.relu(-c_tomorrow) ** 2) * 50.0
    
    total_loss = pde_loss + negative_penalty

    return total_loss