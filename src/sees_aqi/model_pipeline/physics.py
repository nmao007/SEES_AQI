import torch

# --- Physical and Temporal Configuration Scale ---
DX = 10.0         # True 10-meter pixel grid spacing
DT = 3600.0       # 1-hour temporal model step spacing

def compute_comprehensive_physics_loss(pred, inputs):
    """
    Computes PINO residual loss where real satellite biomass, temperature, and 
    wind speed dynamically calculate airborne toxin emission levels.
    
    Args:
        pred (Tensor): Predicted Airborne Toxin Concentration [B, 1, Height, Width, Time]
        inputs (Tensor): 10 Environmental Input Channels [B, 10, Height, Width, Time]
    """
    # 1. Isolate all 10 environmental input layers
    u_2m    = inputs[:, 0:1, ...]
    v_2m    = inputs[:, 1:2, ...]
    u_10m   = inputs[:, 2:3, ...]
    v_10m   = inputs[:, 3:4, ...]
    elev    = inputs[:, 4:5, ...]
    humid   = inputs[:, 5:6, ...]
    precip  = inputs[:, 6:7, ...]
    temp    = inputs[:, 7:8, ...]
    uv      = inputs[:, 8:9, ...]
    biomass = inputs[:, 9:10, ...] # <-- Real satellite biomass input channel

    # 2. Compute Spatiotemporal Derivatives of the Predicted Airborne Toxin (C)
    dC_dt = (pred[..., 1:] - pred[..., :-1]) / DT
    dC_dx = (pred[:, :, 2:, 1:-1, :-1] - pred[:, :, :-2, 1:-1, :-1]) / (2 * DX)
    dC_dy = (pred[:, :, 1:-1, 2:, :-1] - pred[:, :, 1:-1, :-2, :-1]) / (2 * DX)
    
    d2C_dx2 = (pred[:, :, 2:, 1:-1, :-1] - 2 * pred[:, :, 1:-1, 1:-1, :-1] + pred[:, :, :-2, 1:-1, :-1]) / (DX ** 2)
    d2C_dy2 = (pred[:, :, 1:-1, 2:, :-1] - 2 * pred[:, :, 1:-1, 1:-1, :-1] + pred[:, :, 1:-1, :-2, :-1]) / (DX ** 2)

    # 3. Slice Input Layers to match the spatial-temporal interior grid
    u_10m_int   = u_10m[:, :, 1:-1, 1:-1, :-1]
    v_10m_int   = v_10m[:, :, 1:-1, 1:-1, :-1]
    u_2m_int    = u_2m[:, :, 1:-1, 1:-1, :-1]
    v_2m_int    = v_2m[:, :, 1:-1, 1:-1, :-1]
    elev_int    = elev[:, :, 1:-1, 1:-1, :-1]
    temp_int    = temp[:, :, 1:-1, 1:-1, :-1]
    uv_int      = uv[:, :, 1:-1, 1:-1, :-1]
    biomass_int = biomass[:, :, 1:-1, 1:-1, :-1]

    # 4. Calculate Dynamic Turbulent Diffusion (D) via Wind Shear
    wind_speed_10m = torch.sqrt(u_10m_int**2 + v_10m_int**2 + 1e-6)
    wind_speed_2m  = torch.sqrt(u_2m_int**2 + v_2m_int**2 + 1e-6)
    wind_shear     = torch.abs(wind_speed_10m - wind_speed_2m)
    D = 0.01 + 0.05 * wind_shear

    # =========================================================================
    # 5. BIOLOGICAL & MECHANICAL EMISSION COUPLING (The Core Update)
    # =========================================================================
    
    # Step A: Biological Toxin Availability 
    # Toxin synthesis within the biomass scales with temperature and UV exposure.
    # We use a sigmoid trigger around 20°C to simulate exponential toxic cell activation.
    biological_toxin_potential = biomass_int * torch.sigmoid(temp_int - 20.0) * torch.relu(uv_int)
    
    # Step B: Mechanical Aerosolization Pump
    # High wind speeds rip across the lake surface, aerosolizing water droplets.
    # Toxin emission level directly equals (Available Toxin) * (Wind Speed Mechanical Force)
    emission_source_term = 0.005 * wind_speed_10m * biological_toxin_potential

    # =========================================================================
    # 6. Assemble the Airborne Transport PDE Residual
    # =========================================================================
    # Equation: dC/dt + Advection - Diffusion = Emission_Source
    advection_term = (u_10m_int * dC_dx) + (v_10m_int * dC_dy)
    diffusion_term = D * (d2C_dx2 + d2C_dy2)
    
    # The PDE residual should be 0 if the model correctly maps emission to transport
    residual = dC_dt + advection_term - diffusion_term - emission_source_term

    # 7. Apply Topography Masking Weight (Enforce physics only over water pixels)
    water_mask = (elev_int != -2.0).float()
    squared_residual = (residual * water_mask) ** 2
    
    loss = torch.sum(squared_residual) / (torch.sum(water_mask) + 1e-8)
    return loss