import torch

# --- Physical Configuration Scale ---
DX = 40.0         # 40-meter pixel grid spacing (Adjusted for 4x downsampling)
DT = 86400.0      # 24 hours in seconds (60s * 60m * 24h)

def compute_comprehensive_physics_loss(pred, inputs):
    """
    Computes PINO loss for a 24-hour forward forecast.
    Inputs = Day N weather/biomass conditions.
    Pred = Predicted Airborne Toxin at Day N+1 (Passed through softplus).
    """
    # 1. Isolate all 10 input layers (Day N)
    u_10m   = inputs[:, 2:3, ...]
    v_10m   = inputs[:, 3:4, ...]
    elev    = inputs[:, 4:5, ...]
    temp    = inputs[:, 7:8, ...]
    uv      = inputs[:, 8:9, ...]
    biomass = inputs[:, 9:10, ...] 

    # 2. Compute Spatial Derivatives (PyTorch dimensions: [B, C, Height/Y, Width/X])
    # FIXED: Index 2 is rows (Y-axis), Index 3 is columns (X-axis)
    dC_dy = (pred[:, :, 2:, 1:-1] - pred[:, :, :-2, 1:-1]) / (2 * DX)
    dC_dx = (pred[:, :, 1:-1, 2:] - pred[:, :, 1:-1, :-2]) / (2 * DX)
    
    d2C_dy2 = (pred[:, :, 2:, 1:-1] - 2 * pred[:, :, 1:-1, 1:-1] + pred[:, :, :-2, 1:-1]) / (DX ** 2)
    d2C_dx2 = (pred[:, :, 1:-1, 2:] - 2 * pred[:, :, 1:-1, 1:-1] + pred[:, :, 1:-1, :-2]) / (DX ** 2)

    # 3. Temporal Derivative (dC_dt)
    C_today = 0.0
    dC_dt = (pred[:, :, 1:-1, 1:-1] - C_today) / DT

    # 4. Slice Input Layers to match the interior grid spatial dims
    u_10m_int   = u_10m[:, :, 1:-1, 1:-1]
    v_10m_int   = v_10m[:, :, 1:-1, 1:-1]
    temp_int    = temp[:, :, 1:-1, 1:-1]
    uv_int      = uv[:, :, 1:-1, 1:-1]
    biomass_int = biomass[:, :, 1:-1, 1:-1]

    # 5. Calculate Emissions and Diffusion based on Day N conditions
    D = 0.05  # Base diffusion parameter
    biological_toxin_potential = biomass_int * torch.sigmoid(temp_int - 20.0) * torch.relu(uv_int)
    wind_speed_10m = torch.sqrt(u_10m_int**2 + v_10m_int**2 + 1e-6)
    emission_source_term = 0.005 * wind_speed_10m * biological_toxin_potential

    # 6. Assemble the Time-Dependent Transport PDE Residual
    # FIXED: Horizontal wind (u) maps to dC_dx, vertical wind (v) maps to dC_dy
    advection_term = (u_10m_int * dC_dx) + (v_10m_int * dC_dy)
    diffusion_term = D * (d2C_dx2 + d2C_dy2)
    
    # Complete Equation: Change Over Time + Movement - Spreading = New Emissions
    residual = dC_dt + advection_term - diffusion_term - emission_source_term

    # 7. Apply Mask and Calculate Loss
    water_mask = (elev[:, :, 1:-1, 1:-1] != -2.0).float()
    squared_residual = (residual * water_mask) ** 2
    return torch.sum(squared_residual) / (torch.sum(water_mask) + 1e-8)