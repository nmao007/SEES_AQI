import torch

def compute_advection_residual(pred_plume, u_wind, v_wind, dx=10.0, dy=10.0, dt=3600.0):
    """
    Computes the advection equation residual: dC/dt + U*(dC/dx) + V*(dC/dy) = 0
    Shapes expected: [Batch, 1, Height, Width, Time]
    """
    # 1. Time derivative: dC/dt (Forward difference along the Time dimension)
    # Shape becomes [B, 1, H, W, T-1]
    dC_dt = (pred_plume[..., 1:] - pred_plume[..., :-1]) / dt
    
    # 2. Spatial derivatives: dC/dx and dC/dy (Central difference)
    # We slice to align shapes with the interior pixels
    dC_dx = (pred_plume[:, :, :, 2:, :-1] - pred_plume[:, :, :, :-2, :-1]) / (2 * dx)
    dC_dy = (pred_plume[:, :, 2:, :, :-1] - pred_plume[:, :, :-2, :, :-1]) / (2 * dy)
    
    # 3. Align Wind Tensors to match the interior spatial crop [..., 1:-1, 1:-1, :-1]
    u_interior = u_wind[:, :, 1:-1, 1:-1, :-1]
    v_interior = v_wind[:, :, 1:-1, 1:-1, :-1]
    dC_dt_interior = dC_dt[:, :, 1:-1, 1:-1, :]
    
    # 4. Calculate the Advection Residual
    # Physical Law: dC/dt + U*(dC/dx) + V*(dC/dy) should equal 0
    residual = dC_dt_interior + (u_interior * dC_dx[:, :, 1:-1, :]) + (v_interior * dC_dy[:, :, :, 1:-1])
    
    # Return the Mean Squared Error of the physical violation
    return torch.mean(residual ** 2)