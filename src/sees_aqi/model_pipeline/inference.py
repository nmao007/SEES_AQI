import torch
import matplotlib.pyplot as plt
from neuralop.models import FNO
from load_data import load_processed_data

def run_local_inference_and_visualize():
    # 1. Initialize the 2D architecture
    model = FNO(
        n_modes=(16, 16),    # Changed from (12, 12) to match checkpoint
        hidden_channels=24,  # Changed from 12 to 24
        in_channels=10,
        out_channels=1
    )
    
    # 2. Load weights onto CPU
    weights_path = "fno_algae_physics_model.pt" 
    model.load_state_dict(
        torch.load(weights_path, map_location=torch.device('cpu'), weights_only=False)
    )
    model.eval()

    print("loading inputs now")
    
    # 3. Load 2D data map layer tensor
    inputs = load_processed_data("data/final_maps") 
    
    print("inputs loaded! Running prediction now")

    # 4. Run prediction (No gradients tracked)
    with torch.no_grad():
        predictions = model(inputs)
        
    output_grid = predictions.squeeze().numpy()
    print("✨ Local CPU inference successful!")
    
    # 5. Visualize the 2D Output Grid
    plt.figure(figsize=(10, 8))
    # 'viridis' or 'hot' work great for concentration plumes
    plt.imshow(output_grid, cmap='viridis', origin='upper') 
    plt.colorbar(label='Airborne Toxin Concentration')
    
    plt.title('24-Hour Forecasted Airborne Toxin Distribution')
    plt.xlabel('Grid Width (Pixels)')
    plt.ylabel('Grid Height (Pixels)')
    
    # Renders the window directly on your screen
    plt.show()
    
    return output_grid

if __name__ == "__main__":
    run_local_inference_and_visualize()