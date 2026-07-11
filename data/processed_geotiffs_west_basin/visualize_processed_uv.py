import rioxarray as rxr
import matplotlib.pyplot as plt

# 1. Load your generated GeoTIFF and the original reference map
uv_map = rxr.open_rasterio("data/processed_geotiffs_west_basin/processed_uv.tif").sel(band=1)
algae_map = rxr.open_rasterio("data/test_data/test_raw_algae_map/lake_algae_2025.tif").sel(band=1)

# 2. Set up a side-by-side subplot comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot Temperature
uv_plot = uv_map.plot(ax=axes[0], cmap="Dark2", add_colorbar=True)
axes[0].set_title("Processed UV (UV)")

# Plot Algae Reference
algae_plot = algae_map.plot(ax=axes[1], cmap="viridis", add_colorbar=True)
axes[1].set_title("Reference Algae Map")

# Clean up layout and display
plt.tight_layout()
plt.show()