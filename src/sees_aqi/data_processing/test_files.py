import os
import rasterio

# Adjust path if your folder is named differently
data_dir = "data/final_maps"

print(f"Scanning all .tif files in '{data_dir}'...\n")

if not os.path.exists(data_dir):
    print(f"❌ Error: The directory '{data_dir}' does not exist.")
    exit()

# Gather all .tif files in the directory
tif_files = [f for f in os.listdir(data_dir) if f.endswith('.tif')]

if not tif_files:
    print("No .tif files found in the directory.")
else:
    # Sort them so they print in a clean alphabetical list
    for filename in sorted(tif_files):
        file_path = os.path.join(data_dir, filename)
        try:
            with rasterio.open(file_path) as src:
                # src.shape returns (height, width) directly without loading data into memory
                print(f"File: {filename:<32} | Shape: {src.shape} | Bands: {src.count} | CRS: {src.crs is not None}")
        except Exception as e:
            print(f"File: {filename:<32} | ❌ Could not read: {e}")

print("\nScan complete!")