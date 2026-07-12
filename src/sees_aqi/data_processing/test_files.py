import rasterio

file_path = "data/processed_maps/processed_precipitation.tif"

with rasterio.open(file_path) as src:
    if src.crs is not None:
        print("✅ This is a GeoTIFF!")
        print(f"   Coordinate Reference System: {src.crs}")
        print(f"   Origin Transform Matrix:\n{src.transform}")
    else:
        print("❌ This is a standard TIFF image (no geospatial metadata found).")