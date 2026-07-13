import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

def clip_and_align_elevation(raw_elevation_path, reference_path, output_path):
    """
    Warps and clips a raw USGS elevation GeoTIFF to match the exact 
    spatial envelope, resolution, and pixel grid dimensions of a reference raster.
    """
    if not os.path.exists(raw_elevation_path):
        raise FileNotFoundError(f"Cannot find raw elevation file at: {raw_elevation_path}")
    if not os.path.exists(reference_path):
        raise FileNotFoundError(f"Cannot find target reference file at: {reference_path}")

    print(f"1. Extracting target grid footprint from: {reference_path}")
    with rasterio.open(reference_path) as ref_src:
        # Clone the reference metadata exactly (CRS, Transform matrix, Dimensions)
        ref_meta = ref_src.meta.copy()
        ref_transform = ref_src.transform
        ref_crs = ref_src.crs
        ref_width = ref_src.width
        ref_height = ref_src.height
        
    print(f"2. Reprojecting and slicing raw elevation data: {raw_elevation_path}")
    with rasterio.open(raw_elevation_path) as src:
        # Enforce float32 data types and assign your default land mask flag as the NoData value
        ref_meta.update({
            'dtype': 'float32',
            'count': 1,
            'nodata': -2.0  # Matches your background land threshold mask
        })
        
        # Instantiate an empty matrix allocation matching the reference pixel counts
        destination_data = np.zeros((1, ref_height, ref_width), dtype=np.float32)
        
        # Warp the raw data onto the reference spatial coordinates
        reproject(
            source=rasterio.band(src, 1),
            destination=destination_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.bilinear, # Smooths elevation contours across cells smoothly
            dst_nodata=-2.0
        )
        
    print(f"3. Writing aligned, pixel-matched elevation raster to: {output_path}")
    with rasterio.open(output_path, 'w', **ref_meta) as dst:
        dst.write(destination_data)
        
    print("✨ Spatial alignment successful! The output array shape perfectly matches your pipeline requirements.")

if __name__ == "__main__":
    # --- CONFIGURE YOUR PATHS HERE ---
    # Point this to where you stored the massive raw file downloaded from USGS
    RAW_ELEVATION_FILE = "data/raw_data/raw_data_public/Raw_elevation.tif" 
    
    # Point this to your processed satellite map layout template
    REFERENCE_ALGAE_FILE = "data/final_maps/lake_algae_2025_full.tif"
    
    # Where the clean, clipped file should be generated
    OUTPUT_ELEVATION_FILE = "data/final_maps/processed_elevation.tif"
    # ---------------------------------
    
    # Ensure target destination path structure exists
    os.makedirs(os.path.dirname(OUTPUT_ELEVATION_FILE), exist_ok=True)
    
    clip_and_align_elevation(RAW_ELEVATION_FILE, REFERENCE_ALGAE_FILE, OUTPUT_ELEVATION_FILE)