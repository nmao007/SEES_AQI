import os
import rasterio
import numpy as np

# Absolute package import matching your repository setup
import sees_aqi.data_processing.process_function_library as process

# --- Configuration Constants ---
TARGET_YEAR = 2025
TARGET_DOY = 179

# Folder Routes matching your project tree
BASE_DATA_DIR = "data"
RAW_DATA_DIR = os.path.join(BASE_DATA_DIR, "raw_data")
OUTPUT_DIR = os.path.join(BASE_DATA_DIR, "processed_maps")

# --- FILE PATH CONFIGURATION ---
# Base Reference Image 
BASE_TIF_PATH = os.path.join(BASE_DATA_DIR, "test_data", "test_raw_algae_map", "lake_algae_2025.tif")

# Raw CSV Sources (Replace placeholders with your exact raw file names)
WIND_SPEED_CSV = os.path.join(RAW_DATA_DIR, "FILL_IN_WIND_SPEED_CSV_NAME.csv")
WIND_DIR_CSV   = os.path.join(RAW_DATA_DIR, "FILL_IN_WIND_DIRECTION_CSV_NAME.csv")
HUMIDITY_CSV   = os.path.join(RAW_DATA_DIR, "FILL_IN_HUMIDITY_CSV_NAME.csv")
UV_CSV         = os.path.join(RAW_DATA_DIR, "FILL_IN_UV_INDEX_CSV_NAME.csv")
PRECIP_CSV     = os.path.join(RAW_DATA_DIR, "FILL_IN_PRECIPITATION_CSV_NAME.csv")
TEMP_CSV       = os.path.join(RAW_DATA_DIR, "FILL_IN_TEMPERATURE_CSV_NAME.csv")


def save_as_geotiff(data_array, reference_tif_path, output_path):
    """
    Saves a 2D numpy array as a standardized GeoTIFF, completely copying the 
    spatial coordinate metadata and transformation matrix from the base map.
    """
    with rasterio.open(reference_tif_path) as ref:
        meta = ref.meta.copy()
        
    # Update profile to guarantee compatibility with float32 interpolation matrices
    meta.update({
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1
    })
    
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(data_array.astype(np.float32), 1)
    print(f"Successfully exported raster layer to: {output_path}")


def main():
    # Ensure target directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Beginning pipeline interpolation for Year {TARGET_YEAR}, DOY {TARGET_DOY}...")

    # 1. Process and Split Wind Vectors (U and V Channels)
    try:
        u_wind, v_wind = process.map_wind_speed_direction(
            BASE_TIF_PATH, WIND_SPEED_CSV, WIND_DIR_CSV, target_year=TARGET_YEAR, target_doy=TARGET_DOY
        )
        save_as_geotiff(u_wind, BASE_TIF_PATH, os.path.join(OUTPUT_DIR, "processed_u_wind.tif"))
        save_as_geotiff(v_wind, BASE_TIF_PATH, os.path.join(OUTPUT_DIR, "processed_v_wind.tif"))
    except Exception as e:
        print(f"Error executing wind vector pipeline: {e}")

    # 2. Process Specific Humidity (QV2M)
    try:
        humidity = process.generate_humidity_raster(
            BASE_TIF_PATH, HUMIDITY_CSV, target_doy=TARGET_DOY, target_year=TARGET_YEAR
        )
        save_as_geotiff(humidity, BASE_TIF_PATH, os.path.join(OUTPUT_DIR, "processed_humidity.tif"))
    except Exception as e:
        print(f"Error executing humidity interpolation: {e}")

    # 3. Process UV Index (ALLSKY_SFC_UV_INDEX)
    try:
        # Note: generate_uv_raster takes csv_path before tif_path
        uv_index = process.generate_uv_raster(
            UV_CSV, BASE_TIF_PATH, target_doy=TARGET_DOY, target_year=TARGET_YEAR
        )
        save_as_geotiff(uv_index, BASE_TIF_PATH, os.path.join(OUTPUT_DIR, "processed_uv_index.tif"))
    except Exception as e:
        print(f"Error executing UV index interpolation: {e}")

    # 4. Process Precipitation (PRECTOT)
    try:
        precipitation = process.generate_precipitation_raster(
            BASE_TIF_PATH, PRECIP_CSV, target_doy=TARGET_DOY, target_year=TARGET_YEAR, col_name='PRECTOT'
        )
        save_as_geotiff(precipitation, BASE_TIF_PATH, os.path.join(OUTPUT_DIR, "processed_precipitation.tif"))
    except Exception as e:
        print(f"Error executing precipitation interpolation: {e}")

    # 5. Process Temperature (T2M)
    try:
        temperature = process.generate_temperature_raster(
            BASE_TIF_PATH, TEMP_CSV, target_doy=TARGET_DOY, target_year=TARGET_YEAR
        )
        save_as_geotiff(temperature, BASE_TIF_PATH, os.path.join(OUTPUT_DIR, "processed_temperature.tif"))
    except Exception as e:
        print(f"Error executing temperature interpolation: {e}")

    print("\n--- Processing Pipeline Complete ---")
    print(f"All processed grids are aligned to 10m resolution and stored in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()