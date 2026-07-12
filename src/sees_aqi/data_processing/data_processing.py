import os
import rasterio
import numpy as np
import sees_aqi.data_processing.process_function_library as process

# --- Configuration Constants ---
TARGET_YEAR = 2025
TARGET_DOY = 179

# --- FILE PATH CONFIGURATION ---
# Base Reference Image 
BASE_TIF_PATH = 'data/raw_data/raw_algae_map/lake_algae_2025_full.tif'

#Output Dir
OUTPUT_DIR = 'data/final_maps'

# Raw CSV Sources
WIND_SPEED_2M_CSV = 'data/processed_csvs/data_publicCSV/Wind Speed At 2Meters (Western Basin Area) - POWER_Regional_Daily_20250628_20260628.csv'
WIND_DIR_2M_CSV = 'data/processed_csvs/data_publicCSV/Wind Direction At 2Meters (Western Basin Area) - Sheet1.csv'
WIND_SPEED_10M_CSV = 'data/processed_csvs/data_publicCSV/Wind Speed at 10Meters (Western Basin Area) - Sheet1.csv'
WIND_DIR_10M_CSV = 'data/processed_csvs/data_publicCSV/Wind Direction At 10Meters (Western Basin Area) - Sheet1.csv'
HUMIDITY_CSV = 'data/processed_csvs/data_publicCSV/Specific Humidity At 2Meters (Western Basin Area) - Sheet1.csv'
UV_CSV = 'data/processed_csvs/data_publicCSV/All Sky Surface UV Index (Western Basin Area) - Sheet1.csv'
PRECIP_CSV = 'data/processed_csvs/data_publicCSV/Precipitation (Western Basin Area) - Sheet1.csv'
TEMP_CSV = 'data/processed_csvs/data_publicCSV/Temperature At 2Meters (Western Basin Area) - Sheet1.csv'


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

    # 1. Process and Split Wind Vectors 2m & 10m (U and V Channels)
    try:
        u_wind_two, v_wind_two = process.map_wind_speed_direction(
            BASE_TIF_PATH, WIND_SPEED_2M_CSV, WIND_DIR_2M_CSV, 'WS2M', target_year=TARGET_YEAR, target_doy=TARGET_DOY
        )
        u_wind_ten, v_wind_ten = process.map_wind_speed_direction(
            BASE_TIF_PATH, WIND_SPEED_10M_CSV, WIND_DIR_10M_CSV, 'WS10M', target_year=TARGET_YEAR, target_doy=TARGET_DOY
        )
        save_as_geotiff(u_wind_two, BASE_TIF_PATH, os.path.join(OUTPUT_DIR, "processed_2m_u_wind.tif"))
        save_as_geotiff(v_wind_two, BASE_TIF_PATH, os.path.join(OUTPUT_DIR, "processed_2m_v_wind.tif"))
        save_as_geotiff(u_wind_ten, BASE_TIF_PATH, os.path.join(OUTPUT_DIR, "processed_10m_u_wind.tif"))
        save_as_geotiff(v_wind_ten, BASE_TIF_PATH, os.path.join(OUTPUT_DIR, "processed_10m_v_wind.tif"))
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
            BASE_TIF_PATH, UV_CSV, target_doy=TARGET_DOY, target_year=TARGET_YEAR
        )
        save_as_geotiff(uv_index, BASE_TIF_PATH, os.path.join(OUTPUT_DIR, "processed_uv_index.tif"))
    except Exception as e:
        print(f"Error executing UV index interpolation: {e}")

    # 4. Process Precipitation
    try:
        precipitation = process.generate_precipitation_raster(
            BASE_TIF_PATH, PRECIP_CSV, target_doy=TARGET_DOY, target_year=TARGET_YEAR, col_name='PRECTOTCORR'
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