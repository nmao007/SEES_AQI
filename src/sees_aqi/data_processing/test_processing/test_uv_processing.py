import sees_aqi.data_processing.process_function_library as process
import pandas as pd

TIF_FILE_PATH = "data/test_data/test_raw_algae_map/lake_algae_2025.tif" 
CSV_FILE_PATH = "data/processed_csvs/data_publicCSV/All Sky Surface UV Index (Western Basin Area) - Sheet1.csv"


uv_matrix_nn = process.generate_uv_raster(TIF_FILE_PATH, CSV_FILE_PATH, target_year=2025, target_doy=179)

process.map_visualization(uv_matrix_nn, 'UV Index')