import sees_aqi.data_processing.process_function_library as process

TIF_FILE_PATH = "data/test_data/test_raw_algae_map/lake_algae_2025.tif" 
CSV_FILE_PATH = "data/test_data/test_csvs/Wind Speed At 2Meters (Western Basin Area) - POWER_Regional_Daily_20250628_20260628 copy.csv"

wind_matrix_nn = process.map_wind_with_nearest_neighbor(TIF_FILE_PATH, CSV_FILE_PATH, target_year=2025, target_doy=179)

process.map_visualization(wind_matrix_nn, 'Wind Speed at 2m')

wind_matrix_smooth = process.map_wind_smooth_bilinear(TIF_FILE_PATH, CSV_FILE_PATH, target_year=2025, target_doy=179)

process.map_visualization(wind_matrix_smooth, 'Wind Speed at 2m')