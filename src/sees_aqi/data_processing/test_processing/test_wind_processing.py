import sees_aqi.data_processing.process_function_library as process

TIF_FILE_PATH = "data/test_data/test_raw_algae_map/lake_algae_2025.tif" 
SPEED_FILE_PATH = "data/test_data/test_csvs/Wind Speed At 2Meters (Western Basin Area) - POWER_Regional_Daily_20250628_20260628 copy.csv"
DIR_FILE_PATH = "data/test_data/test_csvs/Wind Direction At 2Meters (Western Basin Area) - Sheet1.csv"

wind_matrix_nn = process.map_wind_with_nearest_neighbor(TIF_FILE_PATH, SPEED_FILE_PATH, target_year=2025, target_doy=179)

process.map_visualization(wind_matrix_nn, 'Wind Speed at 2m')

wind_matrix_smooth = process.map_wind_smooth_bilinear(TIF_FILE_PATH, SPEED_FILE_PATH, target_year=2025, target_doy=179)

process.map_visualization(wind_matrix_smooth, 'Wind Speed at 2m')

u_wind, v_wind = process.map_wind_speed_direction(tif_path=TIF_FILE_PATH, speed_csv_path=SPEED_FILE_PATH, dir_csv_path=DIR_FILE_PATH, target_year=2025, target_doy=179)
process.map_visualization(u_wind, 'East-West wind at 2m')
process.map_visualization(v_wind, 'North-South wind at 2m')