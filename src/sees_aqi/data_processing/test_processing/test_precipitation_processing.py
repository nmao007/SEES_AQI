import sees_aqi.data_processing.process_function_library as process

TIF_FILE_PATH = "data/test_data/test_raw_algae_map/lake_algae_2025.tif" 
CSV_FILE_PATH = "data/test_data/test_csvs/Precipitation (Western Basin Area) - Sheet1.csv"

precipitation_matrix_nn = process.generate_precipitation_raster(TIF_FILE_PATH, CSV_FILE_PATH, target_year=2025, target_doy=179)

process.map_visualization(precipitation_matrix_nn, 'Percipitation')