import sees_aqi.data_processing.csv_to_geotiffs as process

TIF_FILE_PATH = "data/test_data/test_raw_algae_map/lake_algae_2025.tif" 
CVS_FILE_PATH = "data/test_data/test_csvs/Wind Speed At 2Meters (Western Basin Area) - POWER_Regional_Daily_20250628_20260628 copy.csv"
OUTPUT_TIF = "data/processed_geotiffs_west_basin/processed_wind_speed.tif"

wind_speed_geotiff = process.csv_table_to_geotiff_match_reference(CVS_FILE_PATH, TIF_FILE_PATH, OUTPUT_TIF, target_year=2025, target_doy=179, value_col="WS2M")

print(wind_speed_geotiff)