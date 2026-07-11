import numpy as np
import rasterio

ref_path = "data/test_data/test_raw_algae_map/lake_algae_2025.tif"
src_path = "data/processed_geotiffs_west_basin/processed_humidity.tif"

# choose nodata value you want (use same as your pipeline)
nodata_val = -999.0

with rasterio.open(ref_path) as ref, rasterio.open(src_path) as src:
    data = src.read(1).astype('float32')
    # replace NaN with nodata (rasterio cannot write NaN as nodata reliably)
    data_out = np.where(np.isnan(data), nodata_val, data)

    meta = ref.meta.copy()
    meta.update(dtype=rasterio.float32, count=1, nodata=nodata_val)

    # overwrite the processed file with ref metadata but keep your data values
    with rasterio.open(src_path, 'w', **meta) as dst:
        dst.write(data_out, 1)

print("Overwrote", src_path, "with reference metadata.")



ref_path = "data/test_data/test_raw_algae_map/lake_algae_2025.tif"
src_path = "data/processed_geotiffs_west_basin/processed_precipitation.tif"

# choose nodata value you want (use same as your pipeline)
nodata_val = -999.0

with rasterio.open(ref_path) as ref, rasterio.open(src_path) as src:
    data = src.read(1).astype('float32')
    # replace NaN with nodata (rasterio cannot write NaN as nodata reliably)
    data_out = np.where(np.isnan(data), nodata_val, data)

    meta = ref.meta.copy()
    meta.update(dtype=rasterio.float32, count=1, nodata=nodata_val)

    # overwrite the processed file with ref metadata but keep your data values
    with rasterio.open(src_path, 'w', **meta) as dst:
        dst.write(data_out, 1)

print("Overwrote", src_path, "with reference metadata.")



ref_path = "data/test_data/test_raw_algae_map/lake_algae_2025.tif"
src_path = "data/processed_geotiffs_west_basin/processed_temp.tif"

# choose nodata value you want (use same as your pipeline)
nodata_val = -999.0

with rasterio.open(ref_path) as ref, rasterio.open(src_path) as src:
    data = src.read(1).astype('float32')
    # replace NaN with nodata (rasterio cannot write NaN as nodata reliably)
    data_out = np.where(np.isnan(data), nodata_val, data)

    meta = ref.meta.copy()
    meta.update(dtype=rasterio.float32, count=1, nodata=nodata_val)

    # overwrite the processed file with ref metadata but keep your data values
    with rasterio.open(src_path, 'w', **meta) as dst:
        dst.write(data_out, 1)

print("Overwrote", src_path, "with reference metadata.")



ref_path = "data/test_data/test_raw_algae_map/lake_algae_2025.tif"
src_path = "data/processed_geotiffs_west_basin/processed_uv.tif"

# choose nodata value you want (use same as your pipeline)
nodata_val = -999.0

with rasterio.open(ref_path) as ref, rasterio.open(src_path) as src:
    data = src.read(1).astype('float32')
    # replace NaN with nodata (rasterio cannot write NaN as nodata reliably)
    data_out = np.where(np.isnan(data), nodata_val, data)

    meta = ref.meta.copy()
    meta.update(dtype=rasterio.float32, count=1, nodata=nodata_val)

    # overwrite the processed file with ref metadata but keep your data values
    with rasterio.open(src_path, 'w', **meta) as dst:
        dst.write(data_out, 1)

print("Overwrote", src_path, "with reference metadata.")



ref_path = "data/test_data/test_raw_algae_map/lake_algae_2025.tif"
src_path = "data/processed_geotiffs_west_basin/processed_wind_direction.tif"

# choose nodata value you want (use same as your pipeline)
nodata_val = -999.0

with rasterio.open(ref_path) as ref, rasterio.open(src_path) as src:
    data = src.read(1).astype('float32')
    # replace NaN with nodata (rasterio cannot write NaN as nodata reliably)
    data_out = np.where(np.isnan(data), nodata_val, data)

    meta = ref.meta.copy()
    meta.update(dtype=rasterio.float32, count=1, nodata=nodata_val)

    # overwrite the processed file with ref metadata but keep your data values
    with rasterio.open(src_path, 'w', **meta) as dst:
        dst.write(data_out, 1)

print("Overwrote", src_path, "with reference metadata.")



ref_path = "data/test_data/test_raw_algae_map/lake_algae_2025.tif"
src_path = "data/processed_geotiffs_west_basin/processed_wind_speed.tif"

# choose nodata value you want (use same as your pipeline)
nodata_val = -999.0

with rasterio.open(ref_path) as ref, rasterio.open(src_path) as src:
    data = src.read(1).astype('float32')
    # replace NaN with nodata (rasterio cannot write NaN as nodata reliably)
    data_out = np.where(np.isnan(data), nodata_val, data)

    meta = ref.meta.copy()
    meta.update(dtype=rasterio.float32, count=1, nodata=nodata_val)

    # overwrite the processed file with ref metadata but keep your data values
    with rasterio.open(src_path, 'w', **meta) as dst:
        dst.write(data_out, 1)

print("Overwrote", src_path, "with reference metadata.")