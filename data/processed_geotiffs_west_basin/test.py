
# import rasterio
# ref="data/test_data/test_raw_algae_map/lake_algae_2025.tif"
# t="data/processed_geotiffs_west_basin/processed_uv.tif"

# with rasterio.open(ref) as r:
#     print("REF CRS:", r.crs)
#     print("REF transform:", r.transform)
#     print("REF width,height:", r.width, r.height)
#     print("REF count:", r.count)

# with rasterio.open(t) as s:
#     print("TST CRS:", s.crs)
#     print("TST transform:", s.transform)
#     print("TST width,height:", s.width, s.height)
#     print("TST count:", s.count)

import numpy as np
import rasterio

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


ref="data/test_data/test_raw_algae_map/lake_algae_2025.tif"
t="data/processed_geotiffs_west_basin/processed_wind_speed.tif"
with rasterio.open(ref) as r: ref_meta=(r.crs, r.transform, r.width, r.height)
with rasterio.open(t) as s: s_meta=(s.crs, s.transform, s.width, s.height)
print("REF:", ref_meta)
print("TST:", s_meta)
print("MATCH:", ref_meta==s_meta)

# import matplotlib.pyplot as plt
# p="data/processed_geotiffs_west_basin/processed_uv.tif"
# with rasterio.open(p) as src:
#     arr = src.read(1)
# plt.figure(figsize=(6,6))
# plt.imshow(arr, cmap='viridis', origin='upper')
# plt.colorbar(label='T2M (°C)')
# plt.title('Processed Temperature (quick view)')
# plt.show()

# # import rioxarray as rxr
# # ref = rxr.open_rasterio("data/test_data/test_raw_algae_map/lake_algae_2025.tif").squeeze()
# # tst = rxr.open_rasterio("data/processed_geotiffs_west_basin/processed_temp.tif").squeeze()
# # # resample ref to tst if needed, then compute difference
# # ref2 = ref.rio.reproject_match(tst)
# # diff = tst - ref2.mean(dim='band', skipna=True) if 'band' in ref2.dims else tst - ref2
# # print("diff stats:", float(diff.min()), float(diff.max()), float(diff.mean()))

# import rasterio, numpy as np
# p="data/processed_geotiffs_west_basin/processed_uv.tif"
# with rasterio.open(p) as src:
#     arr = src.read(1).astype('float32')
# mask = np.isnan(arr) | (arr == src.nodata)
# print("shape:", arr.shape)
# print("min,max,mean,std:", float(np.nanmin(arr)), float(np.nanmax(arr)), float(np.nanmean(arr)), float(np.nanstd(arr)))
# print("percent nodata:", 100.0 * np.count_nonzero(mask) / arr.size)
# print("nodata metadata:", src.nodata)