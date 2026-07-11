import pandas as pd
import xarray as xr
import rioxarray as rxr
import rasterio
import numpy as np
import sys

def csv_table_to_geotiff_match_reference(
    csv_path,
    reference_tif,
    out_tif,
    target_year,
    target_doy,
    value_col="T2M",
    lat_col="LAT",
    lon_col="LON",
    nodata=-999.0
):
    # 1) Programmatically find where the metadata header ends
    skip_rows = 0
    with open(csv_path, 'r') as f:
        for i, line in enumerate(f):
            if "-END HEADER-" in line:
                skip_rows = i + 1
                break
               
    if skip_rows == 0:
        print("Warning: -END HEADER- not found. Attempting to read from top.")

    # 2) Read CSV table skipping the metadata header lines
    df = pd.read_csv(csv_path, skiprows=skip_rows)

    # Clean column names in case there is trailing whitespace
    df.columns = df.columns.str.strip()

    # 3) Filter by year and doy using the argument names expected by your test script
    df = df[(df["YEAR"] == target_year) & (df["DOY"] == target_doy)]
    if df.empty:
        raise ValueError(f"No rows found for YEAR={target_year} DOY={target_doy} in {csv_path}")

    # 4) Pivot to grid: rows = lat, cols = lon
    pivot = df.pivot_table(index=lat_col, columns=lon_col, values=value_col, aggfunc='mean')
    # Sort lat descending so top row is max latitude (north)
    pivot = pivot.sort_index(ascending=False)
    lats = pivot.index.values
    lons = pivot.columns.values
    arr = pivot.values.astype('float32')

    # Replace missing with nodata
    arr = np.where(np.isnan(arr), nodata, arr)

    # 5) Create xarray DataArray with lat/lon coords
    da = xr.DataArray(
        arr,
        dims=("lat", "lon"),
        coords={"lat": lats, "lon": lons},
        name=value_col
    )
   
    # FIX: Explicitly set spatial dimensions so rioxarray knows how to reproject
    da = da.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
   
    # Attach CRS (assume lat/lon WGS84)
    da = da.rio.write_crs("EPSG:4326")

    # 6) Reproject / match the reference GeoTIFF grid
    ref = rxr.open_rasterio(reference_tif)
    # If ref has multiple bands, take the first band for matching
    ref = ref.sel(band=1) if "band" in ref.dims else ref
   
    # FIX: Uses rasterio's native enum tracking or standard string safely
    da_matched = da.rio.reproject_match(ref, resampling=rasterio.enums.Resampling.bilinear)

    # 7) Save to GeoTIFF with same metadata as reference
    da_matched.rio.to_raster(out_tif, dtype='float32', nodata=nodata)
    print(f"Wrote {out_tif} matching {reference_tif}")
   
    return da_matched