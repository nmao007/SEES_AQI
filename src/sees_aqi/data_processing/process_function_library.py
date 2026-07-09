import rasterio
import pandas as pd
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
import matplotlib.pyplot as plt


def map_wind_with_nearest_neighbor(tif_path, csv_path, target_year=2025, target_doy=179):
    # 1. Open the GeoTIFF to get the exact 2D canvas dimensions
    with rasterio.open(tif_path) as src:
        height, width = src.shape
        
        # 2. Load CSV, skipping the 9 metadata rows so Row 10 becomes the header
        df = pd.read_csv(csv_path, skiprows=9)
        
        # Strip any accidental hidden spaces from the header names
        df.columns = df.columns.str.strip()
        
        # 3. Filter the rows down to your specific target day
        day_filter = (df['YEAR'] == target_year) & (df['DOY'] == target_doy)
        filtered_df = df[day_filter]
        
        if filtered_df.empty:
            raise ValueError(f"No data found in CSV for Year {target_year}, DOY {target_doy}!")
            
        # 4. Convert all station coordinates into pixel positions (Row, Col)
        station_pixels = []
        station_speeds = []
        
        for idx, row in filtered_df.iterrows():
            lon = row['LON']
            lat = row['LAT']
            speed = row['WS2M']
            
            # Use the GeoTIFF header math to find the pixel index of the coordinate
            p_row, p_col = src.index(lon, lat)
            
            station_pixels.append([p_row, p_col])
            station_speeds.append(speed)
            
        station_pixels = np.array(station_pixels)
        station_speeds = np.array(station_speeds)
        
        # 5. Build the Spatial Lookup Tree using the station pixel coordinates
        spatial_tree = cKDTree(station_pixels)
        
        # 6. Generate a coordinate list of EVERY single pixel in your 30km map
        rows, cols = np.indices((height, width))
        all_pixel_coords = np.column_stack((rows.ravel(), cols.ravel()))
        
        # 7. Query the tree: Find the closest station for all pixels instantly
        _, closest_station_indices = spatial_tree.query(all_pixel_coords)
        
        # 8. Extract the wind speeds and reshape them back into the 2D grid layout
        flat_wind_grid = station_speeds[closest_station_indices]
        wind_grid_2d = flat_wind_grid.reshape(height, width)
        
        print("--- Execution Summary ---")
        print(f"Skipped metadata headers successfully.")
        print(f"Target Date: Year {target_year}, DOY {target_doy}")
        print(f"Generated Seamless 2D Wind Grid Shape: {wind_grid_2d.shape}")
        
        return wind_grid_2d
    

def map_wind_smooth_bilinear(tif_path, csv_path, target_year=2025, target_doy=179):
    with rasterio.open(tif_path) as src:
        height, width = src.shape
        
        # 1. Load and filter CSV
        df = pd.read_csv(csv_path, skiprows=9)
        df.columns = df.columns.str.strip()
        filtered_df = df[(df['YEAR'] == target_year) & (df['DOY'] == target_doy)]
        
        # 2. Extract unique, sorted grid lines from the NASA CSV
        # Because NASA data is a regular grid, sorting gives us clean axes
        unique_lats = np.sort(filtered_df['LAT'].unique())
        unique_lons = np.sort(filtered_df['LON'].unique())
        
        # 3. Pivot the flat CSV wind speeds into a structured 2D matrix matching the axes
        # Shape will be (len(unique_lats), len(unique_lons))
        pivot_table = filtered_df.pivot(index='LAT', columns='LON', values='WS2M')
        wind_source_matrix = pivot_table.loc[unique_lats, unique_lons].values
        
        # 4. Initialize the True Bilinear Interpolator
        # bounds_error=False tells it to handle pixels sticking out past the weather grid edges
        interp_function = RegularGridInterpolator(
            points=(unique_lats, unique_lons), 
            values=wind_source_matrix, 
            method='linear', # 'linear' on a RegularGridInterpolator IS bilinear
            bounds_error=False,
            fill_value=None # Extrapolates edge values automatically
        )
        
        # 5. Generate target coordinates for every 10m pixel
        rows, cols = np.indices((height, width))
        xs, ys = rasterio.transform.xy(src.transform, rows.ravel(), cols.ravel())
        target_points = np.column_stack((ys, xs)) # Format as (Lat, Lon)
        
        # 6. Run the true bilinear interpolation
        print("Executing true bilinear grid interpolation...")
        flat_bilinear_grid = interp_function(target_points)
        
        return flat_bilinear_grid.reshape(height, width)
    
def map_wind_speed_direction(tif_path, speed_csv_path, dir_csv_path, height_str, target_year=2025, target_doy=179):
    # 1. Load the target GeoTIFF dimensions
    with rasterio.open(tif_path) as src:
        height, width = src.shape
        transform = src.transform
        
    # 2. Load and filter both CSVs (skipping headers)
    df_speed = pd.read_csv(speed_csv_path, skiprows=9)
    df_speed.columns = df_speed.columns.str.strip()
    df_speed = df_speed[(df_speed['YEAR'] == target_year) & (df_speed['DOY'] == target_doy)]
    
    df_dir = pd.read_csv(dir_csv_path, skiprows=9)
    df_dir.columns = df_dir.columns.str.strip()
    df_dir = df_dir[(df_dir['YEAR'] == target_year) & (df_dir['DOY'] == target_doy)]
    
    # 3. Merge the datasets side-by-side using their matching coordinates
    merged_df = pd.merge(df_speed, df_dir, on=['LAT', 'LON'])
    
    if merged_df.empty:
        raise ValueError(f"No matching wind data found for Year: {target_year}, DOY: {target_doy}")
    
    # 4. Calculate U & V vector components
    rad = np.radians(merged_df[height_str].values)
    speed = merged_df[height_str].values
    
    merged_df['U'] = -speed * np.sin(rad)
    merged_df['V'] = -speed * np.cos(rad)
    
    # 5. Extract unique grid lines for interpolation axes
    unique_lats = np.sort(merged_df['LAT'].unique())
    unique_lons = np.sort(merged_df['LON'].unique())
    
    # 6. Pivot into 2D source matrices
    pivot_u = merged_df.pivot(index='LAT', columns='LON', values='U')
    pivot_v = merged_df.pivot(index='LAT', columns='LON', values='V')
    
    u_source_matrix = pivot_u.loc[unique_lats, unique_lons].values
    v_source_matrix = pivot_v.loc[unique_lats, unique_lons].values
    
    # 7. Initialize Interpolators
    interp_u = RegularGridInterpolator((unique_lats, unique_lons), u_source_matrix, method='linear', bounds_error=False, fill_value=None)
    interp_v = RegularGridInterpolator((unique_lats, unique_lons), v_source_matrix, method='linear', bounds_error=False, fill_value=None)
    
    # 8. Generate target coordinates and interpolate
    rows, cols = np.indices((height, width))
    xs, ys = rasterio.transform.xy(transform, rows.ravel(), cols.ravel())
    target_points = np.column_stack((ys, xs)) 
    
    print(f"Merging and interpolating wind vectors for DOY {target_doy}...")
    u_raster = interp_u(target_points).reshape(height, width)
    v_raster = interp_v(target_points).reshape(height, width)
    
    return u_raster, v_raster

def generate_humidity_raster(tif_path, csv_path, target_doy, target_year=2025):
    """
    Extracts high-resolution lat/lon coordinates from a GeoTIFF and interpolates
    sparse NASA MERRA-2 Specific Humidity (QV2M) data to match its exact dimensions.
    """
    # 1. Load the target GeoTIFF dimensions and transformation
    with rasterio.open(tif_path) as src:
        height, width = src.shape
        transform = src.transform

    # 2. Load and filter the NASA CSV data
    # skiprows=9 handles the 9 lines of header seen in your image
    df = pd.read_csv(csv_path, skiprows=9)
    df.columns = df.columns.str.strip()  # Clean up any trailing whitespaces in headers
   
    filtered_df = df[(df['YEAR'] == target_year) & (df['DOY'] == target_doy)]
   
    if filtered_df.empty:
        raise ValueError(f"No data found for Year: {target_year}, DOY: {target_doy}")

    # 3. Extract unique, sorted grid lines to form the interpolation axes
    unique_lats = np.sort(filtered_df['LAT'].unique())
    unique_lons = np.sort(filtered_df['LON'].unique())
   
    # 4. Pivot the flat CSV values into a structured 2D matrix matching the axes
    # We use 'QV2M' as the target value from your dataset
    pivot_table = filtered_df.pivot(index='LAT', columns='LON', values='QV2M')
    humidity_source_matrix = pivot_table.loc[unique_lats, unique_lons].values
   
    # 5. Initialize the Bilinear Interpolator
    interp_function = RegularGridInterpolator(
        points=(unique_lats, unique_lons),
        values=humidity_source_matrix,
        method='linear',
        bounds_error=False,
        fill_value=None  # Extrapolates edge values automatically if the TIF exceeds the CSV bounds
    )
   
    # 6. Generate target coordinates for every single high-res pixel
    rows, cols = np.indices((height, width))
    xs, ys = rasterio.transform.xy(transform, rows.ravel(), cols.ravel())
    target_points = np.column_stack((ys, xs)) # Format precisely as (Lat, Lon) to match points structure
   
    # 7. Execute interpolation and reshape back to original raster dimensions
    print(f"Interpolating QV2M Specific Humidity grid for Year {target_year}, DOY {target_doy}...")
    flat_humidity_grid = interp_function(target_points)
    humidity_raster = flat_humidity_grid.reshape(height, width)
   
    return humidity_raster

def generate_uv_raster(tif_path, csv_path, target_doy, target_year=2025):
    """
    Extracts high-resolution lat/lon coordinates from a GeoTIFF and interpolates
    sparse NASA UV Index data using griddata to match its exact dimensions.
    """
    # 1. Load the target GeoTIFF metadata and geometry
    with rasterio.open(tif_path) as src:
        height, width = src.shape
        transform = src.transform
        
        # Generate matrices of pixel coordinates
        cols, rows = np.meshgrid(np.arange(width), np.arange(height))
        # Convert pixel positions to real-world longitude (xs) and latitude (ys)
        xs, ys = rasterio.transform.xy(transform, rows, cols)
        target_lons = np.array(xs)
        target_lats = np.array(ys)

    # 2. Load and filter the NASA CSV data
    df = pd.read_csv(csv_path, skiprows=9)
    df.columns = df.columns.str.strip()  # Clean up any trailing whitespaces in headers
   
    filtered_df = df[(df['YEAR'] == target_year) & (df['DOY'] == target_doy)]
   
    if filtered_df.empty:
        raise ValueError(f"No UV data found for Year: {target_year}, DOY: {target_doy}")

    # 3. Extract known coordinate anchor points and their UV values
    known_coords = filtered_df[['LAT', 'LON']].values
    uv_values = filtered_df['ALLSKY_SFC_UV_INDEX'].values
   
    # 4. Execute spatial interpolation directly onto the target coordinate mesh
    print(f"Interpolating UV Index grid for Year {target_year}, DOY {target_doy}...")
    flat_uv_raster = griddata(
        points=known_coords,
        values=uv_values,
        xi=(target_lats, target_lons),
        method='linear'
    )

    uv_raster = flat_uv_raster.reshape(height, width)
   
    return uv_raster

def generate_precipitation_raster(tif_path, csv_path, target_doy, target_year=2025, col_name='PRECTOTCORR'):
    """
    Extracts high-resolution lat/lon coordinates from a GeoTIFF and interpolates
    sparse NASA precipitation data (e.g., PRECTOT) to match its exact dimensions.
    """
    # 1. Load the target GeoTIFF dimensions and transformation
    with rasterio.open(tif_path) as src:
        height, width = src.shape
        transform = src.transform

    # 2. Load and filter the NASA CSV data
    # skiprows=9 handles the 9 lines of metadata headers
    df = pd.read_csv(csv_path, skiprows=9)
    df.columns = df.columns.str.strip()  # Clean up any trailing whitespaces in headers
   
    filtered_df = df[(df['YEAR'] == target_year) & (df['DOY'] == target_doy)]
   
    if filtered_df.empty:
        raise ValueError(f"No data found for Year: {target_year}, DOY: {target_doy}")
        
    if col_name not in filtered_df.columns:
        raise KeyError(f"Column '{col_name}' not found in the dataset. Available columns: {list(filtered_df.columns)}")

    # 3. Extract unique, sorted grid lines to form the interpolation axes
    unique_lats = np.sort(filtered_df['LAT'].unique())
    unique_lons = np.sort(filtered_df['LON'].unique())
   
    # 4. Pivot the flat CSV values into a structured 2D matrix matching the axes
    pivot_table = filtered_df.pivot(index='LAT', columns='LON', values=col_name)
    precip_source_matrix = pivot_table.loc[unique_lats, unique_lons].values
   
    # 5. Initialize the Bilinear Interpolator ('linear' on regular grids = bilinear)
    interp_function = RegularGridInterpolator(
        points=(unique_lats, unique_lons),
        values=precip_source_matrix,
        method='linear',
        bounds_error=False,
        fill_value=None  # Extrapolates edge values automatically if the TIF exceeds the CSV bounds
    )
   
    # 6. Generate target coordinates for every single high-res pixel
    rows, cols = np.indices((height, width))
    xs, ys = rasterio.transform.xy(transform, rows.ravel(), cols.ravel())
    target_points = np.column_stack((ys, xs)) # Format precisely as (Lat, Lon) to match points structure
   
    # 7. Execute interpolation and reshape back to original raster dimensions
    print(f"Interpolating {col_name} Precipitation grid for Year {target_year}, DOY {target_doy}...")
    flat_precip_grid = interp_function(target_points)
    precip_raster = flat_precip_grid.reshape(height, width)
   
    return precip_raster

def generate_temperature_raster(tif_path, csv_path, target_doy, target_year=2025):
    """
    Extracts high-resolution lat/lon coordinates from a GeoTIFF and interpolates
    sparse NASA MERRA-2 Temperature at 2 Meters (T2M) data to match its exact dimensions.
    """
    # 1. Load the target GeoTIFF dimensions and transformation
    with rasterio.open(tif_path) as src:
        height, width = src.shape
        transform = src.transform

    # 2. Load and filter the NASA CSV data
    # skiprows=9 handles the 9 lines of header seen in your image
    df = pd.read_csv(csv_path, skiprows=9)
    df.columns = df.columns.str.strip()  # Clean up any trailing whitespaces in headers
   
    filtered_df = df[(df['YEAR'] == target_year) & (df['DOY'] == target_doy)]
   
    if filtered_df.empty:
        raise ValueError(f"No data found for Year: {target_year}, DOY: {target_doy}")

    # 3. Extract unique, sorted grid lines to form the interpolation axes
    unique_lats = np.sort(filtered_df['LAT'].unique())
    unique_lons = np.sort(filtered_df['LON'].unique())
   
    # 4. Pivot the flat CSV values into a structured 2D matrix matching the axes
    # Uses 'T2M' directly from your dataset column header
    pivot_table = filtered_df.pivot(index='LAT', columns='LON', values='T2M')
    temp_source_matrix = pivot_table.loc[unique_lats, unique_lons].values
   
    # 5. Initialize the Bilinear Interpolator
    interp_function = RegularGridInterpolator(
        points=(unique_lats, unique_lons),
        values=temp_source_matrix,
        method='linear',
        bounds_error=False,
        fill_value=None  # Extrapolates edge values automatically if the TIF exceeds the CSV bounds
    )
   
    # 6. Generate target coordinates for every single high-res pixel
    rows, cols = np.indices((height, width))
    xs, ys = rasterio.transform.xy(transform, rows.ravel(), cols.ravel())
    target_points = np.column_stack((ys, xs)) # Format precisely as (Lat, Lon) to match points structure
   
    # 7. Execute interpolation and reshape back to original raster dimensions
    print(f"Interpolating T2M Temperature grid for Year {target_year}, DOY {target_doy}...")
    flat_temp_grid = interp_function(target_points)
    temperature_raster = flat_temp_grid.reshape(height, width)
   
    return temperature_raster

def map_visualization(map, variable):
    # Assuming 'final_wind_channel' is the 2D array generated from the previous script
    plt.figure(figsize=(10, 8), dpi=100)

    # 1. Display the 2D array as an image
    # 'plasma', 'viridis', or 'coolwarm' are great colormaps for wind data
    im = plt.imshow(map, cmap='plasma')

    # 2. Add a colorbar scale on the side so you know what the colors mean
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label(variable, fontsize=12, fontweight='bold')

    # 3. Add titles and axis labels
    plt.title('NASA/POWER Interpolated ' + variable + ' Grid', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Pixel X Column (10m Resolution)', fontsize=10)
    plt.ylabel('Pixel Y Row (10m Resolution)', fontsize=10)

    # 4. Turn on a subtle grid layout over the pixels
    plt.grid(False) 

    # 5. Save the image to your disk (Great for .py scripts)
    #plt.savefig('wind_speed_grid_visualization.png', bbox_inches='tight', dpi=300)

    # 6. Render the plot to the screen
    plt.show()