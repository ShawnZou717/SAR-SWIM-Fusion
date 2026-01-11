import argparse
import sys
import os
import glob
import re
import numpy as np
import xarray as xr
from shapely.geometry import Polygon
from waveutils import config, wavetools, ambiguity_removal
import pandas as pd
import time
import datetime
# Start the timer at the beginning of the script
script_start_time = time.time()

def log_elapsed_time(start_time, label=""):
    """Log the elapsed time since the given start time."""
    elapsed = time.time() - start_time
    formatted_time = str(datetime.timedelta(seconds=int(elapsed)))
    print(f"{label} Elapsed time: {formatted_time} (HH:MM:SS)")

# Function to be called at the end of the script
def log_total_execution_time():
    log_elapsed_time(script_start_time, "Total script execution:")

homo_dist = config.homo_dist
homo_time = config.homo_time

# Access safety parameters
safe_max_lat = config.max_lat
safe_min_lat = config.min_lat

to_fuse_swim_antenna = config.swim_antenna['10']
swim_reference_time = np.datetime64(config.swim_reference_time, 'us')

# Define logs directory

# Create logs directory for this script if it doesn't exist
# Get the name of the current file (without extension)
current_file_name = os.path.splitext(os.path.basename(__file__))[0]
logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), current_file_name)
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)
    print(f"Created logs directory at: {logs_dir}")
else:
    print(f"Using existing logs directory at: {logs_dir}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fuse SWIM L2 and WaveTac L3 data.")
    parser.add_argument('--swim_l2_path', type=str, required=True, help='Path to SWIM L2 data. (Dont get to the date dirs)')
    parser.add_argument('--wavetac_l3_path', type=str, required=True, help='Path to WaveTac L3 data. (Dont get to the date dirs)')
    parser.add_argument('-o', type=str, default=logs_dir, help="Directory to save output files, if not given, would save in the log dir (dir with the file's name under the current dir).")
    parser.add_argument('--year', '-y', type=int, required=False, help='Year of data to process swim (e.g., 2022).')
    parser.add_argument('--month', '-m', type=int, required=False, help='Month of data to process swim (1-12).', choices=range(1, 13))
    return parser.parse_args()


def reconstruct_radar_polygon(box_centroid, phi_orbit_box, mod='swim'):
    '''
    Reconstructing the polygon of the geo box in L2 product given the box centroid and flight direction based on the radar parameters.

    Input:
    ------
    box_centroid: (lon, lat), the central position of the combined geo box, i.e., the left and right boxes are combiend together to form a omnidirectional spectrum.
    phi_orbit_box: the flight direction of the radar, i.e., the angle between the flight direction and the north direction in the clockwise rotation.

    The wave mode for S1A extract wave information from 20km x 20km geo boxes. We reconstruct the polygon of the geo box in S1A product given the box centroid and flight direction based on the radar parameters.
    Ref: https://sentiwiki.copernicus.eu/web/s1-mission#S1-Mission-Wave

    Output:
    -------
    vertices: A list of 4 points, each point is a tuple of (lon, lat) representing the corners of the geo box.
    '''
    # Define size of the box (180km x 180km)
    if mod == 'swim':
        box_size = 180
    elif mod == 's1a':
        box_size = 20
    else:
        raise ValueError("Invalid mode specified. Use 'swim' or 's1a'.")
    
    # Convert box size from km to degrees (approximate)
    # More accurate conversion from km to degrees
    # Earth radius in kilometers
    earth_radius = 6371.0
    
    # Convert box size from km to radians
    box_size_rad = box_size / earth_radius
    
    # Convert to degrees - for latitude (always constant)
    lat_deg = np.degrees(box_size_rad)
    
    # For longitude, the conversion varies with latitude
    # This will be adjusted when calculating the corners
    
    # Convert phi_orbit_box from radians to standard angle
    # phi_orbit_box is the angle between flight direction and north direction (clockwise)
    phi_deg = np.degrees(phi_orbit_box)
    
    # Calculate half size for vertex displacement
    half_lat_deg = lat_deg / 2
    half_lon_deg = half_lat_deg / np.cos(np.radians(box_centroid[1]))  # Adjust for longitude at this latitude
    
    # Calculate displacement vectors for each vertex (in degrees)
    # Taking into account flight direction (phi_orbit_box)
    
    # Calculate angle perpendicular to flight track
    perp_angle = phi_deg + 90
    flight_angle = phi_deg
    
    # Calculate displacement vectors for each corner
    displacements = [
        (-half_lon_deg * np.sin(np.radians(flight_angle)) - half_lon_deg * np.sin(np.radians(perp_angle)),
         -half_lat_deg * np.cos(np.radians(flight_angle)) - half_lat_deg * np.cos(np.radians(perp_angle))),
        
        (half_lon_deg * np.sin(np.radians(flight_angle)) - half_lon_deg * np.sin(np.radians(perp_angle)),
         half_lat_deg * np.cos(np.radians(flight_angle)) - half_lat_deg * np.cos(np.radians(perp_angle))),
        
        (half_lon_deg * np.sin(np.radians(flight_angle)) + half_lon_deg * np.sin(np.radians(perp_angle)),
         half_lat_deg * np.cos(np.radians(flight_angle)) + half_lat_deg * np.cos(np.radians(perp_angle))),
        
        (-half_lon_deg * np.sin(np.radians(flight_angle)) + half_lon_deg * np.sin(np.radians(perp_angle)),
         -half_lat_deg * np.cos(np.radians(flight_angle)) + half_lat_deg * np.cos(np.radians(perp_angle))),
    ]
    
    # Calculate the coordinates of each vertex
    # Calculate the coordinates of each vertex
    vertices = [(box_centroid[0] + dx, box_centroid[1] + dy) for dx, dy in displacements]
    vertices.append(vertices[0])

    polygon = Polygon(vertices)  # Ensure the polygon is closed
    return vertices, polygon


if __name__ == "__main__":
    args = parse_args()
    swim_l2_path = args.swim_l2_path
    wavetac_l3_path = args.wavetac_l3_path
    if not os.path.exists(swim_l2_path):
        raise FileNotFoundError(f"SWIM L2 path '{swim_l2_path}' does not exist.")
    if not os.path.exists(wavetac_l3_path):
        raise FileNotFoundError(f"WaveTac L3 path '{wavetac_l3_path}' does not exist.")
    if args.o:
        output_dir = args.o
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory at: {output_dir}")
        else:
            print(f"Using existing output directory at: {output_dir}")

    year = args.year
    month = args.month
    
    # predefine the assumption for homogenous sea state time and space interval
    homo_sea_time = np.timedelta64(homo_time, 'D')
    # Convert year and month to np.datetime64 if provided
    target_date = None
    if year is not None and month is not None:
        # Create a datetime64 object for the first day of the specified month
        target_date = np.datetime64(f'{year:04d}-{month:02d}-01', 'D')
        print(f"Processing data for {year:04d}-{month:02d}")
    else:
        raise ValueError("In this version, you must specify the year and month to be processed.")
    # Main processing code would go here

    # 0.1 Find all netCDF files in the SWIM L2 directory
    swim_l2_files = []
    for root, dirs, files in os.walk(swim_l2_path):
        # Look for netCDF files (common extensions are .nc, .nc4, .netcdf)
        nc_files = glob.glob(os.path.join(root, "*.nc")) + \
                   glob.glob(os.path.join(root, "*.nc4")) + \
                   glob.glob(os.path.join(root, "*.netcdf"))
        swim_l2_files.extend(nc_files)

    print(f"Found {len(swim_l2_files)} SWIM L2 netCDF files")

    # 0.2 Find all netCDF files in the WaveTac L3 directory
    wavetac_l3_files = []
    for root, dirs, files in os.walk(wavetac_l3_path):
        # Look for netCDF files
        nc_files = glob.glob(os.path.join(root, "*.nc")) + \
                   glob.glob(os.path.join(root, "*.nc4")) + \
                   glob.glob(os.path.join(root, "*.netcdf"))
        wavetac_l3_files.extend(nc_files)
    
    # ignore specific location files.
    
    wavetac_l3_files = [f for f in wavetac_l3_files if '_lon_' not in os.path.basename(f) and '_lat_' not in os.path.basename(f)]

    print(f"Found {len(wavetac_l3_files)} WaveTac L3 netCDF files")

    # 0.3 Create timetable for swim and wavetac files based on their time phase, only the date part is used.
    time_pattern = re.compile(r'(\d{8}T\d{6})')

    # Dictionary to store WaveTac L3 files by their time phase
    wavetac_l3_timetable = {}

    # Extract time from WaveTac L3 filenames
    for wt_file in wavetac_l3_files:
        filename = os.path.basename(wt_file)
        time_match = time_pattern.search(filename)

        if time_match:
            time_str = time_match.group(1)
            # Convert the time string to datetime64 format
            # Format: YYYYMMDDTHHMMSSZ -> YYYY-MM-DDTHH:MM:SS
            formatted_time = f"{time_str[:4]}-{time_str[4:6]}-{time_str[6:8]}T{time_str[9:11]}:{time_str[11:13]}:{time_str[13:15]}"
            time_np = np.datetime64(formatted_time)
            # Truncate the datetime to just the date part (ignore time)
            time_np = time_np.astype('datetime64[D]')
            
            # Store in the dictionary
            if time_np not in wavetac_l3_timetable:
                wavetac_l3_timetable[time_np] = []
            wavetac_l3_timetable[time_np].append(wt_file)
    
    swim_l2_timetable = {}
    # Extract time from SWIM L2 filenames
    for sl_file in swim_l2_files:
        filename = os.path.basename(sl_file)
        time_match = time_pattern.search(filename)

        if time_match:
            # only the first start measurement time is used.
            time_str = time_match.group(1)
            # Convert the time string to datetime64 format
            formatted_time = f"{time_str[:4]}-{time_str[4:6]}-{time_str[6:8]}T{time_str[9:11]}:{time_str[11:13]}:{time_str[13:15]}"
            time_np = np.datetime64(formatted_time)
            # Truncate the datetime to just the date part (ignore time)
            time_np = time_np.astype('datetime64[D]')
            
            # Store in the dictionary
            if time_np not in swim_l2_timetable:
                swim_l2_timetable[time_np] = []
            swim_l2_timetable[time_np].append(sl_file)
    
    
    # 0.4 Filter and sort the timetables by date
    # First, filter both timetables to keep only entries from the specified month
    if target_date is not None:
        # Calculate the next month to set the upper bound
        target_month_end = np.datetime64(f'{year:04d}-{month+1:02d}-01', 'D') if month < 12 else np.datetime64(f'{year+1:04d}-01-01', 'D')
        
        # Filter timetables to keep only dates within the target month
        filtered_wavetac_l3_timetable = {date: files for date, files in wavetac_l3_timetable.items() 
                                        if target_date - homo_sea_time <= date < target_month_end + homo_sea_time}
        
        filtered_swim_l2_timetable = {date: files for date, files in swim_l2_timetable.items() 
                                     if target_date <= date < target_month_end}
        
        # Replace original timetables with filtered versions
        wavetac_l3_timetable = filtered_wavetac_l3_timetable
        swim_l2_timetable = filtered_swim_l2_timetable

    # Sort timetables by date in ascending order
    swim_l2_timetable = dict(sorted(swim_l2_timetable.items()))
    wavetac_l3_timetable = dict(sorted(wavetac_l3_timetable.items()))

    # 0.5 Validate the effectiveness of the found netcdf files.
    for sw_date, sw_files in swim_l2_timetable.items():
        # check if each sw_file could be opened
        valid_ncfiles = []
        for sw_file in sw_files:
            try:
                sw_da = xr.open_dataset(sw_file, decode_times=False)
            except Exception as e:
                print(f"[Warning] Failed to open SWIM L2 file {sw_file}: {e}")
                continue

            try:
                to_test_variables = ['flag_sigma0_shape_box', 
                                        'flag_sigma0_slope_box', 
                                        'flag_sigma0_mean_box', 
                                        'wf_surf_ocean_index_box', 
                                        'nadir_rain_index_box',
                                        'pp_mean',
                                        'k_spectra',
                                        'phi_vector',
                                        'time_nadir_l2',
                                        'min_lat_l2',
                                        'max_lat_l2',
                                        'min_lon_l2',
                                        'max_lon_l2',
                                        'phi_orbit_box',
                                        'u10_ecmwf',
                                        'v10_ecmwf']
                for var in to_test_variables:
                    _ = sw_da[var]
            except KeyError as ke:
                Warning_msg = f"[Warning] KeyError in SWIM L2 file {sw_file}: {ke}. Skipping this file."
                print(Warning_msg)
                continue
            except Exception as e:
                print(f"[Warning] Unexpected error in SWIM L2 file {sw_file}: {e}. Skipping this file.")
                continue
            sw_da.close()
            valid_ncfiles.append(sw_file)
        swim_l2_timetable[sw_date] = valid_ncfiles
    
    # 0.51 validate the effectiveness of the found netcdf files for wavetac
    for wt_date, wt_files in wavetac_l3_timetable.items():
        # check if each wt_file could be opened
        valid_ncfiles = []
        for wt_file in wt_files:
            try:
                wt_da = xr.open_dataset(wt_file, group='obs_params')
            except Exception as e:
                print(f"[Warning] Failed to open WaveTac L3 file {wt_file}: {e}. Skipping this file.")
                continue
            try:
                to_test_variables = ['L2_partition_quality_flag', 
                                        'wave_spec', 
                                        'wavenumber_spec', 
                                        'direction_spec', 
                                        'time',
                                        'latitude',
                                        'longitude',
                                        'resolution_spec']
                for var in to_test_variables:
                    _ = wt_da[var]
            except KeyError as ke:
                Warning_msg = f"[Warning] KeyError in WaveTac L3 file {wt_file}: {ke}. Skipping this file."
                print(Warning_msg)
                continue
            except Exception as e:
                print(f"[Warning] Unexpected error in WaveTac L3 file {wt_file}: {e}. Skipping this file.")
                continue
            wt_da.close()
            valid_ncfiles.append(wt_file)
        wavetac_l3_timetable[wt_date] = valid_ncfiles

    swim_start_date = list(swim_l2_timetable.keys())[0]
    swim_end_date = list(swim_l2_timetable.keys())[-1]
    to_process_file_num_swim = [len(x) for x in swim_l2_timetable.values()]
    to_process_file_num_swim = sum(to_process_file_num_swim)

    wavetac_start_date = list(wavetac_l3_timetable.keys())[0]
    wavetac_end_date = list(wavetac_l3_timetable.keys())[-1]

    to_process_file_num_wavetac = [len(x) for x in wavetac_l3_timetable.values()]
    to_process_file_num_wavetac = sum(to_process_file_num_wavetac)

    print(f"Found {to_process_file_num_swim} days of SWIM L2 data from {swim_start_date} to {swim_end_date}")
    print(f"Found {to_process_file_num_wavetac} corresponding days of WaveTac L3 data from {wavetac_start_date} to {wavetac_end_date}")
    
    # 1.0 Match SWIM L2 files with WaveTac L3 measurements'
    # 1.1 Pre-filtering the SWIM and wavetac data according to the quality indices listed below.
    '''
    * Quality control indices:
        ^ swim:                                     ^ wavetac:
        'flag_sigma0_shape_box',                    'L2_partition_quality_flag',
        'flag_sigma0_slope_box',
        'flag_sigma0_mean_box',
        'sea_ice_coverage' (not included in the current version),
        'land_coverage' (not included in the current version),
        'wf_surf_ocean_index_box',
        'nadir_rain_index_box'
    '''

    quality_filtered_swim_l2_indices = {}

    for sw_date, sw_files in swim_l2_timetable.items():
        # Perform pre-filtering based on quality control indices
        valid_swim_indices = []

        for sw_file in sw_files:
            try:
                sw_da = xr.open_dataset(sw_file, decode_times=False)
            except Exception as e:
                print(f"[Warning] Failed to open SWIM L2 file {sw_file}: {e}. Skipping this file.")
                continue
            sw_file_basename = os.path.basename(sw_file)
            try:
                # Check quality flags - a box is valid only if ALL values are 0 for all phi angles and pos/neg directions
                valid_shape = sw_da['flag_sigma0_shape_box'].isel(n_beam_l2=to_fuse_swim_antenna).all(dim=('n_phi', 'n_posneg')) == 0
                valid_slope = sw_da['flag_sigma0_slope_box'].isel(n_beam_l2=to_fuse_swim_antenna).all(dim=('n_phi', 'n_posneg')) == 0
                valid_mean = sw_da['flag_sigma0_mean_box'].isel(n_beam_l2=to_fuse_swim_antenna).all(dim=('n_phi', 'n_posneg')) == 0
                
                # Additional quality checks if available
                ocean_coverage_ok = sw_da['wf_surf_ocean_index_box'].values >= 90
                rain_flag_ok = sw_da['nadir_rain_index_box'].values <= 10
            except KeyError as e:
                print(f"[Warning] KeyError in file {sw_file_basename}: {e}. Skipping this file.")
                continue
            all_conditions_met = valid_shape & valid_slope & valid_mean & ocean_coverage_ok & rain_flag_ok

            n_box = sw_da.sizes['n_box']
            valid_ratios = np.sum(all_conditions_met) / n_box
            print(f"SWIM L2 date {sw_date}, file name {sw_file_basename} : Valid ratio of boxes = {valid_ratios:.2f}")
            
            valid_swim_indices.append(np.where(all_conditions_met)[0]) # Get indices of valid boxes
            sw_da.close()  # Close the dataset to free resources
        
        if quality_filtered_swim_l2_indices.get(sw_date, None) is None:
            quality_filtered_swim_l2_indices[sw_date] = valid_swim_indices
        else:
            raise ValueError(f"Duplicate SWIM L2 date found: {sw_date}. Please check the input files.")
    
    quality_filtered_wavetac_l3_indices = {}
    for wt_date, wt_files in wavetac_l3_timetable.items():
        # Perform pre-filtering based on quality control indices
        valid_wavetac_indices = []
        for wt_file in wt_files:
            wt_da = xr.open_dataset(wt_file, group='obs_params')
            wt_file_basename = os.path.basename(wt_file)
            # Check quality flags - a box is valid only if ALL values are 0 for all obs
            try:
                partition_flag_ok = wt_da['L2_partition_quality_flag'].values <= 2  # Assuming 0-2 are valid flags
            except KeyError as e:
                print(f"[Warning] KeyError in file {wt_file_basename}: {e}. Skipping this file.")
                continue
            
            n_obs = wt_da.sizes['obs']
            valid_ratios = np.sum(partition_flag_ok) / n_obs
            print(f"WaveTac L3 date {wt_date}, file name {wt_file_basename} : Valid ratio of observations = {valid_ratios:.2f}")

            valid_wavetac_indices.append(np.where(partition_flag_ok)[0])  # Get indices of valid observations
            wt_da.close()  # Close the dataset to free resources
        
        if quality_filtered_wavetac_l3_indices.get(wt_date, None) is None:
            quality_filtered_wavetac_l3_indices[wt_date] = valid_wavetac_indices
        else:
            raise ValueError(f"Duplicate WaveTac L3 date found: {wt_date}. Please check the input files.")
        
        # Proceed with merging or further processing as needed
        # ...

    # 1.2 Merge the SWIM and WaveTac data 
    # For tracking which dates we've already processed to avoid redundant reads
    reading_wavetac_files = {}

    for swim_date, swim_files in swim_l2_timetable.items():
        output_file = os.path.join(output_dir, f'fusion_swim_sar_{swim_date}.nc')
        if os.path.exists(output_file):
            print(f"Output file {output_file} already exists. Skipping processing for date {swim_date}.")
            continue
        # print("[Debugging] Openned here!!!")
        # swim_date = np.datetime64('2022-01-28', 'D')  # Ensure swim_date is a datetime64 object
        reading_swim_files = {}
        reading_swim_files[swim_date] = [xr.open_dataset(sw_file, decode_times=False) for sw_file in swim_l2_timetable[swim_date]]

        # Define the time window for this SWIM date
        wavetac_start_date = swim_date - homo_sea_time
        wavetac_end_date = swim_date + homo_sea_time
        
        print(f"Matching SWIM L2 files for date: {swim_date} (SAR window: {wavetac_start_date} to {wavetac_end_date})")
        current_wavetac_date = wavetac_start_date
        
        # close the previous read wavetac files before read new files.
        to_drop_date = current_wavetac_date - np.timedelta64(1, 'D')
        if reading_wavetac_files.get(to_drop_date, None) is not None:
            to_drop_file_num = len(reading_wavetac_files[to_drop_date])
            for da in reading_wavetac_files[to_drop_date]:
                da.close() # explicitly close the files to reduce expense
            del reading_wavetac_files[to_drop_date]
            print(f"{to_drop_file_num} files of WAVETAC in {to_drop_date} date droped.")

        while current_wavetac_date <= wavetac_end_date:
            if (reading_wavetac_files.get(current_wavetac_date, None) is None) and (wavetac_l3_timetable.get(current_wavetac_date) is not None):
                reading_wavetac_files[current_wavetac_date] = [xr.open_dataset(wt_file, group='obs_params') for wt_file in wavetac_l3_timetable[current_wavetac_date]]
            current_wavetac_date = current_wavetac_date + np.timedelta64(1, 'D')
        
        # 1.21 Merging process.
        ''' 
        The dimension to be concatenated is 'nbox' for swim, and 'obs' for wavetac.
        
        * Parameters to concatenate:
        ^ swim:             ^ wavetac:
        'pp_mean',          'wave_spec',,
        'k_spectra',        'wavenumber_spec',
        'phi_vector',       'direction_spec',
        'time_nadir_l2'     'time',
        'min_lat_l2',       'latitude',
        'max_lat_l2',       'longitude',
        'min_lon_l2',       'resolution_spec'
        'max_lon_l2',
        'phi_orbit_box', 
        'u10_ecmwf', 
        'v10_ecmwf', 

        * Parameters for the output file:
        ^ would ne named as fusion_swim_sar_{swim_date}.nc
        'slope_spec',
        'wavenumber',
        'azimuth', clockwise from north, in degrees.
        'time',
        'vertex_1',
        'vertex_2',
        'vertex_3',
        'vertex_4',
        'u10_ecmwf',
        'v10_ecmwf'
        ''
        '''

        to_match_swim_samples = {'slope_spec': [],
                                 'wavenumber': np.nan,
                                 'azimuth': np.nan,
                                 'north_heading': [],
                                 'time': [],
                                 'vertex_1': [],
                                 'vertex_2': [],
                                 'vertex_3': [],
                                 'vertex_4': [],
                                 'polygon': [],
                                 'u10_ecmwf': [],
                                 'v10_ecmwf': []} # corresponding to the parameters in the output file.
        
        for sw_da, valid_indices in zip(reading_swim_files[swim_date], quality_filtered_swim_l2_indices[swim_date]):

            # Iterate through each valid box
            for box_idx in valid_indices:
                # Extract the bounding box coordinates
                max_lat = sw_da['max_lat_l2'].isel(n_box=box_idx).max(dim="n_posneg").values
                min_lat = sw_da['min_lat_l2'].isel(n_box=box_idx).min(dim="n_posneg").values
                # Check if we need to handle the 180/-180 longitude boundary case
                # Get all longitude values for the box
                all_lons = sw_da['min_lon_l2'].isel(n_box=box_idx).values.tolist() + sw_da['max_lon_l2'].isel(n_box=box_idx).values.tolist()
                                
                # Check if the longitudes cross the international date line
                lon_diff = max(all_lons) - min(all_lons)
                if lon_diff > 180:
                    # Convert longitudes to [0, 360] range for calculation
                    all_lons_adjusted = [(lon + 360) if lon < 0 else lon for lon in all_lons]
                    max_lon = max(all_lons_adjusted)
                    min_lon = min(all_lons_adjusted)
                    # Convert back to [-180, 180] range for storage
                    max_lon = max_lon - 360 if max_lon > 180 else max_lon
                    min_lon = min_lon - 360 if min_lon > 180 else min_lon
                else:
                    # Normal case, no boundary crossing
                    max_lon = max(all_lons)
                    min_lon = min(all_lons)
                # max_lon = sw_da_filtered['max_lon_l2'][box_idx].max(dim="n_posneg").values
                # min_lon = sw_da_filtered['min_lon_l2'][box_idx].min(dim="n_posneg").values
                
                # Calculate the centroid as the average of min and max coordinates
                # Handle the special case where the box crosses the 180/-180 longitude line
                if np.abs(max_lon - min_lon) > 180:
                    # Box crosses the dateline
                    # Adjust longitudes to range [0, 360] for calculation
                    adjusted_min_lon = min_lon if min_lon >= 0 else min_lon + 360
                    adjusted_max_lon = max_lon if max_lon >= 0 else max_lon + 360
                    mean_lon = (adjusted_min_lon + adjusted_max_lon) / 2
                    # Convert back to [-180, 180] range
                    centroid_lon = mean_lon if mean_lon <= 180 else mean_lon - 360
                else:
                    centroid_lon = (max_lon + min_lon) / 2
                centroid_lat = (max_lat + min_lat) / 2
                
                # Skip boxes outside the valid latitude range
                if centroid_lat > safe_max_lat or centroid_lat < safe_min_lat:
                    continue

                # Optionally, reconstruct the polygon for each box
                phi_orbit = sw_da['phi_orbit_box'].isel(n_box=box_idx).values
                to_match_swim_samples['north_heading'].append(phi_orbit)
                swim_vertices, swim_polygon = reconstruct_radar_polygon((centroid_lon, centroid_lat), phi_orbit, mod='swim')
                for i in range(4):
                    to_match_swim_samples[f'vertex_{i+1:1d}'].append(swim_vertices[i])
                
                if to_match_swim_samples['wavenumber'] is np.nan:
                    # Initialize wavenumber if not already set
                    to_match_swim_samples['wavenumber'] = sw_da['k_spectra'].values
                if to_match_swim_samples['azimuth'] is np.nan:
                    # Initialize azimuth if not already set
                    to_match_swim_samples['azimuth'] = np.arange(7.5, 360, 15) * np.pi / 180
                
                left_slope_spec = sw_da['pp_mean'].isel(n_box=box_idx, n_posneg=0, n_beam_l2=to_fuse_swim_antenna).values
                right_slope_spec = sw_da['pp_mean'].isel(n_box=box_idx, n_posneg=1, n_beam_l2=to_fuse_swim_antenna).values
                full_slope_spec = np.concatenate((left_slope_spec/4+right_slope_spec/4, left_slope_spec/4+right_slope_spec/4), axis=1) # (nk, n_phi)
                to_match_swim_samples['slope_spec'].append(full_slope_spec)

                u10_ecmwf = sw_da['u10_ecmwf'].isel(n_box=box_idx).values.mean()
                v10_ecmwf = sw_da['v10_ecmwf'].isel(n_box=box_idx).values.mean()
                to_match_swim_samples['u10_ecmwf'].append(u10_ecmwf)
                to_match_swim_samples['v10_ecmwf'].append(v10_ecmwf)
                to_match_swim_samples['polygon'].append(swim_polygon)

                # Get the flight direction and time for this box
                time_nadir_second = sw_da['time_nadir_l2'].isel(n_box=box_idx, n_tim = 0).values
                time_nadir_us = sw_da['time_nadir_l2'].isel(n_box=box_idx, n_tim = 1).values

                time_nadir_datetime64 = swim_reference_time + np.timedelta64(int(time_nadir_second * 1e6 + time_nadir_us), 'us')
                to_match_swim_samples['time'].append(time_nadir_datetime64)
            
            sw_da.close()

        to_match_wavetac_samples = {'wave_spec': [],
                                    'wavenumber_spec': np.nan,
                                   'direction_spec': np.nan,
                                   'time': [],
                                   'latitude': [],
                                   'longitude': [],
                                   'resolution_spec': [],
                                   'vertex_1': [],
                                   'vertex_2': [],
                                   'vertex_3': [],
                                   'vertex_4': [],
                                   'polygon': []} # corresponding to the parameters in the output file.
        for wavetac_date in reading_wavetac_files.keys():
            wavetac_das_list = reading_wavetac_files[wavetac_date]
            valid_indices_list = quality_filtered_wavetac_l3_indices[wavetac_date]
        # for wavetac_das_list, valid_indices_list in zip(reading_wavetac_files.values(), quality_filtered_wavetac_l3_indices.values()):
            for wt_da, valid_indices in zip(wavetac_das_list, valid_indices_list):
                # Extract the time and polygon for each valid observation
                for obs_idx in valid_indices:
                    # Get the time and polygon for this observation
                    time_obs = wt_da['time'].isel(obs=obs_idx).values
                    # try:
                    #     time_obs = wt_da['time'].isel(obs=obs_idx).values
                    # except Exception as e:
                    #     print(f"Error extracting time for observation {obs_idx} in file {wt_da.file_name}: {e}")
                    #     continue
                    lon = wt_da['longitude'].isel(obs=obs_idx).values
                    lat = wt_da['latitude'].isel(obs=obs_idx).values
                    resolution = wt_da['resolution_spec'].isel(obs=obs_idx).values
                    if to_match_wavetac_samples['wavenumber_spec'] is np.nan:
                        # Initialize wavenumber_spec if not already set
                        to_match_wavetac_samples['wavenumber_spec'] = wt_da['wavenumber_spec'].values
                    if to_match_wavetac_samples['direction_spec'] is np.nan:
                        # Initialize direction_spec if not already set
                        to_match_wavetac_samples['direction_spec'] = (wt_da['direction_spec'].values + 2.5) * np.pi / 180

                    # Skip observations outside the valid latitude range
                    if lat > safe_max_lat or lat < safe_min_lat:
                        continue
                    
                    # Reconstruct the polygon for this observation
                    wavetac_vertices, wavetac_polygon = reconstruct_radar_polygon((lon, lat), 0, mod='s1a')

                    to_match_wavetac_samples['time'].append(time_obs)
                    to_match_wavetac_samples['latitude'].append(lat)
                    to_match_wavetac_samples['longitude'].append(lon)
                    to_match_wavetac_samples['resolution_spec'].append(resolution)
                    to_match_wavetac_samples['wave_spec'].append(wt_da['wave_spec'].isel(obs=obs_idx).values)
                    for i in range(4):
                        to_match_wavetac_samples[f'vertex_{i+1:1d}'].append(wavetac_vertices[i])
                    to_match_wavetac_samples['polygon'].append(wavetac_polygon)

        # 1.3 Matching the SWIM and SAR data
        matching_metrics = wavetools.spatio_temporal_match(to_match_swim_samples['time'],
                                        to_match_wavetac_samples['time'],
                                        to_match_swim_samples['polygon'],
                                        to_match_wavetac_samples['polygon'])
        if len(matching_metrics) == 0:
            Warning_msg = f"[Warning] No matching metrics found for SWIM date {swim_date}. Skipping this date."
            print(Warning_msg)
            continue

        matching_metrics = np.array(matching_metrics)

        for i in range(3):
            print(f"Matching metrics for swim date {swim_date}: {matching_metrics[i, :].tolist()}")

        df_matching_metrics = pd.DataFrame(matching_metrics, columns=['swim_idx', 'wavetac_idx', 'criteria_value'])
        df_matching_metrics.to_csv(os.path.join(logs_dir, f'matching_metrics_{swim_date}.csv'), index=False)
        print(f"Matching metrics saved to {os.path.join(logs_dir, f'matching_metrics_{swim_date}.csv')}")

        matched_swim_samples = {'slope_spec': [],
                                'fused_slope_spec': [],
                                'l2p_slope_spec': [],
                                'swim_partitions': [],
                                'segmentation_label': [],
                                'st_mag': [],
                                'st_direc': [],
                                'st_trans': [],
                                'st_mag_l2p': [],
                                'st_direc_l2p': [],
                                'st_trans_l2p': [],
                                'st_depth': [0, 8, 16],
                                'wavenumber': np.nan,
                                'azimuth': np.nan,
                                'north_heading': [],
                                'time': [],
                                'vertex_1': [],
                                'vertex_2': [],
                                'vertex_3': [],
                                'vertex_4': [],
                                'polygon': [],
                                'u10_ecmwf': [],
                                'v10_ecmwf': []}
        matched_wavetac_samples = {'wave_spec': [],
                                    'wavenumber_spec': np.nan,
                                   'direction_spec': np.nan,
                                   'time': [],
                                   'latitude': [],
                                   'longitude': [],
                                   'resolution_spec': [],
                                   'vertex_1': [],
                                   'vertex_2': [],
                                   'vertex_3': [],
                                   'vertex_4': [],
                                   'polygon': []}
        
        # 1.4 align merged data
        log_count = 0
        log_interval = 10
        for matched_swim_idx, matched_wavetac_idx in zip(matching_metrics[:, 0].astype(int), matching_metrics[:, 1].astype(int)):
            log_count += 1
            if log_count % log_interval == 0:
                print(f"Processed {log_count} matches so far. Total matches: {len(matching_metrics[:, 0])}")
            
            # 2.0 Fuse swim and wavetac spectra
            swim_phi_spectra, swim_k_spectra = np.meshgrid(to_match_swim_samples['azimuth'], to_match_swim_samples['wavenumber'])
            wavetac_phi_spectra, wavetac_k_spectra = np.meshgrid(to_match_wavetac_samples['direction_spec'], to_match_wavetac_samples['wavenumber_spec'])

            # Note: The segmentation and matching algorithm may occasionally produce errors in edge cases.
            # We gracefully handle these exceptions to ensure uninterrupted batch processing.
            try:
                swim_spec_fused, partition_index_swim, partition_index_sar, removed_partitions, remained_partitions, to_merge_wind_parts = ambiguity_removal.sar_swim_fusion(swim_spec=to_match_swim_samples['slope_spec'][matched_swim_idx],
                                                swim_k_spectra=swim_k_spectra,
                                                swim_phi_spectra=swim_phi_spectra,
                                                # swim_heading=to_match_swim_samples['north_heading'][matched_swim_idx],
                                                swim_heading=0,
                                                sar_spec=to_match_wavetac_samples['wave_spec'][matched_wavetac_idx],
                                                sar_k_spectra=wavetac_k_spectra,
                                                sar_phi_spectra=wavetac_phi_spectra,
                                                sar_heading=0,
                                                u10=to_match_swim_samples['u10_ecmwf'][matched_swim_idx],
                                                v10=to_match_swim_samples['v10_ecmwf'][matched_swim_idx])
            except ValueError as e:
                print(f"[Error]: Error during fusion for SWIM date {to_match_swim_samples['time'][matched_swim_idx]} at index {matched_swim_idx}. Error information is: {e}")
                continue

            if len(removed_partitions) == 0 and len(to_merge_wind_parts) == 0:
                print(f"[Warning]: No valid partitions found for SWIM date {to_match_swim_samples['time'][matched_swim_idx]} at index {matched_swim_idx}. Skipping this match.")
                continue

            if np.all(np.isnan(swim_spec_fused)):
                print(f"[Warning]: Nan ratio too high for SWIM date {to_match_swim_samples['time'][matched_swim_idx]} at index {matched_swim_idx}. Skipping this match.")
                continue

            swim_spec_l2p = ambiguity_removal.remove_ambiguity_accord_wind(swim_spec=to_match_swim_samples['slope_spec'][matched_swim_idx],
                                              swim_k_spectra=swim_k_spectra,
                                              swim_phi_spectra=swim_phi_spectra,
                                              swim_heading=to_match_swim_samples['north_heading'][matched_swim_idx],
                                              u10=to_match_swim_samples['u10_ecmwf'][matched_swim_idx],
                                              v10=to_match_swim_samples['v10_ecmwf'][matched_swim_idx])
            
            if np.all(np.isnan(swim_spec_l2p)):
                print(f"[Warning]: Nan ratio too high for WAVETAC date {to_match_wavetac_samples['time'][matched_wavetac_idx]} at {matched_wavetac_idx}. Skipping this match.")
                continue

            # save the remove causibility. 1 if removed by SAR, 2 if merged with wind. 0 for non removed.
            segmentation_label = np.zeros_like(partition_index_swim)
            for part_idx in removed_partitions:
                segmentation_label[partition_index_swim == part_idx] = 1
            
            for part_idx in to_merge_wind_parts:
                segmentation_label[partition_index_swim == part_idx] = 2
            
            matched_swim_samples['segmentation_label'].append(segmentation_label)
            matched_swim_samples['swim_partitions'].append(partition_index_swim)

            # align to the geo north heading
            swim_north_heading = to_match_swim_samples['north_heading'][matched_swim_idx]
            swim_phi_spectra = np.mod(swim_phi_spectra + swim_north_heading, 2*np.pi)
            sorted_indices = np.argsort(swim_phi_spectra[0, :])
            swim_phi_spectra_north_tuned = swim_phi_spectra[:, sorted_indices]

            st_mag_profile, st_dic_profile = [], []
            st_mag_l2p_profile, st_dic_l2p_profile = [], []
            for st_depth in matched_swim_samples['st_depth']:
                st_mag_fusion, st_dic_fusion = wavetools.cal_st(swim_spec_fused, swim_phi_spectra_north_tuned, swim_k_spectra, 'skth', depth=st_depth)
                st_mag_profile.append(st_mag_fusion)
                st_dic_profile.append(st_dic_fusion)
                st_mag_l2p, st_dic_l2p = wavetools.cal_st(swim_spec_l2p, swim_phi_spectra_north_tuned, swim_k_spectra, 'skth', depth=st_depth)
                st_mag_l2p_profile.append(st_mag_l2p)
                st_dic_l2p_profile.append(st_dic_l2p)
            
            matched_swim_samples['fused_slope_spec'].append(swim_spec_fused)
            matched_swim_samples['l2p_slope_spec'].append(swim_spec_l2p)
            matched_swim_samples['st_mag'].append(st_mag_profile)
            matched_swim_samples['st_direc'].append(st_dic_profile)
            matched_swim_samples['st_mag_l2p'].append(st_mag_l2p_profile)
            matched_swim_samples['st_direc_l2p'].append(st_dic_l2p_profile)

            matched_swim_samples['slope_spec'].append(to_match_swim_samples['slope_spec'][matched_swim_idx])
            if matched_swim_samples['wavenumber'] is np.nan:
                # Initialize wavenumber if not already set
                matched_swim_samples['wavenumber'] = to_match_swim_samples['wavenumber']
            if matched_swim_samples['azimuth'] is np.nan:
                # Initialize azimuth if not already set
                matched_swim_samples['azimuth'] = to_match_swim_samples['azimuth']
            matched_swim_samples['north_heading'].append(to_match_swim_samples['north_heading'][matched_swim_idx])
            matched_swim_samples['time'].append(to_match_swim_samples['time'][matched_swim_idx])
            matched_swim_samples['vertex_1'].append(to_match_swim_samples['vertex_1'][matched_swim_idx])
            matched_swim_samples['vertex_2'].append(to_match_swim_samples['vertex_2'][matched_swim_idx])
            matched_swim_samples['vertex_3'].append(to_match_swim_samples['vertex_3'][matched_swim_idx])
            matched_swim_samples['vertex_4'].append(to_match_swim_samples['vertex_4'][matched_swim_idx])
            matched_swim_samples['polygon'].append(to_match_swim_samples['polygon'][matched_swim_idx])
            matched_swim_samples['u10_ecmwf'].append(to_match_swim_samples['u10_ecmwf'][matched_swim_idx])
            matched_swim_samples['v10_ecmwf'].append(to_match_swim_samples['v10_ecmwf'][matched_swim_idx])

            matched_wavetac_samples['wave_spec'].append(to_match_wavetac_samples['wave_spec'][matched_wavetac_idx])
            if matched_wavetac_samples['wavenumber_spec'] is np.nan:
                # Initialize wavenumber_spec if not already set
                matched_wavetac_samples['wavenumber_spec'] = to_match_wavetac_samples['wavenumber_spec']
            if matched_wavetac_samples['direction_spec'] is np.nan:
                # Initialize direction_spec if not already set
                matched_wavetac_samples['direction_spec'] = to_match_wavetac_samples['direction_spec']
            matched_wavetac_samples['time'].append(to_match_wavetac_samples['time'][matched_wavetac_idx])
            matched_wavetac_samples['latitude'].append(to_match_wavetac_samples['latitude'][matched_wavetac_idx])
            matched_wavetac_samples['longitude'].append(to_match_wavetac_samples['longitude'][matched_wavetac_idx])
            matched_wavetac_samples['resolution_spec'].append(to_match_wavetac_samples['resolution_spec'][matched_wavetac_idx])
            matched_wavetac_samples['vertex_1'].append(to_match_wavetac_samples['vertex_1'][matched_wavetac_idx])
            matched_wavetac_samples['vertex_2'].append(to_match_wavetac_samples['vertex_2'][matched_wavetac_idx])
            matched_wavetac_samples['vertex_3'].append(to_match_wavetac_samples['vertex_3'][matched_wavetac_idx])
            matched_wavetac_samples['vertex_4'].append(to_match_wavetac_samples['vertex_4'][matched_wavetac_idx])
            matched_wavetac_samples['polygon'].append(to_match_wavetac_samples['polygon'][matched_wavetac_idx])

        # to_save_parameters = {}
        # to_save_parameters['slope_spec'] = np.array(matched_swim_samples['slope_spec'])
        # to_save_parameters['fused_slope_spec'] = np.array(matched_swim_samples['fused_slope_spec'])
        # to_save_parameters['l2p_slope_spec'] = np.array(matched_swim_samples['l2p_slope_spec'])
        # to_save_parameters['st_mag'] = np.array(matched_swim_samples['st_mag'])
        # to_save_parameters['st_direc'] = np.array(matched_swim_samples['st_direc'])
        # to_save_parameters['st_mag_l2p'] = np.array(matched_swim_samples['st_mag_l2p'])
        # to_save_parameters['st_direc_l2p'] = np.array(matched_swim_samples['st_direc_l2p'])
        # to_save_parameters['st_depth'] = np.array(matched_swim_samples['st_depth'])
        # to_save_parameters['wavenumber'] = matched_swim_samples['wavenumber']
        # to_save_parameters['azimuth'] = matched_swim_samples['azimuth']
        # to_save_parameters['north_heading'] = np.array(matched_swim_samples['north_heading'])
        # to_save_parameters['time'] = np.array(matched_swim_samples['time'])
        # to_save_parameters['vertex_1'] = np.array(matched_swim_samples['vertex_1'])
        # to_save_parameters['vertex_2'] = np.array(matched_swim_samples['vertex_2'])
        # to_save_parameters['vertex_3'] = np.array(matched_swim_samples['vertex_3'])
        # to_save_parameters['vertex_4'] = np.array(matched_swim_samples['vertex_4'])
        # to_save_parameters['u10_ecmwf'] = np.array(matched_swim_samples['u10_ecmwf'])
        # to_save_parameters['v10_ecmwf'] = np.array(matched_swim_samples['v10_ecmwf'])

        # 3.0 Create and save the netCDF file with proper dimensions
        n_time = len(matched_swim_samples['time'])
        # n_k = len(matched_swim_samples['wavenumber'])
        # n_phi = len(matched_swim_samples['azimuth'])
        # n_depth = len(matched_swim_samples['st_depth'])
        # n_vertex = 4

        # Check if we have any matches before attempting to save
        if n_time > 0:
            to_save_parameters = {}
            to_save_parameters['slope_spec'] = np.array(matched_swim_samples['slope_spec'])
            to_save_parameters['fused_slope_spec'] = np.array(matched_swim_samples['fused_slope_spec'])
            to_save_parameters['l2p_slope_spec'] = np.array(matched_swim_samples['l2p_slope_spec'])
            to_save_parameters['st_mag'] = np.array(matched_swim_samples['st_mag'])
            to_save_parameters['st_direc'] = np.array(matched_swim_samples['st_direc'])
            to_save_parameters['st_mag_l2p'] = np.array(matched_swim_samples['st_mag_l2p'])
            to_save_parameters['st_direc_l2p'] = np.array(matched_swim_samples['st_direc_l2p'])
            to_save_parameters['st_depth'] = np.array(matched_swim_samples['st_depth'])
            to_save_parameters['wavenumber'] = matched_swim_samples['wavenumber']
            to_save_parameters['azimuth'] = matched_swim_samples['azimuth']
            to_save_parameters['north_heading'] = np.array(matched_swim_samples['north_heading'])
            to_save_parameters['time'] = np.array(matched_swim_samples['time'])
            to_save_parameters['vertex_1'] = np.array(matched_swim_samples['vertex_1'])
            to_save_parameters['vertex_2'] = np.array(matched_swim_samples['vertex_2'])
            to_save_parameters['vertex_3'] = np.array(matched_swim_samples['vertex_3'])
            to_save_parameters['vertex_4'] = np.array(matched_swim_samples['vertex_4'])
            to_save_parameters['u10_ecmwf'] = np.array(matched_swim_samples['u10_ecmwf'])
            to_save_parameters['v10_ecmwf'] = np.array(matched_swim_samples['v10_ecmwf'])

            # 3.0 Create and save the netCDF file with proper dimensions
            # n_time = len(matched_swim_samples['time'])
            n_k = len(matched_swim_samples['wavenumber'])
            n_phi = len(matched_swim_samples['azimuth'])
            n_depth = len(matched_swim_samples['st_depth'])
            n_vertex = 4

            # Create dimensions and coordinates
            coords = {
                'n_time': np.arange(n_time),
                'wavenumber': ('n_k', matched_swim_samples['wavenumber']),
                'azimuth': ('n_phi', matched_swim_samples['azimuth']),
                'st_depth': ('n_depth', matched_swim_samples['st_depth'])
            }
            
            # Create the xarray dataset
            ds = xr.Dataset(
                data_vars={
                    'slope_spec': (['n_time', 'n_k', 'n_phi'], to_save_parameters['slope_spec']),
                    'partition_index_swim': (['n_time', 'n_k', 'n_phi'], np.array(matched_swim_samples['swim_partitions'])),
                    'segmentation_label': (['n_time', 'n_k', 'n_phi'], np.array(matched_swim_samples['segmentation_label'])),
                    'fused_slope_spec': (['n_time', 'n_k', 'n_phi'], to_save_parameters['fused_slope_spec']),
                    'l2p_slope_spec': (['n_time', 'n_k', 'n_phi'], to_save_parameters['l2p_slope_spec']),
                    'st_mag': (['n_time', 'n_depth'], to_save_parameters['st_mag']),
                    'st_direc': (['n_time', 'n_depth'], to_save_parameters['st_direc']),
                    'st_mag_l2p': (['n_time', 'n_depth'], to_save_parameters['st_mag_l2p']),
                    'st_direc_l2p': (['n_time', 'n_depth'], to_save_parameters['st_direc_l2p']),
                    'north_heading': ('n_time', to_save_parameters['north_heading']),
                    'time': ('n_time', to_save_parameters['time']),
                    'u10_ecmwf': ('n_time', to_save_parameters['u10_ecmwf']),
                    'v10_ecmwf': ('n_time', to_save_parameters['v10_ecmwf']),
                    'vertex_lon': (['n_time', 'n_vertex'], np.column_stack([
                        [to_save_parameters['vertex_1'][i][0] for i in range(n_time)],
                        [to_save_parameters['vertex_2'][i][0] for i in range(n_time)],
                        [to_save_parameters['vertex_3'][i][0] for i in range(n_time)],
                        [to_save_parameters['vertex_4'][i][0] for i in range(n_time)],
                    ])),
                    'vertex_lat': (['n_time', 'n_vertex'], np.column_stack([
                        [to_save_parameters['vertex_1'][i][1] for i in range(n_time)],
                        [to_save_parameters['vertex_2'][i][1] for i in range(n_time)],
                        [to_save_parameters['vertex_3'][i][1] for i in range(n_time)],
                        [to_save_parameters['vertex_4'][i][1] for i in range(n_time)],
                    ]))
                },
                coords=coords,
                attrs={
                    'description': 'The Fused SWIM Slope Spectrum Dataset',
                    'creation_date': str(np.datetime64('now')),
                    'source': f"SWIM L2 data and WaveTac L3 data of Sentinel-1A"
                }
            )
            
            # Add variable attributes
            ds['slope_spec'].attrs = {
                'units': 'm^2/rad',
                'long_name': 'Original SWIM slope spectrum',
                'description': 'Original omnidirectional slope spectrum from SWIM L2 data'
            }

            ds['partition_index_swim'].attrs = {
                'units': 'dimensionless',
                'long_name': 'SWIM partition index',
                'description': 'Index of the SWIM partitions used in the fusion process'
            }

            ds['segmentation_label'].attrs = {
                'units': 'dimensionless',
                'long_name': 'Segmentation label for SWIM partitions',
                'description': 'Indicates which partitions were removed during the fusion process: 0 for no removal, 1 if removed based on SAR, 2 if merged as wind partitions'
            }
            
            ds['fused_slope_spec'].attrs = {
                'units': 'm^2/rad',
                'long_name': 'Fused slope spectrum',
                'description': 'Slope spectrum after SAR-based ambiguity removal'
            }
            
            ds['l2p_slope_spec'].attrs = {
                'units': 'm^2/rad',
                'long_name': 'L2P slope spectrum',
                'description': 'Slope spectrum after wind-based ambiguity removal used in SWIM L2P products'
            }
            
            ds['st_mag'].attrs = {
                'units': 'cm/s',
                'long_name': 'The magnitude of Stokes drift at different depths',
                'description': 'The magnitude of Stokes drift at different depths after fusion of SWIM and WaveTac spectra'
            }
            
            ds['st_direc'].attrs = {
                'units': 'degrees (clockwise from north)',
                'long_name': 'The direction of Stokes drift at different depths',
                'description': 'The direction of Stokes drift at different depths after fusion of SWIM and WaveTac spectra'
            }
            
            ds['st_mag_l2p'].attrs = {
                'units': 'm',
                'long_name': 'The magnitude of Stokes drift at different depths',
                'description': 'The magnitude of Stokes drift at different depths after wind-based ambiguity removal used in SWIM L2P products'
            }
            
            ds['st_direc_l2p'].attrs = {
                'units': 'degrees (clockwise from north)',
                'long_name': 'The direction of Stokes drift at different depths',
                'description': 'The direction of Stokes drift at different depths after wind-based ambiguity removal used in SWIM L2P products'
            }
            
            ds['north_heading'].attrs = {
                'units': 'rad',
                'long_name': 'SWIM north heading angle',
                'description': 'Angle between SWIM flight direction and north, measured clockwise'
            }
            
            # ds['time'].attrs = {
            #     'units': 'datetime64',
            #     'long_name': 'Observation time',
            #     'description': 'Time of the SWIM observation'
            # }
            
            ds['u10_ecmwf'].attrs = {
                'units': 'm/s',
                'long_name': 'ECMWF eastward wind component',
                'description': 'Eastward wind component at 10m from ECMWF'
            }
            
            ds['v10_ecmwf'].attrs = {
                'units': 'm/s',
                'long_name': 'ECMWF northward wind component',
                'description': 'Northward wind component at 10m from ECMWF'
            }
            
            ds['vertex_lon'].attrs = {
                'units': 'degrees',
                'long_name': 'Vertex longitude',
                'description': 'Longitude coordinates of the polygon vertices'
            }
            
            ds['vertex_lat'].attrs = {
                'units': 'degrees',
                'long_name': 'Vertex latitude',
                'description': 'Latitude coordinates of the polygon vertices'
            }
            
            # Save the dataset to netCDF
            output_file = os.path.join(output_dir, f'fusion_swim_sar_{swim_date}.nc')
            ds.to_netcdf(output_file)
            print(f"Saved fused data to {output_file}")
            
            # Print summary stats
            print(f"Number of matched samples: {n_time}")
            print(f"Wavenumber dimensions: {n_k}")
            print(f"Azimuth dimensions: {n_phi}")
            print(f"ST depth levels: {n_depth}")
        else:
            print(f"No matched samples found for date {swim_date}. Skipping file creation.")
    
    log_total_execution_time()






            


        

        



