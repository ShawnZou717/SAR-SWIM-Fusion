import argparse
import pandas as pd
import requests
import time
import os
import math

# --- Internal Station Catalog ---
# West longitudes and South latitudes must be negative.
STATION_CATALOG = {
    "PAPA": {"lat": 50.1, "lon": -144.9},
    "SOFS": {"lat": -47.067, "lon": 142.083},
    "NDBC51002": {"lat": 17.070, "lon": -157.755}
}

def generate_square_polygon(lat_center, lon_center, edge_length_km):
    """
    Generates a bounding box polygon string for a given center point and edge length.
    """
    R = 6371.0  # Earth's mean radius in kilometers
    dist_km = edge_length_km / 2.0  
    
    delta_lat = math.degrees(dist_km / R)
    delta_lon = math.degrees(dist_km / (R * math.cos(math.radians(lat_center))))
    
    lat_min = lat_center - delta_lat
    lat_max = lat_center + delta_lat
    lon_min = lon_center - delta_lon
    lon_max = lon_center + delta_lon
    
    polygon = (
        f"{lon_min:.3f} {lat_min:.3f},"
        f"{lon_min:.3f} {lat_max:.3f},"
        f"{lon_max:.3f} {lat_max:.3f},"
        f"{lon_max:.3f} {lat_min:.3f},"
        f"{lon_min:.3f} {lat_min:.3f}"
    )
    
    return polygon

def request_access_token(username, password):
    """
    Requests a fresh access token from the Copernicus Data Space Ecosystem.
    """
    auth_url = 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token'
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    data = {
        'grant_type': 'password',
        'username': username,
        'password': password,
        'client_id': 'cdse-public',
    }

    response = requests.post(auth_url, headers=headers, data=data)

    if response.status_code == 200:
        return response.json().get('access_token')
    else:
        raise Exception(f"Failed to retrieve token: {response.status_code} - {response.text}")


def OCN_searching(year, lat, lon, polygon_size, satellite):
    """
    Searches for OCN data within the specified temporal and spatial range.
    """
    start_time = f"{year}-01-01T00:00:00.000Z"
    end_time = f"{year}-12-31T23:59:59.000Z"
    temporal_filter = f"ContentDate/Start gt {start_time} and ContentDate/Start lt {end_time}"

    polygon = generate_square_polygon(lat, lon, polygon_size)
    spatial_filter = f"OData.CSC.Intersects(area=geography'SRID=4326;POLYGON(({polygon}))')"

    # Apply Satellite Filter
    satellite_filter = ""
    if satellite.upper() == 'S1A':
        satellite_filter = " and contains(Name,'S1A_')"
    elif satellite.upper() == 'S1B':
        satellite_filter = " and contains(Name,'S1B_')"
    elif satellite.upper() == 'ALL':
        satellite_filter = " and (contains(Name,'S1A_') or contains(Name,'S1B_'))"

    base_url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=Collection/Name eq 'SENTINEL-1' and contains(Name,'OCN'){satellite_filter} and {temporal_filter} and {spatial_filter}&$top=200"
    all_results = []

    next_url = base_url
    while next_url:
        response = requests.get(next_url)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch data: {response.status_code}, {response.text}")
        
        json_data = response.json()
        all_results.extend(json_data['value']) 

        next_url = json_data.get('@odata.nextLink', None)

    df = pd.DataFrame.from_dict(all_results)
    return df


def OCN_downloading(sar_file_id, sar_file_name, access_token, data_save_path, username, password):
    """
    Downloads the SAR file. Refreshes and returns the access token if it expires.
    """
    url = f"https://download.dataspace.copernicus.eu/odata/v1/Products({sar_file_id})/$value"
    zip_file_path = os.path.join(data_save_path, f"{sar_file_name}.zip")
    
    while True:
        try:
            headers = {"Authorization": f"Bearer {access_token}"}
            session = requests.Session()
            session.headers.update(headers)
            
            with session.get(url, headers=headers, stream=True) as response:
                if response.status_code == 200:
                    with open(zip_file_path, "wb") as file:
                        for chunk in response.iter_content(chunk_size=1024*1024):
                            if chunk:
                                file.write(chunk)
                    break
                else:
                    raise requests.exceptions.RequestException(f"response error code: {response.status_code}")
                    
        except requests.exceptions.RequestException as e:
            print(f'An error occurred: {e}')
            time.sleep(5)  # Wait before retrying
            print("Refreshing access token...")
            access_token = request_access_token(username, password)
            
    return access_token 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Sentinel-1 OCN Data based on Space/Time filters.")
    
    # Define Input Arguments
    parser.add_argument("--station_name", type=str, required=True, help="Name of the station (e.g., PAPA, SOFS, NDBC51002)")
    parser.add_argument("--lat", type=float, help="Latitude (Required if station is not in the internal catalog)")
    parser.add_argument("--lon", type=float, help="Longitude (Required if station is not in the internal catalog)")
    parser.add_argument("--year", type=str, required=True, help="Years to download, comma-separated (e.g., 2021,2022,2023)")
    parser.add_argument("--polygon_size", type=float, default=200.0, help="Square polygon edge length in km (Default: 200)")
    parser.add_argument("--save_dir", type=str, required=True, help="Base directory to save the downloaded data")
    
    parser.add_argument("--satellite", type=str, choices=['S1A', 'S1B', 'ALL'], default='ALL', 
                        help="Filter by specific satellite: S1A, S1B, or ALL (Default: ALL)")
    
    # Make sure to replace default strings or pass them via command line
    parser.add_argument("--username", type=str, default="YOUR_EMAIL@gmail.com", help="Dataspace login email")
    parser.add_argument("--password", type=str, default="YOUR_PASSWORD", help="Dataspace login password")

    args = parser.parse_args()

    # --- Coordinate Resolution Logic ---
    station_key = args.station_name.upper() 
    
    if station_key in STATION_CATALOG:
        target_lat = STATION_CATALOG[station_key]["lat"]
        target_lon = STATION_CATALOG[station_key]["lon"]
        print(f"[{station_key}] found in catalog. Using coordinates: Lat {target_lat}, Lon {target_lon}")
    else:
        if args.lat is None or args.lon is None:
            parser.error(f"Station '{args.station_name}' is not in the internal catalog. You MUST provide both --lat and --lon arguments.")
        target_lat = args.lat
        target_lon = args.lon
        print(f"Using custom coordinates for [{args.station_name}]: Lat {target_lat}, Lon {target_lon}")

    # Parse the comma-separated years into a list
    years_to_download = [y.strip() for y in args.year.split(',')]

    print("Requesting initial access token!!!")
    access_token = request_access_token(args.username, args.password)

    # Loop through each provided year
    for current_year in years_to_download:
        print(f"\n{'='*50}")
        print(f"Processing station {args.station_name} for the year {current_year}")
        print(f"{'='*50}")

        current_save_path = os.path.join(args.save_dir, args.station_name, current_year)
        os.makedirs(current_save_path, exist_ok=True)

        print(f"Searching {args.satellite} SAR data around {args.station_name}...")
        sar_ocn_files = OCN_searching(current_year, target_lat, target_lon, args.polygon_size, args.satellite)
        
        if sar_ocn_files.empty:
            print(f"No files found for {current_year}. Moving to next year.")
            continue
            
        print(f"Found {sar_ocn_files.shape[0]} SAR OCN files for {current_year}!!!")

        for sar_ocn_files_id, sar_ocn_files_name in zip(sar_ocn_files['Id'], sar_ocn_files['Name']):
            sar_ocn_files_name_clean = sar_ocn_files_name.replace(".SAFE", "")
            end_marker_path = os.path.join(current_save_path, f"{sar_ocn_files_name_clean}.END")

            # Check if already downloaded
            if os.path.exists(end_marker_path):
                print(f"{sar_ocn_files_name_clean} already downloaded. Skipping.")
                continue

            print(f"Downloading {sar_ocn_files_name_clean}...")
            access_token = OCN_downloading(
                sar_file_id=sar_ocn_files_id, 
                sar_file_name=sar_ocn_files_name_clean, 
                access_token=access_token, 
                data_save_path=current_save_path,
                username=args.username,
                password=args.password
            )
            print(f"Completed {sar_ocn_files_name_clean}")

            # Create download completed label file
            with open(end_marker_path, 'w') as f:
                pass
    
    print("\nAll requested SAR data downloaded successfully!!!")
