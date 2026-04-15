import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
import re


class NDBCWind:
    def __init__(self, data_dir='.'):
        """
        Initializes the class by scanning the target directory for valid NDBC 
        files and building an inventory of available station IDs and years.
        
        :param data_dir: Directory where the .txt files are stored.
        """
        self.data_dir = data_dir
        self.inventory = {}
        self.datasets = {} 
        
        self._scan_directory()

    def _scan_directory(self):
        """
        Internal method to scan the directory and parse filenames 
        matching the pattern {Station_id}s{year}.txt
        """
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Directory not found: {self.data_dir}")

        pattern = re.compile(r'^([a-zA-Z0-9]+)s(\d{4})\.txt$')

        for filename in os.listdir(self.data_dir):
            match = pattern.match(filename)
            if match:
                station_id = match.group(1)
                year = int(match.group(2))
                
                if station_id not in self.inventory:
                    self.inventory[station_id] = []
                self.inventory[station_id].append(year)

        for station_id in self.inventory:
            self.inventory[station_id].sort()

        print(f"Discovered data for {len(self.inventory)} station(s):")
        for station, years in self.inventory.items():
            print(f" - Station {station}: {len(years)} year(s) {years}")

    def load_station(self, station_id):
        """
        Loads and concatenates all discovered years for a specific station.
        Converts wind direction from "coming from" to "going towards".
        """
        station_id = str(station_id)
        if station_id not in self.inventory:
            raise ValueError(f"No data found for station ID: {station_id}")

        years = self.inventory[station_id]
        all_yearly_data = []

        for year in years:
            filepath = os.path.join(self.data_dir, f"{station_id}s{year}.txt")
            
            df = pd.read_csv(
                filepath, 
                sep=r'\s+', 
                header=0, 
                skiprows=[1], 
                na_values=['MM', '99', '999', '99.0', '9999']
            )

            # Map NDBC date/time columns to standard names
            rename_dict = {}
            for col in df.columns:
                if 'YY' in col or '#YY' in col:
                    rename_dict[col] = 'year'
                elif col == 'MM':
                    rename_dict[col] = 'month'
                elif col == 'DD':
                    rename_dict[col] = 'day'
                elif col == 'hh':
                    rename_dict[col] = 'hour'
                elif col == 'mm':
                    rename_dict[col] = 'minute'

            df.rename(columns=rename_dict, inplace=True)
            df['time'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute']])

            # Filter requested columns and rename
            df = df[['time', 'WDIR', 'WSPD']].copy()
            df.rename(columns={'WDIR': 'wind_dir', 'WSPD': 'wind_speed'}, inplace=True)
            
            # --- NEW CONVERSION LOGIC ---
            # Convert wind coming direction to wind going direction
            # Pandas handles NaN values safely here (NaN + 180 % 360 remains NaN)
            df['wind_dir'] = (df['wind_dir'] + 180) % 360

            df.set_index('time', inplace=True)
            
            all_yearly_data.append(df)

        # Concatenate, sort, and remove overlapping duplicates
        combined_df = pd.concat(all_yearly_data)
        combined_df.sort_index(inplace=True)
        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]

        # Convert to Xarray Dataset
        dataset = xr.Dataset.from_dataframe(combined_df)
        
        # --- UPDATED METADATA ---
        dataset.wind_dir.attrs = {
            'standard_name': 'wind_to_direction',
            'units': 'degrees',
            'description': 'Wind direction, clockwise from true north (direction the wind is blowing TOWARDS)'
        }
        dataset.wind_speed.attrs = {
            'standard_name': 'wind_speed',
            'units': 'm/s',
            'description': 'Highest 1 minute wind speed for the hour'
        }

        # Store in the instance dictionary
        self.datasets[station_id] = dataset
        return dataset

    def process_all_to_netcdf(self, output_dir='.'):
        """
        Iterates through all discovered stations, loads their data, 
        and saves each station as its own combined NetCDF file.
        """
        if not self.inventory:
            print("No valid files were found in the directory to process.")
            return

        os.makedirs(output_dir, exist_ok=True)

        for station_id in self.inventory.keys():
            print(f"Processing station {station_id}...")
            dataset = self.load_station(station_id)
            
            output_path = os.path.join(output_dir, f"NDBC{station_id}_wind_combined.nc")
            dataset.to_netcdf(output_path)
            print(f" -> Saved to: {output_path}")


class NDBCWaveSpectra:
    """
    A class to read, organize, and compute directional wave spectra from raw NDBC buoy text files.
    Supports multi-station processing, eager loading, and MEM inversion.
    """
    def __init__(self, data_dir, n=72, compute_strategy='lazy', convention='oceanographic', method='mem'):
        """
        Parameters:
        -----------
        data_dir : str
            Directory containing the NDBC .txt files.
        n : int
            Number of bins to divide the 0-360 degree directional spectrum.
        compute_strategy : str
            'lazy' computes efth only when requested; 'eager' computes immediately.
        convention : str
            'oceanographic' (traveling to) or 'meteorological' (coming from).
        method : str
            'mem' (Maximum Entropy Method - sharp, realistic peaks) or 
            'longuet-higgins' (Truncated Fourier - broad, smooth NDBC default).
        """
        self.data_dir = data_dir
        self.n = n
        self.compute_strategy = compute_strategy.lower()
        self.convention = convention.lower()
        self.method = method.lower()
        
        if self.compute_strategy not in ['lazy', 'eager']:
            raise ValueError("compute_strategy must be either 'lazy' or 'eager'.")
        if self.convention not in ['oceanographic', 'meteorological']:
            raise ValueError("convention must be either 'oceanographic' or 'meteorological'.")
        if self.method not in ['mem', 'longuet-higgins']:
            raise ValueError("method must be either 'mem' or 'longuet-higgins'.")

        self.directions = np.linspace(0, 360, n, endpoint=False) + (360 / n / 2)  # Center of each directional bin
        self.dataset = None
        self._efth_computed = False
        
        if self.compute_strategy == 'eager':
            print("Eager strategy active: Auto-discovering and processing all stations...")
            self.process_all_stations()

    def _parse_ndbc_file(self, filepath):
        with open(filepath, 'r') as f:
            header_line = f.readline()
            
        parts = header_line.split()
        
        try:
            frequencies = [float(x) for x in parts[5:]]
        except ValueError:
            raise ValueError(f"Could not parse frequency headers in {filepath}")

        df = pd.read_csv(
            filepath, 
            sep=r'\s+', 
            skiprows=1,
            header=None,
            names=['YY', 'MM', 'DD', 'hh', 'mm'] + frequencies,
            on_bad_lines='skip'
        )
        
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna(subset=['YY'])
        df['YY'] = df['YY'].apply(lambda x: x + 1900 if x >= 70 and x < 100 else (x + 2000 if x < 70 else x))
        
        times = pd.to_datetime(
            df[['YY', 'MM', 'DD', 'hh', 'mm']].rename(
                columns={'YY': 'year', 'MM': 'month', 'DD': 'day', 'hh': 'hour', 'mm': 'minute'}
            )
        )
        
        df = df.drop(columns=['YY', 'MM', 'DD', 'hh', 'mm'])
        df.index = times
        
        df = df.mask(df == 999.0)
        return df

    def _process_single_station(self, station_id):
        station_id = str(station_id)
        file_types = {'w': 'C11', 'd': 'alpha1', 'i': 'alpha2', 'j': 'r1', 'k': 'r2'}
        
        search_pattern = os.path.join(self.data_dir, f"{station_id}w*.txt")
        w_files = glob.glob(search_pattern)
        years = [f.split('w')[-1].replace('.txt', '') for f in w_files]

        if not years:
            raise ValueError(f"No 'w' files found for station {station_id}.")

        yearly_datasets = []

        for year in years:
            dfs = {}
            files_exist = True
            
            for file_code in file_types.keys():
                fpath = os.path.join(self.data_dir, f"{station_id}{file_code}{year}.txt")
                if not os.path.exists(fpath):
                    files_exist = False
                    break
            
            if not files_exist:
                continue

            for file_code in file_types.keys():
                fpath = os.path.join(self.data_dir, f"{station_id}{file_code}{year}.txt")
                dfs[file_code] = self._parse_ndbc_file(fpath)

            lengths = [len(df) for df in dfs.values() if df is not None]
            if not lengths or len(set(lengths)) > 1:
                continue
                
            base_index = dfs['w'].index
            if not all(df.index.equals(base_index) for df in dfs.values()):
                continue

            time_array = base_index.values 

            ds = xr.Dataset(
                {
                    "C11": (["time", "frequency"], dfs['w'].values),
                    "alpha1": (["time", "frequency"], dfs['d'].values),
                    "alpha2": (["time", "frequency"], dfs['i'].values),
                    "r1": (["time", "frequency"], dfs['j'].values),
                    "r2": (["time", "frequency"], dfs['k'].values),
                },
                coords={
                    "time": time_array,
                    "frequency": dfs['w'].columns.values.astype(float),
                    "direction": self.directions
                }
            )
            yearly_datasets.append(ds)

        if not yearly_datasets:
            raise ValueError(f"No valid, complete years found for station {station_id}.")

        ds_station = xr.concat(yearly_datasets, dim="time").sortby("time")
        
        if self.convention == 'oceanographic':
            ds_station['alpha1'] = (ds_station['alpha1'] + 180.0) % 360.0
            ds_station['alpha2'] = (ds_station['alpha2'] + 180.0) % 360.0
            ds_station['alpha1'].attrs['convention'] = 'oceanographic (traveling to)'
            ds_station['alpha2'].attrs['convention'] = 'oceanographic (traveling to)'
        else:
            ds_station['alpha1'].attrs['convention'] = 'meteorological (coming from)'
            ds_station['alpha2'].attrs['convention'] = 'meteorological (coming from)'
            
        return ds_station

    def process_station(self, station_id):
        self.dataset = self._process_single_station(station_id)
        
        if self.compute_strategy == 'eager':
            print(f"Eager strategy: Computing efth ({self.method}) for station {station_id}...")
            self._compute_efth()

    def process_all_stations(self):
        search_pattern = os.path.join(self.data_dir, "*w*.txt")
        w_files = glob.glob(search_pattern)
        
        station_ids = set([os.path.basename(f).split('w')[0] for f in w_files])
        
        if not station_ids:
            raise ValueError(f"No NDBC wave files found in {self.data_dir}")
            
        station_datasets = []
        valid_stations = []

        for stn in station_ids:
            try:
                ds = self._process_single_station(stn)
                station_datasets.append(ds)
                valid_stations.append(stn)
                print(f"Successfully loaded station {stn}")
            except Exception as e:
                print(f"Skipped station {stn}: {e}")
                
        if not station_datasets:
            raise ValueError("No stations could be processed.")
            
        self.dataset = xr.concat(station_datasets, pd.Index(valid_stations, name="station"))
        
        if self.compute_strategy == 'eager':
            print(f"Eager strategy: Computing 2D directional spectra ({self.method}) for ALL stations...")
            self._compute_efth()

    def _compute_efth(self):
        if self._efth_computed:
            return 
            
        if self.dataset is None:
            raise ValueError("Dataset not loaded.")

        r1 = self.dataset['r1'] * 0.01
        r2 = self.dataset['r2'] * 0.01

        A = np.deg2rad(self.dataset['direction'])
        alpha1_rad = np.deg2rad(self.dataset['alpha1'])
        alpha2_rad = np.deg2rad(self.dataset['alpha2'])

        if self.method == 'longuet-higgins':
            # Original truncated Fourier method (broad/smooth shapes, prone to negative artifacts)
            D = (1.0 / np.pi) * (
                0.5 + 
                r1 * np.cos(A - alpha1_rad) + 
                r2 * np.cos(2 * (A - alpha2_rad))
            )
            D = D.where(D > 0, 0) # Clip negative energy artifacts
            
        elif self.method == 'mem':
            # Lygre & Krogstad (1986) Maximum Entropy Method
            
            # 1. Clamp r1 slightly below 1.0 to prevent mathematical singularities in the denominator
            r1_safe = r1.clip(max=0.995)
            
            # 2. Convert to complex Fourier coefficients
            c1 = r1_safe * np.exp(1j * alpha1_rad)
            c2 = r2 * np.exp(1j * 2 * alpha2_rad)
            
            c1_conj = np.conj(c1)
            c2_conj = np.conj(c2)
            
            # 3. Calculate autoregressive parameters
            denom = 1 - (r1_safe ** 2)
            denom = denom.where(denom > 1e-6, 1e-6) # Safety catch
            
            phi1 = (c1 - c2 * c1_conj) / denom
            phi2 = c2 - c1 * phi1
            
            # 4. Variance parameter (must be strictly real and positive)
            sigma2 = 1 - np.real(phi1 * c1_conj + phi2 * c2_conj)
            sigma2 = sigma2.where(sigma2 > 0, 1e-6)
            
            # 5. Compute MEM Directional Distribution
            e_i_A = np.exp(-1j * A)
            e_i_2A = np.exp(-1j * 2 * A)
            
            D_denom = np.abs(1 - phi1 * e_i_A - phi2 * e_i_2A) ** 2
            D = (1.0 / (2 * np.pi)) * (sigma2 / D_denom)

        # Apply the chosen directional distribution to the total energy
        efth = self.dataset['C11'] * D
        
        efth.attrs['units'] = 'm^2/hz/rad'
        efth.attrs['long_name'] = f'Directional Wave Spectra ({self.method.upper()})'
        efth.attrs['convention'] = self.convention
        
        self.dataset['efth'] = efth
        self._efth_computed = True

    def save_netcdf(self, output_path):
        if self.dataset is None:
            raise ValueError("No data to save.")
            
        if not self._efth_computed:
            print(f"Computing 2D directional spectra ({self.method}) prior to saving...")
            self._compute_efth()
            
        self.dataset.to_netcdf(output_path)
        print(f"Data successfully saved to {output_path}")

    def get_spectrum(self, target_time, station_id=None):
        if self.dataset is None:
            raise ValueError("Dataset not loaded.")
            
        if not self._efth_computed:
            print(f"Lazy strategy triggered: Computing 2D spectra ({self.method}) prior to slicing...")
            self._compute_efth()
        
        nearest_slice = self.dataset.sel(time=target_time, method='nearest')
        
        if station_id is not None:
            if 'station' in nearest_slice.dims:
                nearest_slice = nearest_slice.sel(station=str(station_id))
            else:
                print("Warning: Dataset only contains one station. Ignoring station_id parameter.")
                
        actual_time = pd.to_datetime(nearest_slice.time.values).strftime('%Y-%m-%d %H:%M')
        print(f"Returning data slice for nearest available time: {actual_time}")
        return nearest_slice