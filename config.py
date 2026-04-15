import numpy as np

swim_k = np.array([0.012566, 0.013887, 0.015348, 0.016962, 0.018746, 0.020718, 
                    0.022897, 0.025305, 0.027966, 0.030908, 0.034158, 0.037751, 0.041721, 
                    0.046109, 0.050959, 0.056318, 0.062241, 0.068787, 0.076022, 0.084017, 
                    0.092853, 0.102619, 0.113411, 0.125339, 0.138521, 0.153089, 0.16919, 
                    0.186984, 0.206649, 0.228383, 0.252402, 0.278947])

swim_phi = np.deg2rad(np.linspace(7.5, 360 - 7.5, 24))

sar_k = np.array([0.005235988, 0.00557381, 0.005933429, 0.00631625, 0.00672377, 
                    0.007157583, 0.007619386, 0.008110984, 0.008634299, 0.009191379, 
                    0.0097844, 0.01041568, 0.0110877, 0.01180307, 0.01256459, 0.01337525, 
                    0.01423822, 0.01515686, 0.01613477, 0.01717577, 0.01828394, 0.01946361, 
                    0.02071939, 0.02205619, 0.02347924, 0.02499411, 0.02660671, 0.02832336, 
                    0.03015076, 0.03209607, 0.03416689, 0.03637131, 0.03871796, 0.04121602, 
                    0.04387525, 0.04670605, 0.0497195, 0.05292737, 0.05634221, 0.05997737, 
                    0.06384707, 0.06796645, 0.0723516, 0.07701967, 0.08198893, 0.08727881, 
                    0.09290998, 0.09890447, 0.1052857, 0.1120787, 0.1193099, 0.1270077, 
                    0.1352022, 0.1439253, 0.1532113, 0.1630964, 0.1736193, 0.1848211, 
                    0.1967456, 0.2094395])

sar_phi = np.deg2rad(np.linspace(2.5, 360 - 2.5, 72))

# --- Internal Station Catalog ---
STATION_CATALOG = {
    "SOFS-9": {"lat": -46.984, "lon": 141.81}, # The TriAXY is deployed at different positions near SOFS
    "SOFS-11": {"lat": -46.965, "lon": 141.3511},
    "NDBC51002": {"lat": 17.070, "lon": -157.755},
    "PAPA": {"lat": 50.1, "lon": -144.9}
}

# Wave spectra, time, postition, and quality control factors indices for each dataset
SWIM_QUALITY_INDICES = {'flag_sigma0_shape_box': lambda x: x == 0, # n_phi, n_posneg, n_box, n_beam_l2
                        'flag_sigma0_slope_box': lambda x: x == 0, # n_phi, n_posneg, n_box, n_beam_l2
                        'flag_sigma0_mean_box': lambda x: x == 0, # n_phi, n_posneg, n_box, n_beam_l2
                        'wf_surf_ocean_index_box': lambda x: x >= 90, # n_box
                        'nadir_rain_index_box': lambda x: x <= 10} #n_box

SWIM_PARAM_INDICES = {"wave_spec": "p_combined", # dimension (nk, n_phi, n_posneg, n_box)
                    #   "wavenumber": "k_spectra", # no need to read wavenumber and phi, as it is fixed for all files.
                    #   "direction": "phi"
                    "north_heading": "phi_orbit_box", # dimension (n_box)
                    "lat": "lat_l2", #dimension (n_posneg, n_box)
                    "lon": "lon_l2", #dimension (n_posneg, n_box)
                    "measured_time": "time_spec_l2", #dimension (n_posneg, n_box, n_tim=2(0 to s, 1 to us))
                    "u10": "u10_ecmwf", # dimension (n_posneg, n_box)
                    "v10": "v10_ecmwf"} # dimension (n_posneg, n_box)

WW3_IFREMER_PARAM_INDICES = {"freq":[0.0373, 0.04103, 0.04513299, 0.0496463, 0.05461093, 0.06007202, 
                                    0.06607922, 0.07268714, 0.07995586, 0.08795145, 0.0967466, 0.1064213, 
                                    0.1170634, 0.1287697, 0.1416467, 0.1558114, 0.1713925, 0.1885318, 
                                    0.2073849, 0.2281235, 0.2509358, 0.2760294, 0.3036323, 0.3339956, 
                                    0.3673951, 0.4041346, 0.4445481, 0.4890029, 0.5379032, 0.5916935, 
                                    0.6508629, 0.7159492],
                            "wave_unit_factor": np.pi/180, # Convert from m^2/Hz/rad to m^2/Hz/deg.
                            "direc": np.deg2rad(np.linspace(0, 360 - 15, 24) + 7.5),
                            # use a precomputed argsort to reorder the 'direction' dimension
                            "oceanographic_align": lambda x: x.isel(direction=[6,  5,  4,  3,  2,  1,  0, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10,  9,  8,  7]), # WW3 direction is defined as [90, 75, 60, 45, 30, 15, 0, 345, 330, 315, 300, 285, 270, 255, 240, 225, 210, 195, 180, 165, 150, 135, 120, 105], we need to reorder it to [0, 15, ..., 345] to be consistent with the definition of direction in other datasets. WW3-Ifremer uses the oceanographic convention, denoting the direction where the waves propagate to.
                            "lat": "ww3_lat",
                            "lon": "ww3_lon",
                            }

SAR_QUALITY_INDICES = {}

SAR_PARAM_INDICES = {"wave_spec": "oswPolSpec",
                     "north_heading": "oswHeading",
                     "lat": "oswLat",
                     "lon": "oswLon",
                     "measured_time": "firstMeasurementTime", # NOTICE! The firstMeasurementTime is embedded in the attributes of the dataset.
                     'azimuth_cutoff_wavelength': "oswAzCutoff",
                     }

# WW3IFREMER_PARAM_INDICES = {"wave_spec": "ww3_efth", # dimension (row, cell, frequency, direction)
#                             "lat": "lat",            # dimension (row, cell)
#                             "lon": "lon",            # dimension (row, cell)
#                             "ww3_lat": 'ww3_lat',    # dimension (row, cell)
#                             "ww3_lon": 'ww3_lon',    # dimension (row, cell)
#                             "measured_time": "time", # dimension (row, cell)
#                             "ww3_time": "ww3_time",  # dimension (row, cell)
#                             "wind_speed": "ww3_wnd", # dimension (row, cell)
#                             "wind_dir": "ww3_wnddir" # dimension (row, cell)
#                 }

# PAPA_PARAM_INDICES = {"wave_spec": "waveDirectionalSpectrum", #m^2/Hz/deg
#                      "measured_time": "waveTime"}

NDBC51002_PARAM_INDICES = {"wave_spec": "efth", # m^2/Hz/rad
                           "wave_unit_factor": np.pi/180, # Convert from m^2/Hz/rad to m^2/Hz/deg.
                           "freq": [0.02, 0.0325, 0.0375, 0.0425, 0.0475, 0.0525, 0.0575, 0.0625, 
                                    0.0675, 0.0725, 0.0775, 0.0825, 0.0875, 0.0925, 0.1, 0.11, 0.12, 0.13, 
                                    0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 
                                    0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.365, 0.385, 
                                    0.405, 0.425, 0.445, 0.465, 0.485],
                           "direc": np.deg2rad(np.linspace(2.5, 360 - 2.5, 72)),
                           "measured_time": "time"}

SOFS_PARAM_INDICES = {"wave_spec": "DIR_SPEC", # m^2/Hz/deg, dimension
                      "direction_extended_flag": 1, #The direction spectrum has contained n repeated direction columns, if larger than 1, then the last n cloumns would be removed to keep the processing consistency.
                      "freq": [0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 
                                0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.1, 0.105, 
                                0.11, 0.115, 0.12, 0.125, 0.13, 0.135, 0.14, 0.145, 0.15, 0.155, 0.16, 
                                0.165, 0.17, 0.175, 0.18, 0.185, 0.19, 0.195, 0.2, 0.205, 0.21, 0.215, 
                                0.22, 0.225, 0.23, 0.235, 0.24, 0.245, 0.25, 0.255, 0.26, 0.265, 0.27, 
                                0.275, 0.28, 0.285, 0.29, 0.295, 0.3, 0.305, 0.31, 0.315, 0.32, 0.325, 
                                0.33, 0.335, 0.34, 0.345, 0.35, 0.355, 0.36, 0.365, 0.37, 0.375, 0.38, 
                                0.385, 0.39, 0.395, 0.4, 0.405, 0.41, 0.415, 0.42, 0.425, 0.43, 0.435, 
                                0.44, 0.445, 0.45, 0.455, 0.46, 0.465, 0.47, 0.475, 0.48, 0.485, 0.49, 
                                0.495, 0.5, 0.505, 0.51, 0.515, 0.52, 0.525, 0.53, 0.535, 0.54, 0.545, 
                                0.55, 0.555, 0.56, 0.565, 0.57, 0.575, 0.58, 0.585, 0.59, 0.595, 0.6, 
                                0.605, 0.61, 0.615, 0.62, 0.625, 0.63, 0.635, 0.64],
                      "direc": np.deg2rad(np.linspace(1.5, 360 + 1.5, 121)), 
                      "oceanographic_align": lambda x: x.roll({"DIR_SPEC": x.sizes['DIR_SPEC'] // 2}, roll_coords=False),
                      "measured_time": "TIME"}
