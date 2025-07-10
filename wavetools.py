import numpy as np
from nputils.compat import safe_range as range # to compat with the version diff of numpy
from nputils.compat import safe_zeros as np_zeros, safe_ones as np_ones, safe_full as np_full
from skimage import morphology, measure
from scipy import ndimage
from scipy.interpolate import interp2d, griddata
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from shapely.geometry import Polygon
from geopy.distance import geodesic
from waveutils import config
# import warnings
# warnings.filterwarnings("error", category=RuntimeWarning)

homo_dist = config.homo_dist
homo_time = config.homo_time

# Access safety parameters
max_lat = config.max_lat
min_lat = config.min_lat


def spatio_temporal_match(
        swim_moment: list[np.datetime64] | np.datetime64,
        sar_moment: list[np.datetime64] | np.datetime64,
        swim_polygon: list[Polygon] | Polygon,
        sar_polygon: list[Polygon] | Polygon,
        align_rule: str = 'closest_time',
        thres_t: np.timedelta64 = np.timedelta64(homo_time, 'D'),
        thres_d: float = homo_dist # km
)-> list[tuple[int, int, float]]:
    """
    Match the spatio-temporal information between SWIM and SAR data.
    
    Parameters:
        swim_moment (np.datetime64): The mean timestamp of SWIM L2 Geo boxes.
        sar_moment (np.datetime64): The timestamp of the SAR data.
        swim_polygon (list[Polygon]): Polygon type, illustrating the SWIM L2 Geo boxes. Points in the Polygon should be in (lon, lat) order.
        sar_polygon (list[Polygon]): Polygon type, illustrating the SAR L3 Geo boxes. Points in the Polygon should be in (lon, lat) order.
        align_rule (str): The rule for alignment, can be 'closest_time', 'closest_distance', or 'even_time&distance'.
        thres_t (np.timedelta64): The threshold for time difference, default is 3 days.
        thres_d (float): The threshold for distance difference, default is 180 km.
    
    Returns:
        matched_swim_sar_idx (tuple[int, int, float]): A list of matched indices, where the first element is the index of SWIM and the second element is the index of SAR, the thrid element denotes the corresponding values of criteria.
    """
    # import warnings
    # warnings.filterwarnings("error", category=RuntimeWarning)
    if isinstance(swim_moment, np.datetime64):
        swim_moment = [swim_moment]
    if isinstance(sar_moment, np.datetime64):
        sar_moment = [sar_moment]
    if isinstance(swim_polygon, Polygon):
        swim_polygon = [swim_polygon]
    if isinstance(sar_polygon, Polygon):
        sar_polygon = [sar_polygon]
    if len(swim_moment) != len(swim_polygon):
        raise ValueError("The length of moments and Polygons for SWIM must match.")
    if len(sar_moment) != len(sar_polygon):
        raise ValueError("The length of moments and Polygons for SAR must match.")
    if align_rule not in ['closest_time', 'closest_distance', 'even_time&distance']:
        raise ValueError("align_rule must be one of 'closest_time', 'closest_distance', or 'even_time&distance'.")
    
    thres_t = thres_t.astype('timedelta64[s]')
    
    t_idx_sorted = np.argsort(swim_moment)
    swim_moment = [swim_moment[i] for i in t_idx_sorted]
    swim_polygon = [swim_polygon[i] for i in t_idx_sorted]

    t_idx_sorted = np.argsort(sar_moment)
    sar_moment = [sar_moment[i] for i in t_idx_sorted]
    sar_polygon = [sar_polygon[i] for i in t_idx_sorted]

    # Initialize the matched indices list
    matched_swim_sar_idx = []

    # For each SWIM data point
    for i, sw_moment in enumerate(swim_moment):
        sw_poly = swim_polygon[i]
        matching_indices = []
        
        # For each SAR data point
        for j, sr_moment in enumerate(sar_moment):
            # Check time difference
            time_diff = abs(sr_moment - sw_moment)
            if time_diff <= thres_t:
                # Check spatial overlap
                sr_poly = sar_polygon[j]
                # try:
                #     intersection_area = sw_poly.intersection(sr_poly).area
                # except RuntimeWarning as w:
                #     print(f"Warning: {w} for {i}th SWIM polygon {sw_poly} and {j}th SAR polygon {sr_poly}.")
                # except RuntimeError as e:
                #     print(f"Error: {e} for {i}th SWIM polygon {sw_poly} and {j}th SAR polygon {sr_poly}.")
                #     continue
                intersection_area = sw_poly.intersection(sr_poly).area
                min_area = min(sw_poly.area, sr_poly.area)
                
                if min_area > 0 and intersection_area / min_area >= 0.5: # area based distance determination
                    # Convert shapely Points to lat/lon and calculate geodesic distance in km
                    sw_centroid = sw_poly.centroid
                    sr_centroid = sr_poly.centroid
                    sw_coords = (sw_centroid.y, sw_centroid.x)  # (lat, lon)
                    sr_coords = (sr_centroid.y, sr_centroid.x)  # (lat, lon)
                    dist = geodesic(sw_coords, sr_coords).kilometers

                    weighted_t_dis = time_diff/thres_t + dist/thres_d
                    # Convert time_diff to seconds as a float
                    time_diff_seconds = time_diff.astype('timedelta64[s]').astype(float)
                    matching_indices.append((j, time_diff_seconds, dist, weighted_t_dis))
        
        # Sort matches by time difference if align_rule is 'closest_time'
        # Sort by spatial overlap if align_rule is 'closest_distance'
        # Use a combination if align_rule is 'even_time&distance'
        if matching_indices:
            if align_rule == 'closest_time':
                align_rule_idx = 1
                best_match = min(matching_indices, key=lambda x: x[align_rule_idx])
            elif align_rule == 'closest_distance':
                align_rule_idx = 2
                best_match = min(matching_indices, key=lambda x: x[align_rule_idx])
            elif align_rule == 'even_time&distance':
                align_rule_idx = 3
                best_match = min(matching_indices, key=lambda x: x[align_rule_idx])
            
            matched_swim_sar_idx.append((i, best_match[0], best_match[align_rule_idx]))

    return matched_swim_sar_idx #(swim_idx, sar_idx, criteria_value)


def find_strong_speckle_noise(swim_slope_spec: np.ndarray, quant: float = 0.75, noise_thres: float = 0.85):
    """
    Find the azimutha index whose energy pattern matches the strong speckle noise.
    
    Input:
        swim_slope_spec (np.ndarray): The swim slope spectrum to analyze, shape (n_k, n_phi).
        quant (float): Quantile threshold for background energy.
        noise_thres (float): Threshold for strong speckle noise detection.
    
    Returns:
        azimuth indices or np.nan: Indices of strong speckle noise patterns, or nan if none exist.
    """
    matrix=np.log(swim_slope_spec)
    rows = matrix.shape[0]
    
    background_thres = np.quantile(np.ravel(matrix), quant)
    fbg_mask = matrix.copy()
    fbg_mask[matrix < background_thres] = 0
    fbg_mask[matrix > background_thres] = 1

    high_energy_ratio = np.sum(fbg_mask, axis=0) / rows

    qualifying_indices = np.where(high_energy_ratio > noise_thres)[0]
    
    return qualifying_indices if qualifying_indices.size > 0 else np.nan


def _resign_partition_labels(labels: np.ndarray, background_label: int = 0, set_first_label: int = np.nan):
    # Step 1: Identify unique labels excluding background (0)
    unique_labels = np.unique(labels)

    # Step 2: Filter out the background label (0)
    remaining_labels = [label for label in unique_labels if label != background_label]
    if set_first_label is not np.nan:
        remaining_labels = [set_first_label] + [label for label in remaining_labels if label != set_first_label]

    # Step 3: Create a mapping from old labels to new labels
    new_label_mapping = {old_label: new_label for new_label, old_label in enumerate(remaining_labels, start=1)}

    # Step 4: Create a new matrix to hold the re-labeled values
    relabelled_matrix = np.copy(labels)

    # Step 5: Update the matrix with new labels
    for old_label, new_label in new_label_mapping.items():
        relabelled_matrix[labels == old_label] = new_label
    
    return relabelled_matrix

def resign_partition_labels(labels: np.ndarray, background_label: int = 0, set_first_label: int = np.nan):
    # Step 1: Identify unique labels excluding background (0)
    unique_labels = np.unique(labels)

    # Step 2: Filter out the background label (0)
    remaining_labels = [label for label in unique_labels if label != background_label]
    if set_first_label is not np.nan:
        remaining_labels = [set_first_label] + [label for label in remaining_labels if label != set_first_label]

    # Step 3: Create a mapping from old labels to new labels
    new_label_mapping = {old_label: new_label for new_label, old_label in enumerate(remaining_labels, start=1)}

    # Step 4: Create a new matrix to hold the re-labeled values
    relabelled_matrix = np.copy(labels)

    # Step 5: Update the matrix with new labels
    for old_label, new_label in new_label_mapping.items():
        relabelled_matrix[labels == old_label] = new_label
    
    return relabelled_matrix


def partition_waves_polar(K_mesh: np.ndarray, PHI_mesh: np.ndarray, wave_spectrum: np.ndarray, gaus_blur_ops: int = 2, remove_invalid_parts: bool = True, distinguish_fg_bg = False, version: float=1.0):
    """
    Partition the wave spectrum (in polar coordinate) into a set of waves.

    Parameters
    ----------
    wave_spectrum : np.ndarray
        The wave spectrum to partition.
    
    gaus_blur_ops : int
        The number of Gaussian blur operations to perform.

    Returns
    -------
    partition_num : int
        The number of partitions, background not included.
    
    partitioned_waves : np.ndarray
        The partitioned waves.
    """

    # step 1: Apply Gaussian blur
    blurred_spectrum = wave_spectrum.copy()
    blurred_spectrum = np.hstack((blurred_spectrum[:, -6:], blurred_spectrum, blurred_spectrum[:, :6]))
    for _ in range(gaus_blur_ops):
        blurred_spectrum = ndimage.gaussian_filter(blurred_spectrum, sigma=1, truncate=3.0, mode='nearest')
    extended_spectrum = blurred_spectrum[:, 3:-3]
    
    # Step 2: Thresholding (distinguishing fore- and back-ground)
    ### FIX ME! The watershed methods need to be improved by marking the local maximams in the future version.

    # Version 2.0
    if version == 2.0:
        if distinguish_fg_bg:
            thresh = threshold_otsu(extended_spectrum)
            fbg_labels = extended_spectrum > thresh
        else:
            fbg_labels = np.bool_(np_ones(extended_spectrum.shape))
        
        coords = peak_local_max(extended_spectrum, labels=fbg_labels)
        mask = np_zeros(extended_spectrum.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndimage.label(mask)

        # Step 5: Watershed Algorithm
        labels = watershed(-extended_spectrum, markers, mask=fbg_labels)

        extended_labels = labels.copy()
        labels = labels[:, 3:-3]
    elif version == 1.0:
        if distinguish_fg_bg:
            thresh = threshold_otsu(extended_spectrum)
            binary = extended_spectrum > thresh

            # step 4: Labeling
            markers = measure.label(binary)

            # Step 5: Watershed Algorithm
            labels = watershed(-extended_spectrum, markers, mask=binary)
        else:
            labels = watershed(-extended_spectrum)

        extended_labels = labels.copy()
        labels = labels[:, 3:-3]
    else:
        raise Exception("Invalid version, version should be 1.0 or 2.0.")

    # Step 6: Merge periodic boundary labels
    for i in range(labels.shape[0]):
        if labels[i, 0] != labels[i, -1] and labels[i, 0] != 0 and labels[i, -1] != 0 and extended_labels[i, 2] == extended_labels[i, 3]:
            labels[labels == labels[i, -1]] = labels[i, 0]
    
    partition_num = np.max(labels)
    partitioned_waves = np.copy(labels)

    if remove_invalid_parts:
        # Remove the partitions whoes energy is less than 6.25% Etot, or Hs < 1m
        intermedia_term = wave_spectrum / K_mesh
        extended_spectrum = np.hstack((intermedia_term[:, -1:], intermedia_term))
        PHI_mesh_extended = np.hstack((PHI_mesh[:, -1:]-2*np.pi, PHI_mesh))

        Etot = np.trapz(np.trapz(extended_spectrum, x=PHI_mesh_extended, axis=1), x=K_mesh[:, 0])

        background_label = 0
        for p_label in range(1, partition_num + 1):
            wave_parts = wave_spectrum * (labels == p_label)
            intermedia_term = wave_parts / K_mesh
            extended_spectrum = np.hstack((intermedia_term[:, -1:], intermedia_term))

            E_part = np.trapz(np.trapz(extended_spectrum, x=PHI_mesh_extended, axis=1), x=K_mesh[:, 0])
            if E_part < 0.0625 * Etot and 4 * np.sqrt(E_part) < 1:
                partitioned_waves[labels == p_label] = background_label
                
    partitioned_waves = _resign_partition_labels(partitioned_waves)
    partition_num = np.max(partitioned_waves)
        
    return partition_num, partitioned_waves



def partition_waves(K_mesh: np.ndarray, PHI_mesh: np.ndarray, wave_spectrum: np.ndarray, gaus_blur_ops: int = 2, remove_invalid_parts: bool = True, distinguish_fg_bg = False, version: float=1.0, ambiguity_flag: bool = True):
    """
    Partition the wave spectrum into a set of waves. First transfer wave spectrum from log-polar coord to Cartesian coord, then apply Gaussian blur to the spectrum, and finally apply watershed algorithm to partition the waves.

    Parameters
    ----------
    wave_spectrum : np.ndarray
        The wave spectrum to partition.
    
    gaus_blur_ops : int
        The number of Gaussian blur operations to perform.

    Returns
    -------
    partition_num : int
        The number of partitions, background not included.
    
    partitioned_waves : np.ndarray
        The partitioned waves.
    """

    # Step 0: Transfer wave_spectrum from log-polar to Cartesian coordinates
    to_transfer_spec = np.concatenate((wave_spectrum, wave_spectrum[:, 0:1]), axis=1)
    to_transfer_PHI = np.concatenate((PHI_mesh, PHI_mesh[:, 0:1]+2*np.pi), axis=1)
    to_transfer_K = np.concatenate((K_mesh, K_mesh[:, 0:1]), axis=1)
    wave_spec_cart, kx_mesh, ky_mesh = logpolar_to_cartesian(to_transfer_spec, to_transfer_K, to_transfer_PHI)
    if ambiguity_flag:
        wave_spec_cart = (wave_spec_cart + np.rot90(wave_spec_cart, 2))/2

    # step 1: Apply Gaussian blur
    blurred_spectrum = wave_spec_cart.copy()
    for _ in range(gaus_blur_ops):
        blurred_spectrum = ndimage.gaussian_filter(blurred_spectrum, sigma=1, truncate=3.0, mode='reflect')
    
    # Step 2: Thresholding (distinguishing fore- and back-ground)
    if distinguish_fg_bg:
        thresh = threshold_otsu(blurred_spectrum)
        fbg_labels = blurred_spectrum > thresh
    else:
        fbg_labels = blurred_spectrum > 0
    
    coords = peak_local_max(blurred_spectrum, labels=fbg_labels)
    mask = np_zeros(blurred_spectrum.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndimage.label(mask)

    # Step 5: Watershed Algorithm
    labels = watershed(-blurred_spectrum, markers, mask=fbg_labels)
    
    # partition_num = np.max(labels)
    # partitioned_waves = np.copy(labels)

    # Interpolate using linear interpolation (set NaN outside domain to 0)
    kx0 = K_mesh * np.cos(PHI_mesh)
    ky0 = K_mesh * np.sin(PHI_mesh)
    points = np.column_stack((kx_mesh.ravel(), ky_mesh.ravel()))
    values = labels.ravel()

    labels_polar = griddata(
        points, 
        values, 
        (kx0, ky0), 
        method='nearest', 
        fill_value=0.0
    ) # nearest method could break symmetry when encountering equidistant points, needs further forcing.

    partition_num = np.max(labels_polar)
    partitioned_waves = np.copy(labels_polar)

    ### FIXME Step 6: Merge or remove invalid partitions, investigating.
    ### 2025-05-07. Partition merge method implemented as in function waveutils.merge_partitions.
    if remove_invalid_parts:
        # Remove the partitions whoes energy is less than 6.25% Etot, or Hs < 1m
        intermedia_term = wave_spectrum / K_mesh
        extended_spectrum = np.hstack((intermedia_term[:, -1:], intermedia_term))
        PHI_mesh_extended = np.hstack((PHI_mesh[:, -1:]-2*np.pi, PHI_mesh))

        Etot = np.trapz(np.trapz(extended_spectrum, x=PHI_mesh_extended, axis=1), x=K_mesh[:, 0])

        background_label = 0
        for p_label in range(1, partition_num + 1):
            wave_parts = wave_spectrum * (labels_polar == p_label)
            intermedia_term = wave_parts / K_mesh
            extended_spectrum = np.hstack((intermedia_term[:, -1:], intermedia_term))

            E_part = np.trapz(np.trapz(extended_spectrum, x=PHI_mesh_extended, axis=1), x=K_mesh[:, 0])
            if E_part < 0.0625 * Etot and 4 * np.sqrt(E_part) < 1:
                partitioned_waves[labels_polar == p_label] = background_label
    
    partitioned_waves = _resign_partition_labels(partitioned_waves)
    partition_num = np.max(partitioned_waves)
        
    return partition_num, partitioned_waves


def logpolar_to_cartesian(wave_spectrum: np.ndarray, K_mesh: np.ndarray, PHI_mesh: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Transform wave spectrum from log-polar (K, PHI) to Cartesian (kx, ky) coordinates.
    
    Parameters:
        wave_spectrum: 2D array of shape (n_k, n_phi), wave energy at each (K, PHI).
        K: 1D array of wavenumbers [k0, k1, ..., kn] (must be sorted and positive).
        PHI: 1D array of angles in degrees [0, 360).
    
    Returns:
        cartesian_spectrum: 2D array with shape (2*len(K)-1, 2*len(K)-1), 
                           where kx = ky = [-kn, ..., 0, ..., kn].
        kx: 1D array of Cartesian wavenumbers (symmetric).
        ky: 1D array (same as kx).
    """

    
    kx0 = K_mesh * np.cos(PHI_mesh)  # Original Cartesian points
    ky0 = K_mesh * np.sin(PHI_mesh)
    
    # Flatten polar data for interpolation
    points = np.column_stack((kx0.ravel(), ky0.ravel()))
    values = np.log(wave_spectrum.ravel() + 1)
    
    # Define Cartesian grid (kx, ky)
    kx = K_mesh[:, 0]
    kx = np.concatenate((-kx[::-1], kx))  # Symmetric about 0
    ky = kx.copy()
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    
    # Interpolate using linear interpolation (set NaN outside domain to 0)
    cartesian_spectrum_logged = griddata(
        points, 
        values, 
        (kx_grid, ky_grid), 
        method='linear', 
        fill_value=0.0
    )

    cartesian_spectrum = np.exp(cartesian_spectrum_logged) - 1  # Convert back to original scale
    
    return cartesian_spectrum, kx_grid, ky_grid





# def interpolate_wave_polar(matrix, K0, PHI0, K1, PHI1, log_scale=True):
#     """
#     Interpolates a 2D matrix along the K dimension (rows), and fills with zeros 
#     where the new K1 exceeds the original K0 range.
    
#     Parameters:
#     matrix (2D numpy array): The original matrix with dimensions (K0, PHI0).
#     K0 (1D numpy array): The original K dimension values (k0).
#     PHI1 (1D numpy array): The PHI1 dimension values (PHI1).
#     K1 (1D numpy array): The new K dimension values (K1).
    
#     Returns:
#     2D numpy array: The interpolated matrix with dimensions (K1, PHI1).
#     """

#     if log_scale:
#         matrix = np.log(matrix + 1)
    
#     # 创建扩展数组以处理PHI维度的周期性边界
#     matrix_origin = matrix.copy()
#     matrix_extended = np.concatenate((matrix_origin[:, -1:], matrix_origin, matrix_origin[:, :1]), axis=1)
#     PHI0_extended = np.concatenate((PHI0[-1:]-2*np.pi, PHI0, PHI0[:1]+2*np.pi))
    
#     # 创建源点和目标点网格
#     PHI0_mesh_ext, K0_mesh_ext = np.meshgrid(PHI0_extended, K0)
#     PHI1_mesh, K1_mesh = np.meshgrid(PHI1, K1)
    
#     # 使用griddata进行插值，替换interp2d
#     points = np.column_stack((PHI0_mesh_ext.flatten(), K0_mesh_ext.flatten()))
#     values = matrix_extended.flatten()
#     xi = np.column_stack((PHI1_mesh.flatten(), K1_mesh.flatten()))
    
#     interpolated_values = griddata(points, values, xi, method='cubic', fill_value=0)
#     interpolated_matrix = interpolated_values.reshape(K1_mesh.shape)

#     # For the K1 values that exceed max(K0), set those values to 0
#     PHI_mesh, K_mesh = np.meshgrid(PHI1, K1)
#     PHI0_mesh, K0_mesh = np.meshgrid(PHI0, K0)
#     interpolated_matrix[K_mesh > K0_mesh.max()] = 0
#     interpolated_matrix[interpolated_matrix < 0] = 0

#     if log_scale:
#         interpolated_matrix = np.exp(interpolated_matrix) - 1

#     return interpolated_matrix



def interpolate_wave_polar(matrix, K0, PHI0, K1, PHI1, log_scale=True):
    """
    Interpolates a 2D matrix along the K dimension (rows), and fills with zeros 
    where the new K1 exceeds the original K0 range.
    
    Parameters:
    matrix (2D numpy array): The original matrix with dimensions (K0, PHI0).
    K0 (1D numpy array): The original K dimension values (k0).
    PHI1 (1D numpy array): The PHI1 dimension values (PHI1).
    K1 (1D numpy array): The new K dimension values (K1).
    
    Returns:
    2D numpy array: The interpolated matrix with dimensions (K1, PHI1).
    """

    if log_scale:
        matrix = np.log(matrix + 1)
    
    # Interpolate matrix using bicubic interpolation
    matrix_origin = matrix.copy()
    matrix = np.concatenate((matrix_origin[:, -1:], matrix), axis=1)
    matrix = np.concatenate((matrix, matrix_origin[:, :1]), axis=1)
    PHI0 = np.concatenate((PHI0[-1:]-2*np.pi, PHI0, PHI0[:1]+2*np.pi))
    interpolator = interp2d(PHI0, K0, matrix, kind='cubic', fill_value=0)

    # Interpolate for the new K1
    interpolated_matrix = interpolator(PHI1, K1)

    # For the K1 values that exceed max(K0), set those values to 0
    PHI_mesh, K_mesh = np.meshgrid(PHI1, K1)
    PHI0_mesh, K0_mesh = np.meshgrid(PHI0, K0)
    interpolated_matrix[K_mesh > K0_mesh.max()] = 0
    interpolated_matrix[interpolated_matrix < 0] = 0

    if log_scale:
        interpolated_matrix = np.exp(interpolated_matrix) - 1

    return interpolated_matrix

def refill_invalid_points_periodic(matrix, x, y, method='cubic', extend_cols=10):
    """
    Refill invalid points (inf or nan) in a 2D matrix using scipy.interpolate.griddata.
    The second dimension of the matrix is periodic.

    Parameters:
    matrix (2D numpy array): The original matrix with dimensions (M, N).
    x (2D numpy array): The x-coordinates of the matrix.
    y (2D numpy array): The y-coordinates of the matrix.
    method (str): Interpolation method ('linear', 'nearest', or 'cubic').
    extend_cols (int): Number of columns to extend on each side to handle periodicity.

    Returns:
    2D numpy array: The matrix with invalid points refilled.
    """
    # Create a mask for valid points (neither inf nor nan)
    valid_mask = ~(np.isinf(matrix) | np.isnan(matrix))
    
    # Extract valid points
    valid_x = x[valid_mask]
    valid_y = y[valid_mask]
    valid_matrix = matrix[valid_mask]
    
    # Extend the matrix and coordinates to handle periodicity in the second dimension
    extended_matrix = np.hstack((matrix[:, -extend_cols:], matrix, matrix[:, :extend_cols]))
    extended_x = np.hstack((x[:, -extend_cols:] - 2*np.pi, x, x[:, :extend_cols] + 2*np.pi))
    extended_y = np.hstack((y[:, -extend_cols:], y, y[:, :extend_cols]))
    
    # Create a mask for valid points in the extended matrix
    extended_valid_mask = ~(np.isinf(extended_matrix) | np.isnan(extended_matrix))
    
    # Extract valid points from the extended matrix
    extended_valid_x = extended_x[extended_valid_mask]
    extended_valid_y = extended_y[extended_valid_mask]
    extended_valid_matrix = extended_matrix[extended_valid_mask]
    
    # Create a grid of points to interpolate (all points in extended_x and extended_y)
    grid_x = extended_x.flatten()
    grid_y = extended_y.flatten()
    
    # Interpolate using griddata
    interpolated_matrix = griddata(
        points=(extended_valid_x, extended_valid_y),  # Valid points
        values=extended_valid_matrix,                 # Valid values
        xi=(grid_x, grid_y),                          # Points to interpolate
        method=method,                                # Interpolation method
        fill_value=np.nan                             # Fill value for points outside the convex hull
    )
    
    # Reshape the interpolated values back to the original shape
    interpolated_matrix = interpolated_matrix.reshape(extended_matrix.shape)
    
    # Extract the central part of the interpolated matrix
    refilled_matrix = interpolated_matrix[:, extend_cols:-extend_cols]
    
    # Replace invalid points in the original matrix with the interpolated values
    matrix[~valid_mask] = refilled_matrix[~valid_mask]

    # Truncate the interpolated values to only positive values
    matrix[matrix < 0] = 0
    
    return matrix


def interpolate_wave(matrix, kx0, ky0, kx1, ky1):
    """
    Interpolates a 2D matrix along the kx (cols) and ky (rows) dimension, and fills with zeros 
    where the new wavenumber exceeds the original range.
    
    Parameters:
    matrix (2D numpy array): The original matrix with dimensions (kx0, ky0).
    kx0 (1D numpy array): The original K dimension values (kx0).
    ky0 (1D numpy array): 
    kx1 (1D numpy array): The new K dimension values (kx1).
    ky1 (1D numpy array): 
    
    Returns:
    2D numpy array: The interpolated matrix with dimensions (kx1, ky1).
    """
    
    # Interpolate matrix using bicubic interpolation
    interpolator = interp2d(kx0, ky0, matrix, kind='cubic')

    # Interpolate for the new kx1 and ky1
    interpolated_matrix = interpolator(kx1, ky1)

    # For the K1 values that exceed max(K0), set those values to 0
    kx1_mesh, ky1_mesh = np.meshgrid(kx1, ky1)
    kx0_mesh, ky0_mesh = np.meshgrid(kx0, ky0)

    # set the new wavenumber index to 0 if it exceeds the original range
    interpolated_matrix[np.abs(kx1_mesh) > np.max(np.abs(kx0_mesh))] = 0
    interpolated_matrix[np.abs(ky1_mesh) > np.max(np.abs(ky0_mesh))] = 0

    # truncate the interpolated values to only positive values
    interpolated_matrix[interpolated_matrix < 0] = 0

    return interpolated_matrix


def refill_invalid_points(x, y, A, method='cubic'):
    """
    Interpolate invalid points (inf or nan) in matrix A using coordinates x and y.
    
    Parameters:
        x (numpy.ndarray): 2D array of x-coordinates (M x N).
        y (numpy.ndarray): 2D array of y-coordinates (M x N).
        A (numpy.ndarray): 2D array of values (M x N), containing inf or nan.
        method (str): Interpolation method ('linear', 'nearest', or 'cubic').
    
    Returns:
        A_filled (numpy.ndarray): Matrix A with invalid points interpolated.
    """
    # Create a mask for valid points (neither inf nor nan)
    valid_mask = ~(np.isinf(A) | np.isnan(A))
    
    # Extract valid points
    valid_x = x[valid_mask]
    valid_y = y[valid_mask]
    valid_A = A[valid_mask]
    
    # Create a grid of points to interpolate (all points in x and y)
    grid_x = x.flatten()
    grid_y = y.flatten()
    
    # Interpolate using griddata
    interpolated_A = griddata(
        points=(valid_x, valid_y),  # Valid points
        values=valid_A,             # Valid values
        xi=(grid_x, grid_y),        # Points to interpolate
        method=method,              # Interpolation method
        fill_value=np.nan           # Fill value for points outside the convex hull
    )
    
    # Reshape the interpolated values back to the original shape
    A_filled = interpolated_A.reshape(A.shape)
    
    # Replace invalid points in A with the interpolated values
    A_filled[~valid_mask] = interpolated_A.reshape(A.shape)[~valid_mask]

    # truncate the interpolated values to only positive values
    A_filled[A_filled < 0] = 0
    
    return A_filled


def pair_up_partitions_polar(partition_num: int, partition_index: np.ndarray, dphi: float = None, iou_thres: float = 0.):
    """
    Pair up the partitions in polar coordinates.

    Parameters
    ----------
    partition_num : int
        The number of partitions.
    
    partition_index : np.ndarray
        The partitioned waves. The columns represent Azimuth while the rows represent Wavenumber.
        Azimuth must be evenly spaced.
    
    dphi : float
        The azimuthal resolution.

    iou_thres : float
        The IoU threshold for pairing partitions.

    Returns
    -------
    paired_partitions : np.ndarray
        The paired partitions.
    
    paired_iou : np.ndarray
        The IoU of the paired partitions.
    """

    self_paired_partitions = []
    paired_partitions = []
    paired_iou = []

    if dphi is None:
        dphi = 2*np.pi / partition_index.shape[1]
    
    rolling_column = int(np.round(np.pi / dphi))
    
    # Sort the pairs by IoU in descending order
    iou_pairs = []
    for p_label in range(1, partition_num + 1):
        current_partition = partition_index == p_label
        max_iou = 0
        best_match = None

        for other_label in range(p_label, partition_num + 1):
            partition_index_copy = partition_index.copy()
            other_partition = np.roll(partition_index_copy, rolling_column, axis=1) == other_label
            intersection = np.logical_and(current_partition, other_partition).sum()
            union = np.logical_or(current_partition, other_partition).sum()
            iou = intersection / union

            if iou > max_iou:
                max_iou = iou
                best_match = other_label

        if max_iou > iou_thres:
            iou_pairs.append((p_label, best_match, max_iou))

    iou_pairs.sort(key=lambda x: x[2], reverse=True)

    # Track paired partitions to ensure each partition is paired only once
    paired = set()
    for p1, p2, iou in iou_pairs:
        if p1 not in paired and p2 not in paired and p1 != p2:
            paired_partitions.append((p1, p2))
            paired_iou.append(iou)
            paired.add(p1)
            paired.add(p2)
        elif p1 not in paired and p2 not in paired and p1 == p2:
            self_paired_partitions.append(p1)
            paired.add(p1)

    return paired_partitions, paired_iou, self_paired_partitions


def align_pair_up_paritions_swim(partition_index_swim, paired_swim_parts):
    """
    Align the paired partitions in the swim partition index.

    Parameters
    ----------
    partition_index_swim : np.ndarray
        The partitioned waves in swim coordinates.
    
    paired_swim_parts : list of tuples
        The paired partitions.

    Returns
    -------
    aligned_partition_index : np.ndarray
        The aligned partition index.
    """

    aligned_partition_index = np.zeros_like(partition_index_swim)
    cols_rolling = partition_index_swim.shape[1] // 2
    
    for p1, p2 in paired_swim_parts:
        part01 = partition_index_swim.copy()
        part01[partition_index_swim == p1] = True
        part01[partition_index_swim != p1] = False

        part02 = partition_index_swim.copy()
        part02[partition_index_swim == p2] = True
        part02[partition_index_swim != p2] = False

        aligned_partition_index[np.logical_and(part01, np.roll(part02, cols_rolling, axis=1))] = p1
        aligned_partition_index[np.logical_and(np.roll(part01, cols_rolling, axis=1), part02)] = p2

    aligned_partition_index = _resign_partition_labels(aligned_partition_index, background_label=0)
    
    return aligned_partition_index


def calculate_iou(matrix1, label1, matrix2, label2):
    """Calculate the Intersection over Union (IoU) between two partitions."""
    intersection = np.logical_and(matrix1 == label1, matrix2 == label2)
    union = np.logical_or(matrix1 == label1, matrix2 == label2)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def partition_distances(col_scale, row_scale, nums_x, partition_index_x, nums_y, partition_index_y):
    final_dist = np_full((nums_x, nums_y), np.nan)

    for x_part in range(1, nums_x+1):
        x_locs = np.argwhere(partition_index_x == x_part)
        x_rows = sorted(set(x_locs[:, 0]))
        x_cols = sorted(set(x_locs[:, 1]))
        x_cols_diff = np.diff(np.array(x_cols))
        x_non_continue_index = np.argwhere(x_cols_diff > 1)
        if len(x_non_continue_index) > 0:
            x_cols = np.mod(x_cols - x_cols[x_non_continue_index[0]] - 1, partition_index_x.shape[1])
        
        x_part_rtho, x_part_phi = (x_rows[0] + x_rows[-1])//2, (x_cols[0] + x_cols[-1])//2
        x_part_cartx, x_part_carty = col_scale[x_part_rtho] * np.cos(row_scale[x_part_phi]), col_scale[x_part_rtho] * np.sin(row_scale[x_part_phi])

        for y_part in range(1, nums_y+1):
            y_locs = np.argwhere(partition_index_y == y_part)
            y_rows = sorted(set(y_locs[:, 0]))
            y_cols = sorted(set(y_locs[:, 1]))
            y_cols_diff = np.diff(np.array(y_cols))
            y_non_continue_index = np.argwhere(y_cols_diff > 1)
            if len(y_non_continue_index) > 0:
                y_cols = np.mod(y_cols - y_cols[y_non_continue_index[0]] - 1, partition_index_y.shape[1])
            
            y_part_row, y_part_col = (y_rows[0] + y_rows[-1])//2, (y_cols[0] + y_cols[-1])//2
            y_part_cartx, y_part_carty = col_scale[y_part_row] * np.cos(row_scale[y_part_col]), col_scale[y_part_row] * np.sin(row_scale[y_part_col])

            final_dist[x_part - 1, y_part - 1] = np.sqrt((x_part_cartx - y_part_cartx)**2 + (x_part_carty - y_part_carty)**2)
    
    return final_dist


def partition_iou(nums_x, partition_index_x, nums_y, partition_index_y):
    final_dist = np_full((nums_x, nums_y), np.nan)

    for x_part in range(1, nums_x+1):
        x_locs = (partition_index_x == x_part)
        for y_part in range(1, nums_y+1):
            y_locs = (partition_index_y == y_part)

            intersection = np.logical_and(x_locs, y_locs).sum()
            union = np.logical_or(x_locs, y_locs).sum()

            final_dist[x_part - 1, y_part - 1] = intersection / union
    
    return final_dist


def partition_intersaction(nums_x, partition_index_x, nums_y, partition_index_y):
    final_dist = np_full((nums_x, nums_y), np.nan)

    for x_part in range(1, nums_x+1):
        x_locs = (partition_index_x == x_part)
        for y_part in range(1, nums_y+1):
            y_locs = (partition_index_y == y_part)

            intersection = np.logical_and(x_locs, y_locs).sum()

            final_dist[x_part - 1, y_part - 1] = intersection
    
    return final_dist


def _kernel_based_partition_representation(x: np.ndarray, partition_x: np.ndarray):
    res = np.zeros_like(partition_x).astype(np.float64)
    _, N = x.shape

    partition_num = partition_x.max()
    c_mesh, r_mesh = np.meshgrid(np.arange(x.shape[1]), np.arange(x.shape[0]))
    r_mesh = r_mesh.astype(np.float64)
    c_mesh = c_mesh.astype(np.float64)
    for p_idx in range(1, partition_num + 1):
        x_clone = x.copy()
        partition_index_clone = partition_x.copy()
        x_clone[partition_x != p_idx] = 0
        r, c = np.unravel_index(np.argmax(x_clone.ravel()), x_clone.shape)
        partition_index_clone = np.roll(partition_index_clone, -c+N//2, axis=1)
        r, c = float(r), float(c)

        peak_indices = np.argwhere(partition_index_clone == p_idx)
        r_l, c_l, r_r, c_r = peak_indices[:, 0].min(), peak_indices[:, 1].min(), peak_indices[:, 0].max(), peak_indices[:, 1].max()
        sigma_r = np.max([(r_r - r_l) / 2 / 3, 1])
        sigma_c = np.max([(c_r - c_l) / 2 / 3, 1])
        sigma_r, sigma_c = float(sigma_r), float(sigma_c)

        r_mesh_centeroid, c_mesh_centeroid = r_mesh - r, c_mesh - N//2
        gauss = np.exp(-0.5 * (r_mesh_centeroid**2 / sigma_r**2 + c_mesh_centeroid**2 / sigma_c**2))
        gauss = np.roll(gauss, int(c)-N//2, axis=1)
        gauss[partition_x != p_idx] = 0
        res += gauss
    
    return res


def find_optimal_shift(x: np.ndarray, partition_x: np.ndarray, y: np.ndarray, partition_y: np.ndarray, user_sr: str = 'flat', azimuth_constrain: bool=True):
    """
    Find the optimal shift to maximize the correlation between partition_x and partition_y.
    user_sr = 'flat', 'gaussian', and more to be added. Partition representation methods.
    Rows are non-periodic, and columns are periodic.
    Returns the shift (row_shift, col_shift) and the aligned partition_x.
    """
    partition_x_clone = partition_x.copy()

    # partition representation to facilitate the cross correlation process
    if user_sr == 'flat':
        partition_x = np.int32(np.bool_(partition_x))
        partition_y = np.int32(np.bool_(partition_y))
    elif user_sr == 'ksr':
        partition_x = _kernel_based_partition_representation(x, partition_x)
        partition_y = _kernel_based_partition_representation(y, partition_y)
    elif user_sr == 'origin':
        x_clone = x.copy()
        y_clone = y.copy()
        partition_x = x_clone * np.int32(partition_x > 0)
        partition_y = y_clone * np.int32(partition_y > 0)
    else:
        raise Exception("Not implemented yet")

    # Get the dimensions of the matrices
    rows, cols = partition_x.shape
    
    # Initialize variables to store the maximum correlation and optimal shift
    max_corr = -np.inf
    optimal_shift = (0, 0)
    
    # Iterate over possible row shifts (non-periodic)
    for row_shift in range(-1, 1):
        # Shift partition_x vertically (non-periodic)
        shifted_x = np.roll(partition_x, shift=row_shift, axis=0)
        
        # Handle non-periodic row shifts by cropping invalid regions
        if row_shift > 0:
            shifted_x[:row_shift, :] = 0  # Zero out invalid top rows
        elif row_shift < 0:
            shifted_x[row_shift:, :] = 0  # Zero out invalid bottom rows
        
        # Iterate over possible column shifts (periodic)
        if azimuth_constrain:
            col_shift_range = range(-1, 1)
        else:
            col_shift_range = range(-(cols - 1), cols)

        for col_shift in col_shift_range:
            # Shift partition_x horizontally (periodic)
            shifted_x_col = np.roll(shifted_x, shift=col_shift, axis=1)
            
            # Compute the correlation between shifted_x_col and partition_y
            correlation = np.sum(shifted_x_col * partition_y)
            
            # Update the optimal shift if this correlation is higher
            if correlation > max_corr:
                max_corr = correlation
                optimal_shift = (row_shift, col_shift)
    
    # Apply the optimal shift to partition_x
    row_shift, col_shift = optimal_shift
    aligned_partition_x = np.roll(partition_x_clone, shift=row_shift, axis=0)
    aligned_partition_x = np.roll(aligned_partition_x, shift=col_shift, axis=1)
    
    # Handle non-periodic row shifts by cropping invalid regions
    if row_shift > 0:
        aligned_partition_x[:row_shift, :] = 0  # Zero out invalid top rows
    elif row_shift < 0:
        aligned_partition_x[row_shift:, :] = 0  # Zero out invalid bottom rows
    
    return optimal_shift, aligned_partition_x, max_corr


# turn the efth into slope spectrum of wavenumber domain
def efth2Sk(efth: np.ndarray, frequency: np.ndarray, PHI: np.ndarray) -> np.ndarray:
    PHI_mesh, frequency_mesh = np.meshgrid(PHI, frequency)

    gravity_acceleration = 9.81
    K_mesh = 4 * np.pi ** 2 * frequency_mesh ** 2 / gravity_acceleration
    Sk = efth * frequency_mesh / 2 * 180 / np.pi

    return Sk, K_mesh, PHI_mesh


def cal_hs(wave_spec: np.ndarray, x_mesh: np.ndarray, y_mesh: np.ndarray, spec_mode: str='skth', ) -> float:
    """
    Calculate the significant wave height (Hs) from the wave spectrum.

    Parameters
    ----------
    wave_spec : np.ndarray
        The wave spectrum.
    
    x_mesh : np.ndarray
        The x-coordinates of the wave spectrum.
    
    y_mesh : np.ndarray
        The y-coordinates of the wave spectrum.
    
    spec_mode : str
        The mode of the wave spectrum ('skth', 'fkth', 'fkxy'). skth denotes slope spectrum S(K, theta), fkth represents wave spectrum F(K, theta), while fkxy denotes wave spectrum F(Kx, Ky). Slope spectrum S(K, theta) = K^2*F(K, theta).

    Returns
    -------
    float
        The significant wave height (Hs).
    """

    to_int_term = wave_spec.copy()
    if spec_mode == 'skth':
        to_int_term = to_int_term / y_mesh
        Hs = 4 * np.sqrt(polar_integration(y_mesh, x_mesh, to_int_term))
    elif spec_mode == 'fkth':
        to_int_term = to_int_term * y_mesh
        Hs = 4 * np.sqrt(polar_integration(y_mesh, x_mesh, to_int_term))
    elif spec_mode == 'fkxy':
        Hs = 4 * np.sqrt(np.trapz(np.trapz(to_int_term, y_mesh, axis=0), x_mesh[0, :]))
    else:
        raise Exception("Invalid spec_mode, spec_mode should be one of 'skth', 'fkth', 'fkxy'.")

    return Hs


def cal_st(wave_spec: np.ndarray, x_mesh: np.ndarray, y_mesh: np.ndarray, spec_mode: str='skth', cutoff_wavenumber = np.nan, depth: float = 0) -> tuple[float, float]:
    """
    Calculate the magnitude (cm/s) and direction (degrees) of Stokes drift velocity from the wave spectrum.

    Parameters
    ----------
    wave_spec : np.ndarray
        The wave spectrum.
    
    x_mesh : np.ndarray
        Represents azimuth in '_kth' mode, and Kx in 'fkxy' mode.
    
    y_mesh : np.ndarray
        Represents wavenumber in '_kth' mode, and Ky in 'fkxy' mode.
    
    spec_mode : str
        The mode of the wave spectrum ('skth', 'fkth', 'fkxy'). skth denotes slope spectrum S(K, theta), fkth represents wave spectrum F(K, theta), while fkxy denotes wave spectrum F(Kx, Ky). Slope spectrum S(K, theta) = K^2*F(K, theta).

    Returns
    -------
    st_mag : float
        The magnitude of Stokes drift velocity. unit cm/s

    st_dic : float
        The direction of Stokes drift velocity. unit degrees wrt the Geo north in counter-clockwise rotation.
    """

    G = 9.81
    to_int_term = wave_spec.copy()
    # if spec_mode in 'skth':
    #     K = y_mesh.copy()
    #     if cutoff_wavenumber is not np.nan:
    #         to_int_term[K > cutoff_wavenumber] = 0
    #     PHI = x_mesh.copy()
    #     to_int_term_x = 2 * np.sqrt(G * K) * np.cos(PHI) * to_int_term
    #     to_int_term_y = 2 * np.sqrt(G * K) * np.sin(PHI) * to_int_term
    # elif spec_mode == 'fkth':
    #     K = y_mesh.copy()
    #     if cutoff_wavenumber is not np.nan:
    #         to_int_term[K > cutoff_wavenumber] = 0
    #     PHI = x_mesh.copy()
    #     to_int_term_x = 2 * np.sqrt(G * K) * K**2 * np.cos(PHI) * to_int_term
    #     to_int_term_y = 2 * np.sqrt(G * K) * K**2 * np.sin(PHI) * to_int_term
    if spec_mode in ['skth', 'fkth']:
        K = y_mesh.copy()
        if cutoff_wavenumber is not np.nan:
            to_int_term[K > cutoff_wavenumber] = 0
        PHI = x_mesh.copy()

        # log-polar coords must extend one more column to the right
        # to include the periodic boundary effect.
        K = np.hstack([K, K[:, 0:1]])
        PHI = np.hstack([PHI, PHI[:, 0:1] + 2 * np.pi])
        to_int_term = np.hstack([to_int_term, to_int_term[:, 0:1]])

        compensation_term = 1
        if spec_mode == 'fkth':
            compensation_term = K**2
        to_int_term_x = 2 * np.sqrt(G * K) * np.cos(PHI) * to_int_term
        to_int_term_y = 2 * np.sqrt(G * K) * np.sin(PHI) * to_int_term

        to_int_term_x = 2 * np.sqrt(G * K) * compensation_term * np.cos(PHI) * to_int_term * np.exp(-2 * K * depth)
        to_int_term_y = 2 * np.sqrt(G * K) * compensation_term * np.sin(PHI) * to_int_term * np.exp(-2 * K * depth)

        st_north = np.trapz(np.trapz(to_int_term_x, K, axis=0), PHI[0, :])
        st_east = np.trapz(np.trapz(to_int_term_y, K, axis=0), PHI[0, :])
    elif spec_mode == 'fkxy':
        K = np.sqrt(x_mesh**2 + y_mesh**2)
        if cutoff_wavenumber is not np.nan:
            to_int_term[K > cutoff_wavenumber] = 0
        to_int_term_x = 2 * np.sqrt(G * K) * x_mesh * to_int_term * np.exp(-2 * K * depth)
        to_int_term_y = 2 * np.sqrt(G * K) * y_mesh * to_int_term * np.exp(-2 * K * depth)
        
        st_north = np.trapz(np.trapz(to_int_term_x, y_mesh, axis=0), x_mesh[0, :])
        st_east = np.trapz(np.trapz(to_int_term_y, y_mesh, axis=0), x_mesh[0, :])
    else:
        raise Exception("Invalid spec_mode, spec_mode should be one of 'skth', 'fkth', 'fkxy'.")

    st_mag = np.sqrt(st_north**2 + st_east**2) * 100 # to transfer into cm/s
    st_dic = np.mod(np.arctan2(st_east, st_north)*180/np.pi, 360) # clockwise to the geo-north

    return st_mag, st_dic


def label_wind_wave(u10: float, v10: float, K: np.ndarray, PHI: np.ndarray, partition_index: np.ndarray, slope_spec: np.ndarray, dispersion_relation: str='deep water', ocean_depth: float=np.inf, wave_age_thres: float=1.5) -> np.ndarray:
    """
    Label the wind wave and swell wave in the wave spectrum.

    Parameters
    ----------
    u10 : float
        The eastward component of wind speed at 10m height
    
    v10 : float
        The northward component of wind direction at 10m height.
    
    dispersion_relation : str
        The dispersion relation to use ('deep water' or 'shallow water').
    
    ocean_depth : float
        The ocean depth for shallow water calculations. Default is np.inf (deep water).

    wave_age_thres : float
        The threshold for wave age to distinguish between wind waves and swell waves. Default is 1.5.

    Returns
    -------
    wind_flag : np.ndarray
        The wind flag of the wave spectrum. 1 for wind wave, 0 for swell wave.
    """
    if K.shape[0] != PHI.shape[0]:
        raise ValueError("K and PHI must have the same number of rows.")
    if K.shape[1] != PHI.shape[1]:
        raise ValueError("K and PHI must have the same number of columns.")
    if slope_spec.shape[0] != K.shape[0]:
        raise ValueError("slope_spec and K must have the same number of rows.")
    if slope_spec.shape[1] != K.shape[1]:
        raise ValueError("slope_spec and K must have the same number of columns.")
    if dispersion_relation not in ['deep water', 'shallow water']:
        raise ValueError("dispersion_relation must be either 'deep water' or 'shallow water'.")
    
    if dispersion_relation == 'deep water':
        g = 9.81
        omega = np.sqrt(g * K)
    else:
        g = 9.81
        if ocean_depth <= 0 or ocean_depth is np.nan or ocean_depth is np.inf:
            raise ValueError("Ocean depth must be positive for shallow water.")
        omega = np.sqrt(g * K * np.tanh(K * ocean_depth))

    freq = omega / (2 * np.pi)
    Sf = slope_spec * g / np.pi / omega # denotes f^2 * F(f, theta)
    
    # Step 0: Align the wind coords (east-north) with wave coords (north-west)
    u_north = v10
    u_east = u10
    u_magnitude = np.sqrt(u_north**2 + u_east**2)
    wind_direction = np.arctan2(u_east, u_north) # rad

    wave_age = omega / K / u_magnitude

    wave_age_spread = wave_age_thres * np.cos(PHI - wind_direction)
    wave_age_spread[wave_age_spread < 0] = -np.inf

    wind_flag = np.zeros_like(K)
    wind_flag[wave_age < wave_age_spread] = 1
    wind_flag[wave_age >= wave_age_spread] = 0

    # find all partitions whose peak energy falls into the wind area
    wind_parts = []
    partition_num = partition_index.max()
    for p_idx in range(1, partition_num + 1):
        partition_index_clone = partition_index.copy()
        partition_index_clone[partition_index != p_idx] = 0
        partition_index_clone[partition_index == p_idx] = 1
        spec_part = partition_index_clone * slope_spec

        fp_idx, phip_idx = np.unravel_index(np.argmax(spec_part), spec_part.shape)
        if wind_flag[fp_idx, phip_idx] == 1:
            wind_parts.append(p_idx)

    return wind_flag, wind_parts


def wave_spreading(partition_flag: np.ndarray, K: np.ndarray, PHI: np.ndarray, slope_spec: np.ndarray) -> float:
    """
    Calculate the wave spreading parameter (gamma) from the wave spectrum.

    Parameters
    ----------
    partition_flag : np.ndarray
        The partition flag of the wave spectrum. Filled with 0 and 1, 1 flag is the partition to be calculated.
    
    K : np.ndarray
        The wavenumber array.
    
    PHI : np.ndarray
        The azimuthal angle array.
    
    slope_spec : np.ndarray
        The slope spectrum in the wavenumber domain.

    Returns
    -------
    spreads_coef : float
        The wave spreading parameter (gamma).

    Ref
    -------
    Hanson, J. L., & Phillips, O. M. (2001). Automated Analysis of Ocean Surface Directional Wave Spectra. Journal of Atmospheric and Oceanic Technology, 18(2), 277–293. https://doi.org/10.1175/1520-0426(2001)018<0277:AAOOSD>2.0.CO;2
    """

    # transfer the slope spectrum to the frequency domain
    g = 9.81 # m/s^2
    omega = np.sqrt(g * K)
    freq = omega / (2 * np.pi)
    Sf = 2 * slope_spec # denotes f^2 * F(f, theta)， from "Sf / f * df = Sk / k * dk"
    total_energy = polar_integration(K, PHI, slope_spec / K)

    spec_partitioned = Sf * partition_flag
    fx_bar = polar_integration(freq, PHI, spec_partitioned * np.cos(PHI)) / total_energy
    fy_bar = polar_integration(freq, PHI, spec_partitioned * np.sin(PHI)) / total_energy

    fx_2_bar = polar_integration(freq, PHI, spec_partitioned * freq * np.cos(PHI)**2) / total_energy
    fy_2_bar = polar_integration(freq, PHI, spec_partitioned * freq * np.sin(PHI)**2) / total_energy

    spreads_coef = fx_2_bar - fx_bar**2 + fy_2_bar - fy_bar**2

    if spreads_coef < 0:
        raise ValueError("The wave spreading coefficient is negative, which is not valid.")
    
    return spreads_coef


def peak_distance(partition_flag01: np.ndarray, partition_flag02: np.ndarray, K: np.ndarray, PHI: np.ndarray, slope_spec: np.ndarray):
    """
    Calculate the peak distance between two partitions in the wave spectrum.

    Parameters
    ----------
    partition_flag01 : np.ndarray
        The partition flag of the first wave spectrum. Filled with 0 and 1, 1 flag is the partition to be calculated.
    
    partition_flag02 : np.ndarray
        The partition flag of the second wave spectrum. Filled with 0 and 1, 1 flag is the partition to be calculated.
    
    K : np.ndarray
        The wavenumber array.
    
    PHI : np.ndarray
        The azimuthal angle array.
    
    slope_spec : np.ndarray
        The slope spectrum in the wavenumber domain.

    Returns
    -------
    peak_distance : float
        The peak distance between the two partitions.
    """

    # transfer the slope spectrum to the frequency domain
    g = 9.81 # m/s^2
    omega = np.sqrt(g * K)
    freq = omega / (2 * np.pi)
    Sf = slope_spec * g / np.pi / omega # denotes f^2 * F(f, theta)

    Sf01 = Sf * partition_flag01
    Sf02 = Sf * partition_flag02

    fp_01_idx, phip_01_idx = np.unravel_index(np.argmax(Sf01), Sf01.shape)
    fp_02_idx, phip_02_idx = np.unravel_index(np.argmax(Sf02), Sf02.shape)

    fp_x01 = freq[fp_01_idx, phip_01_idx] * np.cos(PHI[fp_01_idx, phip_01_idx])
    fp_y01 = freq[fp_01_idx, phip_01_idx] * np.sin(PHI[fp_01_idx, phip_01_idx])

    fp_x02 = freq[fp_02_idx, phip_02_idx] * np.cos(PHI[fp_02_idx, phip_02_idx])
    fp_y02 = freq[fp_02_idx, phip_02_idx] * np.sin(PHI[fp_02_idx, phip_02_idx])

    peak_distance = (fp_x01 - fp_x02)**2 + (fp_y01 - fp_y02)**2

    return peak_distance


def polar_integration(rho: np.ndarray, phi: np.ndarray, func: np.ndarray) -> float:
    """
    Perform polar integration of a 2D array.

    Parameters
    ----------
    rho : np.ndarray
        The radial coordinate (r), increasing with rows.
    
    phi : np.ndarray
        The angular coordinate (phi), increasing with columns.
    
    func : np.ndarray
        The function to be integrated.

    Returns
    -------
    res : float
        The result of the polar integration.
    """

    # close the boundary of func by extending the phi dimension
    func_extended = np.hstack((func, func[:, 0:1]))
    phi_extended = np.hstack((phi, phi[:, 0:1] + 2 * np.pi))
    # rho = np.hstack((rho, rho[:, 0:1]))

    temp = np.trapz(func_extended, phi_extended, axis=1)
    res = np.trapz(temp, rho[:, 0], axis=0)

    return res


def merge_partitions(partition_index: np.ndarray, partition_num: int, K: np.ndarray, PHI: np.ndarray, slope_spec: np.ndarray, merge_thres: float=0.4) -> np.ndarray:
    """
    Merge partitions in the wave spectrum.

    Parameters
    ----------
    partition_index : np.ndarray
        The partition index of the wave spectrum.
    
    partition_num : int
        The number of partitions.
    
    partition_merge : list
        The list of partitions to be merged.

    merge_thres : float
        The threshold for merging partitions. Default is 0.4.

    Returns
    -------
    merged_partition_index : np.ndarray
        The merged partition index. Lable 1 is assigned to wind wave if there is one.
    
    start_with_wind_flag : bool
        True if the merged partition starts with wind wave, False otherwise.
    
    References
    Hanson, J. L., & Phillips, O. M. (2001). Automated Analysis of Ocean Surface Directional Wave Spectra. Journal of Atmospheric and Oceanic Technology, 18(2), 277–293. https://doi.org/10.1175/1520-0426(2001)018<0277:AAOOSD>2.0.CO;2
    """

    # Input validation
    labels = np.unique(partition_index)
    labels.sort()
    num_labels = len(labels) if labels[0] != 0 else len(labels) - 1
    if num_labels != partition_num:
        raise ValueError(f"The actual number of partitions ({num_labels}) don't equal the given partition_num({partition_num}) parameter.")
    if not np.all(np.diff(labels) == 1):
        raise ValueError("The labels are not consecutive integers. Please resign the label number before using this function.")
    
    partition_connect_flag = np_zeros((partition_num, partition_num), dtype=np.float64)
    spreads_coef_list = np_zeros((partition_num, 1), dtype=np.float64)

    # calculate the spreading coefficient for each partition, background partition 0 is negelected
    for p_idx in range(1, partition_num + 1):
        partition_index_clone = partition_index.copy()
        partition_index_clone[partition_index != p_idx] = 0
        partition_index_clone[partition_index == p_idx] = 1

        spreads_coef = wave_spreading(partition_index_clone, K, PHI, slope_spec)
        spreads_coef_list[p_idx - 1] = spreads_coef
    
    # calculate the distance between each pair of partitions, and see if all the distance is smaller than the threshold
    for p_idx in range(1, partition_num + 1):
        partition_index_clone = partition_index.copy()
        partition_index_clone[partition_index != p_idx] = 0
        partition_index_clone[partition_index == p_idx] = 1

        for q_idx in range(p_idx + 1, partition_num + 1):
            partition_index_clone2 = partition_index.copy()
            partition_index_clone2[partition_index != q_idx] = 0
            partition_index_clone2[partition_index == q_idx] = 1

            dist = peak_distance(partition_index_clone, partition_index_clone2, K, PHI, slope_spec)

            if (dist < merge_thres * spreads_coef_list[p_idx - 1]) and (dist < merge_thres * spreads_coef_list[q_idx - 1]):
                partition_connect_flag[p_idx - 1, q_idx - 1] = dist
                partition_connect_flag[q_idx - 1, p_idx - 1] = dist
    
    # merge the partitions with lower peak distance, and exclude the partitions labels as wind waves
    partition_index_clone = partition_index.copy()
    groups = find_connected_components(partition_connect_flag)
    for parts_idx in groups:
        parts_idx = [idx for idx in parts_idx]
        parts_idx = np.sort(parts_idx)
        for idx in parts_idx[1:]: # label merged partitions with the first partition label
            partition_index_clone[partition_index == idx + 1] = parts_idx[0] + 1

    merged_partition_index = _resign_partition_labels(partition_index_clone)
    
    return merged_partition_index


def find_connected_components(p2p_matrix):
    n = len(p2p_matrix)
    graph = {i: [] for i in range(n)}
    
    # Build adjacency list (undirected graph)
    for i in range(n):
        for j in range(i + 1, n):  # Use symmetry
            if p2p_matrix[i][j] > 0:
                graph[i].append(j)
                graph[j].append(i)
    
    visited = set()
    components = []
    
    # DFS to find connected nodes
    def dfs(node, component):
        component.append(node)
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor, component)
    
    for node in range(n):
        if node not in visited:
            component = []
            dfs(node, component)
            components.append(component)
    
    return components


def partition_edges(parts, x, y):
    M, N = parts.shape[0], parts.shape[1]
    edges_set = []
    for j in range(1, M - 1):
        for i in range(N):
            dx_left = np.mod(x[j, i] - x[j, (i - 1) % N], 2*np.pi)
            dx_right = np.mod(x[j, (i + 1) % N] - x[j, i], 2*np.pi)
            dy_up = y[j, i] - y[j - 1, i]
            dy_down = y[j + 1, i] - y[j, i]
            if parts[j, i] != parts[j, (i - 1) % N]:
                edges_set.append([(x[j, i]-dx_left/2, y[j, i]-dy_up/2), (x[j, i]-dx_left/2, y[j, i]+dy_down/2)])
            if parts[j, i] != parts[j, (i + 1) % N]:
                edges_set.append([(x[j, i]+dx_right/2, y[j, i]-dy_up/2), (x[j, i]+dx_right/2, y[j, i]+dy_down/2)])
            if parts[j, i] != parts[(j - 1), i]:
                edges_set.append([(x[j, i]-dx_left/2, y[j, i]-dy_up/2), (x[j, i]+dx_right/2, y[j, i]-dy_up/2)])
            if parts[j, i] != parts[(j + 1), i]:
                edges_set.append([(x[j, i]-dx_left/2, y[j, i]+dy_down/2), (x[j, i]+dx_right/2, y[j, i]+dy_down/2)])
    return edges_set