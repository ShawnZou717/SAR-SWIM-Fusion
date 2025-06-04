import numpy as np
from .waveutils import *


def find_min_positions(A, row_pairs):
    N = A.shape[0]  # Get the size of the matrix
    result = []  # List to store the results
    
    # Create a dictionary for fast lookup of paired rows
    row_dict = {}
    for r0, r1 in row_pairs:
        row_dict[r0 - 1] = r1 - 1
        row_dict[r1 - 1] = r0 - 1
    
    # while not np.isnan(np.nanmin(A)) and np.nanmin(A) != 0:  # While there are valid values left
    while np.any(np.isfinite(A)) and np.nanmin(A) != 0:  # While there are valid values left
        # Find the minimum value and its position
        min_value = np.nanmin(A)
        min_pos = np.unravel_index(np.nanargmin(A), A.shape)
        r0, c0 = min_pos
        
        # Store the result (position and value)
        result.append((r0, c0, min_value))
        
        # Find the paired row using the dictionary
        paired_row = row_dict.get(r0, None)
        
        # Exclude the row, column, and paired row
        A[r0, :] = np.nan
        A[:, c0] = np.nan

        # No need to exclude the paired row again, as it will be excluded in the next steps
        # if paired_row is not None:
        #     A[paired_row, :] = np.nan
        
    return result

def sar_swim_fusion(
    swim_spec: np.ndarray,
    swim_k_spectra: np.ndarray,
    swim_phi_spectra: np.ndarray,
    swim_heading: float,
    sar_spec: np.ndarray,
    sar_k_spectra: np.ndarray,
    sar_phi_spectra: np.ndarray,
    sar_heading: float,
    u10: float,
    v10: float,
    non_valid_ratio: float = 0.2,
    tolerance_limit: float = 1E-2
)-> tuple[np.ndarray, np.ndarray, list, list, list]:
    """
    Fuse wave spectra data from SWIM, SAR, and PAPA.

    Parameters
    ----------
    swim_spec : np.ndarray
        Slope spectrum from SWIM, shape (nk, nphi), units: m^2/rad.
    swim_k_spectra : np.ndarray
        Wavenumber array for SWIM spectrum, same shape as swim_spec, units: 1/m.
    swim_phi_spectra : np.ndarray
        Azimuth array for SWIM spectrum, same shape as swim_spec, units: rad.
    swim_heading: float
        Heading of the SWIM sensor to the geo-north, units: rad.
    sar_spec : np.ndarray
        Wave spectrum from SAR, shape (nk_sar, nphi_sar), units: m^4/rad.
    sar_k_spectra : np.ndarray
        Wavenumber array for SAR spectrum, same shape as sar_spec, units: 1/m.
    sar_phi_spectra : np.ndarray
        Azimuth array for SAR spectrum, same shape as sar_spec, units: rad.
    sar_heading: float
        Heading of the SAR sensor to the geo-north, units: rad.
    u10 : float
        10-meter altitude wind speed (eastward component), units: m/s.
    v10 : float
        10-meter altitude wind speed (northward component), units: m/s.
    non_valid_ratio : float
        Ratio of non-valid values in the spectra. If the ratio exceeds this value, the function returns NaN matrix.
        Default is 0.2.
    tolerance_limit : float
        Tolerance limit for the significant wave height (Hs) bias after fusion. If the bias exceeds this limit, the function raises an error.

    Returns
    -------
    swim_spec_fused : np.ndarray
        Fused wave spectrum, shape (nk, nphi), units: m^2/rad. If the input spectra are invalid, returns NaN matrix.
    partition_index_swim : np.ndarray
        Partition index for the SWIM spectrum, shape (nk, nphi). Each partition is labeled with a unique integer.
    removed_partitions : list
        List of removed partitions from the SWIM spectrum. Each partition is represented by its index.
    remained_partitions : list
        List of remaining partitions from the SWIM spectrum. Each partition is represented by its index.
    to_merge_wind_parts : list
        List of partitions to be merged as a wind wave. Each partition is represented by its index.
    """
    # 0.0 Check the input dimensions
    if not (swim_spec.shape == swim_k_spectra.shape == swim_phi_spectra.shape):
        raise ValueError("swim_spec, swim_k_spectra, and swim_phi_spectra must have the same shape")
    if not (sar_spec.shape == sar_k_spectra.shape == sar_phi_spectra.shape):
        raise ValueError("sar_spec, sar_k_spectra, and sar_phi_spectra must have the same shape")
    
    # 0.1 Limiting the valid ratio of swim and sar spectra
    invalid_swim = ~np.isfinite(swim_spec)
    invalid_sar = ~np.isfinite(sar_spec)
    ratio_swim = np.sum(invalid_swim) / swim_spec.size
    ratio_sar = np.sum(invalid_sar) / sar_spec.size
    if ratio_swim > non_valid_ratio or ratio_sar > non_valid_ratio:
        return np.full_like(swim_spec, np.nan), np.full_like(swim_spec, np.nan), [], [], []

    # 0.2 Sort the two spectra according to the geo-north aligned azimuth
    swim_phi_spectra = np.mod(swim_phi_spectra + swim_heading, 2*np.pi)
    sorted_indices = np.argsort(swim_phi_spectra[0, :])

    swim_phi_spectra = swim_phi_spectra[:, sorted_indices]
    swim_spec = swim_spec[:, sorted_indices]

    sar_phi_spectra = np.mod(sar_phi_spectra + sar_heading, 2*np.pi)
    sorted_indices = np.argsort(sar_phi_spectra[0, :])

    sar_phi_spectra = sar_phi_spectra[:, sorted_indices]
    sar_spec = sar_spec[:, sorted_indices]

    # 0.3 Fillin the non-finite values with interpolation
    swim_spec = refill_invalid_points_periodic(swim_spec, swim_phi_spectra, swim_k_spectra, method='cubic')
    sar_spec = refill_invalid_points_periodic(sar_spec, sar_phi_spectra, sar_k_spectra, method='cubic')

    # 1.0 Segmentation of the SWIM spectrum
    nums_swim, partition_index_swim = partition_waves(swim_k_spectra, swim_phi_spectra, swim_spec, gaus_blur_ops=3, remove_invalid_parts=True, distinguish_fg_bg=False)
    # The order-dependent process of watershed breaks the symmetry of the partition for SWIM (which is suppose to be strictly symmetric), so we need to re-align paired up partitions. 
    # Other than that, the tie-breaking mechanism also causes unsymmetry during the coords transformation, another reason that we need to re-align the paired up partitions.
    paired_swim_parts, pair_iou, self_paired_parts_swim = pair_up_partitions_polar(nums_swim, partition_index_swim)
    partition_index_swim = align_pair_up_paritions_swim(partition_index_swim, paired_swim_parts)
    nums_swim = partition_index_swim.max() # refresh the number of partitions

    # 1.0.1 Merge the SWIM partitions, and label the wind wave with the paired ECMWF wind speed in L2 product
    merged_partition_index_swim = merge_partitions(partition_index_swim, nums_swim, swim_k_spectra, swim_phi_spectra, swim_spec)
    if merged_partition_index_swim.max() % 2 != 0:
        raise ValueError("Unexpected merging behavior of SWIM partitions. The number of partitions after merge should be even.")
    
    wind_flag, wind_parts = label_wind_wave(u10, v10, swim_k_spectra, swim_phi_spectra, merged_partition_index_swim, swim_spec, 'deep water')
    paired_swim_parts, pair_iou, self_paired_parts_swim = pair_up_partitions_polar(nums_swim, merged_partition_index_swim)

    # Create a dictionary for fast lookup of paired rows
    swim_part_dict = {}
    for r0, r1 in paired_swim_parts:
        swim_part_dict[r0 - 1] = r1 - 1
        swim_part_dict[r1 - 1] = r0 - 1

    partition_index_swim = merged_partition_index_swim.copy()
    nums_swim = partition_index_swim.max()

    # 1.1 Segmentation of the SAR spectrum
    # Sentinel-1A SAR doesn't contain non-valid values, so we don't need to refill the invalid points
    sar_spec2swim = interpolate_wave_polar(sar_spec, sar_k_spectra[:, 0], sar_phi_spectra[0, :], swim_k_spectra[:, 0], swim_phi_spectra[0, :])
    sar_spec2swim = sar_spec2swim * swim_k_spectra**2 # align the spectra into the same unit

    nums_sar, partition_index_sar = partition_waves(swim_k_spectra, swim_phi_spectra, sar_spec2swim, gaus_blur_ops=1, remove_invalid_parts=False, distinguish_fg_bg=True, ambiguity_flag=False)
    merged_sar_partition_index = merge_partitions(partition_index_sar, nums_sar, swim_k_spectra, swim_phi_spectra, sar_spec2swim)
    partition_index_sar = merged_sar_partition_index.copy()
    nums_sar = partition_index_sar.max()
    paired_sar_parts, sar_pair_iou, self_paired_parts_sar = pair_up_partitions_polar(nums_sar, partition_index_sar, iou_thres=0.5)

    print(f"The paired up SAR partitions are {paired_sar_parts}. The IOU of SAR is {sar_pair_iou}")
    for p1, p2 in paired_sar_parts:
        partition_index_sar[partition_index_sar == p2] = 0
        partition_index_sar[partition_index_sar == p1] = 0
    for part_no in self_paired_parts_sar:
        partition_index_sar[partition_index_sar == part_no] = 0
    
    partition_index_sar = resign_partition_labels(partition_index_sar)
    nums_sar = partition_index_sar.max()

    # 2.1 Fusion the SWIM and SAR spectra, swell wave direction come from SAR, and wind wave direction come from ECMWF (integrated in the SWIM L2 product)
    # 2.1.1 Machting the SWIM and SAR partitions
    removed_partitions = []
    valid_swim_part_found = 0
    if nums_sar > 0: # if there are no SAR partitions, we don't need to match the partitions
        partition_intecs = partition_intersaction(nums_swim, partition_index_swim, nums_sar, partition_index_sar)

        # exclude the non-matched swim parts
        min_positions = find_min_positions(-partition_intecs, paired_swim_parts)
        
        # # Create a dictionary for fast lookup of paired rows
        # swim_part_dict = {}
        # for r0, r1 in paired_swim_parts:
        #     swim_part_dict[r0 - 1] = r1 - 1
        #     swim_part_dict[r1 - 1] = r0 - 1

        swim_spec_fused = swim_spec.copy()

        # Before put the SAR partitions into SWIM, we should notice the conversation of the SWIM energy.
        # It is possible that SAR partitions remove a pair of SWIM partitions, in which case wouldn't cause any biases to the Stokes drift, but the significant wave
        # height would be biased. So we need to take 2 steps to avoid this kind of biases.
        # 1st. We quickly go through the matched SAR and SWIM partitions, to check if there are any pairs of SWIM partitions to be removed, if yes, keep this pair of SWIM partitions by disabling the corresponding SAR partitions.
        # 2nd, the significant wave height must keep exactly the same before and after fusion. If there are biases, then there must have been mistakes during the fusion process.

        # Check if any matched_swim_part_index in min_positions is paired up in swim_part_dict. 
        # Disable all the SAR partitions who matches pairs of SWIM partitions, as it would break the conversation of the SWIM energy if not.
        paired_swim_parts_found = []
        disable_index = []
        for idx, item in enumerate(min_positions):
            matched_swim_part_index = item[0]
            paired_part = swim_part_dict.get(matched_swim_part_index, None)
            if paired_part is not None and paired_part in [pos[0] for pos in min_positions]:
                paired_swim_parts_found.append(matched_swim_part_index)
                disable_index.append(idx)

        if paired_swim_parts_found:
            print(f"These paired up SWIM partitions are excluded from the fusion: {paired_swim_parts_found}.")
        else:
            print("All SAR partitions are valid.")

        for idx, item in enumerate(min_positions):
            swim_part_index = item[0]
            if swim_part_dict.get(swim_part_index, None) is None or idx in disable_index:
                continue
            valid_swim_part_found += 1
            removed_partitions.append(swim_part_dict[swim_part_index])
            to_remove_part = partition_index_swim == (swim_part_dict[swim_part_index] + 1)
            swim_spec_fused[to_remove_part] = 0
            
            to_raise_part = partition_index_swim == (swim_part_index + 1)
            swim_spec_rolled = np.roll(swim_spec, swim_spec.shape[1]//2, axis=1)
            swim_spec_fused[to_raise_part] += swim_spec_rolled[to_raise_part]
    else:
        print("No valid SAR partition found.")
        swim_spec_fused = swim_spec.copy()
    
    # 2.1.2 Labeling the wind wave with the paired ECMWF wind speed in L2 product
    remained_partitions = [swim_part_dict[part_idx] for part_idx in removed_partitions if swim_part_dict.get(part_idx, None) is not None]

    # We stick to the observation information, so if the wind coverd wave parts were fused according to the SAR spectra in advance, no matter removed nor remained, it would need no more processing. Only those never-processed wind parts need to be fused.
    to_merge_wind_parts = [part_no for part_no in wind_parts if part_no - 1 not in removed_partitions and part_no - 1 not in remained_partitions]
    for part_no in to_merge_wind_parts:
        if swim_part_dict.get(part_no, None) is not None:
            valid_swim_part_found += 1
            to_remove_part = partition_index_swim == (swim_part_dict[part_no - 1] + 1)
            swim_spec_fused[to_remove_part] = 0

            to_raise_part = partition_index_swim == (part_no)
            swim_spec_rolled = np.roll(swim_spec, swim_spec.shape[1]//2, axis=1)
            swim_spec_fused[to_raise_part] += swim_spec_rolled[to_raise_part]
    
    hs_fused = cal_hs(swim_spec_fused, swim_phi_spectra, swim_k_spectra, 'skth')
    hs_raw = cal_hs(swim_spec, swim_phi_spectra, swim_k_spectra, 'skth')

    # Fusion causes the amplitude change, and the interpolation method in trapz is affected by these biases to produce a very tiny biaes upon the integration results.
    # That is why we need to set a tolerance to the Hs biases. Here the tolerance is set to 0.01m.
    if np.abs(hs_raw - hs_fused) > tolerance_limit:
        raise ValueError(f"Fusion Hs {hs_fused:.2f}m and raw Hs {hs_raw:.2f}m biases too large.")
    
    return swim_spec_fused, partition_index_swim, removed_partitions, remained_partitions, to_merge_wind_parts

    
def remove_ambiguity_accord_wind(
    swim_spec: np.ndarray,
    swim_k_spectra: np.ndarray,
    swim_phi_spectra: np.ndarray,
    swim_heading: float,
    u10: float,
    v10: float,
    non_valid_ratio: float = 0.2,
    tolerance_limit: float = 1E-2
)-> np.ndarray:
    """
    Remove ambiguity of the SWIM wave spectrum according to the wind speed.

    Parameters
    ----------
    swim_spec : np.ndarray
        Slope spectrum from SWIM, shape (nk, nphi), units: m^2/rad.
    swim_k_spectra : np.ndarray
        Wavenumber array for SWIM spectrum, same shape as swim_spec, units: 1/m.
    swim_phi_spectra : np.ndarray
        Azimuth array for SWIM spectrum, same shape as swim_spec, units: rad.
    swim_heading: float
        Heading of the SWIM sensor to the geo-north, units: rad.
    u10 : float
        10-meter altitude wind speed (eastward component), units: m/s.
    v10 : float
        10-meter altitude wind speed (northward component), units: m/s.
    non_valid_ratio : float
        Ratio of non-valid values in the spectra. If the ratio exceeds this value, the function returns NaN matrix.
        Default is 0.2.
    tolerance_limit : float
        Tolerance limit for the significant wave height (Hs) bias after fusion. If the bias exceeds this limit, the function raises an error.

    Returns
    -------
    swim_spec_fused : np.ndarray
        Fused wave spectrum, shape (nk, nphi), units: m^2/rad. If the input spectra are invalid, returns NaN matrix.
    """

    # 0.0 Check the input dimensions
    if not (swim_spec.shape == swim_k_spectra.shape == swim_phi_spectra.shape):
        raise ValueError("swim_spec, swim_k_spectra, and swim_phi_spectra must have the same shape")
    
    # 0.1 Limiting the valid ratio of swim and sar spectra
    invalid_swim = ~np.isfinite(swim_spec)
    ratio_swim = np.sum(invalid_swim) / swim_spec.size
    if ratio_swim > non_valid_ratio:
        return np.full_like(swim_spec, np.nan)

    # 0.2 Sort the two spectra according to the geo-north aligned azimuth
    swim_phi_spectra = np.mod(swim_phi_spectra + swim_heading, 2*np.pi)
    sorted_indices = np.argsort(swim_phi_spectra[0, :])

    swim_phi_spectra = swim_phi_spectra[:, sorted_indices]
    swim_spec = swim_spec[:, sorted_indices]

    # 0.3 Fillin the non-finite values with interpolation
    swim_spec = refill_invalid_points_periodic(swim_spec, swim_phi_spectra, swim_k_spectra, method='cubic')

    # 1.0 Calculate the wind wave direction
    wind_direction = np.arctan2(u10, v10) # clockwise rotated from the geo-north
    wind_direction = np.mod(wind_direction, 2 * np.pi)  # Convert to [0, 2*pi)

    