import numpy as np
import pytest
from waveutils import merge_partitions

def test_merge_partitions_valid_input():
    partition_index = np.array([
        [1, 1, 0, 2, 2],
        [1, 1, 0, 2, 2],
        [0, 0, 0, 0, 0],
        [3, 3, 0, 4, 4],
        [3, 3, 0, 4, 4]
    ])
    partition_num = 4
    K = np.ones((5, 5))
    PHI = np.ones((5, 5)) * np.pi / 4
    slope_spec = np.ones((5, 5))
    merge_thres = 0.5

    merged_partition_index = merge_partitions(partition_index, partition_num, K, PHI, slope_spec, merge_thres)
    assert merged_partition_index.shape == partition_index.shape
    assert np.all(merged_partition_index >= 0)

def test_merge_partitions_invalid_partition_num():
    partition_index = np.array([
        [1, 1, 0, 2, 2],
        [1, 1, 0, 2, 2],
        [0, 0, 0, 0, 0],
        [3, 3, 0, 4, 4],
        [3, 3, 0, 4, 4]
    ])
    partition_num = 3  # Incorrect partition number
    K = np.ones((5, 5))
    PHI = np.ones((5, 5)) * np.pi / 4
    slope_spec = np.ones((5, 5))
    merge_thres = 0.5

    with pytest.raises(ValueError, match="The actual number of partitions.*"):
        merge_partitions(partition_index, partition_num, K, PHI, slope_spec, merge_thres)

def test_merge_partitions_non_consecutive_labels():
    partition_index = np.array([
        [1, 1, 0, 3, 3],
        [1, 1, 0, 3, 3],
        [0, 0, 0, 0, 0],
        [5, 5, 0, 7, 7],
        [5, 5, 0, 7, 7]
    ])
    partition_num = 4
    K = np.ones((5, 5))
    PHI = np.ones((5, 5)) * np.pi / 4
    slope_spec = np.ones((5, 5))
    merge_thres = 0.5

    with pytest.raises(ValueError, match="The labels are not consecutive integers.*"):
        merge_partitions(partition_index, partition_num, K, PHI, slope_spec, merge_thres)

def test_merge_partitions_empty_input():
    partition_index = np.zeros((5, 5), dtype=int)
    partition_num = 0
    K = np.ones((5, 5))
    PHI = np.ones((5, 5)) * np.pi / 4
    slope_spec = np.ones((5, 5))
    merge_thres = 0.5

    merged_partition_index = merge_partitions(partition_index, partition_num, K, PHI, slope_spec, merge_thres)
    assert np.all(merged_partition_index == 0)

def test_merge_partitions_merge_behavior():
    partition_index = np.array([
        [1, 1, 0, 2, 2],
        [1, 1, 0, 2, 2],
        [0, 0, 0, 0, 0],
        [3, 3, 0, 4, 4],
        [3, 3, 0, 4, 4]
    ])
    partition_num = 4
    K = np.ones((5, 5))
    PHI = np.ones((5, 5)) * np.pi / 4
    slope_spec = np.ones((5, 5))
    merge_thres = 10.0  # High threshold to force merging

    merged_partition_index = merge_partitions(partition_index, partition_num, K, PHI, slope_spec, merge_thres)
    assert np.unique(merged_partition_index).size <= partition_num