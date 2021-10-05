from collections import deque
from typing import Callable, Deque, Sequence, Tuple

import numpy as np
from sklearn_plugins.cluster.spherical_kmeans import SphericalKMeans

from ....common.feature_engineering import pool
from ....common.feature_engineering.skm import proj
from ....config.feature_engineering.pool import PoolConfig


def skm_skl_proj_collate(
    data: Sequence[Tuple[str, Sequence[np.ndarray], np.ndarray, np.ndarray,
                         int]],
    skm: SphericalKMeans,
    copy: bool = False
) -> Sequence[Tuple[str, np.ndarray, np.ndarray, np.ndarray, int]]:
    """For a batch of data, slice and flatten each spectrograms into a list of vectors.

    Args:
        data (Sequence[Tuple[str, Sequence[np.ndarray], np.ndarray, np.ndarray, int]]): (batch_size, ) The data from upstream spectrogram transformation function.
        config (reshape_config.ReshapeConfig): The configuration used to slice spectrogram.

    Returns:
        ret_data (Sequence[Tuple[str, np.ndarray, np.ndarray, np.ndarray, int]]): (batch_size, ) The transformed dataset with each data point being a tuple of (filename, spec_projs, sample_freq, sample_time, label). spec_proj has size (n_slices, n_centroids).
    """
    ret_data: Deque[Tuple[str, np.ndarray, np.ndarray, np.ndarray,
                          int]] = deque()
    for filename, spec_flat_slices, sample_freq, sample_time, label in data:
        spec_projs: np.ndarray = proj.proj_skl_skm(
            spec_flat_slices=spec_flat_slices, skm=skm, copy=copy)
        ret_data.append(
            (filename, spec_projs, sample_freq, sample_time, label))
    return ret_data


def skm_projs_pool_collate(
    data: Sequence[Tuple[str, np.ndarray, np.ndarray, np.ndarray, int]],
    pool_func: Callable[[np.ndarray], np.ndarray], pool_config: PoolConfig
) -> Sequence[Tuple[str, Sequence[np.ndarray], np.ndarray, np.ndarray, int]]:
    """For a batch of data, apply pooling function to each of the projected audio slices.

    Args:
        data (Sequence[Tuple[str, np.ndarray, np.ndarray, np.ndarray, int]]): (batch_size, ) The data from upstream sklearn projection function.
        pool_func (Callable[[np.ndarray], np.ndarray]): The pooling function to be applied to the incoming audio projections.

    Returns:
        ret_data (Sequence[Tuple[str, Sequence[np.ndarray], np.ndarray, np.ndarray, int]]): (batch_size, ) The transformed dataset with each data point being a tuple of (filename, pool_projs, sample_freq, sample_time, lable). pool_projs has size (n_slices_pr, n_output_features).
    """
    ret_data: Deque[Tuple[str, Sequence[np.ndarray], np.ndarray, np.ndarray,
                          int]] = deque()
    for filename, spec_projs, sample_freq, sample_time, label in data:
        pool_projs: Sequence[np.ndarray] = pool.apply_pool_func(
            spec_projs=spec_projs,
            pool_func=pool_func,
            pool_size=pool_config.pool_size,
            stride_size=pool_config.stride_size)
        ret_data.append(
            (filename, pool_projs, sample_freq, sample_time, label))
    return ret_data
