from collections import deque
from typing import Deque, Sequence, Tuple

import numpy as np
from sklearn_plugins.cluster.spherical_kmeans import SphericalKMeans

from ....common.feature_engineering import skm_proj


def skm_skl_proj_collate(
    data: Sequence[Tuple[str, Sequence[np.ndarray], np.ndarray, np.ndarray,
                         int]], skms: Sequence[SphericalKMeans]
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
        spec_projs: np.ndarray = skm_proj.proj_skl_skm(
            spec_flat_slices=spec_flat_slices, skms=skms)
        ret_data.append(
            (filename, spec_projs, sample_freq, sample_time, label))
    return ret_data
