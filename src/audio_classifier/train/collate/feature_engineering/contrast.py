from collections import deque
from typing import Callable, Deque, List, Sequence, Tuple

import numpy as np

from ....common.feature_engineering import contrast
from ....config.feature_engineering.contrast import ContrastConfig


def contrast_collate(
    data: Sequence[Tuple[str, Sequence[np.ndarray], np.ndarray, np.ndarray,
                         int]], threshold_func: Callable[[np.ndarray], float],
    contrast_config: ContrastConfig
) -> Sequence[Tuple[str, Sequence[np.ndarray], np.ndarray, np.ndarray, int]]:
    """For a batch of data, increase contrast to each of the flatten spectrogram slices.

    Args:
        data (Sequence[Tuple[str, np.ndarray, np.ndarray, np.ndarray, int]]): (batch_size, ) The data from upstream slice flatten function.
        threshold_func (Callable[[np.ndarray], float]): The function to calculate the threshold.
        contrast_config (ContrastConfig): The contrast config that specifies the scaling factor.

    Returns:
        Sequence[Tuple[str, Sequence[np.ndarray], np.ndarray, np.ndarray, int]]: (batch_size, ) The transformed dataset with each data point being a tuple of (filename, pool_projs, sample_freq, sample_time, lable). pool_projs has size (n_slices_pr, n_output_features).
    """
    ret_data: Deque[Tuple[str, Sequence[np.ndarray], np.ndarray, np.ndarray,
                          int]] = deque()
    for filename, spec_flat_slices, sample_freq, sample_time, label in data:
        contrast_slices: List[np.ndarray] = [
            contrast.apply_contrast_func(flat_slice=flat_slice,
                                         threshold_func=threshold_func,
                                         alpha=contrast_config.alpha,
                                         beta=contrast_config.beta)
            for flat_slice in spec_flat_slices
        ]
        ret_data.append(
            (filename, contrast_slices, sample_freq, sample_time, label))
    return ret_data
