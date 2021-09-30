from collections import deque
from typing import Deque, List, Sequence, Tuple

import numpy as np

from .....common.preprocessing.spectrogram import reshape
from .....config.preprocessing import reshape as conf_reshape


def slice_flatten_collate(
    data: Sequence[Tuple[str, np.ndarray, np.ndarray, np.ndarray, int]],
    config: conf_reshape.ReshapeConfig,
    copy: bool = False
) -> Sequence[Tuple[str, Sequence[np.ndarray], np.ndarray, np.ndarray, int]]:
    """For a batch of data, slice and flatten each spectrograms into a list of vectors.

    Args:
        data (Sequence[Tuple[str, np.ndarray, np.ndarray, np.ndarray, int]]): (batch_size, ) The data from upstream spectrogram transformation function.
        config (reshape_config.ReshapeConfig): The configuration used to slice spectrogram.

    Returns:
        ret_data (Sequence[Tuple[str, Sequence[np.ndarray], np.ndarray, np.ndarray, int]]): (batch_size, ) The transformed dataset with each data point being a tuple of (filename, flat_slices, sample_freq, sample_time, label).
    """
    ret_data: Deque[Tuple[str, Sequence[np.ndarray], np.ndarray, np.ndarray,
                          int]] = deque()
    for filename, spectrogram, sample_freq, sample_time, label in data:
        slices: Sequence[np.ndarray] = reshape.slice_spectrogram(
            spectrogram=spectrogram,
            slice_size=config.slice_size,
            stride_size=config.stride_size,
            copy=copy)
        flat_slices: List[np.ndarray] = [
            reshape.flatten_slice(slice=slice, copy=copy) for slice in slices
        ]
        ret_data.append(
            (filename, flat_slices, sample_freq, sample_time, label))
    return ret_data
