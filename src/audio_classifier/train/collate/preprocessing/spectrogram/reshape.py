from typing import List, Tuple

import numpy as np

from .....common.preprocessing.spectrogram import reshape
from audio_classifier.config.preprocessing import reshape_config


def slice_flatten_collate(
    data: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, int]],
    config: reshape_config.ReshapeConfig,
    copy: bool = False
) -> List[Tuple[str, List[np.ndarray], np.ndarray, np.ndarray, int]]:
    """For a batch of data, slice and flatten each spectrograms into a list of vectors.

    Args:
        data (List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, int]]): (n_batch, ) The data from upstream spectrogram transformation function.
        config (reshape_config.ReshapeConfig): The configuration used to slice spectrogram.

    Returns:
        ret_data (List[Tuple[str, List[np.ndarray], np.ndarray, np.ndarray, int]]): (n_batch, ) The transformed dataset with each data point being a tuple of (filename, flat_slices, sample_freq, sample_time, label).
    """
    ret_data: List[Tuple[str, List[np.ndarray], np.ndarray, np.ndarray,
                         int]] = list()
    for filename, spectrogram, sample_freq, sample_time, label in data:
        slices: List[np.ndarray] = reshape.slice_spectrogram(
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
