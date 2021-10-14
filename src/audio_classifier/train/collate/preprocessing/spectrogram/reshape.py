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
        copy (bool, optional): Whether or not to copy the incoming spectrogram. Defaults to False.

    Returns:
        ret_data (Sequence[Tuple[str, Sequence[np.ndarray], np.ndarray, np.ndarray, int]]): (batch_size, ) The transformed dataset with each data point being a tuple of (filename, spec_flat_slices, sample_freq, sample_time, label).
    """
    ret_data: Deque[Tuple[str, Sequence[np.ndarray], np.ndarray, np.ndarray,
                          int]] = deque()
    for filename, spectrogram, sample_freq, sample_time, label in data:
        slices: Sequence[np.ndarray] = reshape.slice_spectrogram(
            spectrogram=spectrogram,
            slice_size=config.slice_size,
            stride_size=config.stride_size,
            copy=copy)
        spec_flat_slices: List[np.ndarray] = [
            reshape.flatten_slice(slice=slice, copy=copy) for slice in slices
        ]
        ret_data.append(
            (filename, spec_flat_slices, sample_freq, sample_time, label))
    return ret_data


def slice_flatten_freq_range_colalte(
    data: Sequence[Tuple[str, Sequence[np.ndarray], np.ndarray, np.ndarray,
                         int]],
    config: conf_reshape.ReshapeConfig,
    copy: bool = False
) -> Sequence[Tuple[str, Sequence[Sequence[np.ndarray]], np.ndarray,
                    np.ndarray, int]]:
    """Slice and flatten spectrograms of all the sub freq-range spectrograms.

    Args:
        data (Sequence[Tuple[str, Sequence[np.ndarray], np.ndarray, np.ndarray, int]]): (batch_size, ) The data from upstream spectrogram transformation function.
        config (conf_reshape.ReshapeConfig): The configuration used to slice spectrogram.
        copy (bool, optional): Whether or not to copy the incoming spectrogram. Defaults to False.

    Returns:
        Sequence[Tuple[str, Sequence[Sequence[np.ndarray]], np.ndarray, np.ndarray, int]]: (batch_size, ) The transformed dataset with each data point being a tuple of (filename, split_specs_flat_slices, sample_freq, sample_time, label). split_specs_flat_slices has shape of (n_split_specs, n_slices, n_freq_bins*slcie_size).
    """
    ret_data: Deque[Tuple[str, Sequence[Sequence[np.ndarray]], np.ndarray,
                          np.ndarray, int]] = deque()
    for filename, specs, sample_freq, sample_time, label in data:
        specs_flat_slices: List[List[np.ndarray]] = list()
        for spec in specs:
            slices: Sequence[np.ndarray] = reshape.slice_spectrogram(
                spectrogram=spec,
                slice_size=config.slice_size,
                stride_size=config.stride_size,
                copy=copy)
            spec_flat_slices: List[np.ndarray] = [
                reshape.flatten_slice(slice=slice, copy=copy)
                for slice in slices
            ]
            specs_flat_slices.append(spec_flat_slices)
        ret_data.append(
            (filename, specs_flat_slices, sample_freq, sample_time, label))
    return ret_data
