from typing import List
import numpy as np


def slice_spectrogram(spectrogram: np.ndarray,
                      slice_size: int,
                      stride_size: int,
                      copy: bool = False) -> List[np.ndarray]:
    """Slice a spectrogram.

    Args:
        spectrogram (np.ndarray): (n_sample_freq, n_sample_time) raw spectrogram to be slice.
        shingle_size (int): number of time bins for each sliced spectrogram
        stride_size (int): number of time bins to skip while slicing.

    Returns:
        slices (np.ndarray): (n_slices, n_sample_freq, shingle_size) sliced spectrogram for the given raw_spectrogram.
    """
    slices: List[np.ndarray] = list()
    for i in range(0, spectrogram.shape[1] - slice_size + 1, stride_size):
        curr_slice: np.ndarray = spectrogram[:, i:i + slice_size]
        if copy == True:
            slices.append(np.copy(curr_slice))
        else:
            slices.append(curr_slice)
    return slices


def flatten_slice(slice: np.ndarray, copy: bool = False) -> np.ndarray:
    """Convert a slice of spectrogram in to a row vector.

    Args:
        slice (np.ndarray): (n_sample_freq, shingle_size)

    Returns:
        flat_slice (np.ndarray): (n_sample_freq * shingle_size) `[slice[:, 0].T, slice[:, 1].T, ..., slice[:, shingle_size-1].T]`
    """
    n_sample_freq: int = slice.shape[0]
    shingle_size: int = slice.shape[1]
    flat_slice: np.ndarray = slice.reshape((n_sample_freq * shingle_size),
                                           order="F")
    if copy == True:
        return np.copy(flat_slice)
    return flat_slice
