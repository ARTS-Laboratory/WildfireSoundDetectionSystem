from typing import List
import numpy as np


def slice_spectrogram(spectrogram: np.ndarray,
                      slice_size: int,
                      stride_size: int,
                      copy: bool = False) -> List[np.ndarray]:
    """Slice a spectrogram.

    Args:
        spectrogram (np.ndarray): (n_sample_freq, n_sample_time) Raw spectrogram to be slice.
        slice_size (int): Number of time bins for each sliced spectrogram
        stride_size (int): Number of time bins to skip while slicing.
        copy (bool): if `True`, then the returned slices has no relation with the passed in spectrogram. Defaults to `False`.

    Returns:
        slices (np.ndarray): (n_slices, n_sample_freq, slice_size) Sliced spectrogram for the given raw_spectrogram.
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
        slice (np.ndarray): (n_sample_freq, slice_size) The spectrogram slice to be flatten.
        copy (bool): If `True`, then the returned `flat_slice` has no relation with the passed in `slice`. Defaults to `False`.

    Returns:
        flat_slice (np.ndarray): (n_sample_freq * slice_size) A flattened spectrogram slice in the order of `[slice[:, 0].T, slice[:, 1].T, ..., slice[:, slice_size-1].T]`
    """
    n_sample_freq: int = slice.shape[0]
    slice_size: int = slice.shape[1]
    flat_slice: np.ndarray = slice.reshape((n_sample_freq * slice_size),
                                           order="F")
    if copy == True:
        return np.copy(flat_slice)
    return flat_slice


def unflatten_slice(flat_slice: np.ndarray,
                    slice_size: int,
                    copy: bool = False):
    """Unflatten a slice back to original spectrogram shape.

    Args:
        flat_slice (np.ndarray): (n_sample_freq*slice_size) A flattened spectrogram slice in the order of `[*(slice[:, 0].T), *(slice[:, 1].T), ..., *(slice[:, slice_size-1].T)]` to be restored.
        slice_size (int): Number of time bins for each sliced spectrogram.
        copy (bool, optional): If `True`, then the returned `spec_slice` has no relaition with the passed in `flat_slice`. Defaults to False.

    Returns:
        spec_slice (np.ndarray): (n_sample_freq, slice_size) The restored sliced spectrogram.
    """
    n_sample_freq: int = flat_slice.shape[0] // slice_size
    spec_slice: np.ndarray = flat_slice.reshape((n_sample_freq, slice_size),
                                                order="F")
    if copy == True:
        return np.copy(spec_slice)
    return spec_slice