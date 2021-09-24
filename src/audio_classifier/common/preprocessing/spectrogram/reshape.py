import numpy as np


def slice_spectrogram(spectrogram: np.ndarray, shingle_size: int,
                      stride_size: int) -> np.ndarray:
    """Slice a spectrogram.

    Args:
        spectrogram (np.ndarray): (n_sample_freq, n_sample_time) raw spectrogram to be slice.
        shingle_size (int): number of time bins for each sliced spectrogram
        stride_size (int): number of time bins to skip while slicing.

    Returns:
        slices (np.ndarray): (n_slices, n_sample_freq, shingle_size) sliced spectrogram for the given raw_spectrogram.
    """
    n_sample_freq: int = spectrogram.shape[1]
    slices: np.ndarray = np.array([n_sample_freq, 0], dtype=float)
    for i in range(0, spectrogram.shape[1] - shingle_size + 1, stride_size):
        curr_slice: np.ndarray = spectrogram[:, i:i + shingle_size]
        slices = np.append(slices, curr_slice, axis=1)
    return slices


def flatten_slice(slice: np.ndarray) -> np.ndarray:
    """Convert a slice of spectrogram in to a row vector.

    Args:
        slice (np.ndarray): (n_sample_freq, shingle_size)

    Returns:
        flat_slice (np.ndarray): (n_sample_freq * shingle_size)
    """
    flat_slice: np.ndarray = slice.reshape(order="F")
    return flat_slice
