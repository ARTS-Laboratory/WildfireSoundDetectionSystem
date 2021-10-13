from typing import Callable

import numpy as np


def apply_contrast_func(flat_slice: np.ndarray,
                        threshold_func: Callable[[np.ndarray], float],
                        alpha: float, beta: float) -> np.ndarray:
    """Increase contrast of a flat slice.

    Args:
        flat_slice (np.ndarray): (n_mels * slice_size, ) A flat slice from a spectrogram.
        threshold_func (Callable[[np.ndarray], np.ndarray]): The function that calculates the threshold.
        alpha (float): The value to be multiplied if the value < threshold.
        beta (float): The value to be multiplied if the vlaue is >= threshold.

    Returns:
        contrast_flat_slice (np.ndarray): (n_mels * slice_size, )
    """
    threshold: float = threshold_func(flat_slice)
    contrast_flat_slice: np.ndarray = np.where(flat_slice < threshold,
                                               alpha * flat_slice,
                                               beta * flat_slice)
    return contrast_flat_slice