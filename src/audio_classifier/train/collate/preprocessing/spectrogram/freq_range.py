from collections import deque
from typing import Deque, Tuple, Sequence

import numpy as np

from .....common.preprocessing.spectrogram import freq_range
from .....config.preprocessing import freq_range as conf_freq_range


def split_freq_range_spec_collate(
    data: Sequence[Tuple[str, np.ndarray, np.ndarray, np.ndarray,
                         int]], config: conf_freq_range.FreqRangeConfig
) -> Sequence[Tuple[str, Sequence[np.ndarray], np.ndarray, np.ndarray, int]]:
    """Split a spectrogram based on specified frequency range.

    Args:
        data (Sequence[Tuple[str, np.ndarray, np.ndarray, np.ndarray, int]]): A tuple of (filename, spectrograms, spec_freq, spec_time, label) where spectrogram has shape of (n_freq_bins, n_time_stamps)
        config (conf_freq_range.FreqRangeConfig): The config that used to specify splitting frequencies.

    Returns:
        Sequence[Tuple[str, Sequence[np.ndarray], np.ndarray, np.ndarray, int]]: A tuple of (filename, split_specs, spec_freq, spec_time, label) where split_specs has length of (n_splitted_spec ,) and each element has shape (n_freq_bins_pr, n_time_steps).
    """
    ret_data: Deque[Tuple[str, Sequence[np.ndarray], np.ndarray, np.ndarray,
                          int]] = deque()
    for filename, spec, spec_freq, spec_time, label in data:
        split_specs: Sequence[
            np.ndarray] = freq_range.split_freq_range_spectraogram(
                spectrogram=spec,
                spec_freq=spec_freq,
                split_freq=np.asarray(config.split_freq))
        ret_data.append((filename, split_specs, spec_freq, spec_time, label))
    return ret_data
