from typing import List, Sequence
import numpy as np


def split_freq_range_spectraogram(
        spectrogram: np.ndarray, spec_freq: np.ndarray,
        split_freq: np.ndarray) -> Sequence[np.ndarray]:
    """Split the spectrogram into different frequency range.

    Args:
        spectrogram (np.ndarray): (n_freq_bins, n_frames) The spectrogram to be splitted.
        spec_freq (np.ndarray): (n_freq_bins, ) The frequency that correspond to the freq bins in spectrogram.
        split_freq (np.ndarray): The frequency range to be splitted.

    Returns:
        split_spectrograms: (n_ranges, ) The sub-spectrogram specified by split_freq with eqch as shape = (n_freq_bins, n_frames)
    """
    split_index: np.ndarray = _compute_split_index(spec_freq=spec_freq,
                                                   split_freq=split_freq)
    split_spectrograms: List[np.ndarray] = np.split(
        spectrogram, indices_or_sections=split_index, axis=0)
    # remove spectrogram without that has zero frequency bins.
    split_spectrograms = list(
        filter(lambda spec: spec.shape[0] != 0, split_spectrograms))
    return split_spectrograms


def _compute_split_index(spec_freq: np.ndarray,
                         split_freq: np.ndarray) -> np.ndarray:
    """Compute the indices used to split the array.

    Args:
        spec_freq (np.ndarray): The frequency bins of the spectrogram.
        split_freq (np.ndarray): The frequency to be splitted.

    Returns:
        split_indices (np.ndarray): The indices used to split the spectrogram.
    """
    split_freq = np.expand_dims(split_freq, axis=1)
    freq_diff: np.ndarray = np.abs(spec_freq - split_freq)
    split_indices: np.ndarray = np.argmin(freq_diff, axis=1)
    split_indices = split_indices.reshape(-1)
    if split_indices[0] == 0:
        split_indices = np.delete(split_indices, 0)
    return split_indices
