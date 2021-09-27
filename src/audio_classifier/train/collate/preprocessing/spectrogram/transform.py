from typing import List, Tuple

import numpy as np

from .....common.preprocessing.spectrogram import transform
from .....config.preprocessing import spec_config


def stft_spectrogram_collate(data: List[Tuple[str, np.ndarray, int]],
                                       config: spec_config.STFTSpecConfig):
    """Transfrom a batch of data from time domain signal to frequency domain signal

    Args:
        data (List[Tuple[str, np.ndarray, int]]): (n_batch, ) The data from upstream sound wave dataset loader.
        config (spec_config.MelSpecConfig): The configuration used to generate stft-spectrogram.

    Returns:
        ret_data (List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, int]]): (n_batch, ) The transformed dataset with each data point being a tuple of (filename, stft_spec, stft_freq, stft_time, label).
    """
    ret_data: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray,
                         int]] = list()
    for filename, sound_wave, label in data:
        stft_spec, stft_freq, stft_time = transform.transform_stft_spectrogram(
            sound_wave=sound_wave,
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            window_size=config.window_size,
            hop_size=config.hop_size,
            apply_log=config.apply_log)
        ret_data.append((filename, stft_spec, stft_freq, stft_time, label))
    return ret_data


def mel_spectrogram_collate(data: List[Tuple[str, np.ndarray, int]],
                                      config: spec_config.MelSpecConfig):
    """[summary]

    Args:
        data (List[Tuple[str, np.ndarray, int]]): (n_batch, ) The data from upstream sound wave dataset loader.
        config (spec_config.MelSpecConfig): The configuration used to generate mel-spectrogram.

    Returns:
        [type]: [description]
    """
    ret_data: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray,
                         int]] = list()
    freq_max: float = config.freq_max if config.freq_max > 0.0 else config.sample_rate / 2.0
    for filename, sound_wave, label in data:
        mel_spec, mel_freq, mel_time = transform.transform_mel_spectrogram(
            sound_wave=sound_wave,
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            n_mels=config.n_mels,
            freq_min=config.freq_min,
            freq_max=freq_max,
            window_size=config.window_size,
            hop_size=config.hop_size,
            apply_log=config.apply_log)
        ret_data.append((filename, mel_spec, mel_freq, mel_time, label))
    return ret_data
