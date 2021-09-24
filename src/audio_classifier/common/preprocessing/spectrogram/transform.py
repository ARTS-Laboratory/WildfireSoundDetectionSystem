from typing import Tuple

import librosa.core as rosa_core
import librosa.feature as rosa_feature
import numpy as np


def transform_stft_spectrogram(
        sound_wave: np.ndarray, sample_rate: int, n_fft: int, window_size: int,
        hop_size: int,
        apply_log: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform a sound wave to stft-spectrogram.

    Args:
        sound_wave (np.ndarray): (sample_rate*n_secs) The sound wave to be transformed.
        sample_rate (int): Sample rate of the `sound_wave`.
        n_fft (int): Number of FFT components.
        window_size (int): Each frame of audio is windowed by window of length `window_size` and then padded with zeros to match `n_fft`.
        hop_size (int): Number of audio samples between adjacent STFT columns.
        apply_log (bool): Wheather or not to apply log10 to the `stft_spec` before return.

    Returns:
        stft_spec (np.ndarray): (1 + n_fft/2, n_frames) The stft of the `sample_rate`.
        stft_freq (np.ndarray): (1 + n_fft/2, ) Frequencies corresponding to each bin in `stft_spec`.
        stft_time (np.ndarray): (n_frames, ) Time stamps (in seconds) corresponding to each frame of `stft_spec`.
    """
    stft_spec: np.ndarray = rosa_core.stft(y=sound_wave,
                                           n_fft=n_fft,
                                           win_length=window_size,
                                           hop_length=hop_size,
                                           center=False)
    stft_spec = np.abs(stft_spec)
    stft_freq: np.ndarray = rosa_core.fft_frequencies(sr=sample_rate,
                                                      n_fft=n_fft)
    stft_time: np.ndarray = rosa_core.times_like(X=stft_spec,
                                                 sr=sample_rate,
                                                 hop_length=hop_size,
                                                 n_fft=n_fft)
    if apply_log is True:
        stft_spec = np.log10(stft_spec)
    return stft_spec, stft_freq, stft_time


def transform_mel_spectrogram(
        sound_wave: np.ndarray, sample_rate: int, n_fft: int, n_mels: int,
        freq_min: float, freq_max: float, window_size: int, hop_size: int,
        apply_log: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform a sound wave to mel-spectrogram.

    Args:
        sound_wave (np.ndarray): (sample_rate*n_secs) The sound wave to be transformed.
        sample_rate (int): Sample rate of the `sound_wave`.
        n_fft (int): Number of FFT components.
        n_mels (int): Number of Mel bands to generate.
        freq_min (float): Lowest frequency (in Hz)
        freq_max (float): Highest frequency (in Hz).
        window_size (int): Each frame of audio is windowed by window of length `window_size` and then padded with zeros to match `n_fft`.
        hop_size (int): Number of audio samples between adjacent STFT columns.
        apply_log (bool): Wheather or not to apply log10 to the `mel_spec` before return.

    Returns:
        mel_spec (np.ndarray): (n_mels, n_frames) The mel-spectrogram of the `sample_rate`.
        mel_freq (np.ndarray): (n_mels, ) Frequencies corresponding to each bin in `mel_spec`.
        mel_time (np.ndarray): (n_frames, ) Time stamps (in seconds) corresponding to each frame of `mel_spec`.
    """
    stft_spec: np.ndarray = rosa_core.stft(y=sound_wave,
                                           n_fft=n_fft,
                                           win_length=window_size,
                                           hop_length=hop_size,
                                           center=False)
    stft_spec = np.abs(stft_spec)
    mel_spec: np.ndarray = rosa_feature.melspectrogram(S=stft_spec**2,
                                                       sr=sample_rate,
                                                       n_fft=n_fft,
                                                       hop_length=hop_size,
                                                       win_length=window_size,
                                                       fmin=freq_min,
                                                       fmax=freq_max)
    mel_freq: np.ndarray = rosa_core.mel_frequencies(
        n_mels=n_mels,
        fmin=freq_min,
        fmax=freq_max,
    )
    mel_time: np.ndarray = rosa_core.times_like(X=mel_spec,
                                                sr=sample_rate,
                                                hop_length=hop_size,
                                                n_fft=n_fft)
    if apply_log is True:
        mel_spec = np.log10(mel_spec)
    return mel_spec, mel_freq, mel_time
