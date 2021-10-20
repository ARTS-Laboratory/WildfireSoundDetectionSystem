import numpy as np
from scipy.stats import norm, rv_continuous


def add_white_noise(sound_wave: np.ndarray, snr: float) -> np.ndarray:
    """Add noise to a single passed in sound wave
    
    Implementation based on:
    https://medium.com/analytics-vidhya/adding-noise-to-audio-clips-5d8cee24ccb8

    Args:
        sound_wave (np.ndarray): (n_channels, sample_rate*duration) The sound wave to be processded
        snr (float): The signal to noise ratio.

    Returns:
        sound_wave_noise (np.ndarray): The sound wave with white noise
    """
    rms_sound_sq: float = np.mean(sound_wave**2)
    rms_noise: float = np.sqrt(rms_sound_sq / np.power(10, snr / 10))
    print(rms_noise)
    noise_rv: rv_continuous = norm(0, rms_noise)
    noise: np.ndarray = noise_rv.rvs(size=sound_wave.shape)
    sound_wave_noise: np.ndarray = sound_wave + noise
    return sound_wave_noise
