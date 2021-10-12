#%%
import os
import pickle
from typing import Sequence

import audio_classifier.common.feature_engineering.pool as fe_pool
import audio_classifier.common.feature_engineering.skm.proj as fe_skm_proj
import audio_classifier.common.preprocessing.spectrogram.reshape as spec_reshape
import audio_classifier.common.preprocessing.spectrogram.transform as spec_transform
import audio_classifier.config.feature_engineering.pool as conf_pool
import audio_classifier.config.preprocessing.reshape as conf_reshape
import audio_classifier.config.preprocessing.spec as conf_spec
import librosa.core as rosa_core
import matplotlib.pyplot as plt
import scipy.io.wavfile as scipy_wav
import numpy as np
import script.train.skl_loader.skm as skl_skm_laoder
from scipy.stats import norm, rv_continuous
from sklearn.svm import SVC
from sklearn_plugins.cluster.spherical_kmeans import SphericalKMeans
from IPython.display import Audio as AudioPlayer
#%%
# AUDIO_PATH: str = "../../test/test_dataset/folder_dataset/5_0.wav"
AUDIO_PATH: str = "../../test/test_audio/fire_01.wav"
CONFIG_ROOT_PATH: str = "../../../config"
MODEL_ROOT_PATH: str = "../../../model/spec_00_reshape_00_skm_00"
SPEC_CONFIG_PATH: str = os.path.join(CONFIG_ROOT_PATH,
                                     "preprocessing/spec/00.json")
RESHAPE_CONFIG_PATH: str = os.path.join(CONFIG_ROOT_PATH,
                                        "preprocessing/reshape/00.json")
POOL_CONFIG_PATH: str = os.path.join(CONFIG_ROOT_PATH,
                                     "feature_engineering/pool/00.json")

#%%
CURR_VAL_FOLD: int = 0
VAL_PATH_STUB: str = "val_{:02d}"
CLASS_PATH_STUB: str = "class_{:02d}"
SKM_MODEL_ROOT_PATH: str = os.path.join(MODEL_ROOT_PATH, "skm")
SVM_MODEL_ROOT_PATH: str = os.path.join(MODEL_ROOT_PATH, "svm")

#%%
SNR: float = 30

#%%
spec_config: conf_spec.MelSpecConfig = conf_spec.get_spec_config_from_json(
    SPEC_CONFIG_PATH, conf_spec.MelSpecConfig)
reshape_config: conf_reshape.ReshapeConfig = conf_reshape.get_reshape_config_from_json(
    RESHAPE_CONFIG_PATH)
pool_config: conf_pool.PoolConfig = conf_pool.get_pool_config_from_json(
    POOL_CONFIG_PATH)


#%%
def add_noise(sound_wave: np.ndarray, snr: float):
    #https://medium.com/analytics-vidhya/adding-noise-to-audio-clips-5d8cee24ccb8
    rms_sound_sq: float = np.mean(sound_wave**2)
    rms_noise: float = np.sqrt(rms_sound_sq / np.power(10, snr / 10))
    print(rms_noise)
    noise_rv: rv_continuous = norm(0, rms_noise)
    noise: np.ndarray = noise_rv.rvs(size=sound_wave.shape)
    sound_wave_noise: np.ndarray = sound_wave + noise
    return sound_wave_noise


def chunk_sound_wave(sound_wave: np.ndarray, sample_rate: int,
                     n_secs: float) -> np.ndarray:
    n_dp: int = int(sample_rate * n_secs)
    ret_sound_wave: np.ndarray = sound_wave[0:n_dp]
    return ret_sound_wave


#%%
sound_wave, _ = rosa_core.load(path=AUDIO_PATH,
                               sr=spec_config.sample_rate,
                               mono=True)
sound_wave = chunk_sound_wave(sound_wave=sound_wave,
                              sample_rate=spec_config.sample_rate,
                              n_secs=4.0)

#%%
# AudioPlayer(sound_wave, rate=spec_config.sample_rate, embed=True)
# RAW_FILE_PATH: str = os.path.splitext(AUDIO_PATH)[0] + "_original.wav"
# scipy_wav.write(RAW_FILE_PATH, rate=spec_config.sample_rate, data=sound_wave)

#%%
mel_spec, mel_freq, mel_time = spec_transform.transform_mel_spectrogram(
    sound_wave=sound_wave,
    sample_rate=spec_config.sample_rate,
    n_fft=spec_config.n_fft,
    n_mels=spec_config.n_mels,
    freq_min=spec_config.freq_min,
    freq_max=spec_config.freq_max,
    window_size=spec_config.window_size,
    hop_size=spec_config.hop_size,
    apply_log=spec_config.apply_log)
#%%
plt.pcolormesh(mel_time, mel_freq, mel_spec, shading="auto", cmap="magma")
# plt.gca().set_aspect(1e-5)
plt.show()

#%%
sound_wave = add_noise(sound_wave=sound_wave, snr=SNR)

#%%
# AudioPlayer(sound_wave, rate=spec_config.sample_rate, embed=True)
# NOISY_FILE_PATH: str = str.format(
#     os.path.splitext(AUDIO_PATH)[0] + "_noisy_{}.wav", SNR)
# scipy_wav.write(NOISY_FILE_PATH, rate=spec_config.sample_rate, data=sound_wave)

#%%
mel_spec, mel_freq, mel_time = spec_transform.transform_mel_spectrogram(
    sound_wave=sound_wave,
    sample_rate=spec_config.sample_rate,
    n_fft=spec_config.n_fft,
    n_mels=spec_config.n_mels,
    freq_min=spec_config.freq_min,
    freq_max=spec_config.freq_max,
    window_size=spec_config.window_size,
    hop_size=spec_config.hop_size,
    apply_log=spec_config.apply_log)
#%5
plt.pcolormesh(mel_time, mel_freq, mel_spec, shading="auto", cmap="magma")
# plt.gca().set_aspect(1e-5)
plt.show()
