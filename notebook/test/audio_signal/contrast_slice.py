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
import numpy as np
import script.train.skl_loader.skm as skl_skm_laoder
from scipy.stats import norm, rv_continuous
from sklearn.svm import SVC
from sklearn_plugins.cluster.spherical_kmeans import SphericalKMeans

#%%
AUDIO_PATH: str = "../../test/test_dataset/folder_dataset/0_0.wav"
# AUDIO_PATH: str = "../../test/test_audio/fire_01.wav"
CONFIG_ROOT_PATH: str = "../../../config"
MODEL_ROOT_PATH: str = "../../../model/spec_00_reshape_00_skm_00"
SPEC_CONFIG_PATH: str = os.path.join(CONFIG_ROOT_PATH,
                                     "preprocessing/spec/00.json")
RESHAPE_CONFIG_PATH: str = os.path.join(CONFIG_ROOT_PATH,
                                        "preprocessing/reshape/00.json")
POOL_CONFIG_PATH: str = os.path.join(CONFIG_ROOT_PATH,
                                     "feature_engineering/pool/03.json")

#%%
CURR_VAL_FOLD: int = 0
VAL_PATH_STUB: str = "val_{:02d}"
CLASS_PATH_STUB: str = "class_{:02d}"
SKM_MODEL_ROOT_PATH: str = os.path.join(MODEL_ROOT_PATH, "skm")
SVM_MODEL_ROOT_PATH: str = os.path.join(MODEL_ROOT_PATH, "svm/pool_03")

#%%
SNR: float = 1.0

#%%
spec_config: conf_spec.MelSpecConfig = conf_spec.get_spec_config_from_json(
    SPEC_CONFIG_PATH, conf_spec.MelSpecConfig)
reshape_config: conf_reshape.ReshapeConfig = conf_reshape.get_reshape_config_from_json(
    RESHAPE_CONFIG_PATH)
pool_config: conf_pool.PoolConfig = conf_pool.get_pool_config_from_json(
    POOL_CONFIG_PATH)

#%%
sound_wave, _ = rosa_core.load(path=AUDIO_PATH,
                               sr=spec_config.sample_rate,
                               mono=True)
# sound_wave = add_noise(sound_wave=sound_wave, snr=SNR)
# sound_wave = chunk_sound_wave(sound_wave=sound_wave,
#                               sample_rate=spec_config.sample_rate,
#                               n_secs=4.0)

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
    # apply_log=False)
    apply_log=True)

#%%
slices: Sequence[np.ndarray] = spec_reshape.slice_spectrogram(
    spectrogram=mel_spec,
    slice_size=reshape_config.slice_size,
    stride_size=reshape_config.stride_size)
flat_slices: Sequence[np.ndarray] = list()
for slice in slices:
    flat: np.ndarray = spec_reshape.flatten_slice(slice, copy=True)
    flat_slices.append(flat)
    slice_spec: np.ndarray = spec_reshape.unflatten_slice(
        flat, slice_size=reshape_config.slice_size, copy=True)
    plt.pcolormesh(
        mel_time[0:slice_spec.shape[1]],
        mel_freq,
        #    10 * np.log10(slice_spec),
        slice_spec,
        shading="auto",
        cmap="magma")
    plt.gca().set_aspect(1e-5)
    plt.show()

#%%
filter_slices: Sequence[np.ndarray] = list()
for slice in flat_slices:
    slice_mean: float = np.percentile(slice, 25)
    filter_slice: np.ndarray = np.where(slice > slice_mean, slice, 1.5 * slice)
    filter_slices.append(filter_slice)
    filter_spec: np.ndarray = spec_reshape.unflatten_slice(
        filter_slice, slice_size=reshape_config.slice_size, copy=True)
    plt.pcolormesh(
        mel_time[0:filter_spec.shape[1]],
        mel_freq,
        #    10 * np.log10(filter_spec),
        filter_spec,
        shading="auto",
        cmap="magma")
    plt.gca().set_aspect(1e-5)
    plt.show()

# %%
