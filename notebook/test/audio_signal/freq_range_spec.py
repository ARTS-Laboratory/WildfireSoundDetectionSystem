#%%
import os
from typing import Sequence

import audio_classifier.common.feature_engineering.pool as fe_pool
import audio_classifier.common.feature_engineering.skm.proj as fe_skm_proj
import audio_classifier.common.preprocessing.spectrogram.freq_range as spec_freq_range
import audio_classifier.common.preprocessing.spectrogram.reshape as spec_reshape
import audio_classifier.common.preprocessing.spectrogram.transform as spec_transform
import audio_classifier.config.feature_engineering.pool as conf_pool
import audio_classifier.config.preprocessing.reshape as conf_reshape
import audio_classifier.config.preprocessing.spec as conf_spec
import librosa.core as rosa_core
import matplotlib.pyplot as plt
import numpy as np
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
split_freq: np.ndarray = np.array([0, 5000, 18000])
split_spec: Sequence[
    np.ndarray] = spec_freq_range.split_freq_range_spectraogram(
        spectrogram=mel_spec, spec_freq=mel_freq, split_freq=split_freq)

#%%
for spec in split_spec:
    plt.pcolormesh(
        mel_time,
        mel_freq[0:spec.shape[0]],
        #    10 * np.log10(filter_spec),
        spec,
        shading="auto",
        cmap="magma")
    # plt.gca().set_aspect(1e-2)
    plt.show()

# %%
