#%%
import os
import pickle
from typing import Sequence

import audio_classifier.common.feature_engineering.pool as fe_pool
import audio_classifier.common.feature_engineering.skm_proj as fe_skm_proj
import audio_classifier.common.preprocessing.spectrogram.reshape as spec_reshape
import audio_classifier.common.preprocessing.spectrogram.transform as spec_transform
import audio_classifier.config.feature_engineering.pool as conf_pool
import audio_classifier.config.preprocessing.reshape as conf_reshape
import audio_classifier.config.preprocessing.spec as conf_spec
import librosa.core as rosa_core
import numpy as np
import script.train.skl_loader.skm as skl_skm_laoder
from sklearn.svm import SVC
from sklearn_plugins.cluster.spherical_kmeans import SphericalKMeans

#%%
# AUDIO_PATH: str = "../test/test_dataset/folder_dataset/0_0.wav"
AUDIO_PATH: str = "../test/test_audio/fire_01.wav"
CONFIG_ROOT_PATH: str = "../../config"
MODEL_ROOT_PATH: str = "../../model/binary_fire/spec_00_reshape_00_skm_00"
SPEC_CONFIG_PATH: str = os.path.join(CONFIG_ROOT_PATH,
                                     "preprocessing/spec/00.json")
RESHAPE_CONFIG_PATH: str = os.path.join(CONFIG_ROOT_PATH,
                                        "preprocessing/reshape/00.json")
POOL_CONFIG_PATH: str = os.path.join(CONFIG_ROOT_PATH,
                                     "feature_engineering/pool/02.json")

#%%
CURR_VAL_FOLD: int = 1
VAL_PATH_STUB: str = "val_{:02d}"
CLASS_PATH_STUB: str = "class_{:02d}"
SKM_MODEL_ROOT_PATH: str = os.path.join(MODEL_ROOT_PATH, "skm")
# SVM_MODEL_ROOT_PATH: str = os.path.join(MODEL_ROOT_PATH, "rfc", "pool_02_rfc_02")
SVM_MODEL_ROOT_PATH: str = os.path.join(MODEL_ROOT_PATH, "rfc", "pool_02_pca_00_rfc_02")
# SVM_MODEL_ROOT_PATH: str = os.path.join(MODEL_ROOT_PATH, "rfc", "augment_01_pool_02_pca_00_rfc_02")

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
slices: Sequence[np.ndarray] = spec_reshape.slice_spectrogram(
    spectrogram=mel_spec,
    slice_size=reshape_config.slice_size,
    stride_size=reshape_config.stride_size)
flat_slices: Sequence[np.ndarray] = list()
for slice in slices:
    flat: np.ndarray = spec_reshape.flatten_slice(slice, copy=True)
    flat_slices.append(flat)

#%%
CLASS_SKM_PATH_STUB: str = os.path.join(CLASS_PATH_STUB, "model.pkl")
CURR_VAL_SKM_PATH_STUB: str = skl_skm_laoder.get_curr_val_skm_path_stub(
    curr_val_fold=CURR_VAL_FOLD,
    skm_root_path=SKM_MODEL_ROOT_PATH,
    val_fold_path_stub=VAL_PATH_STUB,
    class_skm_path_stub=CLASS_SKM_PATH_STUB)
skms: Sequence[SphericalKMeans] = skl_skm_laoder.load_skl_skms(
    CURR_VAL_SKM_PATH_STUB, 2)
proj_slices: np.ndarray = fe_skm_proj.proj_skl_skm(flat_slices, skms)
pool_slices: Sequence[np.ndarray] = fe_pool.apply_pool_func(
    spec_projs=proj_slices,
    pool_func=fe_pool.MeanStdPool(),
    pool_size=pool_config.pool_size,
    stride_size=pool_config.stride_size)


#%%
def get_svc_path(curr_val_fold: int,
                 svm_model_root_path: str,
                 val_path_stub: str,
                 file_ext: str = ".pkl"):
    svc_path_stub: str = os.path.join(svm_model_root_path,
                                      val_path_stub + file_ext)
    svc_path = str.format(svc_path_stub, curr_val_fold)
    return svc_path


def get_svc(svc_path: str) -> SVC:
    with open(svc_path, "rb") as svc_file:
        svc: SVC = pickle.load(svc_file)
        return svc


#%%
CURR_VAL_SVC_PATH: str = get_svc_path(curr_val_fold=CURR_VAL_FOLD,
                                      svm_model_root_path=SVM_MODEL_ROOT_PATH,
                                      val_path_stub=VAL_PATH_STUB)
svc: SVC = get_svc(CURR_VAL_SVC_PATH)
#%%
pred = svc.predict(np.asarray(pool_slices))

#%%
np.bincount(pred, minlength=2)
