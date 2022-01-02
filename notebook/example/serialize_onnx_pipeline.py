#%%
import os
import pickle
from typing import List, Sequence, Union

import audio_classifier.common.feature_engineering.pool as fe_pool
import audio_classifier.common.feature_engineering.skm_proj as fe_skm_proj
import audio_classifier.common.preprocessing.spectrogram.reshape as spec_reshape
import audio_classifier.common.preprocessing.spectrogram.transform as spec_transform
import audio_classifier.config.feature_engineering.pool as conf_pool
import audio_classifier.config.preprocessing.reshape as conf_reshape
import audio_classifier.config.preprocessing.spec as conf_spec
import librosa.core as rosa_core
import numpy as np
import skl2onnx
from script.fit_skm_pca_svc import FitSkmPcaSvcResult
from skl2onnx.helpers import onnx_helper
from sklearn.base import ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn_plugins.cluster.spherical_kmeans import SphericalKMeans

#%%
MODEL_DIR_ROOT_PATH: str = "../../model/binary_fire/pipeline_50_100_150_200/"
FILENAME: str = "50_100_150_aug_01_pca_00_svc_01.pkl"

#%%
AUDIO_PATH: str = "../test/test_audio/fire_test.wav"
CONFIG_ROOT_PATH: str = "../../config"
MODEL_ROOT_PATH: str = "../../model/binary_fire/spec_00_reshape_00_skm_00"
SPEC_CONFIG_PATH: str = os.path.join(CONFIG_ROOT_PATH,
                                     "preprocessing/spec/00.json")
RESHAPE_CONFIG_PATH: str = os.path.join(CONFIG_ROOT_PATH,
                                        "preprocessing/reshape/00.json")
POOL_CONFIG_PATH: str = os.path.join(CONFIG_ROOT_PATH,
                                     "feature_engineering/pool/02.json")

#%%
EXPORT_DIR_ROOT_PATH: str = "../../model/binary_fire/spec_00_reshape_00_skm_00/100_100/dtype_float32"
skm_export_dir_path: str = os.path.join(EXPORT_DIR_ROOT_PATH, "skm")
svc_export_dir_path: str = os.path.join(EXPORT_DIR_ROOT_PATH, "svm")
os.makedirs(skm_export_dir_path, exist_ok=True)
os.makedirs(svc_export_dir_path, exist_ok=True)

#%%
dtype = np.float32


#%%
def load_result(result_path: str):
    with open(result_path, mode="rb") as result_file:
        result: FitSkmPcaSvcResult = pickle.load(result_file)
        return result


#%%
spec_config: conf_spec.MelSpecConfig = conf_spec.get_spec_config_from_json(
    SPEC_CONFIG_PATH, conf_spec.MelSpecConfig)
reshape_config: conf_reshape.ReshapeConfig = conf_reshape.get_reshape_config_from_json(
    RESHAPE_CONFIG_PATH)
pool_config: conf_pool.PoolConfig = conf_pool.get_pool_config_from_json(
    POOL_CONFIG_PATH)
#%%
result_path: str = os.path.join(MODEL_DIR_ROOT_PATH, FILENAME)
result: FitSkmPcaSvcResult = load_result(result_path)

#%%
sound_wave, _ = rosa_core.load(path=AUDIO_PATH,
                               sr=spec_config.sample_rate,
                               mono=True)
all_skms: List[List[SphericalKMeans]] = result.skms[1]
classifiers: List[Union[Pipeline, ClassifierMixin]] = result.classifiers[1]
for curr_val, (skms, classifier) in enumerate(zip(all_skms, classifiers)):
    # sample input
    # hop size = window_size
    mel_spec, mel_freq, mel_time = spec_transform.transform_mel_spectrogram(
        sound_wave=sound_wave,
        sample_rate=spec_config.sample_rate,
        n_fft=spec_config.n_fft,
        n_mels=spec_config.n_mels,
        freq_min=spec_config.freq_min,
        freq_max=spec_config.freq_max,
        window_size=spec_config.window_size,
        hop_size=spec_config.window_size,
        apply_log=spec_config.apply_log)
    slices: Sequence[np.ndarray] = spec_reshape.slice_spectrogram(
        spectrogram=mel_spec,
        slice_size=reshape_config.slice_size,
        stride_size=reshape_config.stride_size)
    flat_slices: Sequence[np.ndarray] = list()
    for slice in slices:
        flat: np.ndarray = spec_reshape.flatten_slice(slice, copy=True)
        flat_slices.append(flat)
    proj_slices: np.ndarray = fe_skm_proj.proj_skl_skm(flat_slices, skms)
    pool_slices: Sequence[np.ndarray] = fe_pool.apply_pool_func(
        spec_projs=proj_slices,
        pool_func=fe_pool.MeanStdPool(),
        pool_size=pool_config.pool_size,
        stride_size=pool_config.stride_size)
    # serialize skms
    for curr_class, skm in enumerate(skms):
        curr_skm_export_dir_path: str = os.path.join(
            skm_export_dir_path, str.format("val_{:02d}", curr_val),
            str.format("class_{:02d}", curr_class))
        os.makedirs(curr_skm_export_dir_path, exist_ok=True)
        curr_skm_path: str = os.path.join(curr_skm_export_dir_path,
                                          "model.onnx")
        skm_onnx = skl2onnx.to_onnx(skm,
                                    X=np.asarray([flat_slices[0]],
                                                 dtype=dtype))
        onnx_helper.save_onnx_model(skm_onnx, filename=curr_skm_path)
    # serialze classifier
    curr_classifier_path: str = os.path.join(
        svc_export_dir_path, str.format("val_{:02d}.onnx", curr_val))
    classifier_onnx = skl2onnx.to_onnx(classifier,
                                       X=np.asarray([pool_slices[0]],
                                                    dtype=dtype))
    onnx_helper.save_onnx_model(classifier_onnx, filename=curr_classifier_path)

#%%
