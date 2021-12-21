#%%
import os
from abc import ABC, abstractmethod
from math import ceil
from typing import Callable, Iterable, Iterator, Optional, Sequence

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
from memory_profiler import profile
from onnxruntime import InferenceSession
from overrides import overrides


#%%
class AudioStream(Iterable[np.ndarray], ABC):
    sample_rate: float
    chunk_sec: float
    frame_length: int

    def __init__(self, sample_rate: float, chunk_sec: float) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.chunk_sec = chunk_sec
        self.frame_length = ceil(sample_rate * chunk_sec)

    @abstractmethod
    def __iter__(self) -> Iterator[np.ndarray]:
        pass


class WaveStream(AudioStream):
    src_sample_rate: float
    src_frame_length: int
    wav_path: str

    def __init__(self, wav_path: str, sample_rate: float,
                 chunk_sec: float) -> None:
        super().__init__(sample_rate=sample_rate, chunk_sec=chunk_sec)
        self.wav_path = wav_path
        self.src_sample_rate = rosa_core.get_samplerate(path=self.wav_path)
        self.src_frame_length = ceil(self.src_sample_rate * chunk_sec)

    @overrides
    def __iter__(self) -> Iterator[np.ndarray]:
        for curr_frame in rosa_core.stream(path=self.wav_path,
                                           block_length=1,
                                           frame_length=self.src_frame_length,
                                           hop_length=self.src_frame_length,
                                           mono=True):
            curr_frame = rosa_core.resample(curr_frame,
                                            self.src_sample_rate,
                                            self.sample_rate,
                                            fix=True)
            yield curr_frame


class AudioClassifier():
    skms: Sequence[InferenceSession]
    classifier: InferenceSession
    spec_config: conf_spec.MelSpecConfig
    reshape_config: conf_reshape.ReshapeConfig
    pool_config: conf_pool.PoolConfig

    def __init__(self,
                 skms: Sequence[InferenceSession],
                 classifier: InferenceSession,
                 spec_config: conf_spec.MelSpecConfig,
                 reshape_config: conf_reshape.ReshapeConfig,
                 pool_config: conf_pool.PoolConfig,
                 dtype=np.float64) -> None:
        self.skms = skms
        self.classifier = classifier
        self.spec_config = spec_config
        self.reshape_config = reshape_config
        self.pool_config = pool_config
        self.dtype = dtype

    def __call__(self, chunk: np.ndarray) -> np.ndarray:
        mel_spec, _, _ = spec_transform.transform_mel_spectrogram(
            sound_wave=chunk,
            sample_rate=self.spec_config.sample_rate,
            n_fft=self.spec_config.n_fft,
            n_mels=self.spec_config.n_mels,
            freq_min=self.spec_config.freq_min,
            freq_max=self.spec_config.freq_max,
            window_size=self.spec_config.window_size,
            hop_size=self.spec_config.hop_size,
            apply_log=self.spec_config.apply_log)
        slices: Sequence[np.ndarray] = spec_reshape.slice_spectrogram(
            spectrogram=mel_spec,
            slice_size=self.reshape_config.slice_size,
            stride_size=self.reshape_config.stride_size)
        flat_slices: Sequence[np.ndarray] = list()
        for slice in slices:
            flat: np.ndarray = spec_reshape.flatten_slice(slice, copy=True)
            flat_slices.append(flat)
        proj_slices: np.ndarray = fe_skm_proj.proj_onnx_skm(flat_slices,
                                                            self.skms,
                                                            dtype=self.dtype)
        pool_slices: Sequence[np.ndarray] = fe_pool.apply_pool_func(
            spec_projs=proj_slices,
            pool_func=fe_pool.MeanStdPool(),
            pool_size=self.pool_config.pool_size,
            stride_size=self.pool_config.stride_size)
        labels: np.ndarray = self.classifier.run(
            output_names=["label"],
            input_feed={"X": np.asarray(pool_slices, self.dtype)})[0]
        return labels

    def run(self, audio_stream: AudioStream,
            callback: Optional[Callable[[np.ndarray], None]]):
        for curr_chunk in audio_stream:
            label = self.__call__(curr_chunk)
            if callback is not None:
                callback(label)


#%%
def get_svc_path(curr_val_fold: int,
                 svm_model_root_path: str,
                 val_path_stub: str,
                 file_ext: str = ".onnx"):
    svc_path_stub: str = os.path.join(svm_model_root_path,
                                      val_path_stub + file_ext)
    svc_path = str.format(svc_path_stub, curr_val_fold)
    return svc_path


def get_svc(svc_path: str) -> InferenceSession:
    return InferenceSession(svc_path)


@profile
def main():
    # AUDIO_PATH: str = "../test/test_dataset/folder_dataset/0_0.wav"
    AUDIO_PATH: str = "../../notebook/test/test_audio/fire_01.wav"
    CONFIG_ROOT_PATH: str = "../../config"
    MODEL_ROOT_PATH: str = "../../model/binary_fire/spec_00_reshape_00_skm_00"
    SPEC_CONFIG_PATH: str = os.path.join(CONFIG_ROOT_PATH,
                                         "preprocessing/spec/00.json")
    RESHAPE_CONFIG_PATH: str = os.path.join(CONFIG_ROOT_PATH,
                                            "preprocessing/reshape/00.json")
    POOL_CONFIG_PATH: str = os.path.join(CONFIG_ROOT_PATH,
                                         "feature_engineering/pool/02.json")
    CURR_VAL_FOLD: int = 1
    VAL_PATH_STUB: str = "val_{:02d}"
    CLASS_PATH_STUB: str = "class_{:02d}"
    SKM_MODEL_ROOT_PATH: str = os.path.join(MODEL_ROOT_PATH, "skm")
    # SVM_MODEL_ROOT_PATH: str = os.path.join(MODEL_ROOT_PATH, "svm",
    #                                         "pool_02_svc_04")
    # SVM_MODEL_ROOT_PATH: str = os.path.join(MODEL_ROOT_PATH, "rfc",
    # "pool_02_pca_00_rfc_02")
    SVM_MODEL_ROOT_PATH: str = os.path.join(
        MODEL_ROOT_PATH, "svm", "augment_01_pool_02_pca_00_svc_01")
    CLASS_SKM_PATH_STUB: str = os.path.join(CLASS_PATH_STUB, "model.onnx")
    CURR_VAL_SKM_PATH_STUB: str = skl_skm_laoder.get_curr_val_skm_path_stub(
        curr_val_fold=CURR_VAL_FOLD,
        skm_root_path=SKM_MODEL_ROOT_PATH,
        val_fold_path_stub=VAL_PATH_STUB,
        class_skm_path_stub=CLASS_SKM_PATH_STUB)
    CURR_VAL_SVC_PATH: str = get_svc_path(
        curr_val_fold=CURR_VAL_FOLD,
        svm_model_root_path=SVM_MODEL_ROOT_PATH,
        val_path_stub=VAL_PATH_STUB)

    #%%
    dtype = np.float64

    #%%
    spec_config: conf_spec.MelSpecConfig = conf_spec.get_spec_config_from_json(
        SPEC_CONFIG_PATH, conf_spec.MelSpecConfig)
    reshape_config: conf_reshape.ReshapeConfig = conf_reshape.get_reshape_config_from_json(
        RESHAPE_CONFIG_PATH)
    pool_config: conf_pool.PoolConfig = conf_pool.get_pool_config_from_json(
        POOL_CONFIG_PATH)

    #%%
    skms: Sequence[InferenceSession] = skl_skm_laoder.load_onnx_skms(
        CURR_VAL_SKM_PATH_STUB, 2)
    classifier: InferenceSession = get_svc(CURR_VAL_SVC_PATH)

    #%%
    wav_stream = WaveStream(wav_path=AUDIO_PATH,
                            sample_rate=spec_config.sample_rate,
                            chunk_sec=1.0)
    audio_classifier = AudioClassifier(skms=skms,
                                       classifier=classifier,
                                       spec_config=spec_config,
                                       reshape_config=reshape_config,
                                       pool_config=pool_config)
    #%%
    res = list()
    for curr_chunk in wav_stream:
        curr_label = audio_classifier(curr_chunk)
        res.append(curr_label)

    #%%
    res_np = np.concatenate(tuple(res), axis=None)
    np.bincount(res_np, minlength=2)


if __name__ == '__main__':
    main()
