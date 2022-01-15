import os
import sys
import time
from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, field
from math import ceil
from typing import Callable, Iterable, Iterator, List, Optional, Sequence, Type

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
from onnxruntime import InferenceSession
from onnxruntime.capi.onnxruntime_inference_collection import InferenceSession
from overrides import overrides


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

    def __init__(self, skms: Sequence[InferenceSession],
                 classifier: InferenceSession,
                 spec_config: conf_spec.MelSpecConfig,
                 reshape_config: conf_reshape.ReshapeConfig,
                 pool_config: conf_pool.PoolConfig, dtype) -> None:
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


@dataclass
class StreamWaveConfigs():
    # path
    audio_path: str
    skms_path: str
    classifier_path: str

    # configuration
    spec_config: conf_spec.MelSpecConfig
    reshape_config: conf_reshape.ReshapeConfig
    pool_config: conf_pool.PoolConfig

    # runtime
    dtype: Type = field(default=np.float32)


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


def main(args: List[str]):
    configs = parse_args(args)

    skms: Sequence[InferenceSession] = skl_skm_laoder.load_onnx_skms(
        os.path.join(configs.skms_path, "class_{:02d}", "model.onnx"), 2)
    classifier: InferenceSession = get_svc(configs.classifier_path)

    wav_stream = WaveStream(wav_path=configs.audio_path,
                            sample_rate=configs.spec_config.sample_rate,
                            chunk_sec=1.0)
    audio_classifier = AudioClassifier(skms=skms,
                                       classifier=classifier,
                                       spec_config=configs.spec_config,
                                       reshape_config=configs.reshape_config,
                                       pool_config=configs.pool_config,
                                       dtype=configs.dtype)
    res = list()
    times = list()
    for curr_chunk in wav_stream:
        start_time = time.time()
        curr_label = audio_classifier(curr_chunk)
        end_time = time.time()
        res.append(curr_label)
        proc_time = end_time - start_time
        times.append(proc_time)
    res_np = np.concatenate(tuple(res), axis=None)
    print(np.bincount(res_np, minlength=2))
    print(
        str.format("total_time {} mean_time {} std_time {}", np.sum(times),
                   np.mean(times), np.std(times)))


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(parents=[
        conf_spec.SpecConfigArgumentParser(),
        conf_reshape.ReshapeConfigArgumentParser(),
        conf_pool.PoolConfigArgumentParser(),
    ])
    parser.add_argument("--skms_path",
                        type=str,
                        required=True,
                        help="path to the root of all skm model")
    parser.add_argument("--classifier_path",
                        type=str,
                        required=True,
                        help="path to the classifier to be used")
    parser.add_argument("--audio_path",
                        type=str,
                        required=True,
                        help="path to the wave to be streamed")
    parser.add_argument("--dtype",
                        choices=["float32", "float64"],
                        default="float32")
    return parser


def parse_args(args: List[str]):
    parser = get_parser()
    argv: Namespace = parser.parse_args(args)
    dtype: Type = np.float64
    if argv.dtype == "float32":
        dtype = np.float32
    configs = StreamWaveConfigs(
        audio_path=argv.audio_path,
        skms_path=argv.skms_path,
        classifier_path=argv.classifier_path,
        spec_config=conf_spec.get_spec_config_from_json(
            argv.spec_config_path, conf_spec.MelSpecConfig),
        reshape_config=conf_reshape.get_reshape_config_from_json(
            argv.reshape_config_path),
        pool_config=conf_pool.get_pool_config_from_json(argv.pool_config_path),
        dtype=dtype)
    return configs


if __name__ == '__main__':
    args: List[str] = sys.argv[1:]
    main(args)
