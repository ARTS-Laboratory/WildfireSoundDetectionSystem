import os
from abc import ABC, abstractmethod
from collections import deque
from functools import partial
from math import ceil
from typing import (Any, Callable, Deque, Iterable, Iterator, List,
                    MutableSequence, Optional, Sequence, Tuple)

import audio_classifier.config.feature_engineering.pool as conf_pool
import audio_classifier.config.preprocessing.reshape as conf_reshape
import audio_classifier.config.preprocessing.spec as conf_spec
import librosa.core as rosa_core
import librosa.feature as rosa_feature
import numpy as np
import script.train.skl_loader.skm as skl_skm_laoder
from memory_profiler import profile
from onnxruntime import InferenceSession
from onnxruntime.capi.onnxruntime_inference_collection import InferenceSession
from overrides import overrides


@profile
def transform_stft_spectrogram(
        sound_wave: np.ndarray, sample_rate: int, n_fft: int, window_size: int,
        hop_size: int,
        apply_log: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform a sound wave to stft-spectrogram.

    Args:
        sound_wave (np.ndarray): (sample_rate*n_secs, ) The sound wave to be transformed.
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
        stft_spec = 10 * np.log10(stft_spec)
        stft_spec = np.nan_to_num(stft_spec,
                                  copy=False,
                                  nan=0.0,
                                  posinf=0.0,
                                  neginf=0.0)
    return stft_spec, stft_freq, stft_time


@profile
def transform_mel_spectrogram(
        sound_wave: np.ndarray, sample_rate: int, n_fft: int, n_mels: int,
        freq_min: float, freq_max: float, window_size: int, hop_size: int,
        apply_log: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform a sound wave to mel-spectrogram.

    Args:
        sound_wave (np.ndarray): (sample_rate*n_secs, ) The sound wave to be transformed.
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
                                                       n_mels=n_mels,
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
        mel_spec = 10 * np.log10(mel_spec)
        mel_spec = np.nan_to_num(mel_spec,
                                 copy=False,
                                 nan=0.0,
                                 posinf=0.0,
                                 neginf=0.0)
    return mel_spec, mel_freq, mel_time


@profile
def slice_spectrogram(spectrogram: np.ndarray,
                      slice_size: int,
                      stride_size: int,
                      copy: bool = False) -> Sequence[np.ndarray]:
    """Slice a spectrogram.

    Args:
        spectrogram (np.ndarray): (n_sample_freq, n_sample_time) Raw spectrogram to be slice.
        slice_size (int): Number of time bins for each sliced spectrogram
        stride_size (int): Number of time bins to skip while slicing.
        copy (bool): if `True`, then the returned slices has no relation with the passed in spectrogram. Defaults to `False`.

    Returns:
        slices Sequence[np.ndarray]: (n_slices, n_sample_freq, slice_size) Sliced spectrogram for the given raw_spectrogram.
    """
    spec_slices: Deque[np.ndarray] = deque()
    for i in range(0, spectrogram.shape[1] - slice_size + 1, stride_size):
        curr_slice: np.ndarray = spectrogram[:, i:i + slice_size]
        if copy == True:
            spec_slices.append(np.copy(curr_slice))
        else:
            spec_slices.append(curr_slice)
    return spec_slices


@profile
def flatten_slice(slice: np.ndarray, copy: bool = False) -> np.ndarray:
    """Convert a slice of spectrogram in to a row vector.

    Args:
        slice (np.ndarray): (n_sample_freq, slice_size) The spectrogram slice to be flatten.
        copy (bool): If `True`, then the returned `flat_slice` has no relation with the passed in `slice`. Defaults to `False`.

    Returns:
        flat_slice (np.ndarray): (n_sample_freq * slice_size) A flattened spectrogram slice in the order of `[slice[:, 0].T, slice[:, 1].T, ..., slice[:, slice_size-1].T]`
    """
    n_sample_freq: int = slice.shape[0]
    slice_size: int = slice.shape[1]
    flat_slice: np.ndarray = slice.reshape((n_sample_freq * slice_size),
                                           order="F")
    if copy == True:
        return np.copy(flat_slice)
    return flat_slice


@profile
def unflatten_slice(flat_slice: np.ndarray,
                    slice_size: int,
                    copy: bool = False):
    """Unflatten a slice back to original spectrogram shape.

    Args:
        flat_slice (np.ndarray): (n_sample_freq*slice_size) A flattened spectrogram slice in the order of `[*(slice[:, 0].T), *(slice[:, 1].T), ..., *(slice[:, slice_size-1].T)]` to be restored.
        slice_size (int): Number of time bins for each sliced spectrogram.
        copy (bool, optional): If `True`, then the returned `spec_slice` has no relaition with the passed in `flat_slice`. Defaults to False.

    Returns:
        spec_slice (np.ndarray): (n_sample_freq, slice_size) The restored sliced spectrogram.
    """
    n_sample_freq: int = flat_slice.shape[0] // slice_size
    spec_slice: np.ndarray = flat_slice.reshape((n_sample_freq, slice_size),
                                                order="F")
    if copy == True:
        return np.copy(spec_slice)
    return spec_slice


@profile
def proj_onnx_skm(spec_flat_slices: Sequence[np.ndarray],
                  skms: Sequence[InferenceSession],
                  dtype=np.float64) -> np.ndarray:
    """Project the flat slices of an audio to trained cluster centroids.

    Args:
        audio_flat_slices (Sequence[np.ndarray]): (n_slices, n_mels * slice_size) The flat slices of a spectrogram.
        skm (Sequence[SphericalKMeans]): All the spherical k-means

    Returns:
        spec_projs (np.ndarray): (n_slices, n_total_centroids) The projected slices.
    """
    slices: np.ndarray = np.asarray(spec_flat_slices, dtype)
    spec_projs_list: List[np.ndarray] = [
        skm.run(output_names=["scores"], input_feed={"X": slices})[0]
        for skm in skms
    ]
    if len(spec_projs_list) == 1:
        return spec_projs_list[0]
    spec_projs: np.ndarray = np.concatenate(tuple(spec_projs_list), axis=1)
    return spec_projs


class PoolFunc:
    _pool_funcs: Sequence[Callable[[np.ndarray], np.ndarray]]

    @profile
    def __init__(self, pool_funcs: Sequence[Callable[[np.ndarray],
                                                     np.ndarray]]):
        """Constructor for pooling function that chains multiple pooling function.

        Args:
            pool_funcs (Sequence[Callable[[np.ndarray], np.ndarray]]): Each pooling function takes an np.ndarray of shape (n_slices, n_features) and output and np.ndarray of shape (n_output_features, )
        """
        self._pool_funcs = pool_funcs

    @profile
    def __call__(self, input: np.ndarray) -> np.ndarray:
        """Iterate and apply all the pooling functions and generate an output vector.

        Args:
            input (np.ndarray): (n_slices, n_features)

        Returns:
            output (np.ndarray): (n_output_features, )
        """
        output_list: Tuple[np.ndarray, ...] = tuple(
            [pool_func(input) for pool_func in self._pool_funcs])
        output: np.ndarray = np.concatenate(output_list, axis=0)
        return output


class MeanStdPool(PoolFunc):
    @profile
    def __init__(self):
        super().__init__(
            pool_funcs=[partial(np.mean, axis=0),
                        partial(np.std, axis=0)])

    @profile
    def __call__(self, input: np.ndarray) -> np.ndarray:
        if input.shape[0] == 1:
            return self._pool_funcs[0](input)
        return super().__call__(input)


@profile
def apply_pool_func(
    spec_projs: np.ndarray,
    pool_func: Callable[[np.ndarray], np.ndarray],
    pool_size: int = -1,
    stride_size: int = -1,
) -> Sequence[np.ndarray]:
    """Apply pooling function on the all the projection vectors of a file.

    Args:
        projections (np.ndarray): (n_slices, n_clusters) The input projection vector of an audio.
        pool_func (Callable[[np.ndarray], np.ndarray]): The function used for pooling.
        pool_size (int, optional): The size of the sliding window. Defaults to -1 set the pool_size to len(projections).
        stride_size (int, optional): The stride of the sliding window. Defaults to -1 set the stride_size to pool_size.

    Returns:
        pool_vectors (Sequence[np.ndarray]): (n_slices_prime, n_output_features)
    """
    if pool_size == -1:
        pool_size = len(spec_projs)
    if stride_size == -1:
        stride_size = pool_size
    if len(spec_projs) < pool_size:
        return list()
    pool_projs: MutableSequence[np.ndarray] = deque()
    for begin_idx in range(0, len(spec_projs) - pool_size + 1, stride_size):
        end_idx: int = begin_idx + pool_size
        curr_spec_projs: np.ndarray = spec_projs[begin_idx:end_idx, :]
        curr_pool_proj: np.ndarray = pool_func(curr_spec_projs)
        pool_projs.append(curr_pool_proj)
    pool_projs = list(pool_projs)
    return pool_projs


class AudioStream(Iterable[np.ndarray], ABC):
    sample_rate: float
    chunk_sec: float
    frame_length: int

    @profile
    def __init__(self, sample_rate: float, chunk_sec: float) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.chunk_sec = chunk_sec
        self.frame_length = ceil(sample_rate * chunk_sec)

    @profile
    @abstractmethod
    def __iter__(self) -> Iterator[np.ndarray]:
        pass


class WaveStream(AudioStream):
    src_sample_rate: float
    src_frame_length: int
    wav_path: str

    @profile
    def __init__(self, wav_path: str, sample_rate: float,
                 chunk_sec: float) -> None:
        super().__init__(sample_rate=sample_rate, chunk_sec=chunk_sec)
        self.wav_path = wav_path
        self.src_sample_rate = rosa_core.get_samplerate(path=self.wav_path)
        self.src_frame_length = ceil(self.src_sample_rate * chunk_sec)

    @profile
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

    @profile
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

    @profile
    def __call__(self, chunk: np.ndarray) -> np.ndarray:
        mel_spec, _, _ = transform_mel_spectrogram(
            sound_wave=chunk,
            sample_rate=self.spec_config.sample_rate,
            n_fft=self.spec_config.n_fft,
            n_mels=self.spec_config.n_mels,
            freq_min=self.spec_config.freq_min,
            freq_max=self.spec_config.freq_max,
            window_size=self.spec_config.window_size,
            hop_size=self.spec_config.hop_size,
            apply_log=self.spec_config.apply_log)
        slices: Sequence[np.ndarray] = slice_spectrogram(
            spectrogram=mel_spec,
            slice_size=self.reshape_config.slice_size,
            stride_size=self.reshape_config.stride_size)
        flat_slices: Sequence[np.ndarray] = list()
        for slice in slices:
            flat: np.ndarray = flatten_slice(slice, copy=True)
            flat_slices.append(flat)
        proj_slices: np.ndarray = proj_onnx_skm(flat_slices,
                                                self.skms,
                                                dtype=self.dtype)
        pool_slices: Sequence[np.ndarray] = apply_pool_func(
            spec_projs=proj_slices,
            pool_func=MeanStdPool(),
            pool_size=self.pool_config.pool_size,
            stride_size=self.pool_config.stride_size)
        labels: np.ndarray = self.classifier.run(
            output_names=["label"],
            input_feed={"X": np.asarray(pool_slices, self.dtype)})[0]
        return labels

    @profile
    def run(self, audio_stream: AudioStream,
            callback: Optional[Callable[[np.ndarray], None]]):
        for curr_chunk in audio_stream:
            label = self.__call__(curr_chunk)
            if callback is not None:
                callback(label)


@profile
def get_svc_path(curr_val_fold: int,
                 svm_model_root_path: str,
                 val_path_stub: str,
                 file_ext: str = ".onnx"):
    svc_path_stub: str = os.path.join(svm_model_root_path,
                                      val_path_stub + file_ext)
    svc_path = str.format(svc_path_stub, curr_val_fold)
    return svc_path


@profile
def get_svc(svc_path: str) -> InferenceSession:
    return InferenceSession(svc_path)


@profile
def main():
    AUDIO_PATH: str = "../../notebook/test/test_audio/fire_test.wav"
    CONFIG_ROOT_PATH: str = "../../config"
    MODEL_ROOT_PATH: str = "../../model/binary_fire/spec_00_reshape_00_skm_00/100_100/dtype_float32"
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
    SVM_MODEL_ROOT_PATH: str = os.path.join(MODEL_ROOT_PATH, "svm")
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
    dtype = np.float32

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
                                       pool_config=pool_config,
                                       dtype=dtype)
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

# model/binary_fire/spec_00_reshape_00_skm_00/100_100/dtype_float32/skm/val_01/class_00/model.onnx