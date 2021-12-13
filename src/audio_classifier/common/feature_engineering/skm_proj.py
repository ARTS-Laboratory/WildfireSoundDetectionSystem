from typing import List, Sequence

import numpy as np
from onnxruntime.capi.onnxruntime_inference_collection import InferenceSession
from sklearn_plugins.cluster.spherical_kmeans import SphericalKMeans


def proj_skl_skm(spec_flat_slices: Sequence[np.ndarray],
                 skms: Sequence[SphericalKMeans]) -> np.ndarray:
    """Project the flat slices of an audio to trained cluster centroids.

    Args:
        audio_flat_slices (Sequence[np.ndarray]): (n_slices, n_mels * slice_size) The flat slices of a spectrogram.
        skm (Sequence[SphericalKMeans]): All the spherical k-means

    Returns:
        spec_projs (np.ndarray): (n_slices, n_total_centroids) The projected slices.
    """
    slices: np.ndarray = np.asarray(spec_flat_slices)
    spec_projs_list: List[np.ndarray] = [
        skm.transform(slices, copy=True) for skm in skms
    ]
    if len(spec_projs_list) == 1:
        return spec_projs_list[0]
    spec_projs: np.ndarray = np.concatenate(tuple(spec_projs_list), axis=1)
    return spec_projs


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
