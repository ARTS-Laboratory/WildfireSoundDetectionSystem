from typing import List, Sequence

import numpy as np
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
        skm.predict(slices, copy=True) for skm in skms
    ]
    if len(spec_projs_list) == 1:
        return spec_projs_list[0]
    spec_projs: np.ndarray = np.concatenate(tuple(spec_projs_list), axis=1)
    return spec_projs
