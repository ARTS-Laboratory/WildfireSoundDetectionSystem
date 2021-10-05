from typing import List, Sequence

import numpy as np
from sklearn_plugins.cluster.spherical_kmeans import SphericalKMeans


def proj_skl_skm(spec_flat_slices: Sequence[np.ndarray],
                 skm: SphericalKMeans,
                 copy: bool = True) -> np.ndarray:
    """Project the flat slices of an audio to trained cluster centroids.

    Args:
        audio_flat_slices (Sequence[np.ndarray]): (n_slices, n_mels * slice_size) The flat slices of a spectrogram.
        skm (SphericalKMeans): The spherical k-means instance used to project incoming slices.

    Returns:
        spec_projs (np.ndarray): (n_slices, n_centroids) The projected slices.
    """
    slices: np.ndarray = np.asarray(spec_flat_slices)
    spec_projs: np.ndarray = skm.predict(slices, copy)
    return spec_projs
