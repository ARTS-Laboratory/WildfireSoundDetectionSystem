import numpy as np
from sklearn.decomposition import PCA


def decorrelate_skl_pca(
    spec_projs: np.ndarray,
    pca: PCA,
) -> np.ndarray:
    """Apply pooling function on the all the projection vectors of a file.

    Args:
        projections (np.ndarray): (n_slices, n_clusters) The input projection vector of an audio.
        pca (PCA): the pca instance used to decorrelate data.

    Returns:
        decorrelated_vectors (Sequence[np.ndarray]): (n_slices, n_components)
    """
    decorrelated_vectors: np.ndarray = pca.transform(spec_projs)
    return decorrelated_vectors
