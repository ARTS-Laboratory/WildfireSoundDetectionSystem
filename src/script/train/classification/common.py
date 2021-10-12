import os
import pickle
from typing import Sequence

from sklearn_plugins.cluster.spherical_kmeans import SphericalKMeans


def get_curr_val_skm_path_stub(curr_val_fold: int, skm_root_path: str,
                               val_fold_path_stub: str,
                               class_skm_path_stub: str):
    val_path = str.format(val_fold_path_stub, curr_val_fold)
    curr_val_skm_path_stub = os.path.join(skm_root_path, val_path,
                                          class_skm_path_stub)
    return curr_val_skm_path_stub


def load_skl_skms(curr_val_skm_path_stub: str,
              n_classes: int) -> Sequence[SphericalKMeans]:
    """Load skm model of all classes in current validation fold.

    Args:
        curr_val_skm_path_stub (str): [description]
        n_classes (int): [description]

    Returns:
        [type]: [description]
    """
    skms: Sequence[SphericalKMeans] = list()
    for curr_class in range(n_classes):
        skm_path: str = str.format(
            curr_val_skm_path_stub,
            curr_class) if n_classes > 1 else curr_val_skm_path_stub
        with open(skm_path, "rb") as skm_file:
            skm: SphericalKMeans = pickle.load(skm_file)
            skms.append(skm)
    return skms
