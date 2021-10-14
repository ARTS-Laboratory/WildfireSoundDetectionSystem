import os
from dataclasses import dataclass, field
from typing import Sequence, Tuple

import audio_classifier.train.config.loader as conf_loader
import audio_classifier.train.data.dataset.composite as dataset_composite
import matplotlib
import numpy as np
import script.train.common as script_common

from .. import fit_common

matplotlib.use("agg", force=True)

MetaDataType = script_common.MetaDataType
CollateFuncType = script_common.CollateFuncType


@dataclass
class FreqRangeSliceDataset:
    filenames: Sequence[str] = field()
    range_flat_slices: Sequence[Sequence[np.ndarray]] = field()
    sample_freqs: Sequence[np.ndarray] = field()
    sample_times: Sequence[np.ndarray] = field()
    labels: Sequence[int] = field()


def generate_slice_dataset(
    curr_val_fold: int,
    dataset_generator: dataset_composite.KFoldDatasetGenerator,
    collate_function: CollateFuncType, loader_config: conf_loader.LoaderConfig
) -> Tuple[FreqRangeSliceDataset, FreqRangeSliceDataset]:
    """Generate the frequency range slice dataset.

    Args:
        curr_val_fold (int): The current validation fold number
        dataset_generator (dataset_composite.KFoldDatasetGenerator): The dataset generator.
        collate_function (CollateFuncType): The function used to process sound wave.
        loader_config (conf_loader.LoaderConfig): The loader configuration.

    Returns:
        Tuple[FreqRangeSliceDataset, FreqRangeSliceDataset]: (train_dataset, val_dataset)
    """
    np.seterr(divide="ignore")
    ret_raw_dataset = script_common.generate_dataset(
        curr_val_fold=curr_val_fold,
        dataset_generator=dataset_generator,
        collate_function=collate_function,
        loader_config=loader_config)
    np.seterr(divide="warn")
    ret_dataset: Sequence[FreqRangeSliceDataset] = list()
    for raw_dataset in ret_raw_dataset:
        filenames, all_files_split_flat_slices, sample_freqs, sample_times, labels = raw_dataset
        n_splits: int = len(all_files_split_flat_slices[0])
        range_flat_slices: Sequence[Sequence[np.ndarray]] = [
            [] for _ in range(n_splits)
        ]
        for curr_file_splits_flat_slices in all_files_split_flat_slices:
            for curr_split_idx, curr_split_flat_slices in enumerate(
                    curr_file_splits_flat_slices):
                range_flat_slices[curr_split_idx].extend(
                    curr_split_flat_slices)
        dataset = FreqRangeSliceDataset(filenames=filenames,
                                        range_flat_slices=range_flat_slices,
                                        sample_freqs=sample_freqs,
                                        sample_times=sample_times,
                                        labels=labels)
        ret_dataset.append(dataset)
    return ret_dataset[0], ret_dataset[1]


def get_curr_class_range_path(export_path: str, curr_val_fold: int,
                              curr_class: int, curr_range_path: str) -> str:
    """Get and create curr_class_path

    Args:
        export_path (str): [description]
        curr_val_fold (int): [description]
        curr_class (int): [description]
        curr_range_path (str): [description]

    Returns:
        str: [description]
    """
    curr_class_path: str = fit_common.get_curr_class_path(
        export_path=export_path,
        curr_val_fold=curr_val_fold,
        curr_class=curr_class)
    curr_class_range_path: str = os.path.join(curr_class_path, curr_range_path)
    return curr_class_range_path
