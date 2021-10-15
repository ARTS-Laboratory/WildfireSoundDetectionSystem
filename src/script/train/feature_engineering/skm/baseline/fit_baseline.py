from dataclasses import dataclass, field
from os import path
from typing import List, Sequence, Tuple

import audio_classifier.train.config.loader as conf_loader
import audio_classifier.train.data.dataset.composite as dataset_composite
import numpy as np
import script.train.common as script_common

MetaDataType = script_common.MetaDataType
CollateFuncType = script_common.CollateFuncType


@dataclass
class SliceDataset:
    """

    Attributes:
        filenames (Sequence[str]): (n_files, )
        flat_slices (Sequence[Sequence[np.ndarray]]): (n_files, n_slices, n_freq_bins*slice_size)
        sample_freqs (Sequence[np.ndarray]): (n_files, n_freq_bins)
        sample_times (Sequence[np.ndarray]): (n_files, n_time_stamps)
        labels (Sequence[int]): (n_file, )
    """
    filenames: Sequence[str] = field()
    flat_slices: Sequence[Sequence[np.ndarray]] = field()
    sample_freqs: Sequence[np.ndarray] = field()
    sample_times: Sequence[np.ndarray] = field()
    labels: Sequence[int] = field()


def generate_slice_dataset(
    curr_val_fold: int,
    dataset_generator: dataset_composite.KFoldDatasetGenerator,
    collate_function: CollateFuncType, loader_config: conf_loader.LoaderConfig
) -> Tuple[SliceDataset, SliceDataset]:
    """Generate the slice dataset.

    Args:
        curr_val_fold (int): The current validation fold number
        dataset_generator (dataset_composite.KFoldDatasetGenerator): The dataset generator.
        collate_function (CollateFuncType): The function used to process sound wave.
        loader_config (conf_loader.LoaderConfig): The loader configuration.

    Returns:
        Tuple[SliceDataset, SliceDataset]: (train_dataset, val_dataset)
    """
    np.seterr(divide="ignore")
    ret_raw_dataset = script_common.generate_dataset(
        curr_val_fold=curr_val_fold,
        dataset_generator=dataset_generator,
        collate_function=collate_function,
        loader_config=loader_config)
    np.seterr(divide="warn")
    ret_dataset: Sequence[SliceDataset] = list()
    for raw_dataset in ret_raw_dataset:
        filenames, flat_slices, sample_freqs, sample_times, labels = raw_dataset
        dataset = SliceDataset(filenames=filenames,
                               flat_slices=flat_slices,
                               sample_freqs=sample_freqs,
                               sample_times=sample_times,
                               labels=labels)
        ret_dataset.append(dataset)
    return ret_dataset[0], ret_dataset[1]


def convert_to_ndarray(
        slice_dataset: SliceDataset) -> Tuple[np.ndarray, np.ndarray]:
    """Wrap slices and labels as np.ndarray.

    Args:
        slice_dataset (SliceDataset): The slice dataset to be converted

    Returns:
        slices (np.ndarray): (n_slices, n_sample_freq * slice_size) The converted slices.
        labels (np.ndarray): (n_slices, ) The converted labels.
    """
    slices: np.ndarray = np.asarray(slice_dataset.flat_slices)
    labels: np.ndarray = np.asarray(slice_dataset.labels)
    return slices, labels
