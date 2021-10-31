from collections import deque
from dataclasses import dataclass, field
from typing import Sequence, Tuple, Union

import audio_classifier.train.config.loader as conf_loader
import audio_classifier.train.data.dataset.composite as dataset_composite
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.pipeline import Pipeline

from .. import train_common

MetaDataType = train_common.MetaDataType
CollateFuncType = train_common.CollateFuncType


@dataclass
class ProjDataset:
    filenames: Sequence[str] = field()
    all_file_spec_projs: Sequence[Sequence[np.ndarray]] = field()
    sample_freqs: Sequence[np.ndarray] = field()
    sample_times: Sequence[np.ndarray] = field()
    labels: Sequence[int] = field()


def generate_proj_dataset(
    curr_val_fold: int,
    dataset_generator: dataset_composite.KFoldDatasetGenerator,
    collate_function: CollateFuncType, loader_config: conf_loader.LoaderConfig
) -> Tuple[ProjDataset, ProjDataset]:
    """Generate proj dataset

    Args:
        curr_val_fold (int): Current validation fold
        dataset_generator (dataset_composite.KFoldDatasetGenerator): The dataset generator
        collate_function (CollateFuncType): The collate function used to preprocess raw audio.
        loader_config (conf_loader.LoaderConfig): The loader configuration.

    Returns:
        train_dataset (ProjDataset): Training dataset.
        val_dataset (ProjDataset): Validation dataset.
    """
    np.seterr(divide="ignore")
    ret_raw_datasets = train_common.generate_dataset(
        curr_val_fold=curr_val_fold,
        dataset_generator=dataset_generator,
        collate_function=collate_function,
        loader_config=loader_config)
    np.seterr(divide="warn")
    ret_datasets: Sequence[ProjDataset] = list()
    for curr_raw_dataset in ret_raw_datasets:
        filenames, all_file_spec_projs, sample_freqs, sample_times, labels = curr_raw_dataset
        curr_proj_dataset = ProjDataset(
            filenames=filenames,
            all_file_spec_projs=all_file_spec_projs,
            sample_freqs=sample_freqs,
            sample_times=sample_times,
            labels=labels)
        ret_datasets.append(curr_proj_dataset)
    return ret_datasets[0], ret_datasets[1]


def report_slices_acc(classifier: Union[ClassifierMixin, Pipeline],
                      train: ProjDataset,
                      val: ProjDataset,
                      to_print: bool = True) -> Tuple[float, float]:
    """Return train and validation accuracy for the current classifier.

    Args:
        classifier (Union[ClassifierMixin, Pipeline]): The classifier to be evaluated.
        train (ProjDataset): Training proj dataset.
        val (ProjDataset): Validation proj dataset.

    Return:
        train_acc (float): The accuracy on training set.
        val_acc (float): The accuracy on validation set.
    """
    train_slices, train_labels = convert_to_ndarray(train.all_file_spec_projs,
                                                    train.labels)
    val_slices, val_labels = convert_to_ndarray(val.all_file_spec_projs,
                                                val.labels)
    return report_slices_acc_np(classifier=classifier,
                                train_slices=train_slices,
                                train_labels=train_labels,
                                val_slices=val_slices,
                                val_labels=val_labels,
                                to_print=to_print)


def report_slices_acc_np(classifier: Union[ClassifierMixin, Pipeline],
                         train_slices: np.ndarray,
                         train_labels: np.ndarray,
                         val_slices: np.ndarray,
                         val_labels: np.ndarray,
                         to_print: bool = True) -> Tuple[float, float]:
    """Return train and validation accuracy for the current classifier.

    Args:
        classifier (Union[ClassifierMixin, Pipeline]): The classifier to be evaluated.
        train_slices (np.ndarray): (n_slices, n_clusters) Trainig slices.
        train_labels (np.ndarray): (n_slices, ) Trainig class labels.
        val_slices (np.ndarray): (n_slices, n_clusters) Validation slices.
        val_labels (np.ndarray): (n_slices, ) Validation class labels.

    Return:
        train_acc (float): The accuracy on training set.
        val_acc (float): The accuracy on validation set.
    """
    train_acc: float = classifier.score(train_slices, train_labels)
    val_acc: float = classifier.score(val_slices, val_labels)
    if to_print:
        info_str: str = str.format("train: {:.5f} val: {:.5f}", train_acc,
                                   val_acc)
        print(info_str)
    return train_acc, val_acc


def convert_to_ndarray(all_file_spec_projs: Sequence[Sequence[np.ndarray]],
                       labels: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Convert all the spectrogram projection vectors from python Sequnce to ndarray.

    Args:
        all_file_spec_projs (Sequence[Sequence[np.ndarray]]): (n_files, n_slices, n_clusters) The spctrogram slices of all files.
        labels (Sequence[int]): (n_slices, )  Class labels of all files.

    Returns:
        spec_projs (np.ndarray): (n_slices, n_clusters) The current training projs
        spec_labels (np.ndarray): (n_slices, ) The class label correspond to each slice.
    """
    spec_projs_list: Sequence[np.ndarray] = deque()
    labels_list: Sequence[int] = deque()
    for curr_file_spec_projs, curr_label in zip(all_file_spec_projs, labels):
        spec_projs_list.extend(curr_file_spec_projs)
        labels_list.extend([curr_label] * len(curr_file_spec_projs))
    spec_projs: np.ndarray = np.array(spec_projs_list)
    spec_labels: np.ndarray = np.array(labels_list)
    return spec_projs, spec_labels
