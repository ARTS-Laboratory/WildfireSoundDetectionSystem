from collections import deque
from dataclasses import dataclass, field
from typing import Sequence, Tuple, Union

import audio_classifier.train.config.loader as conf_loader
import audio_classifier.train.data.dataset.composite as dataset_composite
import numpy as np
import script.train.common as script_common
from sklearn.base import ClassifierMixin
from sklearn.pipeline import Pipeline

MetaDataType = script_common.MetaDataType
CollateFuncType = script_common.CollateFuncType


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
    np.seterr(divide="ignore")
    ret_raw_datasets = script_common.generate_dataset(
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
                      train: ProjDataset, val: ProjDataset):
    train_slices, train_labels = convert_to_ndarray(train.all_file_spec_projs,
                                                    train.labels)
    val_slices, val_labels = convert_to_ndarray(val.all_file_spec_projs,
                                                val.labels)
    report_slices_acc_np(classifier=classifier,
                         train_slices=train_slices,
                         train_labels=train_labels,
                         val_slices=val_slices,
                         val_labels=val_labels)


def report_slices_acc_np(classifier: Union[ClassifierMixin, Pipeline],
                         train_slices: np.ndarray, train_labels: np.ndarray,
                         val_slices: np.ndarray, val_labels: np.ndarray):
    train_acc: float = classifier.score(train_slices, train_labels)
    val_acc: float = classifier.score(val_slices, val_labels)
    info_str: str = str.format("train: {:.5f} val: {:.5f}", train_acc, val_acc)
    print(info_str)


def convert_to_ndarray(all_file_spec_projs: Sequence[Sequence[np.ndarray]],
                       labels: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    train_slices_list: Sequence[np.ndarray] = deque()
    train_labels_list: Sequence[int] = deque()
    for spec_projs, label in zip(all_file_spec_projs, labels):
        train_slices_list.extend(spec_projs)
        train_labels_list.extend([label] * len(spec_projs))
    train_slices: np.ndarray = np.array(train_slices_list)
    train_labels: np.ndarray = np.array(train_labels_list)
    return train_slices, train_labels
