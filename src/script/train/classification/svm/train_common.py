import pickle
from collections import deque
from dataclasses import dataclass, field
from os import path
from typing import Sequence, Tuple

import audio_classifier.train.config.loader as conf_loader
import audio_classifier.train.data.dataset.composite as dataset_composite
import numpy as np
import script.train.common as script_common
from audio_classifier.train.config.alg import SVCConfig
from sklearn.svm import SVC

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


def train_svc(curr_val_fold: int,
              dataset: ProjDataset,
              svc_config: SVCConfig,
              export_path: str,
              model_path_stub: str = "val_{:02d}.pkl") -> SVC:
    train_slices, train_labels = _create_slices_set(
        all_file_spec_projs=dataset.all_file_spec_projs, labels=dataset.labels)
    svc: SVC = train_svc_np(curr_val_fold=curr_val_fold,
                            train_slices=train_slices,
                            train_labels=train_labels,
                            svc_config=svc_config,
                            export_path=export_path,
                            model_path_stub=model_path_stub)
    return svc


def train_svc_np(curr_val_fold: int,
                 train_slices: np.ndarray,
                 train_labels: np.ndarray,
                 svc_config: SVCConfig,
                 export_path: str,
                 model_path_stub: str = "val_{:02d}.pkl") -> SVC:
    curr_val_svc_path = path.join(export_path,
                                  str.format(model_path_stub, curr_val_fold))
    svc = SVC(C=svc_config.C,
              kernel=svc_config.kernel,
              degree=svc_config.degree,
              gamma=svc_config.gamma,
              coef0=svc_config.coef0)
    svc.fit(train_slices, train_labels)
    with open(curr_val_svc_path, "wb") as svc_file:
        pickle.dump(svc, svc_file)
    return svc


def report_slices_acc(svc: SVC, train: ProjDataset, val: ProjDataset):
    train_slices, train_labels = _create_slices_set(train.all_file_spec_projs,
                                                    train.labels)
    val_slices, val_labels = _create_slices_set(val.all_file_spec_projs,
                                                val.labels)
    report_slices_acc_np(svc=svc,
                         train_slices=train_slices,
                         train_labels=train_labels,
                         val_slices=val_slices,
                         val_labels=val_labels)


def report_slices_acc_np(svc: SVC, train_slices: np.ndarray,
                         train_labels: np.ndarray, val_slices: np.ndarray,
                         val_labels: np.ndarray):
    train_acc: float = svc.score(train_slices, train_labels)
    val_acc: float = svc.score(val_slices, val_labels)
    info_str: str = str.format("train: {:.5f} val: {:.5f}", train_acc, val_acc)
    print(info_str)


def _create_slices_set(all_file_spec_projs: Sequence[Sequence[np.ndarray]],
                       labels: Sequence[int]):
    train_slices_list: Sequence[np.ndarray] = deque()
    train_labels_list: Sequence[int] = deque()
    for spec_projs, label in zip(all_file_spec_projs, labels):
        train_slices_list.extend(spec_projs)
        train_labels_list.extend([label] * len(spec_projs))
    train_slices: np.ndarray = np.array(train_slices_list)
    train_labels: np.ndarray = np.array(train_labels_list)
    return train_slices, train_labels
