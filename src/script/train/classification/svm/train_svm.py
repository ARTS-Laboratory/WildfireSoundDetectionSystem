import os
import pickle
from argparse import ArgumentParser, Namespace
from collections import deque
from dataclasses import dataclass, field
from functools import partial
from os import path
from typing import Callable, List, Sequence, Tuple

import audio_classifier.common.feature_engineering.pool as feature_pool
import audio_classifier.config.feature_engineering.pool as conf_pool
import audio_classifier.config.preprocessing.reshape as conf_reshape
import audio_classifier.config.preprocessing.spec as conf_spec
import audio_classifier.train.collate.base as collate_base
import audio_classifier.train.collate.feature_engineering.skm as collate_skm
import audio_classifier.train.collate.preprocessing.spectrogram.reshape as collate_reshape
import audio_classifier.train.collate.preprocessing.spectrogram.transform as collate_transform
import audio_classifier.train.config.dataset as conf_dataset
import audio_classifier.train.config.loader as conf_loader
import audio_classifier.train.data.dataset.composite as dataset_composite
import numpy as np
import script.train.common as script_common
from sklearn.svm import SVC
from sklearn_plugins.cluster.spherical_kmeans import SphericalKMeans

MetaDataType = script_common.MetaDataType
CollateFuncType = script_common.CollateFuncType


@dataclass
class ProjDataset:
    filenames: Sequence[str] = field()
    all_file_spec_projs: Sequence[Sequence[np.ndarray]] = field()
    sample_freqs: Sequence[np.ndarray] = field()
    sample_times: Sequence[np.ndarray] = field()
    labels: Sequence[int] = field()


def get_argparse() -> ArgumentParser:
    parser = ArgumentParser(parents=[
        conf_dataset.DatasetConfigArgumentParser(),
        conf_spec.SpecConfigArgumentParser(),
        conf_reshape.ReshapeConfigArgumentParser(),
        conf_pool.PoolConfigArgumentParser(),
        conf_loader.LoaderConfigArgumentParser()
    ])
    parser.add_argument("--skm_root_path",
                        type=str,
                        required=True,
                        help="path to the root of all skm model")
    parser.add_argument("--export_path",
                        type=str,
                        required=True,
                        help="the export path")
    return parser


def parse_args(args: List[str]) -> Namespace:
    parser = get_argparse()
    argv = parser.parse_args(args)
    return argv


def get_config(argv: Namespace):
    DATASET_CONFIG_PATH: str = argv.dataset_config_path
    SPEC_CONFIG_PATH: str = argv.spec_config_path
    RESHAPE_CONFIG_PATH: str = argv.reshape_config_path
    LOADER_CONFIG_PATH: str = argv.loader_config_path
    POOL_CONFIG_PATH: str = argv.pool_config_path
    dataset_config: conf_dataset.PreSplitFoldDatasetConfig = conf_dataset.get_dataset_config_from_json(
        DATASET_CONFIG_PATH, argv, conf_dataset.PreSplitFoldDatasetConfig)
    mel_spec_config: conf_spec.MelSpecConfig = conf_spec.get_spec_config_from_json(
        SPEC_CONFIG_PATH, conf_spec.MelSpecConfig)
    reshape_config: conf_reshape.ReshapeConfig = conf_reshape.get_reshape_config_from_json(
        RESHAPE_CONFIG_PATH)
    loader_config: conf_loader.LoaderConfig = conf_loader.get_loader_config_from_json(
        LOADER_CONFIG_PATH)
    pool_config: conf_pool.PoolConfig = conf_pool.get_pool_config_from_json(
        POOL_CONFIG_PATH)
    return dataset_config, mel_spec_config, reshape_config, loader_config, pool_config


def get_curr_val_skm_path_stub(curr_val_fold: int, skm_root_path: str,
                               val_fold_path_stub: str,
                               class_skm_path_stub: str):
    val_path = str.format(val_fold_path_stub, curr_val_fold)
    curr_val_skm_path_stub = os.path.join(skm_root_path, val_path,
                                          class_skm_path_stub)
    return curr_val_skm_path_stub


def load_skms(curr_val_skm_path_stub: str,
              n_classes: int) -> Sequence[SphericalKMeans]:
    """Load skm model of all 

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


def get_collate_func(
    skms: Sequence[SphericalKMeans],
    mel_spec_config: conf_spec.MelSpecConfig,
    reshape_config: conf_reshape.ReshapeConfig,
    pool_config: conf_pool.PoolConfig,
    pool_func: Callable[[np.ndarray], np.ndarray] = feature_pool.MeanStdPool()
) -> CollateFuncType:
    collate_func: CollateFuncType = collate_base.EnsembleCollateFunction(
        collate_funcs=[
            partial(collate_transform.mel_spectrogram_collate,
                    config=mel_spec_config),
            partial(collate_reshape.slice_flatten_collate,
                    config=reshape_config),
            partial(collate_skm.skm_skl_proj_collate, skms=skms),
            partial(collate_skm.skm_projs_pool_collate,
                    pool_func=pool_func,
                    pool_config=pool_config)
        ])
    return collate_func


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
              export_path: str,
              model_path_stub: str = "val{:02d}.pkl"):
    curr_val_svc_path = path.join(export_path,
                                  str.format(model_path_stub, curr_val_fold))
    train_slices, train_labels = _create_slices_set(
        all_file_spec_projs=dataset.all_file_spec_projs, labels=dataset.labels)
    svc = SVC()
    svc.fit(train_slices, train_labels)
    with open(curr_val_svc_path, "wb") as svc_file:
        pickle.dump(svc, svc_file)
    return svc


def report_slices_acc(svc: SVC, train: ProjDataset, val: ProjDataset):
    train_slices, train_labels = _create_slices_set(train.all_file_spec_projs,
                                                    train.labels)
    val_slices, val_labels = _create_slices_set(val.all_file_spec_projs,
                                                val.labels)
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
