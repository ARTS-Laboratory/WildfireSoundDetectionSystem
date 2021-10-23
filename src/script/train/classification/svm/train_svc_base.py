import pickle
from argparse import ArgumentParser, Namespace
from os import path
from typing import List

import audio_classifier.config.feature_engineering.pool as conf_pool
import audio_classifier.config.preprocessing.reshape as conf_reshape
import audio_classifier.config.preprocessing.spec as conf_spec
import audio_classifier.train.config.alg as conf_alg
import audio_classifier.train.config.dataset as conf_dataset
import audio_classifier.train.config.loader as conf_loader
import numpy as np
from sklearn.svm import SVC

from ... import train_common
from .. import classify_common

MetaDataType = train_common.MetaDataType
CollateFuncType = train_common.CollateFuncType


def get_argparse() -> ArgumentParser:
    parser = ArgumentParser(parents=[
        conf_dataset.DatasetConfigArgumentParser(),
        conf_spec.SpecConfigArgumentParser(),
        conf_reshape.ReshapeConfigArgumentParser(),
        conf_pool.PoolConfigArgumentParser(),
        conf_alg.SVCArgumentParser(),
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
    POOL_CONFIG_PATH: str = argv.pool_config_path
    SVC_CONFIG_PATH: str = argv.svc_config_path
    LOADER_CONFIG_PATH: str = argv.loader_config_path
    dataset_config: conf_dataset.PreSplitFoldDatasetConfig = conf_dataset.get_dataset_config_from_json(
        DATASET_CONFIG_PATH, argv, conf_dataset.PreSplitFoldDatasetConfig)
    mel_spec_config: conf_spec.MelSpecConfig = conf_spec.get_spec_config_from_json(
        SPEC_CONFIG_PATH, conf_spec.MelSpecConfig)
    reshape_config: conf_reshape.ReshapeConfig = conf_reshape.get_reshape_config_from_json(
        RESHAPE_CONFIG_PATH)
    loader_config: conf_loader.LoaderConfig = conf_loader.get_loader_config_from_json(
        LOADER_CONFIG_PATH)
    svc_config: conf_alg.SVCConfig = conf_alg.get_alg_config_from_json(
        SVC_CONFIG_PATH, conf_alg.SVCConfig)
    pool_config: conf_pool.PoolConfig = conf_pool.get_pool_config_from_json(
        POOL_CONFIG_PATH)
    return dataset_config, mel_spec_config, reshape_config, pool_config, svc_config, loader_config


def train_svc(curr_val_fold: int,
              dataset: classify_common.ProjDataset,
              svc_config: conf_alg.SVCConfig,
              export_path: str,
              model_path_stub: str = "val_{:02d}.pkl") -> SVC:
    train_slices, train_labels = classify_common.convert_to_ndarray(
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
                 svc_config: conf_alg.SVCConfig,
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
