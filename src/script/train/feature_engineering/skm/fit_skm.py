from collections import deque
import sys
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, field
from functools import partial
from os import path
from typing import Deque, Dict, List, Optional, Sequence, Tuple, TypeVar

import audio_classifier.config.preprocessing.reshape as conf_reshape
import audio_classifier.config.preprocessing.spec as conf_spec
import audio_classifier.train.collate.base as collate_base
import audio_classifier.train.collate.preprocessing.spectrogram.reshape as collate_reshape
import audio_classifier.train.collate.preprocessing.spectrogram.transform as collate_transform
import audio_classifier.train.config.alg as conf_alg
import audio_classifier.train.config.dataset as conf_dataset
import audio_classifier.train.config.loader as conf_loader
import audio_classifier.train.data.dataset.base as dataset_base
import audio_classifier.train.data.dataset.composite as dataset_composite
import audio_classifier.train.data.metadata.query as metadata_query
import audio_classifier.train.data.metadata.reader as metadata_reader
import numpy as np
import script.train.common as script_common
from sklearn_plugins.cluster.spherical_kmeans import SphericalKMeans

MetaDataType = script_common.MetaDataType
CollateFuncType = script_common.CollateFuncType


@dataclass
class SliceDataset:
    filenames: Sequence[str] = field()
    flat_slices: Sequence[np.ndarray] = field()
    sample_freqs: Sequence[np.ndarray] = field()
    sample_times: Sequence[np.ndarray] = field()
    labels: Sequence[int] = field()


def get_argparse() -> ArgumentParser:
    parser = ArgumentParser(parents=[
        conf_dataset.DatasetConfigArgumentParser(),
        conf_spec.SpecConfigArgumentParser(),
        conf_reshape.ReshapeConfigArgumentParser(),
        conf_alg.SKMArgumentParser(),
        conf_loader.LoaderConfigArgumentParser()
    ])
    return parser


def parse_args(args: List[str]) -> Namespace:
    parser = get_argparse()
    argv = parser.parse_args(args)
    return argv


def get_config(argv: Namespace):
    DATASET_CONFIG_PATH: str = argv.dataset_config_path
    SPEC_CONFIG_PATH: str = argv.spec_config_path
    RESHAPE_CONFIG_PATH: str = argv.reshape_config_path
    SKM_CONFIG_PATH: str = argv.skm_config_path
    LOADER_CONFIG_PATH: str = argv.loader_config_path
    dataset_config: conf_dataset.PreSplitFoldDatasetConfig = conf_dataset.get_dataset_config_from_json(
        DATASET_CONFIG_PATH, conf_dataset.PreSplitFoldDatasetConfig)
    mel_spec_config: conf_spec.MelSpecConfig = conf_spec.get_spec_config_from_json(
        SPEC_CONFIG_PATH, conf_spec.MelSpecConfig)
    reshape_config: conf_reshape.ReshapeConfig = conf_reshape.get_reshape_config_from_json(
        RESHAPE_CONFIG_PATH)
    skm_config: conf_alg.SKMConfig = conf_alg.get_alg_config_from_json(
        SKM_CONFIG_PATH, conf_alg.SKMConfig)
    loader_config: conf_loader.LoaderConfig = conf_loader.get_loader_config_from_json(
        LOADER_CONFIG_PATH)
    return dataset_config, mel_spec_config, reshape_config, skm_config, loader_config


def generate_slice_dataset(
    curr_val_fold: int,
    dataset_generator: dataset_composite.KFoldDatasetGenerator,
    collate_function: CollateFuncType, loader_config: conf_loader.LoaderConfig
) -> Tuple[SliceDataset, SliceDataset]:
    ret_raw_dataset = script_common.generate_dataset(
        curr_val_fold=curr_val_fold,
        dataset_generator=dataset_generator,
        collate_function=collate_function,
        loader_config=loader_config)
    ret_dataset: Sequence[SliceDataset] = list()
    for raw_dataset in ret_raw_dataset:
        filenames, all_files_flat_slices, sample_freqs, sample_times, labels = raw_dataset
        flat_slices: List[np.ndarray] = [
            slices for curr_file_flat_slices in all_files_flat_slices
            for slices in curr_file_flat_slices
        ]
        dataset = SliceDataset(filenames=filenames,
                               flat_slices=flat_slices,
                               sample_freqs=sample_freqs,
                               sample_times=sample_times,
                               labels=labels)
        ret_dataset.append(dataset)
    return ret_dataset[0], ret_dataset[1]
