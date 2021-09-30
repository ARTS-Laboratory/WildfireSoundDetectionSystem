import sys
from argparse import ArgumentParser, Namespace
from functools import partial
from os import path
from typing import Dict, List, Optional, Tuple, TypeVar

import audio_classifier.config.preprocessing.reshape as conf_reshape
import audio_classifier.config.preprocessing.spec as conf_spec
import audio_classifier.train.collate.base as collate_base
import audio_classifier.train.collate.preprocessing.spectrogram.reshape as collate_reshape
import audio_classifier.train.collate.preprocessing.spectrogram.transform as collate_transform
import audio_classifier.train.config.alg as conf_alg
import audio_classifier.train.config.dataset as conf_dataset
import audio_classifier.train.data.dataset.base as dataset_base
import audio_classifier.train.data.dataset.composite as dataset_composite
import audio_classifier.train.data.metadata.query as metadata_query
import audio_classifier.train.data.metadata.reader as metadata_reader
import numpy as np
from sklearn_plugins.cluster.spherical_kmeans import SphericalKMeans


def create_argparse() -> ArgumentParser:
    parser = ArgumentParser(parents=[
        conf_dataset.DatasetConfigArgumentParser(),
        conf_spec.SpecConfigArgumentParser(),
        conf_reshape.ReshapeConfigArgumentParser(),
        conf_alg.SKMArgumentParser()
    ])
    return parser


def parse_args(args: List[str]) -> Namespace:
    parser = create_argparse()
    argv = parser.parse_args(args)
    return argv


def get_config(args: List[str]):
    argv: Namespace = parse_args(args)
    DATASET_CONFIG_PATH: str = argv.dataset_config_path
    SPEC_CONFIG_PATH: str = argv.spec_config_path
    RESHAPE_CONFIG_PATH: str = argv.reshape_config_path
    SKM_CONFIG_PATH: str = argv.skm_config_path
    dataset_config: conf_dataset.PreSplitFoldDatasetConfig = conf_dataset.get_dataset_config_from_json(
        DATASET_CONFIG_PATH, conf_dataset.PreSplitFoldDatasetConfig)
    mel_spec_config: conf_spec.MelSpecConfig = conf_spec.get_spec_config_from_json(
        SPEC_CONFIG_PATH, conf_spec.MelSpecConfig)
    reshape_config: conf_reshape.ReshapeConfig = conf_reshape.get_reshape_config_from_json(
        RESHAPE_CONFIG_PATH)
    skm_config: conf_alg.SKMConfig = conf_alg.get_alg_config_from_json(
        SKM_CONFIG_PATH, conf_alg.SKMConfig)
    return dataset_config, mel_spec_config, reshape_config, skm_config
