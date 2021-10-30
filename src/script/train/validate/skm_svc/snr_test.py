from argparse import ArgumentParser, Namespace
from typing import List

import audio_classifier.config.feature_engineering.pool as conf_pool
import audio_classifier.config.preprocessing.reshape as conf_reshape
import audio_classifier.config.preprocessing.spec as conf_spec
import audio_classifier.train.config.dataset as conf_dataset
import audio_classifier.train.config.loader as conf_loader


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
    parser.add_argument("--classifier_path",
                        type=str,
                        required=True,
                        help="path to the clasifier")
    parser.add_argument(
        "--snr_range",
        type=float,
        nargs=2,
        default=[5.0, 50.0],
        required=True,
        help="the range of the tested snr in the format of (min_snr, max_snr)")
    parser.add_argument("--snr_step",
                        type=float,
                        default=5.0,
                        required=True,
                        help="the step size of snr")
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
    LOADER_CONFIG_PATH: str = argv.loader_config_path
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
    return dataset_config, mel_spec_config, reshape_config, pool_config, loader_config
