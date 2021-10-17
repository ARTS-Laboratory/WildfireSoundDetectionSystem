import sys
from argparse import ArgumentParser, Namespace
from functools import partial
from typing import List, Union

import audio_classifier.config.preprocessing.reshape as conf_reshape
import audio_classifier.config.preprocessing.spec as conf_spec
import audio_classifier.train.collate.base as collate_base
import audio_classifier.train.collate.preprocessing.spectrogram.reshape as collate_reshape
import audio_classifier.train.collate.preprocessing.spectrogram.transform as collate_transform
import audio_classifier.train.config.alg as conf_alg
import audio_classifier.train.config.dataset as conf_dataset
import audio_classifier.train.config.loader as conf_loader
import numpy as np
import script.train.common as script_common
from audio_classifier.train.data.dataset.composite import KFoldDatasetGenerator
from script.train.feature_engineering.skm import fit_baseline, fit_common

MetaDataType = script_common.MetaDataType
CollateFuncType = script_common.CollateFuncType


def get_argparse() -> ArgumentParser:
    parser = ArgumentParser(parents=[
        conf_dataset.DatasetConfigArgumentParser(),
        conf_spec.SpecConfigArgumentParser(),
        conf_reshape.ReshapeConfigArgumentParser(),
        conf_alg.SKMArgumentParser(),
        conf_loader.LoaderConfigArgumentParser()
    ])
    parser.add_argument("--export_path",
                        type=str,
                        required=True,
                        help="the export path")
    parser.add_argument("--k_min", type=int, required=True, help="minimum k")
    parser.add_argument("--k_max", type=int, required=True, help="maximum k")
    parser.add_argument("--k_step",
                        type=int,
                        required=True,
                        help="k step size")
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
        DATASET_CONFIG_PATH, argv, conf_dataset.PreSplitFoldDatasetConfig)
    mel_spec_config: conf_spec.MelSpecConfig = conf_spec.get_spec_config_from_json(
        SPEC_CONFIG_PATH, conf_spec.MelSpecConfig)
    reshape_config: conf_reshape.ReshapeConfig = conf_reshape.get_reshape_config_from_json(
        RESHAPE_CONFIG_PATH)
    skm_config: conf_alg.SKMConfig = conf_alg.get_alg_config_from_json(
        SKM_CONFIG_PATH, conf_alg.SKMConfig)
    loader_config: conf_loader.LoaderConfig = conf_loader.get_loader_config_from_json(
        LOADER_CONFIG_PATH)
    return dataset_config, mel_spec_config, reshape_config, skm_config, loader_config


def main(args: List[str]):
    argv: Namespace = parse_args(args)
    export_path: str = argv.export_path
    k_min: int = argv.k_min
    k_max: int = argv.k_max
    k_step: int = argv.k_step
    dataset_config, mel_spec_config, reshape_config, skm_config, loader_config = get_config(
        argv=argv)
    metadata: MetaDataType = script_common.get_metadata(dataset_config)
    dataset_generator: KFoldDatasetGenerator = script_common.get_dataset_generator(
        metadata=metadata,
        dataset_config=dataset_config,
        mel_spec_config=mel_spec_config)
    collate_func: CollateFuncType = collate_base.EnsembleCollateFunction(
        collate_funcs=[
            partial(collate_transform.mel_spectrogram_collate,
                    config=mel_spec_config),
            partial(collate_reshape.slice_flatten_collate,
                    config=reshape_config)
        ])
    for curr_val_fold in range(dataset_config.k_folds):
        train, _ = fit_baseline.generate_slice_dataset(
            curr_val_fold=curr_val_fold,
            dataset_generator=dataset_generator,
            collate_function=collate_func,
            loader_config=loader_config)
        slices, labels = fit_baseline.convert_to_ndarray(slice_dataset=train)
        unique_labels: np.ndarray = np.unique(labels)
        for curr_label in unique_labels:
            curr_class_path: str = fit_common.get_curr_class_path(
                export_path, curr_val_fold, curr_label)
            curr_slices: np.ndarray = fit_common.get_curr_class_slices(
                curr_label, slices, labels)
            k_value: Union[int, None] = fit_common.try_k_elbow(
                curr_class_path=curr_class_path,
                slices=curr_slices,
                skm_config=skm_config,
                k_range=range(k_min, k_max, k_step))
            if k_value is None:
                continue
            skm, model_path = fit_common.fit_skm(
                curr_class_path=curr_class_path,
                slices=curr_slices,
                skm_config=skm_config,
                k_value=k_value)
            is_pass: bool = fit_common.verify_model(slices=slices,
                                                    skm=skm,
                                                    model_path=model_path)
            if is_pass == False:
                print("Discrepency between src model and exported model",
                      file=sys.stderr)
                sys.exit(1)
            fit_common.plot_centroids(curr_plot_path=curr_class_path,
                                      skm=skm,
                                      spec_config=mel_spec_config,
                                      reshape_config=reshape_config,
                                      skm_config=skm_config,
                                      sample_freq=train.sample_freqs[0],
                                      sample_time=train.sample_times[0])
            fit_common.plot_silhouette(curr_plot_path=curr_class_path,
                                       slices=curr_slices,
                                       skm=skm,
                                       k_value=k_value)


if __name__ == "__main__":
    args: List[str] = sys.argv[1:]
    main(args)
