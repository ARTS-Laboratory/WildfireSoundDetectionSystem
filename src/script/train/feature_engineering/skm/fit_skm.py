import csv
import os
import pickle
import traceback
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, field
from os import path
from typing import List, Sequence, Tuple, Union

import audio_classifier.config.preprocessing.reshape as conf_reshape
import audio_classifier.config.preprocessing.spec as conf_spec
import audio_classifier.train.config.alg as conf_alg
import audio_classifier.train.config.dataset as conf_dataset
import audio_classifier.train.config.loader as conf_loader
import audio_classifier.train.data.dataset.composite as dataset_composite
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import script.train.common as script_common
from sklearn_plugins.cluster.spherical_kmeans import SphericalKMeans
from yellowbrick_plugins.cluster import (SphericalKElbowVisualizer,
                                         SphericalSilhouetteVisualizer)

matplotlib.use("agg", force=True)

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


def get_curr_class_slices(curr_class: int, slices: np.ndarray,
                          labels: np.ndarray) -> np.ndarray:
    curr_indices: np.ndarray = np.argwhere(labels == curr_class)
    curr_slices: np.ndarray = slices[curr_indices]
    return curr_slices


def get_curr_class_path(export_path: str, curr_val_fold: int,
                        curr_class: int) -> str:
    """Get and create curr_class_path

    Args:
        export_path (str): [description]
        curr_val_fold (int): [description]
        curr_class (int): [description]

    Returns:
        str: [description]
    """
    curr_class_path: str = path.join(export_path,
                                     str.format("val_{:02d}", curr_val_fold),
                                     str.format("class_{:02d}", curr_class))
    os.makedirs(curr_class_path)
    return curr_class_path


def try_k_elbow(curr_class_path: str,
                slices: np.ndarray,
                skm_config: conf_alg.SKMConfig,
                k_range: range,
                filename_sub: str = "elbow_{}.png") -> Union[int, None]:
    """Search for optimal k value.

    Args:
        curr_class_path (str): path to the root to contain metrics plot
        slices (np.ndarray): (n_slices, n_sample_freq * slice_size). Class conditioned feature vectors dataset.
        feature_vector_config (FeatureVectorConfig): feature vector configuration instance.
        k_test_range (range): range object used to generate k value to be tested
        filename_stub (str): the filename stub for the plot, must include a placeholder for current k value. Defaults to "elbow_{}.png".

    Returns:
        optimal_k (int): optimal k value found by the algorihtm
    """
    # iterate through all possible metrics to get optimal k value
    METRIC: str = "distortion"
    figure, axes = plt.subplots()
    skm = SphericalKMeans(n_components=skm_config.n_components,
                          normalize=skm_config.normalize,
                          standardize=skm_config.standardize,
                          whiten=skm_config.whiten,
                          copy=True,
                          max_iter=10000)
    try:
        visualizer = SphericalKElbowVisualizer(estimator=skm,
                                               ax=axes,
                                               k=k_range,
                                               metric=METRIC,
                                               locate_elbow=True)
    except np.linalg.LinAlgError:
        traceback.print_exc()
        return None
    visualizer.fit(slices)
    visualizer.finalize()
    fig_filename: str = path.join(curr_class_path,
                                  str.format(filename_sub, METRIC))
    figure.savefig(fname=fig_filename, dpi=300)
    plt.close(fig=figure)
    return visualizer.elbow_value_


# def log_history_csv(hist: np.ndarray,
#                     log_path: str,
#                     filename: str = "history.csv") -> None:
#     """Log history matrix to csv file with the last 2 row being the mean and median across all folds.

#     Args:
#         hist (np.ndarray): (n_folds, n_class) the optimal k identified.
#         log_path (str): the path to log the file
#         filename (str, optional): the filename of the csv file. Defaults to "history.csv".
#     """
#     with open(path.join(log_path, filename), mode="w") as hist_file:
#         hist_writer = csv.writer(hist_file)
#         hist_writer.writerows(hist)
#         hist_writer.writerow(np.nanmean(a=hist, axis=0))
#         hist_writer.writerow(np.nanstd(a=hist[0:-1, :], axis=0))
#         hist_writer.writerow(np.nanmedian(a=hist[0:-2, :], axis=0))


def fit_skm(curr_class_path: str,
            slices: np.ndarray,
            skm_config: conf_alg.SKMConfig,
            k_value: int,
            filename: str = "model.pkl") -> SphericalKMeans:
    """Train and serialize SphericalKMeans.

    Args:
        model_filename (str): The filename including the path the trained model is serialzied to.
        slices (np.ndarray): (n_slices, n_sample_freq * slice_size). Class conditioned feature vectors dataset.
        feature_vector_config (FeatureVectorConfig): Feature vector configuration instance.
        k_value (int): the k value used to train SphericalKMeans.

    Returns:
        SphericalKMeans: [description]
    """
    file_path: str = path.join(curr_class_path, filename)
    skm = SphericalKMeans(n_clusters=k_value,
                          n_components=skm_config.n_components,
                          normalize=skm_config.normalize,
                          standardize=skm_config.standardize,
                          whiten=skm_config.whiten,
                          copy=True,
                          max_iter=10000)
    skm.fit(slices)
    with open(file_path, "wb") as file:
        pickle.dump(skm, file)
    return skm


def plot_silhouette(curr_class_path: str,
                    slices: np.ndarray,
                    skm: SphericalKMeans,
                    k_value: int,
                    filename_stub: str = "silhouette_k_{}.png"):
    figure, axes = plt.subplots()
    visualizer = SphericalSilhouetteVisualizer(estimator=skm,
                                               ax=axes,
                                               is_fitted=True)
    visualizer.fit(slices)
    visualizer.finalize()
    fig_path: str = path.join(curr_class_path,
                              str.format(filename_stub, k_value))
    figure.savefig(fname=fig_path, dpi=300)
    plt.close(fig=figure)
