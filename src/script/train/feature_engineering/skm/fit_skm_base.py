import os
import pickle
import traceback
from os import path
from typing import List, Optional, Tuple, Union

import audio_classifier.config.preprocessing.reshape as conf_reshape
import audio_classifier.config.preprocessing.spec as conf_spec
import audio_classifier.train.config.alg as conf_alg
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn_plugins.cluster.spherical_kmeans import SphericalKMeans
from yellowbrick_plugins.cluster import (SphericalKElbowVisualizer,
                                         SphericalSilhouetteVisualizer)

from ... import train_common

matplotlib.use("agg", force=True)

MetaDataType = train_common.MetaDataType
CollateFuncType = train_common.CollateFuncType


def get_curr_class_slices(curr_class: int, slices: np.ndarray,
                          labels: np.ndarray) -> np.ndarray:
    """Get current class slices.

    Args:
        curr_class (int): The current class label.
        slices (np.ndarray): (n_slices, n_dims) All the slices in the dataset.
        labels (np.ndarray): (n_slices, ) All the class labels.

    Returns:
        curr_class_slices (np.ndarray): (n_slices_curr_class, n_dims) Slices of current class.
    """
    curr_indices: np.ndarray = np.argwhere(labels == curr_class).flatten()
    curr_slices: np.ndarray = slices[curr_indices, :]
    return curr_slices


def get_curr_class_path(export_path: str, curr_val_fold: int,
                        curr_class: int) -> str:
    """Get and create curr_class_path

    Args:
        export_path (str): The path to current class folder.
        curr_val_fold (int): The current validation fold number.
        curr_class (int): The current number.

    Returns:
        str: The path to current class path.
    """
    curr_class_path: str = path.join(export_path,
                                     str.format("val_{:02d}", curr_val_fold),
                                     str.format("class_{:02d}", curr_class))
    os.makedirs(curr_class_path, exist_ok=True)
    return curr_class_path


def try_k_elbow(curr_class_path: str,
                slices: np.ndarray,
                skm_config: conf_alg.SKMConfig,
                k_range: range,
                metric_filename_stub: str = "elbow_{}") -> Union[int, None]:
    """Search for optimal k value.

    Args:
        curr_class_path (str): path to the root to contain metrics plot
        slices (np.ndarray): (n_slices, n_sample_freq * slice_size). Class conditioned feature vectors dataset.
        feature_vector_config (FeatureVectorConfig): feature vector configuration instance.
        k_test_range (range): range object used to generate k value to be tested
        metric_filename_stub (str): the filename stub for the plot and metrics, must include a placeholder for current k value. Defaults to "elbow_{}".

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
    # plot elbow
    visualizer.fit(slices)
    visualizer.finalize()
    plot_path: str = path.join(
        curr_class_path, str.format(metric_filename_stub + ".png", METRIC))
    figure.savefig(fname=plot_path, dpi=300)
    plt.close(fig=figure)
    # serialize data used for plotting
    metric_path: str = path.join(
        curr_class_path, str.format(metric_filename_stub + ".pkl", METRIC))
    with open(metric_path, mode="wb") as metric_file:
        metrics: Tuple[Optional[int], List[int], List[float],
                       List[float]] = (visualizer.elbow_value_,
                                       visualizer.k_values_,
                                       visualizer.k_timers_,
                                       visualizer.k_scores_)
        pickle.dump(metrics, metric_file)
    return visualizer.elbow_value_


def fit_skm(curr_class_path: str,
            slices: np.ndarray,
            skm_config: conf_alg.SKMConfig,
            k_value: int,
            filename: str = "model.pkl") -> Tuple[SphericalKMeans, str]:
    """Train and serialize SphericalKMeans.

    Args:
        model_filename (str): The filename including the path the trained model is serialzied to.
        slices (np.ndarray): (n_slices, n_sample_freq * slice_size). Class conditioned feature vectors dataset.
        feature_vector_config (FeatureVectorConfig): Feature vector configuration instance.
        k_value (int): the k value used to train SphericalKMeans.

    Returns:
        skm (SphericalKMeans): The fitted skm model.
        file_path (str): The path to exported model.
    """
    model_path: str = path.join(curr_class_path, filename)
    skm = SphericalKMeans(n_clusters=k_value,
                          n_components=skm_config.n_components,
                          normalize=skm_config.normalize,
                          standardize=skm_config.standardize,
                          whiten=skm_config.whiten,
                          copy=True,
                          max_iter=10000)
    skm.fit(slices)
    with open(model_path, "wb") as file:
        pickle.dump(skm, file)
    return skm, model_path


def verify_model(slices: np.ndarray, skm: SphericalKMeans,
                 model_path: str) -> bool:
    """Verify the model

    Args:
        slices (np.ndarray): The slices used to verify.
        skm (SphericalKMeans): The original skm.
        model_path (str): Path to load the exported skm.

    Returns:
        is_pass (bool): Whether a model passed the consistency test or not.
    """
    with open(model_path, "rb") as file:
        skm_prime: SphericalKMeans = pickle.load(file)
    res: np.ndarray = skm.predict(slices)
    res_prime: np.ndarray = skm_prime.predict(slices)
    is_pass: bool = np.array_equal(res, res_prime)
    return is_pass


def plot_centroids(curr_plot_path: str,
                   skm: SphericalKMeans,
                   spec_config: conf_spec.MelSpecConfig,
                   reshape_config: conf_reshape.ReshapeConfig,
                   skm_config: conf_alg.SKMConfig,
                   sample_freq: np.ndarray,
                   sample_time: np.ndarray,
                   filename_stub: str = "k_{:02d}.png"):
    """Plot centorids

    Args:
        curr_plot_path (str): the path to the directory where all the plot are saved
        skm (SphericalKMeans): A trained SphericalKMeans instance
        spec_config (conf_spec.MelSpecConfig): spectrogram configuration
        reshape_config (conf_reshape.ReshapeConfig): reshape configuration
        skm_config (conf_alg.SKMConfig): spherical k means configuration
        sample_freq (np.ndarray): A sample freq of mel spectrogram.
        sample_time (np.ndarray): A sample time.
        filename_stub (str, optional): The filename stub for the plots. Defaults to "k_{:02d}.png".
    """
    centers: np.ndarray
    if skm.cluster_centers_ is not None:
        centers = skm.cluster_centers_
    else:
        return
    # recover to original spectrogram space
    centers = skm.pca_.inverse_transform(centers)
    if skm_config.standardize and (skm.std_scalar_ is not None):
        centers = skm.std_scalar_.inverse_transform(centers, copy=False)
    n_centers: int = len(centers)
    slice_size: int = reshape_config.slice_size
    n_mels: int = spec_config.n_mels
    centers = np.reshape(a=centers,
                         newshape=(n_centers, n_mels, slice_size),
                         order='F')
    sample_freq = sample_freq
    sample_time = sample_time[0:slice_size]
    curr_plot_root_path: str = path.join(curr_plot_path, "centroids")
    spec_plot_path: str = path.join(curr_plot_root_path, "spec")
    raw_plot_path: str = path.join(curr_plot_root_path, "raw")
    os.makedirs(spec_plot_path, exist_ok=True)
    os.makedirs(raw_plot_path, exist_ok=True)
    for i, center in enumerate(centers):
        if spec_config.apply_log == False:
            center = 10 * np.log10(center)
        fig_filename: str = str.format(filename_stub, i)
        fig_filename = path.join(spec_plot_path, fig_filename)
        plt.pcolormesh(sample_time,
                       sample_freq,
                       center,
                       shading="auto",
                       cmap="magma")
        plt.gca().set_aspect(1e-5)
        plt.savefig(fname=fig_filename, dpi=300)
        plt.close()
        fig_filename = str.format(filename_stub, i)
        fig_filename = path.join(raw_plot_path, fig_filename)
        plt.pcolormesh(center, shading="auto", cmap="gray")
        plt.gca().set_aspect(2.0)
        plt.savefig(fname=fig_filename, dpi=300)
        plt.close()


def plot_silhouette(curr_plot_path: str,
                    slices: np.ndarray,
                    skm: SphericalKMeans,
                    k_value: int,
                    metric_filename_stub: str = "silhouette_k_{}"):
    """Plot Silhouette score for a skm.

    Args:
        curr_plot_path (str): the path to the directory where all the plot are saved
        slices (np.ndarray): The slices used to fit the skm.
        skm (SphericalKMeans): The fitted skm, assumed previously fitted.
        k_value (int): The k value of the skm.
        metric_filename_stub (str, optional): The filename stub of the plot. Defaults to "silhouette_k_{}".
    """
    figure, axes = plt.subplots()
    visualizer = SphericalSilhouetteVisualizer(estimator=skm,
                                               ax=axes,
                                               is_fitted=True)
    visualizer.fit(slices)
    visualizer.finalize()
    fig_path: str = path.join(
        curr_plot_path, str.format(metric_filename_stub + ".png", k_value))
    figure.savefig(fname=fig_path, dpi=300)
    plt.close(fig=figure)
    # TODO serialize silhouette score
