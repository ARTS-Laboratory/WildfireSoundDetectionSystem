#%%
import os
import pickle
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from script.validate.bias_variance import (BiasVarianceResult,
                                           BiasVarianceResultBase)

#%%
plt.style.use(["science", "ieee"])

#%%
METRICS_DIR_PATH: str = "../../../metrics/binary_fire/spec_00_reshape_00_skm_00/bias_variance/svc_01"
DATA_PER_FOLD: int = 588


#%%
def load_result(result_path: str):
    with open(result_path, mode="rb") as result_file:
        result: BiasVarianceResult = pickle.load(result_file)
        return result


#%%
def plot_acc_vs_n_samples(result: BiasVarianceResultBase):
    fig, ax = plt.subplots()
    ax.plot(np.linspace(588, 588 * len(result.train_accs),
                        len(result.train_accs)),
            result.train_accs,
            label="Train")
    ax.plot(np.linspace(588, 588 * len(result.val_accs), len(result.val_accs)),
            result.val_accs,
            label="Validation")
    ax.plot(np.linspace(588, 588 * len(result.test_accs),
                        len(result.test_accs)),
            result.test_accs,
            label="Test")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Number of Training Samples")
    ax.legend(loc="center left", bbox_to_anchor=(1.04, 0.5))
    fig.set_size_inches(fig.get_size_inches()[0] * 2, fig.get_size_inches()[1])
    plt.tight_layout()
    # plt.show()
    return fig, ax


def plot_elbow(result: BiasVarianceResult):
    plots: List[List[Tuple[Figure, Axes]]] = list()
    n_clusters: np.ndarray = np.linspace(result.k_min, result.k_max,
                                         (result.k_max - result.k_min) //
                                         result.k_step)
    for curr_fold_k_vals, curr_fold_k_scores in zip(result.k_vals,
                                                    result.k_scores):
        curr_plots: List[Tuple[Figure, Axes]] = list()
        plots.append(curr_plots)
        for curr_class_k_val, curr_class_k_scores in zip(
                curr_fold_k_vals, curr_fold_k_scores):
            fig, ax = plt.subplots()
            ax.plot(n_clusters, curr_class_k_scores, label="Distortion Score")
            ax.axvline(curr_class_k_val,
                       linewidth=0.5,
                       linestyle='--',
                       label="Elbow Value")
            ax.set_ylabel("Distortion Score")
            ax.set_xlabel("Number of Clusters (k)")
            ax.legend(loc='best', fontsize='medium')
            curr_plots.append((fig, ax))
    return plots


#%%
# plot accuracy
filenames: List[str] = os.listdir(METRICS_DIR_PATH)
filenames = list(
    filter(lambda fname: os.path.splitext(fname)[1] == ".pkl", filenames))
for filename in filenames:
    result_path: str = os.path.join(METRICS_DIR_PATH, filename)
    result = load_result(result_path)
    # elbow plot
    plots: List[List[Tuple[Figure, Axes]]] = plot_elbow(result)
    for curr_label, curr_class_plots in enumerate(plots[-2]):
        elbow_fig_filename: str = str.format(
            "elbow_{}_", curr_label) + os.path.splitext(filename)[0] + ".png"
        elbow_fig_path: str = os.path.join(METRICS_DIR_PATH,
                                           elbow_fig_filename)
        curr_class_plots[0].savefig(elbow_fig_path, dpi=300)
    # accuracy vs number of data points
    acc_fig, acc_ax = plot_acc_vs_n_samples(result)
    acc_fig_filename: str = "acc_" + os.path.splitext(filename)[0] + ".png"
    acc_fig_path: str = os.path.join(METRICS_DIR_PATH, acc_fig_filename)
    acc_fig.savefig(acc_fig_path, dpi=300)
plt.close("all")

# %%
