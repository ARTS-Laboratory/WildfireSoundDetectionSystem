#%%
import os
import pickle
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colorbar import Colorbar
from script.fit_skm_pca_svc import FitSkmPcaSvcResult
import sklearn.metrics
#%%
plt.style.use(["science", "ieee"])

#%%
MODEL_DIR_ROOT_PATH: str = "../../../model/binary_fire/pipeline_50_100_150_200/"

# FILENAME: str = "50_100_150_200_aug_02_pca_00_svc_04.pkl"


#%%
def load_result(result_path: str):
    with open(result_path, mode="rb") as result_file:
        result: FitSkmPcaSvcResult = pickle.load(result_file)
        return result


def plot_cfm(cfms: np.ndarray):
    # cfm for confusion matrix
    # average cfm across all folds, round average, and cast to int
    avg_cfm: np.ndarray = np.mean(np.asarray(cfms), axis=0)
    avg_cfm = np.round(avg_cfm).astype(int)
    dsp = sklearn.metrics.ConfusionMatrixDisplay(avg_cfm,
                                                 display_labels=np.asarray(
                                                     ["fire", "forest"]))
    dsp.plot(cmap='Greys', values_format="d", colorbar=False)
    cbar: Colorbar = dsp.figure_.colorbar(dsp.im_, ax=dsp.ax_)
    cbar.set_label("Number of Slices")
    dsp.ax_.set_xlabel("Prediction")
    dsp.ax_.set_ylabel("Truth")
    # dsp.figure_.
    plt.tight_layout()
    return dsp.figure_, dsp.ax_


def compute_stats(cfms: np.ndarray):
    accs: List[float] = list()
    precisions: List[float] = list()
    recalls: List[float] = list()
    for cfm in cfms:
        acc: float = np.sum(np.diag(cfm)) / np.sum(cfm)
        precision: float = cfm[0, 0] / (cfm[0, 0] + cfm[1, 0])
        recall: float = cfm[0, 0] / (cfm[0, 0] + cfm[0, 1])
        accs.append(acc)
        precisions.append(precision)
        recalls.append(recall)
    return accs, precisions, recalls


def print_test_summary(accs: List[float], precisions: List[float],
                       recalls: List[float]):
    print(
        str.format("acc_mean: {:.3f} acc_std: {:.3f}", np.mean(accs),
                   np.std(accs)))
    print(
        str.format("precision_mean: {:.3f} precision_std: {:.3f}",
                   np.mean(precisions), np.std(precisions)))
    print(
        str.format("recall_mean: {:.3f} recall_std: {:.3f}", np.mean(recalls),
                   np.std(recalls)))


def print_train_val_summary(train_accs: List[float], val_accs: List[float]):
    print(
        str.format("train_acc_mean: {:.3f} train_acc_std: {:.3f}",
                   np.mean(np.asarray(train_accs)),
                   np.std(np.asarray(train_accs))))
    print(
        str.format("val_acc_mean: {:.3f} val_acc_std: {:.3f}",
                   np.mean(np.asarray(val_accs)),
                   np.std(np.asarray(val_accs))))
    pass


#%%
# result_path: str = os.path.join(MODEL_DIR_ROOT_PATH, FILENAME)
# result: FitSkmPcaSvcResult = load_result(result_path)

#%%
filenames = os.listdir(MODEL_DIR_ROOT_PATH)
filenames = list(
    filter(
        lambda fname: os.path.isfile(os.path.join(MODEL_DIR_ROOT_PATH, fname)),
        filenames))
filenames = list(
    filter(lambda fname: os.path.splitext(fname)[1] == ".pkl", filenames))
for filename in filenames:
    result_path: str = os.path.join(MODEL_DIR_ROOT_PATH, filename)
    result: FitSkmPcaSvcResult = load_result(result_path)
    print("========================================================")
    print(os.path.splitext(filename)[0])
    export_dir_path: str = os.path.join(MODEL_DIR_ROOT_PATH,
                                        os.path.splitext(filename)[0])
    os.makedirs(export_dir_path, exist_ok=True)
    for k_val, cfms in zip(result.k_vals, result.confusion_mats):
        cfm_fig, cfm_ax = plot_cfm(np.asarray(cfms))
        cfm_fig_path: str = os.path.join(export_dir_path,
                                         str.format("{}.png", k_val))
        cfm_fig.savefig(cfm_fig_path, dpi=300)
        accs, precisions, recalls = compute_stats(np.asarray(cfms))
        print(str.format("k_val {}", k_val))
        # print_train_val_summary(train_accs, val_accs)
        print_test_summary(accs, precisions, recalls)
        plt.close("all")
    print("========================================================")

#%%
# print("=============================")
# print(os.path.splitext(FILENAME)[0])
# export_dir_path: str = os.path.join(MODEL_DIR_ROOT_PATH,
#                                     os.path.splitext(FILENAME)[0])
# os.makedirs(export_dir_path, exist_ok=True)
# for k_val, cfms in zip(result.k_vals, result.confusion_mats):
#     cfm_fig, cfm_ax = plot_cfm(np.asarray(cfms))
#     cfm_fig_path: str = os.path.join(export_dir_path,
#                                      str.format("{}.png", k_val))
#     cfm_fig.savefig(cfm_fig_path, dpi=300)
#     accs, precisions, recalls = compute_stats(np.asarray(cfms))
#     print(str.format("k_val {}", k_val))
#     print_summary(accs, precisions, recalls)
#     plt.close("all")
# print("=============================")

#%%
# #%%
# def plot_acc_vs_n_samples(result: BiasVarianceResultBase):
#     fig, ax = plt.subplots()
#     ax.plot(np.linspace(588, 588 * len(result.train_accs),
#                         len(result.train_accs)),
#             result.train_accs,
#             label="Train")
#     ax.plot(np.linspace(588, 588 * len(result.val_accs), len(result.val_accs)),
#             result.val_accs,
#             label="Validation")
#     ax.plot(np.linspace(588, 588 * len(result.test_accs),
#                         len(result.test_accs)),
#             result.test_accs,
#             label="Test")
#     ax.set_ylabel("Accuracy")
#     ax.set_xlabel("Number of Training Samples")
#     ax.legend(loc="center left", bbox_to_anchor=(1.04, 0.5))
#     fig.set_size_inches(fig.get_size_inches()[0] * 2, fig.get_size_inches()[1])
#     plt.tight_layout()
#     # plt.show()
#     return fig, ax

# def plot_elbow(result: BiasVarianceResult):
#     plots: List[List[Tuple[Figure, Axes]]] = list()
#     n_clusters: np.ndarray = np.linspace(result.k_min, result.k_max,
#                                          (result.k_max - result.k_min) //
#                                          result.k_step)
#     for curr_fold_k_vals, curr_fold_k_scores in zip(result.k_vals,
#                                                     result.k_scores):
#         curr_plots: List[Tuple[Figure, Axes]] = list()
#         plots.append(curr_plots)
#         for curr_class_k_val, curr_class_k_scores in zip(
#                 curr_fold_k_vals, curr_fold_k_scores):
#             fig, ax = plt.subplots()
#             ax.plot(n_clusters, curr_class_k_scores, label="Distortion Score")
#             ax.axvline(curr_class_k_val,
#                        linewidth=0.5,
#                        linestyle='--',
#                        label="Elbow Value")
#             ax.legend(loc='best', fontsize='medium')
#             curr_plots.append((fig, ax))
#     return plots

# #%%
# # plot accuracy
# filenames: List[str] = os.listdir(METRICS_DIR_PATH)
# filenames = list(
#     filter(lambda fname: os.path.splitext(fname)[1] == ".pkl", filenames))
# for filename in filenames:
#     result_path: str = os.path.join(METRICS_DIR_PATH, filename)
#     result = load_result(result_path)
#     # elbow plot
#     plots: List[List[Tuple[Figure, Axes]]] = plot_elbow(result)
#     for curr_label, curr_class_plots in enumerate(plots[-2]):
#         elbow_fig_filename: str = str.format(
#             "elbow_{}_", curr_label) + os.path.splitext(filename)[0] + ".png"
#         elbow_fig_path: str = os.path.join(METRICS_DIR_PATH,
#                                            elbow_fig_filename)
#         curr_class_plots[0].savefig(elbow_fig_path, dpi=300)
#     # accuracy vs number of data points
#     acc_fig, acc_ax = plot_acc_vs_n_samples(result)
#     acc_fig_filename: str = "acc_" + os.path.splitext(filename)[0] + ".png"
#     acc_fig_path: str = os.path.join(METRICS_DIR_PATH, acc_fig_filename)
#     acc_fig.savefig(acc_fig_path, dpi=300)
# plt.close("all")

# # %%
