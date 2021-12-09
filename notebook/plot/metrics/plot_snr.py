#%%
import csv
from math import floor
import pickle
from os import path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from script.train.validate.validate_common import ClassificationMetrics

#%%
PROJECT_ROOT_PATH: str = "../../../"
LABELS: Optional[List[str]] = None
LABELS = ["fire", "forest"]
SNR_RANGE = (-5.0, 5.0)
SNR_STEP = 0.5
#%%
metric_file_path: str = path.join(
    PROJECT_ROOT_PATH,
    "metrics/binary_fire/spec_00_reshape_00_skm_00/snr_augment_01_pool_02_pca_00_svc_04.pkl"
)
# metric_file_path: str = path.join(
#     PROJECT_ROOT_PATH,
#     "metrics/binary_fire/spec_00_reshape_00_skm_00/snr_pool_02_pca_00_svc_04.pkl"
# )

with open(metric_file_path, mode="rb") as metric_file:
    metrics: Sequence[Sequence[Tuple[ClassificationMetrics,
                                     ClassificationMetrics]]] = pickle.load(
                                         metric_file)

#%%
train_conf: List[List[np.ndarray]] = list()
train_acc: List[List[float]] = list()
train_auc: List[List[float]] = list()
val_conf: List[List[np.ndarray]] = list()
val_acc: List[List[float]] = list()
val_auc: List[List[float]] = list()
for curr_fold, fold_metrics in enumerate(metrics):
    train_conf.append(list())
    train_acc.append(list())
    train_auc.append(list())
    val_conf.append(list())
    val_acc.append(list())
    val_auc.append(list())
    for train_m, val_m in fold_metrics:
        train_conf[curr_fold].append(train_m.confusion_matrix)
        train_acc[curr_fold].append(train_m.accuracy)
        train_auc[curr_fold].append(train_m.roc_auc)
        val_conf[curr_fold].append(val_m.confusion_matrix)
        val_acc[curr_fold].append(val_m.accuracy)
        val_auc[curr_fold].append(val_m.roc_auc)

#%%
train_acc_avg: np.ndarray = np.mean(np.asarray(train_acc), axis=0)
train_auc_avg: np.ndarray = np.mean(np.asarray(train_auc), axis=0)

SNR_AXIS = np.linspace(SNR_RANGE[0], SNR_RANGE[1],
                       floor((SNR_RANGE[1] - SNR_RANGE[0]) // SNR_STEP) + 1)
#%%
plt.plot(SNR_AXIS, train_acc_avg[1:-1], 'k', label="SNR")
plt.axhline(train_acc_avg[0],
            color='black',
            linestyle="--",
            linewidth=1,
            label="No SNR")
plt.legend()
ax = plt.gca()
ax.get_yaxis().get_major_formatter().set_useOffset(False)
# plt.xlim(SNR_RANGE[0], SNR_RANGE[1])
plt.xticks(SNR_AXIS, rotation=90)
plt.show()
#%%
plt.plot(SNR_AXIS, train_auc_avg[1:-1], 'k', label="SNR")
plt.axhline(train_auc_avg[0],
            color='black',
            linestyle="--",
            linewidth=1,
            label="No SNR")
plt.legend()
ax = plt.gca()
ax.get_yaxis().get_major_formatter().set_useOffset(False)
plt.xticks(SNR_AXIS, rotation=90)
plt.show()

#%%
val_acc_avg: np.ndarray = np.mean(np.asarray(val_acc), axis=0)
val_auc_avg: np.ndarray = np.mean(np.asarray(val_auc), axis=0)

SNR_AXIS = np.linspace(SNR_RANGE[0], SNR_RANGE[1],
                       floor((SNR_RANGE[1] - SNR_RANGE[0]) // SNR_STEP) + 1)

#%%
val_acc_fig_path: str = path.join(
    path.splitext(metric_file_path)[0] + "_acc.png")
plt.plot(SNR_AXIS, val_acc_avg[1:-1], 'k', label="SNR")
plt.axhline(val_acc_avg[0],
            color='black',
            linestyle="--",
            linewidth=1,
            label="No SNR")
plt.xlabel("SNR")
plt.ylabel("Accuracy")
plt.legend()
ax = plt.gca()
ax.get_yaxis().get_major_formatter().set_useOffset(False)
# plt.xlim(SNR_RANGE[0], SNR_RANGE[1])
plt.xticks(SNR_AXIS, rotation=90)
plt.tight_layout()
plt.savefig(val_acc_fig_path, dpi=300)

plt.show()

#%%
val_roc_auc_fig_path: str = path.join(
    path.splitext(metric_file_path)[0] + "_roc_auc.png")
plt.plot(SNR_AXIS, val_auc_avg[1:-1], 'k', label="SNR")
plt.axhline(val_auc_avg[0],
            color='black',
            linestyle="--",
            linewidth=1,
            label="No SNR")
plt.xlabel("SNR")
plt.ylabel("ROC AUC")
plt.legend()
ax = plt.gca()
ax.get_yaxis().get_major_formatter().set_useOffset(False)
plt.xticks(SNR_AXIS, rotation=90)
plt.tight_layout()
plt.savefig(val_roc_auc_fig_path, dpi=300)
plt.show()


# %%
