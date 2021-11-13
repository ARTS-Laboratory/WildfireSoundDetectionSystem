#%%
import csv
import pickle
from os import path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from script.train.validate.validate_common import ClassificationMetrics
from sklearn.metrics import ConfusionMatrixDisplay

#%%
PROJECT_ROOT_PATH: str = "../../../"
LABELS: Optional[List[str]] = None
LABELS = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music",
]
# LABELS = [
#     "Dog ",
#     "Rain ",
#     "Crying baby ",
#     "Door knock ",
#     "Helicopter",
#     "Rooster ",
#     "Sea waves ",
#     "Sneezing ",
#     "Mouse click ",
#     "Chainsaw",
#     "Pig ",
#     "Crackling fire ",
#     "Clapping ",
#     "Keyboard typing ",
#     "Siren",
#     "Cow ",
#     "Crickets ",
#     "Breathing ",
#     "Door, wood creaks ",
#     "Car horn",
#     "Frog ",
#     "Chirping birds ",
#     "Coughing ",
#     "Can opening ",
#     "Engine",
#     "Cat ",
#     "Water drops ",
#     "Footsteps ",
#     "Washing machine ",
#     "Train",
#     "Hen ",
#     "Wind ",
#     "Laughing ",
#     "Vacuum cleaner ",
#     "Church bells",
#     "Insects (flying) ",
#     "Pouring water ",
#     "Brushing teeth ",
#     "Clock alarm ",
#     "Airplane",
#     "Sheep ",
#     "Toilet flush ",
#     "Snoring ",
#     "Clock tick ",
#     "Fireworks",
#     "Crow ",
#     "Thunderstorm ",
#     "Drinking, sipping ",
#     "Glass breaking ",
#     "Hand saw",
# ]

#%%
# "metrics/esc_50/spec_00_reshape_00_skm_00/val_pool_02_pca_01_svc_05.pkl"
# "metrics/urban_sound/spec_00_reshape_00_skm_00/val_pool_02_pca_00_svc_05.pkl"
metric_file_path: str = path.join(
    PROJECT_ROOT_PATH,
    "metrics/urban_sound/spec_00_reshape_00_skm_00/val_pool_02_pca_00_svc_05.pkl"
)
# metric_file_path: str = path.join(
#     PROJECT_ROOT_PATH,
#     "metrics/esc_50/spec_00_reshape_00_skm_00/val_pool_02_pca_01_svc_05.pkl")
with open(metric_file_path, mode="rb") as metric_file:
    metrics: Sequence[Tuple[ClassificationMetrics,
                            ClassificationMetrics]] = pickle.load(metric_file)

# %%
train_cm: Sequence[np.ndarray] = list()
val_cm: Sequence[np.ndarray] = list()
for train_metric, val_metric in metrics:
    train_cm.append(train_metric.confusion_matrix)
    val_cm.append(val_metric.confusion_matrix)

#%%
sum_train_cm = np.sum(np.asarray(train_cm), axis=0)
sum_val_cm = np.sum(np.asarray(val_cm), axis=0)


def calculate_accuracy(mat: np.ndarray) -> float:
    acc = np.sum(np.diag(mat)) / np.sum(mat)
    return acc


def calculate_precission(mat: np.ndarray) -> np.ndarray:
    precission = np.diag(mat) / np.sum(mat, axis=0)
    return precission


def calculate_recall(mat: np.ndarray) -> np.ndarray:
    recall = np.diag(mat) / np.sum(mat, axis=1)
    return recall


#%%
sum_train_cm_vis = ConfusionMatrixDisplay(
    sum_train_cm,
    display_labels=[str.replace(l, "_", " ").title() for l in LABELS])
train_acc = calculate_accuracy(sum_train_cm)
train_precision = calculate_precission(sum_train_cm)
train_recall = calculate_recall(sum_train_cm)
sum_train_cm_vis.plot()
if LABELS is not None:
    sum_train_cm_vis.figure_
    for tick in sum_train_cm_vis.ax_.get_xticklabels():
        tick.set_rotation(90)
new_fig_size = sum_train_cm_vis.figure_.get_size_inches() / 10.0 * len(LABELS)
sum_train_cm_vis.figure_.set_size_inches(new_fig_size)

#%%
sum_val_cm_vis = ConfusionMatrixDisplay(
    sum_val_cm,
    display_labels=[str.replace(l, "_", " ").title() for l in LABELS])
val_acc = calculate_accuracy(sum_val_cm)
val_precision = calculate_precission(sum_val_cm)
val_recall = calculate_recall(sum_val_cm)
sum_val_cm_vis.plot()
if LABELS is not None:
    for tick in sum_val_cm_vis.ax_.get_xticklabels():
        tick.set_rotation(90)
new_fig_size = sum_val_cm_vis.figure_.get_size_inches() / 10.0 * len(LABELS)
sum_val_cm_vis.figure_.set_size_inches(new_fig_size)
#%%
cm_plot_path: str = path.join(path.splitext(metric_file_path)[0] + ".png")
sum_val_cm_vis.figure_.savefig(cm_plot_path,
                               dpi=300,
                               transparent=True,
                               pad_inches=0.3)

#%%
header = [str.replace(l, "_", " ").title() for l in LABELS]
header.insert(0, "Metrics")
precision_row = [str(p) for p in val_precision]
precision_row.insert(0, "Precision")
recall_row = [str(r) for r in val_recall]
recall_row.insert(0, "Recall")
metrics_csv_path: str = path.join(path.splitext(metric_file_path)[0] + ".csv")
with open(metrics_csv_path, "w") as metric_csv:
    writer = csv.writer(metric_csv)
    writer.writerow(header)
    writer.writerow(precision_row)
    writer.writerow(recall_row)
#%%