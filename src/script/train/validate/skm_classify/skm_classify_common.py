from typing import List, Sequence, Union

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline

from ...classification import classify_common
from .. import validate_common


def calculate_metrics(
    classifier: Union[ClassifierMixin, Pipeline],
    dataset: classify_common.ProjDataset,
) -> validate_common.ClassificationMetrics:
    """Calculate the metrics given the dataset and prediction of all slices of all files.

    Args:
        classifier (Union[ClassifierMixin, Pipeline]): The classifier to be evaluated.
        dataset (classify_common.ProjDataset): The projection dataset

    Returns:
        validate_common.ClassificationMetrics: The validation classification metrics.
    """
    pred_labels: Sequence[int] = list()
    pred_probs: Sequence[np.ndarray] = list()
    n_classes: int = max(dataset.labels) + 1
    for spec_projs in dataset.all_file_spec_projs:
        curr_slices_preds: np.ndarray = classifier.predict(
            X=np.asarray(spec_projs))
        pred_label_count: np.ndarray = np.bincount(curr_slices_preds,
                                                   minlength=n_classes)
        pred_label: int = np.argmax(pred_label_count)
        pred_labels.append(pred_label)
        if n_classes > 2:
            pred_prob: np.ndarray = pred_label_count / np.sum(pred_label_count)
            pred_probs.append(pred_prob)
    metrics = validate_common.ClassificationMetrics(
        accuracy=accuracy_score(y_true=dataset.labels, y_pred=pred_labels),
        roc_auc=roc_auc_score(
            y_true=dataset.labels,
            y_score=pred_labels if n_classes == 2 else pred_probs,
            multi_class="ovo"),
        confusion_matrix=confusion_matrix(y_true=dataset.labels,
                                          y_pred=pred_labels))
    return metrics