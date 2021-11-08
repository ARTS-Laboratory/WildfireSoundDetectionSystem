from dataclasses import dataclass, field

import numpy as np


@dataclass
class ClassificationMetrics:
    accuracy: float = field()
    roc_auc: float = field()
    confusion_matrix: np.ndarray = field(repr=False)
