import pickle
from typing import Union
from sklearn.base import ClassifierMixin
from sklearn.pipeline import Pipeline


def load_classifier(classifier_path: str) -> Union[ClassifierMixin, Pipeline]:
    """Load classifier from a given path.

    Args:
        classifier_path (str): The path to the classifier.

    Returns:
        classifier (Union[ClassifierMixin, Pipeline]): The loaded classifier.
    """
    with open(classifier_path, "rb") as classifier_file:
        classifier: Union[ClassifierMixin,
                          Pipeline] = pickle.load(classifier_file)
        return classifier