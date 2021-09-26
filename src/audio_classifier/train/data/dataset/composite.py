from typing import Callable, List, Tuple

from torch.utils.data import ConcatDataset, Dataset


class KFoldDatasetGenerator:

    __k_folds: int
    __all_fold_datasets: List[Dataset]

    def __init__(self, k_folds: int,
                 sub_dataset_generator: Callable[[int], Dataset]) -> None:
        """Constructor for KFoldDatasetGenerator

        The validation fold number is zero indexed.

        Args:
            k_folds (int): The total number of validation folds in the dataset.
            sub_dataset_generator (Callable[[int], Dataset]): The function that takes in the val_fold number and returns the corresponding `Dataset`.
        """
        self.__k_folds = k_folds

        for k in range(self.__k_folds):
            dataset: Dataset = sub_dataset_generator(k)
            self.__all_fold_datasets.append(dataset)

    def get_train_val_dataset(self,
                              curr_val_fold: int) -> Tuple[Dataset, Dataset]:
        """Get the current train and validation dataset.

        The validation fold number is zero indexed.

        Args:
            curr_val_fold (int): The zero indexed current validation fold number.

        Returns:
            train_dataset (Dataset): The current training dataset.
            val-dataset (Dataset): The current validation dataset.

        Raises:
            ValueError: If `curr_val_fold` is bigger than `self.__k_folds`.
        """
        if curr_val_fold >= self.__k_folds:
            raise ValueError(
                str.format("curr_val_fols {} is bigger than self.__k_folds {}",
                           curr_val_fold, self.__k_folds))
        train_dataset_list = [
            dataset for i, dataset in self.__all_fold_datasets
            if i != curr_val_fold
        ]
        train_dataset: Dataset = ConcatDataset(train_dataset_list)
        val_dataset: Dataset = self.__all_fold_datasets[curr_val_fold]
        return train_dataset, val_dataset
