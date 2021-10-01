from collections import deque
from functools import partial
from os import path
from typing import Any, Callable, Deque, Dict, Optional, Sequence, Tuple, Union

import audio_classifier.config.preprocessing.spec as conf_spec
import audio_classifier.train.config.dataset as conf_dataset
import audio_classifier.train.config.loader as conf_loader
import audio_classifier.train.data.dataset.base as dataset_base
import audio_classifier.train.data.dataset.composite as dataset_composite
import audio_classifier.train.data.dataset.utils.batch as batch_utils
import audio_classifier.train.data.metadata.query as metadata_query
import audio_classifier.train.data.metadata.reader as metadata_reader
from torch.utils.data import DataLoader

MetaDataType = Sequence[Dict[str, str]]
CollateFuncType = Callable[[Union[Sequence[Tuple], Sequence]],
                           Sequence[Sequence]]


def get_metadata(
        dataset_config: conf_dataset.PreSplitFoldDatasetConfig
) -> MetaDataType:
    """Read metadata from the path specified in dataset_config.

    Args:
        dataset_config (conf_dataset.PreSplitFoldDatasetConfig): The dataset_config to be used.

    Returns:
        MetaDataType: The loaded metadata.
    """
    metadata = metadata_reader.read_csv_metadata(dataset_config.metadata_path)
    return metadata


def get_dataset_generator(
    metadata: MetaDataType,
    dataset_config: conf_dataset.PreSplitFoldDatasetConfig,
    mel_spec_config: conf_spec.MelSpecConfig
) -> dataset_composite.KFoldDatasetGenerator:
    """Create dataset generator.

    Args:
        metadata (MetaDataType): The metadata correspond the dataset.
        dataset_config (conf_dataset.PreSplitFoldDatasetConfig): The dataset config to be used.
        mel_spec_config (conf_spec.MelSpecConfig): The mel-spectrogram config to be used.

    Returns:
        dataset_composite.KFoldDatasetGenerator: The dataset generator.
    """
    dataset_generator = dataset_composite.KFoldDatasetGenerator(
        k_folds=dataset_config.k_folds,
        sub_dataset_generator=partial(_sub_dataset_generator,
                                      dataset_config=dataset_config,
                                      spec_config=mel_spec_config,
                                      metadata=metadata))
    return dataset_generator


def generate_dataset(
    curr_val_fold: int,
    dataset_generator: dataset_composite.KFoldDatasetGenerator,
    collate_function: CollateFuncType, loader_config: conf_loader.LoaderConfig
) -> Sequence[Sequence[Sequence[Any]]]:
    """Generate the train and validation dataset with the dataset_generator

    Args:
        curr_val_fold (int): The fold number to be used as validation dataset. Zero-indexed.
        dataset_generator (dataset_composite.KFoldDatasetGenerator): The dataset generator to be used.
        collate_function (CollateFuncType): The collate function used to process the time series data.
        loader_config (conf_loader.LoaderConfig): The loader config used to load the dataset.

    Returns:
        Sequence[Sequence[Sequence[Any]]]: (2, n_items_output, n_data_points)
    """
    datasets = dataset_generator.get_train_val_dataset(
        curr_val_fold=curr_val_fold)
    ret_dataset: Deque[Sequence[Sequence[Any]]] = deque()
    for dataset in datasets:
        loader = DataLoader(dataset=dataset,
                            collate_fn=collate_function,
                            num_workers=loader_config.num_workers,
                            batch_size=loader_config.batch_size)
        batches: Deque[Sequence[Sequence]] = deque()
        for batch in loader:
            batches.append(batch)
        curr_ret_dataset: Sequence[
            Sequence[Any]] = batch_utils.combine_batches(batches)
        ret_dataset.append(curr_ret_dataset)
    return ret_dataset


def _sub_dataset_generator(
        curr_fold: int,
        dataset_config: conf_dataset.PreSplitFoldDatasetConfig,
        spec_config: conf_spec.STFTSpecConfig,
        metadata: Optional[MetaDataType] = None) -> dataset_base.FolderDataset:
    """The functor used to generate each sub dataset.

    Args:
        curr_fold (int): The current fold to be generated
        dataset_config (conf_dataset.PreSplitFoldDatasetConfig): The dataset configuration to be used.
        spec_config (conf_spec.STFTSpecConfig): The spectrogram configuration to be used.
        metadata (Optional[MetaDataType], optional): The metadata used to map filename to class lable. If not provided, one will be loaded. Defaults to None.

    Returns:
        dataset_base.FolderDataset: The current fold dataset.
    """
    folder_path: str = path.join(dataset_config.root_path,
                                 dataset_config.fold_path_stub)
    folder_path = str.format(folder_path, curr_fold + 1)
    metadata = get_metadata(dataset_config) if metadata is None else metadata
    metadata_querier = metadata_query.DictMetaDataQuerier(
        metadata=metadata,
        filename_key=dataset_config.filename_key,
        label_key=dataset_config.label_key)
    folder_dataset = dataset_base.FolderDataset(
        folder_path=folder_path,
        sample_rate=spec_config.sample_rate,
        filename_to_label_func=metadata_querier)
    return folder_dataset
