from functools import partial
from os import path
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import audio_classifier.config.preprocessing.spec as conf_spec
import audio_classifier.train.config.dataset as conf_dataset
import audio_classifier.train.data.dataset.base as dataset_base
import audio_classifier.train.data.dataset.composite as dataset_composite
import audio_classifier.train.data.metadata.query as metadata_query
import audio_classifier.train.data.metadata.reader as metadata_reader
from torch.utils.data import DataLoader

MetaDataType = Sequence[Dict[str, str]]
CollateFuncType = Callable[[Union[Sequence[Tuple], Sequence]],
                           Sequence[Sequence]]


def get_metadata(
        dataset_config: conf_dataset.PreSplitFoldDatasetConfig
) -> MetaDataType:
    metadata = metadata_reader.read_csv_metadata(dataset_config.metadata_path)
    return metadata


def get_dataset_generator(
    metadata: MetaDataType,
    dataset_config: conf_dataset.PreSplitFoldDatasetConfig,
    mel_spec_config: conf_spec.MelSpecConfig
) -> dataset_composite.KFoldDatasetGenerator:
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
        collate_function: CollateFuncType,
        num_workers: int = 1):
    datasets = dataset_generator.get_train_val_dataset(
        curr_val_fold=curr_val_fold)
    for dataset in datasets:
        loader = DataLoader(dataset=dataset,
                            collate_fn=collate_function,
                            num_workers=num_workers)
        for batch in loader:
            pass


def _sub_dataset_generator(
        curr_fold: int,
        dataset_config: conf_dataset.PreSplitFoldDatasetConfig,
        spec_config: conf_spec.STFTSpecConfig,
        metadata: Optional[MetaDataType] = None) -> dataset_base.FolderDataset:
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
