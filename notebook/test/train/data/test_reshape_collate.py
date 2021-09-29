#%%
from functools import partial
from typing import Dict, List

import audio_classifier.train.collate.base as base_collate
import audio_classifier.train.collate.preprocessing.spectrogram.reshape as reshape_collate
import audio_classifier.train.collate.preprocessing.spectrogram.transform as transform_collate
import audio_classifier.train.data.dataset.base as dataset_base
import audio_classifier.train.data.metadata.query as metadata_query
import audio_classifier.train.data.metadata.reader as metadata_reader
import numpy as np
from audio_classifier.common.preprocessing.spectrogram import reshape
from audio_classifier.config.preprocessing.reshape import ReshapeConfig
from audio_classifier.config.preprocessing.spec import MelSpecConfig
from torch.utils.data import DataLoader

#%%
PATH_TO_METADATA: str = "../../test_dataset/metadata.csv"
PATH_TO_FOLDER_DATASET: str = "../../test_dataset/folder_dataset"

#%%
metadata: List[Dict[str, str]] = metadata_reader.read_csv_metadata(
    path_to_metadata=PATH_TO_METADATA)

#%%
query = metadata_query.DictMetaDataQuerier(metadata=metadata,
                                           filename_key="slice_file_name",
                                           label_key="classID")

#%%
mel_config: MelSpecConfig = MelSpecConfig()
reshape_config: ReshapeConfig = ReshapeConfig()
dataset = dataset_base.FolderDataset(folder_path=PATH_TO_FOLDER_DATASET,
                                     sample_rate=mel_config.sample_rate,
                                     filename_to_label_func=query,
                                     cache=True)
#%%
spec_collate = base_collate.EnsembleCollateFunction(collate_funcs=[
    partial(transform_collate.mel_spectrogram_collate, config=mel_config)
])

#%%
loader = DataLoader(dataset=dataset,
                    batch_size=2,
                    num_workers=3,
                    collate_fn=spec_collate)
gt_data: Dict[str, List[np.ndarray]] = dict()
for filenames, mel_specs, mel_freqs, mel_times, labels in loader:
    for filename, mel_spec, mel_freq, mel_time, label in zip(
            filenames, mel_specs, mel_freqs, mel_times, labels):
        slices: List[np.ndarray] = reshape.slice_spectrogram(
            spectrogram=mel_spec,
            slice_size=reshape_config.slice_size,
            stride_size=reshape_config.stride_size,
            copy=True)
        flat_slices: List[np.ndarray] = [
            reshape.flatten_slice(slice=slice, copy=False) for slice in slices
        ]
        gt_data[filename] = flat_slices

# %%
collate_func = base_collate.EnsembleCollateFunction(collate_funcs=[
    partial(transform_collate.mel_spectrogram_collate, config=mel_config),
    partial(reshape_collate.slice_flatten_collate, config=reshape_config)
])
loader = DataLoader(dataset=dataset,
                    batch_size=2,
                    num_workers=3,
                    collate_fn=collate_func)
for filenames, batch_flat_slices, mel_freqs, mel_times, labels in loader:
    for filename, curr_flat_slices, mel_freq, mel_time, label in zip(
            filenames, batch_flat_slices, mel_freqs, mel_times, labels):
        print(np.array_equal(gt_data[filename], curr_flat_slices),
              gt_data[filename] is not curr_flat_slices)
