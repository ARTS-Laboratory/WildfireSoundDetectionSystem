#%%
from os import path
from typing import Any, Dict, List

import numpy as np

import audio_classifier.train.data.dataset.base as dataset_base
import audio_classifier.train.data.metadata.query as metadata_query
import audio_classifier.train.data.metadata.reader as metadata_reader
import librosa.core as rosa_core
from torch.utils.data import DataLoader, get_worker_info

#%%
PATH_TO_METADATA: str = "../../test_dataset/metadata.csv"
PATH_TO_FOLDER_DATASET: str = "../../test_dataset/folder_dataset"

#%%
metadata: List[Dict[str, str]] = metadata_reader.read_csv_metadata(
    path_to_metadata=PATH_TO_METADATA)

#%%
query = metadata_query.DictMetaDataQuery(metadata=metadata,
                                         filename_key="slice_file_name",
                                         label_key="classID")

#%%
print(query("529_1.wav"))

# #%%
dataset = dataset_base.FolderDataset(path_to_folder=PATH_TO_FOLDER_DATASET,
                                     sample_rate=44100,
                                     filename_to_label_func=query,
                                     cache=True)


#%%
def verify_collate_function(data):
    print(get_worker_info())
    n_items: int = len(data[0])
    ret_data: List[List[Any]] = [list() for _ in range(0, n_items + 1)]
    for data_point in data:
        print(type(data_point))
        for j, item in enumerate(data_point):
            ret_data[j].append(item)
        path_to_file: str = path.join(PATH_TO_FOLDER_DATASET, data_point[0])
        sound_wave_gt, _ = rosa_core.load(path=path_to_file,
                                          sr=44100,
                                          mono=True)
        is_pass: bool = np.array_equal(sound_wave_gt, data_point[1])
        ret_data[-1].append(is_pass)
    return ret_data


#%%
loader = DataLoader(dataset=dataset,
                    batch_size=2,
                    num_workers=3,
                    collate_fn=verify_collate_function)
for filename, sound_wave, label, is_pass in loader:
    print(
        str.format("{} {} {} {}", type(filename), type(sound_wave),
                   type(label), np.all(is_pass)))

#%%
from audio_classifier.train.data.collate import base

loader = DataLoader(dataset=dataset,
                    batch_size=2,
                    num_workers=3,
                    collate_fn=base.identity_collate_function)
for filename, sound_wave, label in loader:
    print(
        str.format("{} {} {}", type(filename), type(sound_wave),
                   type(label)))
# %%
