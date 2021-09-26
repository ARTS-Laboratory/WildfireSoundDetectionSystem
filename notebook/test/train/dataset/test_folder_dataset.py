#%%
from typing import Any, Dict, List

import audio_classifier.train.dataset.base as dataset_base
import audio_classifier.train.dataset.utils.metadata.query as metadata_query
import audio_classifier.train.dataset.utils.metadata.reader as metadata_reader
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
def identity_collate_function(data: List[List]):
    n_items: int = len(data[0])
    ret_data: List[List[Any]] = [list() for _ in range(0, n_items)]
    for data_point in data:
        for j, item in enumerate(data_point):
            ret_data[j].append(item)
    return ret_data


#%%
loader = DataLoader(dataset=dataset,
                    batch_size=2,
                    num_workers=3,
                    collate_fn=identity_collate_function)
for filename, sound_wave, label in loader:
    print(str.format("{} {} {}", type(filename), type(sound_wave),
                     type(label)))