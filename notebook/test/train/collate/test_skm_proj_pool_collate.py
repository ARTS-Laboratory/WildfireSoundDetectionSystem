#%%
from functools import partial
from typing import Dict, List, Sequence

import audio_classifier.common.feature_engineering.pool as feature_pool
import audio_classifier.train.collate.base as base_collate
import audio_classifier.train.collate.feature_engineering.skm as collate_skm
import audio_classifier.train.collate.preprocessing.spectrogram.reshape as reshape_collate
import audio_classifier.train.collate.preprocessing.spectrogram.transform as transform_collate
import audio_classifier.train.data.dataset.base as dataset_base
import audio_classifier.train.data.metadata.query as metadata_query
import audio_classifier.train.data.metadata.reader as metadata_reader
import numpy as np
from audio_classifier.config.feature_engineering.pool import PoolConfig
from audio_classifier.config.preprocessing.reshape import ReshapeConfig
from audio_classifier.config.preprocessing.spec import MelSpecConfig
from script.train.classification.svm.train_svm import load_skms
from torch.utils.data import DataLoader

#%%
METADATA_PATH: str = "../../test_dataset/metadata.csv"
FOLDER_DATASET_PATH: str = "../../test_dataset/folder_dataset"

CURR_VAL_SKM_PATH_STUB: str = "../../../../model/skm/spec_00_reshape_00_skm_00/val_00/class_{:02d}/model.pkl"

#%%
metadata: Sequence[Dict[str, str]] = metadata_reader.read_csv_metadata(
    path_to_metadata=METADATA_PATH)

#%%
query = metadata_query.DictMetaDataQuerier(metadata=metadata,
                                           filename_key="slice_file_name",
                                           label_key="classID")

#%%
mel_config: MelSpecConfig = MelSpecConfig()
reshape_config: ReshapeConfig = ReshapeConfig(slice_size=16, stride_size=16)
dataset = dataset_base.FolderDataset(folder_path=FOLDER_DATASET_PATH,
                                     sample_rate=mel_config.sample_rate,
                                     filename_to_label_func=query,
                                     cache=True)
pool_config = PoolConfig(pool_size=5, stride_size=5)
#%%
skms = load_skms(curr_val_skm_path_stub=CURR_VAL_SKM_PATH_STUB, n_classes=2)
slice_collate = base_collate.EnsembleCollateFunction(collate_funcs=[
    partial(transform_collate.mel_spectrogram_collate, config=mel_config),
    partial(reshape_collate.slice_flatten_collate, config=reshape_config),
    partial(collate_skm.skm_skl_proj_collate, skms=skms),
])

#%%
loader = DataLoader(dataset=dataset,
                    batch_size=2,
                    num_workers=3,
                    collate_fn=slice_collate)

#%%
gt_data: Dict[str, Sequence[np.ndarray]] = dict()
for filenames, batch_projs, mel_freqs, mel_times, labels in loader:
    for filename, curr_projs, mel_freq, mel_time, label in zip(
            filenames, batch_projs, mel_freqs, mel_times, labels):

        curr_pool: Sequence[np.ndarray] = feature_pool.apply_pool_func(
            spec_projs=curr_projs,
            pool_func=feature_pool.MeanStdPool(),
            pool_size=pool_config.pool_size,
            stride_size=pool_config.stride_size)
        # curr_pool_mean: np.ndarray = np.mean(curr_projs, axis=0)
        # curr_pool_std: np.ndarray = np.std(curr_projs, axis=0)
        # curr_pool: np.ndarray = np.concatenate((curr_pool_mean, curr_pool_std),
        #                                        axis=0)
        gt_data[filename] = curr_pool

# %%
collate_func = base_collate.EnsembleCollateFunction(collate_funcs=[
    partial(transform_collate.mel_spectrogram_collate, config=mel_config),
    partial(reshape_collate.slice_flatten_collate, config=reshape_config),
    partial(collate_skm.skm_skl_proj_collate, skms=skms),
    partial(collate_skm.skm_projs_pool_collate,
            pool_func=feature_pool.MeanStdPool(),
            pool_config=pool_config)
])
loader = DataLoader(dataset=dataset,
                    batch_size=2,
                    num_workers=3,
                    collate_fn=collate_func)
for filenames, batch_projs, mel_freqs, mel_times, labels in loader:
    for filename, curr_projs, mel_freq, mel_time, label in zip(
            filenames, batch_projs, mel_freqs, mel_times, labels):
        print(np.array_equal(gt_data[filename], curr_projs),
              gt_data[filename] is not curr_projs)

# %%
