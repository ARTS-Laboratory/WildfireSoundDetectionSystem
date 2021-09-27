#%%
from functools import partial
from os import path
from typing import Dict, List

import audio_classifier.train.collate.base as base_collate
import audio_classifier.train.collate.preprocessing.spectrogram.transform as transform_collate
import audio_classifier.train.data.dataset.base as dataset_base
import audio_classifier.train.data.metadata.query as metadata_query
import audio_classifier.train.data.metadata.reader as metadata_reader
import librosa.core as rosa_core
import numpy as np
from audio_classifier.common.preprocessing.spectrogram.transform import \
    transform_mel_spectrogram
from audio_classifier.config.preprocessing.spec_config import MelSpecConfig
from torch.utils.data import DataLoader

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
config: MelSpecConfig = MelSpecConfig()
dataset = dataset_base.FolderDataset(path_to_folder=PATH_TO_FOLDER_DATASET,
                                     sample_rate=config.sample_rate,
                                     filename_to_label_func=query,
                                     cache=True)
#%%
collate_function = base_collate.EnsembleCollateFunction(collate_funcs=[
    partial(transform_collate.mel_spectrogram_collate, config=config)
])

#%%
loader = DataLoader(dataset=dataset,
                    batch_size=2,
                    num_workers=3,
                    collate_fn=collate_function)
for filenames, mel_specs, mel_freqs, mel_times, labels in loader:
    for filename, mel_spec, mel_freq, mel_time, label in zip(
            filenames, mel_specs, mel_freqs, mel_times, labels):
        file_path: str = path.join(PATH_TO_FOLDER_DATASET, filename)
        sound_wave, _ = rosa_core.load(path=file_path,
                                       sr=config.sample_rate,
                                       mono=True)
        mel_spec_pr, mel_freq_pr, mel_time_pr = transform_mel_spectrogram(
            sound_wave=sound_wave,
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            n_mels=config.n_mels,
            freq_min=config.freq_min,
            freq_max=config.freq_max,
            window_size=config.window_size,
            hop_size=config.hop_size,
            apply_log=config.apply_log)
        is_pass: bool = True
        is_pass = is_pass and np.array_equal(mel_spec, mel_spec_pr)
        is_pass = is_pass and np.array_equal(mel_freq, mel_freq_pr)
        is_pass = is_pass and np.array_equal(mel_time, mel_time_pr)
        print(str.format("{} {}", filename, is_pass))

# %%
