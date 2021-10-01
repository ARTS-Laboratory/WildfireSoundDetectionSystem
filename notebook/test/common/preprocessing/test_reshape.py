#%%
from typing import List, Sequence
import librosa.core as rosa_core
import numpy as np
from audio_classifier.common.preprocessing.spectrogram import (reshape,
                                                               transform)

#%%
AUDIO_PATH: str = "../../test_audio/17592-5-1-1.wav"
SAMPLE_RATE: int = 44100
WINDOW_SIZE: int = 1024
NFFT: int = 1024
NMELS: int = 40

#%%
sound_wave, _ = rosa_core.load(AUDIO_PATH, sr=SAMPLE_RATE, mono=True)

#%%[markdown]
# # Generate mel-spectrogram with transform.transform_mel_spectrogram

#%%
mel_spec, mel_sample_freq, mel_sample_time = transform.transform_mel_spectrogram(
    sound_wave=sound_wave,
    sample_rate=SAMPLE_RATE,
    n_fft=NFFT,
    n_mels=NMELS,
    freq_min=0,
    freq_max=44100 / 2,
    hop_size=WINDOW_SIZE,
    window_size=WINDOW_SIZE,
    apply_log=True)


#%%
def verify_slice(slice: np.ndarray, flat_slice: np.ndarray) -> bool:
    for j in range(0, slice.shape[1]):  # iterate over the time axis
        for i in range(0, slice.shape[0]):  # iterate over the freq axis
            curr_idx = (j * slice.shape[0]) + i
            if slice[i, j] != flat_slice[curr_idx]:
                return False
    return True


#%% [markdown]
# # Slice the spectrogram every 0.1 seconds.

#%%
spec_time_res: float = mel_sample_time[1] - mel_sample_time[0]
SLICE_SIZE: int = int(0.1 / spec_time_res)
STRIDE_SIZE: int = int(0.1 / spec_time_res)
mel_spec_slices: Sequence[np.ndarray] = reshape.slice_spectrogram(
    spectrogram=mel_spec, slice_size=SLICE_SIZE, stride_size=STRIDE_SIZE)

#%% [markdown]
# ## result verified

#%%
flatten_slices: List[np.ndarray] = list()
is_succeed: bool = True
for curr_slice in mel_spec_slices:
    curr_flat_slice = reshape.flatten_slice(curr_slice)
    is_succeed = is_succeed and verify_slice(curr_slice, curr_flat_slice)
    if is_succeed == False:
        print("failed")
        break
    flatten_slices.append(curr_flat_slice)
if is_succeed == True:
    print("pass")
#%%
is_succeed = True
for curr_flat_slice, curr_slice in zip(flatten_slices, mel_spec_slices):
    curr_slice_restored = reshape.unflatten_slice(flat_slice=curr_flat_slice,
                                                  slice_size=SLICE_SIZE)
    is_succeed = is_succeed and np.all(
        np.equal(curr_slice, curr_slice_restored))
    if is_succeed == False:
        print("failed")
        break
if is_succeed == True:
    print("pass")
# %%
