#%%
import librosa.core as rosa_core
import librosa.feature.spectral as rosa_spectral
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

#%%
AUDIO_PATH: str = "../test_audio/17592-5-1-1.wav"
SAMPLE_RATE: int = 44100
WINDOW_SIZE: int = 1024
NFFT: int = 1024
NMELS: int = 40

# %% [markdown]
# # Generate stft spectrogram with librosa library

#%%
amplitude, _ = rosa_core.load(AUDIO_PATH, sr=SAMPLE_RATE, mono=True)

#%%
stft_sample_freq: np.ndarray = rosa_core.fft_frequencies(sr=SAMPLE_RATE,
                                                         n_fft=NFFT)
stft_spec: np.ndarray = rosa_core.stft(y=amplitude,
                                       n_fft=NFFT,
                                       win_length=WINDOW_SIZE,
                                       hop_length=WINDOW_SIZE,
                                       center=False)
stft_spec = np.abs(stft_spec)
stft_sample_time: np.ndarray = rosa_core.times_like(stft_spec,
                                                    sr=SAMPLE_RATE,
                                                    hop_length=WINDOW_SIZE,
                                                    n_fft=NFFT)

#%%
plt.pcolormesh(stft_sample_time,
               stft_sample_freq,
               10 * np.log10(stft_spec),
               shading='auto',
               cmap="magma")

#%% [markdown]
# # Generate spectrogram with scipy.signal

#%%
stft_sample_freq_prime, stft_sample_time_prime, stft_spec_prime = signal.spectrogram(
    amplitude,
    fs=SAMPLE_RATE,
    nperseg=WINDOW_SIZE,
    nfft=NFFT,
    noverlap=0,
    window="hann")
stft_spec_prime = np.abs(stft_spec_prime)

#%%
plt.pcolormesh(stft_sample_time_prime,
               stft_sample_freq_prime,
               10 * np.log10(stft_spec_prime),
               shading='auto',
               cmap="magma")

#%% [markdown]
# ## Cannot replicate the result

#%%
print(np.all(np.equal(stft_spec, stft_spec_prime)))

#%% [markdown]
# # Use librosa melspectrogram built in pipeline to generate melspectrogram

#%%
mel_sample_freq: np.ndarray = rosa_core.mel_frequencies(n_mels=NMELS)
mel_spec: np.ndarray = rosa_spectral.melspectrogram(y=amplitude,
                                                    n_fft=NFFT,
                                                    sr=SAMPLE_RATE,
                                                    n_mels=NMELS,
                                                    win_length=WINDOW_SIZE,
                                                    hop_length=WINDOW_SIZE,
                                                    center=False)
mel_sample_time: np.ndarray = rosa_core.times_like(mel_spec,
                                                   sr=SAMPLE_RATE,
                                                   hop_length=WINDOW_SIZE,
                                                   n_fft=NFFT)

#%%
plt.pcolormesh(mel_sample_time,
               mel_sample_freq,
               10 * np.log10(mel_spec),
               shading='auto',
               cmap="magma")

#%% [markdown]
# # Use the stft spectrogram to replicate the built in melspectrogram pipeline result

# %%
mel_sample_freq_prime: np.ndarray = rosa_core.mel_frequencies(n_mels=NMELS)
mel_spec_prime: np.ndarray = rosa_spectral.melspectrogram(
    S=stft_spec**2,
    n_fft=NFFT,
    sr=SAMPLE_RATE,
    n_mels=NMELS,
    win_length=WINDOW_SIZE,
    hop_length=WINDOW_SIZE)
mel_sample_time_prime: np.ndarray = rosa_core.times_like(
    mel_spec_prime, sr=SAMPLE_RATE, hop_length=WINDOW_SIZE, n_fft=NFFT)

#%%
plt.pcolormesh(mel_sample_time_prime,
               mel_sample_freq_prime,
               10 * np.log10(mel_spec_prime),
               shading='auto',
               cmap="magma")

#%% [markdown]
# ## Equivalent result verified

#%%
print(np.all(np.equal(mel_spec, mel_spec_prime)))
print(mel_spec is mel_spec_prime)
