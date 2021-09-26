#%%
import librosa.core as rosa_core
import librosa.feature.spectral as rosa_spectral
import matplotlib.pyplot as plt
import numpy as np
from audio_classifier.common.preprocessing.spectrogram import transform

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
plt.pcolormesh(mel_sample_time,
               mel_sample_freq,
               mel_spec,
               shading='auto',
               cmap="magma")

#%%[markdown]
# # Generate mel-spectrogram with rosa_spectral.transform_mel_spectrogram

# %%
mel_sample_freq_prime: np.ndarray = rosa_core.mel_frequencies(n_mels=NMELS)
mel_spec_prime: np.ndarray = rosa_spectral.melspectrogram(
    y=sound_wave,
    n_fft=NFFT,
    sr=SAMPLE_RATE,
    n_mels=NMELS,
    fmax=SAMPLE_RATE / 2,
    win_length=WINDOW_SIZE,
    hop_length=WINDOW_SIZE,
    center=False)
mel_sample_time_prime: np.ndarray = rosa_core.times_like(
    mel_spec_prime, sr=SAMPLE_RATE, hop_length=WINDOW_SIZE, n_fft=NFFT)
plt.pcolormesh(mel_sample_time_prime,
               mel_sample_freq_prime,
               10 * np.log10(mel_spec_prime),
               shading="auto",
               cmap="magma")

#%% [markdown]
# ## Equivalent result verified

#%%
print(np.all(np.equal(mel_spec, 10 * np.log10(mel_spec_prime))))
print(mel_spec is mel_spec_prime)

#%%[markdown]
# # Generate stft-spectrogram with transform.transform_stft_spectrogram

#%%
stft_spec, stft_sample_freq, stft_sample_time = transform.transform_stft_spectrogram(
    sound_wave=sound_wave,
    sample_rate=SAMPLE_RATE,
    n_fft=NFFT,
    window_size=WINDOW_SIZE,
    hop_size=WINDOW_SIZE,
    apply_log=True)
plt.pcolormesh(stft_sample_time,
               stft_sample_freq,
               stft_spec,
               shading="auto",
               cmap="magma")

#%%
stft_sample_freq_prime: np.ndarray = rosa_core.fft_frequencies(sr=SAMPLE_RATE,
                                                               n_fft=NFFT)
stft_spec_prime: np.ndarray = rosa_core.stft(y=sound_wave,
                                             n_fft=NFFT,
                                             win_length=WINDOW_SIZE,
                                             hop_length=WINDOW_SIZE,
                                             center=False)
stft_spec_prime = np.abs(stft_spec_prime)
stft_sample_time_prime: np.ndarray = rosa_core.times_like(
    stft_spec_prime, sr=SAMPLE_RATE, hop_length=WINDOW_SIZE, n_fft=NFFT)
plt.pcolormesh(stft_sample_time_prime,
               stft_sample_freq_prime,
               10 * np.log10(stft_spec_prime),
               shading='auto',
               cmap="magma")

#%%
print(np.all(np.equal(stft_spec, 10 * np.log10(stft_spec_prime))))
print(stft_spec is stft_spec_prime)
