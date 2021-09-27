"""The module contains all the functions and classes related to spectrogram configuration.
"""
import sys
from dataclasses import dataclass, field
from typing import Type, TypeVar

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass(init=True,
           repr=True,
           eq=True,
           order=False,
           unsafe_hash=False,
           frozen=False)
class STFTSpecConfig:
    """A struct that contains spectrogram configuration

    Attributes:
        sample_rate (int): The sample rate of the sound wave. Defaults to 44100.
        n_fft (int): Number of FFT components. Defaults to 256.
        window_size (int): Each frame of audio is windowed by window of length window_size and then padded with zeros to match n_fft. Defaults to 256.
        hop_size (int): Number of audio samples between adjacent STFT columns. Defaults to 256.
        apply_log (bool): Wheather or not to apply 10*log10 to the stft_spec before return. Defaults to True.
    """
    sample_rate: int = field(default=44100)
    n_fft: int = field(default=256)
    window_size: int = field(default=256)
    hop_size: int = field(default=256)
    apply_log: bool = field(default=True)


@dataclass_json
@dataclass(init=True,
           repr=True,
           eq=True,
           order=False,
           unsafe_hash=False,
           frozen=False)
class MelSpecConfig(STFTSpecConfig):
    """A struct that contains spectrogram configuration

    Attributes:
        n_mels (int): Number of Mel bands to generate. Defaults to 40.
        freq_min (float): Lowest frequency (in Hz). Defaults to 0.0.
        freq_max (float): Highest frequency (in Hz). Defaults to -1.0 will be set to sample_rate/2.0 during runtime.
    """
    n_mels: int = field(default=40)
    freq_min: float = field(default=0.0)
    freq_max: float = field(default=-1.0)


SpectrogramConfigType = TypeVar("SpectrogramConfigType", STFTSpecConfig,
                                MelSpecConfig)


def get_spec_config_from_json(
    config_file_path: str,
    ConfigType: Type[SpectrogramConfigType] = STFTSpecConfig
) -> SpectrogramConfigType:
    """Get SpectrogramConfig from a json file.

    If exception encountered while reading the json file, default value will be assigned to SpectrogramConfig.

    Args:
        config_file_path (str): path to config *.json file
        ConfigType (Type): All the subclass of `STFTConfig`.

    Returns:
        config (ConfigType): an instance of ConfigType set according to passed in file.
    """
    config: ConfigType
    try:
        with open(config_file_path, mode="r") as config_file:
            config = ConfigType.from_json(config_file.read())
    except Exception as e:
        print("spectrogram_config", file=sys.stderr)
        print(str(e), file=sys.stderr)
        config = ConfigType()
    if isinstance(config, MelSpecConfig) and config.freq_max < 0.0:
        config.freq_max = config.sample_rate / 2.0
    return config
