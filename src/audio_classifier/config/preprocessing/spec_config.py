"""The module contains all the functions and classes related to spectrogram configuration.
"""
import sys
from argparse import ArgumentParser, HelpFormatter
from dataclasses import dataclass, field
from typing import Type, TypeVar

from dataclasses_json import dataclass_json
from overrides import overrides


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

    def __post_init__(self):
        return


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

    @overrides
    def __post_init__(self):
        if self.freq_max < 0.0:
            self.freq_max = self.sample_rate / 2.0


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
    config.__post_init__()
    return config


class SpecConfigArgumentParser(ArgumentParser):
    def __init__(self,
                 prog=None,
                 usage=None,
                 description=None,
                 epilog=None,
                 parents=[],
                 formatter_class=HelpFormatter,
                 prefix_chars='-',
                 fromfile_prefix_chars=None,
                 argument_default=None,
                 conflict_handler='error',
                 add_help=False,
                 allow_abbrev=True):
        super().__init__(prog=prog,
                         usage=usage,
                         description=description,
                         epilog=epilog,
                         parents=parents,
                         formatter_class=formatter_class,
                         prefix_chars=prefix_chars,
                         fromfile_prefix_chars=fromfile_prefix_chars,
                         argument_default=argument_default,
                         conflict_handler=conflict_handler,
                         add_help=add_help,
                         allow_abbrev=allow_abbrev)
        self.add_argument(
            "--spectrogram_config_file",
            required=True,
            type=str,
            help="path to the spectrogram configuration *.json file")
