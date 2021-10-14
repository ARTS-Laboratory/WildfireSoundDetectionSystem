"""The module contains all the functions and classes related to spectrogram configuration.
"""
import sys
from argparse import ArgumentParser, HelpFormatter
from dataclasses import dataclass, field
from typing import List, Sequence

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass(init=True,
           repr=True,
           eq=True,
           order=False,
           unsafe_hash=False,
           frozen=False)
class FreqRangeConfig:
    """A struct that contains FreqRangeSpec configuration

    Attributes:
        split_freq (Sequence[float]): The starting frequency of each spectrogram.
    """
    split_freq: List[float] = field(
        default_factory=lambda: [0.0, 5000.0, 18000.0])

    def __post_init__(self):
        return


def get_freq_range_config(config_file_path: str) -> FreqRangeConfig:
    """Get FreqRangeConfig from a json file.

    If exception encountered while reading the json file, default value will be assigned to FreqRangeConfig.

    Args:
        config_file_path (str): path to config *.json file

    Returns:
        config (FreqRangeConfig): an instance of FreqRangeConfig set according to passed in file.
    """
    try:
        with open(config_file_path, mode="r") as config_file:
            config = FreqRangeConfig.from_json(config_file.read())
    except Exception as e:
        print("freq_range_config", file=sys.stderr)
        print(str(e), file=sys.stderr)
        config = FreqRangeConfig()
    config.__post_init__()
    return config


class FreqRangeArgumentParser(ArgumentParser):
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
            "--freq_range_config_path",
            required=True,
            type=str,
            help="path to the frequency range configuration *.json file")
