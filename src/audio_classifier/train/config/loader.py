from abc import ABC
from argparse import ArgumentParser, HelpFormatter
from dataclasses import dataclass, field
from typing import Optional, Type, TypeVar

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass(init=True,
           repr=True,
           eq=True,
           order=False,
           unsafe_hash=False,
           frozen=False)
class LoaderConfig():

    num_workers: int = field(default=0)
    batch_size: int = field(default=1)

    def __post_init__(self):
        pass


def get_loader_config_from_json(config_file_path: str) -> LoaderConfig:
    """Get SpectrogramConfig from a json file.

    If exception encountered while reading the json file, default value will be assigned to SpectrogramConfig.

    Args:
        config_file_path (str): path to config *.json file
        ConfigType (Type): All the subclass of `STFTConfig`.

    Returns:
        config (ConfigType): an instance of ConfigType set according to passed in file.
    """
    config: LoaderConfig
    try:
        with open(config_file_path, mode="r") as config_file:
            config = LoaderConfig.from_json(config_file.read())
    except Exception as e:
        config = LoaderConfig()
    config.__post_init__()
    return config


class LoaderConfigArgumentParser(ArgumentParser):
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
        self.add_argument("--loader_config_path",
                          type=str,
                          required=False,
                          help="path to the loader configuration *.json file")