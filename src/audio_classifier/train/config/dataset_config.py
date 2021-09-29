from abc import ABC
from argparse import ArgumentParser, HelpFormatter
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
class DatasetConfigBase(ABC):
    def __post_init__(self):
        pass


@dataclass_json
@dataclass(init=True,
           repr=True,
           eq=True,
           order=False,
           unsafe_hash=False,
           frozen=False)
class PreSplitFoldDataset(ABC):

    path_to_root: str = field()
    path_to_metadata: str = field()

    def __post_init__(self):
        pass


DatasetConfigType = TypeVar("DatasetConfigType", DatasetConfigBase,
                            PreSplitFoldDataset)


def get_dataset_config_from_json(
        config_file_path: str,
        ConfigType: Type[DatasetConfigType]) -> DatasetConfigType:
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
        raise e
    config.__post_init__()
    return config


class PreSplitFoldDatasetArgumentParser(ArgumentParser):
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
        self.add_argument("--path_to_dataset",
                          type=str,
                          required=True,
                          help="path to the dataset root path")
        self.add_argument("--path_to_metadata",
                          type=str,
                          required=True,
                          help="path to the metadata")
