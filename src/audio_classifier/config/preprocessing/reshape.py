"""The module contains all the functions and classes related to module configuration.
"""
import dataclasses
import sys
from argparse import ArgumentParser, HelpFormatter
from dataclasses import dataclass

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass(init=True,
           repr=True,
           eq=True,
           order=False,
           unsafe_hash=False,
           frozen=False)
class ReshapeConfig:
    """FeatureVectorConfig class

    Attributes:
        slice_size (int): Defaults to 5
        stride_size (int): Defaults to 5
    """
    slice_size: int = dataclasses.field(default=5)
    stride_size: int = dataclasses.field(default=5)


def get_reshape_config_from_json(config_file_path: str) -> ReshapeConfig:
    """Get FeatureVectorConfig from a json file.

    If exception encountered while reading the json file, default value will be assigned to FeatureVectorConfig.

    Args:
        config_file_path (str): path to config *.json file

    Returns:
        config (FeatureVectorConfig): an instance of FeatureVectorConfig set according to passed in file
    """
    config: ReshapeConfig
    try:
        with open(config_file_path, mode="r") as config_file:
            config = ReshapeConfig.from_json(config_file.read())
    except Exception as e:
        print("module_config", file=sys.stderr)
        print(str(e), file=sys.stderr)
        config = ReshapeConfig()
    return config


class ReshapeConfigArgumentParser(ArgumentParser):
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
        self.add_argument("--reshape_config_path",
                          required=True,
                          type=str,
                          help="path to the reshape configuration *.json file")
