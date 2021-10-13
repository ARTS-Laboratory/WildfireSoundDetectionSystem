"""The module contains all the functions and classes related to contrast configuration.
"""
import sys
from argparse import ArgumentParser, HelpFormatter
from dataclasses import dataclass, field

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass(init=True,
           repr=True,
           eq=True,
           order=False,
           unsafe_hash=False,
           frozen=False)
class ContrastConfig:
    """ContrastConfig class

    Attributes:
        alpha (float): The value to be multiplied if the value < threshold. Defaults to 1.0.
        beta (float): The value to be multiplied if the vlaue is >= threshold. Defaults to 1.3.
    """
    alpha: float = field(default=1.0)
    beta: float = field(default=1.3)


def get_contrast_config_from_json(config_file_path: str) -> ContrastConfig:
    """Get ContrastConfig from a json file.

    If exception encountered while reading the json file, default value will be assigned to ContrastConfig.

    Args:
        config_file_path (str): path to config *.json file

    Returns:
        config (ContrastConfig): an instance of ContrastConfig set according to passed in file
    """
    config: ContrastConfig
    try:
        with open(config_file_path, mode="r") as config_file:
            config = ContrastConfig.from_json(config_file.read())
    except Exception as e:
        print("contrast_config", file=sys.stderr)
        print(str(e), file=sys.stderr)
        config = ContrastConfig()
    return config


class ContrastConfigArgumentParser(ArgumentParser):
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
            "--contrast_config_path",
            required=True,
            type=str,
            help="path to the contrast configuration *.json file")
