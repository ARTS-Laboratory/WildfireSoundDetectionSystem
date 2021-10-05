"""The module contains all the functions and classes related to module configuration.
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
class PoolConfig:
    """PoolConfig class

    Attributes:
        pool_size (int): The size of the sliding window. Defaults to -1 set the pool_size to len(projections).
        stride_size (int): The stride of the sliding window. Defaults to -1 set the stride_size to pool_size.
    """
    pool_size: int = field(default=-1)
    stride_size: int = field(default=-1)


def get_pool_config_from_json(config_file_path: str) -> PoolConfig:
    """Get PoolConfig from a json file.

    If exception encountered while reading the json file, default value will be assigned to PoolConfig.

    Args:
        config_file_path (str): path to config *.json file

    Returns:
        config (PoolConfig): an instance of PoolConfig set according to passed in file
    """
    config: PoolConfig
    try:
        with open(config_file_path, mode="r") as config_file:
            config = PoolConfig.from_json(config_file.read())
    except Exception as e:
        print("module_config", file=sys.stderr)
        print(str(e), file=sys.stderr)
        config = PoolConfig()
    return config


class PoolConfigArgumentParser(ArgumentParser):
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
        self.add_argument("--pool_config_path",
                          required=True,
                          type=str,
                          help="path to the pool configuration *.json file")
