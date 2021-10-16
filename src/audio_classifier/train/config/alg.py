"""The module contains all the functions and classes related to training time specific algorithm configuration.
"""
from enum import Enum
import sys
from abc import ABC
from argparse import ArgumentParser, HelpFormatter
from dataclasses import dataclass, field
from typing import Type, TypeVar, Union

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass(init=True,
           repr=True,
           eq=True,
           order=False,
           unsafe_hash=False,
           frozen=False)
class MLConfigBase(ABC):
    def __post_init__(self):
        pass


@dataclass_json
@dataclass(init=True,
           repr=True,
           eq=True,
           order=False,
           unsafe_hash=False,
           frozen=False)
class SKMConfig(MLConfigBase):
    n_components: float = field(default=0.8)
    normalize: bool = field(default=True)
    standardize: bool = field(default=True)
    whiten: bool = field(default=True)


@dataclass_json
@dataclass(init=True,
           repr=True,
           eq=True,
           order=False,
           unsafe_hash=False,
           frozen=False)
class NuSVCConfig(MLConfigBase):
    nu: float = field(default=0.5)
    kernel: str = field(default="rbf")
    degree: int = field(default=3)
    gamma: Union[str, float] = field(default="scale")
    coef0: float = field(default=0.0)

    def __post_init__(self):
        self.kernel = str.lower(self.kernel)
        if isinstance(self.gamma, str):
            self.gamma = str.lower(self.gamma)


MLConfigType = TypeVar("MLConfigType", MLConfigBase, SKMConfig, NuSVCConfig)


def get_alg_config_from_json(config_file_path: str,
                             ConfigType: Type[MLConfigBase]) -> "ConfigType":
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
        print("alg_config", file=sys.stderr)
        print(str(e), file=sys.stderr)
        config = ConfigType()
    config.__post_init__()
    return config


class SKMArgumentParser(ArgumentParser):
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
        self.add_argument("--skm_config_path",
                          required=True,
                          type=str,
                          help="path to the SKM configuration *.json file")


class NuSVCArgumentParser(ArgumentParser):
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
        self.add_argument("--svc_config_path",
                          required=True,
                          type=str,
                          help="path to the NuSVC configuration *.json file")
