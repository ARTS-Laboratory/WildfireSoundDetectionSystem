"""The module contains all the functions and classes related to training time specific algorithm configuration.
"""
import sys
from abc import ABC
from argparse import ArgumentParser, HelpFormatter
from dataclasses import dataclass, field
from typing import Optional, Type, TypeVar, Union

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
class PCAConfig(MLConfigBase):
    n_components: float = field(default=0.8)
    whiten: bool = field(default=True)


@dataclass_json
@dataclass(init=True,
           repr=True,
           eq=True,
           order=False,
           unsafe_hash=False,
           frozen=False)
class SKMConfig(PCAConfig):
    normalize: bool = field(default=True)
    standardize: bool = field(default=True)


@dataclass_json
@dataclass(init=True,
           repr=True,
           eq=True,
           order=False,
           unsafe_hash=False,
           frozen=False)
class SVCConfig(MLConfigBase):
    C: float = field(default=0.5)
    kernel: str = field(default="rbf")
    degree: int = field(default=3)
    gamma: Union[str, float] = field(default="scale")
    coef0: float = field(default=0.0)

    def __post_init__(self):
        self.kernel = str.lower(self.kernel)
        if isinstance(self.gamma, str):
            self.gamma = str.lower(self.gamma)


@dataclass_json
@dataclass(init=True,
           repr=True,
           eq=True,
           order=False,
           unsafe_hash=False,
           frozen=False)
class RFCConfig(MLConfigBase):
    n_estimators: int = field(default=100)
    criterion: str = field(default="gini")
    max_depth: Optional[int] = field(default=None)
    min_samples_split: Union[int, float] = field(default=2)
    min_samples_leaf: Union[int, float] = field(default=1)
    min_weight_fraction_leaf: float = field(default=0.0)
    max_features: Union[int, float, str] = field(default="auto")
    max_leaf_nodes: Optional[int] = field(default=None)
    min_impurity_decrease: float = field(default=0.0)
    boot_strap: bool = field(default=True)
    oob_score: bool = field(default=False)

    def __post_init__(self):
        self.criterion = str.lower(self.criterion)
        if self.min_samples_split > 1:
            self.min_samples_split = int(self.min_samples_split)


MLConfigType = TypeVar("MLConfigType", MLConfigBase, PCAConfig, SKMConfig,
                       SVCConfig, RFCConfig)


def get_alg_config_from_json(config_file_path: str,
                             ConfigType: Type[MLConfigBase]) -> "MLConfigType":
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


class PCAArgumentParser(ArgumentParser):
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
        self.add_argument("--pca_config_path",
                          required=True,
                          type=str,
                          help="path to the PCA configuration *.json file")


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


class SVCArgumentParser(ArgumentParser):
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
                          help="path to the SVC configuration *.json file")


class RFCArgumentParser(ArgumentParser):
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
        self.add_argument("--rfc_config_path",
                          required=True,
                          type=str,
                          help="path to the RFC configuration *.json file")
