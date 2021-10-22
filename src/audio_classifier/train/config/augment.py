from abc import ABC
from argparse import ArgumentParser, HelpFormatter, Namespace
from dataclasses import dataclass, field
from typing import Tuple, Type, TypeVar

from dataclasses_json import dataclass_json
from overrides import overrides


@dataclass_json
@dataclass(init=True,
           repr=True,
           eq=True,
           order=False,
           unsafe_hash=False,
           frozen=False)
class AugmentConfigBase(ABC):
    def __post_init__(self):
        pass

    def _post_process(self, argv: Namespace):
        pass


@dataclass_json
@dataclass(init=True,
           repr=True,
           eq=True,
           order=False,
           unsafe_hash=False,
           frozen=False)
class SoundWaveAugmentConfig(AugmentConfigBase):

    snr_range: Tuple[float, float] = field(default_factory=lambda: (15, 30))
    augment_ratio: float = field(default=0.5)  # ratio being augmented

    @overrides
    def __post_init__(self):
        if self.augment_ratio > 1.0 or self.augment_ratio < 0.0:
            self.augment_ratio = 0.5


AugmentConfigType = TypeVar("AugmentConfigType", AugmentConfigBase,
                            SoundWaveAugmentConfig)


def get_augment_config_from_json(
        config_file_path: str, argv: Namespace,
        ConfigType: Type[AugmentConfigType]) -> "ConfigType":
    """Get AugmentConfigBase from a json file.

    If exception encountered while reading the json file, default value will be assigned to AugmentConfigBase.

    Args:
        config_file_path (str): path to config *.json file
        ConfigType (Type): All the subclass of `AugmentConfigBase`.

    Returns:
        config (ConfigType): an instance of ConfigType set according to passed in file.
    """
    config: ConfigType
    try:
        with open(config_file_path, mode="r") as config_file:
            config = ConfigType.from_json(config_file.read())
    except Exception as e:
        raise e
    config._post_process(argv)
    return config


class SoundWaveAugmentConfigArgumentParser(ArgumentParser):
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
            "--sound_wave_augment_config_path",
            type=str,
            required=True,
            help="path to the sound wave augment configuration *.json file")
