from abc import ABC
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
