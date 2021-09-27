"""The module contains all the functions and classes related to module configuration.
"""
import dataclasses
import sys
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
        slice_size (int): Defaults to 130
        stride_size (int): Defaults to 100
    """
    slice_size: int = dataclasses.field(default=5)
    stride_size: int = dataclasses.field(default=5)


def get_feature_vector_config_from_json(
        config_file_path: str) -> ReshapeConfig:
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
