from abc import ABC, abstractmethod
from typing import Dict, List
from overrides import overrides


class MetadataQueryBase(ABC):
    @abstractmethod
    def __call__(self, filename: str) -> int:
        """Query the class label correspond to the given filename.

        Args:
            filename (str): The filename to be queried

        Returns:
            label (int): The class label corresponds to the filename

        Raises:
            NotImplementedError: "MetaDataQueryBase must be overrideen by subclass"
        """
        raise NotImplementedError(
            "MetaDataQueryBase must be overridden by subclass")


class DictMetaDataQuery(MetadataQueryBase):

    __metadata: List[Dict[str, str]]
    __filename_key: str
    __label_key: str
    __filename_to_label_dict: Dict[str, int]

    def __init__(self, metadata: List[Dict[str, str]], filename_key: str,
                 label_key: str):
        super().__init__()
        self.__metadata = metadata
        self.__filename_key = filename_key
        self.__label_key = label_key
        self.__filename_to_label_dict = {
            entry[self.__filename_key]: int(entry[self.__label_key])
            for entry in self.__metadata
        }

    @overrides
    def __call__(self, filename: str) -> int:
        return self.__filename_to_label_dict[filename]
