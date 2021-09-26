import os
from typing import Callable, List, Optional, Tuple, Union

import librosa.core as rosa_core
import numpy as np
from overrides import overrides
from torch.utils.data import Dataset

DataPointType = Tuple[np.ndarray, int]

class FolderDataset(Dataset):

    __path_to_folder: str
    __sample_rate: int
    __filename_to_label_func: Callable[[str], int]
    __cache: bool
    __filenames: List[str]

    __data_: Optional[List[Union[DataPointType, None]]]

    def __init__(self,
                 path_to_folder: str,
                 sample_rate: int,
                 filename_to_label_func: Callable[[str], int],
                 cache: bool = True):
        """Constructor for FolderDataset

        Args:
            path_to_folder (str): The path to the folder containing *.wav files.
            sample_rate (int): The sample rate of the audio.
            filename_to_label_func (Callable[[str], int]): The function that takes in a filename and returns the corresponding class label.
            cache (bool, optional): Wheather or not to cache the loaded audio. Defaults to True.
        """
        super().__init__()
        self.__path_to_folder = path_to_folder
        self.__sample_rate = sample_rate
        self.__filename_to_label_func = filename_to_label_func
        self.__cache = cache
        self.__filenames = os.listdir(self.__path_to_folder)
        self.__filenames = list(
            filter(lambda fn: (os.path.splitext(fn)[-1] == ".wav"),
                   self.__filenames))
        if self.__cache == True:
            self.__data_ = [None for _ in range(0, len(self.__filenames))]
        else:
            self.__data_ = None

    @overrides
    def __getitem__(self, index: int) -> Tuple[str, np.ndarray, int]:
        """Get an item from the dataset.

        Args:
            index (int): The index of the queried data point.

        Returns:
            filename (str): The filename of the queried data point.
            sound_wave (np.ndarray): (sample_rate*n_secs, ) The sound wave of the quereid data point.
            label (int): The class lable of the queried data point.
        """
        if self.__cache == False:
            return self.__load_single_audio(index=index)
        return self.__get_item_from_cache(index=index)
    
    def __len__(self):
        return len(self.__filenames)

    def __load_single_audio(self, index: int) -> Tuple[str, np.ndarray, int]:
        """Load the sound wave from the filesystem.

        Args:
            index (int): The index of the queried data point.

        Returns:
            filename (str): The filename of the queried data point.
            sound_wave (np.ndarray): (sample_rate*n_secs, ) The sound wave of the quereid data point.
            label (int): The class lable of the queried data point.
        """
        filename: str = self.__filenames[index]
        path_to_audio: str = os.path.join(self.__path_to_folder, filename)
        sound_wave, _ = rosa_core.load(path=path_to_audio,
                                       sr=self.__sample_rate,
                                       mono=True)
        label: int = self.__filename_to_label_func(filename)
        return filename, sound_wave, label

    def __get_item_from_cache(self, index: int) -> Tuple[str, np.ndarray, int]:
        """Load the sound wave from the cache.

        If the queried data point exists in the cache, returns the copy from the cache.
        Load from filesystem and store to cache if the queried data point isn't currently available.

        Args:
            index (int): The index of the queried data point.

        Returns:
            filename (str): The filename of the queried data point.
            sound_wave (np.ndarray): (sample_rate*n_secs, ) The sound wave of the quereid data point.
            label (int): The class lable of the queried data point.

        Raises:
            ValueError: Attempt to access self.__data_ when self.__cache is set to False
        """
        if self.__data_ is None:
            raise ValueError(
                "Attempt to access self.__data_ when self.__cache is set to False"
            )
        filename: str = self.__filenames[index]
        data_point: Union[DataPointType, None] = self.__data_[index]
        if data_point is not None:
            return filename, *data_point
        filename, sound_wave, label = self.__load_single_audio(index=index)
        self.__data_[index] = (sound_wave, label)
        return filename, sound_wave, label
