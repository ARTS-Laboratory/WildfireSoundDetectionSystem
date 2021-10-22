import random
from collections import deque
from typing import Deque, Sequence, Tuple

import numpy as np

from ...config import augment as conf_augment
from ...data.augment import sound_wave as aug_sound_wave


def add_white_noise_collate(
    data: Sequence[Tuple[str, np.ndarray,
                         int]], config: conf_augment.SoundWaveAugmentConfig
) -> Sequence[Tuple[str, np.ndarray, int]]:
    """Blend white noise to a batch of data.

    Args:
        data (Sequence[Tuple[str, np.ndarray, int]]): (batch_size, ) The data from upstream sound wave dataset loader.
        config (conf_augment.SoundWaveAugmentConfig): The configuration used to add white noise.

    Returns:
        ret_data (Sequence[Tuple[str, np.ndarray, int]]): (batch_size, ) The transformed dataset with each data point being a tuple of (filename, sound_wave, label).
    """
    ret_data: Deque[Tuple[str, np.ndarray, int]] = deque()
    for filename, sound_wave, label in data:
        if random.random() <= config.augment_ratio:
            snr: float = random.uniform(config.snr_range[0],
                                        config.snr_range[1])
            sound_wave = aug_sound_wave.add_white_noise(sound_wave=sound_wave,
                                                        snr=snr)
        ret_data.append((filename, sound_wave, label))
    return ret_data
