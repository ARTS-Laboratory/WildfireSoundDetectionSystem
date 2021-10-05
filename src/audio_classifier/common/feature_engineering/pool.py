from collections import deque
from typing import Callable, Deque, Sequence, Tuple, Union, MutableSequence

import numpy as np


class PoolFunc:
    __pool_funcs: Sequence[Callable[[np.ndarray], np.ndarray]]

    def __init__(self, pool_funcs: Sequence[Callable[[np.ndarray],
                                                     np.ndarray]]):
        """[summary]

        Args:
            pool_funcs (Sequence[Callable[[np.ndarray], np.ndarray]]): Each pooling function takes an np.ndarray of shape (n_slices, n_features) and output and np.ndarray of shape (n_output_features, )
        """
        self.__pool_funcs = pool_funcs

    def __call__(self, input: np.ndarray) -> np.ndarray:
        """[summary]

        Args:
            input (np.ndarray): (n_slices, n_features)

        Returns:
            output (np.ndarray): (n_output_features, )
        """
        output_list: Tuple[np.ndarray, ...] = tuple(
            [pool_func(input) for pool_func in self.__pool_funcs])
        output: np.ndarray = np.concatenate(output_list, axis=0)
        return output


def apply_pool_func(
    spec_projs: np.ndarray,
    pool_func: Callable[[np.ndarray], np.ndarray],
    pool_size: int = -1,
    stride_size: int = -1,
) -> Sequence[np.ndarray]:
    """Apply pooling function on the projection vectors of a file.

    Args:
        projections (np.ndarray): (n_slices, n_clusters) The input projection vector of an audio.
        pool_func (Callable[[np.ndarray], np.ndarray]): The function used for pooling.
        pool_size (int, optional): The size of the sliding window. Defaults to -1 set the pool_size to len(projections).
        stride_size (int, optional): The stride of the sliding window. Defaults to -1 set the stride_size to pool_size.

    Returns:
        pool_vectors (Sequence[np.ndarray]): (n_slices_prime, n_output_features)
    """
    if pool_size == -1:
        pool_size = len(spec_projs)
    if stride_size == -1:
        stride_size = pool_size
    pool_projs: MutableSequence[np.ndarray] = deque()
    for begin_idx in range(0, len(spec_projs) - pool_size + 1, stride_size):
        end_idx: int = begin_idx + pool_size
        curr_spec_projs: np.ndarray = spec_projs[begin_idx:end_idx, :]
        curr_pool_proj: np.ndarray = pool_func(curr_spec_projs)
        pool_projs.append(curr_pool_proj)
    return pool_projs