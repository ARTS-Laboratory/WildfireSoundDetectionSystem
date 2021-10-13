from collections import deque
from functools import partial
from typing import Callable, MutableSequence, Sequence, Tuple

import numpy as np


class PoolFunc:
    __pool_funcs: Sequence[Callable[[np.ndarray], np.ndarray]]

    def __init__(self, pool_funcs: Sequence[Callable[[np.ndarray],
                                                     np.ndarray]]):
        """Constructor for pooling function that chains multiple pooling function.

        Args:
            pool_funcs (Sequence[Callable[[np.ndarray], np.ndarray]]): Each pooling function takes an np.ndarray of shape (n_slices, n_features) and output and np.ndarray of shape (n_output_features, )
        """
        self.__pool_funcs = pool_funcs

    def __call__(self, input: np.ndarray) -> np.ndarray:
        """Iterate and apply all the pooling functions and generate an output vector.

        Args:
            input (np.ndarray): (n_slices, n_features)

        Returns:
            output (np.ndarray): (n_output_features, )
        """
        output_list: Tuple[np.ndarray, ...] = tuple(
            [pool_func(input) for pool_func in self.__pool_funcs])
        output: np.ndarray = np.concatenate(output_list, axis=0)
        return output


class MeanStdPool(PoolFunc):
    def __init__(self):
        super().__init__(
            pool_funcs=[partial(np.mean, axis=0),
                        partial(np.std, axis=0)])

    def __call__(self, input: np.ndarray) -> np.ndarray:
        if input.shape[0] == 1:
            return self.__pool_funcs[0](input)
        return super().__call__(input)


def apply_pool_func(
    spec_projs: np.ndarray,
    pool_func: Callable[[np.ndarray], np.ndarray],
    pool_size: int = -1,
    stride_size: int = -1,
) -> Sequence[np.ndarray]:
    """Apply pooling function on the all the projection vectors of a file.

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
    pool_projs = list(pool_projs)
    return pool_projs
