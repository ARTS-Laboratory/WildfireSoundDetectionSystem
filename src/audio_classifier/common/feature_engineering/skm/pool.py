from typing import Callable, Sequence, Tuple, Union

import numpy as np


class PoolFunc:
    __pool_funcs: Sequence[Callable[[np.ndarray], np.ndarray]]

    def __init__(self, pool_funcs: Sequence[Callable[[np.ndarray],
                                                     np.ndarray]]):
        self.__pool_funcs = pool_funcs

    def __call__(self, input: np.ndarray) -> np.ndarray:
        output_list: Tuple[np.ndarray, ...] = tuple(
            [pool_func(input) for pool_func in self.__pool_funcs])
        output: np.ndarray = np.concatenate(output_list, axis=1)
        return output


def apply_pool_func(
    spec_projs: np.ndarray,
    pool_func: Callable[[np.ndarray], np.ndarray],
    pool_size: int = -1,
    stride_size: int = -1,
) -> np.ndarray:
    """Apply pooling function on the projection vectors of a file.

    Args:
        projections (np.ndarray): (n_slices, n_clusters) The input projection vector of an audio.
        pool_func (Callable[[np.ndarray], np.ndarray]): The function used for pooling.
        pool_size (int, optional): The size of the sliding window. Defaults to -1 set the pool_size to len(projections).
        stride_size (int, optional): The stride of the sliding window. Defaults to -1 set the stride_size to pool_size.

    Returns:
        pool_vectors (np.ndarray): (n_slices_prime, pool_func_out_dim)
    """
    if pool_size == -1:
        pool_size = len(spec_projs)
    if stride_size == -1:
        stride_size = pool_size
    pool_vectors: Union[np.ndarray, None] = None
    for begin_idx in range(0, len(spec_projs) - pool_size + 1, stride_size):
        end_idx: int = begin_idx + pool_size
        curr_proj: np.ndarray = spec_projs[begin_idx:end_idx]
        curr_pool: np.ndarray = pool_func(curr_proj)
        if pool_vectors is not None:
            pool_vectors = np.vstack((pool_vectors, curr_pool))
        else:
            pool_vectors = curr_pool
    if pool_vectors is None:
        raise ValueError("pool_vectors is None")
    return pool_vectors
