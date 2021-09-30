from collections import deque
from typing import Any, Callable, Deque, List, Sequence, MutableSequence, Tuple, Union


def identity_collate_function(
        data: Union[Sequence[Tuple], Sequence]) -> Sequence[Sequence]:
    """Transform the `data` from "batch" major to "items" major.

    Args:
        data (Sequence[Tuple]): (batch_size, n_items) A list of tuple of items returned from upstream dataset.

    Returns:
        ret_data (Sequence[Sequence[Any]]): (n_items, batch_size)
    """
    n_items: int = len(data[0])
    # use deque for better append operation
    ret_data: List[MutableSequence[Any]] = [deque() for _ in range(0, n_items)]
    for data_point in data:
        for j, item in enumerate(data_point):
            ret_data[j].append(item)
    # convert deque back to list for better random access performance
    for i, item in enumerate(ret_data):
        ret_data[i] = list(item)
    return ret_data


class EnsembleCollateFunction():

    __collate_funcs: Sequence[Callable[[Union[Sequence[Tuple], Sequence]],
                                       Union[Sequence[Tuple], Sequence]]]

    def __init__(
        self,
        collate_funcs: Sequence[Callable[[Union[Sequence[Tuple], Sequence]],
                                         Union[Sequence[Tuple], Sequence]]]
    ) -> None:
        """The constructor for EnsembleCollateFunction

        Args:
            collate_funcs (Sequence[Callable[[Sequence[Tuple]], Sequence[Tuple]]]): (n_collate_funcs, ) Each collate function should take a Sequence[Tuple] of size (n_batch, n_items_input) and returns a Sequence of size (n_batch, n_items_output).
        """
        self.__collate_funcs = collate_funcs

    def __call__(self, data: Union[Sequence[Tuple],
                                   Sequence]) -> Sequence[Sequence]:
        """Transform the data with multiple collate function in `collate_funcs` and reshape with `identity_collate_function`.

        Args:
            data (Sequence[Tuple]): (batch_size, n_items) A list of tuple of items returned from upstream dataset.

        Returns:
            ret_data (Sequence[Sequence[Any]]): (n_items_output, batch_size)
        """
        for collate_func in self.__collate_funcs:
            data = collate_func(data)
        ret_data = identity_collate_function(data)
        return ret_data
