from typing import Any, Callable, List, Sequence, Tuple, Union


def identity_collate_function(
        data: Union[Sequence[Tuple], Sequence]) -> List[List]:
    """Transform the `data` from "batch" major to "items" major.

    Args:
        data (Sequence[Tuple]): (n_batch, n_items) A list of tuple of items returned from upstream dataset.

    Returns:
        ret_data (List[List[Any]]): (n_items, n_batch)
    """
    n_items: int = len(data[0])
    ret_data: List[List[Any]] = [list() for _ in range(0, n_items)]
    for data_point in data:
        for j, item in enumerate(data_point):
            ret_data[j].append(item)
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

    def __call__(self, data: Union[Sequence[Tuple], Sequence]) -> List[List]:
        """Transform the data with multiple collate function in `collate_funcs` and reshape with `identity_collate_function`.

        Args:
            data (Sequence[Tuple]): (n_batch, n_items) A list of tuple of items returned from upstream dataset.

        Returns:
            ret_data (List[List[Any]]): (n_items_output, n_batch)
        """
        for collate_func in self.__collate_funcs:
            data = collate_func(data)
        ret_data = identity_collate_function(data)
        return ret_data
