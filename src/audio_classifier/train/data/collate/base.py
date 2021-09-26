from typing import Any, Callable, List, Sequence, Tuple, Union


def identity_collate_function(data: Union[List[Tuple], List]) -> List[List]:
    """Transform the `data` from "batch" major to "items" major.

    Args:
        data (List[Tuple]): (n_batch, n_items) A list of tuple of items returned from upstream dataset.

    Returns:
        ret_data (List[List[Any]]): (n_items, n_batch)
    """
    n_items: int = len(data[0])
    ret_data: List[List[Any]] = [list() for _ in range(0, n_items)]
    for data_point in data:
        for j, item in enumerate(data_point):
            ret_data[j].append(item)
    return ret_data


def ensemble_collate_function(
    data: List[Tuple],
    collate_funcs: Sequence[Callable[[List[Tuple]],
                                     List[Tuple]]]) -> List[List]:
    """Transform the data with multiple collate function in `collate_funcs` and reshape with `identity_collate_function`.

    Args:
        data (List[Tuple]): (n_batch, n_items) A list of tuple of items returned from upstream dataset.
        collate_funcs (Sequence[Callable[[List[Tuple]], List[Tuple]]]): (n_collate_funcs, ) Each collate function should take a List[Tuple] of size (n_batch, n_items_input) and returns a List of size (n_batch, n_items_output).

    Returns:
        ret_data (List[List[Any]]): (n_items_output, n_batch)
    """
    for collate_func in collate_funcs:
        data = collate_func(data)
    ret_data = identity_collate_function(data)
    return ret_data
