from collections import deque
import enum
from typing import Any, Deque, List, MutableSequence, Sequence


def combine_batches(
        batches: Deque[Sequence[Sequence[Any]]]) -> Sequence[Sequence[Any]]:
    """Combine multiple batches into one big dataset.

    Args:
        batches (Deque[Sequence[Sequence[Any]]]): (n_batches, n_items, batch_size)

    Returns:
        ret_data Sequence[Sequence[Any]]: (n_items_output, n_total_data_points)
    """
    n_items_output: int = len(batches[0])
    ret_data: List[MutableSequence[Any]] = [
        deque() for _ in range(n_items_output)
    ]
    while len(batches) > 0:
        curr_batch: Sequence[Sequence] = batches.popleft()
        for i, curr_item in enum(curr_batch):
            ret_data[i].extend(curr_item)
    return ret_data
