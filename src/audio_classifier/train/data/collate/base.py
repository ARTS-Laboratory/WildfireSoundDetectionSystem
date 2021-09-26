from typing import Any, List


def identity_collate_function(data):
    n_items: int = len(data[0])
    ret_data: List[List[Any]] = [list() for _ in range(0, n_items)]
    for data_point in data:
        for j, item in enumerate(data_point):
            ret_data[j].append(item)
    return ret_data
