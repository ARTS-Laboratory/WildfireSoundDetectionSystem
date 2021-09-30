import csv
from collections import deque
from typing import Dict, List, MutableSequence, Optional, Sequence


def read_csv_metadata(
        path_to_metadata: str,
        fieldnames: Optional[List[str]] = None) -> Sequence[Dict[str, str]]:
    """Read a csv metadata

    Args:
        path_to_metadata (str): The path to the metadata
        fieldnames (Optional[List[str]], optional): Required if csv file does not have a header row. Defaults to None.

    Returns:
        metadata (List[Dict[str, str]]): The metadata.
    """
    with open(path_to_metadata, mode="r") as csv_file:
        csv_reader = csv.DictReader(csv_file, fieldnames=fieldnames)
        metadata: MutableSequence[Dict[str, str]] = deque()
        for entry in csv_reader:
            metadata.append(entry)
        metadata = list(metadata)
        return metadata
