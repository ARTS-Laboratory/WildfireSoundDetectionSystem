import csv
import os
import random
import shutil
import sys
from argparse import ArgumentParser, Namespace
from os import path
from typing import Dict, List, Tuple

import numpy as np
import soundfile
from librosa import core as rosa_core


def create_parser() -> ArgumentParser:
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("--dataset_src_path", required=True, type=str)
    parser.add_argument("--dataset_export_path", required=True, type=str)
    return parser


def process_metadata(metadata_path: str) -> Tuple[List[str], List[int]]:
    filenames: List[str] = list()
    fold: List[int] = list()
    with open(metadata_path, mode="r") as metadata_file:
        metadata = csv.DictReader(metadata_file)
        for curr_row in metadata:
            filenames.append(curr_row["filename"])
            fold.append(int(curr_row["fold"]))
    return filenames, fold


def trim_export_audio(src_path: str, export_path: str):
    audio: np.ndarray
    sr: float
    audio, sr = rosa_core.audio.load(src_path, sr=44100, mono=True)
    audio_export: np.ndarray = np.trim_zeros(audio, "b")
    soundfile.write(file=export_path, data=audio_export, samplerate=sr)


def export_metadata(metadata_src_path: str, metadata_export_path: str):
    content: List[str] = list()
    with open(metadata_src_path, mode="r") as metadata_file:
        content = metadata_file.readlines()
    if len(content) == 0:
        return
    content[0] = "slice_file_name,fold,classID,class,esc10,src_file,take\n"
    with open(metadata_export_path, mode="w+") as metadata_file:
        metadata_file.writelines(content)


def main(args: List[str]):
    parser = create_parser()
    argv: Namespace = parser.parse_args(args)
    DATASET_SRC_PATH: str = argv.dataset_src_path
    DATASET_EXPORT_PATH: str = argv.dataset_export_path
    METADATA_SRC_PATH: str = os.path.join(DATASET_SRC_PATH, "meta",
                                          "esc50.csv")
    AUDIO_EXPORT_PATH: str = os.path.join(DATASET_EXPORT_PATH, "audio")
    METADATA_EXPORT_PATH: str = os.path.join(DATASET_EXPORT_PATH, "metedata")
    filenames, folds = process_metadata(METADATA_SRC_PATH)
    for filename, fold in zip(filenames, folds):
        curr_val_path: str = os.path.join(AUDIO_EXPORT_PATH,
                                          str.format("val{}", fold))
        os.makedirs(curr_val_path, exist_ok=True)
        src_path: str = os.path.join(DATASET_SRC_PATH, "audio", filename)
        export_path: str = os.path.join(curr_val_path, filename)
        trim_export_audio(src_path, export_path)
    os.makedirs(METADATA_EXPORT_PATH, exist_ok=True)
    export_metadata(METADATA_SRC_PATH,
                    os.path.join(METADATA_EXPORT_PATH, "metadata.csv"))


if __name__ == '__main__':
    args: List[str] = sys.argv[1:]
    main(args)
