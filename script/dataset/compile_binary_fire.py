import csv
import os
import random
import sys
from argparse import ArgumentParser, Namespace
from os import path
from typing import Dict, List, Tuple

import numpy as np
import soundfile
from librosa import core as rosa_core


def create_parser() -> ArgumentParser:
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("--raw_root_path", required=True, type=str)
    parser.add_argument("--dataset_export_path", required=True, type=str)
    parser.add_argument("--split_sec", required=True, type=float)
    parser.add_argument("--num_samples_per_class", required=True, type=int)
    parser.add_argument("--num_folds", required=True, type=int)
    parser.add_argument("--sample_rate",
                        required=True,
                        type=int,
                        default=44100)
    return parser


def split_audio(
        file_path: str,
        split_sec: float,
        sample_rate: float = 44100) -> Tuple[List[np.ndarray], List[str]]:
    """Split the audio every `split_sec`

    Args:
        file_path (str): Path to the audio file.
        split_sec (float): Split the audio every `split_sec`
        sample_rate (float, optional): The target sample rate. Defaults to 44100.

    Returns:
        List[np.ndarray]: The list of all the splitted audio. Each np.ndarray has shape [n_channels, split_sec * sample_rate]
        List[str]: [curr_filename] * len(audio_splits)
    """
    audio: np.ndarray
    sr: float
    audio, sr = rosa_core.audio.load(file_path, sr=sample_rate, mono=False)
    # if mono channel, add channel dimension
    if len(audio.shape) != 2:
        audio = np.expand_dims(audio, axis=0)
    SPLIT_SIZE: int = int(sr * split_sec)
    # make array splittable
    audio = audio[:, :-(audio.shape[1] % SPLIT_SIZE)]
    audio_splits: List[np.ndarray] = np.split(audio,
                                              int(audio.shape[1] / SPLIT_SIZE),
                                              axis=1)
    curr_filename: str = path.split(file_path)[-1]
    filenames: List[str] = [curr_filename] * len(audio_splits)
    return audio_splits, filenames


def sample_current_file(curr_audio_splits: List[np.ndarray],
                        curr_filenames: List[str], curr_labels: List[int],
                        fold_audio_splits: List[List[np.ndarray]],
                        fold_filenames: List[List[str]],
                        fold_labels: List[List[int]],
                        num_samples_per_fold_per_class: int):
    """Sample splitted file from current file for each fold without replacement.

    Args:
        curr_audio_splits (List[np.ndarray]): The list of all splitted audio for current file.
        curr_filenames (List[str]): [curr_filename] * len(audio_splits)
        curr_labels (List[int]): [curr_label] * len(audio_splits)
        fold_audio_splits (List[List[np.ndarray]]): The list containing the audio splits for all folds. Has shape (n_folds,  n_samples_per_fold, n_channels, n_samples)
        fold_filenames (List[List[str]]): The list containing the filenames for all folds. Has shape (n_folds,  n_samples_per_fold)
        fold_labels (List[List[int]]): The list contianing the lables for all folds. Has shahpe (n_folds, n_samples_per_fold)
        num_samples_per_fold (int): number of samples per fold per class
    """
    num_folds: int = len(fold_audio_splits)
    sample_indices: List[int] = random.sample(
        range(len(curr_audio_splits)),
        k=num_samples_per_fold_per_class * num_folds)
    for curr_fold in range(num_folds):
        begin_idx: int = curr_fold * num_samples_per_fold_per_class
        end_idx: int = begin_idx + num_samples_per_fold_per_class
        curr_indices: List[int] = sample_indices[begin_idx:end_idx]
        curr_fold_audio_splits: List[np.ndarray] = [
            curr_audio_splits[i] for i in curr_indices
        ]
        curr_fold_filenames: List[str] = [
            curr_filenames[i] for i in curr_indices
        ]
        curr_fold_labels: List[int] = [curr_labels[i] for i in curr_indices]
        # the following is the error
        fold_audio_splits[curr_fold].extend(curr_fold_audio_splits)
        fold_filenames[curr_fold].extend(curr_fold_filenames)
        fold_labels[curr_fold].extend(curr_fold_labels)


def process_class_folder(class_path: str,
                         fold_audio_splits: List[List[np.ndarray]],
                         fold_filenames: List[List[str]],
                         fold_labels: List[List[int]],
                         num_samples_per_class: int,
                         split_sec: float,
                         curr_class_label: int,
                         sample_rate: float = 44100):
    """Split all the audio in the current class folder.

    Args:
        class_path (str): The path to the current class.
        split_sec (float): Split the audio every `split_sec`.
        curr_class_label (int): The int representation of the current class label.
        sample_rate (float, optional): The target sample rate. Defaults to 44100.

    Returns:
        List[np.ndarray]: The list of all the splitted audio. Each np.ndarray has shape [n_channels, split_sec * sample_rate]
        List[str]: The list of filenames that each splitted audio came from.
        List[int]: The list of curr_class_label.

    """
    file_paths: List[str] = os.listdir(class_path)
    file_paths = [path.join(class_path, fp) for fp in file_paths]
    # make file_paths only contains *.wav file
    file_paths = list(
        filter(lambda p: (path.splitext(p)[1] == ".wav"), file_paths))
    num_raw_files: int = len(file_paths)
    num_samples_per_fold_per_class: int = int(num_samples_per_class /
                                              num_raw_files)
    # split each file
    for file_path in file_paths:
        curr_audio_splits, curr_filenames = split_audio(
            file_path=file_path, split_sec=split_sec, sample_rate=sample_rate)
        curr_labels: List[int] = [curr_class_label] * len(curr_filenames)
        sample_current_file(
            curr_audio_splits=curr_audio_splits,
            curr_filenames=curr_filenames,
            curr_labels=curr_labels,
            fold_audio_splits=fold_audio_splits,
            fold_filenames=fold_filenames,
            fold_labels=fold_labels,
            num_samples_per_fold_per_class=num_samples_per_fold_per_class)


def process_raw_dataset(raw_root_path: str,
                        fold_audio_splits: List[List[np.ndarray]],
                        fold_filenames: List[List[str]],
                        fold_labels: List[List[int]],
                        num_samples_per_class: int,
                        split_sec: float,
                        sample_rate: float = 44100):
    """Process the raw dataset.

    Args:
        raw_root_path (str): Path to the root of the raw dataset.
        fold_audio_splits (List[List[np.ndarray]]): The list to contain the splitted audio of all folds.
        fold_filenames (List[List[str]]): The list to contain the raw filename of all folds.
        fold_labels (List[List[int]]): The list to contain teh lables of all folds.
        num_samples_per_class (int): The number of samples per class.
        split_sec (float): Split the audio every `split_sec`.
        sample_rate (float, optional): The target sample rate. Defaults to 44100.

    Returns:
        Dict[str, int]: The dictionary that maps the class string representaiton to its int representation.
    """
    # filter out non-folder path in class_paths
    class_paths: List[str] = os.listdir(raw_root_path)
    class_paths = [path.join(raw_root_path, cp) for cp in class_paths]
    class_paths = list(filter(lambda p: (path.isdir(p) == True), class_paths))
    # map class string to int representation
    class_dict: Dict[str, int] = {
        path.split(class_path)[-1]: label
        for label, class_path in enumerate(class_paths)
    }
    for class_path in class_paths:
        curr_class: str = path.split(class_path)[-1]
        process_class_folder(class_path=class_path,
                             fold_audio_splits=fold_audio_splits,
                             fold_filenames=fold_filenames,
                             fold_labels=fold_labels,
                             num_samples_per_class=num_samples_per_class,
                             split_sec=split_sec,
                             curr_class_label=class_dict[curr_class],
                             sample_rate=sample_rate)
    return class_dict


def write_curr_fold_audio(curr_fold_path: str,
                          curr_fold_audio_splits: List[np.ndarray],
                          curr_fold_labels: List[int], sample_rate: int,
                          audio_file_count: int) -> Tuple[int, List[str]]:
    """Export *.wav file for current fold

    Args:
        curr_fold_path (str): The path to the root of current fold folder
        curr_fold_audio_splits (List[List[np.ndarray]]): All the splitted auido that belongs to current fold.
        curr_fold_labels (List[List[int]]): All the class labels for each file in current fold.
        sample_rate (int): The sapmle rate used to export the audio.
        audio_file_count (int): The counter for the files that have been exported.

    Returns:
        int: The updated audio_file_count.
        List[str]: The list containing filenames of the exported audio file.
    """
    curr_fold_export_filenames: List[str] = list()
    # print(len(curr_fold_audio_splits))
    for curr_audio, curr_label in zip(curr_fold_audio_splits,
                                      curr_fold_labels):
        filename: str = str.format("{}_{}.wav", audio_file_count, curr_label)
        file_path: str = path.join(curr_fold_path, filename)
        soundfile.write(file=file_path,
                        data=curr_audio.T,
                        samplerate=sample_rate)
        curr_fold_export_filenames.append(filename)
        audio_file_count = audio_file_count + 1
    return audio_file_count, curr_fold_export_filenames


def write_fold_audio(dataset_export_path: str,
                     fold_audio_splits: List[List[np.ndarray]],
                     fold_labels: List[List[int]],
                     sample_rate: int) -> List[List[str]]:
    """Write all folds dataset audio.

    Args:
        dataset_export_path (str): The path to the root of dataset.
        fold_audio_splits (List[List[np.ndarray]]): The splitted audio of all folds.
        fold_labels (List[List[int]]): All the class labels for each file in current fold.
        sample_rate (int): The sapmle rate used to export the audio.

    Returns:
        List[str]: The exported filenames of all folds.
    """
    audio_file_count: int = 0
    fold_export_filenames: List[List[str]] = list()
    for curr_fold, (curr_fold_audio_splits, curr_fold_labels) in enumerate(
            zip(fold_audio_splits, fold_labels)):
        curr_fold_dir: str = str.format("fold{}", curr_fold + 1)
        curr_fold_path: str = path.join(dataset_export_path, "audio",
                                        curr_fold_dir)
        os.makedirs(curr_fold_path, exist_ok=True)
        audio_file_count, curr_fold_export_filenames = write_curr_fold_audio(
            curr_fold_path=curr_fold_path,
            curr_fold_audio_splits=curr_fold_audio_splits,
            curr_fold_labels=curr_fold_labels,
            sample_rate=sample_rate,
            audio_file_count=audio_file_count)
        fold_export_filenames.append(curr_fold_export_filenames)
    return fold_export_filenames


def write_metadata(dataset_export_path: str,
                   fold_export_filenames: List[List[str]],
                   fold_filenames: List[List[str]],
                   fold_labels: List[List[int]], class_dict: Dict[str, int]):
    """Write metadata to the dataset export path.

    Args:
        dataset_export_path (str): The path to the root of dataset.
        fold_export_filenames (List[List[str]]): The exported filenames of all folds.
        fold_filenames (List[List[str]]): The raw filenames of all folds.
        fold_labels (List[List[int]]): The class labels of all folds.
        class_dict (Dict[str, int]): The dictionary that maps the class string representaiton to its int representation.
    """
    os.makedirs(path.join(dataset_export_path, "metadata"))
    metadata_path: str = path.join(dataset_export_path, "metadata",
                                   "metadata.csv")
    inv_class_dict: Dict[int, str] = {
        label: class_str
        for class_str, label in class_dict.items()
    }
    with open(metadata_path, mode='w') as metadata:
        metadata_writer = csv.writer(metadata)
        metadata_writer.writerow([
            "slice_file_name", "raw_file_name", "salience", "fold", "classID",
            "class"
        ])
        num_folds: int = len(fold_export_filenames)
        for i in range(num_folds):
            curr_fold: int = i + 1
            for exp_filename, raw_filename, label in zip(
                    fold_export_filenames[i], fold_filenames[i],
                    fold_labels[i]):
                metadata_writer.writerow([
                    exp_filename, raw_filename,
                    str(1),
                    str(curr_fold),
                    str(label), inv_class_dict[label]
                ])


def main(argc: int, argv: List[str]):
    parser = create_parser()
    args: Namespace = parser.parse_args(argv)
    # read constant
    RAW_ROOT_PATH: str = args.raw_root_path
    DATASET_EXPORT_PATH: str = args.dataset_export_path
    SPLIT_SEC: float = args.split_sec
    NUM_SAMPLES_PER_CLASS: int = args.num_samples_per_class
    NUM_FOLDS: int = args.num_folds
    SAMPLE_RATE: int = args.sample_rate
    # initialize container for dataset
    fold_audio_splits: List[List[np.ndarray]] = [[] for _ in range(NUM_FOLDS)]
    fold_filenames: List[List[str]] = [[] for _ in range(NUM_FOLDS)]
    fold_labels: List[List[int]] = [[] for _ in range(NUM_FOLDS)]
    class_dict: Dict[str, int] = process_raw_dataset(
        raw_root_path=RAW_ROOT_PATH,
        fold_audio_splits=fold_audio_splits,
        fold_filenames=fold_filenames,
        fold_labels=fold_labels,
        num_samples_per_class=NUM_SAMPLES_PER_CLASS,
        split_sec=SPLIT_SEC,
        sample_rate=SAMPLE_RATE)
    os.makedirs(DATASET_EXPORT_PATH, exist_ok=True)
    fold_export_filenames: List[List[str]] = write_fold_audio(
        dataset_export_path=DATASET_EXPORT_PATH,
        fold_audio_splits=fold_audio_splits,
        fold_labels=fold_labels,
        sample_rate=SAMPLE_RATE)
    write_metadata(dataset_export_path=DATASET_EXPORT_PATH,
                   fold_export_filenames=fold_export_filenames,
                   fold_filenames=fold_filenames,
                   fold_labels=fold_labels,
                   class_dict=class_dict)


if __name__ == '__main__':
    args: List[str] = sys.argv[1:]
    main(len(args), args)
