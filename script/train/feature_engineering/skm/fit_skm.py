from argparse import Namespace
import sys
from functools import partial
from typing import List

import audio_classifier.train.collate.base as collate_base
import audio_classifier.train.collate.preprocessing.spectrogram.reshape as collate_reshape
import audio_classifier.train.collate.preprocessing.spectrogram.transform as collate_transform
import script.train.common as script_common
from audio_classifier.train.data.dataset.composite import KFoldDatasetGenerator
from script.train.feature_engineering.skm import fit_skm

MetaDataType = script_common.MetaDataType
CollateFuncType = script_common.CollateFuncType


def main(args: List[str]):
    argv: Namespace = fit_skm.parse_args(args)
    dataset_config, mel_spec_config, reshape_config, skm_config, loader_config = fit_skm.get_config(
        argv=argv)
    metadata: MetaDataType = script_common.get_metadata(dataset_config)
    dataset_generator: KFoldDatasetGenerator = script_common.get_dataset_generator(
        metadata=metadata,
        dataset_config=dataset_config,
        mel_spec_config=mel_spec_config)
    collate_func: CollateFuncType = collate_base.EnsembleCollateFunction(
        collate_funcs=[
            partial(collate_transform.mel_spectrogram_collate,
                    config=mel_spec_config),
            partial(collate_reshape.slice_flatten_collate,
                    config=reshape_config)
        ])
    for curr_val_fold in range(dataset_config.k_folds):
        train, val = fit_skm.generate_slice_dataset(
            curr_val_fold=curr_val_fold,
            dataset_generator=dataset_generator,
            collate_function=collate_func,
            loader_config=loader_config)
        print(
            str.format("{} {} {} {} {}", len(train.filenames),
                       len(train.flat_slices), len(train.sample_freqs),
                       len(train.sample_times), len(train.labels)))
        print(
            str.format("{} {} {} {} {}", len(val.filenames),
                       len(val.flat_slices), len(val.sample_freqs),
                       len(val.sample_times), len(val.labels)))

if __name__ == "__main__":
    args: List[str] = sys.argv[1:]
    main(args)
