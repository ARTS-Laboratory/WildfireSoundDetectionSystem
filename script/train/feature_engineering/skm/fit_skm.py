import sys
from argparse import Namespace
from functools import partial
from typing import List, Union

import audio_classifier.train.collate.base as collate_base
import audio_classifier.train.collate.preprocessing.spectrogram.reshape as collate_reshape
import audio_classifier.train.collate.preprocessing.spectrogram.transform as collate_transform
import numpy as np
import script.train.common as script_common
from audio_classifier.train.data.dataset.composite import KFoldDatasetGenerator
from script.train.feature_engineering.skm import fit_skm

MetaDataType = script_common.MetaDataType
CollateFuncType = script_common.CollateFuncType


def main(args: List[str]):
    argv: Namespace = fit_skm.parse_args(args)
    export_path: str = argv.export_path
    k_min: int = argv.k_min
    k_max: int = argv.k_max
    k_step: int = argv.k_step
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
        train, _ = fit_skm.generate_slice_dataset(
            curr_val_fold=curr_val_fold,
            dataset_generator=dataset_generator,
            collate_function=collate_func,
            loader_config=loader_config)
        slices, labels = fit_skm.convert_to_ndarray(slice_dataset=train)
        unique_labels: np.ndarray = np.unique(labels)
        for curr_label in unique_labels:
            curr_class_path: str = fit_skm.get_curr_class_path(
                export_path, curr_val_fold, curr_label)
            curr_slices: np.ndarray = fit_skm.get_curr_class_slices(
                curr_label, slices, labels)
            k_value: Union[int, None] = fit_skm.try_k_elbow(
                curr_class_path=curr_class_path,
                slices=curr_slices,
                skm_config=skm_config,
                k_range=range(k_min, k_max, k_step))
            if k_value is None:
                continue
            skm, model_path = fit_skm.fit_skm(curr_class_path=curr_class_path,
                                              slices=curr_slices,
                                              skm_config=skm_config,
                                              k_value=k_value)
            is_pass: bool = fit_skm.verify_model(slices=slices,
                                                 skm=skm,
                                                 model_path=model_path)
            if is_pass == False:
                print("Discrepency between src model and exported model",
                      file=sys.stderr)
                sys.exit(1)
            fit_skm.plot_silhouette(curr_class_path=curr_class_path,
                                    slices=curr_slices,
                                    skm=skm,
                                    k_value=k_value)


if __name__ == "__main__":
    args: List[str] = sys.argv[1:]
    main(args)
