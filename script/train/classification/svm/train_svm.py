import sys
from argparse import Namespace
from typing import List, Sequence

import script.train.common as script_common
from audio_classifier.train.data.dataset.composite import KFoldDatasetGenerator
from script.train.classification.svm import train_svm
from sklearn.svm import SVC
from sklearn_plugins.cluster.spherical_kmeans import SphericalKMeans

MetaDataType = script_common.MetaDataType
CollateFuncType = script_common.CollateFuncType


def main(args: List[str]):
    argv: Namespace = train_svm.parse_args(args)
    skm_root_path: str = argv.skm_root_path
    val_fold_path_stub: str = "val_{:02d}"
    class_skm_path_stub: str = "class_{:02d}/model.pkl"
    export_path: str = argv.export_path
    dataset_config, mel_spec_config, reshape_config, loader_config, pool_config = train_svm.get_config(
        argv=argv)
    metadata: MetaDataType = script_common.get_metadata(dataset_config)
    dataset_generator: KFoldDatasetGenerator = script_common.get_dataset_generator(
        metadata=metadata,
        dataset_config=dataset_config,
        mel_spec_config=mel_spec_config)
    for curr_val_fold in range(dataset_config.k_folds):
        curr_val_skm_path_stub: str = train_svm.get_curr_val_skm_path_stub(
            curr_val_fold=curr_val_fold,
            skm_root_path=skm_root_path,
            val_fold_path_stub=val_fold_path_stub,
            class_skm_path_stub=class_skm_path_stub)
        skms: Sequence[SphericalKMeans] = train_svm.load_skms(
            curr_val_skm_path_stub=curr_val_skm_path_stub,
            n_classes=dataset_config.n_classes)
        collate_func: CollateFuncType = train_svm.get_collate_func(
            skms=skms,
            mel_spec_config=mel_spec_config,
            reshape_config=reshape_config,
            pool_config=pool_config)
        train, val = train_svm.generate_proj_dataset(
            curr_val_fold=curr_val_fold,
            dataset_generator=dataset_generator,
            collate_function=collate_func,
            loader_config=loader_config)
        svc: SVC = train_svm.train_svc(curr_val_fold=curr_val_fold,
                                       dataset=train,
                                       export_path=export_path)


if __name__ == "__main__":
    args: List[str] = sys.argv[1:]
    main(args)
