import os
import pickle
import sys
from argparse import Namespace
from functools import partial
from typing import List, Sequence, Tuple, Union

import audio_classifier.common.feature_engineering.pool as feature_pool
import audio_classifier.train.collate.base as collate_base
import audio_classifier.train.collate.feature_engineering.pool as collate_pool
import audio_classifier.train.collate.feature_engineering.skm as collate_skm
import audio_classifier.train.collate.preprocessing.spectrogram.reshape as collate_reshape
import audio_classifier.train.collate.preprocessing.spectrogram.transform as collate_transform
import script.train.skl_loader.classifier as skl_classifier_loader
import script.train.skl_loader.skm as skl_skm_laoder
from audio_classifier.train.data.dataset.composite import KFoldDatasetGenerator
from script.train import train_common
from script.train.classification import classify_common
from script.train.validate import validate_common
from script.train.validate.skm_classify import skm_classify_common, snr_test, validate
from sklearn.base import ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn_plugins.cluster.spherical_kmeans import SphericalKMeans

MetaDataType = train_common.MetaDataType
CollateFuncType = train_common.CollateFuncType


def main(args: List[str]):
    argv: Namespace = validate.parse_args(args)
    skm_root_path: str = argv.skm_root_path
    val_fold_path_stub: str = "val_{:02d}"
    class_skm_path_stub: str = "class_{:02d}/model.pkl"
    classifier_path_stub: str = os.path.join(argv.classifier_path,
                                             val_fold_path_stub + ".pkl")
    metric_path: str = os.path.join(argv.classifier_path, "metric.pkl")
    # os.makedirs(export_path, exist_ok=True)
    dataset_config, mel_spec_config, reshape_config, pool_config, loader_config = snr_test.get_config(
        argv=argv)
    metadata: MetaDataType = train_common.get_metadata(dataset_config)
    dataset_generator: KFoldDatasetGenerator = train_common.get_dataset_generator(
        metadata=metadata,
        dataset_config=dataset_config,
        mel_spec_config=mel_spec_config)
    metrics: List[Tuple[validate_common.ClassificationMetrics,
                        validate_common.ClassificationMetrics]] = list()
    for curr_val_fold in range(dataset_config.k_folds):
        curr_val_skm_path_stub: str = skl_skm_laoder.get_curr_val_skm_path_stub(
            curr_val_fold=curr_val_fold,
            skm_root_path=skm_root_path,
            val_fold_path_stub=val_fold_path_stub,
            class_skm_path_stub=class_skm_path_stub)
        # load models
        skms: Sequence[SphericalKMeans] = skl_skm_laoder.load_skl_skms(
            curr_val_skm_path_stub=curr_val_skm_path_stub,
            n_classes=dataset_config.n_classes)
        classifier_path: str = str.format(classifier_path_stub, curr_val_fold)
        classifier: Union[ClassifierMixin,
                          Pipeline] = skl_classifier_loader.load_classifier(
                              classifier_path=classifier_path)
        # generate collate function
        collate_func: CollateFuncType = collate_base.EnsembleCollateFunction(
            collate_funcs=[
                partial(collate_transform.mel_spectrogram_collate,
                        config=mel_spec_config),
                partial(collate_reshape.slice_flatten_collate,
                        config=reshape_config),
                partial(collate_skm.skm_skl_proj_collate, skms=skms),
                partial(collate_pool.pool_collate,
                        pool_func=feature_pool.MeanStdPool(),
                        pool_config=pool_config)
            ])
        train, val = classify_common.generate_proj_dataset(
            curr_val_fold=curr_val_fold,
            dataset_generator=dataset_generator,
            collate_function=collate_func,
            loader_config=loader_config)
        metrics_t = skm_classify_common.calculate_metrics(
            classifier=classifier, dataset=train)
        metrics_v = skm_classify_common.calculate_metrics(
            classifier=classifier, dataset=val)
        metrics.append((metrics_t, metrics_v))
        print(
            str.format("val:{} train: {} val: {}", curr_val_fold, metrics_t,
                       metrics_v))
    with open(metric_path, mode="wb") as metric_file:
        pickle.dump(metrics, metric_file)


if __name__ == "__main__":
    args: List[str] = sys.argv[1:]
    main(args)
