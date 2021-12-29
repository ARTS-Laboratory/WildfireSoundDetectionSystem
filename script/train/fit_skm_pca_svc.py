import os
import pickle
import sys
from argparse import ArgumentParser, Namespace
from typing import List, Union

import audio_classifier.config.feature_engineering.pool as conf_pool
import audio_classifier.config.preprocessing.reshape as conf_reshape
import audio_classifier.config.preprocessing.spec as conf_spec
import audio_classifier.train.config.alg as conf_alg
import audio_classifier.train.config.augment as conf_augment
import audio_classifier.train.config.dataset as conf_dataset
import audio_classifier.train.config.loader as conf_loader
import numpy as np
from audio_classifier.train.data.dataset.composite import KFoldDatasetGenerator
from script import fit_skm_pca_svc
from script.train import train_common
from sklearn.base import ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn_plugins.cluster import SphericalKMeans
from sklearn_plugins.cluster.spherical_kmeans import SphericalKMeans

MetaDataType = train_common.MetaDataType
CollateFuncType = train_common.CollateFuncType


def main(args: List[str]):
    argv: Namespace = parse_args(args)
    configs = fit_skm_pca_svc.FitSkmPcaSvcConfigs(argv)
    metadata: MetaDataType = train_common.get_metadata(configs.dataset_config)
    dataset_generator: KFoldDatasetGenerator = train_common.get_dataset_generator(
        metadata=metadata,
        dataset_config=configs.dataset_config,
        mel_spec_config=configs.mel_spec_config)
    result = fit_skm_pca_svc.FitSkmPcaSvcResult(k_vals=configs.k_vals)
    for k_val in configs.k_vals:
        skms: List[List[SphericalKMeans]] = list()
        classifiers: List[Union[Pipeline, ClassifierMixin]] = list()
        confusion_mats: List[np.ndarray] = list()
        for curr_fold in range(configs.dataset_config.k_folds):
            train_dataset, val_dataset = dataset_generator.get_train_val_dataset(
                curr_fold)
            curr_skms = fit_skm_pca_svc.fit_skms((k_val, k_val), train_dataset,
                                                 configs)
            classifier, train_acc = fit_skm_pca_svc.train_classifier(
                dataset=train_dataset, skms=curr_skms, configs=configs)
            val_acc: float = fit_skm_pca_svc.val_classifier(
                dataset=val_dataset,
                skms=curr_skms,
                classifier=classifier,
                configs=configs)
            pos_bincount: np.ndarray = fit_skm_pca_svc.infer_single_audio(
                file_path=configs.pos_test_audio_path,
                skms=curr_skms,
                classifier=classifier,
                configs=configs)
            pos_recall: float = pos_bincount[0] / (pos_bincount[0] +
                                                   pos_bincount[1])
            neg_bincount: np.ndarray = fit_skm_pca_svc.infer_single_audio(
                file_path=configs.neg_test_audio_path,
                skms=curr_skms,
                classifier=classifier,
                configs=configs)
            neg_recall: float = neg_bincount[0] / (neg_bincount[0] +
                                                   neg_bincount[1])
            confusion_mat: np.ndarray = np.stack((pos_bincount, neg_bincount),
                                                 axis=0)
            print(
                str.format(
                    "n_folds {}: train {} val {} test_pos_recall {} test_neg_recall {}",
                    curr_fold + 1, train_acc, val_acc, pos_recall, neg_recall))
            # update result
            skms.append(curr_skms)
            classifiers.append(classifier)
            confusion_mats.append(confusion_mat)
        result.skms.append(skms)
        result.classifiers.append(classifiers)
        result.confusion_mats.append(confusion_mats)
    os.makedirs(configs.export_path, exist_ok=True)
    result_path: str = os.path.join(configs.export_path,
                                    configs.export_filename)
    with open(result_path, mode="wb") as result_file:
        pickle.dump(result, result_file)


def get_argparse() -> ArgumentParser:
    parser = ArgumentParser(parents=[
        conf_dataset.DatasetConfigArgumentParser(),
        conf_augment.SoundWaveAugmentConfigArgumentParser(),
        conf_spec.SpecConfigArgumentParser(),
        conf_reshape.ReshapeConfigArgumentParser(),
        conf_alg.SKMArgumentParser(),
        conf_pool.PoolConfigArgumentParser(),
        conf_alg.PCAArgumentParser(),
        conf_alg.SVCArgumentParser(),
        conf_loader.LoaderConfigArgumentParser()
    ])
    parser.add_argument("--pos_test_audio_path", type=str, required=True)
    parser.add_argument("--neg_test_audio_path", type=str, required=True)
    parser.add_argument("--k_vals", type=int, nargs="*", required=True)
    parser.add_argument("--export_path",
                        type=str,
                        required=True,
                        help="the export path")
    parser.add_argument("--export_filename", type=str, default="metrics.pkl")
    return parser


def parse_args(args: List[str]) -> Namespace:
    parser = get_argparse()
    argv = parser.parse_args(args)
    return argv


if __name__ == "__main__":
    args: List[str] = sys.argv[1:]
    main(args)
