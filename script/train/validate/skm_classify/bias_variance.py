import os
import pickle
import sys
from argparse import ArgumentParser, Namespace
from typing import List

import audio_classifier.config.feature_engineering.pool as conf_pool
import audio_classifier.config.preprocessing.reshape as conf_reshape
import audio_classifier.config.preprocessing.spec as conf_spec
import audio_classifier.train.config.alg as conf_alg
import audio_classifier.train.config.augment as conf_augment
import audio_classifier.train.config.dataset as conf_dataset
import audio_classifier.train.config.loader as conf_loader
import audio_classifier.train.data.dataset.base as dataset_base
from script.train import train_common
from script.validate import bias_variance
from torch.utils.data.dataset import ConcatDataset

MetaDataType = train_common.MetaDataType
CollateFuncType = train_common.CollateFuncType


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
    parser.add_argument("--test_audio_path", type=str, required=True)
    parser.add_argument("--k_min", type=int, required=True, help="minimum k")
    parser.add_argument("--k_max", type=int, required=True, help="maximum k")
    parser.add_argument("--k_step",
                        type=int,
                        required=True,
                        help="k step size")
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


def main(args: List[str]):
    argv: Namespace = parse_args(args)
    configs = bias_variance.BiasVarianceConfig(argv)
    metadata: MetaDataType = train_common.get_metadata(configs.dataset_config)
    datasets: List[
        dataset_base.FolderDataset] = bias_variance.generate_fold_datasets(
            configs=configs, metadata=metadata)
    result = bias_variance.BiasVarianceResult(k_min=configs.k_min,
                                              k_max=configs.k_max,
                                              k_step=configs.k_step)
    val_dataset = datasets[-1]
    for curr_fold in range(configs.dataset_config.k_folds):
        curr_dataset = ConcatDataset(datasets[0:curr_fold + 1])
        skms, k_vals, k_scores = bias_variance.fit_skms(curr_dataset, configs)
        classifier, train_acc = bias_variance.train_classifier(
            dataset=curr_dataset, skms=skms, configs=configs)
        val_acc: float = bias_variance.val_classifier(dataset=val_dataset,
                                                      skms=skms,
                                                      classifier=classifier,
                                                      configs=configs)
        test_acc: float = bias_variance.infer_single_audio(
            skms=skms, classifier=classifier, configs=configs)
        print(
            str.format("n_folds {}: k_vals {} train {} val {} test {}",
                       curr_fold + 1, k_vals, train_acc, val_acc, test_acc))
        # update result
        result.k_vals.append(k_vals)
        result.k_scores.append(k_scores)
        result.train_accs.append(train_acc)
        result.val_accs.append(val_acc)
        result.test_accs.append(test_acc)
        result.skms.append(skms)
        result.classifiers.append(classifier)
    os.makedirs(configs.export_path, exist_ok=True)
    result_path: str = os.path.join(configs.export_path,
                                    configs.export_filename)
    with open(result_path, mode="wb") as result_file:
        pickle.dump(result, result_file)


if __name__ == "__main__":
    args: List[str] = sys.argv[1:]
    main(args)
