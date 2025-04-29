import argparse
from typing import List

from caspar.config import MEK_FILE
from caspar.utils.argparse_utils import ExclusiveArgumentGroup


def parse_args():
    parser = argparse.ArgumentParser(description='Train CASPAR neural network model')
    parser.add_argument('--mekfile', type=str, default=MEK_FILE,
                        help='Path to Mek file (txt)')
    parser.add_argument('--version', '-v', action='version', version='caspar 1.0',
                        help='Show version and exit')
    groups: List[ExclusiveArgumentGroup] = []
    ###########

    dataset_group = ExclusiveArgumentGroup(parser, "Dataset Handling")
    dataset_group.add_argument('--name-datasets', action='store_true',
                               help='Rename the datasets in the datasets directory')
    dataset_group.add_argument('--parse-datasets', action='store_true',
                               help='Compile the datasets with pre-tags from the raw game action data')
    dataset_group.add_argument('--test-size', type=float,
                               help='Percent from 0 to 1 of the dataset to use for testing')
    dataset_group.add_argument('--validation-size', type=float,
                               help='Percent from 0 to 1 of the dataset to use for validation')
    dataset_group.add_argument('--oversample', action='store_true',
                               help='Oversample the training data to balance the classes, default behavior is to undersample')
    feature_extraction_group = dataset_group.add_mutually_exclusive_group(required=False)
    feature_extraction_group.add_argument('--extract-features', action='store_true',
                                          help='Extract features from datasets and create untagged training data')
    groups.append(dataset_group)
    ###########

    test_model_group = ExclusiveArgumentGroup(
        parser, "Test Model", "Load a model and test it against the test and validation datasets")
    test_model_group.add_argument('--s3-model', type=str, help='Path to trained model')
    groups.append(test_model_group)
    ###########

    experiment_group = parser.add_argument_group(
        "Experiment",
        description="Setup name and experiment for training and/or optimization")
    experiment_group.add_argument('--experiment-name', type=str,
                                help='MLflow experiment name')
    experiment_group.add_argument('--run-name', type=str, required=False,
                                  help='Name of the run')
    experiment_group.add_argument('--feature-correlation', action='store_true',
                                  help='Check feature correlation')
    experiment_group.add_argument('--model-name', type=str,
                                  help='Name of the model')
    ###########

    training_group = ExclusiveArgumentGroup(parser, "Training And Model Architecture")
    training_group.add_argument('--epochs', type=int,
                        help='Number of training epochs')
    training_group.add_argument('--dropout-rate', type=float,
                        help='Dropout rate')
    training_group.add_argument('--batch-size', type=int,
                        help='Batch size for training')
    training_group.add_argument('--learning-rate', type=float,
                        help='Learning rate for optimization')
    training_group.add_argument('--hidden-layers', nargs='+', type=int,
                            help='Hidden layers formats')
    groups.append(training_group)
    ###########

    hyperparameter_group = ExclusiveArgumentGroup(parser, "Hyperparameter Search")
    hyperparameter_group.add_argument('--optimize', action='store_true',
                                help='Run hyperparameter optimization')
    hyperparameter_group.add_argument('--n-trials', type=int,
                                help='Number of trials for hyperparameter optimization')
    hyperparameter_group.add_argument('--n-jobs', type=int,
                                help='Number of parallel jobs for optimization (-1 uses all cores)')
    groups.append(hyperparameter_group)

    args = parser.parse_args()

    used_groups = []
    for group in groups:
        if any(getattr(args, arg) for arg in group.args):
            used_groups.append(group.name)

    if len(used_groups) > 1:
        parser.error(
            f"Arguments from mutually exclusive groups {', '.join(used_groups)} cannot be used simultaneously.")

    return args
