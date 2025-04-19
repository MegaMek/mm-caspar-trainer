# Copyright (C) 2025-2025 The MegaMek Team. All Rights Reserved.
#
# This file is part of MM-Caspar-Trainer.
#
# MM-Caspar-Trainer is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License (GPL),
# version 3 or (at your option) any later version,
# as published by the Free Software Foundation.
#
# MM-Caspar-Trainer is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# A copy of the GPL should have been included with this project;
# if not, see <https://www.gnu.org/licenses/>.
#
# NOTICE: The MegaMek organization is a non-profit group of volunteers
# creating free software for the BattleTech community.
#
# MechWarrior, BattleMech, `Mech and AeroTech are registered trademarks
# of The Topps Company, Inc. All Rights Reserved.
#
# Catalyst Game Labs and the Catalyst Game Labs logo are trademarks of
# InMediaRes Productions, LLC.
import argparse
import json
import os
import pickle

from dotenv import load_dotenv
import tensorflow as tf
import numpy as np
import tempfile

import boto3

from caspar.data.data_loader import load_datasets, load_data_as_numpy_arrays, load_tagged_datasets_classifier
from caspar.data import FeatureExtractor
from caspar.data.feature_extractor import ClassifierFeatureExtractor
from caspar.model.model import CasparClassificationModel
from caspar.training.trainer import ClassificationModelTrainer
from caspar.utils.mlflow_utils import setup_mlflow
from caspar.config import DATASETS_DIR, MEK_FILE, MODEL_CONFIG, TRAINING_CONFIG, MLFLOW_CONFIG, DOTENV_PATH, DATA_DIR, \
    DATASETS_TAGGED_DIR
from caspar.data.training_dataset_processor import TrainingDatasetProcessor, ClassificationTrainingDatasetProcessor
from caspar.data.tagger import tag_action
from caspar.hyperparameter_search import optimize_architecture
from caspar.utils.argparse_utils import ExclusiveArgumentGroup
load_dotenv(DOTENV_PATH)


def parse_args():
    parser = argparse.ArgumentParser(description='Train CASPAR neural network model')
    parser.add_argument('--mekfile', type=str, default=MEK_FILE,
                        help='Path to Mek file (txt)')
    groups = []
    ###########


    dataset_group = ExclusiveArgumentGroup(parser, "Dataset Handling")
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


def main():
    args = parse_args()

    if args.model_name:
        MLFLOW_CONFIG['model_name'] = args.model_name

    if args.test_size:
        TRAINING_CONFIG['test_size'] = args.test_size

    if args.validation_size:
        TRAINING_CONFIG['validation_size'] = args.validation_size

    if args.epochs:
        TRAINING_CONFIG['epochs'] = args.epochs

    if args.dropout_rate:
        MODEL_CONFIG['dropout_rate'] = args.dropout_rate

    if args.batch_size:
        TRAINING_CONFIG['batch_size'] = args.batch_size

    if args.learning_rate:
        MODEL_CONFIG['learning_rate'] = args.learning_rate

    if args.hidden_layers:
        MODEL_CONFIG['hidden_layers'] = args.hidden_layers

    if args.run_name:
        MLFLOW_CONFIG['run_name'] = args.run_name

    if args.experiment_name:
        MLFLOW_CONFIG['experiment_name'] = args.experiment_name

    if args.parse_datasets:
        make_tagged_datasets()
        print("Finished compiling datasets")
        return

    if args.extract_features:
        make_test_train_val_data_classifier(oversample=args.oversample if args.oversample else False)
        print("Finished extracting features")
        return

    # Set up MLflow
    setup_mlflow(
        tracking_uri=MLFLOW_CONFIG['tracking_uri'],
        experiment_name=MLFLOW_CONFIG['experiment_name'],
    )

    run_name = MLFLOW_CONFIG['run_name']
    experiment_name = MLFLOW_CONFIG['experiment_name']
    model_name = MLFLOW_CONFIG['model_name']
    batch_size = TRAINING_CONFIG['batch_size']
    epochs = TRAINING_CONFIG['epochs']
    learning_rate = MODEL_CONFIG['learning_rate']
    dropout_rate = MODEL_CONFIG['dropout_rate']
    hidden_layers = MODEL_CONFIG['hidden_layers']

    x_train, x_val, x_test, y_train, y_val, y_test = load_data_as_numpy_arrays()

    # Load class information
    with open(os.path.join(DATA_DIR, 'class_info.json'), 'r') as f:
        class_info = json.load(f)
    num_classes = class_info['num_classes']

    input_shape = x_train.shape[1]
    print(f"Extracted {input_shape} features for {x_train.shape[0]} samples")
    print(f"Classification model with {num_classes} movement classes")
    print("Building model...")
    print("Input shape:", input_shape, "Hidden layers:", hidden_layers)


    if args.optimize:
        hidden_layers, best_params = optimize_architecture(args, x_train, y_train, x_val, y_val,
                                                           n_jobs=args.n_jobs or -1,
                                                           n_trials=args.n_trials or 20,
                                                           num_classes=num_classes)
        print("Input shape:", input_shape, "Hidden layers:", input_shape, "x", len(hidden_layers))
        dropout_rate = best_params['dropout_rate']
        learning_rate = best_params['learning_rate']

    model = CasparClassificationModel(
        input_shape=input_shape,
        num_classes=num_classes,
        hidden_layers=hidden_layers,
        dropout_rate=dropout_rate,
    )

    if args.s3_model:
        pre_loaded_model = load_model_from_s3(args.s3_model)
        model.set_model(pre_loaded_model)
    else:
        model.build_model()
        optimizer = tf.keras.optimizers.legacy.Adagrad(learning_rate=learning_rate)
        model.compile_model(optimizer=optimizer, loss='categorical_crossentropy')

    model.summary()

    # Train model
    print("Training classification model...")
    class_weights = ClassifierFeatureExtractor.create_class_weights(y_train)
    print("Class weights:", class_weights)

    trainer = ClassificationModelTrainer(
        model=model,
        experiment_name=experiment_name
    )

    if args.s3_model:
        trainer.test(x_val, y_val, model_name=model_name, run_name=run_name, batch_size=batch_size)
    else:
        trainer.train(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            epochs=epochs,
            batch_size=batch_size,
            model_name=model_name,
            run_name=run_name,
            class_weight=class_weights
        )

    ClassifierFeatureExtractor().analyze_features(x_val, y_val, model.model, visualize=True)


def load_model_from_s3(address):

    print(f"Loading model from {address}...")
    # Parse S3 path
    s3_path = address.replace("s3://", "")
    bucket_name = s3_path.split("/")[0]
    key = "/".join(s3_path.split("/")[1:])

    # Initialize S3 client
    s3_client = boto3.client('s3')

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_file:
        temp_path = temp_file.name

        # Download model to the temporary file
        s3_client.download_file(Bucket=bucket_name, Key=key, Filename=temp_path)

        # Load model from the temporary file
        return tf.keras.models.load_model(temp_path)


def make_old_test_train_val_data():
    # Load data
    print(f"Loading data from {DATASETS_DIR}...")

    unit_actions, game_states, game_boards = load_datasets()

    print(f"Loaded {len(unit_actions)} unit actions and {len(game_states)} game states")

    # Extract features
    print("Extracting features...")
    feature_extractor = FeatureExtractor()
    x, y = feature_extractor.extract_features(unit_actions, game_states, game_boards)
    processor = TrainingDatasetProcessor(x, y, 0.1, 0.1, 7077)
    processor.split_and_save()

def calculate_quality(unit_actions, game_states, game_boards):
    """
    Quality is calculated as the ratio of human players to bot players
    in the game. The higher the ratio, the better the quality.
    """

    human_players = set(
        action['player_id'] for action in unit_actions
        if action.get('is_bot') == 0
    )

    unique_units = {}
    for game_states_array in game_states:
        for game_state in game_states_array:
            unit_key = str(game_state.get('entity_id'))
            if unit_key not in unique_units:
                unique_units[unit_key] = {
                    'isBot': game_state.get('player_id') not in human_players,
                    'bv': game_state.get('bv', 0)
                }

    human_bv = 0
    bot_bv = 0

    for unit in unique_units.values():
        if unit['isBot']:
            bot_bv += unit['bv']
        else:
            human_bv += unit['bv']

    # Mark datasets with low human player presence as lower quality
    adjusted_bot_bv = bot_bv * 4
    if adjusted_bot_bv == 0:
        return 100

    return min((human_bv / adjusted_bot_bv * 100), 50)


def make_tagged_datasets():
    unit_actions_tuple, game_states_tuple, game_boards_tuple = load_datasets()
    for i in range(len(unit_actions_tuple)):
        quality = int(calculate_quality(unit_actions_tuple[i][1], game_states_tuple[i][1], game_boards_tuple[i][1]))
        with open(os.path.join(DATASETS_TAGGED_DIR, f'tagged_dataset-quality={quality:03d}-actions={len(unit_actions_tuple[i][1])}-id={i}.json'), 'w') as f:
            json.dump({
                "unitActions": unit_actions_tuple[i][1],
                "gameStates": game_states_tuple[i][1],
                "gameBoard": game_boards_tuple[i][1],
                "tags": tag_action(unit_actions_tuple[i][1], game_states_tuple[i][1], game_boards_tuple[i][1]),
                "notes": ""
                },
            f)


def make_test_train_val_data_classifier(oversample: bool):
    # Load data
    print(f"Loading data from {DATASETS_DIR}...")
    unit_actions, game_states, game_boards, tags = load_tagged_datasets_classifier()

    print(f"Loaded {len(unit_actions)} unit actions and {len(game_states)} game states")

    # Extract features and classify movements
    print("Extracting features and classifying movements...")
    feature_extractor = ClassifierFeatureExtractor()

    x_acc = []
    y_acc = []

    for i in range(len(unit_actions)):
        x, y = feature_extractor.extract_classification_features(unit_actions[i][1], game_states[i][1], game_boards[i][1], tags[i][1], i)
        x_acc.append(x)
        y_acc.append(y)

    # Save number of classes for model configuration
    num_classes = feature_extractor.num_classes
    x = np.concatenate(x_acc)
    y = np.concatenate(y_acc)
    # Process and save the datasets
    test_percentage = TRAINING_CONFIG["test_size"]
    validation_percentage = TRAINING_CONFIG["validation_size"]
    processor = ClassificationTrainingDatasetProcessor(x, y, test_percentage, validation_percentage, 7)
    dataset_info = processor.split_and_save(oversample)

    zero_value = feature_extractor.features_always_zero(dataset_info['x'])
    feature_extractor.save_feature_statistics(dataset_info['x'], f"_00_{len(x)}_feature_statistics.csv", comments=", ".join(zero_value))

    with open(os.path.join(DATA_DIR, 'class_info.json'), 'w') as f:
        json.dump({
            'num_classes': num_classes,
            'class_mapping': feature_extractor.movement_classes
        }, f)

    classes_frequency = ClassifierFeatureExtractor.describe_class_frequency(dataset_info['y'])
    print("Classes frequencies:", classes_frequency)
    print(f"Feature matrix shape: {dataset_info['x'].shape}")
    print(f"Class labels shape: {dataset_info['y'].shape}")
    print(f"Number of movement classes: {num_classes}")
    print(f"Training data shape: {dataset_info['x_train_shape']}")
    print(f"Validation data shape: {dataset_info['x_val_shape']}")
    print(f"Test data shape: {dataset_info['x_test_shape']}")


def test():
    x_train, x_val, x_test, y_train, y_val, y_test = load_data_as_numpy_arrays()

    with open('../checkpoints/single_input.txt', 'w') as f:
        f.write("private static final double[] x_test = new double[]{" + ", ".join(map(str, x_train[0])) + "};\n")
        f.write("private static final double y_test = " + str(y_train[0]) + ";\n")



if __name__ == "__main__":
    main()
