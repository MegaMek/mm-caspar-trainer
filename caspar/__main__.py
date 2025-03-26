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

from dotenv import load_dotenv
import tensorflow as tf

from caspar.data.data_loader import load_datasets, load_data_as_numpy_arrays
from caspar.data import FeatureExtractor
from caspar.data.feature_extractor import ClassifierFeatureExtractor
from caspar.model.model import CasparClassificationModel
from caspar.training.trainer import ClassificationModelTrainer
from caspar.utils.mlflow_utils import setup_mlflow
from caspar.config import DATASETS_DIR, MEK_FILE, MODEL_CONFIG, TRAINING_CONFIG, MLFLOW_CONFIG, DOTENV_PATH, DATA_DIR
from caspar.data.training_dataset_processor import TrainingDatasetProcessor, ClassificationTrainingDatasetProcessor

load_dotenv(DOTENV_PATH)


def parse_args():
    parser = argparse.ArgumentParser(description='Train CASPAR neural network model')
    parser.add_argument('--mekfile', type=str, default=MEK_FILE,
                        help='Path to Mek file (txt)')
    parser.add_argument('--data', action='store_true',
                        help='Recompile the datasets')
    parser.add_argument('--epochs', type=int, default=TRAINING_CONFIG['epochs'],
                        help='Number of training epochs')
    parser.add_argument('--dropout-rate', type=float, default=MODEL_CONFIG['dropout_rate'],
                        help='Dropout rate')
    parser.add_argument('--hidden-layers',  nargs='+', type=int, default=MODEL_CONFIG['hidden_layers'],
                        help='Hidden layers formats')
    parser.add_argument('--batch-size', type=int, default=TRAINING_CONFIG['batch_size'],
                        help='Batch size for training')
    parser.add_argument('--experiment-name', type=str, default=MLFLOW_CONFIG['experiment_name'],
                        help='MLflow experiment name')
    parser.add_argument('--optimize', action='store_true',
                        help='Run hyperparameter optimization')
    parser.add_argument('--n-trials', type=int, default=100,
                        help='Number of trials for hyperparameter optimization')
    parser.add_argument('--test-size', type=float, default=TRAINING_CONFIG['test_size'],
                        help='Percent from 0 to 1 of the dataset to use for testing')
    parser.add_argument('--n-jobs', type=int, default=-1,
                        help='Number of parallel jobs for optimization (-1 uses all cores)')
    parser.add_argument('--learning-rate', type=float, default=MODEL_CONFIG['learning_rate'],
                        help='Learning rate for optimization')
    parser.add_argument('--run-name', type=str, required=False,
                        help='Name of the run')

    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()

    # Set up MLflow
    setup_mlflow(
        tracking_uri=MLFLOW_CONFIG['tracking_uri'],
        experiment_name=args.experiment_name
    )

    if args.data:
        make_test_train_val_data()
        print("Finished compiling datasets")
        return


    x_train, x_val, x_test, y_train, y_val, y_test = load_data_as_numpy_arrays()

    run_name = args.run_name

    # Load class information
    with open(os.path.join(DATA_DIR, 'class_info.json'), 'r') as f:
        class_info = json.load(f)
    num_classes = class_info['num_classes']

    run_name = args.run_name

    # Create and compile model
    input_shape = x_train.shape[1]
    print(f"Extracted {input_shape} features for {x_train.shape[0]} samples")
    print(f"Classification model with {num_classes} movement classes")
    print("Building model...")
    print("Input shape:", input_shape, "Hidden layers:", args.hidden_layers)
    model = CasparClassificationModel(
        input_shape=input_shape,
        num_classes=num_classes,
        hidden_layers=args.hidden_layers,
        dropout_rate=args.dropout_rate,
    )
    model.build_model()
    optimizer = tf.keras.optimizers.legacy.Adagrad(learning_rate=args.learning_rate)
    model.compile_model(optimizer=optimizer, loss='categorical_crossentropy')

    model.summary()

    # Train model
    print("Training classification model...")
    class_weights = ClassifierFeatureExtractor.create_class_weights(y_train)
    print("Class weights:", class_weights)

    trainer = ClassificationModelTrainer(
        model=model,
        experiment_name=args.experiment_name
    )

    trainer.train(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_name=f"CASPAR-Classification-00",
        run_name=run_name,
        class_weight=class_weights
    )

def make_test_train_val_data():
    # Load data
    print(f"Loading data from {DATASETS_DIR}...")

    unit_actions, game_states, game_boards = load_datasets()

    print(f"Loaded {len(unit_actions)} unit actions and {len(game_states)} game states")

    # Extract features
    print("Extracting features...")
    feature_extractor = FeatureExtractor()
    x, y = feature_extractor.extract_features(unit_actions, game_states)
    processor = TrainingDatasetProcessor(x, y, 0.1, 0.1, 7077)
    processor.split_and_save()


def make_test_train_val_data_classifier():
    # Load data
    print(f"Loading data from {DATASETS_DIR}...")
    unit_actions, game_states, game_boards = load_datasets()
    print(f"Loaded {len(unit_actions)} unit actions and {len(game_states)} game states")

    # Extract features and classify movements
    print("Extracting features and classifying movements...")
    feature_extractor = ClassifierFeatureExtractor()
    x, y = feature_extractor.extract_classification_features(unit_actions, game_states, game_boards)

    # Save number of classes for model configuration
    num_classes = feature_extractor.num_classes

    # Process and save the datasets
    processor = ClassificationTrainingDatasetProcessor(x, y, 0.1, 0.1, 7077)
    dataset_info = processor.split_and_save()
    with open(os.path.join(DATA_DIR, 'class_info.json'), 'w') as f:
        json.dump({
            'num_classes': num_classes,
            'class_mapping': feature_extractor.movement_classes
        }, f)

    print(f"Feature matrix shape: {x.shape}")
    print(f"Class labels shape: {y.shape}")
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
    # make_test_train_val_data_classifier()
    # main()
    # make_test_train_val_data()
    # test()

    unit_actions, game_states, game_boards = load_datasets()
    with open('game_log.json', 'w') as f:
        json.dump({
            "unitActions": unit_actions[0][1],
            "gameStates": game_states[0][1],
            "gameBoard": game_boards[0][1]},
            f)
