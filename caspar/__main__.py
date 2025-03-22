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

from dotenv import load_dotenv
import tensorflow as tf

from caspar.data.data_loader import load_datasets, load_data_as_numpy_arrays
from caspar.data import FeatureExtractor
from caspar.model.model import CasparModel
from caspar.training.trainer import ModelTrainer
from caspar.utils.mlflow_utils import setup_mlflow
from caspar.config import DATASETS_DIR, MEK_FILE, MODEL_CONFIG, TRAINING_CONFIG, MLFLOW_CONFIG, DOTENV_PATH
from caspar.hyperparameter_search import optimize_architecture
from caspar.data.training_dataset_processor import TrainingDatasetProcessor


load_dotenv(DOTENV_PATH)


def parse_args():
    parser = argparse.ArgumentParser(description='Train CASPAR neural network model')
    parser.add_argument('--mekfile', type=str, default=MEK_FILE,
                        help='Path to Mek file (txt)')
    parser.add_argument('--data', action='store_true',
                        help='Recompile the datasets')
    parser.add_argument('--convert-model', type=str, required=False,
                        help='Convert a specific model `name` to tflite so it can be used in the game')
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

    if args.convert_model:
        convert_data(args.convert_model)
        print("Finished converting data")
        return

    x_train, x_val, x_test, y_train, y_val, y_test = load_data_as_numpy_arrays()

    run_name = args.run_name

    # Create and compile model
    input_shape = x_train.shape[1]
    print(f"Extracted {x_train.shape[1]} features for {x_train.shape[0]} samples")
    print("Building model...")

    if args.optimize:
        hidden_layers, best_params = optimize_architecture(args, x_train, y_train, x_val, y_val, n_jobs=args.n_jobs, n_trials=args.n_trials)
        print("Input shape:", input_shape, "Hidden layers:", input_shape, "x", len(hidden_layers))

        # Create model with best architecture
        model = CasparModel(
            input_shape=input_shape,
            hidden_layers=hidden_layers,
            dropout_rate=best_params['dropout_rate'],
        )

        # Use optimized learning rate
        optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=best_params['learning_rate'], momentum=0.1)
        model.compile_model(optimizer=optimizer)

    else:
        print("Input shape:", input_shape, "Hidden layers:", args.hidden_layers)
        model = CasparModel(
            input_shape=input_shape,
            hidden_layers=args.hidden_layers,
            dropout_rate=args.dropout_rate,
        )
        optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=args.learning_rate, momentum=0.9, nesterov=True)
        model.compile_model(optimizer=optimizer)

    model.summary()

    # Train model
    print("Training model...")

    trainer = ModelTrainer(
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
        model_name=f"CASPAR-00",
        run_name=run_name,
    )

def make_test_train_val_data():
    # Load data
    print(f"Loading data from {DATASETS_DIR}...")

    unit_actions, game_states = load_datasets()

    print(f"Loaded {len(unit_actions)} unit actions and {len(game_states)} game states")

    # Extract features
    print("Extracting features...")
    feature_extractor = FeatureExtractor()
    x, y = feature_extractor.extract_features(unit_actions, game_states)
    processor = TrainingDatasetProcessor(x, y, 0.1, 0.1, 7077)
    processor.split_and_save()


def convert_data(model_name):
    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model("models:/" + model_name)
    tflite_model = converter.convert()

    # Save the model.
    with open(model_name + '.tflite', 'wb') as f:
        f.write(tflite_model)


if __name__ == "__main__":
    main()
