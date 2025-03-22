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
import os
from typing import Dict, Any

import mlflow
import mlflow.tensorflow
import numpy as np
import tensorflow as tf

from caspar.model.model import CasparModel
from caspar.config import CHECKPOINT_DIR


class ModelTrainer:
    """
    Handles training of the CASPAR neural network model with MlFlow tracking.
    """

    def __init__(self,
                 model: CasparModel,
                 experiment_name: str = "caspar-model",
                 mlflow_tracking_uri: str = None):
        """
        Initialize the trainer.

        Args:
            model: CasparModel instance
            experiment_name: Name for MlFlow experiment
            mlflow_tracking_uri: URI for MlFlow tracking server
        """
        self.model = model
        self.experiment_name = experiment_name

        # Set up MlFlow
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name)

    def train(self,
              x_train: np.ndarray,
              y_train: np.ndarray,
              x_val: np.ndarray,
              y_val: np.ndarray,
              epochs: int = 16,
              batch_size: int = 32,
              model_name: str = None,
              run_name: str = None,
              hyperparams: Dict[str, Any] = None):
        """
        Train the model and log results with MlFlow.

        Args:
            x_train: Feature matrix
            y_train: Target values
            x_val: Feature matrix
            y_val: Target values
            test_size: Proportion of data to use for testing
            epochs: Number of training epochs
            batch_size: Batch size for training
            model_name: Name for the model
            run_name: Name for the MlFlow run
            hyperparams: Additional hyperparameters to log

        Returns:
            History object from training
        """
        tf.keras.backend.clear_session()
        # Split data into train and test sets

        # Default hyperparams if none provided
        if hyperparams is None:
            hyperparams = {}

        # Start MlFlow run
        with mlflow.start_run(run_name=run_name, log_system_metrics=True) as run:
            # Log model parameters
            self.mlflow_setup_log(batch_size, epochs, hyperparams)

            checkpoint_dir = os.path.join(CHECKPOINT_DIR, f"{run.info.run_id}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, "model_epoch_{epoch:04d}_val_loss_{val_loss:.4f}.h5")
            # Set up early stopping
            checkout_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                save_freq='epoch',
                initial_value_threshold=0.019,
            )

            # Train the model
            history = self.model.model.fit(
                x_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_val, y_val),
                callbacks=[checkout_callback,],
                use_multiprocessing=True,
                verbose=1
            )

            # Evaluate the model
            test_loss, *test_metrics = self.model.model.evaluate(x_val, y_val)

            # Log metrics manually
            mlflow.log_metric("test_loss", test_loss)

            # Log the model with its signature
            model_artifact_path = model_name if model_name else "model"
            model_uri = f"runs:/{run.info.run_id}/{model_artifact_path}"
            mlflow.tensorflow.log_model(
                self.model.model,
                model_uri,
                signature=self.model.get_mlflow_signature()
            )
            mlflow.register_model(model_uri, model_name)

            return history

    def mlflow_setup_log(self, batch_size, epochs, hyperparams):
        mlflow.tensorflow.autolog()
        mlflow.log_param("input_shape", self.model.input_shape)
        mlflow.log_param("hidden_layers", self.model.hidden_layers)
        mlflow.log_param("dropout_rate", self.model.dropout_rate)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        for param_name, param_value in hyperparams.items():
            mlflow.log_param(param_name, param_value)
