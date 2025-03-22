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
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.metrics import MeanSquaredError
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec


class CasparModel:
    """
    Neural network model for CASPAR bot move path ranking.
    """

    def __init__(self, input_shape: int, hidden_layers: list = None, dropout_rate: float = 0.2, l2_reg: float = 0.001):
        """
        Initialize the model with given architecture parameters.

        Args:
            input_shape: Number of input features
            hidden_layers: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
            l2_reg: L2 regularization factor
        """
        if hidden_layers is None:
            hidden_layers = [300, 300]

        self.input_shape = input_shape
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.model = self.__build_model()

    def __build_model(self):
        """
        Build the neural network model architecture.

        Returns:
            Compiled Keras model
        """
        model = Sequential()
        layer_num = 0
        # Input layer
        model.add(Dense(self.hidden_layers[0],
                        input_shape=(self.input_shape,),
                        activation='sigmoid',
                        name='hidden_layer_0'))
        model.add(BatchNormalization(name='batch_norm_0'))
        model.add(Dropout(self.dropout_rate, name='dropout_0'))

        # Hidden layers
        for units in self.hidden_layers[1:]:
            layer_num += 1
            model.add(Dense(units,
                            activation='sigmoid',
                            name='hidden_layer_' + str(layer_num)))
            model.add(BatchNormalization(name='batch_norm_' + str(layer_num)))
            model.add(Dropout(self.dropout_rate, name='dropout_' + str(layer_num)))

        # Output layer - single neuron
        model.add(Dense(1, activation='sigmoid', name='output'))

        return model

    def compile_model(self, optimizer='sgd', loss='mse', metrics=None):
        """
        Compile the Keras model with specified optimizer, loss, and metrics.
        SGD for regression
        Args:
            optimizer: Optimizer to use (default: 'sgd')
            loss: Loss function (default: 'mse')
            metrics: List of metrics to track, default is mean squared error
        """
        if metrics is None:
            metrics = [MeanSquaredError()]

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def get_mlflow_signature(self):
        """
        Generate MLflow model signature for the input and output schemas.

        Returns:
            ModelSignature object for MLflow
        """
        input_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, self.input_shape), "features")])

        output_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, 1), "score")])
        return ModelSignature(inputs=input_schema, outputs=output_schema)

    def summary(self):
        """
        Get model summary.
        """
        return self.model.summary()


    def predict(self, features: np.ndarray) -> float:
        """
        Predict the score for given features.
        """
        score = None
        output = self.model.predict(x=features, use_multiprocessing=False, verbose=1)
        if output is not None:
            score = output[0][0]
        return score