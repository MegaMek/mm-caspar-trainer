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
from sklearn.model_selection import train_test_split

from caspar.config import DATA_DIR


class TrainingDatasetProcessor:
    def __init__(self, x: np.ndarray, y: np.ndarray, test_percentage, validation_percentage, random_state):
        self.x = x
        self.y = y
        self.test_percentage = test_percentage
        self.validation_percentage = validation_percentage
        self.random_state = random_state

    def split_and_save(self):
        # remove all indices from X that sum to 0
        # and remove the corresponding entries from Y
        clean_x = self.x[~np.all(self.x == 0, axis=1)]
        clean_y = self.y[~np.all(self.x == 0, axis=1)]

        x_train, x_test, y_train, y_test = train_test_split(clean_x, clean_y, test_size=self.test_percentage, random_state=self.random_state)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=self.validation_percentage, random_state=self.random_state)

        np.save(DATA_DIR + '/x_train.npy', x_train)
        np.save(DATA_DIR + '/x_val.npy', x_val)
        np.save(DATA_DIR + '/x_test.npy', x_test)
        np.save(DATA_DIR + '/y_train.npy', y_train)
        np.save(DATA_DIR + '/y_val.npy', y_val)
        np.save(DATA_DIR + '/y_test.npy', y_test)
        print("Saved training and test datasets to data directory")


class ClassificationTrainingDatasetProcessor(TrainingDatasetProcessor):
    """
    Extends the TrainingDatasetProcessor to handle classification data.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, test_percentage, validation_percentage, random_state):
        """
        Initialize the processor.

        Args:
            x: Feature matrix
            y: One-hot encoded class labels
            test_percentage: Percentage of data to use for testing
            validation_percentage: Percentage of training data to use for validation
            random_state: Random state for reproducibility
        """
        super().__init__(x, y, test_percentage, validation_percentage, random_state)

    def split_and_save(self):
        """
        Split the data into training, validation, and test sets and save them.
        """
        # Remove all indices from X that sum to 0
        # and remove the corresponding entries from Y
        non_zero_indices = ~np.all(self.y == 0, axis=1)
        clean_x = self.x[non_zero_indices]
        clean_y = self.y[non_zero_indices]

        # Split data into training, validation, and test sets
        x_train, x_test, y_train, y_test = train_test_split(
            clean_x, clean_y, test_size=self.test_percentage, random_state=self.random_state
        )

        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=self.validation_percentage, random_state=self.random_state
        )

        # Save the datasets
        np.save(DATA_DIR + '/x_train.npy', x_train)
        np.save(DATA_DIR + '/x_val.npy', x_val)
        np.save(DATA_DIR + '/x_test.npy', x_test)
        np.save(DATA_DIR + '/y_train.npy', y_train)
        np.save(DATA_DIR + '/y_val.npy', y_val)
        np.save(DATA_DIR + '/y_test.npy', y_test)

        print("Saved training, validation, and test datasets to data directory")

        # Return information about the datasets
        return {
            'x_train_shape': x_train.shape,
            'y_train_shape': y_train.shape,
            'x_val_shape': x_val.shape,
            'y_val_shape': y_val.shape,
            'x_test_shape': x_test.shape,
            'y_test_shape': y_test.shape
        }