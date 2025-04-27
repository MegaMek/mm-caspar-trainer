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
import os
from sklearn.model_selection import train_test_split
import pickle

from caspar.config import DATA_DIR


class TrainingDatasetProcessor:
    def __init__(self, x: np.ndarray, y: np.ndarray, test_percentage, validation_percentage, random_state):
        self.x = x
        self.y = y
        self.test_percentage = test_percentage
        self.validation_percentage = validation_percentage
        self.random_state = random_state

    def split_and_save(self, *args, **kwargs):
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

    def split_and_save(self, oversample: bool):
        """
        Normalizes the dataset, then split the data into training, validation, and test sets and save them.
        """
        # Remove all indices from X that sum to 0
        # and remove the corresponding entries from Y
        non_zero_indices = ~np.all(self.x == 0, axis=1)
        clean_x = self.x[non_zero_indices]
        clean_y = self.y[non_zero_indices]
        
        save_ratios = []
        for i in range(len(clean_x[0])):
            max_dim = np.max(clean_x[:, i])
            min_dim = np.min(clean_x[:, i])

            clean_x[:, i] = (clean_x[:, i] - min_dim) / (max_dim - min_dim + 1.0e-7)
            save_ratios.append((min_dim, max_dim))

        save_normalization_params(save_ratios, os.path.join(DATA_DIR, 'min_max_feature_normalization.csv'))

        balanced_x, balanced_y = normalize_classes_distribution(clean_x, clean_y, random_state=self.random_state, oversample=oversample)

        with open(os.path.join(DATA_DIR, 'training_data.csv'), 'w') as f:
            for (vals, res) in zip(balanced_x, balanced_y):
                if np.sum(vals) == 0:
                    continue
                a = [str(v) for v in vals]
                f.write(",".join(a) + "," + str(np.argmax(res)) + "\n")

        # Split data into training, validation, and test sets
        _x_train, x_test, _y_train, y_test = train_test_split(
            balanced_x, balanced_y, test_size=self.test_percentage, random_state=self.random_state
        )

        x_train, x_val, y_train, y_val = train_test_split(
            _x_train, _y_train, test_size=self.validation_percentage, random_state=self.random_state
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
            'y_test_shape': y_test.shape,
            'x': balanced_x,
            'y': balanced_y,
        }


def normalize_classes_distribution(clean_x, clean_y, random_state, oversample=True):
    """
    Normalize the classes distribution in the dataset.
    This function ensures that each class is represented equally in the dataset.

    Args:
        clean_x: Feature matrix
        clean_y: One-hot encoded class labels
        random_state: Random state for reproducibility
        oversample: If True, oversample minority classes to match the majority class.
                   If False, undersample majority classes to match the minority class.

    Returns:
        Tuple of (normalized_x, normalized_y)
    """
    np.random.seed(random_state)
    y_indices = np.argmax(clean_y, axis=1) if len(clean_y.shape) > 1 else clean_y
    unique_classes, class_counts = np.unique(y_indices, return_counts=True)
    target_samples = max(class_counts) if oversample else min(class_counts)
    class_indices = {cls: np.where(y_indices == cls)[0] for cls in unique_classes}

    balanced_x = []
    balanced_y = []

    for cls in unique_classes:
        cls_indices = class_indices[cls]

        if len(cls_indices) == target_samples:
            balanced_indices = cls_indices
        elif len(cls_indices) < target_samples:
            if oversample:
                additional_indices = np.random.choice(
                    cls_indices,
                    size=target_samples - len(cls_indices),
                    replace=True
                )
                balanced_indices = np.concatenate([cls_indices, additional_indices])
            else:
                # Should not happen when undersample=False
                balanced_indices = cls_indices
        else:
            # Too many samples - undersample
            balanced_indices = np.random.choice(
                cls_indices,
                size=target_samples,
                replace=False
            )

        balanced_x.append(clean_x[balanced_indices])
        balanced_y.append(clean_y[balanced_indices])

    # Combine all balanced classes
    balanced_x = np.vstack(balanced_x)
    balanced_y = np.vstack(balanced_y)

    # Shuffle the balanced dataset
    shuffle_indices = np.random.permutation(len(balanced_x))
    balanced_x = balanced_x[shuffle_indices]
    balanced_y = balanced_y[shuffle_indices]

    strategy_name = "oversampling minority classes" if oversample else "undersampling majority classes"
    print(f"Normalized class distribution using {strategy_name}. All classes now have {target_samples} samples.")

    # Return the new balanced arrays instead of modifying in place
    return balanced_x, balanced_y

# Save normalization parameters
def save_normalization_params(save_ratios, file_path):
    with open(file_path, 'w') as f:
        f.write(f"feature,min,max\n")
        for i, (min_val, max_val) in enumerate(save_ratios):
            f.write(f"{i},{min_val},{max_val}\n")

# Load normalization parameters
def load_normalization_params(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        min_max_values = []
        for line in lines[1:]:
            parts = line.strip().split(',')
            min_max_values.append((float(parts[1]), float(parts[2])))
    return min_max_values
