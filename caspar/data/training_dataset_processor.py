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
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=self.test_percentage, random_state=self.random_state)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=self.validation_percentage, random_state=self.random_state)

        np.save(DATA_DIR + '/x_train.npy', x_train)
        np.save(DATA_DIR + '/x_val.npy', x_val)
        np.save(DATA_DIR + '/x_test.npy', x_test)
        np.save(DATA_DIR + '/y_train.npy', y_train)
        np.save(DATA_DIR + '/y_val.npy', y_val)
        np.save(DATA_DIR + '/y_test.npy', y_test)
        print("Saved training and test datasets to data directory")

    def apply_normalization_factor(self, x: np.ndarray, factor: np.ndarray):
        if len(x.shape) == 1:
            x[:] *= factor[0]
            return x

        for i in range(x.shape[1]):
            x[:, i] *= factor[i]
        return x

    def min_max_normalize_factor(self, x: np.ndarray) -> float:
        max_value = x.max()
        min_value = x.min()
        if max_value == min_value:
            return 1
        return 1 / (max_value - min_value)

    def normalize_minmax(self, x: np.ndarray) :
        if len(x.shape) == 1 :
            minmax_factor = np.zeros(1)
            minmax_factor[0] = self.min_max_normalize_factor(x)
            x[:] *= minmax_factor[0]

            return x, minmax_factor
        else:
            minmax_factor = np.zeros(x.shape[1])
            for i in range(x.shape[1]):
                minmax_factor[i] = self.min_max_normalize_factor(x[:, i])
                x[:, i] *= minmax_factor[i]

            return x, minmax_factor
