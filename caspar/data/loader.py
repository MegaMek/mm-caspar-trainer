# Copyright (C) 2025 The MegaMek Team. All Rights Reserved.
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
import json
import logging
import math
import os
from typing import List, Dict

import numpy as np
from tqdm import tqdm

from caspar.config import RAW_GAMEPLAY_LOGS_DIR, MEK_FILE, DATASETS_TAGGED_DIR, DATA_DIR
from caspar.data.parser.log_parser import RawGameDataParser, ActionAndState

logger = logging.getLogger(__name__)


class LoadUnitState:

    def __init__(self, *args, **kwargs): ...

    @classmethod
    def accepts(cls, action) -> bool:
        # We currently only accept ActionAndState objects
        return isinstance(action, ActionAndState)

    def filter_unit_states(self, round_number: int, action: Dict, unit_states: List[Dict]) -> List[Dict]:
        """
        Returns the unit states
        """
        return unit_states

    def new_instance(self, *args, **kwargs):
        """
        Returns a new instance of the LoadUnitState class
        """
        return LoadUnitState()


class LoadUnitStateDoubleBlind(LoadUnitState):

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed = kwargs.get('seed', 0)

    @classmethod
    def accepts(cls, action) -> bool:
        return isinstance(action, ActionAndState)

    def new_instance(self, *args, **kwargs):
        """
        Returns a new instance of the LoadUnitState class
        """
        return LoadUnitStateDoubleBlind(seed=kwargs.get('seed') or self.seed)

    def filter_unit_states(self, round_number: int, action: Dict, unit_states: List[Dict]) -> List[Dict]:
        """
        Returns the unit states applying double-blind to the action and states.

        Args:
            round_number: The round number
            action: The action dictionary
            unit_states: The list of state builders

        Returns:
            List of delayed unit state builders
        """
        team_id = action['team_id']
        np.random.seed(round_number + self.seed)
        sensor_range_brackets = [16, 32, 48]
        enemy_positions = [(state, np.random.choice(sensor_range_brackets, 1)[0]) for state in unit_states if state['team_id'] != team_id]
        team_positions = [(state, np.random.choice(sensor_range_brackets, 1)[0]) for state in unit_states if state['team_id'] == team_id]

        ret_states = []
        for red, _ in enemy_positions:
            for blue, sensor_range in team_positions:
                distance = math.sqrt((red['x'] - blue['x']) ** 2 + (red['y'] - blue['x']) ** 2)

                if distance < 9 or ((distance >= (sensor_range-16)) and (distance <= sensor_range)):
                    ret_states.append(red)
                    break

        ret_states += [state for state, _ in team_positions]

        return ret_states


def load_tagged_datasets_classifier():
    game_states = []
    unit_actions = []
    game_boards = []
    tags = []
    i = 0
    stats = {
        "total_actions": 0,
        "total_actions_100q": 0,
        "average_actions": 0,
        "average_quality": 0,
        "weighted_average_quality": 0.0
    }

    for root, _, files in os.walk(DATASETS_TAGGED_DIR):
        with tqdm(total=len(files), desc=" " * 60) as t:
            filtered_files = [file for file in files if file.endswith(".json")]

            for file in filtered_files:
                file_path = os.path.join(root, file)
                try:
                    with (open(file_path, "r", encoding='utf-8') as f):
                        value = json.load(f)

                    t.set_description("Loading ..." + file_path[-60:] if len(file_path) > 60 else file_path + " " * (60 - len(file_path)))
                    t.update()

                    _, quality_part, actions_part, _ = file.split("-")
                    quality_value = int(quality_part.split("=")[-1])
                    actions_value = int(actions_part.split("=")[-1])
                    stats["total_actions"] += actions_value
                    stats["average_actions"] += actions_value
                    stats["average_quality"] += quality_value
                    stats["weighted_average_quality"] += quality_value * actions_value
                    stats["total_actions_100q"] += actions_value if quality_value == 100 else 0
                    unit_actions.append((i, value["unitActions"]))
                    game_states.append((i, value["gameStates"]))
                    game_boards.append((i, value["gameBoard"]))
                    tags.append((i, value["tags"]))

                    i += 1
                except Exception as e:
                    logger.error("Error when reading thing", e)

    stats["average_actions"] = stats["average_actions"] / i
    stats["average_quality"] = stats["average_quality"] / i
    stats["weighted_average_quality"] = stats["weighted_average_quality"] / stats["total_actions"]

    return unit_actions, game_states, game_boards, tags


def load_datasets(double_blind: bool = False):
    game_states = []
    unit_actions = []
    game_boards = []
    file_names = []
    data_loader = RawGameDataParser(MEK_FILE)
    i = 0

    for root, _, files in os.walk(RAW_GAMEPLAY_LOGS_DIR):

        filtered_files = [file for file in files if file.endswith(".tsv")]
        if not filtered_files:
            continue

        with tqdm(total=len(filtered_files), desc=" " * 60) as t:

            for file in filtered_files:
                file_path = os.path.join(root, file)

                desc_text = file_path[-60:] if len(file_path) > 60 else file_path + " " * (60 - len(file_path))
                t.set_description(f"{desc_text}")
                t.update()

                loaded_unit_actions, loaded_game_states, loaded_game_board = (
                    load_dataset_from_file(data_loader, file_path, double_blind, seed=i)
                )
                if not loaded_unit_actions:
                    continue

                unit_actions.append((i, loaded_unit_actions))
                game_states.append((i, loaded_game_states))
                game_boards.append((i, loaded_game_board))
                file_names.append(file_path)
                i += 1

    return unit_actions, game_states, game_boards, file_names


def load_dataset_from_file(
        data_loader,
        file_path,
        double_blind: bool = False,
        seed: int = 0
):

    parsed_data = data_loader.parse(str(file_path))

    if parsed_data is None:
        return None, None, None

    unit_state_loader = LoadUnitState() if not double_blind else LoadUnitStateDoubleBlind(seed=seed)
    loaded_unit_actions, loaded_game_states, loaded_game_board = parsed_data.get_actions_and_states(
        unit_state_loader
    )

    return loaded_unit_actions, loaded_game_states, loaded_game_board


def load_data_as_numpy_arrays():
    x_train = np.load(DATA_DIR + '/x_train.npy')
    x_val = np.load(DATA_DIR + '/x_val.npy')
    x_test = np.load(DATA_DIR + '/x_test.npy')

    y_train = np.load(DATA_DIR + '/y_train.npy')
    y_val = np.load(DATA_DIR + '/y_val.npy')
    y_test = np.load(DATA_DIR + '/y_test.npy')

    return x_train, x_val, x_test, y_train, y_val, y_test