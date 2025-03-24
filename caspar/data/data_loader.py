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
from typing import Tuple, List, Dict
from enum import Enum
import re

import numpy as np

from caspar.config import DATA_DIR, DATASETS_DIR, MEK_FILE
from caspar.data.game_board import GameBoardRepr


class LineType(Enum):
    """Enum for different header types in the dataset file"""
    MOVE_ACTION_HEADER_V1 = "PLAYER_ID\tENTITY_ID\tCHASSIS\tMODEL\tFACING\tFROM_X\tFROM_Y\tTO_X\tTO_Y\tHEXES_MOVED\tDISTANCE\tMP_USED\tMAX_MP\tMP_P\tHEAT_P\tARMOR_P\tINTERNAL_P\tJUMPING\tPRONE\tLEGAL\tSTEPS"
    MOVE_ACTION_HEADER_V2 = re.compile(r"^PLAYER_ID\tENTITY_ID\tCHASSIS\tMODEL\tFACING\tFROM_X\tFROM_Y\tTO_X\tTO_Y\tHEXES_MOVED\tDISTANCE\tMP_USED\tMAX_MP\tMP_P\tHEAT_P\tARMOR_P\tINTERNAL_P\tJUMPING\tPRONE\tLEGAL\tSTEPS\tTEAM_ID.*$")
    MOVE_ACTION_HEADER_V3 = re.compile(r"^PLAYER_ID\tENTITY_ID\tCHASSIS\tMODEL\tFACING\tFROM_X\tFROM_Y\tTO_X\tTO_Y\tHEXES_MOVED\tDISTANCE\tMP_USED\tMAX_MP\tMP_P\tHEAT_P\tARMOR_P\tINTERNAL_P\tJUMPING\tPRONE\tLEGAL\tSTEPS\tTEAM_ID.*$")
    STATE_HEADER_V1 = "ROUND\tPHASE\tPLAYER_ID\tENTITY_ID\tCHASSIS\tMODEL\tTYPE\tROLE\tX\tY\tFACING\tMP\tHEAT\tPRONE\tAIRBORNE\tOFF_BOARD\tCRIPPLED\tDESTROYED\tARMOR_P\tINTERNAL_P\tDONE"
    STATE_HEADER_V2 = "ROUND\tPHASE\tTEAM_ID\tPLAYER_ID\tENTITY_ID\tCHASSIS\tMODEL\tTYPE\tROLE\tX\tY\tFACING\tMP\tHEAT\tPRONE\tAIRBORNE\tOFF_BOARD\tCRIPPLED\tDESTROYED\tARMOR_P\tINTERNAL_P\tDONE"
    STATE_HEADER_V3 = re.compile(r"^ROUND\tPHASE\tPLAYER_ID\tENTITY_ID\tCHASSIS\tMODEL\tTYPE\tROLE\tX\tY\tFACING\tMP\tHEAT\tPRONE\tAIRBORNE\tOFF_BOARD\tCRIPPLED\tDESTROYED\tARMOR_P\tINTERNAL_P\tDONE.*$")
    ATTACK_ACTION_HEADER = "PLAYER_ID\tENTITY_ID\tCHASSIS\tMODEL\tFACING\tFROM_X\tFROM_Y\tTO_X\tTO_Y\tHEXES_MOVED\tDISTANCE\tMP_USED\tMAX_MP\tMP_P\tHEAT_P\tARMOR_P\tINTERNAL_P\tJUMPING\tPRONE\tLEGAL\tSTEPS"
    ROUND = "ROUND"
    ACTION_HEADER = "PLAYER_ID\tENTITY_ID"
    BOARD = "BOARD_NAME\tWIDTH\tHEIGHT"


class ActionAndState:
    """Container for an action and its corresponding state"""
    def __init__(self, round_number: int, board: GameBoardRepr, action: Dict, state_builders: List):
        self.round_number = round_number
        self.board = board
        self.action = action
        self.state_builders = state_builders

    @property
    def states(self):
        return [builder.build() for builder in self.state_builders]



class DataLoader:
    """
    Class that parses a dataset file into unit actions and states
    """

    def __init__(self, mek_extras_file: str):
        self.meks_extras = dict()
        self._action_and_states = list()
        self.mek_extras_file = mek_extras_file
        self.entities = dict()
        self.game_board = None
        self.__load_meks_extras()

    def __load_meks_extras(self):
        data = dict()
        with open(self.mek_extras_file, "r") as meks_extras_file:
            for line in meks_extras_file:
                line = line.strip()
                if line == "":
                    continue
                if line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue

                if parts[0] == "Chassis\tModel":
                    continue

                data[f'{parts[0]} {parts[1]}'] = {
                    "bv": int(parts[2]),
                    "armor": int(parts[3]),
                    "internal": int(parts[4]),
                    "ecm": int(parts[5]),
                    "max_range":int(parts[6]),
                    "total_damage": int(parts[7]),
                    "role": parts[8],
                }

        self.meks_extras = data

    def parse(self, file_path: str) -> 'DataLoader':
        """
        Parses a dataset from a file. Can be chained with other parse calls to create a large single training dataset.

        Args:
            file_path: Path to the file to parse

        Returns:
            The parser instance
        """
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()
                content = ''.join(lines)
                # Parse the game board first
                self.game_board = GameBoardRepr(content)
                print(f"Parsed game board: {self.game_board.width}x{self.game_board.height}")

            self._action_and_states = list()
            self.entities = dict()

            board = None
            i = 0
            while i < len(lines):
                line = lines[i].strip()

                # Skip empty lines
                if not line:
                    i += 1
                    continue

                # Check for action headers
                if (line == LineType.MOVE_ACTION_HEADER_V1.value or
                        self._matches_pattern(line, LineType.MOVE_ACTION_HEADER_V2.value) or
                        self._matches_pattern(line, LineType.MOVE_ACTION_HEADER_V3.value)):

                    # Parse action line
                    i += 1
                    if i >= len(lines):
                        break

                    action_line = lines[i].strip()
                    action = self._parse_unit_action(action_line.split('\t'))

                    # Parse state block
                    i += 1
                    if i >= len(lines):
                        raise RuntimeError(f"Invalid line after action: {action_line}")

                    state_header = lines[i].strip()
                    if not (state_header == LineType.STATE_HEADER_V1.value or
                            state_header == LineType.STATE_HEADER_V2.value or
                            self._matches_pattern(state_header, LineType.STATE_HEADER_V3.value)):
                        raise RuntimeError(f"Invalid state header after action: {state_header}")

                    states = []
                    current_round = None
                    i += 1

                    while i < len(lines):
                        line = lines[i].strip()

                        # Check for end of state block
                        if not line or line.startswith(LineType.ACTION_HEADER.value) or line.startswith(
                                LineType.ROUND.value):
                            break

                        _round, state = self._parse_unit_state(line.split('\t'))

                        if current_round is None:
                            current_round = _round
                        elif current_round != _round:
                            raise RuntimeError("State block has inconsistent rounds")

                        states.append(state)
                        i += 1

                    if current_round is None:
                        raise RuntimeError("State block has no valid states")

                    self._action_and_states.append(ActionAndState(current_round, self.game_board, action, states))

                    # We're now at the next action header or at the end
                    continue

                # If none of the above, move to next line
                i += 1

            return self

        except Exception as e:
            raise RuntimeError(f"Error reading file {file_path}: {str(e)}") from e

    @classmethod
    def _matches_pattern(cls, line: str, pattern) -> bool:
        """Check if a line matches a regex pattern"""
        if isinstance(pattern, re.Pattern):
            return pattern.match(line) is not None
        return line.startswith(pattern)

    def get_actions_and_states(self) -> Tuple[List[Dict], List[List[Dict]], List[GameBoardRepr]]:
        """
        Returns the parsed actions and states in a format similar to the load_data method.

        Returns:
            Tuple containing lists of unit actions and game states
        """
        unit_actions = []
        game_states = []
        game_boards = []
        for action_and_state in self._action_and_states:
            unit_actions.append(action_and_state.action)
            game_states.append(action_and_state.states)
            game_boards.append(action_and_state.board)
        return unit_actions, game_states, game_boards

    def _parse_unit_action(self, data: List[str]) -> Dict:
        """
        Parse a single unit action line from the TSV.

        Args:
            data: List of string values from TSV

        Returns:
            Dictionary containing unit action data
        """
        if len(data) < 20:  # Check we have enough fields
            return {}

        mek = self.meks_extras.get(f'{data[2]} {data[3]}', {})

        action = {
            'player_id': int(data[0]),
            'entity_id': int(data[1]),
            'chassis': data[2],
            'model': data[3],
            'facing': int(data[4]),
            'from_x': int(data[5]),
            'from_y': int(data[6]),
            'x': int(data[5]),
            'y': int(data[6]),
            'to_x': int(data[7]),
            'to_y': int(data[8]),
            'hexes_moved': int(data[9]),
            'distance': int(data[10]),
            'mp_used': int(data[11]),
            'mp': int(data[11]),
            'max_mp': int(data[12]),
            'mp_p': float(data[13]),
            'heat_p': float(data[14]),
            'armor_p': float(data[15]),
            'internal_p': float(data[16]),
            'jumping': int(data[17]),
            'prone': int(data[18]),
            'legal': int(data[19]),
            'steps': data[20] if len(data) > 20 else "",
            'team_id': int(data[21]) if len(data) > 21 else 0,
            'chance_of_failure': float(data[22]) if len(data) > 22 else 0.0,
            'is_bot': int(data[23]) if len(data) > 23 else 0,
            'armor': mek.get("armor", -1),
            'internal': mek.get("internal", -1),
            'max_range': mek.get("max_range", -1),
            'total_damage': mek.get("total_damage", -1),
            'ecm': mek.get("ecm", 0),
            'type': mek.get('role', None),
            'role': mek.get('role', None),
            'bv': mek.get('bv', -1)
        }
        self.entities[action['entity_id']] = action
        return action

    def _parse_unit_state(self, data: List[str]) -> tuple[int, 'DelayedUnitStateBuilder']:
        """
        Parse a single unit state line from the TSV.

        Args:
            data: List of string values from TSV

        Returns:
            Dictionary containing unit state data
        """
        if len(data) < 20:  # Check we have enough fields
            raise RuntimeError(f"Invalid state data: {data}")

        return int(data[0]), DelayedUnitStateBuilder(data, self)


class DelayedUnitStateBuilder:

    def __init__(self, data: list, data_loader: DataLoader):
        self.data_loader = data_loader
        self._unit_state = dict()
        self.__data = "\t".join(data)

    def build(self) -> dict:
        data = self.__data.split("\t")
        try:
            action = self.data_loader.entities.get(int(data[3]), None)
        except ValueError:
            action = self.data_loader.entities.get(int(data[2]), None)
        mek = {}
        chassis = data[4]
        model = None
        if not action:
            for key in self.data_loader.meks_extras.keys():
                if key.startswith(chassis):
                    mek = self.data_loader.meks_extras[key]
                    model = key.split(chassis, 1)[-1]
                    break
        else:
            model = action["model"]
            mek = self.data_loader.meks_extras[f'{chassis} {model}']

        return {
            'round': int(data[0]),
            'phase': data[1],
            'player_id': int(data[2]),
            'entity_id': int(data[3]),
            'chassis': chassis,
            'model': model,
            'type': data[6],
            'role': data[7],
            'x': int(data[8]),
            'y': int(data[9]),
            'facing': int(data[10]),
            'mp': float(data[11]),
            'heat': float(data[12]),
            'heat_p': float(data[12]) / (40 if "Mek" in data[6] else 999),
            'prone': int(data[13]),
            'airborne': int(data[14]),
            'off_board': int(data[15]),
            'crippled': int(data[16]),
            'destroyed': int(data[17]),
            'armor_p': float(data[18]),
            'internal_p': float(data[19]),
            'done': int(data[20]),
            'team_id': int(data[23]),
            'armor': mek.get("armor", -1),
            'internal': mek.get("internal", -1),
            'max_range': mek.get("max_range", -1),
            'total_damage': mek.get("total_damage", -1),
            'ecm': mek.get("ecm", 0),
            'bv': mek.get('bv', -1)
        }


def load_datasets():
    game_states = []
    unit_actions = []
    game_boards = []
    data_loader = DataLoader(MEK_FILE)
    # Process each path in args.data
    for path in [DATASETS_DIR]:
        if os.path.isfile(path):
            # Process a single file
            try:
                loaded_unit_actions, loaded_game_states, loaded_game_boards = data_loader.parse(path).get_actions_and_states()
                unit_actions.extend(loaded_unit_actions)
                game_states.extend(loaded_game_states)
                game_boards.append(loaded_game_boards)
                print(f"Loaded file: {path}")
            except Exception as e:
                print(f"Failed to load {path}: {str(e)}")
        elif os.path.isdir(path):
            # Walk through directory and process all files
            for root, _, files in os.walk(path):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        loaded_unit_actions, loaded_game_states, loaded_game_boards = data_loader.parse(path).get_actions_and_states()
                        unit_actions.extend(loaded_unit_actions)
                        game_states.extend(loaded_game_states)
                        game_boards.append(loaded_game_boards)
                        print(f"Loaded file: {file_path}")
                    except Exception as e:
                        print(f"Failed to load {file_path}: {str(e)}")
        else:
            print(f"Path not found: {path}")

    return unit_actions, game_states, game_boards


def load_data_as_numpy_arrays():
    x_train = np.load(DATA_DIR + '/x_train.npy')
    x_val = np.load(DATA_DIR + '/x_val.npy')
    x_test = np.load(DATA_DIR + '/x_test.npy')

    y_train = np.load(DATA_DIR + '/y_train.npy')
    y_val = np.load(DATA_DIR + '/y_val.npy')
    y_test = np.load(DATA_DIR + '/y_test.npy')

    return x_train, x_val, x_test, y_train, y_val, y_test