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
import logging
import os
from typing import Any
from typing import Tuple, List, Dict, Union, Optional

from tqdm import tqdm

from caspar.config import RAW_GAMEPLAY_LOGS_DIR, MEK_FILE
from caspar.data.game_board import GameBoardRepr
from caspar.data.parser.base_parser import ParserFactory, BaseParser

logger = logging.getLogger(__name__)


class ActionAndState:
    """Container for an action and its corresponding state"""
    def __init__(self, round_number: int, action: Dict, state_builders: list['DelayedUnitStateBuilder']):
        self.round_number = round_number
        self.action = action
        self.state_builders = state_builders

    @property
    def states(self):
        return [builder.build() for builder in self.state_builders]


class AttackActionAndState:
    """Container for an attack action and its corresponding state"""
    def __init__(self, round_number: int, attack_actions: list[dict], state_builders: list['DelayedUnitStateBuilder']):
        self.round_number = round_number
        self.attack_actions = attack_actions
        self.state_builders = state_builders

    @property
    def states(self):
        return [builder.build() for builder in self.state_builders]


class RawGameDataParser:
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
        cached_parser = None
        with open(self.mek_extras_file, "r") as meks_extras_file:
            for line in meks_extras_file:
                if not cached_parser:
                    if not (cached_parser := ParserFactory.get_parser(line)):
                        continue
                if cached_parser is not None and cached_parser.is_parseable_line(line):
                    if unit := cached_parser.parse(line):
                        data[f'{unit["chassis"]} {unit["model"]}'] = unit

        self.meks_extras = data

    def parse(self, file_path: str) -> Optional['RawGameDataParser']:
        """
        Parses a dataset from a file. Can be chained with other parse calls to create a large single training dataset.

        Args:
            file_path: Path to the file to parse

        Returns:
            The parser instance
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
        except UnicodeDecodeError:
            print(f"Error reading file {file_path}: UnicodeDecodeError. Skipping file.")
            return None

        self.game_board = GameBoardRepr(lines)
        self._action_and_states = list()
        self.entities = dict()
        states = []
        attacks = []
        action = None
        current_round = -1
        cached_parser = None
        for line in lines:
            line = line.strip()
            if not line:  # Skip empty lines
                continue

            parser = ParserFactory.get_parser(line)
            if parser is None:
                continue
            if cached_parser is None:
                cached_parser = parser
            elif parser.TYPE == cached_parser.TYPE:  # this is a latch that keeps the parser for the same type
                parser = cached_parser
            else:
                cached_parser = None

            if data := parser.parse(line):
                match data["_type"]:
                    case "UnitState":
                        states.append(DelayedUnitStateBuilder(data, self))
                        current_round = data['round']

                    case "UnitAction":
                        self._commit_previous_action(current_round, action, attacks, states)
                        self.entity_post_processing(data)
                        action = data
                        attacks = []
                        states = []

                    case "UnitAttack":
                        self._commit_previous_action(current_round, action, attacks, states)
                        attacks.append(data)
                        action = None
                        states = []

        self._commit_previous_action(current_round, action, attacks, states)

        return self

    def entity_post_processing(self, unit: dict[str, Any]):
        mek = self.meks_extras.get(f'{unit["chassis"]} {unit["model"]}', {})
        for mek_key in mek:
            if mek_key == "version" or mek_key == "_type":
                continue
            if mek_key not in unit:
                unit[mek_key] = mek[mek_key]
            elif mek[mek_key] is not None and mek[mek_key] != unit[mek_key] and unit[mek_key] == -1:
                unit[mek_key] = mek[mek_key]

        self.entities[unit['id']] = unit

    def _commit_previous_action(self, current_round: int, action: dict, attacks: list[dict], states: list['DelayedUnitStateBuilder']):
        if attacks:
            self._action_and_states.append(AttackActionAndState(current_round, attacks, states))
        elif action:
            self._action_and_states.append(ActionAndState(current_round, action, states))

    def get_actions_and_states(
            self,
            load_unit_state
    ) -> Tuple[List[Dict], List[List[Dict]], Union[GameBoardRepr, Dict]]:
        """
        Returns the parsed actions and states in a format similar to the load_data method.

        Returns:
            Tuple containing lists of unit actions and game states
        """
        unit_actions = []
        game_states = []
        game_board = self.game_board.to_dict()
        for action_and_state in self._action_and_states:
            # We currently don't do anything with attack actions, so we skip them
            if not load_unit_state.accepts(action_and_state):
                continue

            unit_actions.append(action_and_state.action)
            game_states.append(
                load_unit_state.filter_unit_states(
                    action_and_state.round_number,
                    action_and_state.action,
                    action_and_state.states
                )
            )

        return unit_actions, game_states, game_board


class DelayedUnitStateBuilder:

    def __init__(self, unit: dict, data_loader: RawGameDataParser):
        self.data_loader = data_loader
        self._unit_state = dict()
        self.unit = unit

    def build(self) -> dict:
        if self.unit['chassis'] == self.unit['model']:
            action = self.data_loader.entities.get(self.unit['id'], {})
            self.unit['model'] = action['model']
        chassis = self.unit['chassis']
        model = self.unit['model']
        mek = self.data_loader.meks_extras.get(f'{chassis} {model}', {})
        for mek_key in mek:
            if mek_key == "version" or mek_key == "_type":
                continue
            if mek_key not in self.unit:
                self.unit[mek_key] = mek[mek_key]
            elif (
                    mek[mek_key] is not None and
                    mek[mek_key] != self.unit.get(mek_key) and
                    self.unit.get(mek_key) in (-1, None)
            ):
                self.unit[mek_key] = mek[mek_key]

        return self.unit
