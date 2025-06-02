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
import re
from typing import Optional

from caspar.data.parser.base_parser import BaseParser
from caspar.data.parser.weapon_data_parser import weapon_data_parser

logger = logging.getLogger(__name__)


class UnitMoveActionParserV1(BaseParser):
    TYPE = "UnitAction"
    HEADER = BaseParser.make_header_regex_for(
        "player_id entity_id facing from_x from_y to_x to_y hexes_moved distance mp_used max_mp mp_p heat_p armor_p internal_p jumping prone legal step")
    LINE = BaseParser.make_line_regex_for_groups(
        "player_id id facing from_x from_y to_x to_y hexes_moved distance mp_used max_mp mp_p heat_p armor_p internal_p jumping prone legal step")


class UnitMoveActionParserV2(BaseParser):
    TYPE = "UnitAction"
    HEADER = BaseParser.make_header_regex_for(
        "player_id entity_id facing from_x from_y to_x to_y hexes_moved distance mp_used max_mp mp_p heat_p armor_p internal_p jumping prone legal step team_id")
    LINE = BaseParser.make_line_regex_for_groups(
        "player_id id facing from_x from_y to_x to_y hexes_moved distance mp_used max_mp mp_p heat_p armor_p internal_p jumping prone legal step team_id")


class UnitMoveActionParserV3(BaseParser):
    TYPE = "UnitAction"
    HEADER = BaseParser.make_header_regex_for(
        "player_id entity_id facing from_x from_y to_x to_y hexes_moved distance mp_used max_mp mp_p heat_p armor_p internal_p jumping prone legal step team_id chance_of_failure is_bot")
    LINE = BaseParser.make_line_regex_for_groups(
        "player_id id facing from_x from_y to_x to_y hexes_moved distance mp_used max_mp mp_p heat_p armor_p internal_p jumping prone legal step team_id chance_of_failure is_bot")


class UnitMoveActionParserV4(BaseParser):
    TYPE = "UnitAction"
    HEADER = BaseParser.make_header_regex_for(
        "entity_id player_id team_id facing from_x from_y to_x to_y hexes_moved distance mp_used max_mp mp_p heat_p armor_p internal_p jumping prone legal chance_of_failure is_bot step")
    LINE = BaseParser.make_line_regex_for_groups(
        "id player_id team_id facing from_x from_y to_x to_y hexes_moved distance mp_used max_mp mp_p heat_p armor_p internal_p jumping prone legal chance_of_failure is_bot step")


class UnitMoveActionParserV5(BaseParser):
    TYPE = "UnitAction"
    HEADER = BaseParser.make_header_regex_for(
        "entity_id player_id team_id facing from_x from_y to_x to_y hexes_moved distance mp_used max_mp mp_p heat_p armor_p internal_p jumping prone legal chance_of_failure step is_bot")
    LINE = BaseParser.make_line_regex_for_groups(
        "id player_id team_id facing from_x from_y to_x to_y hexes_moved distance mp_used max_mp mp_p heat_p armor_p internal_p jumping prone legal chance_of_failure step is_bot")


class UnitMoveActionParserV6(BaseParser):
    TYPE = "UnitAction"
    HEADER = BaseParser.make_header_regex_for(
        "entity_id player_id team_id facing from_x from_y to_x to_y hexes_moved distance mp_used max_mp mp_p heat_p armor_p internal_p jumping prone legal chance_of_failure step is_bot ha_ecm armor internal bc max_range armor_front_p armor_left_p armor_right_p armor_back_p role weapon_dmg_facing_short_medium_long_range")
    LINE = BaseParser.make_line_regex_for_groups(
        "id player_id team_id facing from_x from_y to_x to_y hexes_moved distance mp_used max_mp mp_p heat_p armor_p internal_p jumping prone legal chance_of_failure step is_bot ha_ecm armor internal bc max_range armor_front_p armor_left_p armor_right_p armor_back_p role weapon_data")
    SPECIAL_PARSERS = {"weapon_data": weapon_data_parser}


class UnitMoveActionParserV7(BaseParser):
    TYPE = "UnitAction"
    HEADER = BaseParser.make_header_regex_for(
        "version entity_id player_id team_id facing from_x from_y to_x to_y hexes_moved distance mp_used max_mp mp_p heat_p armor_p internal_p jumping prone legal chance_of_failure step is_bot ha_ecm armor internal bc max_range armor_front_p armor_left_p armor_right_p armor_back_p role weapon_dmg_facing_short_medium_long_range")
    LINE = BaseParser.make_line_regex_for_groups(
        version_field="UnitAction",
        line="version id player_id team_id facing from_x from_y to_x to_y hexes_moved distance mp_used max_mp mp_p heat_p armor_p internal_p jumping prone legal chance_of_failure step is_bot ha_ecm armor internal bc max_range armor_front_p armor_left_p armor_right_p armor_back_p role weapon_data")
    VERSION = "31052025"
    SPECIAL_PARSERS = {"weapon_data": weapon_data_parser}
