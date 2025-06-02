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

from caspar.data.parser.base_parser import BaseParser
from caspar.data.parser.weapon_data_parser import weapon_data_parser

logger = logging.getLogger(__name__)

class UnitStateParserV1(BaseParser):
    TYPE = "UnitState"
    HEADER = BaseParser.make_header_regex_for("round phase player_id entity_id chassis model type role x y facing mp heat prone airborne off_board crippled destroyed armor_p internal_p done max_range total_damage team_id")
    LINE = BaseParser.make_line_regex_for_groups("round phase player_id id chassis model type role x y facing mp heat prone airborne off_board crippled destroyed armor_p internal_p done max_range total_damage team_id")

class UnitStateParserV2(BaseParser):
    TYPE = "UnitState"
    HEADER = BaseParser.make_header_regex_for("round phase player_id entity_id chassis model type role x y facing mp heat prone airborne off_board crippled destroyed armor_p internal_p done max_range total_damage team_id armor internal bv")
    LINE = BaseParser.make_line_regex_for_groups("round phase player_id id chassis model type role x y facing mp heat prone airborne off_board crippled destroyed armor_p internal_p done max_range total_damage team_id armor internal bv")


class UnitStateParserV3(BaseParser):
    TYPE = "UnitState"
    HEADER = BaseParser.make_header_regex_for("id phase team_id round player_id chassis model type role x y facing mp heat prone airborne off_board crippled destroyed armor_p internal_p done max_range total_damage armor internal bv is_bot has_ecm armor_front_p armor_left_p armor_right_p armor_back_p weapon_dmg_facing_short_medium_long_range")
    LINE = BaseParser.make_line_regex_for_groups("id phase team_id round player_id chassis model type role x y facing mp heat prone airborne off_board crippled destroyed armor_p internal_p done max_range total_damage armor internal bv is_bot has_ecm armor_front_p armor_left_p armor_right_p armor_back_p weapon_data")
    SPECIAL_PARSERS = {"weapon_data": weapon_data_parser}


class UnitStateParserV4(BaseParser):
    TYPE = "UnitState"
    HEADER = BaseParser.make_header_regex_for(
        "version id phase team_id round player_id chassis model type role x y facing mp heat prone airborne off_board crippled destroyed armor_p internal_p done max_range total_damage armor internal bv is_bot has_ecm armor_front_p armor_left_p armor_right_p armor_back_p weapon_dmg_facing_short_medium_long_range")
    LINE = BaseParser.make_line_regex_for_groups(
        version_field="UnitState",
        line="version id phase team_id round player_id chassis model type role x y facing mp heat prone airborne off_board crippled destroyed armor_p internal_p done max_range total_damage armor internal bv is_bot has_ecm armor_front_p armor_left_p armor_right_p armor_back_p weapon_data")
    VERSION = "31052025"
    SPECIAL_PARSERS = {"weapon_data": weapon_data_parser}

