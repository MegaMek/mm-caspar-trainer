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

logger = logging.getLogger(__name__)


class UnitAttackActionParserV1(BaseParser):
    TYPE = "UnitAttack"
    HEADER = BaseParser.make_header_regex_for("round player_id entity_id type role x y facing target_player_id target_id target_type target_role target_x target_y target_facing aiming_loc aiming_mode weapon_id ammo_id ata atg gtg gta to_hit turns_to_hit spotter_id")
    LINE = BaseParser.make_line_regex_for_groups("round player_id id type role x y facing target_player_id target_id target_type target_role target_x target_y target_facing aiming_loc aiming_mode weapon_id ammo_id ata atg gtg gta to_hit turns_to_hit spotter_id")


class UnitAttackActionParserV2(BaseParser):
    TYPE = "UnitAttack"
    HEADER = BaseParser.make_header_regex_for(
        "version round player_id entity_id type role x y facing target_player_id target_id target_type target_role target_x target_y target_facing aiming_loc aiming_mode weapon_id ammo_id ata atg gtg gta to_hit turns_to_hit spotter_id")
    LINE = BaseParser.make_line_regex_for_groups(
        version_field="UnitAttack",
        line = "version round player_id id type role x y facing target_player_id target_id target_type target_role target_x target_y target_facing aiming_loc aiming_mode weapon_id ammo_id ata atg gtg gta to_hit turns_to_hit spotter_id")
    VERSION = "31052025"
