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


class UnitEnrichmentParserV1(BaseParser):
    TYPE = "UnitEnrichment"
    HEADER = BaseParser.make_header_regex_for("version chassis model type role bv walk_mp run_mp jump_mp heat armor internal height has_ecm has_ams max_range max_damage armor_front armor_left armor_right armor_back weapon_dmg_facing_short_medium_long_range")
    LINE = BaseParser.make_line_regex_for_groups(
        version_field="UnitEnrichment",
        line="version chassis model type role bv walk_mp run_mp jump_mp heat armor internal height has_ecm has_ams max_range max_damage armor_front armor_left armor_right armor_back weapon_data")
    VERSION = "31052025"
