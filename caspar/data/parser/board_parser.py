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
import re
import logging

from caspar.data.parser.base_parser import BaseParser

logger = logging.getLogger(__name__)


class BoardParserV1(BaseParser):
    TYPE = "BoardData"
    HEADER = BaseParser.make_header_regex_for("board_name width height")
    LINE = re.compile(r"Board #(?P<board_id>\d+)\s(?P<WIDTH>\d+)\s(?P<HEIGHT>\d+)")


class BoardParserV2(BaseParser):
    TYPE = "BoardData"
    HEADER = BaseParser.make_header_regex_for("version board_name width height")
    BaseParser.make_line_regex_for_groups(version_field="BoardData", line="version board_id width height")
    VERSION = "31052025"
