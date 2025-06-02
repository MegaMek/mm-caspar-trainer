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
from typing import Any

logger = logging.getLogger(__name__)

arc_map_table = {
    0: ["arc_0", "arc_1", "arc_2", "arc_3", "arc_4", "arc_5"],  # ARC_360
    1: ["arc_0", "arc_1", "arc_5"],  # ARC_FORWARD
    2: ["arc_0", "arc_5"],  # ARC_LEFTARM
    3: ["arc_0", "arc_1"],  # ARC_RIGHTARM
    4: ["arc_3"],  # ARC_REAR
    5: ["arc_1"],  # ARC_LEFTSIDE
    6: ["arc_5"],  # ARC_RIGHTSIDE
    7: ["arc_0"],  # ARC_MAINGUN
    8: ["arc_0"],  # ARC_NORTH
    9: ["arc_1", "arc_2"],  # ARC_EAST
    10: ["arc_5", "arc_4"],  # ARC_WEST
    11: ["arc_0"],  # ARC_NOSE
    12: ["arc_0", "arc_5"],  # ARC_LWING
    13: ["arc_0", "arc_1"],  # ARC_RWING
    14: ["arc_5", "arc_4"],  # ARC_LWINGA
    15: ["arc_1", "arc_2"],  # ARC_RWINGA
    16: ["arc_0", "arc_4", "arc_5", "arc_3"],  # ARC_LEFTSIDE_SPHERE
    17: ["arc_0", "arc_1", "arc_2", "arc_3"],  # ARC_RIGHTSIDE_SPHERE
    18: ["arc_4", "arc_3"],  # ARC_LEFTSIDEA_SPHERE
    19: ["arc_2", "arc_3"],  # ARC_RIGHTSIDEA_SPHERE
    20: ["arc_4", "arc_5"],  # ARC_LEFT_BROADSIDE
    21: ["arc_1", "arc_2"],  # ARC_RIGHT_BROADSIDE
    22: ["arc_3"],  # ARC_AFT
    23: ["arc_0", "arc_3", "arc_4", "arc_5"],  # ARC_LEFT_SPHERE_GROUND
    24: ["arc_0", "arc_1", "arc_2", "arc_3"],  # ARC_RIGHT_SPHERE_GROUND
    25: ["arc_0", "arc_1", "arc_2", "arc_3", "arc_4", "arc_5"],  # ARC_TURRET
    26: ["arc_4", "arc_5"],  # ARC_SPONSON_TURRET_LEFT
    27: ["arc_1", "arc_2"],  # ARC_SPONSON_TURRET_RIGHT
    28: ["arc_4", "arc_5"],  # ARC_PINTLE_TURRET_LEFT
    29: ["arc_1", "arc_2"],  # ARC_PINTLE_TURRET_RIGHT
    30: ["arc_0", "arc_1", "arc_5"],  # ARC_PINTLE_TURRET_FRONT
    31: ["arc_3", "arc_2", "arc_4"],  # ARC_PINTLE_TURRET_REAR
    32: ["arc_0"],  # ARC_VGL_FRONT
    33: ["arc_0", "arc_1"],  # ARC_VGL_RF
    34: ["arc_0", "arc_5"],  # ARC_VGL_RR
    35: ["arc_3"],  # ARC_VGL_REAR
    36: ["arc_1", "arc_2"],  # ARC_VGL_LR
    37: ["arc_4", "arc_5"],  # ARC_VGL_LF
    38: ["arc_0"],  # ARC_NOSE_WPL
    39: ["arc_5"],  # ARC_LWING_WPL
    40: ["arc_1"],  # ARC_RWING_WPL
    41: ["arc_4", "arc_5"],  # ARC_LWINGA_WPL
    42: ["arc_1", "arc_2"],  # ARC_RWINGA_WPL
    43: ["arc_5"],  # ARC_LEFTSIDE_SPHERE_WPL
    44: ["arc_1"],  # ARC_RIGHTSIDE_SPHERE_WPL
    45: ["arc_4"],  # ARC_LEFTSIDEA_SPHERE_WPL
    46: ["arc_2"],  # ARC_RIGHTSIDEA_SPHERE_WPL
    47: ["arc_3"],  # ARC_AFT_WPL
    48: ["arc_4", "arc_5"],  # ARC_LEFT_BROADSIDE_WPL
    49: ["arc_1", "arc_2"],  # ARC_RIGHT_BROADSIDE_WPL
}

def weapon_data_parser(weapon_data: str) -> dict[str, dict[str, dict[str, int]] | list[int]]:
    """
    Parse the weapon data from the action line.

    Args:
        weapon_data: String containing weapon data

    Returns:
        Dictionary with parsed weapon data
    """
    if not weapon_data:
        return {}

    # Split the weapon data by spaces and parse each part
    parts = weapon_data.split()
    weapons = {}
    arcs = {
        'arc_0': [],
        'arc_1': [],
        'arc_2': [],
        'arc_3': [],
        'arc_4': [],
        'arc_5': [],
        'weapons': {},
    }
    weapon_id = 0
    for i in range(len(parts), 5):
        weapon = {
            "id": weapon_id,
            "damage": int(parts[i]),
            "weapon_arc_code": int(parts[i + 1]),
            "short_range": int(parts[i + 2]),
            "medium_range": int(parts[i + 3]),
            "long_range": int(parts[i + 4]),
        }
        weapons[weapon_id] = weapon
        weapon_id += 1
        for arc in arc_map_table.get(weapon["weapon_arc_code"], []):
            if arc in arcs:
                arcs[arc].append(weapon["id"])

    arcs['weapons'] = weapons

    return arcs
