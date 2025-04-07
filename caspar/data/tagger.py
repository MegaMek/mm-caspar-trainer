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


def tag_action(unit_actions, game_states_for_actions, game_board):
    tags = {}
    for i, (unit_action, game_states) in enumerate(zip(unit_actions, game_states_for_actions)):
        tags[str(i)] = {
            "entity_id": unit_action["entity_id"],
            "classification": movement_class(unit_action, game_states),
            "quality": movement_quality(unit_action, game_states),
        }

    return tags


def movement_class(unit_action, game_states):
    start = (unit_action["from_x"], unit_action["from_y"])
    end = (unit_action["to_x"], unit_action["to_y"])
    moved = unit_action["hexes_moved"]
    team_id = unit_action["team_id"]

    dist_to_closest_enemy_at_start = distance_to_closest_enemy(start, team_id, game_states)
    dist_to_closest_enemy_at_end = distance_to_closest_enemy(end, team_id, game_states)
    enemy_distance_delta = dist_to_closest_enemy_at_start - dist_to_closest_enemy_at_end

    def is_hold_position():
        return moved == 0 or (start[0] == end[0] and start[1] == end[1])

    def is_offensive():
        return moved > 2 and enemy_distance_delta > 0

    def is_defensive():
        return moved > 2 and enemy_distance_delta < 0

    def is_reposition():
        return moved < 3

    if is_hold_position():
        return "HOLD_POSITION"

    if is_offensive():
        return "OFFENSIVE"

    if is_defensive():
        return "DEFENSIVE"

    if is_reposition():
        return "DEFENSIVE"

    return "DEFENSIVE"

def distance_to_closest_enemy(position, team_id, game_states):
    enemies = filter(lambda e: e.get("team_id", -1) != team_id, game_states)
    closest_distance = 99999999999
    for enemy in enemies:
        dist = distance(position, (enemy.get("x"), enemy.get("y")))
        closest_distance = min(closest_distance, dist)
    return closest_distance


def distance(coordA, coordB):
    return np.sqrt((coordA[0] - coordB[0]) ** 2 + (coordA[1] - coordB[1]) ** 2)


def movement_quality(unit_action, game_states):
    quality = [
        "HIGH_QUALITY",
        "LOW_QUALITY",
        "IGNORE"
    ]

    if unit_action.get('from_x', -1) == -1:
        return quality[-1]

    if (unit_action.get("max_mp", 0) / 4 * 3) <= unit_action.get("hexes_moved"):
        return quality[0]

    return quality[1]