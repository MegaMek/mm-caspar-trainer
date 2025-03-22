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
from typing import List, Dict, Tuple

import numpy as np
import math
from tqdm import tqdm

def clamp01(value: float) -> float:
    return max(min(value, 1), 0)


class FeatureExtractor:
    """
    Class for extracting features from unit actions and game states.
    """
    ENEMIES_N = 4
    ALLIES_N = 3
    RADAR_GRID_SIZE = 9
    MIN_DISTANCE_THRESHOLD = {
        "MISSILE_BOAT": 8,
        "JUGGERNAUT": 4,
        "SNIPER": 10,
        "SCOUT": 8,
        "STRIKER": 6,
        "AMBUSHER": 6,
        "BRAWLER": 1,
        "SKIRMISHER": 6
    }

    def __init__(self):
        # Define feature names and their indices for clarity
        _features = [
            "mp_percentage", "heat_percentage", "armor_percentage", "internal_percentage", "jumping",
            "distance_traveled", "hexes_moved", "is_facing_enemy", "enemy_in_range", "cover_value",
            "allies_nearby", "enemies_nearby", "team_cohesion", "unit_health_average", "unit_health_front",
            "unit_health_right", "unit_health_left", "unit_health_back", "position_crowding", "damage_ratio",
            "ecm_coverage", "enemy_ecm_coverage", "environmental_cover", "environmental_hazards",
            "favorite_target_in_range", "flanking_position", "formation_cohesion", "formation_separation",
            "formation_alignment", "friendly_artillery_fire", "covering_units", "heat_management",
            "enemy_vip_distance", "is_crippled",
            "moving_toward_waypoint", "unit_role", "threat_by_sniper", "threat_by_missile_boat",
            "threat_by_juggernaut", "unit_tmm", "piloting_caution", "retreat", "scouting",
            "standing_still", "strategic_goal", "target_health", "target_within_optimal_range",
            "turns_to_encounter", "chance_of_failure", "self_threat_level"
        ]

        for n in range(16):
            _features.append("unit_role_" + str(n))

        for n in range(100):
            _features.append("enemy_threat_heatmap_" + str(n))

        for n in range(100):
            _features.append("friendly_threat_heatmap_" + str(n))

        for n in range(self.ENEMIES_N):
            _features.append("closest_enemy_" + str(n))

        for n in range(self.ALLIES_N):
            _features.append("closest_friend_" + str(n))

        # Add radar features - square grid
        for y in range(self.RADAR_GRID_SIZE):
            for x in range(self.RADAR_GRID_SIZE):
                _features.append(f"radar_{y}_{x}")

        self.features = {feature: n for n, feature in enumerate(_features)}

        self.num_features = len(self.features)

    @classmethod
    def filter_actions(cls, action):
        return action.get("is_bot", 0) == 0 and action.get("type", 'BipedMek') not in {
            'AeroSpaceFighter', 'Infantry', 'FixedWingSupport', 'ConvFighter', 'Dropship', 'EjectedCrew', 'MekWarrior',
        'GunEmplacement', 'BattleArmor'}

    def extract_features(
            self, unit_actions: List[Dict], game_states: List[List[Dict]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features from unit actions and game states.

        Args:
            unit_actions: List of unit action dictionaries
            game_states: List of game state dictionaries

        Returns:
            X: Feature matrix
            y: Target values (currently set to 1.0 for human actions, 0.0 for bot actions)
        """
        if len(unit_actions) != len(game_states):
            raise ValueError("Number of unit actions must match number of game states")
        size = len(list(filter(lambda g: self.filter_actions(g), unit_actions)))
        x = np.zeros((size, self.num_features))
        y = np.zeros(size)

        for i, (action, state) in enumerate(tqdm(zip(unit_actions, game_states))):
            if action.get("is_bot", 0) == 1:
                continue

            if i >= len(x):
                break

            state = {unit["entity_id"]: unit for unit in state}

            # Basic features directly from unit action
            x[i, self.features["mp_percentage"]] = action.get("mp_p", 0)
            x[i, self.features["heat_percentage"]] = action.get("heat_p", 0)
            x[i, self.features["armor_percentage"]] = action.get("armor_p", 0)
            x[i, self.features["internal_percentage"]] = action.get("internal_p", 0)
            x[i, self.features["jumping"]] = action.get("jumping", 0)
            x[i, self.features["distance_traveled"]] = 0 if action.get("max_mp", 10.0) == 0 else (
                    action.get("distance", 0) / action.get("max_mp", 10.0))  # Normalize
            x[i, self.features["hexes_moved"]] = 0 if action.get("max_mp", 10.0) == 0 else (
                    action.get("hexes_moved", 0) / action.get("max_mp", 10.0))  # Normalize
            x[i, self.features["chance_of_failure"]] = action.get("chance_of_failure", 0.0)

            # More complex features derived from both action and state
            # These would normally be computed by your Java calculators
            # Here we're using simpler approximations
            x[i, self.features["is_facing_enemy"]] = self._calculate_facing_enemy(action, state)
            x[i, self.features["enemy_in_range"]] = self._calculate_enemy_in_range(action, state)
            x[i, self.features["cover_value"]] = self._calculate_cover(action, state)
            x[i, self.features["allies_nearby"]] = self._calculate_allies_nearby(action, state)
            x[i, self.features["enemies_nearby"]] = self._calculate_enemies_nearby(action, state)
            x[i, self.features["team_cohesion"]] = self._calculate_team_cohesion(action, state)
            x[i, self.features["unit_health_average"]] = self._calculate_unit_health_average(action)
            x[i, self.features["unit_health_front"]] = self._calculate_unit_health_front(action)
            x[i, self.features["unit_health_right"]] = self._calculate_unit_health_right(action)
            x[i, self.features["unit_health_left"]] = self._calculate_unit_health_left(action)
            x[i, self.features["unit_health_back"]] = self._calculate_unit_health_back(action)
            x[i, self.features["position_crowding"]] = self._calculate_position_crowding(action, state)
            x[i, self.features["damage_ratio"]] = self._calculate_damage_ratio(action, state)
            x[i, self.features["ecm_coverage"]] = self._calculate_ecm_coverage(action, state)
            x[i, self.features["enemy_ecm_coverage"]] = self._calculate_enemy_ecm_coverage(action, state)
            x[i, self.features["environmental_cover"]] = self._calculate_environmental_cover(action, state)
            x[i, self.features["environmental_hazards"]] = self._calculate_environmental_hazards(action)
            x[i, self.features["favorite_target_in_range"]] = self._calculate_favorite_target_in_range(action, state)
            x[i, self.features["flanking_position"]] = self._calculate_flanking_position(action, state)
            x[i, self.features["formation_cohesion"]] = self._calculate_formation_cohesion(action, state)
            x[i, self.features["formation_separation"]] = self._calculate_formation_separation(action, state)
            x[i, self.features["formation_alignment"]] = self._calculate_formation_alignment(action, state)
            x[i, self.features["friendly_artillery_fire"]] = self._calculate_friendly_artillery_fire(action, state)
            x[i, self.features["covering_units"]] = self._calculate_covering_units(action, state)
            x[i, self.features["heat_management"]] = self._calculate_heat_management(action, state)
            x[i, self.features["enemy_vip_distance"]] = self._calculate_enemy_vip_distance(action, state)
            x[i, self.features["is_crippled"]] = self._calculate_is_crippled(action, state)
            x[i, self.features["moving_toward_waypoint"]] = self._calculate_moving_toward_waypoint(action, state)
            x[i, self.features["unit_role"]] = self._calculate_unit_role(action, state)
            x[i, self.features["threat_by_sniper"]] = self._calculate_threat_by_sniper(action, state)
            x[i, self.features["threat_by_missile_boat"]] = self._calculate_threat_by_missile_boat(action, state)
            x[i, self.features["threat_by_juggernaut"]] = self._calculate_threat_by_juggernaut(action, state)
            x[i, self.features["unit_tmm"]] = self._calculate_unit_tmm(action)
            x[i, self.features["piloting_caution"]] = self._calculate_piloting_caution(action)
            x[i, self.features["retreat"]] = self._calculate_retreat(action, state)
            x[i, self.features["scouting"]] = self._calculate_scouting(action, state)
            x[i, self.features["standing_still"]] = self._calculate_standing_still(action)
            x[i, self.features["strategic_goal"]] = self._calculate_strategic_goal(action, state)
            x[i, self.features["target_health"]] = self._calculate_target_health(action, state)
            x[i, self.features["target_within_optimal_range"]] = self._calculate_target_within_optimal_range(action, state)
            x[i, self.features["turns_to_encounter"]] = self._calculate_turns_to_encounter(action, state)

            for n, value in enumerate(self._hot_encode_unit_role(action)):
                x[i, self.features["unit_role_" + str(n)]] = value

            for n, value in enumerate(self._n_threat(action, state, self.ENEMIES_N)):
                x[i, self.features["closest_enemy_" + str(n)]] = value

            for n, value in enumerate(self._n_threat_allies(action, state, self.ALLIES_N)):
                x[i, self.features["closest_friend_" + str(n)]] = value

            x[i, self.features["self_threat_level"]] = self._self_threat(action, state)

            enemy_threat_heatmap = self._calculate_enemy_threat_heatmap(action, state)
            friendly_threat_heatmap = self._calculate_friendly_threat_heatmap(action, state)

            for n in range(100):
                x[i, self.features["enemy_threat_heatmap_" + str(n)]] = enemy_threat_heatmap[n]
            for n in range(100):
                x[i, self.features["friendly_threat_heatmap_" + str(n)]] = friendly_threat_heatmap[n]

            # Calculate and add radar features
            radar = self._calculate_unit_radar_features(action, state)
            for y_idx in range(self.RADAR_GRID_SIZE):
                for x_idx in range(self.RADAR_GRID_SIZE):
                    feature_name = f"radar_{y_idx}_{x_idx}"
                    feature_idx = self.features[feature_name]
                    x[i, feature_idx] = radar[y_idx, x_idx]

            reward = self.reward_calculator(action, state, game_states[i:])

            y[i] = reward

        return x, y

    @classmethod
    def _hot_encode_unit_role(cls, action):
        role = action.get("role", "UNDETERMINED")
        roles = [
            "UNDETERMINED",
            "NONE",
            "AMBUSHER",
            "BRAWLER",

            "JUGGERNAUT",
            "MISSILE_BOAT",
            "SCOUT",
            "SKIRMISHER",

            "SNIPER",
            "STRIKER",
            "ATTACK_FIGHTER",
            "DOGFIGHTER",

            "FAST_DOGFIGHTER",
            "FIRE_SUPPORT",
            "INTERCEPTOR",
            "TRANSPORT"
        ]

        return [1.0 if r == role else 0.0 for r in roles]

    def _calculate_unit_radar_features(self, action: Dict, state: Dict) -> np.ndarray:
        """
        Create a 5x5 radar feature matrix representing unit positions relative to the moving unit.

        Args:
            action: Dictionary containing action data for the moving unit
            state: Dictionary mapping entity_id to unit state data

        Returns:
            np.ndarray: 5x5 matrix with unit position features
        """
        # Get the position of the unit being moved
        unit_x = action.get("to_x", action.get("from_x", 0))
        unit_y = action.get("to_y", action.get("from_y", 0))
        unit_team_id = action.get("team_id", -1)
        unit_entity_id = action.get("entity_id", -1)

        # Initialize the 5x5 radar matrix
        # We'll use different values for:
        # - 0: Empty space
        # - Positive values: Friendly units (1 for regular, 2 for shadows)
        # - Negative values: Enemy units (-1 for regular, -2 for shadows)
        radar = np.zeros((self.RADAR_GRID_SIZE, self.RADAR_GRID_SIZE), dtype=float)

        # Mark the center position (where our unit is)
        center_x, center_y = math.floor(self.RADAR_GRID_SIZE / 2), math.floor(self.RADAR_GRID_SIZE / 2)
        radar[center_y, center_x] = 3  # Special value for the unit itself

        # Process each unit in the state
        for entity_id, unit in state.items():
            # Skip the unit being moved
            if entity_id == unit_entity_id:
                continue

            # Get the unit's position
            other_x = unit.get("x", 0)
            other_y = unit.get("y", 0)

            # Calculate relative position
            rel_x = other_x - unit_x
            rel_y = other_y - unit_y

            # Determine if this is a friendly or enemy unit
            is_friendly = unit.get("team_id", -1) == unit_team_id
            unit_value = 1 if is_friendly else -1
            shadow_value = 2 if is_friendly else -2

            # Convert to radar coordinates
            radar_x = center_x + rel_x
            radar_y = center_y + rel_y

            # If the unit is within the radar bounds, mark it directly
            if 0 <= radar_x < self.RADAR_GRID_SIZE and 0 <= radar_y < self.RADAR_GRID_SIZE:
                radar[radar_y, radar_x] = unit_value
            else:
                # Unit is outside radar bounds, so we need to mark it at the edge
                # Calculate direction vector from center to unit
                dir_x, dir_y = rel_x, rel_y

                # Normalize direction vector
                length = max(abs(dir_x), abs(dir_y))
                if length > 0:
                    dir_x = dir_x / length
                    dir_y = dir_y / length

                # Find where this ray intersects the radar edge
                edge_x, edge_y = self._find_radar_edge_intersection(dir_x, dir_y)

                # Map to radar coordinates
                edge_radar_x = center_x + edge_x
                edge_radar_y = center_y + edge_y

                # Clamp to radar bounds
                edge_radar_x = max(0, min(self.RADAR_GRID_SIZE-1, int(round(edge_radar_x))))
                edge_radar_y = max(0, min(self.RADAR_GRID_SIZE-1, int(round(edge_radar_y))))

                # Mark the edge position
                radar[edge_radar_y, edge_radar_x] = unit_value

        return radar

    def _find_radar_edge_intersection(self, dir_x: float, dir_y: float) -> Tuple[float, float]:
        """
        Find where a ray from the center in direction (dir_x, dir_y) intersects the edge of the 5x5 grid.

        Args:
            dir_x: X component of direction vector
            dir_y: Y component of direction vector

        Returns:
            Tuple containing the x, y coordinates of the intersection point
        """
        # Grid boundaries (relative to center at 0,0)
        min_x, max_x = -2, 2
        min_y, max_y = -2, 2

        # Initialize to a large value
        min_t = float('inf')
        result_x, result_y = 0, 0

        # Check intersection with each of the four edges
        if dir_x != 0:
            # Left edge
            t_left = (min_x - 0) / dir_x
            if t_left > 0:
                y_left = t_left * dir_y
                if min_y <= y_left <= max_y and t_left < min_t:
                    min_t = t_left
                    result_x, result_y = min_x, y_left

            # Right edge
            t_right = (max_x - 0) / dir_x
            if t_right > 0:
                y_right = t_right * dir_y
                if min_y <= y_right <= max_y and t_right < min_t:
                    min_t = t_right
                    result_x, result_y = max_x, y_right

        if dir_y != 0:
            # Bottom edge
            t_bottom = (min_y - 0) / dir_y
            if t_bottom > 0:
                x_bottom = t_bottom * dir_x
                if min_x <= x_bottom <= max_x and t_bottom < min_t:
                    min_t = t_bottom
                    result_x, result_y = x_bottom, min_y

            # Top edge
            t_top = (max_y - 0) / dir_y
            if t_top > 0:
                x_top = t_top * dir_x
                if min_x <= x_top <= max_x and t_top < min_t:
                    min_t = t_top
                    result_x, result_y = x_top, max_y

        return result_x, result_y

    @classmethod
    def round_from_state(cls, state) -> int:
        for val in state.values():
            return val["round"]
        return -1

    @classmethod
    def unit_health_retention(cls, action, state, game_states) -> float:
        entity_id = action["entity_id"]
        current_round = cls.round_from_state(state)
        initial_health = (action['armor_p'] + action['internal_p']) / 2
        final_health = initial_health
        for gs in game_states:
            temp_round = gs[0]["round"]
            if temp_round == current_round + 1:
                for unit in gs:
                    if unit["entity_id"] == entity_id:
                        final_health = (unit["armor_p"] + unit["internal_p"]) / 2
                        break

        health_retention = min(1.0, final_health / initial_health) if initial_health > 0 else 0.0
        return health_retention

    @classmethod
    def unit_survival_bonus(cls, action, state, game_states) -> float:
        entity_id = action["entity_id"]
        current_round = cls.round_from_state(state)

        for gs in game_states:
            temp_round = gs[0]["round"]
            if temp_round == current_round + 1:
                for unit in gs:
                    if unit["entity_id"] == entity_id:
                        return 1.0
                break
            elif temp_round < current_round:
                return 1.0

        return 0.0

    @classmethod
    def team_bv_calculator(cls, action, state, game_states) -> tuple[float, float]:
        team_id = action["team_id"]
        total_allied_bv = 0
        total_enemy_bv = 0
        current_round = cls.round_from_state(state)

        for gs in game_states:
            temp_round = gs[0]["round"]
            if temp_round == current_round + 1:
                for unit in gs:
                    if unit["team_id"] == team_id:
                        total_allied_bv += unit["bv"]
                    else:
                        total_enemy_bv += unit["bv"]
                break
            elif temp_round < current_round:
                for unit in state.values():
                    if unit["team_id"] == team_id:
                        total_allied_bv += unit["bv"]
                    else:
                        total_enemy_bv += unit["bv"]
                break

        total_bv = max(total_allied_bv + total_enemy_bv, 1)
        return clamp01(total_allied_bv / total_bv), clamp01(total_enemy_bv / total_bv)

    @classmethod
    def reward_calculator(cls, action, state, game_states) -> float:
        allied_bv, enemy_bv = cls.team_bv_calculator(action, state, game_states)
        health_retention = cls.unit_health_retention(action, state, game_states)
        survival_bonus = cls.unit_survival_bonus(action, state, game_states)
        distance_mod = cls.distance_reward(action, state) if action.get("is_bot", 1.0) == 0.0 else 1.0
        return (0.5 * allied_bv + 0.3 * health_retention + 0.2 * survival_bonus) * distance_mod

    @classmethod
    def distance_reward(cls, action, state):
        if cls.MIN_DISTANCE_THRESHOLD:
            min_distance = cls.MIN_DISTANCE_THRESHOLD.get(action.get("role", "NONE"), 3)

            mod = clamp01(cls._distance_from_closest_enemy(action, state) / min_distance)
        else:
            mod = 1
        return mod

    @classmethod
    def _closest_enemy(cls, action, state) -> tuple[float, dict]:
        x = action.get("to_x", 0)
        y = action.get("to_y", 0)
        dist = float('inf')
        closest = None
        for unit in state.values():
            if unit.get("team_id", -1) != action.get("team_id", -2):
                if (d := cls._distance(x, y, unit)) < dist:
                    dist = d
                    closest = unit
        return dist, closest

    @classmethod
    def _distance_from_closest_enemy(cls, action, state) -> float:
        dist, _ = cls._closest_enemy(action, state)
        return dist

    @classmethod
    def _number_of_enemies_closer_than(cls, action, state, distance):
        x = action.get("to_x", 0)
        y = action.get("to_y", 0)
        team_id = action.get("team_id", -2)
        count = 0
        for unit in state.values():
            if unit.get("team_id", -1) != team_id:
                if cls._distance(x, y, unit) < distance:
                    count += 1
        return count

    @classmethod
    def _number_of_friendlies_closer_than(cls, action, state, distance):
        x = action.get("to_x", 0)
        y = action.get("to_y", 0)
        count = 0
        for unit in state.values():
            if unit.get("team_id", -1) == action.get("team_id", -2):
                if cls._distance(x, y, unit) < distance:
                    count += 1
        return count
    
    @classmethod
    def _distance(cls, x, y, unit):
        return np.sqrt((unit.get("x", 0) - x) ** 2 + (unit.get("y", 0) - y) ** 2)

    @classmethod
    def _calculate_unit_health_average(cls, action) -> float:
        return clamp01(action.get("armor_p", 0) + action.get("internal_p", 0) / 2)

    @classmethod
    def _calculate_unit_health_front(cls, action):
        return cls._calculate_unit_health_average(action)

    @classmethod
    def _calculate_unit_health_right(cls, action):
        return cls._calculate_unit_health_average(action)

    @classmethod
    def _calculate_unit_health_left(cls, action):
        return cls._calculate_unit_health_average(action)

    @classmethod
    def _calculate_unit_health_back(cls, action):
        return cls._calculate_unit_health_average(action)

    @classmethod
    def _calculate_position_crowding(cls, action, state):
        enemy_count = cls._number_of_enemies_closer_than(action, state, 8) / 2
        allies_count = cls._number_of_friendlies_closer_than(action, state, 4) / 3
        return clamp01(enemy_count * 0.7 + allies_count * 0.3)

    @classmethod
    def _threatening_enemies(cls, action, state) -> set[dict]:
        unit_x = action.get("to_x", action.get("from_x", 0))
        unit_y = action.get("to_y", action.get("from_y", 0))
        team_id = action.get("team_id", -2)
        enemies = []
        for unit in state.values():
            if unit.get("team_id", -1) != team_id:
                distance = cls._distance(unit_x, unit_y, unit)

                if distance <= unit.get("max_range", 10):
                    enemies.append(unit)
        return enemies

    @classmethod
    def _calculate_facing_enemy(cls, action, state):
        """
        Calculate how well the unit is facing potential enemy threats.
        Similar to FacingEnemyCalculator.
        """
        unit_facing = action.get("facing", 0)
        threatening_enemies = cls._threatening_enemies(action, state)

        if threatening_enemies:
            dx = 0
            dy = 0
            n = len(threatening_enemies)
            for unit in threatening_enemies:
                dx += unit.get("x", 0)
                dy += unit.get("y", 0)

            dx /= n
            dy /= n
            ideal_facing = (int(np.arctan2(dy, dx) * 3 / np.pi) + 3) % 6
        else:
            ideal_facing = unit_facing

        # Calculate facing difference (0-3, where 0 is perfect, 3 is opposite)
        facing_diff = min(abs(unit_facing - ideal_facing), 6 - abs(unit_facing - ideal_facing))
        
        # Convert to 0-1 score (1 is perfect facing, 0 is worst)
        return 1.0 - (facing_diff / 3.0)

    @classmethod
    def _calculate_enemy_in_range(cls, action, state):
        """
        Calculate how many enemies are within weapon range.
        """
        max_range = action.get("max_range")
        if max_range == 0:
            return 0.0
        
        # Count enemies in range
        dist, enemy = cls._closest_enemy(action, state)

        # Normalize (capping at 5 enemies)
        return clamp01(1 - (dist / max_range))

    @classmethod
    def _calculate_cover(cls, action, state):
        """
        Calculate how much cover the unit has at its position.
        Related to EnvironmentalCoverCalculator.
        """
        # This is a simplified approach. In practice, you'd calculate actual cover
        # based on terrain, buildings, etc. But here I want to just say how much the cover would
        # improve the unit situation
        
        # Get distance to the closest enemy to adjust cover value
        dist = cls._distance_from_closest_enemy(action, state)
        
        if dist > 20:  # Far from enemies, cover matters less
            base_cover = 0.3
        elif dist > 12:  # Medium distance
            base_cover = 0.6
        else:  # Close to enemies, cover matters more
            base_cover = 0.9
        
        # Adjust based on unit's role and movement
        unit_id = action.get("entity_id", -1)
        unit_state = state.get(unit_id, {})
        
        if unit_state.get("role", "") in {"SNIPER", "MISSILE_BOAT", "SCOUT", "STRIKER"}:
            # Snipers benefit more from cover
            base_cover *= 1.2
        
        if action.get("jumping", 0) == 1:
            # Jumping doesnt need much cover
            base_cover *= 0.7

        if action.get("prone", 0) == 1:
            # Prone units are hard to hit
            base_cover *= 1.2

        return clamp01(base_cover)

    @classmethod
    def _calculate_allies_nearby(cls, action, state):
        """
        Calculate number of allied units nearby.
        Related to CoveringUnitsCalculator.
        """
        unit_x = action.get("to_x", action.get("from_x", 0))
        unit_y = action.get("to_y", action.get("from_y", 0))
        team_id = action.get("team_id", -1)
        
        allies_count = 0
        for unit in state.values():
            if unit.get("team_id", -1) == team_id and unit.get("entity_id", -1) != action.get("entity_id", -1):
                ally_x = unit.get("x", 0)
                ally_y = unit.get("y", 0)
                distance = np.sqrt((ally_x - unit_x)**2 + (ally_y - unit_y)**2)
                
                # Count allies within a reasonable distance (e.g., 8 hexes)
                if distance <= 8:
                    allies_count += 1
        
        # Normalize (capping at 6 allies)
        return min(allies_count / 6.0, 1.0)

    @classmethod
    def _calculate_enemies_nearby(cls, action, state):
        """
        Calculate number of enemy units nearby.
        Related to NearbyEnemyCountCalculator.
        """
        unit_x = action.get("to_x", action.get("from_x", 0))
        unit_y = action.get("to_y", action.get("from_y", 0))
        team_id = action.get("team_id", -1)
        
        enemies_count = 0
        for unit in state.values():
            if unit.get("team_id", -1) != team_id:
                enemy_x = unit.get("x", 0)
                enemy_y = unit.get("y", 0)
                distance = np.sqrt((enemy_x - unit_x)**2 + (enemy_y - unit_y)**2)
                
                # Count enemies within a threatening distance (e.g., 10 hexes)
                if distance <= 10:
                    enemies_count += 1
        
        # Normalize (capping at 8 enemies)
        return clamp01(enemies_count / 8.0)

    @classmethod
    def _calculate_team_cohesion(cls, action, state):
        """
        Calculate how well the unit stays with its team.
        Related to FormationCohesionCalculator.
        """
        distance_to_center = cls.distance_to_team_center(action, state)

        # Normalize distance (assuming a reasonable max distance of 20)
        
        return clamp01(1 - (distance_to_center / 20.0))

    @classmethod
    def _calculate_damage_ratio(cls, action, state):
        """
        Calculates expected damage ratio.
        Based on DamageRatioCalculator.
        """
        unit_damage = action.get("total_damage")
        unit_x = action.get("to_x", action.get("from_x", 0))
        unit_y = action.get("to_y", action.get("from_y", 0))
        max_range = action.get("max_range")

        best_ratio = 0.0

        for enemy in state.values():
            if enemy.get("team_id", -1) != action.get("team_id", -2):
                distance = cls._distance(unit_x, unit_y, enemy)

                if distance <= max_range:
                    enemy_armor = enemy.get("armor", 0)  # Approximate armor value
                    enemy_internal = enemy.get("internal", 0)  # Approximate internal structure
                    total_health = enemy_armor + enemy_internal

                    if total_health > 0:
                        damage_ratio = unit_damage / total_health
                        best_ratio = max(best_ratio, damage_ratio)

        # Normalize for a reasonable damage ratio range
        return clamp01(best_ratio)

    @classmethod
    def _calculate_ecm_coverage(cls, action, state):
        """
        Calculate ECM coverage of the unit.
        Based on EcmCoverageCalculator.
        """
        has_ecm = action.get("ecm", 0) == 1

        if not has_ecm:
            return 0.0
        
        # Count nearby friendly units with ECM
        unit_x = action.get("to_x", action.get("from_x", 0))
        unit_y = action.get("to_y", action.get("from_y", 0))
        team_id = action.get("team_id", -1)
        
        overlapping_ecm = 0
        for unit in state.values():
            if unit.get("team_id", -1) == team_id and unit.get("entity_id", -1) != action.get("entity_id", -1):
                overlapping_ecm = cls.count_overlapping_ecm(overlapping_ecm, unit, unit_x, unit_y)
        
        # Calculate ECM efficiency (1.0 for no overlap, decreases with overlaps)
        ecm_efficiency = clamp01(1.0 / (overlapping_ecm + 1.0))
        
        return ecm_efficiency

    @classmethod
    def count_overlapping_ecm(cls, overlapping_ecm, unit, unit_x, unit_y):
        if unit.get("ecm", 0) == 1:
            ally_x = unit.get("x", 0)
            ally_y = unit.get("y", 0)
            distance = np.sqrt((ally_x - unit_x) ** 2 + (ally_y - unit_y) ** 2)

            # ECM typically has a range of 6 hexes
            if distance <= 6:
                overlapping_ecm += 1
        return overlapping_ecm

    @classmethod
    def unit_has_ecm(cls, action, state):
        unit_id = action.get("entity_id", -1)
        unit_state = state.get(unit_id, {})
        # Simple logic: if unit has ECM, calculate coverage
        return unit_state.get("ecm", False)  # This would come from your data

    @classmethod
    def _calculate_enemy_ecm_coverage(cls, action, state):
        """
        Calculate enemy ECM affecting the unit.
        Based on UnderEnemyEcmCoverageCalculator.
        """
        unit_x = action.get("to_x", action.get("from_x", 0))
        unit_y = action.get("to_y", action.get("from_y", 0))
        team_id = action.get("team_id", -1)
        
        # Count nearby enemy units with ECM
        overlapping_ecm = 0
        for unit in state.values():
            if unit.get("team_id", -1) != team_id:
                overlapping_ecm = cls.count_overlapping_ecm(overlapping_ecm, unit, unit_x, unit_y)

        return clamp01(1 /(overlapping_ecm + 1))

    @classmethod
    def _calculate_environmental_cover(cls, action, state):
        """
        Calculate cover from terrain.
        Based on EnvironmentalCoverCalculator.
        """
        # This would need information about terrain, which isn't directly available in our dataset
        # Simplified approach based on distance to enemies
        dist, _ = cls._closest_enemy(action, state)
        
        # Higher cover value when enemies are nearby
        if dist < 7:
            return 0.8
        elif dist < 12:
            return 0.6
        elif dist < 24:
            return 0.4
        else:
            return 0.2

    @classmethod
    def _calculate_environmental_hazards(cls, action):
        """
        Calculate terrain hazards at position.
        Based on EnvironmentalHazardsCalculator.
        """
        # Without terrain data, use a heuristic based on movement points
        mp_p = action.get("mp_p", 0)

        mp_used = action.get("mp_used", 0)
        hexes_moved = action.get("hexes_moved", 0)

        hazard_score = 0.0
        if mp_used > hexes_moved:
            hazard_score = 0.3 * mp_p

        # Also consider jumping (which might avoid hazards)
        if action.get("jumping", 0) == 1:
            hazard_score = hazard_score * 0.5
        
        return clamp01(hazard_score)

    @classmethod
    def _calculate_flanking_position(cls, action, state):
        """
        Calculate flanking position score.
        Based on FlankingPositionCalculator.
        """
        unit_x = action.get("to_x", action.get("from_x", 0))
        unit_y = action.get("to_y", action.get("from_y", 0))
        team_id = action.get("team_id", -1)
        
        best_flanking_score = 0.0
        
        for enemy in state.values():
            if enemy.get("team_id", -1) != team_id:
                enemy_x = enemy.get("x", 0)
                enemy_y = enemy.get("y", 0)
                enemy_facing = enemy.get("facing", 0)
                
                # Calculate distance to enemy
                distance = np.sqrt((enemy_x - unit_x)**2 + (enemy_y - unit_y)**2)
                
                # Skip enemies that are too far away
                if distance > 12:
                    continue
                
                # Calculate angle between unit and enemy (simplified)
                dx = unit_x - enemy_x
                dy = unit_y - enemy_y
                direction_to_unit = (int(np.arctan2(dy, dx) * 3 / np.pi) + 3) % 6
                
                # Calculate how far the unit is from the enemy's facing
                angle_diff = abs(direction_to_unit - enemy_facing)
                if angle_diff > 3:
                    angle_diff = 6 - angle_diff
                
                # Determine attack zone (front, side, rear)
                if angle_diff <= 1:
                    # Front arc - lowest flanking value
                    zone_score = 0.2
                elif angle_diff == 3:
                    # Rear arc - highest flanking value
                    zone_score = 1.0
                else:
                    # Side arcs
                    zone_score = 0.75
                
                # Adjust score based on distance
                distance_factor = clamp01(1.0 - (distance / 12.0))

                # Calculate combined flanking score
                flanking_score = zone_score * distance_factor
                
                # Keep track of best score
                best_flanking_score = max(best_flanking_score, flanking_score)
        
        return best_flanking_score

    @classmethod
    def _calculate_formation_cohesion(cls, action, state):
        """
        Calculate formation cohesion score.
        Based on FormationCohesionCalculator.
        """
        # Calculate distance to team center (similar to team_cohesion but different weighting)
        distance_to_center = cls.distance_to_team_center(action, state)

        # Calculate movement pattern match
        movement_pattern_match = cls._calculate_movement_pattern_match(action, state)

        # Combine factors with weights
        distance_weight = 0.6
        movement_weight = 0.4

        distance_factor = max(0, 1 - (distance_to_center / 15.0))

        return (distance_weight * distance_factor) + (movement_weight * movement_pattern_match)

    @classmethod
    def _calculate_movement_pattern_match(cls, action, state):
        """
        Calculate how well the unit maintains proper formation with nearby allies.
        Good formation means units are between 1-4 hexes away from their closest allies,
        allowing for box or cross formations.
        """
        unit_x = action.get("to_x", action.get("from_x", 0))
        unit_y = action.get("to_y", action.get("from_y", 0))
        player_id = action.get("player_id", -1)
        unit_id = action.get("entity_id", -1)

        # Find allied units
        own_units = []
        for unit in state.values():
            if unit.get("player_id", -1) == player_id and unit.get("entity_id", -1) != unit_id:
                own_units.append({
                    "x": unit.get("x", 0),
                    "y": unit.get("y", 0),
                    "distance": np.sqrt((unit.get("x", 0) - unit_x) ** 2 + (unit.get("y", 0) - unit_y) ** 2)
                })

        # No allies to form pattern with
        if len(own_units) < 1:
            return 0.5  # Neutral score when no allies

        # Sort allies by distance
        own_units.sort(key=lambda u: u["distance"])

        # Take up to 4 closest allies for formation evaluation
        closest_allies = own_units[:min(4, len(own_units))]

        # Calculate formation score
        formation_score = 0.0
        good_spacing_count = 0

        for ally in closest_allies:
            distance = ally["distance"]
            # Ideal distance is between 1 and 4 hexes
            if 1.0 <= distance <= 4.0:
                # Perfect spacing gets full points
                if 3 <= distance <= 4.0:
                    spacing_score = 1.0
                # Less ideal but still acceptable spacing
                else:
                    spacing_score = 0.7
                good_spacing_count += 1
            # Too close or too far
            else:
                # Calculate penalty based on how far outside ideal range
                if distance < 1.0:
                    # Too close - severe penalty
                    spacing_score = 0.0
                else:
                    # Too far - moderate penalty based on excess
                    spacing_score = max(0, 1 - (distance - 4.0) / 3.0)

            formation_score += spacing_score

        # Average the formation score
        if closest_allies:
            formation_score /= len(closest_allies)

        # Bonus for having more units in good formation
        formation_bonus = clamp01(good_spacing_count / 4.0) * 0.2

        return clamp01(formation_score + formation_bonus)

    @classmethod
    def distance_to_team_center(cls, action, state):
        unit_x = action.get("to_x", action.get("from_x", 0))
        unit_y = action.get("to_y", action.get("from_y", 0))
        team_id = action.get("team_id", -1)
        # Calculate center of team
        team_center_x = 0
        team_center_y = 0
        team_count = 0
        for unit in state.values():
            if unit.get("team_id", -1) == team_id:
                team_center_x += unit.get("x", 0)
                team_center_y += unit.get("y", 0)
                team_count += 1
        if team_count > 0:
            team_center_x /= team_count
            team_center_y /= team_count
        # Calculate distance to team center
        distance_to_center = np.sqrt((team_center_x - unit_x) ** 2 + (team_center_y - unit_y) ** 2)
        return distance_to_center

    @classmethod
    def _calculate_formation_separation(cls, action, state):
        """
        Calculate formation separation score.
        Based on FormationSeparationCalculator.
        """
        unit_x = action.get("to_x", action.get("from_x", 0))
        unit_y = action.get("to_y", action.get("from_y", 0))
        unit_id = action.get("entity_id", -1)
        team_id = action.get("team_id", -1)
        
        # Minimum desirable distance between units
        min_distance = 2
        
        # Check distance to each friendly unit
        closest_friendly_distance = float('inf')
        
        for unit in state.values():
            if unit.get("team_id", -1) == team_id and unit.get("entity_id", -1) != unit_id:
                friendly_x = unit.get("x", 0)
                friendly_y = unit.get("y", 0)
                distance = np.sqrt((friendly_x - unit_x)**2 + (friendly_y - unit_y)**2)
                
                closest_friendly_distance = min(closest_friendly_distance, distance)
        
        if closest_friendly_distance == float('inf'):
            return 1.0  # No other friendlies
        
        # Calculate separation factor
        if closest_friendly_distance < min_distance:
            # Too close - apply penalty
            separation_factor = closest_friendly_distance / min_distance
        else:
            # Sufficient separation
            separation_factor = 1.0
        
        return separation_factor

    @classmethod
    def _calculate_formation_alignment(cls, action, state):
        """
        Calculate formation alignment score.
        Based on FormationAlignmentCalculator.
        """
        unit_facing = action.get("facing", 0)
        team_id = action.get("team_id", -1)
        
        # Determine team's primary facing direction
        facings = [unit.get("facing", 0) for unit in state.values() if unit.get("team_id", -1) == team_id]
        
        if not facings:
            return 0.5  # No data
        
        # Find most common facing direction
        facing_counts = {}
        for f in facings:
            facing_counts[f] = facing_counts.get(f, 0) + 1
        
        leader_facing = max(facing_counts, key=facing_counts.get)
        
        # Calculate facing alignment with leader
        facing_diff = min(abs(unit_facing - leader_facing), 6 - abs(unit_facing - leader_facing))
        facing_alignment = 1.0 - (facing_diff / 3.0)
        
        # Also consider position alignment in formation (placeholder)
        position_alignment = 0.6  # Simplified
        
        # Combine with weights
        return (0.4 * facing_alignment) + (0.6 * position_alignment)

    @classmethod
    def _calculate_covering_units(cls, action, state):
        """
        Calculate number of units covering this unit.
        Based on CoveringUnitsCalculator.
        """
        unit_x = action.get("to_x", action.get("from_x", 0))
        unit_y = action.get("to_y", action.get("from_y", 0))
        team_id = action.get("team_id", -1)
        
        covering_units = 0
        
        for unit in state.values():
            if unit.get("team_id", -1) == team_id and unit.get("entity_id", -1) != action.get("entity_id", -1):
                ally_x = unit.get("x", 0)
                ally_y = unit.get("y", 0)
                distance = np.sqrt((ally_x - unit_x)**2 + (ally_y - unit_y)**2)
                
                # Check if ally can cover this unit based on distance and weapon range
                max_weapon_range = unit.get("max_range", 0)
                
                if distance <= max_weapon_range:
                    covering_units += 1
        
        # Normalize (capping at 5 covering units)
        return clamp01(covering_units / 5.0)

    @classmethod
    def _calculate_heat_management(cls, action, state):
        """
        Calculate heat management efficiency.
        Based on HeatManagementCalculator.
        """
        unit_id = action.get("entity_id", -1)
        unit_state = state.get(unit_id, {})
        
        heat_p = unit_state.get("heat_p", 0)
        # Calculate heat efficiency (1 = cool, 0 = overheating)
        return clamp01(1 - heat_p)

    @classmethod
    def _calculate_enemy_vip_distance(cls, action, state):
        """
        Calculate distance to enemy VIPs.
        Based on EnemyVipDistanceCalculator.
        """
        unit_x = action.get("to_x", action.get("from_x", 0))
        unit_y = action.get("to_y", action.get("from_y", 0))
        team_id = action.get("team_id", -1)
        
        # Find closest "VIP" enemy (here we'll just use the enemy with the highest damage)
        closest_vip_distance = float('inf')
        
        for unit in state.values():
            if unit.get("team_id", -1) != team_id:
                # Determine if unit is VIP based on role or other criteria
                # For simplicity, we'll consider high-damage units as VIPs
                is_vip = unit.get("total_damage", 0) > 60
                
                if is_vip:
                    enemy_x = unit.get("x", 0)
                    enemy_y = unit.get("y", 0)
                    distance = np.sqrt((enemy_x - unit_x)**2 + (enemy_y - unit_y)**2)
                    
                    closest_vip_distance = min(closest_vip_distance, distance)
        
        if closest_vip_distance == float('inf'):
            return 0.0  # No VIPs found
        
        # Normalize distance (34 is approximately 2 boards width)
        max_distance = action.get("max_range")
        normalized_distance = max(0, 1 - (closest_vip_distance / max_distance))
        
        return normalized_distance

    @classmethod
    def _calculate_is_crippled(cls, action, state):
        """
        Calculate if unit is crippled.
        Based on IsCrippledCalculator.
        """
        unit_id = action.get("entity_id", -1)
        unit_state = state.get(unit_id, {})
        
        is_crippled = unit_state.get("crippled", 0)
        
        return float(is_crippled)

    @classmethod
    def _calculate_moving_toward_waypoint(cls, action, state):
        """
        Calculate if the unit is moving toward a waypoint.
        Based on MovingTowardWaypointCalculator.
        """
        # This requires knowledge of waypoints, which aren't directly in our data
        # For simplicity, we'll check if the unit is moving toward enemy team centroid
        
        from_x = action.get("from_x", 0)
        from_y = action.get("from_y", 0)
        to_x = action.get("to_x", 0)
        to_y = action.get("to_y", 0)
        team_id = action.get("team_id", -1)
        
        # Calculate team centroid (as approximation of waypoint)
        team_center_x = 0
        team_center_y = 0
        team_count = 0
        
        for unit in state.values():
            if unit.get("team_id", -1) != team_id:
                team_center_x += unit.get("x", 0)
                team_center_y += unit.get("y", 0)
                team_count += 1
        
        if team_count > 0:
            team_center_x /= team_count
            team_center_y /= team_count
        
        # Calculate distances
        dist_from_start = np.sqrt((team_center_x - from_x)**2 + (team_center_y - from_y)**2)
        dist_from_end = np.sqrt((team_center_x - to_x)**2 + (team_center_y - to_y)**2)
        
        # Moving toward waypoint if end position is closer
        moving_toward = dist_from_end < dist_from_start
        
        return 1.0 if moving_toward else 0.0

    @classmethod
    def _calculate_unit_role(cls, action, state):
        """
        Calculate how well the unit performs its role.
        Based on UnitRoleCalculator.
        """
        role = action.get("role", "NONE")
        max_range = action.get("max_range", 0)
        # Calculate score based on role
        role_score = 0.5  # Default
        
        if role == "SNIPER":
            # Snipers prefer distance from enemies
            dist, _ = cls._closest_enemy(action, state)
            role_score = 1 - (abs(dist - (max_range * 0.7)) / (max_range * 0.3))
        elif role == "STRIKER":
            # Strikers prefer medium distance
            dist, _ = cls._closest_enemy(action, state)
            role_score = 1 - (dist / 10.0)
        elif role == "JUGGERNAUT":
            # Juggernauts prefer close combat
            dist, _ = cls._closest_enemy(action, state)
            role_score = 1 - (dist / 7.0)
        elif role == "MISSILE_BOAT":
            # Missile boats like ranges that match their weapons
            dist, _ = cls._closest_enemy(action, state)
            role_score = 1 - (abs(dist - max_range) / max_range)
        elif role == "SCOUT":
            # Scouts prefer high TMM and to be under low threat from enemy
            role_score = 1 / (len(cls._threatening_enemies(action, state)) + 1)
            role_score = role_score * cls._calculate_unit_tmm(action)
        elif role == "BRAWLER":
            # Brawlers prefer close combat with up to 3 units
            role_score = 3 / (len(cls._threatening_enemies(action, state)) + 1)
        elif role == "SKIRMISHER":
            damage_ratio = cls._calculate_damage_ratio(action, state)
            tmm_score = cls._calculate_unit_tmm(action)
            role_score = tmm_score * 0.6 + damage_ratio * 0.4
        elif role == "AMBUSHER":
            enemies_close_by = 1 / (len(cls._threatening_enemies(action, state)) + 1)
            distance = cls._distance_from_closest_enemy(action, state)
            to_divide_for = min(7, action.get("max_range", 7))
            role_score = clamp01(0.7 * enemies_close_by) * (clamp01(1 - (distance / to_divide_for)) * 0.3)

        return clamp01(role_score)

    @staticmethod
    def _calculate_unit_tmm(action):
        """
        Calculate target movement modifier.
        Based on UnitTmmCalculator.
        """
        hexes_moved = action.get("hexes_moved", 0)
        jumped = action.get("jumped", 0) == 1
        prone = action.get("prone", 0) == 1

        tmm = 1.0 if jumped else 0.0
        tmm -= 2.0 if prone else 0.0
        # In BattleTech, TMM typically increases with hexes moved
        if hexes_moved >= 25:
            tmm += 6
        elif hexes_moved >= 18:
            tmm += 5
        elif hexes_moved >= 10:
            tmm += 4
        elif hexes_moved >= 7:
            tmm += 3
        elif hexes_moved >= 5:
            tmm += 2
        elif hexes_moved >= 3:
            tmm += 1
        if hexes_moved == 0:
            tmm -= 4

        return clamp01(tmm / 7)

    def _calculate_piloting_caution(self, action):
        """
        Calculate piloting caution.
        Based on PilotingCautionCalculator.
        """
        # Consider chance of failure (PSR checks)
        chance_of_failure = action.get("chance_of_failure", 0.0)
        
        # Also consider terrain hazards
        hazard_level = self._calculate_environmental_hazards(action)
        
        # Combine factors
        caution_score = max(chance_of_failure, hazard_level)
        
        return caution_score

    @classmethod
    def _calculate_standing_still(cls, action):
        """
        Calculate if the unit is standing still.
        Based on StandingStillCalculator.
        """
        hexes_moved = action.get("hexes_moved", 0)
        
        # Simple check for movement
        return 1.0 if hexes_moved == 0 else 0.0

    def _calculate_target_health(self, action, state):
        """
        Calculate target health.
        Based on TargetHealthCalculator.
        """
        # Find the closest enemy
        _, closest_enemy = self._closest_enemy(action, state)
        
        if closest_enemy is None:
            return 0.5  # No target
        
        # Get enemy health
        armor_percentage = closest_enemy.get("armor_p", 0)
        internal_percentage = closest_enemy.get("internal_p", 0)

        return (armor_percentage + internal_percentage) / 2

    @classmethod
    def _calculate_target_within_optimal_range(cls, action, state):
        """
        Calculate if targets are within optimal weapon range.
        Based on TargetWithinOptimalRangeCalculator.
        """
        unit_x = action.get("to_x", action.get("from_x", 0))
        unit_y = action.get("to_y", action.get("from_y", 0))
        team_id = action.get("team_id", -1)
        max_range = action.get("max_range")
        
        # Assume optimal range is 70% of maximum range
        optimal_min = int(max_range * 0.7)
        
        # Calculate score based on enemies in optimal range
        enemies_in_optimal_range = 0
        total_enemies = 0
        
        for unit in state.values():
            if unit.get("team_id", -1) != team_id:
                enemy_x = unit.get("x", 0)
                enemy_y = unit.get("y", 0)
                distance = np.sqrt((enemy_x - unit_x)**2 + (enemy_y - unit_y)**2)
                
                total_enemies += 1
                if optimal_min <= distance <= max_range:
                    enemies_in_optimal_range += 1
        
        if total_enemies == 0:
            return 0.0
        
        return enemies_in_optimal_range / total_enemies

    @classmethod
    def _n_threat(cls, action, state, n):
        """
        Calculate the distance of the closest 4 enemies and return
        as an array.
        """
        
        x = action.get("to_x", 0)
        y = action.get("to_y", 0)
        team_id = action.get("team_id", -2)
        max_range = max(action.get("max_range"), 1.0)
        total_health = action.get("armor", 0) + action.get("internal", 0)
        # normalizes from 1 to 0.3
        my_tmm = 1 - (((cls._calculate_unit_tmm(action) + 1) / 3) * 2)
        threat = []

        for unit in state.values():
            if unit.get("team_id", -1) != team_id:
                targeting_dist = max(unit.get("max_range"), 1.0)
                dist = cls._distance(x, y, unit)

                i_can_target = 0.8 if dist <= max_range else 1.0
                i_can_be_targeted = 1.0 if dist <= targeting_dist else 0.5

                damage = unit.get("total_damage", 1)
                damage_ratio = damage / total_health

                t = damage_ratio / my_tmm * i_can_target * i_can_be_targeted
                threat.append(clamp01(t))

        return cls.sorted_clipped_array(threat, n)

    @classmethod
    def sorted_clipped_array(cls, threat, n):
        sorted_threat = sorted(threat, reverse=True)
        output = [0] * n
        last_value = 0
        for i in range(n):
            if i < len(sorted_threat):
                last_value = clamp01(sorted_threat[i])
            output[i] = last_value

        return output

    @classmethod
    def _self_threat(cls, action, state):
        """
        Calculate the distance of the closest 4 enemies and return
        as an array.
        """
        max_range = max(action.get("max_range"), 1.0)
        damage = action.get("total_damage", 0)

        dist, enemy = cls._closest_enemy(action, state)

        if not enemy:
            return 1.0

        total_health = enemy.get("armor", 0) + enemy.get("internal", 0)
        targeting_dist = max(enemy.get("max_range"), 1.0)

        i_can_target = 1.0 if dist <= max_range else 0.6
        i_can_be_targeted = 0.8 if dist <= targeting_dist else 1.0

        damage_ratio = damage / total_health

        threat = damage_ratio * i_can_target * i_can_be_targeted

        return clamp01(threat)

    @classmethod
    def _n_threat_allies(cls, action, state, n):
        """
        Calculate the distance of the closest 4 enemies and return
        as an array.
        """
        my_id = action.get("entity_id", -1)
        x = action.get("to_x", 0)
        y = action.get("to_y", 0)
        team_id = action.get("team_id", -2)

        _, enemy = cls._closest_enemy(action, state)
        if not enemy:
            return [0] * n
        total_health = enemy.get("armor", 0) + enemy.get("internal", 0)
        threat = []

        for unit in state.values():
            if unit.get("team_id", -1) == team_id and unit.get("entity_id") != my_id:
                targeting_dist = max(unit.get("max_range"), 1.0)
                dist = cls._distance(x, y, unit)

                cover_me = 1.0 if dist <= targeting_dist else 0.5

                damage = unit.get("total_damage", 0)
                damage_ratio = damage / total_health

                t = damage_ratio * cover_me
                threat.append(clamp01(t))

        return cls.sorted_clipped_array(threat, n)


    @classmethod
    def _calculate_turns_to_encounter(cls, action, state):
        """
        Calculate estimated turns until enemy encounter.
        Based on TurnsToEncounterCalculator.
        """
        closest_distance, enemy = cls._closest_enemy(action, state)
        if closest_distance == float('inf'):
            return 0.0  # No enemies
        
        # Estimate turns based on average movement per turn (e.g., 5 hexes)
        avg_movement_per_turn = enemy.get("mp", 0)
        if avg_movement_per_turn == 0:
            avg_movement_per_turn = float('inf')

        turns_to_encounter = closest_distance / avg_movement_per_turn
        
        # Normalize (assuming max 5 turns is far future)
        return clamp01(1 - (turns_to_encounter / 5))

    @classmethod
    def _calculate_enemy_threat_heatmap(cls, action, state):
        """
        Calculate a 10x10 heatmap of enemy threat.
        Based on EnemyThreatHeatmapCalculator.
        """
        smoothed_heatmap = cls.calculate_threat_heatmap(
            state, lambda unit: action.get('team_id') != unit.get('team_id'))

        # Normalize values
        norm = np.linalg.norm(smoothed_heatmap)

        if norm == 0:
            return smoothed_heatmap

        return smoothed_heatmap / norm

    @classmethod
    def _calculate_enemy_threat_radius(cls, action, state):
        max_x = max(unit.get('x') for unit in state.values())
        max_y = max(unit.get('y') for unit in state.values())
        max_index_pos = max_x * max_y
        heatmap = np.zeros(max_index_pos)
        team_id = action.get("team_id", -1)

        for unit in state.values():
            if unit.get("team_id", -1) != team_id:
                index = cls.calculate_heatmap_index(max_x, max_y, unit)

                if 0 <= index < max_index_pos:
                    threat_value = unit.get("max_range", 0)
                    heatmap[index] = threat_value

        # Smooth the heatmap
        smoothed_heatmap = cls.calculate_smoothed_heatmap(heatmap, max_x, max_y)

        return smoothed_heatmap


    @classmethod
    def _calculate_friendly_threat_radius(cls, action, state):
        team_id = action.get("team_id", -1)
        unit_id = action.get("entity_id", -1)
        max_x = max(unit.get('x') for unit in state.values())
        max_y = max(unit.get('y') for unit in state.values())
        max_index_pos = max_x * max_y
        heatmap = np.zeros(max_index_pos)

        for unit in state.values():
            if unit.get("team_id", -1) == team_id and unit.get("entity_id", -1) != unit_id:
                index = cls.calculate_heatmap_index(max_x, max_y, unit)

                if 0 <= index < max_index_pos:
                    threat_value = unit.get("max_range", 0)
                    heatmap[index] = threat_value

        # Smooth the heatmap
        smoothed_heatmap = cls.calculate_smoothed_heatmap(heatmap, max_x, max_y)

        return smoothed_heatmap

    @classmethod
    def calculate_smoothed_heatmap(cls, heatmap, max_x, max_y):
        smoothed_heatmap = np.copy(heatmap)
        for y in range(max_y):
            for x in range(max_x):
                index = y * max_y + x
                center_value = heatmap[index]
                if center_value != 0:
                    for dy in range(-center_value, center_value):
                        for dx in range(-center_value, center_value):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < max_x and 0 <= ny < max_y and (dx != 0 or dy != 0):
                                adjacent_index = ny * max_y + nx
                                dist = np.sqrt((nx - x) ** 2 + (ny - y) ** 2)
                                # noinspection PyTypeChecker
                                smoothed_heatmap[adjacent_index] = max(
                                    smoothed_heatmap[adjacent_index],
                                    min(center_value - dist, 0)
                                )
        # Normalize values
        norm = np.linalg.norm(smoothed_heatmap)
        if norm > 0:
            smoothed_heatmap = smoothed_heatmap / norm
        return smoothed_heatmap

    @classmethod
    def _calculate_friendly_threat_heatmap(cls, action, state):
        """
                Calculate a 10x10 heatmap of enemy threat.
                Based on EnemyThreatHeatmapCalculator.
                """
        # Create a 10x10 heatmap (100 values)
        smoothed_heatmap = cls.calculate_threat_heatmap(
            state, lambda unit: action.get('team_id') == unit.get('team_id'))

        # Normalize values
        norm = np.linalg.norm(smoothed_heatmap)

        if norm == 0:
            return smoothed_heatmap

        return smoothed_heatmap / norm

    @classmethod
    def calculate_threat_heatmap(cls, state, filter_fn):
        heatmap = np.zeros(100)
        max_x = 10
        max_y = 10
        for unit in state.values():
            max_x = max(max_x, unit.get("x", 0))
            max_y = max(max_y, unit.get("y", 0))
        board_width = max_x
        board_height = max_y
        # Scale factor to convert from board coordinates to 10x10 grid
        scale_x = 10.0 / board_width
        scale_y = 10.0 / board_height
        for unit in state.values():
            if filter_fn(unit):
                index = cls.calculate_heatmap_index(scale_x, scale_y, unit)

                # Add threat based on range
                if 0 <= index < 100:
                    heatmap[index] = unit.get("max_range", heatmap[index])
        # Add decreasing threat in surrounding cells
        smoothed_heatmap = np.copy(heatmap)
        for y in range(10):
            for x in range(10):
                index = y * 10 + x
                center_value = heatmap[index]

                # Skip cells with no threat
                if center_value <= 0:
                    continue

                # Apply decay to surrounding cells based on distance
                decay_radius = 3  # How far the threat spreads
                for ny in range(max(0, y - decay_radius), min(10, y + decay_radius + 1)):
                    for nx in range(max(0, x - decay_radius), min(10, x + decay_radius + 1)):
                        # Skip the center cell itself
                        if nx == x and ny == y:
                            continue

                        # Calculate distance
                        distance = np.sqrt((nx - x) ** 2 + (ny - y) ** 2)

                        # Calculate decay factor (closer = more threat)
                        if distance <= decay_radius:
                            decay_factor = center_value / (distance + 1)
                            # Update the smoothed heatmap with maximum threat at this position
                            smoothed_index = ny * 10 + nx
                            smoothed_heatmap[smoothed_index] = max(float(smoothed_heatmap[smoothed_index]),
                                                                   decay_factor)
        return smoothed_heatmap

    @staticmethod
    def calculate_heatmap_index(scale_x, scale_y, unit):
        enemy_x = unit.get("x", 0)
        enemy_y = unit.get("y", 0)
        # Convert to heatmap coordinates
        heatmap_x = min(int(enemy_x * scale_x), 9)
        heatmap_y = min(int(enemy_y * scale_y), 9)
        # Calculate index in flattened 10x10 grid
        index = heatmap_y * 10 + heatmap_x
        return index

    def _calculate_retreat(self, action, state):
        """
        Calculate retreat score.
        Based on RetreatCalculator.
        """
        # Consider unit health and enemy proximity
        unit_health = clamp01((action.get("armor_p", 0) + action.get("internal_p", 0)) / 2.0)
        
        # Get distance from the closest enemy
        dist = self._distance_from_closest_enemy(action, state)
        
        # Calculate retreat score (higher when damaged and enemies are close)
        retreat_score = clamp01(1.0 - unit_health) * (1.0 / max(1, dist / 3.0))
        
        return clamp01(retreat_score)

    @staticmethod
    def _calculate_scouting(action, state):
        """
        Calculate scouting value.
        Based on ScoutingCalculator.
        """
        # Consider speed and role
        unit_id = action.get("entity_id", -1)
        unit_state = state.get(unit_id, {})
        
        role = unit_state.get("role", "NONE")
        max_mp = action.get("max_mp", 0)
        hexes_moved = action.get("hexes_moved", 0)
        
        # Scout role and high movement get bonus
        is_scout = role == "SCOUT"
        is_fast = max_mp >= 8
        
        # Also consider distance from team
        unit_x = action.get("to_x", action.get("from_x", 0))
        unit_y = action.get("to_y", action.get("from_y", 0))
        team_id = action.get("team_id", -1)
        
        # Calculate distance from team center
        team_positions = [(unit.get("x", 0), unit.get("y", 0)) for unit in state.values() 
                        if unit.get("team_id", -1) == team_id and unit.get("entity_id", -1) != unit_id]
        
        if not team_positions:
            avg_team_distance = 0
        else:
            team_x = sum(x for x, _ in team_positions) / len(team_positions)
            team_y = sum(y for _, y in team_positions) / len(team_positions)
            avg_team_distance = np.sqrt((team_x - unit_x)**2 + (team_y - unit_y)**2)
        
        # Calculate scouting score
        scouting_score = 0.0
        
        if is_scout:
            scouting_score += 0.4
        if is_fast:
            scouting_score += 0.3
        
        # Movement utilization
        movement_factor = min(hexes_moved / max(1, max_mp), 1.0) * 0.2
        
        # Distance from team (scouts should be forward)
        distance_factor = min(avg_team_distance / 15.0, 1.0) * 0.1
        
        return clamp01(scouting_score + movement_factor + distance_factor)

    @classmethod
    def _calculate_strategic_goal(cls, action, state):
        """
        Calculate alignment with strategic goals.
        Based on StrategicGoalCalculator.
        """
        # Extract action data
        from_x = action.get("from_x", 0)
        from_y = action.get("from_y", 0)
        to_x = action.get("to_x", 0)
        to_y = action.get("to_y", 0)
        team_id = action.get("team_id", -1)
        unit_id = action.get("entity_id", -1)

        # Calculate control areas (using control radius of 7 hexes)
        control_radius = 7
        grid_size = 40  # Resolution of the grid

        # Get all units positions by team
        friendly_units = []
        enemy_units = []

        for unit in state.values():
            if unit.get("entity_id", -1) != unit_id:  # Exclude current unit's old position
                if unit.get("team_id", -1) == team_id:
                    friendly_units.append((unit.get("x", 0), unit.get("y", 0)))
                else:
                    enemy_units.append((unit.get("x", 0), unit.get("y", 0)))

        # Add current unit's position
        friendly_units_before = friendly_units + [(from_x, from_y)]
        friendly_units_after = friendly_units + [(to_x, to_y)]

        # Calculate area control before and after move
        control_before = cls._calculate_area_control(friendly_units_before, control_radius, grid_size)
        control_after = cls._calculate_area_control(friendly_units_after, control_radius, grid_size)

        # Calculate disputed territory before and after move
        disputed_before = cls._calculate_disputed_territory(friendly_units_before, enemy_units, control_radius)
        disputed_after = cls._calculate_disputed_territory(friendly_units_after, enemy_units, control_radius)

        # Calculate strategic improvement
        control_improvement = (control_after - control_before) / max(control_before, 1)
        dispute_improvement = (disputed_before - disputed_after) / max(disputed_before, 1)

        # Weight factors
        control_weight = 0.7
        dispute_weight = 0.3

        # Calculate final strategic score
        strategic_score = (control_weight * clamp01(control_improvement + 0.5)) + \
                          (dispute_weight * clamp01(dispute_improvement + 0.5))

        return clamp01(strategic_score)

    @classmethod
    def _calculate_area_control(cls, team_units, radius, grid_size):
        """
        Calculate approximate area controlled by a team using a grid-based approach.

        Returns a normalized score representing controlled area.
        """
        # Use a grid to approximate control
        grid = cls._make_grid(team_units, radius, grid_size)

        # Calculate total controlled area
        controlled_area = np.sum(grid)
        max_area = grid_size * grid_size

        return controlled_area / max_area

    @classmethod
    def _make_grid(cls, units, radius, grid_size):
        grid = np.zeros((grid_size, grid_size))

        # Find min/max coordinates to determine bounds

        if not units:
            return grid

        min_x = min(x for x, _ in units) - radius
        max_x = max(x for x, _ in units) + radius
        min_y = min(y for _, y in units) - radius
        max_y = max(y for _, y in units) + radius

        # Adjust for grid bounds
        width = max(1, max_x - min_x)
        height = max(1, max_y - min_y)

        # Mark control for each friendly unit
        for x, y in units:
            # Convert unit position to grid coordinates
            grid_x = int((x - min_x) * (grid_size / width))
            grid_y = int((y - min_y) * (grid_size / height))

            # Mark control in radius
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx * dx + dy * dy <= radius * radius:  # Within circular radius
                        gx = grid_x + dx
                        gy = grid_y + dy
                        if 0 <= gx < grid_size and 0 <= gy < grid_size:
                            grid[gx, gy] = 1
        return grid

    @classmethod
    def _calculate_disputed_territory(cls, team_units, enemy_units, radius):
        """
        Calculate approximate area that is disputed (controlled by both teams).

        Returns a normalized score representing disputed area.
        """
        # Use a grid to approximate control
        grid_size = 40  # Resolution of the grid
        friendly_grid = cls._make_grid(team_units, radius, grid_size)
        enemy_grid = cls._make_grid(enemy_units, radius, grid_size)

        # Calculate disputed area (where both teams have control)
        disputed_grid = friendly_grid * enemy_grid
        disputed_area = np.sum(disputed_grid)
        max_area = grid_size * grid_size

        return disputed_area / max_area

    @classmethod
    def _calculate_favorite_target_in_range(cls, action, state):
        """
        Calculate if preferred target types are in range.
        Based on FavoriteTargetInRangeCalculator.
        """
        unit_id = action.get("entity_id", -1)
        unit_state = state.get(unit_id, {})
        
        unit_x = action.get("to_x", action.get("from_x", 0))
        unit_y = action.get("to_y", action.get("from_y", 0))
        team_id = action.get("team_id", -1)
        
        role = unit_state.get("role", "NONE")
        max_range = unit_state.get("max_range", 0)

        preferred_targets = {"MISSILE_BOAT": 0.9, "SNIPER": 0.9, "STRIKER": 0.5, "SCOUT": 0.3, "JUGGERNAUT": 0.7}

        if role == "SNIPER":
            preferred_targets = {"MISSILE_BOAT": 0.9, "SNIPER": 0.9, "STRIKER": 0.5, "SCOUT": 0.5, "JUGGERNAUT": 0.7}
        elif role == "JUGGERNAUT":
            preferred_targets = {"MISSILE_BOAT": 0.9, "SNIPER": 0.9, "STRIKER": 0.5, "SCOUT": 0.3, "JUGGERNAUT": 1.0}
        elif role == "STRIKER":
            preferred_targets = {"MISSILE_BOAT": 0.9, "SNIPER": 0.9, "STRIKER": 0.6, "SCOUT": 0.5, "JUGGERNAUT": 0.4}
        elif role == "MISSILE_BOAT":
            preferred_targets = {"MISSILE_BOAT": 0.9, "SNIPER": 0.9, "STRIKER": 0.7, "SCOUT": 0.6, "JUGGERNAUT": 0.3}

        
        # Look for preferred targets in range
        best_target_score = 0.0
        
        for unit in state.values():
            if unit.get("team_id", -1) != team_id:
                enemy_role = unit.get("role", "NONE")
                enemy_x = unit.get("x", 0)
                enemy_y = unit.get("y", 0)
                
                distance = np.sqrt((enemy_x - unit_x)**2 + (enemy_y - unit_y)**2)
                
                if distance <= max_range and enemy_role in preferred_targets:
                    target_score = preferred_targets[enemy_role] * (1 - (distance / max_range))
                    best_target_score = max(best_target_score, target_score)
        
        return clamp01(best_target_score)

    @classmethod
    def _calculate_friendly_artillery_fire(cls, action, state):
        """
        Calculate friendly artillery coverage.
        Based on FriendlyArtilleryFireCalculator.
        """
        unit_x = action.get("to_x", action.get("from_x", 0))
        unit_y = action.get("to_y", action.get("from_y", 0))
        team_id = action.get("team_id", -1)
        unit_id = action.get("entity_id", -1)
        self_max_range = action.get("max_range")
        artillery_support = 0
        
        for unit in state.values():
            if unit.get("team_id", -1) == team_id and unit.get("entity_id", -1) != unit_id:
                max_range = unit.get("max_range", 0)
                # Consider units with very long range as artillery
                if max_range >= 22:
                    friendly_x = unit.get("x", 0)
                    friendly_y = unit.get("y", 0)
                    
                    # Artillery typically provides support over a wide area
                    distance = np.sqrt((friendly_x - unit_x)**2 + (friendly_y - unit_y)**2)
                    
                    if distance <= 17:  # Artillery support range
                        artillery_support = max(artillery_support, max_range - 17)

        return clamp01(artillery_support / (self_max_range + 1))

    @classmethod
    def _calculate_threat_by_sniper(cls, action, state):
        """
        Calculate threat posed by enemy snipers.
        Based on ThreatByRoleCalculator.
        """
        return cls._calculate_threat_by_role(action, state, "SNIPER")

    @classmethod
    def _calculate_threat_by_missile_boat(cls, action, state):
        """
        Calculate threat posed by enemy missile boats.
        Based on ThreatByRoleCalculator.
        """
        return cls._calculate_threat_by_role(action, state, "MISSILE_BOAT")

    @classmethod
    def _calculate_threat_by_juggernaut(cls, action, state):
        """
        Calculate threat posed by enemy juggernauts.
        Based on ThreatByRoleCalculator.
        """
        return cls._calculate_threat_by_role(action, state, "JUGGERNAUT")

    @classmethod
    def _calculate_threat_by_role(cls, action, state, target_role):
        """
        Helper method to calculate threat by a specific role.
        """
        unit_x = action.get("to_x", action.get("from_x", 0))
        unit_y = action.get("to_y", action.get("from_y", 0))
        team_id = action.get("team_id", -2)
        unit_health = max(action.get("armor", 0) + action.get("internal", 0), 1)
        threat_level = 0.0
        
        for unit in state.values():
            if unit.get("team_id", -1) != team_id and unit.get("role", "") == target_role:
                enemy_x = unit.get("x", 0)
                enemy_y = unit.get("y", 0)
                
                distance = np.sqrt((enemy_x - unit_x)**2 + (enemy_y - unit_y)**2)
                
                # Calculate threat based on distance and damage potential
                max_range = unit.get("max_range", 0)
                total_damage = unit.get("total_damage", 0)
                
                # Only consider units that can reach us
                if distance <= max_range:
                    # Normalize damage and apply distance modifier
                    damage_factor = clamp01(total_damage / unit_health)
                    distance_factor = 1 - (distance / max_range)
                    
                    threat_level += damage_factor * distance_factor
        
        # Cap at 1.0
        return clamp01(threat_level)

    @classmethod
    def calculate_reward(cls, previous_state, current_state, action_taken, all_previous_units, all_current_units):
        """
        Calculate the reward for transitioning from previous_state to current_state
        after taking action_taken.

        Returns a scalar reward value.
        """
        # 1. Damage dealt/received component
        damage_reward = cls.calculate_damage_reward(current_state, all_previous_units, all_current_units)

        # 2. Tactical positioning reward
        position_reward = cls.calculate_position_reward(action_taken, current_state, all_current_units)

        # 3. Heat management reward
        heat_reward = cls.calculate_heat_reward(previous_state, current_state, action_taken)

        # 4. Unit preservation reward
        preservation_reward = cls.calculate_preservation_reward(previous_state, current_state, all_previous_units,
                                                                all_current_units)

        # 5. Team coordination reward
        coordination_reward = cls.calculate_coordination_reward(previous_state, current_state, all_current_units)

        # 6. Victory progress reward
        victory_reward = cls.calculate_victory_reward(current_state, all_previous_units, all_current_units)

        is_bot_reward = 0.1 if action_taken.get("is_bot", 1.0) == 1.0 else 1.0

        # Combine all rewards with appropriate weights
        total_reward = clamp01(
                0.5 * damage_reward +
                0.2 * position_reward +
                0.1 * heat_reward +
                0.3 * preservation_reward +
                0.2 * coordination_reward +
                1.0 * victory_reward
        ) * is_bot_reward

        return total_reward

    @classmethod
    def sigmoid(cls, x):
        return 1 / (1 + np.exp(-x))

    @classmethod
    def calculate_preservation_reward(cls, previous_state, current_state, all_previous_units, all_current_units):
        """
        Reward for preserving units and penalize losses.
        """
        reward = 0

        # Get team ID
        team_id = current_state['team_id']

        # Calculate team BV value before and after
        previous_team_bv = sum(u['bv'] for u in all_previous_units.values() if u['team_id'] == team_id and not u['destroyed'])
        current_team_bv = sum(u['bv'] for u in all_current_units.values() if u['team_id'] == team_id and not u['destroyed'])

        # Calculate enemy BV value before and after
        previous_enemy_bv = sum(u['bv'] for u in all_previous_units.values() if u['team_id'] != team_id and not u['destroyed'])
        current_enemy_bv = sum(u['bv'] for u in all_current_units.values() if u['team_id'] != team_id and not u['destroyed'])

        # Calculate BV exchange ratio
        team_bv_lost = previous_team_bv - current_team_bv
        enemy_bv_lost = previous_enemy_bv - current_enemy_bv

        # Calculate BV-weighted reward
        if team_bv_lost > 0:
            # We lost BV value - check if trade was favorable
            if enemy_bv_lost > team_bv_lost:
                reward += 0.5 * (enemy_bv_lost / team_bv_lost - 1.0)  # Favorable trade
            else:
                reward -= 0.5  # Unfavorable trade
        elif enemy_bv_lost > 0:
            # We didn't lose any BV but enemy did
            reward += 0.5 * (enemy_bv_lost / 1000.0)  # Scale by typical BV values

        # Special penalty for crippling our unit
        if not previous_state['crippled'] and current_state['crippled']:
            reward -= 1.0

        # Special penalty for destroying our unit
        if not previous_state['destroyed'] and current_state['destroyed']:
            reward -= 2.0

        return reward

    @classmethod
    def calculate_damage_reward(cls, current_state, all_previous_units, all_current_units):
        """
        Reward for damage dealt to enemies and penalty for damage received.
        """
        # Get team ID
        team_id = current_state['team_id']

        # Calculate damage dealt to enemies
        damage_dealt = 0
        for prev_unit in all_previous_units.values():
            if prev_unit['team_id'] != team_id:
                # Find the same unit in current state
                current_unit = next((u for u in all_current_units.values() if u['entity_id'] == prev_unit['entity_id']), None)
                if current_unit:
                    # Calculate armor and internal damage
                    armor_damage = (prev_unit['armor'] * prev_unit['armor_p']) - (
                                current_unit['armor'] * current_unit['armor_p'])
                    internal_damage = (prev_unit['internal'] * prev_unit['internal_p']) - (
                                current_unit['internal'] * current_unit['internal_p'])
                    damage_dealt += max(0, armor_damage + internal_damage)

        # Calculate damage received by friendly units
        damage_received = 0
        for prev_unit in all_previous_units.values():
            if prev_unit['team_id'] == team_id:
                # Find the same unit in current state
                current_unit = next((u for u in all_current_units.values() if u['entity_id'] == prev_unit['entity_id']), None)
                if current_unit:
                    # Calculate armor and internal damage
                    armor_damage = (prev_unit['armor'] * prev_unit['armor_p']) - (
                                current_unit['armor'] * current_unit['armor_p'])
                    internal_damage = (prev_unit['internal'] * prev_unit['internal_p']) - (
                                current_unit['internal'] * current_unit['internal_p'])
                    damage_received += max(0, armor_damage + internal_damage)

        # Net damage reward (weighted to prioritize dealing damage over avoiding it)
        net_damage_reward = (1.0 * damage_dealt) - (0.7 * damage_received)

        # Scale the reward
        return net_damage_reward / 100.0  # Scale based on typical damage values in your game

    @classmethod
    def calculate_position_reward(cls, action, current_unit, all_current_units):
        """
        Reward for good tactical positioning.
        """
        reward = 0

        # Get enemy units
        enemy_units = [u for u in all_current_units.values() if u['team_id'] != current_unit['team_id']]
        if not enemy_units:
            return 0  # No enemies to position against

        # 1. Reward for moving to optimal weapon range
        optimal_range = current_unit['max_range'] * 0.7
        closest_enemy_distance = min([cls.calculate_distance(current_unit, enemy) for enemy in enemy_units])
        range_optimality = 1.0 - min(abs(closest_enemy_distance - optimal_range) / optimal_range, 1.0)
        reward += 0.3 * range_optimality

        # 2. Reward for facing toward enemies
        angle_to_enemy = cls._calculate_facing_enemy(action, all_current_units)
        facing_reward = 1.0 if angle_to_enemy == current_unit['facing'] else 0.0
        reward += 0.2 * facing_reward

        # 4. Reward for taking cover (simplified - would need terrain data)
        # Here we're just checking if position is even/odd as a placeholder for actual cover mechanics
        if current_unit['x'] % 2 == 0 and current_unit['y'] % 2 == 0:
            reward += 0.1

        # 5. Penalty for ending move in the open when enemies can target you
        exposure_penalty = 0
        for enemy in enemy_units:
            if cls.calculate_distance(current_unit, enemy) <= enemy['max_range']:
                exposure_penalty += 0.1
        reward -= min(exposure_penalty, 0.3)  # Cap the penalty

        return reward

    @classmethod
    def calculate_heat_reward(cls, previous_state, current_state, action_taken):
        reward = 0

        # Penalty for increasing heat beyond safe levels
        if current_state['heat_p'] > 0.7 and current_state['heat_p'] > previous_state['heat_p']:
            reward -= 0.2 * (current_state['heat_p'] - 0.7)

        # Reward for decreasing dangerous heat levels
        if previous_state['heat_p'] > 0.7 and current_state['heat_p'] < previous_state['heat_p']:
            reward += 0.1 * (previous_state['heat_p'] - current_state['heat_p'])

        # Reward for maintaining efficient heat levels
        if current_state['heat_p'] < 0.5:
            reward += 0.05

        # Penalty for shutdown risk
        if current_state['heat_p'] > 0.9:
            reward -= 0.3

        # Jumping heat efficiency (if jumping was more efficient than walking)
        if action_taken['jumping'] and action_taken['distance'] > 0:
            heat_per_hex_jumped = action_taken['heat_p'] / action_taken['distance']
            walking_heat_per_hex = 0.05  # Approximate heat for walking 1 hex
            if heat_per_hex_jumped < walking_heat_per_hex:
                reward += 0.1

        return reward

    @classmethod
    def calculate_coordination_reward(cls, previous_state, current_state, all_current_units):
        """
        Reward for coordinating with team members.
        """
        reward = 0

        # Get team ID
        team_id = current_state['team_id']

        # Get friendly units
        friendly_units = [u for u in all_current_units.values() if
                          u['team_id'] == team_id and u['entity_id'] != current_state['entity_id']]
        if not friendly_units:
            return 0  # No teammates to coordinate with

        # Calculate team centroid
        team_x = sum(u['x'] for u in friendly_units) / len(friendly_units)
        team_y = sum(u['y'] for u in friendly_units) / len(friendly_units)
        team_centroid = {'x': team_x, 'y': team_y}

        # Calculate previous and current distances to team centroid
        prev_distance_to_team = cls.calculate_distance(previous_state, team_centroid)
        current_distance_to_team = cls.calculate_distance(current_state, team_centroid)

        # Reward for maintaining formation (not too far, not too close)
        optimal_formation_distance = 3  # Adjust based on your game's tactics
        prev_formation_score = 1.0 - min(
            abs(prev_distance_to_team - optimal_formation_distance) / optimal_formation_distance, 1.0)
        current_formation_score = 1.0 - min(
            abs(current_distance_to_team - optimal_formation_distance) / optimal_formation_distance, 1.0)

        # Reward improvement in formation position
        formation_improvement = current_formation_score - prev_formation_score
        reward += 0.2 * formation_improvement

        # Reward for mutual support (units covering each other)
        mutual_support_count = sum(1 for unit in friendly_units if cls.calculate_distance(current_state, unit) <= 8)
        reward += 0.05 * min(mutual_support_count, 3)  # Cap the reward

        return reward

    @classmethod
    def calculate_victory_reward(cls, current_action, all_previous_units, all_current_units):
        """
        Reward for making progress toward victory conditions.
        """
        reward = 0

        # Get team ID
        team_id = current_action['team_id']

        # Count friendly and enemy units before and after
        prev_friendly_bv = sum(u['bv'] for u in all_previous_units.values() if u['team_id'] == team_id)
        prev_enemy_bv = sum(u['bv'] for u in all_previous_units.values() if u['team_id'] != team_id)

        current_friendly_bv = sum(u['bv'] for u in all_current_units.values() if u['team_id'] == team_id)
        current_enemy_bv = sum(u['bv'] for u in all_current_units.values() if u['team_id'] != team_id)

        # Calculate relative force ratio improvement
        if prev_enemy_bv == 0:
            prev_ratio = 0
        else:
            prev_ratio = prev_friendly_bv / max(prev_enemy_bv, 1)
        if current_enemy_bv == 0:
            current_ratio = 0
        else:
            current_ratio = current_friendly_bv / max(current_enemy_bv, 1)

        ratio_improvement = current_ratio - prev_ratio

        # Reward for improving force ratio
        reward += ratio_improvement

        # Major rewards for victory progress
        if prev_enemy_bv > 0 and current_enemy_bv == 0:
            reward += 10.0  # Eliminated all enemies

        if current_enemy_bv < prev_enemy_bv:
            reward += 2.0 * (prev_enemy_bv - current_enemy_bv)  # Destroyed enemy units

        # Penalty for team losses
        if current_friendly_bv < prev_friendly_bv:
            reward -= 1.0 * (prev_friendly_bv - current_friendly_bv)  # Lost friendly units

        # Reward for maintaining numerical advantage
        if current_friendly_bv > current_enemy_bv:
            reward += 0.1 * (current_friendly_bv - current_enemy_bv)

        return reward

    @classmethod
    def calculate_distance(cls, unit1, unit2):
        # Hex grid distance calculation
        dx = unit1['x'] - unit2['x']
        dy = unit1['y'] - unit2['y']
        return max(abs(dx), abs(dy))
