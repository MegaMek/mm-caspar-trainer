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

import re
from typing import Optional, List


class Hex:
    """Represents a single hex on the game board with all its terrain features."""

    def __init__(self, feature_str_in: str):
        """Initialize a hex from a feature string."""
        self.elevation = 0
        self.has_water = False
        self.depth = 0
        self.has_woods = False
        self.is_heavy_woods = False
        self.has_pavement = False
        self.has_building = False
        self.has_swamp = False
        self.has_magma = False
        self.has_bridge = False
        self.bridge_elevation = 0
        self.build_elevation = 0
        self.has_rough = False
        self.has_road = False
        self.has_rubble = False
        self.has_mud = False
        self.has_ice = False
        self.has_rapids = False
        self.has_geyser = False
        self.has_fire = False
        self.has_smoke = False
        self.has_ultra_sublevel = False
        self.has_hazardous_liquid = False
        self.has_fuel_tank = False

        feature_str = feature_str_in.lower()
        # Extract level information
        level_match = re.match(r"level: (-?\d+)", feature_str)
        if level_match:
            self.elevation = int(level_match.group(1))

        # Check for water
        if "water" in feature_str:
            self.has_water = True
            depth_match = re.search(r"water, depth: (\d+)", feature_str)
            if depth_match:
                self.depth = int(depth_match.group(1))

        # Check for woods
        if "woods" in feature_str:
            self.has_woods = True
            self.is_heavy_woods = "heavy woods" in feature_str

        # Check for roads and pavements
        if "pavement" in feature_str:
            self.has_pavement = True
        if "Road" in feature_str:
            self.has_road = True
            self.has_pavement = True

        # Check for buildings
        if "building" in feature_str:
            building_elevation_match = re.search(r"building\((\d+)", feature_str)
            if building_elevation_match:
                self.build_elevation = int(building_elevation_match.group(1))
            self.has_building = True

        # Check for bridges
        if "bridge" in feature_str:
            self.has_bridge = True
            bridge_elev_match = re.search(r"bridge_elev\((\d+)", feature_str)
            if bridge_elev_match:
                self.bridge_elevation = int(bridge_elev_match.group(1))

        if "rough" in feature_str:
            self.has_rough = True

        if "swamp" in feature_str:
            self.has_swamp = True

        if "magma" in feature_str:
            self.has_magma = True

        if "rubble" in feature_str:
            self.has_rubble = True

        if "mud" in feature_str:
            self.has_mud = True

        if "ice" in feature_str:
            self.has_ice = True

        if "rapids" in feature_str:
            self.has_rapids = True

        if "geyser" in feature_str:
            self.has_geyser = True

        if "fire" in feature_str:
            self.has_fire = True

        if "smoke" in feature_str:
            self.has_smoke = True

        if "ultra_sublevel" in feature_str:
            self.has_ultra_sublevel = True

        if "hazardous_liquid" in feature_str:
            self.has_hazardous_liquid = True

        if "fuel_tank" in feature_str:
            self.has_fuel_tank = True

    def __repr__(self) -> str:
        """String representation of the hex for debugging."""
        features = [f"elev:{self.elevation}"]
        if self.has_water:
            features.append(f"water(d:{self.depth})")
        if self.has_woods:
            wood_type = "heavy" if self.is_heavy_woods else "light"
            features.append(f"{wood_type}_woods")
        if self.has_pavement:
            features.append("pavement")
        if self.has_road:
            features.append("road")
        if self.has_building:
            features.append(f"building(e:{self.build_elevation})")
        if self.has_bridge:
            features.append(f"bridge(e:{self.bridge_elevation})")
        if self.has_rough:
            features.append("rough")
        return "Hex " + ", ".join(features)


class GameBoardRepr:
    """Represents the game board with dimensions and hexes."""

    def __init__(self, data: List[str]):
        """Initialize the game board from the parsed data."""
        self.width = 0
        self.height = 0
        self.hexes = []
        self._parse(data)

    def _parse(self, lines: List[str]):
        """Parse the game board data."""
        # Find and parse board dimensions
        for i, line in enumerate(lines):
            if line.startswith("BOARD_NAME"):
                # Next line contains dimensions
                dim_line = lines[i + 1]
                parts = dim_line.split('\t')
                if len(parts) >= 3:
                    self.width = int(parts[2]) # I inverted them both in the logger :facepalm:
                    self.height = int(parts[1]) # I inverted them both in the logger :facepalm:
                break

        # Initialize empty board
        self.hexes = [[None for _ in range(self.height)] for _ in range(self.width)]

        # Find where the row data starts
        row_start = 0
        for i, line in enumerate(lines):
            if line.startswith("ROW_0"):
                row_start = i
                break

        # Process each row
        for y in range(self.height):
            row_line = lines[row_start + y]
            # Split the row by tabs
            cells = row_line.split('\t')

            # Process each cell in the row (skipping the ROW_X cell)
            for x in range(0, self.width):
                feature_str = cells[x + 1]
                self.hexes[x][y] = Hex(feature_str)

    def __getitem__(self, x):
        """Allow bracket indexing for x coordinate."""
        if 0 <= x < self.width:
            return self.hexes[x]
        raise IndexError(f"X coordinate {x} out of bounds (0-{self.width - 1})")

    def get_hex(self, x: int, y: int) -> Optional[Hex]:
        """Get the hex at the specified coordinates."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.hexes[x][y]
        return None

    def __repr__(self) -> str:
        """String representation of the game board."""
        return f"GameBoard({self.width}x{self.height})"

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "hexes": [[
                {
                    "elevation": hex.elevation,
                    "has_water": hex.has_water,
                    "depth": hex.depth,
                    "has_woods": hex.has_woods,
                    "is_heavy_woods": hex.is_heavy_woods,
                    "has_pavement": hex.has_pavement,
                    "has_building": hex.has_building,
                    "has_bridge": hex.has_bridge,
                    "bridge_elevation": hex.bridge_elevation,
                    "has_rough": hex.has_rough,
                    "has_road": hex.has_road,
                    "floor": hex.elevation + hex.build_elevation - hex.depth,
                    "building_elevation": hex.build_elevation,
                }
                for hex in row
            ] for row in self.hexes]
        }

def parse_board_data(file_content: str) -> GameBoardRepr:
    """Parse board data from file content."""
    return GameBoardRepr(file_content)


# Example usage:
if __name__ == "__main__":
    # Replace with actual file reading or input text
    with open("map_data.txt", "r") as f:
        data = f.read()

    game_board = parse_board_data(data)

    # Demo accessing some properties
    print(f"Board dimensions: {game_board.width}x{game_board.height}")

    # Check some random hexes
    for coords in [(5, 5), (10, 10), (15, 15)]:
        x, y = coords
        if x < game_board.width and y < game_board.height:
            hex = game_board[x][y]
            print(f"Hex at {x},{y}: {hex}")
            print(f"  Elevation: {hex.elevation}")
            print(f"  Has water: {hex.has_water} (depth: {hex.depth})")
            print(f"  Has woods: {hex.has_woods} (heavy: {hex.is_heavy_woods})")
            print(f"  Has pavement: {hex.has_pavement}")
            print(f"  Has building: {hex.has_building}")
            print(f"  Has bridge: {hex.has_bridge} (elev: {hex.bridge_elevation})")