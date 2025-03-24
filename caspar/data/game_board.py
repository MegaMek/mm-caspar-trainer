import re
from typing import Dict, List, Tuple, Optional, Any


class Hex:
    """Represents a single hex on the game board with all its terrain features."""

    def __init__(self, feature_str: str):
        """Initialize a hex from a feature string."""
        self.elevation = 0
        self.has_water = False
        self.depth = 0
        self.has_woods = False
        self.is_heavy_woods = False
        self.has_pavement = False
        self.has_building = False
        self.has_bridge = False
        self.bridge_elevation = 0
        self.has_rough = False
        self.has_road = False

        # Extract level information
        level_match = re.match(r"Level: (-?\d+)", feature_str)
        if level_match:
            self.elevation = int(level_match.group(1))

        # Check for water
        if "Water" in feature_str:
            self.has_water = True
            depth_match = re.search(r"Water, depth: (\d+)", feature_str)
            if depth_match:
                self.depth = int(depth_match.group(1))

        # Check for woods
        if "Woods" in feature_str:
            self.has_woods = True
            self.is_heavy_woods = "Heavy Woods" in feature_str

        # Check for roads and pavements
        if "pavement" in feature_str:
            self.has_pavement = True
        if "Road" in feature_str:
            self.has_road = True
            self.has_pavement = True

        # Check for buildings
        if "building" in feature_str:
            self.has_building = True

        # Check for bridges
        if "bridge" in feature_str:
            self.has_bridge = True
            bridge_elev_match = re.search(r"bridge_elev\((\d+)", feature_str)
            if bridge_elev_match:
                self.bridge_elevation = int(bridge_elev_match.group(1))

        # Check for rough terrain
        if "Rough" in feature_str:
            self.has_rough = True

    def __repr__(self) -> str:
        """String representation of the hex for debugging."""
        features = []
        features.append(f"elev:{self.elevation}")
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
            features.append("building")
        if self.has_bridge:
            features.append(f"bridge(e:{self.bridge_elevation})")
        if self.has_rough:
            features.append("rough")
        return ", ".join(features)


class GameBoardRepr:
    """Represents the game board with dimensions and hexes."""

    def __init__(self, data: str):
        """Initialize the game board from the parsed data."""
        self.width = 0
        self.height = 0
        self.hexes = []
        self._parse(data)

    def _parse(self, data: str):
        """Parse the game board data."""
        lines = data.strip().split('\n')

        # Find and parse board dimensions
        for i, line in enumerate(lines):
            if line.startswith("BOARD_NAME"):
                # Next line contains dimensions
                dim_line = lines[i + 1]
                parts = dim_line.split('\t')
                if len(parts) >= 3:
                    self.width = int(parts[1])
                    self.height = int(parts[2])
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
            if row_start + y >= len(lines):
                break

            row_line = lines[row_start + y]
            if not row_line.startswith(f"ROW_{y}"):
                continue

            # Split the row by tabs
            cells = row_line.split('\t')

            # Process each cell in the row (skipping the ROW_X cell)
            for x in range(min(self.width, len(cells) - 1)):
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