"""
Environment Module for Autonomous Underwater Survey Vehicle (AUV)
Handles map parsing, directional current dynamics, and graph transitions.
"""

from typing import List, Tuple, Dict, Optional, Set

# Directions: (delta_row, delta_col, name, symbol)
DIRECTIONS = {
    'N': (-1, 0),
    'S': (1, 0),
    'E': (0, 1),
    'W': (0, -1)
}

OPPOSITE_DIR = {
    'N': 'S',
    'S': 'N',
    'E': 'W',
    'W': 'E'
}

CURRENT_SYMBOLS = {
    '^': 'N',
    'v': 'S',
    '>': 'E',
    '<': 'W',
    '.': None
}

class MapEnvironment:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.rows: int = 0
        self.cols: int = 0
        self.battery_capacity: float = 0.0
        self.terrain: List[List[str]] = []
        self.currents: List[List[Optional[str]]] = []
        
        self.home_base: Optional[Tuple[int, int]] = None
        self.charging_docks: List[Tuple[int, int]] = []
        self.survey_sites: Dict[str, Tuple[int, int]] = {}
        
        self._load_map(filepath)

    def _load_map(self, filepath: str):
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        idx = 0
        while idx < len(lines):
            line = lines[idx]
            if line.startswith("ROWS"):
                self.rows = int(line.split()[1])
            elif line.startswith("COLS"):
                self.cols = int(line.split()[1])
            elif line.startswith("BATTERY"):
                self.battery_capacity = float(line.split()[1])
            elif line == "TERRAIN":
                idx += 1
                for _ in range(self.rows):
                    row_str = lines[idx]
                    self.terrain.append(list(row_str))
                    idx += 1
                continue
            elif line == "CURRENTS":
                idx += 1
                for _ in range(self.rows):
                    row_str = lines[idx]
                    curr_row = [CURRENT_SYMBOLS.get(ch, None) for ch in row_str]
                    self.currents.append(curr_row)
                    idx += 1
                continue
            idx += 1

        # Scan special sites
        for r in range(self.rows):
            for c in range(self.cols):
                tile = self.terrain[r][c]
                if tile == 'H':
                    self.home_base = (r, c)
                elif tile == 'C':
                    self.charging_docks.append((r, c))
                elif tile.isdigit():
                    self.survey_sites[tile] = (r, c)

    @property
    def all_chargers(self) -> List[Tuple[int, int]]:
        chargers = []
        if self.home_base:
            chargers.append(self.home_base)
        chargers.extend(self.charging_docks)
        return chargers

    def is_valid_tile(self, r: int, c: int) -> bool:
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return self.terrain[r][c] != '#'
        return False

    def get_base_terrain_cost(self, r: int, c: int) -> float:
        tile = self.terrain[r][c]
        if tile == '~':
            return 6.0
        elif tile == '#':
            return float('inf')
        else:
            # '.', 'H', 'C', '1'-'9' are open water
            return 2.0

    def get_movement_cost(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> float:
        """
        Calculates directional movement cost when entering to_pos.
        """
        r1, c1 = from_pos
        r2, c2 = to_pos

        if not self.is_valid_tile(r2, c2):
            return float('inf')

        base_cost = self.get_base_terrain_cost(r2, c2)
        if base_cost == float('inf'):
            return float('inf')

        # Determine direction of travel
        dr = r2 - r1
        dc = c2 - c1
        travel_dir = None
        for d_name, (mdr, mdc) in DIRECTIONS.items():
            if mdr == dr and mdc == dc:
                travel_dir = d_name
                break

        if not travel_dir:
            return float('inf')

        target_current = self.currents[r2][c2]
        if target_current is None:
            multiplier = 1.0
        elif travel_dir == target_current:
            multiplier = 0.5  # Moving with current
        elif travel_dir == OPPOSITE_DIR[target_current]:
            multiplier = 2.0  # Moving against current
        else:
            multiplier = 1.0  # Moving across current

        return base_cost * multiplier

    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[Tuple[int, int], float]]:
        """Returns valid (neighbor_pos, move_cost) pairs."""
        r, c = pos
        neighbors = []
        for dr, dc in DIRECTIONS.values():
            nr, nc = r + dr, c + dc
            if self.is_valid_tile(nr, nc):
                cost = self.get_movement_cost((r, c), (nr, nc))
                if cost < float('inf'):
                    neighbors.append(((nr, nc), cost))
        return neighbors
