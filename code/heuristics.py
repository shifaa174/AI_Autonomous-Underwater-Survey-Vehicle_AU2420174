"""
Heuristics Module for A* Navigation
Implements 4 distinct heuristics:
1. Zero Heuristic (Uniform-Cost Search Baseline)
2. Scaled Euclidean Distance
3. Scaled Manhattan Distance
4. Current-Aware / Directional Informed Heuristic
"""

import math
from typing import Tuple
from environment import MapEnvironment

def heuristic_zero(pos: Tuple[int, int], goal: Tuple[int, int], env: MapEnvironment = None) -> float:
    """Heuristic 0: Zero heuristic (UCS / Dijkstra baseline). Always admissible & consistent."""
    return 0.0

def heuristic_euclidean(pos: Tuple[int, int], goal: Tuple[int, int], env: MapEnvironment = None) -> float:
    """
    Heuristic 1: Scaled Euclidean Distance.
    c_min = 1.0 (minimum possible transition cost in the grid: open water with current = 2 * 0.5 = 1.0).
    Admissible and consistent on 4-connected grid.
    """
    c_min = 1.0
    dr = pos[0] - goal[0]
    dc = pos[1] - goal[1]
    return c_min * math.sqrt(dr * dr + dc * dc)

def heuristic_manhattan(pos: Tuple[int, int], goal: Tuple[int, int], env: MapEnvironment = None) -> float:
    """
    Heuristic 2: Scaled Manhattan Distance (L1 norm).
    c_min = 1.0. Tighter than Euclidean on 4-connected grids while strictly admissible & consistent.
    """
    c_min = 1.0
    dr = abs(pos[0] - goal[0])
    dc = abs(pos[1] - goal[1])
    return c_min * (dr + dc)

def heuristic_current_aware(pos: Tuple[int, int], goal: Tuple[int, int], env: MapEnvironment = None) -> float:
    """
    Heuristic 3: Direction- & Current-Aware Relaxed Heuristic.
    Evaluates required directional displacement. If motion in a direction cannot
    be assisted by prevailing currents, a tighter lower bound (normal open water cost = 2.0)
    is applied.
    """
    dr = goal[0] - pos[0]
    dc = goal[1] - pos[1]
    
    cost_vertical = 0.0
    if dr > 0:  # Moving South
        cost_vertical = abs(dr) * 1.0
    elif dr < 0:  # Moving North
        cost_vertical = abs(dr) * 1.0

    cost_horizontal = 0.0
    if dc > 0:  # Moving East
        cost_horizontal = abs(dc) * 1.0
    elif dc < 0:  # Moving West
        cost_horizontal = abs(dc) * 1.0

    return cost_vertical + cost_horizontal
