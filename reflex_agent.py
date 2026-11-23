"""
Reflex Agent Module: Reactive Control & Safety Invariant Monitoring
"""

from typing import Tuple, List, Optional, Callable
from environment import MapEnvironment
from planner import AStarPlanner

class ReflexAgent:
    def __init__(self, env: MapEnvironment, planner: AStarPlanner, heuristic_fn: Callable):
        self.env = env
        self.planner = planner
        self.heuristic_fn = heuristic_fn
        self.battery = env.battery_capacity
        self.position = env.home_base

    def refill_battery(self):
        self.battery = self.env.battery_capacity

    def check_reflex_rules(self, next_tile: Tuple[int, int]) -> Tuple[str, Optional[Tuple[int, int]], Optional[List[Tuple[int, int]]]]:
        """
        Evaluates the 4 core rules before executing a move:
        Returns action: ('PROCEED', None, None) or ('RECHARGE', target_charger, path_to_charger)
        """
        cap = self.env.battery_capacity

        # Rule 4: Opportunistic Recharging (25% <= battery <= 40% and charger reached with cost <= 4)
        if 0.25 * cap <= self.battery <= 0.40 * cap:
            for charger in self.env.all_chargers:
                path, cost, _ = self.planner.plan_path(self.position, charger, self.heuristic_fn)
                if path and cost <= 4.0:
                    return 'RECHARGE', charger, path

        # Rule 1: Emergency Recharge (Battery < 25%)
        if self.battery < 0.25 * cap:
            best_c, path, cost = self.planner.get_cheapest_charger(self.position, self.heuristic_fn)
            return 'RECHARGE', best_c, path

        # Rule 3: Lookahead Safety Invariant Check
        # Check that after moving to next_tile, battery >= cost to reach nearest charger from next_tile
        move_cost = self.env.get_movement_cost(self.position, next_tile)
        battery_after_move = self.battery - move_cost
        
        _, _, min_cost_to_charger_from_next = self.planner.get_cheapest_charger(next_tile, self.heuristic_fn)
        
        if battery_after_move < min_cost_to_charger_from_next:
            # Unsafe move! Divert to charger directly from current position
            best_c, path, _ = self.planner.get_cheapest_charger(self.position, self.heuristic_fn)
            return 'RECHARGE', best_c, path

        # Rule 2: Reef Avoidance is inherently handled by valid tile checks
        return 'PROCEED', None, None
