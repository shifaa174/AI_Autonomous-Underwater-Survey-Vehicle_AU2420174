"""
Planner Module: A* Path Planning and High-Level Survey Tour Sequencing
"""

import heapq
import time
import itertools
from typing import Tuple, List, Dict, Callable, Optional
from environment import MapEnvironment

class AStarPlanner:
    def __init__(self, env: MapEnvironment):
        self.env = env

    def plan_path(self, start: Tuple[int, int], goal: Tuple[int, int], 
                  heuristic_fn: Callable) -> Tuple[Optional[List[Tuple[int, int]]], float, Dict]:
        """
        Runs A* search from start to goal using the supplied heuristic function.
        Returns: (path, total_cost, search_stats)
        """
        start_time = time.perf_counter()
        
        if start == goal:
            return [start], 0.0, {'nodes_expanded': 0, 'nodes_generated': 1, 'time_ms': 0.0}

        # Priority queue stores: (f_score, g_score, current_node)
        open_set = []
        h_start = heuristic_fn(start, goal, self.env)
        heapq.heappush(open_set, (h_start, 0.0, start))

        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start: 0.0}
        closed_set = set()

        nodes_expanded = 0
        nodes_generated = 1

        while open_set:
            f, g, current = heapq.heappop(open_set)

            if current in closed_set:
                continue
            closed_set.add(current)
            nodes_expanded += 1

            if current == goal:
                # Reconstruct path
                path = []
                curr = current
                while curr in came_from:
                    path.append(curr)
                    curr = came_from[curr]
                path.append(start)
                path.reverse()
                
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                stats = {
                    'nodes_expanded': nodes_expanded,
                    'nodes_generated': nodes_generated,
                    'time_ms': elapsed_ms
                }
                return path, g, stats

            for neighbor, move_cost in self.env.get_neighbors(current):
                tentative_g = g + move_cost
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    h_val = heuristic_fn(neighbor, goal, self.env)
                    f_val = tentative_g + h_val
                    heapq.heappush(open_set, (f_val, tentative_g, neighbor))
                    nodes_generated += 1

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return None, float('inf'), {'nodes_expanded': nodes_expanded, 'nodes_generated': nodes_generated, 'time_ms': elapsed_ms}

    def get_cheapest_charger(self, from_pos: Tuple[int, int], heuristic_fn: Callable) -> Tuple[Tuple[int, int], List[Tuple[int, int]], float]:
        """Finds the charger (H or C) that is cheapest to reach from from_pos."""
        best_charger = None
        best_path = None
        min_cost = float('inf')

        for charger in self.env.all_chargers:
            path, cost, _ = self.plan_path(from_pos, charger, heuristic_fn)
            if path and cost < min_cost:
                min_cost = cost
                best_charger = charger
                best_path = path

        return best_charger, best_path, min_cost

    def find_optimal_survey_order(self, current_pos: Tuple[int, int], 
                                  remaining_sites: List[str], 
                                  heuristic_fn: Callable) -> List[str]:
        """
        Solves multi-goal tour ordering using nearest-neighbor / TSP heuristic.
        """
        if not remaining_sites:
            return []
        
        # If small number of sites (<= 6), compute exact optimal permutation
        if len(remaining_sites) <= 6:
            best_order = None
            min_est_cost = float('inf')
            site_coords = [self.env.survey_sites[s] for s in remaining_sites]
            
            for perm in itertools.permutations(remaining_sites):
                est_cost = 0.0
                curr = current_pos
                for s in perm:
                    target = self.env.survey_sites[s]
                    est_cost += heuristic_fn(curr, target, self.env)
                    curr = target
                # Return to home
                est_cost += heuristic_fn(curr, self.env.home_base, self.env)
                
                if est_cost < min_est_cost:
                    min_est_cost = est_cost
                    best_order = list(perm)
            return best_order
        else:
            # Greedy nearest neighbor for larger counts
            unvisited = remaining_sites.copy()
            curr = current_pos
            order = []
            while unvisited:
                next_site = min(unvisited, key=lambda s: heuristic_fn(curr, self.env.survey_sites[s], self.env))
                order.append(next_site)
                curr = self.env.survey_sites[next_site]
                unvisited.remove(next_site)
            return order
