"""
Simulation and Driver Module for AUV Mission
"""

from typing import List, Dict, Callable
from environment import MapEnvironment
from heuristics import (heuristic_zero, heuristic_euclidean, 
                        heuristic_manhattan, heuristic_current_aware)
from planner import AStarPlanner
from reflex_agent import ReflexAgent

def run_mission(map_path: str, heuristic_fn: Callable) -> Dict:
    env = MapEnvironment(map_path)
    planner = AStarPlanner(env)
    agent = ReflexAgent(env, planner, heuristic_fn)

    unvisited_sites = sorted(list(env.survey_sites.keys()))
    visited_sites = []
    
    total_battery_consumed = 0.0
    total_steps = 0
    recharge_events = 0
    path_history = [agent.position]
    battery_history = [agent.battery]
    event_log = []

    # High-level tour sequencing
    tour_order = planner.find_optimal_survey_order(agent.position, unvisited_sites, heuristic_fn)
    event_log.append(f"Initial planned tour order: {tour_order}")

    current_target_idx = 0

    while unvisited_sites or agent.position != env.home_base:
        # Determine next destination
        if unvisited_sites:
            target_site_id = tour_order[current_target_idx]
            goal_pos = env.survey_sites[target_site_id]
            goal_desc = f"Survey Site {target_site_id}"
        else:
            goal_pos = env.home_base
            goal_desc = "Home Base H"

        # Plan route to goal
        leg_path, leg_cost, _ = planner.plan_path(agent.position, goal_pos, heuristic_fn)
        if not leg_path or len(leg_path) < 2:
            if agent.position == goal_pos:
                if unvisited_sites and target_site_id in unvisited_sites:
                    unvisited_sites.remove(target_site_id)
                    visited_sites.append(target_site_id)
                    current_target_idx += 1
                continue
            else:
                event_log.append(f"Failed to find path to {goal_desc}")
                break

        # Execute leg step-by-step with reflex monitoring
        step_idx = 1
        diverted = False
        while step_idx < len(leg_path):
            next_tile = leg_path[step_idx]

            # Evaluate reflex rules
            action, charger, recharge_path = agent.check_reflex_rules(next_tile)

            if action == 'RECHARGE':
                diverted = True
                recharge_events += 1
                event_log.append(f"Step {total_steps}: Reflex triggered recharge to {charger} (Battery: {agent.battery:.1f})")
                
                # Navigate to charger
                for c_step in recharge_path[1:]:
                    m_cost = env.get_movement_cost(agent.position, c_step)
                    agent.battery -= m_cost
                    total_battery_consumed += m_cost
                    agent.position = c_step
                    total_steps += 1
                    path_history.append(agent.position)
                    battery_history.append(agent.battery)

                # Recharge to full
                agent.refill_battery()
                battery_history.append(agent.battery)
                event_log.append(f"Step {total_steps}: Recharged full at {agent.position}")

                # Re-plan tour for remaining unvisited sites
                tour_order = planner.find_optimal_survey_order(agent.position, unvisited_sites, heuristic_fn)
                current_target_idx = 0
                break

            # Execute normal move
            move_cost = env.get_movement_cost(agent.position, next_tile)
            agent.battery -= move_cost
            total_battery_consumed += move_cost
            agent.position = next_tile
            total_steps += 1
            path_history.append(agent.position)
            battery_history.append(agent.battery)

            if unvisited_sites and agent.position == goal_pos:
                unvisited_sites.remove(target_site_id)
                visited_sites.append(target_site_id)
                current_target_idx += 1
                event_log.append(f"Step {total_steps}: Visited Site {target_site_id} (Battery: {agent.battery:.1f})")

            step_idx += 1

    return {
        'total_steps': total_steps,
        'total_battery_consumed': total_battery_consumed,
        'recharge_events': recharge_events,
        'path_history': path_history,
        'battery_history': battery_history,
        'event_log': event_log
    }
