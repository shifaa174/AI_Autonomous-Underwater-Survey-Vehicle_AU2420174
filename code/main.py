if __name__ == "__main__":
    print("=" * 60)
    print("AUV Survey Mission Simulation - AU2420174 (Shifataj Haroon Shaikh)")
    print("=" * 60)
    
    heuristics = {
        "H0: Zero (UCS)": heuristic_zero,
        "H1: Euclidean": heuristic_euclidean,
        "H2: Manhattan": heuristic_manhattan,
        "H3: Current-Aware": heuristic_current_aware
    }

    for name, h_fn in heuristics.items():
        res = run_mission("map_main.txt", h_fn)
        print(f"\nResults for {name}:")
        print(f"  - Total Steps: {res['total_steps']}")
        print(f"  - Total Battery Consumed: {res['total_battery_consumed']:.1f}")
        print(f"  - Recharge Events: {res['recharge_events']}")
