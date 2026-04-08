"""
Generate ground-truth debris positions for all 3 scenarios.

Run this once to calibrate the target_debris values in each scenario file.
The target positions MUST come from the same physics engine to ensure
deterministic grading.
"""

import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forensic_hawkeye_env.server.physics import PhysicsWorld, compute_distance_errors
from forensic_hawkeye_env.server.scenarios.task1_property_strike import PropertyStrikeScenario
from forensic_hawkeye_env.server.scenarios.task2_pedestrian import PedestrianParadoxScenario
from forensic_hawkeye_env.server.scenarios.task3_momentum import MomentumScenario


def main():
    physics = PhysicsWorld()

    # ══════════════════════════════════════════════════════════════
    # Task 1: Property Strike
    # Ground truth: Car_A @ speed=54, steering=-8
    # ══════════════════════════════════════════════════════════════
    t1 = PropertyStrikeScenario()
    r1 = physics.simulate(
        t1.get_physics_config(),
        {"Car_A": {"speed": 54.0, "steering": -8.0}},
    )
    print("=" * 60)
    print("TASK 1: The Public Property Strike")
    print(f"  Ground truth params: Car_A speed=54, steering=-8")
    print(f"  Final positions: {r1.final_positions}")
    print(f"  Collision: {r1.was_collision}")

    # Verify that 30 mph (the lie) produces different results
    r1_lie = physics.simulate(
        t1.get_physics_config(),
        {"Car_A": {"speed": 30.0, "steering": -8.0}},
    )
    errors, total = compute_distance_errors(r1.final_positions, r1_lie.final_positions)
    print(f"  30mph lie positions: {r1_lie.final_positions}")
    print(f"  Error vs truth: {total:.2f}m")

    # ══════════════════════════════════════════════════════════════
    # Task 2: Pedestrian Paradox
    # Ground truth: Car_A @ speed=52, steering=-15; Pedestrian velocity=1.2
    # ══════════════════════════════════════════════════════════════
    t2 = PedestrianParadoxScenario()
    r2 = physics.simulate(
        t2.get_physics_config(),
        {
            "Car_A": {"speed": 52.0, "steering": -15.0},
            "Pedestrian": {"velocity": 1.2},
        },
    )
    print("\n" + "=" * 60)
    print("TASK 2: The Temporal Pedestrian Paradox")
    print(f"  Ground truth params: Car_A speed=52 steering=-15, Ped velocity=1.2")
    print(f"  Final positions: {r2.final_positions}")
    print(f"  Collision: {r2.was_collision}")

    # ══════════════════════════════════════════════════════════════
    # Task 3: Multi-Vehicle Momentum
    # Ground truth: A=42mph/offset=0, B=85mph/offset=0.2, C=28mph/offset=0
    # ══════════════════════════════════════════════════════════════
    t3 = MomentumScenario()
    r3 = physics.simulate(
        t3.get_physics_config(),
        {
            "Car_A": {"speed": 42.0, "timing_offset": 0.0},
            "Car_B": {"speed": 85.0, "timing_offset": 0.2},
            "Car_C": {"speed": 28.0, "timing_offset": 0.0},
        },
    )
    print("\n" + "=" * 60)
    print("TASK 3: The Multi-Vehicle Origin")
    print(f"  Ground truth params: A=42/0, B=85/0.2, C=28/0")
    print(f"  Final positions: {r3.final_positions}")
    print(f"  Collision: {r3.was_collision}")

    # ── Determinism check ──
    print("\n" + "=" * 60)
    print("DETERMINISM CHECK (10 runs of Task 1):")
    positions = []
    for i in range(10):
        r = physics.simulate(
            t1.get_physics_config(),
            {"Car_A": {"speed": 54.0, "steering": -8.0}},
        )
        positions.append(r.final_positions["Car_A"])
    
    all_same = all(p == positions[0] for p in positions)
    print(f"  All 10 runs identical: {all_same}")
    print(f"  Position: {positions[0]}")

    print("\n" + "=" * 60)
    print("\nCOPY THESE VALUES INTO THE SCENARIO FILES:")
    print(f"\n  Task 1 _TARGET_DEBRIS = {r1.final_positions}")
    print(f"  Task 2 _TARGET_DEBRIS = {r2.final_positions}")
    print(f"  Task 3 _TARGET_DEBRIS = {r3.final_positions}")


if __name__ == "__main__":
    main()
