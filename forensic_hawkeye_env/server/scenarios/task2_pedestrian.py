"""
Task 2: The Temporal Pedestrian Paradox (Medium)

Scenario: Car A swerves into a pole, claiming a pedestrian ran into the road.
The pedestrian claims they were walking slowly. The agent must:
1. Reconstruct the pole crash
2. Run a counterfactual (steering=0) to check if the straight-line path
   would have intersected the pedestrian's walking path
3. Prove the slow-walking pedestrian would NOT have been hit

Variables: Car_A.speed, Car_A.steering, Pedestrian.velocity (3 variables)
"""

from typing import Dict, List, Tuple

from .base import BaseScenario
from ..physics import ScenarioPhysicsConfig, EntityConfig


class PedestrianParadoxScenario(BaseScenario):
    """
    Medium task — 3 variables.

    The agent must:
    1. Match Car_A's crash into the pole (tune speed + steering)
    2. Run a counterfactual with steering=0 to see where Car_A would have gone
    3. Use Pedestrian.velocity to determine if a slow-walking pedestrian
       would have been intersected
    4. Conclude the driver overreacted — the pedestrian was never in danger
    """

    _TARGET_DEBRIS = {
        "Car_A": (130.8, 63.79),
        "Pedestrian": (65.0, 74.73),
    }

    @property
    def task_id(self) -> int:
        return 2

    @property
    def task_name(self) -> str:
        return "The Temporal Pedestrian Paradox"

    @property
    def max_steps(self) -> int:
        return 25

    @property
    def testimony(self) -> str:
        return (
            "Driver of Car A states: 'I was driving my mid-size coupe at about 35 mph. "
            "It was a bright, sunny day with bone-dry roads. A pedestrian "
            "suddenly sprinted into the road from my right side. I swerved hard to the left "
            "and hit a utility pole. The pole barely scratched my bumper, just a minor dent.'\n\n"
            "Pedestrian states: 'I was walking slowly across the crosswalk. "
            "The car came out of nowhere going very fast. The driver was "
            "definitely speeding.'"
        )

    @property
    def target_debris(self) -> Dict[str, Tuple[float, float]]:
        return self._TARGET_DEBRIS

    @property
    def ground_truth_liable(self) -> str:
        return "Car_A"

    @property
    def ground_truth_cause(self) -> str:
        return "Overreaction"

    @property
    def ground_truth_friction(self) -> float:
        return 0.8  # Dry day

    @property
    def ground_truth_restitution(self) -> float:
        return 0.5  # Minor dent

    @property
    def ground_truth_masses(self) -> Dict[str, float]:
        return {"Car_A": 1400.0, "Pedestrian": 75.0}

    @property
    def error_threshold(self) -> float:
        return 2.0  # meters total across both entities

    @property
    def available_entities(self) -> List[str]:
        return ["Car_A", "Pedestrian"]

    @property
    def available_parameters(self) -> List[str]:
        return ["speed", "steering", "velocity"]

    def get_physics_config(self) -> ScenarioPhysicsConfig:
        return ScenarioPhysicsConfig(
            entities=[
                EntityConfig(
                    name="Car_A",
                    mass=1400.0,
                    width=2.0,
                    height=4.5,
                    start_x=20.0,
                    start_y=75.0,
                    heading_deg=0.0,  # Driving rightward
                    entity_type="car",
                ),
                EntityConfig(
                    name="Pedestrian",
                    mass=75.0,
                    width=0.5,
                    height=0.5,
                    start_x=65.0,
                    start_y=55.0,
                    heading_deg=90.0,  # Walking upward across road
                    entity_type="pedestrian",
                ),
            ],
            static_obstacles=[
                {"name": "Utility_Pole", "x": 95.0, "y": 60.0, "w": 0.5, "h": 0.5},
            ],
            world_bounds=(150.0, 150.0),
        )

    def check_contradiction(
        self, sim_params: Dict[str, Dict[str, float]]
    ) -> bool:
        """
        Contradiction if:
        - Car_A speed is high (> 40 mph = speeding, supports pedestrian)
        - Pedestrian velocity is slow (< 3 m/s = walking, contradicts driver)
        """
        car_a = sim_params.get("Car_A", {})
        ped = sim_params.get("Pedestrian", {})

        speed = car_a.get("speed", 0)
        ped_vel = ped.get("velocity", 0)

        # Driver claimed ~35mph and pedestrian sprinting.
        # If physics shows speed > 45 and pedestrian < 3 m/s (walking), it's a contradiction
        return speed > 45.0 and ped_vel < 3.0
