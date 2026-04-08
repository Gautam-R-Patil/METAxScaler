"""
Task 1: The Public Property Strike (Easy)

Scenario: Car A hits a bus stop. Driver claims they were driving the 30 mph
speed limit. Physics proves the true speed was 54 mph.

Variables: Car_A.speed, Car_A.steering (2 variables)
"""

from typing import Dict, List, Tuple

from .base import BaseScenario
from ..physics import ScenarioPhysicsConfig, EntityConfig


class PropertyStrikeScenario(BaseScenario):
    """
    Easy task — 2 variables.

    The agent must:
    1. Iteratively run simulations with different speed/steering values
    2. Match the final debris position of Car_A to within 1.0m
    3. Determine that Car_A was speeding (true speed = 54 mph, not 30 mph)
    """

    # ── Pre-computed ground truth ──
    # These are the final positions when Car_A starts at (20, 100) going
    # at 54 mph with steering = -8 degrees and hits the bus stop at (80, 85).
    _TARGET_DEBRIS = {
        "Car_A": (137.64, 81.95),
    }

    @property
    def task_id(self) -> int:
        return 1

    @property
    def task_name(self) -> str:
        return "The Public Property Strike"

    @property
    def max_steps(self) -> int:
        return 15

    @property
    def testimony(self) -> str:
        return (
            "Driver of Car A states: 'I was driving at exactly 30 mph, the posted "
            "speed limit, when a stray dog ran across the road. I swerved slightly "
            "to avoid it and lost control, sliding into the bus stop. I was not "
            "speeding at any point. The road was a bit wet which made me slide further "
            "than expected.'"
        )

    @property
    def target_debris(self) -> Dict[str, Tuple[float, float]]:
        return self._TARGET_DEBRIS

    @property
    def ground_truth_liable(self) -> str:
        return "Car_A"

    @property
    def ground_truth_cause(self) -> str:
        return "Speeding"

    @property
    def error_threshold(self) -> float:
        return 1.0  # meters

    @property
    def available_entities(self) -> List[str]:
        return ["Car_A"]

    @property
    def available_parameters(self) -> List[str]:
        return ["speed", "steering"]

    def get_physics_config(self) -> ScenarioPhysicsConfig:
        return ScenarioPhysicsConfig(
            entities=[
                EntityConfig(
                    name="Car_A",
                    mass=1500.0,
                    width=2.0,
                    height=4.5,
                    start_x=20.0,
                    start_y=100.0,
                    heading_deg=0.0,  # Driving rightward
                    entity_type="car",
                ),
            ],
            static_obstacles=[
                {"name": "Bus_Stop", "x": 80.0, "y": 85.0, "w": 3.0, "h": 2.0},
            ],
            world_bounds=(150.0, 150.0),
        )

    def check_contradiction(
        self, sim_params: Dict[str, Dict[str, float]]
    ) -> bool:
        """Contradiction exists when the speed needed is significantly above 30 mph."""
        car_a = sim_params.get("Car_A", {})
        speed = car_a.get("speed", 0)
        # If agent discovers the speed must be > 40 mph, testimony is contradicted
        return speed > 40.0
