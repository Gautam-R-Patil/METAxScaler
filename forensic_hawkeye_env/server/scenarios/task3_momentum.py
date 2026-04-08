"""
Task 3: The Multi-Vehicle Origin — Conservation of Momentum (Hard)

Scenario: 3-car intersection pileup. All cars scattered. Nobody admits to
running the red light. Due to conservation of momentum, there is only one
valid set of entry speeds that produces the specific scatter pattern.

Variables: speed + timing_offset for Cars A, B, C (6 variables total)
Ground truth: Car B was going 85 mph (way over the limit)
"""

from typing import Dict, List, Tuple

from .base import BaseScenario
from ..physics import ScenarioPhysicsConfig, EntityConfig


class MomentumScenario(BaseScenario):
    """
    Hard task — 6 variables.

    The agent must:
    1. Optimize speed and timing_offset for all 3 cars
    2. Use conservation of momentum to find the unique solution
    3. Achieve combined distance_error < 1.5m
    4. Identify Car B as the speeder (85 mph)
    """

    _TARGET_DEBRIS = {
        "Car_A": (141.16, 100.0),
        "Car_B": (106.45, 172.07),
        "Car_C": (173.71, 100.0),
    }

    @property
    def task_id(self) -> int:
        return 3

    @property
    def task_name(self) -> str:
        return "The Multi-Vehicle Origin"

    @property
    def max_steps(self) -> int:
        return 30

    @property
    def testimony(self) -> str:
        return (
            "Driver of Car A states: 'I was going about 40 mph through the intersection "
            "on a green light. Car B came from my left and Car C from straight ahead. "
            "They both ran their red lights.'\n\n"
            "Driver of Car B states: 'I was driving at about 35 mph. My light was green. "
            "Car A came speeding through the red light from my right, and Car C was already "
            "in the intersection illegally.'\n\n"
            "Driver of Car C states: 'I was traveling about 30 mph. I had the right of way. "
            "Both Car A and Car B ran their lights. Car B seemed to be going especially fast.'"
        )

    @property
    def target_debris(self) -> Dict[str, Tuple[float, float]]:
        return self._TARGET_DEBRIS

    @property
    def ground_truth_liable(self) -> str:
        return "Car_B"

    @property
    def ground_truth_cause(self) -> str:
        return "Speeding"

    @property
    def error_threshold(self) -> float:
        return 1.5  # meters combined

    @property
    def available_entities(self) -> List[str]:
        return ["Car_A", "Car_B", "Car_C"]

    @property
    def available_parameters(self) -> List[str]:
        return ["speed", "timing_offset"]

    def get_physics_config(self) -> ScenarioPhysicsConfig:
        return ScenarioPhysicsConfig(
            entities=[
                EntityConfig(
                    name="Car_A",
                    mass=1500.0,
                    width=2.0,
                    height=4.5,
                    start_x=50.0,
                    start_y=100.0,
                    heading_deg=0.0,  # Driving rightward toward intersection
                    entity_type="car",
                ),
                EntityConfig(
                    name="Car_B",
                    mass=1800.0,
                    width=2.2,
                    height=5.0,
                    start_x=100.0,
                    start_y=50.0,
                    heading_deg=90.0,  # Driving upward toward intersection
                    entity_type="car",
                ),
                EntityConfig(
                    name="Car_C",
                    mass=1300.0,
                    width=1.8,
                    height=4.2,
                    start_x=150.0,
                    start_y=100.0,
                    heading_deg=180.0,  # Driving leftward toward intersection
                    entity_type="car",
                ),
            ],
            static_obstacles=[],
            world_bounds=(200.0, 200.0),
        )

    def check_contradiction(
        self, sim_params: Dict[str, Dict[str, float]]
    ) -> bool:
        """
        Contradiction when Car_B's speed is revealed to be much higher
        than any testimony claims.
        """
        car_b = sim_params.get("Car_B", {})
        speed = car_b.get("speed", 0)
        # Car B claimed 35 mph. If physics shows > 70, strong contradiction.
        return speed > 70.0
