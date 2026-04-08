"""Scenario registry for the Forensic Hawkeye environment."""

from .base import BaseScenario
from .task1_property_strike import PropertyStrikeScenario
from .task2_pedestrian import PedestrianParadoxScenario
from .task3_momentum import MomentumScenario

SCENARIOS = {
    1: PropertyStrikeScenario,
    2: PedestrianParadoxScenario,
    3: MomentumScenario,
}

__all__ = ["BaseScenario", "SCENARIOS"]
