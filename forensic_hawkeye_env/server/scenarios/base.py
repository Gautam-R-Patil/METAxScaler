"""Abstract base class for all forensic scenarios."""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from ..physics import ScenarioPhysicsConfig


class BaseScenario(ABC):
    """
    Abstract base scenario for the Forensic Hawkeye environment.

    Each scenario defines:
    - The physical layout (entities, obstacles, positions)
    - The ground truth (what actually happened)
    - The human testimony (the lie to disprove)
    - The grading criteria
    """

    @property
    @abstractmethod
    def task_id(self) -> int:
        """Unique task identifier."""
        ...

    @property
    @abstractmethod
    def task_name(self) -> str:
        """Human-readable task name."""
        ...

    @property
    @abstractmethod
    def max_steps(self) -> int:
        """Maximum simulation attempts allowed."""
        ...

    @property
    @abstractmethod
    def testimony(self) -> str:
        """The human testimony (contains lies the agent must disprove)."""
        ...

    @property
    @abstractmethod
    def target_debris(self) -> Dict[str, Tuple[float, float]]:
        """Ground truth final positions of all entities after the crash."""
        ...

    @property
    @abstractmethod
    def ground_truth_liable(self) -> str:
        """The entity actually at fault."""
        ...

    @property
    @abstractmethod
    def ground_truth_cause(self) -> str:
        """The actual root cause."""
        ...

    @property
    @abstractmethod
    def error_threshold(self) -> float:
        """Maximum acceptable total distance error for a valid reconstruction."""
        ...

    @abstractmethod
    def get_physics_config(self) -> ScenarioPhysicsConfig:
        """Return the physics configuration for this scenario."""
        ...

    @property
    @abstractmethod
    def available_entities(self) -> List[str]:
        """Entities the agent can configure."""
        ...

    @property
    @abstractmethod
    def available_parameters(self) -> List[str]:
        """Parameters the agent can set per entity."""
        ...

    def check_contradiction(
        self, sim_params: Dict[str, Dict[str, float]]
    ) -> bool:
        """Check if the simulation parameters contradict the testimony."""
        return False  # Subclasses override
