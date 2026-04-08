# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Forensic Hawkeye Environment.

Forensic accident reconstruction environment where an AI agent acts as a
liability auditor, iteratively tuning physics parameters to reproduce crash
debris patterns and disprove false human testimony.
"""

import json
from typing import Dict, List, Optional, Tuple, Literal
from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field, field_validator


class ForensicHawkeyeAction(Action):
    """
    Action for the Forensic Hawkeye environment.

    Two action types:
    - RUN_SIMULATION: Provide physics parameters to simulate the crash
    - SUBMIT_VERDICT: Submit a final liability determination
    """

    action_type: Literal["RUN_SIMULATION", "SUBMIT_VERDICT"] = Field(
        ..., description="Type of action to perform"
    )

    # Used for RUN_SIMULATION
    sim_parameters: Optional[Dict[str, Dict[str, float]]] = Field(
        default=None,
        description='Physics parameters per entity, e.g. {"Car_A": {"speed": 45.0, "steering": 0.0}}',
    )

    # Used for SUBMIT_VERDICT
    liable_party: Optional[str] = Field(
        default=None, description="The entity determined to be at fault"
    )
    root_cause: Optional[str] = Field(
        default=None, description="The root cause of the accident"
    )

    @field_validator("sim_parameters", mode="before")
    @classmethod
    def parse_sim_parameters(cls, v):
        """Auto-parse JSON strings from the web playground into dicts."""
        if isinstance(v, str):
            if not v.strip():
                return None
            return json.loads(v)
        return v


class ForensicHawkeyeObservation(Observation):
    """
    Observation from the Forensic Hawkeye environment.

    Provides target debris positions (ground truth from the crash scene),
    simulated debris positions (from the agent's last simulation), and
    distance errors between them.
    """

    # done: bool and reward: Optional[float] inherited from Observation

    task_id: int = Field(default=1, description="Current task identifier (1-3)")
    task_name: str = Field(default="", description="Human-readable task name")

    target_debris: Dict[str, List[float]] = Field(
        default_factory=dict,
        description="Ground truth final positions from crash scene {entity: [x, y]}",
    )
    simulated_debris: Dict[str, List[float]] = Field(
        default_factory=dict,
        description="Simulated final positions from last run {entity: [x, y]}",
    )
    distance_errors: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-entity distance between target and simulated positions",
    )
    total_distance_error: float = Field(
        default=999.0,
        description="Sum of all entity distance errors",
    )

    human_testimony: str = Field(
        default="", description="The witness testimony to evaluate"
    )
    active_contradiction_flag: bool = Field(
        default=False,
        description="True if physics results contradict the human testimony",
    )

    step_number: int = Field(default=0, description="Current step in this episode")
    max_steps: int = Field(default=15, description="Maximum steps allowed")
    message: str = Field(default="", description="Feedback message from the environment")

    available_entities: List[str] = Field(
        default_factory=list,
        description="Entities that can be configured in sim_parameters",
    )
    available_parameters: List[str] = Field(
        default_factory=list,
        description="Parameters that can be set per entity (e.g., speed, steering)",
    )


class ForensicHawkeyeState(State):
    """
    Internal state tracking for the Forensic Hawkeye environment.
    """

    # episode_id and step_count inherited from State
    task_id: int = 1
    best_total_error: float = 999.0
    verdict_submitted: bool = False
    simulation_count: int = 0
