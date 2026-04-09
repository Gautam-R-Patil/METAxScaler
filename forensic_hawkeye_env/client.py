# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Forensic Hawkeye Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import ForensicHawkeyeAction, ForensicHawkeyeObservation, ForensicHawkeyeState


class ForensicHawkeyeEnv(
    EnvClient[ForensicHawkeyeAction, ForensicHawkeyeObservation, ForensicHawkeyeState]
):
    """
    Client for the Forensic Hawkeye Environment.

    Maintains a persistent WebSocket connection to the environment server.
    Each client instance has its own dedicated environment session.

    Example:
        >>> with ForensicHawkeyeEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     obs = result.observation
        ...     print(obs.task_name, obs.total_distance_error)
        ...
        ...     action = ForensicHawkeyeAction(
        ...         action_type="RUN_SIMULATION",
        ...         sim_parameters={"Car_A": {"speed": 50.0, "steering": -5.0}}
        ...     )
        ...     result = client.step(action)
        ...     print(result.observation.total_distance_error)
    """

    def _step_payload(self, action: ForensicHawkeyeAction) -> Dict:
        """Convert ForensicHawkeyeAction to JSON payload for step message."""
        payload = {"action_type": action.action_type}

        if action.sim_parameters is not None:
            payload["sim_parameters"] = action.sim_parameters
        if action.friction_coefficient is not None:
            payload["friction_coefficient"] = action.friction_coefficient
        if action.restitution is not None:
            payload["restitution"] = action.restitution
        if action.mass_overrides is not None:
            payload["mass_overrides"] = action.mass_overrides
        if action.impact_offset_y is not None:
            payload["impact_offset_y"] = action.impact_offset_y
        
        if action.liable_party is not None:
            payload["liable_party"] = action.liable_party
        if action.root_cause is not None:
            payload["root_cause"] = action.root_cause

        return payload

    def _parse_result(self, payload: Dict) -> StepResult[ForensicHawkeyeObservation]:
        """Parse server response into StepResult[ForensicHawkeyeObservation]."""
        obs_data = payload.get("observation", {})

        observation = ForensicHawkeyeObservation(
            done=payload.get("done", False),
            reward=payload.get("reward"),
            task_id=obs_data.get("task_id", 1),
            task_name=obs_data.get("task_name", ""),
            target_debris=obs_data.get("target_debris", {}),
            simulated_debris=obs_data.get("simulated_debris", {}),
            distance_errors=obs_data.get("distance_errors", {}),
            total_distance_error=obs_data.get("total_distance_error", 999.0),
            human_testimony=obs_data.get("human_testimony", ""),
            weather_conditions=obs_data.get("weather_conditions", ""),
            vehicle_descriptions=obs_data.get("vehicle_descriptions", {}),
            damage_description=obs_data.get("damage_description", ""),
            active_contradiction_flag=obs_data.get("active_contradiction_flag", False),
            step_number=obs_data.get("step_number", 0),
            max_steps=obs_data.get("max_steps", 15),
            message=obs_data.get("message", ""),
            available_entities=obs_data.get("available_entities", []),
            available_parameters=obs_data.get("available_parameters", []),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> ForensicHawkeyeState:
        """Parse server response into ForensicHawkeyeState object."""
        return ForensicHawkeyeState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", 1),
            best_total_error=payload.get("best_total_error", 999.0),
            verdict_submitted=payload.get("verdict_submitted", False),
            simulation_count=payload.get("simulation_count", 0),
        )
