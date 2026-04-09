# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Forensic Hawkeye Environment Implementation.

A forensic accident reconstruction environment where an AI agent iteratively
tunes physics simulation parameters to reconstruct crash evidence and
disprove false human testimony.

Uses headless pymunk physics with a 2.5D weight transfer trick for realism.
"""

import math
from uuid import uuid4
from typing import Dict, Optional, Tuple, List

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import (
        ForensicHawkeyeAction,
        ForensicHawkeyeObservation,
        ForensicHawkeyeState,
    )
except ImportError:
    from models import (
        ForensicHawkeyeAction,
        ForensicHawkeyeObservation,
        ForensicHawkeyeState,
    )

from .physics import PhysicsWorld, compute_distance_errors
from .grader import compute_step_reward, compute_final_score
from .scenarios import SCENARIOS
from .scenarios.base import BaseScenario


def _extract_hints(testimony: str) -> dict:
    """Extract structured hints from plain-English testimony for observation fields."""
    text = testimony.lower()

    # Weather / friction hints
    weather = "unknown"
    if any(w in text for w in ["rain", "downpour", "wet", "slick", "slippery"]):
        weather = "wet/rainy — expect low friction (0.2–0.4)"
    elif any(w in text for w in ["ice", "icy", "freezing", "frozen", "snow"]):
        weather = "icy — expect very low friction (0.1–0.2)"
    elif any(w in text for w in ["dry", "sunny", "clear", "bright"]):
        weather = "dry/clear — expect high friction (0.7–0.9)"

    # Vehicle type / mass hints
    vehicles = {}
    for car_id in ["car_a", "car_b", "car_c"]:
        display = car_id.replace("_", " ").title()
        if any(w in text for w in ["truck", "delivery truck", "heavy truck"]) and car_id == "car_b":
            vehicles[display] = "heavy truck (~6000-10000 kg)"
        elif any(w in text for w in ["compact", "small car"]) and car_id in text:
            vehicles[display] = "compact car (~1200-1400 kg)"
        elif any(w in text for w in ["sedan", "coupe", "mid-size"]):
            vehicles[display] = "sedan/coupe (~1400-1600 kg)"

    # Damage / restitution hints
    damage = "unknown"
    if any(w in text for w in ["crushed", "totaled", "crumpled", "destroyed", "completely"]):
        damage = "severe crush damage — expect low restitution (0.1–0.2)"
    elif any(w in text for w in ["minor dent", "scratch", "barely"]):
        damage = "minor damage — expect higher restitution (0.4–0.6)"

    return {"weather": weather, "vehicles": vehicles, "damage": damage}


class ForensicHawkeyeEnvironment(Environment):
    """
    Forensic Accident Reconstruction Environment.

    The agent acts as a forensic liability auditor. Given crash scene evidence
    (final debris positions) and witness testimony, it must:
    1. Iteratively run physics simulations with different parameters
    2. Converge on parameter values that reproduce the debris pattern
    3. Identify contradictions between physics and testimony
    4. Submit a verdict identifying the liable party and root cause

    Three tasks of increasing difficulty:
    - Task 1 (Easy): 2 variables — single car, property damage
    - Task 2 (Medium): 3 variables — car + pedestrian, counterfactual analysis
    - Task 3 (Hard): 6 variables — 3-car pileup, conservation of momentum
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task_id: int = 1):
        """Initialize the environment with a specific task."""
        self._task_id = task_id
        self._physics = PhysicsWorld()
        self._scenario: Optional[BaseScenario] = None
        self._state = ForensicHawkeyeState(
            episode_id=str(uuid4()), step_count=0, task_id=task_id
        )
        self._previous_total_error = 999.0
        self._best_total_error = 999.0
        self._simulated_debris: Dict[str, Tuple[float, float]] = {}
        self._distance_errors: Dict[str, float] = {}
        self._total_error = 999.0
        self._contradiction_flag = False
        self._last_sim_params: Dict[str, Dict[str, float]] = {}
        self._rewards: List[float] = []

    def reset(self, task_id: int = None, **kwargs) -> ForensicHawkeyeObservation:
        """
        Reset the environment for a new episode.

        Args:
            task_id: Which task to run (1, 2, or 3). Defaults to self._task_id.

        Returns:
            Initial observation with target debris and testimony.
        """
        if task_id is not None:
            self._task_id = task_id

        # Clamp task_id to valid range
        self._task_id = max(1, min(3, self._task_id))

        # Initialize scenario
        scenario_cls = SCENARIOS.get(self._task_id)
        if scenario_cls is None:
            scenario_cls = SCENARIOS[1]
        self._scenario = scenario_cls()

        # Reset state
        self._state = ForensicHawkeyeState(
            episode_id=str(uuid4()),
            step_count=0,
            task_id=self._task_id,
        )
        self._previous_total_error = 999.0
        self._best_total_error = 999.0
        self._simulated_debris = {}
        self._distance_errors = {}
        self._total_error = 999.0
        self._contradiction_flag = False
        self._last_sim_params = {}
        self._rewards = []

        # Build initial observation
        target = self._scenario.target_debris
        target_as_lists = {k: list(v) for k, v in target.items()}

        # Extract structured hints from testimony
        hints = _extract_hints(self._scenario.testimony)

        return ForensicHawkeyeObservation(
            done=False,
            reward=0.0,
            task_id=self._task_id,
            task_name=self._scenario.task_name,
            target_debris=target_as_lists,
            simulated_debris={},
            distance_errors={},
            total_distance_error=999.0,
            human_testimony=self._scenario.testimony,
            weather_conditions=hints["weather"],
            vehicle_descriptions=hints["vehicles"],
            damage_description=hints["damage"],
            active_contradiction_flag=False,
            step_number=0,
            max_steps=self._scenario.max_steps,
            message=(
                f"Task {self._task_id}: {self._scenario.task_name}. "
                f"Analyze the crash scene evidence and testimony. "
                f"Use RUN_SIMULATION to test parameters, then SUBMIT_VERDICT "
                f"when you've identified the liable party. "
                f"You have {self._scenario.max_steps} steps."
            ),
            available_entities=self._scenario.available_entities,
            available_parameters=self._scenario.available_parameters,
        )

    def step(self, action: ForensicHawkeyeAction) -> ForensicHawkeyeObservation:  # type: ignore[override]
        """
        Execute a step in the environment.

        Args:
            action: Either RUN_SIMULATION or SUBMIT_VERDICT

        Returns:
            Updated observation with simulation results or verdict response.
        """
        try:
            self._state.step_count += 1
            scenario = self._scenario

            if scenario is None:
                return self._error_obs("Environment not reset. Call reset() first.")

            # Check max steps
            if self._state.step_count > scenario.max_steps:
                final_score = compute_final_score(
                    self._best_total_error,
                    scenario.error_threshold,
                    False,
                    self._state.simulation_count,
                )
                return self._done_obs(
                    reward=0.0,
                    score=final_score,
                    message="Maximum steps exceeded. Episode ended.",
                )

            if action.action_type == "RUN_SIMULATION":
                return self._handle_simulation(action)
            elif action.action_type == "SUBMIT_VERDICT":
                return self._handle_verdict(action)
            else:
                return self._error_obs(
                    f"Unknown action_type: {action.action_type}. "
                    f"Use 'RUN_SIMULATION' or 'SUBMIT_VERDICT'."
                )
        except Exception as e:
            # Absolute fallback to ensure we NEVER crash the container and ALWAYS return a valid score (between 0 and 1)
            # if the evaluator tries an invalid action payload or syntax.
            return self._done_obs(
                reward=0.0,
                score=0.001,
                message=f"Internal scenario implementation error: {str(e)}"
            )

    def _handle_simulation(
        self, action: ForensicHawkeyeAction
    ) -> ForensicHawkeyeObservation:
        """Run a physics simulation with the agent's parameters."""
        scenario = self._scenario

        # Validate parameters - now allows either sim_parameters OR the global parameters
        sim_params = action.sim_parameters or {}
        if not sim_params and action.friction_coefficient is None and action.restitution is None and action.mass_overrides is None and action.impact_offset_y is None:
            return self._step_obs(
                reward=-0.05,
                message="No parameters provided. You must provide physics parameters.",
            )

        # Run physics simulation
        physics_config = scenario.get_physics_config()
        result = self._physics.simulate(physics_config, sim_params, action)

        # Compute distance errors
        target = scenario.target_debris
        self._simulated_debris = result.final_positions
        self._distance_errors, self._total_error = compute_distance_errors(
            target, result.final_positions
        )

        # Track best error
        if self._total_error < self._best_total_error:
            self._best_total_error = self._total_error

        # Check for testimony contradiction
        self._contradiction_flag = scenario.check_contradiction(sim_params)
        self._last_sim_params = sim_params
        self._state.simulation_count += 1

        # Compute reward
        reward = compute_step_reward(
            "RUN_SIMULATION",
            self._total_error,
            self._previous_total_error,
            scenario.error_threshold,
        )
        self._previous_total_error = self._total_error
        self._rewards.append(reward)

        # Build message
        direction = ""
        if self._total_error < self._previous_total_error + 0.01:
            direction = "↓ Error decreased — getting closer!"
        elif self._total_error > self._previous_total_error:
            direction = "↑ Error increased — try different values."
        else:
            direction = "→ Error unchanged."

        message = (
            f"Simulation #{self._state.simulation_count} complete. "
            f"Total distance error: {self._total_error:.2f}m "
            f"(threshold: {scenario.error_threshold:.1f}m). {direction}"
        )

        if self._contradiction_flag:
            message += (
                " ⚠ CONTRADICTION DETECTED: Physics results conflict "
                "with witness testimony!"
            )

        return self._step_obs(reward=reward, message=message)

    def _handle_verdict(
        self, action: ForensicHawkeyeAction
    ) -> ForensicHawkeyeObservation:
        """Process a verdict submission."""
        scenario = self._scenario

        liable = (action.liable_party or "").strip()
        cause = (action.root_cause or "").strip()

        if not liable or not cause:
            return self._step_obs(
                reward=-0.1,
                message="Verdict requires both 'liable_party' and 'root_cause'. "
                "Try again.",
            )

        # Check correctness
        liable_correct = liable.lower() == scenario.ground_truth_liable.lower()
        cause_correct = cause.lower() == scenario.ground_truth_cause.lower()
        verdict_correct = liable_correct and cause_correct

        self._state.verdict_submitted = True

        # Compute rewards
        reward = compute_step_reward(
            "SUBMIT_VERDICT",
            self._best_total_error,
            self._previous_total_error,
            scenario.error_threshold,
            verdict_correct,
        )
        self._rewards.append(reward)

        # Calculate physics bonus
        physics_bonus = 0.0
        if action.friction_coefficient is not None:
            if abs(action.friction_coefficient - scenario.ground_truth_friction) < 0.1:
                physics_bonus += 0.05
        if action.restitution is not None:
            if abs(action.restitution - scenario.ground_truth_restitution) < 0.1:
                physics_bonus += 0.05
        if action.mass_overrides:
            mass_correct = True
            for entity, truth_mass in scenario.ground_truth_masses.items():
                guessed_mass = action.mass_overrides.get(entity)
                if guessed_mass is None or abs(guessed_mass - truth_mass) > 100.0:
                    mass_correct = False
                    break
            if mass_correct:
                physics_bonus += 0.05

        # Final score
        final_score = compute_final_score(
            self._best_total_error,
            scenario.error_threshold,
            verdict_correct,
            self._state.simulation_count,
            physics_bonus=physics_bonus,
        )

        # Message
        if verdict_correct:
            message = (
                f"✓ CORRECT VERDICT! {liable} is liable for {cause}. "
                f"Final reconstruction error: {self._best_total_error:.2f}m. "
                f"Score: {final_score:.3f}"
            )
        else:
            parts = []
            if not liable_correct:
                parts.append(f"Wrong liable party (you said '{liable}')")
            if not cause_correct:
                parts.append(f"Wrong root cause (you said '{cause}')")
            message = (
                f"✗ INCORRECT VERDICT. {'; '.join(parts)}. "
                f"Score: {final_score:.3f}"
            )

        return self._done_obs(reward=reward, score=final_score, message=message)

    def _step_obs(
        self, reward: float, message: str
    ) -> ForensicHawkeyeObservation:
        """Build a mid-episode observation."""
        scenario = self._scenario
        target_as_lists = {k: list(v) for k, v in scenario.target_debris.items()}
        sim_as_lists = {k: list(v) for k, v in self._simulated_debris.items()}

        hints = _extract_hints(scenario.testimony)

        return ForensicHawkeyeObservation(
            done=False,
            # Hackathon: Intermediate steps MUST return 0.0 so episode sum is exactly final_score
            reward=0.0,
            task_id=self._task_id,
            task_name=scenario.task_name,
            target_debris=target_as_lists,
            simulated_debris=sim_as_lists,
            distance_errors=self._distance_errors,
            total_distance_error=self._total_error,
            human_testimony=scenario.testimony,
            weather_conditions=hints["weather"],
            vehicle_descriptions=hints["vehicles"],
            damage_description=hints["damage"],
            active_contradiction_flag=self._contradiction_flag,
            step_number=self._state.step_count,
            max_steps=scenario.max_steps,
            message=message,
            available_entities=scenario.available_entities,
            available_parameters=scenario.available_parameters,
        )

    def _done_obs(
        self, reward: float, score: float, message: str
    ) -> ForensicHawkeyeObservation:
        """Build an end-of-episode observation."""
        scenario = self._scenario
        target_as_lists = {k: list(v) for k, v in scenario.target_debris.items()}
        sim_as_lists = {k: list(v) for k, v in self._simulated_debris.items()}

        hints = _extract_hints(scenario.testimony)

        return ForensicHawkeyeObservation(
            done=True,
            reward=score,  # Final observation uses the grader score as reward
            task_id=self._task_id,
            task_name=scenario.task_name,
            target_debris=target_as_lists,
            simulated_debris=sim_as_lists,
            distance_errors=self._distance_errors,
            total_distance_error=self._best_total_error,
            human_testimony=scenario.testimony,
            weather_conditions=hints["weather"],
            vehicle_descriptions=hints["vehicles"],
            damage_description=hints["damage"],
            active_contradiction_flag=self._contradiction_flag,
            step_number=self._state.step_count,
            max_steps=scenario.max_steps,
            message=message,
            available_entities=scenario.available_entities,
            available_parameters=scenario.available_parameters,
        )

    def _error_obs(self, message: str) -> ForensicHawkeyeObservation:
        """Build an error observation."""
        return ForensicHawkeyeObservation(
            done=False,
            reward=0.0,
            task_id=self._task_id,
            task_name="",
            target_debris={},
            simulated_debris={},
            distance_errors={},
            total_distance_error=999.0,
            human_testimony="",
            active_contradiction_flag=False,
            step_number=0,
            max_steps=15,
            message=message,
            available_entities=[],
            available_parameters=[],
        )

    @property
    def state(self) -> ForensicHawkeyeState:
        """Get the current environment state."""
        return self._state
