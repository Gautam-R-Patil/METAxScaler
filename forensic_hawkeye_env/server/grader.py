"""
Grader and reward shaping for the Forensic Hawkeye environment.

Provides continuous reward signals (not sparse binary) measuring:
- Simulation quality (did error decrease?)
- Verdict correctness (right party + cause?)
- Final score (0.0–1.0) for hackathon grading
"""

from typing import Dict, Optional, Tuple


def compute_step_reward(
    action_type: str,
    current_total_error: float,
    previous_total_error: float,
    error_threshold: float,
    verdict_correct: Optional[bool] = None,
) -> float:
    """
    Compute the reward for a single step.

    Reward schedule (from PRD):
        +0.1: Valid RUN_SIMULATION
        +0.2: distance_error decreased
        -0.1: distance_error increased
        -0.5: Premature SUBMIT_VERDICT (error still high)
        +0.5: Correct final verdict
    """
    reward = 0.0

    if action_type == "RUN_SIMULATION":
        # Base reward for valid simulation
        reward += 0.1

        # Reward improvement, penalize regression
        if current_total_error < previous_total_error:
            # Scale by how much improvement: bigger drops get more reward
            improvement_ratio = (previous_total_error - current_total_error) / max(
                previous_total_error, 0.01
            )
            reward += 0.2 + min(improvement_ratio * 0.3, 0.3)
        elif current_total_error > previous_total_error:
            reward -= 0.1

    elif action_type == "SUBMIT_VERDICT":
        if current_total_error > error_threshold * 3:
            # Premature verdict — error still way too high
            reward -= 0.5
        elif verdict_correct is True:
            # Correct verdict
            reward += 0.5
        elif verdict_correct is False:
            # Wrong verdict
            reward -= 0.3

    return round(reward, 3)


def compute_final_score(
    total_error: float,
    error_threshold: float,
    verdict_correct: bool,
    simulation_count: int,
    physics_bonus: float = 0.0,
) -> float:
    """
    Compute the final grader score (0.0–1.0) for hackathon evaluation.

    Scoring:
    - 0.0: Agent never got close or wrong verdict
    - 0.5: Good reconstruction but wrong/no verdict
    - 0.7: Decent reconstruction + correct verdict
    - 1.0: Perfect reconstruction (< threshold) + correct verdict

    Args:
        total_error: Final total distance error
        error_threshold: Scenario-specific threshold
        verdict_correct: Whether the agent identified the right party + cause
        simulation_count: Number of simulations run
    """
    # Physics accuracy component (0.0 – 0.6)
    if total_error <= error_threshold:
        accuracy_score = 0.6
    elif total_error <= error_threshold * 2:
        # Linear interpolation
        ratio = 1.0 - (total_error - error_threshold) / error_threshold
        accuracy_score = 0.3 + ratio * 0.3
    elif total_error <= error_threshold * 5:
        ratio = 1.0 - (total_error - error_threshold * 2) / (error_threshold * 3)
        accuracy_score = max(0.1 * ratio, 0.0)
    else:
        accuracy_score = 0.0

    # Verdict component (0.0 – 0.4)
    verdict_score = 0.4 if verdict_correct else 0.0

    # Small penalty if agent used too few simulations (didn't explore)
    if simulation_count < 2 and verdict_correct:
        # Likely guessed — slight penalty
        accuracy_score *= 0.8

    final_score = accuracy_score + verdict_score + physics_bonus
    
    # Bound strictly between 0 and 1 exclusive (not 0.0, not 1.0)
    final_score = max(0.001, min(final_score, 0.999))
    
    return round(final_score, 3)
