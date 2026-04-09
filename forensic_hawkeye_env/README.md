---
title: Forensic Hawkeye Environment Server
emoji: "🚗"
colorFrom: blue
colorTo: gray
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Forensic Hawkeye Environment

A fully deterministic, 2.5D physics-based text environment for the **Meta-PyTorch Hackathon**. This environment trains AI agents to act as **Forensic Crash Investigators**, requiring them to iteratively run physics simulations, analyze ground-truth debris patterns, and logically disprove false human testimony to determine accident liability.

## Motivation

Accident reconstruction is a real-world forensic task performed by insurance investigators and law enforcement. It demands spatial reasoning, physics intuition, and logical deduction — capabilities that are poorly benchmarked in existing LLM evaluation suites. This environment provides a structured, deterministic testbed for exactly these skills.

## Action Space

The agent sends a `ForensicHawkeyeAction` (Pydantic model) with one of two action types:

| Field | Type | Description |
|-------|------|-------------|
| `action_type` | `"RUN_SIMULATION"` or `"SUBMIT_VERDICT"` | Which action to perform |
| `thought` | `str` (optional) | Agent's internal reasoning chain |
| `sim_parameters` | `Dict[str, Dict[str, float]]` | Per-entity physics parameters (for `RUN_SIMULATION`) |
| `friction_coefficient` | `float` (optional) | Global surface friction (0.15=ice, 0.3=rain, 0.8=dry) |
| `restitution` | `float` (optional) | Collision elasticity (0.1=crushed, 0.5=minor dent) |
| `mass_overrides` | `Dict[str, float]` (optional) | Custom mass per entity in kg |
| `impact_offset_y` | `float` (optional) | Y-axis collision offset for angular momentum |
| `liable_party` | `str` | The entity at fault (for `SUBMIT_VERDICT`) |
| `root_cause` | `str` | Why the accident happened (for `SUBMIT_VERDICT`) |

The agent extracts hidden physics variables from plain-English testimony (weather → friction, damage → restitution, vehicle type → mass) and iteratively tunes all parameters to match target debris.

**Example simulation action (neuro-symbolic):**
```json
{
  "action_type": "RUN_SIMULATION",
  "thought": "Testimony says torrential downpour (friction ~0.3) and compact sedan (mass ~1500). Error is 12m, increasing speed.",
  "sim_parameters": {"Car_A": {"speed": 54.0, "steering": -8.0}},
  "friction_coefficient": 0.3,
  "restitution": 0.1,
  "mass_overrides": {"Car_A": 1500.0},
  "impact_offset_y": 0.0
}
```

**Example verdict action:**
```json
{"action_type": "SUBMIT_VERDICT", "liable_party": "Car_A", "root_cause": "Speeding"}
```

## Observation Space

Each step returns a `ForensicHawkeyeObservation` with:

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | `int` | Current task (1-3) |
| `task_name` | `str` | Human-readable task name |
| `target_debris` | `Dict[str, [x, y]]` | Ground-truth crash debris positions |
| `simulated_debris` | `Dict[str, [x, y]]` | Debris from last simulation run |
| `distance_errors` | `Dict[str, float]` | Per-entity error (meters) between target and simulated |
| `total_distance_error` | `float` | Sum of all distance errors |
| `human_testimony` | `str` | Witness statement to evaluate |
| `active_contradiction_flag` | `bool` | Whether physics contradicts testimony |
| `available_entities` | `List[str]` | Entities that can be configured |
| `available_parameters` | `List[str]` | Tunable parameters per entity |
| `done` | `bool` | Episode finished |
| `reward` | `float` | Reward signal (0.0–1.0) |

## Reward Function

The reward provides **continuous partial-progress signal** (not sparse):

- **Per-step reward**: Based on improvement in `total_distance_error` from the previous step. Getting closer to ground truth = positive reward.
- **Final score** (on `SUBMIT_VERDICT`): Weighted combination of debris accuracy (70%) and verdict correctness (30%). Ranges 0.0–1.0.

## Tasks

Three progressive difficulty scenarios:

### Task 1: The Public Property Strike (Easy)
- **Variables**: 2 (`speed`, `steering` for Car_A)
- **Scenario**: A car strikes a bus stop. The agent must find the speed and steering angle that reproduce the ground-truth debris pattern.
- **Testimony**: The driver claims they were going 25 mph. Physics will show otherwise.

### Task 2: The Pedestrian Paradox (Medium)
- **Variables**: 3 (`speed`, `steering` for Car_A + `velocity` for Pedestrian)
- **Scenario**: A car nearly strikes a pedestrian near a utility pole. The agent must determine if the driver's claimed trajectory is physically possible.
- **Challenge**: Counterfactual reasoning — the agent must prove the testimony is impossible.

### Task 3: Multi-Vehicle Momentum (Hard)
- **Variables**: 6 (speed + timing for 3 cars)
- **Scenario**: Three-car intersection collision. The agent must use conservation of momentum to reconstruct how each vehicle entered the intersection.
- **Challenge**: High-dimensional search space with coupled momentum transfer.

## System Architecture

- **Physics Engine**: Headless `pymunk` simulation with deterministic stepping (dt=1/60, 300 steps per simulation).
- **2.5D Weight Transfer**: Dynamic friction adjustment under braking — simulates longitudinal weight transfer for more realistic deceleration physics.
- **Determinism**: 10/10 identical runs verified. Same inputs always produce the same debris positions.

## Setup & Usage

```bash
# Install
uv sync

# Run locally
openenv start forensic_hawkeye_env

# Or with uvicorn directly
uvicorn server.app:app --port 8000
```

## Inference

```bash
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export HF_TOKEN="your_hf_token"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export ENV_URL="https://Axelrod15-forensic-hawkeye-env.hf.space"

python inference.py
```

## Baseline Scores

| Task | Difficulty | Baseline Score | Steps |
|------|-----------|---------------|-------|
| Task 1: Property Strike | Easy | ~0.45 | 8-10 |
| Task 2: Pedestrian Paradox | Medium | ~0.30 | 10-12 |
| Task 3: Multi-Vehicle Momentum | Hard | ~0.15 | 12-15 |

*Baseline measured with Llama-3.1-8B-Instruct. Frontier models (GPT-4, Claude) expected to score significantly higher.*
