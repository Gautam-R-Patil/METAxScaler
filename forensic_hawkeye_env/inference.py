"""
Forensic Hawkeye Environment — Baseline Inference Script
Meta-PyTorch Hackathon Round 1

Reads environment variables:
  API_BASE_URL  — The API endpoint for the LLM.
  MODEL_NAME    — The model identifier to use for inference.
  HF_TOKEN      — Your Hugging Face / API key.
  ENV_URL       — (optional) Environment server URL, defaults to http://localhost:8000
"""

import os
import sys
import json
import logging
import textwrap
from pathlib import Path
from typing import Dict, Any, List, Optional

# Suppress noisy HTTP logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Ensure parent directory is on path for package imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
from forensic_hawkeye_env.client import ForensicHawkeyeEnv
from forensic_hawkeye_env.models import ForensicHawkeyeAction

# ── Configuration ─────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")
BENCHMARK = "forensic_hawkeye_env"
MAX_STEPS = 35
TEMPERATURE = 0.1
MAX_TOKENS = 512

SYSTEM_PROMPT = textwrap.dedent("""\
    You are Watson, an elite AI Forensic Auditor and detective. Your job is to reconstruct car accidents
    using a ruthless, unfeeling physics simulator to find the exact configuration that perfectly
    matches the target debris, thereby disproving false human testimony.

    You must respond with ONLY a valid JSON object. No markdown, no explanation outside the JSON.

    To run a simulation (gradient descent thought process):
    {
      "action_type": "RUN_SIMULATION",
      "thought": "Testimony claims rainy road (friction=0.3). Distance error is 12m short. I will increase speed.",
      "sim_parameters": {"Car_A": {"speed": 45.0, "steering": -5.0}},
      "friction_coefficient": 0.3,
      "restitution": 0.5,
      "mass_overrides": {"Car_A": 1500.0},
      "impact_offset_y": 0.0
    }

    To submit your final verdict (after error < threshold):
    {
      "action_type": "SUBMIT_VERDICT", 
      "liable_party": "Car_A", 
      "root_cause": "Speeding"
    }

    Strategy:
    1. Study testimony to extract the Four Pillars:
       - friction_coefficient (0.15 for ice, 0.3 for rain, 0.8 for dry)
       - restitution (0.1 for crushed, 0.5 for minor dent)
       - mass_overrides (1500 sedan, 8000 heavy truck, 75 pedestrian)
       - impact_offset_y (0.0 unless angular spin needed)
    2. Adjust 'sim_parameters' iteratively like gradient descent to minimize distance error.
    3. Once distance error < threshold, submit verdict.
""")


# ── Logging helpers (strict hackathon format) ─────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ── Action parsing ────────────────────────────────────────
def parse_llm_response(response_text: str) -> ForensicHawkeyeAction:
    """Extract and parse the JSON block from the LLM's response."""
    try:
        start = response_text.find('{')
        end = response_text.rfind('}')
        if start != -1 and end != -1:
            json_str = response_text[start:end + 1]
        else:
            json_str = response_text

        data = json.loads(json_str)
        action_type = data.get("action_type", "RUN_SIMULATION")
        action = ForensicHawkeyeAction(action_type=action_type)

        if action_type == "RUN_SIMULATION":
            action.sim_parameters = data.get("sim_parameters")
            action.friction_coefficient = data.get("friction_coefficient")
            action.restitution = data.get("restitution")
            action.mass_overrides = data.get("mass_overrides")
            action.impact_offset_y = data.get("impact_offset_y")
        elif action_type == "SUBMIT_VERDICT":
            action.liable_party = data.get("liable_party")
            action.root_cause = data.get("root_cause")

        return action
    except Exception:
        return ForensicHawkeyeAction(
            action_type="RUN_SIMULATION",
            sim_parameters={"Car_A": {"speed": 0}}
        )


# ── LLM interaction ──────────────────────────────────────
def build_user_prompt(obs) -> str:
    return textwrap.dedent(f"""\
        Task: {obs.task_name}
        Testimony: {obs.human_testimony}
        Target Debris (Ground Truth): {obs.target_debris}
        Simulated Debris (Last Run): {obs.simulated_debris}
        Distance Errors: {obs.distance_errors}
        Total Distance Error: {obs.total_distance_error}m
        Contradiction Detected: {obs.active_contradiction_flag}
        Available Entities: {obs.available_entities}
        Available Parameters: {obs.available_parameters}
        Env Message: {obs.message}

        What is your next action (JSON only)?
    """)


def get_model_action(client: OpenAI, messages: list, obs) -> str:
    user_msg = build_user_prompt(obs)
    messages.append({"role": "user", "content": user_msg})

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        messages.append({"role": "assistant", "content": text})
        return text
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return '{"action_type": "RUN_SIMULATION", "sim_parameters": {"Car_A": {"speed": 30}}}'


# ── Main task runner ──────────────────────────────────────
def run_task(task_id: int) -> None:
    global HF_TOKEN, API_BASE_URL, MODEL_NAME
    if not HF_TOKEN:
        HF_TOKEN = "dummy"

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    with ForensicHawkeyeEnv(base_url=ENV_URL).sync() as env:
        # Reset environment
        result = env.reset(task_id=task_id)
        obs = result.observation

        # [START] log
        task_name = f"task_{task_id}"
        log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        done = False

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            steps_taken = step

            try:
                # Get LLM response
                llm_text = get_model_action(client, messages, obs)
                action = parse_llm_response(llm_text)

                # Step the environment
                result = env.step(action)
                obs = result.observation
                done = obs.done
                reward = obs.reward if obs.reward is not None else 0.0
                rewards.append(reward)

                if done:
                    score = reward

                # [STEP] log
                log_step(step=step, action=action.action_type, reward=reward, done=done, error=None)

            except Exception as e:
                log_step(step=step, action="ERROR", reward=0.0, done=done, error=str(e))
                break

        success = score > 0.5

        # [END] log
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    for task_id in range(1, 4):
        run_task(task_id=task_id)
