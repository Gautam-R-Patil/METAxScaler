<p align="center">
  <img src="https://img.shields.io/badge/OpenEnv-Forensic%20Hawkeye-crimson?style=for-the-badge&logo=pytorch&logoColor=white" alt="Forensic Hawkeye" />
  <img src="https://img.shields.io/badge/Meta%20PyTorch-Hackathon-blue?style=for-the-badge" alt="Hackathon" />
  <img src="https://img.shields.io/badge/Python-3.10+-green?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
</p>

<h1 align="center">🦅 Forensic Hawkeye Environment</h1>

<p align="center">
  <strong>A neuro-symbolic accident reconstruction environment for AI agent evaluation</strong>
  <br />
  <em>Can an AI detective disprove false human testimony using physics alone?</em>
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/Axelrod15/forensic-hawkeye-env">🤗 Live Demo</a> •
  <a href="#tasks">📋 Tasks</a> •
  <a href="#quickstart">🚀 Quickstart</a> •
  <a href="#architecture">🏗️ Architecture</a>
</p>

---

## 🔍 Overview

**Forensic Hawkeye** is an OpenEnv-compliant reinforcement learning environment that challenges AI agents to act as **forensic liability auditors**. Given crash scene evidence (final debris positions) and witness testimony, the agent must:

1. **Iteratively run physics simulations** with different parameter configurations
2. **Converge** on values that reproduce the observed debris pattern
3. **Identify contradictions** between physics results and human testimony
4. **Submit a verdict** naming the liable party and root cause

The environment uses a headless **pymunk** physics engine with a 2.5D weight transfer model to simulate realistic vehicle collisions, making it a genuine test of an agent's scientific reasoning and iterative problem-solving capabilities.

---

## 🎯 Tasks

Three tasks of increasing difficulty, each requiring the agent to reconstruct a different accident scenario:

| # | Task Name | Difficulty | Variables | Description |
|---|-----------|:----------:|:---------:|-------------|
| 1 | **The Public Property Strike** | 🟢 Easy | 2 | Single car hits a bus stop. Driver claims 30 mph — physics proves 54 mph. |
| 2 | **The Pedestrian Crosswalk** | 🟡 Medium | 3 | Car strikes pedestrian. Driver claims they stopped — skid marks say otherwise. |
| 3 | **The Highway Pileup** | 🔴 Hard | 6 | Three-vehicle chain collision. Conservation of momentum reveals the true initiator. |

### The Four Pillars

Agents must deduce four hidden global physics constants from contextual clues in testimony:

| Parameter | Clue Source | Range |
|-----------|------------|-------|
| `friction_coefficient` | Weather (rain → 0.3, ice → 0.15, dry → 0.8) | 0.1 – 0.9 |
| `restitution` | Damage severity (crushed → 0.1, dent → 0.5) | 0.1 – 0.8 |
| `mass_overrides` | Vehicle type (sedan → 1500kg, truck → 8000kg) | per entity |
| `impact_offset_y` | Angular spin / side-impact evidence | -5.0 – 5.0 |

---

## 🚀 Quickstart

### Prerequisites

```bash
pip install openenv-core pymunk
```

### Run Locally

```bash
# Clone the repository
git clone https://github.com/Gautam-R-Patil/METAxScaler.git
cd METAxScaler/forensic_hawkeye_env

# Install dependencies
uv sync

# Start the environment server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# In a separate terminal, run the baseline agent
python inference.py
```

### Run with Docker

```bash
cd forensic_hawkeye_env
docker build -f server/Dockerfile -t forensic-hawkeye .
docker run -p 8000:8000 forensic-hawkeye
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_BASE_URL` | LLM API endpoint | `https://api-inference.huggingface.co/v1` |
| `MODEL_NAME` | Model identifier | `meta-llama/Llama-3.1-8B-Instruct` |
| `HF_TOKEN` | HuggingFace API key | — |
| `ENV_URL` | Environment server URL | `http://localhost:8000` |

---

## 🏗️ Architecture

```
forensic_hawkeye_env/
├── openenv.yaml              # OpenEnv spec metadata
├── inference.py              # Baseline agent (Watson persona)
├── models.py                 # Pydantic Action/Observation/State models
├── client.py                 # OpenEnv WebSocket client
├── pyproject.toml            # Dependencies
└── server/
    ├── app.py                # FastAPI application (create_app)
    ├── forensic_hawkeye_env_environment.py  # Core environment logic
    ├── grader.py             # Reward shaping & final scoring
    ├── physics.py            # Headless pymunk simulation engine
    └── scenarios/
        ├── base.py           # Abstract scenario interface
        ├── task1_property_strike.py
        ├── task2_pedestrian.py
        └── task3_momentum.py
```

### Agent Loop

```
┌──────────────┐     reset(task_id)     ┌──────────────────┐
│              │ ──────────────────────▸ │                  │
│   AI Agent   │                        │   Environment    │
│  (Watson)    │ ◂────────────────────  │   (pymunk)       │
│              │   observation + reward  │                  │
│              │                        │                  │
│              │  step(RUN_SIMULATION)   │                  │
│              │ ──────────────────────▸ │  Physics Engine  │
│              │ ◂────────────────────  │  ┌────────────┐  │
│              │  debris positions +     │  │ Grader     │  │
│              │  distance errors        │  └────────────┘  │
│              │                        │                  │
│              │  step(SUBMIT_VERDICT)   │                  │
│              │ ──────────────────────▸ │  Final Score     │
│              │ ◂────────────────────  │  (0.0 – 1.0)    │
└──────────────┘                        └──────────────────┘
```

### Reward Design

| Signal | Value | Trigger |
|--------|:-----:|---------|
| Valid simulation | +0.1 | Each `RUN_SIMULATION` step |
| Error decreased | +0.2 to +0.5 | Distance error improved vs previous |
| Error increased | -0.1 | Distance error worsened |
| Correct verdict | +0.5 | Right party + right cause |
| Wrong verdict | -0.3 | Incorrect identification |
| Premature verdict | -0.5 | Submitted before error < 3× threshold |

### Final Score Composition

```
Final Score = Accuracy Component (0–0.6) + Verdict Component (0–0.4) + Physics Bonus (0–0.15)
```

- **Accuracy**: Based on how close the agent's best simulation matched target debris
- **Verdict**: Binary — correct liable party AND root cause
- **Physics Bonus**: Correctly deducing the Four Pillars values

---

## 📊 Baseline Scores

Measured with `meta-llama/Llama-3.1-8B-Instruct`:

| Task | Difficulty | Baseline Score | Avg Steps |
|------|:----------:|:--------------:|:---------:|
| Task 1 — Property Strike | 🟢 Easy | ~0.45 | 8 |
| Task 2 — Pedestrian | 🟡 Medium | ~0.30 | 12 |
| Task 3 — Highway Pileup | 🔴 Hard | ~0.15 | 15 |

*Frontier models (GPT-4o, Claude Sonnet) are expected to score significantly higher due to superior physics reasoning.*

---

## 🔗 Links

- **Hugging Face Space**: [Axelrod15/forensic-hawkeye-env](https://huggingface.co/spaces/Axelrod15/forensic-hawkeye-env)
- **OpenEnv Framework**: [openenv-core](https://pypi.org/project/openenv-core/)

---

## 📜 License

This project is licensed under the BSD-style license. See the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Built for the <strong>Meta PyTorch Hackathon × Scaler School of Technology</strong>
</p>
