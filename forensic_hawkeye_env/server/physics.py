"""
Headless pymunk physics engine for accident reconstruction.

Runs 2D rigid-body simulations with a 2.5D trick: longitudinal weight transfer
under braking dynamically adjusts friction coefficients, providing realistic
physics without 3D computational overhead.

The LLM never sees this module. It only interacts via Pydantic models.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import pymunk


# ─── Constants ────────────────────────────────────────────────────────────────

GRAVITY = (0.0, 0.0)  # Top-down view, no gravity pull
DT = 1.0 / 120.0  # Fixed timestep for determinism
MAX_SIM_STEPS = 3000  # ~25 seconds of sim time
REST_VELOCITY_THRESHOLD = 0.5  # m/s — body considered at rest below this
DEFAULT_FRICTION = 0.7
DEFAULT_ELASTICITY = 0.3

# 2.5D weight transfer constants
DEFAULT_COG_HEIGHT = 0.55  # meters — center of gravity height
DEFAULT_WHEELBASE = 2.7  # meters
G_ACCEL = 9.81  # m/s^2


# ─── Data Classes ─────────────────────────────────────────────────────────────


@dataclass
class EntityConfig:
    """Configuration for a single physics entity (car, pedestrian, obstacle)."""

    name: str
    mass: float  # kg
    width: float = 2.0  # meters
    height: float = 4.5  # meters (length for cars)
    start_x: float = 0.0
    start_y: float = 0.0
    heading_deg: float = 0.0  # Base travel direction in degrees (0=right, 90=up)
    is_static: bool = False
    entity_type: str = "car"  # car, pedestrian, obstacle


@dataclass
class ScenarioPhysicsConfig:
    """Full physics configuration for a scenario."""

    entities: List[EntityConfig] = field(default_factory=list)
    static_obstacles: List[Dict] = field(default_factory=list)  # {name, x, y, w, h}
    world_bounds: Tuple[float, float] = (200.0, 200.0)  # meters


@dataclass
class SimulationResult:
    """Result of a physics simulation run."""

    final_positions: Dict[str, Tuple[float, float]]
    was_collision: bool = False


class PhysicsWorld:
    """
    Headless pymunk-based 2D physics simulation for crash reconstruction.

    Supports the 2.5D weight transfer trick: when a car brakes, longitudinal
    weight transfer increases front-axle load, dynamically adjusting the
    effective friction coefficient.
    """

    def __init__(self):
        self._space: Optional[pymunk.Space] = None

    def _compute_dynamic_friction(
        self,
        speed: float,
        deceleration: float,
        mass: float,
        cog_height: float = DEFAULT_COG_HEIGHT,
        wheelbase: float = DEFAULT_WHEELBASE,
    ) -> float:
        """
        Compute the 2.5D dynamic friction coefficient.

        When braking, weight transfers to the front axle:
            ΔW_front = (mass × deceleration × cog_height) / wheelbase
            μ_effective = μ_base × (1 + ΔW_front / W_static_per_axle)
        """
        if speed < 1.0 or deceleration < 0.1:
            return DEFAULT_FRICTION

        static_weight_per_axle = (mass * G_ACCEL) / 2.0
        weight_transfer = (mass * deceleration * cog_height) / wheelbase

        mu_effective = DEFAULT_FRICTION * (
            1.0 + weight_transfer / max(static_weight_per_axle, 1.0)
        )
        # Clamp to physically reasonable range
        return min(mu_effective, 1.2)

    def simulate(
        self,
        physics_config: ScenarioPhysicsConfig,
        sim_params: Dict[str, Dict[str, float]],
    ) -> SimulationResult:
        """
        Run a full physics simulation with given parameters.

        Args:
            physics_config: Scenario layout (entities, obstacles, bounds)
            sim_params: Agent-provided parameters per entity
                        e.g. {"Car_A": {"speed": 54.0, "steering": -5.0}}

        Returns:
            SimulationResult with final positions of all dynamic entities
        """
        space = pymunk.Space()
        space.gravity = GRAVITY
        space.damping = 0.95  # 1.0 = no damping, lower = more friction

        bodies: Dict[str, pymunk.Body] = {}
        was_collision = False

        # Track collisions using pymunk 7 callback
        def on_collision(arbiter, space, data):
            nonlocal was_collision
            was_collision = True
            return True

        space.on_collision = on_collision

        # ── Add world boundaries ──
        bw, bh = physics_config.world_bounds
        walls = [
            pymunk.Segment(space.static_body, (0, 0), (bw, 0), 1),
            pymunk.Segment(space.static_body, (bw, 0), (bw, bh), 1),
            pymunk.Segment(space.static_body, (bw, bh), (0, bh), 1),
            pymunk.Segment(space.static_body, (0, bh), (0, 0), 1),
        ]
        for w in walls:
            w.elasticity = 0.2
            w.friction = DEFAULT_FRICTION
        space.add(*walls)

        # ── Add static obstacles ──
        for obs in physics_config.static_obstacles:
            obs_body = pymunk.Body(body_type=pymunk.Body.STATIC)
            obs_body.position = (obs["x"], obs["y"])
            hw, hh = obs.get("w", 2.0) / 2, obs.get("h", 2.0) / 2
            obs_shape = pymunk.Poly(
                obs_body,
                [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)],
            )
            obs_shape.elasticity = DEFAULT_ELASTICITY
            obs_shape.friction = DEFAULT_FRICTION
            space.add(obs_body, obs_shape)

        # ── Add dynamic entities ──
        for entity_cfg in physics_config.entities:
            if entity_cfg.is_static:
                continue

            params = sim_params.get(entity_cfg.name, {})
            speed = params.get("speed", 0.0)
            steering_deg = params.get("steering", 0.0)
            timing_offset = params.get("timing_offset", 0.0)
            velocity_param = params.get("velocity", None)  # For pedestrians

            # Convert speed from mph to m/s
            speed_ms = speed * 0.44704

            # For pedestrians, use velocity directly (m/s)
            if entity_cfg.entity_type == "pedestrian" and velocity_param is not None:
                speed_ms = velocity_param

            # Total heading = base heading + agent steering offset
            total_heading_deg = entity_cfg.heading_deg + steering_deg
            heading_rad = math.radians(total_heading_deg)

            # Compute velocity components
            vx = speed_ms * math.cos(heading_rad)
            vy = speed_ms * math.sin(heading_rad)

            # Create body
            moment = pymunk.moment_for_box(
                entity_cfg.mass, (entity_cfg.width, entity_cfg.height)
            )
            body = pymunk.Body(entity_cfg.mass, moment)
            body.position = (
                entity_cfg.start_x + timing_offset * vx * 0.5,
                entity_cfg.start_y + timing_offset * vy * 0.5,
            )
            body.velocity = (vx, vy)

            # Shape
            hw, hh = entity_cfg.width / 2, entity_cfg.height / 2
            if entity_cfg.entity_type == "pedestrian":
                shape = pymunk.Circle(body, entity_cfg.width / 2)
            else:
                shape = pymunk.Poly(
                    body, [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
                )

            # Apply 2.5D dynamic friction
            decel = speed_ms * 0.3  # Approximate deceleration from friction
            mu = self._compute_dynamic_friction(
                speed_ms, decel, entity_cfg.mass
            )
            shape.friction = mu
            shape.elasticity = DEFAULT_ELASTICITY

            space.add(body, shape)
            bodies[entity_cfg.name] = body

        # ── Run simulation ──
        for _ in range(MAX_SIM_STEPS):
            space.step(DT)

            # Check if all bodies are at rest
            all_at_rest = all(
                body.velocity.length < REST_VELOCITY_THRESHOLD
                for body in bodies.values()
            )
            if all_at_rest:
                break

        # ── Collect final positions ──
        final_positions = {}
        for name, body in bodies.items():
            final_positions[name] = (
                round(body.position.x, 2),
                round(body.position.y, 2),
            )

        return SimulationResult(
            final_positions=final_positions,
            was_collision=was_collision,
        )


def compute_distance_errors(
    target: Dict[str, Tuple[float, float]],
    simulated: Dict[str, Tuple[float, float]],
) -> Tuple[Dict[str, float], float]:
    """
    Compute per-entity and total distance errors.

    Args:
        target: Ground truth final positions
        simulated: Simulated final positions

    Returns:
        (per_entity_errors, total_error)
    """
    errors = {}
    for name in target:
        if name in simulated:
            tx, ty = target[name]
            sx, sy = simulated[name]
            dist = math.sqrt((tx - sx) ** 2 + (ty - sy) ** 2)
            errors[name] = round(dist, 3)
        else:
            errors[name] = 999.0

    total = sum(errors.values())
    return errors, round(total, 3)
