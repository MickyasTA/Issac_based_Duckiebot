"""The SPEC v2 S5.5 terminations and truncation, vectorised over ``N`` envs.

Pure torch, no Isaac import, so every condition is testable on CPU.

S5.5 in full
------------

``terminated`` is the union of five conditions:

1. **off-drivable** - any of four test points (robot centre, both wheel ground-contact points,
   chassis front mid) leaves the drivable tiles. Four points rather than one because a single
   centre test lets the robot hang a wheel over the edge for free, and a full polygon test costs
   more than it buys at this tile scale.
2. **obstacle** - the geometric safety circle ``dist(centre, obstacle centre) < 0.12 + r_obs``.
   This is the primary mechanism; the kinematic colliders on the obstacle bodies are the
   backstop, which is the decision critic item H asked for explicitly.
3. **rollover** - ``|roll| > 30 deg`` or ``|pitch| > 30 deg``.
4. **stall** - body speed below 0.03 m/s continuously for longer than 2 s.
5. **spin** - integrated ``|yaw rate|`` beyond ``3 * pi`` while the net displacement from the
   spawn point is under 0.2 m. Both halves are required: a lap of a small loop legitimately
   integrates several turns of yaw, so the displacement clause is what separates "drove around"
   from "span on the spot".

``truncated`` is the 450-step horizon and nothing else. The two flags are returned separately
and must stay separate all the way into GAE: terminated zeroes the bootstrap, truncated
bootstraps from the captured terminal observation (S6.4). Collapsing them is the single most
common from-scratch PPO bug.

Stateful conditions
-------------------

Stall and spin are not functions of the current state alone, so :class:`TerminationState` owns
the two counters. It is reset per env id, exactly like the rest of the environment, and it
serialises so a checkpoint resume does not silently hand every env a fresh stall counter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from duckiebot_rl.assets.params import DUCKIEBOT, DuckiebotParams

__all__ = [
    "OBSTACLE_MARGIN_M",
    "ROLLOVER_LIMIT_RAD",
    "SPIN_MIN_DISPLACEMENT_M",
    "SPIN_YAW_LIMIT_RAD",
    "STALL_SPEED_M_S",
    "STALL_TIMEOUT_S",
    "TerminationFlags",
    "TerminationState",
    "obstacle_contact",
    "off_drivable",
    "rollover",
    "test_points",
    "truncated_by_horizon",
]

STALL_SPEED_M_S: float = 0.03
"""Body speed below which the robot counts as stalled (S5.4 stall term and S5.5 condition 4)."""

STALL_TIMEOUT_S: float = 2.0
"""Continuous stall duration that terminates the episode."""

ROLLOVER_LIMIT_RAD: float = math.radians(30.0)
"""Roll or pitch magnitude that terminates the episode."""

SPIN_YAW_LIMIT_RAD: float = 3.0 * math.pi
"""Integrated absolute yaw beyond which the spin guard can fire."""

SPIN_MIN_DISPLACEMENT_M: float = 0.2
"""Net displacement below which the spin guard fires once the yaw limit is exceeded."""

OBSTACLE_MARGIN_M: float = 0.12
"""Geometric safety radius added to each obstacle radius (S5.5 condition 2)."""


@dataclass
class TerminationFlags:
    """Per-condition boolean masks plus the two flags the env returns.

    Every field is ``(N,)`` bool. The per-condition masks exist because the S6.8 diagnostics log
    a termination histogram, and because a failure-mode histogram is one of the M12 release
    artifacts; recovering "why" from a single collapsed flag afterwards is not possible.

    Attributes:
        off_drivable: Condition 1.
        obstacle: Condition 2.
        rollover: Condition 3.
        stall: Condition 4.
        spin: Condition 5.
        terminated: Logical OR of the five conditions.
        truncated: Time limit reached, and NOT already terminated.
    """

    off_drivable: torch.Tensor
    obstacle: torch.Tensor
    rollover: torch.Tensor
    stall: torch.Tensor
    spin: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor

    def counts(self) -> dict[str, int]:
        """Return the number of envs each condition fired for, for logging.

        Returns:
            ``{condition_name: count}`` over the five conditions plus the two flags.
        """
        return {name: int(mask.sum().item()) for name, mask in self.__dict__.items()}


# =============================================================================================
# Stateless conditions
# =============================================================================================


def test_points(
    x: torch.Tensor,
    y: torch.Tensor,
    yaw: torch.Tensor,
    params: DuckiebotParams = DUCKIEBOT,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the four S5.5 off-drivable test points in world coordinates.

    The points are the robot centre, the two wheel ground-contact points at ``+/-b/2`` on the
    axle, and the chassis front mid-point. The MuJoCo twin
    (``duckiebot_rl.sim2sim.env.MjDuckiebotEnv._test_points``) uses the identical four offsets,
    which is what makes an off-drivable termination mean the same thing in both simulators.

    Args:
        x: ``(N,)`` world x of the robot origin (the axle midpoint) in metres.
        y: ``(N,)`` world y in metres.
        yaw: ``(N,)`` heading in radians, counter-clockwise from ``+x``.
        params: Parameter set supplying the baseline and the chassis geometry.

    Returns:
        ``(px, py)``, each ``(N, 4)``: centre, left wheel, right wheel, chassis front.
    """
    half = params.half_baseline_m
    front = params.chassis_center_base_frame_m[0] + 0.5 * params.chassis_size_m[0]
    local_x = torch.tensor([0.0, 0.0, 0.0, front], dtype=x.dtype, device=x.device)
    local_y = torch.tensor([0.0, half, -half, 0.0], dtype=x.dtype, device=x.device)
    cos_yaw = torch.cos(yaw).unsqueeze(-1)
    sin_yaw = torch.sin(yaw).unsqueeze(-1)
    px = x.unsqueeze(-1) + cos_yaw * local_x - sin_yaw * local_y
    py = y.unsqueeze(-1) + sin_yaw * local_x + cos_yaw * local_y
    return px, py


def off_drivable(
    lane_graph: object,
    variant_idx: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    yaw: torch.Tensor,
    params: DuckiebotParams = DUCKIEBOT,
) -> torch.Tensor:
    """Return True where any of the four test points has left the drivable tiles.

    Args:
        lane_graph: A :class:`duckiebot_rl.city.lane_graph.BatchedLaneGraph`. Typed as ``object``
            so this module stays importable without building a lane graph first; the only method
            used is ``is_drivable(variant_idx, x, y)``.
        variant_idx: ``(N,)`` long tensor of per-env map variant indices.
        x: ``(N,)`` world x of the robot origin.
        y: ``(N,)`` world y.
        yaw: ``(N,)`` heading in radians.
        params: Parameter set supplying the test-point offsets.

    Returns:
        ``(N,)`` bool mask.
    """
    px, py = test_points(x, y, yaw, params)
    num_envs, num_points = px.shape
    flat_variant = variant_idx.reshape(-1, 1).expand(num_envs, num_points).reshape(-1)
    drivable = lane_graph.is_drivable(flat_variant, px.reshape(-1), py.reshape(-1))  # type: ignore[attr-defined]
    return ~drivable.view(num_envs, num_points).all(dim=1)


def obstacle_contact(gap: torch.Tensor) -> torch.Tensor:
    """Return True where the robot centre has entered an obstacle's safety circle.

    Args:
        gap: ``(N,)`` signed clearance ``dist - (OBSTACLE_MARGIN_M + r_obs)`` to the nearest
            obstacle, in metres. ``+inf`` on an obstacle-free map.

    Returns:
        ``(N,)`` bool mask, all False where the gap is infinite.
    """
    return torch.isfinite(gap) & (gap < 0.0)


def rollover(
    roll: torch.Tensor,
    pitch: torch.Tensor,
    limit_rad: float = ROLLOVER_LIMIT_RAD,
) -> torch.Tensor:
    """Return True where the chassis has tipped past the roll/pitch limit.

    Args:
        roll: ``(N,)`` roll angle in radians.
        pitch: ``(N,)`` pitch angle in radians.
        limit_rad: Magnitude limit applied to both, 30 degrees by S5.5.

    Returns:
        ``(N,)`` bool mask.
    """
    return (roll.abs() > limit_rad) | (pitch.abs() > limit_rad)


def truncated_by_horizon(episode_length_buf: torch.Tensor, max_episode_length: int) -> torch.Tensor:
    """Return True where the episode has reached its step horizon.

    Isaac Lab increments ``episode_length_buf`` before ``_get_dones`` is called, so the
    comparison is ``>=`` against ``max_episode_length`` and the horizon fires on step 450 of a
    450-step episode rather than on step 451.

    Args:
        episode_length_buf: ``(N,)`` long tensor of elapsed control steps.
        max_episode_length: The horizon in control steps.

    Returns:
        ``(N,)`` bool mask.
    """
    return episode_length_buf >= max_episode_length


# =============================================================================================
# Stateful conditions
# =============================================================================================


class TerminationState:
    """Per-env counters behind the stall and spin conditions.

    Args:
        num_envs: Number of parallel envs.
        control_dt: Control period in seconds.
        device: Torch device.
        stall_speed: Body speed below which a step counts as stalled.
        stall_timeout_s: Continuous stall duration that terminates.
        spin_yaw_limit_rad: Integrated absolute yaw beyond which the spin guard can fire.
        spin_min_displacement_m: Net displacement below which the spin guard fires.
    """

    def __init__(
        self,
        num_envs: int,
        control_dt: float = DUCKIEBOT.control_dt_s,
        device: torch.device | str = "cpu",
        stall_speed: float = STALL_SPEED_M_S,
        stall_timeout_s: float = STALL_TIMEOUT_S,
        spin_yaw_limit_rad: float = SPIN_YAW_LIMIT_RAD,
        spin_min_displacement_m: float = SPIN_MIN_DISPLACEMENT_M,
    ) -> None:
        self.num_envs = int(num_envs)
        self.control_dt = float(control_dt)
        self.device = torch.device(device)
        self.stall_speed = float(stall_speed)
        self.stall_timeout_s = float(stall_timeout_s)
        self.spin_yaw_limit_rad = float(spin_yaw_limit_rad)
        self.spin_min_displacement_m = float(spin_min_displacement_m)

        self.stall_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.yaw_integral = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.spawn_xy = torch.zeros(self.num_envs, 2, dtype=torch.float32, device=self.device)

    @property
    def stall_step_limit(self) -> int:
        """Return the number of consecutive stalled steps that terminates the episode."""
        return math.floor(self.stall_timeout_s / self.control_dt)

    def reset(self, env_ids: torch.Tensor | None = None, spawn_xy: torch.Tensor | None = None) -> None:
        """Clear the counters of the given envs and record their new spawn position.

        Args:
            env_ids: Long tensor of env indices, or None for all envs.
            spawn_xy: ``(len(env_ids), 2)`` world spawn position. Required for the spin
                condition to mean anything; when None the stored position is left untouched,
                which is only correct for a state that has never been used.
        """
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids.to(self.device)
        self.stall_steps[ids] = 0
        self.yaw_integral[ids] = 0.0
        if spawn_xy is not None:
            self.spawn_xy[ids] = spawn_xy.to(device=self.device, dtype=self.spawn_xy.dtype)

    def update(
        self,
        body_speed: torch.Tensor,
        yaw_rate: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance the counters by one control step and evaluate both conditions.

        Args:
            body_speed: ``(N,)`` body-frame forward speed in m/s.
            yaw_rate: ``(N,)`` yaw rate in rad/s.
            x: ``(N,)`` world x.
            y: ``(N,)`` world y.

        Returns:
            ``(stall, spin)``, each a ``(N,)`` bool mask.
        """
        stalled_now = body_speed.abs() < self.stall_speed
        self.stall_steps = torch.where(stalled_now, self.stall_steps + 1, torch.zeros_like(self.stall_steps))
        stall = self.stall_steps > self.stall_step_limit

        self.yaw_integral = self.yaw_integral + yaw_rate.abs() * self.control_dt
        displacement = torch.sqrt((x - self.spawn_xy[:, 0]) ** 2 + (y - self.spawn_xy[:, 1]) ** 2)
        spin = (self.yaw_integral > self.spin_yaw_limit_rad) & (displacement < self.spin_min_displacement_m)
        return stall, spin

    def evaluate(
        self,
        lane_graph: object,
        variant_idx: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        yaw: torch.Tensor,
        roll: torch.Tensor,
        pitch: torch.Tensor,
        body_speed: torch.Tensor,
        yaw_rate: torch.Tensor,
        gap: torch.Tensor,
        episode_length_buf: torch.Tensor,
        max_episode_length: int,
        params: DuckiebotParams = DUCKIEBOT,
    ) -> TerminationFlags:
        """Evaluate all five terminations and the truncation for one control step.

        Args:
            lane_graph: A ``BatchedLaneGraph``.
            variant_idx: ``(N,)`` per-env map variant index.
            x: ``(N,)`` world x.
            y: ``(N,)`` world y.
            yaw: ``(N,)`` heading in radians.
            roll: ``(N,)`` roll in radians.
            pitch: ``(N,)`` pitch in radians.
            body_speed: ``(N,)`` body-frame forward speed in m/s.
            yaw_rate: ``(N,)`` yaw rate in rad/s.
            gap: ``(N,)`` signed clearance to the nearest obstacle safety circle.
            episode_length_buf: ``(N,)`` elapsed control steps.
            max_episode_length: Horizon in control steps.
            params: Parameter set supplying the test-point offsets.

        Returns:
            The populated :class:`TerminationFlags`. ``truncated`` excludes envs that also
            terminated, so the two masks are disjoint and GAE never sees a step flagged both
            ways.
        """
        off = off_drivable(lane_graph, variant_idx, x, y, yaw, params)
        hit = obstacle_contact(gap)
        tipped = rollover(roll, pitch)
        stall, spin = self.update(body_speed, yaw_rate, x, y)
        terminated = off | hit | tipped | stall | spin
        truncated = truncated_by_horizon(episode_length_buf, max_episode_length) & ~terminated
        return TerminationFlags(
            off_drivable=off,
            obstacle=hit,
            rollover=tipped,
            stall=stall,
            spin=spin,
            terminated=terminated,
            truncated=truncated,
        )

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Serialise the counters for a checkpoint.

        Returns:
            ``{"stall_steps", "yaw_integral", "spawn_xy"}`` as CPU-mappable tensors.
        """
        return {
            "stall_steps": self.stall_steps.clone(),
            "yaw_integral": self.yaw_integral.clone(),
            "spawn_xy": self.spawn_xy.clone(),
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Restore counters produced by :meth:`state_dict`.

        Args:
            state: The mapping returned by :meth:`state_dict`.

        Raises:
            KeyError: If a field is missing.
        """
        for key in ("stall_steps", "yaw_integral", "spawn_xy"):
            if key not in state:
                raise KeyError(f"termination state is missing field {key!r}")
        self.stall_steps.copy_(state["stall_steps"].to(self.stall_steps.device))
        self.yaw_integral.copy_(state["yaw_integral"].to(self.yaw_integral.device))
        self.spawn_xy.copy_(state["spawn_xy"].to(self.spawn_xy.device))
