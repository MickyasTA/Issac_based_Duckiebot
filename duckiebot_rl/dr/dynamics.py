"""Dynamics domain-randomization sampler (SPEC v2 S7.3, layer 3).

SPEC v2 S7.1 layer 3 is explicit about where dynamics DR lives: *everything* motor- and
kinematics-related is applied in the **action path** (S5.3) as per-env tensors resampled on
reset. Nothing here writes USD, calls ``randomize_actuator_gains`` or touches an Isaac actuator
API, because a per-reset actuator write forces a CPU sync every episode (critic item E).

This module is therefore a pure sampler: it owns the S7.3 table (one
:class:`~duckiebot_rl.dr.curriculum.Range` per axis, so ``alpha_dyn`` gates them all) and hands
out per-env parameter tensors. The env, the MuJoCo harness and the sysid scripts all read the
same :class:`DynamicsParams`.

Axis coverage (S7.3):

* D1 wheel radius (+ left/right asymmetry), D2 baseline
* D3 tire friction, per wheel, with ``mu_d <= mu_s`` enforced
* D4 mass / Izz / CoM offset
* D5 motor gain and trim, D6 battery sag with in-episode decay, D7 dead-band (independent L/R)
* D8 actuation delay (integer control steps + sub-step fraction) and first-order motor lag
* D9 observation latency (camera stream only; the vec observation is never delayed)
* D10 control-period jitter, D11 drag, D12 wheel slip, D13 encoder noise and dropout
* D14 external pushes, D15 floor-patch friction, D16 spawn pose
* D17 wheel effort limit (written once at STARTUP per env, never per reset)
* D18 brake authority
* Actuator armature and joint friction (S2), which are the stage-2 sysid targets

The dominant axis, per the research, is D8 actuation latency (50-250 ms); it is implemented by
:class:`duckiebot_rl.dr.delay.DelayBuffer` in the action path, and only its *parameters* are
sampled here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

from duckiebot_rl.dr.curriculum import Range, RangeBook

__all__ = [
    "ENCODER_TICKS_PER_REV",
    "MOTOR_K_RAD_PER_S",
    "NOMINAL_BASELINE_M",
    "NOMINAL_WHEEL_RADIUS_M",
    "DynamicsDRCfg",
    "DynamicsParams",
    "DynamicsRandomizer",
    "default_dynamics_ranges",
    "quantize_encoder",
]

NOMINAL_WHEEL_RADIUS_M: float = 0.0318
"""Nominal wheel radius (SPEC v2 S2, [C]/[S])."""

NOMINAL_BASELINE_M: float = 0.100
"""Nominal wheel baseline / track (SPEC v2 S2, [C])."""

MOTOR_K_RAD_PER_S: float = 27.0
"""Motor constant: wheel rad/s per unit duty at gain 1.0 (SPEC v2 S2, [C])."""

ENCODER_TICKS_PER_REV: int = 135
"""Encoder resolution (SPEC v2 S2, [S]/[C]); the vec observation is quantized to it."""


def default_dynamics_ranges() -> dict[str, Range]:
    """Return the dynamics DR axes of SPEC v2 S7.3.

    Every entry carries the nominal (best-estimate real robot) value so that ``alpha_dyn = 0``
    reproduces the nominal robot exactly and ``alpha_dyn = 1`` reproduces the full table.

    Returns:
        A range book keyed by parameter name.
    """
    return {
        # D1 wheel radius
        "wheel_radius_scale": Range(0.95, 1.05, 1.0, "linear", "x nominal"),
        "wheel_radius_asym": Range(-0.06, 0.06, 0.0, "linear", "fraction, +L/-R"),
        # D2 baseline
        "baseline_m": Range(0.090, 0.110, NOMINAL_BASELINE_M, "linear", "m"),
        # D3 tire friction (sampled per wheel; kinetic is a ratio of static so mu_d <= mu_s)
        "friction_static": Range(0.40, 1.40, 1.0, "linear", ""),
        "friction_kinetic_ratio": Range(0.7, 1.0, 1.0, "linear", ""),
        # D4 mass / inertia / CoM (base body; the two wheels stay at 0.05 kg each)
        "mass_kg": Range(0.85, 1.40, 1.00, "linear", "kg"),
        "izz_scale": Range(0.7, 1.4, 1.0, "linear", "x nominal"),
        "com_x_m": Range(-0.035, 0.005, -0.015, "linear", "m"),
        "com_y_m": Range(-0.020, 0.020, 0.0, "linear", "m"),
        "com_z_m": Range(0.005, 0.025, 0.015, "linear", "m"),
        # D5 motor gain / trim
        "motor_gain": Range(0.60, 1.40, 1.0, "linear", ""),
        "motor_trim": Range(-0.12, 0.12, 0.0, "linear", ""),
        # D6 battery sag (start level + slow in-episode decay)
        "battery_sag": Range(0.90, 1.04, 1.0, "linear", "x nominal"),
        "battery_decay_per_s": Range(0.0, 0.002, 0.0, "linear", "1/s"),
        # D7 dead-band, independent per wheel; below it the wheel COASTS (S5.3 step 3)
        "deadband_duty": Range(0.0, 0.15, 0.0, "linear", "duty"),
        # D8 actuation delay. Nominal 2 steps + 0.25 sub-step ~ the 0.150 s modeled delay at the
        # 15 Hz control rate, so alpha_dyn = 0 is the nominal robot rather than a delay-free one.
        "delay_steps": Range(1.0, 3.0, 2.0, "int", "control steps"),
        "delay_frac": Range(0.0, 0.9, 0.25, "linear", "control steps"),
        # First-order motor lag. The identity (1.0) is outside the S7.3 table, so the nominal is
        # the least-lag end of the table; alpha_dyn = 0 gives alpha_lag = 0.9, not 1.0.
        "motor_lag_alpha": Range(0.4, 0.9, 0.9, "linear", ""),
        # D9 observation latency (camera stream only)
        "obs_delay_steps": Range(0.0, 2.0, 0.0, "int", "control steps"),
        # D10 control-period jitter
        "control_jitter": Range(0.9, 1.3, 1.0, "linear", "x dt"),
        # D11 drag (the u1/w1 equivalents of the Duckietown dynamics model)
        "drag_u1": Range(3.5, 7.0, 5.0, "linear", ""),
        "drag_w1": Range(2.5, 6.0, 4.0, "linear", ""),
        # D12 wheel slip: per-step multiplicative noise on the commanded wheel speed
        "slip_std": Range(0.0, 0.10, 0.0, "linear", "fraction"),
        # D13 encoder dropout (the 135-tick quantization itself is always on)
        "encoder_dropout_p": Range(0.0, 0.02, 0.0, "linear", "probability"),
        # D14 external push
        "push_interval_s": Range(3.0, 8.0, 5.5, "linear", "s"),
        "push_dv": Range(-0.15, 0.15, 0.0, "linear", "m/s"),
        "push_dyaw": Range(-0.6, 0.6, 0.0, "linear", "rad/s"),
        # D15 floor patches (per-tile friction multiplier)
        "tile_friction_scale": Range(0.7, 1.3, 1.0, "linear", "x nominal"),
        # D16 spawn pose
        "spawn_lateral_m": Range(-0.06, 0.06, 0.0, "linear", "m"),
        "spawn_heading_deg": Range(-25.0, 25.0, 0.0, "linear", "deg"),
        # D17 wheel effort limit (startup-only write; never per reset)
        "effort_limit_nm": Range(0.06, 0.25, 0.15, "linear", "N.m"),
        # D18 brake authority
        "brake_beta": Range(0.4, 1.0, 1.0, "linear", ""),
        # Actuator armature / joint friction: the S2 numbers and the stage-2 sysid targets
        "armature_scale": Range(0.5, 2.0, 1.0, "log", "x nominal"),
        "joint_friction_nm": Range(0.005, 0.03, 0.010, "linear", "N.m"),
    }


@dataclass
class DynamicsParams:
    """Per-env dynamics parameters.

    Scalars are ``(N,)`` tensors; per-wheel quantities are ``(N, 2)`` ordered ``[left, right]``;
    the CoM offset is ``(N, 3)``. ``delay_steps`` and ``obs_delay_steps`` are ``torch.long``.

    Attributes:
        wheel_radius_m: Per-wheel effective radius after D1 scale and asymmetry.
        baseline_m: Wheel baseline.
        friction_static: Per-wheel static friction coefficient.
        friction_kinetic: Per-wheel kinetic friction, always <= static.
        mass_kg: Base-body mass.
        izz_scale: Yaw-inertia multiplier.
        com_offset_m: Base-body CoM offset in the base frame.
        motor_gain: Motor gain.
        motor_trim: Motor trim (left/right asymmetry, applied as ``1 -/+ trim``).
        battery_sag: Battery output multiplier at episode start.
        battery_decay_per_s: Linear decay of ``battery_sag`` over the episode.
        deadband_duty: Per-wheel PWM dead-band; below it the wheel coasts.
        delay_steps: Integer actuation delay in control steps.
        delay_frac: Sub-step actuation delay in control steps.
        motor_lag_alpha: First-order motor-lag coefficient.
        obs_delay_steps: D9 camera-stream latency in control steps.
        control_jitter: Control-period multiplier.
        drag_u1: Longitudinal drag equivalent.
        drag_w1: Angular drag equivalent.
        slip_std: Per-step commanded-speed noise fraction.
        encoder_dropout_p: Encoder dropout probability.
        push_interval_s: Mean interval between external pushes.
        push_dv: Linear-velocity impulse of a push.
        push_dyaw: Yaw-rate impulse of a push.
        tile_friction_scale: Per-env floor-patch friction multiplier.
        spawn_lateral_m: Spawn lateral offset from the lane centerline.
        spawn_heading_deg: Spawn heading offset.
        effort_limit_nm: Wheel effort limit (startup write).
        brake_beta: Brake-authority slew factor.
        armature_scale: Joint-armature multiplier.
        joint_friction_nm: Joint dry friction.
    """

    wheel_radius_m: torch.Tensor
    baseline_m: torch.Tensor
    friction_static: torch.Tensor
    friction_kinetic: torch.Tensor
    mass_kg: torch.Tensor
    izz_scale: torch.Tensor
    com_offset_m: torch.Tensor
    motor_gain: torch.Tensor
    motor_trim: torch.Tensor
    battery_sag: torch.Tensor
    battery_decay_per_s: torch.Tensor
    deadband_duty: torch.Tensor
    delay_steps: torch.Tensor
    delay_frac: torch.Tensor
    motor_lag_alpha: torch.Tensor
    obs_delay_steps: torch.Tensor
    control_jitter: torch.Tensor
    drag_u1: torch.Tensor
    drag_w1: torch.Tensor
    slip_std: torch.Tensor
    encoder_dropout_p: torch.Tensor
    push_interval_s: torch.Tensor
    push_dv: torch.Tensor
    push_dyaw: torch.Tensor
    tile_friction_scale: torch.Tensor
    spawn_lateral_m: torch.Tensor
    spawn_heading_deg: torch.Tensor
    effort_limit_nm: torch.Tensor
    brake_beta: torch.Tensor
    armature_scale: torch.Tensor
    joint_friction_nm: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        """Return the parameters as a plain dict.

        Returns:
            ``{field_name: tensor}`` for every field.
        """
        return dict(self.__dict__)

    def index(self, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """Select a subset of envs.

        Args:
            env_ids: Long tensor of env indices.

        Returns:
            ``{field_name: tensor[env_ids]}``.
        """
        return {k: v[env_ids] for k, v in self.__dict__.items()}


@dataclass
class DynamicsDRCfg:
    """Configuration of the dynamics DR sampler.

    Attributes:
        ranges: Range book; defaults to :func:`default_dynamics_ranges`.
        per_wheel_friction: If True, D3 is sampled independently per wheel (a superset of the
            S7.3 row, which specifies one material). Left/right friction asymmetry is a real
            Duckiebot failure mode (worn tyre, dusty patch), so it is on by default.
    """

    ranges: dict[str, Range] = field(default_factory=default_dynamics_ranges)
    per_wheel_friction: bool = True


class DynamicsRandomizer:
    """Samples per-env dynamics parameters (SPEC v2 S7.3).

    The randomizer owns persistent ``(N, ...)`` tensors so that a partial reset
    (``resample(env_ids)``) only touches the envs that reset, exactly like the Isaac event
    manager does. All draws use one explicit generator, so a seeded run is reproducible.

    Args:
        num_envs: Number of parallel envs.
        cfg: Configuration; defaults to :class:`DynamicsDRCfg`.
        device: Torch device.
        generator: Torch generator (determinism).
    """

    def __init__(
        self,
        num_envs: int,
        cfg: DynamicsDRCfg | None = None,
        *,
        device: Any = None,
        generator: torch.Generator | None = None,
    ) -> None:
        self.num_envs = int(num_envs)
        self.cfg = cfg or DynamicsDRCfg()
        self.device = device
        self.generator = generator
        self._params = self._build(alpha=0.0)

    @property
    def params(self) -> DynamicsParams:
        """The current per-env parameters."""
        return self._params

    @property
    def ranges(self) -> RangeBook:
        """The range book in use."""
        return self.cfg.ranges

    def _draw(
        self,
        name: str,
        shape: int | tuple[int, ...],
        alpha: float | torch.Tensor,
        boundary: torch.Tensor | None,
    ) -> torch.Tensor:
        """Draw one axis.

        Args:
            name: Axis name in the range book.
            shape: Output shape.
            alpha: Curriculum scalar.
            boundary: Optional ADR boundary-probe mask (broadcast to ``shape``).

        Returns:
            The sampled tensor.
        """
        bnd = boundary
        if bnd is not None and not isinstance(shape, int):
            bnd = bnd.view(-1, *([1] * (len(shape) - 1))).expand(*shape)
        return self.cfg.ranges[name].sample(
            shape, alpha, generator=self.generator, device=self.device, boundary=bnd
        )

    def _build(
        self,
        alpha: float | torch.Tensor = 1.0,
        *,
        n: int | None = None,
        boundary: torch.Tensor | None = None,
    ) -> DynamicsParams:
        """Sample a full parameter set for ``n`` envs.

        Args:
            alpha: Curriculum scalar ``alpha_dyn``.
            n: Number of envs; defaults to ``num_envs``.
            boundary: Optional ADR boundary-probe mask of shape ``(n,)``.

        Returns:
            A freshly sampled :class:`DynamicsParams`.
        """
        n = self.num_envs if n is None else int(n)
        d = self._draw
        scale = d("wheel_radius_scale", n, alpha, boundary)
        asym = d("wheel_radius_asym", n, alpha, boundary)
        radius = NOMINAL_WHEEL_RADIUS_M * scale.view(n, 1) * torch.stack([1.0 + asym, 1.0 - asym], dim=1)
        fric_shape: int | tuple[int, ...] = (n, 2) if self.cfg.per_wheel_friction else n
        mu_s = d("friction_static", fric_shape, alpha, boundary)
        if isinstance(fric_shape, int):
            mu_s = mu_s.view(n, 1).expand(n, 2).contiguous()
        mu_ratio = d("friction_kinetic_ratio", (n, 2), alpha, boundary)
        com = torch.stack(
            [
                d("com_x_m", n, alpha, boundary),
                d("com_y_m", n, alpha, boundary),
                d("com_z_m", n, alpha, boundary),
            ],
            dim=1,
        )
        return DynamicsParams(
            wheel_radius_m=radius,
            baseline_m=d("baseline_m", n, alpha, boundary),
            friction_static=mu_s,
            friction_kinetic=mu_s * mu_ratio,
            mass_kg=d("mass_kg", n, alpha, boundary),
            izz_scale=d("izz_scale", n, alpha, boundary),
            com_offset_m=com,
            motor_gain=d("motor_gain", n, alpha, boundary),
            motor_trim=d("motor_trim", n, alpha, boundary),
            battery_sag=d("battery_sag", n, alpha, boundary),
            battery_decay_per_s=d("battery_decay_per_s", n, alpha, boundary),
            deadband_duty=d("deadband_duty", (n, 2), alpha, boundary),
            delay_steps=d("delay_steps", n, alpha, boundary),
            delay_frac=d("delay_frac", n, alpha, boundary),
            motor_lag_alpha=d("motor_lag_alpha", n, alpha, boundary),
            obs_delay_steps=d("obs_delay_steps", n, alpha, boundary),
            control_jitter=d("control_jitter", n, alpha, boundary),
            drag_u1=d("drag_u1", n, alpha, boundary),
            drag_w1=d("drag_w1", n, alpha, boundary),
            slip_std=d("slip_std", n, alpha, boundary),
            encoder_dropout_p=d("encoder_dropout_p", n, alpha, boundary),
            push_interval_s=d("push_interval_s", n, alpha, boundary),
            push_dv=d("push_dv", n, alpha, boundary),
            push_dyaw=d("push_dyaw", n, alpha, boundary),
            tile_friction_scale=d("tile_friction_scale", n, alpha, boundary),
            spawn_lateral_m=d("spawn_lateral_m", n, alpha, boundary),
            spawn_heading_deg=d("spawn_heading_deg", n, alpha, boundary),
            effort_limit_nm=d("effort_limit_nm", n, alpha, boundary),
            brake_beta=d("brake_beta", n, alpha, boundary),
            armature_scale=d("armature_scale", n, alpha, boundary),
            joint_friction_nm=d("joint_friction_nm", n, alpha, boundary),
        )

    def resample(
        self,
        env_ids: torch.Tensor | None = None,
        alpha: float | torch.Tensor = 1.0,
        *,
        boundary: torch.Tensor | None = None,
    ) -> DynamicsParams:
        """Resample the parameters of the given envs in place (called on episode reset).

        Args:
            env_ids: Long tensor of env indices, or ``None`` for all envs.
            alpha: Curriculum scalar ``alpha_dyn``.
            boundary: Optional ADR boundary-probe mask, shaped like ``env_ids``.

        Returns:
            The updated :class:`DynamicsParams` (the same object every call).
        """
        if env_ids is None:
            self._params = self._build(alpha, boundary=boundary)
            return self._params
        ids = env_ids.to(torch.long)
        fresh = self._build(alpha, n=int(ids.numel()), boundary=boundary)
        cur = self._params.__dict__
        for k, v in fresh.__dict__.items():
            cur[k][ids] = v.to(cur[k].dtype)
        return self._params

    def sample_push(
        self,
        env_ids: torch.Tensor | None = None,
        alpha: float | torch.Tensor = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a fresh D14 external push.

        The push interval itself is a per-env parameter (``params.push_interval_s``); this draws
        the impulse applied when that interval elapses.

        Args:
            env_ids: Envs being pushed, or ``None`` for all.
            alpha: Curriculum scalar ``alpha_dyn``.

        Returns:
            ``(dv, dyaw)``, both ``(len(env_ids),)`` tensors.
        """
        n = self.num_envs if env_ids is None else int(env_ids.numel())
        return (
            self._draw("push_dv", n, alpha, None),
            self._draw("push_dyaw", n, alpha, None),
        )

    def spawn_pose(
        self,
        env_ids: torch.Tensor | None = None,
        alpha: float | torch.Tensor = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample the D16 spawn perturbation.

        Args:
            env_ids: Envs being spawned, or ``None`` for all.
            alpha: Curriculum scalar ``alpha_dyn``.

        Returns:
            ``(lateral_m, heading_rad)``, both ``(len(env_ids),)`` tensors.
        """
        n = self.num_envs if env_ids is None else int(env_ids.numel())
        lateral = self._draw("spawn_lateral_m", n, alpha, None)
        heading = self._draw("spawn_heading_deg", n, alpha, None) * (math.pi / 180.0)
        return lateral, heading

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Serialize the current parameter tensors.

        Returns:
            ``{field_name: tensor}``.
        """
        return self._params.as_dict()

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Restore parameter tensors produced by :meth:`state_dict`.

        Args:
            state: The dict returned by :meth:`state_dict`.

        Raises:
            KeyError: If a field is missing.
        """
        cur = self._params.__dict__
        for k in cur:
            if k not in state:
                raise KeyError(f"dynamics DR state is missing field {k!r}")
            cur[k][...] = state[k].to(cur[k].dtype)


def quantize_encoder(
    wheel_speed: torch.Tensor,
    dt: float,
    dropout_p: torch.Tensor,
    *,
    tick_noise: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Apply the D13 encoder model to a wheel-speed observation.

    The real encoder reports whole ticks accumulated over the control period, so the observed
    speed is quantized to ``2*pi / (ticks_per_rev * dt)`` rad/s, with +/- 1 tick of noise and an
    occasional dropped reading (repeat of nothing -> zero).

    Args:
        wheel_speed: ``(N, 2)`` true wheel speeds in rad/s.
        dt: Control period in seconds.
        dropout_p: ``(N,)`` dropout probability per env.
        tick_noise: Standard deviation of the additive tick noise, in ticks.
        generator: Torch generator (determinism).

    Returns:
        The quantized, noisy ``(N, 2)`` observation.
    """
    quantum = 2.0 * math.pi / (ENCODER_TICKS_PER_REV * dt)
    ticks = torch.round(wheel_speed / quantum)
    if tick_noise > 0.0:
        noise = torch.randn(ticks.shape, generator=generator, device=ticks.device, dtype=ticks.dtype)
        ticks = torch.round(ticks + noise * tick_noise)
    out = ticks * quantum
    drop = torch.rand(out.shape, generator=generator, device=out.device, dtype=out.dtype)
    keep = drop >= dropout_p.to(out.device, out.dtype).view(-1, 1)
    return out * keep.to(out.dtype)
