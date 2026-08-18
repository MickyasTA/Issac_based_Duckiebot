"""MuJoCo Duckiebot environment with the same contract as the Isaac Lab task (SPEC v2 S5, S8.3).

Parity is the entire point, so the things that must be identical are identical *by construction*
rather than by inspection:

======================  ==============================================================================
Quantity                How parity is enforced here
======================  ==============================================================================
Robot geometry / mass   :mod:`duckiebot_rl.sim2sim.mjcf` reads ``duckiebot_rl.assets.params``
Physics rate            ``sim_dt = 1/240`` and ``decimation = 16`` come from the same params object
Control rate            derived, asserted to be an exact integer division (S8.3 item 3)
Action path             :class:`ActionPath` implements S5.3 steps 1-6 verbatim, in numpy
Observation chain       imported from the shared S4.3 module, never re-implemented
Camera pose             :func:`duckiebot_rl.sim2sim._resolve.mj_camera_xyaxes`, one conversion
Reward / terminations   S5.4 and S5.5, evaluated against the same lane-graph quantities
======================  ==============================================================================

The physics-rate confound the critic flagged (item J: Isaac ran 1/120 with decimation 8 while
MuJoCo ran 1/240 with decimation 16, a 2x integration-rate difference silently folded into the
headline C1-vs-C5 delta) is resolved by SPEC v2 S5.2: both simulators now run 1/240 with decimation
16. :class:`MjEnvCfg` still exposes ``physics_dt`` and ``decimation``, so the confound can be
re-introduced *as a named ablation* (``MjEnvCfg(physics_dt=1/120, decimation=8)``); doing so emits a
warning and sets ``info["rate_confound"]`` on every step, so no report can quote such a run as C5
without the flag showing up in its metadata.

What condition C6 randomizes, and what it does not
--------------------------------------------------

``MjEnvCfg(dynamics_dr=True)`` is the S8.4 condition C6. It is a *subset* of the S7.3 axes, and the
subset is stated here rather than left to be discovered by reading ``_apply_body_dr``:

=====================================  ======================================================
Applied                                Where
=====================================  ======================================================
D3 tyre friction (static and kinetic)  ``_apply_body_dr`` -> ``geom_friction``
D4 base mass and yaw inertia           ``_apply_body_dr`` -> ``body_mass``, ``body_inertia``
D5 centre of mass                      ``_apply_body_dr`` -> ``body_ipos``
D17 effort limit                       ``_apply_body_dr`` -> ``actuator_forcerange``
armature, joint friction (S2 Drive)    ``_apply_body_dr`` -> ``dof_armature``, ``dof_frictionloss``
D6-D8, D12, D13, D18                   :class:`ActionPath`, per control step
D16 spawn pose                         ``LaneGraph.sample_spawn``
V14-V19 photometric                    the shared preprocess chain (``photometric_dr=True``)
=====================================  ======================================================

Not applied, and each for a structural reason rather than an oversight: the scene-graph visual axes
V1-V8 and V13 have no MuJoCo counterpart; D10 control-period jitter is excluded because S8.3 item 3
locks the decimation to an exact integer division in both simulators; and D11 aerodynamic drag, D14
external pushes and D15 per-tile friction patches are not modelled by this harness. The same list
appears in ``CONDITIONS['C6'].description`` so it reaches the evaluation report as well.

The armature and joint-friction axes matter beyond coverage bookkeeping: they are the two
parameters stage 2 of the S8.2 identification fits, and S8.2 acceptance requires the identified
delta to be COVERED by these ranges. If C6 did not randomize them, that acceptance check would
couple to nothing.

Observation modes
-----------------
``"rgb_vec"``
    The real thing: renders 192x128, runs the shared S4.3 chain, returns the stacked
    ``(48, 96, 9)`` uint8 image plus the 8-vector. Needs the shared preprocess module, but *not*
    torch: that module ships a numpy implementation of the chain and a numpy ``FrameStack``, so the
    whole vision path runs in the tools venv as it stands.
``"vec"``
    Proprioception only. Runs anywhere, used by system identification and by the M3/M5-equivalent
    state-based checks.
``"none"``
    No observation assembly at all. Used by the kinematics parity tests.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import _resolve
from . import mjcf as _mjcf
from ._resolve import (
    DRRanges,
    ObsParams,
    RobotParams,
    SharedModuleUnavailable,
    SimParams,
    resolve_dr_ranges,
    resolve_preprocess,
    resolve_robot_params,
    resolve_sim_params,
)
from .track import LOOP_5X5, LaneGraph, MapSpec, TrackScene, build_track

__all__ = [
    "NO_OBSTACLE_DISTANCE",
    "TERMINAL_PENALIZED",
    "TERMINAL_PENALTY",
    "ActionPath",
    "EpisodeMetrics",
    "MjDuckiebotEnv",
    "MjEnvCfg",
]

_ISAAC_REFERENCE_RATE = (1.0 / 240.0, 16)
_DEG30 = math.radians(30.0)

#: Distance reported for ``dist_nearest_obstacle`` when the scene contains no obstacle at all.
#: Far outside any map, so the critic reads it as "nothing anywhere near", which 0.0 did not.
NO_OBSTACLE_DISTANCE = 100.0

#: SPEC v2 S5.4 terminal reward, added to the per-step sum before the [-20, +20] clip.
TERMINAL_PENALTY = -10.0
#: The S5.5 termination reasons S5.4 prices. Truncation is bootstrapped and scores 0.
TERMINAL_PENALIZED = frozenset({"off_drivable", "obstacle"})


@dataclass
class MjEnvCfg:
    """Configuration for :class:`MjDuckiebotEnv`.

    Attributes:
        map: map source accepted by :func:`duckiebot_rl.sim2sim.track.load_map`.
        asset_dir: where tile textures are written.
        physics_dt: MuJoCo integration step. Changing it away from 1/240 re-introduces critic item
            J's integration-rate confound and is flagged on every step.
        decimation: physics steps per control step. Must divide the control period exactly.
        episode_length_s: truncation horizon. SPEC v2 uses 30 s for training and a 45 s cap for the
            S8.4 evaluation conditions.
        obs_mode: ``"rgb_vec"``, ``"vec"`` or ``"none"``.
        dynamics_dr: apply the S7.3 dynamics randomization on reset (condition C6).
        photometric_dr: apply the shared layer-1 photometric randomization inside preprocess
            (condition C6); ignored unless ``obs_mode`` is ``"rgb_vec"``.
        dr_alpha: curriculum scalar in ``[0, 1]`` scaling every DR range around its nominal.
        obstacles: obstacles to place beyond the ones the map itself declares.
        walls: emit the visual-only perimeter walls.
        stall_speed: body speed below which the robot counts as stalled, in m/s.
        stall_timeout_s: stall duration that terminates the episode.
        obstacle_margin: the S5.5 geometric safety radius added to each obstacle radius.
        seed: RNG seed. ``reset(seed=...)`` overrides it.
        texture_provider: appearance provider override; None resolves the shared city generator.
    """

    map: Any = field(default_factory=lambda: LOOP_5X5)
    asset_dir: str = "."
    physics_dt: float | None = None
    decimation: int | None = None
    episode_length_s: float = 30.0
    obs_mode: str = "vec"
    dynamics_dr: bool = False
    photometric_dr: bool = False
    dr_alpha: float = 1.0
    obstacles: Sequence[Any] = ()
    walls: bool = True
    stall_speed: float = 0.03
    stall_timeout_s: float = 2.0
    obstacle_margin: float = 0.12
    seed: int = 0
    texture_provider: Any = None


# --------------------------------------------------------------------------------- action path
@dataclass
class _EpisodeDR:
    """Per-episode dynamics-randomization sample (SPEC v2 S7.3)."""

    radius_left: float
    radius_right: float
    baseline: float
    gain: float
    trim: float
    motor_k: float
    deadband_left: float
    deadband_right: float
    delay_steps: int
    delay_substep: float
    lag_alpha: float
    brake_beta: float
    battery_sag: float
    slip_frac: float
    obs_latency: int
    encoder_dropout: float


class ActionPath:
    """SPEC v2 S5.3 action path, in numpy, one control step at a time.

    This is the torch-free twin of the Isaac env's action path. It is written to match the spec text
    literally, including step 4's asymmetric ``max`` form, because a "cleverer" symmetric brake here
    would be a silent divergence from the trained-against dynamics. Any change to this class must be
    made simultaneously in the ``[env]``-owned Isaac implementation and covered by
    ``tests/unit/test_action_path.py``.

    Attributes:
        robot: the shared robot parameters.
        dr: the per-episode randomization sample currently in force.
    """

    def __init__(self, robot: RobotParams, control_dt: float) -> None:
        """Create an action path.

        Args:
            robot: shared robot parameters.
            control_dt: control period in seconds.
        """
        self.robot = robot
        self.control_dt = float(control_dt)
        self.dr = self.nominal(robot)
        self._queue: list[np.ndarray] = []
        self._lagged = np.zeros(2, dtype=np.float64)

    def nominal(self, robot: RobotParams | None = None) -> _EpisodeDR:
        """Return the zero-randomization sample (every axis at its nominal value).

        The whole-step delay is ``actuation_delay_s / control_dt`` with **this instance's** control
        period, never a literal: the module exists to keep one copy of every rate, and a second
        hardcoded 1/15 s would round to the same 2 steps today and silently go wrong the moment the
        decimation or the physics step changes.

        Args:
            robot: robot parameters; defaults to this instance's.

        Returns:
            The nominal :class:`_EpisodeDR`.
        """
        robot = robot if robot is not None else self.robot
        return _EpisodeDR(
            radius_left=robot.wheel_radius,
            radius_right=robot.wheel_radius,
            baseline=robot.wheel_separation,
            gain=1.0,
            trim=0.0,
            motor_k=robot.motor_k,
            deadband_left=0.0,
            deadband_right=0.0,
            delay_steps=round(robot.actuation_delay_s / max(1e-9, self.control_dt)),
            delay_substep=0.0,
            lag_alpha=1.0,
            brake_beta=1.0,
            battery_sag=1.0,
            slip_frac=0.0,
            obs_latency=0,
            encoder_dropout=0.0,
        )

    def sample(self, rng: np.random.Generator, ranges: DRRanges, alpha: float) -> _EpisodeDR:
        """Draw a per-episode DR sample, interpolated from nominal by ``alpha``.

        Args:
            rng: numpy generator.
            ranges: the S7.3 clamps.
            alpha: curriculum scalar; 0 gives the nominal, 1 gives the full range.

        Returns:
            The sampled :class:`_EpisodeDR`.
        """
        robot = self.robot

        def band(bounds: tuple[float, float], nominal: float) -> float:
            lo = nominal + alpha * (bounds[0] - nominal)
            hi = nominal + alpha * (bounds[1] - nominal)
            return float(rng.uniform(min(lo, hi), max(lo, hi)))

        radius_scale = band(ranges.dr_wheel_radius_scale, 1.0)
        asymmetry = band(ranges.dr_wheel_radius_asymmetry, 0.0)
        base_radius = robot.wheel_radius * radius_scale
        delay_lo, delay_hi = ranges.dr_delay_control_steps
        nominal_delay = robot.actuation_delay_s / self.control_dt
        delay = round(band((float(delay_lo), float(delay_hi)), nominal_delay))
        latency_lo, latency_hi = ranges.dr_obs_latency_steps
        latency = round(band((float(latency_lo), float(latency_hi)), 0.0))
        self.dr = _EpisodeDR(
            radius_left=base_radius * (1.0 + asymmetry),
            radius_right=base_radius * (1.0 - asymmetry),
            baseline=band(ranges.dr_baseline_m, robot.wheel_separation),
            gain=band(ranges.dr_motor_gain, 1.0),
            trim=band(ranges.dr_motor_trim, 0.0),
            motor_k=robot.motor_k,
            deadband_left=band(ranges.dr_deadband_duty, 0.0),
            deadband_right=band(ranges.dr_deadband_duty, 0.0),
            delay_steps=max(0, delay),
            delay_substep=band(ranges.dr_delay_substep, 0.0),
            lag_alpha=band(ranges.dr_motor_lag_alpha, 1.0),
            brake_beta=band(ranges.dr_brake_authority_beta, 1.0),
            battery_sag=band(ranges.dr_battery_sag, 1.0),
            slip_frac=band(ranges.dr_wheel_slip_frac, 0.0),
            obs_latency=max(0, latency),
            encoder_dropout=band(ranges.dr_encoder_dropout, 0.0),
        )
        # The ring depth follows delay_steps, so a sample drawn after reset() has to re-warm it.
        self.reset()
        return self.dr

    def reset(self) -> None:
        """Clear the motor-lag state and pre-fill the delay ring with zero targets.

        The ring is filled rather than emptied on purpose. With an empty ring the index
        ``len(queue) - 1 - delay_steps`` clamps to the newest entry, so the first ``delay_steps``
        control steps of every episode would bypass the S5.3 actuation delay entirely and the robot
        would answer the very first command at full commanded wheel speed. S2 calls the 0.150 s
        actuation delay the dominant dynamics gap, and that divergence would land on the first two
        control steps of every reset, on every episode of every condition, and on every open-loop
        system-identification program. Pre-filling makes step 0 behave exactly like step 100 of a
        run that has been commanding zero.
        """
        depth = self.dr.delay_steps + 2
        self._queue = [np.zeros(2, dtype=np.float64) for _ in range(depth)]
        self._lagged = np.zeros(2, dtype=np.float64)

    def __call__(
        self, action: np.ndarray, wheel_velocity: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Map a policy action to a pair of wheel-velocity targets.

        Args:
            action: the raw policy action; clipped to ``[-1, 1]`` here, as the Isaac env does.
            wheel_velocity: current measured ``(left, right)`` wheel speeds in rad/s.
            rng: generator for the per-step slip noise.

        Returns:
            The ``(left, right)`` wheel-velocity targets in rad/s.
        """
        dr, robot = self.dr, self.robot
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        v_cmd = 0.5 * robot.v_max * (a[0] + 1.0)
        om_cmd = robot.omega_max * a[1]

        # 1. inverse kinematics with randomized gain, trim, baseline and per-wheel radius
        half_turn = 0.5 * om_cmd * dr.baseline
        target = np.array(
            [
                (v_cmd - half_turn) / dr.radius_left * dr.gain * (1.0 - dr.trim),
                (v_cmd + half_turn) / dr.radius_right * dr.gain * (1.0 + dr.trim),
            ]
        )

        # 2. actuation delay D8: whole-step ring plus a sub-step linear interpolation, then a
        #    first-order motor lag.
        self._queue.append(target.copy())
        depth = dr.delay_steps + 2
        if len(self._queue) > depth:
            del self._queue[0 : len(self._queue) - depth]
        newer = self._queue[max(0, len(self._queue) - 1 - dr.delay_steps)]
        older = self._queue[max(0, len(self._queue) - 2 - dr.delay_steps)]
        delayed = older + (1.0 - dr.delay_substep) * (newer - older)
        self._lagged = self._lagged + dr.lag_alpha * (delayed - self._lagged)
        target = self._lagged.copy()

        # 3. dead-band D7: below the release duty the hardware COASTS. Commanding the current wheel
        #    speed makes the velocity servo produce ~zero torque, which is a coast, not a brake.
        duty = target / max(1e-9, dr.motor_k)
        deadband = np.array([dr.deadband_left, dr.deadband_right])
        coasting = np.abs(duty) < np.maximum(deadband, robot.pwm_release_duty)
        target = np.where(coasting, wheel_velocity, target)

        # 4. brake authority D18: back-EMF braking is partial, so bound how fast a target may fall.
        #    The spec's asymmetric max() form is reproduced verbatim; see the class docstring.
        floor = wheel_velocity - dr.brake_beta * robot.brake_dw_max
        target = np.maximum(target, floor)

        # 5. wheel slip D12 and battery sag D6 scale the realized target
        slip = 1.0 + rng.uniform(-dr.slip_frac, dr.slip_frac, size=2)
        target = target * dr.battery_sag * slip

        return np.clip(target, -robot.velocity_limit, robot.velocity_limit)


# ------------------------------------------------------------------------------------- metrics
@dataclass
class EpisodeMetrics:
    """Per-episode statistics for the SPEC v2 S8.4 evaluation protocol.

    Attributes:
        steps: number of control steps executed.
        duration_s: survival time in seconds.
        distance_m: lane-frame consecutive distance, the AI-DO headline metric.
        laps: fractional laps, or NaN when the map has no closed lane cycle.
        lane_rms_m: root-mean-square lateral error.
        lane_max_m: maximum absolute lateral error.
        lane_abs_integral_ms: time-integrated ``|d|`` in metre-seconds (the corner-cut watch, S5.4).
        out_of_lane_integral_ms: time-integrated out-of-lane excursion in metre-seconds.
        p_far: fraction of steps with ``|d| > 0.6 * w_lane / 2``.
        collisions: number of safety-circle violations.
        success: True when the episode ran to truncation without terminating.
        reason: failure reason, or ``"timeout"``.
        return_: undiscounted sum of the S5.4 reward.
    """

    steps: int = 0
    duration_s: float = 0.0
    distance_m: float = 0.0
    laps: float = float("nan")
    lane_rms_m: float = float("nan")
    lane_max_m: float = float("nan")
    lane_abs_integral_ms: float = 0.0
    out_of_lane_integral_ms: float = 0.0
    p_far: float = float("nan")
    collisions: int = 0
    success: bool = False
    reason: str = "timeout"
    return_: float = 0.0

    def as_dict(self) -> dict[str, float | int | str | bool]:
        """Return the metrics as a plain dict (the field ``return_`` is renamed ``return``)."""
        out: dict[str, float | int | str | bool] = {}
        for key, value in self.__dict__.items():
            out["return" if key == "return_" else key] = value
        return out


# ----------------------------------------------------------------------------------- the env
class MjDuckiebotEnv:
    """Single-robot MuJoCo lane-following environment.

    Attributes:
        cfg: the configuration.
        scene: the generated track scene.
        model: the compiled ``mujoco.MjModel``.
        data: the ``mujoco.MjData``.
        metrics: metrics for the episode in progress.
    """

    def __init__(self, cfg: MjEnvCfg | None = None, scene: TrackScene | None = None) -> None:
        """Build the environment.

        Args:
            cfg: configuration; defaults are the SPEC v2 training settings.
            scene: a pre-built track, to avoid regenerating textures across parallel workers.

        Raises:
            ImportError: if ``mujoco`` is not importable in this interpreter.
            ValueError: if the requested rates do not produce an exact integer decimation.
        """
        try:
            import mujoco
        except ImportError as exc:  # pragma: no cover - depends on the interpreter
            raise ImportError(
                "mujoco is required by duckiebot_rl.sim2sim.env. Use the tools venv: "
                "d:/Personal/personal/mujoco_venv/Scripts/python.exe"
            ) from exc
        self._mj = mujoco
        self.cfg = cfg if cfg is not None else MjEnvCfg()

        self.robot, self._robot_source = resolve_robot_params()
        sim, self._sim_source = resolve_sim_params()
        self.sim = self._apply_rate_overrides(sim)
        self.obs_params = ObsParams()
        self.dr_ranges, self._dr_source = resolve_dr_ranges()

        mjcf_cfg = _mjcf.MjcfCfg.from_shared(
            sim=self.sim,
            obs=self.obs_params,
            include_camera=self.cfg.obs_mode == "rgb_vec",
        )
        self.scene = (
            scene
            if scene is not None
            else build_track(
                self.cfg.map,
                cfg=mjcf_cfg,
                asset_dir=self.cfg.asset_dir,
                texture_provider=self.cfg.texture_provider,
                obstacles=self.cfg.obstacles,
                walls=self.cfg.walls,
            )
        )
        self.lane: LaneGraph = self.scene.lane
        self.map: MapSpec = self.scene.map

        self.model = mujoco.MjModel.from_xml_string(self.scene.xml)
        self.model.opt.timestep = self.sim.physics_dt
        self.data = mujoco.MjData(self.model)

        self._name2id = mujoco.mj_name2id
        self._obj = mujoco.mjtObj
        self._base_body = self._id(self._obj.mjOBJ_BODY, self.robot.base_link_name)
        self._wheel_dofs = [
            self.model.jnt_dofadr[self._id(self._obj.mjOBJ_JOINT, name)]
            for name in (self.robot.left_wheel_joint_name, self.robot.right_wheel_joint_name)
        ]
        self._camera_id = self._id(self._obj.mjOBJ_CAMERA, _mjcf.CAMERA_NAME, required=False)

        self.action_path = ActionPath(self.robot, self.sim.control_dt)
        self.rng = np.random.default_rng(self.cfg.seed)
        self._renderer: Any = None
        self._preprocess: Any = None
        self._stack: Any = None
        self._visual_dr: tuple[Any, Any] | None = None
        self._prev_actions = np.zeros((2, 2), dtype=np.float32)
        self._prev_wheel_pos = np.zeros(2, dtype=np.float64)
        self._encoder_speed = np.zeros(2, dtype=np.float64)
        self.metrics = EpisodeMetrics()
        self._mobile_warned = False
        self._reset_episode_state()

        if self.cfg.obs_mode == "rgb_vec":
            self._preprocess = resolve_preprocess()
            self._check_preprocess_constants()
            # The shared FrameStack owns S4.3 steps 9 and 10 (the depth-5 ring, the D9 observation
            # delay and the (t, t-2, t-4) stack). Re-implementing it here would be a second copy of
            # the exact logic whose rollout-boundary bug SPEC v2 item 4 exists to kill.
            self._stack = self._preprocess.FrameStack(
                num_envs=1,
                obs_hw=(self._preprocess.OBS_H, self._preprocess.OBS_W),
                offsets=tuple(self._preprocess.FRAME_STACK_OFFSETS),
                max_obs_delay=int(self._preprocess.MAX_OBS_DELAY),
                backend="numpy",
            )
            self._renderer = mujoco.Renderer(self.model, self._preprocess.RENDER_H, self._preprocess.RENDER_W)

    # -- construction helpers ------------------------------------------------------------------
    def _apply_rate_overrides(self, sim: SimParams) -> SimParams:
        """Apply the cfg rate overrides and warn when they re-introduce the item-J confound."""
        import dataclasses

        changes: dict[str, Any] = {}
        if self.cfg.physics_dt is not None:
            changes["physics_dt"] = float(self.cfg.physics_dt)
        if self.cfg.decimation is not None:
            changes["decimation"] = int(self.cfg.decimation)
        out = dataclasses.replace(sim, **changes) if changes else sim
        out.validate()
        control_hz = out.control_hz
        if abs(control_hz - round(control_hz)) > 1e-9:
            raise ValueError(
                f"physics_dt {out.physics_dt} and decimation {out.decimation} give a control rate "
                f"of {control_hz} Hz, which is not an integer; SPEC v2 S8.3 item 3 requires an "
                f"exact integer division in both simulators"
            )
        self.rate_confound = (out.physics_dt, out.decimation) != _ISAAC_REFERENCE_RATE
        if self.rate_confound:
            warnings.warn(
                f"MuJoCo is running physics_dt={out.physics_dt} decimation={out.decimation}, not "
                f"the Isaac reference {_ISAAC_REFERENCE_RATE}. This is critic item J's "
                f"integration-rate confound; every step will carry info['rate_confound']=True and "
                f"no result from this env may be reported as a plain C5 or C6.",
                RuntimeWarning,
                stacklevel=3,
            )
        return out

    def _id(self, objtype: Any, name: str, required: bool = True) -> int:
        """Look up a MuJoCo name, raising a useful error when it is absent.

        Args:
            objtype: the ``mjtObj`` enum member.
            name: the element name.
            required: raise instead of returning -1 when the name is missing.

        Returns:
            The element id, or -1 when ``required`` is False and the name is absent.

        Raises:
            KeyError: if the name is missing and ``required``.
        """
        index = self._name2id(self.model, objtype, name)
        if index < 0 and required:
            raise KeyError(f"MuJoCo model has no {objtype} named {name!r}")
        return int(index)

    # -- state accessors -----------------------------------------------------------------------
    @property
    def control_dt(self) -> float:
        """Control period in seconds."""
        return self.sim.control_dt

    def pose(self) -> tuple[float, float, float]:
        """Return the robot's world ``(x, y, yaw)``."""
        q = self.data.qpos
        yaw = math.atan2(2.0 * (q[3] * q[6] + q[4] * q[5]), 1.0 - 2.0 * (q[5] ** 2 + q[6] ** 2))
        return float(q[0]), float(q[1]), yaw

    def roll_pitch(self) -> tuple[float, float]:
        """Return the robot's roll and pitch in radians."""
        w, x, y, z = (float(v) for v in self.data.qpos[3:7])
        roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        sin_pitch = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
        return roll, math.asin(sin_pitch)

    def wheel_velocity(self) -> np.ndarray:
        """Return the measured ``(left, right)`` wheel speeds in rad/s."""
        return np.array([self.data.qvel[d] for d in self._wheel_dofs], dtype=np.float64)

    def body_speed(self) -> float:
        """Return the forward body-frame speed in m/s."""
        _x, _y, yaw = self.pose()
        c, s = math.cos(yaw), math.sin(yaw)
        return float(c * self.data.qvel[0] + s * self.data.qvel[1])

    # -- episode lifecycle ---------------------------------------------------------------------
    def _reset_episode_state(self) -> None:
        """Clear all per-episode accumulators."""
        self.metrics = EpisodeMetrics()
        self._step_count = 0
        self._lateral: list[float] = []
        self._yaw_integral = 0.0
        self._start_xy = np.zeros(2)
        self._stall_steps = 0
        self._prev_gap = 0.0
        self._loop_length = float("nan")
        self._prev_actions[:] = 0.0

    def reset(
        self, seed: int | None = None, pose: tuple[float, float, float] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset the episode.

        Args:
            seed: reseed the environment RNG before sampling the spawn pose and the DR sample.
            pose: explicit ``(x, y, yaw)`` spawn; None samples the S7.3 D16 distribution.

        Returns:
            ``(observation, info)``.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._mj.mj_resetData(self.model, self.data)
        self._reset_episode_state()
        # The DR sample has to be in force BEFORE the delay ring is warmed, because the ring depth
        # follows delay_steps. ActionPath.sample() re-warms it itself; the nominal branch does so
        # explicitly here.
        if self.cfg.dynamics_dr:
            self.action_path.sample(self.rng, self.dr_ranges, self.cfg.dr_alpha)
            self._apply_body_dr()
        else:
            self.action_path.dr = self.action_path.nominal(self.robot)
            self.action_path.reset()

        x, y, yaw = pose if pose is not None else self.lane.sample_spawn(self.rng)
        self.data.qpos[0:2] = (x, y)
        self.data.qpos[2] = self.robot.base_height + 1e-6
        self.data.qpos[3:7] = (math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw))
        self._mj.mj_forward(self.model, self.data)

        self._start_xy = np.array([x, y])
        self._prev_wheel_pos = self._wheel_positions()
        self._encoder_speed[:] = 0.0
        query = self.lane.query(x, y, yaw)  # free global match: this is the spawn
        self._prev_match = (query.segment, query.s)
        self._loop_length = self.lane.cycle_length(query.segment)
        self._prev_gap = self._nearest_obstacle_gap(x, y)
        if self._stack is not None:
            self._stack.set_obs_delay(np.array([self.action_path.dr.obs_latency], dtype=np.int64))
        observation = self._observation(first=True)
        return observation, {"spawn": (x, y, yaw), "rate_confound": self.rate_confound}

    def _apply_body_dr(self) -> None:
        """Write every model-side S7.3 axis MuJoCo can represent into the compiled model.

        Covered here: D4 base mass and yaw inertia, D5 centre of mass, D3 tyre friction (static and
        the kinetic scale MuJoCo carries as the torsional/rolling pair), D17 effort limit, and the
        drive-train pair the system identification actually moves, joint armature and joint
        friction. The action-path axes (D6 to D8, D12, D13, D18) are applied per step by
        :class:`ActionPath` instead.

        Not representable here, and therefore a declared C6 coverage gap rather than a silent
        omission: D10 control-period jitter (the harness runs a fixed decimation by S8.3 item 3),
        D11 aerodynamic drag, D14 external pushes and D15 per-tile friction patches. They are
        listed in the module docstring and in ``CONDITIONS['C6'].description`` next to the visual
        axes V1-V8 and V13.
        """
        ranges, robot, rng = self.dr_ranges, self.robot, self.rng
        alpha = self.cfg.dr_alpha

        def band(bounds: tuple[float, float], nominal: float) -> float:
            lo = nominal + alpha * (bounds[0] - nominal)
            hi = nominal + alpha * (bounds[1] - nominal)
            return float(rng.uniform(min(lo, hi), max(lo, hi)))

        self.model.body_mass[self._base_body] = band(ranges.dr_base_mass_kg, robot.base_mass)
        com = np.array(robot.base_com, dtype=np.float64)
        com[0] += band(ranges.dr_com_xy_m, 0.0)
        com[1] += band(ranges.dr_com_xy_m, 0.0)
        com[2] += band(ranges.dr_com_z_m, 0.0)
        self.model.body_ipos[self._base_body] = com

        # D4's yaw-inertia term. Only Izz is randomized: it is the one that sets how hard the base
        # resists the differential-drive turn, and it is the term the S8.2 arc residuals are
        # sensitive to.
        inertia = np.array(robot.base_inertia_diag, dtype=np.float64)
        inertia[2] *= band(ranges.dr_base_inertia_zz_scale, 1.0)
        self.model.body_inertia[self._base_body] = inertia

        mu = band(ranges.dr_tire_friction_static, robot.wheel_friction)
        kinetic = mu * band(ranges.dr_tire_friction_dynamic_scale, 1.0)
        for link in (robot.left_wheel_link_name, robot.right_wheel_link_name):
            geom = self._id(self._obj.mjOBJ_GEOM, f"{link}_collision")
            # MuJoCo's geom_friction is (sliding, torsional, rolling). The sliding coefficient is
            # the static one; the kinetic scale D3 is carried on the torsional/rolling pair, which
            # is what resists a wheel that is already sliding.
            self.model.geom_friction[geom, 0] = mu
            self.model.geom_friction[geom, 1] = 0.005 * kinetic
            self.model.geom_friction[geom, 2] = 1.0e-4 * kinetic

        # The drive train. armature and frictionloss are the two parameters stage 2 of the system
        # identification fits, and S8.2 requires the identified delta to be COVERED by these
        # ranges, so C6 has to actually randomize them or that acceptance check couples to nothing.
        armature = robot.joint_armature * band(ranges.dr_armature_scale, 1.0)
        friction = band(ranges.dr_joint_friction_nm, robot.joint_friction)
        effort = band(ranges.dr_effort_limit_nm, robot.effort_limit)
        for name in (robot.left_wheel_joint_name, robot.right_wheel_joint_name):
            joint = self._id(self._obj.mjOBJ_JOINT, name)
            dof = int(self.model.jnt_dofadr[joint])
            self.model.dof_armature[dof] = armature
            self.model.dof_frictionloss[dof] = friction
        for index in range(self.model.nu):
            self.model.actuator_forcerange[index] = (-effort, effort)

    # -- stepping ------------------------------------------------------------------------------
    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Advance one control step.

        Args:
            action: the policy action in ``[-1, 1]^2``.

        Returns:
            ``(observation, reward, terminated, truncated, info)``.
        """
        action = np.asarray(action, dtype=np.float32).reshape(2)
        before = np.array(self.pose()[:2])

        target = self.action_path(action, self.wheel_velocity(), self.rng)
        self.data.ctrl[:] = target
        self._drive_mobile_obstacles()
        for _ in range(self.sim.decimation):
            self._mj.mj_step(self.model, self.data)

        self._step_count += 1
        self._update_encoders()
        x, y, yaw = self.pose()
        query = self.lane.query(x, y, yaw, prev_match=getattr(self, "_prev_match", None))
        self._prev_match = (query.segment, query.s)

        after = np.array([x, y])
        ds = float((after - before) @ np.asarray(query.tangent))
        gap = self._nearest_obstacle_gap(x, y)
        step_reward = self._reward(action, query, ds, gap)
        terminated, reason = self._terminations(x, y, yaw, gap)
        truncated = (not terminated) and (
            self._step_count >= round(self.cfg.episode_length_s / self.control_dt)
        )
        # S5.4: R_terminal = -10 on collision or off-drivable, 0 on truncation (which is
        # bootstrapped, not a failure). The clip is applied to R *after* the terminal penalty,
        # because the spec clips R, not the per-step sum. Rollover, stall and spin are S5.5
        # terminations that S5.4 does not price, so they carry no penalty here either.
        terminal = TERMINAL_PENALTY if (terminated and reason in TERMINAL_PENALIZED) else 0.0
        reward = float(np.clip(step_reward + terminal, -20.0, 20.0))

        self._accumulate(query, ds, reward, terminated, truncated, reason)
        self._prev_actions[1] = self._prev_actions[0]
        self._prev_actions[0] = np.clip(action, -1.0, 1.0)
        self._prev_gap = gap

        info: dict[str, Any] = {
            "d": query.d,
            "psi": query.psi,
            "ds": ds,
            "reason": reason,
            "step_reward": step_reward,
            "terminal_reward": terminal,
            "rate_confound": self.rate_confound,
            "metrics": self.metrics,
        }
        return self._observation(), reward, terminated, truncated, info

    def _drive_mobile_obstacles(self) -> None:
        """Advance the mocap obstacles. Documented no-op until the shared motion model lands.

        The S7.4 NPC and pedestrian motion models are owned by ``[env]``
        (``tasks/direct/lane_following/obstacles.py``). When that module lands, this method should
        call it so both simulators move obstacles with the same policy. Inventing a second motion
        model here would be a divergence dressed up as a feature.

        A scene that actually contains a mobile obstacle warns once per environment, because such a
        scene silently evaluates a *different* task from the Isaac one: the movers stand still.
        """
        if self._mobile_warned or not any(o.mobile for o in self.scene.obstacles):
            return
        self._mobile_warned = True
        warnings.warn(
            f"{sum(1 for o in self.scene.obstacles if o.mobile)} mobile obstacle(s) in this scene "
            f"will NOT move: the shared S7.4 motion model ([env]-owned) is not available, and this "
            f"harness refuses to invent a second one. Any number from this scene is a "
            f"static-obstacle number.",
            RuntimeWarning,
            stacklevel=2,
        )

    def _wheel_positions(self) -> np.ndarray:
        """Return the ``(left, right)`` wheel joint angles in radians."""
        adr = [
            self.model.jnt_qposadr[self._id(self._obj.mjOBJ_JOINT, name)]
            for name in (self.robot.left_wheel_joint_name, self.robot.right_wheel_joint_name)
        ]
        return np.array([self.data.qpos[a] for a in adr], dtype=np.float64)

    def _update_encoders(self) -> None:
        """Update the quantized wheel-speed estimate (S7.3 axis D13)."""
        positions = self._wheel_positions()
        tick = 2.0 * math.pi / float(self.robot.encoder_ticks_per_rev)
        ticks = np.round((positions - self._prev_wheel_pos) / tick)
        self._prev_wheel_pos = self._prev_wheel_pos + ticks * tick
        speed = ticks * tick / self.control_dt
        dropout = self.action_path.dr.encoder_dropout
        if dropout > 0.0:
            keep = self.rng.random(2) >= dropout
            speed = np.where(keep, speed, self._encoder_speed)
        self._encoder_speed = speed

    def _nearest_obstacle_gap(self, x: float, y: float) -> float:
        """Return the signed clearance to the nearest obstacle safety circle, in metres."""
        if not self.scene.obstacles:
            return float("inf")
        best = math.inf
        for obstacle in self.scene.obstacles:
            distance = math.hypot(x - obstacle.pos[0], y - obstacle.pos[1])
            best = min(best, distance - (self.cfg.obstacle_margin + obstacle.radius))
        return best

    # -- reward and terminations ---------------------------------------------------------------
    def _reward(self, action: np.ndarray, query: Any, ds: float, gap: float) -> float:
        """Evaluate the six per-step terms of the SPEC v2 S5.4 reward.

        The result is deliberately **not** clipped here. S5.4 clips ``R``, and ``R`` includes the
        terminal penalty, so the clip belongs after that penalty is added; :meth:`step` owns it.

        Args:
            action: the action taken this step.
            query: the lane projection after the step.
            ds: lane-frame forward progress this step, in metres.
            gap: signed clearance to the nearest obstacle safety circle, in metres.

        Returns:
            The unclipped weighted sum of the six per-step terms.
        """
        w_lane = self.lane.city.lane_width
        d, psi = query.d, query.psi
        psi_target = -np.clip(d / 0.05, -1.0, 1.0) * (math.pi / 4.0)
        error = psi - psi_target

        def leaky_cos(value: float) -> float:
            return math.cos(value) if abs(value) < math.pi else -1.0 - 0.05 * (abs(value) - math.pi)

        r_head = 0.5 * (
            leaky_cos(math.pi * error / math.radians(10.0)) + leaky_cos(math.pi * error / math.radians(50.0))
        )
        gate = 0.5 * (w_lane - self.robot.robot_width) + 0.02
        r_prog = ds / (self.robot.v_max * self.control_dt) if (ds > 0.0 and d < gate) else 0.0
        r_lat = -(1.0 - 0.001 ** (abs(d) / (0.5 * w_lane)))
        r_smooth = -float(np.sum((action - self._prev_actions[0]) ** 2))
        # S5.4 defines p as the nearest safety-circle OVERLAP, i.e. p <= 0, so r_prox pays only for
        # opening an overlap that already exists. Feeding the raw signed clearance instead would
        # pay up to 1.5 per step for moving away from an obstacle anywhere on the map, comparable
        # to the 6.0-weighted progress term, turning a collision-recovery term into a general
        # keep-away bonus and shaping C5/C6 returns differently from the Isaac reward.
        if math.isinf(gap) or math.isinf(self._prev_gap):
            r_prox = 0.0
        else:
            p = min(0.0, gap)
            p_prev = min(0.0, self._prev_gap)
            r_prox = float(np.clip(-(p_prev - p) * 50.0, 0.0, 1.5))
        stall = 1.0 if abs(self.body_speed()) < self.cfg.stall_speed else 0.0
        total = 1.0 * r_head + 6.0 * r_prog + 0.5 * r_lat + 0.10 * r_smooth + 1.0 * r_prox - 0.5 * stall
        return float(total)

    def _test_points(self, x: float, y: float, yaw: float) -> list[tuple[float, float]]:
        """Return the four S5.5 off-drivable test points in world coordinates."""
        c, s = math.cos(yaw), math.sin(yaw)
        half = 0.5 * self.robot.wheel_separation
        front = self.robot.chassis_center[0] + 0.5 * self.robot.chassis_size[0]
        local = ((0.0, 0.0), (0.0, half), (0.0, -half), (front, 0.0))
        return [(x + c * lx - s * ly, y + s * lx + c * ly) for lx, ly in local]

    def _terminations(self, x: float, y: float, yaw: float, gap: float) -> tuple[bool, str]:
        """Evaluate the SPEC v2 S5.5 termination conditions.

        Args:
            x: world x in metres.
            y: world y in metres.
            yaw: heading in radians.
            gap: signed clearance to the nearest obstacle safety circle.

        Returns:
            ``(terminated, reason)`` where ``reason`` is ``"timeout"`` when nothing fired.
        """
        if any(not self.lane.is_drivable(px, py) for px, py in self._test_points(x, y, yaw)):
            return True, "off_drivable"
        if gap < 0.0:
            self.metrics.collisions += 1
            return True, "obstacle"
        roll, pitch = self.roll_pitch()
        if abs(roll) > _DEG30 or abs(pitch) > _DEG30:
            return True, "rollover"
        if abs(self.body_speed()) < self.cfg.stall_speed:
            self._stall_steps += 1
        else:
            self._stall_steps = 0
        if self._stall_steps * self.control_dt > self.cfg.stall_timeout_s:
            return True, "stall"
        self._yaw_integral += abs(self.data.qvel[5]) * self.control_dt
        displacement = float(np.linalg.norm(np.array([x, y]) - self._start_xy))
        if self._yaw_integral > 3.0 * math.pi and displacement < 0.2:
            return True, "spin"
        return False, "timeout"

    def _accumulate(
        self,
        query: Any,
        ds: float,
        reward: float,
        terminated: bool,
        truncated: bool,
        reason: str,
    ) -> None:
        """Fold this step into :attr:`metrics`.

        Args:
            query: the lane projection for the pose after the step.
            ds: lane-frame forward progress this step, in metres.
            reward: the S5.4 reward for this step.
            terminated: whether a termination condition fired.
            truncated: whether the episode hit its horizon.
            reason: the termination reason.
        """
        metrics = self.metrics
        w_half = 0.5 * self.lane.city.lane_width
        metrics.steps = self._step_count
        metrics.duration_s = self._step_count * self.control_dt
        metrics.distance_m += max(0.0, ds)
        metrics.return_ += reward
        self._lateral.append(query.d)
        metrics.lane_abs_integral_ms += abs(query.d) * self.control_dt
        metrics.out_of_lane_integral_ms += max(0.0, abs(query.d) - w_half) * self.control_dt
        if terminated or truncated:
            lateral = np.asarray(self._lateral, dtype=np.float64)
            metrics.lane_rms_m = float(np.sqrt(np.mean(lateral**2)))
            metrics.lane_max_m = float(np.max(np.abs(lateral)))
            metrics.p_far = float(np.mean(np.abs(lateral) > 0.6 * w_half))
            metrics.laps = (
                metrics.distance_m / self._loop_length
                if self._loop_length == self._loop_length
                else float("nan")
            )
            metrics.success = truncated and not terminated
            metrics.reason = "timeout" if metrics.success else reason

    # -- observations --------------------------------------------------------------------------
    def _vec(self) -> np.ndarray:
        """Return the 8-dimensional proprioceptive observation (S5.2)."""
        return np.array(
            [
                self._prev_actions[0, 0],
                self._prev_actions[0, 1],
                self._prev_actions[1, 0],
                self._prev_actions[1, 1],
                self._encoder_speed[0],
                self._encoder_speed[1],
                float(self.data.qvel[5]),
                self.body_speed(),
            ],
            dtype=np.float32,
        )

    def vec_priv(self) -> np.ndarray:
        """Return the 14-dimensional privileged critic observation (S5.2).

        ``dist_nearest_obstacle`` is the true centre-to-centre distance, and an obstacle-free map
        reports :data:`NO_OBSTACLE_DISTANCE` rather than 0.0. Mapping "no obstacle anywhere" onto
        0.0 encoded the emptiest possible scene as the most dangerous one the critic can see, and
        every map in the current default set is obstacle-free. ``rel_speed_nearest_obstacle`` is
        the body velocity projected on the bearing to that obstacle, positive when closing.

        Returns:
            The 14-vector: the 8 proprioceptive entries followed by ``d``, ``psi``, curvature,
            distance to the nearest obstacle, closing speed, and the along-lane speed.
        """
        x, y, yaw = self.pose()
        query = self.lane.query(x, y, yaw, prev_match=getattr(self, "_prev_match", None))
        distance, closing = self._nearest_obstacle_state(x, y, yaw)
        extra = np.array(
            [
                query.d,
                query.psi,
                query.curvature,
                distance,
                closing,
                self.body_speed() * math.cos(query.psi),
            ],
            dtype=np.float32,
        )
        return np.concatenate([self._vec(), extra])

    def _nearest_obstacle_state(self, x: float, y: float, yaw: float) -> tuple[float, float]:
        """Return ``(distance, closing_speed)`` to the nearest obstacle centre.

        Args:
            x: world x in metres.
            y: world y in metres.
            yaw: heading in radians.

        Returns:
            The centre-to-centre distance in metres (:data:`NO_OBSTACLE_DISTANCE` when the scene
            has no obstacle) and the component of the body velocity along the bearing to it in
            m/s, positive when closing.
        """
        del yaw
        if not self.scene.obstacles:
            return NO_OBSTACLE_DISTANCE, 0.0
        best = min(
            self.scene.obstacles,
            key=lambda o: math.hypot(x - o.pos[0], y - o.pos[1]),
        )
        dx, dy = best.pos[0] - x, best.pos[1] - y
        distance = math.hypot(dx, dy)
        if distance < 1e-9:
            return 0.0, 0.0
        vx, vy = float(self.data.qvel[0]), float(self.data.qvel[1])
        return distance, (vx * dx + vy * dy) / distance

    def render_frame(self) -> np.ndarray:
        """Render one 192x128 RGB frame from the robot camera.

        Returns:
            An ``(H, W, 3)`` uint8 array.

        Raises:
            RuntimeError: if the environment was built without a camera.
        """
        if self._renderer is None or self._camera_id < 0:
            raise RuntimeError("this environment has no camera; build it with MjEnvCfg(obs_mode='rgb_vec')")
        self._renderer.update_scene(self.data, camera=self._camera_id)
        return self._renderer.render()

    def _observation(self, first: bool = False) -> dict[str, np.ndarray]:
        """Assemble the observation dict for the current state.

        Args:
            first: True on the post-reset call, which refills the frame ring with copies of the
                first frame exactly as the Isaac env does (SPEC v2 S6.3).

        Returns:
            The observation dict. ``obs_mode='none'`` returns an empty dict.
        """
        if self.cfg.obs_mode == "none":
            return {}
        vec = self._vec()
        if self.cfg.obs_mode == "vec":
            return {"vec": vec}
        frame = self._preprocess_frame(self.render_frame())[None]
        if first:
            self._stack.reset(frame=frame)
        else:
            self._stack.push(frame)
        return {"rgb": np.asarray(self._stack.get())[0], "vec": vec}

    def _check_preprocess_constants(self) -> None:
        """Assert the shared preprocess constants match this package's copy.

        Raises:
            ValueError: on any mismatch. A silent divergence here would mean the MuJoCo policy sees
                a differently shaped or differently cropped image from the one it was trained on,
                which is exactly the class of bug the S8.3 parity requirements exist to prevent.
        """
        module, obs = self._preprocess, self.obs_params
        expected = {
            "RENDER_W": obs.render_w,
            "RENDER_H": obs.render_h,
            "OBS_W": obs.obs_w,
            "OBS_H": obs.obs_h,
            "CROP_TOP": obs.crop_top,
            "BOX": obs.box,
        }
        bad = {
            name: (getattr(module, name), want)
            for name, want in expected.items()
            if getattr(module, name, want) != want
        }
        if bad:
            raise ValueError(
                f"{module.__name__} disagrees with duckiebot_rl.sim2sim about the S4.3 constants "
                f"{bad} (module value, sim2sim value). preprocess.py changes need [ppo]+[sim2sim]"
                f"+[deploy] sign-off; update ObsParams in _resolve.py to match."
            )

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Run the shared S4.3 chain over one rendered frame.

        Steps 5 to 8 (fixed 5-tap blur, exact 2x2 box downsample, 16-row crop, uint8 quantize) come
        from the shared module, so the MuJoCo path cannot drift from the Isaac path. Photometric
        randomization (step 3) is passed through as a callable for condition C6.

        Args:
            frame: the ``(128, 192, 3)`` uint8 render.

        Returns:
            The ``(48, 96, 3)`` uint8 preprocessed frame.
        """
        photometric = self._photometric if self.cfg.photometric_dr else None
        return np.asarray(self._preprocess.preprocess_frame_np(frame[None], photometric=photometric))[0]

    def _photometric(self, image: np.ndarray) -> np.ndarray:
        """Layer-1 photometric randomization for condition C6 (SPEC v2 S4.3 step 3, S7.2 V12-V18).

        Delegates to the ``[dr]``-owned :class:`duckiebot_rl.dr.visual.VisualDR`, the same object the
        Isaac env drives, so C6 gets the identical chain in the identical order rather than a
        MuJoCo-only augmentation that would not be comparable with C1.

        That module is torch-only, which is the one place the vision path in this package does need
        torch. The rest of it (the S4.3 tail, the frame stack, the tile textures) has a numpy
        implementation and runs without.

        Args:
            image: NCHW float32 batch in ``[0, 1]`` at render resolution, as handed over by
                ``preprocess_frame_np``.

        Returns:
            The randomized batch, same shape and dtype.

        Raises:
            SharedModuleUnavailable: if torch or the shared visual module is missing.
        """
        if self._visual_dr is None:
            self._visual_dr = self._build_visual_dr()
        torch, visual_dr = self._visual_dr
        tensor = torch.from_numpy(np.ascontiguousarray(image))
        speed = torch.tensor(
            [min(1.0, abs(self.body_speed()) / max(self.robot.v_max, 1e-6))], dtype=tensor.dtype
        )
        params = visual_dr.sample(self.cfg.dr_alpha, speed_frac=speed)
        with torch.no_grad():
            out = visual_dr.apply(tensor, params)
        return out.cpu().numpy().astype(np.float32)

    def _build_visual_dr(self) -> tuple[Any, Any]:
        """Construct the shared photometric randomizer, seeded from this environment's RNG.

        Returns:
            ``(torch_module, VisualDR_instance)``.

        Raises:
            SharedModuleUnavailable: if torch or ``duckiebot_rl.dr.visual`` cannot be imported, or
                if the module does not expose the ``sample``/``apply`` pair.
        """
        from ._resolve import _try_import

        torch = _try_import("torch")
        if torch is None:
            raise SharedModuleUnavailable(
                "condition C6's photometric half needs torch: duckiebot_rl.dr.visual is torch-only. "
                "Everything else in the MuJoCo vision path runs on numpy, so C5 works today and "
                "only C6 is blocked. Install CPU torch into the tools venv (SPEC v2 M0):\n  "
                "d:/Personal/personal/mujoco_venv/Scripts/python.exe -m pip install torch "
                "--index-url https://download.pytorch.org/whl/cpu"
            )
        visual = _try_import("duckiebot_rl.dr.visual")
        factory = getattr(visual, "VisualDR", None) if visual is not None else None
        if factory is None:
            raise SharedModuleUnavailable(
                "condition C6 needs duckiebot_rl.dr.visual.VisualDR, which is owned by [dr]. A "
                "MuJoCo-only augmentation invented here would not be comparable with C1."
            )
        generator = torch.Generator()
        generator.manual_seed(int(self.rng.integers(0, 2**31 - 1)))
        instance = factory(num_envs=1, generator=generator)
        for method in ("sample", "apply"):
            if not callable(getattr(instance, method, None)):
                raise SharedModuleUnavailable(
                    f"duckiebot_rl.dr.visual.VisualDR has no {method}() method; sim2sim drives it "
                    f"through sample(alpha, speed_frac=...) and apply(nchw, params)."
                )
        return torch, instance

    def close(self) -> None:
        """Release the offscreen renderer."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def provenance(self) -> dict[str, str]:
        """Return where every parameter group came from, for inclusion in evaluation reports."""
        return {
            "robot_params": self._robot_source,
            "sim_rates": self._sim_source,
            "dr_ranges": self._dr_source,
            "city_geometry": _resolve.CITY_SOURCE,
            "lane_offset_tiles": f"{self.lane.city.lane_offset_tiles:.4f}",
            "lane_width_m": f"{self.lane.city.lane_width:.4f}",
            "textures": type(self.scene.texture_provider).__name__,
            "obs_mode": self.cfg.obs_mode,
            "rate_confound": str(self.rate_confound),
        }
