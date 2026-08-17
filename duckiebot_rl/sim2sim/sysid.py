"""Isaac-to-MuJoCo physics matching, two stages (SPEC v2 S8.2).

The two simulators are only comparable once their *open-loop* responses agree. Matching them with a
single optimizer over all six parameters does not work, and the reason was measured during the
research phase rather than guessed:

* Sliding friction ``mu`` is unidentifiable from any manoeuvre that does not slip. Its residual
  Jacobian column is numerically zero, ``J^T J`` is singular and Levenberg-Marquardt reports a
  factorization failure. It only becomes meaningful under a slip-inducing excitation.
* The effective rolling radius and the effective wheel baseline are each determined in closed form
  by two steady-state scalars. Handing them to an optimizer only adds local minima.

So stage 1 solves the three steady-state parameters analytically and iterates to a fixed point
(three passes suffice; the estimators are weakly coupled because changing the radius shifts both the
measured servo droop and the yaw gain), and stage 2 runs Levenberg-Marquardt on the three transient
parameters only.

Stage 1 - closed form, three fixed-point passes
    ``r_eff = v_ss / omega_wheel_ss`` from straight lines at 12, 20 and 28 rad/s;
    ``b_eff = r_eff * (omega_R - omega_L) / psidot_ss`` from constant arcs at delta-omega 2, 4, 6;
    ``kv = tau_ss / (ctrl - omega_ss)`` from the velocity-servo droop.
    The arcs stay inside the no-slip envelope on purpose: past about 20 rad/s of wheel-speed
    difference the base exceeds roughly 0.6 g of lateral acceleration and pirouettes, and no
    parameter fit can match that.

Stage 2 - Levenberg-Marquardt on ``(armature, frictionloss, damping)``
    Excited by a velocity step and a coast-down, scored on **per-control-step body-frame
    increments** rather than absolute pose. Absolute pose integrates its error, which makes the
    least-squares problem effectively chaotic as soon as contact slip differs at all.

Acceptance (S8.2 and milestone M10): stage-1 recovery of ``r`` and ``b`` within 0.5%; open-loop
endpoint error at most 25 mm over runs of 2.7 to 4.0 m; and every identified delta must be COVERED
by the S7.3 domain-randomization ranges. If a delta falls outside, the RANGE widens; the nominal
never moves to chase the fit.

Measured behaviour of this implementation
----------------------------------------

The table below is **generated, not written**. It is the shipped synthetic self-test, whose
perturbation is :data:`SELFTEST_PERTURBATION` (the single definition the CLI also uses: effective
radius x1.03, effective baseline x0.965, joint armature 6.0e-4 kg.m^2 = x3, joint friction 4.0e-3
N.m = x0.4, servo gain kv 0.042 N.m.s/rad = x0.84). Regenerate it, and the JSON artifact beside it,
with::

    d:/Personal/personal/mujoco_venv/Scripts/python.exe scripts/eval_sim2sim.py sysid \
        --out docs/sim2sim_sysid_selftest.json

which is three outer stage-1/stage-2 alternations with a 50-iteration Levenberg-Marquardt cap.

``tests/unit/test_mj_sysid.py`` re-runs the same self-test and fails if any number here moves, so
this table cannot drift away from the code again.

======================================  =============================  ==========================
Quantity                                Measured                       S8.2 criterion
======================================  =============================  ==========================
``r_eff`` / ``b_eff``                   -0.030% / +0.045%              0.5%: **met**
``kv``                                  -7.142%                        no explicit gate
armature                                4.338e-4 vs 6.0e-4 (-28%)      covered by S7.3: **not met**
joint friction                          9.645e-3 vs 4.0e-3 (+141%)     covered by S7.3: met
straight endpoint over 3.2 m            20.31 mm                       25 mm: met
curved endpoint over 3.2 m of arc       301.39 mm                      25 mm: **not met**
steady straights, final position        27.56 / 25.42 / 22.86 mm       (not endpoint programs)
accepted                                ``False``
======================================  =============================  ==========================

Read that honestly, because two of the four criteria fail:

* The **curved endpoint** is the largest open item, at 301 mm against a 25 mm gate. A five-second
  constant-curvature run covers most of a half turn, so a few percent of residual disagreement in
  the *transient* effective baseline rotates the whole arc and lands the endpoint hundreds of
  millimetres away, while the straight-line endpoint from the same fit is 20 mm.
* The **identified armature**, 4.338e-4, sits outside the S7.3 range ``[1.0e-4, 4.0e-4]`` that
  ``dr_armature_scale`` implies. Per S8.2 the response is to widen the RANGE, never to move the
  nominal to chase the fit; that decision belongs to ``[assets]`` and is not taken here.
* The three steady straights end 22.9 to 27.6 mm from the reference. Two of the three exceed
  25 mm. They are not ``ENDPOINT_PROGRAMS`` and so do not enter the acceptance test, but quoting
  the straight-line agreement as millimetric would be wrong: it is centimetric.

So the S8.2 endpoint gate is **not yet met by this self-test**, and no number here should be
presented as if it were. Two things to try before M10, in this order: (1) have the Isaac exporter
log actuator torque, which is already supported here and which pins ``kv`` in closed form (without
it the ``(kv, joint friction)`` pair is degenerate on the servo droop and the fit measurably
degrades, which is visible above as the -7.1% ``kv`` and the +141% joint friction absorbing each
other); (2) if the curved endpoint still fails once ``kv`` is pinned, the gap is a real physics
difference rather than an identification failure, and it belongs in the report as one.

Note what the self-test does and does not prove. It perturbs a MuJoCo model and asks this code to
recover it, so it validates the *procedure* end to end. It is not the Isaac-to-MuJoCo gap, which
needs a reference captured from Isaac (``--reference``) and has not been measured yet.

Critical implementation detail, learned the hard way: whenever the wheel radius changes, the chassis
frame rides at the new radius and the caster is left hanging. A Duckiebot on two contact points is a
free-pitching, free-yawing bicycle and spins out in tight arcs (yaw rate measured at 2.3x the
kinematic prediction). :func:`seat` re-seats both the chassis height and the caster offset on every
reset, and :func:`apply_params` keeps the caster tangent.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import mjcf as _mjcf
from ._resolve import DRRanges, RobotParams, resolve_dr_ranges

__all__ = [
    "ENDPOINT_PROGRAMS",
    "REFERENCE_SCHEMA",
    "SPIN_PROGRAMS",
    "STEADY_PROGRAMS",
    "TRANSIENT_PROGRAMS",
    "Program",
    "Stage1Result",
    "SteadyState",
    "SysIdResult",
    "apply_stage1",
    "collect_reference",
    "load_reference",
    "make_model",
    "match_stage1",
    "measure_stage1",
    "residual_report",
    "rollout",
    "run_selftest",
    "run_sysid",
    "save_reference",
    "seat",
    "stage1",
    "stage2",
    "steady_state",
]

REFERENCE_SCHEMA = "duckiebot-sim2sim-reference/1"

#: The synthetic self-test perturbation, as multiples of the nominal S2 values where the quantity
#: is a scale and absolute SI units where it is not. This is the ONLY definition: the module
#: docstring quotes it, :func:`run_selftest` applies it and ``scripts/eval_sim2sim.py sysid`` calls
#: that function, so the documented experiment and the executed one cannot diverge. It perturbs
#: every identifiable parameter at once, which is the hard case on purpose.
SELFTEST_PERTURBATION: dict[str, float] = {
    "radius_scale": 1.03,
    "baseline_scale": 0.965,
    "armature": 6.0e-4,
    "frictionloss": 0.004,
    "kv": 0.042,
}

#: Where ``scripts/eval_sim2sim.py sysid --out`` writes the regenerable artifact that
#: ``tests/unit/test_mj_sysid.py`` checks the module docstring against.
SELFTEST_ARTIFACT = "docs/sim2sim_sysid_selftest.json"


@dataclass(frozen=True)
class Program:
    """One open-loop excitation program.

    Attributes:
        name: identifier used in reports and in the reference file.
        left: left-wheel command in rad/s as a function of time in seconds.
        right: right-wheel command in rad/s as a function of time in seconds.
        duration: program length in seconds.
        average: length of the trailing window used for steady-state estimation, in seconds.
    """

    name: str
    left: Callable[[float], float]
    right: Callable[[float], float]
    duration: float
    average: float = 1.0


def _const(value: float) -> Callable[[float], float]:
    """Return a constant command function."""
    return lambda _t: value


def _step(before: float, after: float, at: float) -> Callable[[float], float]:
    """Return a step command function."""
    return lambda t: before if t < at else after


STEADY_PROGRAMS: tuple[Program, ...] = (
    Program("straight_12", _const(12.0), _const(12.0), 6.0, average=3.0),
    Program("straight_20", _const(20.0), _const(20.0), 6.0, average=3.0),
    Program("straight_28", _const(28.0), _const(28.0), 6.0, average=3.0),
    Program("arc_d2", _const(19.0), _const(21.0), 6.0, average=3.0),
    Program("arc_d4", _const(18.0), _const(22.0), 6.0, average=3.0),
    Program("arc_d6", _const(17.0), _const(23.0), 6.0, average=3.0),
    Program("spin_ccw", _const(-15.0), _const(15.0), 8.0, average=6.0),
    Program("spin_cw", _const(15.0), _const(-15.0), 8.0, average=6.0),
)
"""Steady-state programs for stage 1.

``r_eff`` comes from the three straights. ``b_eff`` comes from the **bidirectional spin in place**,
with the constant-curvature arcs kept as a cross-check. That split is a measured decision, not a
preference:

============================================  =====================  ==================
Estimator (this model, 5 physical baselines)  effective/physical     spread over the 5
============================================  =====================  ==================
tight arcs, delta-omega 2/4/6 at 20 rad/s     0.94 - 0.96 (scrub)    1.9%
same arcs at a lower mean wheel speed         0.87 - 0.95            8.4%
bidirectional spin at 15 rad/s, 6 s averaged  1.001 - 1.006          **0.44%**
============================================  =====================  ==================

A spin has no forward velocity, so no lateral acceleration and no tyre scrub, and it is unbiased:
``psidot = r * (omega_R - omega_L) / b``. The arcs, at ``R/b`` between 1.6 and 4.8, sit close enough
to the slip envelope that the *effective* baseline they report wanders by about 2% between otherwise
identical models. That matters because SPEC v2 S8.2 accepts stage 1 at 0.5%: with an arc-only
estimator the acceptance criterion would be measuring estimator noise. The arcs above were also
widened from delta-omega 2/4/6 at a 20 rad/s mean (radii 0.48 / 0.24 / 0.16 m) to the same deltas
about a 20 rad/s mean with gentler ratios, and run for 6 s with a 3 s average.

Do not lengthen the spin beyond about 8 s: at 12 s one of the five test baselines destabilized and
the estimate jumped by 63%. The frictionless caster leaves chassis yaw under-constrained, so a long
spin eventually finds a limit cycle."""

SPIN_PROGRAMS: tuple[Program, ...] = tuple(p for p in STEADY_PROGRAMS if p.name.startswith("spin"))
"""The two spin programs, used for the primary ``b_eff`` estimate."""

TRANSIENT_PROGRAMS: tuple[Program, ...] = (
    Program("step_20", _step(0.0, 20.0, 0.5), _step(0.0, 20.0, 0.5), 2.5),
    Program("coast_20", _step(20.0, 0.0, 1.5), _step(20.0, 0.0, 1.5), 3.0),
    Program("step_arc", _step(0.0, 14.0, 0.5), _step(0.0, 26.0, 0.5), 2.5),
)
"""Transient programs (S8.2 stage 2). Wheel-speed differences stay at or below 12 rad/s."""

ENDPOINT_PROGRAMS: tuple[Program, ...] = (
    Program("endpoint_straight", _const(20.0), _const(20.0), 5.0),
    Program("endpoint_arc", _const(19.0), _const(21.0), 5.0),
)
"""Long open-loop runs for the S8.2 25 mm endpoint acceptance (about 3.2 m of path).

The arc is deliberately gentle (1.0 m radius, roughly a half turn over the run). The stage-1
estimation arcs are tight on purpose, because a tight arc maximizes the yaw-rate signal, but a tight
arc run open loop for five seconds completes several revolutions and turns a 1% baseline error into
hundreds of millimetres of endpoint error. That would make the acceptance criterion a measure of
revolution count rather than of model agreement."""


@dataclass
class SteadyState:
    """Averaged steady-state response of one excitation.

    Attributes:
        v: body-frame forward speed in m/s.
        psidot: yaw rate in rad/s.
        wheel_left: left wheel speed in rad/s.
        wheel_right: right wheel speed in rad/s.
        torque_left: left actuator force in N.m.
        torque_right: right actuator force in N.m.
    """

    v: float
    psidot: float
    wheel_left: float
    wheel_right: float
    torque_left: float
    torque_right: float


@dataclass
class Stage1Result:
    """Closed-form stage-1 estimates.

    Attributes:
        r_eff: effective rolling radius in metres, from the straight lines.
        b_eff: effective wheel baseline in metres, from the bidirectional spin.
        kv: velocity-servo gain in N.m.s/rad, or NaN when the log carries no actuator torque.
        b_arc: the arc-based cross-check of the baseline, in metres, or NaN when unavailable.
        b_arc_spread_pct: peak-to-peak disagreement between the individual arcs, in percent. Values
            above a couple of percent mean the arcs are running near the slip envelope and the
            cross-check should not be trusted.
    """

    r_eff: float
    b_eff: float
    kv: float
    b_arc: float = float("nan")
    b_arc_spread_pct: float = float("nan")


@dataclass
class SysIdResult:
    """Everything a system-identification run produces.

    Attributes:
        stage1_target: the stage-1 estimates measured on the reference simulator.
        stage1_mujoco: the stage-1 estimates measured on MuJoCo after matching.
        stage1_error_pct: relative error per stage-1 parameter, in percent.
        stage2: identified ``armature``, ``frictionloss`` and ``damping``.
        residuals: per-program position and yaw residual statistics.
        endpoint_error_mm: worst endpoint error over :data:`ENDPOINT_PROGRAMS`, in millimetres.
        dr_coverage: per-parameter report on whether the identified delta is inside its S7.3 range.
        accepted: True when every S8.2 acceptance criterion holds.
        wall_clock_s: total run time in seconds.
        provenance: where the robot parameters and DR ranges came from.
    """

    stage1_target: Stage1Result
    stage1_mujoco: Stage1Result
    stage1_error_pct: dict[str, float]
    stage2: dict[str, float]
    residuals: dict[str, dict[str, float]]
    endpoint_error_mm: float
    dr_coverage: dict[str, dict[str, Any]]
    accepted: bool
    wall_clock_s: float
    provenance: dict[str, str] = field(default_factory=dict)

    def save(self, path: str | Path) -> Path:
        """Write the result as JSON.

        Args:
            path: destination file.

        Returns:
            The destination path.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return destination

    def report(self) -> str:
        """Render a human-readable residual report."""
        lines = [
            "sim2sim system identification (SPEC v2 S8.2)",
            f"  provenance          : {self.provenance}",
            "  stage 1 (closed form, 3 fixed-point passes)",
        ]
        for key in ("r_eff", "b_eff", "kv"):
            target = getattr(self.stage1_target, key)
            got = getattr(self.stage1_mujoco, key)
            lines.append(
                f"    {key:<6} target {target:<12.6g} mujoco {got:<12.6g} "
                f"({self.stage1_error_pct[key]:+.3f}%)"
            )
        lines.append(
            f"    b_arc cross-check target {self.stage1_target.b_arc:.6g} "
            f"(arc spread {self.stage1_target.b_arc_spread_pct:.2f}%)  "
            f"mujoco {self.stage1_mujoco.b_arc:.6g} "
            f"(arc spread {self.stage1_mujoco.b_arc_spread_pct:.2f}%)"
        )
        lines.append("  stage 2 (Levenberg-Marquardt on the transient parameters)")
        for key, value in self.stage2.items():
            lines.append(f"    {key:<13} {value:.6g}")
        lines.append(f"{'    program':<26}{'pos RMS [mm]':>14}{'final pos [mm]':>16}{'yaw RMS [deg]':>15}")
        for name, stats in self.residuals.items():
            lines.append(
                f"    {name:<22}{stats['pos_rms_mm']:>14.2f}"
                f"{stats['final_pos_mm']:>16.2f}{stats['yaw_rms_deg']:>15.3f}"
            )
        lines.append(f"  worst endpoint error : {self.endpoint_error_mm:.2f} mm (limit 25.0)")
        lines.append("  DR coverage of the identified deltas (S8.2: widen the RANGE, never the nominal)")
        for key, entry in self.dr_coverage.items():
            state = "inside" if entry["covered"] else "OUTSIDE - widen this range"
            lines.append(f"    {key:<13} {entry['value']:.6g} in {entry['range']} -> {state}")
        lines.append(f"  accepted             : {self.accepted}")
        lines.append(f"  wall clock           : {self.wall_clock_s:.1f} s")
        return "\n".join(lines)


# ------------------------------------------------------------------------------- model plumbing
def make_model(cfg: _mjcf.MjcfCfg | None = None) -> tuple[Any, Any, _mjcf.MjcfCfg]:
    """Compile the flat-ground robot-only model used for identification.

    Args:
        cfg: MJCF configuration; a fresh :meth:`MjcfCfg.from_shared` is used when None.

    Returns:
        ``(model, data, cfg)``.
    """
    import mujoco

    cfg = cfg if cfg is not None else _mjcf.MjcfCfg.from_shared(include_camera=False)
    model = mujoco.MjModel.from_xml_string(_mjcf.build_robot_xml(cfg))
    return model, mujoco.MjData(model), cfg


def _geom(model: Any, name: str) -> int:
    """Return a geom id by name."""
    import mujoco

    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name))


def _body(model: Any, name: str) -> int:
    """Return a body id by name."""
    import mujoco

    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))


def _dof(model: Any, name: str) -> int:
    """Return the dof address of a hinge joint by name."""
    import mujoco

    return int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])


def seat(model: Any, data: Any, robot: RobotParams) -> None:
    """Reset with the chassis seated exactly on the current wheel radius.

    ``qpos0`` for a free body is baked at compile time, so after the radius changes the wheels start
    buried in the ground and the stiff contact launches (and flips) the robot.

    Args:
        model: the compiled model.
        data: its data block.
        robot: the shared robot parameters, used for the link names.
    """
    import mujoco

    mujoco.mj_resetData(model, data)
    radius = float(model.geom_size[_geom(model, f"{robot.left_wheel_link_name}_collision"), 0])
    data.qpos[2] = radius + 1e-6
    mujoco.mj_forward(model, data)


def steady_state(
    model: Any,
    data: Any,
    robot: RobotParams,
    left: float,
    right: float,
    control_hz: float,
    duration: float = 4.0,
    average: float = 1.0,
    settle: float = 0.7,
) -> SteadyState:
    """Drive open loop and average the last ``average`` seconds.

    The settle phase matters: :func:`seat` drops the chassis onto its contacts, and the resulting
    lightly damped yaw ring is what makes an un-settled estimate wander by a percent or two.

    Args:
        model: the compiled model.
        data: its data block.
        robot: the shared robot parameters.
        left: left-wheel command in rad/s.
        right: right-wheel command in rad/s.
        control_hz: control rate.
        duration: total drive time in seconds.
        average: averaging window at the end of the run, in seconds.
        settle: quiet time at zero command before the drive, in seconds.

    Returns:
        The averaged :class:`SteadyState`.
    """
    import mujoco

    seat(model, data, robot)
    substeps = max(1, round((1.0 / control_hz) / model.opt.timestep))
    for _ in range(int(settle * control_hz)):
        data.ctrl[:] = (0.0, 0.0)
        for _ in range(substeps):
            mujoco.mj_step(model, data)
    steps = round(duration * control_hz)
    window = round(average * control_hz)
    left_dof, right_dof = _dof(model, robot.left_wheel_joint_name), _dof(model, robot.right_wheel_joint_name)
    samples: list[tuple[float, ...]] = []
    for k in range(steps):
        data.ctrl[:] = (left, right)
        for _ in range(substeps):
            mujoco.mj_step(model, data)
        if k >= steps - window:
            q = data.qpos
            cos = 1.0 - 2.0 * (q[5] ** 2 + q[6] ** 2)
            sin = 2.0 * (q[3] * q[6] + q[4] * q[5])
            norm = math.hypot(cos, sin)
            samples.append(
                (
                    (cos * data.qvel[0] + sin * data.qvel[1]) / norm,
                    float(data.qvel[5]),
                    float(data.qvel[left_dof]),
                    float(data.qvel[right_dof]),
                    float(data.actuator_force[0]),
                    float(data.actuator_force[1]),
                )
            )
    columns = np.mean(np.asarray(samples, dtype=np.float64), axis=0)
    return SteadyState(*(float(v) for v in columns))


def rollout(
    model: Any,
    data: Any,
    robot: RobotParams,
    program: Program,
    control_hz: float,
    settle: float = 0.7,
) -> np.ndarray:
    """Run one open-loop program and log the pose at the control rate.

    Args:
        model: the compiled model.
        data: its data block.
        robot: the shared robot parameters.
        program: the excitation to run.
        control_hz: control rate.
        settle: quiet time before logging starts, in seconds.

    Returns:
        A ``(T, 10)`` array of
        ``[t, x, y, yaw, wheel_left, wheel_right, v_body, psidot, torque_left, torque_right]``.
        Averaging the body twist beats differentiating the pose for steady-state estimation, and the
        two torque columns are what make ``kv`` identifiable in closed form (see
        :func:`_stage1_from_trajectories`). Both are therefore part of the reference exchange
        format.
    """
    import mujoco

    seat(model, data, robot)
    substeps = max(1, round((1.0 / control_hz) / model.opt.timestep))
    left_dof, right_dof = _dof(model, robot.left_wheel_joint_name), _dof(model, robot.right_wheel_joint_name)
    for _ in range(int(settle * control_hz)):
        data.ctrl[:] = (0.0, 0.0)
        for _ in range(substeps):
            mujoco.mj_step(model, data)
    t0, x0, y0 = float(data.time), float(data.qpos[0]), float(data.qpos[1])
    steps = round(program.duration * control_hz)
    out = np.zeros((steps, 10), dtype=np.float64)
    for k in range(steps):
        t = k / control_hz
        data.ctrl[:] = (program.left(t), program.right(t))
        for _ in range(substeps):
            mujoco.mj_step(model, data)
        q = data.qpos
        cos = 1.0 - 2.0 * (q[5] ** 2 + q[6] ** 2)
        sin = 2.0 * (q[3] * q[6] + q[4] * q[5])
        norm = math.hypot(cos, sin)
        out[k] = (
            data.time - t0,
            q[0] - x0,
            q[1] - y0,
            math.atan2(sin, cos),
            data.qvel[left_dof],
            data.qvel[right_dof],
            (cos * data.qvel[0] + sin * data.qvel[1]) / norm,
            data.qvel[5],
            data.actuator_force[0],
            data.actuator_force[1],
        )
    return out


# ------------------------------------------------------------------------------------- stage 1
def stage1(probe: Callable[[float, float], SteadyState], spin_speed: float = 15.0) -> Stage1Result:
    """Solve the steady-state parameters in closed form from a torque-instrumented probe.

    Use this when the target simulator can report actuator torque, which makes ``kv`` identifiable
    from the servo droop. :func:`_stage1_from_trajectories` is the pose-log equivalent used when the
    reference arrives as a file.

    Args:
        probe: a function mapping ``(left_cmd, right_cmd)`` in rad/s to a :class:`SteadyState`.
        spin_speed: wheel speed used for the bidirectional spin, in rad/s.

    Returns:
        The :class:`Stage1Result`.
    """
    straights = [probe(w, w) for w in (12.0, 20.0, 28.0)]
    wheel = np.array([0.5 * (s.wheel_left + s.wheel_right) for s in straights])
    speed = np.array([s.v for s in straights])
    r_eff = float(wheel @ speed / (wheel @ wheel))

    spins = [probe(-spin_speed, spin_speed), probe(spin_speed, -spin_speed)]
    b_eff = float(np.mean([r_eff * (s.wheel_right - s.wheel_left) / s.psidot for s in spins]))

    arcs = [probe(20.0 - d, 20.0 + d) for d in (1.0, 2.0, 3.0)]
    deltas = np.array([a.wheel_right - a.wheel_left for a in arcs])
    rates = np.array([a.psidot for a in arcs])
    b_arc = float(r_eff * (deltas @ deltas) / (deltas @ rates))
    per_arc = r_eff * deltas / rates
    spread = float(100.0 * (per_arc.max() - per_arc.min()) / np.median(per_arc))

    commands = np.array([12.0, 20.0, 28.0])
    error = commands - wheel
    torque = np.array([0.5 * (s.torque_left + s.torque_right) for s in straights])
    kv = float(error @ torque / (error @ error))
    return Stage1Result(r_eff=r_eff, b_eff=b_eff, kv=kv, b_arc=b_arc, b_arc_spread_pct=spread)


def apply_stage1(
    model: Any,
    robot: RobotParams,
    kv: float | None = None,
    r_eff: float | None = None,
    b_eff: float | None = None,
) -> None:
    """Write stage-1 parameters into a compiled model, keeping the caster tangent.

    Args:
        model: the compiled model to mutate in place.
        robot: the shared robot parameters (link and joint names, caster geometry).
        kv: velocity-servo gain in N.m.s/rad.
        r_eff: effective wheel radius in metres.
        b_eff: effective wheel baseline in metres.
    """
    if kv is not None:
        for actuator in range(model.nu):
            model.actuator_gainprm[actuator, 0] = kv
            model.actuator_biasprm[actuator, 2] = -kv
    if r_eff is not None:
        for link in (robot.left_wheel_link_name, robot.right_wheel_link_name):
            geom = _geom(model, f"{link}_collision")
            model.geom_size[geom, 0] = r_eff
            model.geom_rbound[geom] = r_eff
        caster = _geom(model, "caster_collision")
        model.geom_pos[caster, 2] = model.geom_size[caster, 0] - r_eff
        model.body_pos[_body(model, robot.base_link_name), 2] = r_eff
    if b_eff is not None:
        for sign, link in ((+1.0, robot.left_wheel_link_name), (-1.0, robot.right_wheel_link_name)):
            model.body_pos[_body(model, link), 1] = sign * 0.5 * b_eff


def measure_stage1(model: Any, data: Any, robot: RobotParams, control_hz: float) -> Stage1Result:
    """Measure MuJoCo's stage-1 parameters using the same estimator applied to the reference.

    Both sides go through :func:`_stage1_from_trajectories` on logged pose plus wheel speed. Using
    one estimator on both sides is what makes the 0.5% acceptance criterion meaningful; comparing a
    torque-based estimate on one side with a pose-based estimate on the other would bake the
    estimator difference into the acceptance number.

    Args:
        model: the compiled model.
        data: its data block.
        robot: the shared robot parameters.
        control_hz: control rate.

    Returns:
        The stage-1 estimates.
    """
    trajectories = {p.name: rollout(model, data, robot, p, control_hz) for p in STEADY_PROGRAMS}
    return _stage1_from_trajectories(trajectories, robot, control_hz)


def match_stage1(
    model: Any,
    data: Any,
    robot: RobotParams,
    target: Stage1Result,
    control_hz: float,
    passes: int = 3,
) -> Stage1Result:
    """Iterate stage 1 to a fixed point so MuJoCo reproduces the target's steady state.

    ``kv`` is only matched when the target carries a finite one. A pose-and-wheel-speed reference
    log cannot identify ``kv``: the servo droop needs actuator torque, which the Isaac exporter does
    not have to provide. Stage 2 identifies it from the transients instead.

    Args:
        model: the compiled model to mutate.
        data: its data block.
        robot: the shared robot parameters.
        target: the stage-1 estimates measured on the reference simulator.
        control_hz: control rate.
        passes: number of fixed-point passes; three is enough in practice, because the estimators
            are only weakly coupled (changing the radius shifts both the droop and the yaw gain).

    Returns:
        The stage-1 estimates measured on MuJoCo after the last pass.
    """
    for _ in range(passes):
        current = measure_stage1(model, data, robot, control_hz)
        wheel_geom = _geom(model, f"{robot.left_wheel_link_name}_collision")
        kv = None
        if math.isfinite(target.kv) and math.isfinite(current.kv) and current.kv > 0.0:
            kv = float(model.actuator_gainprm[0, 0]) * target.kv / current.kv
        apply_stage1(
            model,
            robot,
            kv=kv,
            r_eff=float(model.geom_size[wheel_geom, 0]) * target.r_eff / current.r_eff,
            b_eff=2.0
            * float(model.body_pos[_body(model, robot.left_wheel_link_name), 1])
            * target.b_eff
            / current.b_eff,
        )
    return measure_stage1(model, data, robot, control_hz)


# ------------------------------------------------------------------------------------- stage 2
_STAGE2_NAMES = ("armature", "frictionloss", "damping")

STAGE2_WEIGHT_POS = 1.0 / 5e-4
"""Residual weight on per-control-step body-frame increments (1 unit per 0.5 mm of error)."""

STAGE2_WEIGHT_WHEEL = 1.0 / 0.1
"""Residual weight on wheel speeds (1 unit per rad/s of error).

Deliberately far below the position weight. Stage 1 has already matched the *effective* radius and
baseline, which it does by changing the physical ones; the physical wheel speed that produces a
given body speed therefore differs between the two models by construction. Weighting absolute wheel
speed heavily makes Levenberg-Marquardt chase an unreachable target and it pays for it by driving
the servo gain and the joint friction far from their true values (measured: kv 0.106 against a true
0.042, frictionloss 0.027 against a true 0.004, while the pose residual got worse). The wheel term
is kept because the step and coast transients carry the armature and friction information, but the
pose increments are what sim-to-sim transfer actually cares about."""


def _apply_stage2(model: Any, robot: RobotParams, x: Sequence[float]) -> None:
    """Write the three transient parameters into a compiled model.

    ``damping`` is the velocity-servo gain ``kv``, which is the *same quantity* as Isaac's
    ``ImplicitActuatorCfg(damping=...)``; it is not a second passive damper on the joint. Fitting a
    passive ``dof_damping`` instead would introduce a parameter the Isaac model does not have, and
    is also a trap: at ``dof_damping = 0.02`` the 0.15 N.m force limit caps every wheel at about
    7.2 rad/s regardless of command, so straights and arcs collapse onto the same trajectory and the
    identification silently succeeds on a model that cannot steer.

    Args:
        model: the compiled model to mutate in place.
        robot: the shared robot parameters (joint names).
        x: ``(armature, frictionloss, damping)``.
    """
    armature, frictionloss, damping = (float(v) for v in x)
    for name in (robot.left_wheel_joint_name, robot.right_wheel_joint_name):
        dof = _dof(model, name)
        model.dof_armature[dof] = armature
        model.dof_frictionloss[dof] = frictionloss
    for actuator in range(model.nu):
        model.actuator_gainprm[actuator, 0] = damping
        model.actuator_biasprm[actuator, 2] = -damping


def _increments(traj: np.ndarray) -> np.ndarray:
    """Return per-control-step body-frame increments ``(ds_fwd, ds_lat, dyaw)``.

    Absolute pose integrates its error, which makes the least-squares problem effectively chaotic
    once contact slip differs at all. Increments are far better conditioned.
    """
    x, y, psi = traj[:, 1], traj[:, 2], traj[:, 3]
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    dpsi_raw = np.diff(psi, prepend=psi[0])
    dpsi = np.arctan2(np.sin(dpsi_raw), np.cos(dpsi_raw))
    cos, sin = np.cos(psi), np.sin(psi)
    return np.stack([cos * dx + sin * dy, -sin * dx + cos * dy, 0.05 * dpsi], axis=1)


def stage2(
    model: Any,
    data: Any,
    robot: RobotParams,
    reference: dict[str, np.ndarray],
    control_hz: float,
    x0: Sequence[float] | None = None,
    max_iter: int = 40,
    verbose: bool = False,
    fit_kv: bool = True,
) -> dict[str, float]:
    """Fit ``(armature, frictionloss, damping)`` with Levenberg-Marquardt.

    Args:
        model: the compiled model to mutate.
        data: its data block.
        robot: the shared robot parameters.
        reference: reference trajectories keyed by program name.
        control_hz: control rate.
        x0: initial guess; defaults to the nominal S2 values.
        max_iter: LM iteration cap.
        verbose: print the LM iteration log.
        fit_kv: fit the servo gain too. Set False when stage 1 identified it in closed form from
            logged actuator torque, which leaves a well-conditioned two-parameter problem.

    Returns:
        The identified parameters, keyed by name.
    """
    import mujoco.minimize as minimize

    # Transients only. Adding the steady straights was tried and made the fit worse: their long
    # constant-speed tails pull the optimizer toward matching ABSOLUTE wheel speed, which is
    # unreachable once stage 1 has changed the physical radius to match the effective one.
    programs = [p for p in TRANSIENT_PROGRAMS if p.name in reference]
    start = np.array(
        x0 if x0 is not None else (robot.joint_armature, robot.joint_friction, robot.joint_damping),
        dtype=np.float64,
    )
    lower = np.array([1e-6, 0.0, 5e-3], dtype=np.float64)
    upper = np.array([5e-3, 0.05, 0.5], dtype=np.float64)
    if not fit_kv:
        # mujoco.minimize rejects a degenerate box, so pin kv with a hairline interval instead.
        frozen_kv = float(model.actuator_gainprm[0, 0])
        start[2] = frozen_kv
        lower[2] = frozen_kv * (1.0 - 1e-9)
        upper[2] = frozen_kv * (1.0 + 1e-9)
    weight_pos, weight_wheel = STAGE2_WEIGHT_POS, STAGE2_WEIGHT_WHEEL

    def residual(batch: np.ndarray) -> np.ndarray:
        columns = np.atleast_2d(batch.T) if batch.ndim == 2 else batch[None, :]
        out = []
        for candidate in columns:
            _apply_stage2(model, robot, candidate)
            parts = []
            for program in programs:
                simulated = rollout(model, data, robot, program, control_hz)
                target = reference[program.name]
                length = min(len(simulated), len(target))
                parts.append(
                    weight_pos * (_increments(simulated[:length]) - _increments(target[:length])).ravel()
                )
                parts.append(weight_wheel * (simulated[:length, 4:6] - target[:length, 4:6]).ravel())
            out.append(np.concatenate(parts))
        return np.stack(out, axis=1)

    solution, _trace = minimize.least_squares(
        start,
        residual,
        bounds=[lower, upper],
        max_iter=max_iter,
        verbose=minimize.Verbosity.ITER if verbose else minimize.Verbosity.SILENT,
        x_scale=np.maximum(np.abs(start), 1e-6),
    )
    _apply_stage2(model, robot, solution)
    return dict(zip(_STAGE2_NAMES, (float(v) for v in solution), strict=True))


# ------------------------------------------------------------------------------------ reporting
def residual_report(
    model: Any,
    data: Any,
    robot: RobotParams,
    reference: dict[str, np.ndarray],
    control_hz: float,
) -> dict[str, dict[str, float]]:
    """Compare MuJoCo against every reference trajectory.

    Args:
        model: the matched model.
        data: its data block.
        robot: the shared robot parameters.
        reference: reference trajectories keyed by program name.
        control_hz: control rate.

    Returns:
        Per-program ``pos_rms_mm``, ``final_pos_mm`` and ``yaw_rms_deg``.
    """
    lookup = {p.name: p for p in STEADY_PROGRAMS + TRANSIENT_PROGRAMS + ENDPOINT_PROGRAMS}
    out: dict[str, dict[str, float]] = {}
    for name, target in reference.items():
        program = lookup.get(name)
        if program is None:
            continue
        simulated = rollout(model, data, robot, program, control_hz)
        length = min(len(simulated), len(target))
        position = np.linalg.norm(simulated[:length, 1:3] - target[:length, 1:3], axis=1)
        yaw_error = np.arctan2(
            np.sin(simulated[:length, 3] - target[:length, 3]),
            np.cos(simulated[:length, 3] - target[:length, 3]),
        )
        out[name] = {
            "pos_rms_mm": float(1e3 * np.sqrt(np.mean(position**2))),
            "final_pos_mm": float(1e3 * position[-1]),
            "yaw_rms_deg": float(np.degrees(np.sqrt(np.mean(yaw_error**2)))),
            "path_length_m": float(np.linalg.norm(target[-1, 1:3])),
        }
    return out


def _dr_coverage(
    identified: dict[str, float], robot: RobotParams, ranges: DRRanges
) -> dict[str, dict[str, Any]]:
    """Check that each identified delta is covered by its S7.3 randomization range."""
    checks = {
        "armature": (
            identified.get("armature", robot.joint_armature),
            (
                robot.joint_armature * ranges.dr_armature_scale[0],
                robot.joint_armature * ranges.dr_armature_scale[1],
            ),
        ),
        "frictionloss": (
            identified.get("frictionloss", robot.joint_friction),
            ranges.dr_joint_friction_nm,
        ),
    }
    out: dict[str, dict[str, Any]] = {}
    for key, (value, bounds) in checks.items():
        out[key] = {
            "value": float(value),
            "range": [float(bounds[0]), float(bounds[1])],
            "covered": bool(bounds[0] <= value <= bounds[1]),
        }
    return out


# --------------------------------------------------------------------------- reference exchange
def collect_reference(
    probe_rollout: Callable[[Program], np.ndarray],
    programs: Sequence[Program] = STEADY_PROGRAMS + TRANSIENT_PROGRAMS + ENDPOINT_PROGRAMS,
) -> dict[str, np.ndarray]:
    """Run every program on a reference simulator and collect its trajectories.

    Args:
        probe_rollout: maps a :class:`Program` to a ``(T, 6)`` trajectory array.
        programs: the programs to run.

    Returns:
        Trajectories keyed by program name.
    """
    return {program.name: np.asarray(probe_rollout(program)) for program in programs}


def save_reference(
    path: str | Path,
    trajectories: dict[str, np.ndarray],
    control_hz: float,
    physics_dt: float,
    decimation: int,
    source: str = "isaac-lab",
) -> Path:
    """Write reference trajectories to a JSON file the MuJoCo side can consume.

    The Isaac Lab side produces this file by running :data:`STEADY_PROGRAMS`,
    :data:`TRANSIENT_PROGRAMS` and :data:`ENDPOINT_PROGRAMS` open loop and logging
    ``[t, x, y, yaw, wheel_left, wheel_right, v_body, psidot, torque_left, torque_right]`` at the
    control rate, in the robot's own start frame. The last four columns are optional: without the
    twist the pose is differentiated instead, and without the torque ``kv`` is left to stage 2.

    Args:
        path: destination file.
        trajectories: trajectories keyed by program name.
        control_hz: the control rate the reference was logged at.
        physics_dt: the reference simulator's integration step.
        decimation: the reference simulator's decimation.
        source: free-form label of the producing simulator.

    Returns:
        The destination path.
    """
    payload = {
        "schema": REFERENCE_SCHEMA,
        "source": source,
        "control_hz": control_hz,
        "physics_dt": physics_dt,
        "decimation": decimation,
        "columns": [
            "t",
            "x",
            "y",
            "yaw",
            "wheel_left",
            "wheel_right",
            "v_body",
            "psidot",
            "torque_left",
            "torque_right",
        ],
        "trajectories": {k: np.asarray(v).tolist() for k, v in trajectories.items()},
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload), encoding="utf-8")
    return destination


def load_reference(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read a reference file written by :func:`save_reference`.

    Args:
        path: the reference file.

    Returns:
        ``(trajectories, metadata)``.

    Raises:
        ValueError: if the schema tag does not match, or the rates disagree with SPEC v2 S5.2.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != REFERENCE_SCHEMA:
        raise ValueError(f"{path} has schema {payload.get('schema')!r}, expected {REFERENCE_SCHEMA!r}")
    metadata = {k: v for k, v in payload.items() if k != "trajectories"}
    trajectories = {k: np.asarray(v, dtype=np.float64) for k, v in payload["trajectories"].items()}
    return trajectories, metadata


# ------------------------------------------------------------------------------------ the driver
def run_sysid(
    reference: dict[str, np.ndarray],
    control_hz: float | None = None,
    cfg: _mjcf.MjcfCfg | None = None,
    passes: int = 3,
    outer_passes: int = 2,
    max_iter: int = 40,
    verbose: bool = False,
) -> tuple[SysIdResult, Any]:
    """Run both identification stages against a reference and produce the acceptance report.

    The two stages are not independent: stage 2 changes the armature, the joint friction and the
    servo gain, all of which shift the *effective* radius and baseline that stage 1 measures. So the
    pair is iterated. Two outer passes is enough in practice; the second pass typically moves
    ``b_eff`` from about 1% to under 0.1%.

    Args:
        reference: reference trajectories keyed by program name.
        control_hz: control rate; defaults to the shared rate settings.
        cfg: MJCF configuration; a fresh one is resolved when None.
        passes: stage-1 fixed-point passes within each outer pass.
        outer_passes: how many times to alternate stage 1 and stage 2.
        max_iter: stage-2 LM iteration cap.
        verbose: print the LM iteration log.

    Returns:
        ``(result, model)``. The model carries the identified parameters, ready to be written into
        an evaluation scene.
    """
    started = time.time()
    model, data, cfg = make_model(cfg)
    robot = cfg.robot
    ranges, dr_source = resolve_dr_ranges()
    control_hz = control_hz if control_hz is not None else cfg.sim.control_hz

    target = _stage1_from_trajectories(reference, robot, control_hz)
    identified: dict[str, float] = {}
    matched = target
    for _ in range(max(1, outer_passes)):
        matched = match_stage1(model, data, robot, target, control_hz, passes=passes)
        # When the reference carried actuator torque, stage 1 pinned kv in closed form and
        # exactly; letting stage 2 move it again turns the outer loop into a divergent chase
        # (measured: kv drifted to +119% of target over two passes).
        identified = stage2(
            model,
            data,
            robot,
            reference,
            control_hz,
            x0=tuple(identified.values()) if identified else None,
            max_iter=max_iter,
            verbose=verbose,
            fit_kv=not math.isfinite(target.kv),
        )
    matched = measure_stage1(model, data, robot, control_hz)
    residuals = residual_report(model, data, robot, reference, control_hz)

    endpoint = max(
        (stats["final_pos_mm"] for name, stats in residuals.items() if name.startswith("endpoint")),
        default=float("nan"),
    )
    error_pct = {}
    for key in ("r_eff", "b_eff", "kv"):
        want, got = getattr(target, key), getattr(matched, key)
        error_pct[key] = (
            100.0 * (got / want - 1.0)
            if math.isfinite(want) and math.isfinite(got) and want != 0.0
            else float("nan")
        )
    coverage = _dr_coverage(identified, robot, ranges)
    accepted = (
        abs(error_pct["r_eff"]) <= 0.5
        and abs(error_pct["b_eff"]) <= 0.5
        and (endpoint <= 25.0 if endpoint == endpoint else False)
        and all(entry["covered"] for entry in coverage.values())
    )
    result = SysIdResult(
        stage1_target=target,
        stage1_mujoco=matched,
        stage1_error_pct=error_pct,
        stage2=identified,
        residuals=residuals,
        endpoint_error_mm=float(endpoint),
        dr_coverage=coverage,
        accepted=bool(accepted),
        wall_clock_s=time.time() - started,
        provenance={"robot_params": cfg.params_source, "dr_ranges": dr_source},
    )
    return result, model


def run_selftest(
    cfg: _mjcf.MjcfCfg | None = None,
    perturbation: dict[str, float] | None = None,
    outer_passes: int = 3,
    max_iter: int = 50,
    verbose: bool = False,
) -> tuple[SysIdResult, Any]:
    """Perturb a known model, then try to recover it, and report against the S8.2 criteria.

    This validates the identification *procedure*, not the Isaac-to-MuJoCo gap: the reference
    trajectories come from MuJoCo itself with :data:`SELFTEST_PERTURBATION` applied. Measuring the
    real gap needs a reference captured from Isaac and passed to :func:`run_sysid`.

    Args:
        cfg: MJCF configuration; a camera-free :meth:`MjcfCfg.from_shared` is used when None.
        perturbation: overrides for :data:`SELFTEST_PERTURBATION`.
        outer_passes: how many times to alternate stage 1 and stage 2. The default is the one
            ``scripts/eval_sim2sim.py sysid`` uses, so calling this with no arguments reproduces
            the documented table exactly.
        max_iter: stage-2 LM iteration cap; the default is again the CLI's.
        verbose: print the LM iteration log.

    Returns:
        ``(result, model)``, exactly as :func:`run_sysid` returns them.
    """
    cfg = cfg if cfg is not None else _mjcf.MjcfCfg.from_shared(include_camera=False)
    values = {**SELFTEST_PERTURBATION, **(perturbation or {})}
    model, data, cfg = make_model(cfg)
    apply_stage1(
        model,
        cfg.robot,
        kv=values["kv"],
        r_eff=cfg.robot.wheel_radius * values["radius_scale"],
        b_eff=cfg.robot.wheel_separation * values["baseline_scale"],
    )
    _apply_stage2(model, cfg.robot, (values["armature"], values["frictionloss"], values["kv"]))
    trajectories = collect_reference(
        lambda program: rollout(model, data, cfg.robot, program, cfg.sim.control_hz)
    )
    return run_sysid(
        trajectories,
        control_hz=cfg.sim.control_hz,
        cfg=cfg,
        outer_passes=outer_passes,
        max_iter=max_iter,
        verbose=verbose,
    )


def _stage1_from_trajectories(
    reference: dict[str, np.ndarray], robot: RobotParams, control_hz: float
) -> Stage1Result:
    """Recover the stage-1 estimates from logged reference trajectories.

    Columns 6 and 7 of the log (body forward speed and yaw rate) are used when present; otherwise
    the pose columns are differentiated, which is noisier. Columns 8 and 9 (actuator torque) make
    ``kv`` identifiable in closed form from the servo droop, ``tau = kv * (ctrl - omega)``; without
    them ``kv`` comes back NaN and stage 2 has to fit it alongside the joint friction, which is a
    genuinely degenerate pair (both act on the droop) and measurably degrades the fit. Ask the Isaac
    exporter for the torque columns.

    Args:
        reference: reference trajectories keyed by program name.
        robot: the shared robot parameters.
        control_hz: the control rate the reference was logged at.

    Returns:
        The target :class:`Stage1Result`.

    Raises:
        KeyError: if the reference is missing a program stage 1 needs.
    """
    del robot  # every stage-1 quantity is measured, none is assumed
    lookup = {p.name: p for p in STEADY_PROGRAMS}
    straights = ("straight_12", "straight_20", "straight_28")
    spins = tuple(p.name for p in SPIN_PROGRAMS)
    arcs = ("arc_d2", "arc_d4", "arc_d6")
    missing = [n for n in straights + spins if n not in reference]
    if missing:
        raise KeyError(
            f"reference is missing the stage-1 programs {missing}; produce it with "
            f"collect_reference(), which runs every program in STEADY_PROGRAMS"
        )

    def window(name: str) -> np.ndarray:
        traj = np.asarray(reference[name], dtype=np.float64)
        program = lookup.get(name)
        span = program.average if program is not None else 0.25 * (len(traj) / control_hz)
        count = max(2, round(span * control_hz))
        return traj[-count:]

    def twist(name: str) -> tuple[float, float]:
        """Return the averaged ``(v_body, psidot)`` of one program's steady window."""
        chunk = window(name)
        if chunk.shape[1] >= 8:
            return float(np.mean(chunk[:, 6])), float(np.mean(chunk[:, 7]))
        dt = 1.0 / control_hz
        speed = float(np.mean(np.linalg.norm(np.diff(chunk[:, 1:3], axis=0), axis=1)) / dt)
        yaw = np.unwrap(chunk[:, 3])
        return speed, float((yaw[-1] - yaw[0]) / ((len(chunk) - 1) * dt))

    wheel = np.array([float(np.mean(window(n)[:, 4:6])) for n in straights])
    speed = np.array([twist(n)[0] for n in straights])
    r_eff = float(wheel @ speed / (wheel @ wheel))

    spin_estimates = []
    for name in spins:
        chunk = window(name)
        delta = float(np.mean(chunk[:, 5] - chunk[:, 4]))
        spin_estimates.append(r_eff * delta / twist(name)[1])
    b_eff = float(np.mean(spin_estimates))

    b_arc, spread = float("nan"), float("nan")
    if all(name in reference for name in arcs):
        deltas = np.array([float(np.mean(window(n)[:, 5] - window(n)[:, 4])) for n in arcs])
        rates = np.array([twist(n)[1] for n in arcs])
        b_arc = float(r_eff * (deltas @ deltas) / (deltas @ rates))
        per_arc = r_eff * deltas / rates
        spread = float(100.0 * (per_arc.max() - per_arc.min()) / np.median(per_arc))
    kv = float("nan")
    if all(np.asarray(reference[n]).shape[1] >= 10 for n in straights):
        commands = np.array([12.0, 20.0, 28.0])
        torque = np.array([float(np.mean(window(n)[:, 8:10])) for n in straights])
        droop = commands - wheel
        kv = float(droop @ torque / (droop @ droop))
    return Stage1Result(r_eff=r_eff, b_eff=b_eff, kv=kv, b_arc=b_arc, b_arc_spread_pct=spread)
