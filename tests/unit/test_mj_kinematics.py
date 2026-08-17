r"""Differential-drive parity tests for the MuJoCo model (SPEC v2 S8, owner ``[sim2sim]``).

This is the test that matters. Everything else in the sim-to-sim package is plumbing around one
claim: that the MuJoCo Duckiebot moves the way a differential drive is supposed to move, and
therefore the way a correctly configured PhysX Duckiebot moves. The reference is the closed-form
kinematic model, evaluated at the *measured* wheel speeds so that servo droop is not counted as a
modelling error::

    v      = r * (omega_L + omega_R) / 2
    psidot = r * (omega_R - omega_L) / b
    R      = (b / 2) * (omega_R + omega_L) / (omega_R - omega_L)

The two locked model decisions are exercised as ablations rather than merely asserted structurally,
because a structural assertion cannot tell you that the finding still holds after somebody changes a
solver setting:

* :func:`test_cylinder_wheel_contact_destroys_yaw` rebuilds the model with cylinder wheel colliders
  and shows the yaw error blowing up: 15.9% against the sphere's 0.04% on the (10, 30) rad/s arc at
  the SPEC v2 parameters and dt 1/240, a factor of 400. The research-phase headline of -74% was
  measured at the v1 parameters and does not reproduce; the finding does, at this magnitude.
* :func:`test_integrator_choice_is_locked_and_its_margin_measured` switches the compiled model to
  ``Euler`` and measures how far it moves. The v1 claim of -87% forward speed does **not** reproduce
  against the corrected SPEC v2 armature; the test documents the real number and keeps the lock for
  the structural reason instead. See its docstring.

Interpreter: only ``mujoco`` and ``numpy`` are needed, so these run today in the tools venv
``d:/Personal/personal/mujoco_venv/Scripts/python.exe``. Run with::

    d:/Personal/personal/mujoco_venv/Scripts/python.exe -m pytest tests/unit/test_mj_kinematics.py \\
        --run-mujoco -q
"""

from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

#: The project's opt-in marker (conftest.py, pyproject.toml). Without it these tests could not be
#: SELECTED by `pytest -m mujoco --run-mujoco`, which is the mechanism CI uses to run them on a
#: runner that has the mujoco wheel installed; they would only ever run when a human remembered to
#: point the tools venv at this file. The importorskip below stays as a belt-and-braces guard for
#: anyone who runs the file directly.
pytestmark = pytest.mark.mujoco

mujoco = pytest.importorskip("mujoco", reason="run these with the tools venv (mujoco_venv)")

from duckiebot_rl.sim2sim import mjcf as _mjcf  # noqa: E402

# Agreement with the kinematic model, measured on this configuration: 0.07% worst case over
# straights, gentle arcs, tight arcs and a spin. The gate is set an order of magnitude looser so it
# catches a broken model without failing on solver noise.
RELATIVE_TOLERANCE = 0.02


@dataclass(frozen=True)
class Response:
    """Averaged steady-state response of one open-loop command.

    Attributes:
        v: measured body-frame forward speed in m/s.
        psidot: measured yaw rate in rad/s.
        wheel_left: measured left wheel speed in rad/s.
        wheel_right: measured right wheel speed in rad/s.
        drift: lateral displacement from the initial heading line, in metres.
        travel: total distance travelled, in metres.
    """

    v: float
    psidot: float
    wheel_left: float
    wheel_right: float
    drift: float
    travel: float

    @property
    def kinematic_v(self) -> float:
        """Forward speed the differential-drive model predicts from the measured wheel speeds."""
        return _RADIUS * (self.wheel_left + self.wheel_right) / 2.0

    @property
    def kinematic_psidot(self) -> float:
        """Yaw rate the differential-drive model predicts from the measured wheel speeds."""
        return _RADIUS * (self.wheel_right - self.wheel_left) / _BASELINE

    @property
    def kinematic_radius(self) -> float:
        """Turn radius the differential-drive model predicts, in metres."""
        delta = self.wheel_right - self.wheel_left
        if abs(delta) < 1e-9:
            return math.inf
        return 0.5 * _BASELINE * (self.wheel_right + self.wheel_left) / delta

    @property
    def measured_radius(self) -> float:
        """Turn radius implied by the measured twist, in metres."""
        if abs(self.psidot) < 1e-9:
            return math.inf
        return self.v / self.psidot


_CFG = _mjcf.MjcfCfg.from_shared(include_camera=False)
_RADIUS = _CFG.robot.wheel_radius
_BASELINE = _CFG.robot.wheel_separation


def _drive(model, left: float, right: float, seconds: float = 4.0, tail: float = 0.4) -> Response:
    """Drive both wheels open loop and average the trailing part of the run.

    Args:
        model: a compiled model.
        left: left wheel command in rad/s.
        right: right wheel command in rad/s.
        seconds: run length.
        tail: fraction of the run to average over.

    Returns:
        The averaged :class:`Response`.
    """
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    decimation = _CFG.sim.decimation
    steps = round(seconds * _CFG.sim.control_hz)
    start = int((1.0 - tail) * steps)
    samples: list[tuple[float, ...]] = []
    path = 0.0
    previous = np.array([float(data.qpos[0]), float(data.qpos[1])])
    for index in range(steps):
        data.ctrl[:] = (left, right)
        for _ in range(decimation):
            mujoco.mj_step(model, data)
        current = np.array([float(data.qpos[0]), float(data.qpos[1])])
        path += float(np.linalg.norm(current - previous))
        previous = current
        if index >= start:
            q = data.qpos
            cos = 1.0 - 2.0 * (q[5] ** 2 + q[6] ** 2)
            sin = 2.0 * (q[3] * q[6] + q[4] * q[5])
            norm = math.hypot(cos, sin)
            samples.append(
                (
                    (cos * data.qvel[0] + sin * data.qvel[1]) / norm,
                    float(data.qvel[5]),
                    float(data.qvel[6]),
                    float(data.qvel[7]),
                )
            )
    means = np.mean(np.asarray(samples, dtype=np.float64), axis=0)
    return Response(
        v=float(means[0]),
        psidot=float(means[1]),
        wheel_left=float(means[2]),
        wheel_right=float(means[3]),
        drift=abs(float(data.qpos[1])),
        travel=path,
    )


@pytest.fixture(scope="module")
def model():
    """A compiled robot-on-a-plane model, spawned at the origin facing +x."""
    return mujoco.MjModel.from_xml_string(_mjcf.build_robot_xml(_CFG))


# ------------------------------------------------------------------------------- straight line
@pytest.mark.parametrize("command", [12.0, 20.0, 28.0])
def test_equal_wheel_speeds_drive_straight(model, command: float) -> None:
    """Equal wheel speeds produce a straight line at the kinematically predicted speed."""
    response = _drive(model, command, command)
    assert response.v == pytest.approx(response.kinematic_v, rel=RELATIVE_TOLERANCE)
    assert abs(response.psidot) < 1e-3, "a symmetric command must not yaw"
    drift_per_metre = response.drift / max(response.travel, 1e-9)
    assert drift_per_metre < 0.02, (
        f"lateral drift {1e3 * drift_per_metre:.2f} mm/m exceeds the M1 gate of 20 mm/m"
    )


def test_wheel_speeds_track_the_command_within_servo_droop(model) -> None:
    """The velocity servo tracks its command; the residual droop is the joint friction over kv."""
    response = _drive(model, 20.0, 20.0)
    expected_droop = _CFG.robot.joint_friction / _CFG.robot.joint_damping
    droop = 20.0 - 0.5 * (response.wheel_left + response.wheel_right)
    assert 0.0 <= droop < 3.0 * expected_droop + 0.2, (
        f"droop {droop:.3f} rad/s is not explained by frictionloss/kv = {expected_droop:.3f}"
    )


# ---------------------------------------------------------------------------- constant curvature
@pytest.mark.parametrize(
    ("left", "right"),
    [(19.0, 21.0), (17.0, 23.0), (14.0, 26.0), (10.0, 30.0), (23.0, 17.0)],
)
def test_differential_command_gives_the_predicted_turn_radius(model, left: float, right: float) -> None:
    """A differential command produces the analytically predicted yaw rate and turn radius.

    This is the parity test. It is evaluated at the measured wheel speeds, so it isolates the
    contact and integration behaviour from the velocity servo's steady-state droop.
    """
    response = _drive(model, left, right)
    assert response.psidot == pytest.approx(response.kinematic_psidot, rel=RELATIVE_TOLERANCE), (
        f"yaw rate {response.psidot:.4f} rad/s against a kinematic prediction of "
        f"{response.kinematic_psidot:.4f} rad/s"
    )
    assert response.v == pytest.approx(response.kinematic_v, rel=RELATIVE_TOLERANCE)
    assert response.measured_radius == pytest.approx(response.kinematic_radius, rel=2 * RELATIVE_TOLERANCE), (
        f"turn radius {response.measured_radius:.4f} m against a kinematic prediction of "
        f"{response.kinematic_radius:.4f} m"
    )
    assert math.copysign(1.0, response.psidot) == math.copysign(1.0, right - left), (
        "a faster right wheel must turn the robot to the left (positive yaw)"
    )


def test_spin_in_place(model) -> None:
    """Opposed wheels spin the robot in place at the predicted rate with no net translation."""
    response = _drive(model, -15.0, 15.0)
    assert response.psidot == pytest.approx(response.kinematic_psidot, rel=RELATIVE_TOLERANCE)
    assert abs(response.v) < 0.05, "a pure spin must not translate"


# ------------------------------------------------------------------------------------ ablations
def _cylinder_wheel_model() -> object:
    """Return the same model with cylinder wheel colliders instead of spheres."""
    xml = _mjcf.build_robot_xml(_CFG)
    half_width = 0.5 * _CFG.robot.wheel_width
    replacement = f'<geom type="cylinder" size="{_RADIUS:.9g} {half_width:.9g}" euler="{math.pi / 2:.9g} 0 0"'
    patched, count = re.subn(
        rf'<geom type="sphere" size="{_RADIUS:.9g}" group="3" condim="3" priority="2"',
        replacement + ' group="3" condim="3" priority="2"',
        xml,
        count=1,
    )
    assert count == 1, "the wheel collision default did not match; update this ablation"
    return mujoco.MjModel.from_xml_string(patched)


def test_cylinder_wheel_contact_destroys_yaw(model) -> None:
    """Rebuilding with cylinder wheel colliders breaks differential-drive yaw.

    A cylinder makes a two-point line contact whose torsional couple fights the turn. Measured here
    at the SPEC v2 parameters, dt 1/240, on the (10, 30) rad/s arc: the sphere tracks the kinematic
    prediction to 0.04% and the cylinder is 15.9% short of it, turning at 79% of the sphere's yaw
    rate. The research-phase headline of -74% was measured at the v1 parameters and does not
    reproduce at this magnitude; the finding itself does, and the sphere is not a simplification but
    the correct model.

    The gate is relative on purpose: what has to hold is that the two contact models are far apart,
    not that the gap has a particular size, since the size moves with the friction and solver
    settings that system identification is allowed to touch.
    """
    command = (10.0, 30.0)
    sphere = _drive(model, *command)
    cylinder = _drive(_cylinder_wheel_model(), *command)
    sphere_error = abs(sphere.psidot / sphere.kinematic_psidot - 1.0)
    cylinder_error = abs(cylinder.psidot / cylinder.kinematic_psidot - 1.0)
    assert sphere_error < RELATIVE_TOLERANCE
    assert cylinder_error > 5.0 * max(sphere_error, 0.01), (
        f"cylinder wheels gave a yaw error of {100 * cylinder_error:.1f}% against the sphere's "
        f"{100 * sphere_error:.2f}%; the two are too close for this ablation to be meaningful, so "
        f"either the contact model changed or the finding no longer reproduces"
    )


def test_integrator_choice_is_locked_and_its_margin_measured(model) -> None:
    """Record how much the integrator choice actually moves the answer, and keep the lock.

    The v1 architecture justified locking ``implicitfast`` with a measured -87% forward speed and
    +133% yaw rate under ``Euler`` on a tight arc. **That finding does not reproduce against the
    SPEC v2 parameter set, and this test is the evidence.** Measured here at ``dt = 1/240`` on the
    (10, 30) rad/s arc, ``Euler`` and ``implicitfast`` agree to 0.00002% on forward speed and
    0.00001% on yaw rate. Sweeping the armature over its whole S7.3 domain-randomization range
    (x0.5 to x2.0 of 2.0e-4) and the servo gain up to 0.4 N.m.s/rad does not separate them either.

    The reason is SPEC v2 S1 item 26: v1 left the joint armature unset and the prototype used
    2.0e-5, ten times smaller than the corrected rotor-inertia-times-48-squared figure of 2.0e-4.
    Explicit integration of the ``kv`` term is stable while ``dt * kv / I_effective`` stays below
    about 2; at the v1 numbers that ratio was roughly 8, and at the v2 numbers it is roughly 0.9.
    Fixing the armature fixed the Euler blow-up as a side effect.

    The lock stays, for two reasons that do not depend on the old measurement. First, PhysX
    integrates its implicit drive implicitly, so ``implicitfast`` is the structurally matching
    choice and matching structure is the entire job of this package. Second, the margin is under a
    factor of three and system identification is free to move the armature and the gain, so the
    stability guarantee is not something to spend on.

    What this test asserts is therefore what is actually true: ``implicitfast`` reproduces the
    kinematic model, and the configuration layer refuses to build anything else.
    """
    command = (10.0, 30.0)
    reference = _drive(model, *command)
    broken = mujoco.MjModel.from_xml_string(_mjcf.build_robot_xml(_CFG))
    broken.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    euler = _drive(broken, *command)

    assert reference.v == pytest.approx(reference.kinematic_v, rel=RELATIVE_TOLERANCE)
    assert reference.psidot == pytest.approx(reference.kinematic_psidot, rel=RELATIVE_TOLERANCE)

    from duckiebot_rl.sim2sim import _resolve

    with pytest.raises(ValueError, match="implicitfast"):
        _resolve.SimParams(integrator="Euler").validate()

    # The gate is set at 2%, which is 5 orders of magnitude above the measured 0.00002% and still
    # far below anything that would matter physically. A 50% tolerance, which is what this used to
    # be, would have let a 45% forward-speed regression pass silently: a guard set four orders of
    # magnitude looser than the measurement is documentation with an assert attached.
    separation = abs(euler.v / max(reference.v, 1e-9) - 1.0)
    yaw_separation = abs(euler.psidot / max(abs(reference.psidot), 1e-9) - 1.0)
    assert separation < 0.02, (
        f"Euler and implicitfast now differ by {100 * separation:.4f}% on forward speed, against "
        f"the 0.00002% this model measures. If this fires, the model has moved back toward the "
        f"explicit-integration danger zone (check that the joint armature is still ~2.0e-4 and dt "
        f"is still 1/240) and the docstring above needs rewriting to match reality again."
    )
    assert yaw_separation < 0.02, (
        f"Euler and implicitfast now differ by {100 * yaw_separation:.4f}% on yaw rate, against "
        f"the 0.00001% this model measures."
    )


# -------------------------------------------------------------------------------- env-level check
def test_environment_reaches_the_commanded_speed() -> None:
    """The full action path, driven at maximum forward command, reaches the commanded speed.

    This closes the loop from policy action to body velocity through the S5.3 chain: inverse
    kinematics, actuation delay, dead band, brake authority and the velocity servo.
    """
    import tempfile

    from duckiebot_rl.sim2sim.env import MjDuckiebotEnv, MjEnvCfg

    with tempfile.TemporaryDirectory() as tmp:
        env = MjDuckiebotEnv(MjEnvCfg(asset_dir=tmp, obs_mode="none", episode_length_s=6.0))
        try:
            # Spawn on the right-hand lane centre of the east-bound lane through the tile at the
            # origin. The offset is read from the resolved [city] geometry, never written here as a
            # literal: a hardcoded 0.1287 m is exactly the 0.22-tile figure that put the lane graph
            # 11.7 mm off the painted lane in the first place.
            env.reset(seed=0, pose=(0.0, -env.lane.city.lane_offset_m, 0.0))
            speeds = []
            for _ in range(60):
                env.step(np.array([1.0, 0.0], dtype=np.float32))
                speeds.append(env.body_speed())
        finally:
            env.close()
    settled = float(np.mean(speeds[-20:]))
    assert settled == pytest.approx(env.robot.v_max, rel=0.15), (
        f"a full-forward action settled at {settled:.3f} m/s, not the commanded {env.robot.v_max:.3f} m/s"
    )
