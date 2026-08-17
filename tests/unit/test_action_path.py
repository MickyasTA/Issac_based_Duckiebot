"""The SPEC v2 S5.3 action path, asserted step by step (owner ``[sim2sim]``).

Interpreter: numpy only, no ``mujoco`` and no torch, so this runs in every venv and in CI.

S5.3 says the chain is "one class, with a torch-free numpy twin verified equal by unit test". The
Isaac half of that pair (``[env]``-owned) has not landed yet, so what this file can enforce today is
the other half: that :class:`duckiebot_rl.sim2sim.env.ActionPath` matches an *independent*
transcription of the six S5.3 steps, written here from the specification text rather than from the
implementation. When the Isaac twin lands, :func:`reference_action_path` is the fixture to drive it
against; the assertion becomes a three-way equality and nothing else in this file changes.

The specific defect this file was written for: the delay ring used to start EMPTY, and the index
``len(queue) - 1 - delay_steps`` then clamped to the newest entry, so the first ``delay_steps``
control steps of every episode bypassed the 0.150 s actuation delay entirely and the robot answered
its very first command at full commanded wheel speed. S2 calls that delay the dominant dynamics gap,
and the divergence landed on every reset of every episode of every condition.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from duckiebot_rl.sim2sim._resolve import DRRanges, RobotParams, resolve_robot_params  # noqa: E402
from duckiebot_rl.sim2sim.env import ActionPath  # noqa: E402

CONTROL_DT = 1.0 / 15.0


@pytest.fixture(scope="module")
def robot() -> RobotParams:
    """The shared robot parameters."""
    return resolve_robot_params()[0]


def reference_action_path(
    robot: RobotParams,
    dr,
    actions: np.ndarray,
    wheel_velocities: np.ndarray,
    slips: np.ndarray,
) -> np.ndarray:
    """Transcribe SPEC v2 S5.3 steps 1-6 directly, as a reference for the shipped implementation.

    Written from the specification text, not from :class:`ActionPath`: the point of the comparison
    is that two independent readings of S5.3 agree. The delay line is a plain list of the last
    ``delay_steps + 2`` targets, pre-filled with zeros so that step 0 of an episode sees the same
    history as step 100 of a run that has been commanding zero.

    Args:
        robot: shared robot parameters.
        dr: the per-episode randomization sample in force.
        actions: ``(T, 2)`` policy actions.
        wheel_velocities: ``(T, 2)`` measured wheel speeds fed back each step.
        slips: ``(T, 2)`` per-step slip factors, standing in for the D12 noise draw.

    Returns:
        ``(T, 2)`` wheel-velocity targets in rad/s.
    """
    queue = [np.zeros(2) for _ in range(dr.delay_steps + 2)]
    lagged = np.zeros(2)
    out = []
    for action, measured, slip in zip(actions, wheel_velocities, slips, strict=True):
        a = np.clip(action, -1.0, 1.0)
        v_cmd = 0.3 * (a[0] + 1.0)
        om_cmd = 4.0 * a[1]

        # 1. inverse kinematics with randomized gain, trim, baseline and per-wheel radius
        left = (v_cmd - 0.5 * om_cmd * dr.baseline) / dr.radius_left * dr.gain * (1.0 - dr.trim)
        right = (v_cmd + 0.5 * om_cmd * dr.baseline) / dr.radius_right * dr.gain * (1.0 + dr.trim)
        target = np.array([left, right])

        # 2. delay ring, sub-step interpolation, first-order motor lag
        queue.append(target.copy())
        del queue[0 : len(queue) - (dr.delay_steps + 2)]
        newer = queue[len(queue) - 1 - dr.delay_steps]
        older = queue[len(queue) - 2 - dr.delay_steps]
        delayed = older + (1.0 - dr.delay_substep) * (newer - older)
        lagged = lagged + dr.lag_alpha * (delayed - lagged)
        target = lagged.copy()

        # 3. dead band: below the release duty the hardware coasts
        duty = target / dr.motor_k
        for i, band in enumerate((dr.deadband_left, dr.deadband_right)):
            if abs(duty[i]) < max(band, robot.pwm_release_duty):
                target[i] = measured[i]

        # 4. brake authority
        target = np.maximum(target, measured - dr.brake_beta * robot.brake_dw_max)

        # 5. slip and battery sag
        target = target * dr.battery_sag * slip

        # 6. the velocity target, clipped to the actuator limit
        out.append(np.clip(target, -robot.velocity_limit, robot.velocity_limit))
    return np.asarray(out)


def test_the_actuation_delay_applies_from_the_very_first_step(robot: RobotParams) -> None:
    """A fresh episode holds the commanded target for ``delay_steps`` steps, then releases it.

    With the nominal 0.150 s delay at 15 Hz that is two steps of zero followed by the first
    commanded target. The empty-ring bug produced the full 18.868 rad/s target on step 0.
    """
    path = ActionPath(robot, CONTROL_DT)
    path.reset()
    assert path.dr.delay_steps == 2

    rng = np.random.default_rng(0)
    action = np.array([1.0, 0.0])
    outputs = [path(action, np.zeros(2), rng) for _ in range(4)]

    commanded = 0.5 * robot.v_max * 2.0 / robot.wheel_radius
    assert outputs[0] == pytest.approx(np.zeros(2), abs=1e-12), (
        f"step 0 already emitted {outputs[0]}, so the S5.3 actuation delay is not applied at the "
        f"start of an episode"
    )
    assert outputs[1] == pytest.approx(np.zeros(2), abs=1e-12)
    assert outputs[2] == pytest.approx(np.full(2, commanded), rel=1e-12)
    assert outputs[3] == pytest.approx(np.full(2, commanded), rel=1e-12)


def test_reset_rewarms_the_ring_between_episodes(robot: RobotParams) -> None:
    """State from the previous episode cannot leak past a reset."""
    path = ActionPath(robot, CONTROL_DT)
    path.reset()
    rng = np.random.default_rng(0)
    for _ in range(10):
        path(np.array([1.0, 0.0]), np.zeros(2), rng)
    assert path(np.array([1.0, 0.0]), np.zeros(2), rng)[0] > 1.0

    path.reset()
    assert path(np.array([1.0, 0.0]), np.zeros(2), rng) == pytest.approx(np.zeros(2), abs=1e-12)


def test_the_nominal_delay_derives_from_the_control_period(robot: RobotParams) -> None:
    """``delay_steps`` is ``actuation_delay_s / control_dt``, never a hardcoded period.

    A second copy of the control period would round to the same 2 steps at 15 Hz and silently go
    wrong the moment the decimation or the physics step moves.
    """
    assert ActionPath(robot, 1.0 / 15.0).nominal().delay_steps == 2
    assert ActionPath(robot, 1.0 / 60.0).nominal().delay_steps == 9
    assert ActionPath(robot, 1.0 / 100.0).nominal().delay_steps == 15
    expected = round(robot.actuation_delay_s / (1.0 / 15.0))
    assert ActionPath(robot, 1.0 / 15.0).nominal().delay_steps == expected


def test_a_longer_sampled_delay_is_honoured_from_step_zero(robot: RobotParams) -> None:
    """Sampling a 3-step delay warms a 3-step ring, not the ring the previous sample needed."""
    path = ActionPath(robot, CONTROL_DT)
    path.reset()
    ranges = DRRanges(dr_delay_control_steps=(3, 3), dr_delay_substep=(0.0, 0.0))
    path.sample(np.random.default_rng(0), ranges, alpha=1.0)
    assert path.dr.delay_steps == 3

    rng = np.random.default_rng(1)
    outputs = [path(np.array([1.0, 0.0]), np.zeros(2), rng) for _ in range(5)]
    for index in range(3):
        assert outputs[index] == pytest.approx(np.zeros(2), abs=1e-9), f"step {index} broke the delay"
    assert outputs[3][0] > 1.0


def test_matches_an_independent_transcription_of_spec_s5_3(robot: RobotParams) -> None:
    """The shipped chain equals a fresh reading of S5.3 over a randomized action sequence.

    Both sides get the same actions, the same DR sample and the same wheel-velocity feedback, and
    the slip draw is neutralized (``dr_wheel_slip_frac`` clamped to zero) so the comparison does not
    depend on the two sides consuming the generator identically.
    """
    path = ActionPath(robot, CONTROL_DT)
    ranges = DRRanges(dr_wheel_slip_frac=(0.0, 0.0))
    sample = path.sample(np.random.default_rng(7), ranges, alpha=1.0)
    assert sample.slip_frac == 0.0

    rng = np.random.default_rng(11)
    steps = 40
    actions = rng.uniform(-1.5, 1.5, size=(steps, 2))
    measured = rng.uniform(-5.0, 25.0, size=(steps, 2))

    got = np.asarray([path(actions[i], measured[i], rng) for i in range(steps)])
    want = reference_action_path(robot, sample, actions, measured, np.ones((steps, 2)))
    assert got == pytest.approx(want, rel=1e-12, abs=1e-12)


def test_dead_band_coasts_instead_of_braking(robot: RobotParams) -> None:
    """Below the release duty the target becomes the *current* wheel speed, which is a coast.

    Commanding zero there would be a brake, and the hardware cannot brake with the motor released.
    """
    path = ActionPath(robot, CONTROL_DT)
    ranges = DRRanges(
        dr_deadband_duty=(0.5, 0.5),
        dr_wheel_slip_frac=(0.0, 0.0),
        dr_motor_lag_alpha=(1.0, 1.0),
        dr_delay_control_steps=(0, 0),
        dr_delay_substep=(0.0, 0.0),
        dr_battery_sag=(1.0, 1.0),
        dr_brake_authority_beta=(1.0, 1.0),
        dr_motor_gain=(1.0, 1.0),
        dr_motor_trim=(0.0, 0.0),
    )
    path.sample(np.random.default_rng(0), ranges, alpha=1.0)
    rng = np.random.default_rng(0)
    measured = np.array([4.0, 4.0])
    out = path(np.array([-1.0, 0.0]), measured, rng)
    assert out == pytest.approx(measured, rel=1e-9)


def test_brake_authority_bounds_how_fast_a_target_may_fall(robot: RobotParams) -> None:
    """A full stop command cannot pull the target below ``w - beta * DW_MAX`` in one step."""
    path = ActionPath(robot, CONTROL_DT)
    ranges = DRRanges(
        dr_brake_authority_beta=(0.5, 0.5),
        dr_delay_control_steps=(0, 0),
        dr_delay_substep=(0.0, 0.0),
        dr_motor_lag_alpha=(1.0, 1.0),
        dr_wheel_slip_frac=(0.0, 0.0),
        dr_battery_sag=(1.0, 1.0),
        dr_deadband_duty=(0.0, 0.0),
        dr_motor_gain=(1.0, 1.0),
        dr_motor_trim=(0.0, 0.0),
    )
    path.sample(np.random.default_rng(0), ranges, alpha=1.0)
    rng = np.random.default_rng(0)
    measured = np.array([25.0, 25.0])
    # A command of 5 rad/s: far below the measured speed, but above the release duty, so the
    # dead-band coast of step 3 does not fire and step 4's bound is what is being measured.
    action_v = 5.0 * robot.wheel_radius / (0.5 * robot.v_max) - 1.0
    out = path(np.array([action_v, 0.0]), measured, rng)
    floor = 25.0 - 0.5 * robot.brake_dw_max
    assert out == pytest.approx(np.full(2, floor), rel=1e-9)


def test_targets_are_clipped_to_the_actuator_velocity_limit(robot: RobotParams) -> None:
    """No target ever leaves the chain outside the S2 velocity limit."""
    path = ActionPath(robot, CONTROL_DT)
    path.reset()
    rng = np.random.default_rng(3)
    for _ in range(20):
        out = path(np.array([1.0, 1.0]), np.zeros(2), rng)
        assert np.all(np.abs(out) <= robot.velocity_limit + 1e-12)
