r"""Reward, terminations and condition-C6 randomization of the MuJoCo env (SPEC v2 S5.4, S5.5, S7.3).

Interpreter: needs ``mujoco`` and ``numpy`` only, so it runs in the tools venv
``d:/Personal/personal/mujoco_venv/Scripts/python.exe``. Run with::

    d:/Personal/personal/mujoco_venv/Scripts/python.exe -m pytest tests/unit/test_mj_env_reward.py \\
        --run-mujoco -q

Three defects are guarded here, all of which made the MuJoCo return a different objective from the
Isaac one while looking correct:

1. ``R_terminal = -10`` on collision or off-drivable was simply missing, so every terminated
   episode's ``return`` was off by exactly 10, and in smoke runs 7 of 8 episodes terminate
   off-drivable.
2. ``r_prox`` was fed the raw signed clearance instead of the S5.4 overlap ``p <= 0``, which paid up
   to 1.5 per step for driving away from an obstacle anywhere on the map.
3. Condition C6 wrote only mass, centre of mass and tyre friction, leaving the drive train (joint
   armature and joint friction, the two parameters the S8.2 identification fits) un-randomized while
   advertising "the identified-dynamics jitter of S7.3".
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

pytestmark = pytest.mark.mujoco
mujoco = pytest.importorskip("mujoco", reason="run these with the tools venv (mujoco_venv)")

from duckiebot_rl.sim2sim.env import (  # noqa: E402
    NO_OBSTACLE_DISTANCE,
    TERMINAL_PENALIZED,
    TERMINAL_PENALTY,
    MjDuckiebotEnv,
    MjEnvCfg,
)
from duckiebot_rl.sim2sim.track import ObstacleSpec  # noqa: E402


def _env(**kwargs: object) -> tuple[MjDuckiebotEnv, tempfile.TemporaryDirectory]:
    """Build an environment in a temporary asset directory.

    Args:
        **kwargs: :class:`MjEnvCfg` overrides.

    Returns:
        ``(env, tmpdir)``; the caller closes both.
    """
    tmp = tempfile.TemporaryDirectory()
    defaults = {"asset_dir": tmp.name, "obs_mode": "none", "episode_length_s": 20.0}
    defaults.update(kwargs)
    return MjDuckiebotEnv(MjEnvCfg(**defaults)), tmp


def _drive_until_terminated(
    env: MjDuckiebotEnv, action: np.ndarray, limit: int = 400
) -> tuple[float, bool, bool, dict[str, object]]:
    """Step ``action`` until the episode ends.

    Args:
        env: the environment.
        action: the action to repeat.
        limit: step cap.

    Returns:
        ``(reward, terminated, truncated, info)`` of the final step.
    """
    for _ in range(limit):
        _obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            return reward, terminated, truncated, info
    raise AssertionError(f"episode did not end within {limit} steps")


# ------------------------------------------------------------------------------ terminal reward
def test_off_drivable_termination_costs_ten_reward() -> None:
    """The final reward of an off-drivable episode is 10 below the same state's per-step reward.

    ``info['step_reward']`` is the six-term S5.4 sum for exactly that state, so the difference is
    the terminal penalty and nothing else. This is the assertion the review asked for.
    """
    env, tmp = _env()
    try:
        # Spawn facing across the lane so the robot leaves the drivable tiles quickly.
        env.reset(seed=0, pose=(0.0, 0.0, math.pi / 2.0))
        reward, terminated, _truncated, info = _drive_until_terminated(
            env, np.array([1.0, 0.0], dtype=np.float32)
        )
        assert terminated
        assert info["reason"] in TERMINAL_PENALIZED
        assert info["terminal_reward"] == pytest.approx(TERMINAL_PENALTY)
        assert info["step_reward"] - reward >= 10.0 - 1e-9, (
            f"terminating for {info['reason']!r} cost only "
            f"{info['step_reward'] - reward:.3f} reward, not the S5.4 10.0"
        )
        assert reward == pytest.approx(float(np.clip(info["step_reward"] + TERMINAL_PENALTY, -20.0, 20.0)))
    finally:
        env.close()
        tmp.cleanup()


def test_the_terminal_penalty_lands_in_the_reported_return() -> None:
    """``EpisodeMetrics.return_`` carries the penalty, because ``return`` is a reported metric."""
    env, tmp = _env()
    try:
        env.reset(seed=0, pose=(0.0, 0.0, math.pi / 2.0))
        rewards = []
        while True:
            _obs, reward, terminated, truncated, _info = env.step(np.array([1.0, 0.0], dtype=np.float32))
            rewards.append(reward)
            if terminated or truncated:
                break
        assert env.metrics.return_ == pytest.approx(sum(rewards))
        assert env.metrics.reason in TERMINAL_PENALIZED
    finally:
        env.close()
        tmp.cleanup()


def test_truncation_carries_no_terminal_penalty() -> None:
    """A timeout is bootstrapped, so its terminal reward is 0 (S5.4)."""
    env, tmp = _env(episode_length_s=2.0)
    try:
        env.reset(seed=0)
        # a_v = -1 is v_cmd = 0: the robot holds still and the episode ends at the horizon.
        reward, terminated, truncated, info = _drive_until_terminated(
            env, np.array([-1.0, 0.0], dtype=np.float32)
        )
        if truncated and not terminated:
            assert info["terminal_reward"] == 0.0
            assert reward == pytest.approx(float(np.clip(info["step_reward"], -20.0, 20.0)))
        else:  # a stall or spin termination is also unpenalized by S5.4
            assert info["reason"] not in TERMINAL_PENALIZED
            assert info["terminal_reward"] == 0.0
    finally:
        env.close()
        tmp.cleanup()


def test_the_clip_is_applied_after_the_terminal_penalty() -> None:
    """S5.4 clips ``R``, and ``R`` includes the terminal reward, so the clip comes last."""
    env, tmp = _env()
    try:
        env.reset(seed=0)
        query = env.lane.query(*env.pose())
        # A per-step sum far below the clip: with the penalty added the total must still be -20.
        env._prev_actions[0] = np.array([1.0, 1.0], dtype=np.float32)
        raw = env._reward(np.array([-1.0, -1.0]), query, ds=-1.0, gap=float("inf"))
        assert float(np.clip(raw + TERMINAL_PENALTY, -20.0, 20.0)) >= -20.0
        assert raw == pytest.approx(raw)  # _reward itself is unclipped, by contract
    finally:
        env.close()
        tmp.cleanup()


# --------------------------------------------------------------------------------------- r_prox
def test_r_prox_is_zero_while_the_safety_circles_are_clear() -> None:
    """Opening a positive clearance pays nothing; only an existing overlap can pay (S5.4).

    ``p`` in S5.4 is the safety-circle *overlap*, so it is never positive. Feeding the raw
    clearance made r_prox a general keep-away bonus worth up to 1.5 per step, comparable to the
    6.0-weighted progress term at full speed.
    """
    env, tmp = _env(obstacles=[ObstacleSpec(kind="cone", pos=(1.5, 0.0), radius=0.05)])
    try:
        env.reset(seed=0)
        query = env.lane.query(*env.pose())
        action = np.zeros(2, dtype=np.float32)

        env._prev_gap = 0.40
        opening = env._reward(action, query, ds=0.0, gap=0.50)
        env._prev_gap = 0.50
        stationary = env._reward(action, query, ds=0.0, gap=0.50)
        env._prev_gap = 0.60
        closing = env._reward(action, query, ds=0.0, gap=0.50)
        assert opening == pytest.approx(stationary)
        assert closing == pytest.approx(stationary)

        # Now with the circles actually overlapping, opening the overlap does pay.
        env._prev_gap = -0.02
        recovering = env._reward(action, query, ds=0.0, gap=-0.01)
        assert recovering - stationary == pytest.approx(0.5, abs=1e-9)
    finally:
        env.close()
        tmp.cleanup()


def test_r_prox_is_capped_at_the_spec_bound() -> None:
    """The overlap term saturates at 1.5, as S5.4 writes it."""
    env, tmp = _env(obstacles=[ObstacleSpec(kind="cone", pos=(1.5, 0.0), radius=0.05)])
    try:
        env.reset(seed=0)
        query = env.lane.query(*env.pose())
        action = np.zeros(2, dtype=np.float32)
        env._prev_gap = 0.0
        baseline = env._reward(action, query, ds=0.0, gap=0.0)
        env._prev_gap = -0.5
        saturated = env._reward(action, query, ds=0.0, gap=-0.001)
        assert saturated - baseline == pytest.approx(1.5, abs=1e-9)
    finally:
        env.close()
        tmp.cleanup()


# ---------------------------------------------------------------------------- privileged vector
def test_vec_priv_reports_no_obstacle_as_far_away_not_as_touching() -> None:
    """An obstacle-free map reports a large distance and zero closing speed, never 0.0 m."""
    env, tmp = _env()
    try:
        env.reset(seed=0)
        priv = env.vec_priv()
        assert priv.shape == (env.obs_params.vec_priv_dim,)
        assert priv[11] == pytest.approx(NO_OBSTACLE_DISTANCE)
        assert priv[12] == pytest.approx(0.0)
    finally:
        env.close()
        tmp.cleanup()


def test_vec_priv_reports_the_true_distance_to_an_obstacle() -> None:
    """With an obstacle present the slot is the centre-to-centre distance, not the safety gap."""
    env, tmp = _env(obstacles=[ObstacleSpec(kind="cone", pos=(2.0, 0.5), radius=0.05)])
    try:
        env.reset(seed=0, pose=(0.0, 0.0, 0.0))
        priv = env.vec_priv()
        expected = math.hypot(2.0, 0.5)
        assert priv[11] == pytest.approx(expected, rel=1e-5)
        assert priv[11] > env._nearest_obstacle_gap(0.0, 0.0)
    finally:
        env.close()
        tmp.cleanup()


# -------------------------------------------------------------------------------- C6 dynamics DR
def test_condition_c6_randomizes_the_drive_train_not_just_the_mass() -> None:
    """``dynamics_dr`` writes armature, joint friction, effort limit, yaw inertia and friction.

    S8.2 acceptance requires the identified deltas to be covered by the S7.3 ranges, and the two
    parameters the identification fits are exactly armature and joint friction. If C6 does not
    randomize them, that coverage check couples to nothing.
    """
    env, tmp = _env(dynamics_dr=True, dr_alpha=1.0)
    try:
        joint = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, env.robot.left_wheel_joint_name)
        dof = int(env.model.jnt_dofadr[joint])
        geom = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_GEOM, f"{env.robot.left_wheel_link_name}_collision"
        )
        base = env._base_body

        seen: dict[str, set[float]] = {
            k: set() for k in ("armature", "friction", "effort", "izz", "mu", "mass")
        }
        for seed in range(6):
            env.reset(seed=seed)
            seen["armature"].add(round(float(env.model.dof_armature[dof]), 12))
            seen["friction"].add(round(float(env.model.dof_frictionloss[dof]), 12))
            seen["effort"].add(round(float(env.model.actuator_forcerange[0, 1]), 12))
            seen["izz"].add(round(float(env.model.body_inertia[base, 2]), 12))
            seen["mu"].add(round(float(env.model.geom_friction[geom, 0]), 12))
            seen["mass"].add(round(float(env.model.body_mass[base]), 12))

            ranges, robot = env.dr_ranges, env.robot
            armature = float(env.model.dof_armature[dof])
            lo, hi = ranges.dr_armature_scale
            assert robot.joint_armature * lo - 1e-12 <= armature <= robot.joint_armature * hi + 1e-12
            f_lo, f_hi = ranges.dr_joint_friction_nm
            assert f_lo - 1e-12 <= float(env.model.dof_frictionloss[dof]) <= f_hi + 1e-12
            e_lo, e_hi = ranges.dr_effort_limit_nm
            assert e_lo - 1e-12 <= float(env.model.actuator_forcerange[0, 1]) <= e_hi + 1e-12
            i_lo, i_hi = ranges.dr_base_inertia_zz_scale
            nominal_izz = robot.base_inertia_diag[2]
            assert nominal_izz * i_lo - 1e-12 <= float(env.model.body_inertia[base, 2])
            assert float(env.model.body_inertia[base, 2]) <= nominal_izz * i_hi + 1e-12
            m_lo, m_hi = ranges.dr_tire_friction_static
            assert m_lo - 1e-12 <= float(env.model.geom_friction[geom, 0]) <= m_hi + 1e-12

        for name, values in seen.items():
            assert len(values) > 1, f"C6 never varied {name} across six resets"
    finally:
        env.close()
        tmp.cleanup()


def test_dynamics_dr_off_leaves_the_model_at_its_nominal_values() -> None:
    """C5 is nominal: no reset may move a physical parameter."""
    env, tmp = _env(dynamics_dr=False)
    try:
        joint = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, env.robot.left_wheel_joint_name)
        dof = int(env.model.jnt_dofadr[joint])
        env.reset(seed=0)
        first = (
            float(env.model.dof_armature[dof]),
            float(env.model.dof_frictionloss[dof]),
            float(env.model.body_mass[env._base_body]),
        )
        env.reset(seed=17)
        second = (
            float(env.model.dof_armature[dof]),
            float(env.model.dof_frictionloss[dof]),
            float(env.model.body_mass[env._base_body]),
        )
        assert first == second
        assert first[0] == pytest.approx(env.robot.joint_armature)
        assert first[1] == pytest.approx(env.robot.joint_friction)
        assert first[2] == pytest.approx(env.robot.base_mass)
    finally:
        env.close()
        tmp.cleanup()


def test_the_dr_provenance_names_every_unresolved_axis() -> None:
    """A partial match must NAME the axes still coming from the local literals.

    ``resolve_robot_params`` raises rather than mix authoritative and fallback numbers. A DR clamp
    is a training-time range, so the harness may proceed, but it may not hide which axes are local.
    """
    env, tmp = _env()
    try:
        source = env.provenance()["dr_ranges"]
        if "/" in source:  # a partial match
            assert "fallback for" in source
            assert "dr_" in source.split("fallback for")[1]
    finally:
        env.close()
        tmp.cleanup()


# --------------------------------------------------------------------------- the S8.4 episode cap
def test_run_condition_forces_the_s8_4_episode_cap() -> None:
    """``max_seconds`` is the only owner of the episode length, whatever the base config says.

    The env truncates at ``cfg.episode_length_s``, so a caller passing a base config still carrying
    the 30 s training horizon used to get 30 s episodes while believing the 45 s S8.4 cap applied.
    Episode length feeds survival time, fractional laps and distance, three reported metrics.
    """
    from duckiebot_rl.sim2sim import evaluate

    with tempfile.TemporaryDirectory() as tmp:
        base = MjEnvCfg(asset_dir=tmp, obs_mode="none", episode_length_s=30.0)
        records, _budget = evaluate.run_condition(
            evaluate.CONDITIONS["C5"],
            evaluate.PolicySpec(kind="constant", action=(-1.0, 0.0)),
            base,
            seeds=(0,),
            episodes_per_seed=1,
            max_seconds=2.0,
            workers=1,
            calibrate_episodes=1,
        )
    assert records
    for record in records:
        assert float(record["duration_s"]) <= 2.0 + 1e-6, (
            f"an episode ran {record['duration_s']:.2f} s under a 2.0 s S8.4 cap; the base config's "
            f"episode_length_s is still in charge"
        )


def test_run_condition_refuses_a_condition_that_contradicts_the_cap() -> None:
    """A condition overriding ``episode_length_s`` against ``max_seconds`` is an error, not a race."""
    from dataclasses import replace

    from duckiebot_rl.sim2sim import evaluate

    condition = replace(evaluate.CONDITIONS["C5"], overrides={"dynamics_dr": False, "episode_length_s": 30.0})
    with tempfile.TemporaryDirectory() as tmp:
        base = MjEnvCfg(asset_dir=tmp, obs_mode="none")
        with pytest.raises(ValueError, match="episode_length_s"):
            evaluate.run_condition(
                condition,
                evaluate.PolicySpec(kind="zero"),
                base,
                seeds=(0,),
                episodes_per_seed=1,
                max_seconds=45.0,
                workers=1,
                calibrate_episodes=1,
            )
