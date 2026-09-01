"""The SPEC v2 S5.5 terminations, each fired exactly when it should (owner ``[env]``).

Every condition gets two tests: one that it fires when the specification says it must, and one
that it does NOT fire in the nearby case that looks similar. A termination that is merely
"triggerable" is not enough - a spin guard that also fires on a legitimate lap, or an
off-drivable test that only checks the robot centre, silently truncates good episodes and the
resulting return curve looks like a learning failure rather than a bug.

The stateful conditions (stall, spin) also get a test that the counters are cleared per env id on
reset, because a leaked counter terminates the FOLLOWING episode of that env for something that
happened in the previous one.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from duckiebot_rl.assets.params import DUCKIEBOT  # noqa: E402
from duckiebot_rl.city.lane_graph import BatchedLaneGraph, LaneGraph  # noqa: E402
from duckiebot_rl.city.maps import builtin_map  # noqa: E402
from duckiebot_rl.envs import terminations as tm  # noqa: E402

CONTROL_DT = DUCKIEBOT.control_dt_s


@pytest.fixture(scope="module")
def lane() -> BatchedLaneGraph:
    """A real batched lane graph, so the drivable mask comes from the production code path."""
    return BatchedLaneGraph([LaneGraph(builtin_map("loop_small"))])


@pytest.fixture(scope="module")
def drivable_pose(lane: BatchedLaneGraph) -> tuple[float, float, float]:
    """A pose safely inside a drivable tile: the midpoint of the longest straight lane segment."""
    from duckiebot_rl.envs.obstacles import lane_frame_to_world

    lengths = lane.seg_length[0].clone()
    lengths[lane.seg_is_arc[0]] = -1.0
    lengths[~lane.seg_valid[0]] = -1.0
    segment = int(torch.argmax(lengths))
    x, y, yaw = lane_frame_to_world(
        lane,
        torch.zeros(1, dtype=torch.long),
        torch.full((1,), segment, dtype=torch.long),
        lane.seg_length[0, segment].reshape(1) * 0.5,
        torch.zeros(1),
    )
    return float(x), float(y), float(yaw)


# =============================================================================================
# Condition 1: off drivable
# =============================================================================================


def test_four_test_points_are_the_spec_offsets():
    """Centre, both wheel contact points and the chassis front mid, in that order."""
    px, py = tm.test_points(torch.zeros(1), torch.zeros(1), torch.zeros(1))
    assert px.shape == (1, 4)
    half = DUCKIEBOT.half_baseline_m
    front = DUCKIEBOT.chassis_center_base_frame_m[0] + 0.5 * DUCKIEBOT.chassis_size_m[0]
    assert px[0].tolist() == pytest.approx([0.0, 0.0, 0.0, front], abs=1e-6)
    assert py[0].tolist() == pytest.approx([0.0, half, -half, 0.0], abs=1e-6)


def test_test_points_rotate_with_the_robot():
    """At 90 degrees the left wheel must be behind the robot in world -x, not in world +y."""
    px, py = tm.test_points(torch.zeros(1), torch.zeros(1), torch.tensor([math.pi / 2.0]))
    half = DUCKIEBOT.half_baseline_m
    assert float(px[0, 1]) == pytest.approx(-half, abs=1e-6)
    assert float(py[0, 1]) == pytest.approx(0.0, abs=1e-6)


def test_off_drivable_is_false_inside_a_lane(lane, drivable_pose):
    x, y, yaw = drivable_pose
    flag = tm.off_drivable(
        lane, torch.zeros(1, dtype=torch.long), torch.tensor([x]), torch.tensor([y]), torch.tensor([yaw])
    )
    assert not bool(flag[0])


def test_off_drivable_fires_far_outside_the_map(lane):
    flag = tm.off_drivable(
        lane,
        torch.zeros(1, dtype=torch.long),
        torch.tensor([50.0]),
        torch.tensor([50.0]),
        torch.tensor([0.0]),
    )
    assert bool(flag[0])


def test_off_drivable_uses_all_four_points_not_just_the_centre(lane, drivable_pose):
    """Slide sideways until one wheel leaves the tile; a centre-only test would miss it.

    The assertion is that SOME offset exists where the four-point test fires and a bare centre
    test does not. That gap is exactly what the extra three points buy, and the previous version
    of this environment had none of them.
    """
    x, y, yaw = drivable_pose
    normal_x, normal_y = -math.sin(yaw), math.cos(yaw)
    variant = torch.zeros(1, dtype=torch.long)
    found = False
    for offset in [0.01 * i for i in range(1, 80)]:
        for sign in (+1.0, -1.0):
            px = torch.tensor([x + sign * offset * normal_x])
            py = torch.tensor([y + sign * offset * normal_y])
            four = bool(tm.off_drivable(lane, variant, px, py, torch.tensor([yaw]))[0])
            centre = not bool(lane.is_drivable(variant, px, py)[0])
            if four and not centre:
                found = True
                break
        if found:
            break
    assert found, "no lateral offset distinguishes the four-point test from a centre-only test"


# =============================================================================================
# Condition 2: obstacle safety circle
# =============================================================================================


def test_obstacle_contact_fires_only_on_a_negative_gap():
    flags = tm.obstacle_contact(torch.tensor([0.05, 0.0, -1e-4, -0.2]))
    assert flags.tolist() == [False, False, True, True]


def test_obstacle_contact_is_false_on_an_obstacle_free_map():
    """An infinite gap is the sentinel for "no obstacle anywhere" and must never terminate."""
    flags = tm.obstacle_contact(torch.tensor([float("inf"), float("inf")]))
    assert not bool(flags.any())


# =============================================================================================
# Condition 3: rollover
# =============================================================================================


def test_rollover_fires_past_thirty_degrees_on_either_axis():
    roll = torch.tensor([0.0, math.radians(31.0), 0.0, -math.radians(31.0)])
    pitch = torch.tensor([math.radians(31.0), 0.0, 0.0, 0.0])
    assert tm.rollover(roll, pitch).tolist() == [True, True, False, True]


def test_rollover_does_not_fire_just_inside_the_limit():
    """29.9 degrees is a bump, not a rollover."""
    angle = torch.tensor([math.radians(29.9)])
    assert not bool(tm.rollover(angle, torch.zeros(1))[0])
    assert not bool(tm.rollover(torch.zeros(1), angle)[0])


# =============================================================================================
# Condition 4: stall
# =============================================================================================


def test_stall_fires_after_two_seconds_of_standing_still():
    """The limit is floor(2.0 / control_dt) = 30 consecutive steps at 15 Hz."""
    state = tm.TerminationState(1, CONTROL_DT)
    assert state.stall_step_limit == 30
    for step in range(1, 40):
        stall, _spin = state.update(torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1))
        assert bool(stall[0]) == (step > 30), f"step {step}"


def test_stall_counter_resets_when_the_robot_moves_again():
    """A momentary stop inside a corner must not accumulate toward a termination."""
    state = tm.TerminationState(1, CONTROL_DT)
    for _ in range(25):
        state.update(torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1))
    state.update(torch.tensor([0.3]), torch.zeros(1), torch.zeros(1), torch.zeros(1))
    assert int(state.stall_steps[0]) == 0
    for _ in range(25):
        stall, _ = state.update(torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1))
    assert not bool(stall[0])


def test_stall_uses_the_speed_magnitude():
    """Reversing at 0.2 m/s is not a stall."""
    state = tm.TerminationState(1, CONTROL_DT)
    for _ in range(40):
        stall, _ = state.update(torch.tensor([-0.2]), torch.zeros(1), torch.zeros(1), torch.zeros(1))
    assert not bool(stall[0])


# =============================================================================================
# Condition 5: spin
# =============================================================================================


def test_spin_fires_only_when_the_robot_also_stayed_put():
    """Both halves are required: 3 * pi of yaw AND under 0.2 m of net displacement."""
    state = tm.TerminationState(1, CONTROL_DT)
    state.reset(torch.zeros(1, dtype=torch.long), torch.zeros(1, 2))
    yaw_rate = torch.tensor([4.0])
    spin = torch.tensor([False])
    for _ in range(60):
        _stall, spin = state.update(torch.zeros(1), yaw_rate, torch.zeros(1), torch.zeros(1))
        if bool(spin[0]):
            break
    assert bool(spin[0])
    assert float(state.yaw_integral[0]) > tm.SPIN_YAW_LIMIT_RAD


def test_spin_does_not_fire_on_a_lap_that_actually_went_somewhere():
    """A closed loop integrates 2 pi per lap; displacement is what separates it from a pirouette."""
    state = tm.TerminationState(1, CONTROL_DT)
    state.reset(torch.zeros(1, dtype=torch.long), torch.zeros(1, 2))
    for i in range(120):
        distance = torch.tensor([0.05 * (i + 1)])
        _stall, spin = state.update(torch.tensor([0.5]), torch.tensor([4.0]), distance, torch.zeros(1))
        assert not bool(spin[0]), f"spin fired at step {i} with {float(distance)} m travelled"


def test_spin_measures_displacement_from_the_recorded_spawn():
    """The reference point is the spawn, not the origin, or every map but one is mismeasured."""
    state = tm.TerminationState(1, CONTROL_DT)
    state.reset(torch.zeros(1, dtype=torch.long), torch.tensor([[10.0, -4.0]]))
    for _ in range(60):
        _stall, spin = state.update(
            torch.zeros(1), torch.tensor([4.0]), torch.tensor([10.05]), torch.tensor([-4.05])
        )
    assert bool(spin[0])


# =============================================================================================
# Truncation and the assembled evaluation
# =============================================================================================


def test_truncation_fires_at_the_horizon_not_after_it():
    """Isaac increments episode_length_buf before _get_dones, so the test is >=."""
    buf = torch.tensor([448, 449, 450, 451])
    assert tm.truncated_by_horizon(buf, 450).tolist() == [False, False, True, True]


def test_reset_clears_only_the_named_envs():
    """A partial reset must not clear the counters of the envs that are still running."""
    state = tm.TerminationState(3, CONTROL_DT)
    for _ in range(10):
        state.update(torch.zeros(3), torch.ones(3), torch.zeros(3), torch.zeros(3))
    state.reset(torch.tensor([1]), torch.zeros(1, 2))
    assert state.stall_steps.tolist() == [10, 0, 10]
    assert float(state.yaw_integral[1]) == 0.0
    assert float(state.yaw_integral[0]) > 0.0


def test_evaluate_returns_disjoint_terminated_and_truncated(lane, drivable_pose):
    """GAE must never see a step flagged both ways; a terminated env is not also truncated."""
    x, y, yaw = drivable_pose
    state = tm.TerminationState(2, CONTROL_DT)
    state.reset(None, torch.zeros(2, 2))
    flags = state.evaluate(
        lane_graph=lane,
        variant_idx=torch.zeros(2, dtype=torch.long),
        x=torch.tensor([x, 50.0]),
        y=torch.tensor([y, 50.0]),
        yaw=torch.tensor([yaw, 0.0]),
        roll=torch.zeros(2),
        pitch=torch.zeros(2),
        body_speed=torch.tensor([0.3, 0.3]),
        yaw_rate=torch.zeros(2),
        gap=torch.full((2,), float("inf")),
        episode_length_buf=torch.tensor([450, 450]),
        max_episode_length=450,
    )
    assert flags.terminated.tolist() == [False, True]
    assert flags.truncated.tolist() == [True, False]
    assert not bool((flags.terminated & flags.truncated).any())


def test_evaluate_reports_a_per_condition_histogram(lane, drivable_pose):
    """The failure-mode histogram is an M12 release artifact; it cannot be recovered later."""
    x, y, yaw = drivable_pose
    state = tm.TerminationState(2, CONTROL_DT)
    state.reset(None, torch.zeros(2, 2))
    flags = state.evaluate(
        lane_graph=lane,
        variant_idx=torch.zeros(2, dtype=torch.long),
        x=torch.tensor([x, x]),
        y=torch.tensor([y, y]),
        yaw=torch.tensor([yaw, yaw]),
        roll=torch.tensor([0.0, math.radians(45.0)]),
        pitch=torch.zeros(2),
        body_speed=torch.tensor([0.3, 0.3]),
        yaw_rate=torch.zeros(2),
        gap=torch.tensor([-0.01, float("inf")]),
        episode_length_buf=torch.zeros(2, dtype=torch.long),
        max_episode_length=450,
    )
    counts = flags.counts()
    assert counts["obstacle"] == 1
    assert counts["rollover"] == 1
    assert counts["off_drivable"] == 0
    assert counts["terminated"] == 2


def test_state_dict_round_trips():
    """The stall and spin counters are part of what a resume must not silently rewind."""
    state = tm.TerminationState(2, CONTROL_DT)
    for _ in range(7):
        state.update(torch.zeros(2), torch.ones(2), torch.zeros(2), torch.zeros(2))
    snapshot = state.state_dict()
    restored = tm.TerminationState(2, CONTROL_DT)
    restored.load_state_dict(snapshot)
    assert restored.stall_steps.tolist() == state.stall_steps.tolist()
    assert restored.yaw_integral.tolist() == pytest.approx(state.yaw_integral.tolist())


def test_load_state_dict_rejects_an_incomplete_payload():
    state = tm.TerminationState(1, CONTROL_DT)
    with pytest.raises(KeyError, match="yaw_integral"):
        state.load_state_dict({"stall_steps": torch.zeros(1, dtype=torch.long)})


# =============================================================================================
# counts(): one host sync, not seven (profile rank 5)
# =============================================================================================


def _reference_counts(flags: tm.TerminationFlags) -> dict[str, int]:
    """Return the counts the obvious per-field implementation produced.

    Args:
        flags: A ``TerminationFlags``.

    Returns:
        ``{field_name: int}`` computed one ``.item()`` at a time.
    """
    return {name: int(mask.sum().item()) for name, mask in flags.__dict__.items()}


def _flags_from_seed(num_envs: int, seed: int) -> tm.TerminationFlags:
    """Return a ``TerminationFlags`` with pseudo-random masks.

    Args:
        num_envs: Number of environments.
        seed: Generator seed.

    Returns:
        The flags.
    """
    generator = torch.Generator().manual_seed(seed)
    masks = {
        name: torch.rand(num_envs, generator=generator) < p
        for name, p in (
            ("off_drivable", 0.2),
            ("obstacle", 0.05),
            ("rollover", 0.1),
            ("stall", 0.3),
            ("spin", 0.0),
            ("terminated", 0.4),
            ("truncated", 0.15),
        )
    }
    return tm.TerminationFlags(**masks)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("num_envs", [1, 64, 256])
def test_counts_matches_the_per_field_reference(num_envs, seed):
    """Stacking before summing changes no integer, at any env count, including all-False axes."""
    flags = _flags_from_seed(num_envs, seed)
    assert flags.counts() == _reference_counts(flags)


def test_counts_keeps_field_order():
    """``scripts/train.py:termination_reason`` iterates the condition names; order is interface."""
    flags = _flags_from_seed(8, 0)
    assert list(flags.counts()) == [
        "off_drivable",
        "obstacle",
        "rollover",
        "stall",
        "spin",
        "terminated",
        "truncated",
    ]


def test_counts_costs_one_host_transfer_not_seven(count_host_syncs):
    """Profile rank 5: this was 7 of the 23.4 synchronising calls per control step.

    ``tolist`` is the single transfer; nothing else inside ``counts`` may reach the host.
    """
    flags = _flags_from_seed(32, 5)
    with count_host_syncs() as syncs:
        flags.counts()
        assert syncs() == 1


def test_spin_does_not_fire_when_a_closed_lap_returns_to_the_spawn():
    """The regression: on a closed loop, finishing where you started is success, not a pirouette.

    ``test_spin_does_not_fire_on_a_lap_that_actually_went_somewhere`` drives in a straight line,
    so displacement from the spawn only ever grows and the old spawn-anchored guard passed it.
    Every map this project trains and evaluates on is a CLOSED loop, where the last few percent
    of a lap brings the robot back inside the 0.2 m radius of its spawn with the yaw integral
    long past 3 pi. The guard therefore fired on the finish line and charged the terminal
    penalty: measured in the S8 C5 matrix, 117 of 120 episodes died at a median 0.964 laps while
    holding 3.3 cm lane RMS.

    This walks the 8.12 m loop as a circle through the spawn, with enough corrective steering
    that the yaw integral passes the limit inside a single lap, and requires silence throughout.
    """
    radius = 8.12 / (2.0 * math.pi)
    steps = 240
    state = tm.TerminationState(1, CONTROL_DT)
    state.reset(torch.zeros(1, dtype=torch.long), torch.zeros(1, 2))

    # 4 pi of integrated yaw over one lap: 2 pi of geometry plus that much again of wiggle,
    # which is what the real rollouts accumulate on a four-corner loop.
    yaw_rate = torch.tensor([4.0 * math.pi / (steps * CONTROL_DT)])
    returned = False
    for i in range(steps + 1):
        theta = 2.0 * math.pi * i / steps
        x = torch.tensor([radius * math.sin(theta)])
        y = torch.tensor([radius * (1.0 - math.cos(theta))])
        _stall, spin = state.update(torch.tensor([0.5]), yaw_rate, x, y)
        distance_to_spawn = float(torch.sqrt(x**2 + y**2)[0])
        if i > steps // 2 and distance_to_spawn < tm.SPIN_MIN_DISPLACEMENT_M:
            returned = True
        assert not bool(spin[0]), (
            f"spin fired at step {i} of a clean lap ({distance_to_spawn:.3f} m from spawn, "
            f"yaw integral {float(state.yaw_integral[0]):.2f} rad): the guard is punishing success"
        )
    assert returned, "the fixture must actually close the loop back onto the spawn"


def test_spin_still_fires_when_the_robot_pirouettes_far_from_its_spawn():
    """The moving anchor must not blind the guard once the robot has driven somewhere.

    The complement of the test above, and the reason the fix moves the anchor rather than simply
    dropping the displacement clause: a robot that drives away and THEN starts turning on the
    spot is still spinning, and a guard anchored to the spawn forever would never notice.
    """
    state = tm.TerminationState(1, CONTROL_DT)
    state.reset(torch.zeros(1, dtype=torch.long), torch.zeros(1, 2))
    for i in range(40):  # drive out to 2 m, well clear of the spawn
        state.update(torch.tensor([0.5]), torch.tensor([0.0]), torch.tensor([0.05 * (i + 1)]), torch.zeros(1))

    spin = torch.tensor([False])
    for _ in range(200):  # now turn on the spot, parked at x = 2 m
        _stall, spin = state.update(torch.zeros(1), torch.tensor([4.0]), torch.tensor([2.0]), torch.zeros(1))
        if bool(spin[0]):
            break
    assert bool(spin[0]), "a pirouette away from the spawn must still be caught"
