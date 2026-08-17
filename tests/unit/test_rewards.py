"""The SPEC v2 S5.4 reward, term by term against hand-computed fixtures (owner ``[env]``).

Two halves.

The first half checks each term in isolation against a number worked out by hand from the
specification text, including the boundary cases (the leaky-cosine seam at exactly ``pi``, the
progress gate at exactly the lane edge, the proximity term on an obstacle-free map).

The second half is the part the research asked for: three degenerate policies that a
lane-following reward is classically hacked by - spinning in place, hugging one line, and driving
backwards - are rolled out against a REAL lane graph and must all score strictly worse than clean
lane following. A reward that only passes the per-term fixtures can still be hacked; a reward
that passes both is at least not hackable in the three ways that are already known.
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
from duckiebot_rl.city.lane_graph import BatchedLaneGraph, LaneGraph, progress_delta  # noqa: E402
from duckiebot_rl.city.maps import builtin_map  # noqa: E402
from duckiebot_rl.envs import rewards as rw  # noqa: E402
from duckiebot_rl.envs.obstacles import lane_frame_to_world  # noqa: E402

CONTROL_DT = DUCKIEBOT.control_dt_s
V_MAX = DUCKIEBOT.v_cmd_max_m_s
ROBOT_W = DUCKIEBOT.robot_width_m
LANE_W = 0.23


def t(*values: float) -> torch.Tensor:
    """Return a 1-D float32 tensor of the given values."""
    return torch.tensor(values, dtype=torch.float32)


# =============================================================================================
# leaky_cos
# =============================================================================================


def test_leaky_cos_is_a_cosine_inside_the_half_period():
    """Inside |x| < pi the term is exactly cos(x)."""
    x = t(0.0, 0.5, -1.0, 3.0)
    torch.testing.assert_close(rw.leaky_cos(x), torch.cos(x))


def test_leaky_cos_leaks_linearly_outside():
    """At |x| >= pi it becomes -1 - 0.05 * (|x| - pi), which removes the periodicity."""
    assert float(rw.leaky_cos(t(math.pi))) == pytest.approx(-1.0, abs=1e-6)
    assert float(rw.leaky_cos(t(2.0 * math.pi))) == pytest.approx(-1.0 - 0.05 * math.pi, abs=1e-6)
    assert float(rw.leaky_cos(t(-2.0 * math.pi))) == pytest.approx(-1.0 - 0.05 * math.pi, abs=1e-6)


def test_leaky_cos_is_monotone_in_the_error_magnitude_far_out():
    """The leak exists so a heading error of one full period does not score as well as zero."""
    values = rw.leaky_cos(t(math.pi, 2.0 * math.pi, 4.0 * math.pi))
    assert float(values[0]) > float(values[1]) > float(values[2])


# =============================================================================================
# psi_target and r_heading
# =============================================================================================


def test_psi_target_points_back_at_the_centreline_for_both_signs():
    """``d > 0`` is left of the centreline, so the target heading must be negative (rightward)."""
    assert float(rw.psi_target(t(0.10))) == pytest.approx(-math.pi / 4.0, abs=1e-6)
    assert float(rw.psi_target(t(-0.10))) == pytest.approx(+math.pi / 4.0, abs=1e-6)
    assert float(rw.psi_target(t(0.0))) == pytest.approx(0.0, abs=1e-9)


def test_psi_target_saturates_at_45_degrees():
    """The clip is at |d| = 0.05 m; beyond it the request stops growing."""
    assert float(rw.psi_target(t(0.05))) == pytest.approx(-math.pi / 4.0, abs=1e-6)
    assert float(rw.psi_target(t(5.00))) == pytest.approx(-math.pi / 4.0, abs=1e-6)


def test_psi_target_is_linear_inside_the_clip():
    """Half the saturating error asks for half the corrective heading."""
    assert float(rw.psi_target(t(0.025))) == pytest.approx(-math.pi / 8.0, abs=1e-6)


def test_heading_reward_is_one_on_the_centreline_facing_along_the_lane():
    """Both lobes peak together, so a perfect pose scores exactly 1.0."""
    assert float(rw.r_heading(t(0.0), t(0.0))) == pytest.approx(1.0, abs=1e-6)


def test_heading_reward_at_a_ten_degree_error():
    """Hand-computed: the narrow lobe is at its seam (-1) and the wide lobe is cos(pi/5)."""
    value = float(rw.r_heading(t(0.0), t(math.radians(10.0))))
    expected = 0.5 * (-1.0 + math.cos(math.pi / 5.0))
    assert value == pytest.approx(expected, abs=1e-6)


def test_heading_reward_decreases_monotonically_over_the_useful_range():
    """The wide lobe exists so the gradient does not vanish at large errors."""
    errors = t(0.0, math.radians(10.0), math.radians(25.0), math.radians(50.0))
    values = rw.r_heading(torch.zeros_like(errors), errors)
    assert float(values[0]) > float(values[1]) > float(values[2]) > float(values[3])


def test_heading_reward_rewards_correcting_a_lateral_error():
    """Off to the left, turning right must beat holding the lane tangent."""
    d = t(0.08, 0.08)
    psi = t(-math.pi / 4.0, 0.0)
    values = rw.r_heading(d, psi)
    assert float(values[0]) > float(values[1])


# =============================================================================================
# r_progress
# =============================================================================================


def test_progress_is_one_at_full_speed_on_the_centreline():
    """The term is normalised by the distance a full-speed control step covers."""
    ds = t(V_MAX * CONTROL_DT)
    value = rw.r_progress(ds, t(0.0), LANE_W, ROBOT_W, V_MAX, CONTROL_DT)
    assert float(value) == pytest.approx(1.0, abs=1e-6)


def test_progress_is_zero_when_driving_backwards():
    """The ds > 0 half of the gate is the direction lock."""
    ds = t(-0.01)
    assert float(rw.r_progress(ds, t(0.0), LANE_W, ROBOT_W, V_MAX, CONTROL_DT)) == 0.0


def test_progress_gate_uses_the_episode_lane_width_not_a_literal():
    """Critic item H: 0.115 was half of a nominal lane and is wrong on every randomised episode."""
    ds = t(0.02, 0.02)
    d = t(0.069, 0.069)
    narrow = rw.r_progress(ds, d, 0.17, ROBOT_W, V_MAX, CONTROL_DT)
    wide = rw.r_progress(ds, d, 0.28, ROBOT_W, V_MAX, CONTROL_DT)
    # gate(0.17) = 0.5*(0.17-0.131)+0.02 = 0.0395  -> closed at d = 0.069
    # gate(0.28) = 0.5*(0.28-0.131)+0.02 = 0.0945  -> open   at d = 0.069
    assert float(narrow[0]) == 0.0
    assert float(wide[0]) > 0.0


def test_progress_gate_boundary_is_exactly_the_spec_expression():
    """Just inside the gate pays; just outside pays nothing."""
    gate = 0.5 * (LANE_W - ROBOT_W) + rw.PROGRESS_GATE_MARGIN_M
    ds = t(0.02, 0.02)
    d = t(gate - 1e-4, gate + 1e-4)
    values = rw.r_progress(ds, d, LANE_W, ROBOT_W, V_MAX, CONTROL_DT)
    assert float(values[0]) > 0.0
    assert float(values[1]) == 0.0


def test_progress_gate_does_not_punish_the_white_edge_side():
    """The gate is one-sided: it blocks the oncoming lane, not the outer shoulder."""
    ds = t(0.02)
    assert float(rw.r_progress(ds, t(-0.20), LANE_W, ROBOT_W, V_MAX, CONTROL_DT)) > 0.0


# =============================================================================================
# r_lateral
# =============================================================================================


def test_lateral_penalty_is_zero_on_the_centreline():
    assert float(rw.r_lateral(t(0.0), LANE_W)) == pytest.approx(0.0, abs=1e-9)


def test_lateral_penalty_at_the_lane_edge():
    """At |d| = w/2 the exponent is 1, so the term is -(1 - 0.001) = -0.999."""
    assert float(rw.r_lateral(t(0.5 * LANE_W), LANE_W)) == pytest.approx(-0.999, abs=1e-6)


def test_lateral_penalty_at_half_the_lane_edge():
    """Hand-computed: -(1 - 0.001 ** 0.5) = -0.968377."""
    assert float(rw.r_lateral(t(0.25 * LANE_W), LANE_W)) == pytest.approx(-0.968377, abs=1e-6)


def test_lateral_penalty_is_symmetric_and_bounded():
    """Bounded in (-1, 0]: an unbounded penalty would dominate the early return.

    The bound is asserted as ``>= -1``, not ``> -1``. Mathematically the term never reaches -1,
    but at ``|d| = 0.4`` on a 0.23 m lane the exponential is ``4e-11`` and float32 rounds
    ``-(1 - 4e-11)`` to exactly -1.0. That saturation is the desired behaviour - it is what
    "bounded" buys - and asserting the open bound would only be asserting a property of the
    dtype.
    """
    values = rw.r_lateral(t(-0.4, -0.05, 0.05, 0.4), LANE_W)
    assert float(values[0]) == pytest.approx(float(values[3]), abs=1e-9)
    assert float(values[1]) == pytest.approx(float(values[2]), abs=1e-9)
    assert bool((values >= -1.0).all()) and bool((values <= 0.0).all())
    assert float(rw.r_lateral(t(0.5 * LANE_W), LANE_W)) > -1.0


def test_lateral_penalty_scales_with_the_episode_lane_width():
    """The same absolute offset hurts more in a narrow lane, which is the correct behaviour."""
    narrow = float(rw.r_lateral(t(0.06), 0.17))
    wide = float(rw.r_lateral(t(0.06), 0.28))
    assert narrow < wide


# =============================================================================================
# r_smooth, r_proximity, stall
# =============================================================================================


def test_smoothness_penalty_is_the_squared_action_delta():
    action = torch.tensor([[1.0, 0.0], [0.5, -0.5]])
    previous = torch.tensor([[0.0, 0.0], [0.5, -0.5]])
    values = rw.r_smooth(action, previous)
    assert float(values[0]) == pytest.approx(-1.0, abs=1e-6)
    assert float(values[1]) == pytest.approx(0.0, abs=1e-9)


def test_proximity_pays_only_for_opening_an_existing_overlap():
    """``p = min(gap, 0)``, so a positive clearance that grows earns nothing."""
    assert float(rw.r_proximity(t(0.06), t(0.05))) == 0.0
    assert float(rw.r_proximity(t(-0.01), t(-0.02))) == pytest.approx(0.5, abs=1e-6)


def test_proximity_never_pays_for_closing_the_gap():
    """Deepening an overlap is clipped to 0, not turned into a negative reward."""
    assert float(rw.r_proximity(t(-0.02), t(-0.01))) == 0.0


def test_proximity_is_clipped_at_the_spec_ceiling():
    """A large single-step recovery is worth at most 1.5."""
    assert float(rw.r_proximity(t(-0.001), t(-0.10))) == pytest.approx(1.5, abs=1e-6)


def test_proximity_is_zero_on_an_obstacle_free_map():
    """An infinite gap must not produce NaN through inf - inf."""
    infinite = torch.full((2,), float("inf"))
    value = rw.r_proximity(infinite, infinite)
    assert torch.isfinite(value).all()
    assert float(value[0]) == 0.0


def test_stall_indicator_uses_the_speed_magnitude():
    """Reversing at 0.2 m/s is not a stall; it is penalised by the progress gate instead."""
    values = rw.stall_indicator(t(0.0, 0.02, 0.05, -0.2))
    assert values.tolist() == [1.0, 1.0, 0.0, 0.0]


# =============================================================================================
# The assembled reward
# =============================================================================================


def _nominal_kwargs(**overrides: object) -> dict:
    """Return a clean-driving argument set for :func:`compute_reward`, with overrides applied."""
    base = {
        "d": t(0.0),
        "psi": t(0.0),
        "ds": t(V_MAX * CONTROL_DT),
        "action": torch.tensor([[1.0, 0.0]]),
        "prev_action": torch.tensor([[1.0, 0.0]]),
        "body_speed": t(V_MAX),
        "lane_width": LANE_W,
        "control_dt": CONTROL_DT,
    }
    base.update(overrides)
    return base


def test_total_reward_is_the_weighted_sum_of_the_terms():
    """Assemble the fixture by hand from the S5.4 weights."""
    reward, terms = rw.compute_reward(**_nominal_kwargs())
    weights = rw.RewardWeights()
    expected = (
        weights.heading * float(terms.heading)
        + weights.progress * float(terms.progress)
        + weights.lateral * float(terms.lateral)
        + weights.smooth * float(terms.smooth)
        + weights.proximity * float(terms.proximity)
        - weights.stall * float(terms.stall)
    )
    assert float(reward) == pytest.approx(expected, abs=1e-5)
    assert float(reward) == pytest.approx(1.0 * 1.0 + 6.0 * 1.0, abs=1e-4)


def test_terminal_penalty_is_added_before_the_clip():
    """S5.4 clips R *including* the terminal penalty, so a good step cannot absorb a collision."""
    clean, _ = rw.compute_reward(**_nominal_kwargs())
    crashed, _ = rw.compute_reward(**_nominal_kwargs(terminated=torch.tensor([True])))
    assert float(crashed) == pytest.approx(float(clean) - 10.0, abs=1e-5)


def test_truncation_carries_no_penalty():
    """A time limit is a harness artifact; it is bootstrapped, not punished (S6.4)."""
    reward, _ = rw.compute_reward(**_nominal_kwargs(terminated=torch.tensor([False])))
    clean, _ = rw.compute_reward(**_nominal_kwargs())
    assert float(reward) == pytest.approx(float(clean), abs=1e-9)


def test_reward_is_clipped_to_the_spec_range():
    """A pathological progress delta cannot inflate the return past +/-20."""
    huge, _ = rw.compute_reward(**_nominal_kwargs(ds=t(100.0)))
    assert float(huge) == pytest.approx(rw.REWARD_CLIP, abs=1e-6)


def test_reward_terms_are_reported_unweighted_for_logging():
    """S6.8 logs per-term means; they must be the raw terms, not pre-multiplied."""
    _reward, terms = rw.compute_reward(**_nominal_kwargs())
    assert float(terms.progress) == pytest.approx(1.0, abs=1e-4)
    assert set(terms.as_dict()) == {
        "heading",
        "progress",
        "lateral",
        "smooth",
        "proximity",
        "stall",
        "total",
    }


def test_reward_is_finite_without_any_obstacle_arguments():
    """The obstacle-free map is the default training condition for stages 0 and 1."""
    reward, terms = rw.compute_reward(**_nominal_kwargs())
    assert torch.isfinite(reward).all()
    assert float(terms.proximity) == 0.0


# =============================================================================================
# Reward hacking: three degenerate policies on a real lane graph
# =============================================================================================


@pytest.fixture(scope="module")
def straight_lane() -> tuple[BatchedLaneGraph, int, float]:
    """Return a batched lane graph and the longest straight segment of a built-in map.

    A real lane graph rather than a hand-written straight line, so that ``d``, ``psi`` and the
    tangent used by ``progress_delta`` all come from the production code path.
    """
    lane = BatchedLaneGraph([LaneGraph(builtin_map("loop_small"))])
    lengths = lane.seg_length[0].clone()
    lengths[lane.seg_is_arc[0]] = -1.0
    lengths[~lane.seg_valid[0]] = -1.0
    segment = int(torch.argmax(lengths))
    return lane, segment, float(lane.seg_length[0, segment])


def _rollout(
    lane: BatchedLaneGraph,
    segment: int,
    length: float,
    mode: str,
    steps: int = 18,
) -> float:
    """Roll one behaviour along a lane segment and return the total S5.4 reward.

    Args:
        lane: The batched lane graph.
        segment: Segment index to drive along.
        length: Length of that segment in metres.
        mode: ``"clean"``, ``"spin"``, ``"hug"`` or ``"backward"``.
        steps: Number of control steps.

    Returns:
        The summed reward over the rollout.
    """
    variant = torch.zeros(1, dtype=torch.long)
    seg = torch.full((1,), segment, dtype=torch.long)
    speed = 0.3
    step_m = speed * CONTROL_DT
    start = 0.5 * length - 0.5 * steps * step_m
    lateral = 0.09 if mode == "hug" else 0.0

    total = 0.0
    prev_xy: torch.Tensor | None = None
    prev_action = torch.zeros(1, 2)
    lane_width = lane.lane_width[:1]
    for i in range(steps):
        if mode == "clean" or mode == "hug":
            arc = torch.tensor([start + i * step_m])
            heading, body_speed = 0.0, speed
            action = torch.tensor([[0.0, 0.0]])
        elif mode == "backward":
            arc = torch.tensor([start + (steps - 1 - i) * step_m])
            heading, body_speed = math.pi, speed
            action = torch.tensor([[0.0, 0.0]])
        elif mode == "spin":
            arc = torch.tensor([start])
            heading, body_speed = 0.4 * i, 0.0
            action = torch.tensor([[0.0, 1.0 if i % 2 else -1.0]])
        else:  # pragma: no cover - guarded by the caller
            raise ValueError(mode)

        x, y, tangent_yaw = lane_frame_to_world(
            lane, variant, seg, arc, torch.tensor([lateral], dtype=torch.float32)
        )
        yaw = tangent_yaw + heading
        query = lane.query(variant, x, y, yaw)
        xy = torch.stack([x, y], dim=-1)
        if prev_xy is None:
            ds = torch.zeros(1)
        else:
            ds = progress_delta(prev_xy[:, 0], prev_xy[:, 1], x, y, query.tangent_x, query.tangent_y)
        prev_xy = xy

        reward, _ = rw.compute_reward(
            d=query.d,
            psi=query.psi,
            ds=ds,
            action=action,
            prev_action=prev_action,
            body_speed=torch.tensor([body_speed]),
            lane_width=lane_width,
            control_dt=CONTROL_DT,
        )
        prev_action = action
        total += float(reward)
    return total


def test_clean_lane_following_scores_best(straight_lane):
    """The three known reward hacks must all lose to simply driving down the lane."""
    lane, segment, length = straight_lane
    clean = _rollout(lane, segment, length, "clean")
    spin = _rollout(lane, segment, length, "spin")
    hug = _rollout(lane, segment, length, "hug")
    backward = _rollout(lane, segment, length, "backward")
    assert clean > spin, f"spinning in place scored {spin} against clean {clean}"
    assert clean > hug, f"hugging the yellow line scored {hug} against clean {clean}"
    assert clean > backward, f"driving backwards scored {backward} against clean {clean}"


def test_spinning_in_place_scores_negative(straight_lane):
    """Not merely worse than clean driving: a stationary spin must be actively punished."""
    lane, segment, length = straight_lane
    assert _rollout(lane, segment, length, "spin") < 0.0


def test_driving_backwards_earns_no_progress_credit(straight_lane):
    """Reversing along the lane collects zero progress and pays the heading penalty."""
    lane, segment, length = straight_lane
    assert _rollout(lane, segment, length, "backward") < 0.0


def test_hugging_the_oncoming_lane_earns_no_progress_credit(straight_lane):
    """At d = 0.09 the robot is past the gate, so the 6.0-weighted term pays nothing."""
    lane, segment, length = straight_lane
    hug = _rollout(lane, segment, length, "hug")
    clean = _rollout(lane, segment, length, "clean")
    assert hug < 0.25 * clean
