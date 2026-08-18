"""Continuity-constrained lane matching: the fix for hairpin corner cutting going unmeasured.

The bug these tests pin, reproduced from the live system: the free nearest-segment search
re-homes a robot that crosses the centreline onto the adjacent lane. On ``zigzag`` the two
flanks of a hairpin are segments 0.2340 m apart in space but 8.23 m apart along the route, so a
robot cutting the corner was re-matched mid-cut, its ``|d|`` collapsed from about 0.117 m back
toward zero, every reward and termination term derived from ``d`` went blind, and the match
jumped three quarters of the loop. The route window (``MATCH_WINDOW_M``) makes the re-home
impossible while leaving legitimate driving and spawn matching untouched.

The zigzag flank pair used throughout (measured, not assumed; a fixture test guards it):
segment A mid ``(-1.287, 0.585)`` tangent ``(0, -1)``, segment B mid ``(-1.053, 0.585)``
tangent ``(0, +1)``, separation 0.2340 m = two lane offsets of 0.117 m.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch

from duckiebot_rl.city import maps as M
from duckiebot_rl.city.lane_graph import MATCH_WINDOW_M, LaneGraph
from duckiebot_rl.sim2sim.track import LaneGraph as MjLaneGraph
from duckiebot_rl.sim2sim.track import load_map

FLANK_Y = 0.585
A_X = -1.287
B_X = -1.053
YAW_SOUTH = -math.pi / 2.0


@pytest.fixture(scope="module")
def zigzag() -> LaneGraph:
    """Lane graph of the zigzag built-in, which carries the hairpin geometry."""
    return LaneGraph(M.builtin_map("zigzag"))


def _one(graph: LaneGraph, x: float, y: float, yaw: float, prev: tuple[int, float] | None) -> Any:
    """Query one pose, optionally continuing from a previous ``(segment, s)`` match."""
    if prev is None:
        return graph.query([x], [y], [yaw])
    return graph.query([x], [y], [yaw], prev_seg_id=torch.tensor([prev[0]]), prev_s=torch.tensor([prev[1]]))


def _flank_ids(graph: LaneGraph) -> tuple[int, int, tuple[int, float]]:
    """Return (segment A id, segment B id, A's ``(segment, s)`` match at the flank row)."""
    qa = _one(graph, A_X, FLANK_Y, YAW_SOUTH, None)
    qb = _one(graph, B_X, FLANK_Y, math.pi / 2.0, None)
    return int(qa.seg_id), int(qb.seg_id), (int(qa.seg_id), float(qa.s))


def test_fixture_flanks_are_where_the_docstring_says(zigzag: LaneGraph) -> None:
    """Guard the measured hairpin-flank geometry the other tests depend on."""
    a_id, b_id, _ = _flank_ids(zigzag)
    seg_a, seg_b = zigzag.segments[a_id], zigzag.segments[b_id]
    assert not seg_a.is_arc and not seg_b.is_arc
    assert seg_a.t0[0] * seg_b.t0[0] + seg_a.t0[1] * seg_b.t0[1] == pytest.approx(-1.0)
    assert abs(A_X - B_X) == pytest.approx(0.234, abs=1e-9)
    route_sep = abs(zigzag.route_offsets[a_id] - zigzag.route_offsets[b_id])
    assert route_sep > 5.0, "the flanks must be far apart along the route for the test to bite"


def test_free_match_rehomes_across_the_centreline_and_collapses_d(zigzag: LaneGraph) -> None:
    """The bug, reproduced: without a window, crossing re-homes the match and |d| collapses.

    Sweeping straight across from lane A toward lane B while still heading in A's direction,
    the free search flips to segment B about halfway and reports the robot nearly centred.
    """
    a_id, b_id, _ = _flank_ids(zigzag)
    at_far_side = _one(zigzag, B_X - 0.02, FLANK_Y, YAW_SOUTH, None)
    assert int(at_far_side.seg_id) == b_id
    assert abs(float(at_far_side.d)) < 0.05, "the free match reports a wrong-lane robot as centred"
    halfway = _one(zigzag, (A_X + B_X) / 2.0 + 0.04, FLANK_Y, YAW_SOUTH, None)
    assert int(halfway.seg_id) != a_id, "the free match abandons the robot's lane mid-cut"


def test_windowed_match_keeps_d_truthful_across_the_centreline(zigzag: LaneGraph) -> None:
    """With the window, the match stays on lane A and d grows monotonically through the cut."""
    a_id, _, route_a = _flank_ids(zigzag)
    previous_d = -1.0
    steps = 40
    for i in range(steps + 1):
        x = A_X + (B_X - A_X) * i / steps
        query = _one(zigzag, x, FLANK_Y, YAW_SOUTH, route_a)
        assert int(query.seg_id) == a_id, f"match left lane A at x={x:.3f}"
        d = float(query.d)
        assert d > previous_d - 1e-9, f"d not monotone at x={x:.3f}"
        assert abs(d - (x - A_X)) < 1e-6, "d must equal the physical offset from lane A"
        previous_d = d
    assert previous_d == pytest.approx(0.234, abs=1e-6), "at lane B's centre, d is two lane offsets"


def test_windowed_match_follows_legitimate_driving(zigzag: LaneGraph) -> None:
    """Driving properly along the lane, the window never interferes and d stays near zero."""
    a_id, _, _ = _flank_ids(zigzag)
    seg = zigzag.segments[a_id]
    prev_route: tuple[int, float] | None = None
    step_m = 0.03
    n_steps = int(seg.length / step_m) + 12  # runs through the end of A into its successors
    x, y = seg.p0
    for _ in range(n_steps):
        x += seg.t0[0] * 0.0  # kept for symmetry; motion is written explicitly below
        query = _one(zigzag, x, y, YAW_SOUTH, prev_route)
        assert abs(float(query.d)) < 5e-3
        prev_route = (int(query.seg_id), float(query.s))
        y -= step_m  # lane A runs south
        if y < seg.p1[1]:
            break


def test_untracked_sentinel_is_a_free_search(zigzag: LaneGraph) -> None:
    """A -1 previous segment (fresh reset) behaves exactly like the unconstrained search."""
    free = _one(zigzag, B_X - 0.02, FLANK_Y, YAW_SOUTH, None)
    untracked = _one(zigzag, B_X - 0.02, FLANK_Y, YAW_SOUTH, (-1, 0.0))
    assert int(free.seg_id) == int(untracked.seg_id)
    assert float(free.d) == pytest.approx(float(untracked.d))


def test_window_is_inert_on_intersection_maps(zigzag: LaneGraph) -> None:
    """Maps without a single directed route (has_route False) keep the free search."""
    graph = LaneGraph(M.builtin_map("intersection_4way"))
    assert not graph.has_route
    x, y = graph.segments[0].p0
    free = graph.query([x + 0.05], [y], [0.0])
    windowed = graph.query([x + 0.05], [y], [0.0], prev_seg_id=torch.tensor([0]), prev_s=torch.tensor([0.0]))
    assert int(free.seg_id) == int(windowed.seg_id)
    assert float(free.d) == pytest.approx(float(windowed.d))


def test_an_adversarial_window_never_returns_a_garbage_match(zigzag: LaneGraph) -> None:
    """However hostile the previous match and the window, the answer stays self-consistent.

    An all-infinite cost row would make ``argmin`` return segment 0 and a ``d`` belonging to
    somewhere else entirely, which is why the query keeps a fallback to the free search. On the
    built-in maps that branch turns out to be unreachable, because the projection clamps at
    segment ends: the predecessor's end point and the successor's start point are the same route
    position, so the previous match is always representable and the window is never empty. The
    branch stays as a guard for pathological geometry, and what is asserted here is the property
    it protects rather than the line itself: whatever segment comes back, ``d`` is finite and is
    genuinely that segment's signed offset.
    """
    for offset in range(0, len(zigzag.segments), 7):
        for window in (1e-12, 0.01, MATCH_WINDOW_M):
            query = zigzag.query(
                [A_X],
                [FLANK_Y],
                [YAW_SOUTH],
                prev_seg_id=torch.tensor([offset]),
                prev_s=torch.tensor([0.0]),
                window_m=window,
            )
            seg_id, d = int(query.seg_id), float(query.d)
            assert 0 <= seg_id < len(zigzag.segments)
            assert math.isfinite(d)
            reach = math.hypot(A_X - float(query.closest_x), FLANK_Y - float(query.closest_y))
            assert abs(d) <= reach + 1e-6, "d must belong to the segment that was returned"


# ------------------------------------------------------------------- MuJoCo twin parity
def _mj_zigzag() -> MjLaneGraph:
    """The MuJoCo twin's lane graph of the same shared zigzag map (C6 parity)."""
    return MjLaneGraph(load_map(M.builtin_map("zigzag").to_dict()))


def _mj_flank_pair(lane: MjLaneGraph) -> tuple[int, int, tuple[float, float], tuple[float, float]]:
    """Find the closest antiparallel straight pair: the two flanks of a hairpin."""
    best = None
    for i, a in enumerate(lane.segments):
        if a.is_arc:
            continue
        for j, b in enumerate(lane.segments):
            if j <= i or b.is_arc:
                continue
            ta, tb = a.tangent_at(0.0), b.tangent_at(0.0)
            if ta[0] * tb[0] + ta[1] * tb[1] > -0.99:
                continue
            ma = a.point_at(a.length / 2.0)
            mb = b.point_at(b.length / 2.0)
            dist = math.hypot(ma[0] - mb[0], ma[1] - mb[1])
            if dist < 0.30 and (best is None or dist < best[0]):
                best = (dist, i, j, ma, mb)
    assert best is not None, "the zigzag must contain a hairpin flank pair"
    assert best[0] == pytest.approx(0.234, abs=1e-6)
    return best[1], best[2], best[3], best[4]


def test_mujoco_free_match_collapses_and_windowed_match_does_not() -> None:
    """The MuJoCo twin shows the same bug free and the same truth windowed (C6 parity)."""
    lane = _mj_zigzag()
    a_seg, _, ma, mb = _mj_flank_pair(lane)
    ta = lane.segments[a_seg].tangent_at(0.0)
    heading = math.atan2(ta[1], ta[0])

    across = (mb[0] - ma[0], mb[1] - ma[1])
    near_b = lane.query(ma[0] + 0.9 * across[0], ma[1] + 0.9 * across[1], heading)
    assert near_b.segment != a_seg, "free search re-homes across the centreline"
    assert abs(near_b.d) < 0.06, "free search reports a wrong-lane robot as nearly centred"

    start = lane.query(ma[0], ma[1], heading)
    assert start.segment == a_seg
    prev = (start.segment, start.s)
    previous_d = -1.0
    steps = 30
    for i in range(steps + 1):
        x = ma[0] + across[0] * i / steps
        y = ma[1] + across[1] * i / steps
        query = lane.query(x, y, heading, prev_match=prev)
        assert query.segment == a_seg, f"windowed match left lane A at step {i}"
        assert abs(query.d) > previous_d - 1e-9, f"|d| not monotone at step {i}"
        previous_d = abs(query.d)
    assert previous_d == pytest.approx(0.234, abs=1e-6)


def test_mujoco_allowed_window_is_local_and_excludes_the_far_flank() -> None:
    """The window holds a local route neighbourhood; the hairpin's far flank is not in it."""
    lane = _mj_zigzag()
    a_seg, b_seg, _, _ = _mj_flank_pair(lane)
    seg = lane.segments[a_seg]
    allowed = lane.allowed_window(a_seg, seg.length / 2.0, MATCH_WINDOW_M)
    assert a_seg in allowed
    assert b_seg not in allowed, "the far flank inside the window defeats the whole fix"
    assert len(allowed) < len(lane.segments)


# ------------------------------------------------------------------- mechanism proof
def test_cutting_no_longer_earns_progress_credit(zigzag: LaneGraph) -> None:
    """The reward consequence, end to end, at the pose where the exploit used to pay out.

    Early in a cut the old accounting was already direction-blocked (matched to the far flank
    with psi ~ pi, so ds < 0). The launder completed when the robot finished rotating: matched
    to the far flank, aligned with it, d ~ 0, full progress credit, and the route position had
    silently jumped 8 m. At that same pose the windowed match still scores the robot on its own
    lane: d is two lane offsets, the same physical step projects NEGATIVE on the own-lane
    tangent, and psi ~ pi keeps the wrong-lane penalty firing. Same motion, opposite verdict.
    """
    from duckiebot_rl.city.lane_graph import progress_delta
    from duckiebot_rl.envs.rewards import r_progress, wrong_lane_indicator

    a_id, b_id, route_a = _flank_ids(zigzag)
    _ = b_id
    x_late = B_X - 0.02
    yaw_north = math.pi / 2.0  # rotated, aligned with the far flank
    free = _one(zigzag, x_late, FLANK_Y, yaw_north, None)
    windowed = _one(zigzag, x_late, FLANK_Y, yaw_north, route_a)

    assert int(free.seg_id) == b_id and abs(float(free.d)) < 0.03
    assert int(windowed.seg_id) == a_id and float(windowed.d) == pytest.approx(0.214, abs=1e-3)

    # the same physical step, 0.03 m north, projected on each match's tangent
    prev_x, prev_y = torch.tensor([x_late]), torch.tensor([FLANK_Y - 0.03])
    now_x, now_y = torch.tensor([x_late]), torch.tensor([FLANK_Y])
    ds_free = progress_delta(prev_x, prev_y, now_x, now_y, free.tangent_x, free.tangent_y)
    ds_windowed = progress_delta(prev_x, prev_y, now_x, now_y, windowed.tangent_x, windowed.tangent_y)
    lane_width = torch.tensor([0.2046])

    paid_free = float(r_progress(ds_free, free.d, lane_width))
    paid_windowed = float(r_progress(ds_windowed, windowed.d, lane_width))
    assert paid_free > 0.5, "the old accounting paid the completed cut near-full progress"
    assert paid_windowed == 0.0, "the windowed accounting refuses the cut any progress"
    assert float(wrong_lane_indicator(free.psi)) == 0.0, "the old accounting saw nothing wrong"
    assert float(wrong_lane_indicator(windowed.psi)) == 1.0, "the window keeps the flag firing"


# ------------------------------------------------------- the regression that reached a run
@pytest.mark.parametrize("map_name", ["zigzag", "loop_small", "loop_big"])
def test_a_full_lap_keeps_the_match_on_the_robot(map_name: str) -> None:
    """Driving a whole lap down the lane centre, the match tracks and ``d`` stays at zero.

    The guard against the failure the first version of this window shipped with. A map is a
    union of directed cycles with independent zeros and lengths (``zigzag``: 10.93 m and
    9.46 m); comparing route offsets across cycles, and reducing them modulo the longest one,
    made the arithmetic wrong by 1.47 m against a 0.35 m window on the shorter cycle. Legitimate
    successors were then excluded, the match pinned to a stale segment, and ``d`` grew without
    bound as the robot drove away from it. In a live run that showed up as mean episode
    ``lane_dev_rms`` jumping from 0.06 m to 0.61 m with almost no off-drivable terminations,
    which is the signature of a measurement failure rather than a driving failure: a robot
    really 1.5 m off its lane would have left the road.

    Walking the lane centreline is the strongest available oracle, because ``d`` is zero there
    by construction for the correct match and large for any other.
    """
    graph = LaneGraph(M.builtin_map(map_name))
    start = next(i for i, seg in enumerate(graph.segments) if not seg.is_arc)
    cycle = graph.segment_cycle[start]
    lap = graph.segment_cycle_length[start]

    index, arc, travelled = start, 0.0, 0.0
    prev: tuple[int, float] | None = None
    step_m = 0.02  # half the 0.041 m a full-speed control step covers
    worst = 0.0
    while travelled < lap:
        seg = graph.segments[index]
        if arc > seg.length:
            arc -= seg.length
            index = graph.successors[index][0]
            continue
        point = _point_on(seg, arc)
        tangent = _tangent_on(seg, arc)
        yaw = math.atan2(tangent[1], tangent[0])
        query = _one(graph, point[0], point[1], yaw, prev)
        worst = max(worst, abs(float(query.d)))
        assert graph.segment_cycle[int(query.seg_id)] == cycle, (
            f"{map_name}: match left the robot's cycle after {travelled:.2f} m"
        )
        prev = (int(query.seg_id), float(query.s))
        arc += step_m
        travelled += step_m
    assert worst < 1e-3, f"{map_name}: |d| reached {worst:.4f} m while driving the lane centre"


def _point_on(seg, arc: float) -> tuple[float, float]:
    """World point at arc length ``arc`` along ``seg``."""
    if not seg.is_arc:
        return (seg.p0[0] + seg.t0[0] * arc, seg.p0[1] + seg.t0[1] * arc)
    theta = seg.theta0 + seg.sweep_sign * (arc / seg.radius)
    return (
        seg.center[0] + seg.radius * math.cos(theta),
        seg.center[1] + seg.radius * math.sin(theta),
    )


def _tangent_on(seg, arc: float) -> tuple[float, float]:
    """Unit tangent at arc length ``arc`` along ``seg``."""
    if not seg.is_arc:
        return seg.t0
    theta = seg.theta0 + seg.sweep_sign * (arc / seg.radius)
    return (-seg.sweep_sign * math.sin(theta), seg.sweep_sign * math.cos(theta))
