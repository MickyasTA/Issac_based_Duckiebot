"""Unit tests for the lane graph: the reward's ground truth, and its sign conventions.

This is the file that pins SPEC v2 S2's sign conventions in executable form. If any of these
tests are edited to make an implementation pass, the reward silently changes meaning, so each
assertion below carries the hand-computed number it checks.

The fixtures use ``loop_small`` at the nominal 0.585 m pitch. Its tile ``(2, 1)`` is a
``straight/N`` centred on ``(-0.585, 0)`` and its tile ``(1, 1)`` is a ``curve_left/N`` centred
on ``(-0.585, 0.585)`` whose arc centre is the tile's south-east corner ``(-0.2925, 0.2925)``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from duckiebot_rl.city import maps as M
from duckiebot_rl.city import spec as S
from duckiebot_rl.city import tiles as T
from duckiebot_rl.city.lane_graph import (
    BatchedLaneGraph,
    LaneGraph,
    build_lane_segments,
    progress_delta,
    wrap_to_pi,
)

PITCH = S.NOMINAL_TILE_SPEC.pitch_m  # 0.585
HALF = S.NOMINAL_TILE_SPEC.half_m  # 0.2925
LANE = S.NOMINAL_TILE_SPEC.lane_center_offset_m  # 0.117
R_IN = S.NOMINAL_TILE_SPEC.curve_radius_inner_m  # 0.1755
R_OUT = S.NOMINAL_TILE_SPEC.curve_radius_outer_m  # 0.4095

STRAIGHT_CENTER = (-0.585, 0.0)
CURVE_CENTER = (-0.585, 0.585)
CURVE_CORNER = (-0.2925, 0.2925)


@pytest.fixture(scope="module")
def graph() -> LaneGraph:
    """Lane graph of the small loop at nominal geometry."""
    return LaneGraph(M.builtin_map("loop_small"))


def q1(graph: LaneGraph, x: float, y: float, yaw: float, **kwargs: float):
    """Query a single pose and return the result with scalar Python floats where useful."""
    return graph.query([x], [y], [yaw], **kwargs)


# ---------------------------------------------------------------------------- map fixtures
def test_fixture_tiles_are_where_the_docstring_says(graph: LaneGraph) -> None:
    """Guard the hand-computed coordinates the rest of this file depends on."""
    city = graph.city
    assert M.format_tile(city.tiles[2][1]) == "straight/N"
    assert city.tile_center_xy(2, 1) == pytest.approx(STRAIGHT_CENTER)
    assert M.format_tile(city.tiles[1][1]) == "curve_left/N"
    assert city.tile_center_xy(1, 1) == pytest.approx(CURVE_CENTER)
    assert (CURVE_CENTER[0] + HALF, CURVE_CENTER[1] - HALF) == pytest.approx(CURVE_CORNER)


# ------------------------------------------------------------------------ segment structure
def test_lane_counts_per_tile_kind() -> None:
    """Each drivable tile carries one lane per ordered pair of open edges: 2 / 2 / 6 / 12."""
    counts = {
        "straight": 2,
        "curve": 2,
        "threeway": 6,
        "fourway": 12,
    }
    city = M.builtin_map("intersection_4way")
    segments = build_lane_segments(city)
    per_kind: dict[str, int] = {}
    for seg in segments:
        per_kind[city.tiles[seg.row][seg.col].kind] = per_kind.get(city.tiles[seg.row][seg.col].kind, 0) + 1
    tiles_of_kind = {kind: sum(1 for _, _, t in city.iter_tiles() if t.kind == kind) for kind in counts}
    for kind, per_tile in counts.items():
        assert per_kind[kind] == per_tile * tiles_of_kind[kind], kind
    assert len(segments) == 68


def test_straight_and_curve_segment_geometry(graph: LaneGraph) -> None:
    """Straight lanes are one tile long; curve lanes are exact quarter arcs at 0.30 / 0.70 tile."""
    straights = [s for s in graph.segments if not s.is_arc]
    arcs = [s for s in graph.segments if s.is_arc]
    assert len(straights) == 8 and len(arcs) == 8
    for seg in straights:
        assert seg.length == pytest.approx(PITCH)
        assert seg.curvature == 0.0
    radii = sorted({round(s.radius, 6) for s in arcs})
    assert radii == [pytest.approx(R_IN), pytest.approx(R_OUT)]
    for seg in arcs:
        assert seg.sweep == pytest.approx(math.pi / 2)
        assert seg.length == pytest.approx(seg.radius * math.pi / 2)
        assert abs(seg.curvature) == pytest.approx(1.0 / seg.radius)
    total = 8 * PITCH + 4 * (R_IN + R_OUT) * math.pi / 2
    assert graph.total_lane_length == pytest.approx(total)


def test_successor_segments_join_exactly(graph: LaneGraph) -> None:
    """A lane's exit point and tangent equal its successor's entry point and tangent."""
    for i, seg in enumerate(graph.segments):
        successors = graph.successors[i]
        assert successors, f"segment {i} has no successor on a closed loop"
        for j in successors:
            nxt = graph.segments[j]
            assert nxt.p0 == pytest.approx(seg.p1, abs=1e-12)
            assert nxt.t0 == pytest.approx(seg.t1, abs=1e-12)


def test_intersection_segments_join_exactly() -> None:
    """The same continuity holds through 3-way and 4-way tiles, where lanes branch."""
    graph = LaneGraph(M.builtin_map("intersection_4way"))
    branching = 0
    for i, seg in enumerate(graph.segments):
        assert graph.successors[i]
        branching += len(graph.successors[i]) > 1
        for j in graph.successors[i]:
            assert graph.segments[j].p0 == pytest.approx(seg.p1, abs=1e-12)
            assert graph.segments[j].t0 == pytest.approx(seg.t1, abs=1e-12)
    assert branching > 0
    assert not graph.has_route  # a branching network has no single cumulative arc length


def test_closed_loops_expose_a_cumulative_route(graph: LaneGraph) -> None:
    """On a pure loop every segment has one successor, so route offsets are well defined."""
    assert graph.has_route
    assert all(len(s) == 1 for s in graph.successors)
    offsets = sorted(graph.route_offsets)
    assert offsets[0] == pytest.approx(0.0)
    assert max(offsets) < graph.total_lane_length


# --------------------------------------------------------- straight tile, hand-computed
def test_straight_on_lane_pose_is_the_zero_of_the_error_coordinates(graph: LaneGraph) -> None:
    """A robot on the right-hand lane centreline heading with the lane has d = psi = 0."""
    x = STRAIGHT_CENTER[0] + LANE
    result = q1(graph, x, 0.0, math.pi / 2)
    assert float(result.d) == pytest.approx(0.0, abs=1e-6)
    assert float(result.psi) == pytest.approx(0.0, abs=1e-6)
    assert float(result.curvature) == pytest.approx(0.0)
    assert (float(result.tangent_x), float(result.tangent_y)) == pytest.approx((0.0, 1.0), abs=1e-6)
    # the segment starts at the tile's southern edge, so s is half a tile at the tile centre
    assert float(result.s) == pytest.approx(HALF, abs=1e-6)
    assert (float(result.closest_x), float(result.closest_y)) == pytest.approx((x, 0.0), abs=1e-6)


def test_straight_d_is_positive_toward_the_yellow_tape(graph: LaneGraph) -> None:
    """SPEC v2 S2: d > 0 is displaced LEFT of the centreline, toward the yellow centre tape.

    The tile centreline (and so the yellow tape) is at x = -0.585; the right-hand lane centre
    for a northbound robot is at x = -0.468. Moving toward the yellow tape means x decreasing.
    """
    x = STRAIGHT_CENTER[0] + LANE
    toward_yellow = q1(graph, x - 0.03, 0.0, math.pi / 2)
    toward_white = q1(graph, x + 0.03, 0.0, math.pi / 2)
    assert float(toward_yellow.d) == pytest.approx(+0.03, abs=1e-6)
    assert float(toward_white.d) == pytest.approx(-0.03, abs=1e-6)
    # and the same in the southbound lane, where "toward yellow" is x increasing
    x_south = STRAIGHT_CENTER[0] - LANE
    assert float(q1(graph, x_south + 0.03, 0.0, -math.pi / 2).d) == pytest.approx(+0.03, abs=1e-6)
    assert float(q1(graph, x_south - 0.03, 0.0, -math.pi / 2).d) == pytest.approx(-0.03, abs=1e-6)


def test_straight_psi_is_positive_counter_clockwise(graph: LaneGraph) -> None:
    """SPEC v2 S2: psi > 0 means the heading is rotated left of the lane tangent."""
    x = STRAIGHT_CENTER[0] + LANE
    for offset in (-1.2, -0.4, 0.0, 0.4, 1.2):
        result = q1(graph, x, 0.0, math.pi / 2 + offset)
        assert float(result.psi) == pytest.approx(offset, abs=1e-6)


def test_psi_target_steers_back_to_the_centreline_for_both_signs(graph: LaneGraph) -> None:
    """The S5.4 heading target must reduce |d| whichever side of the lane the robot is on."""
    x = STRAIGHT_CENTER[0] + LANE
    for lateral in (-0.06, -0.02, 0.02, 0.06):
        result = q1(graph, x - lateral, 0.0, math.pi / 2)
        d = float(result.d)
        psi_target = -np.clip(d / 0.05, -1.0, 1.0) * (math.pi / 4)
        # heading at psi_target, expressed in world, must have a lateral velocity opposing d
        tangent = np.array([float(result.tangent_x), float(result.tangent_y)])
        left_normal = np.array([-tangent[1], tangent[0]])
        heading = math.atan2(tangent[1], tangent[0]) + psi_target
        velocity = np.array([math.cos(heading), math.sin(heading)])
        assert float(velocity @ left_normal) * d < 0.0


def test_the_two_lanes_of_a_road_are_separated_by_twice_the_lane_offset(graph: LaneGraph) -> None:
    """The two lanes of a road are 234 mm apart and each is matched on its own side."""
    north = q1(graph, STRAIGHT_CENTER[0] + LANE, 0.0, math.pi / 2)
    south = q1(graph, STRAIGHT_CENTER[0] - LANE, 0.0, -math.pi / 2)
    assert float(north.dist) == pytest.approx(0.0, abs=1e-6)
    assert float(south.dist) == pytest.approx(0.0, abs=1e-6)
    assert int(north.seg_id) != int(south.seg_id)
    assert pytest.approx(0.234) == 2 * LANE
    # halfway between them (on the yellow tape) the two lanes are equidistant
    middle = q1(graph, STRAIGHT_CENTER[0], 0.0, math.pi / 2)
    assert float(middle.dist) == pytest.approx(LANE, abs=1e-6)


# ------------------------------------------------------------ curve tile, hand-computed
def test_curve_inner_lane_is_a_right_turn_of_radius_0_1755(graph: LaneGraph) -> None:
    """Entering the curve from the south heading north, the lane is a 0.1755 m right turn."""
    # 2 degrees into the arc: the entry point itself is shared with the preceding straight
    # segment, where either match is geometrically correct.
    theta_entry = math.radians(180.0 - 2.0)
    entry = (
        CURVE_CORNER[0] + R_IN * math.cos(theta_entry),
        CURVE_CORNER[1] + R_IN * math.sin(theta_entry),
    )
    result = q1(graph, entry[0], entry[1], math.pi / 2 - math.radians(2.0))
    assert float(result.d) == pytest.approx(0.0, abs=1e-6)
    assert float(result.psi) == pytest.approx(0.0, abs=1e-6)
    assert float(result.s) == pytest.approx(R_IN * math.radians(2.0), abs=1e-6)
    assert float(result.curvature) == pytest.approx(-1.0 / R_IN, rel=1e-6)  # negative = right

    # the 45 degree point of the arc, computed by hand from the corner
    mid = (
        CURVE_CORNER[0] + R_IN * math.cos(math.radians(135.0)),
        CURVE_CORNER[1] + R_IN * math.sin(math.radians(135.0)),
    )
    assert mid == pytest.approx((-0.41659, 0.41659), abs=1e-5)
    result = q1(graph, mid[0], mid[1], math.radians(45.0))
    assert float(result.d) == pytest.approx(0.0, abs=1e-6)
    assert float(result.psi) == pytest.approx(0.0, abs=1e-6)
    assert float(result.s) == pytest.approx(R_IN * math.pi / 4, abs=1e-6)
    assert (float(result.tangent_x), float(result.tangent_y)) == pytest.approx(
        (math.cos(math.radians(45.0)), math.sin(math.radians(45.0))), abs=1e-6
    )


def test_curve_d_sign_matches_the_straight_convention(graph: LaneGraph) -> None:
    """On the inner lane the yellow arc is at the larger radius, so d > 0 is radially outward."""
    theta = math.radians(135.0)
    heading = math.radians(45.0)
    for delta in (-0.03, 0.03):
        radius = R_IN + delta
        point = (
            CURVE_CORNER[0] + radius * math.cos(theta),
            CURVE_CORNER[1] + radius * math.sin(theta),
        )
        result = q1(graph, point[0], point[1], heading)
        assert float(result.d) == pytest.approx(delta, abs=1e-6)
    # sanity: the yellow arc sits at half a tile, further out than the 0.1755 m lane centre
    assert S.NOMINAL_TILE_SPEC.curve_radius_yellow_m > R_IN


def test_curve_outer_lane_is_a_left_turn_of_radius_0_4095(graph: LaneGraph) -> None:
    """Entering the same tile from the east heading west is a 0.4095 m left turn."""
    theta_entry = math.radians(90.0 + 2.0)
    entry = (
        CURVE_CORNER[0] + R_OUT * math.cos(theta_entry),
        CURVE_CORNER[1] + R_OUT * math.sin(theta_entry),
    )
    result = q1(graph, entry[0], entry[1], math.pi + math.radians(2.0))
    assert float(result.d) == pytest.approx(0.0, abs=1e-6)
    assert float(result.psi) == pytest.approx(0.0, abs=1e-6)
    assert float(result.curvature) == pytest.approx(+1.0 / R_OUT, rel=1e-6)  # positive = left
    theta = math.radians(135.0)
    for delta in (-0.03, 0.03):
        radius = R_OUT + delta
        point = (
            CURVE_CORNER[0] + radius * math.cos(theta),
            CURVE_CORNER[1] + radius * math.sin(theta),
        )
        # travelling counter-clockwise, the heading at 135 degrees is 225 degrees
        result = q1(graph, point[0], point[1], math.radians(225.0))
        assert float(result.d) == pytest.approx(-delta, abs=1e-6)


# ------------------------------------------------------------------------- direction lock
def test_a_robot_facing_backwards_stays_matched_to_its_own_lane(graph: LaneGraph) -> None:
    """The heading tie-break must never flip the match into the oncoming lane.

    If it did, driving backwards down one's own lane would earn positive progress, defeating
    the SPEC v2 S5.4 direction lock.
    """
    x = STRAIGHT_CENTER[0] + LANE
    forward = q1(graph, x, 0.0, math.pi / 2)
    backward = q1(graph, x, 0.0, -math.pi / 2)
    assert int(forward.seg_id) == int(backward.seg_id)
    assert abs(float(backward.psi)) == pytest.approx(math.pi, abs=1e-6)
    assert float(backward.d) == pytest.approx(0.0, abs=1e-6)
    # ... and the progress a backwards step earns is negative
    step = 0.02
    ds = progress_delta(
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        torch.tensor([-step]),
        backward.tangent_x,
        backward.tangent_y,
    )
    assert float(ds) == pytest.approx(-step, abs=1e-6)


def test_heading_tie_break_cannot_outweigh_lane_separation() -> None:
    """The default tie-break weight is far below the cost of jumping to the other lane."""
    from duckiebot_rl.city.lane_graph import DEFAULT_HEADING_WEIGHT

    max_heading_cost = 2.0 * DEFAULT_HEADING_WEIGHT
    lane_separation_cost = (2.0 * LANE) ** 2
    assert max_heading_cost < 0.1 * lane_separation_cost


def test_heading_tie_break_resolves_overlapping_intersection_lanes() -> None:
    """At an intersection entry several lanes coincide; heading picks the right one."""
    graph = LaneGraph(M.builtin_map("intersection_4way"))
    city = graph.city
    row, col = next((r, c) for r, c, t in city.iter_tiles() if t.kind == "fourway")
    cx, cy = city.tile_center_xy(row, col)
    entry = (cx - LANE, cy + HALF)  # entering from the north, heading south
    result = graph.query([entry[0]], [entry[1]], [-math.pi / 2])
    assert float(result.dist) == pytest.approx(0.0, abs=1e-6)
    assert float(result.psi) == pytest.approx(0.0, abs=1e-6)
    seg = graph.segments[int(result.seg_id)]
    assert (seg.row, seg.col) == (row, col)
    assert seg.entry_edge == "N"


# --------------------------------------------------------------- vectorised vs reference
def _dense_reference(
    graph: LaneGraph, x: float, y: float, yaw: float, step: float = 2e-4
) -> tuple[float, float, float, int]:
    """Brute-force nearest point on the lane network by dense sampling.

    An independent implementation: it never touches the analytic projection the module uses.

    Args:
        graph: The lane graph.
        x: World x.
        y: World y.
        yaw: Heading in radians.
        step: Sampling interval along each segment, in metres.

    Returns:
        ``(distance, d, psi, segment_index)``.
    """
    best = (float("inf"), 0.0, 0.0, -1)
    for index, seg in enumerate(graph.segments):
        n = max(2, round(seg.length / step) + 1)
        s = np.linspace(0.0, seg.length, n)
        if seg.is_arc:
            theta = seg.theta0 + seg.sweep_sign * (s / seg.radius)
            px = seg.center[0] + seg.radius * np.cos(theta)
            py = seg.center[1] + seg.radius * np.sin(theta)
            tx = -seg.sweep_sign * np.sin(theta)
            ty = seg.sweep_sign * np.cos(theta)
        else:
            px = seg.p0[0] + s * seg.t0[0]
            py = seg.p0[1] + s * seg.t0[1]
            tx = np.full_like(s, seg.t0[0])
            ty = np.full_like(s, seg.t0[1])
        dist = np.hypot(px - x, py - y)
        k = int(dist.argmin())
        if dist[k] < best[0]:
            d = tx[k] * (y - py[k]) - ty[k] * (x - px[k])
            error = yaw - math.atan2(ty[k], tx[k])
            psi = math.atan2(math.sin(error), math.cos(error))
            best = (float(dist[k]), float(d), float(psi), index)
    return best


@pytest.mark.parametrize("map_name", ["loop_small", "intersection_4way"])
def test_analytic_query_matches_a_dense_sampling_reference(map_name: str) -> None:
    """The vectorised analytic projection agrees with brute force to sub-sampling precision."""
    graph = LaneGraph(M.builtin_map(map_name))
    rng = np.random.default_rng(0)
    extent = graph.city.half_extent_m
    xs = rng.uniform(-extent, extent, 40)
    ys = rng.uniform(-extent, extent, 40)
    yaws = rng.uniform(-math.pi, math.pi, 40)
    batched = graph.query(xs, ys, yaws, heading_weight=0.0)
    # The reference discretises each segment at `step`, so its tangent direction carries an
    # error of up to step / R_IN radians; d and psi tolerances follow from that.
    angle_tol = 2e-4 / R_IN + 1e-5
    compared = 0
    for i in range(len(xs)):
        ref_dist, ref_d, ref_psi, ref_seg = _dense_reference(graph, xs[i], ys[i], yaws[i])
        assert float(batched.dist[i]) == pytest.approx(ref_dist, abs=2e-4)
        if int(batched.seg_id[i]) == ref_seg:
            compared += 1
            assert float(batched.d[i]) == pytest.approx(
                ref_d, abs=2e-4 + abs(ref_d) * angle_tol + ref_dist * angle_tol
            )
            assert float(batched.psi[i]) == pytest.approx(ref_psi, abs=angle_tol)
    assert compared > 0.7 * len(xs)


def test_batched_query_equals_single_pose_queries() -> None:
    """Querying N poses at once gives exactly what querying them one at a time gives."""
    graphs = [LaneGraph(M.builtin_map(n)) for n in M.BUILTIN_MAP_NAMES]
    batched = BatchedLaneGraph(graphs)
    rng = np.random.default_rng(5)
    n = 96
    variant = torch.from_numpy(rng.integers(0, len(graphs), n))
    x = torch.from_numpy(rng.uniform(-2.0, 2.0, n)).float()
    y = torch.from_numpy(rng.uniform(-2.0, 2.0, n)).float()
    yaw = torch.from_numpy(rng.uniform(-math.pi, math.pi, n)).float()
    together = batched.query(variant, x, y, yaw)
    for i in range(n):
        one = batched.query(variant[i : i + 1], x[i : i + 1], y[i : i + 1], yaw[i : i + 1])
        for field in ("d", "psi", "s", "curvature", "dist"):
            assert float(getattr(one, field)[0]) == pytest.approx(
                float(getattr(together, field)[i]), abs=1e-6
            ), field
        assert int(one.seg_id[0]) == int(together.seg_id[i])


def test_batched_graph_matches_its_single_map_graphs() -> None:
    """A variant inside a BatchedLaneGraph answers exactly as its standalone LaneGraph does."""
    graphs = [LaneGraph(M.builtin_map(n)) for n in ("loop_small", "zigzag")]
    batched = BatchedLaneGraph(graphs)
    rng = np.random.default_rng(9)
    x = rng.uniform(-1.5, 1.5, 30)
    y = rng.uniform(-1.5, 1.5, 30)
    yaw = rng.uniform(-math.pi, math.pi, 30)
    for index, graph in enumerate(graphs):
        alone = graph.query(x, y, yaw)
        joint = batched.query(torch.full((30,), index), x, y, yaw)
        assert torch.allclose(alone.d, joint.d, atol=1e-6)
        assert torch.allclose(alone.psi, joint.psi, atol=1e-6)


# ---------------------------------------------------------------- lane graph vs textures
def _texture_sampler(res: int = 1024) -> dict[str, np.ndarray]:
    """Render every drivable tile kind sharply, for overlay checks."""
    return {kind: T.render_tile(kind, res=res, supersample=1) for kind in ("straight", "curve")}


def test_lane_centreline_lies_midway_between_the_rendered_tapes() -> None:
    """SPEC v2 milestone M2: the lane graph overlays the rendered markings within 5 mm.

    For points along every lane of a loop map, march perpendicular to the lane in the tile's own
    texture and find the first white texel on the right and the first yellow texel on the left.
    Both must be half the clear lane width away.
    """
    spec = S.NOMINAL_TILE_SPEC
    res = 1024
    textures = _texture_sampler(res)
    city = M.builtin_map("loop_small")
    graph = LaneGraph(city, spec)
    step = 2e-4
    reach = spec.clear_lane_mm / 1000.0 / 2.0 + spec.white_tape_mm / 1000.0 + 0.02
    expected = spec.clear_lane_mm / 1000.0 / 2.0

    def first_hit(kind: str, qx: float, qy: float, nx: float, ny: float, color) -> float | None:
        texture = textures[kind]
        target = np.rint(np.asarray(color) * 255.0).astype(np.int16)
        for k in range(1, int(reach / step)):
            sx, sy = qx + nx * step * k, qy + ny * step * k
            if max(abs(sx), abs(sy)) >= spec.half_m:
                return None
            row, col = T.tile_local_to_pixel(spec, res, sx, sy)
            if np.abs(texture[row, col].astype(np.int16) - target).max() <= 8:
                return step * k
        return None

    checked_white = checked_yellow = 0
    for seg in graph.segments:
        kind = city.tiles[seg.row][seg.col].kind
        angle = -seg.row * 0.0  # placeholder, replaced below by the tile rotation
        del angle
        rot = city.tiles[seg.row][seg.col].rot
        theta = -rot * math.pi / 2.0
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        cx, cy = city.tile_center_xy(seg.row, seg.col)
        for frac in np.linspace(0.12, 0.88, 9):
            s = frac * seg.length
            if seg.is_arc:
                a = seg.theta0 + seg.sweep_sign * (s / seg.radius)
                px = seg.center[0] + seg.radius * math.cos(a)
                py = seg.center[1] + seg.radius * math.sin(a)
                tx = -seg.sweep_sign * math.sin(a)
                ty = seg.sweep_sign * math.cos(a)
            else:
                px, py = seg.p0[0] + s * seg.t0[0], seg.p0[1] + s * seg.t0[1]
                tx, ty = seg.t0

            # into the tile's own (unrotated) texture frame
            wx, wy = px - cx, py - cy
            qx = cos_t * wx - sin_t * wy
            qy = sin_t * wx + cos_t * wy
            lx_w, ly_w = -ty, tx  # left normal in world
            lx = cos_t * lx_w - sin_t * ly_w
            ly = sin_t * lx_w + cos_t * ly_w

            right = first_hit(kind, qx, qy, -lx, -ly, (1.0, 1.0, 1.0))
            assert right is not None, (kind, seg.entry_edge, seg.exit_edge, frac)
            assert right == pytest.approx(expected, abs=0.005)
            checked_white += 1

            left = first_hit(kind, qx, qy, lx, ly, (1.0, 1.0, 0.0))
            if left is not None:  # None means the sample fell in a dash gap
                assert left == pytest.approx(expected, abs=0.005)
                checked_yellow += 1

    assert checked_white == 16 * 9
    assert checked_yellow > 0.4 * checked_white


def test_lane_offset_tracks_a_rescaled_spec() -> None:
    """A map with a different pitch gets lanes scaled in tile units, not shifted in metres."""
    city = M.builtin_map("loop_small", tile_size=0.615)
    graph = LaneGraph(city, S.NOMINAL_TILE_SPEC)
    expected_offset = S.NOMINAL_TILE_SPEC.lane_center_offset_tile * 0.615
    straight = next(
        s
        for s in graph.segments
        if not s.is_arc and city.tiles[s.row][s.col].rot % 2 == 0  # north-south road
    )
    cx, _ = city.tile_center_xy(straight.row, straight.col)
    assert abs(straight.p0[0] - cx) == pytest.approx(expected_offset, abs=1e-9)
    batched = BatchedLaneGraph([graph])
    assert float(batched.lane_width[0]) == pytest.approx(
        S.NOMINAL_TILE_SPEC.clear_lane_mm / 585.0 * 0.615, abs=1e-6
    )


# -------------------------------------------------------------------------- other queries
def test_progress_delta_projects_onto_the_lane_tangent() -> None:
    """Progress is the world displacement projected on the tangent: signed and continuous."""
    tangent_x = torch.tensor([0.0, 1.0])
    tangent_y = torch.tensor([1.0, 0.0])
    prev_x = torch.tensor([0.0, 0.0])
    prev_y = torch.tensor([0.0, 0.0])
    x = torch.tensor([0.05, -0.05])
    y = torch.tensor([0.10, 0.20])
    ds = progress_delta(prev_x, prev_y, x, y, tangent_x, tangent_y)
    assert ds.tolist() == pytest.approx([0.10, -0.05])


def test_curvature_at_lookahead_follows_the_track_into_the_next_tile(graph: LaneGraph) -> None:
    """Approaching a curve, the 0.3 m lookahead reports the curve's signed curvature."""
    # just before the end of the straight that feeds the north-west curve
    x = STRAIGHT_CENTER[0] + LANE
    near_end = q1(graph, x, HALF - 0.05, math.pi / 2)
    assert float(near_end.curvature) == 0.0
    variant = torch.zeros(1, dtype=torch.long)
    batched = BatchedLaneGraph([graph])
    result = batched.query(variant, [x], [HALF - 0.05], [math.pi / 2])
    ahead = batched.curvature_at_lookahead(variant, result.seg_id, result.s, 0.30)
    assert abs(float(ahead)) == pytest.approx(1.0 / R_IN, rel=1e-5)
    here = batched.curvature_at_lookahead(variant, result.seg_id, result.s, 0.0)
    assert float(here) == 0.0


def test_route_progress_is_monotonic_around_a_closed_loop(graph: LaneGraph) -> None:
    """Walking the loop in 2 cm steps increases the cumulative route arc length monotonically."""
    batched = BatchedLaneGraph([graph])
    positions: list[tuple[float, float, float]] = []
    index = 0
    for _ in range(len(graph.segments) // 2 + 1):
        seg = graph.segments[index]
        for frac in np.linspace(0.05, 0.95, 12):
            s = frac * seg.length
            if seg.is_arc:
                a = seg.theta0 + seg.sweep_sign * (s / seg.radius)
                px = seg.center[0] + seg.radius * math.cos(a)
                py = seg.center[1] + seg.radius * math.sin(a)
                yaw = math.atan2(seg.sweep_sign * math.cos(a), -seg.sweep_sign * math.sin(a))
            else:
                px, py = seg.p0[0] + s * seg.t0[0], seg.p0[1] + s * seg.t0[1]
                yaw = math.atan2(seg.t0[1], seg.t0[0])
            positions.append((px, py, yaw))
        index = graph.successors[index][0]
    n = len(positions)
    variant = torch.zeros(n, dtype=torch.long)
    result = batched.query(
        variant,
        torch.tensor([p[0] for p in positions]),
        torch.tensor([p[1] for p in positions]),
        torch.tensor([p[2] for p in positions]),
    )
    route = batched.route_progress(variant, result.seg_id, result.s)
    steps = torch.diff(route)
    # monotonic all the way round, with at most one wrap back to the start of the cycle
    assert int((steps <= 0).sum()) <= 1
    assert float(steps[steps > 0].min()) > 0.0


def test_tile_index_and_drivability(graph: LaneGraph) -> None:
    """Drivability is a grid lookup; off-map points are not drivable."""
    batched = BatchedLaneGraph([graph])
    city = graph.city
    rows, cols, xs, ys = [], [], [], []
    for row in range(city.n_rows):
        for col in range(city.n_cols):
            x, y = city.tile_center_xy(row, col)
            rows.append(row)
            cols.append(col)
            xs.append(x)
            ys.append(y)
    variant = torch.zeros(len(xs), dtype=torch.long)
    got_row, got_col = batched.tile_index(variant, xs, ys)
    assert got_row.tolist() == rows
    assert got_col.tolist() == cols
    drivable = batched.is_drivable(variant, xs, ys)
    expected = [city.tiles[r][c].drivable for r, c in zip(rows, cols, strict=True)]
    assert drivable.tolist() == expected
    off_map = batched.is_drivable(torch.zeros(2, dtype=torch.long), [50.0, -50.0], [0.0, 0.0])
    assert not bool(off_map.any())


def test_off_drivable_boundary_is_where_the_tile_grid_says(graph: LaneGraph) -> None:
    """The drivable test flips exactly at the analytic tile boundary (M3 uses this)."""
    batched = BatchedLaneGraph([graph])
    city = graph.city
    # tile (2, 1) is drivable, tile (2, 0) is grass; their shared boundary is at x = -0.8775
    boundary = city.tile_center_xy(2, 1)[0] - HALF
    variant = torch.zeros(2, dtype=torch.long)
    result = batched.is_drivable(variant, [boundary + 1e-4, boundary - 1e-4], [0.0, 0.0])
    assert result.tolist() == [True, False]


# ---------------------------------------------------------------------------- error paths
def test_query_validates_its_inputs(graph: LaneGraph) -> None:
    """Shape and range mistakes are reported rather than silently broadcast."""
    batched = BatchedLaneGraph([graph])
    with pytest.raises(ValueError, match="batch shapes disagree"):
        batched.query(torch.zeros(2, dtype=torch.long), [0.0, 1.0, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="variant index out of range"):
        batched.query(torch.tensor([3]), [0.0], [0.0], [0.0])
    with pytest.raises(ValueError, match="at least one LaneGraph"):
        BatchedLaneGraph([])


def test_wrap_to_pi_is_the_documented_interval() -> None:
    """Angles come back in (-pi, pi]."""
    angles = torch.tensor([0.0, math.pi, -math.pi, 3 * math.pi, -3.5 * math.pi])
    wrapped = wrap_to_pi(angles)
    assert torch.all(wrapped > -math.pi - 1e-6)
    assert torch.all(wrapped <= math.pi + 1e-6)
    assert float(wrapped[0]) == pytest.approx(0.0)
    assert abs(float(wrapped[1])) == pytest.approx(math.pi)


def test_lane_construction_rejects_inconsistent_geometry() -> None:
    """The builder asserts its own arcs; a spec whose lane offset breaks them is caught."""
    city = M.builtin_map("loop_small")
    segments = build_lane_segments(city, S.TileSpec(lane_center_offset_mm=0.22 * 585.0))
    # a different but still self-consistent convention simply moves the radii
    arcs = [s for s in segments if s.is_arc]
    assert sorted({round(s.radius / city.tile_size, 4) for s in arcs}) == [0.28, 0.72]


# =============================================================================================
# curvature_at_lookahead: branchless hops (profile rank 5)
# =============================================================================================


def _curvature_with_early_break(
    batched: BatchedLaneGraph,
    variant_idx: torch.Tensor,
    seg_id: torch.Tensor,
    s: torch.Tensor,
    distance: float,
    max_hops: int = 4,
) -> torch.Tensor:
    """Return the lookahead curvature using the ORIGINAL early-breaking hop loop.

    Transcribed from the implementation as it stood before the branch was removed, so that the
    branchless version is compared against the thing it replaced rather than against itself.

    Args:
        batched: A ``BatchedLaneGraph``.
        variant_idx: ``(B,)`` variant index.
        seg_id: ``(B,)`` current segment.
        s: ``(B,)`` arc length within the segment.
        distance: Lookahead distance in metres.
        max_hops: Maximum segment transitions to follow.

    Returns:
        ``(B,)`` signed curvature.
    """
    vidx = torch.as_tensor(variant_idx, dtype=torch.long, device=batched.device).reshape(-1)
    cur = seg_id.clone()
    remaining = s + float(distance)
    for _ in range(max_hops):
        seg_len = batched.seg_length[vidx, cur]
        overflow = remaining > seg_len
        if not bool(overflow.any()):
            break
        nxt = batched.seg_primary[vidx, cur]
        remaining = torch.where(overflow, remaining - seg_len, remaining)
        cur = torch.where(overflow, nxt, cur)
    return batched.seg_curvature[vidx, cur]


@pytest.mark.parametrize("distance", [0.0, 0.05, 0.3, 1.0, 5.0, 40.0])
def test_branchless_curvature_is_bit_identical_to_the_early_breaking_loop(graph, distance):
    """Once ``overflow`` is all-False both ``where`` calls are the identity, so the hops are free.

    The distances span a lookahead that never overflows (where the old loop broke on hop 1), the
    S5.2 lookahead of 0.3 m, and one long enough to exhaust every hop on every env.
    """
    batched = BatchedLaneGraph([graph])
    generator = torch.Generator().manual_seed(4)
    count = 96
    num_segments = int(batched.seg_valid[0].sum())
    variant_idx = torch.zeros(count, dtype=torch.long)
    seg_id = torch.randint(0, num_segments, (count,), generator=generator)
    s = torch.rand(count, generator=generator) * batched.seg_length[0, seg_id]

    branchless = batched.curvature_at_lookahead(variant_idx, seg_id, s, distance)
    reference = _curvature_with_early_break(batched, variant_idx, seg_id, s, distance)
    assert torch.equal(branchless, reference)


def test_curvature_lookahead_performs_no_host_sync(graph, count_host_syncs):
    """Profile rank 5: the early break cost up to 4 syncs per control step on the critic path."""
    batched = BatchedLaneGraph([graph])
    seg_id = torch.zeros(16, dtype=torch.long)
    with count_host_syncs() as syncs:
        batched.curvature_at_lookahead(torch.zeros(16, dtype=torch.long), seg_id, torch.zeros(16), 0.3)
        assert syncs() == 0
