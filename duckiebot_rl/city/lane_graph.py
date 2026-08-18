"""Analytic lane-centreline graph and the ground-truth queries the reward needs.

This module turns a :class:`~.maps.CityMap` plus a :class:`~.spec.TileSpec` into an exact
geometric description of every lane, and answers, for a batch of robot poses, the three
quantities SPEC v2 S5.4 needs: the signed lateral offset ``d``, the heading error ``psi``, and
lane-frame forward progress.

Sign conventions (SPEC v2 S2, and the reason this module has a test file of its own)
------------------------------------------------------------------------------------
* ``d > 0`` means the robot is displaced to the **left** of its lane centreline, i.e. **toward
  the yellow centre tape**. ``d < 0`` is toward the white edge tape. Concretely, ``d`` is the
  cross product ``tangent x (robot - closest_point)``, so with a lane tangent pointing east a
  robot displaced north has ``d > 0``.
* ``psi > 0`` means the robot heading is rotated **counter-clockwise (left)** of the lane
  tangent. ``psi = wrap(yaw - atan2(t_y, t_x))`` in ``(-pi, pi]``.
* Signed curvature ``kappa > 0`` means the lane turns left.
* With these, ``psi_target = -clip(d / 0.05, -1, 1) * 45 deg`` steers back toward the centreline
  for both signs of ``d``, which is what S5.4 requires.

Geometry
--------
Right-hand traffic: every ordered pair of open edges ``(entry, exit)`` of a drivable tile carries
one lane. Entering through edge ``e`` the robot travels along ``-n_e`` and its lane centre sits at
``+lane_offset * u_e`` from the edge midpoint (``n_e`` is the outward normal, ``u_e = R(+90) n_e``);
leaving through edge ``f`` it travels along ``+n_f`` with its lane centre at ``-lane_offset * u_f``.
Opposite edges give a straight segment, adjacent edges an exact quarter-circle arc centred on the
shared tile corner. That yields 2 lanes per straight or curve tile, 6 per 3-way and 12 per 4-way,
matching the Duckietown lane templates.

The lane offset is taken from the :class:`~.spec.TileSpec` in **tile units** and scaled by the
map's own ``tile_size``, because the marking textures are painted per geometry bucket while the
layout carries its own pitch. Use :meth:`~.spec.TileSpec.rescaled` if you want the millimetre
figures for a particular map.

Matching
--------
A pose is matched to a lane by exact analytic projection onto lane segments (there are tens,
not thousands, so no spatial index or polyline sampling is needed and no discretisation error is
introduced), choosing the segment that minimises ``dist^2 + heading_weight * (1 - cos(psi))``.
The heading term is a *tie-break*: the default weight of 0.002 m^2 can shift the decision by at
most 0.063 m of equivalent distance, so for a robot inside its own lane the match is simply the
nearest directed centreline.

The tie-break is NOT sufficient for a robot that leaves its lane, and this is load bearing for
the reward: the two lanes of a road are only 0.234 m apart, so a free nearest-segment search
re-homes a robot about halfway across the centreline onto the adjacent lane, collapsing ``|d|``
exactly when it should be growing and blinding every reward and termination term derived from
it. At a hairpin the adjacent lane belongs to the robot's own route a few segments ahead, so a
corner cut was reported as flawless driving plus a free jump along the route. Matching is
therefore *continuity constrained*: callers that track poses over time pass ``prev_route_pos``,
and the search is restricted to segments within :data:`MATCH_WINDOW_M` of route arc length,
with a free search at spawn and as the fallback if the window ever empties. Under the window,
``d`` stays truthful however far the robot strays, driving backwards yields ``psi ~ pi`` and
negative ``ds``, and cutting pays its real cost.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final, NamedTuple

import numpy as np
import torch

from .maps import CityMap
from .spec import NOMINAL_TILE_SPEC, TileSpec
from .tiles import EDGE_BASIS

__all__ = [
    "DEFAULT_HEADING_WEIGHT",
    "MATCH_WINDOW_M",
    "BatchedLaneGraph",
    "LaneGraph",
    "LaneQuery",
    "LaneSegment",
    "build_lane_segments",
    "progress_delta",
    "wrap_to_pi",
]

MATCH_WINDOW_M: Final[float] = 0.35
"""Half-width of the route window for continuity-constrained lane matching, in metres.

Chosen against the two failure directions. Too small and a legitimate fast step could leave the
window: the robot covers at most ``v_max * dt_c`` = 0.62 * 0.067 = 0.041 m per control step, so
0.35 m is an 8-step cushion, and even a physics glitch cannot strand the match because a window
that excludes everything falls back to the free search. Too large and the window stops excluding
what it exists to exclude: the apex arc of the tightest hairpin is only about 0.5 m of route, so
a window much above that lets a mid-cut robot be matched to the post-apex return lane it has not
yet earned, which is the d-collapsing re-home this window was built to prevent.
"""

DEFAULT_HEADING_WEIGHT: Final[float] = 0.002
"""Tie-break weight on the heading term of the lane-matching cost, in metres squared."""

_OPPOSITE: Final[dict[str, str]] = {"N": "S", "S": "N", "E": "W", "W": "E"}
_EDGE_STEP: Final[dict[str, tuple[int, int]]] = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1)}


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angles into ``(-pi, pi]``.

    Args:
        angle: Angles in radians, any shape.

    Returns:
        The wrapped angles.
    """
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def progress_delta(
    prev_x: torch.Tensor,
    prev_y: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    tangent_x: torch.Tensor,
    tangent_y: torch.Tensor,
) -> torch.Tensor:
    """Lane-frame forward progress over one control step.

    This is ``ds`` in the SPEC v2 S5.4 reward: the world displacement projected onto the lane
    tangent at the *current* pose. Projecting onto the tangent rather than differencing an arc
    length means the value is continuous across lane-segment and tile boundaries, and it is
    negative when the robot drives against the lane direction, which is the direction lock.

    Args:
        prev_x: Previous world x, shape ``(B,)``.
        prev_y: Previous world y, shape ``(B,)``.
        x: Current world x, shape ``(B,)``.
        y: Current world y, shape ``(B,)``.
        tangent_x: Lane tangent x at the current pose, shape ``(B,)``.
        tangent_y: Lane tangent y at the current pose, shape ``(B,)``.

    Returns:
        Signed progress in metres, shape ``(B,)``.
    """
    return (x - prev_x) * tangent_x + (y - prev_y) * tangent_y


@dataclass(frozen=True)
class LaneSegment:
    """One lane centreline: a straight segment or an exact circular arc.

    Attributes:
        is_arc: ``True`` for an arc, ``False`` for a straight segment.
        p0: Start point ``(x, y)`` in world metres.
        p1: End point ``(x, y)`` in world metres.
        t0: Unit tangent at ``p0``.
        t1: Unit tangent at ``p1``.
        length: Arc length in metres.
        center: Arc centre ``(x, y)``; ``(0, 0)`` for straight segments.
        radius: Arc radius in metres; ``1.0`` placeholder for straight segments.
        theta0: Arc start angle in radians; ``0.0`` for straight segments.
        sweep_sign: ``+1`` for a counter-clockwise (left) turn, ``-1`` for clockwise;
            ``+1`` placeholder for straight segments.
        sweep: Absolute angular extent in radians; ``0.0`` for straight segments.
        curvature: Signed curvature ``sweep_sign / radius``; ``0.0`` for straight segments.
        row: Grid row of the tile the lane lies on.
        col: Grid column of the tile.
        entry_edge: Tile edge the lane enters through.
        exit_edge: Tile edge the lane leaves through.
    """

    is_arc: bool
    p0: tuple[float, float]
    p1: tuple[float, float]
    t0: tuple[float, float]
    t1: tuple[float, float]
    length: float
    center: tuple[float, float]
    radius: float
    theta0: float
    sweep_sign: float
    sweep: float
    curvature: float
    row: int
    col: int
    entry_edge: str
    exit_edge: str


def _unit(vx: float, vy: float) -> tuple[float, float]:
    """Normalise a 2-vector.

    Args:
        vx: x component.
        vy: y component.

    Returns:
        The unit vector.

    Raises:
        ValueError: If the vector is degenerate.
    """
    norm = math.hypot(vx, vy)
    if norm < 1e-12:
        raise ValueError("cannot normalise a zero-length vector")
    return vx / norm, vy / norm


def build_lane_segments(
    city: CityMap, spec: TileSpec = NOMINAL_TILE_SPEC, tol: float = 1e-9
) -> list[LaneSegment]:
    """Enumerate every lane centreline of a map.

    Args:
        city: A validated map.
        spec: Marking geometry; only its lane-centre offset *in tile units* is used, scaled by
            the map's own ``tile_size``.
        tol: Tolerance for the internal geometric consistency assertions, in metres/radians.

    Returns:
        Lane segments in a deterministic order: tiles row-major, then entry edge, then exit edge,
        both in ``N, E, S, W`` order.

    Raises:
        ValueError: If a constructed segment fails its geometric consistency check, which would
            mean the tile connectivity and the marking geometry disagree.
    """
    pitch = city.tile_size
    half = 0.5 * pitch
    lane = spec.lane_center_offset_tile * pitch
    order = ("N", "E", "S", "W")
    segments: list[LaneSegment] = []

    for row, col, tile in city.iter_tiles():
        if not tile.drivable:
            continue
        cx, cy = city.tile_center_xy(row, col)
        open_edges = [e for e in order if e in tile.open_edges]
        for entry in open_edges:
            (enx, eny), (eux, euy) = EDGE_BASIS[entry]
            p_in = (cx + half * enx + lane * eux, cy + half * eny + lane * euy)
            t_in = (-enx, -eny)
            for exit_edge in open_edges:
                if exit_edge == entry:
                    continue
                (fnx, fny), (fux, fuy) = EDGE_BASIS[exit_edge]
                p_out = (cx + half * fnx - lane * fux, cy + half * fny - lane * fuy)
                t_out = (fnx, fny)
                if abs(enx + fnx) < tol and abs(eny + fny) < tol:
                    # opposite edges: a straight segment
                    dx, dy = p_out[0] - p_in[0], p_out[1] - p_in[1]
                    length = math.hypot(dx, dy)
                    tangent = _unit(dx, dy)
                    if abs(tangent[0] - t_in[0]) > 1e-6 or abs(tangent[1] - t_in[1]) > 1e-6:
                        raise ValueError(
                            f"straight lane {entry}->{exit_edge} on tile ({row}, {col}) has tangent "
                            f"{tangent} but enters heading {t_in}"
                        )
                    segments.append(
                        LaneSegment(
                            is_arc=False,
                            p0=p_in,
                            p1=p_out,
                            t0=tangent,
                            t1=tangent,
                            length=length,
                            center=(0.0, 0.0),
                            radius=1.0,
                            theta0=0.0,
                            sweep_sign=1.0,
                            sweep=0.0,
                            curvature=0.0,
                            row=row,
                            col=col,
                            entry_edge=entry,
                            exit_edge=exit_edge,
                        )
                    )
                    continue
                # adjacent edges: quarter arc centred on the shared tile corner
                corner_x = cx + half * (enx + fnx)
                corner_y = cy + half * (eny + fny)
                r_in = math.hypot(p_in[0] - corner_x, p_in[1] - corner_y)
                r_out = math.hypot(p_out[0] - corner_x, p_out[1] - corner_y)
                if abs(r_in - r_out) > 1e-9:
                    raise ValueError(
                        f"arc lane {entry}->{exit_edge} on tile ({row}, {col}) is not circular: "
                        f"entry radius {r_in:.9f} m, exit radius {r_out:.9f} m"
                    )
                theta0 = math.atan2(p_in[1] - corner_y, p_in[0] - corner_x)
                theta1 = math.atan2(p_out[1] - corner_y, p_out[0] - corner_x)
                delta = (theta1 - theta0 + math.pi) % (2.0 * math.pi) - math.pi
                sign = 1.0 if delta > 0.0 else -1.0
                sweep = abs(delta)
                if abs(sweep - math.pi / 2.0) > 1e-9:
                    raise ValueError(
                        f"arc lane {entry}->{exit_edge} on tile ({row}, {col}) sweeps "
                        f"{math.degrees(sweep):.4f} deg, expected 90"
                    )
                tan0 = (-sign * math.sin(theta0), sign * math.cos(theta0))
                if abs(tan0[0] - t_in[0]) > 1e-6 or abs(tan0[1] - t_in[1]) > 1e-6:
                    raise ValueError(
                        f"arc lane {entry}->{exit_edge} on tile ({row}, {col}) starts with tangent "
                        f"{tan0} but enters heading {t_in}"
                    )
                tan1 = (-sign * math.sin(theta1), sign * math.cos(theta1))
                if abs(tan1[0] - t_out[0]) > 1e-6 or abs(tan1[1] - t_out[1]) > 1e-6:
                    raise ValueError(
                        f"arc lane {entry}->{exit_edge} on tile ({row}, {col}) ends with tangent "
                        f"{tan1} but exits heading {t_out}"
                    )
                segments.append(
                    LaneSegment(
                        is_arc=True,
                        p0=p_in,
                        p1=p_out,
                        t0=tan0,
                        t1=tan1,
                        length=r_in * sweep,
                        center=(corner_x, corner_y),
                        radius=r_in,
                        theta0=theta0,
                        sweep_sign=sign,
                        sweep=sweep,
                        curvature=sign / r_in,
                        row=row,
                        col=col,
                        entry_edge=entry,
                        exit_edge=exit_edge,
                    )
                )
    return segments


class LaneQuery(NamedTuple):
    """Result of a batched lane query. Every field has shape ``(B,)`` unless noted.

    Attributes:
        d: Signed lateral offset in metres; positive is left of the centreline, toward yellow.
        psi: Heading error in radians, wrapped to ``(-pi, pi]``; positive is left of the tangent.
        s: Arc length from the start of the matched lane segment, in metres.
        seg_id: Index of the matched segment within its variant's segment table.
        tangent_x: Lane tangent x at the closest point.
        tangent_y: Lane tangent y at the closest point.
        curvature: Signed lane curvature in ``1/m``; positive turns left.
        closest_x: World x of the closest point on the lane centreline.
        closest_y: World y of the closest point on the lane centreline.
        dist: Euclidean distance to the closest point, in metres. Equals ``|d|`` except at
            segment ends, where the projection clamps.
    """

    d: torch.Tensor
    psi: torch.Tensor
    s: torch.Tensor
    seg_id: torch.Tensor
    tangent_x: torch.Tensor
    tangent_y: torch.Tensor
    curvature: torch.Tensor
    closest_x: torch.Tensor
    closest_y: torch.Tensor
    dist: torch.Tensor


class LaneGraph:
    """The lane network of a single map, with successor links and a drivable-tile mask.

    Attributes:
        city: The source map.
        spec: The marking geometry the lanes were derived from.
        segments: Every lane segment, in the order produced by :func:`build_lane_segments`.
    """

    def __init__(self, city: CityMap, spec: TileSpec = NOMINAL_TILE_SPEC) -> None:
        """Build the lane graph of a map.

        Args:
            city: A map; it is validated before use.
            spec: Marking geometry.
        """
        self.city = city.validate()
        self.spec = spec
        self.segments = build_lane_segments(city, spec)
        self.successors = self._build_successors()
        self.primary_successors = self._build_primary()
        self.route_offsets, self.has_route = self._build_route()
        self._batched: BatchedLaneGraph | None = None

    # ------------------------------------------------------------------------ graph structure
    def _build_successors(self) -> list[list[int]]:
        """Map each segment to the segments that may follow it across a tile boundary."""
        by_entry: dict[tuple[int, int, str], list[int]] = {}
        for i, seg in enumerate(self.segments):
            by_entry.setdefault((seg.row, seg.col, seg.entry_edge), []).append(i)
        out: list[list[int]] = []
        for seg in self.segments:
            d_row, d_col = _EDGE_STEP[seg.exit_edge]
            key = (seg.row + d_row, seg.col + d_col, _OPPOSITE[seg.exit_edge])
            out.append(list(by_entry.get(key, ())))
        return out

    def _build_primary(self) -> list[int]:
        """Pick the straightest successor of each segment, used for lookahead curvature."""
        out: list[int] = []
        for i, seg in enumerate(self.segments):
            options = self.successors[i]
            if not options:
                out.append(i)
                continue
            best = min(
                options,
                key=lambda j: abs(
                    (
                        math.atan2(self.segments[j].t0[1], self.segments[j].t0[0])
                        - math.atan2(seg.t1[1], seg.t1[0])
                        + math.pi
                    )
                    % (2.0 * math.pi)
                    - math.pi
                ),
            )
            out.append(best)
        return out

    def _build_route(self) -> tuple[list[float], bool]:
        """Cumulative arc length along each directed lane cycle, when the map is a pure loop.

        A map is a UNION of directed cycles, not one cycle: a plain loop map carries one cycle
        per travel direction, and on ``zigzag`` those are 10.93 m and 9.46 m long. Offsets
        therefore restart at zero in each cycle and are only comparable within one, which is why
        this also records :attr:`segment_cycle` and :attr:`segment_cycle_length`. Comparing two
        offsets without checking they share a cycle, or reducing them modulo the wrong cycle's
        length, silently mixes up two different places on the map.

        Returns:
            ``(route_offsets, has_route)``. When every segment has exactly one successor the map
            is a union of directed cycles and the offsets are cumulative arc lengths within each
            cycle; otherwise the offsets are all zero and ``has_route`` is ``False``.
        """
        offsets = [0.0] * len(self.segments)
        self.segment_cycle = [-1] * len(self.segments)
        self.segment_cycle_length = [0.0] * len(self.segments)
        if any(len(s) != 1 for s in self.successors):
            return offsets, False
        seen: set[int] = set()
        for start in range(len(self.segments)):
            if start in seen:
                continue
            cycle_id = len(set(self.segment_cycle) - {-1})
            cur, total, members = start, 0.0, []
            while cur not in seen:
                seen.add(cur)
                members.append(cur)
                offsets[cur] = total
                total += self.segments[cur].length
                self.segment_cycle[cur] = cycle_id
                cur = self.successors[cur][0]
            for member in members:
                self.segment_cycle_length[member] = total
        return offsets, True

    @property
    def total_lane_length(self) -> float:
        """Sum of all lane-segment lengths in metres (both directions of travel)."""
        return float(sum(seg.length for seg in self.segments))

    def drivable_mask(self) -> np.ndarray:
        """Boolean ``(n_rows, n_cols)`` mask of drivable grid cells."""
        return np.array([[t.drivable for t in line] for line in self.city.tiles], dtype=bool)

    def query(
        self,
        x: Any,
        y: Any,
        yaw: Any,
        heading_weight: float = DEFAULT_HEADING_WEIGHT,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        prev_seg_id: torch.Tensor | None = None,
        prev_s: torch.Tensor | None = None,
        window_m: float = MATCH_WINDOW_M,
    ) -> LaneQuery:
        """Query this single map. See :meth:`BatchedLaneGraph.query`.

        Args:
            x: World x, scalar or shape ``(B,)``.
            y: World y, scalar or shape ``(B,)``.
            yaw: Heading in radians, scalar or shape ``(B,)``.
            heading_weight: Tie-break weight in metres squared.
            device: Torch device; defaults to CPU.
            dtype: Floating dtype.
            prev_seg_id: Previously matched segments for continuity-constrained matching,
                ``-1`` for a free search; see :meth:`BatchedLaneGraph.query`.
            prev_s: Arc length along ``prev_seg_id``.
            window_m: Route half-width of the window in metres.

        Returns:
            The lane query result.
        """
        if (
            self._batched is None
            or self._batched.dtype != dtype
            or (device is not None and self._batched.device != torch.device(device))
        ):
            self._batched = BatchedLaneGraph([self], device=device, dtype=dtype)
        batched = self._batched
        n = torch.as_tensor(x, dtype=dtype, device=batched.device).reshape(-1).numel()
        variant = torch.zeros(n, dtype=torch.long, device=batched.device)
        return batched.query(
            variant,
            x,
            y,
            yaw,
            heading_weight=heading_weight,
            prev_seg_id=prev_seg_id,
            prev_s=prev_s,
            window_m=window_m,
        )


class BatchedLaneGraph:
    """Torch engine answering lane queries for ``N`` environments over ``V`` map variants.

    Every per-variant table is padded to the largest variant and masked, so a single batched
    gather serves all environments. Matching is exact analytic projection onto every segment;
    with tens of segments per map this costs a handful of elementwise kernels on an
    ``(N, S_max)`` tensor and introduces no discretisation error.

    Attributes:
        graphs: The per-variant lane graphs, in variant order.
        device: Torch device the tables live on.
        dtype: Floating dtype of the tables.
        num_variants: ``V``.
        max_segments: ``S_max``.
        lane_width: ``(V,)`` clear lane width in metres, the ``w_ep`` of the S5.4 reward.
    """

    def __init__(
        self,
        graphs: Sequence[LaneGraph],
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Stack lane graphs into padded tensors.

        Args:
            graphs: One :class:`LaneGraph` per city variant, in variant order.
            device: Torch device; defaults to CPU.
            dtype: Floating dtype; SPEC v2 S6.7 forbids reduced precision in the training path.

        Raises:
            ValueError: If ``graphs`` is empty.
        """
        if not graphs:
            raise ValueError("BatchedLaneGraph needs at least one LaneGraph")
        self.graphs = list(graphs)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype
        self.num_variants = len(self.graphs)
        self.max_segments = max(len(g.segments) for g in self.graphs)

        v, s = self.num_variants, self.max_segments
        zeros = lambda *shape: torch.zeros(*shape, dtype=dtype)  # noqa: E731
        is_arc = torch.zeros((v, s), dtype=torch.bool)
        valid = torch.zeros((v, s), dtype=torch.bool)
        p0, t0, center = zeros(v, s, 2), zeros(v, s, 2), zeros(v, s, 2)
        length, radius, theta0 = zeros(v, s), torch.ones((v, s), dtype=dtype), zeros(v, s)
        sign, curvature, route = torch.ones((v, s), dtype=dtype), zeros(v, s), zeros(v, s)
        sweep = zeros(v, s)
        primary = torch.zeros((v, s), dtype=torch.long)
        tile_row = torch.zeros((v, s), dtype=torch.long)
        tile_col = torch.zeros((v, s), dtype=torch.long)

        for vi, graph in enumerate(self.graphs):
            for si, seg in enumerate(graph.segments):
                valid[vi, si] = True
                is_arc[vi, si] = seg.is_arc
                p0[vi, si] = torch.tensor(seg.p0, dtype=dtype)
                t0[vi, si] = torch.tensor(seg.t0, dtype=dtype)
                center[vi, si] = torch.tensor(seg.center, dtype=dtype)
                length[vi, si] = seg.length
                radius[vi, si] = seg.radius
                theta0[vi, si] = seg.theta0
                sign[vi, si] = seg.sweep_sign
                sweep[vi, si] = seg.sweep
                curvature[vi, si] = seg.curvature
                route[vi, si] = graph.route_offsets[si]
                primary[vi, si] = graph.primary_successors[si]
                tile_row[vi, si] = seg.row
                tile_col[vi, si] = seg.col

        self.seg_is_arc = is_arc.to(self.device)
        self.seg_valid = valid.to(self.device)
        self.seg_p0 = p0.to(self.device)
        self.seg_t0 = t0.to(self.device)
        self.seg_center = center.to(self.device)
        self.seg_length = length.to(self.device)
        self.seg_radius = radius.to(self.device)
        self.seg_theta0 = theta0.to(self.device)
        self.seg_sign = sign.to(self.device)
        self.seg_sweep = sweep.to(self.device)
        self.seg_curvature = curvature.to(self.device)
        self.seg_route_s = route.to(self.device)
        self.seg_primary = primary.to(self.device)
        self.seg_tile_row = tile_row.to(self.device)
        self.seg_tile_col = tile_col.to(self.device)

        self.max_rows = max(g.city.n_rows for g in self.graphs)
        self.max_cols = max(g.city.n_cols for g in self.graphs)
        grid = torch.zeros((v, self.max_rows, self.max_cols), dtype=torch.bool)
        for vi, graph in enumerate(self.graphs):
            mask = torch.from_numpy(graph.drivable_mask())
            grid[vi, : mask.shape[0], : mask.shape[1]] = mask
        self.grid_drivable = grid.to(self.device)
        self.grid_rows = torch.tensor([g.city.n_rows for g in self.graphs], device=self.device)
        self.grid_cols = torch.tensor([g.city.n_cols for g in self.graphs], device=self.device)
        self.origin = torch.tensor(
            [list(g.city.origin_xy) for g in self.graphs], dtype=dtype, device=self.device
        )
        self.pitch = torch.tensor([g.city.tile_size for g in self.graphs], dtype=dtype, device=self.device)
        self.lane_width = torch.tensor(
            [g.spec.clear_lane_mm / g.spec.tile_pitch_mm * g.city.tile_size for g in self.graphs],
            dtype=dtype,
            device=self.device,
        )
        self.variant_has_route = torch.tensor([g.has_route for g in self.graphs], device=self.device)
        cycle_id = torch.full((v, s), -1, dtype=torch.long)
        cycle_len = torch.ones((v, s), dtype=dtype)
        for vi, graph in enumerate(self.graphs):
            for si in range(len(graph.segments)):
                cycle_id[vi, si] = graph.segment_cycle[si]
                cycle_len[vi, si] = max(graph.segment_cycle_length[si], 1e-6)
        self.seg_cycle = cycle_id.to(self.device)
        self.seg_cycle_length = cycle_len.to(self.device)

        # loop circumference per variant; 1.0 (not 0) where there is no route so that the
        # windowed-matching remainder below never divides by zero on intersection maps, where
        # the window is masked off anyway

    # ------------------------------------------------------------------------------- helpers
    def _as_batch(self, value: Any) -> torch.Tensor:
        """Coerce a scalar, array or tensor to a 1-D tensor on this graph's device and dtype."""
        return torch.as_tensor(value, dtype=self.dtype, device=self.device).reshape(-1)

    # --------------------------------------------------------------------------------- query
    def query(
        self,
        variant_idx: Any,
        x: Any,
        y: Any,
        yaw: Any,
        heading_weight: float = DEFAULT_HEADING_WEIGHT,
        prev_seg_id: torch.Tensor | None = None,
        prev_s: torch.Tensor | None = None,
        window_m: float = MATCH_WINDOW_M,
    ) -> LaneQuery:
        """Match a batch of poses to lanes and return the reward ground truth.

        Args:
            variant_idx: ``(B,)`` integer index of each environment's map variant.
            x: ``(B,)`` world x in metres.
            y: ``(B,)`` world y in metres.
            yaw: ``(B,)`` heading in radians, counter-clockwise from ``+x``.
            heading_weight: Tie-break weight in metres squared; see the module docstring.
            prev_seg_id: ``(B,)`` segment each environment was matched to on the PREVIOUS step,
                with ``-1`` meaning "not tracked yet", which asks for a free global match. When
                given, matching is constrained to segments reachable within ``window_m`` of route
                arc length IN THE SAME CYCLE. This is what makes ``d`` truthful when the robot
                leaves its lane: without it, the nearest-segment search re-homes a robot that
                crosses the centreline onto the adjacent lane, so ``|d|`` collapses exactly when
                it should be growing, every reward term derived from it goes blind, and cutting a
                hairpin is reported as flawless driving a few segments further along. Measured on
                ``city_000`` before the window existed: crossing the centreline flipped
                ``seg_id`` 2 to 3 and collapsed ``|d|`` 0.103 m to 0.003 m.

                The cycle check is not a detail. A map is a union of directed cycles, one per
                travel direction, each with its own zero and its own length (``zigzag``: 10.93 m
                and 9.46 m). An earlier version of this window compared offsets across cycles and
                reduced them modulo the longest one; on the shorter cycle the arithmetic was off
                by 1.47 m against a 0.35 m window, so legitimate successors were excluded, the
                match pinned to a stale segment, and ``d`` grew without bound as the robot drove
                away from it. That regression reached a live training run and is what
                ``test_a_full_lap_keeps_the_match_on_the_robot`` now guards.
            prev_s: ``(B,)`` arc length along ``prev_seg_id``; zeros when omitted.
            window_m: Half-width of the allowed route interval in metres. The default clears one
                tile comfortably while keeping the return lane of a hairpin, one lane separation
                away in space but several tiles away along the route, out of reach.

        Returns:
            A :class:`LaneQuery` whose fields all have shape ``(B,)``.

        Raises:
            ValueError: If the batch shapes disagree or a variant index is out of range.
        """
        px = self._as_batch(x)
        py = self._as_batch(y)
        yaw_t = self._as_batch(yaw)
        vidx = torch.as_tensor(variant_idx, dtype=torch.long, device=self.device).reshape(-1)
        if vidx.numel() == 1 and px.numel() > 1:
            vidx = vidx.expand(px.numel())
        if not (px.shape == py.shape == yaw_t.shape == vidx.shape):
            raise ValueError(
                f"batch shapes disagree: x {tuple(px.shape)}, y {tuple(py.shape)}, "
                f"yaw {tuple(yaw_t.shape)}, variant_idx {tuple(vidx.shape)}"
            )
        if int(vidx.max()) >= self.num_variants or int(vidx.min()) < 0:
            raise ValueError(f"variant index out of range for {self.num_variants} variants")

        is_arc = self.seg_is_arc[vidx]
        valid = self.seg_valid[vidx]
        p0 = self.seg_p0[vidx]
        t0 = self.seg_t0[vidx]
        center = self.seg_center[vidx]
        seg_len = self.seg_length[vidx]
        radius = self.seg_radius[vidx]
        theta0 = self.seg_theta0[vidx]
        sign = self.seg_sign[vidx]
        sweep = self.seg_sweep[vidx]

        qx = px[:, None]
        qy = py[:, None]

        # --- straight branch -------------------------------------------------------------
        dx, dy = qx - p0[..., 0], qy - p0[..., 1]
        s_line = torch.minimum(torch.clamp(dx * t0[..., 0] + dy * t0[..., 1], min=0.0), seg_len)
        cx_line = p0[..., 0] + s_line * t0[..., 0]
        cy_line = p0[..., 1] + s_line * t0[..., 1]

        # --- arc branch ------------------------------------------------------------------
        vx, vy = qx - center[..., 0], qy - center[..., 1]
        theta = torch.atan2(vy, vx)
        travelled = torch.remainder((theta - theta0) * sign, 2.0 * math.pi)
        # points beyond the far end wrap around; send them to whichever end is closer
        past = travelled > sweep
        beyond = travelled - sweep
        before = 2.0 * math.pi - travelled
        travelled = torch.where(
            past, torch.where(beyond <= before, sweep, torch.zeros_like(sweep)), travelled
        )
        theta_c = theta0 + sign * travelled
        cos_c, sin_c = torch.cos(theta_c), torch.sin(theta_c)
        cx_arc = center[..., 0] + radius * cos_c
        cy_arc = center[..., 1] + radius * sin_c
        tx_arc = -sign * sin_c
        ty_arc = sign * cos_c
        s_arc = radius * travelled

        # --- select branch ---------------------------------------------------------------
        closest_x = torch.where(is_arc, cx_arc, cx_line)
        closest_y = torch.where(is_arc, cy_arc, cy_line)
        tan_x = torch.where(is_arc, tx_arc, t0[..., 0])
        tan_y = torch.where(is_arc, ty_arc, t0[..., 1])
        s_all = torch.where(is_arc, s_arc, s_line)

        off_x = qx - closest_x
        off_y = qy - closest_y
        dist2 = off_x * off_x + off_y * off_y
        d_all = tan_x * off_y - tan_y * off_x
        psi_all = wrap_to_pi(yaw_t[:, None] - torch.atan2(tan_y, tan_x))
        cost = dist2 + heading_weight * (1.0 - torch.cos(psi_all))
        cost = torch.where(valid, cost, torch.full_like(cost, float("inf")))
        if prev_seg_id is not None:
            prev_seg = torch.as_tensor(prev_seg_id, dtype=torch.long, device=self.device).reshape(-1)
            prev_arc = torch.zeros_like(px) if prev_s is None else self._as_batch(prev_s)
            tracked = prev_seg >= 0
            safe_seg = torch.where(tracked, prev_seg, torch.zeros_like(prev_seg))
            prev_route = self.seg_route_s[vidx, safe_seg] + prev_arc
            prev_cycle = self.seg_cycle[vidx, safe_seg]
            # every offset is relative to its OWN cycle, so the modulus is the previous
            # segment's cycle length and a candidate in a different cycle is never comparable
            length = self.seg_cycle_length[vidx, safe_seg].unsqueeze(1)
            candidate = self.seg_route_s[vidx] + s_all
            ahead = torch.remainder(candidate - prev_route[:, None], length)
            circular = torch.minimum(ahead, length - ahead)
            same_cycle = self.seg_cycle[vidx] == prev_cycle[:, None]
            reachable = same_cycle & (circular <= window_m)
            constrain = tracked[:, None] & self.variant_has_route[vidx].unsqueeze(1) & ~reachable
            windowed = torch.where(constrain, torch.full_like(cost, float("inf")), cost)
            # a window that excludes every segment (teleport, a push, a giant integration step)
            # must fall back to the free match rather than return garbage from an all-inf argmin
            stuck = torch.isinf(windowed).all(dim=1, keepdim=True)
            cost = torch.where(stuck, cost, windowed)
        best = torch.argmin(cost, dim=1)

        rows = torch.arange(px.numel(), device=self.device)
        gather = lambda t: t[rows, best]  # noqa: E731
        return LaneQuery(
            d=gather(d_all),
            psi=gather(psi_all),
            s=gather(s_all),
            seg_id=best,
            tangent_x=gather(tan_x),
            tangent_y=gather(tan_y),
            curvature=self.seg_curvature[vidx, best],
            closest_x=gather(closest_x),
            closest_y=gather(closest_y),
            dist=torch.sqrt(gather(dist2)),
        )

    def route_progress(self, variant_idx: Any, seg_id: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Cumulative arc length along the directed lane cycle, for logging and diagnostics.

        Only meaningful for pure-loop maps (see :attr:`LaneGraph.has_route`); on intersection
        maps the offsets are all zero and this reduces to ``s``.

        Args:
            variant_idx: ``(B,)`` variant index.
            seg_id: ``(B,)`` matched segment index.
            s: ``(B,)`` arc length within the segment.

        Returns:
            ``(B,)`` cumulative arc length in metres.
        """
        vidx = torch.as_tensor(variant_idx, dtype=torch.long, device=self.device).reshape(-1)
        return self.seg_route_s[vidx, seg_id] + s

    def curvature_at_lookahead(
        self, variant_idx: Any, seg_id: torch.Tensor, s: torch.Tensor, distance: float, max_hops: int = 4
    ) -> torch.Tensor:
        """Signed lane curvature a fixed distance ahead along the straightest continuation.

        This is the ``curvature at +0.3 m lookahead`` entry of the privileged critic observation
        in SPEC v2 S5.2.

        Args:
            variant_idx: ``(B,)`` variant index.
            seg_id: ``(B,)`` current segment index.
            s: ``(B,)`` arc length within the current segment.
            distance: Lookahead distance in metres.
            max_hops: Maximum number of segment transitions to follow.

        Returns:
            ``(B,)`` signed curvature in ``1/m``; positive turns left.

        Note:
            The hop loop is branchless on purpose. It used to break out of the loop as soon as
            ``overflow.any()`` was False, which cost one device-to-host sync per hop: the M-phase
            census measured 2.78 of them per control step, on the critical path of the
            privileged observation, in a rollout that was already spending 88% of its GPU idle.
            Once ``overflow`` is all-False both ``torch.where`` calls are the identity, so
            running the remaining hops unconditionally produces the identical tensor and lets the
            whole lane query stay in the queue.
        """
        vidx = torch.as_tensor(variant_idx, dtype=torch.long, device=self.device).reshape(-1)
        cur = seg_id.clone()
        remaining = s + float(distance)
        for _ in range(max_hops):
            seg_len = self.seg_length[vidx, cur]
            overflow = remaining > seg_len
            nxt = self.seg_primary[vidx, cur]
            remaining = torch.where(overflow, remaining - seg_len, remaining)
            cur = torch.where(overflow, nxt, cur)
        return self.seg_curvature[vidx, cur]

    def tile_index(self, variant_idx: Any, x: Any, y: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Grid cell of a batch of world points.

        Args:
            variant_idx: ``(B,)`` variant index.
            x: ``(B,)`` world x in metres.
            y: ``(B,)`` world y in metres.

        Returns:
            ``(row, col)`` integer tensors of shape ``(B,)``. Out-of-map points give ``-1``.
        """
        vidx = torch.as_tensor(variant_idx, dtype=torch.long, device=self.device).reshape(-1)
        px, py = self._as_batch(x), self._as_batch(y)
        origin = self.origin[vidx]
        pitch = self.pitch[vidx]
        col = torch.floor((px - origin[:, 0]) / pitch).to(torch.long)
        row_south = torch.floor((py - origin[:, 1]) / pitch).to(torch.long)
        row = self.grid_rows[vidx] - 1 - row_south
        inside = (col >= 0) & (col < self.grid_cols[vidx]) & (row >= 0) & (row < self.grid_rows[vidx])
        missing_row = -torch.ones_like(row)
        missing_col = -torch.ones_like(col)
        return torch.where(inside, row, missing_row), torch.where(inside, col, missing_col)

    def is_drivable(self, variant_idx: Any, x: Any, y: Any) -> torch.Tensor:
        """Whether a batch of world points lies on a drivable tile.

        This is the geometric test behind the SPEC v2 S5.5 off-drivable termination; the caller
        supplies the four robot test points.

        Args:
            variant_idx: ``(B,)`` variant index.
            x: ``(B,)`` world x in metres.
            y: ``(B,)`` world y in metres.

        Returns:
            ``(B,)`` boolean tensor; ``False`` outside the map.
        """
        vidx = torch.as_tensor(variant_idx, dtype=torch.long, device=self.device).reshape(-1)
        row, col = self.tile_index(vidx, x, y)
        inside = (row >= 0) & (col >= 0)
        safe_row = torch.clamp(row, min=0)
        safe_col = torch.clamp(col, min=0)
        return self.grid_drivable[vidx, safe_row, safe_col] & inside
