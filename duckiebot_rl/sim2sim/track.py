"""Build a MuJoCo track scene, and its lane graph, from a shared city map (SPEC v2 S3.3, S8.1).

Design decisions that are not obvious and are load bearing:

**Exactly one collidable surface.** Isaac has zero physics colliders in the city USD; physics is one
authored plane at ``/World/ground`` (S3.3, resolving critic item K). This module mirrors that: tiles,
walls and signs are emitted with ``contype="0" conaffinity="0"`` and the only collidable geom is the
ground plane from :func:`duckiebot_rl.sim2sim.mjcf.ground_geom`. Tile-edge contacts, phantom steps
and the "which tile did the wheel hit" class of divergence therefore cannot exist here either.

**The lane graph is derived from connectivity, not from tile-name orientation suffixes.** A road tile
knows which of its four neighbours are also road tiles; two opposite neighbours make a straight, two
perpendicular neighbours make a quarter arc about the shared corner, three or four make an
intersection. No ``curve_left/W``-style convention is ever parsed, so no convention can be
misinterpreted. When the map does declare a kind, :meth:`LaneGraph.check_against_map` reports any
disagreement instead of silently trusting either side.

**Lane geometry comes from [city], never from a literal here.** The yellow centre line runs down
the tile centreline and the right-hand lane centre is offset :attr:`CityParams.lane_offset_tiles`
to the right of it. That offset is resolved from ``duckiebot_rl.city.spec`` by
:func:`duckiebot_rl.sim2sim._resolve.resolve_city_params`, which is the same module that paints the
textures, so the lane graph and the pixels cannot disagree: at the nominal geometry the offset is
0.20 tile (117.0 mm), giving curve radii of 0.30 and 0.70 tile units (175.5 and 409.5 mm). The
0.22-tile figure that also appears in SPEC v2 S3.3 is the duckietown-world value and is *not* ours;
using it would place ``d = 0`` 11.7 mm from the centre of the painted lane and put a constant bias
into the headline S8.4 ``lane_rms_m`` and ``lane_max_m`` metrics. ``CityParams.validate()`` rejects
that combination outright, and ``tests/unit/test_sim2sim_city_params.py`` measures the lane centre
straight out of the rendered texture. Lateral error ``d`` is positive to the LEFT of the right-lane
centre, i.e. toward the yellow tape, which is the S2 sign convention.

**Textures.** For a vision evaluation the tiles must carry the *same* markings the ``[city]`` module
bakes into the Isaac city, otherwise C5 measures a texture difference rather than a physics or
renderer difference. :class:`CityTextureProvider` binds to ``duckiebot_rl.city`` and raises a precise
error naming what has to land if it cannot. :class:`FlatColorProvider` exists for physics-only work
(system identification, the kinematics tests, obstacle geometry checks) and is explicitly not valid
for reporting any C-number.
"""

from __future__ import annotations

import math
import struct
import warnings
import xml.etree.ElementTree as ET
import zlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import mjcf as _mjcf
from ._resolve import CityParams, SharedModuleUnavailable, resolve_city, resolve_city_params

__all__ = [
    "CityTextureProvider",
    "FlatColorProvider",
    "LaneGraph",
    "LaneQuery",
    "MapSpec",
    "ObstacleSpec",
    "TextureProvider",
    "TrackScene",
    "build_track",
    "kind_and_rotation",
    "load_map",
    "write_png",
]

# Tile kinds that carry lane markings and are part of the road network. Matching is by prefix so
# that "curve_left/W", "curve/NE" and plain "curve" all classify identically. The shared
# [city]-owned classifier is preferred when it is importable; these prefixes are the fallback for
# maps written before that module existed.
ROAD_PREFIXES = ("straight", "curve", "3way", "4way", "threeway", "fourway", "turn")
NON_ROAD_PREFIXES = ("grass", "floor", "asphalt", "empty", "blank", "wall", "void")

_DIRS: dict[str, tuple[int, int]] = {"N": (0, +1), "S": (0, -1), "E": (+1, 0), "W": (-1, 0)}
_OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}
# One counter-clockwise quarter turn acting on edge labels, the same table [city] uses in
# duckiebot_rl.city.maps. Used only when that module cannot be imported (it needs PyYAML, which the
# tools venv does not have); kind_and_rotation prefers the shared implementation.
_ROT_CCW = {"E": "N", "N": "W", "W": "S", "S": "E"}

_RGBA_ROAD = (0.10, 0.10, 0.11, 1.0)
_RGBA_GRASS = (0.37, 0.51, 0.16, 1.0)
_RGBA_WALL = (0.82, 0.82, 0.80, 1.0)
_RGBA_OBSTACLE = (0.90, 0.62, 0.10, 1.0)


def _rotate_edges(edges: Iterable[str], rot: int) -> frozenset[str]:
    """Rotate a set of edge labels ``rot`` quarter turns counter-clockwise."""
    out = frozenset(edges)
    for _ in range(rot % 4):
        out = frozenset(_ROT_CCW[e] for e in out)
    return out


def kind_and_rotation(kind: str, connections: Sequence[str]) -> tuple[str, int]:
    """Return the canonical tile kind and the quarter turns needed to match ``connections``.

    The shared tile generator paints exactly one orientation per kind (a straight connecting N-S,
    a curve connecting S-E), and every other orientation is that painting rotated. The Isaac city
    applies the rotation through ``duckiebot_rl.city.tiles.rotated_uv_corners``; MuJoCo has no
    per-face UV control on a box geom, so the rotation is baked into the texture instead. Both
    paths therefore use the same rotation index, derived here by the same rule.

    Args:
        kind: the declared tile kind, which may carry an orientation suffix or be unknown.
        connections: compass directions in which the tile has road neighbours.

    Returns:
        ``(canonical_kind, rot)`` with ``rot`` in ``0 .. 3`` counter-clockwise. Non-road tiles and
        tiles with fewer than two connections return ``("asphalt", 0)``.
    """
    edges = frozenset(connections)
    if len(edges) < 2:
        return "asphalt", 0
    maps = _try_import_city_maps()
    if maps is not None:
        try:
            return maps.kind_rot_for_edges(edges)
        except Exception:
            pass
    canonical = {
        "straight": frozenset({"N", "S"}),
        "curve": frozenset({"S", "E"}),
        "threeway": frozenset({"N", "S", "W"}),
        "fourway": frozenset({"N", "S", "E", "W"}),
    }
    token = kind.strip().lower().split("/")[0]
    order = sorted(canonical, key=lambda k: (k != token,))
    for candidate in order:
        for rot in range(4):
            if _rotate_edges(canonical[candidate], rot) == edges:
                return candidate, rot
    return "asphalt", 0


def _try_import_city_maps() -> Any:
    """Return ``duckiebot_rl.city.maps`` if it is importable, else None."""
    city = resolve_city()
    if city is None:
        return None
    try:
        import importlib

        return importlib.import_module("duckiebot_rl.city.maps")
    except Exception:
        return None


# ------------------------------------------------------------------------------------ map input
@dataclass
class ObstacleSpec:
    """One obstacle placed in the scene.

    Attributes:
        kind: free-form label used for the geom name and the failure histogram.
        pos: world ``(x, y)`` of the obstacle centre.
        radius: safety-circle radius used by the S5.5 geometric termination test.
        height: geom height in metres.
        mobile: True to emit a mocap body the environment repositions every control step
            (the MuJoCo equivalent of Isaac's ``kinematic_enabled=True`` movers).
    """

    kind: str
    pos: tuple[float, float]
    radius: float = 0.04
    height: float = 0.08
    mobile: bool = False


@dataclass
class MapSpec:
    """A MapFormat1-compatible map.

    ``tiles`` is indexed ``tiles[row][col]`` with row 0 the north-most row, which is how the YAML
    files are written. World coordinates put tile ``(row, col)`` at
    ``(col * tile_size, (nrows - 1 - row) * tile_size)``.

    Attributes:
        tiles: grid of tile-kind strings.
        tile_size: tile pitch in metres.
        objects: raw object entries carried through from the map file.
        name: map identifier used in file names and reports.
    """

    tiles: list[list[str]]
    tile_size: float = field(default_factory=lambda: resolve_city_params()[0].tile_pitch)
    objects: list[dict[str, Any]] = field(default_factory=list)
    name: str = "map"

    @property
    def nrows(self) -> int:
        """Number of tile rows."""
        return len(self.tiles)

    @property
    def ncols(self) -> int:
        """Number of tile columns."""
        return len(self.tiles[0]) if self.tiles else 0

    def kind(self, row: int, col: int) -> str:
        """Return the tile kind at ``(row, col)``, or ``""`` outside the grid."""
        if 0 <= row < self.nrows and 0 <= col < self.ncols:
            return str(self.tiles[row][col])
        return ""

    def is_road(self, row: int, col: int) -> bool:
        """Return True if ``(row, col)`` is a marked road tile."""
        return is_road_kind(self.kind(row, col))

    def center(self, row: int, col: int) -> tuple[float, float]:
        """Return the world ``(x, y)`` centre of tile ``(row, col)``."""
        return (col * self.tile_size, (self.nrows - 1 - row) * self.tile_size)

    def cell_of(self, x: float, y: float) -> tuple[int, int]:
        """Return the ``(row, col)`` whose tile contains world point ``(x, y)``."""
        col = math.floor(x / self.tile_size + 0.5)
        row = self.nrows - 1 - math.floor(y / self.tile_size + 0.5)
        return row, col

    def road_cells(self) -> list[tuple[int, int]]:
        """Return every road cell as ``(row, col)``."""
        return [(r, c) for r in range(self.nrows) for c in range(self.ncols) if self.is_road(r, c)]

    def extent(self) -> tuple[float, float, float, float]:
        """Return ``(xmin, xmax, ymin, ymax)`` of the tiled area including the outer half-tile."""
        half = 0.5 * self.tile_size
        return (
            -half,
            (self.ncols - 1) * self.tile_size + half,
            -half,
            (self.nrows - 1) * self.tile_size + half,
        )


def is_road_kind(kind: str) -> bool:
    """Classify a tile-kind string as road (marked, drivable) or not.

    Args:
        kind: the tile-kind string from the map, e.g. ``"curve_left/W"`` or ``"grass"``.

    Returns:
        True if the tile is part of the marked road network.
    """
    token = kind.strip().lower().split("/")[0]
    tiles = city_tiles()
    if tiles is not None and token in set(getattr(tiles, "TILE_KINDS", ())):
        return bool(tiles.is_drivable_kind(token))
    if not token or any(token.startswith(p) for p in NON_ROAD_PREFIXES):
        return False
    return any(token.startswith(p) for p in ROAD_PREFIXES)


def city_tiles() -> Any:
    """Return the ``[city]``-owned tiles module, or None when it cannot be imported.

    Imported lazily and defensively. ``duckiebot_rl.city`` pulls in pyyaml and torch through other
    submodules, neither of which the tools venv necessarily has yet, and a missing optional
    dependency over there must not take the MuJoCo harness down with it.

    Returns:
        The module, or None.
    """
    from ._resolve import _try_import

    return _try_import("duckiebot_rl.city.tiles")


def load_map(source: str | Path | dict[str, Any] | MapSpec) -> MapSpec:
    """Load a map from the shared city module, a YAML file, or a plain dict.

    Args:
        source: a :class:`MapSpec`, a mapping with ``tiles`` / ``tile_size`` / ``objects`` keys, or a
            path to a MapFormat1 YAML file.

    Returns:
        The parsed :class:`MapSpec`.

    Raises:
        SharedModuleUnavailable: if a file path is given and neither ``duckiebot_rl.city`` nor
            ``pyyaml`` is importable in this interpreter.
        ValueError: if the mapping has no usable ``tiles`` entry.
    """
    if isinstance(source, MapSpec):
        return source
    if isinstance(source, dict):
        raw = source
        name = str(raw.get("name", "map"))
    else:
        path = Path(source)
        from ._resolve import _try_import

        loader = None
        for holder in (_try_import("duckiebot_rl.city.maps"), resolve_city()):
            loader = getattr(holder, "load_map", None) if holder is not None else None
            if callable(loader):
                break
        if callable(loader):
            loaded = loader(str(path))
            return loaded if isinstance(loaded, MapSpec) else load_map(dict(loaded))
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - depends on the interpreter
            raise SharedModuleUnavailable(
                f"cannot read {path}: this interpreter has neither duckiebot_rl.city.load_map nor "
                f"pyyaml. Install pyyaml into the tools venv (SPEC v2 M0):\n"
                f"  d:/Personal/personal/mujoco_venv/Scripts/python.exe -m pip install pyyaml"
            ) from exc
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        name = path.stem
    tiles_raw = raw.get("tiles")
    if not tiles_raw:
        raise ValueError(f"map {name!r} has no 'tiles' entry")
    tiles = [[str(cell).strip() for cell in row] for row in tiles_raw]
    return MapSpec(
        tiles=tiles,
        tile_size=float(raw.get("tile_size", resolve_city_params()[0].tile_pitch)),
        objects=list(raw.get("objects", []) or []),
        name=name,
    )


MATCH_WINDOW_M: float = 0.35
"""Route half-width for continuity-constrained lane matching, in metres.

The same value and the same reasoning as ``duckiebot_rl.city.lane_graph.MATCH_WINDOW_M``: an
8-step cushion at full speed, small enough that the post-apex return lane of a hairpin stays
unmatchable until the robot has actually driven the arc. The two simulators must window
identically or the sim-to-sim reward comparison (C6) measures the matcher, not the policy.
"""


# ------------------------------------------------------------------------------------ lane graph
@dataclass(frozen=True)
class LaneQuery:
    """Result of projecting a robot pose onto the lane graph.

    Attributes:
        d: signed lateral error in metres, positive to the LEFT of the right-lane centre, i.e.
            toward the yellow tape (SPEC v2 S2 sign convention).
        psi: heading error in radians, positive when the robot points counterclockwise of the lane
            tangent.
        curvature: signed lane curvature (1/m) at the lookahead distance, positive for a left turn.
        segment: index of the closest directed lane segment.
        s: arc length along that segment, in metres.
        tangent: unit lane tangent at the projection point.
        point: the projection point on the lane centre.
    """

    d: float
    psi: float
    curvature: float
    segment: int
    s: float
    tangent: tuple[float, float]
    point: tuple[float, float]


@dataclass
class _Segment:
    """One directed lane segment: either a straight line or a circular arc."""

    is_arc: bool
    length: float
    cell: tuple[int, int]
    enter: str
    exit: str
    # line fields
    p0: tuple[float, float] = (0.0, 0.0)
    tangent: tuple[float, float] = (1.0, 0.0)
    # arc fields
    center: tuple[float, float] = (0.0, 0.0)
    radius: float = 0.0
    theta0: float = 0.0
    dtheta: float = 0.0

    @property
    def curvature(self) -> float:
        """Signed curvature: positive for a left (counterclockwise) turn."""
        if not self.is_arc:
            return 0.0
        return math.copysign(1.0 / self.radius, self.dtheta)

    def point_at(self, s: float) -> tuple[float, float]:
        """Return the lane-centre point at arc length ``s``."""
        s = min(max(s, 0.0), self.length)
        if not self.is_arc:
            return (self.p0[0] + self.tangent[0] * s, self.p0[1] + self.tangent[1] * s)
        theta = self.theta0 + math.copysign(s / self.radius, self.dtheta)
        return (
            self.center[0] + self.radius * math.cos(theta),
            self.center[1] + self.radius * math.sin(theta),
        )

    def tangent_at(self, s: float) -> tuple[float, float]:
        """Return the unit lane tangent at arc length ``s``."""
        if not self.is_arc:
            return self.tangent
        s = min(max(s, 0.0), self.length)
        theta = self.theta0 + math.copysign(s / self.radius, self.dtheta)
        sign = 1.0 if self.dtheta > 0.0 else -1.0
        return (-sign * math.sin(theta), sign * math.cos(theta))

    def end_point(self) -> tuple[float, float]:
        """Return the lane-centre point at the end of the segment."""
        return self.point_at(self.length)

    def end_tangent(self) -> tuple[float, float]:
        """Return the unit lane tangent at the end of the segment."""
        return self.tangent_at(self.length)


def _wrap(angle: float) -> float:
    """Wrap an angle to ``(-pi, pi]``."""
    return math.atan2(math.sin(angle), math.cos(angle))


class LaneGraph:
    """Directed right-hand lane centrelines derived from a map's road connectivity.

    Attributes:
        map: the source :class:`MapSpec`.
        city: the S3.3 dimensional constants used to place the lane centre.
        segments: the directed lane segments, in construction order.
    """

    def __init__(self, map_spec: MapSpec, city: CityParams | None = None) -> None:
        """Build the lane graph.

        Args:
            map_spec: the map to derive lanes from.
            city: dimensional constants; defaults to the SPEC v2 S3.3 values.
        """
        self.map = map_spec
        city = city if city is not None else resolve_city_params()[0]
        if abs(city.tile_pitch - map_spec.tile_size) > 1e-9:
            # The map declares its own pitch. Markings hold their fraction of the tile, so the
            # lane width and tape widths scale with it; lane_offset_tiles is dimensionless.
            city = city.rescaled(map_spec.tile_size)
        city.validate()
        self.city = city
        self.segments: list[_Segment] = []
        self._build()
        self._pack()
        self._successors = self._build_successors()

    # -- construction -------------------------------------------------------------------------
    def _connections(self, row: int, col: int) -> list[str]:
        """Return the compass directions in which ``(row, col)`` has a road neighbour."""
        out = []
        for name, (dcol, drow_world) in _DIRS.items():
            # world +y is decreasing row, so a north neighbour is row - 1
            nrow = row - drow_world
            ncol = col + dcol
            if self.map.is_road(nrow, ncol):
                out.append(name)
        return out

    def _edge_midpoint(self, center: tuple[float, float], direction: str) -> tuple[float, float]:
        """Return the midpoint of the tile edge facing ``direction``."""
        half = 0.5 * self.map.tile_size
        dx, dy = _DIRS[direction]
        return (center[0] + dx * half, center[1] + dy * half)

    def _build(self) -> None:
        """Populate :attr:`segments` from the road connectivity of every road tile."""
        pitch = self.map.tile_size
        offset = self.city.lane_offset_tiles * pitch
        for row, col in self.map.road_cells():
            center = self.map.center(row, col)
            dirs = self._connections(row, col)
            if len(dirs) < 2:
                continue
            for enter in dirs:
                for exit_ in dirs:
                    if enter == exit_:
                        continue
                    seg = self._make_segment((row, col), center, enter, exit_, offset, pitch)
                    if seg is not None:
                        self.segments.append(seg)

    def _make_segment(
        self,
        cell: tuple[int, int],
        center: tuple[float, float],
        enter: str,
        exit_: str,
        offset: float,
        pitch: float,
    ) -> _Segment | None:
        """Return the directed lane segment entering at ``enter`` and leaving at ``exit_``."""
        start = self._edge_midpoint(center, enter)
        end = self._edge_midpoint(center, exit_)
        if _OPPOSITE[enter] == exit_:
            heading = (end[0] - start[0], end[1] - start[1])
            norm = math.hypot(*heading)
            tangent = (heading[0] / norm, heading[1] / norm)
            right = (tangent[1], -tangent[0])
            p0 = (start[0] + right[0] * offset, start[1] + right[1] * offset)
            return _Segment(
                is_arc=False, length=norm, cell=cell, enter=enter, exit=exit_, p0=p0, tangent=tangent
            )
        # perpendicular: quarter arc about the corner shared by the two edges
        corner = (
            center[0] + 0.5 * pitch * (_DIRS[enter][0] + _DIRS[exit_][0]),
            center[1] + 0.5 * pitch * (_DIRS[enter][1] + _DIRS[exit_][1]),
        )
        theta0 = math.atan2(start[1] - corner[1], start[0] - corner[0])
        theta1 = math.atan2(end[1] - corner[1], end[0] - corner[0])
        dtheta = _wrap(theta1 - theta0)
        if abs(abs(dtheta) - math.pi / 2.0) > 1e-6:
            return None
        # A left (counterclockwise) turn keeps the corner on the left, so the right-hand lane sits
        # further from the corner; a right turn puts it closer.
        radius = 0.5 * pitch + (offset if dtheta > 0.0 else -offset)
        return _Segment(
            is_arc=True,
            length=abs(dtheta) * radius,
            cell=cell,
            enter=enter,
            exit=exit_,
            center=corner,
            radius=radius,
            theta0=theta0,
            dtheta=dtheta,
        )

    def _pack(self) -> None:
        """Pack the segments into numpy arrays so :meth:`query` is vectorized over all of them."""
        lines = [(i, s) for i, s in enumerate(self.segments) if not s.is_arc]
        arcs = [(i, s) for i, s in enumerate(self.segments) if s.is_arc]
        self._line_idx = np.array([i for i, _ in lines], dtype=np.int64)
        self._line_p0 = np.array([s.p0 for _, s in lines], dtype=np.float64).reshape(-1, 2)
        self._line_t = np.array([s.tangent for _, s in lines], dtype=np.float64).reshape(-1, 2)
        self._line_len = np.array([s.length for _, s in lines], dtype=np.float64)
        self._arc_idx = np.array([i for i, _ in arcs], dtype=np.int64)
        self._arc_c = np.array([s.center for _, s in arcs], dtype=np.float64).reshape(-1, 2)
        self._arc_r = np.array([s.radius for _, s in arcs], dtype=np.float64)
        self._arc_t0 = np.array([s.theta0 for _, s in arcs], dtype=np.float64)
        self._arc_dt = np.array([s.dtheta for _, s in arcs], dtype=np.float64)

    def _build_successors(self) -> list[int]:
        """Map each segment to the successor that continues straightest through the next tile."""
        starts = np.array([s.point_at(0.0) for s in self.segments], dtype=np.float64)
        start_tangents = np.array([s.tangent_at(0.0) for s in self.segments], dtype=np.float64)
        successors: list[int] = []
        for seg in self.segments:
            end = np.asarray(seg.end_point())
            end_tangent = np.asarray(seg.end_tangent())
            gap = np.linalg.norm(starts - end, axis=1)
            aligned = start_tangents @ end_tangent
            feasible = np.nonzero((gap < 1e-6) & (aligned > 0.0))[0]
            if feasible.size == 0:
                successors.append(-1)
                continue
            successors.append(int(feasible[np.argmax(aligned[feasible])]))
        return successors

    # -- queries ------------------------------------------------------------------------------
    def allowed_window(self, segment: int, s: float, window_m: float = MATCH_WINDOW_M) -> set[int]:
        """Return the segment indices reachable within ``window_m`` of route from ``(segment, s)``.

        The window always contains the current segment (side-slip changes ``d``, not ``s``), the
        predecessor chain within the window (the projection clamp can walk ``s`` slightly
        backwards through a corner), and the successor chain as far as ``window_m`` ahead.

        Args:
            segment: index of the currently matched segment.
            s: arc length along it, in metres.
            window_m: route half-width of the window, in metres.

        Returns:
            The allowed segment indices.
        """
        # Strict budget, mirroring the Isaac-side circular route window exactly: a segment is
        # allowed only if some part of it lies within window_m of route distance from (segment,
        # s). Adding the remaining current-segment length on top, the obvious-looking variant,
        # lets the post-hairpin flank into the window from mid-segment on small maps, which is
        # precisely the re-home the window exists to prevent (caught by the 3x2 ring test).
        allowed = {segment}
        ahead = self.segments[segment].length - s
        index = segment
        for _ in range(len(self.segments)):
            if ahead >= window_m:
                break
            nxt = self._successors[index]
            if nxt < 0 or nxt in allowed:
                break
            allowed.add(nxt)
            ahead += self.segments[nxt].length
            index = nxt
        behind = s
        index = segment
        predecessors = {succ: idx for idx, succ in enumerate(self._successors) if succ >= 0}
        for _ in range(len(self.segments)):
            if behind >= window_m:
                break
            prev = predecessors.get(index, -1)
            if prev < 0 or prev in allowed:
                break
            allowed.add(prev)
            behind += self.segments[prev].length
            index = prev
        return allowed

    def query(
        self,
        x: float,
        y: float,
        yaw: float,
        lookahead: float = 0.3,
        prev_match: tuple[int, float] | None = None,
        window_m: float = MATCH_WINDOW_M,
    ) -> LaneQuery:
        """Project a pose onto the nearest directed lane centre.

        Args:
            x: world x in metres.
            y: world y in metres.
            yaw: robot heading in radians.
            lookahead: distance ahead at which the reported curvature is evaluated, in metres.
            prev_match: ``(segment, s)`` of the previous step's match, or None for a free global
                search. When given, matching is constrained to :meth:`allowed_window` around it,
                mirroring the Isaac-side route window: without the constraint a robot crossing
                the centreline is re-homed onto the adjacent lane and ``|d|`` collapses exactly
                when it should grow, which is how hairpin cutting goes unmeasured.
            window_m: route half-width of the window, in metres.

        Returns:
            The :class:`LaneQuery`.

        Raises:
            RuntimeError: if the map produced no lane segments at all.
        """
        if not self.segments:
            raise RuntimeError(f"map {self.map.name!r} produced no lane segments")
        if prev_match is not None:
            allowed = self.allowed_window(prev_match[0], prev_match[1], window_m)
            result = self._query_free(x, y, yaw, lookahead, allowed)
            if result is not None:
                return result
        result = self._query_free(x, y, yaw, lookahead, None)
        if result is None:  # pragma: no cover - self.segments is non-empty here
            raise RuntimeError(f"map {self.map.name!r} produced no matchable segments")
        return result

    def _query_free(
        self, x: float, y: float, yaw: float, lookahead: float, allowed: set[int] | None
    ) -> LaneQuery | None:
        """Nearest-lane projection over ``allowed`` segments, or all of them when None.

        Args:
            x: world x in metres.
            y: world y in metres.
            yaw: robot heading in radians.
            lookahead: curvature lookahead in metres.
            allowed: segment indices to consider, or None for every segment.

        Returns:
            The query, or None when ``allowed`` excludes every segment.
        """
        p = np.array([x, y], dtype=np.float64)
        best_dist = math.inf
        best_index = -1
        best_s = 0.0
        line_keep = (
            slice(None) if allowed is None else np.isin(self._line_idx, np.fromiter(allowed, dtype=np.int64))
        )
        line_idx = self._line_idx[line_keep]
        if line_idx.size:
            rel = p[None, :] - self._line_p0[line_keep]
            t0 = self._line_t[line_keep]
            proj = np.clip((rel * t0).sum(axis=1), 0.0, self._line_len[line_keep])
            near = self._line_p0[line_keep] + t0 * proj[:, None]
            dist = np.linalg.norm(p[None, :] - near, axis=1)
            k = int(np.argmin(dist))
            if dist[k] < best_dist:
                best_dist, best_index, best_s = float(dist[k]), int(line_idx[k]), float(proj[k])
        arc_keep = (
            slice(None) if allowed is None else np.isin(self._arc_idx, np.fromiter(allowed, dtype=np.int64))
        )
        arc_idx = self._arc_idx[arc_keep]
        if arc_idx.size:
            rel = p[None, :] - self._arc_c[arc_keep]
            ang = np.arctan2(rel[:, 1], rel[:, 0])
            arc_dt = self._arc_dt[arc_keep]
            sign = np.sign(arc_dt)
            delta = np.mod((ang - self._arc_t0[arc_keep]) * sign, 2.0 * math.pi)
            span = np.abs(arc_dt)
            # Points beyond the arc end snap to whichever endpoint is angularly closer.
            beyond = delta > span
            delta = np.where(beyond & (delta - span > (2.0 * math.pi - delta)), 0.0, delta)
            delta = np.clip(delta, 0.0, span)
            theta = self._arc_t0[arc_keep] + sign * delta
            arc_r = self._arc_r[arc_keep]
            near = self._arc_c[arc_keep] + arc_r[:, None] * np.stack([np.cos(theta), np.sin(theta)], axis=1)
            dist = np.linalg.norm(p[None, :] - near, axis=1)
            k = int(np.argmin(dist))
            if dist[k] < best_dist:
                best_dist = float(dist[k])
                best_index = int(arc_idx[k])
                best_s = float(delta[k] * arc_r[k])
        if best_index < 0:
            return None
        seg = self.segments[best_index]
        point = seg.point_at(best_s)
        tangent = seg.tangent_at(best_s)
        left = (-tangent[1], tangent[0])
        d = (x - point[0]) * left[0] + (y - point[1]) * left[1]
        psi = _wrap(yaw - math.atan2(tangent[1], tangent[0]))
        return LaneQuery(
            d=d,
            psi=psi,
            curvature=self.curvature_ahead(best_index, best_s, lookahead),
            segment=best_index,
            s=best_s,
            tangent=tangent,
            point=point,
        )

    def curvature_ahead(self, segment: int, s: float, lookahead: float) -> float:
        """Return the signed lane curvature ``lookahead`` metres ahead of ``(segment, s)``.

        Args:
            segment: index of the current segment.
            s: arc length along it, in metres.
            lookahead: distance to travel forward along the lane, in metres.

        Returns:
            Signed curvature in 1/m, positive for a left turn.
        """
        index, remaining = segment, s + lookahead
        for _ in range(8):
            seg = self.segments[index]
            if remaining <= seg.length:
                return seg.curvature
            remaining -= seg.length
            nxt = self._successors[index]
            if nxt < 0:
                return seg.curvature
            index = nxt
        return self.segments[index].curvature

    def cycle_length(self, segment: int) -> float:
        """Return the closed-loop length in metres through ``segment``, or NaN if it is not a loop.

        Args:
            segment: index of a segment on the loop.

        Returns:
            Total arc length of the cycle, or ``float('nan')`` when the successor chain does not
            close (which is the normal case for maps containing intersections).
        """
        total, index = 0.0, segment
        for _ in range(len(self.segments) + 1):
            total += self.segments[index].length
            nxt = self._successors[index]
            if nxt < 0:
                return float("nan")
            if nxt == segment:
                return total
            index = nxt
        return float("nan")

    def is_drivable(self, x: float, y: float) -> bool:
        """Return True if world point ``(x, y)`` lies on a marked road tile."""
        row, col = self.map.cell_of(x, y)
        return self.map.is_road(row, col)

    def sample_spawn(
        self, rng: np.random.Generator, lateral: float = 0.06, heading_deg: float = 25.0
    ) -> tuple[float, float, float]:
        """Sample a SPEC v2 D16 spawn pose: any lane, lane direction, jittered.

        Args:
            rng: the numpy generator to draw from.
            lateral: half-width of the uniform lateral offset, in metres.
            heading_deg: half-width of the uniform heading offset, in degrees.

        Returns:
            ``(x, y, yaw)`` in world coordinates.
        """
        seg = self.segments[int(rng.integers(len(self.segments)))]
        s = float(rng.uniform(0.0, seg.length))
        px, py = seg.point_at(s)
        tx, ty = seg.tangent_at(s)
        offset = float(rng.uniform(-lateral, lateral))
        yaw = math.atan2(ty, tx) + math.radians(float(rng.uniform(-heading_deg, heading_deg)))
        return (px - ty * offset, py + tx * offset, yaw)

    def check_against_map(self) -> list[str]:
        """Compare the derived connectivity with any declared tile kinds.

        Returns:
            A list of human-readable disagreements; empty when the map is self-consistent.
        """
        issues: list[str] = []
        tiles = city_tiles()
        shared = getattr(tiles, "KIND_CONNECTIONS", None) if tiles is not None else None
        expected = {"straight": 2, "curve": 2, "turn": 2, "3way": 3, "4way": 4, "threeway": 3, "fourway": 4}
        if shared:
            expected = {k: len(v) for k, v in shared.items() if v}
        for row, col in self.map.road_cells():
            kind = self.map.kind(row, col).split("/")[0].lower()
            count = len(self._connections(row, col))
            want = expected.get(kind)
            if want is not None and count != want:
                issues.append(
                    f"tile ({row},{col}) declares {kind!r} ({want} connections) but has {count} "
                    f"road neighbours; the lane graph follows the neighbours"
                )
        return issues


# --------------------------------------------------------------------------------- tile textures
def write_png(path: str | Path, rgb: np.ndarray) -> Path:
    """Write an 8-bit RGB PNG using only the standard library.

    Pillow is not installed in the tools venv, and the track builder must be able to emit textures
    without it. Only the city *generator* needs Pillow; writing the result does not.

    Args:
        path: destination file.
        rgb: ``(H, W, 3)`` array; float input in ``[0, 1]`` is scaled, uint8 is used as is.

    Returns:
        The destination path.

    Raises:
        ValueError: if the array is not ``(H, W, 3)``.
    """
    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) array, got shape {array.shape}")
    if array.dtype != np.uint8:
        array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    height, width, _ = array.shape
    raw = b"".join(b"\x00" + array[row].tobytes() for row in range(height))

    def chunk(tag: bytes, payload: bytes) -> bytes:
        body = tag + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))

    blob = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(blob)
    return destination


class TextureProvider:
    """Base class for tile-appearance providers."""

    valid_for_vision: bool = False

    def texture(self, kind: str, connections: Sequence[str], size: int) -> np.ndarray | None:
        """Return an ``(size, size, 3)`` texture for a tile, or None to use a flat colour.

        Args:
            kind: the declared tile kind.
            connections: compass directions in which the tile has road neighbours.
            size: texture edge length in pixels.

        Returns:
            The texture array, or None.
        """
        raise NotImplementedError


class FlatColorProvider(TextureProvider):
    """Untextured tiles: dark road, green surroundings. Physics-only work only.

    A C5 or C6 number produced with this provider would be measuring a missing-markings ablation,
    not sim-to-sim transfer, so :meth:`TrackScene.assert_vision_ready` refuses it.
    """

    valid_for_vision = False

    def texture(self, kind: str, connections: Sequence[str], size: int) -> np.ndarray | None:
        """Return None, so the caller falls back to a flat rgba.

        Args:
            kind: ignored.
            connections: ignored.
            size: ignored.

        Returns:
            None.
        """
        del kind, connections, size
        return None


class CityTextureProvider(TextureProvider):
    """Paint the tiles with the ``[city]`` generator, so both simulators show the same markings.

    This is what makes a C5 number mean anything: the Isaac city and the MuJoCo track come from one
    texture generator and one ``TileSpec``, so the C1-versus-C5 delta measures physics and renderer
    differences rather than a difference in where somebody drew the yellow line.

    Attributes:
        spec: the marking-geometry spec (one of the 16 S3.3 buckets).
        style: the colour and wear style.
        resolution: texture edge length in pixels, or None to use the caller's request.
        seed: generator seed for the procedural wear and noise.
    """

    valid_for_vision = True

    def __init__(
        self,
        spec: Any = None,
        style: Any = None,
        resolution: int | None = None,
        bucket: int | None = None,
        seed: int = 0,
    ) -> None:
        """Bind to the shared tile generator.

        Args:
            spec: a ``duckiebot_rl.city.TileSpec``; None uses the nominal one.
            style: a ``duckiebot_rl.city.TileStyle``; None uses the generator default.
            resolution: texture edge length in pixels; None uses the caller's request.
            bucket: index into the shared marking-geometry buckets, used when ``spec`` is None and
                the city module exposes ``geometry_buckets``.
            seed: generator seed.

        Raises:
            SharedModuleUnavailable: if the shared tile generator cannot be imported.
        """
        tiles = city_tiles()
        if tiles is None or not hasattr(tiles, "render_tile"):
            raise SharedModuleUnavailable(
                "duckiebot_rl.city.tiles.render_tile is not importable in this interpreter. It is "
                "owned by [city]. Until it is available the MuJoCo track can only be built with "
                "FlatColorProvider, which paints no lane markings and is not valid for reporting "
                "C5 or C6. If the module exists but fails to import, check that pyyaml and Pillow "
                "are installed in the tools venv (SPEC v2 M0)."
            )
        self._tiles = tiles
        self.resolution = resolution
        self.style = style
        self.seed = seed
        if spec is None and bucket is not None and hasattr(tiles, "geometry_buckets"):
            buckets = tiles.geometry_buckets()
            spec = buckets[bucket % len(buckets)]
        self.spec = spec if spec is not None else getattr(tiles, "NOMINAL_TILE_SPEC", None)

    def texture(self, kind: str, connections: Sequence[str], size: int) -> np.ndarray | None:
        """Render a tile texture through the shared city generator.

        An unrecognised kind string is resolved by *connection count*, which is the same
        connectivity the lane graph is derived from, so a tile the generator has never heard of
        still gets the right markings.

        Args:
            kind: the declared tile kind.
            connections: compass directions in which the tile has road neighbours.
            size: texture edge length in pixels, used when no resolution was configured.

        Returns:
            The ``(H, W, 3)`` texture.
        """
        known = set(getattr(self._tiles, "TILE_KINDS", ()))
        token, rot = kind_and_rotation(kind, connections)
        if token not in known:
            token = "asphalt"
            rot = 0
        kwargs: dict[str, Any] = {"seed": self.seed}
        if self.spec is not None:
            kwargs["spec"] = self.spec
        if self.style is not None:
            kwargs["style"] = self.style
        kwargs["res"] = self.resolution if self.resolution is not None else size
        array = np.asarray(self._tiles.render_tile(token, **kwargs))
        # The generator paints one canonical orientation per kind; every other orientation is that
        # painting rotated. Row 0 of the texture is north and column 0 is west, both in the
        # generator and on a MuJoCo box +z face (measured, and locked by
        # tests/unit/test_mj_tile_orientation.py), so a counter-clockwise world rotation is exactly
        # np.rot90. Without this an east-west straight showed a yellow line running north-south,
        # across the road instead of along it, and every curve showed the canonical S-E arc
        # whatever way it actually bent.
        return np.ascontiguousarray(np.rot90(array, rot)) if rot % 4 else array


# --------------------------------------------------------------------------------- scene builder
@dataclass
class TrackScene:
    """A generated MuJoCo track: the XML, its lane graph and the assets it references.

    Attributes:
        xml: the MJCF document.
        map: the source map.
        lane: the derived lane graph.
        obstacles: the obstacles placed in the scene, in mocap-body order for the mobile ones.
        asset_dir: directory the XML's ``texturedir`` points at.
        texture_provider: the provider used, recorded so reports can state whether the markings came
            from the shared city generator.
    """

    xml: str
    map: MapSpec
    lane: LaneGraph
    obstacles: list[ObstacleSpec]
    asset_dir: Path
    texture_provider: TextureProvider

    @property
    def mobile_obstacles(self) -> list[ObstacleSpec]:
        """The obstacles emitted as mocap bodies, in mocap index order."""
        return [o for o in self.obstacles if o.mobile]

    def assert_vision_ready(self) -> None:
        """Raise unless the tiles carry the shared city markings.

        Raises:
            RuntimeError: if the scene was built with a provider that is not valid for vision.
        """
        if not self.texture_provider.valid_for_vision:
            raise RuntimeError(
                "this track was built with "
                f"{type(self.texture_provider).__name__}, which paints no lane markings. Any C5 or "
                "C6 number from it would measure missing textures, not sim-to-sim transfer."
            )

    def write(self, path: str | Path) -> Path:
        """Write the MJCF next to its assets.

        Args:
            path: destination ``.xml`` path.

        Returns:
            The destination path.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.xml, encoding="utf-8")
        return destination


def _tile_geoms(
    map_spec: MapSpec,
    lane: LaneGraph,
    provider: TextureProvider,
    asset_dir: Path,
    texture_size: int,
    thickness: float,
) -> tuple[list[ET.Element], list[ET.Element]]:
    """Emit one visual-only box per tile plus the textures and materials they reference."""
    geoms: list[ET.Element] = []
    assets: list[ET.Element] = []
    materials: dict[str, str] = {}
    written_tiles: dict[str, str] = {}
    half = 0.5 * map_spec.tile_size
    for row in range(map_spec.nrows):
        for col in range(map_spec.ncols):
            kind = map_spec.kind(row, col)
            road = map_spec.is_road(row, col)
            connections = lane._connections(row, col) if road else []
            key = f"{kind or 'empty'}|{''.join(sorted(connections))}"
            material = materials.get(key)
            if material is None and road:
                array = provider.texture(kind, connections, texture_size)
                if array is not None:
                    material = f"tile_mat_{len(materials)}"
                    filename = f"tile_{len(materials)}.png"
                    write_png(asset_dir / filename, array)
                    written_tiles[filename] = f"{kind or 'empty'} ({''.join(sorted(connections))})"
                    assets.append(
                        ET.Element(
                            "texture",
                            {"name": f"tile_tex_{len(materials)}", "type": "2d", "file": filename},
                        )
                    )
                    assets.append(
                        ET.Element(
                            "material",
                            {
                                "name": material,
                                "texture": f"tile_tex_{len(materials)}",
                                "texrepeat": "1 1",
                                "texuniform": "false",
                                "specular": "0.05",
                                "shininess": "0.05",
                            },
                        )
                    )
                    materials[key] = material
            cx, cy = map_spec.center(row, col)
            attrs = {
                "name": f"tile_{row}_{col}",
                "type": "box",
                "size": f"{half:.9g} {half:.9g} {0.5 * thickness:.9g}",
                "pos": f"{cx:.9g} {cy:.9g} {0.5 * thickness:.9g}",
                "contype": "0",
                "conaffinity": "0",
                "density": "0",
                "group": "2",
            }
            if material is not None:
                attrs["material"] = material
            else:
                attrs["rgba"] = " ".join(f"{v:.4g}" for v in (_RGBA_ROAD if road else _RGBA_GRASS))
            geoms.append(ET.Element("geom", attrs))
    _write_tile_manifest(asset_dir, provider, written_tiles)
    return geoms, assets


def _write_tile_manifest(asset_dir: Path, provider: TextureProvider, written_tiles: dict[str, str]) -> None:
    """Record that these tile PNGs were generated here, for the S3.4 clean-room gate.

    The gate refuses any image it cannot trace to a generator, which is the whole point: an
    unmanifested PNG cannot be shown to be ours rather than copied from Duckietown, whose assets
    carry no redistribution grant. The city generator writes its own MANIFEST.yaml; the
    sim-to-sim scene writes tiles too, so it owes the same proof. Without this the gate fails
    the moment anyone builds a MuJoCo scene, which is exactly when they need it to pass.

    Args:
        asset_dir: directory the tiles were written into.
        provider: the texture provider used, named in the manifest as the proximate generator.
        written_tiles: mapping of tile filename to the tile kind and connections it renders.
    """
    if not written_tiles:
        return
    lines = [
        "generator: duckiebot_rl.sim2sim.track.build_track",
        f"package: {type(provider).__module__}",
        f"provider: {type(provider).__name__}",
        "entries:",
    ]
    for filename, description in sorted(written_tiles.items()):
        lines.append(f"  {filename}:")
        lines.append("    generator: duckiebot_rl.sim2sim.track._tile_geoms")
        lines.append(f"    tile: {description}")
    text = "\n".join(lines) + "\n"
    (asset_dir / "MANIFEST.yaml").write_text(text, encoding="utf-8")


def _wall_geoms(map_spec: MapSpec, city: CityParams) -> list[ET.Element]:
    """Emit visual-only perimeter walls (S3.3: walls never collide; containment is the MDP's job)."""
    xmin, xmax, ymin, ymax = map_spec.extent()
    height, thickness = city.wall_height, city.wall_thickness
    spans = (
        ("wall_south", (0.5 * (xmin + xmax), ymin), (0.5 * (xmax - xmin), 0.5 * thickness)),
        ("wall_north", (0.5 * (xmin + xmax), ymax), (0.5 * (xmax - xmin), 0.5 * thickness)),
        ("wall_west", (xmin, 0.5 * (ymin + ymax)), (0.5 * thickness, 0.5 * (ymax - ymin))),
        ("wall_east", (xmax, 0.5 * (ymin + ymax)), (0.5 * thickness, 0.5 * (ymax - ymin))),
    )
    out = []
    for name, (cx, cy), (hx, hy) in spans:
        out.append(
            ET.Element(
                "geom",
                {
                    "name": name,
                    "type": "box",
                    "size": f"{hx:.9g} {hy:.9g} {0.5 * height:.9g}",
                    "pos": f"{cx:.9g} {cy:.9g} {0.5 * height:.9g}",
                    "contype": "0",
                    "conaffinity": "0",
                    "density": "0",
                    "group": "2",
                    "rgba": " ".join(f"{v:.4g}" for v in _RGBA_WALL),
                },
            )
        )
    return out


def _obstacle_elements(obstacles: Sequence[ObstacleSpec]) -> list[ET.Element]:
    """Emit obstacles: static geoms for parked ones, mocap bodies for movers.

    A MuJoCo mocap body is exactly the counterpart of Isaac's ``kinematic_enabled=True`` rigid body:
    it collides, it is driven by pose writes at the control rate, and contacts never move it.
    """
    out: list[ET.Element] = []
    for index, obstacle in enumerate(obstacles):
        geom_attrs = {
            "name": f"obstacle_{index}_{obstacle.kind}",
            "type": "cylinder",
            "size": f"{obstacle.radius:.9g} {0.5 * obstacle.height:.9g}",
            "condim": "3",
            "friction": "0.7 0.01 0.001",
            "rgba": " ".join(f"{v:.4g}" for v in _RGBA_OBSTACLE),
        }
        if obstacle.mobile:
            body = ET.Element(
                "body",
                {
                    "name": f"obstacle_body_{index}",
                    "mocap": "true",
                    "pos": f"{obstacle.pos[0]:.9g} {obstacle.pos[1]:.9g} {0.5 * obstacle.height:.9g}",
                },
            )
            body.append(ET.Element("geom", geom_attrs))
            out.append(body)
        else:
            geom_attrs["pos"] = f"{obstacle.pos[0]:.9g} {obstacle.pos[1]:.9g} {0.5 * obstacle.height:.9g}"
            out.append(ET.Element("geom", geom_attrs))
    return out


def _objects_to_obstacles(map_spec: MapSpec) -> list[ObstacleSpec]:
    """Convert the map's ``objects`` entries into :class:`ObstacleSpec` records."""
    out: list[ObstacleSpec] = []
    for entry in map_spec.objects:
        kind = str(entry.get("kind", "object"))
        pos = entry.get("pos", (0.0, 0.0))
        scale = float(entry.get("height", 0.08))
        world = (float(pos[0]) * map_spec.tile_size, float(pos[1]) * map_spec.tile_size)
        out.append(
            ObstacleSpec(
                kind=kind,
                pos=world,
                radius=float(entry.get("radius", 0.04)),
                height=scale,
                mobile=bool(entry.get("mobile", False)),
            )
        )
    return out


def build_track(
    source: str | Path | dict[str, Any] | MapSpec,
    cfg: _mjcf.MjcfCfg | None = None,
    asset_dir: str | Path = ".",
    texture_provider: TextureProvider | None = None,
    obstacles: Iterable[ObstacleSpec] | None = None,
    walls: bool = True,
    spawn: tuple[float, float, float] | None = None,
    texture_size: int = 512,
    city: CityParams | None = None,
) -> TrackScene:
    """Build a complete MuJoCo track scene from a map.

    Args:
        source: the map, as a :class:`MapSpec`, a dict or a path to a MapFormat1 YAML file.
        cfg: MJCF configuration; a fresh :meth:`MjcfCfg.from_shared` is used when None.
        asset_dir: directory to write tile textures into and to point ``texturedir`` at.
        texture_provider: appearance provider; defaults to :class:`CityTextureProvider` when the
            ``[city]`` module is importable and :class:`FlatColorProvider` otherwise (with a
            warning, because the fallback is not valid for vision evaluation).
        obstacles: obstacles to add on top of the map's own ``objects`` entries.
        walls: emit visual-only perimeter walls.
        spawn: initial ``(x, y, yaw)``; defaults to the start of the first lane segment.
        texture_size: tile texture edge length in pixels.
        city: dimensional constants; defaults to the SPEC v2 S3.3 values.

    Returns:
        The assembled :class:`TrackScene`.
    """
    map_spec = load_map(source)
    directory = Path(asset_dir)
    directory.mkdir(parents=True, exist_ok=True)
    cfg = cfg if cfg is not None else _mjcf.MjcfCfg.from_shared()
    # <compiler texturedir> must be ABSOLUTE, because this scene is compiled through two
    # different MuJoCo entry points that resolve a relative value against different bases:
    # `from_xml_path` resolves it against the directory of the XML (so a value of `directory`
    # looked under `<directory>/<directory>/tile_N.png`, the failure that blocked the whole S8
    # harness with "Error opening file" while the tiles sat there valid), and
    # `from_xml_string` has no file context at all, so it resolves against the process CWD
    # (where "." finds nothing). An absolute path is used verbatim by both.
    cfg.texturedir = str(directory.resolve())

    if texture_provider is None:
        try:
            texture_provider = CityTextureProvider()
        except SharedModuleUnavailable as exc:
            warnings.warn(
                f"{exc} Falling back to FlatColorProvider; physics is unaffected but no lane "
                "markings will be rendered.",
                RuntimeWarning,
                stacklevel=2,
            )
            texture_provider = FlatColorProvider()

    # The lane graph is derived from the SAME TileSpec the provider paints with, so a randomized
    # (S7.2 axis V9) or rescaled marking geometry moves the pixels and the reward reference line
    # together. Falling back to the nominal spec here would re-introduce exactly the disagreement
    # this coupling exists to remove.
    if city is None:
        city, _city_source = resolve_city_params(getattr(texture_provider, "spec", None))
    lane = LaneGraph(map_spec, city)
    city = lane.city

    all_obstacles = _objects_to_obstacles(map_spec) + list(obstacles or ())
    tiles, tile_assets = _tile_geoms(
        map_spec, lane, texture_provider, directory, texture_size, city.tile_visual_thickness
    )

    xmin, xmax, ymin, ymax = map_spec.extent()
    ground_half = 0.5 * max(xmax - xmin, ymax - ymin) + map_spec.tile_size
    children: list[ET.Element] = [_mjcf.ground_geom(cfg, ground_half)]
    children.extend(tiles)
    if walls:
        children.extend(_wall_geoms(map_spec, city))
    children.extend(_obstacle_elements(all_obstacles))

    if spawn is None:
        first = lane.segments[0]
        px, py = first.point_at(0.25 * first.length)
        tx, ty = first.tangent_at(0.25 * first.length)
        spawn = (px, py, math.atan2(ty, tx))
    children.append(_mjcf.build_robot_body(cfg, pos=(spawn[0], spawn[1], 0.0), yaw=spawn[2]))

    xml = _mjcf.build_scene_xml(cfg, world_children=children, assets=tile_assets)
    return TrackScene(
        xml=xml,
        map=map_spec,
        lane=lane,
        obstacles=all_obstacles,
        asset_dir=directory,
        texture_provider=texture_provider,
    )


LOOP_5X5: dict[str, Any] = {
    "name": "loop_5x5",
    "tile_size": resolve_city_params()[0].tile_pitch,
    "tiles": [
        ["curve", "straight", "straight", "straight", "curve"],
        ["straight", "grass", "grass", "grass", "straight"],
        ["straight", "grass", "grass", "grass", "straight"],
        ["straight", "grass", "grass", "grass", "straight"],
        ["curve", "straight", "straight", "straight", "curve"],
    ],
    "objects": [],
}
"""A 5x5 single-loop map used by the unit tests and by the sysid smoke run.

It is a placeholder for the ``[city]``-owned ``maps/*.yaml`` set, not a substitute: the real
evaluation runs read the 4 frozen held-out maps through :func:`load_map`.
"""
