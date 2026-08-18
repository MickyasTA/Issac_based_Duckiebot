"""Map format, loader, validator, built-in layouts and a random closed-loop generator.

Map format
----------
The on-disk format is a MapFormat1-shaped YAML document: a ``tiles`` grid of tile strings, a
``tile_size`` in metres, and an ``objects`` list. It is *shaped* like MapFormat1 -- same keys,
same tile-kind names, same ``kind/orientation`` string syntax -- but the orientation letter has
an explicitly documented meaning here rather than an implicit one, and the file is authored by
this package rather than copied from anywhere.

.. code-block:: yaml

   name: loop_small
   tile_size: 0.585
   tiles:
     - [grass, grass,         grass,       grass,        grass]
     - [grass, curve_left/N,  straight/E,  curve_left/W, grass]
     - [grass, straight/N,    asphalt,     straight/N,   grass]
     - [grass, curve_left/E,  straight/E,  curve_left/S, grass]
     - [grass, grass,         grass,       grass,        grass]
   objects: []
   meta: {walls: true, geometry_bucket: 0}

Grid and world conventions
--------------------------
* ``tiles[row][col]``. Row ``0`` is the **northernmost** row, so the YAML reads like a printed
  map. The one place that flip is encoded is :meth:`CityMap.tile_center_xy`.
* World position of tile ``(row, col)`` is ``origin + ((col + 0.5) * pitch,
  (n_rows - 1 - row + 0.5) * pitch)``. By default ``origin`` centres the map on ``(0, 0)`` so
  every city fits inside the per-env AABB of SPEC v2 S5.1 (half extent 3.6 m).
* The orientation letter is the tile's rotation, counter-clockwise about ``+z``:
  ``N`` = 0, ``E`` = 1, ``S`` = 2, ``W`` = 3 quarter turns. The canonical (``N``) connectivity of
  every kind is given by :data:`~.tiles.KIND_CONNECTIONS`. With that convention ``straight/N`` is
  a north-south road and ``straight/E`` an east-west one.
* ``curve_right`` and ``3way_right`` are accepted on input and normalised to ``curve_left`` and
  ``3way_left`` with a shifted rotation, because they are rotations of the same shape.
  ``floor`` is accepted as an alias of ``empty``.

Validity
--------
A map is valid when the grid is rectangular, every tile string parses, every **open** edge of
every drivable tile faces a tile whose facing edge is also open, and the whole layout fits inside
the per-env AABB. Closed edges may face anything, which is what lets a loop run alongside itself.
The Duckietown topological guidance that intersections should not be adjacent to curves or other
intersections is reported by :meth:`CityMap.topology_warnings` rather than enforced.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
import yaml

from .spec import NOMINAL_TILE_SPEC
from .tiles import DRIVABLE_KINDS, EDGE_BASIS, KIND_CONNECTIONS, TILE_KINDS

__all__ = [
    "BUILTIN_MAP_NAMES",
    "DIFFICULTY_NAMES",
    "DIFFICULTY_PROFILES",
    "ENV_HALF_EXTENT_M",
    "CityMap",
    "DifficultyProfile",
    "LoopComplexity",
    "MapObject",
    "MapValidationError",
    "Tile",
    "builtin_map",
    "difficulty_profile",
    "eval_maps",
    "format_tile",
    "kind_rot_for_edges",
    "load_map",
    "loop_complexity",
    "map_from_cycle",
    "parse_tile",
    "random_loop_map",
    "rotate_edges",
    "save_map",
    "variant_maps",
]

#: Half extent of the per-env AABB every city must fit inside (SPEC v2 S5.1, env_spacing 8.0).
ENV_HALF_EXTENT_M: Final[float] = 3.6

#: Rotation index -> orientation letter.
ROT_TO_ORIENT: Final[tuple[str, str, str, str]] = ("N", "E", "S", "W")

#: Orientation letter -> rotation index (quarter turns counter-clockwise).
ORIENT_TO_ROT: Final[dict[str, int]] = {letter: i for i, letter in enumerate(ROT_TO_ORIENT)}

#: One counter-clockwise quarter turn, acting on edge labels.
_ROT_CCW: Final[dict[str, str]] = {"E": "N", "N": "W", "W": "S", "S": "E"}

#: Canonical kind -> the MapFormat1 name emitted when serialising.
_KIND_TO_FORMAT1: Final[dict[str, str]] = {
    "straight": "straight",
    "curve": "curve_left",
    "threeway": "3way_left",
    "fourway": "4way",
    "asphalt": "asphalt",
    "grass": "grass",
    "empty": "empty",
}

#: MapFormat1 name -> (canonical kind, extra rotation added to the orientation letter).
_FORMAT1_TO_KIND: Final[dict[str, tuple[str, int]]] = {
    "straight": ("straight", 0),
    "curve_left": ("curve", 0),
    "curve_right": ("curve", 3),
    "3way_left": ("threeway", 0),
    "3way_right": ("threeway", 2),
    "4way": ("fourway", 0),
    "asphalt": ("asphalt", 0),
    "grass": ("grass", 0),
    "empty": ("empty", 0),
    "floor": ("empty", 0),
}

#: Grid step of each edge, as ``(d_row, d_col)``. Row 0 is north, so N is ``row - 1``.
_EDGE_STEP: Final[dict[str, tuple[int, int]]] = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1)}

#: The edge of the neighbouring tile that faces a given edge.
_OPPOSITE: Final[dict[str, str]] = {"N": "S", "S": "N", "E": "W", "W": "E"}


class MapValidationError(ValueError):
    """Raised when a map cannot describe a physically buildable city."""


def rotate_edges(edges: Iterable[str], rot: int) -> frozenset[str]:
    """Rotate a set of edge labels counter-clockwise.

    Args:
        edges: Edge labels drawn from ``{"N", "E", "S", "W"}``.
        rot: Quarter turns counter-clockwise (any integer).

    Returns:
        The rotated edge set.
    """
    out = frozenset(edges)
    for _ in range(rot % 4):
        out = frozenset(_ROT_CCW[e] for e in out)
    return out


def kind_rot_for_edges(edges: Iterable[str]) -> tuple[str, int]:
    """Find the tile kind and rotation whose open edges equal ``edges``.

    Args:
        edges: The required set of open edges.

    Returns:
        ``(kind, rot)`` where ``rot`` is quarter turns counter-clockwise.

    Raises:
        MapValidationError: If no drivable tile kind has that connectivity.
    """
    want = frozenset(edges)
    for kind in DRIVABLE_KINDS:
        for rot in range(4):
            if rotate_edges(KIND_CONNECTIONS[kind], rot) == want:
                return kind, rot
    raise MapValidationError(f"no tile kind connects exactly the edges {sorted(want)}")


@dataclass(frozen=True)
class Tile:
    """One grid cell: a canonical kind plus a rotation.

    Attributes:
        kind: One of :data:`~.tiles.TILE_KINDS`.
        rot: Quarter turns counter-clockwise about ``+z``, in ``0 .. 3``.
    """

    kind: str
    rot: int = 0

    def __post_init__(self) -> None:
        """Validate the kind and normalise the rotation.

        Raises:
            MapValidationError: If ``kind`` is unknown.
        """
        if self.kind not in TILE_KINDS:
            raise MapValidationError(f"unknown tile kind {self.kind!r}")
        object.__setattr__(self, "rot", int(self.rot) % 4)

    @property
    def drivable(self) -> bool:
        """Whether a robot may drive on this tile."""
        return self.kind in DRIVABLE_KINDS

    @property
    def open_edges(self) -> frozenset[str]:
        """The tile edges this tile connects roads through, after rotation."""
        return rotate_edges(KIND_CONNECTIONS[self.kind], self.rot)


def parse_tile(text: str) -> Tile:
    """Parse a MapFormat1 tile string such as ``"curve_right/W"``.

    Args:
        text: Tile string, optionally with an ``/N|/E|/S|/W`` orientation suffix.

    Returns:
        The canonical :class:`Tile`.

    Raises:
        MapValidationError: If the kind or orientation letter is unknown.
    """
    raw = str(text).strip()
    if "/" in raw:
        name, orient = raw.split("/", 1)
        orient = orient.strip().upper()
        if orient not in ORIENT_TO_ROT:
            raise MapValidationError(f"unknown orientation {orient!r} in tile {raw!r}")
        base_rot = ORIENT_TO_ROT[orient]
    else:
        name, base_rot = raw, 0
    name = name.strip()
    if name not in _FORMAT1_TO_KIND:
        raise MapValidationError(f"unknown tile kind {name!r} in tile {raw!r}")
    kind, extra = _FORMAT1_TO_KIND[name]
    return Tile(kind, base_rot + extra)


def format_tile(tile: Tile) -> str:
    """Serialise a :class:`Tile` back to a MapFormat1 string.

    Non-drivable kinds carry no orientation suffix because their textures are isotropic.

    Args:
        tile: The tile to serialise.

    Returns:
        A string that :func:`parse_tile` maps back to the same tile.
    """
    name = _KIND_TO_FORMAT1[tile.kind]
    if not tile.drivable:
        return name
    if tile.kind == "fourway":
        return name
    return f"{name}/{ROT_TO_ORIENT[tile.rot]}"


@dataclass
class MapObject:
    """A static or dynamic prop placed in the city.

    Attributes:
        kind: Free-form object class, e.g. ``"duckie"``, ``"cone"``, ``"duckiebot"``,
            ``"sign"``, ``"barrier"``.
        x: World x in metres, in the map's own frame.
        y: World y in metres.
        yaw_deg: Heading in degrees, counter-clockwise from ``+x``.
        static: ``False`` marks a mover driven at control rate (SPEC v2 S5.1 obstacles).
        params: Any extra per-kind fields (radius, speed, tag id, ...).
    """

    kind: str
    x: float
    y: float
    yaw_deg: float = 0.0
    static: bool = True
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the YAML-serialisable form."""
        out: dict[str, Any] = {
            "kind": self.kind,
            "pose": [float(self.x), float(self.y), float(self.yaw_deg)],
            "static": bool(self.static),
        }
        if self.params:
            out["params"] = dict(self.params)
        return out

    @staticmethod
    def from_dict(data: dict[str, Any]) -> MapObject:
        """Build a :class:`MapObject` from its YAML form.

        Args:
            data: Mapping with ``kind`` and ``pose`` keys.

        Returns:
            The parsed object.

        Raises:
            MapValidationError: If required keys are missing or malformed.
        """
        if "kind" not in data or "pose" not in data:
            raise MapValidationError(f"object needs 'kind' and 'pose' keys, got {sorted(data)}")
        pose = list(data["pose"])
        if len(pose) != 3:
            raise MapValidationError(f"object pose must be [x, y, yaw_deg], got {pose}")
        return MapObject(
            kind=str(data["kind"]),
            x=float(pose[0]),
            y=float(pose[1]),
            yaw_deg=float(pose[2]),
            static=bool(data.get("static", True)),
            params=dict(data.get("params", {})),
        )


@dataclass
class CityMap:
    """A validated city layout.

    Attributes:
        name: Map name; also the stem of the generated ``.usda`` and YAML files.
        tiles: ``tiles[row][col]``, row 0 northernmost.
        tile_size: Tile pitch in metres.
        objects: Props to place.
        meta: Free-form metadata. Recognised keys: ``walls`` (bool), ``geometry_bucket`` (int),
            ``palette_index`` (int), ``variant_index`` (int).
        origin: World coordinate of the map's minimum-x, minimum-y corner. ``None`` centres the
            map on ``(0, 0)``.
    """

    name: str
    tiles: list[list[Tile]]
    tile_size: float = NOMINAL_TILE_SPEC.pitch_m
    objects: list[MapObject] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    origin: tuple[float, float] | None = None

    # -------------------------------------------------------------------------- basic shape
    @property
    def n_rows(self) -> int:
        """Number of grid rows."""
        return len(self.tiles)

    @property
    def n_cols(self) -> int:
        """Number of grid columns."""
        return len(self.tiles[0]) if self.tiles else 0

    @property
    def origin_xy(self) -> tuple[float, float]:
        """World coordinate of the minimum-x, minimum-y corner of the grid."""
        if self.origin is not None:
            return self.origin
        return (-0.5 * self.n_cols * self.tile_size, -0.5 * self.n_rows * self.tile_size)

    @property
    def half_extent_m(self) -> float:
        """Largest ``|x|`` or ``|y|`` reached by any tile corner."""
        ox, oy = self.origin_xy
        xs = (abs(ox), abs(ox + self.n_cols * self.tile_size))
        ys = (abs(oy), abs(oy + self.n_rows * self.tile_size))
        return max(max(xs), max(ys))

    def tile_center_xy(self, row: int, col: int) -> tuple[float, float]:
        """World centre of a grid cell.

        This is the single place where the "YAML row 0 is north" flip is applied.

        Args:
            row: Grid row, 0 northernmost.
            col: Grid column, 0 westernmost.

        Returns:
            ``(x, y)`` in metres.
        """
        ox, oy = self.origin_xy
        return (ox + (col + 0.5) * self.tile_size, oy + (self.n_rows - 1 - row + 0.5) * self.tile_size)

    def tile_at(self, row: int, col: int) -> Tile | None:
        """Return the tile at ``(row, col)``, or ``None`` if outside the grid."""
        if 0 <= row < self.n_rows and 0 <= col < self.n_cols:
            return self.tiles[row][col]
        return None

    def cell_of_xy(self, x: float, y: float) -> tuple[int, int] | None:
        """Return the grid cell containing a world point, or ``None`` if outside the map.

        Args:
            x: World x in metres.
            y: World y in metres.

        Returns:
            ``(row, col)`` or ``None``.
        """
        ox, oy = self.origin_xy
        col = int((x - ox) // self.tile_size)
        row_from_south = int((y - oy) // self.tile_size)
        row = self.n_rows - 1 - row_from_south
        if 0 <= row < self.n_rows and 0 <= col < self.n_cols:
            return row, col
        return None

    def layout_signature(self) -> tuple[tuple[str, ...], ...]:
        """Hashable identity of the tile layout, ignoring pitch, objects and metadata.

        Two maps with the same signature describe the same road network, which is what
        "held out" has to mean for the evaluation layouts.

        Returns:
            The grid of canonical tile strings.
        """
        return tuple(tuple(format_tile(t) for t in line) for line in self.tiles)

    def iter_tiles(self) -> Iterator[tuple[int, int, Tile]]:
        """Yield ``(row, col, tile)`` for every grid cell in row-major order."""
        for row, line in enumerate(self.tiles):
            for col, tile in enumerate(line):
                yield row, col, tile

    def drivable_cells(self) -> list[tuple[int, int]]:
        """All ``(row, col)`` cells that carry lane geometry."""
        return [(r, c) for r, c, t in self.iter_tiles() if t.drivable]

    # -------------------------------------------------------------------------- validation
    def validate(self, half_extent_m: float = ENV_HALF_EXTENT_M) -> CityMap:
        """Check that the map describes a buildable city.

        Args:
            half_extent_m: AABB half extent the map must fit inside.

        Returns:
            ``self``, so calls can be chained.

        Raises:
            MapValidationError: On a non-rectangular grid, an empty grid, a non-positive tile
                size, a dangling open edge, no drivable tiles, or an oversized footprint.
        """
        if self.n_rows == 0 or self.n_cols == 0:
            raise MapValidationError(f"map {self.name!r} has an empty tile grid")
        widths = {len(line) for line in self.tiles}
        if len(widths) != 1:
            raise MapValidationError(f"map {self.name!r} grid is ragged: row widths {sorted(widths)}")
        if self.tile_size <= 0.0:
            raise MapValidationError(f"map {self.name!r} tile_size must be > 0, got {self.tile_size}")
        if not self.drivable_cells():
            raise MapValidationError(f"map {self.name!r} has no drivable tiles")
        for row, col, tile in self.iter_tiles():
            for edge in tile.open_edges:
                d_row, d_col = _EDGE_STEP[edge]
                neighbour = self.tile_at(row + d_row, col + d_col)
                if neighbour is None:
                    raise MapValidationError(
                        f"map {self.name!r}: tile ({row}, {col}) {format_tile(tile)!r} opens {edge} "
                        f"off the edge of the grid"
                    )
                if _OPPOSITE[edge] not in neighbour.open_edges:
                    raise MapValidationError(
                        f"map {self.name!r}: tile ({row}, {col}) {format_tile(tile)!r} opens {edge} "
                        f"onto ({row + d_row}, {col + d_col}) {format_tile(neighbour)!r}, which is "
                        f"closed on its {_OPPOSITE[edge]} edge"
                    )
        if self.half_extent_m > half_extent_m + 1e-9:
            raise MapValidationError(
                f"map {self.name!r} half extent {self.half_extent_m:.3f} m exceeds the per-env "
                f"AABB limit of {half_extent_m:.3f} m"
            )
        return self

    def topology_warnings(self) -> list[str]:
        """Soft Duckietown topology checks, reported rather than enforced.

        Returns:
            Human-readable warnings: intersections adjacent to curves or other intersections.
        """
        warnings: list[str] = []
        for row, col, tile in self.iter_tiles():
            if tile.kind not in ("threeway", "fourway"):
                continue
            for edge in tile.open_edges:
                d_row, d_col = _EDGE_STEP[edge]
                neighbour = self.tile_at(row + d_row, col + d_col)
                if neighbour is not None and neighbour.kind in ("curve", "threeway", "fourway"):
                    warnings.append(
                        f"{self.name}: intersection at ({row}, {col}) is adjacent to "
                        f"{format_tile(neighbour)} at ({row + d_row}, {col + d_col})"
                    )
        return warnings

    def is_closed_loop(self) -> bool:
        """Whether every drivable tile has exactly two open edges and they form one cycle.

        Intersection maps return ``False`` by construction; this is a property of pure loops.

        Returns:
            ``True`` if the drivable tiles form a single closed 2-regular cycle.
        """
        cells = self.drivable_cells()
        if not cells:
            return False
        if any(len(self.tiles[r][c].open_edges) != 2 for r, c in cells):
            return False
        start = cells[0]
        seen = {start}
        prev: tuple[int, int] | None = None
        cur = start
        for _ in range(len(cells) + 1):
            tile = self.tiles[cur[0]][cur[1]]
            nxt = None
            for edge in sorted(tile.open_edges):
                d_row, d_col = _EDGE_STEP[edge]
                cand = (cur[0] + d_row, cur[1] + d_col)
                if cand != prev:
                    nxt = cand
                    break
            if nxt is None:
                return False
            prev, cur = cur, nxt
            if cur == start:
                return len(seen) == len(cells)
            if cur in seen:
                return False
            seen.add(cur)
        return False

    # -------------------------------------------------------------------------- (de)serialise
    def to_dict(self) -> dict[str, Any]:
        """Return the YAML-serialisable form of this map."""
        out: dict[str, Any] = {
            "name": self.name,
            "tile_size": float(self.tile_size),
            "tiles": [[format_tile(t) for t in line] for line in self.tiles],
            "objects": [o.to_dict() for o in self.objects],
        }
        if self.meta:
            out["meta"] = dict(self.meta)
        if self.origin is not None:
            out["origin"] = [float(self.origin[0]), float(self.origin[1])]
        return out

    def to_yaml(self) -> str:
        """Serialise this map to a YAML document."""
        return yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=None, width=200)

    @staticmethod
    def from_dict(data: dict[str, Any], validate: bool = True) -> CityMap:
        """Build a :class:`CityMap` from its YAML form.

        Args:
            data: Parsed YAML mapping.
            validate: Run :meth:`validate` before returning.

        Returns:
            The parsed map.

        Raises:
            MapValidationError: If required keys are missing or the map is invalid.
        """
        if "tiles" not in data:
            raise MapValidationError(f"map document has no 'tiles' key (keys: {sorted(data)})")
        raw_size = data.get("tile_size", NOMINAL_TILE_SPEC.pitch_m)
        if isinstance(raw_size, dict):
            sx, sy = float(raw_size.get("x", 0.0)), float(raw_size.get("y", 0.0))
            if abs(sx - sy) > 1e-9:
                raise MapValidationError(f"non-square tile_size {raw_size} is not supported")
            tile_size = sx
        else:
            tile_size = float(raw_size)
        tiles = [[parse_tile(cell) for cell in line] for line in data["tiles"]]
        origin_raw = data.get("origin")
        origin = (float(origin_raw[0]), float(origin_raw[1])) if origin_raw is not None else None
        city = CityMap(
            name=str(data.get("name", "unnamed")),
            tiles=tiles,
            tile_size=tile_size,
            objects=[MapObject.from_dict(o) for o in data.get("objects") or []],
            meta=dict(data.get("meta") or {}),
            origin=origin,
        )
        return city.validate() if validate else city


def load_map(path: str | Path, validate: bool = True) -> CityMap:
    """Load a map from a YAML file.

    Args:
        path: Path to the YAML document.
        validate: Run :meth:`CityMap.validate` before returning.

    Returns:
        The loaded map. Its ``name`` falls back to the file stem.

    Raises:
        MapValidationError: If the document is not a mapping or the map is invalid.
    """
    text = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise MapValidationError(f"{path}: expected a YAML mapping, got {type(data).__name__}")
    data.setdefault("name", Path(path).stem)
    return CityMap.from_dict(data, validate=validate)


def save_map(city: CityMap, path: str | Path) -> Path:
    """Write a map to a YAML file, creating parent directories.

    Args:
        city: The map to write.
        path: Destination file.

    Returns:
        The path written.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(city.to_yaml(), encoding="utf-8")
    return out


# ------------------------------------------------------------------------------ construction
def map_from_cycle(
    name: str,
    n_rows: int,
    n_cols: int,
    cycle: Sequence[tuple[int, int]],
    tile_size: float = NOMINAL_TILE_SPEC.pitch_m,
    fill: str = "grass",
    inner_fill: str | None = None,
    objects: Sequence[MapObject] = (),
    meta: dict[str, Any] | None = None,
) -> CityMap:
    """Build a pure-loop map from an ordered list of road cells.

    Args:
        name: Map name.
        n_rows: Grid rows.
        n_cols: Grid columns.
        cycle: Ordered, closed, 4-adjacent list of ``(row, col)`` road cells with no repeats.
        tile_size: Tile pitch in metres.
        fill: Kind used for every non-road cell.
        inner_fill: Kind used for non-road cells fully surrounded by the grid interior; useful
            for putting ``asphalt`` inside a loop and ``grass`` outside it. ``None`` uses ``fill``
            everywhere.
        objects: Props to attach.
        meta: Metadata to attach.

    Returns:
        A validated :class:`CityMap`.

    Raises:
        MapValidationError: If the cycle is not a simple closed 4-adjacent cycle, or if the
            resulting map fails validation.
    """
    cells = [(int(r), int(c)) for r, c in cycle]
    if len(cells) < 4:
        raise MapValidationError(f"{name}: a closed loop needs at least 4 cells, got {len(cells)}")
    if len(set(cells)) != len(cells):
        raise MapValidationError(f"{name}: cycle repeats a cell")
    for i, (r, c) in enumerate(cells):
        if not (0 <= r < n_rows and 0 <= c < n_cols):
            raise MapValidationError(f"{name}: cycle cell ({r}, {c}) is outside the {n_rows}x{n_cols} grid")
        pr, pc = cells[(i + 1) % len(cells)]
        if abs(pr - r) + abs(pc - c) != 1:
            raise MapValidationError(f"{name}: cycle cells ({r}, {c}) and ({pr}, {pc}) are not adjacent")

    grid = [[Tile(fill) for _ in range(n_cols)] for _ in range(n_rows)]
    if inner_fill is not None:
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        for r in range(min(rows), max(rows) + 1):
            for c in range(min(cols), max(cols) + 1):
                grid[r][c] = Tile(inner_fill)
    for i, (r, c) in enumerate(cells):
        edges = set()
        for other in (cells[(i - 1) % len(cells)], cells[(i + 1) % len(cells)]):
            d_row, d_col = other[0] - r, other[1] - c
            edges.add(next(e for e, step in _EDGE_STEP.items() if step == (d_row, d_col)))
        kind, rot = kind_rot_for_edges(edges)
        grid[r][c] = Tile(kind, rot)
    return CityMap(
        name=name,
        tiles=grid,
        tile_size=tile_size,
        objects=list(objects),
        meta=dict(meta or {}),
    ).validate()


def map_from_rows(
    name: str,
    rows: Sequence[str],
    tile_size: float = NOMINAL_TILE_SPEC.pitch_m,
    objects: Sequence[MapObject] = (),
    meta: dict[str, Any] | None = None,
) -> CityMap:
    """Build a map from whitespace-separated tile strings, one string per grid row.

    Args:
        name: Map name.
        rows: One string per row, northernmost first; cells separated by whitespace.
        tile_size: Tile pitch in metres.
        objects: Props to attach.
        meta: Metadata to attach.

    Returns:
        A validated :class:`CityMap`.
    """
    grid = [[parse_tile(cell) for cell in line.split()] for line in rows]
    return CityMap(
        name=name, tiles=grid, tile_size=tile_size, objects=list(objects), meta=dict(meta or {})
    ).validate()


# ------------------------------------------------------------------- trajectory complexity
@dataclass(frozen=True)
class LoopComplexity:
    """How demanding one layout's driving line is.

    The definitions are those of the trajectory-complexity audit of the 68 layouts under
    ``build/city``, so a number produced here is directly comparable with that table. A *turn* is
    a cycle cell whose incoming and outgoing grid steps differ; a *chicane* is an adjacent
    opposite-handed pair in the cyclic turn-only sequence, which is the pattern that forces a
    steering reversal; the *longest straight* is the longest cyclic run of non-turning cells, and
    a long one is where a lane-following policy can coast.

    Attributes:
        name: Map name, carried through for reporting.
        drivable_tiles: Number of drivable cells.
        loop_tiles: Cells on the single closed cycle, or ``0`` when the layout is not one.
        turns: Cycle cells that change direction.
        chicanes: Adjacent opposite-handed pairs in the cyclic turn sequence.
        longest_straight: Longest cyclic run of consecutive straight cells.
        intersections: Three- and four-way tiles.
        is_loop: Whether the drivable tiles form one closed 2-regular cycle.
        turn_sequence: One character per cycle cell, ``"L"``, ``"R"`` or ``"S"``; empty when the
            layout is not a single closed cycle.
    """

    name: str
    drivable_tiles: int
    loop_tiles: int
    turns: int
    chicanes: int
    longest_straight: int
    intersections: int
    is_loop: bool
    turn_sequence: str

    @property
    def score(self) -> int:
        """Aggregate difficulty: ``turns + 2 * chicanes + 3 * intersections``."""
        return int(self.turns + 2 * self.chicanes + 3 * self.intersections)


def _road_adjacency(city: CityMap) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """Adjacency of the drivable cells, through mutually open tile edges.

    Args:
        city: The map to walk.

    Returns:
        ``{cell: [neighbour, ...]}`` over the drivable cells only, neighbours in ``N, E, S, W``
        edge-label order so any walk built on it is deterministic.
    """
    tiles = {(r, c): city.tiles[r][c] for r, c in city.drivable_cells()}
    adjacency: dict[tuple[int, int], list[tuple[int, int]]] = {cell: [] for cell in tiles}
    for (row, col), tile in tiles.items():
        for edge in sorted(tile.open_edges):
            d_row, d_col = _EDGE_STEP[edge]
            neighbour = (row + d_row, col + d_col)
            if neighbour in tiles and _OPPOSITE[edge] in tiles[neighbour].open_edges:
                adjacency[(row, col)].append(neighbour)
    return adjacency


def loop_complexity(city: CityMap) -> LoopComplexity:
    """Measure the trajectory complexity of a layout.

    Layouts that are not a single closed cycle (the intersection maps) still report their
    drivable-tile and intersection counts; their turn statistics are reported as ``0`` because a
    single cyclic turn sequence is not defined for them.

    Args:
        city: The map to measure.

    Returns:
        The metrics, as a :class:`LoopComplexity`.
    """
    cells = city.drivable_cells()
    intersections = sum(1 for r, c in cells if city.tiles[r][c].kind in ("threeway", "fourway"))
    base: dict[str, Any] = {
        "name": city.name,
        "drivable_tiles": len(cells),
        "intersections": intersections,
    }
    if not city.is_closed_loop():
        return LoopComplexity(
            **base,
            loop_tiles=0,
            turns=0,
            chicanes=0,
            longest_straight=0,
            is_loop=False,
            turn_sequence="",
        )

    adjacency = _road_adjacency(city)
    start = min(cells)
    cycle = [start]
    prev, cur = start, sorted(adjacency[start])[0]
    while cur != start:
        cycle.append(cur)
        prev, cur = cur, next(n for n in adjacency[cur] if n != prev)

    labels: list[str] = []
    for i, (row, col) in enumerate(cycle):
        before, after = cycle[i - 1], cycle[(i + 1) % len(cycle)]
        d_in = (row - before[0], col - before[1])
        d_out = (after[0] - row, after[1] - col)
        if d_in == d_out:
            labels.append("S")
            continue
        # World axes are x = +col and y = -row, so the z component of d_in x d_out is
        # d_in.x * d_out.y - d_in.y * d_out.x with those substitutions. Positive turns left.
        cross = d_in[1] * (-d_out[0]) - (-d_in[0]) * d_out[1]
        labels.append("L" if cross > 0 else "R")

    turn_seq = [x for x in labels if x != "S"]
    chicanes = (
        sum(1 for i, x in enumerate(turn_seq) if x != turn_seq[(i + 1) % len(turn_seq)])
        if len(turn_seq) > 1
        else 0
    )
    longest = run = 0
    for label in labels + labels:
        run = run + 1 if label == "S" else 0
        longest = max(longest, run)
    return LoopComplexity(
        **base,
        loop_tiles=len(cycle),
        turns=len(turn_seq),
        chicanes=chicanes,
        longest_straight=min(longest, labels.count("S")),
        is_loop=True,
        turn_sequence="".join(labels),
    )


# ------------------------------------------------------------------------- random generation
def _random_spanning_tree(
    rows: int, cols: int, rng: np.random.Generator, straight_bias: float
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Randomised depth-first spanning tree over a ``rows x cols`` grid graph.

    Args:
        rows: Grid rows.
        cols: Grid columns.
        rng: A ``numpy.random.Generator``.
        straight_bias: Probability in ``[0, 1)`` of preferring to continue in the direction the
            walk arrived from, which lengthens the straights in the resulting track.

    Returns:
        ``rows * cols - 1`` undirected edges as ``((r0, c0), (r1, c1))`` pairs.
    """
    start = (int(rng.integers(rows)), int(rng.integers(cols)))
    visited = {start}
    stack: list[tuple[tuple[int, int], tuple[int, int]]] = [(start, (0, 0))]
    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    while stack:
        (r, c), came = stack[-1]
        options = []
        for d_row, d_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + d_row, c + d_col
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                options.append((d_row, d_col))
        if not options:
            stack.pop()
            continue
        straight = [d for d in options if d == came]
        if straight and float(rng.random()) < straight_bias:
            choice = straight[0]
        else:
            choice = options[int(rng.integers(len(options)))]
        nxt = (r + choice[0], c + choice[1])
        visited.add(nxt)
        edges.append(((r, c), nxt))
        stack.append((nxt, choice))
    return edges


def _hamiltonian_cycle(
    coarse_rows: int, coarse_cols: int, rng: np.random.Generator, straight_bias: float
) -> list[tuple[int, int]]:
    """Build a Hamiltonian cycle on a ``2*coarse_rows x 2*coarse_cols`` grid.

    A spanning tree of the coarse grid induces exactly one closed loop through every fine cell:
    each coarse cell contributes a 4-cycle over its four fine cells, and each tree edge splices
    two of those cycles into one by swapping the pair of facing fine edges. With
    ``coarse_rows * coarse_cols - 1`` tree edges, all cycles merge into a single cycle.

    Args:
        coarse_rows: Coarse grid rows.
        coarse_cols: Coarse grid columns.
        rng: A ``numpy.random.Generator``.
        straight_bias: Forwarded to :func:`_random_spanning_tree`.

    Returns:
        The cycle as an ordered list of ``(row, col)`` fine cells.

    Raises:
        MapValidationError: If the construction fails to produce a single cycle (never expected;
            the assertion is kept because the rest of the generator relies on it).
    """
    adjacency: dict[tuple[int, int], set[tuple[int, int]]] = {}

    def link(a: tuple[int, int], b: tuple[int, int]) -> None:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)

    def unlink(a: tuple[int, int], b: tuple[int, int]) -> None:
        adjacency[a].discard(b)
        adjacency[b].discard(a)

    for i in range(coarse_rows):
        for j in range(coarse_cols):
            nw, ne = (2 * i, 2 * j), (2 * i, 2 * j + 1)
            sw, se = (2 * i + 1, 2 * j), (2 * i + 1, 2 * j + 1)
            link(nw, ne)
            link(ne, se)
            link(se, sw)
            link(sw, nw)

    for (i0, j0), (i1, j1) in _random_spanning_tree(coarse_rows, coarse_cols, rng, straight_bias):
        if (i0, j0) > (i1, j1):
            i0, j0, i1, j1 = i1, j1, i0, j0
        if i0 == i1:  # horizontal tree edge: (i0, j0) - (i0, j0 + 1)
            a, b = (2 * i0, 2 * j0 + 1), (2 * i0 + 1, 2 * j0 + 1)
            c, d = (2 * i0, 2 * j0 + 2), (2 * i0 + 1, 2 * j0 + 2)
        else:  # vertical tree edge: (i0, j0) - (i0 + 1, j0)
            a, b = (2 * i0 + 1, 2 * j0), (2 * i0 + 1, 2 * j0 + 1)
            c, d = (2 * i0 + 2, 2 * j0), (2 * i0 + 2, 2 * j0 + 1)
        unlink(a, b)
        unlink(c, d)
        link(a, c)
        link(b, d)

    total = 4 * coarse_rows * coarse_cols
    if any(len(v) != 2 for v in adjacency.values()) or len(adjacency) != total:
        raise MapValidationError("Hamiltonian cycle construction produced a non 2-regular graph")
    start = (0, 0)
    cycle = [start]
    prev, cur = start, sorted(adjacency[start])[0]
    while cur != start:
        cycle.append(cur)
        nxt = next(n for n in sorted(adjacency[cur]) if n != prev)
        prev, cur = cur, nxt
    if len(cycle) != total:
        raise MapValidationError(f"expected a single cycle of {total} cells, walked {len(cycle)}")
    return cycle


def random_loop_map(
    name: str,
    seed: int = 0,
    coarse_rows: int = 2,
    coarse_cols: int = 2,
    border: int = 1,
    tile_size: float = NOMINAL_TILE_SPEC.pitch_m,
    straight_bias: float = 0.7,
    meta: dict[str, Any] | None = None,
) -> CityMap:
    """Generate a random, always-valid, closed drivable loop.

    The loop is a Hamiltonian cycle over a ``2*coarse_rows x 2*coarse_cols`` block of road tiles,
    surrounded by ``border`` rings of grass. Because the construction is a spanning-tree splice
    it can never fail or self-intersect, so every seed yields a valid map.

    Args:
        name: Map name.
        seed: Seed for the layout.
        coarse_rows: Half the number of road rows.
        coarse_cols: Half the number of road columns.
        border: Rings of grass around the road block; at least 1 so no road opens off-grid.
        tile_size: Tile pitch in metres.
        straight_bias: Preference for continuing straight when growing the spanning tree.
        meta: Metadata to attach; the seed and generator name are added automatically.

    Returns:
        A validated :class:`CityMap` whose drivable tiles form one closed loop.

    Raises:
        MapValidationError: If ``border`` is below 1 or the coarse grid is degenerate.
    """
    if border < 1:
        raise MapValidationError(f"border must be >= 1 so no road opens off-grid, got {border}")
    if coarse_rows < 1 or coarse_cols < 1:
        raise MapValidationError("coarse_rows and coarse_cols must be >= 1")
    rng = np.random.default_rng(seed)
    fine = _hamiltonian_cycle(coarse_rows, coarse_cols, rng, straight_bias)
    cycle = [(r + border, c + border) for r, c in fine]
    full_meta = {"generator": "random_loop_map", "seed": int(seed), **(meta or {})}
    return map_from_cycle(
        name=name,
        n_rows=2 * coarse_rows + 2 * border,
        n_cols=2 * coarse_cols + 2 * border,
        cycle=cycle,
        tile_size=tile_size,
        fill="grass",
        meta=full_meta,
    )


#: Coarse-grid shapes the random generator cycles through, as ``(rows, cols, border)``.
#: A shape of ``(r, c, b)`` yields a ``(2r + 2b) x (2c + 2b)`` grid with ``4rc`` road tiles.
#: Every shape stays within the 5x5 .. 8x8 grid range of SPEC v2 S3.3 and inside the per-env AABB.
_RANDOM_SHAPES: Final[tuple[tuple[int, int, int], ...]] = (
    (2, 2, 1),
    (2, 2, 2),
    (2, 3, 1),
    (3, 2, 1),
    (3, 3, 1),
)


#: Coarse-grid shapes the ``"hard"`` profile draws from. Every entry is a 24- or 36-tile loop
#: inside an 8x8 grid, the largest SPEC v2 S3.3 allows. Six slots in eight are the 36-tile 8x8
#: shape, because that is the only shape whose score ceiling (16 turns, 6 chicanes, so 28) is
#: above the nominal set's median of 20; the two 24-tile slots keep the grid aspect ratio varied
#: without dragging the distribution down. The two 24-tile slots sit at indices 5 and 7 so that
#: the four eval layouts, which take shape indices 0 to 3, are all the 36-tile shape. Measured
#: over ``variant_maps(64, difficulty="hard")``: score min 20, median 25, mean 24.5, against
#: nominal's min 4, median 20, mean 17.5.
_HARD_SHAPES: Final[tuple[tuple[int, int, int], ...]] = (
    (3, 3, 1),
    (3, 3, 1),
    (3, 3, 1),
    (3, 3, 1),
    (3, 3, 1),
    (3, 2, 1),
    (3, 3, 1),
    (2, 3, 1),
)


@dataclass(frozen=True)
class DifficultyProfile:
    """How the random layout generator trades trajectory complexity against gentleness.

    The generator cannot simply "add a turn": a layout is a Hamiltonian cycle over the fine grid,
    so the turn count is a property of which spanning tree was drawn rather than a free
    parameter. The three levers that do work are the coarse shape, which fixes how many cells the
    cycle has; ``straight_bias``, which biases the spanning-tree walk; and *selection*, which
    draws several layouts for one slot and keeps the one that ranks best under
    :func:`loop_complexity`. ``candidates = 1`` disables selection entirely, which is what keeps
    the ``"nominal"`` profile bit-for-bit identical to the layouts already on disk.

    Attributes:
        name: Profile name, as accepted by :func:`difficulty_profile`.
        shapes: Coarse ``(rows, cols, border)`` shapes cycled through by attempt index; a shape
            ``(r, c, b)`` yields a ``(2r + 2b) x (2c + 2b)`` grid with ``4rc`` road tiles.
        straight_bias: Forwarded to :func:`random_loop_map`; lower values grow twistier trees.
        candidates: Layouts sampled per attempt before one is kept. ``1`` keeps the first sample,
            which is the historical behaviour.
        sense: ``+1`` keeps the hardest-ranked candidate, ``-1`` the gentlest.
        straight_weight: Weight of the longest straight run in the ranking, so the hard sense
            also avoids layouts that are mostly one long straight.
        use_builtins: Whether the hand-authored built-ins lead the variant sequence. The hard
            profile turns this off: ``loop_small`` has four turns and ``intersection_4way`` is
            not a loop at all, so keeping them would put the five easiest maps in the set.
    """

    name: str
    shapes: tuple[tuple[int, int, int], ...]
    straight_bias: float
    candidates: int = 1
    sense: int = 1
    straight_weight: float = 0.0
    use_builtins: bool = True

    def __post_init__(self) -> None:
        """Validate the profile.

        Raises:
            ValueError: If the profile cannot drive the generator.
        """
        if not self.shapes:
            raise ValueError(f"difficulty profile {self.name!r} has no coarse shapes")
        if self.candidates < 1:
            raise ValueError(f"difficulty profile {self.name!r} needs candidates >= 1")
        if self.sense not in (1, -1):
            raise ValueError(f"difficulty profile {self.name!r} needs sense in (1, -1)")
        if not 0.0 <= self.straight_bias <= 1.0:
            raise ValueError(f"difficulty profile {self.name!r} needs straight_bias in [0, 1]")

    def rank(self, complexity: LoopComplexity) -> float:
        """Score one candidate layout; the highest-ranked candidate is kept.

        Args:
            complexity: Metrics of the candidate.

        Returns:
            The ranking value.
        """
        return float(self.sense) * (
            complexity.score - self.straight_weight * float(complexity.longest_straight)
        )


#: The named difficulty profiles. ``"nominal"`` is the historical generator and must stay so:
#: ``build/city`` was generated with it and a paused training run references those exact layouts.
#:
#: There is deliberately no ``"easy"`` preset. The construction's floor is already easy: the
#: smallest loop it can emit is a 16-tile Hamiltonian cycle, which always has exactly 8 turns and
#: 2 chicanes and so always scores 12, and the nominal set is already mostly those. A profile
#: built from the small shapes with ``sense = -1`` was measured at 16 variants and moved the mean
#: score from 13.6 to 12.5 with an identical median, while shrinking the pool of distinct layouts
#: from thousands to about 38 -- a placebo that costs layout diversity. Build one with
#: :class:`DifficultyProfile` if a curriculum genuinely needs it; it is not shipped as a name.
DIFFICULTY_PROFILES: Final[dict[str, DifficultyProfile]] = {
    "nominal": DifficultyProfile(name="nominal", shapes=_RANDOM_SHAPES, straight_bias=0.7),
    "hard": DifficultyProfile(
        name="hard",
        shapes=_HARD_SHAPES,
        straight_bias=0.0,
        candidates=24,
        sense=1,
        straight_weight=0.5,
        use_builtins=False,
    ),
}

#: Difficulty names accepted on the command line, easiest first.
DIFFICULTY_NAMES: Final[tuple[str, ...]] = ("nominal", "hard")


def difficulty_profile(difficulty: str | DifficultyProfile) -> DifficultyProfile:
    """Resolve a difficulty name to its profile.

    Args:
        difficulty: A name from :data:`DIFFICULTY_NAMES`, or a profile to pass through.

    Returns:
        The profile.

    Raises:
        ValueError: If the name is not one of :data:`DIFFICULTY_NAMES`.
    """
    if isinstance(difficulty, DifficultyProfile):
        return difficulty
    key = str(difficulty).strip().lower()
    if key not in DIFFICULTY_PROFILES:
        raise ValueError(f"unknown difficulty {difficulty!r}; expected one of {DIFFICULTY_NAMES}")
    return DIFFICULTY_PROFILES[key]


def _distinct_random_loop(
    name: str,
    index: int,
    seed: int,
    tile_size: float,
    meta: dict[str, Any],
    seen: set[tuple[tuple[str, ...], ...]],
    max_attempts: int = 256,
    profile: DifficultyProfile | None = None,
) -> CityMap:
    """Generate a random loop whose layout is not already in ``seen``.

    Each attempt draws ``profile.candidates`` layouts from the same coarse shape, ranks them with
    :meth:`DifficultyProfile.rank` and returns the best-ranked one that is not already taken.
    With the nominal profile's ``candidates = 1`` that reduces to drawing exactly one layout per
    attempt from exactly the historical seed, so nominal output is unchanged.

    Args:
        name: Map name.
        index: Position in the variant sequence; mixed into the seed.
        seed: Base seed.
        tile_size: Tile pitch in metres.
        meta: Metadata to attach.
        seen: Layout signatures already taken.
        max_attempts: Attempts before giving up and returning the last candidate.
        profile: Difficulty profile; ``None`` uses the nominal one.

    Returns:
        A validated map, distinct from ``seen`` unless ``max_attempts`` was exhausted.
    """
    prof = profile if profile is not None else DIFFICULTY_PROFILES["nominal"]
    city: CityMap | None = None
    for attempt in range(max_attempts):
        rows, cols, border = prof.shapes[(index + attempt) % len(prof.shapes)]
        base_seed = seed + index * 1009 + attempt * 7919
        ranked: list[tuple[float, int, CityMap]] = []
        for draw in range(prof.candidates):
            candidate = random_loop_map(
                name,
                seed=base_seed + draw * 104729,
                coarse_rows=rows,
                coarse_cols=cols,
                border=border,
                tile_size=tile_size,
                straight_bias=prof.straight_bias,
                meta=meta,
            )
            # Sort key: best rank first, draw order breaking ties, so the pick is deterministic.
            ranked.append((-prof.rank(loop_complexity(candidate)), draw, candidate))
        ranked.sort(key=lambda item: (item[0], item[1]))
        city = ranked[0][2]
        for _, _, candidate in ranked:
            if candidate.layout_signature() not in seen:
                return candidate
    assert city is not None
    return city


# ------------------------------------------------------------------------------- built-ins
_LOOP_SMALL_CYCLE: Final[tuple[tuple[int, int], ...]] = (
    (1, 1),
    (1, 2),
    (1, 3),
    (2, 3),
    (3, 3),
    (3, 2),
    (3, 1),
    (2, 1),
)

_LOOP_BIG_CYCLE: Final[tuple[tuple[int, int], ...]] = (
    (1, 1),
    (1, 2),
    (1, 3),
    (1, 4),
    (1, 5),
    (2, 5),
    (3, 5),
    (4, 5),
    (5, 5),
    (5, 4),
    (5, 3),
    (5, 2),
    (5, 1),
    (4, 1),
    (3, 1),
    (2, 1),
)

_ZIGZAG_CYCLE: Final[tuple[tuple[int, int], ...]] = (
    (1, 1),
    (1, 2),
    (2, 2),
    (2, 3),
    (2, 4),
    (1, 4),
    (1, 5),
    (2, 5),
    (3, 5),
    (4, 5),
    (5, 5),
    (5, 4),
    (4, 4),
    (4, 3),
    (4, 2),
    (5, 2),
    (5, 1),
    (4, 1),
    (3, 1),
    (2, 1),
)

_INTERSECTION_4WAY_ROWS: Final[tuple[str, ...]] = (
    "grass grass       grass       grass       grass       grass        grass",
    "grass curve_left/N straight/E 3way_left/E straight/E  curve_left/W grass",
    "grass straight/N   grass      straight/N  grass       straight/N   grass",
    "grass 3way_left/S  straight/E 4way        straight/E  3way_left/N  grass",
    "grass straight/N   grass      straight/N  grass       straight/N   grass",
    "grass curve_left/E straight/E 3way_left/W straight/E  curve_left/S grass",
    "grass grass       grass       grass       grass       grass        grass",
)

#: Names of every built-in map.
BUILTIN_MAP_NAMES: Final[tuple[str, ...]] = (
    "loop_small",
    "loop_big",
    "zigzag",
    "intersection_4way",
    "obstacles_dynamic",
)


def _lane_seed_point(city: CityMap, row: int, col: int) -> tuple[float, float, float]:
    """A point on a lane centreline at the edge of a drivable tile.

    Used only to seed obstacle poses in the built-in maps. The offset is taken from the nominal
    tile spec in *tile units* and scaled by the map's pitch, so it tracks the pitch without this
    module having to know which geometry bucket the variant will use. The environment refines
    obstacle poses with :mod:`.lane_graph` at spawn time.

    Args:
        city: The map.
        row: Grid row of a drivable tile.
        col: Grid column of a drivable tile.

    Returns:
        ``(x, y, yaw_deg)`` on the lane that enters the tile through its first open edge.
    """
    tile = city.tiles[row][col]
    edge = sorted(tile.open_edges)[0]
    (nx, ny), (ux, uy) = EDGE_BASIS[edge]
    cx, cy = city.tile_center_xy(row, col)
    half = 0.5 * city.tile_size
    lane = NOMINAL_TILE_SPEC.lane_center_offset_tile * city.tile_size
    x = cx + nx * half * 0.9 + ux * lane
    y = cy + ny * half * 0.9 + uy * lane
    yaw = float(np.degrees(np.arctan2(-ny, -nx)))
    return x, y, yaw


def _obstacle_objects(city: CityMap) -> list[MapObject]:
    """Place a scripted obstacle set on the lane centres of a loop map.

    Args:
        city: The map to place obstacles on.

    Returns:
        Two moving NPC duckiebots, four duckies and four cones, matching the
        ``RigidObjectCollection`` budget of SPEC v2 S5.1.
    """
    cells = city.drivable_cells()
    picks = [cells[i % len(cells)] for i in (0, 3, 5, 7, 9, 11, 13, 15, 2, 4)]
    objects: list[MapObject] = []
    for i, (row, col) in enumerate(picks):
        x, y, yaw = _lane_seed_point(city, row, col)
        if i < 2:
            objects.append(
                MapObject("duckiebot", x, y, yaw, static=False, params={"speed": 0.10, "radius": 0.11})
            )
        elif i < 6:
            objects.append(MapObject("duckie", x, y, yaw, static=(i % 2 == 0), params={"radius": 0.04}))
        else:
            objects.append(MapObject("cone", x, y, yaw, static=True, params={"radius": 0.03}))
    return objects


def builtin_map(name: str, tile_size: float = NOMINAL_TILE_SPEC.pitch_m) -> CityMap:
    """Build one of the maps listed in :data:`BUILTIN_MAP_NAMES`.

    Args:
        name: Map name.
        tile_size: Tile pitch in metres.

    Returns:
        A validated :class:`CityMap`.

    Raises:
        KeyError: If ``name`` is not a built-in map.
    """
    if name == "loop_small":
        return map_from_cycle(
            "loop_small",
            5,
            5,
            _LOOP_SMALL_CYCLE,
            tile_size,
            inner_fill="asphalt",
            meta={"walls": True},
        )
    if name == "loop_big":
        return map_from_cycle("loop_big", 7, 7, _LOOP_BIG_CYCLE, tile_size, meta={"walls": True})
    if name == "zigzag":
        return map_from_cycle("zigzag", 7, 7, _ZIGZAG_CYCLE, tile_size, meta={"walls": True})
    if name == "intersection_4way":
        return map_from_rows("intersection_4way", _INTERSECTION_4WAY_ROWS, tile_size, meta={"walls": True})
    if name == "obstacles_dynamic":
        base = map_from_cycle("obstacles_dynamic", 7, 7, _LOOP_BIG_CYCLE, tile_size, meta={"walls": True})
        base.objects = _obstacle_objects(base)
        return base.validate()
    raise KeyError(f"unknown built-in map {name!r}; expected one of {BUILTIN_MAP_NAMES}")


def variant_maps(
    count: int = 64,
    seed: int = 0,
    geometry_buckets: int = 16,
    prefix: str = "city",
    tile_sizes: Sequence[float] | None = None,
    difficulty: str | DifficultyProfile = "nominal",
) -> list[CityMap]:
    """Build the deterministic set of training city variants (SPEC v2 S3.3, K = 64).

    Variant ``i`` is assigned geometry bucket ``i % geometry_buckets`` and palette index ``i``;
    the tile pitch cycles through ``tile_sizes`` so the V9 pitch axis is represented in the
    layouts themselves. The first five variants are the hand-authored built-ins so that a
    debugging run at ``num_envs <= 5`` always sees readable maps.

    Layouts are deduplicated by :meth:`CityMap.layout_signature`. Small coarse grids admit only
    a handful of spanning trees, so without this the 64 "distinct" variants would silently
    contain repeats.

    Args:
        count: Number of variants.
        seed: Base seed for the random layouts.
        geometry_buckets: Number of marking-geometry texture buckets to spread across.
        prefix: Filename prefix; variant ``i`` is named ``f"{prefix}_{i:03d}"``.
        tile_sizes: Tile pitches to cycle through; defaults to ``(0.570, 0.585, 0.600, 0.615)``.
        difficulty: A name from :data:`DIFFICULTY_NAMES` or a :class:`DifficultyProfile`. The
            default reproduces the historical generator exactly, layout for layout, which is what
            lets ``build/city`` be regenerated byte-identically while a training run references
            it. Non-nominal profiles record their name under the ``difficulty`` metadata key;
            nominal deliberately does not, because writing the key would change every YAML on
            disk.

    Returns:
        ``count`` validated maps with distinct layouts.

    Raises:
        ValueError: If ``count`` or ``geometry_buckets`` is not positive, or the difficulty is
            not a known name.
    """
    if count <= 0 or geometry_buckets <= 0:
        raise ValueError("count and geometry_buckets must be > 0")
    profile = difficulty_profile(difficulty)
    pitches = tuple(tile_sizes) if tile_sizes else (0.570, 0.585, 0.600, 0.615)
    out: list[CityMap] = []
    seen: set[tuple[tuple[str, ...], ...]] = set()
    for i in range(count):
        pitch = pitches[i % len(pitches)]
        meta: dict[str, Any] = {
            "variant_index": i,
            "geometry_bucket": i % geometry_buckets,
            "palette_index": i,
            "walls": (i % 3) != 0,
        }
        if profile.name != "nominal":
            meta["difficulty"] = profile.name
        name = f"{prefix}_{i:03d}"
        if i < len(BUILTIN_MAP_NAMES) and profile.use_builtins:
            city = builtin_map(BUILTIN_MAP_NAMES[i], tile_size=pitch)
            city.name = name
            city.meta.update(meta)
            city.validate()
        else:
            city = _distinct_random_loop(name, i, seed, pitch, meta, seen, profile=profile)
        seen.add(city.layout_signature())
        out.append(city)
    return out


def eval_maps(
    count: int = 4,
    seed: int = 90210,
    tile_size: float = NOMINAL_TILE_SPEC.pitch_m,
    exclude: set[tuple[tuple[str, ...], ...]] | None = None,
    train_count: int = 64,
    train_seed: int = 0,
    difficulty: str | DifficultyProfile = "nominal",
) -> list[CityMap]:
    """Build the frozen held-out evaluation layouts (SPEC v2 S3.3, ``eval_00 .. eval_03``).

    "Held out" is enforced, not assumed: each candidate layout is rejected if its
    :meth:`CityMap.layout_signature` appears in the training set. A disjoint seed range is not
    sufficient on its own, because small coarse grids admit only a handful of distinct loops.

    Args:
        count: Number of eval maps.
        seed: Base seed.
        tile_size: Tile pitch in metres.
        exclude: Layout signatures that must not recur. ``None`` excludes the layouts of
            ``variant_maps(train_count, seed=train_seed)``.
        train_count: Training-variant count used to build the default exclusion set.
        train_seed: Training-variant seed used to build the default exclusion set.
        difficulty: Difficulty profile for the eval layouts, and for the training set the default
            exclusion is computed from. Held-out maps are only a fair test of the training
            distribution when they come from the same profile, so passing ``"hard"`` here is what
            gives a hard build genuinely hard eval maps.

    Returns:
        ``count`` validated maps whose layouts appear in neither the training set nor each other.
    """
    profile = difficulty_profile(difficulty)
    if exclude is None:
        exclude = {
            c.layout_signature() for c in variant_maps(train_count, seed=train_seed, difficulty=profile)
        }
    seen = set(exclude)
    out: list[CityMap] = []
    for i in range(count):
        meta: dict[str, Any] = {
            "eval": True,
            "geometry_bucket": 0,
            "palette_index": 0,
            "walls": i % 2 == 0,
        }
        if profile.name != "nominal":
            meta["difficulty"] = profile.name
        city = _distinct_random_loop(f"eval_{i:02d}", i, seed, tile_size, meta, seen, profile=profile)
        seen.add(city.layout_signature())
        out.append(city)
    return out
