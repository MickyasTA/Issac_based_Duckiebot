"""Procedural tile-texture generation, pure numpy, with a dependency-free PNG codec.

Every texture in the city is painted from the millimetre dimensions in :mod:`.spec`. Nothing is
loaded from disk, so the repo carries no Duckietown image assets (SPEC v2 S3.1 / S3.4).

Dependencies
------------
This module needs **numpy only**. The PNG writer and reader are implemented here on top of
``zlib`` and ``struct`` from the standard library, so texture generation runs in the MuJoCo tools
venv (which has neither Pillow nor OpenCV) and in CI. Pillow and OpenCV are never imported.

Canonical tile orientations
---------------------------
Every drivable tile kind is defined at rotation 0 ("north", see :mod:`.maps`) by the set of tile
edges it connects. Map layouts obtain the other three orientations by rotating the tile
counter-clockwise about ``+z`` in quarter turns; textures are shared across rotations and the
rotation is carried by the UV assignment (see :func:`rotated_uv_corners`).

===========  ============================  =================================================
kind         canonical connected edges     geometry
===========  ============================  =================================================
straight     ``{N, S}``                    road along +/-y, yellow dashed line at ``x = 0``
curve        ``{S, E}``                    quarter turn, arc centre at the ``(+x, -y)`` corner
threeway     ``{N, S, W}``                 T junction, the ``E`` edge is closed
fourway      ``{N, S, E, W}``              full crossroads
asphalt      ``{}``                        featureless road surface (non drivable)
grass        ``{}``                        green surround (non drivable)
empty        ``{}``                        neutral floor (non drivable); alias ``floor``
===========  ============================  =================================================

Texture axis convention
-----------------------
Image column ``0`` is the tile's minimum ``x`` (west) and image row ``0`` is the tile's maximum
``y`` (north), i.e. the usual top-left image origin. The USD ``st`` primvar uses ``(0, 0)`` at
the minimum-``x``, minimum-``y`` corner, which is the image's *bottom* left, matching the
standard OpenGL texture origin. :func:`tile_local_to_pixel` and :func:`tile_local_to_uv` are the
only places this convention is encoded.
"""

from __future__ import annotations

import math
import struct
import zlib
from pathlib import Path
from typing import Final

import numpy as np

from .spec import IDEAL_PALETTE, NOMINAL_TILE_SPEC, ColorPalette, TileSpec

__all__ = [
    "DEFAULT_TEXTURE_RES",
    "DRIVABLE_KINDS",
    "EDGES",
    "EDGE_BASIS",
    "KIND_CONNECTIONS",
    "NON_DRIVABLE_KINDS",
    "SIGN_CARD_ASPECT",
    "SIGN_KINDS",
    "TILE_KINDS",
    "TileStyle",
    "is_drivable_kind",
    "read_png",
    "render_sign",
    "render_tile",
    "render_tile_set",
    "rotated_uv_corners",
    "save_sign_set",
    "save_tile_set",
    "tile_local_to_pixel",
    "tile_local_to_uv",
    "write_png",
]

#: Working precision of the texture rasteriser. float32 halves the memory traffic of the 2x
#: supersampled buffers; its 1e-7 relative precision is 30 nm at tile scale, far below the
#: 1.1 mm per texel of a 512 px tile.
_FLOAT = np.float32

#: Tile edges in counter-clockwise order starting at north.
EDGES: Final[tuple[str, str, str, str]] = ("N", "W", "S", "E")

#: ``edge -> (outward normal, along-edge basis vector)``. The along-edge vector ``u`` is the
#: outward normal rotated by +90 degrees (counter-clockwise).
EDGE_BASIS: Final[dict[str, tuple[tuple[float, float], tuple[float, float]]]] = {
    "N": ((0.0, 1.0), (-1.0, 0.0)),
    "W": ((-1.0, 0.0), (0.0, -1.0)),
    "S": ((0.0, -1.0), (1.0, 0.0)),
    "E": ((1.0, 0.0), (0.0, 1.0)),
}

#: Canonical (rotation 0) connectivity of every tile kind.
KIND_CONNECTIONS: Final[dict[str, frozenset[str]]] = {
    "straight": frozenset({"N", "S"}),
    "curve": frozenset({"S", "E"}),
    "threeway": frozenset({"N", "S", "W"}),
    "fourway": frozenset({"N", "S", "E", "W"}),
    "asphalt": frozenset(),
    "grass": frozenset(),
    "empty": frozenset(),
}

#: Tile kinds a robot may legally drive on (SPEC v2 S5.5 off-drivable termination).
DRIVABLE_KINDS: Final[tuple[str, ...]] = ("straight", "curve", "threeway", "fourway")

#: Tile kinds that carry no lane geometry.
NON_DRIVABLE_KINDS: Final[tuple[str, ...]] = ("asphalt", "grass", "empty")

#: Every kind, drivable first.
TILE_KINDS: Final[tuple[str, ...]] = DRIVABLE_KINDS + NON_DRIVABLE_KINDS

#: Width divided by height of the Duckietown traffic-sign card (85 mm x 155 mm, SPEC v2 S3.3).
SIGN_CARD_ASPECT: Final[float] = 85.0 / 155.0

#: Texture edge length in pixels per kind (SPEC v2 S5.6 texture budget: 512 for the drivable
#: markings, 256 for the featureless surrounds).
DEFAULT_TEXTURE_RES: Final[dict[str, int]] = {
    **dict.fromkeys(DRIVABLE_KINDS, 512),
    **dict.fromkeys(NON_DRIVABLE_KINDS, 256),
}


def is_drivable_kind(kind: str) -> bool:
    """Return whether ``kind`` is a drivable tile kind.

    Args:
        kind: Canonical tile kind name.

    Returns:
        ``True`` if a lane graph exists on this tile.
    """
    return kind in DRIVABLE_KINDS


# --------------------------------------------------------------------------------------- PNG
def write_png(path: str | Path, rgb_u8: np.ndarray, compress_level: int = 6) -> Path:
    """Write an 8-bit RGB PNG with no third-party dependency.

    Rows are written with filter type 0 (None), which keeps :func:`read_png` trivial.

    Args:
        path: Destination file.
        rgb_u8: ``(H, W, 3)`` array of ``uint8``.
        compress_level: zlib compression level, 0 .. 9.

    Returns:
        The path written.

    Raises:
        ValueError: If ``rgb_u8`` is not an ``(H, W, 3)`` ``uint8`` array.
    """
    arr = np.asarray(rgb_u8)
    if arr.ndim != 3 or arr.shape[2] != 3 or arr.dtype != np.uint8:
        raise ValueError(f"expected an (H, W, 3) uint8 array, got shape {arr.shape} dtype {arr.dtype}")
    height, width, _ = arr.shape
    stride = width * 3
    raw = np.empty((height, stride + 1), dtype=np.uint8)
    raw[:, 0] = 0
    raw[:, 1:] = arr.reshape(height, stride)

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw.tobytes(), compress_level))
        + chunk(b"IEND", b"")
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(png)
    return out


def read_png(path: str | Path) -> np.ndarray:
    """Read an 8-bit RGB PNG written by :func:`write_png`.

    Only the subset this module emits is supported: colour type 2 (RGB), bit depth 8, no
    interlacing, and filter type 0 on every scanline.

    Args:
        path: Source file.

    Returns:
        ``(H, W, 3)`` array of ``uint8``.

    Raises:
        ValueError: If the file is not a PNG this reader understands.
    """
    data = Path(path).read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path} is not a PNG")
    pos = 8
    width = height = 0
    idat = bytearray()
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        tag = data[pos + 4 : pos + 8]
        body = data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if tag == b"IHDR":
            width, height, depth, color, comp, filt, interlace = struct.unpack(">IIBBBBB", body)
            if (depth, color, comp, filt, interlace) != (8, 2, 0, 0, 0):
                raise ValueError(f"{path}: unsupported PNG variant {(depth, color, comp, filt, interlace)}")
        elif tag == b"IDAT":
            idat += body
        elif tag == b"IEND":
            break
    raw = np.frombuffer(zlib.decompress(bytes(idat)), dtype=np.uint8)
    stride = width * 3
    raw = raw.reshape(height, stride + 1)
    if np.any(raw[:, 0] != 0):
        raise ValueError(f"{path}: only PNG filter type 0 is supported by this reader")
    return raw[:, 1:].reshape(height, width, 3).copy()


# ------------------------------------------------------------------------------ coordinates
def tile_local_to_pixel(spec: TileSpec, res: int, x_m: float, y_m: float) -> tuple[int, int]:
    """Map a tile-local metre coordinate to the ``(row, col)`` of the texture pixel containing it.

    Args:
        spec: Tile geometry (only the pitch is used).
        res: Texture edge length in pixels.
        x_m: Tile-local x in metres, in ``[-pitch/2, +pitch/2]``.
        y_m: Tile-local y in metres, in ``[-pitch/2, +pitch/2]``.

    Returns:
        ``(row, col)``, clipped to the texture.
    """
    half = spec.half_m
    col = int(np.clip(np.floor((x_m + half) / spec.pitch_m * res), 0, res - 1))
    row = int(np.clip(np.floor((half - y_m) / spec.pitch_m * res), 0, res - 1))
    return row, col


def tile_local_to_uv(spec: TileSpec, x_m: float, y_m: float) -> tuple[float, float]:
    """Map a tile-local metre coordinate to the USD ``st`` texture coordinate.

    Args:
        spec: Tile geometry (only the pitch is used).
        x_m: Tile-local x in metres.
        y_m: Tile-local y in metres.

    Returns:
        ``(u, v)`` with ``(0, 0)`` at the minimum-x, minimum-y tile corner.
    """
    half = spec.half_m
    return (x_m + half) / spec.pitch_m, (y_m + half) / spec.pitch_m


def rotated_uv_corners(rot: int) -> tuple[tuple[float, float], ...]:
    """UV coordinates for the four tile-quad vertices under a quarter-turn rotation.

    The quad vertices are always emitted in the world order ``(x0,y0) (x1,y0) (x1,y1) (x0,y1)``
    (south-west, south-east, north-east, north-west). Rotating the tile counter-clockwise by
    ``rot`` quarter turns rotates the painted texture with it, which is achieved by rotating this
    UV list.

    Args:
        rot: Quarter turns counter-clockwise, any integer (taken modulo 4).

    Returns:
        Four ``(u, v)`` pairs in south-west, south-east, north-east, north-west vertex order.
    """
    base = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    r = rot % 4
    return tuple(base[(i - r) % 4] for i in range(4))


# ------------------------------------------------------------------------------ rasterising
def _tile_grid(res: int, half: float) -> tuple[np.ndarray, np.ndarray]:
    """Build the tile-local metre coordinates of every texel centre.

    Args:
        res: Texture edge length in pixels.
        half: Half tile pitch in metres.

    Returns:
        ``(X, Y)``, each ``(res, res)``. ``X`` increases with the column index (east),
        ``Y`` decreases with the row index (row 0 is north).
    """
    t = (np.arange(res, dtype=_FLOAT) + 0.5) / res
    x = -half + t * (2.0 * half)
    y = half - t * (2.0 * half)
    return np.meshgrid(x, y)


def _paint(
    img: np.ndarray,
    signed_dist: np.ndarray,
    color: tuple[float, float, float],
    soft_m: float = 0.0,
    keep: np.ndarray | None = None,
) -> None:
    """Paint a flat colour where ``signed_dist <= 0``, in place.

    With ``soft_m == 0`` (the default, since the 2x supersample already anti-aliases) this is a
    boolean-indexed assignment touching only the covered texels, which is far cheaper than
    compositing a full-image alpha. A positive ``soft_m`` switches to alpha blending over a soft
    edge of that width.

    Args:
        img: ``(res, res, 3)`` float image, modified in place.
        signed_dist: Negative inside the shape, positive outside, in metres.
        color: sRGB triple in ``[0, 1]``.
        soft_m: Soft edge width in metres.
        keep: Optional boolean wear mask; texels where it is ``False`` are left untouched.
    """
    if soft_m <= 0.0:
        mask = signed_dist <= 0.0
        if keep is not None:
            mask = mask & keep
        if not mask.any():
            return
        img[mask] = color
        return
    alpha = np.clip(0.5 - signed_dist / soft_m, 0.0, 1.0)
    if keep is not None:
        alpha = alpha * keep
    a = alpha[..., None]
    img *= 1.0 - a
    img += a * np.asarray(color, dtype=_FLOAT)


def _band_sd(coord: np.ndarray, center: float, half_width: float) -> np.ndarray:
    """Signed distance to a slab of half-width ``half_width`` centred on ``center``."""
    return np.abs(coord - center) - half_width


def _interval_sd(coord: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Signed distance to the interval ``[lo, hi]`` along one axis."""
    return np.maximum(lo - coord, coord - hi)


def _rect_sd(X: np.ndarray, Y: np.ndarray, x0: float, x1: float, y0: float, y1: float) -> np.ndarray:
    """Signed distance (Chebyshev, exact outside the axis-aligned rectangle sides)."""
    return np.maximum(_interval_sd(X, x0, x1), _interval_sd(Y, y0, y1))


def _arc_sd(
    X: np.ndarray,
    Y: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    half_width: float,
    th0: float,
    sweep: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Signed distance to a circular band, plus the arc-length coordinate along it.

    Args:
        X: Tile-local x grid.
        Y: Tile-local y grid.
        cx: Arc centre x.
        cy: Arc centre y.
        radius: Arc radius.
        half_width: Half the band width.
        th0: Start angle in radians.
        sweep: Positive angular extent in radians.

    Returns:
        ``(signed_distance, arc_length)`` where ``arc_length`` is measured from ``th0``.
    """
    dx = X - cx
    dy = Y - cy
    r = np.hypot(dx, dy)
    radial = np.abs(r - radius) - half_width
    t = np.mod(np.arctan2(dy, dx) - th0, 2.0 * np.pi)
    over = np.minimum(t - sweep, 2.0 * np.pi - t)
    angular = np.where(t <= sweep, -np.inf, over * radius)
    return np.maximum(radial, angular), t * radius


def _dash_sd(arc_len: np.ndarray, period: float, mark: float, phase: float) -> np.ndarray:
    """Signed distance along a dashed pattern (negative inside a mark).

    Args:
        arc_len: Distance along the stripe, in metres.
        period: Mark + gap length, in metres.
        mark: Mark length, in metres.
        phase: Pattern phase as a fraction of the period, in ``[0, 1)``.

    Returns:
        Signed distance in metres.
    """
    q = np.mod(arc_len / period + phase, 1.0) * period
    return np.maximum(-q, q - mark)


def _box_blur_axis(a: np.ndarray, radius: int, axis: int) -> np.ndarray:
    """Moving-average blur along one axis with edge padding, in O(n) via cumulative sums.

    Args:
        a: 2-D float array.
        radius: Half window; the window is ``2 * radius + 1`` wide.
        axis: Axis to blur along.

    Returns:
        The blurred array, same shape as ``a``.
    """
    if radius < 1:
        return a
    moved = np.moveaxis(a, axis, 0)
    width = 2 * radius + 1
    padded = np.pad(moved, ((radius + 1, radius), (0, 0)), mode="edge")
    cumulative = np.cumsum(padded, axis=0)
    out = (cumulative[width:] - cumulative[:-width]) / width
    return np.moveaxis(out, 0, axis)


def _low_freq_field(rng: np.random.Generator, res: int, cells: int) -> np.ndarray:
    """A smooth random field in ``[0, 1]`` used for wear and mottling.

    Args:
        rng: Seeded generator.
        res: Output edge length.
        cells: Number of random cells per axis before smoothing.

    Returns:
        ``(res, res)`` float array, roughly uniform in ``[0, 1]``.
    """
    cells = max(2, min(cells, res))
    coarse = rng.random((cells, cells), dtype=_FLOAT)
    reps = int(np.ceil(res / cells))
    field = np.kron(coarse, np.ones((reps, reps)))[:res, :res]
    # Two box blurs of the cell width per axis give a smooth, separable, triangular kernel and
    # cost O(res^2) via cumulative sums, which matters because this runs at 2x supersampled size.
    radius = max(1, reps)
    for axis in (0, 1):
        for _ in range(2):
            field = _box_blur_axis(field, radius, axis)
    lo, hi = float(field.min()), float(field.max())
    return (field - lo) / (hi - lo + 1e-12)


# ---------------------------------------------------------------------------------- styling
class TileStyle:
    """Appearance knobs applied on top of the geometry in :class:`~.spec.TileSpec`.

    Attributes:
        palette: Colours to paint with.
        noise: Standard deviation of per-texel additive grain, in linear ``[0, 1]`` units.
        wear: Fraction of tape area scuffed away, in ``[0, 1)``. Implemented as a smooth random
            field thresholded at the ``wear`` quantile; the scuffed texels revert to the road
            colour, so it is the S7.2 "worn tape" appearance axis.
        mottle: Amplitude of low-frequency road-surface luminance variation.
    """

    def __init__(
        self,
        palette: ColorPalette = IDEAL_PALETTE,
        noise: float = 0.0,
        wear: float = 0.0,
        mottle: float = 0.0,
    ) -> None:
        """Initialise a style.

        Args:
            palette: Colours to paint with.
            noise: Per-texel grain standard deviation.
            wear: Tape scuff fraction in ``[0, 1)``.
            mottle: Low-frequency road luminance amplitude.

        Raises:
            ValueError: If ``wear`` is outside ``[0, 1)`` or ``noise``/``mottle`` are negative.
        """
        if not 0.0 <= wear < 1.0:
            raise ValueError(f"wear must be in [0, 1), got {wear}")
        if noise < 0.0 or mottle < 0.0:
            raise ValueError("noise and mottle must be non-negative")
        self.palette = palette
        self.noise = float(noise)
        self.wear = float(wear)
        self.mottle = float(mottle)


# ---------------------------------------------------------------------------------- painters
def _paint_surface(
    img: np.ndarray, rng: np.random.Generator, color: tuple[float, float, float], style: TileStyle
) -> None:
    """Fill ``img`` with a flat colour plus optional low-frequency mottling."""
    img[:] = np.asarray(color, dtype=_FLOAT)
    if style.mottle > 0.0:
        field = _low_freq_field(rng, img.shape[0], 8)
        img *= (1.0 + style.mottle * (field - 0.5) * 2.0)[..., None]


def _wear_mask(rng: np.random.Generator, res: int, wear: float) -> np.ndarray | None:
    """Return a boolean keep-mask for tape coverage, or ``None`` when ``wear`` is 0.

    Args:
        rng: Seeded generator.
        res: Texture edge length in pixels.
        wear: Fraction of tape area to scuff away, in ``[0, 1)``.

    Returns:
        ``(res, res)`` boolean array, ``False`` on scuffed texels, or ``None``.
    """
    if wear <= 0.0:
        return None
    field = _low_freq_field(rng, res, 24)
    threshold = float(np.quantile(field, wear))
    return field > threshold


def _paint_straight(
    img: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    spec: TileSpec,
    style: TileStyle,
    keep: np.ndarray | None,
    soft: float,
) -> None:
    """Paint the canonical straight tile (connects N and S, road along +/-y)."""
    half = spec.half_m
    off_w = spec.white_center_offset_m
    for sign in (-1.0, 1.0):
        sd = np.maximum(_band_sd(X, sign * off_w, spec.white_half_m), _interval_sd(Y, -half, half))
        _paint(img, sd, style.palette.white, soft, keep)
    dash_sd = _dash_sd(Y + half, spec.dash_period_m, spec.dash_mm / 1000.0, spec.dash_phase)
    sd = np.maximum(_band_sd(X, 0.0, spec.yellow_half_m), dash_sd)
    _paint(img, sd, style.palette.yellow, soft, keep)


def _paint_curve(
    img: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    spec: TileSpec,
    style: TileStyle,
    keep: np.ndarray | None,
    soft: float,
) -> None:
    """Paint the canonical curve tile (connects S and E, arc centre at the +x/-y corner)."""
    half = spec.half_m
    cx, cy = half, -half
    th0, sweep = np.pi / 2.0, np.pi / 2.0
    for radius in (half - spec.white_center_offset_m, half + spec.white_center_offset_m):
        sd, _ = _arc_sd(X, Y, cx, cy, radius, spec.white_half_m, th0, sweep)
        _paint(img, sd, style.palette.white, soft, keep)
    sd, arc = _arc_sd(X, Y, cx, cy, half, spec.yellow_half_m, th0, sweep)
    sd = np.maximum(sd, _dash_sd(arc, spec.dash_period_m, spec.dash_mm / 1000.0, spec.dash_phase))
    _paint(img, sd, style.palette.yellow, soft, keep)


def _paint_intersection(
    img: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    spec: TileSpec,
    style: TileStyle,
    keep: np.ndarray | None,
    soft: float,
    connections: frozenset[str],
) -> None:
    """Paint a 3-way or 4-way intersection tile.

    The layering follows the official assembly instructions: red tape is placed *underneath* the
    yellow and white tape, so it is painted first.

    Args:
        img: Float image being painted, modified in place.
        X: Tile-local x grid.
        Y: Tile-local y grid.
        spec: Tile geometry.
        style: Appearance.
        keep: Optional wear keep-mask.
        soft: Soft edge width in metres.
        connections: Which tile edges are open roads.
    """
    half = spec.half_m
    off_w = spec.white_center_offset_m
    lane = spec.lane_center_offset_m
    red_w = spec.red_tape_mm / 1000.0
    red_len = spec.red_tape_len_mm / 1000.0
    inset = spec.stop_line_inset_mm / 1000.0
    stub = spec.yellow_stub_mm / 1000.0
    corner = spec.corner_white_mm / 1000.0

    basis: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for edge in EDGES:
        (nx, ny), (ux, uy) = EDGE_BASIS[edge]
        basis[edge] = (X * nx + Y * ny, X * ux + Y * uy)

    # 1. red stop bars on every incoming lane
    for edge in sorted(connections):
        a, b = basis[edge]
        sd = np.maximum(
            _interval_sd(a, half - inset - red_w, half - inset),
            _interval_sd(b, lane - red_len / 2.0, lane + red_len / 2.0),
        )
        _paint(img, sd, style.palette.red, soft, keep)

    # 2. white: full edge line on closed edges, corner brackets on open edges.
    #    A corner arm may never intrude into the crossing box, so its length is clamped to
    #    ``half - off_w``; two adjacent arms then meet as an L bracket in the tile corner
    #    ("add white tape at the corners", official 4-way assembly instructions).
    arm = min(corner, half - off_w)
    for edge in EDGES:
        a, b = basis[edge]
        if edge in connections:
            for sign in (-1.0, 1.0):
                sd = np.maximum(
                    _band_sd(b, sign * off_w, spec.white_half_m),
                    _interval_sd(a, half - arm, half),
                )
                _paint(img, sd, style.palette.white, soft, keep)
        else:
            sd = np.maximum(_band_sd(a, off_w, spec.white_half_m), _interval_sd(b, -half, half))
            _paint(img, sd, style.palette.white, soft, keep)

    # 3. yellow centre stubs on every open edge
    for edge in sorted(connections):
        a, b = basis[edge]
        sd = np.maximum(_band_sd(b, 0.0, spec.yellow_half_m), _interval_sd(a, half - stub, half))
        _paint(img, sd, style.palette.yellow, soft, keep)


def _paint_grass(img: np.ndarray, rng: np.random.Generator, style: TileStyle) -> None:
    """Paint the grass surround: base green plus two scales of mottling."""
    res = img.shape[0]
    img[:] = np.asarray(style.palette.grass, dtype=_FLOAT)
    coarse = _low_freq_field(rng, res, 6)
    fine = _low_freq_field(rng, res, 48)
    modulation = 1.0 + 0.18 * (coarse - 0.5) * 2.0 + 0.10 * (fine - 0.5) * 2.0
    img *= modulation[..., None]


def render_tile(
    kind: str,
    spec: TileSpec = NOMINAL_TILE_SPEC,
    style: TileStyle | None = None,
    res: int | None = None,
    supersample: int = 2,
    seed: int = 0,
) -> np.ndarray:
    """Render one tile texture.

    Args:
        kind: One of :data:`TILE_KINDS` (``"floor"`` is accepted as an alias of ``"empty"``).
        spec: Millimetre geometry to paint.
        style: Appearance; defaults to the idealised palette with no noise or wear.
        res: Output edge length in pixels; defaults to :data:`DEFAULT_TEXTURE_RES` for the kind.
        supersample: Integer supersampling factor. The texture is painted at ``res * supersample``
            and box-averaged down, which is the same energy-preserving downsample the observation
            pipeline uses (SPEC v2 S4.3 step 6).
        seed: Seed for every stochastic element (grain, wear, mottle). Rendering is a pure
            function of ``(kind, spec, style, res, supersample, seed)``.

    Returns:
        ``(res, res, 3)`` array of ``uint8``.

    Raises:
        ValueError: If ``kind`` is unknown, or ``res``/``supersample`` are not positive.
    """
    kind = "empty" if kind == "floor" else kind
    if kind not in KIND_CONNECTIONS:
        raise ValueError(f"unknown tile kind {kind!r}; expected one of {sorted(KIND_CONNECTIONS)}")
    if supersample < 1:
        raise ValueError(f"supersample must be >= 1, got {supersample}")
    res = DEFAULT_TEXTURE_RES[kind] if res is None else int(res)
    if res < 1:
        raise ValueError(f"res must be >= 1, got {res}")
    spec.validate()
    style = style or TileStyle()

    big = res * supersample
    rng = np.random.default_rng(seed)
    img = np.empty((big, big, 3), dtype=_FLOAT)
    soft = spec.edge_soft_mm / 1000.0

    if kind == "grass":
        _paint_grass(img, rng, style)
    elif kind in ("asphalt", "empty"):
        _paint_surface(img, rng, style.palette.asphalt, style)
    else:
        _paint_surface(img, rng, style.palette.road, style)
        X, Y = _tile_grid(big, spec.half_m)
        keep = _wear_mask(rng, big, style.wear)
        if kind == "straight":
            _paint_straight(img, X, Y, spec, style, keep, soft)
        elif kind == "curve":
            _paint_curve(img, X, Y, spec, style, keep, soft)
        else:
            _paint_intersection(img, X, Y, spec, style, keep, soft, KIND_CONNECTIONS[kind])

    if style.noise > 0.0:
        img += rng.normal(0.0, style.noise, size=(big, big, 1))
    np.clip(img, 0.0, 1.0, out=img)

    if supersample > 1:
        img = img.reshape(res, supersample, res, supersample, 3).mean(axis=(1, 3))
    return np.rint(img * 255.0).astype(np.uint8)


#: Procedural sign faces available for the roadside distractor cards.
SIGN_KINDS: Final[tuple[str, ...]] = ("stop", "yield", "priority", "turn", "blank")


def _polygon_sd(X: np.ndarray, Y: np.ndarray, sides: int, circumradius: float, rotation: float) -> np.ndarray:
    """Signed distance to a regular polygon centred on the origin.

    Args:
        X: x grid.
        Y: y grid.
        sides: Number of sides.
        circumradius: Distance from the centre to a vertex.
        rotation: Rotation of the first edge normal, in radians.

    Returns:
        Signed distance, negative inside.
    """
    apothem = circumradius * math.cos(math.pi / sides)
    sd = np.full_like(X, -np.inf)
    for k in range(sides):
        angle = rotation + 2.0 * math.pi * k / sides
        sd = np.maximum(sd, X * math.cos(angle) + Y * math.sin(angle) - apothem)
    return sd


def render_sign(
    kind: str = "stop",
    res: int = 128,
    aspect: float = SIGN_CARD_ASPECT,
    palette: ColorPalette = IDEAL_PALETTE,
    supersample: int = 2,
    seed: int = 0,
) -> np.ndarray:
    """Render a procedural traffic-sign card face.

    SPEC v2 S3.3 reclassifies signs and tags as *visual distractors only*: at the observation
    resolution a 65 mm feature spans 2-4 px and is not decodable (S4.2). These faces are
    therefore plausible rather than standards-accurate, and no AprilTag imagery is vendored.

    Args:
        kind: One of :data:`SIGN_KINDS`.
        res: Output height in pixels; the width follows ``aspect``.
        aspect: Card width divided by card height (85 / 155 for the Duckietown sign card).
        palette: Colours to draw with.
        supersample: Integer supersampling factor.
        seed: Seed for the paper grain.

    Returns:
        ``(res, round(res * aspect), 3)`` array of ``uint8``.

    Raises:
        ValueError: If ``kind`` is unknown.
    """
    if kind not in SIGN_KINDS:
        raise ValueError(f"unknown sign kind {kind!r}; expected one of {SIGN_KINDS}")
    height = res * supersample
    width = max(1, round(res * aspect)) * supersample
    rng = np.random.default_rng(seed)
    img = np.tile(np.array(palette.white, dtype=_FLOAT), (height, width, 1))

    # Coordinates normalised so the sign face is the unit square in the upper part of the card.
    u = (np.arange(width) + 0.5) / width * 2.0 - 1.0
    v = 1.0 - (np.arange(height) + 0.5) / height * 2.0
    X, Y = np.meshgrid(u, v)
    face_scale = aspect
    Yf = (Y - (1.0 - face_scale)) / face_scale

    dark = (0.08, 0.08, 0.08)
    if kind == "stop":
        _paint(img, _polygon_sd(X, Yf, 8, 0.85, math.pi / 8), palette.red)
        _paint(img, _polygon_sd(X, Yf, 8, 0.62, math.pi / 8), palette.white)
        _paint(img, _rect_sd(X, Yf, -0.42, 0.42, -0.13, 0.13), palette.red)
    elif kind == "yield":
        _paint(img, _polygon_sd(X, -Yf, 3, 0.95, math.pi / 2), palette.red)
        _paint(img, _polygon_sd(X, -Yf, 3, 0.66, math.pi / 2), palette.white)
    elif kind == "priority":
        _paint(img, _polygon_sd(X, Yf, 4, 0.92, math.pi / 4), palette.yellow)
        _paint(img, _polygon_sd(X, Yf, 4, 0.60, math.pi / 4), palette.white)
    elif kind == "turn":
        radial = np.hypot(X, Yf)
        _paint(img, radial - 0.88, (0.10, 0.20, 0.75))
        _paint(img, _rect_sd(X, Yf, -0.50, 0.50, -0.14, 0.14), palette.white)
        _paint(img, _rect_sd(X, Yf, 0.10, 0.50, -0.45, 0.45), palette.white)
    _paint(img, -_rect_sd(X, Y, -0.94, 0.94, -0.96, 0.96), dark)
    img += rng.normal(0.0, 0.01, size=(height, width, 1))
    np.clip(img, 0.0, 1.0, out=img)
    if supersample > 1:
        img = img.reshape(res, supersample, width // supersample, supersample, 3).mean(axis=(1, 3))
    return np.rint(img * 255.0).astype(np.uint8)


def save_sign_set(
    outdir: str | Path,
    res: int = 128,
    palette: ColorPalette = IDEAL_PALETTE,
    supersample: int = 2,
    seed: int = 0,
    prefix: str = "",
) -> dict[str, Path]:
    """Render and write one PNG per entry of :data:`SIGN_KINDS`.

    Args:
        outdir: Directory to write into.
        res: Card height in pixels.
        palette: Colours to draw with.
        supersample: Supersampling factor.
        seed: Base seed.
        prefix: Filename prefix.

    Returns:
        Mapping from sign kind to the written path.
    """
    out = Path(outdir)
    return {
        kind: write_png(
            out / f"{prefix}sign_{kind}.png",
            render_sign(kind, res=res, palette=palette, supersample=supersample, seed=seed + i),
        )
        for i, kind in enumerate(SIGN_KINDS)
    }


def render_tile_set(
    spec: TileSpec = NOMINAL_TILE_SPEC,
    style: TileStyle | None = None,
    kinds: tuple[str, ...] = TILE_KINDS,
    res: dict[str, int] | None = None,
    supersample: int = 2,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Render every tile kind with one shared geometry and style.

    Args:
        spec: Millimetre geometry.
        style: Appearance.
        kinds: Which kinds to render.
        res: Optional per-kind output resolution overriding :data:`DEFAULT_TEXTURE_RES`.
        supersample: Supersampling factor.
        seed: Base seed; each kind gets ``seed + index`` so the kinds decorrelate.

    Returns:
        Mapping from kind to ``(res, res, 3)`` ``uint8`` texture.
    """
    res = res or {}
    return {
        kind: render_tile(
            kind, spec=spec, style=style, res=res.get(kind), supersample=supersample, seed=seed + i
        )
        for i, kind in enumerate(kinds)
    }


def save_tile_set(
    outdir: str | Path,
    spec: TileSpec = NOMINAL_TILE_SPEC,
    style: TileStyle | None = None,
    kinds: tuple[str, ...] = TILE_KINDS,
    res: dict[str, int] | None = None,
    supersample: int = 2,
    seed: int = 0,
    prefix: str = "",
) -> dict[str, Path]:
    """Render and write a full tile set as PNGs.

    Args:
        outdir: Directory to write into; created if missing.
        spec: Millimetre geometry.
        style: Appearance.
        kinds: Which kinds to render.
        res: Optional per-kind output resolution.
        supersample: Supersampling factor.
        seed: Base seed.
        prefix: Filename prefix, e.g. ``"bucket00_"``.

    Returns:
        Mapping from kind to the written path.
    """
    textures = render_tile_set(
        spec=spec, style=style, kinds=kinds, res=res, supersample=supersample, seed=seed
    )
    out = Path(outdir)
    return {kind: write_png(out / f"{prefix}{kind}.png", tex) for kind, tex in textures.items()}
