"""Unit tests for the procedural tile textures and the dimensional spec they are painted from.

The point of these tests is that a texel measured in the PNG lands where the millimetre spec says
it should. Every geometric assertion is written in millimetres and converted to pixels once, so a
change to :mod:`duckiebot_rl.city.spec` that silently moves a marking fails here.

The tests deliberately render with ``supersample=1`` and a zero-noise style so that the painted
colours are exact and edges are single texels; the production path uses ``supersample=2``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from duckiebot_rl.city import spec as S
from duckiebot_rl.city import tiles as T

RES = 512
SHARP = {"res": RES, "supersample": 1}


@pytest.fixture(scope="module")
def nominal() -> S.TileSpec:
    """The nominal tile geometry."""
    return S.NOMINAL_TILE_SPEC


@pytest.fixture(scope="module")
def straight() -> np.ndarray:
    """A sharply rendered canonical straight tile."""
    return T.render_tile("straight", **SHARP)


@pytest.fixture(scope="module")
def curve() -> np.ndarray:
    """A sharply rendered canonical curve tile."""
    return T.render_tile("curve", **SHARP)


def px_per_m(spec: S.TileSpec, res: int = RES) -> float:
    """Texels per metre of a ``res``-pixel texture of this tile."""
    return res / spec.pitch_m


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous ``True`` runs of a 1-D boolean array as inclusive ``(start, end)`` pairs.

    Args:
        mask: 1-D boolean array.

    Returns:
        The runs, in order.
    """
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([idx[0]], idx[breaks + 1]))
    ends = np.concatenate((idx[breaks], [idx[-1]]))
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def is_color(img: np.ndarray, rgb: tuple[float, float, float], tol: int = 8) -> np.ndarray:
    """Boolean mask of texels within ``tol`` of an sRGB colour given in ``[0, 1]``."""
    target = np.rint(np.asarray(rgb) * 255.0).astype(np.int16)
    return (np.abs(img.astype(np.int16) - target) <= tol).all(axis=-1)


# ------------------------------------------------------------------------------ spec sanity
def test_nominal_spec_is_self_consistent(nominal: S.TileSpec) -> None:
    """The nominal spec validates and its derived quantities match the S3.3 numbers."""
    nominal.validate()
    assert nominal.pitch_m == pytest.approx(0.585)
    assert nominal.lane_center_offset_m == pytest.approx(0.117)
    assert nominal.lane_center_offset_tile == pytest.approx(0.20)
    assert nominal.white_center_offset_m == pytest.approx(0.246)
    assert nominal.white_inner_offset_m == pytest.approx(0.222)
    assert nominal.curve_radius_inner_m == pytest.approx(0.1755)
    assert nominal.curve_radius_outer_m == pytest.approx(0.4095)
    assert nominal.curve_radius_inner_m / nominal.pitch_m == pytest.approx(0.30)
    assert nominal.curve_radius_outer_m / nominal.pitch_m == pytest.approx(0.70)
    assert nominal.dash_period_m == pytest.approx(0.075)


def test_lane_center_override_reproduces_duckietown_world(nominal: S.TileSpec) -> None:
    """Forcing the 0.22 convention gives the duckietown-world offset and radii."""
    forced = S.TileSpec(lane_center_offset_mm=S.DUCKIETOWN_WORLD_LANE_OFFSET_TILE * 585.0)
    assert forced.lane_center_offset_m == pytest.approx(0.1287)
    assert forced.lane_center_offset_tile == pytest.approx(0.22)
    assert forced.curve_radius_inner_m / forced.pitch_m == pytest.approx(0.28)
    assert forced.curve_radius_outer_m / forced.pitch_m == pytest.approx(0.72)
    # ... and differs from the derived convention by the documented 11.7 mm.
    assert abs(forced.lane_center_offset_m - nominal.lane_center_offset_m) == pytest.approx(0.0117)


def test_rescale_preserves_tile_relative_geometry(nominal: S.TileSpec) -> None:
    """Rescaling to another pitch keeps every marking at the same fraction of the tile."""
    scaled = nominal.rescaled(0.615)
    assert scaled.pitch_m == pytest.approx(0.615)
    assert scaled.lane_center_offset_tile == pytest.approx(nominal.lane_center_offset_tile)
    assert scaled.white_center_offset_m / scaled.pitch_m == pytest.approx(
        nominal.white_center_offset_m / nominal.pitch_m
    )
    assert nominal.rescaled(0.615).rescaled(0.585).tile_pitch_mm == pytest.approx(585.0)


def test_spec_validation_rejects_markings_that_do_not_fit() -> None:
    """Markings wider than the tile, and degenerate values, are rejected."""
    with pytest.raises(ValueError, match="do not fit"):
        S.TileSpec(clear_lane_mm=400.0).validate()
    with pytest.raises(ValueError, match="must be > 0"):
        S.TileSpec(white_tape_mm=0.0).validate()
    with pytest.raises(ValueError, match="dash_phase"):
        S.TileSpec(dash_phase=1.0).validate()


def test_geometry_buckets_are_deterministic_and_valid() -> None:
    """The 16 buckets are reproducible, distinct, valid, and bucket 0 is nominal."""
    first = S.geometry_buckets(16, seed=3)
    second = S.geometry_buckets(16, seed=3)
    assert first == second
    assert first[0] == S.NOMINAL_TILE_SPEC
    assert len({b.tile_pitch_mm for b in first}) == 16
    for bucket in first:
        bucket.validate()
    assert S.geometry_buckets(16, seed=4) != first
    # alpha = 0 collapses the ranges onto the nominal spec.
    assert all(b == S.NOMINAL_TILE_SPEC for b in S.geometry_buckets(4, seed=3, alpha=0.0))


def test_sampled_clear_lane_is_clamped_to_the_feasible_interval() -> None:
    """The SPEC V9 upper bound of 280 mm does not fit any tile; sampling clamps instead."""
    ranges = S.TileSpecRanges()
    rng = np.random.default_rng(12345)
    for _ in range(500):
        sampled = ranges.sample(rng)
        sampled.validate()
        cap = ranges.feasible_clear_lane_mm(
            sampled.tile_pitch_mm, sampled.yellow_tape_mm, sampled.white_tape_mm
        )
        assert sampled.clear_lane_mm <= cap + 1e-9


# --------------------------------------------------------------------------------- textures
def test_render_is_a_pure_function_of_its_arguments() -> None:
    """Identical arguments give byte-identical textures; a different seed does not."""
    a = T.render_tile("straight", seed=7, style=T.TileStyle(noise=0.02, wear=0.2))
    b = T.render_tile("straight", seed=7, style=T.TileStyle(noise=0.02, wear=0.2))
    c = T.render_tile("straight", seed=8, style=T.TileStyle(noise=0.02, wear=0.2))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_every_kind_renders_at_the_documented_size() -> None:
    """All seven kinds render, drivable at 512 and non-drivable at 256."""
    textures = T.render_tile_set()
    assert set(textures) == set(T.TILE_KINDS)
    for kind, tex in textures.items():
        expected = 512 if kind in T.DRIVABLE_KINDS else 256
        assert tex.shape == (expected, expected, 3)
        assert tex.dtype == np.uint8
    assert np.array_equal(T.render_tile("floor"), T.render_tile("empty"))


def test_unknown_kind_and_bad_arguments_raise() -> None:
    """The painter refuses unknown kinds and non-positive sizes."""
    with pytest.raises(ValueError, match="unknown tile kind"):
        T.render_tile("roundabout")
    with pytest.raises(ValueError, match="supersample"):
        T.render_tile("straight", supersample=0)
    with pytest.raises(ValueError, match="wear"):
        T.TileStyle(wear=1.0)


def test_straight_tile_white_tape_matches_the_millimetre_spec(
    straight: np.ndarray, nominal: S.TileSpec
) -> None:
    """Both white edge lines sit at +/-246 mm and are 48 mm wide, to within one texel."""
    scale = px_per_m(nominal)
    row = straight[RES // 2]
    white_runs = runs(is_color(row[None, :], nominal_white := (1.0, 1.0, 1.0))[0])
    del nominal_white
    assert len(white_runs) == 2
    for (start, end), expected_center_m in zip(white_runs, (-0.246, +0.246), strict=True):
        width_px = end - start + 1
        assert width_px == pytest.approx(nominal.white_tape_mm / 1000.0 * scale, abs=1.0)
        center_px = (start + end + 1) / 2.0
        center_m = center_px / RES * nominal.pitch_m - nominal.half_m
        assert center_m == pytest.approx(expected_center_m, abs=1.5 / scale)


def test_straight_tile_yellow_dashes_match_the_millimetre_spec(
    straight: np.ndarray, nominal: S.TileSpec
) -> None:
    """The centre line is 24 mm wide with a 50 / 25 mm dash pattern on the tile centreline."""
    scale = px_per_m(nominal)
    yellow = is_color(straight, (1.0, 1.0, 0.0))

    # width, measured across a row that lands inside a dash
    row_index = int(yellow.sum(axis=1).argmax())
    width_runs = runs(yellow[row_index])
    assert len(width_runs) == 1
    start, end = width_runs[0]
    assert end - start + 1 == pytest.approx(nominal.yellow_tape_mm / 1000.0 * scale, abs=1.0)
    center_m = (start + end + 1) / 2.0 / RES * nominal.pitch_m - nominal.half_m
    assert center_m == pytest.approx(0.0, abs=1.5 / scale)

    # dash and gap lengths along the road direction
    column = yellow[:, RES // 2]
    marks = runs(column)
    assert len(marks) >= 6
    interior = marks[1:-1]
    for mark_start, mark_end in interior:
        assert mark_end - mark_start + 1 == pytest.approx(nominal.dash_mm / 1000.0 * scale, abs=1.5)
    starts = np.array([m[0] for m in marks])
    assert np.diff(starts).mean() == pytest.approx(nominal.dash_period_m * scale, abs=1.0)


def test_straight_tile_colors_at_sampled_points(straight: np.ndarray, nominal: S.TileSpec) -> None:
    """Named points sample the colour the spec says they should."""

    def at(x_m: float, y_m: float) -> tuple[int, int, int]:
        row, col = T.tile_local_to_pixel(nominal, RES, x_m, y_m)
        return tuple(int(v) for v in straight[row, col])

    assert at(0.246, 0.0) == (255, 255, 255)
    assert at(-0.246, 0.0) == (255, 255, 255)
    assert at(0.10, 0.0) == (0, 0, 0)
    assert at(0.285, 0.0) == (0, 0, 0)  # asphalt shoulder outside the white tape
    # first dash starts at the southern tile edge with phase 0
    assert at(0.0, -nominal.half_m + 0.01) == (255, 255, 0)
    assert at(0.0, -nominal.half_m + 0.06) == (0, 0, 0)  # inside the 25 mm gap


def test_curve_tile_arcs_are_centred_on_the_tile_corner(curve: np.ndarray, nominal: S.TileSpec) -> None:
    """Every marking of a curve tile lies at its analytic radius from the +x/-y corner."""
    scale = px_per_m(nominal)
    corner = np.array([nominal.half_m, -nominal.half_m])
    cols = (np.arange(RES) + 0.5) / RES * nominal.pitch_m - nominal.half_m
    rows = nominal.half_m - (np.arange(RES) + 0.5) / RES * nominal.pitch_m
    grid_x, grid_y = np.meshgrid(cols, rows)
    radius = np.hypot(grid_x - corner[0], grid_y - corner[1])

    for mask, expected_r, expected_w in (
        (is_color(curve, (1.0, 1.0, 1.0)), nominal.half_m - nominal.white_center_offset_m, 0.048),
        (is_color(curve, (1.0, 1.0, 0.0)), nominal.half_m, 0.024),
    ):
        near = mask & (radius < nominal.half_m + 0.05)
        assert near.any()
        sampled = radius[near]
        # The band midpoint, not the mean: an annulus has more area at larger radius, so the
        # mean radius of a painted band is biased outward by w^2 / (12 r).
        assert (sampled.min() + sampled.max()) / 2.0 == pytest.approx(expected_r, abs=2.0 / scale)
        assert sampled.max() - sampled.min() == pytest.approx(expected_w, abs=3.0 / scale)

    outer = is_color(curve, (1.0, 1.0, 1.0)) & (radius > nominal.half_m + 0.05)
    outer_radius = radius[outer]
    assert (outer_radius.min() + outer_radius.max()) / 2.0 == pytest.approx(
        nominal.half_m + nominal.white_center_offset_m, abs=2.0 / scale
    )


def test_curve_tile_joins_a_straight_tile_flush(curve: np.ndarray, straight: np.ndarray) -> None:
    """The curve's south edge carries the same markings as a straight tile's south edge."""
    curve_edge = curve[-1]
    straight_edge = straight[-1]
    for color in ((1.0, 1.0, 1.0), (1.0, 1.0, 0.0)):
        curve_runs = runs(is_color(curve_edge[None, :], color)[0])
        straight_runs = runs(is_color(straight_edge[None, :], color)[0])
        assert curve_runs == straight_runs, color


def test_intersection_stop_bars_sit_on_the_incoming_lane(nominal: S.TileSpec) -> None:
    """The 48 x 210 mm red bar covers exactly the clear lane a robot enters the tile on."""
    tile = T.render_tile("fourway", **SHARP)
    scale = px_per_m(nominal)
    red = is_color(tile, (1.0, 0.0, 0.0))
    # Northern edge: a robot entering heads south, so its lane centre is at x = -117 mm.
    row, _ = T.tile_local_to_pixel(nominal, RES, 0.0, nominal.half_m - 0.030)
    bar = runs(red[row])
    assert len(bar) == 1
    start, end = bar[0]
    span_m = np.array([start, end + 1]) / RES * nominal.pitch_m - nominal.half_m
    assert span_m[1] - span_m[0] == pytest.approx(nominal.red_tape_len_mm / 1000.0, abs=2.0 / scale)
    assert span_m.mean() == pytest.approx(-nominal.lane_center_offset_m, abs=2.0 / scale)
    # thickness along the road direction
    col = int((start + end) / 2)
    thickness = runs(red[:, col])
    assert len(thickness) == 1
    assert thickness[0][1] - thickness[0][0] + 1 == pytest.approx(
        nominal.red_tape_mm / 1000.0 * scale, abs=1.5
    )


def test_threeway_closes_its_fourth_edge_with_a_solid_white_line(nominal: S.TileSpec) -> None:
    """The canonical 3-way has no road through its east edge, so a white line crosses it."""
    tile = T.render_tile("threeway", **SHARP)
    white = is_color(tile, (1.0, 1.0, 1.0))
    col, _ = T.tile_local_to_pixel(nominal, RES, nominal.white_center_offset_m, 0.0)[::-1]
    assert white[:, col].mean() > 0.95
    mirrored, _ = T.tile_local_to_pixel(nominal, RES, -nominal.white_center_offset_m, 0.0)[::-1]
    assert white[:, mirrored].mean() < 0.5


def test_texture_geometry_tracks_the_spec(nominal: S.TileSpec) -> None:
    """Widening the clear lane moves the white tape outward in the rendered PNG."""
    wide = S.TileSpec(clear_lane_mm=170.0)
    tile = T.render_tile("straight", spec=wide, **SHARP)
    row = tile[RES // 2]
    white_runs = runs(is_color(row[None, :], (1.0, 1.0, 1.0))[0])
    assert len(white_runs) == 2
    center_px = (white_runs[1][0] + white_runs[1][1] + 1) / 2.0
    center_m = center_px / RES * wide.pitch_m - wide.half_m
    assert center_m == pytest.approx(wide.white_center_offset_m, abs=2.0 / px_per_m(wide))
    assert wide.white_center_offset_m < nominal.white_center_offset_m


# -------------------------------------------------------------------------------- PNG codec
def test_png_round_trip(tmp_path: Path) -> None:
    """The dependency-free writer and reader are inverses on every kind."""
    for kind in T.TILE_KINDS:
        tex = T.render_tile(kind, res=64, supersample=1, seed=2)
        path = T.write_png(tmp_path / f"{kind}.png", tex)
        assert np.array_equal(T.read_png(path), tex)


def test_png_is_readable_by_pillow(tmp_path: Path) -> None:
    """The PNGs are standard files, not just readable by our own reader."""
    pil = pytest.importorskip("PIL.Image", reason="Pillow is not installed in this venv")
    tex = T.render_tile("fourway", res=96, supersample=1, seed=1)
    path = T.write_png(tmp_path / "fourway.png", tex)
    with pil.open(path) as handle:
        assert np.array_equal(np.asarray(handle.convert("RGB")), tex)


def test_png_writer_rejects_bad_arrays(tmp_path: Path) -> None:
    """Only (H, W, 3) uint8 arrays are accepted."""
    with pytest.raises(ValueError, match="uint8"):
        T.write_png(tmp_path / "bad.png", np.zeros((4, 4, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="uint8"):
        T.write_png(tmp_path / "bad.png", np.zeros((4, 4), dtype=np.uint8))


def test_read_png_rejects_non_png(tmp_path: Path) -> None:
    """A non-PNG file gives a clear error."""
    path = tmp_path / "not.png"
    path.write_bytes(b"nope")
    with pytest.raises(ValueError, match="not a PNG"):
        T.read_png(path)


# ------------------------------------------------------------------------------ UV rotation
def test_rotated_uv_corners_form_a_cyclic_group() -> None:
    """Four quarter turns is the identity, and each turn shifts the corner list by one."""
    base = T.rotated_uv_corners(0)
    assert base == ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    assert T.rotated_uv_corners(4) == base
    assert T.rotated_uv_corners(-1) == T.rotated_uv_corners(3)
    for rot in range(4):
        corners = T.rotated_uv_corners(rot)
        assert set(corners) == set(base)
        assert corners[0] == base[(-rot) % 4]


def test_rotated_uv_matches_an_inverse_rotation_of_the_tile(nominal: S.TileSpec) -> None:
    """Rotated UVs equal an inverse rotation of the tile-local point.

    Sampling a rotated tile at a world point must equal sampling the canonical tile at the
    inverse-rotated tile-local point, which is what the UV cycling has to implement.
    """
    half = nominal.half_m
    quad = ((-half, -half), (half, -half), (half, half), (-half, half))
    for rot in range(4):
        corners = T.rotated_uv_corners(rot)
        angle = -rot * np.pi / 2.0
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        for (vx, vy), (u, v) in zip(quad, corners, strict=True):
            qx = cos_a * vx - sin_a * vy
            qy = sin_a * vx + cos_a * vy
            expected = T.tile_local_to_uv(nominal, qx, qy)
            assert (u, v) == pytest.approx(expected, abs=1e-9)


def test_tile_local_to_pixel_and_uv_agree_on_the_axis_convention(nominal: S.TileSpec) -> None:
    """Row 0 is maximum y, column 0 is minimum x, and v = 0 is minimum y."""
    half = nominal.half_m
    assert T.tile_local_to_pixel(nominal, RES, -half + 1e-6, half - 1e-6) == (0, 0)
    assert T.tile_local_to_pixel(nominal, RES, half - 1e-6, -half + 1e-6) == (RES - 1, RES - 1)
    assert T.tile_local_to_uv(nominal, -half, -half) == pytest.approx((0.0, 0.0))
    assert T.tile_local_to_uv(nominal, half, half) == pytest.approx((1.0, 1.0))


# ------------------------------------------------------------------------------------ signs
def test_signs_render_deterministically_at_the_card_aspect() -> None:
    """Sign faces have the 85:155 card aspect and are reproducible."""
    for kind in T.SIGN_KINDS:
        card = T.render_sign(kind, res=155, seed=1)
        assert card.shape == (155, round(155 * T.SIGN_CARD_ASPECT), 3)
        assert np.array_equal(card, T.render_sign(kind, res=155, seed=1))
    assert not np.array_equal(T.render_sign("stop"), T.render_sign("yield"))
    with pytest.raises(ValueError, match="unknown sign kind"):
        T.render_sign("roundabout")


def test_save_sets_write_every_file(tmp_path: Path) -> None:
    """The batch writers produce one PNG per kind with the expected names."""
    tiles_written = T.save_tile_set(tmp_path / "tiles", res=dict.fromkeys(T.TILE_KINDS, 32))
    assert set(tiles_written) == set(T.TILE_KINDS)
    assert all(path.is_file() for path in tiles_written.values())
    signs_written = T.save_sign_set(tmp_path / "signs", res=32)
    assert set(signs_written) == set(T.SIGN_KINDS)
    assert all(path.is_file() for path in signs_written.values())
