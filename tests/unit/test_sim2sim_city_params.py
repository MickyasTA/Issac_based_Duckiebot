"""The sim2sim track geometry must be the geometry [city] paints (SPEC v2 S3.3, owner ``[sim2sim]``).

Interpreter: these tests need **only** numpy, so they run in the Isaac venv, in the tools venv and
in CI. There is deliberately no ``mujoco`` import here: the defect they guard is a *dimensional*
disagreement between two modules, and it is measurable straight out of the texture generator.

The defect: :class:`duckiebot_rl.sim2sim._resolve.CityParams` used to hardcode a lane-centre offset
of 0.22 tile, the duckietown-world figure, while ``duckiebot_rl.city.spec`` (the source of truth)
paints the clear lane centred at 0.20 tile. The lane graph therefore placed ``d = 0`` 11.7 mm away
from the line the camera can see, and ``lane_rms_m`` and ``lane_max_m``, the two headline S8.4
metrics, carried that bias into every C1-vs-C5 comparison. The first test below is the real guard:
it measures the painted lane centre out of the rendered pixels and asserts the lane graph agrees to
within 1 mm.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from duckiebot_rl.city.spec import (  # noqa: E402
    DUCKIETOWN_WORLD_LANE_OFFSET_TILE,
    NOMINAL_TILE_SPEC,
    TileSpec,
)
from duckiebot_rl.city.tiles import render_tile  # noqa: E402
from duckiebot_rl.sim2sim._resolve import (  # noqa: E402
    CityParams,
    SpecFallbackWarning,
    resolve_city_params,
)
from duckiebot_rl.sim2sim.track import LOOP_5X5, LaneGraph, MapSpec, load_map  # noqa: E402

#: Texture resolution used for the pixel measurement. At the nominal 585 mm pitch one texel is
#: 0.571 mm, so a half-texel quantization error is 0.29 mm, comfortably inside the 1 mm gate.
MEASURE_RES = 1024


def _painted_lane_center_m(spec: TileSpec, res: int = MEASURE_RES) -> float:
    """Measure the centre of the painted right-hand clear lane out of a straight tile texture.

    The canonical straight tile runs north-south with the yellow tape on the tile centreline, so
    the clear lane on the +x side of every scanline is bounded by the right edge of the yellow tape
    and the left edge of the right-hand white tape. Rows falling in a dash *gap* carry no yellow
    and are skipped, which is why the measurement is taken over the rows that do.

    Args:
        spec: the tile geometry to render.
        res: texture edge length in pixels.

    Returns:
        The mean centre of the painted clear lane, in metres from the tile centreline.
    """
    image = render_tile("straight", spec, res=res).astype(np.int16)
    red, green, blue = image[..., 0], image[..., 1], image[..., 2]
    yellow = (red > 200) & (green > 200) & (blue < 80)
    white = (red > 200) & (green > 200) & (blue > 200)

    centers: list[float] = []
    for row in range(res):
        yellow_cols = np.nonzero(yellow[row])[0]
        if yellow_cols.size == 0:
            continue
        right_of_yellow = int(yellow_cols.max())
        white_cols = np.nonzero(white[row])[0]
        white_cols = white_cols[white_cols > right_of_yellow]
        if white_cols.size == 0:
            continue
        left_of_white = int(white_cols.min())
        # Centroid of the clear run, converted from a column index to a tile-local metre.
        column = 0.5 * (right_of_yellow + left_of_white) + 0.5
        centers.append(column / res * spec.pitch_m - spec.half_m)
    assert len(centers) > res // 4, f"only {len(centers)} scanlines carried yellow tape"
    return float(np.mean(centers))


def test_lane_graph_zero_lies_on_the_centre_of_the_painted_lane() -> None:
    """``d = 0`` is the centre of the clear lane the texture paints, to within 1 mm.

    This is the whole finding in one assertion. It renders the *shared* tile texture, measures
    where the paint puts the drivable lane, and asks the lane graph what lateral error a robot
    standing exactly there would be scored with. Anything but zero is a constant bias on
    ``lane_rms_m`` and ``lane_max_m``, and at the old 0.22-tile offset it was -11.7 mm, 11% of a
    half lane.
    """
    spec = NOMINAL_TILE_SPEC
    painted = _painted_lane_center_m(spec)

    map_spec = MapSpec(tiles=[["straight"], ["straight"], ["straight"]], tile_size=spec.pitch_m, name="ns")
    lane = LaneGraph(map_spec)
    cx, cy = map_spec.center(1, 0)
    # Heading north, so the right-hand lane centre is at +x of the tile centreline.
    query = lane.query(cx + painted, cy, math.pi / 2.0)

    assert abs(query.d) < 1.0e-3, (
        f"a robot standing on the centre of the painted lane ({painted * 1e3:.2f} mm from the tile "
        f"centreline) is scored with a lateral error of {query.d * 1e3:+.2f} mm. The lane graph and "
        f"the texture disagree, so lane_rms_m and lane_max_m carry that bias."
    )


def test_the_measured_lane_centre_matches_the_city_spec_property() -> None:
    """The pixel measurement agrees with ``TileSpec.lane_center_offset_m``, and both are 0.20 tile."""
    spec = NOMINAL_TILE_SPEC
    painted = _painted_lane_center_m(spec)
    assert painted == pytest.approx(spec.lane_center_offset_m, abs=1.0e-3)
    assert spec.lane_center_offset_tile == pytest.approx(0.20, abs=1e-9)
    assert spec.lane_center_offset_tile != pytest.approx(DUCKIETOWN_WORLD_LANE_OFFSET_TILE)


def test_city_params_are_resolved_from_the_shared_spec_not_from_literals() -> None:
    """Every dimensional field comes from ``duckiebot_rl.city.spec``, and the provenance says so."""
    params, source = resolve_city_params()
    spec = NOMINAL_TILE_SPEC
    assert "duckiebot_rl.city.spec" in source
    assert params.tile_pitch == pytest.approx(spec.pitch_m)
    assert params.lane_offset_tiles == pytest.approx(spec.lane_center_offset_tile)
    assert params.lane_width == pytest.approx(spec.lane_width_m)
    assert params.white_tape_w == pytest.approx(spec.white_tape_mm / 1000.0)
    assert params.yellow_tape_w == pytest.approx(spec.yellow_tape_mm / 1000.0)


def test_curve_radii_follow_the_city_spec() -> None:
    """The arc radii of a curve tile are the [city] ones, not radii derived from a local literal."""
    params, _ = resolve_city_params()
    spec = NOMINAL_TILE_SPEC
    assert params.curve_radius_inner_m == pytest.approx(spec.curve_radius_inner_m)
    assert params.curve_radius_outer_m == pytest.approx(spec.curve_radius_outer_m)
    assert params.curve_radius_inner_m == pytest.approx(0.1755)
    assert params.curve_radius_outer_m == pytest.approx(0.4095)


def test_loop_length_matches_the_closed_form_from_the_city_radii() -> None:
    """The 5x5 loop length is 12 straights plus four quarter arcs at the [city] inner radius.

    The closed form is written from ``duckiebot_rl.city.spec`` properties rather than from the lane
    graph's own constant, so this is a cross-module check rather than the circular one it would be
    if both sides used the same local literal.
    """
    map_spec = load_map(LOOP_5X5)
    lane = LaneGraph(map_spec)
    spec = NOMINAL_TILE_SPEC
    expected = 12.0 * spec.pitch_m + 4.0 * (math.pi / 2.0) * spec.curve_radius_inner_m
    assert lane.cycle_length(0) == pytest.approx(expected, abs=1e-9)


def test_validate_rejects_the_duckietown_world_offset() -> None:
    """A 0.22-tile lane offset is refused: it is not the centre of the painted lane."""
    params = CityParams(lane_offset_tiles=DUCKIETOWN_WORLD_LANE_OFFSET_TILE)
    with pytest.raises(ValueError, match="painted clear lane"):
        params.validate()


def test_validate_accepts_the_resolved_geometry_and_rejects_impossible_markings() -> None:
    """``validate`` passes the shared geometry and still catches markings that do not fit."""
    resolve_city_params()[0].validate()
    with pytest.raises(ValueError, match="markings do not fit"):
        CityParams(lane_width=0.5, yellow_tape_w=0.024, white_tape_w=0.048).validate()


def test_resolver_raises_when_a_field_cannot_be_mapped() -> None:
    """A city spec missing an adapter field is a hard error, never a silent partial fallback."""

    class Partial:
        """A TileSpec-shaped object that has lost its lane-width properties."""

        pitch_m = 0.585
        lane_center_offset_tile = 0.20
        white_tape_mm = 48.0
        yellow_tape_mm = 24.0

    with pytest.raises(ValueError, match="lane_width"):
        resolve_city_params(Partial())


def test_resolver_rejects_a_spec_whose_lane_centre_is_not_the_painted_centre() -> None:
    """An explicit ``lane_center_offset_mm`` override is refused by the resolver, not absorbed."""
    forced = TileSpec(lane_center_offset_mm=0.22 * 585.0)
    with pytest.raises(ValueError, match="painted clear lane"):
        resolve_city_params(forced)


def test_rescaling_follows_the_map_tile_size() -> None:
    """A map declaring its own pitch scales the markings with it and keeps the lane centred."""
    params, _ = resolve_city_params()
    scaled = params.rescaled(0.61)
    assert scaled.tile_pitch == pytest.approx(0.61)
    assert scaled.lane_offset_tiles == pytest.approx(params.lane_offset_tiles)
    assert scaled.lane_width == pytest.approx(params.lane_width * 0.61 / params.tile_pitch)
    scaled.validate()

    map_spec = MapSpec(tiles=[["straight"], ["straight"]], tile_size=0.61, name="wide")
    lane = LaneGraph(map_spec)
    assert lane.city.tile_pitch == pytest.approx(0.61)
    assert lane.city.lane_width == pytest.approx(params.lane_width * 0.61 / params.tile_pitch)


def test_a_randomized_tile_spec_moves_the_lane_graph_with_the_paint() -> None:
    """An S7.2 axis-V9 geometry sample keeps ``d = 0`` on the painted lane centre.

    This is the same assertion as the headline test, run on a *randomized* marking geometry, which
    is what condition C6 evaluates. It is the reason the resolver takes a ``TileSpec`` rather than
    reading a module-level nominal.
    """
    spec = TileSpec(tile_pitch_mm=600.0, yellow_tape_mm=30.0, clear_lane_mm=190.0, white_tape_mm=42.0)
    params, _ = resolve_city_params(spec)
    painted = _painted_lane_center_m(spec)
    assert painted == pytest.approx(params.lane_offset_m, abs=1.0e-3)

    map_spec = MapSpec(tiles=[["straight"], ["straight"], ["straight"]], tile_size=spec.pitch_m, name="v9")
    lane = LaneGraph(map_spec, params)
    cx, cy = map_spec.center(1, 0)
    assert abs(lane.query(cx + painted, cy, math.pi / 2.0).d) < 1.0e-3


def test_fallback_literals_agree_with_the_shared_spec() -> None:
    """The literal fallback is a faithful transcription, so an offline build cannot drift.

    If ``[city]`` moves a dimension, this fails and the transcription has to be updated with it.
    """
    fallback = CityParams()
    spec = NOMINAL_TILE_SPEC
    assert fallback.tile_pitch == pytest.approx(spec.pitch_m)
    assert fallback.lane_offset_tiles == pytest.approx(spec.lane_center_offset_tile)
    assert fallback.lane_width == pytest.approx(spec.lane_width_m)
    assert fallback.white_tape_w == pytest.approx(spec.white_tape_mm / 1000.0)
    assert fallback.yellow_tape_w == pytest.approx(spec.yellow_tape_mm / 1000.0)
    fallback.validate()


def test_missing_city_module_warns_rather_than_silently_falling_back(monkeypatch) -> None:
    """With the shared spec unavailable the resolver warns; it never quietly uses the literals."""
    monkeypatch.setattr("duckiebot_rl.sim2sim._resolve._CITY_MODULES", ("duckiebot_rl.city.__absent__",))
    with pytest.warns(SpecFallbackWarning, match="city.spec is not importable"):
        params, source = resolve_city_params()
    assert "fallback" in source
    assert params.lane_offset_tiles == pytest.approx(0.20)
