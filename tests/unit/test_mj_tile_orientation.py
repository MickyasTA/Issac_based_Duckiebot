r"""Tile textures must be painted in the orientation the tile actually has (SPEC v2 S3.3, S8.1).

Interpreter: needs ``mujoco`` and ``numpy``. Run with::

    d:/Personal/personal/mujoco_venv/Scripts/python.exe -m pytest \\
        tests/unit/test_mj_tile_orientation.py --run-mujoco -q

The shared tile generator paints exactly one orientation per kind: a straight connecting north to
south, a curve connecting south to east. The Isaac city rotates that painting into place through
``duckiebot_rl.city.tiles.rotated_uv_corners``. A MuJoCo box geom has no per-face UV control, so the
rotation has to be baked into the texture instead; :func:`duckiebot_rl.sim2sim.track.kind_and_rotation`
derives the same rotation index and :class:`CityTextureProvider` applies it with ``np.rot90``.

Without it, an east-west straight showed a yellow line running north-south, i.e. straight across the
road rather than along it, and every curve showed the canonical south-east arc whichever way it
actually bent. That is a texture difference between the two simulators, which is precisely what a
C1-vs-C5 comparison must not contain.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

pytestmark = pytest.mark.mujoco
mujoco = pytest.importorskip("mujoco", reason="run these with the tools venv (mujoco_venv)")

from duckiebot_rl.city.tiles import write_png  # noqa: E402
from duckiebot_rl.sim2sim.track import (  # noqa: E402
    CityTextureProvider,
    kind_and_rotation,
)

RES = 256


def _yellow(image: np.ndarray) -> np.ndarray:
    """Return the boolean mask of yellow-tape pixels."""
    array = image.astype(np.int16)
    return (array[..., 0] > 200) & (array[..., 1] > 200) & (array[..., 2] < 80)


def test_rotation_index_matches_the_shared_convention() -> None:
    """Every orientation maps onto the canonical kind plus quarter turns counter-clockwise."""
    assert kind_and_rotation("straight", ["N", "S"]) == ("straight", 0)
    assert kind_and_rotation("straight", ["E", "W"]) == ("straight", 1)
    assert kind_and_rotation("curve", ["S", "E"]) == ("curve", 0)
    assert kind_and_rotation("curve", ["N", "E"]) == ("curve", 1)
    assert kind_and_rotation("curve", ["N", "W"]) == ("curve", 2)
    assert kind_and_rotation("curve", ["S", "W"]) == ("curve", 3)
    assert kind_and_rotation("4way", ["N", "S", "E", "W"]) == ("fourway", 0)
    assert kind_and_rotation("grass", []) == ("asphalt", 0)


def test_a_north_south_straight_paints_its_yellow_line_north_south() -> None:
    """The canonical orientation is unrotated: the yellow tape runs down the image columns."""
    provider = CityTextureProvider(resolution=RES)
    image = provider.texture("straight", ["N", "S"], RES)
    mask = _yellow(image)
    rows_with_yellow = int(np.count_nonzero(mask.any(axis=1)))
    cols_with_yellow = int(np.count_nonzero(mask.any(axis=0)))
    assert rows_with_yellow > RES // 2, "the dashed centre line should span most of the rows"
    assert cols_with_yellow < RES // 8, "the centre line should occupy only a narrow column band"


def test_an_east_west_straight_paints_its_yellow_line_east_west() -> None:
    """The rotated orientation puts the yellow tape along the road, not across it.

    This is the failing case before the fix: the same unrotated texture was used for both, so the
    east-west road carried a centre line running perpendicular to the direction of travel.
    """
    provider = CityTextureProvider(resolution=RES)
    image = provider.texture("straight", ["E", "W"], RES)
    mask = _yellow(image)
    rows_with_yellow = int(np.count_nonzero(mask.any(axis=1)))
    cols_with_yellow = int(np.count_nonzero(mask.any(axis=0)))
    assert cols_with_yellow > RES // 2, (
        "an east-west straight must carry its dashed centre line along the road; the texture is "
        "still the unrotated north-south painting"
    )
    assert rows_with_yellow < RES // 8


def test_the_two_straight_orientations_are_rotations_of_one_painting() -> None:
    """The east-west texture is exactly the north-south one turned a quarter turn."""
    provider = CityTextureProvider(resolution=RES)
    north_south = provider.texture("straight", ["N", "S"], RES)
    east_west = provider.texture("straight", ["E", "W"], RES)
    assert np.array_equal(east_west, np.rot90(north_south, 1))


def test_every_curve_orientation_hugs_its_own_corner() -> None:
    """Each of the four curve orientations puts its arc around the corner it actually connects."""
    provider = CityTextureProvider(resolution=RES)
    # The yellow arc has radius half-a-tile about the corner shared by the two open edges, so it
    # sweeps entirely through that corner's quadrant. (connections -> quadrant), rows north-first.
    cases = {
        ("S", "E"): (slice(RES // 2, RES), slice(RES // 2, RES)),
        ("N", "E"): (slice(0, RES // 2), slice(RES // 2, RES)),
        ("N", "W"): (slice(0, RES // 2), slice(0, RES // 2)),
        ("S", "W"): (slice(RES // 2, RES), slice(0, RES // 2)),
    }
    for connections, quadrant in cases.items():
        mask = _yellow(provider.texture("curve", list(connections), RES))
        counts = {
            "NW": int(mask[0 : RES // 2, 0 : RES // 2].sum()),
            "NE": int(mask[0 : RES // 2, RES // 2 : RES].sum()),
            "SW": int(mask[RES // 2 : RES, 0 : RES // 2].sum()),
            "SE": int(mask[RES // 2 : RES, RES // 2 : RES].sum()),
        }
        corner_count = int(mask[quadrant].sum())
        assert corner_count == max(counts.values()), (
            f"curve {connections} has its arc drawn around the wrong corner; quadrant yellow "
            f"counts were {counts}"
        )
        assert corner_count > 4 * sorted(counts.values())[-2], (
            f"curve {connections} does not concentrate its arc in one quadrant: {counts}"
        )


def test_a_mujoco_box_face_reads_row_zero_as_north_and_column_zero_as_west() -> None:
    """Lock the convention the rotation depends on, by measuring it.

    ``duckiebot_rl.city.tiles`` paints row 0 at +y (north) and column 0 at -x (west). The rotation
    baked into the texture is only correct if a MuJoCo box +z face maps the image the same way, so
    this renders a four-colour quadrant chart on a tile-sized box from directly above and checks
    where each quadrant lands. If MuJoCo ever changes that mapping, this fails rather than the
    lane markings quietly turning ninety degrees.
    """
    with tempfile.TemporaryDirectory() as tmp:
        chart = np.zeros((64, 64, 3), dtype=np.uint8)
        chart[:32, :32] = (255, 255, 255)  # north-west
        chart[:32, 32:] = (255, 0, 0)  # north-east
        chart[32:, :32] = (0, 255, 0)  # south-west
        chart[32:, 32:] = (0, 0, 255)  # south-east
        write_png(Path(tmp) / "chart.png", chart)
        xml = f"""
        <mujoco>
          <compiler texturedir="{tmp}"/>
          <asset>
            <texture name="tex" type="2d" file="chart.png"/>
            <material name="mat" texture="tex" texrepeat="1 1" texuniform="false"/>
          </asset>
          <worldbody>
            <light pos="0 0 3"/>
            <geom name="tile" type="box" size="0.5 0.5 0.01" pos="0 0 0" material="mat"/>
            <camera name="top" pos="0 0 2" xyaxes="1 0 0 0 1 0"/>
          </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        try:
            renderer = mujoco.Renderer(model, 128, 128)
        except Exception as exc:  # pragma: no cover - depends on the local OpenGL stack
            pytest.skip(f"no offscreen OpenGL context available: {exc}")
        try:
            renderer.update_scene(data, camera="top")
            pixels = renderer.render()
        finally:
            renderer.close()

    # The camera looks down -z with image right = world +x and image up = world +y.
    def dominant(rows: slice, cols: slice) -> int:
        return int(np.argmax(pixels[rows, cols].reshape(-1, 3).mean(axis=0)))

    assert dominant(slice(10, 50), slice(78, 118)) == 0, "world north-east is not texture row 0/right"
    assert dominant(slice(78, 118), slice(10, 50)) == 1, "world south-west is not texture bottom-left"
    assert dominant(slice(78, 118), slice(78, 118)) == 2, "world south-east is not texture bottom-right"
    north_west = pixels[10:50, 10:50].reshape(-1, 3).mean(axis=0)
    assert north_west.min() > 60.0, "world north-west is not the white texture row 0/column 0"
