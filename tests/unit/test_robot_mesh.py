"""The real DB21 visual: which file is chosen, how it is oriented, and what it may not touch.

Three claims are locked down here, none of which needs Isaac Sim.

**Which file.** ``db21j`` used to mean ``_refs/visual_mesh/db21/main.gltf``, which
duckietown-world publishes under a directory called ``duckiebot3``. That file is export_DB18's
asset: it is the DB18-era robot, not a DB21. The real latest-generation model is only published
inside Duckietown's own Duckiematrix engine, and ``scripts/fetch_visual_mesh.py`` extracts it to
``_refs/visual_mesh/db21j/main.obj``. The resolution order therefore has to prefer the OBJ, and
the glTF has to keep working as a labelled fallback.

**How it is oriented.** The extraction is metres and Y-up with the mast along +Y, so it needs a
fixed rotation into Isaac's Z-up, +X-forward, +Y-left frame. The existing "is it lying down?"
heuristic CANNOT do that job: it compares the Y extent against the Z extent with a 1.5x
threshold, and this model is 0.1216 m tall against 0.1340 m wide, a ratio of 1.10. The rotation
is a constant, and these tests check the constant against the geometry it claims to describe.

**What it may not touch.** Attaching the mesh must not change what the policy sees, and for
geometry that is provable offline: every one of the 281,367 vertices is mapped into the camera's
own frame and shown to fall inside the 0.05 m near plane, at the nominal camera pose and at every
corner of the declared V10 camera-pose randomisation box. What no offline test can settle is
indirect lighting and the robot's own cast shadow, which is why ``scripts/check_obs.py`` grew a
``--robot-mesh`` flag: run it once per mesh and diff the two reports.

The vertex-level tests skip when the mesh is absent, which is the normal state of a fresh clone:
the model is not redistributable, so ``_refs/`` is gitignored and nothing here may assume it.
"""

from __future__ import annotations

import functools
import math
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from duckiebot_rl.assets.params import DUCKIEBOT
from duckiebot_rl.envs import viz_env
from scripts import fetch_visual_mesh as fetch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_OBJ = _REPO_ROOT / "_refs" / "visual_mesh" / "db21j" / "main.obj"

_needs_obj = pytest.mark.skipif(
    not _OBJ.is_file(),
    reason="the Duckiematrix DB21 is not on this machine; run scripts/fetch_visual_mesh.py",
)

#: Bounding box of the extraction in its OWN frame (X length, Y height, Z width), in metres.
_SOURCE_SIZE_M = (0.215083, 0.121565, 0.134002)

#: Lowest point of the extraction along its up axis: the wheel and caster contact patches [m].
_SOURCE_GROUND_Y_M = 0.000319


# =============================================================================== helpers


@functools.lru_cache(maxsize=1)
def _load_vertices(path: Path) -> np.ndarray:
    """Read an OBJ's vertex positions, once per session.

    Two economies, both of them about the same constraint. ``np.fromiter`` streams straight into a
    numpy buffer instead of building 281,367 Python tuples first, and the result is cached and
    frozen so that the seven tests below share one 6.8 MB array rather than allocating seven. The
    machine this project targets dies of commit exhaustion, not of VRAM, and its unit tests run
    beside an 8 GB Kit process: a test suite that adds to that pressure is a test suite that fails
    for reasons that have nothing to do with the code.

    Args:
        path: The ``.obj`` file.

    Returns:
        A read-only ``(N, 3)`` float64 array.
    """

    def values() -> Any:
        """Yield every vertex coordinate in file order.

        Yields:
            One float per coordinate.
        """
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.startswith("v "):
                    _, x, y, z = line.split()
                    yield float(x)
                    yield float(y)
                    yield float(z)

    vertices = np.fromiter(values(), dtype=np.float64).reshape(-1, 3)
    vertices.setflags(write=False)
    return vertices


def _group_extents(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Measure each OBJ group's bounding box.

    Args:
        path: The ``.obj`` file.

    Returns:
        Group name to ``(minimum, maximum)``, both ``(3,)`` arrays.
    """
    vertices = _load_vertices(path)
    boxes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    current = ""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("g "):
                current = line[2:].strip()
            elif line.startswith("f "):
                for token in line.split()[1:]:
                    point = vertices[int(token.split("/")[0]) - 1]
                    low, high = boxes.get(current, (point, point))
                    boxes[current] = (np.minimum(low, point), np.maximum(high, point))
    return boxes


def _rotate_xyz(degrees: tuple[float, float, float]) -> np.ndarray:
    """Build the matrix a USD ``RotateXYZOp`` applies.

    USD rotates about X, then Y, then Z, which composes as ``Rz @ Ry @ Rx`` on a column vector.

    Args:
        degrees: The ``(x, y, z)`` Euler triple in degrees.

    Returns:
        A ``(3, 3)`` rotation matrix.
    """
    x, y, z = (math.radians(value) for value in degrees)
    rx = np.array([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]])
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rz = np.array([[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]])
    return rz @ ry @ rx


def _to_base_link(vertices: np.ndarray) -> np.ndarray:
    """Map source vertices into the robot's ``base_link`` frame.

    The numpy twin of what the attachment authors in USD: the fixed rotation, then the drop that
    stands the wheels on the ground while ``base_link`` rides one wheel radius above it.

    Args:
        vertices: An ``(N, 3)`` array in the OBJ's own frame.

    Returns:
        An ``(N, 3)`` array in ``base_link``.
    """
    rotated = vertices @ _rotate_xyz(viz_env.DB21_OBJ_ROTATE_XYZ).T
    return rotated - np.array([0.0, 0.0, rotated[:, 2].min() + DUCKIEBOT.base_link_height_m])


def _forward_depth(points: np.ndarray, forward_m: float, height_m: float, pitch_deg: float) -> np.ndarray:
    """Return each point's depth along the camera's optical axis.

    Args:
        points: An ``(N, 3)`` array in ``base_link``.
        forward_m: Camera x offset in ``base_link`` [m].
        height_m: Camera height above the ground [m].
        pitch_deg: Downward pitch of the optical axis [deg].

    Returns:
        An ``(N,)`` array of view-space depths. Negative means behind the camera.
    """
    pitch = math.radians(pitch_deg)
    origin = np.array([forward_m, 0.0, height_m - DUCKIEBOT.base_link_height_m])
    axis = np.array([math.cos(pitch), 0.0, -math.sin(pitch)])
    return (points - origin) @ axis


# =============================================================== which source file is used


def test_db21j_sources_are_documented_in_preference_order() -> None:
    """The Duckiematrix OBJ first: it is the only one of the two that is a real DB21."""
    assert [kind for kind, _location in viz_env.DB21J_SOURCES] == ["duckiematrix-obj", "db18-gltf"]
    assert viz_env.DB21J_SOURCES[0][1] == "_refs/visual_mesh/db21j/main.obj"
    assert viz_env.DB21J_SOURCES[1][1] == "_refs/visual_mesh/db21/main.gltf"


def test_the_obj_wins_when_both_sources_are_on_disk(monkeypatch) -> None:
    """The regression this whole change exists to fix: the glTF is not a DB21."""
    monkeypatch.setattr(viz_env, "_find_db21j_obj", lambda: Path("obj/main.obj"))
    monkeypatch.setattr(viz_env, "_find_db21j_gltf", lambda: Path("gltf/main.gltf"))
    kind, path = viz_env.resolve_db21j_source()
    assert kind == "duckiematrix-obj"
    assert path == Path("obj/main.obj")


def test_the_gltf_is_used_only_when_the_obj_is_absent(monkeypatch) -> None:
    """A user who has not fetched the engine build still gets something Duckiebot-shaped."""
    monkeypatch.setattr(viz_env, "_find_db21j_obj", lambda: None)
    monkeypatch.setattr(viz_env, "_find_db21j_gltf", lambda: Path("gltf/main.gltf"))
    assert viz_env.resolve_db21j_source() == ("db18-gltf", Path("gltf/main.gltf"))


def test_no_source_at_all_reports_none(monkeypatch) -> None:
    """None, rather than an exception: a missing mesh downgrades the view, it does not fail."""
    monkeypatch.setattr(viz_env, "_find_db21j_obj", lambda: None)
    monkeypatch.setattr(viz_env, "_find_db21j_gltf", lambda: None)
    assert viz_env.resolve_db21j_source() is None


def test_db21j_survives_when_only_the_obj_is_present(monkeypatch) -> None:
    """The OBJ alone is enough; the glTF is not a prerequisite for anything."""
    monkeypatch.setattr(viz_env, "_find_db21j_obj", lambda: Path("obj/main.obj"))
    monkeypatch.setattr(viz_env, "_find_db21j_gltf", lambda: None)
    assert viz_env.resolve_robot_mesh("db21j") == "db21j"


def test_db21j_downgrades_to_db17_when_neither_source_is_present(monkeypatch, capsys) -> None:
    """And it says which two files it looked for, both of them, by path."""
    monkeypatch.setattr(viz_env, "_find_db21j_obj", lambda: None)
    monkeypatch.setattr(viz_env, "_find_db21j_gltf", lambda: None)
    monkeypatch.setattr(viz_env, "_find_visual_mesh_dir", lambda: "somewhere/meshes")
    assert viz_env.resolve_robot_mesh("db21j") == "db17"
    printed = capsys.readouterr().out
    assert "_refs/visual_mesh/db21j/main.obj" in printed
    assert "_refs/visual_mesh/db21/main.gltf" in printed


@_needs_obj
def test_the_installed_mesh_is_found_where_the_fetcher_puts_it() -> None:
    """No monkeypatching: the real lookup, against the real file."""
    kind, path = viz_env.resolve_db21j_source()
    assert kind == "duckiematrix-obj"
    assert Path(path) == _OBJ


# ============================================================= the deterministic rotation


def test_the_obj_orientation_is_a_constant() -> None:
    """Not a heuristic. The lying-down rule could not fire on this model anyway."""
    assert viz_env.DB21_OBJ_ROTATE_XYZ == (90.0, 0.0, 180.0)
    # 0.1340 m wide against 0.1216 m tall is a ratio of 1.10, well under the heuristic's 1.5x.
    assert _SOURCE_SIZE_M[2] / _SOURCE_SIZE_M[1] < 1.5


def test_the_rotation_maps_the_source_frame_onto_the_isaac_frame() -> None:
    """``(x, y, z) -> (-x, z, y)``: mast to +Z, front to +X, left to +Y."""
    matrix = _rotate_xyz(viz_env.DB21_OBJ_ROTATE_XYZ)
    for source, expected in (
        ((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
        ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    ):
        assert np.allclose(matrix @ np.asarray(source), np.asarray(expected), atol=1e-12)


def test_a_plain_ninety_degrees_about_x_would_leave_the_robot_driving_backwards() -> None:
    """Why the constant is not just the Y-up correction the glTF path applies."""
    naive = _rotate_xyz((90.0, 0.0, 0.0))
    # the source's front is -X; a bare +90 about X leaves it on -X, which is 180 deg out
    assert (naive @ np.array([-1.0, 0.0, 0.0]))[0] == pytest.approx(-1.0)
    assert (_rotate_xyz(viz_env.DB21_OBJ_ROTATE_XYZ) @ np.array([-1.0, 0.0, 0.0]))[0] == pytest.approx(1.0)


def test_the_expected_box_is_the_source_box_reordered_by_that_rotation() -> None:
    """The number the attachment checks against is derived from the source, not from hope."""
    rotated = np.abs(_rotate_xyz(viz_env.DB21_OBJ_ROTATE_XYZ) @ np.asarray(_SOURCE_SIZE_M))
    assert np.allclose(rotated, np.asarray(viz_env.DB21_OBJ_EXPECTED_SIZE_M), atol=1e-3)
    length, width, height = viz_env.DB21_OBJ_EXPECTED_SIZE_M
    assert length == pytest.approx(0.2151, abs=1e-4)
    assert width == pytest.approx(0.1340, abs=1e-4)
    assert height == pytest.approx(0.1216, abs=1e-4)


@pytest.mark.parametrize(
    "point",
    [(0.0, 0.0, 0.0), (0.1308, 0.1219, 0.067), (-0.0843, 0.0942, -0.0058), (0.05, 0.02, -0.03)],
)
def test_the_python_twin_of_the_transform_matches_the_usd_one(point: tuple[float, float, float]) -> None:
    """``duckiematrix_point_to_base_link`` is what lets the transform be checked without Kit."""
    height = DUCKIEBOT.base_link_height_m
    expected = _rotate_xyz(viz_env.DB21_OBJ_ROTATE_XYZ) @ np.asarray(point)
    expected = expected - np.array([0.0, 0.0, _SOURCE_GROUND_Y_M + height])
    got = viz_env.duckiematrix_point_to_base_link(point, height, _SOURCE_GROUND_Y_M)
    assert np.allclose(np.asarray(got), expected, atol=1e-12)


# ================================================================ the file that is on disk


@_needs_obj
def test_the_installed_obj_is_the_model_the_constants_describe() -> None:
    """Counts and box, straight off the file. A different build must not pass silently."""
    vertices, faces, size = fetch.measure_obj(_OBJ)
    assert vertices == fetch.EXPECTED_VERTICES == 281_367
    assert faces == fetch.EXPECTED_FACES == 184_838
    assert np.allclose(np.asarray(size), np.asarray(_SOURCE_SIZE_M), atol=1e-4)
    assert fetch.verify_obj(_OBJ) is True


@_needs_obj
def test_the_fetcher_and_the_attachment_agree_on_the_same_box() -> None:
    """Two modules, one measurement: ``EXPECTED_SIZE_M`` rotated is ``DB21_OBJ_EXPECTED_SIZE_M``."""
    rotated = np.abs(_rotate_xyz(viz_env.DB21_OBJ_ROTATE_XYZ) @ np.asarray(fetch.EXPECTED_SIZE_M))
    assert np.allclose(rotated, np.asarray(viz_env.DB21_OBJ_EXPECTED_SIZE_M), atol=1e-3)


@_needs_obj
def test_after_the_rotation_the_height_is_on_z_and_the_length_on_x() -> None:
    """The claim the measured-bbox print exists to make, made here without booting Kit."""
    points = _to_base_link(_load_vertices(_OBJ))
    size = points.max(axis=0) - points.min(axis=0)
    assert size[0] == pytest.approx(0.2151, abs=1e-3)
    assert size[1] == pytest.approx(0.1340, abs=1e-3)
    assert size[2] == pytest.approx(0.1216, abs=1e-3)
    # the wheels stand on the ground: base_link rides exactly one wheel radius above the lowest
    assert points[:, 2].min() == pytest.approx(-DUCKIEBOT.base_link_height_m, abs=1e-9)
    # and the model is centred laterally, which is what identifies the lateral axis at attach time
    assert abs(points[:, 1].min() + points[:, 1].max()) < 1e-3


@_needs_obj
def test_the_front_of_the_model_ends_up_forward() -> None:
    """The camera mast and the front bumper on +X, the back bumper on -X."""
    groups = _group_extents(_OBJ)
    matrix = _rotate_xyz(viz_env.DB21_OBJ_ROTATE_XYZ)

    def forward_span(needle: str) -> tuple[float, float]:
        """Return the min and max +X reach of every group whose path contains ``needle``.

        Args:
            needle: Case-insensitive substring of the group path.

        Returns:
            ``(minimum, maximum)`` along the rotated X axis.
        """
        reach = [
            (matrix @ corner)[0] for name, box in groups.items() if needle in name.lower() for corner in box
        ]
        assert reach, f"no group matched {needle!r}"
        return min(reach), max(reach)

    assert forward_span("camera_mount")[1] > 0.049
    assert forward_span("front_bumper")[0] > 0.0
    assert forward_span("back_bumper")[1] < 0.0

    # The independent confirmation that the orientation is right rather than merely consistent:
    # the real robot's caster and wheel axle land on the clean-room robot's own numbers, which
    # were derived from Duckietown's published dimensions and not from this mesh. A model that
    # was 180 deg out would put the caster in front of the axle instead of 85 mm behind it.
    caster = forward_span("caster_wheel")
    assert sum(caster) / 2.0 == pytest.approx(DUCKIEBOT.caster_center_base_frame_m[0], abs=5e-3)
    axle = forward_span("/wheel/")
    assert sum(axle) / 2.0 == pytest.approx(0.0, abs=5e-3)


@_needs_obj
def test_only_the_wheels_reach_past_the_lateral_threshold() -> None:
    """The geometric backstop that guarantees black tyres whatever the group names become."""
    groups = _group_extents(_OBJ)
    matrix = _rotate_xyz(viz_env.DB21_OBJ_ROTATE_XYZ)
    threshold = viz_env._DB21_WHEEL_HALF_WIDTH_M
    wide = {
        name for name, box in groups.items() if max(abs((matrix @ corner)[1]) for corner in box) >= threshold
    }
    assert wide, "no group reaches the wheel threshold; the classifier would colour nothing black"
    assert all("wheel" in name.lower() for name in wide), sorted(wide)
    # and every one of them is caught by the name rules too, so the two agree rather than compete
    assert all(viz_env._db21_material_name(name) == "tyre" for name in wide)


# ================================================== the policy camera contract, proven offline


@_needs_obj
def test_the_robot_mesh_never_reaches_its_own_camera_near_plane() -> None:
    """The geometric half of "visual only": no vertex is in front of the near plane.

    Depth along the optical axis is linear in position, so a triangle's deepest point is one of
    its vertices: checking all 281,367 vertices checks the whole surface. The margin at the
    nominal pose is large (the model tops out around 18 mm into a 50 mm near plane), which is what
    a robot whose camera is mounted at the very front of its own chassis looks like.
    """
    points = _to_base_link(_load_vertices(_OBJ))
    forward_m, _lateral, up_m = DUCKIEBOT.camera_pos_base_frame_m
    depth = _forward_depth(
        points,
        forward_m=forward_m,
        height_m=up_m + DUCKIEBOT.base_link_height_m,
        pitch_deg=DUCKIEBOT.camera_pitch_down_deg,
    )
    near = DUCKIEBOT.camera_clipping_range_m[0]
    assert float(depth.max()) < near
    assert float(depth.max()) == pytest.approx(0.0177, abs=5e-4)


@_needs_obj
def test_the_mesh_stays_behind_the_near_plane_across_the_whole_camera_dr_box() -> None:
    """Every corner of the declared V10 camera-pose randomisation, not just the nominal pose.

    The V10 clamps are not applied by the environment today; they are declared in
    ``DuckiebotParams`` and this test is what will notice if implementing them ever pushes the
    robot's own bumper through its own near plane. The worst corner (camera pulled back to
    0.058 m, raised to 0.120 m, pitched 28 deg down) leaves about 1.8 mm of margin, so the
    tolerance below is deliberately tight rather than comfortable.
    """
    points = _to_base_link(_load_vertices(_OBJ))
    near = DUCKIEBOT.camera_clipping_range_m[0]
    worst = -math.inf
    for forward_m in DUCKIEBOT.dr_camera_forward_m:
        for height_m in DUCKIEBOT.dr_camera_height_m:
            for pitch_deg in DUCKIEBOT.dr_camera_pitch_down_deg:
                depth = _forward_depth(points, forward_m, height_m, pitch_deg)
                worst = max(worst, float(depth.max()))
    assert worst < near
    assert worst == pytest.approx(0.0482, abs=5e-4)


@_needs_obj
def test_no_vertex_is_inside_the_camera_frustum_at_the_nominal_pose() -> None:
    """The same conclusion by the full frustum test, near plane and both field-of-view limits."""
    points = _to_base_link(_load_vertices(_OBJ))
    forward_m, _lateral, up_m = DUCKIEBOT.camera_pos_base_frame_m
    pitch = math.radians(DUCKIEBOT.camera_pitch_down_deg)
    origin = np.array([forward_m, 0.0, up_m])
    axis = np.array([math.cos(pitch), 0.0, -math.sin(pitch)])
    up = np.array([math.sin(pitch), 0.0, math.cos(pitch)])
    offsets = points - origin
    depth = offsets @ axis
    horizontal = np.abs(offsets[:, 1])
    vertical = np.abs(offsets @ up)
    inside = (
        (depth >= DUCKIEBOT.camera_clipping_range_m[0])
        & (horizontal <= depth * math.tan(math.radians(DUCKIEBOT.camera_hfov_deg) / 2.0))
        & (vertical <= depth * math.tan(math.radians(DUCKIEBOT.camera_vfov_deg) / 2.0))
    )
    assert int(inside.sum()) == 0


# ===================================================================== attachment plumbing


def test_the_robot_prim_path_addresses_any_environment() -> None:
    """One attachment pass covers a whole training scene because the stage names every robot."""
    assert viz_env.robot_prim_path() == "/World/envs/env_0/Robot"
    assert viz_env.robot_prim_path(63) == "/World/envs/env_63/Robot"


def test_the_environment_count_comes_from_the_scene() -> None:
    """``num_envs`` on the env, else on its scene, else one."""
    assert viz_env._env_count(types.SimpleNamespace(num_envs=64)) == 64
    assert viz_env._env_count(types.SimpleNamespace(scene=types.SimpleNamespace(num_envs=8))) == 8
    assert viz_env._env_count(object()) == 1


def test_one_line_is_printed_per_distinct_outcome_not_per_environment(capsys) -> None:
    """64 identical messages bury the one that differs."""
    viz_env._report({"DB21 attached": 63, "no base_link under /World/envs/env_9/Robot": 1})
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(lines) == 2
    assert "DB21 attached (x63 envs)" in lines[0]
    assert lines[1].endswith("Robot")


def test_attaching_primitives_touches_nothing(monkeypatch, capsys) -> None:
    """The primitive path must not look for a mesh, let alone import Kit."""

    def explode() -> object:
        raise AssertionError("the primitive path must not look for meshes")

    monkeypatch.setattr(viz_env, "_find_db21j_obj", explode)
    monkeypatch.setattr(viz_env, "_find_visual_mesh_dir", explode)
    assert viz_env.attach_robot_visuals(object(), robot_mesh="primitive", num_envs=64) == "primitive"
    assert "keeping the environment's own visuals" in capsys.readouterr().out


class _ComposedPrim:
    """A prim that already exists, the way a clone of env_0 composes its attachment."""

    def IsValid(self) -> bool:
        return True


class _ComposedStage:
    """A stage on which every queried prim is already composed."""

    def GetPrimAtPath(self, path: str) -> _ComposedPrim:
        return _ComposedPrim()


def _fake_pxr(monkeypatch) -> None:
    """Install an inert ``pxr`` module: the guard under test must return before touching it."""
    module = types.ModuleType("pxr")
    module.Gf = types.SimpleNamespace()
    module.UsdGeom = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "pxr", module)


def test_attach_db21_obj_skips_a_clone_that_already_composes_the_mesh(monkeypatch) -> None:
    """A clone that already composes the mesh is counted as attached and left alone.

    The robot spawns with ``copy_from_source=False``, so env_1..N are internal references to
    env_0 and compose its attachment live. Authoring the xform ops again on such a clone raises
    ("xformOp:translate already exists in xformOpOrder"), which took the WHOLE attachment down to
    primitives on any multi-env scene (observed on a real 4-env ``check_obs`` boot).
    """
    _fake_pxr(monkeypatch)
    message = viz_env._attach_db21_obj(
        _ComposedStage(), None, "main_colored.usd", "/World/envs/env_1/Robot", verbose=False
    )
    assert message == "DB21 (Duckiematrix extraction) attached"


def test_attach_db21j_gltf_skips_a_clone_that_already_composes_the_mesh(monkeypatch, tmp_path) -> None:
    _fake_pxr(monkeypatch)
    usd = tmp_path / "main.usd"
    usd.write_bytes(b"")
    message = viz_env._attach_db21j_gltf(
        _ComposedStage(),
        None,
        lambda src, dst: False,
        tmp_path / "main.gltf",
        "/World/envs/env_1/Robot",
        verbose=False,
    )
    assert message == "DB18-era glTF attached (duckietown-world 'duckiebot3', not a real DB21)"


def test_attaching_rejects_an_unknown_selector() -> None:
    """A typo is a programming error; a missing file is not. Only the first one raises."""
    with pytest.raises(ValueError, match="unknown robot_mesh"):
        viz_env.attach_robot_visuals(object(), robot_mesh="db99")


def test_a_missing_mesh_downgrades_instead_of_raising(monkeypatch, capsys) -> None:
    """A training run must never die because the pretty robot was not downloaded."""
    monkeypatch.setattr(viz_env, "_find_db21j_obj", lambda: None)
    monkeypatch.setattr(viz_env, "_find_db21j_gltf", lambda: None)
    monkeypatch.setattr(viz_env, "_find_visual_mesh_dir", lambda: None)
    assert viz_env.attach_robot_visuals(object(), robot_mesh="db21j", num_envs=64) == "primitive"
    assert "primitive" in capsys.readouterr().out


# ============================================================== the training-side command line


@pytest.fixture(scope="module")
def isaac_free_scripts() -> tuple[Any, Any]:
    """Import ``scripts/train.py`` and ``scripts/check_obs.py`` with a stubbed ``isaaclab.app``.

    Both import ``AppLauncher`` at module scope, which is Isaac Lab's documented launch rule and
    not something to work around in production code. The stub lets their parsers be tested on a
    runner with no Isaac Sim, exactly as ``tests/unit/test_ppo_perf.py`` already does.

    Returns:
        ``(train_module, check_obs_module)``.
    """
    if "isaaclab.app" not in sys.modules:

        class _StubAppLauncher:
            """Stands in for ``isaaclab.app.AppLauncher`` while importing the scripts."""

            @staticmethod
            def add_app_launcher_args(parser: Any) -> None:
                """Add the launcher flags the scripts read back.

                Args:
                    parser: The argument parser being built at module scope.
                """
                parser.add_argument("--headless", action="store_true")
                parser.add_argument("--enable_cameras", action="store_true", default=None)
                parser.add_argument("--device", default="cuda:0")

        package = types.ModuleType("isaaclab")
        package.__path__ = []  # type: ignore[attr-defined]
        app = types.ModuleType("isaaclab.app")
        app.AppLauncher = _StubAppLauncher  # type: ignore[attr-defined]
        sys.modules.setdefault("isaaclab", package)
        sys.modules["isaaclab.app"] = app

    import scripts.check_obs as check_obs
    import scripts.train as train

    return train, check_obs


def test_training_draws_the_real_db21_by_default(isaac_free_scripts) -> None:
    """The flag exists on the trainer, and its default is the latest-generation robot."""
    train, _check_obs = isaac_free_scripts
    args = train.build_parser().parse_args([])
    assert args.robot_mesh == "db21j"
    assert train.DEFAULT_ROBOT_MESH == "db21j"


@pytest.mark.parametrize("choice", ["db21j", "db17", "primitive"])
def test_the_trainer_accepts_every_mesh_choice(isaac_free_scripts, choice: str) -> None:
    """Same vocabulary as the viewer, from the same tuple."""
    train, _check_obs = isaac_free_scripts
    assert train.build_parser().parse_args(["--robot-mesh", choice]).robot_mesh == choice
    assert train.ROBOT_MESH_CHOICES == viz_env.ROBOT_MESH_CHOICES


def test_the_trainer_rejects_an_unknown_mesh_before_booting_kit(isaac_free_scripts) -> None:
    """A typo must cost nothing; argparse refuses it at parse time."""
    train, _check_obs = isaac_free_scripts
    with pytest.raises(SystemExit):
        train.build_parser().parse_args(["--robot-mesh", "db99"])


def test_the_trainer_attaches_to_every_environment(isaac_free_scripts) -> None:
    """Not just env 0: the count comes from the settings the scene was built with."""
    import inspect

    train, _check_obs = isaac_free_scripts
    source = inspect.getsource(train.train)
    assert "attach_robot_visuals(env, robot_mesh=args.robot_mesh, num_envs=settings.num_envs)" in source
    # and it happens after the scene exists, because it addresses prims on the stage
    assert source.index("DuckiebotLaneFollowEnv(cfg)") < source.index("attach_robot_visuals(env")


def test_the_mesh_choice_lands_in_the_run_config(isaac_free_scripts) -> None:
    """``config.yaml`` has to record which robot was drawn, or a run cannot be reproduced."""
    train, _check_obs = isaac_free_scripts
    args = train.build_parser().parse_args(["--robot-mesh", "db17"])
    config = train.run_config(args, types.SimpleNamespace(summary=lambda: {}))
    assert config["cli"]["robot_mesh"] == "db17"


def test_the_observation_check_takes_the_same_flag_and_the_same_default(isaac_free_scripts) -> None:
    """The two reports a verifier diffs have to be produced by one script and one vocabulary."""
    _train, check_obs = isaac_free_scripts
    assert check_obs.build_parser().parse_args([]).robot_mesh == viz_env.DEFAULT_ROBOT_MESH
    assert check_obs.build_parser().parse_args(["--robot-mesh", "primitive"]).robot_mesh == "primitive"


# ==================================================================== the fetcher's own logic


def test_only_the_engine_asset_files_are_extracted() -> None:
    """123 MB in, about a third of it skipped: no IL2CPP metadata, no plugins, no executable."""
    names = [
        "Duckiematrix.exe",
        "GameAssembly.dll",
        "Duckiematrix_Data/level0",
        "Duckiematrix_Data/resources.assets",
        "Duckiematrix_Data/resources.assets.resS",
        "Duckiematrix_Data/il2cpp_data/Metadata/global-metadata.dat",
        "Duckiematrix_Data/Plugins/x86_64/lib_burst_generated.dll",
        "Duckiematrix_BurstDebugInformation_DoNotShip/",
    ]
    assert fetch.engine_members(names) == [
        "Duckiematrix_Data/level0",
        "Duckiematrix_Data/resources.assets",
        "Duckiematrix_Data/resources.assets.resS",
    ]


@pytest.mark.parametrize(
    ("path", "skipped"),
    [
        ("DB21/Curve_002", True),
        ("DB21/footprint/base/left_motor/wheel/Curve_002", False),
        ("DB21/footprint/base/bottom_plate/top_plate/FakeCamera", True),
        ("DB21/footprint/base/bottom_plate/top_plate/FakeCamera/lens", True),
        ("DB21/footprint/base/bottom_plate/top_plate/camera_mount", False),
    ],
)
def test_the_export_drops_the_placeholder_geometry_only(path: str, skipped: bool) -> None:
    """The wheels carry nodes named ``Curve_002`` too, and those ARE the tyres."""
    assert fetch.skips_mesh(path) is skipped


def test_measuring_an_obj_counts_and_sizes_it(tmp_path: Path) -> None:
    """The cheap verification that runs in milliseconds instead of minutes into a Kit boot."""
    obj = tmp_path / "cube.obj"
    obj.write_text("# test\nv 0.0 0.0 0.0\nv 1.0 2.0 3.0\nv 0.5 0.5 0.5\ng part\nf 1 2 3\n", encoding="utf-8")
    vertices, faces, size = fetch.measure_obj(obj)
    assert (vertices, faces) == (3, 1)
    assert size == pytest.approx((1.0, 2.0, 3.0))


def test_measuring_an_empty_obj_does_not_divide_by_an_infinite_box(tmp_path: Path) -> None:
    """A failed export writes a header and nothing else; that must report zero, not NaN."""
    obj = tmp_path / "empty.obj"
    obj.write_text("# nothing here\n", encoding="utf-8")
    assert fetch.measure_obj(obj) == (0, 0, (0.0, 0.0, 0.0))


def test_a_fresh_export_invalidates_the_cached_conversions(tmp_path: Path, capsys) -> None:
    """Otherwise a re-fetched model would keep drawing as the one it replaced, for ever."""
    obj = tmp_path / "main.obj"
    obj.write_text("v 0 0 0\n", encoding="utf-8")
    geometry = tmp_path / "main.usd"
    coloured = tmp_path / "main_colored.usd"
    for stale in (geometry, coloured):
        stale.write_text("stale", encoding="utf-8")
    fetch.drop_derived_usd(obj)
    assert not geometry.exists()
    assert not coloured.exists()
    assert obj.exists()
    assert "rebuilt on next launch" in capsys.readouterr().out


def test_dropping_conversions_that_were_never_built_is_not_an_error(tmp_path: Path) -> None:
    """The normal case: a first fetch on a clean machine."""
    obj = tmp_path / "main.obj"
    obj.write_text("v 0 0 0\n", encoding="utf-8")
    fetch.drop_derived_usd(obj)
    assert obj.exists()


def test_a_different_model_is_reported_rather_than_accepted(tmp_path: Path, capsys) -> None:
    """The counts are a fingerprint of one upstream build; a new one must be looked at."""
    obj = tmp_path / "wrong.obj"
    obj.write_text("v 0 0 0\nv 1 1 1\nf 1 2 2\n", encoding="utf-8")
    assert fetch.verify_obj(obj) is False
    assert "DB21_OBJ_ROTATE_XYZ" in capsys.readouterr().out
