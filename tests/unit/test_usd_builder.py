"""Unit tests for the text-USD city authoring.

These tests skip when no USD runtime is importable, which is the documented degraded mode:
texture and map generation never need USD. Everything they assert is a clean-room gate
requirement from SPEC v2 S3.3 / S3.4 or a stage-structure requirement from S5.1.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from duckiebot_rl.city import lane_graph as LG
from duckiebot_rl.city import maps as M
from duckiebot_rl.city import spec as S
from duckiebot_rl.city import usd_builder as U
from scripts import build_city
from scripts.check_clean_room import run_gate


@pytest.fixture(scope="module")
def usd() -> U.UsdModules:
    """The pxr modules, or skip the module if none can be found."""
    try:
        return U.ensure_usd()
    except U.UsdUnavailableError as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"no USD runtime: {exc}")


def test_missing_usd_error_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    """The failure mode a fresh clone hits must name the fix, not just the symptom.

    Both discovery paths are blocked, so this exercises the real raise in ``ensure_usd`` rather
    than the class docstring: someone deleting the instructions from the message fails here.
    """

    def no_pxr(source: str) -> U.UsdModules:
        raise ImportError("no module named pxr")

    monkeypatch.setattr(U, "_USD_CACHE", None)
    monkeypatch.setattr(U, "_import_pxr", no_pxr)
    monkeypatch.setattr(U, "_isaac_usd_dir", lambda: None)
    with pytest.raises(U.UsdUnavailableError) as excinfo:
        U.ensure_usd()
    message = str(excinfo.value)
    assert "pip install usd-core" in message
    assert "DUCKIEBOT_ISAAC_USD_DIR" in message
    with pytest.raises(U.UsdUnavailableError, match="pip install usd-core"):
        U.build_city_usda(M.builtin_map("loop_small"), "city.usda")


def test_a_non_usda_extension_is_refused_before_any_authoring() -> None:
    """The clean-room gate bans binary USD, so the builder refuses to write it."""
    with pytest.raises(ValueError, match=r"\.usda"):
        U.build_city_usda(M.builtin_map("loop_small"), "city.usd")


def test_vertex_budget_is_four_points_per_tile(usd: U.UsdModules, tmp_path: Path) -> None:
    """Clean-room rule 3: at most 4 * tile_count + 64 mesh points in a city file."""
    city = M.builtin_map("intersection_4way")
    path = U.build_city_usda(city, tmp_path / "city.usda", max_signs=8)
    stage = usd.Usd.Stage.Open(str(path))
    meshes = [p for p in stage.Traverse() if p.IsA(usd.UsdGeom.Mesh)]
    total = sum(len(usd.UsdGeom.Mesh(p).GetPointsAttr().Get()) for p in meshes)
    assert total == U.city_vertex_count(city, n_signs=8)
    assert total <= 4 * city.n_rows * city.n_cols + 64
    road = [p for p in meshes if p.GetPath().name == "RoadMesh"]
    assert len(road) == 1
    assert len(usd.UsdGeom.Mesh(road[0]).GetPointsAttr().Get()) == 4 * city.n_rows * city.n_cols


def test_sign_cap_protects_the_vertex_budget() -> None:
    """More than 16 textured cards would overflow the 64-point slack, so it is refused."""
    with pytest.raises(ValueError, match="max_signs must be <= 16"):
        U.build_city_usda(M.builtin_map("loop_small"), "x.usda", max_signs=17)


def test_subsets_partition_the_road_mesh(usd: U.UsdModules, tmp_path: Path) -> None:
    """One GeomSubset per tile kind, forming a valid partition, each with a bound material."""
    city = M.builtin_map("intersection_4way")
    path = U.build_city_usda(city, tmp_path / "city.usda")
    stage = usd.Usd.Stage.Open(str(path))
    mesh = usd.UsdGeom.Mesh(stage.GetPrimAtPath("/City/RoadMesh"))
    valid, reason = usd.UsdGeom.Subset.ValidateFamily(mesh, usd.UsdGeom.Tokens.face, "materialBind")
    assert valid, reason
    subsets = usd.UsdGeom.Subset.GetGeomSubsets(mesh, usd.UsdGeom.Tokens.face, "materialBind")
    kinds_present = {tile.kind for _, _, tile in city.iter_tiles()}
    assert {s.GetPrim().GetName().removeprefix("subset_") for s in subsets} == kinds_present
    total_faces = sum(len(s.GetIndicesAttr().Get()) for s in subsets)
    assert total_faces == city.n_rows * city.n_cols
    for subset in subsets:
        bound = usd.UsdShade.MaterialBindingAPI(subset).ComputeBoundMaterial()[0]
        assert bound, subset.GetPrim().GetName()


def test_city_stage_carries_no_colliders(usd: U.UsdModules, tmp_path: Path) -> None:
    """SPEC v2 S3.3: the only physics surface in the scene is the separate ground plane."""
    path = U.build_city_usda(M.builtin_map("loop_big"), tmp_path / "city.usda")
    stage = usd.Usd.Stage.Open(str(path))
    for prim in stage.Traverse():
        assert not prim.HasAPI(usd.UsdPhysics.CollisionAPI), prim.GetPath()
    assert "CollisionAPI" not in path.read_text(encoding="utf-8")


def test_materials_expose_the_three_runtime_dr_scalars(usd: U.UsdModules, tmp_path: Path) -> None:
    """The OmniPBR handles the S7.1 layer-2b reset terms write must exist and be authored."""
    path = U.build_city_usda(
        M.builtin_map("loop_small"),
        tmp_path / "city.usda",
        tint=(0.9, 0.95, 1.0),
        albedo_brightness=1.1,
        roughness=0.6,
    )
    stage = usd.Usd.Stage.Open(str(path))
    shader = usd.UsdShade.Shader(stage.GetPrimAtPath("/City/Looks/mat_straight/OmniPBR"))
    assert shader
    assert tuple(shader.GetInput("diffuse_tint").Get()) == pytest.approx((0.9, 0.95, 1.0))
    assert shader.GetInput("albedo_brightness").Get() == pytest.approx(1.1)
    assert shader.GetInput("reflection_roughness_constant").Get() == pytest.approx(0.6)
    preview = usd.UsdShade.Shader(stage.GetPrimAtPath("/City/Looks/mat_straight/Preview"))
    assert preview.GetIdAttr().Get() == "UsdPreviewSurface"


def test_texture_paths_are_relative_and_forward_slashed(usd: U.UsdModules, tmp_path: Path) -> None:
    """Asset paths must resolve from the .usda file on Windows and Linux alike."""
    path = U.build_city_usda(
        M.builtin_map("loop_small"),
        tmp_path / "usd" / "city.usda",
        texture_dir="../textures/bucket_03",
        sign_texture_dir="../textures/signs",
    )
    text = path.read_text(encoding="utf-8")
    assert "../textures/bucket_03/straight.png" in text
    assert "../textures/signs/sign_stop.png" in text
    assert "\\" not in text.split("asset")[1][:400]


def test_stage_metadata_records_the_geometry(usd: U.UsdModules, tmp_path: Path) -> None:
    """A consumer can recover the pitch and lane width from the stage without the YAML."""
    city = M.builtin_map("loop_small", tile_size=0.6)
    path = U.build_city_usda(city, tmp_path / "city.usda")
    stage = usd.Usd.Stage.Open(str(path))
    root = stage.GetDefaultPrim()
    assert root.GetPath() == "/City"
    assert root.GetCustomDataByKey("duckiebot:map_name") == "loop_small"
    assert root.GetCustomDataByKey("duckiebot:tile_size_m") == pytest.approx(0.6)
    assert usd.UsdGeom.GetStageUpAxis(stage) == usd.UsdGeom.Tokens.z
    assert usd.UsdGeom.GetStageMetersPerUnit(stage) == pytest.approx(1.0)
    # No key may end in "tile": scripts/check_clean_room.py parses the header for an integer
    # tile count, and a float-valued key with such a name used to be read as "zero tiles".
    custom = root.GetCustomDataByKey("duckiebot")
    assert not [key for key in custom if key.endswith("tile")], sorted(custom)
    assert "lane_center_offset_frac" in custom


def test_layer_metadata_declares_the_tile_count_the_gate_reads(usd: U.UsdModules, tmp_path: Path) -> None:
    """Rule 3 of the clean-room gate sizes its budget from ``int tiles = N`` in the header.

    The generator did not author it at all, so every real city file fell back to the gate's
    slack-only budget and was reported as a licensing violation. Assert both the USD-level value
    and the exact text form the gate's regex anchors on.
    """
    from scripts.check_clean_room import _count_tiles

    city = M.builtin_map("intersection_4way")
    path = U.build_city_usda(city, tmp_path / "city.usda")
    stage = usd.Usd.Stage.Open(str(path))
    layer_data = stage.GetRootLayer().customLayerData
    assert layer_data["tiles"] == city.n_rows * city.n_cols
    assert layer_data["rows"] == city.n_rows
    assert layer_data["cols"] == city.n_cols
    text = path.read_text(encoding="utf-8")
    assert f"int tiles = {city.n_rows * city.n_cols}" in text
    assert _count_tiles(text) == city.n_rows * city.n_cols


def test_recorded_lane_width_is_the_width_the_stage_actually_shows(usd: U.UsdModules, tmp_path: Path) -> None:
    """``duckiebot:lane_width_m`` is ``w_ep``, the SPEC v2 S5.4 reward denominator.

    The texture bucket carries its own sampled pitch while the map carries an independently
    chosen ``tile_size``, and the texture is UV-mapped onto the map-pitch quad, so every marking
    scales by ``map_pitch / bucket_pitch``. Writing the bucket's raw ``lane_width_m`` was wrong
    by up to 7.9%. Build the worst pairing the shipped set contains and check the stage against
    the lane graph, which is the other consumer of the same number.
    """
    buckets = S.geometry_buckets(count=16, seed=0, alpha=1.0)
    bucket = max(buckets, key=lambda b: b.tile_pitch_mm)  # widest pitch against the narrowest map
    city = M.builtin_map("loop_small", tile_size=0.570)
    assert bucket.tile_pitch_mm / 1000.0 > city.tile_size
    path = U.build_city_usda(city, tmp_path / "city.usda", spec=bucket)
    stage = usd.Usd.Stage.Open(str(path))
    root = stage.GetDefaultPrim()
    recorded = root.GetCustomDataByKey("duckiebot:lane_width_m")

    graph = LG.BatchedLaneGraph([LG.LaneGraph(city, bucket)])
    assert recorded == pytest.approx(float(graph.lane_width[0]), abs=1e-9)
    assert recorded == pytest.approx(bucket.clear_lane_mm / bucket.tile_pitch_mm * 0.570, abs=1e-9)
    # The bug wrote the bucket's own width, which is more than 3x the M2 5 mm tolerance out.
    assert abs(recorded - bucket.lane_width_m) > 0.015
    # The offset key is scale free, so it survives the rescale unchanged.
    assert root.GetCustomDataByKey("duckiebot:lane_center_offset_frac") == pytest.approx(
        bucket.lane_center_offset_tile, abs=1e-12
    )


def test_walls_are_visual_only_and_optional(usd: U.UsdModules, tmp_path: Path) -> None:
    """Walls are Cube prims (no mesh points, no colliders) and can be turned off."""
    with_walls = U.build_city_usda(M.builtin_map("loop_small"), tmp_path / "a.usda", walls=True)
    without = U.build_city_usda(M.builtin_map("loop_small"), tmp_path / "b.usda", walls=False)
    stage = usd.Usd.Stage.Open(str(with_walls))
    cubes = [p for p in stage.Traverse() if p.GetPath().pathString.startswith("/City/Walls")]
    assert cubes
    assert not usd.Usd.Stage.Open(str(without)).GetPrimAtPath("/City/Walls")


def test_ground_plane_is_the_only_collider(usd: U.UsdModules, tmp_path: Path) -> None:
    """build_ground_usda emits one four-point quad with a collider and a physics material."""
    path = U.build_ground_usda(tmp_path / "ground.usda", half_extent_m=25.0)
    stage = usd.Usd.Stage.Open(str(path))
    mesh = usd.UsdGeom.Mesh(stage.GetPrimAtPath("/Ground/Plane"))
    assert len(mesh.GetPointsAttr().Get()) == 4
    assert mesh.GetPrim().HasAPI(usd.UsdPhysics.CollisionAPI)
    material = usd.UsdPhysics.MaterialAPI(stage.GetPrimAtPath("/Ground/Looks/physics_ground"))
    assert material.GetStaticFrictionAttr().Get() == pytest.approx(1.0)
    assert material.GetDynamicFrictionAttr().Get() == pytest.approx(0.9)
    with pytest.raises(ValueError, match=r"\.usda"):
        U.build_ground_usda(tmp_path / "ground.usd")


def test_every_variant_builds(usd: U.UsdModules, tmp_path: Path) -> None:
    """A sample of the training variants and the eval maps all author cleanly and reopen."""
    for city in M.variant_maps(8) + M.eval_maps(2):
        path = U.build_city_usda(city, tmp_path / f"{city.name}.usda")
        stage = usd.Usd.Stage.Open(str(path))
        assert stage.GetDefaultPrim()
        mesh = usd.UsdGeom.Mesh(stage.GetPrimAtPath("/City/RoadMesh"))
        assert len(mesh.GetPointsAttr().Get()) == 4 * city.n_rows * city.n_cols


# ------------------------------------------------------ the M2 acceptance contract, end to end
@pytest.mark.parametrize(("map_name", "expected_grid"), [("loop_small", (5, 5)), ("city_063", (8, 8))])
def test_generated_city_tree_passes_the_clean_room_gate(
    usd: U.UsdModules, tmp_path: Path, map_name: str, expected_grid: tuple[int, int]
) -> None:
    """SPEC v2 M2: ``check_clean_room.py`` exits 0 on the generated set.

    Nothing used to run the gate against generated output. The builder asserted its own vertex
    invariant in Python, the gate asserted a different one by parsing ``.usda`` text, the two
    disagreed, and both halves stayed green. This test is the contract: it drives the real
    ``build_city.py`` into a temporary ``assets/`` tree (maps, textures, USD stages and the
    manifest, exactly as milestone M2 ships them) and then runs the real gate over it.

    Covers the smallest and the largest layouts the generator emits, because the budget is
    per-file and the 8x8 grid is the one that overflowed.
    """
    out = tmp_path / "assets" / "city"
    manifest = tmp_path / "assets" / "MANIFEST.yaml"
    argv = [
        "--map",
        map_name,
        "--out",
        str(out),
        "--manifest",
        str(manifest),
        "--signs",
        "8",
        "--distractors",
        "2",
    ]
    if map_name.startswith("city_"):
        # The procedural variants are not built-in names, so write the YAML first and pass it.
        variant = next(c for c in M.variant_maps(64) if c.name == map_name)
        yaml_path = tmp_path / f"{map_name}.yaml"
        M.save_map(variant, yaml_path)
        argv[1] = str(yaml_path)
        assert (variant.n_rows, variant.n_cols) == expected_grid
    else:
        builtin = M.builtin_map(map_name)
        assert (builtin.n_rows, builtin.n_cols) == expected_grid

    assert build_city.main(argv) == 0
    assert manifest.is_file()

    result = run_gate(tmp_path, min_files=1)
    assert result.ok, "\n".join(result.violations)
    stages = sorted((out / "usd").glob("*.usda"))
    pngs = sorted((out / "textures").rglob("*.png"))
    assert len(stages) == 2  # the city plus ground.usda
    assert pngs
    assert {p.resolve() for p in stages + pngs} <= result.inspected


def test_the_gate_rejects_a_stage_padded_with_an_imported_mesh(usd: U.UsdModules, tmp_path: Path) -> None:
    """The other direction of the same contract: a real stage plus imported geometry must fail."""
    city = M.builtin_map("loop_small")
    path = U.build_city_usda(city, tmp_path / "usd" / "city.usda")
    budget = 4 * city.n_rows * city.n_cols + 64
    points = ", ".join("(0, 0, 0)" for _ in range(budget + 1))
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f'\ndef Mesh "imported" {{ point3f[] points = [{points}] }}\n')
    result = run_gate(tmp_path)
    assert not result.ok
    assert any("R3" in v for v in result.violations), result.violations
