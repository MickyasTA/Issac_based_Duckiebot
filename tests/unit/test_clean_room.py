"""Self-tests for the clean-room asset gate (SPEC v2 S3.4).

A gate that cannot fail protects nothing, and a gate that inspects nothing is worse: it reports
PASS. These tests therefore do three things.

1. They contaminate small fixture trees and assert that each rule fires.
2. They build the fixtures the *real* generator produces, by calling
   :func:`duckiebot_rl.city.usd_builder.build_city_usda`, rather than hand-writing ``.usda``
   text. Hand-written fixtures are how the R3 contract silently diverged: the fixtures wrote
   ``customLayerData = { int tiles = 25 }``, which the generator never emitted, so eight self
   tests stayed green while every one of the 68 real city files was flagged.
3. They assert the gate cannot go vacuous again: the generator's default output directory must
   not be shadowed by :data:`~scripts.check_clean_room.EXCLUDED_DIRS`, and every asset the
   generator writes must be opened by a content rule.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import build_city
from scripts.check_clean_room import (
    EXCLUDED_DIRS,
    GENERATED_OUTPUT_DIRS,
    _check_configuration,
    run_gate,
)


def _rules(violations: list[str]) -> set[str]:
    """Extract the rule identifiers from violation lines.

    Args:
        violations: Violation strings of the form ``"[R1] path: message"``.

    Returns:
        The set of rule identifiers that fired.
    """
    return {line.split("]")[0].lstrip("[") for line in violations}


@pytest.fixture(scope="module")
def usd():
    """The pxr modules, or skip every test that needs to author a real stage."""
    from duckiebot_rl.city import usd_builder

    try:
        return usd_builder.ensure_usd()
    except usd_builder.UsdUnavailableError as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"no USD runtime: {exc}")


def _real_city_usda(directory: Path, map_name: str = "loop_small") -> Path:
    """Author one city stage with the production builder.

    Args:
        directory: Where to write it.
        map_name: Built-in map to build.

    Returns:
        The written ``.usda`` path.
    """
    from duckiebot_rl.city import maps, usd_builder

    return usd_builder.build_city_usda(maps.builtin_map(map_name), directory / f"{map_name}.usda")


# ------------------------------------------------------------------ anti-vacuity (rule 0)
def test_the_gate_configuration_cannot_hide_generated_output() -> None:
    """No generator output directory may be shadowed by an exclusion.

    This is the exact defect that made the whole gate vacuous: ``EXCLUDED_DIRS`` held ``"build"``
    while ``build_city.py`` defaulted to ``--out build/city``.
    """
    for entry in GENERATED_OUTPUT_DIRS:
        for part in entry.split("/"):
            assert part not in EXCLUDED_DIRS, entry
    _check_configuration()  # raises if the two constants ever disagree again


def test_the_gate_knows_the_generator_default_output_directory(monkeypatch) -> None:
    """The gate's generator-output list must track ``build_city.py``'s actual ``--out`` default.

    Read out of the live parser rather than copied, so moving the default breaks this test
    instead of quietly taking the generated tree out of rule 0's reach.
    """
    captured: dict[str, object] = {}

    def capture(args: object) -> int:
        captured["args"] = args
        return 0

    monkeypatch.setattr(build_city, "build", capture)
    build_city.main(["--builtin"])
    default_out = Path(captured["args"].out).as_posix()  # type: ignore[attr-defined]
    assert default_out in GENERATED_OUTPUT_DIRS, (
        f"scripts/build_city.py defaults to --out {default_out!r}, which the clean-room gate "
        f"does not list in GENERATED_OUTPUT_DIRS {GENERATED_OUTPUT_DIRS}; rule 0 would not "
        f"notice if that tree were skipped"
    )


def test_an_uninspected_generated_tree_fails_the_gate(tmp_path: Path, usd) -> None:
    """Rule 0 fires when generated assets exist but no content rule opened them."""
    from scripts import check_clean_room as gate

    outdir = tmp_path / "build" / "city" / "usd"
    outdir.mkdir(parents=True)
    _real_city_usda(outdir)
    (tmp_path / "build" / "city" / "MANIFEST.yaml").write_text("entries: {}\n", encoding="utf-8")

    assert run_gate(tmp_path).ok, run_gate(tmp_path).violations

    # Simulate the regression: exclude the generated tree from the content walk only.
    original = gate.EXCLUDED_DIRS
    try:
        gate.EXCLUDED_DIRS = original | {"build"}
        result = run_gate(tmp_path)
    finally:
        gate.EXCLUDED_DIRS = original
    assert not result.ok
    assert _rules(result.violations) == {"R0"}
    assert "never inspected" in result.violations[0]


def test_the_minimum_inspection_floor_fires_on_an_empty_tree(tmp_path: Path) -> None:
    """A tree with nothing to read must not be certified clean."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "notes.md").write_text("no assets here", encoding="utf-8")
    assert run_gate(tmp_path).ok
    result = run_gate(tmp_path, min_files=1)
    assert not result.ok
    assert _rules(result.violations) == {"R0"}
    assert "inspected only 0 file(s)" in result.violations[0]


# --------------------------------------------------------------------------- rules 1 to 4
def test_clean_tree_passes(tmp_path: Path, usd) -> None:
    """A tree of text assets and real generated stages is clean."""
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "duckiebot.urdf").write_text(
        "<robot name='db'><link name='base_link'/></robot>", encoding="utf-8"
    )
    _real_city_usda(tmp_path / "assets")
    result = run_gate(tmp_path, min_files=2)
    assert result.ok, result.violations
    assert result.scanned == 2


def test_binary_mesh_formats_fire_rule_one(tmp_path: Path) -> None:
    """Every banned geometry container is caught, wherever it hides."""
    (tmp_path / "assets" / "meshes").mkdir(parents=True)
    for name in ("duckiebot.obj", "city.glb", "wheel.stl", "scene.usdc", "robot.usd"):
        (tmp_path / "assets" / "meshes" / name).write_bytes(b"\x00binary")
    result = run_gate(tmp_path)
    assert not result.ok
    assert _rules(result.violations) == {"R1"}
    assert len(result.violations) == 5


def test_a_container_renamed_to_png_fires_rule_one(tmp_path: Path) -> None:
    """Renaming a mesh to .png must not smuggle it past the format rule."""
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "tile.png").write_bytes(b"v 0.0 0.0 0.0\nv 1.0 0.0 0.0\n")
    (tmp_path / "assets" / "MANIFEST.yaml").write_text(
        "tile.png: generated by duckiebot_rl.city.tiles\n", encoding="utf-8"
    )
    result = run_gate(tmp_path)
    assert not result.ok
    assert "R1" in _rules(result.violations)
    assert "PNG signature" in result.violations[0]


def test_upstream_string_in_an_asset_fires_rule_two(tmp_path: Path) -> None:
    """A converted upstream artifact usually still carries its provenance string."""
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "tile.usda").write_text(
        '#usda 1.0\ndef Xform "tile" { string source = "gym-duckietown/meshes/tile" }\n',
        encoding="utf-8",
    )
    result = run_gate(tmp_path)
    assert not result.ok
    assert "R2" in _rules(result.violations)


def test_a_real_generated_stage_carries_no_provenance_string(tmp_path: Path, usd) -> None:
    """Rule 2 must be satisfied by what the generator actually writes, not by a fixture."""
    path = _real_city_usda(tmp_path)
    assert "duckietown" not in path.read_text(encoding="utf-8").lower()
    assert run_gate(tmp_path, min_files=1).ok


def test_prose_mentions_are_allowed(tmp_path: Path) -> None:
    """The acknowledgement in README and NOTICE is legitimate and must not fire the gate."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "README.md").write_text(
        "Dimensional facts from the Duckietown specification.", encoding="utf-8"
    )
    (tmp_path / "NOTICE").write_text("No Duckietown assets are redistributed.", encoding="utf-8")
    (tmp_path / "docs" / "architecture.md").write_text("Duckietown tile pitch is 0.585 m.", encoding="utf-8")
    result = run_gate(tmp_path)
    assert result.ok, result.violations


def test_robot_usda_may_not_contain_a_mesh_prim(tmp_path: Path) -> None:
    """Robot geometry is primitives only; a Mesh prim means an imported asset."""
    (tmp_path / "assets" / "usd").mkdir(parents=True)
    (tmp_path / "assets" / "usd" / "robot_body.usda").write_text(
        '#usda 1.0\ndef Mesh "chassis" { point3f[] points = [(0,0,0), (1,0,0)] }\n',
        encoding="utf-8",
    )
    result = run_gate(tmp_path)
    assert not result.ok
    assert "R3" in _rules(result.violations)


@pytest.mark.parametrize("map_name", ["loop_small", "intersection_4way", "loop_big"])
def test_real_generated_stages_are_within_the_vertex_budget(tmp_path: Path, usd, map_name: str) -> None:
    """Rule 3 must pass on the generator's own output, which is what M2 signs off.

    The old fixture wrote ``customLayerData = { int tiles = 25 }`` by hand and the old regex read
    ``tile = 0`` out of the generator's float metadata, so this direction of the contract was
    never exercised. It is now, on three real layouts.
    """
    from duckiebot_rl.city import maps

    directory = tmp_path / "usd"
    directory.mkdir()
    path = _real_city_usda(directory, map_name)
    city = maps.builtin_map(map_name)
    text = path.read_text(encoding="utf-8")
    assert f"int tiles = {city.n_rows * city.n_cols}" in text
    result = run_gate(tmp_path, min_files=1)
    assert result.ok, result.violations


def test_an_imported_mesh_appended_to_a_real_stage_fires_rule_three(tmp_path: Path, usd) -> None:
    """Padding a legitimate stage with an imported mesh must be caught."""
    path = _real_city_usda(tmp_path)
    points = ", ".join("(0, 0, 0)" for _ in range(600))
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f'\ndef Mesh "imported" {{ point3f[] points = [{points}] }}\n')
    result = run_gate(tmp_path)
    assert not result.ok
    assert "R3" in _rules(result.violations)
    assert "exceed the quad-per-tile budget" in result.violations[0]


def test_a_float_key_ending_in_tile_can_no_longer_be_read_as_a_tile_count(tmp_path: Path) -> None:
    """The R3 regex must never again parse ``lane_center_offset_tile = 0.2`` as "zero tiles"."""
    from scripts.check_clean_room import _count_tiles

    assert _count_tiles('(\n    double lane_center_offset_tile = 0.2\n)\ndef Xform "C" {}\n') is None
    assert _count_tiles('(\n    int tiles = 36\n)\ndef Xform "C" {}\n') == 36
    # A hint smuggled inside a prim body is ignored: only the layer header counts.
    assert _count_tiles('#usda 1.0\ndef Mesh "m" {\n int tiles = 9999\n}\n') is None


def test_an_undeclared_usda_gets_the_slack_and_nothing_more(tmp_path: Path) -> None:
    """A .usda that declares no tile count is not from this generator and gets 64 points."""
    (tmp_path / "assets").mkdir()
    points = ", ".join("(0, 0, 0)" for _ in range(65))
    (tmp_path / "assets" / "unknown.usda").write_text(
        f'#usda 1.0\ndef Mesh "m" {{ point3f[] points = [{points}] }}\n', encoding="utf-8"
    )
    result = run_gate(tmp_path)
    assert not result.ok
    assert "R3" in _rules(result.violations)
    assert "declares no 'int tiles = N'" in result.violations[0]


def test_unmanifested_png_fires_rule_four(tmp_path: Path) -> None:
    """Images must name their generator or their BSD-2 source."""
    (tmp_path / "assets" / "textures").mkdir(parents=True)
    (tmp_path / "assets" / "textures" / "yellow_dash.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
    result = run_gate(tmp_path)
    assert not result.ok
    assert "R4" in _rules(result.violations)

    (tmp_path / "assets" / "MANIFEST.yaml").write_text(
        "textures/yellow_dash.png: generated by duckiebot_rl/city/tiles.py\n", encoding="utf-8"
    )
    assert run_gate(tmp_path).ok


def test_a_manifest_key_from_another_directory_does_not_launder_an_image(tmp_path: Path) -> None:
    """Rule 4 resolves keys as paths; a bare name match would let any image inherit provenance."""
    (tmp_path / "assets" / "textures").mkdir(parents=True)
    (tmp_path / "assets" / "textures" / "straight.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
    (tmp_path / "assets" / "MANIFEST.yaml").write_text(
        "entries:\n  city/textures/bucket_00/straight.png:\n    generator: render_tile\n",
        encoding="utf-8",
    )
    result = run_gate(tmp_path)
    assert not result.ok
    assert "R4" in _rules(result.violations)


def test_research_and_reference_trees_are_excluded(tmp_path: Path) -> None:
    """The contaminated prototypes are dimensional reference only and are gitignored."""
    (tmp_path / "_research" / "prototypes" / "db2").mkdir(parents=True)
    (tmp_path / "_research" / "prototypes" / "db2" / "duckiebot.obj").write_bytes(b"v 0 0 0")
    (tmp_path / "_refs").mkdir()
    (tmp_path / "_refs" / "jetbot.usd").write_bytes(b"\x00")
    result = run_gate(tmp_path)
    assert result.ok, result.violations


# ------------------------------------------------------------------------- the real repository
def test_the_real_repository_is_clean(repo_root: Path) -> None:
    """The actual repository passes its own gate. This is the executable license claim."""
    result = run_gate(repo_root, min_files=1)
    assert result.ok, "\n".join(result.violations)


def test_the_real_repository_gate_reads_every_generated_asset(repo_root: Path) -> None:
    """When the M2 tree has been built, the gate must open all of it, not a subset.

    Skipped on a bare checkout, where the generated tree is gitignored and absent; CI builds it
    first, so the CI leg always exercises this.
    """
    generated = repo_root / "build" / "city"
    if not generated.is_dir():
        pytest.skip("run scripts/build_city.py --all --out build/city first")
    on_disk = {
        p.resolve() for p in generated.rglob("*") if p.is_file() and p.suffix.lower() in {".usda", ".png"}
    }
    assert len(on_disk) > 100, len(on_disk)
    result = run_gate(repo_root, min_files=len(on_disk))
    assert result.ok, "\n".join(result.violations)
    assert on_disk <= result.inspected
