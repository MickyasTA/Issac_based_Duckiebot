"""Unit tests for the map format, the validator, the built-in layouts and the random generator.

The two properties that matter downstream are: (1) a map that validates can always be built into
a city whose roads join up, and (2) the random generator can never emit an invalid map, because
nothing checks it again once training starts.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from duckiebot_rl.city import maps as M
from duckiebot_rl.city.tiles import DRIVABLE_KINDS, KIND_CONNECTIONS


# ------------------------------------------------------------------------------ tile strings
def test_orientation_letters_are_quarter_turns() -> None:
    """N, E, S, W index 0 .. 3 quarter turns counter-clockwise about +z."""
    assert M.ORIENT_TO_ROT == {"N": 0, "E": 1, "S": 2, "W": 3}
    assert M.parse_tile("straight/N").open_edges == frozenset({"N", "S"})
    assert M.parse_tile("straight/E").open_edges == frozenset({"E", "W"})
    assert M.parse_tile("curve_left/N").open_edges == frozenset({"S", "E"})


def test_right_hand_aliases_are_rotations_of_the_left_hand_shapes() -> None:
    """curve_right and 3way_right normalise onto the canonical shapes."""
    assert M.parse_tile("curve_right/N") == M.Tile("curve", 3)
    assert M.parse_tile("curve_right/N").open_edges == frozenset({"S", "W"})
    assert M.parse_tile("curve_right/E") == M.parse_tile("curve_left/N")
    assert M.parse_tile("3way_right/N").open_edges == frozenset({"N", "S", "E"})
    assert M.parse_tile("3way_right/E") == M.parse_tile("3way_left/W")
    assert M.parse_tile("floor") == M.parse_tile("empty")


def test_tile_string_round_trip_is_stable() -> None:
    """format(parse(x)) parses back to the same tile, for every kind and rotation."""
    for kind in KIND_CONNECTIONS:
        for rot in range(4):
            tile = M.Tile(kind, rot)
            reparsed = M.parse_tile(M.format_tile(tile))
            assert reparsed.kind == tile.kind
            assert reparsed.open_edges == tile.open_edges


def test_bad_tile_strings_raise() -> None:
    """Unknown kinds and orientation letters are rejected with a readable message."""
    with pytest.raises(M.MapValidationError, match="unknown tile kind"):
        M.parse_tile("roundabout/N")
    with pytest.raises(M.MapValidationError, match="unknown orientation"):
        M.parse_tile("straight/Q")
    with pytest.raises(M.MapValidationError, match="unknown tile kind"):
        M.Tile("roundabout")


def test_rotate_edges_and_kind_lookup_are_inverses() -> None:
    """Every drivable connectivity resolves back to a (kind, rotation) pair."""
    for kind in DRIVABLE_KINDS:
        for rot in range(4):
            edges = M.rotate_edges(KIND_CONNECTIONS[kind], rot)
            found_kind, found_rot = M.kind_rot_for_edges(edges)
            assert M.rotate_edges(KIND_CONNECTIONS[found_kind], found_rot) == edges
    with pytest.raises(M.MapValidationError, match="no tile kind"):
        M.kind_rot_for_edges({"N"})


# ---------------------------------------------------------------------------------- geometry
def test_row_zero_is_north_and_column_zero_is_west() -> None:
    """The single place the YAML row flip lives is tile_center_xy; pin its behaviour."""
    city = M.builtin_map("loop_small")
    north_west = city.tile_center_xy(0, 0)
    south_east = city.tile_center_xy(city.n_rows - 1, city.n_cols - 1)
    assert north_west[1] > south_east[1]
    assert north_west[0] < south_east[0]
    assert city.tile_center_xy(0, 0) == pytest.approx(
        (-2.5 * city.tile_size + 0.5 * city.tile_size, 2.5 * city.tile_size - 0.5 * city.tile_size)
    )


def test_cell_lookup_inverts_tile_center() -> None:
    """cell_of_xy recovers the cell that tile_center_xy produced, and rejects outside points."""
    city = M.builtin_map("loop_big")
    for row in range(city.n_rows):
        for col in range(city.n_cols):
            x, y = city.tile_center_xy(row, col)
            assert city.cell_of_xy(x, y) == (row, col)
    assert city.cell_of_xy(100.0, 0.0) is None
    assert city.cell_of_xy(0.0, -100.0) is None


def test_maps_are_centred_and_fit_the_per_env_aabb() -> None:
    """Every generated layout stays inside the 3.6 m half extent of SPEC v2 S5.1."""
    for city in M.variant_maps(64) + M.eval_maps(4):
        assert city.half_extent_m <= M.ENV_HALF_EXTENT_M
        ox, oy = city.origin_xy
        assert ox == pytest.approx(-0.5 * city.n_cols * city.tile_size)
        assert oy == pytest.approx(-0.5 * city.n_rows * city.tile_size)


# -------------------------------------------------------------------------------- built-ins
def test_every_builtin_map_validates() -> None:
    """All five built-ins parse, validate and contain drivable tiles."""
    assert len(M.BUILTIN_MAP_NAMES) == 5
    for name in M.BUILTIN_MAP_NAMES:
        city = M.builtin_map(name)
        city.validate()
        assert city.name == name
        assert city.drivable_cells()
        assert city.n_rows in (5, 7)


def test_builtin_loops_are_closed_and_the_intersection_map_is_not() -> None:
    """The four loop maps form a single cycle; the intersection map is a road network."""
    for name in ("loop_small", "loop_big", "zigzag", "obstacles_dynamic"):
        assert M.builtin_map(name).is_closed_loop(), name
    intersection = M.builtin_map("intersection_4way")
    assert not intersection.is_closed_loop()
    kinds = {tile.kind for _, _, tile in intersection.iter_tiles()}
    assert {"fourway", "threeway", "curve", "straight"} <= kinds


def test_intersection_map_respects_the_duckietown_adjacency_guidance() -> None:
    """No intersection touches a curve or another intersection in the shipped layout."""
    assert M.builtin_map("intersection_4way").topology_warnings() == []


def test_topology_warnings_fire_when_they_should() -> None:
    """A hand-made map with a 4-way next to a curve reports the soft violation."""
    city = M.map_from_rows(
        "tight",
        (
            "grass grass        grass       grass        grass",
            "grass curve_left/N 3way_left/E curve_left/W grass",
            "grass straight/N   straight/N  straight/N   grass",
            "grass curve_left/E 3way_left/W curve_left/S grass",
            "grass grass        grass       grass        grass",
        ),
    )
    warnings = city.topology_warnings()
    assert warnings
    assert all("intersection at" in line for line in warnings)


def test_obstacles_map_carries_movers_and_static_props() -> None:
    """The dynamic-obstacle map ships the S5.1 obstacle budget with two movers."""
    city = M.builtin_map("obstacles_dynamic")
    assert len(city.objects) == 10
    movers = [o for o in city.objects if not o.static]
    assert len(movers) >= 2
    assert {o.kind for o in city.objects} == {"duckiebot", "duckie", "cone"}
    for obj in city.objects:
        assert abs(obj.x) <= city.half_extent_m and abs(obj.y) <= city.half_extent_m


def test_unknown_builtin_name_raises() -> None:
    """Asking for a map that does not exist lists the ones that do."""
    with pytest.raises(KeyError, match="unknown built-in map"):
        M.builtin_map("nonexistent")


# ------------------------------------------------------------------------------- validation
def test_validator_rejects_a_dangling_open_edge() -> None:
    """A road that opens onto grass is a hole in the track and must not validate."""
    city = M.builtin_map("loop_small")
    city.tiles[1][2] = M.Tile("straight", 0)  # was straight/E, now opens north onto grass
    with pytest.raises(M.MapValidationError, match="closed on its"):
        city.validate()


def test_validator_rejects_a_road_leaving_the_grid() -> None:
    """A road on the border opening outward is caught before it can be built."""
    grid = [[M.Tile("grass") for _ in range(3)] for _ in range(3)]
    grid[1][1] = M.Tile("straight", 0)
    with pytest.raises(M.MapValidationError, match="closed on its"):
        M.CityMap("edge", grid).validate()
    with pytest.raises(M.MapValidationError, match="off the edge of the grid"):
        M.CityMap("tiny", [[M.Tile("straight", 0)]]).validate()


def test_validator_rejects_ragged_and_empty_grids() -> None:
    """Structural problems are reported before any geometry is derived."""
    with pytest.raises(M.MapValidationError, match="empty tile grid"):
        M.CityMap("empty", []).validate()
    ragged = [[M.Tile("grass")], [M.Tile("grass"), M.Tile("grass")]]
    with pytest.raises(M.MapValidationError, match="ragged"):
        M.CityMap("ragged", ragged).validate()
    with pytest.raises(M.MapValidationError, match="no drivable tiles"):
        M.CityMap("bare", [[M.Tile("grass")] * 2] * 2).validate()


def test_validator_rejects_an_oversized_footprint() -> None:
    """A map larger than the per-env AABB cannot be spawned at env_spacing 8.0."""
    city = M.builtin_map("loop_big")
    city.tile_size = 2.0
    with pytest.raises(M.MapValidationError, match="exceeds the per-env"):
        city.validate()


def test_map_from_cycle_rejects_broken_cycles() -> None:
    """Non-adjacent, repeated and too-short cycles are rejected."""
    with pytest.raises(M.MapValidationError, match="not adjacent"):
        M.map_from_cycle("bad", 5, 5, [(1, 1), (1, 2), (3, 3), (2, 1)])
    with pytest.raises(M.MapValidationError, match="repeats a cell"):
        M.map_from_cycle("bad", 5, 5, [(1, 1), (1, 2), (1, 1), (1, 2)])
    with pytest.raises(M.MapValidationError, match="at least 4 cells"):
        M.map_from_cycle("bad", 5, 5, [(1, 1), (1, 2)])
    with pytest.raises(M.MapValidationError, match="outside the"):
        M.map_from_cycle("bad", 5, 5, [(0, 0), (0, 1), (-1, 1), (-1, 0)])


# ------------------------------------------------------------------------------------ YAML
def test_yaml_round_trip_preserves_every_field(tmp_path: Path) -> None:
    """Saving and loading a map reproduces its tiles, pitch, objects and metadata."""
    original = M.builtin_map("obstacles_dynamic")
    path = M.save_map(original, tmp_path / "map.yaml")
    loaded = M.load_map(path)
    assert loaded.name == original.name
    assert loaded.tile_size == pytest.approx(original.tile_size)
    assert loaded.meta == original.meta
    assert len(loaded.objects) == len(original.objects)
    for a, b in zip(loaded.objects, original.objects, strict=True):
        assert a.kind == b.kind and a.static == b.static
        assert (a.x, a.y, a.yaw_deg) == pytest.approx((b.x, b.y, b.yaw_deg))
    for row_a, row_b in zip(loaded.tiles, original.tiles, strict=True):
        assert [M.format_tile(t) for t in row_a] == [M.format_tile(t) for t in row_b]


def test_yaml_document_shape_is_mapformat1_like(tmp_path: Path) -> None:
    """The document has the documented keys and a grid of tile strings."""
    path = M.save_map(M.builtin_map("loop_small"), tmp_path / "loop.yaml")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert set(data) >= {"name", "tile_size", "tiles", "objects"}
    assert data["tiles"][1] == ["grass", "curve_left/N", "straight/E", "curve_left/W", "grass"]
    assert data["tiles"][2][2] == "asphalt"


def test_loader_accepts_the_vector_tile_size_form(tmp_path: Path) -> None:
    """A ``tile_size: {x: .., y: ..}`` document loads; a non-square one does not."""
    data = M.builtin_map("loop_small").to_dict()
    data["tile_size"] = {"x": 0.585, "y": 0.585}
    assert M.CityMap.from_dict(data).tile_size == pytest.approx(0.585)
    data["tile_size"] = {"x": 0.585, "y": 0.6}
    with pytest.raises(M.MapValidationError, match="non-square"):
        M.CityMap.from_dict(data)


def test_loader_reports_bad_documents(tmp_path: Path) -> None:
    """Missing keys and non-mapping documents give actionable errors."""
    path = tmp_path / "bad.yaml"
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(M.MapValidationError, match="expected a YAML mapping"):
        M.load_map(path)
    with pytest.raises(M.MapValidationError, match="no 'tiles' key"):
        M.CityMap.from_dict({"name": "x"})
    with pytest.raises(M.MapValidationError, match="'kind' and 'pose'"):
        M.MapObject.from_dict({"kind": "duckie"})
    with pytest.raises(M.MapValidationError, match="pose must be"):
        M.MapObject.from_dict({"kind": "duckie", "pose": [0.0, 0.0]})


def test_name_falls_back_to_the_file_stem(tmp_path: Path) -> None:
    """A document without a name is named after its file."""
    data = M.builtin_map("loop_small").to_dict()
    data.pop("name")
    path = tmp_path / "my_track.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert M.load_map(path).name == "my_track"


# ------------------------------------------------------------------------ random generation
@pytest.mark.parametrize("shape", [(1, 1), (2, 2), (2, 3), (3, 2), (3, 3)])
def test_random_generator_always_produces_a_valid_closed_loop(shape: tuple[int, int]) -> None:
    """Across many seeds and shapes every generated map validates and is one closed loop."""
    rows, cols = shape
    for seed in range(60):
        city = M.random_loop_map("r", seed=seed, coarse_rows=rows, coarse_cols=cols)
        city.validate()
        assert city.is_closed_loop(), (shape, seed)
        assert len(city.drivable_cells()) == 4 * rows * cols
        assert city.n_rows == 2 * rows + 2
        assert city.n_cols == 2 * cols + 2


def test_random_generator_is_deterministic_and_seed_sensitive() -> None:
    """The same seed reproduces the layout; different seeds usually do not."""
    a = M.random_loop_map("a", seed=11, coarse_rows=3, coarse_cols=3)
    b = M.random_loop_map("b", seed=11, coarse_rows=3, coarse_cols=3)
    assert a.to_dict()["tiles"] == b.to_dict()["tiles"]
    layouts = {
        tuple(map(tuple, M.random_loop_map("x", seed=s, coarse_rows=3, coarse_cols=3).to_dict()["tiles"]))
        for s in range(24)
    }
    assert len(layouts) > 12


def test_random_generator_rejects_a_missing_border() -> None:
    """Without a grass border the loop would open off the grid, so border 0 is refused."""
    with pytest.raises(M.MapValidationError, match="border must be >= 1"):
        M.random_loop_map("r", border=0)
    with pytest.raises(M.MapValidationError, match="must be >= 1"):
        M.random_loop_map("r", coarse_rows=0)


def test_generated_loops_use_both_straights_and_curves() -> None:
    """A degenerate generator that emitted only curves would still be a loop; check it does not."""
    kinds: set[str] = set()
    for seed in range(10):
        city = M.random_loop_map("r", seed=seed, coarse_rows=3, coarse_cols=3)
        kinds |= {tile.kind for _, _, tile in city.iter_tiles() if tile.drivable}
    assert kinds == {"straight", "curve"}


# --------------------------------------------------------------------------------- variants
def test_variant_set_is_deterministic_and_covers_every_geometry_bucket() -> None:
    """64 variants, deterministic, with bucket i % 16 and a mix of tile pitches."""
    first = M.variant_maps(64, seed=0)
    second = M.variant_maps(64, seed=0)
    assert len(first) == 64
    assert [c.name for c in first] == [f"city_{i:03d}" for i in range(64)]
    assert [c.to_dict()["tiles"] for c in first] == [c.to_dict()["tiles"] for c in second]
    assert {c.meta["geometry_bucket"] for c in first} == set(range(16))
    assert {c.meta["variant_index"] for c in first} == set(range(64))
    assert len({round(c.tile_size, 4) for c in first}) == 4
    assert all(c.validate() is c for c in first)


def test_first_variants_are_the_readable_built_ins() -> None:
    """Debug runs at num_envs <= 5 always see the hand-authored layouts."""
    variants = M.variant_maps(8, seed=0)
    for i, name in enumerate(M.BUILTIN_MAP_NAMES):
        reference = M.builtin_map(name, tile_size=variants[i].tile_size)
        assert variants[i].to_dict()["tiles"] == reference.to_dict()["tiles"]


def test_eval_maps_are_disjoint_from_the_training_variants() -> None:
    """The four held-out layouts never coincide with a training layout."""
    training = {tuple(map(tuple, c.to_dict()["tiles"])) for c in M.variant_maps(64, seed=0)}
    evaluation = M.eval_maps(4)
    assert len(evaluation) == 4
    assert [c.name for c in evaluation] == [f"eval_{i:02d}" for i in range(4)]
    for city in evaluation:
        city.validate()
        assert city.meta["eval"] is True
        assert tuple(map(tuple, city.to_dict()["tiles"])) not in training


def test_layouts_are_deduplicated() -> None:
    """Only the deliberate loop_big / obstacles_dynamic pair shares a layout."""
    variants = M.variant_maps(64, seed=0)
    signatures = [c.layout_signature() for c in variants]
    duplicates = [i for i, sig in enumerate(signatures) if signatures.count(sig) > 1]
    assert duplicates == [1, 4]  # loop_big and obstacles_dynamic are the same track by design
    assert len(set(signatures)) == 63


def test_eval_maps_accept_an_explicit_exclusion_set() -> None:
    """Callers can hold out against any layout set they choose."""
    forbidden = {M.builtin_map("loop_small").layout_signature()}
    evaluation = M.eval_maps(3, exclude=forbidden)
    assert all(c.layout_signature() not in forbidden for c in evaluation)
    assert len({c.layout_signature() for c in evaluation}) == 3


def test_variant_arguments_are_checked() -> None:
    """Non-positive counts are rejected rather than silently returning an empty list."""
    with pytest.raises(ValueError, match="must be > 0"):
        M.variant_maps(0)
    with pytest.raises(ValueError, match="must be > 0"):
        M.variant_maps(4, geometry_buckets=0)
