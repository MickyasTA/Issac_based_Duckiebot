"""The trajectory-complexity metric and the generator's difficulty knob.

Three properties are worth locking down, and they pull against each other:

* **The metric agrees with the audit.** ``loop_complexity`` reimplements the definitions the
  68-layout audit of ``build/city`` used. If it drifts, every difficulty claim made against it
  silently changes meaning, so a few layouts with hand-checkable geometry pin the numbers.
* **Nominal output never moves.** ``build/city`` was generated with the nominal profile and a
  paused training run references those exact layouts, so regenerating them has to produce the
  same YAML byte for byte. That is asserted against the files on disk when they are present.
* **Hard is measurably harder.** Not "looks harder": the score distributions have to separate
  under the audit's own metric, with the hard minimum above the nominal median.
"""

from __future__ import annotations

import statistics
from pathlib import Path

import pytest

from duckiebot_rl.city import maps as M

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BUILT_MAPS = _REPO_ROOT / "build" / "city" / "maps"


# ------------------------------------------------------------------------------- the metric
def test_metric_matches_the_audit_on_a_hand_checkable_loop() -> None:
    """``loop_small`` is a 2x2 ring: four right turns, four straights, no chicane."""
    metrics = M.loop_complexity(M.builtin_map("loop_small"))
    assert metrics.is_loop
    assert (metrics.loop_tiles, metrics.turns, metrics.chicanes) == (8, 4, 0)
    assert metrics.longest_straight == 1
    assert metrics.intersections == 0
    assert metrics.turn_sequence == "RSRSRSRS"
    assert metrics.score == 4


def test_metric_counts_chicanes_only_where_the_handedness_reverses() -> None:
    """``zigzag`` reverses handedness repeatedly; ``loop_big`` never does."""
    zigzag = M.loop_complexity(M.builtin_map("zigzag"))
    loop_big = M.loop_complexity(M.builtin_map("loop_big"))
    assert set(zigzag.turn_sequence) == {"L", "R", "S"}
    assert zigzag.chicanes > 0
    assert set(loop_big.turn_sequence) == {"R", "S"}
    assert loop_big.chicanes == 0
    assert loop_big.turns == 4
    assert zigzag.score > loop_big.score


def test_metric_reports_intersection_maps_without_a_turn_sequence() -> None:
    """A non-loop layout has no single cyclic turn sequence, and says so rather than guessing."""
    metrics = M.loop_complexity(M.builtin_map("intersection_4way"))
    assert not metrics.is_loop
    assert metrics.intersections == 5
    assert (metrics.loop_tiles, metrics.turns, metrics.chicanes) == (0, 0, 0)
    assert metrics.turn_sequence == ""
    assert metrics.score == 15


def test_longest_straight_is_measured_cyclically() -> None:
    """The run may wrap the start of the cycle, and can never exceed the straight count."""
    for name in M.BUILTIN_MAP_NAMES:
        city = M.builtin_map(name)
        metrics = M.loop_complexity(city)
        if not metrics.is_loop:
            continue
        assert metrics.longest_straight <= metrics.turn_sequence.count("S")
        assert metrics.turns + metrics.turn_sequence.count("S") == metrics.loop_tiles


# --------------------------------------------------------------------------- profile plumbing
def test_only_measured_profiles_are_named() -> None:
    """Shipping a name promises a measured effect; ``easy`` is documented as absent on purpose."""
    assert M.DIFFICULTY_NAMES == ("nominal", "hard")
    assert set(M.DIFFICULTY_PROFILES) == set(M.DIFFICULTY_NAMES)
    assert M.difficulty_profile("HARD ") is M.DIFFICULTY_PROFILES["hard"]
    assert M.difficulty_profile(M.DIFFICULTY_PROFILES["hard"]) is M.DIFFICULTY_PROFILES["hard"]


def test_unknown_difficulty_is_rejected() -> None:
    """A typo must not silently fall back to nominal and produce the wrong city."""
    with pytest.raises(ValueError, match="unknown difficulty"):
        M.difficulty_profile("brutal")
    with pytest.raises(ValueError, match="unknown difficulty"):
        M.variant_maps(2, difficulty="brutal")
    with pytest.raises(ValueError, match="unknown difficulty"):
        M.eval_maps(1, difficulty="brutal")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"shapes": ()}, "no coarse shapes"),
        ({"candidates": 0}, "candidates >= 1"),
        ({"sense": 2}, r"sense in \(1, -1\)"),
        ({"straight_bias": 1.5}, r"straight_bias in \[0, 1\]"),
    ],
)
def test_profile_fields_are_validated(kwargs: dict, match: str) -> None:
    """A malformed profile fails at construction, not thousands of layouts later."""
    base = {"name": "probe", "shapes": ((2, 2, 1),), "straight_bias": 0.5}
    with pytest.raises(ValueError, match=match):
        M.DifficultyProfile(**{**base, **kwargs})


def test_the_nominal_profile_still_describes_the_historical_generator() -> None:
    """Changing any of these fields silently regenerates ``build/city`` into different layouts."""
    nominal = M.DIFFICULTY_PROFILES["nominal"]
    assert nominal.shapes == M._RANDOM_SHAPES
    assert nominal.straight_bias == pytest.approx(0.7)
    assert nominal.candidates == 1
    assert nominal.use_builtins


# ------------------------------------------------------------------- nominal output is frozen
def test_nominal_variants_are_unchanged_by_the_difficulty_knob() -> None:
    """Passing the default explicitly, or not at all, must give identical layouts."""
    implicit = M.variant_maps(24, seed=0)
    explicit = M.variant_maps(24, seed=0, difficulty="nominal")
    assert [c.to_yaml() for c in implicit] == [c.to_yaml() for c in explicit]
    assert [c.to_yaml() for c in M.eval_maps(4)] == [
        c.to_yaml() for c in M.eval_maps(4, difficulty="nominal")
    ]


def test_nominal_metadata_carries_no_difficulty_key() -> None:
    """Nominal maps must serialise exactly as before, so the key is only written when non-default."""
    assert "difficulty" not in M.variant_maps(8)[7].meta
    assert M.variant_maps(8, difficulty="hard")[7].meta["difficulty"] == "hard"
    assert M.eval_maps(1, difficulty="hard", train_count=8)[0].meta["difficulty"] == "hard"


@pytest.mark.skipif(not _BUILT_MAPS.is_dir(), reason="build/city has not been generated here")
def test_regenerating_nominal_reproduces_the_maps_on_disk_byte_for_byte() -> None:
    """The paused training run references ``build/city``; regeneration must not move it."""
    written = sorted(p.stem for p in _BUILT_MAPS.glob("*.yaml"))
    train = M.variant_maps(count=64, seed=0, geometry_buckets=16)
    evaluation = M.eval_maps(count=4)
    assert sorted(c.name for c in train + evaluation) == written
    for city in train + evaluation:
        on_disk = (_BUILT_MAPS / f"{city.name}.yaml").read_text(encoding="utf-8")
        assert city.to_yaml() == on_disk, f"{city.name} would be regenerated differently"


# -------------------------------------------------------------------------------- hard is hard
def test_hard_variants_are_valid_closed_loops_over_many_seeds() -> None:
    """Nothing revalidates a layout once training starts, so every seed has to be buildable."""
    for seed in range(8):
        for city in M.variant_maps(count=8, seed=seed, difficulty="hard"):
            city.validate()
            assert city.is_closed_loop()
            assert M.loop_complexity(city).is_loop
            assert city.half_extent_m <= M.ENV_HALF_EXTENT_M + 1e-9


def test_hard_eval_layouts_are_closed_loops_held_out_from_the_hard_training_set() -> None:
    """A held-out map that recurs in training measures memorisation, not generalisation."""
    train = M.variant_maps(count=32, seed=0, difficulty="hard")
    evaluation = M.eval_maps(count=4, train_count=32, difficulty="hard")
    trained = {c.layout_signature() for c in train}
    assert len({c.layout_signature() for c in evaluation}) == 4
    for city in evaluation:
        city.validate()
        assert city.is_closed_loop()
        assert city.layout_signature() not in trained


def test_hard_and_nominal_score_distributions_separate() -> None:
    """The whole point of the knob: measurably harder under the audit's own metric."""
    nominal = [M.loop_complexity(c).score for c in M.variant_maps(48, seed=0)]
    hard = [M.loop_complexity(c).score for c in M.variant_maps(48, seed=0, difficulty="hard")]
    assert min(hard) > statistics.median(nominal)
    assert statistics.mean(hard) > statistics.mean(nominal) + 5.0
    assert statistics.median(hard) > statistics.median(nominal)


def test_hard_layouts_turn_more_often_and_never_reuse_the_gentle_builtins() -> None:
    """Turn and chicane counts, not just the aggregate, have to move in the right direction."""
    nominal = [M.loop_complexity(c) for c in M.variant_maps(48, seed=0)]
    hard = [M.loop_complexity(c) for c in M.variant_maps(48, seed=0, difficulty="hard")]
    assert statistics.mean([m.turns for m in hard]) > statistics.mean([m.turns for m in nominal])
    assert statistics.mean([m.chicanes for m in hard]) > statistics.mean([m.chicanes for m in nominal])
    # loop_small at variant 0 is the softest layout in the nominal set, and intersection_4way at
    # variant 3 is not a loop at all; the hard profile must not inherit either.
    builtin_signatures = {M.builtin_map(name).layout_signature() for name in M.BUILTIN_MAP_NAMES}
    assert not builtin_signatures & {c.layout_signature() for c in M.variant_maps(48, difficulty="hard")}


def test_hard_layouts_stay_inside_the_spec_grid_range() -> None:
    """SPEC v2 S3.3 caps the grid at 8x8, and S5.1 caps the footprint at the per-env AABB."""
    for city in M.variant_maps(count=32, seed=3, difficulty="hard"):
        assert 5 <= city.n_rows <= 8
        assert 5 <= city.n_cols <= 8
        assert city.half_extent_m <= M.ENV_HALF_EXTENT_M + 1e-9


def test_hard_generation_is_deterministic_in_the_seed() -> None:
    """Same seed, same maps: a rebuild has to be able to reproduce a hard set exactly."""
    first = M.variant_maps(count=16, seed=11, difficulty="hard")
    second = M.variant_maps(count=16, seed=11, difficulty="hard")
    other = M.variant_maps(count=16, seed=12, difficulty="hard")
    assert [c.to_yaml() for c in first] == [c.to_yaml() for c in second]
    assert [c.to_yaml() for c in first] != [c.to_yaml() for c in other]
    assert [c.to_yaml() for c in M.eval_maps(3, difficulty="hard", train_count=16)] == [
        c.to_yaml() for c in M.eval_maps(3, difficulty="hard", train_count=16)
    ]


def test_hard_layouts_are_distinct() -> None:
    """Repeated layouts would silently shrink the training distribution."""
    variants = M.variant_maps(count=64, seed=0, difficulty="hard")
    assert len({c.layout_signature() for c in variants}) == 64
