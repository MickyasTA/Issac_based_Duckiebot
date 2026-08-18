"""Which city stage the live viewer actually shows, and proof the training path did not move.

The bug being locked out: ``MultiUsdFileCfg`` assigns stages to environments by ``index % len``
(SPEC v2 S7.1, critic item D), and the viewer builds exactly one environment. Env 0 can only ever
receive ``usd_paths[0]``. The viewer used to respond to ``--map city_37`` by growing the list to
38 entries, which left ``usd_paths[0] == city_000``, so the requested layout was never on screen
and nothing said so. Selecting a layout therefore has to mean *replacing* the list, not extending
it.

The other half of the file is the guard on the training path: ``CitySettings()`` with no variant
request must still resolve ``city_000 .. city_063`` in that order, because a paused training run
references exactly those stages.

None of this needs Isaac. ``resolve_city_selection`` returns plain settings and
``resolve_city_assets`` is pure path resolution, so both run on a CPU-only runner.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from duckiebot_rl.envs import env_cfg as ec
from duckiebot_rl.envs import viz_env

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CITY_ROOT = _REPO_ROOT / "build" / "city"

_needs_city = pytest.mark.skipif(
    not (_CITY_ROOT / "usd").is_dir(), reason="build/city has not been generated here"
)


# ------------------------------------------------------------------------------ name parsing
@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        ("city_37", "city_037"),
        ("city_037", "city_037"),
        ("city_0", "city_000"),
        ("eval_2", "eval_02"),
        ("eval_02", "eval_02"),
        ("  city_7  ", "city_007"),
        ("loop_small", "loop_small"),
        ("my_custom_stage", "my_custom_stage"),
    ],
)
def test_stage_stem_restores_the_generator_zero_padding(requested: str, expected: str) -> None:
    """A human types ``city_7``; the generator wrote ``city_007``."""
    assert viz_env.stage_stem(requested) == expected


# --------------------------------------------------------------- the training path is frozen
def test_default_settings_still_resolve_the_paused_run_asset_list() -> None:
    """No variant request means the historical 64-entry list, in order, unchanged."""
    city = ec.CitySettings()
    assert city.variant_names is None
    assert city.num_variants == 64


@_needs_city
def test_default_num_envs_64_resolution_is_bit_for_bit_the_historical_list() -> None:
    """The exact list ``MultiUsdFileCfg`` receives for a 64-env training run."""
    usd_paths, map_paths, root = ec.resolve_city_assets(ec.CitySettings())
    expected_stems = [f"city_{index:03d}" for index in range(64)]
    assert [Path(p).stem for p in usd_paths] == expected_stems
    assert [Path(p).stem for p in map_paths] == expected_stems
    assert root == _CITY_ROOT.as_posix()
    # index % len over 64 envs and 64 stages is the identity, which is the property critic item
    # D relies on: every layout gets exactly the same number of envs.
    assert [usd_paths[env % len(usd_paths)] for env in range(64)] == usd_paths


@_needs_city
def test_eval_layouts_still_append_after_the_training_ones() -> None:
    """``include_eval`` is unaffected by the new field."""
    usd_paths, _, _ = ec.resolve_city_assets(ec.CitySettings(), include_eval=True)
    assert [Path(p).stem for p in usd_paths[-4:]] == ["eval_00", "eval_01", "eval_02", "eval_03"]
    assert len(usd_paths) == 68


# ------------------------------------------------------------------- explicit variant_names
def test_variant_names_must_agree_with_the_variant_count() -> None:
    """The two describe the same list; letting them drift is how env 0 got the wrong stage."""
    with pytest.raises(ValueError, match="variant_names has 1 entries but num_variants is 4"):
        ec.CitySettings(num_variants=4, variant_names=("city_007",))


@_needs_city
def test_variant_names_replaces_the_generated_sequence() -> None:
    """One name in, one stage out, and it is the requested one."""
    city = ec.CitySettings(num_variants=1, variant_names=("city_037",))
    usd_paths, map_paths, _ = ec.resolve_city_assets(city)
    assert [Path(p).stem for p in usd_paths] == ["city_037"]
    assert [Path(p).stem for p in map_paths] == ["city_037"]


@_needs_city
def test_a_missing_named_stage_still_reports_the_directories_tried() -> None:
    """The actionable error survives the new path."""
    city = ec.CitySettings(num_variants=1, variant_names=("city_999",))
    with pytest.raises(FileNotFoundError, match=r"scripts/build_city.py"):
        ec.resolve_city_assets(city)


# ------------------------------------------------------------------------ viewer selection
@_needs_city
@pytest.mark.parametrize("requested", ["city_37", "city_037", "city_63", "eval_2", "eval_03"])
def test_the_viewer_puts_the_requested_layout_at_index_zero(requested: str) -> None:
    """The regression itself: env 0 gets the requested stage, not ``city_000``."""
    city = viz_env.resolve_city_selection(requested)
    usd_paths, _, _ = ec.resolve_city_assets(city)
    assert len(usd_paths) == 1
    assert Path(usd_paths[0]).stem == viz_env.stage_stem(requested)


@_needs_city
def test_the_viewer_no_longer_shows_city_000_for_every_request() -> None:
    """Directly contrasts the two behaviours on the same input."""
    shown = Path(ec.resolve_city_assets(viz_env.resolve_city_selection("city_37"))[0][0]).stem
    assert shown == "city_037"
    assert shown != "city_000"


@_needs_city
def test_a_builtin_map_name_resolves_to_the_variant_carrying_that_layout() -> None:
    """``variant_maps`` puts the built-ins at variants 0 to 4; the claim is verified, not assumed."""
    from duckiebot_rl.city.maps import BUILTIN_MAP_NAMES, builtin_map, load_map

    for index, name in enumerate(BUILTIN_MAP_NAMES):
        city = viz_env.resolve_city_selection(name)
        usd_paths, map_paths, _ = ec.resolve_city_assets(city)
        assert Path(usd_paths[0]).stem == f"city_{index:03d}"
        assert load_map(map_paths[0]).layout_signature() == builtin_map(name).layout_signature()


@_needs_city
def test_an_unknown_map_name_falls_back_loudly_rather_than_refusing_to_start(capsys) -> None:
    """The viewer is a debugging tool: a typo must not cost a Kit boot."""
    city = viz_env.resolve_city_selection("no_such_map")
    usd_paths, _, _ = ec.resolve_city_assets(city)
    assert Path(usd_paths[0]).stem == "city_000"
    assert "no generated stage matches" in capsys.readouterr().out


@_needs_city
@pytest.mark.parametrize("subdir,suffix", [("maps", ".yaml"), ("usd", ".usda")])
def test_a_path_selects_both_the_build_root_and_the_layout(subdir: str, suffix: str) -> None:
    """This is how a layout from a non-default build root, such as a hard set, is viewed."""
    source = _CITY_ROOT / subdir / f"city_012{suffix}"
    city = viz_env.resolve_city_selection(source.as_posix())
    assert city.root == _CITY_ROOT.as_posix()
    usd_paths, _, _ = ec.resolve_city_assets(city)
    assert Path(usd_paths[0]).stem == "city_012"


def test_a_path_that_does_not_exist_is_not_treated_as_a_path() -> None:
    """A stale path falls through to name handling instead of pointing at a missing root."""
    assert viz_env._selection_from_path("build/nope/maps/city_000.yaml") is None
    assert viz_env._selection_from_path("city_007") is None
