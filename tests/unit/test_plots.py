"""Unit tests for the plotting layer: the metric table, the smoothing, and every panel.

The panels are tested against the shapes real runs actually produce, including the ugly ones: no
rows at all, a single row, a column that is entirely NaN, a key that only starts appearing halfway
through. A dashboard that raises on one of those is worse than no dashboard, because it fails
exactly when a run has gone wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from duckiebot_rl.viz import plots
from duckiebot_rl.viz.plots import (
    PALETTE,
    PANELS,
    SERIES,
    MetricTable,
    build_panels,
    ema,
    fmt_duration,
    fmt_int,
    fmt_metric,
    matplotlib_available,
    prettify,
)

pytestmark = pytest.mark.skipif(not matplotlib_available(), reason="needs the [viz] extra (matplotlib)")


# ------------------------------------------------------------------------------------- fixtures


def make_rows(count: int = 40) -> list[dict[str, float]]:
    """Build a plausible metrics table.

    Args:
        count: Number of iterations.

    Returns:
        Rows in file order.
    """
    rows = []
    for index in range(1, count + 1):
        progress = index / count
        rows.append(
            {
                "iteration": index,
                "total_timesteps": index * 4096,
                "wall_clock_s": index * 12.0,
                "ep_return_mean": 100.0 * progress,
                "ep_return_std": 10.0 * (1.0 - progress),
                "ep_len_mean": 200.0 + 500.0 * progress,
                "policy_loss": -0.02 * (1.0 - progress),
                "value_loss": 40.0 * (1.0 - progress) + 1.0,
                "entropy": 2.3 - progress,
                "approx_kl": 0.014 + 0.002 * progress,
                "clipfrac": 0.1 + 0.05 * progress,
                "target_kl": 0.015,
                "explained_variance": -0.3 + 1.2 * progress,
                "grad_norm": 2.0 - progress,
                "learning_rate": 3e-4 * (1.0 - 0.5 * progress),
                "lane_dev_rms_m": 0.12 * (1.0 - progress) + 0.01,
                "lane_dev_max_m": 0.25 * (1.0 - progress) + 0.03,
                "success_rate": progress,
                "alpha_vis": min(1.0, 0.05 * index),
                "alpha_dyn": min(1.0, 0.03 * index),
            }
        )
    return rows


@pytest.fixture
def axes():
    """Yield a fresh axes on a throwaway figure.

    Yields:
        A matplotlib axes created from a 1x1 gridspec, which is what the panels that split their
        cell need.
    """
    with plots.style_context():
        figure = plots.plt.figure(figsize=(6, 3), dpi=60)
        grid = figure.add_gridspec(1, 1)
        yield figure.add_subplot(grid[0, 0])
        plots.plt.close(figure)


# --------------------------------------------------------------------------------------- palette


def test_the_palette_slots_are_distinct_and_are_never_cycled():
    assert len(SERIES) == len(set(SERIES)) == 8
    assert SERIES[0] == "#2a78d6"
    assert all(colour.startswith("#") and len(colour) == 7 for colour in SERIES)


def test_status_colours_are_not_reused_as_series_colours():
    status = {PALETTE.good, PALETTE.warning, PALETTE.serious, PALETTE.critical}
    assert status.isdisjoint(set(SERIES))


# ------------------------------------------------------------------------------------ formatting


def test_fmt_int_handles_missing_and_non_finite():
    assert fmt_int(1234567) == "1,234,567"
    assert fmt_int(None) == "-"
    assert fmt_int(float("nan")) == "-"


def test_fmt_metric_keeps_magnitude():
    assert fmt_metric(0.123456) == "0.1235"
    assert fmt_metric(None) == "-"
    assert fmt_metric(float("nan")) == "nan"
    assert "e" in fmt_metric(3e-7)
    assert "e" in fmt_metric(2.5e9)


def test_fmt_duration_adds_days_when_needed():
    assert fmt_duration(0) == "0:00:00"
    assert fmt_duration(3661) == "1:01:01"
    assert fmt_duration(2 * 86400 + 3661) == "2d 01:01:01"
    assert fmt_duration(None) == "-"
    assert fmt_duration(float("nan")) == "-"


def test_prettify_turns_keys_into_titles():
    assert prettify("lane_dev_rms_m") == "lane dev rms m"


# ------------------------------------------------------------------------------------- smoothing


def test_ema_follows_a_constant_series():
    smoothed = ema(np.full(20, 3.0), alpha=0.3)
    assert np.allclose(smoothed, 3.0)


def test_ema_keeps_nan_gaps_but_carries_state_across_them():
    values = np.array([1.0, np.nan, 1.0, 1.0])
    smoothed = ema(values, alpha=0.5)
    assert np.isnan(smoothed[1]), "a gap in the raw series must stay a gap in the smoothed one"
    assert np.isfinite(smoothed[2])
    assert smoothed[0] == pytest.approx(1.0)


def test_ema_of_an_all_nan_column_is_all_nan():
    assert np.all(np.isnan(ema(np.full(5, np.nan))))


def test_ema_of_an_empty_series_is_empty():
    assert ema(np.array([])).shape == (0,)


# ----------------------------------------------------------------------------------- the table


def test_table_columns_are_nan_filled_for_missing_keys():
    table = MetricTable.from_rows([{"iteration": 1, "a": 1.0}, {"iteration": 2, "b": 2.0}])
    assert np.isnan(table.column("a")[1])
    assert np.isnan(table.column("b")[0])
    assert table.has("a") and table.has("b")
    assert not table.has("never_logged")
    assert np.all(np.isnan(table.column("never_logged")))


def test_table_ignores_non_numeric_values():
    table = MetricTable.from_rows([{"iteration": 1, "note": "hello", "flag": True}])
    assert "note" not in table.columns
    assert table.column("flag")[0] == 1.0


def test_table_x_falls_back_to_a_counter_when_iteration_is_missing():
    table = MetricTable.from_rows([{"a": 1.0}, {"a": 2.0}, {"a": 3.0}])
    assert list(table.x) == [1.0, 2.0, 3.0]


def test_table_last_returns_the_most_recent_finite_value():
    table = MetricTable.from_rows([{"a": 1.0}, {"a": float("nan")}, {"a": None}])
    assert table.last("a") == pytest.approx(1.0)
    assert table.last("missing") is None


def test_extra_keys_are_discovered_in_first_seen_order():
    table = MetricTable.from_rows([{"iteration": 1, "ep_return_mean": 1.0, "zeta": 1.0, "alpha_extra": 2.0}])
    assert table.extra_keys(("iteration", "ep_return_mean")) == ["zeta", "alpha_extra"]


def test_empty_table_is_safe():
    table = MetricTable.from_rows([])
    assert len(table) == 0
    assert table.x.shape == (0,)
    assert table.column("anything").shape == (0,)
    assert table.last("anything") is None


# --------------------------------------------------------------------------------- panel drawing


@pytest.mark.parametrize("panel", PANELS, ids=[panel.slug for panel in PANELS])
def test_every_panel_draws_a_full_run(panel, axes):
    panel.draw(axes, MetricTable.from_rows(make_rows()))


@pytest.mark.parametrize("panel", PANELS, ids=[panel.slug for panel in PANELS])
def test_every_panel_survives_zero_rows(panel, axes):
    panel.draw(axes, MetricTable.from_rows([]))


@pytest.mark.parametrize("panel", PANELS, ids=[panel.slug for panel in PANELS])
def test_every_panel_survives_one_row(panel, axes):
    panel.draw(axes, MetricTable.from_rows(make_rows(1)))


@pytest.mark.parametrize("panel", PANELS, ids=[panel.slug for panel in PANELS])
def test_every_panel_survives_an_all_nan_table(panel, axes):
    rows = make_rows(6)
    for row in rows:
        for key in row:
            if key != "iteration":
                row[key] = float("nan")
    panel.draw(axes, MetricTable.from_rows(rows))


@pytest.mark.parametrize("panel", PANELS, ids=[panel.slug for panel in PANELS])
def test_every_panel_survives_a_metric_that_starts_halfway(panel, axes):
    rows = make_rows(20)
    for row in rows[:10]:
        row.pop("success_rate", None)
        row.pop("explained_variance", None)
        row.pop("clipfrac", None)
    panel.draw(axes, MetricTable.from_rows(rows))


def test_a_single_row_draws_a_marker_not_an_invisible_line(axes):
    table = MetricTable.from_rows(make_rows(1))
    plots.panel_ep_return(axes, table)
    assert axes.lines or axes.collections, "one row must still put something on the canvas"


def test_the_raw_series_is_drawn_under_the_smoothed_one(axes):
    plots.panel_ep_len(axes, MetricTable.from_rows(make_rows(40)))
    alphas = [line.get_alpha() for line in axes.lines]
    assert any(alpha is not None and alpha < 0.5 for alpha in alphas), "raw series must be visible"
    assert any(alpha is None or alpha >= 0.9 for alpha in alphas), "smoothed series must be on top"


def test_ppo_health_draws_the_target_kl_line(axes):
    plots.panel_ppo_health(axes, MetricTable.from_rows(make_rows(20)))
    horizontals = [line for line in axes.lines if len(set(np.round(line.get_ydata(), 9))) == 1]
    assert horizontals, "the target KL threshold must be drawn"


def test_explained_variance_draws_the_zero_reference(axes):
    plots.panel_explained_variance(axes, MetricTable.from_rows(make_rows(20)))
    assert any(np.allclose(line.get_ydata(), 0.0) for line in axes.lines)


def test_success_rate_is_pinned_to_the_full_scale(axes):
    plots.panel_success(axes, MetricTable.from_rows(make_rows(20)))
    low, high = axes.get_ylim()
    assert low <= 0.0 and high >= 1.0


# ------------------------------------------------------------------------------- panel discovery


def test_build_panels_appends_one_panel_per_unclaimed_key():
    rows = make_rows(5)
    for row in rows:
        row["collisions_per_min"] = 1.0
        row["fps_env"] = 50000.0
    panels = build_panels(MetricTable.from_rows(rows))
    slugs = [panel.slug for panel in panels]
    assert slugs[: len(PANELS)] == [panel.slug for panel in PANELS]
    assert "extra_collisions_per_min" in slugs
    assert "extra_fps_env" in slugs
    assert "extra_grad_norm" in slugs, "grad_norm has no fixed panel, so it must be discovered"
    assert "extra_target_kl" not in slugs, "target_kl is consumed by the PPO health panel"


def test_build_panels_caps_the_number_of_discovered_panels():
    rows = [{"iteration": 1, **{f"m{index}": float(index) for index in range(50)}}]
    panels = build_panels(MetricTable.from_rows(rows), max_extra=4)
    assert len(panels) == len(PANELS) + 4


def test_discovered_panels_draw(axes):
    rows = make_rows(10)
    for row in rows:
        row["weird/key name"] = 1.0
    panels = build_panels(MetricTable.from_rows(rows))
    extra = next(panel for panel in panels if panel.slug.startswith("extra_weird"))
    extra.draw(axes, MetricTable.from_rows(rows))


def test_standalone_panel_render_writes_a_png(tmp_path):
    path = tmp_path / "panel.png"
    plots.render_panel_standalone(PANELS[0], MetricTable.from_rows(make_rows(30)), str(path))
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert path.stat().st_size > 5_000
