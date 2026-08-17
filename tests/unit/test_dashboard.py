"""Unit tests for the composite dashboard and the watch loop.

Every rendering test checks a real PNG: the magic bytes, a plausible size, and that Pillow can
decode it to the expected pixel width. "It did not raise" is not evidence that a figure exists.
"""

from __future__ import annotations

import json
import threading
import time

import pytest

from duckiebot_rl.viz.dashboard import (
    STALE_AFTER_S,
    describe_state,
    render_dashboard,
    summarise,
    watch,
)
from duckiebot_rl.viz.logger import TrainLogger
from duckiebot_rl.viz.plots import PALETTE, matplotlib_available
from duckiebot_rl.viz.run_dir import RunDir, RunStatus, utc_now_iso

pytestmark = pytest.mark.skipif(not matplotlib_available(), reason="needs the [viz] extra (matplotlib)")

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def assert_valid_png(path, min_bytes: int = 8_000, expect_width: int | None = 1600) -> None:
    """Assert that a path holds a decodable PNG of a sensible size.

    Args:
        path: The PNG path.
        min_bytes: Floor on file size; a blank figure is roughly 6 kB.
        expect_width: Expected pixel width, or None to skip the check.
    """
    from PIL import Image

    assert path.exists(), f"{path} was not written"
    raw = path.read_bytes()
    assert raw[:8] == PNG_MAGIC, "not a PNG"
    assert len(raw) > min_bytes, f"{path} is only {len(raw)} bytes"
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        if expect_width is not None:
            assert image.width == expect_width, f"expected {expect_width} px wide, got {image.width}"
        assert image.height > 400


def make_run(tmp_path, rows: list[dict], state: str = "finished", run_id: str = "20260817T104500Z_t_seed0"):
    """Write a run directory with the given metrics rows and terminal state.

    Args:
        tmp_path: Temporary directory.
        rows: Metrics rows to append.
        state: Terminal state to record, or ``"running"``.
        run_id: Run id to use.

    Returns:
        The :class:`RunDir`.
    """
    run = RunDir.create(tmp_path, run_id=run_id)
    run.write_config({"seed": 0})
    for row in rows:
        run.append_metrics(row)
    run.write_status(
        RunStatus(
            state=state,
            iteration=len(rows),
            total_timesteps=len(rows) * 4096,
            wall_clock_s=len(rows) * 11.0,
            steps_per_s=372.0,
            best_metric_name="ep_return_mean",
            best_metric_value=1.0 if rows else None,
            best_iteration=len(rows) or None,
            num_envs=4096,
            device="cuda:0",
            vram_used_mb=6800.0,
        )
    )
    return run


def full_rows(count: int = 30) -> list[dict]:
    """Build rows carrying every required metric key.

    Args:
        count: Number of iterations.

    Returns:
        The rows.
    """
    rows = []
    for index in range(1, count + 1):
        progress = index / count
        rows.append(
            {
                "iteration": index,
                "total_timesteps": index * 4096,
                "wall_clock_s": index * 11.0,
                "ep_return_mean": 90.0 * progress,
                "ep_return_std": 12.0 * (1 - progress),
                "ep_len_mean": 150.0 + 600.0 * progress,
                "policy_loss": -0.02 * (1 - progress),
                "value_loss": 30.0 * (1 - progress) + 1.0,
                "entropy": 2.2 - progress,
                "approx_kl": 0.012 + 0.004 * progress,
                "clipfrac": 0.1 + 0.06 * progress,
                "explained_variance": -0.2 + 1.1 * progress,
                "grad_norm": 1.8 - 0.9 * progress,
                "learning_rate": 3e-4 * (1 - 0.6 * progress),
                "lane_dev_rms_m": 0.1 * (1 - progress) + 0.012,
                "lane_dev_max_m": 0.2 * (1 - progress) + 0.03,
                "success_rate": progress,
                "alpha_vis": min(1.0, 0.04 * index),
                "alpha_dyn": min(1.0, 0.02 * index),
            }
        )
    return rows


# ------------------------------------------------------------------------------ the happy path


def test_renders_a_full_run(tmp_path):
    run = make_run(tmp_path, full_rows(60))
    figure = render_dashboard(run)
    assert figure == run.dashboard_figure
    assert_valid_png(figure, min_bytes=40_000)


def test_renders_standalone_panels_too(tmp_path):
    run = make_run(tmp_path, full_rows(20))
    render_dashboard(run, panels=True)
    # The composite lives in graphs/ next to the house-standard overviews; its standalone
    # panels live in graphs/panels/ so graphs/ stays one-file-per-tag plus the overviews.
    assert run.dashboard_figure.name in {p.name for p in run.graphs_dir.glob("*.png")}
    written = {path.name for path in run.panels_dir.glob("*.png")}
    for slug in ("ep_return", "lane_deviation", "ppo_health", "dr_curriculum", "entropy_lr"):
        assert f"{slug}.png" in written, f"missing standalone panel {slug}"
        assert_valid_png(run.panel_path(slug), min_bytes=5_000, expect_width=None)


def test_the_composite_is_written_atomically(tmp_path):
    run = make_run(tmp_path, full_rows(10))
    render_dashboard(run)
    assert not (run.graphs_dir / "_dashboard.png.tmp").exists()
    first = run.dashboard_figure.stat().st_size
    render_dashboard(run)
    assert run.dashboard_figure.stat().st_size > 0 and first > 0


def test_extra_metric_keys_grow_the_figure(tmp_path):
    plain = make_run(tmp_path / "plain", full_rows(20))
    rows = full_rows(20)
    for row in rows:
        row["collisions_per_min"] = 2.0
        row["duckie_squashes"] = 0.0
        row["corner_cut_m"] = 0.03
        row["survival_s"] = 42.0
    extra = make_run(tmp_path / "extra", rows)

    from PIL import Image

    with Image.open(render_dashboard(plain)) as small, Image.open(render_dashboard(extra)) as big:
        assert big.height > small.height, "discovered panels must be appended, not dropped"


# ------------------------------------------------------------------------- the degenerate cases


def test_renders_zero_rows(tmp_path):
    run = make_run(tmp_path, [], state="running")
    assert_valid_png(render_dashboard(run))


def test_renders_a_run_directory_with_nothing_in_it_at_all(tmp_path):
    run = RunDir.create(tmp_path, run_id="20260817T104500Z_bare_seed0")
    assert_valid_png(render_dashboard(run))


def test_renders_one_row(tmp_path):
    run = make_run(tmp_path, full_rows(1))
    assert_valid_png(render_dashboard(run))


def test_renders_an_all_nan_column(tmp_path):
    rows = full_rows(15)
    for row in rows:
        row["value_loss"] = float("nan")
        row["success_rate"] = float("nan")
    run = make_run(tmp_path, rows)
    assert_valid_png(render_dashboard(run))


def test_renders_a_metric_that_only_appears_halfway(tmp_path):
    rows = full_rows(20)
    for index, row in enumerate(rows):
        if index >= 10:
            row["eval_rms_m"] = 0.05
    run = make_run(tmp_path, rows)
    assert_valid_png(render_dashboard(run))


def test_renders_a_crashed_run(tmp_path):
    run = make_run(tmp_path, full_rows(8), state="crashed")
    assert_valid_png(render_dashboard(run))
    status = run.read_status()
    assert describe_state(status, 1.0) == ("CRASHED", PALETTE.critical)


def test_renders_when_status_json_is_corrupt(tmp_path):
    run = make_run(tmp_path, full_rows(5))
    run.status_path.write_bytes(b"{ this is not json")
    assert_valid_png(render_dashboard(run))


def test_renders_when_a_torn_row_is_in_flight(tmp_path):
    run = make_run(tmp_path, full_rows(12))
    with run.metrics_path.open("ab") as handle:
        handle.write(b'{"iteration": 13, "ep_ret')
    assert_valid_png(render_dashboard(run))


def test_renders_a_run_whose_metrics_are_all_strings(tmp_path):
    run = RunDir.create(tmp_path, run_id="20260817T104500Z_odd_seed0")
    for index in range(4):
        run.append_metrics({"iteration": index, "note": "no numbers here"})
    assert_valid_png(render_dashboard(run))


# --------------------------------------------------------------------------------- header state


def test_describe_state_covers_every_case():
    assert describe_state(None, None)[0] == "NO STATUS YET"
    assert describe_state(RunStatus(state="finished", last_update_utc=utc_now_iso()), 1.0) == (
        "FINISHED",
        PALETTE.good,
    )
    assert describe_state(RunStatus(state="running"), 1.0) == ("RUNNING", PALETTE.good)
    label, colour = describe_state(RunStatus(state="running"), STALE_AFTER_S + 1.0)
    assert label.startswith("STALE") and colour == PALETTE.warning
    assert describe_state(RunStatus(state="crashed"), 1.0) == ("CRASHED", PALETTE.critical)


def test_summarise_reports_the_run_without_matplotlib_state(tmp_path):
    log = TrainLogger.create(tmp_path, name="sum", seed=1, tensorboard=False, warn_missing_keys=False)
    log.log_iteration({"iteration": 1, "total_timesteps": 4096, "ep_return_mean": 3.0})
    log.save_checkpoint(lambda path: path.write_bytes(b"ckpt"), metric=3.0)
    log.finish(render=False)
    text = summarise(log.run)
    assert "FINISHED" in text
    assert "sum_seed1" in text
    assert "best.pt" in text
    assert "latest.pt" in text


# ------------------------------------------------------------------------------------ watch mode


def test_watch_renders_and_exits_on_a_finished_run(tmp_path):
    run = make_run(tmp_path, full_rows(6), state="finished")
    seen: list[int] = []
    figure = watch(run, interval=0.05, on_render=lambda path, count: seen.append(count))
    assert figure == run.dashboard_figure
    assert_valid_png(figure)
    assert seen, "watch must render at least once before exiting"


def test_watch_tolerates_a_run_directory_that_does_not_exist_yet(tmp_path):
    missing = RunDir.open(tmp_path / "not_created_yet")
    assert watch(missing, interval=0.01, max_iterations=3) is None


def test_watch_re_renders_when_metrics_grow(tmp_path):
    run = make_run(tmp_path, full_rows(4), state="running")
    renders: list[int] = []

    def grow() -> None:
        for index in range(5, 9):
            time.sleep(0.06)
            run.append_metrics(full_rows(index)[-1])
        run.write_status(RunStatus(state="finished", iteration=8))

    writer = threading.Thread(target=grow, daemon=True)
    writer.start()
    watch(run, interval=0.03, on_render=lambda path, count: renders.append(count))
    writer.join(timeout=10)
    assert len(renders) >= 3, f"expected repeated renders, got {len(renders)}"
    assert_valid_png(run.dashboard_figure)


def test_watch_can_be_asked_to_keep_going_after_the_run_ends(tmp_path):
    run = make_run(tmp_path, full_rows(3), state="finished")
    renders: list[int] = []
    watch(
        run,
        interval=0.01,
        max_iterations=4,
        exit_when_done=False,
        on_render=lambda path, count: renders.append(count),
    )
    assert len(renders) == 4, "a terminal run must still be re-rendered when asked"


def test_watch_survives_a_render_failure(tmp_path, monkeypatch):
    run = make_run(tmp_path, full_rows(3), state="finished")

    def explode(*args: object, **kwargs: object) -> None:
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr("duckiebot_rl.viz.dashboard.render_dashboard", explode)
    assert watch(run, interval=0.01, max_iterations=2) is None


# ------------------------------------------------------------------------------------ end to end


def test_logger_to_dashboard_end_to_end(tmp_path):
    log = TrainLogger.create(
        tmp_path,
        name="e2e",
        seed=0,
        config={"seed": 0, "num_envs": 8},
        num_envs=8,
        device="cpu",
        tensorboard=False,
    )
    for row in full_rows(25):
        log.log_iteration(row)
        if row["iteration"] % 10 == 0:
            log.save_checkpoint(lambda path: path.write_bytes(b"x"), metric=row["ep_return_mean"])
    log.finish(render=True)

    assert_valid_png(log.run.dashboard_figure, min_bytes=30_000)
    assert json.loads(log.run.index_path.read_text())["best"]["file"] == "model_best.pth"
    status = log.run.read_status()
    assert status is not None and status.state == "finished"
    assert len(log.run.read_metrics()) == 25
