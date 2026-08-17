"""Unit tests for the run-directory contract and the TrainLogger API.

The interesting tests here are the concurrency ones. The whole hot-reload story rests on the
claim that a reader never observes a partial file, so that claim is tested by actually running a
reader in a loop against a live writer, on this platform, rather than by trusting ``os.replace``.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from duckiebot_rl.viz.logger import TrainLogger
from duckiebot_rl.viz.run_dir import (
    REQUIRED_METRIC_KEYS,
    SCHEMA_VERSION,
    RunDir,
    RunStatus,
    atomic_write_json,
    atomic_write_text,
    find_latest_run,
    json_safe,
    make_run_id,
    parse_run_id,
    read_json_tolerant,
    read_text_tolerant,
    sha256_file,
    status_age_s,
)

# ------------------------------------------------------------------------------------- run ids


def test_run_id_has_the_contract_shape():
    when = datetime(2026, 8, 17, 10, 45, 0, tzinfo=UTC)
    assert make_run_id("lanefollow", 0, when=when) == "20260817T104500Z_lanefollow_seed0"


def test_run_id_sanitises_and_round_trips():
    run_id = make_run_id("lane follow/v2", 7, when=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC))
    assert run_id == "20260102T030405Z_lane-follow-v2_seed7"
    parts = parse_run_id(run_id)
    assert parts == {"stamp": "20260102T030405Z", "name": "lane-follow-v2", "seed": "7"}


def test_run_id_rejects_an_empty_name():
    with pytest.raises(ValueError, match="empty after sanitising"):
        make_run_id("///", 0)


def test_parse_run_id_returns_none_for_a_hand_named_directory():
    assert parse_run_id("my_scratch_run") is None


def test_find_latest_run_prefers_the_newest_stamp(tmp_path):
    for stamp in ("20260101T000000Z", "20260817T104500Z", "20260501T120000Z"):
        (tmp_path / f"{stamp}_x_seed0").mkdir()
    (tmp_path / "hand_named").mkdir()
    latest = find_latest_run(tmp_path)
    assert latest is not None and latest.name == "20260817T104500Z_x_seed0"


def test_find_latest_run_on_an_empty_root(tmp_path):
    assert find_latest_run(tmp_path / "nope") is None


# ------------------------------------------------------------------------------------ the tree


def test_create_makes_every_directory_of_the_contract(tmp_path):
    run = RunDir.create(tmp_path / "runs", name="lanefollow", seed=3)
    for directory in (
        run.root,
        run.checkpoints_dir,
        run.metrics_dir,
        run.graphs_dir,
        run.videos_dir,
        run.obs_dir,
    ):
        assert directory.is_dir()
    assert run.root.parent.name == "runs"
    assert run.run_id.endswith("_lanefollow_seed3")


def test_paths_are_the_only_place_the_layout_is_written_down(tmp_path):
    run = RunDir.create(tmp_path, run_id="r")
    assert run.config_path.name == "config.yaml"
    assert run.status_path.name == "status.json"
    assert run.metrics_path.name == "metrics.jsonl"
    assert run.latest_checkpoint.name == "model_latest.pth"
    assert run.best_checkpoint.name == "model_best.pth"
    assert run.final_checkpoint.name == "model_final.pth"
    assert run.index_path.name == "index.json"
    assert run.archive_checkpoint(42).name == "model_episode_42.pth"
    assert run.overview_figure.name == "_overview.png"
    assert run.dashboard_figure.name == "_dashboard.png"
    # House standard: per-tag graphs are the tag with non-alphanumerics collapsed to "__".
    assert run.graph_path("episode/reward").name == "episode__reward.png"
    assert run.latest_video_mp4.name == "latest_rollout.mp4"
    assert run.latest_video_gif.name == "latest_rollout.gif"
    assert run.latest_obs.name == "latest_obs.png"


def test_open_does_not_create_unless_asked(tmp_path):
    run = RunDir.open(tmp_path / "ghost")
    assert not run.exists()
    assert run.read_status() is None
    assert run.read_metrics() == []
    assert run.read_index() == {}


def test_create_is_idempotent_so_a_resume_reuses_the_tree(tmp_path):
    first = RunDir.create(tmp_path, run_id="r")
    first.append_metrics({"iteration": 1})
    second = RunDir.create(tmp_path, run_id="r")
    assert second.root == first.root
    assert len(second.read_metrics()) == 1


# ------------------------------------------------------------------------------ atomic writing


def test_atomic_write_leaves_no_temporary_behind(tmp_path):
    target = tmp_path / "a.json"
    atomic_write_json(target, {"x": 1})
    assert json.loads(target.read_text()) == {"x": 1}
    assert not (tmp_path / "a.json.tmp").exists()


def test_a_reader_in_a_loop_never_sees_a_partial_file(tmp_path):
    """The atomicity rule, tested rather than assumed.

    One thread rewrites the same path with two payloads of very different sizes; another reads it
    as fast as it can. Every single read must decode, and must equal one of the two payloads. A
    non-atomic writer fails this within a few hundred iterations on Windows.
    """
    target = tmp_path / "status.json"
    small = {"state": "running", "iteration": 1}
    large = {"state": "running", "iteration": 2, "pad": "x" * 200_000}
    atomic_write_json(target, small)

    stop = threading.Event()
    failures: list[str] = []
    reads = [0]

    def reader() -> None:
        while not stop.is_set():
            payload = read_json_tolerant(target)
            if payload is None:
                failures.append("reader saw a vanished file")
                continue
            if payload not in (small, large):
                failures.append(f"reader saw a partial payload of {len(str(payload))} chars")
            reads[0] += 1

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    try:
        for index in range(120):
            atomic_write_json(target, large if index % 2 else small, fsync=False)
    finally:
        stop.set()
        thread.join(timeout=10)

    assert not failures, failures[:3]
    assert reads[0] > 50, f"reader only managed {reads[0]} reads, the race was never exercised"


def test_the_tolerant_reader_survives_a_file_appearing_late(tmp_path):
    target = tmp_path / "late.json"
    assert read_json_tolerant(target) is None
    assert read_text_tolerant(target) is None
    atomic_write_text(target, '{"ok": true}')
    assert read_json_tolerant(target) == {"ok": True}


def test_the_tolerant_reader_returns_none_on_undecodable_json(tmp_path):
    target = tmp_path / "junk.json"
    target.write_bytes(b"{not json at all")
    assert read_json_tolerant(target, attempts=2) is None


def test_sha256_file_matches_hashlib_and_tolerates_a_missing_file(tmp_path):
    import hashlib

    target = tmp_path / "blob.bin"
    target.write_bytes(b"duckiebot")
    assert sha256_file(target) == hashlib.sha256(b"duckiebot").hexdigest()
    assert sha256_file(tmp_path / "missing.bin") == ""


# ------------------------------------------------------------------------------------- json_safe


def test_json_safe_coerces_numpy_and_kills_non_finite_floats():
    payload = json_safe(
        {
            "np_scalar": np.float32(1.5),
            "np_int": np.int64(7),
            "array": np.arange(3),
            "nan": float("nan"),
            "inf": float("inf"),
            "nested": {"path": __import__("pathlib").Path("a/b")},
            "flag": True,
        }
    )
    assert payload["np_scalar"] == pytest.approx(1.5)
    assert payload["np_int"] == 7
    assert payload["array"] == [0, 1, 2]
    assert payload["nan"] is None
    assert payload["inf"] is None
    assert payload["nested"]["path"] == "a/b"
    assert payload["flag"] is True
    json.dumps(payload)


# --------------------------------------------------------------------------------------- status


def test_status_round_trips_through_disk(tmp_path):
    run = RunDir.create(tmp_path, run_id="r")
    run.write_status(RunStatus(state="running", iteration=12, total_timesteps=999, device="cuda:0"))
    status = run.read_status()
    assert status is not None
    assert (status.iteration, status.total_timesteps, status.device) == (12, 999, "cuda:0")
    assert status.run_id == "r"
    assert status.schema_version == SCHEMA_VERSION
    assert status.last_update_utc.endswith("Z")


def test_status_json_carries_every_contract_field(tmp_path):
    run = RunDir.create(tmp_path, run_id="r")
    run.write_status(RunStatus())
    payload = json.loads(run.status_path.read_text())
    expected = {
        "schema_version",
        "run_id",
        "pid",
        "state",
        "iteration",
        "total_timesteps",
        "wall_clock_s",
        "steps_per_s",
        "best_metric_name",
        "best_metric_value",
        "best_iteration",
        "last_update_utc",
        "vram_used_mb",
        "num_envs",
        "device",
        "git_commit",
    }
    assert set(payload) == expected


def test_status_age_reports_staleness():
    old = RunStatus(last_update_utc="2026-08-17T10:00:00Z")
    now = datetime(2026, 8, 17, 10, 5, 0, tzinfo=UTC)
    assert status_age_s(old, now=now) == pytest.approx(300.0)
    assert status_age_s(RunStatus(), now=now) is None
    assert status_age_s(None) is None
    future = RunStatus(last_update_utc=(now + timedelta(seconds=30)).strftime("%Y-%m-%dT%H:%M:%SZ"))
    assert status_age_s(future, now=now) == 0.0


def test_unknown_status_keys_are_ignored_so_an_older_run_still_reads():
    status = RunStatus.from_dict({"state": "finished", "some_future_field": 1})
    assert status.state == "finished"


# -------------------------------------------------------------------------------------- metrics


def test_metrics_round_trip_including_non_finite(tmp_path):
    run = RunDir.create(tmp_path, run_id="r")
    rows = [
        {"iteration": 1, "ep_return_mean": 1.5, "note": "hello"},
        {"iteration": 2, "ep_return_mean": float("nan")},
        {"iteration": 3, "ep_return_mean": float("inf")},
    ]
    for row in rows:
        run.append_metrics(row)
    read = run.read_metrics()
    assert [row["iteration"] for row in read] == [1, 2, 3]
    assert read[0]["ep_return_mean"] == pytest.approx(1.5)
    assert read[0]["note"] == "hello"
    assert read[1]["ep_return_mean"] is None
    assert read[2]["ep_return_mean"] is None


def test_metrics_reader_drops_a_torn_trailing_line(tmp_path):
    run = RunDir.create(tmp_path, run_id="r")
    run.append_metrics({"iteration": 1})
    run.append_metrics({"iteration": 2})
    with run.metrics_path.open("ab") as handle:
        handle.write(b'{"iteration": 3, "ep_ret')
    rows = run.read_metrics()
    assert [row["iteration"] for row in rows] == [1, 2]


def test_metrics_fingerprint_moves_when_the_file_grows(tmp_path):
    run = RunDir.create(tmp_path, run_id="r")
    assert run.metrics_fingerprint() == (-1, -1.0)
    run.append_metrics({"iteration": 1})
    first = run.metrics_fingerprint()
    run.append_metrics({"iteration": 2})
    assert run.metrics_fingerprint()[0] > first[0]


def test_config_round_trips(tmp_path):
    run = RunDir.create(tmp_path, run_id="r")
    config = {"seed": 0, "ppo": {"clip": 0.2, "lr": 3e-4}, "envs": [1, 2, 3]}
    run.write_config(config)
    assert run.read_config() == config


# ---------------------------------------------------------------------------- checkpoint index


def _writer(payload: bytes) -> Callable[[Path], None]:
    return lambda path: path.write_bytes(payload)


def test_index_tracks_best_only_when_the_metric_improves(tmp_path):
    log = TrainLogger.create(
        tmp_path, name="r", seed=0, config={"a": 1}, tensorboard=False, warn_missing_keys=False
    )

    log.log_iteration({"iteration": 1, "total_timesteps": 10, "ep_return_mean": 1.0})
    log.save_checkpoint(_writer(b"one"), metric=1.0)
    index = log.run.read_index()
    assert index["latest"]["iteration"] == 1
    assert index["best"]["iteration"] == 1
    assert index["best"]["file"] == "model_best.pth"
    assert index["best"]["metric_value"] == pytest.approx(1.0)

    log.log_iteration({"iteration": 2, "total_timesteps": 20, "ep_return_mean": 5.0})
    log.save_checkpoint(_writer(b"two"), metric=5.0)
    index = log.run.read_index()
    assert index["best"]["iteration"] == 2 and index["best"]["metric_value"] == pytest.approx(5.0)

    log.log_iteration({"iteration": 3, "total_timesteps": 30, "ep_return_mean": 0.5})
    log.save_checkpoint(_writer(b"three"), metric=0.5)
    index = log.run.read_index()
    assert index["latest"]["iteration"] == 3, "latest always moves"
    assert index["best"]["iteration"] == 2, "best must not regress"
    assert log.run.best_checkpoint.read_bytes() == b"two"
    assert log.run.latest_checkpoint.read_bytes() == b"three"
    log.finish(render=False)


def test_best_mode_min_selects_the_smallest(tmp_path):
    log = TrainLogger.create(
        tmp_path,
        name="r",
        seed=0,
        best_metric="lane_dev_rms_m",
        best_mode="min",
        tensorboard=False,
        warn_missing_keys=False,
    )
    for iteration, metric in ((1, 0.10), (2, 0.02), (3, 0.30)):
        log.log_iteration({"iteration": iteration, "total_timesteps": iteration, "lane_dev_rms_m": metric})
        log.save_checkpoint(_writer(f"c{iteration}".encode()), metric=metric)
    assert log.run.read_index()["best"]["iteration"] == 2
    assert log.status.best_metric_value == pytest.approx(0.02)
    log.finish(render=False)


def test_index_entry_hash_matches_the_file_on_disk(tmp_path):
    log = TrainLogger.create(tmp_path, name="r", seed=0, tensorboard=False, warn_missing_keys=False)
    log.log_iteration({"iteration": 1, "total_timesteps": 1, "ep_return_mean": 1.0})
    log.save_checkpoint(_writer(b"payload"), metric=1.0)
    entry = log.run.read_index()["latest"]
    assert entry["sha256"] == sha256_file(log.run.latest_checkpoint)
    assert entry["size_bytes"] == len(b"payload")
    log.finish(render=False)


def test_bad_best_mode_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="best_mode"):
        TrainLogger(tmp_path / "r", best_mode="sideways")


# ---------------------------------------------------------------------------------- TrainLogger


def test_logger_writes_config_status_and_metrics(tmp_path):
    log = TrainLogger.create(
        tmp_path, name="lanefollow", seed=2, config={"seed": 2}, num_envs=64, device="cpu"
    )
    row = dict.fromkeys(REQUIRED_METRIC_KEYS, 0.0)
    row.update({"iteration": 1, "total_timesteps": 4096})
    log.log_iteration(row)
    log.finish(render=False)

    assert log.run.config_path.exists()
    status = log.run.read_status()
    assert status is not None
    assert status.state == "finished"
    assert status.iteration == 1
    assert status.total_timesteps == 4096
    assert status.num_envs == 64
    assert status.device == "cpu"
    assert len(log.run.read_metrics()) == 1


def test_logger_fills_in_bookkeeping_keys(tmp_path):
    log = TrainLogger.create(tmp_path, name="r", seed=0, tensorboard=False, warn_missing_keys=False)
    first = log.log_iteration({"ep_return_mean": 1.0})
    second = log.log_iteration({"ep_return_mean": 2.0})
    assert (first["iteration"], second["iteration"]) == (1, 2)
    assert math.isfinite(first["wall_clock_s"])
    log.finish(render=False)


def test_logger_warns_once_about_missing_required_keys(tmp_path):
    log = TrainLogger.create(tmp_path, name="r", seed=0, tensorboard=False)
    with pytest.warns(RuntimeWarning, match="missing required key"):
        log.log_iteration({"iteration": 1, "ep_return_mean": 1.0})
    log.log_iteration({"iteration": 2, "ep_return_mean": 1.0})
    log.finish(render=False)


def test_logger_marks_a_crash_when_the_loop_raises(tmp_path):
    with (
        pytest.raises(RuntimeError, match="boom"),
        TrainLogger.create(tmp_path, name="r", seed=0, tensorboard=False, warn_missing_keys=False) as log,
    ):
        log.log_iteration({"iteration": 1, "total_timesteps": 1, "ep_return_mean": 0.0})
        raise RuntimeError("boom")
    status = RunDir.open(log.run.root).read_status()
    assert status is not None and status.state == "crashed"


def test_logger_finish_is_idempotent(tmp_path):
    log = TrainLogger.create(tmp_path, name="r", seed=0, tensorboard=False)
    log.finish(render=False)
    log.finish(state="crashed", render=False)
    status = log.run.read_status()
    assert status is not None and status.state == "finished"


def test_logger_archives_periodically(tmp_path):
    log = TrainLogger.create(
        tmp_path, name="r", seed=0, archive_every=2, tensorboard=False, warn_missing_keys=False
    )
    for iteration in range(1, 5):
        log.log_iteration({"iteration": iteration, "total_timesteps": iteration, "ep_return_mean": 0.0})
        log.save_checkpoint(_writer(b"x"), metric=float(iteration))
    archives = [path.name for path in log.run.archived_checkpoints()]
    assert archives == ["model_episode_2.pth", "model_episode_4.pth"]
    log.finish(render=False)


def test_logger_accepts_a_mapping_and_writes_it_with_torch(tmp_path):
    torch = pytest.importorskip("torch")
    log = TrainLogger.create(tmp_path, name="r", seed=0, tensorboard=False, warn_missing_keys=False)
    log.log_iteration({"iteration": 1, "total_timesteps": 1, "ep_return_mean": 1.0})
    log.save_checkpoint({"weights": torch.zeros(3)}, metric=1.0)
    payload = torch.load(log.run.latest_checkpoint, weights_only=False)
    assert payload["weights"].shape == (3,)
    log.finish(render=False)


def test_the_heartbeat_is_rewritten_every_iteration(tmp_path):
    log = TrainLogger.create(tmp_path, name="r", seed=0, tensorboard=False, warn_missing_keys=False)
    seen: list[int] = []
    for iteration in range(1, 4):
        log.log_iteration({"iteration": iteration, "total_timesteps": iteration * 10})
        status = log.run.read_status()
        assert status is not None
        seen.append(status.iteration)
    assert seen == [1, 2, 3]
    log.finish(render=False)


def test_a_dashboard_reader_survives_a_live_writer(tmp_path):
    """A reader polling a run being written must never see a broken status or a torn row."""
    log = TrainLogger.create(tmp_path, name="r", seed=0, tensorboard=False, warn_missing_keys=False)
    reader = RunDir.open(log.run.root)
    stop = threading.Event()
    problems: list[str] = []

    def poll() -> None:
        while not stop.is_set():
            status = reader.read_status()
            if status is None or status.state not in ("running", "finished"):
                problems.append(f"bad status: {status}")
            rows = reader.read_metrics()
            iterations = [row.get("iteration") for row in rows]
            if iterations != sorted(iterations):
                problems.append("rows out of order")
            time.sleep(0.001)

    thread = threading.Thread(target=poll, daemon=True)
    thread.start()
    try:
        for iteration in range(1, 200):
            log.log_iteration({"iteration": iteration, "total_timesteps": iteration * 512})
    finally:
        stop.set()
        thread.join(timeout=10)
    log.finish(render=False)
    assert not problems, problems[:3]
    assert len(reader.read_metrics()) == 199
