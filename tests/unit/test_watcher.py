"""CheckpointWatcher: detects new checkpoints, refuses corrupt and partial ones.

No Isaac, no GPU, no torch: the watcher only ever hashes bytes and reads JSON, so these tests
write plain byte blobs where a real run would write a ``.pt``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from duckiebot_rl.viz.run_dir import RunDir
from duckiebot_rl.viz.watcher import CheckpointWatcher, atomic_replace


def _write_checkpoint(run: RunDir, payload: bytes, name: str = "latest.pt") -> Path:
    """Write a fake checkpoint file into the run's checkpoints directory."""
    run.ensure_tree()
    path = run.checkpoints_dir / name
    path.write_bytes(payload)
    return path


def _publish(run: RunDir, payload: bytes, iteration: int, metric: float, name: str = "latest.pt") -> Path:
    """Write a fake checkpoint and record it in the index exactly as the trainer does."""
    path = _write_checkpoint(run, payload, name=name)
    run.record_checkpoint(
        path, iteration=iteration, metric_name="ep_return_mean", metric_value=metric, kinds=("latest",)
    )
    return path


@pytest.fixture
def run(tmp_path: Path) -> RunDir:
    return RunDir.open(tmp_path / "20260817T104500Z_lanefollow_seed0", create=True)


def test_missing_run_directory_never_raises(tmp_path):
    watcher = CheckpointWatcher(tmp_path / "does" / "not" / "exist")
    assert watcher.poll() is None
    assert watcher.poll() is None
    assert watcher.last_error
    assert watcher.current is None


def test_index_present_but_checkpoint_absent(run):
    run.ensure_tree()
    path = _write_checkpoint(run, b"x" * 64)
    run.record_checkpoint(path, iteration=1, metric_value=1.0)
    path.unlink()

    watcher = CheckpointWatcher(run.root)
    assert watcher.poll() is None
    assert "not present" in watcher.last_error


def test_detects_a_new_checkpoint_once(run):
    _publish(run, b"alpha" * 100, iteration=3, metric=1.25)
    watcher = CheckpointWatcher(run.root)

    found = watcher.poll()
    assert found is not None
    assert found.which == "latest"
    assert found.iteration == 3
    assert found.metric_name == "ep_return_mean"
    assert found.metric_value == pytest.approx(1.25)
    assert found.verified is True
    assert found.size == len(b"alpha" * 100)
    assert "iteration=3" in found.describe()

    # A second poll on unchanged state reports nothing new.
    assert watcher.poll() is None
    assert watcher.current is found


def test_detects_a_swapped_checkpoint(run):
    _publish(run, b"alpha" * 100, iteration=3, metric=1.0)
    watcher = CheckpointWatcher(run.root)
    first = watcher.poll()
    assert first is not None

    _publish(run, b"bravo" * 137, iteration=9, metric=4.5)
    second = watcher.poll()
    assert second is not None
    assert second.sha256 != first.sha256
    assert second.iteration == 9
    assert second.metric_value == pytest.approx(4.5)
    assert watcher.poll() is None


def test_rejects_a_corrupt_checkpoint_hash_mismatch(run):
    path = _publish(run, b"alpha" * 100, iteration=3, metric=1.0)
    # The index now records the hash of the ORIGINAL bytes; rewrite the file underneath it.
    path.write_bytes(b"corrupt" * 100)

    watcher = CheckpointWatcher(run.root, hash_retries=0, hash_retry_delay=0.0)
    assert watcher.poll() is None
    assert watcher.rejected_count == 1
    assert "does not match sha256" in watcher.last_error
    assert watcher.current is None

    # Repeated polls on the same bad file do not re-hash it into an ever-growing rejection count.
    assert watcher.poll() is None
    assert watcher.rejected_count == 1


def test_rejects_a_partially_written_checkpoint(run):
    path = _publish(run, b"z" * 4096, iteration=7, metric=2.0)
    path.write_bytes(b"z" * 1024)  # truncated: the writer had not finished

    watcher = CheckpointWatcher(run.root, hash_retries=0, hash_retry_delay=0.0)
    assert watcher.poll() is None
    assert watcher.rejected_count == 1


def test_recovers_after_a_rejection_when_the_file_is_republished(run):
    path = _publish(run, b"z" * 4096, iteration=7, metric=2.0)
    path.write_bytes(b"z" * 1024)

    watcher = CheckpointWatcher(run.root, hash_retries=0, hash_retry_delay=0.0)
    assert watcher.poll() is None
    assert watcher.rejected_count == 1

    _publish(run, b"z" * 4096, iteration=8, metric=2.5)
    found = watcher.poll()
    assert found is not None
    assert found.iteration == 8
    assert found.verified is True


def test_unreadable_index_is_tolerated(run):
    _publish(run, b"alpha" * 100, iteration=1, metric=1.0)
    run.index_path.write_text("{ this is not json", encoding="utf-8")

    watcher = CheckpointWatcher(run.root)
    assert watcher.poll() is None
    assert watcher.last_error


def test_index_without_sha256_is_refused_by_default_and_allowed_when_opted_out(run):
    path = _write_checkpoint(run, b"alpha" * 100)
    run.index_path.write_text(json.dumps({"latest": {"file": path.name, "iteration": 4}}), encoding="utf-8")

    strict = CheckpointWatcher(run.root, require_index=True)
    assert strict.poll() is None
    assert "no sha256" in strict.last_error

    lenient = CheckpointWatcher(run.root, require_index=False, hash_retry_delay=0.0)
    found = lenient.poll()
    assert found is not None
    assert found.verified is False
    assert found.iteration == 4
    assert lenient.poll() is None


def test_best_slot_is_followed_independently(run):
    latest = _write_checkpoint(run, b"latest-bytes" * 10, name="latest.pt")
    best = _write_checkpoint(run, b"best-bytes" * 10, name="best.pt")
    run.record_checkpoint(latest, iteration=10, metric_value=1.0, kinds=("latest",))
    run.record_checkpoint(best, iteration=6, metric_value=9.0, kinds=("best",))

    watcher = CheckpointWatcher(run.root, which="best")
    found = watcher.poll()
    assert found is not None
    assert found.path.name == "best.pt"
    assert found.iteration == 6
    assert found.metric_value == pytest.approx(9.0)


def test_reset_re_offers_the_current_checkpoint(run):
    _publish(run, b"alpha" * 100, iteration=3, metric=1.0)
    watcher = CheckpointWatcher(run.root)
    assert watcher.poll() is not None
    assert watcher.poll() is None
    watcher.reset()
    assert watcher.poll() is not None


def test_wait_for_new_times_out_without_raising(run):
    watcher = CheckpointWatcher(run.root, poll_interval=0.01)
    assert watcher.wait_for_new(timeout=0.15) is None


def test_wait_for_new_honours_the_stop_predicate(run):
    _publish(run, b"alpha" * 100, iteration=1, metric=1.0)
    watcher = CheckpointWatcher(run.root, poll_interval=0.01)
    assert watcher.wait_for_new(timeout=1.0, stop=lambda: True) is None


def test_atomic_replace_leaves_no_temporary_file(tmp_path):
    source = tmp_path / "payload.tmp"
    destination = tmp_path / "payload.bin"
    source.write_bytes(b"contents")

    returned = atomic_replace(source, destination)

    assert returned == destination
    assert destination.read_bytes() == b"contents"
    assert not source.exists()
    assert list(tmp_path.glob("*.tmp")) == []
