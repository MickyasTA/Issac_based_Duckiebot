"""Detect when a training run has published a new checkpoint worth loading.

This is the reader half of the checkpoint contract. The writer half already exists and is not
duplicated here: :meth:`duckiebot_rl.viz.run_dir.RunDir.record_checkpoint` hashes a written
checkpoint and records it in ``checkpoints/index.json``, and
:meth:`duckiebot_rl.viz.logger.TrainLogger.save_checkpoint` calls that for the trainer. So the
integration contract with ``scripts/train.py`` is simply *keep using the logger*; this module
consumes what the logger already writes and adds nothing the trainer has to remember.

:class:`CheckpointWatcher` polls the index plus the size and mtime of the checkpoint file, and
reports only checkpoints it has not handed out before **and** whose content matches the SHA-256
the writer recorded.

Why hash verification is not paranoia
-------------------------------------
The run-directory contract says every writer writes ``<name>.tmp`` and then replaces. That makes a
reader safe from torn files only if the reader also refuses anything whose content disagrees with
the hash the writer recorded. Two things still go wrong without that check:

1. the index is updated before the ``.pt`` lands, or the other way round, so the reader sees a
   stale file paired with fresh metadata and loads yesterday's policy while printing today's
   iteration;
2. the file is genuinely truncated or corrupt, and ``torch.load`` dies inside the render loop.

A mismatch is retried a few times, which is the normal case of hashing a file mid-replace, then
recorded as a rejection and skipped until the file changes again. The watcher never raises for a
missing run directory, a missing index, a vanished file or a half-written one; it returns ``None``
and explains itself in :attr:`CheckpointWatcher.last_error`.

Windows note on :func:`atomic_replace`
--------------------------------------
On Windows, ``os.replace`` onto a path another process currently holds open fails with
``PermissionError``. That is exactly the live-viewer situation. Readers here hold a checkpoint
open only for the duration of one read, and :func:`atomic_replace` retries the collision. It is
public because :mod:`duckiebot_rl.viz.render` needs it for files an external video encoder wrote,
which none of the ``atomic_write_*`` helpers in :mod:`duckiebot_rl.viz.run_dir` can cover.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from duckiebot_rl.viz.run_dir import RunDir, sha256_file

__all__ = [
    "DEFAULT_POLL_INTERVAL_S",
    "CheckpointInfo",
    "CheckpointWatcher",
    "atomic_replace",
]

DEFAULT_POLL_INTERVAL_S = 2.0
"""Default seconds between :meth:`CheckpointWatcher.poll` attempts in blocking waits."""

_REPLACE_RETRIES = 20
_REPLACE_DELAY_S = 0.05


def atomic_replace(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    retries: int = _REPLACE_RETRIES,
    delay: float = _REPLACE_DELAY_S,
) -> Path:
    """Move ``source`` onto ``destination`` atomically, retrying the Windows sharing violation.

    Args:
        source: Existing temporary file.
        destination: Final path.
        retries: Extra attempts after a ``PermissionError``.
        delay: Seconds to wait between attempts.

    Returns:
        The destination path.

    Raises:
        PermissionError: If every attempt loses the race with a concurrent reader.
    """
    src = Path(source)
    dst = Path(destination)
    for attempt in range(retries + 1):
        try:
            src.replace(dst)
        except PermissionError:
            if attempt == retries:
                raise
            time.sleep(delay)
        else:
            return dst
    return dst  # pragma: no cover - the loop above always returns or raises


@dataclass(frozen=True)
class CheckpointInfo:
    """A verified, fully written checkpoint the viewer may load.

    Attributes:
        which: Index slot it came from, ``"latest"`` or ``"best"``.
        path: Path of the ``.pt`` file.
        iteration: PPO iteration recorded in the index.
        metric_name: Model-selection metric name recorded in the index.
        metric_value: Model-selection metric value, or None.
        sha256: Verified content hash, or ``""`` in unverified fallback mode.
        size: File size in bytes at verification time.
        mtime: File mtime at verification time.
        verified: True when the content hash was checked against the index.
    """

    which: str
    path: Path
    iteration: int
    metric_name: str
    metric_value: float | None
    sha256: str
    size: int
    mtime: float
    verified: bool

    def describe(self) -> str:
        """Return the one-line reload banner the viewer prints.

        Returns:
            A human-readable summary naming the file, the iteration and the metric.
        """
        metric = "n/a"
        if self.metric_value is not None:
            name = self.metric_name or "metric"
            metric = f"{name}={self.metric_value:.4g}"
        digest = self.sha256[:12] if self.sha256 else "unverified"
        return f"{self.which}={self.path.name} iteration={self.iteration} {metric} sha256={digest}"


class CheckpointWatcher:
    """Poll a run directory and report when a new, verified checkpoint appears.

    The watcher is stateful: :meth:`poll` returns a :class:`CheckpointInfo` only the first time it
    sees a given checkpoint, and None afterwards. That makes the caller's loop trivial::

        watcher = CheckpointWatcher(run_dir, which="latest")
        while True:
            found = watcher.poll()
            if found is not None:
                host.load_from_info(found)

    Args:
        run_dir: The run directory to watch. It does not need to exist yet.
        which: Index slot to follow, ``"latest"`` or ``"best"``.
        poll_interval: Default seconds between attempts in :meth:`wait_for_new`.
        require_index: When True, the contract-compliant default, a checkpoint is only offered
            once ``index.json`` carries a ``sha256`` for it. When False the watcher falls back to
            a size-stability check, which is weaker but lets the viewer follow a run directory
            produced by something that does not write an index.
        hash_retries: Extra hash attempts before a mismatch becomes a rejection, covering the
            normal case of hashing a file mid-replace.
        hash_retry_delay: Seconds between hash attempts.

    Attributes:
        run: The :class:`~duckiebot_rl.viz.run_dir.RunDir` being watched.
        which: The index slot being followed.
        poll_interval: Seconds between attempts in :meth:`wait_for_new`.
        require_index: Whether an index-recorded hash is mandatory.
        hash_retries: Extra hash attempts before a rejection.
        hash_retry_delay: Seconds between hash attempts.
        rejected_count: Number of checkpoints refused because their content did not match the
            hash recorded in the index.
        last_error: Human-readable reason the most recent poll returned nothing, or ``""``.
    """

    def __init__(
        self,
        run_dir: str | os.PathLike[str] | RunDir,
        which: str = "latest",
        poll_interval: float = DEFAULT_POLL_INTERVAL_S,
        require_index: bool = True,
        hash_retries: int = 4,
        hash_retry_delay: float = 0.10,
    ) -> None:
        self.run = run_dir if isinstance(run_dir, RunDir) else RunDir.open(run_dir)
        self.which = str(which)
        self.poll_interval = float(poll_interval)
        self.require_index = bool(require_index)
        self.hash_retries = int(hash_retries)
        self.hash_retry_delay = float(hash_retry_delay)
        self.rejected_count = 0
        self.last_error = ""
        self._seen_key: str | None = None
        self._rejected: dict[str, tuple[int, int]] = {}
        self._current: CheckpointInfo | None = None

    @property
    def run_dir(self) -> Path:
        """The watched run directory."""
        return self.run.root

    @property
    def current(self) -> CheckpointInfo | None:
        """The most recent checkpoint this watcher handed out, or None."""
        return self._current

    def reset(self) -> None:
        """Forget every checkpoint seen so far, so the next poll re-offers the current one."""
        self._seen_key = None
        self._rejected.clear()
        self._current = None
        self.last_error = ""

    def _entry(self) -> dict[str, Any] | None:
        """Return this watcher's slot from the index, or None.

        Returns:
            The index entry mapping, or None when the index or the slot is absent.
        """
        entry = self.run.read_index().get(self.which)
        return entry if isinstance(entry, dict) else None

    def _resolve_path(self, entry: dict[str, Any] | None) -> Path:
        """Return the checkpoint path an index entry points at.

        The house standard keeps the models at the run-directory ROOT
        (``model_best.pth`` / ``model_latest.pth``), while ``checkpoints/`` holds only the
        integrity index. The index records a bare basename, so resolve it against the root
        first and fall back to the canonical path for this slot.

        Args:
            entry: The index entry, or None.

        Returns:
            The checkpoint path. Existing candidates win over non-existent ones so a
            half-migrated run directory still resolves.
        """
        canonical = self.run.best_checkpoint if self.which == "best" else self.run.latest_checkpoint
        name = str(entry.get("file", "")) if entry else ""
        if not name:
            return canonical
        for candidate in (self.run.root / name, self.run.checkpoints_dir / name):
            if candidate.is_file():
                return candidate
        return self.run.root / name

    def _verify(self, path: Path, expected: str) -> tuple[bool, os.stat_result | None]:
        """Hash ``path`` and compare against ``expected``, retrying a file that is mid-write.

        The file is stat-ed before and after hashing. If it changed underneath, the attempt is
        discarded rather than reported, because the digest would describe neither version.

        Args:
            path: Checkpoint file.
            expected: Hex digest recorded in the index.

        Returns:
            ``(matched, stat)``. ``stat`` is None when the file vanished during every attempt.
        """
        for attempt in range(self.hash_retries + 1):
            try:
                before = path.stat()
                digest = sha256_file(path)
                after = path.stat()
            except OSError:
                if attempt == self.hash_retries:
                    return False, None
                time.sleep(self.hash_retry_delay)
                continue
            stable = (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
            if stable and digest == expected:
                return True, after
            if attempt == self.hash_retries:
                return False, after
            time.sleep(self.hash_retry_delay)
        return False, None  # pragma: no cover - the loop above always returns first

    def poll(self) -> CheckpointInfo | None:
        """Check for a new checkpoint.

        Returns:
            A :class:`CheckpointInfo` the first time a given verified checkpoint is seen, else
            None. Never raises: a missing run directory, a missing index, a vanished file and a
            hash mismatch all return None and set :attr:`last_error`.
        """
        entry = self._entry()
        if entry is None and self.require_index:
            self.last_error = f"no {self.which!r} entry in {self.run.index_path.as_posix()}"
            return None

        path = self._resolve_path(entry)
        try:
            stat = path.stat()
        except OSError:
            self.last_error = f"checkpoint not present: {path.as_posix()}"
            return None

        expected = str(entry.get("sha256", "")) if entry else ""
        if not expected:
            if self.require_index:
                self.last_error = f"index entry {self.which!r} carries no sha256"
                return None
            return self._offer(entry, path, stat, sha256="", verified=False)

        if self._seen_key == expected:
            return None
        rejected_at = self._rejected.get(expected)
        if rejected_at is not None and rejected_at == (stat.st_size, stat.st_mtime_ns):
            self.last_error = f"checkpoint {path.name} still fails its index hash; skipping"
            return None

        matched, verified_stat = self._verify(path, expected)
        if not matched or verified_stat is None:
            self.rejected_count += 1
            self._rejected[expected] = (stat.st_size, stat.st_mtime_ns)
            self.last_error = (
                f"content of {path.as_posix()} does not match sha256 {expected[:12]} recorded in "
                "the index (partial write or corrupt file); refusing to load it"
            )
            return None

        self._rejected.pop(expected, None)
        self._seen_key = expected
        return self._offer(entry, path, verified_stat, sha256=expected, verified=True)

    def _offer(
        self,
        entry: dict[str, Any] | None,
        path: Path,
        stat: os.stat_result,
        sha256: str,
        verified: bool,
    ) -> CheckpointInfo | None:
        """Build and record the :class:`CheckpointInfo` this poll is handing out.

        In unverified mode the file is stat-ed a second time first: a file still being written
        almost always changes size between the two samples, so this rejects the common partial
        write even without a recorded hash.

        Args:
            entry: The index entry, possibly None.
            path: Checkpoint file.
            stat: A stat of the file.
            sha256: The verified digest, or ``""``.
            verified: Whether the digest was checked.

        Returns:
            The new :class:`CheckpointInfo`, or None when an unverified file is still settling.
        """
        if not verified:
            key = f"{stat.st_size}:{stat.st_mtime_ns}"
            if self._seen_key == key:
                return None
            time.sleep(self.hash_retry_delay)
            try:
                again = path.stat()
            except OSError:
                self.last_error = f"checkpoint vanished while sizing it: {path.as_posix()}"
                return None
            if (again.st_size, again.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
                self.last_error = f"{path.name} is still changing size; waiting for it to settle"
                return None
            self._seen_key = key
            stat = again

        self.last_error = ""
        self._current = CheckpointInfo(
            which=self.which,
            path=path,
            iteration=int(entry.get("iteration", -1)) if entry else -1,
            metric_name=str(entry.get("metric_name", "")) if entry else "",
            metric_value=_as_float(entry.get("metric_value")) if entry else None,
            sha256=sha256,
            size=int(stat.st_size),
            mtime=float(stat.st_mtime),
            verified=verified,
        )
        return self._current

    def wait_for_new(
        self,
        timeout: float | None = None,
        poll_interval: float | None = None,
        stop: Callable[[], bool] | None = None,
    ) -> CheckpointInfo | None:
        """Block until a new checkpoint appears.

        Args:
            timeout: Seconds to wait, or None to wait forever.
            poll_interval: Override for this call's polling period.
            stop: Optional predicate polled alongside; returning True aborts the wait.

        Returns:
            The new checkpoint, or None on timeout or abort.
        """
        interval = self.poll_interval if poll_interval is None else float(poll_interval)
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        while True:
            if stop is not None and stop():
                return None
            found = self.poll()
            if found is not None:
                return found
            if deadline is not None and time.monotonic() >= deadline:
                return None
            time.sleep(min(interval, 0.25) if deadline is not None else interval)


def _as_float(value: Any) -> float | None:
    """Coerce an index field to float, returning None when it is absent or not numeric.

    Args:
        value: Raw JSON value.

    Returns:
        The float, or None.
    """
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
