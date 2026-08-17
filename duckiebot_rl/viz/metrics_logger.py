"""Per-episode CSV and the session summary JSON, in the house standard's layout.

    metrics/
        episodes_<session>.csv   one row per FINISHED episode
        summary_<session>.json   session_info + performance_metrics + recent_performance

Where this deliberately differs from the reference
--------------------------------------------------
The reference trainer runs one environment, so "an episode ends" and "a step of the training loop
completes" are the same event and the CSV is written once per loop iteration. PPO here is
vectorised over 128 to 256 environments, so episodes end continuously and asynchronously: a single
PPO iteration finishes anywhere between zero and a few hundred of them, in no particular order.

So this logger is an **append-per-finished-episode** log rather than an append-per-iteration one.
:meth:`EpisodeMetricsLogger.log_episodes` takes the whole batch a vectorised step produced and
numbers them in arrival order. Everything the reference guarantees still holds: one row per
episode, in a stable column order, appended and flushed as it happens, never rewritten.

The property that matters most is preserved exactly: **resume reads the CSV**. On restart,
:meth:`EpisodeMetricsLogger.resume_state` recovers the episode counter and the global step from
the last well-formed row, so a second stage continues the numbering instead of restarting it.

Memory
------
The reference keeps every row in memory to compute its summary, which on a 455k-episode run means
a 62 MB CSV parsed into 455k dicts. This one keeps running aggregates plus a bounded deque of the
most recent rows, so the summary is identical and the footprint is constant. That matters here:
256 environments finish episodes far faster than one does.
"""

from __future__ import annotations

import csv
import io
import math
import os
import time
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from duckiebot_rl.viz.run_dir import DEFAULT_SESSION, atomic_write_json, read_text_tolerant

__all__ = ["EPISODE_FIELDS", "EpisodeMetricsLogger", "EpisodeRecord", "ResumeState"]

EPISODE_FIELDS: tuple[str, ...] = (
    "episode",
    "score",
    "steps",
    "duration",
    "global_step",
    "timestamp",
    "lane_dev_rms",
    "lane_dev_max",
    "success",
    "termination_reason",
)
"""The CSV column order. Fixed: an existing file's header is never rewritten, so a run resumed by
a future version keeps the columns it started with."""

RECENT_WINDOW = 100
"""How many trailing episodes ``recent_performance`` is computed over."""

_RETRY_ATTEMPTS = 60
_RETRY_DELAY_S = 0.01


@dataclass
class EpisodeRecord:
    """One finished episode.

    Attributes:
        score: Undiscounted return.
        steps: Control steps the episode lasted.
        duration: Simulated seconds the episode lasted.
        global_step: Total environment steps consumed across all envs when it ended.
        timestamp: POSIX time it was recorded.
        lane_dev_rms: RMS lateral deviation from the lane centre, in metres.
        lane_dev_max: Peak lateral deviation, in metres.
        success: Whether the episode reached its horizon without a terminating failure.
        termination_reason: Why it ended, for example ``off_lane`` or ``truncated``.
    """

    score: float = 0.0
    steps: int = 0
    duration: float = 0.0
    global_step: int = 0
    timestamp: float = 0.0
    lane_dev_rms: float = float("nan")
    lane_dev_max: float = float("nan")
    success: bool = False
    termination_reason: str = ""

    def row(self, episode: int) -> list[Any]:
        """Render this record as a CSV row in :data:`EPISODE_FIELDS` order.

        Args:
            episode: The episode number to stamp.

        Returns:
            The row values.
        """
        values: dict[str, Any] = {
            "episode": int(episode),
            "score": float(self.score),
            "steps": int(self.steps),
            "duration": float(self.duration),
            "global_step": int(self.global_step),
            "timestamp": float(self.timestamp),
            "lane_dev_rms": float(self.lane_dev_rms),
            "lane_dev_max": float(self.lane_dev_max),
            "success": bool(self.success),
            "termination_reason": str(self.termination_reason),
        }
        return [values[name] for name in EPISODE_FIELDS]


@dataclass(frozen=True)
class ResumeState:
    """What a restart recovers from an existing CSV.

    Attributes:
        episode: The highest episode number in the file, 0 when there is none.
        global_step: The environment step count that episode ended at.
        rows: How many well-formed rows were read.
    """

    episode: int = 0
    global_step: int = 0
    rows: int = 0


@dataclass
class _Aggregate:
    """Running totals over every episode ever logged into this CSV, history included.

    Attributes:
        count: Episodes seen.
        score_sum: Sum of returns.
        score_max: Highest return.
        score_min: Lowest return.
        steps_sum: Sum of episode lengths.
        steps_max: Longest episode.
        duration_sum: Sum of episode durations.
        success_count: Episodes that ended in success.
        lane_dev_sum: Sum of RMS lane deviations over the episodes that reported one.
        lane_dev_count: How many episodes reported an RMS lane deviation.
        lane_dev_max: Worst peak lane deviation seen.
    """

    count: int = 0
    score_sum: float = 0.0
    score_max: float = float("-inf")
    score_min: float = float("inf")
    steps_sum: int = 0
    steps_max: int = 0
    duration_sum: float = 0.0
    success_count: int = 0
    lane_dev_sum: float = 0.0
    lane_dev_count: int = 0
    lane_dev_max: float = float("-inf")

    def add(self, record: EpisodeRecord) -> None:
        """Fold one episode into the totals.

        Args:
            record: The finished episode.
        """
        self.count += 1
        self.score_sum += record.score
        self.score_max = max(self.score_max, record.score)
        self.score_min = min(self.score_min, record.score)
        self.steps_sum += record.steps
        self.steps_max = max(self.steps_max, record.steps)
        self.duration_sum += record.duration
        self.success_count += int(bool(record.success))
        if math.isfinite(record.lane_dev_rms):
            self.lane_dev_sum += record.lane_dev_rms
            self.lane_dev_count += 1
        if math.isfinite(record.lane_dev_max):
            self.lane_dev_max = max(self.lane_dev_max, record.lane_dev_max)


class EpisodeMetricsLogger:
    """Appends one CSV row per finished episode and writes the session summary.

    Attributes:
        metrics_dir: The ``metrics/`` directory.
        session_id: The ``<session>`` in both file names.
        csv_path: The per-episode CSV.
        summary_path: The session summary JSON.
        start_time: POSIX time the session started, stamped by the caller.
        episode: The episode counter, continued from the CSV on a resume.
    """

    def __init__(self, metrics_dir: str | os.PathLike[str], session_id: str = DEFAULT_SESSION) -> None:
        """Open (creating or continuing) the CSV for a session.

        A fresh file gets the :data:`EPISODE_FIELDS` header. An existing one is streamed to
        recover the episode counter, the global step and the aggregates, and is then appended to
        under **its own** header, so a file written by an earlier column layout is never corrupted.

        Args:
            metrics_dir: The ``metrics/`` directory; created if absent.
            session_id: The session id.
        """
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = str(session_id)
        self.csv_path = self.metrics_dir / f"episodes_{self.session_id}.csv"
        self.summary_path = self.metrics_dir / f"summary_{self.session_id}.json"
        self.start_time: float | None = None
        self._aggregate = _Aggregate()
        self._recent: deque[EpisodeRecord] = deque(maxlen=RECENT_WINDOW)
        self.episode = 0
        self._global_step = 0

        if self.csv_path.is_file() and self.csv_path.stat().st_size > 0:
            self.fields = self._existing_header() or list(EPISODE_FIELDS)
            self._replay()
        else:
            self.fields = list(EPISODE_FIELDS)
            self._write_header()

    # ------------------------------------------------------------------------ resume

    def _existing_header(self) -> list[str] | None:
        """Read the header of an existing CSV.

        Returns:
            The column names, or None if the file could not be read.
        """
        text = read_text_tolerant(self.csv_path)
        if not text:
            return None
        first = text.splitlines()[0] if text.splitlines() else ""
        header = next(csv.reader([first]), [])
        return [name.strip() for name in header] or None

    def _write_header(self) -> None:
        """Create the CSV with its header row."""
        buffer = io.StringIO()
        csv.writer(buffer, lineterminator="\n").writerow(self.fields)
        self._append_text(buffer.getvalue())

    def _replay(self) -> None:
        """Stream an existing CSV to recover the counters and the aggregates.

        Malformed rows are skipped rather than raised on, exactly as the reference does: one bad
        line written during a crash must not stop the next stage from starting.
        """
        text = read_text_tolerant(self.csv_path)
        if not text:
            return
        for row in csv.DictReader(io.StringIO(text)):
            record = _record_from_row(row)
            if record is None:
                continue
            try:
                number = int(float(row.get("episode", 0) or 0))
            except (TypeError, ValueError):
                continue
            self.episode = max(self.episode, number)
            self._global_step = max(self._global_step, record.global_step)
            self._aggregate.add(record)
            self._recent.append(record)

    def resume_state(self) -> ResumeState:
        """Report what the CSV said about where the previous stage stopped.

        Returns:
            A :class:`ResumeState`. All-zero when the CSV is new, which is the correct starting
            point for a fresh run.
        """
        return ResumeState(episode=self.episode, global_step=self._global_step, rows=self._aggregate.count)

    # ----------------------------------------------------------------------- logging

    def set_start_time(self, when: float) -> None:
        """Stamp the session start, used for ``session_info.duration``.

        Args:
            when: POSIX time the session began.
        """
        self.start_time = float(when)

    def log_episode(self, record: EpisodeRecord) -> int:
        """Append one finished episode.

        Args:
            record: The episode. Its ``timestamp`` is filled in here when the caller left it at 0.

        Returns:
            The episode number that was assigned.
        """
        return self.log_episodes([record])[-1]

    def log_episodes(self, records: Iterable[EpisodeRecord]) -> list[int]:
        """Append a whole batch of finished episodes in one write.

        This is the vectorised path: one PPO iteration ends many episodes at once, and appending
        them individually would mean one open/write/fsync per episode on the training thread.

        Args:
            records: The finished episodes, in arrival order.

        Returns:
            The episode numbers assigned, in the same order. Empty for an empty batch.
        """
        batch = list(records)
        if not batch:
            return []
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        numbers: list[int] = []
        now = time.time()
        for record in batch:
            if not record.timestamp:
                record.timestamp = now
            self.episode += 1
            numbers.append(self.episode)
            self._global_step = max(self._global_step, record.global_step)
            self._aggregate.add(record)
            self._recent.append(record)
            writer.writerow(record.row(self.episode))
        self._append_text(buffer.getvalue())
        return numbers

    def _append_text(self, text: str) -> None:
        """Append text to the CSV, retrying the Windows sharing violation.

        Args:
            text: One or more complete lines, each already newline-terminated.
        """
        payload = text.encode("utf-8")
        for attempt in range(_RETRY_ATTEMPTS):
            try:
                with self.csv_path.open("ab") as handle:
                    handle.write(payload)
                    handle.flush()
                return
            except PermissionError:
                if attempt == _RETRY_ATTEMPTS - 1:
                    raise
                time.sleep(_RETRY_DELAY_S)

    # ----------------------------------------------------------------------- summary

    @property
    def episode_count(self) -> int:
        """Episodes recorded in this CSV, history included.

        Returns:
            The row count.
        """
        return self._aggregate.count

    @property
    def global_step(self) -> int:
        """The highest global step any recorded episode ended at.

        Returns:
            The step count.
        """
        return self._global_step

    def recent(self) -> list[EpisodeRecord]:
        """The trailing window used by ``recent_performance``.

        Returns:
            Up to :data:`RECENT_WINDOW` most recent episodes, oldest first.
        """
        return list(self._recent)

    def write_summary(
        self,
        now: float | None = None,
        total_episodes: int | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> Path | None:
        """Write ``summary_<session>.json`` with the reference's three blocks.

        Args:
            now: POSIX time to compute ``duration`` against; defaults to now.
            total_episodes: Episode count to report; defaults to the counter.
            extra: Extra keys merged into ``session_info``, for example the run id.

        Returns:
            The path written, or None when no episode has been recorded yet, since a summary of
            nothing is worse than no summary at all.
        """
        aggregate = self._aggregate
        if aggregate.count == 0:
            return None
        stamp = time.time() if now is None else float(now)
        recent = self.recent()
        payload = {
            "session_info": {
                "session_id": self.session_id,
                "start_time": self.start_time,
                "duration": None if self.start_time is None else stamp - self.start_time,
                "total_episodes": int(self.episode if total_episodes is None else total_episodes),
                **(dict(extra) if extra else {}),
            },
            "performance_metrics": {
                "avg_score": aggregate.score_sum / aggregate.count,
                "max_score": aggregate.score_max,
                "min_score": aggregate.score_min,
                "avg_episode_length": aggregate.steps_sum / aggregate.count,
                "max_episode_length": aggregate.steps_max,
                "avg_episode_duration": aggregate.duration_sum / aggregate.count,
                "success_rate": aggregate.success_count / aggregate.count,
                "failure_rate": 1.0 - aggregate.success_count / aggregate.count,
                "avg_lane_dev_rms_m": (
                    aggregate.lane_dev_sum / aggregate.lane_dev_count if aggregate.lane_dev_count else None
                ),
                "max_lane_dev_m": aggregate.lane_dev_max if aggregate.lane_dev_count else None,
            },
            "recent_performance": _recent_block(recent),
        }
        return atomic_write_json(self.summary_path, payload)


def _recent_block(recent: Sequence[EpisodeRecord]) -> dict[str, Any]:
    """Build the ``recent_performance`` block.

    Args:
        recent: The trailing episode window.

    Returns:
        The block; an empty window yields nulls rather than a division by zero.
    """
    if not recent:
        return {
            "window": 0,
            "recent_avg_score": None,
            "recent_avg_length": None,
            "recent_success_rate": None,
            "recent_avg_lane_dev_rms_m": None,
        }
    deviations = [record.lane_dev_rms for record in recent if math.isfinite(record.lane_dev_rms)]
    return {
        "window": len(recent),
        "recent_avg_score": sum(record.score for record in recent) / len(recent),
        "recent_avg_length": sum(record.steps for record in recent) / len(recent),
        "recent_success_rate": sum(1 for record in recent if record.success) / len(recent),
        "recent_avg_lane_dev_rms_m": (sum(deviations) / len(deviations) if deviations else None),
    }


def _record_from_row(row: Mapping[str, Any]) -> EpisodeRecord | None:
    """Rebuild an :class:`EpisodeRecord` from a decoded CSV row.

    Args:
        row: One ``csv.DictReader`` row.

    Returns:
        The record, or None when the row is malformed. A missing optional column reads as its
        default, so a CSV written by an older column layout still replays.
    """
    try:
        return EpisodeRecord(
            score=float(row.get("score", 0.0) or 0.0),
            steps=int(float(row.get("steps", 0) or 0)),
            duration=float(row.get("duration", 0.0) or 0.0),
            global_step=int(float(row.get("global_step", 0) or 0)),
            timestamp=float(row.get("timestamp", 0.0) or 0.0),
            lane_dev_rms=_optional_float(row.get("lane_dev_rms")),
            lane_dev_max=_optional_float(row.get("lane_dev_max")),
            success=str(row.get("success", "")).strip().lower() in ("true", "1", "yes"),
            termination_reason=str(row.get("termination_reason", "") or ""),
        )
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float:
    """Coerce a CSV cell to float, mapping anything unusable to NaN.

    Args:
        value: The raw cell.

    Returns:
        The float, or NaN. NaN is the right sentinel here because the aggregates skip it, so an
        episode that did not report a lane deviation does not drag the mean toward zero.
    """
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
