"""The run-directory contract, implemented exactly once.

Every training run owns one directory tree, in the house standard layout::

    training_results/<run_dir>/
        model_best.pth             best by the run's model-selection metric (the viewer follows it)
        model_latest.pth           frequent resume copy, rewritten every save
        model_episode_<N>.pth      numbered archive
        model_final.pth            written once, at the end of training
        train.log                  the console transcript
        config.yaml                full resolved config, written once at start
        status.json                heartbeat, rewritten atomically every iteration
        videos/                    rollout clips written by scripts/live_view.py --record
        obs/
            latest_obs.png         what the policy actually sees
        checkpoints/
            index.json             INTERNAL integrity aid: {latest, best} with iteration + sha256
        metrics/
            episodes_<session>.csv one row per FINISHED EPISODE, the resume source of truth
            summary_<session>.json session_info + performance_metrics + recent_performance
            metrics.jsonl          append-only, one flat JSON object per PPO iteration, fsync'd
            runs/<session>/        TensorBoard event files (torch interpreters only)
            graphs/
                <tag>.png          one PNG per scalar tag, house style
                _overview.png      the 3-column grid of every tag
                _dashboard.png     the richer composite with the status header
                _series.json       every series, reloaded at startup so curves CONTINUE
                panels/<slug>.png  the composite's panels, standalone

This module is the single source of truth for that layout. Nothing else in the repository may
hardcode one of those paths; import :class:`RunDir` and ask it instead.

**Why the models sit at the root.** The house standard puts ``model_best.pth`` where a human
lands when they open the run folder, and the live viewer hot-reloads exactly that file. The
``checkpoints/index.json`` alongside it is not a second copy of the layout: it records the
SHA-256 of each published file so a reader can reject a torn one, which the bare filename cannot.

**Atomicity.** Readers must never observe a partial file, so every whole-file writer here writes
``<name>.tmp`` beside the target and then :meth:`pathlib.Path.replace` s it into place, which is
atomic on Windows and POSIX alike. ``metrics.jsonl`` and ``episodes_<session>.csv`` are the two
exceptions: they are append-only, so a reader can instead see a torn *final line*, and both
readers drop it.

**Windows sharing violations.** ``os.replace`` needs delete access to the destination, and CPython
opens files without ``FILE_SHARE_DELETE``. A reader that happens to hold ``status.json`` open for
the microsecond the writer swaps it therefore makes the writer raise ``PermissionError`` on
Windows, where the same code never fails on Linux. Both sides retry: writers retry the replace,
readers retry the open. That retry pair is the whole reason a dashboard can safely poll a live run.

**No third-party imports.** This module is pure standard library on purpose: the dashboard must be
runnable from the MuJoCo venv, which has no torch, and from CI, which has no GPU.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = [
    "CHECKPOINT_BEST",
    "CHECKPOINT_FINAL",
    "CHECKPOINT_INDEX",
    "CHECKPOINT_LATEST",
    "DASHBOARD_NAME",
    "DEFAULT_RESULTS_ROOT",
    "DEFAULT_SESSION",
    "OVERVIEW_NAME",
    "REQUIRED_METRIC_KEYS",
    "RUN_STATES",
    "SCHEMA_VERSION",
    "SERIES_NAME",
    "CheckpointEntry",
    "RunDir",
    "RunStatus",
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_text",
    "find_latest_run",
    "json_safe",
    "make_run_id",
    "parse_run_id",
    "parse_utc",
    "read_bytes_tolerant",
    "read_json_tolerant",
    "read_text_tolerant",
    "sha256_file",
    "status_age_s",
    "tag_filename",
    "utc_now",
    "utc_now_iso",
]

SCHEMA_VERSION = 2
"""Bumped whenever the meaning of a ``status.json`` field changes. Version 2 is the house layout:
models at the run-dir root as ``model_*.pth``, metrics under ``metrics/``, graphs under
``metrics/graphs/``."""

DEFAULT_RESULTS_ROOT = "training_results"
"""Default parent of every run directory, relative to the repository root. The house standard is
one folder per training *continuum*, not per stage: a second stage resumes into the same folder
and its curves continue, which is what ``metrics/graphs/_series.json`` exists to make true."""

DEFAULT_SESSION = "train"
"""Session id used when the caller does not name one. It becomes the ``<session>`` in
``episodes_<session>.csv``, ``summary_<session>.json`` and ``metrics/runs/<session>/``."""

CONFIG_NAME = "config.yaml"
STATUS_NAME = "status.json"
TRAIN_LOG_NAME = "train.log"
METRICS_NAME = "metrics.jsonl"
METRICS_DIRNAME = "metrics"
GRAPHS_DIRNAME = "graphs"
TB_RUNS_DIRNAME = "runs"
PANELS_DIRNAME = "panels"
VIDEOS_DIRNAME = "videos"
CHECKPOINT_LATEST = "model_latest.pth"
CHECKPOINT_BEST = "model_best.pth"
CHECKPOINT_FINAL = "model_final.pth"
CHECKPOINT_INDEX = "index.json"
DASHBOARD_NAME = "_dashboard.png"
OVERVIEW_NAME = "_overview.png"
SERIES_NAME = "_series.json"
VIDEO_MP4 = "latest_rollout.mp4"
VIDEO_GIF = "latest_rollout.gif"
OBS_LATEST = "latest_obs.png"

RUN_STATES: tuple[str, ...] = ("running", "finished", "crashed")

REQUIRED_METRIC_KEYS: tuple[str, ...] = (
    "iteration",
    "total_timesteps",
    "wall_clock_s",
    "ep_return_mean",
    "ep_return_std",
    "ep_len_mean",
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "clipfrac",
    "explained_variance",
    "grad_norm",
    "learning_rate",
    "lane_dev_rms_m",
    "lane_dev_max_m",
    "success_rate",
    "alpha_vis",
    "alpha_dyn",
)
"""Keys every ``metrics.jsonl`` row must carry. Extra keys are welcome and are discovered by the
dashboard; these are the ones its fixed panels are built from."""

_RETRY_ATTEMPTS = 60
_RETRY_DELAY_S = 0.01
_RUN_ID_RE = re.compile(r"^(?P<stamp>\d{8}T\d{6}Z)_(?P<name>.+)_seed(?P<seed>\d+)$")
_UNSAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_TAG_UNSAFE_RE = re.compile(r"[^0-9a-zA-Z._-]+")
_ARCHIVE_RE = re.compile(r"^model_episode_(?P<n>\d+)\.pth$")


def _safe_session(session: str) -> str:
    """Sanitise a session id so it is always a legal single path segment.

    Args:
        session: The caller's session id.

    Returns:
        The id with unsafe characters replaced by ``-``, or :data:`DEFAULT_SESSION` when nothing
        usable is left.
    """
    clean = _UNSAFE_NAME_RE.sub("-", str(session)).strip("-")
    return clean or DEFAULT_SESSION


def tag_filename(tag: str) -> str:
    """Collapse a scalar tag into the house standard's PNG stem.

    Every run of characters outside ``[0-9A-Za-z._-]`` becomes a literal double underscore, so
    ``episode/reward`` lands as ``episode__reward.png``, exactly as the reference writer names it.
    This mangling is part of the contract: a person who knows the tag can find the file.

    Args:
        tag: The scalar tag, for example ``loss/policy``.

    Returns:
        The file stem, never empty.
    """
    return _TAG_UNSAFE_RE.sub("__", str(tag)).strip("_") or "metric"


# ------------------------------------------------------------------------------------ time utils


def utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC datetime.

    Returns:
        ``datetime.now`` in UTC.
    """
    return datetime.now(UTC)


def utc_now_iso() -> str:
    """Return the current UTC time as ``YYYY-MM-DDTHH:MM:SSZ``.

    Returns:
        A second-resolution ISO-8601 timestamp with a literal ``Z`` suffix.
    """
    return utc_now().strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc(stamp: str | None) -> datetime | None:
    """Parse a timestamp written by :func:`utc_now_iso`, tolerating junk.

    Args:
        stamp: The string to parse, or None.

    Returns:
        A timezone-aware datetime, or None if ``stamp`` is missing or unparseable.
    """
    if not stamp:
        return None
    try:
        parsed = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


# -------------------------------------------------------------------------- atomic write helpers


def _replace_with_retry(source: Path, destination: Path, attempts: int = _RETRY_ATTEMPTS) -> None:
    """Move ``source`` onto ``destination``, retrying Windows sharing violations.

    Args:
        source: The temporary file to move.
        destination: The final path.
        attempts: How many times to retry before giving up.

    Raises:
        PermissionError: If every attempt lost the race with a reader.
    """
    for attempt in range(attempts):
        try:
            source.replace(destination)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(_RETRY_DELAY_S)


def json_safe(value: Any) -> Any:
    """Coerce a value into something :func:`json.dumps` accepts without a custom encoder.

    numpy and torch scalars go through their ``.item()``; non-finite floats become None, because
    JSON has no NaN and a reader that has to special-case ``NaN`` literals is a reader that
    crashes at 3 a.m.; Paths become forward-slash strings; anything else falls back to ``repr``.

    Args:
        value: The value to coerce.

    Returns:
        A JSON-serialisable structure.
    """
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value).replace("\\", "/")
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, bytes | bytearray):
        return bytes(value).decode("utf-8", errors="replace")
    item_fn = getattr(value, "item", None)
    if callable(item_fn) and not isinstance(value, Iterable):
        try:
            return json_safe(item_fn())
        except (ValueError, TypeError, RuntimeError):
            pass
    if isinstance(value, Iterable):
        return [json_safe(item) for item in value]
    if hasattr(value, "__float__"):
        try:
            return json_safe(float(value))
        except (ValueError, TypeError):
            pass
    return repr(value)


def atomic_write_bytes(path: str | os.PathLike[str], data: bytes, fsync: bool = True) -> Path:
    """Write bytes so that no reader ever observes a partial file.

    The payload goes to ``<name>.tmp`` in the same directory (same filesystem, so the move is a
    rename and not a copy), is flushed and optionally fsync'd, then replaced into place.

    Args:
        path: Destination path. Parent directories are created.
        data: Payload.
        fsync: Force the bytes to the platter before the swap. Disable only in tests.

    Returns:
        The destination path.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name + ".tmp")
    with tmp.open("wb") as handle:
        handle.write(data)
        handle.flush()
        if fsync:
            os.fsync(handle.fileno())
    _replace_with_retry(tmp, destination)
    return destination


def atomic_write_text(path: str | os.PathLike[str], text: str, fsync: bool = True) -> Path:
    """Write UTF-8 text atomically, with LF line endings on every platform.

    Args:
        path: Destination path.
        text: Payload.
        fsync: Force the bytes to the platter before the swap.

    Returns:
        The destination path.
    """
    return atomic_write_bytes(path, text.encode("utf-8"), fsync=fsync)


def atomic_write_json(path: str | os.PathLike[str], payload: Any, fsync: bool = True) -> Path:
    """Serialise ``payload`` through :func:`json_safe` and write it atomically.

    Args:
        path: Destination path.
        payload: Anything :func:`json_safe` can coerce.
        fsync: Force the bytes to the platter before the swap.

    Returns:
        The destination path.
    """
    text = json.dumps(json_safe(payload), indent=2, sort_keys=False) + "\n"
    return atomic_write_text(path, text, fsync=fsync)


# ----------------------------------------------------------------------------- tolerant readers


def read_bytes_tolerant(path: str | os.PathLike[str], attempts: int = _RETRY_ATTEMPTS) -> bytes | None:
    """Read a file that a writer may be replacing underneath us.

    Args:
        path: File to read.
        attempts: How many times to retry a sharing violation.

    Returns:
        The file contents, or None if the file does not exist (yet) or stayed unreadable.
    """
    target = Path(path)
    for attempt in range(attempts):
        try:
            return target.read_bytes()
        except FileNotFoundError:
            return None
        except OSError:
            if attempt == attempts - 1:
                return None
            time.sleep(_RETRY_DELAY_S)
    return None


def read_text_tolerant(path: str | os.PathLike[str], attempts: int = _RETRY_ATTEMPTS) -> str | None:
    """Read a UTF-8 text file, tolerating concurrent replacement.

    Args:
        path: File to read.
        attempts: How many times to retry a sharing violation.

    Returns:
        The decoded text, or None if the file does not exist.
    """
    raw = read_bytes_tolerant(path, attempts=attempts)
    return None if raw is None else raw.decode("utf-8", errors="replace")


def read_json_tolerant(path: str | os.PathLike[str], attempts: int = _RETRY_ATTEMPTS) -> Any | None:
    """Read a JSON file, tolerating concurrent replacement and torn writes.

    A decode failure is retried, because it can only mean the file was written by something that
    did not go through :func:`atomic_write_json`. After the last attempt None is returned rather
    than raised: a dashboard must never die on a bad heartbeat.

    Args:
        path: File to read.
        attempts: How many times to retry.

    Returns:
        The decoded object, or None if unreadable.
    """
    for attempt in range(attempts):
        raw = read_bytes_tolerant(path, attempts=attempts)
        if raw is None:
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            if attempt == attempts - 1:
                return None
            time.sleep(_RETRY_DELAY_S)
    return None


def sha256_file(path: str | os.PathLike[str], chunk: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest of a file.

    Args:
        path: File to hash.
        chunk: Read block size in bytes.

    Returns:
        A 64-character lowercase hex digest, or ``""`` if the file could not be read.
    """
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            while block := handle.read(chunk):
                digest.update(block)
    except OSError:
        return ""
    return digest.hexdigest()


# --------------------------------------------------------------------------------------- run ids


def make_run_id(name: str, seed: int, when: datetime | None = None) -> str:
    """Build a run id of the contract form ``<UTCtimestamp>_<name>_seed<N>``.

    Args:
        name: Human-readable run name, for example ``lanefollow``. Characters outside
            ``[A-Za-z0-9._-]`` become ``-`` so the id is always a legal path segment.
        seed: The run's master seed.
        when: Timestamp to stamp; defaults to now (UTC).

    Returns:
        For example ``20260817T104500Z_lanefollow_seed0``.

    Raises:
        ValueError: If ``name`` is empty once sanitised.
    """
    stamp = (when or utc_now()).astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    clean = _UNSAFE_NAME_RE.sub("-", name).strip("-")
    if not clean:
        raise ValueError(f"run name {name!r} is empty after sanitising")
    return f"{stamp}_{clean}_seed{int(seed)}"


def parse_run_id(run_id: str) -> dict[str, str] | None:
    """Split a run id back into its three parts.

    Args:
        run_id: The directory name to parse.

    Returns:
        ``{"stamp", "name", "seed"}``, or None if ``run_id`` is not of the contract form.
    """
    match = _RUN_ID_RE.match(run_id)
    return None if match is None else match.groupdict()


def find_latest_run(runs_root: str | os.PathLike[str]) -> Path | None:
    """Return the newest run directory under ``runs_root``.

    Ordering is by the run id's UTC stamp when it parses and by directory mtime otherwise, so a
    hand-named directory still resolves.

    Args:
        runs_root: The results root to scan, ``training_results/``.

    Returns:
        The newest run directory, or None if there is none.
    """
    root = Path(runs_root)
    if not root.is_dir():
        return None
    candidates = [child for child in root.iterdir() if child.is_dir()]
    if not candidates:
        return None

    def sort_key(path: Path) -> tuple[int, str, float]:
        parsed = parse_run_id(path.name)
        stamp = parsed["stamp"] if parsed else ""
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        return (1 if parsed else 0, stamp, mtime)

    return max(candidates, key=sort_key)


# ---------------------------------------------------------------------------------------- status


@dataclass
class RunStatus:
    """The ``status.json`` heartbeat payload.

    Attributes:
        run_id: The run directory name.
        state: One of ``running``, ``finished``, ``crashed``.
        iteration: Completed PPO iterations.
        total_timesteps: Environment steps consumed across all envs.
        wall_clock_s: Seconds since the logger was created.
        steps_per_s: Throughput over the whole run so far.
        best_metric_name: Which metric drives model selection.
        best_metric_value: Its best value so far, or None before the first checkpoint.
        best_iteration: The iteration that produced ``best_metric_value``.
        vram_used_mb: Reserved CUDA memory in MiB, or None on CPU.
        num_envs: Parallel environment count.
        device: Torch device string.
        git_commit: Commit the run was launched from.
        pid: Process id of the trainer, so a stale heartbeat can be attributed to a process.
        schema_version: Layout version, see :data:`SCHEMA_VERSION`.
        last_update_utc: Stamped by :meth:`RunDir.write_status` on every write.
    """

    run_id: str = ""
    state: str = "running"
    iteration: int = 0
    total_timesteps: int = 0
    wall_clock_s: float = 0.0
    steps_per_s: float = 0.0
    best_metric_name: str = ""
    best_metric_value: float | None = None
    best_iteration: int | None = None
    vram_used_mb: float | None = None
    num_envs: int | None = None
    device: str = ""
    git_commit: str = "unknown"
    pid: int = field(default_factory=os.getpid)
    schema_version: int = SCHEMA_VERSION
    last_update_utc: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON payload in the contract's field order.

        Returns:
            A plain dict ready for :func:`atomic_write_json`.
        """
        raw = asdict(self)
        order = (
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
        )
        return {key: raw[key] for key in order}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> RunStatus:
        """Rebuild a status from a decoded ``status.json``, ignoring unknown keys.

        Args:
            payload: The decoded mapping, or None.

        Returns:
            A :class:`RunStatus`; an all-defaults instance when ``payload`` is falsy.
        """
        if not payload:
            return cls()
        known = {f.name for f in fields(cls)}
        return cls(**{key: value for key, value in payload.items() if key in known})


def status_age_s(status: RunStatus | Mapping[str, Any] | None, now: datetime | None = None) -> float | None:
    """Return how many seconds ago the heartbeat was written.

    Args:
        status: A :class:`RunStatus`, a decoded status mapping, or None.
        now: Reference time; defaults to now (UTC).

    Returns:
        Age in seconds, never negative, or None if there is no usable timestamp.
    """
    if status is None:
        return None
    stamp = status.last_update_utc if isinstance(status, RunStatus) else status.get("last_update_utc")
    written = parse_utc(stamp)
    if written is None:
        return None
    return max(0.0, ((now or utc_now()) - written).total_seconds())


# ------------------------------------------------------------------------------ checkpoint index


@dataclass
class CheckpointEntry:
    """One row of ``checkpoints/index.json``.

    Attributes:
        file: Checkpoint file name, relative to the run-dir root.
        iteration: The iteration the checkpoint was written at.
        metric_name: Model-selection metric name.
        metric_value: Its value at that iteration.
        sha256: Digest of the file as written, so a copied checkpoint can be verified.
        mtime: POSIX mtime of the file.
        size_bytes: File size.
        utc: When the entry was recorded.
    """

    file: str
    iteration: int
    metric_name: str = ""
    metric_value: float | None = None
    sha256: str = ""
    mtime: float = 0.0
    size_bytes: int = 0
    utc: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return the entry as a plain dict.

        Returns:
            A JSON-ready mapping.
        """
        return asdict(self)


# -------------------------------------------------------------------------------------- run dir


@dataclass(frozen=True)
class RunDir:
    """A handle on one run directory. The only place the layout is known.

    Attributes:
        root: The run directory itself, ``.../training_results/<run_id>``.
        run_id: Its name.
    """

    root: Path
    run_id: str

    # ------------------------------------------------------------------ construction

    @classmethod
    def create(
        cls,
        runs_root: str | os.PathLike[str],
        name: str = "run",
        seed: int = 0,
        run_id: str | None = None,
        when: datetime | None = None,
    ) -> RunDir:
        """Create ``runs_root/<run_id>`` with every subdirectory of the contract.

        Args:
            runs_root: The results root, ``training_results/``; created if absent.
            name: Run name used to build the id when ``run_id`` is not given.
            seed: Master seed used to build the id.
            run_id: Use this exact id instead of generating one.
            when: Timestamp for the generated id.

        Returns:
            The created :class:`RunDir`. Re-creating an existing tree is not an error, so a
            resume reuses its directory.
        """
        resolved = run_id or make_run_id(name, seed, when=when)
        run = cls(root=Path(runs_root) / resolved, run_id=resolved)
        run.ensure_tree()
        return run

    @classmethod
    def open(cls, path: str | os.PathLike[str], create: bool = False) -> RunDir:
        """Open an existing run directory.

        Args:
            path: The run directory itself, not the results root.
            create: Create the tree if it is missing instead of merely describing it.

        Returns:
            A :class:`RunDir`. Nothing is read here, so this is safe to call before training
            has started and even before the directory exists.
        """
        root = Path(path)
        run = cls(root=root, run_id=root.name)
        if create:
            run.ensure_tree()
        return run

    def ensure_tree(self) -> RunDir:
        """Create every directory of the contract.

        Returns:
            ``self``, for chaining.
        """
        for directory in (
            self.root,
            self.checkpoints_dir,
            self.metrics_dir,
            self.graphs_dir,
            self.videos_dir,
            self.obs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return self

    def exists(self) -> bool:
        """Whether the run directory is present on disk.

        Returns:
            True if ``root`` is a directory.
        """
        return self.root.is_dir()

    # ------------------------------------------------------------------------- paths

    @property
    def config_path(self) -> Path:
        """Path of ``config.yaml``.

        Returns:
            The resolved-config path.
        """
        return self.root / CONFIG_NAME

    @property
    def status_path(self) -> Path:
        """Path of ``status.json``.

        Returns:
            The heartbeat path.
        """
        return self.root / STATUS_NAME

    @property
    def train_log_path(self) -> Path:
        """Path of ``train.log``, the console transcript.

        Returns:
            The log path.
        """
        return self.root / TRAIN_LOG_NAME

    @property
    def metrics_dir(self) -> Path:
        """The ``metrics/`` directory of the house standard.

        Returns:
            Directory path.
        """
        return self.root / METRICS_DIRNAME

    @property
    def metrics_path(self) -> Path:
        """Path of ``metrics/metrics.jsonl``, the per-iteration append-only log.

        This is *our* addition to the house standard, not a replacement for the episode CSV: PPO
        here is vectorised, so an iteration and an episode are different units and both are
        recorded. The composite dashboard reads this file; the CSV is the resume source of truth.

        Returns:
            The append-only metrics path.
        """
        return self.metrics_dir / METRICS_NAME

    def episodes_csv(self, session: str = DEFAULT_SESSION) -> Path:
        """Path of ``metrics/episodes_<session>.csv``.

        Args:
            session: The session id.

        Returns:
            The per-episode CSV path.
        """
        return self.metrics_dir / f"episodes_{_safe_session(session)}.csv"

    def summary_json(self, session: str = DEFAULT_SESSION) -> Path:
        """Path of ``metrics/summary_<session>.json``.

        Args:
            session: The session id.

        Returns:
            The summary path.
        """
        return self.metrics_dir / f"summary_{_safe_session(session)}.json"

    def tb_run_dir(self, session: str = DEFAULT_SESSION) -> Path:
        """Path of ``metrics/runs/<session>/``, where TensorBoard event files land.

        Args:
            session: The session id.

        Returns:
            Directory path. Only a torch interpreter ever writes into it.
        """
        return self.metrics_dir / TB_RUNS_DIRNAME / _safe_session(session)

    @property
    def graphs_dir(self) -> Path:
        """The ``metrics/graphs/`` directory: one PNG per scalar tag, plus the overviews.

        Returns:
            Directory path.
        """
        return self.metrics_dir / GRAPHS_DIRNAME

    @property
    def panels_dir(self) -> Path:
        """The ``metrics/graphs/panels/`` directory: the composite's panels, standalone.

        Kept out of ``graphs/`` itself so that directory stays exactly what the house standard
        says it is: one file per scalar tag, plus the overviews.

        Returns:
            Directory path.
        """
        return self.graphs_dir / PANELS_DIRNAME

    @property
    def videos_dir(self) -> Path:
        """The ``videos/`` directory.

        Returns:
            Directory path.
        """
        return self.root / VIDEOS_DIRNAME

    @property
    def obs_dir(self) -> Path:
        """The ``obs/`` directory.

        Returns:
            Directory path.
        """
        return self.root / "obs"

    @property
    def checkpoints_dir(self) -> Path:
        """The ``checkpoints/`` directory, which holds only ``index.json``.

        The published models live at the run-dir root under their house names. This directory
        exists for the integrity index, which is what lets a reader reject a torn ``.pth``.

        Returns:
            Directory path.
        """
        return self.root / "checkpoints"

    @property
    def latest_checkpoint(self) -> Path:
        """Path of ``model_latest.pth``.

        Returns:
            Checkpoint path.
        """
        return self.root / CHECKPOINT_LATEST

    @property
    def best_checkpoint(self) -> Path:
        """Path of ``model_best.pth``, the file the live viewer hot-reloads.

        Returns:
            Checkpoint path.
        """
        return self.root / CHECKPOINT_BEST

    @property
    def final_checkpoint(self) -> Path:
        """Path of ``model_final.pth``.

        Returns:
            Checkpoint path.
        """
        return self.root / CHECKPOINT_FINAL

    @property
    def index_path(self) -> Path:
        """Path of ``checkpoints/index.json``.

        Returns:
            Index path.
        """
        return self.checkpoints_dir / CHECKPOINT_INDEX

    @property
    def series_path(self) -> Path:
        """Path of ``metrics/graphs/_series.json``, the cross-stage continuity record.

        Returns:
            Series path.
        """
        return self.graphs_dir / SERIES_NAME

    @property
    def overview_figure(self) -> Path:
        """Path of ``metrics/graphs/_overview.png``, the house standard's 3-column grid.

        Returns:
            Figure path.
        """
        return self.graphs_dir / OVERVIEW_NAME

    @property
    def dashboard_figure(self) -> Path:
        """Path of ``metrics/graphs/_dashboard.png``, the richer composite.

        Returns:
            Figure path.
        """
        return self.graphs_dir / DASHBOARD_NAME

    @property
    def latest_video_mp4(self) -> Path:
        """Path of ``videos/latest_rollout.mp4``.

        Returns:
            Video path.
        """
        return self.videos_dir / VIDEO_MP4

    @property
    def latest_video_gif(self) -> Path:
        """Path of ``videos/latest_rollout.gif``.

        Returns:
            Video path.
        """
        return self.videos_dir / VIDEO_GIF

    @property
    def latest_obs(self) -> Path:
        """Path of ``obs/latest_obs.png``.

        Returns:
            Observation-preview path.
        """
        return self.obs_dir / OBS_LATEST

    def archive_checkpoint(self, episode: int) -> Path:
        """Path of the numbered archive checkpoint.

        Args:
            episode: The episode (or iteration) number to stamp into the name.

        Returns:
            ``model_episode_<N>.pth``, unpadded, as the house standard writes it.
        """
        return self.root / f"model_episode_{int(episode)}.pth"

    def archived_checkpoints(self) -> list[Path]:
        """List the numbered archive checkpoints, lowest number first.

        Returns:
            Every ``model_episode_<N>.pth`` in the run directory, sorted numerically. Sorting on
            the parsed integer and not on the name matters: lexicographic order would put
            ``model_episode_1000`` before ``model_episode_999``.
        """
        found: list[tuple[int, Path]] = []
        try:
            children = list(self.root.iterdir())
        except OSError:
            return []
        for child in children:
            match = _ARCHIVE_RE.match(child.name)
            if match is not None:
                found.append((int(match.group("n")), child))
        return [path for _, path in sorted(found, key=lambda item: item[0])]

    def resolve_checkpoint(self, name: str) -> Path:
        """Resolve a checkpoint file name recorded in the index to a full path.

        Args:
            name: A bare file name such as ``model_best.pth``. Any directory component is
                discarded, so a hand-edited index can never point outside the run directory.

        Returns:
            The path under the run-dir root.
        """
        return self.root / (Path(str(name)).name or CHECKPOINT_LATEST)

    def graph_path(self, tag: str) -> Path:
        """Path of one scalar tag's PNG.

        Args:
            tag: The scalar tag, for example ``episode/reward``.

        Returns:
            ``metrics/graphs/<mangled tag>.png``; see :func:`tag_filename`.
        """
        return self.graphs_dir / f"{tag_filename(tag)}.png"

    def panel_path(self, panel: str) -> Path:
        """Path of a standalone dashboard-panel PNG.

        Args:
            panel: Panel slug, for example ``ep_return``.

        Returns:
            ``metrics/graphs/panels/<panel>.png``.
        """
        slug = _UNSAFE_NAME_RE.sub("-", panel).strip("-") or "panel"
        return self.panels_dir / f"{slug}.png"

    # ------------------------------------------------------------------------ config

    def write_config(self, config: Any) -> Path:
        """Write the fully resolved config once, atomically.

        YAML is used when PyYAML is importable and JSON otherwise. JSON is a subset of YAML, so
        ``config.yaml`` parses either way and readers need no special case.

        Args:
            config: Any mapping or JSON-coercible object.

        Returns:
            The path written.
        """
        payload = json_safe(config)
        try:
            import yaml
        except ImportError:
            text = json.dumps(payload, indent=2) + "\n"
        else:
            text = yaml.safe_dump(payload, sort_keys=False, default_flow_style=False, allow_unicode=True)
        return atomic_write_text(self.config_path, text)

    def read_config(self) -> Any | None:
        """Read ``config.yaml`` back.

        Returns:
            The decoded config, or None if it is missing or unreadable.
        """
        text = read_text_tolerant(self.config_path)
        if text is None:
            return None
        try:
            import yaml
        except ImportError:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return None
        try:
            return yaml.safe_load(text)
        except yaml.YAMLError:
            return None

    # ------------------------------------------------------------------------ status

    def write_status(self, status: RunStatus) -> Path:
        """Stamp and write the heartbeat atomically.

        ``last_update_utc`` is set here rather than by the caller, so the freshness the dashboard
        reports is always the freshness of the write.

        Args:
            status: The heartbeat to write; its ``run_id`` is filled in when empty.

        Returns:
            The path written.
        """
        if not status.run_id:
            status.run_id = self.run_id
        status.last_update_utc = utc_now_iso()
        status.schema_version = SCHEMA_VERSION
        return atomic_write_json(self.status_path, status.to_dict())

    def read_status(self) -> RunStatus | None:
        """Read the heartbeat, tolerating a concurrent replacement.

        Returns:
            A :class:`RunStatus`, or None if ``status.json`` does not exist yet.
        """
        payload = read_json_tolerant(self.status_path)
        return None if payload is None else RunStatus.from_dict(payload)

    # ----------------------------------------------------------------------- metrics

    def append_metrics(self, row: Mapping[str, Any], fsync: bool = True) -> None:
        """Append one flat JSON object to ``metrics.jsonl``.

        The row is written as a single ``write`` of one line including its newline, then flushed
        and fsync'd. A reader therefore sees either nothing or a complete line except inside the
        write itself, and :meth:`read_metrics` drops a torn trailing line anyway.

        Args:
            row: The iteration's metrics. Values are coerced by :func:`json_safe`, so non-finite
                floats land as ``null`` rather than as invalid JSON.
            fsync: Force the line to the platter. Disable only in tests.

        Raises:
            PermissionError: If the file stayed locked for the whole retry window.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        line = (json.dumps(json_safe(dict(row)), sort_keys=False) + "\n").encode("utf-8")
        for attempt in range(_RETRY_ATTEMPTS):
            try:
                with self.metrics_path.open("ab") as handle:
                    handle.write(line)
                    handle.flush()
                    if fsync:
                        os.fsync(handle.fileno())
                return
            except PermissionError:
                if attempt == _RETRY_ATTEMPTS - 1:
                    raise
                time.sleep(_RETRY_DELAY_S)

    def read_metrics(self) -> list[dict[str, Any]]:
        """Read every complete row of ``metrics.jsonl``.

        A trailing torn line (the writer was mid-append) is silently dropped, and so is any line
        that fails to decode: losing one row of a live plot beats crashing the dashboard.

        Returns:
            The rows in file order; an empty list when the file is absent or empty.
        """
        return list(self.iter_metrics())

    def iter_metrics(self) -> Iterator[dict[str, Any]]:
        """Iterate the rows of ``metrics.jsonl``.

        Yields:
            One decoded row per complete, well-formed line.
        """
        raw = read_bytes_tolerant(self.metrics_path)
        if not raw:
            return
        for line in raw.decode("utf-8", errors="replace").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row

    def metrics_fingerprint(self) -> tuple[int, float]:
        """Cheap change detector for watch mode.

        Returns:
            ``(size_bytes, mtime)`` of ``metrics.jsonl``, or ``(-1, -1.0)`` when it is absent.
        """
        try:
            stat = self.metrics_path.stat()
        except OSError:
            return (-1, -1.0)
        return (stat.st_size, stat.st_mtime)

    # ------------------------------------------------------------------------- index

    def read_index(self) -> dict[str, Any]:
        """Read ``checkpoints/index.json``.

        Returns:
            The decoded index, or ``{}`` if it is missing or unreadable.
        """
        payload = read_json_tolerant(self.index_path)
        return payload if isinstance(payload, dict) else {}

    def record_checkpoint(
        self,
        path: str | os.PathLike[str],
        iteration: int,
        metric_name: str = "",
        metric_value: float | None = None,
        kinds: Iterable[str] = ("latest",),
    ) -> CheckpointEntry:
        """Hash a written checkpoint and record it under one or more index keys.

        Args:
            path: The checkpoint file that was just written.
            iteration: Iteration it belongs to.
            metric_name: Model-selection metric name.
            metric_value: Its value at ``iteration``.
            kinds: Index keys to update, typically ``("latest",)`` or ``("latest", "best")``.

        Returns:
            The entry that was recorded.
        """
        target = Path(path)
        try:
            stat = target.stat()
            mtime, size = stat.st_mtime, stat.st_size
        except OSError:
            mtime, size = 0.0, 0
        entry = CheckpointEntry(
            file=target.name,
            iteration=int(iteration),
            metric_name=metric_name,
            metric_value=None if metric_value is None else float(metric_value),
            sha256=sha256_file(target),
            mtime=mtime,
            size_bytes=size,
            utc=utc_now_iso(),
        )
        index = self.read_index()
        for kind in kinds:
            index[kind] = entry.to_dict()
        atomic_write_json(self.index_path, index)
        return entry
