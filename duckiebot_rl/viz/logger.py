"""``TrainLogger``: the whole integration surface between a training loop and the run directory.

This API is deliberately tiny and is meant to stay stable, because ``scripts/train.py`` is owned
by someone else. Four calls do everything::

    from duckiebot_rl.viz import TrainLogger

    log = TrainLogger.create("runs", name="lanefollow", seed=0, config=cfg,
                             num_envs=cfg.num_envs, device="cuda:0")
    for iteration in range(1, n_iters + 1):
        ...
        log.log_iteration({"iteration": iteration, "total_timesteps": steps, ...})
        log.save_checkpoint(lambda path: save_checkpoint(path, learner, iteration, steps, curric),
                            metric=metrics["ep_return_mean"])
    log.finish()

Or, more simply, as a context manager, which also records ``state: crashed`` when the loop raises::

    with TrainLogger.create("runs", name="lanefollow", seed=0, config=cfg) as log:
        ...

Everything the logger writes goes through :mod:`duckiebot_rl.viz.run_dir`, so no path is
hardcoded here either. Torch is imported lazily: the module imports cleanly in a torch-free
interpreter, which is what lets ``scripts/dashboard.py`` share the package.
"""

from __future__ import annotations

import os
import shutil
import time
import warnings
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from duckiebot_rl.viz.metrics_logger import EpisodeMetricsLogger, EpisodeRecord, ResumeState
from duckiebot_rl.viz.plot_writer import DEFAULT_REFRESH_S, make_plot_writer
from duckiebot_rl.viz.run_dir import (
    DEFAULT_SESSION,
    REQUIRED_METRIC_KEYS,
    RunDir,
    RunStatus,
    _replace_with_retry,
    utc_now_iso,
)

__all__ = ["TrainLogger"]

CheckpointWriter = Callable[[Path], Any]
"""A callback that writes a checkpoint to the path it is given.

Passing one of these to :meth:`TrainLogger.save_checkpoint` is the preferred form, because it
lets the caller delegate to :func:`duckiebot_rl.ppo.checkpoint.save_checkpoint`, which already
saves the model, optimiser, normalisers, curriculum and every RNG stream.
"""


_CHART_KEYS = frozenset(
    {
        "ep_return_mean",
        "ep_return_std",
        "ep_len_mean",
        "success_rate",
        "steps_per_s",
        "wall_clock_s",
        "lane_dev_rms_m",
        "lane_dev_max_m",
    }
)
_LOSS_KEYS = frozenset(
    {
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clipfrac",
        "explained_variance",
        "grad_norm",
        "learning_rate",
        "bounds_loss",
    }
)


def _scalar_tag(key: str) -> str:
    """Group a flat metrics key into a house-standard TensorBoard tag.

    The reference layout groups scalars as ``charts/``, ``episode/`` and ``loss/``, which is what
    gives per-tag PNGs names like ``charts__ep_rew_mean.png``. A key that already carries a slash
    is passed through untouched, so callers keep full control when they want it.

    Args:
        key: A flat metrics-row key, or an already-grouped tag.

    Returns:
        The grouped tag.
    """
    if "/" in key:
        return key
    if key in _CHART_KEYS:
        return f"charts/{key}"
    if key in _LOSS_KEYS:
        return f"loss/{key}"
    if key.startswith("reward_") or key.startswith("reward/"):
        return f"reward/{key.removeprefix('reward_').removeprefix('reward/')}"
    if key.startswith("alpha_"):
        return f"curriculum/{key}"
    return f"metrics/{key}"


def _git_commit() -> str:
    """Return the current git commit, reusing the PPO checkpoint helper when torch is present.

    Returns:
        A 40-character hex commit, or ``"unknown"``.
    """
    try:
        from duckiebot_rl.ppo.checkpoint import git_commit
    except ImportError:
        return "unknown"
    try:
        return git_commit()
    except OSError:
        return "unknown"


def _vram_used_mb() -> float | None:
    """Return CUDA memory reserved by this process, in MiB.

    Reserved rather than allocated: the caching allocator's reservation is what the 8 GiB budget
    is actually spent against, and it is what shows up in ``nvidia-smi``.

    Returns:
        MiB reserved, or None when torch is absent or no CUDA device is in use.
    """
    try:
        import torch
    except ImportError:
        return None
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            return None
        return float(torch.cuda.memory_reserved()) / (1024.0 * 1024.0)
    except (RuntimeError, AssertionError):
        return None


def _atomic_torch_save(payload: Any, destination: Path) -> None:
    """Serialise ``payload`` with ``torch.save`` under the atomicity rule.

    Args:
        payload: Anything ``torch.save`` accepts.
        destination: Final path; written via ``<name>.tmp`` then replaced.

    Raises:
        ImportError: If torch is not installed in this interpreter.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch is a core dependency of training
        raise ImportError(
            "TrainLogger.save_checkpoint(mapping) needs torch. Either install torch, or pass a "
            "callable that writes the checkpoint itself."
        ) from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name + ".tmp")
    with tmp.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    _replace_with_retry(tmp, destination)


def _atomic_copy(source: Path, destination: Path) -> None:
    """Copy a file into place atomically.

    Args:
        source: Existing file.
        destination: Final path; the copy lands on ``<name>.tmp`` first.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name + ".tmp")
    shutil.copyfile(source, tmp)
    _replace_with_retry(tmp, destination)


class TrainLogger:
    """Writes the run directory: config, heartbeat, metrics, checkpoints and figures.

    The logger owns model selection. Give it the metric value each iteration and it decides what
    ``best.pt`` is, records it in ``checkpoints/index.json`` and reports it in ``status.json``.

    Attributes:
        run: The :class:`~duckiebot_rl.viz.run_dir.RunDir` being written.
        status: The live :class:`~duckiebot_rl.viz.run_dir.RunStatus` heartbeat.
    """

    def __init__(
        self,
        run_dir: RunDir | str | os.PathLike[str],
        config: Any = None,
        best_metric: str = "ep_return_mean",
        best_mode: str = "max",
        num_envs: int | None = None,
        device: str = "",
        tensorboard: bool = True,
        archive_every: int = 0,
        warn_missing_keys: bool = True,
        session: str = DEFAULT_SESSION,
        graphs: bool = True,
        graph_refresh_s: float = DEFAULT_REFRESH_S,
    ) -> None:
        """Open (and create) a run directory and write ``config.yaml``.

        Args:
            run_dir: A :class:`RunDir`, or the path of the run directory to create.
            config: The fully resolved config. Written once, verbatim, to ``config.yaml``.
            best_metric: Metric key that drives model selection.
            best_mode: ``"max"`` or ``"min"``.
            num_envs: Parallel environment count, reported in the heartbeat.
            device: Torch device string, reported in the heartbeat.
            tensorboard: Mirror scalars to real TensorBoard event files under
                ``metrics/runs/<session>/`` when torch is importable.
            archive_every: Also write ``model_episode_<N>.pth`` every N checkpoints; 0 disables.
            warn_missing_keys: Warn once if the first logged row omits a contract-required key.
            session: Session id, used to name ``episodes_<session>.csv``,
                ``summary_<session>.json`` and ``metrics/runs/<session>/``.
            graphs: Write the house-standard live PNGs (one per scalar tag plus
                ``_overview.png``) into ``metrics/graphs/`` on a background thread.
            graph_refresh_s: Seconds between background graph re-renders.

        Raises:
            ValueError: If ``best_mode`` is neither ``"max"`` nor ``"min"``.
        """
        if best_mode not in ("max", "min"):
            raise ValueError(f"best_mode must be 'max' or 'min', got {best_mode!r}")
        self.run = run_dir if isinstance(run_dir, RunDir) else RunDir.open(run_dir, create=True)
        self.run.ensure_tree()
        self.best_metric = best_metric
        self.best_mode = best_mode
        self.archive_every = int(archive_every)
        self._warn_missing_keys = warn_missing_keys
        self._checked_keys = False
        self._checkpoint_count = 0
        self._closed = False
        self._start = time.perf_counter()
        self._tb_writer: Any | None = None
        self._tb_requested = tensorboard
        self.session = str(session)

        # House standard: one PNG per scalar tag plus _overview.png in metrics/graphs/, with
        # series persisted so a resumed run EXTENDS the curves. TBPlotWriter also emits real
        # TensorBoard events into metrics/runs/<session>/; PngPlotWriter is the torch-free
        # fallback. Graph failures must never take training down, so this is best-effort.
        self._plots: Any | None = None
        if graphs:
            try:
                self._plots = make_plot_writer(
                    graph_dir=self.run.graphs_dir,
                    tb_log_dir=self.run.tb_run_dir(self.session) if tensorboard else None,
                    refresh_sec=graph_refresh_s,
                )
            except Exception as exc:
                warnings.warn(f"live graphs disabled: {exc!r}", RuntimeWarning, stacklevel=2)
                self._plots = None

        # Per-episode CSV is the house standard's source of truth for resume. Replaying it on
        # construction recovers the episode counter and global_step across restarts.
        self._episodes = EpisodeMetricsLogger(self.run.metrics_dir, session_id=self.session)
        self._last_archived_episode = self._episodes.episode_count

        self.status = RunStatus(
            run_id=self.run.run_id,
            state="running",
            best_metric_name=best_metric,
            num_envs=num_envs,
            device=device,
            git_commit=_git_commit(),
        )
        if config is not None:
            self.run.write_config(config)
        self.run.write_status(self.status)

    # -------------------------------------------------------------------- construction

    @classmethod
    def create(
        cls,
        runs_root: str | os.PathLike[str],
        name: str = "run",
        seed: int = 0,
        config: Any = None,
        run_id: str | None = None,
        **kwargs: Any,
    ) -> TrainLogger:
        """Resolve a contract-shaped run id and open a logger on it.

        Args:
            runs_root: The ``runs/`` directory.
            name: Run name, for example ``lanefollow``.
            seed: Master seed; both go into the run id.
            config: Resolved config, written to ``config.yaml``.
            run_id: Use this exact run id instead of generating one, which is how a resume
                re-attaches to an existing run directory.
            **kwargs: Forwarded to :class:`TrainLogger`.

        Returns:
            A ready logger whose run directory already exists.
        """
        run = RunDir.create(runs_root, name=name, seed=seed, run_id=run_id)
        return cls(run, config, **kwargs)

    # ------------------------------------------------------------------------ metrics

    def log_iteration(self, metrics: Mapping[str, Any]) -> dict[str, Any]:
        """Append one iteration's metrics and refresh the heartbeat.

        Missing bookkeeping keys are filled in: ``iteration`` continues the logger's own count,
        ``wall_clock_s`` is measured here, and ``total_timesteps`` carries forward. Every other
        key is passed through untouched, so a caller can log whatever it likes and the dashboard
        will discover it and give it a panel.

        Args:
            metrics: A flat mapping of scalars. Non-finite values are written as ``null``.

        Returns:
            The row exactly as written to ``metrics.jsonl``.
        """
        row: dict[str, Any] = dict(metrics)
        row.setdefault("iteration", self.status.iteration + 1)
        row.setdefault("total_timesteps", self.status.total_timesteps)
        row["wall_clock_s"] = float(row.get("wall_clock_s", time.perf_counter() - self._start))

        self._check_required_keys(row)
        self.run.append_metrics(row)

        self.status.iteration = int(row["iteration"])
        self.status.total_timesteps = int(row["total_timesteps"] or 0)
        self.status.wall_clock_s = float(row["wall_clock_s"])
        self.status.steps_per_s = (
            self.status.total_timesteps / self.status.wall_clock_s if self.status.wall_clock_s > 0 else 0.0
        )
        self.status.vram_used_mb = _vram_used_mb()
        self.run.write_status(self.status)
        self._mirror_to_tensorboard(row)
        return row

    def heartbeat(self) -> None:
        """Rewrite ``status.json`` without logging a metrics row.

        Useful inside a long iteration, so a dashboard does not paint the run as stale while a
        slow evaluation or video rollout is in progress.
        """
        self.status.wall_clock_s = time.perf_counter() - self._start
        self.status.vram_used_mb = _vram_used_mb()
        self.run.write_status(self.status)

    def _check_required_keys(self, row: Mapping[str, Any]) -> None:
        """Warn once if the contract's required metric keys are not all present.

        Args:
            row: The first row logged.
        """
        if self._checked_keys or not self._warn_missing_keys:
            return
        self._checked_keys = True
        missing = [key for key in REQUIRED_METRIC_KEYS if key not in row]
        if missing:
            warnings.warn(
                "metrics.jsonl rows are missing required key(s): "
                + ", ".join(missing)
                + ". The dashboard will show those panels as 'not logged'.",
                RuntimeWarning,
                stacklevel=3,
            )

    # -------------------------------------------------------------------- checkpoints

    def save_checkpoint(
        self,
        state: CheckpointWriter | Mapping[str, Any],
        metric: float | None = None,
        is_best: bool | None = None,
        iteration: int | None = None,
        archive: bool | None = None,
    ) -> dict[str, Path]:
        """Write ``latest.pt``, promote ``best.pt`` when warranted, and update the index.

        Args:
            state: Either a callable that writes the checkpoint to the path it is handed (the
                preferred form, so the caller can delegate to
                :func:`duckiebot_rl.ppo.checkpoint.save_checkpoint`), or a mapping that is
                serialised here with ``torch.save``.
            metric: The model-selection metric's value at this iteration. Omit it and the
                checkpoint is written but never promoted to ``best.pt``.
            is_best: Force or forbid the promotion. When None (the default) the logger decides by
                comparing ``metric`` against the best seen so far under ``best_mode``.
            iteration: Iteration to record; defaults to the last logged iteration.
            archive: Force or forbid the periodic ``iter_%08d.pt`` copy. When None, ``archive_every``
                decides.

        Returns:
            Mapping of ``{"latest": path}`` plus ``"best"`` and ``"archive"`` when those were
            written.
        """
        step = self.status.iteration if iteration is None else int(iteration)
        latest = self.run.latest_checkpoint
        if callable(state):
            state(latest)
        else:
            _atomic_torch_save(dict(state), latest)

        self._checkpoint_count += 1
        written: dict[str, Path] = {"latest": latest}

        promote = self._is_improvement(metric) if is_best is None else bool(is_best)
        if promote and metric is not None:
            self.status.best_metric_value = float(metric)
            self.status.best_iteration = step
        if promote:
            _atomic_copy(latest, self.run.best_checkpoint)
            written["best"] = self.run.best_checkpoint

        # House standard names the archive by EPISODE (model_episode_<N>.pth), so key it off the
        # episode counter and fall back to the iteration when no episodes have been logged yet.
        episode = self._episodes.episode_count
        keep = archive if archive is not None else self._should_archive()
        if keep:
            archive_path = self.run.archive_checkpoint(episode or step)
            _atomic_copy(latest, archive_path)
            written["archive"] = archive_path
            self._last_archived_episode = episode

        self.run.record_checkpoint(
            latest, iteration=step, metric_name=self.best_metric, metric_value=metric, kinds=["latest"]
        )
        if promote:
            self.run.record_checkpoint(
                self.run.best_checkpoint,
                iteration=step,
                metric_name=self.best_metric,
                metric_value=metric,
                kinds=["best"],
            )
        self.run.write_status(self.status)
        return written

    def _is_improvement(self, metric: float | None) -> bool:
        """Whether ``metric`` beats the best value seen so far.

        Args:
            metric: Candidate value, or None.

        Returns:
            True if the checkpoint should become ``best.pt``. The first non-None metric always
            wins, so a run always has a ``best.pt``.
        """
        if metric is None:
            return False
        current = self.status.best_metric_value
        if current is None:
            return True
        return metric > current if self.best_mode == "max" else metric < current

    def _should_archive(self) -> bool:
        """Whether this checkpoint also lands in the numbered archive.

        The house standard archives every N EPISODES, not every N checkpoints, so this measures
        the episode counter. Before any episode is logged it falls back to counting checkpoints,
        which keeps the archive working for a trainer that never reports episodes.

        Returns:
            True when ``archive_every`` is set and an Nth interval has elapsed.
        """
        if self.archive_every <= 0:
            return False
        episode = self._episodes.episode_count
        if episode <= 0:
            return self._checkpoint_count % self.archive_every == 0
        return episode - self._last_archived_episode >= self.archive_every

    # ------------------------------------------------------------- scalars, graphs, episodes

    @property
    def plots(self) -> Any | None:
        """The scalar plot writer driving ``metrics/graphs/``.

        Returns:
            A :class:`~duckiebot_rl.viz.plot_writer.ScalarPlotWriter`, or None when graphs are
            disabled or matplotlib is missing.
        """
        return self._plots

    @property
    def tensorboard(self) -> Any | None:
        """The underlying TensorBoard ``SummaryWriter``, when one is in use.

        The plot writer owns it, so scalars reach TensorBoard and the PNGs through one call.

        Returns:
            The writer, or None when TensorBoard is unavailable or was disabled.
        """
        return getattr(self._plots, "tb", None) if self._plots is not None else None

    def _mirror_to_tensorboard(self, row: Mapping[str, Any]) -> None:
        """Feed the numeric entries of one row to the plot writer.

        Iteration-scalar tags are grouped house-style (``charts/`` for throughput and episode
        aggregates, ``loss/`` for optimiser diagnostics, ``metrics/`` for the rest) so the
        per-tag PNG filenames come out as ``charts__ep_rew_mean.png`` and friends.

        Args:
            row: The metrics row just written.
        """
        writer = self._plots
        if writer is None:
            return
        step = int(row.get("total_timesteps") or row.get("iteration") or 0)
        for key, value in row.items():
            if key in ("iteration", "total_timesteps") or isinstance(value, bool):
                continue
            if isinstance(value, int | float):
                writer.add_scalar(_scalar_tag(key), float(value), step)

    def log_episode(self, record: EpisodeRecord) -> int:
        """Append one finished episode to ``metrics/episodes_<session>.csv``.

        PPO here is vectorized, so episodes finish continuously across many environments rather
        than one at a time. Each is appended as it ends, carrying its own ``global_step``.

        Args:
            record: The finished episode.

        Returns:
            The episode number assigned.
        """
        return self._episodes.log_episode(record)

    def log_episodes(self, records: Iterable[EpisodeRecord]) -> list[int]:
        """Append a batch of finished episodes, the usual vectorized case.

        Args:
            records: The finished episodes from this step.

        Returns:
            The episode numbers assigned, in order.
        """
        return self._episodes.log_episodes(records)

    def write_log(self, message: str) -> None:
        """Append one timestamped line to ``train.log``.

        The house standard keeps a plain-text log beside the models, so a run is readable without
        any tooling. Failures are swallowed: losing a log line must never end a training run.

        Args:
            message: The line to append, without a trailing newline.
        """
        try:
            with self.run.train_log_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(f"{utc_now_iso()} {message}\n")
        except OSError:
            pass

    def resume_state(self) -> ResumeState:
        """Episode counter and global step recovered from the episode CSV.

        The CSV is the house standard's source of truth for resume, so a restarted or subsequent
        stage continues its counters from here rather than from zero.

        Returns:
            The replayed :class:`~duckiebot_rl.viz.metrics_logger.ResumeState`.
        """
        return self._episodes.resume_state()

    # --------------------------------------------------------------------------- figures

    def render_dashboard(self, panels: bool = False) -> Path | None:
        """Render ``figures/latest.png`` from inside the training process.

        Training does not need this: ``scripts/dashboard.py --watch`` renders out of process for
        free. It exists so a run that finishes unattended still leaves a final figure behind.

        Args:
            panels: Also write each panel as its own PNG.

        Returns:
            The composite figure path, or None when matplotlib is not installed.
        """
        try:
            from duckiebot_rl.viz.dashboard import render_dashboard
        except ImportError as exc:
            warnings.warn(f"dashboard not rendered: {exc}", RuntimeWarning, stacklevel=2)
            return None
        try:
            return render_dashboard(self.run, panels=panels)
        except Exception as exc:  # deliberately broad: a plotting bug must never kill a training run
            warnings.warn(f"dashboard render failed: {exc!r}", RuntimeWarning, stacklevel=2)
            return None

    # ---------------------------------------------------------------------------- finish

    def finish(self, state: str = "finished", render: bool = True) -> None:
        """Write the terminal heartbeat, close TensorBoard and optionally render a last figure.

        Calling this twice is harmless, which matters because the context manager calls it on the
        way out whether or not the loop already did.

        Args:
            state: ``"finished"`` or ``"crashed"``.
            render: Render the overview and the composite one last time.
        """
        if self._closed:
            return
        self._closed = True
        self.status.state = state
        self.status.wall_clock_s = time.perf_counter() - self._start
        if self.status.wall_clock_s > 0:
            self.status.steps_per_s = self.status.total_timesteps / self.status.wall_clock_s
        self.status.vram_used_mb = _vram_used_mb()
        self.run.write_status(self.status)

        # summary_<session>.json, the house standard's end-of-session block.
        try:
            self._episodes.write_summary(time.time())
        except OSError as exc:
            warnings.warn(f"summary not written: {exc}", RuntimeWarning, stacklevel=2)

        # model_final.pth: the house standard's end-of-training snapshot. A crashed run still gets
        # one, because the last reachable weights are exactly what you want to inspect after a
        # crash. Absent a latest checkpoint there is nothing to copy, which is not an error.
        if state == "finished" or self.run.latest_checkpoint.is_file():
            try:
                if self.run.latest_checkpoint.is_file():
                    _atomic_copy(self.run.latest_checkpoint, self.run.final_checkpoint)
            except OSError as exc:
                warnings.warn(f"model_final not written: {exc}", RuntimeWarning, stacklevel=2)
        self.write_log(
            f"run {state}: {self.status.iteration} iterations, {self.status.total_timesteps} steps"
        )

        # Closing the plot writer renders one final time and stops the background thread. It
        # owns the TensorBoard writer, so this closes that too.
        if self._plots is not None:
            try:
                self._plots.close(render=render)
            except Exception as exc:
                warnings.warn(f"final graph render failed: {exc!r}", RuntimeWarning, stacklevel=2)
            self._plots = None
        self._tb_writer = None
        if render:
            self.render_dashboard(panels=True)

    def __enter__(self) -> TrainLogger:
        """Enter the context manager.

        Returns:
            ``self``.
        """
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: Any) -> None:
        self.finish(state="crashed" if exc_type is not None else "finished")
