"""Composes ``figures/latest.png``: the one image that tells you how a run is doing.

The composite is a header strip over a three-column grid of panels. The header carries the run
identity and its liveness, because the first question about a multi-day run is never "what is the
KL doing", it is "is this thing still alive". A heartbeat older than two minutes turns the strip
amber; a crashed run turns it red; a finished run reads green.

Panels come from :mod:`duckiebot_rl.viz.plots`. Any metric key the training loop logged that no
fixed panel claims gets a panel of its own appended, so nothing a run bothered to record is
silently dropped.

Writes are atomic. ``latest.png`` is rendered to bytes, then swapped into place through
:func:`duckiebot_rl.viz.run_dir.atomic_write_bytes`, so an image viewer polling the file never
loads a half-written PNG.
"""

from __future__ import annotations

import io
import math
import os
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from duckiebot_rl.viz import plots
from duckiebot_rl.viz.plots import (
    PALETTE,
    MetricTable,
    Panel,
    build_panels,
    fmt_duration,
    fmt_int,
    fmt_metric,
    require_matplotlib,
    style_context,
)
from duckiebot_rl.viz.run_dir import (
    RunDir,
    RunStatus,
    atomic_write_bytes,
    status_age_s,
    utc_now_iso,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = ["describe_state", "render_dashboard", "watch"]

STALE_AFTER_S = 120.0
"""A heartbeat older than this is shown as stale. Two minutes is longer than any single PPO
iteration on the target hardware, so amber always means something is actually wrong."""

NCOLS = 3
WIDTH_IN = 16.0
"""16 inches at the default 100 dpi is the required 1600 px reading width."""

# Geometry in inches. Matplotlib's automatic layout engines are not used: ``constrained`` gives
# up on this figure (one panel replaces its cell with a nested sub-gridspec, and the engine
# reports "axes sizes collapsed to zero"), and ``tight`` is order-dependent. Fixed inches are
# predictable, and every gap below is sized for the thing that has to fit in it.
PAD_LEFT_IN = 0.68  # y tick labels plus a rotated y axis title
PAD_RIGHT_IN = 0.36  # the right-hand twin axis of the PPO health panel
TOP_PAD_IN = 0.18
HEADER_IN = 1.08  # three text lines
HEADER_GAP_IN = 0.46
PANEL_H_IN = 2.05
PANEL_GAP_IN = 0.90  # x axis label of the row above plus the title of the row below
COL_GAP_IN = 1.15
FOOTER_IN = 0.68  # x tick labels and the x axis title of the bottom row, plus the provenance line


def _blend(color: str, background: str, weight: float) -> tuple[float, float, float]:
    """Mix a colour into a background.

    Args:
        color: Foreground hex colour.
        background: Background hex colour.
        weight: Amount of foreground in [0, 1].

    Returns:
        An RGB tuple, used for the header's tinted fill.
    """
    from matplotlib.colors import to_rgb

    fore = to_rgb(color)
    back = to_rgb(background)
    return tuple(weight * f + (1.0 - weight) * b for f, b in zip(fore, back, strict=True))  # type: ignore[return-value]


def describe_state(status: RunStatus | None, age_s: float | None) -> tuple[str, str]:
    """Turn a heartbeat into a label and an accent colour.

    Args:
        status: The decoded heartbeat, or None when ``status.json`` does not exist yet.
        age_s: Seconds since the heartbeat was written, or None.

    Returns:
        ``(label, hex_colour)``. The label always says the state in words, so the colour is
        reinforcement and never the only carrier of meaning.
    """
    if status is None:
        return "NO STATUS YET", PALETTE.ink_muted
    state = (status.state or "").lower()
    if state == "crashed":
        return "CRASHED", PALETTE.critical
    if state == "finished":
        return "FINISHED", PALETTE.good
    if age_s is not None and age_s > STALE_AFTER_S:
        return f"STALE, no heartbeat for {fmt_duration(age_s)}", PALETTE.warning
    return "RUNNING", PALETTE.good


def _artifact_note(run: RunDir) -> str:
    """Summarise which rollout artifacts exist, for the footer.

    Args:
        run: The run directory.

    Returns:
        A short human string, for example ``"video 2.4 MB - obs 61 KB"``.
    """
    parts: list[str] = []
    for label, path in (
        ("video", run.latest_video_gif if run.latest_video_gif.exists() else run.latest_video_mp4),
        ("obs", run.latest_obs),
    ):
        try:
            size = path.stat().st_size
        except OSError:
            continue
        parts.append(f"{label} {size / 1024:.0f} KB" if size < 1 << 20 else f"{label} {size / 2**20:.1f} MB")
    return " - ".join(parts) if parts else "no rollout artifacts yet"


def _header(ax: Axes, run: RunDir, status: RunStatus | None, table: MetricTable) -> None:
    """Draw the header strip: identity, liveness and the headline numbers.

    Args:
        ax: A full-width axes with its frame turned off.
        run: The run directory being reported.
        status: The decoded heartbeat, or None.
        table: The metrics table, used as a fallback when there is no heartbeat.
    """
    age = status_age_s(status)
    label, accent = describe_state(status, age)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_facecolor(_blend(accent, PALETTE.plane, 0.10))
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.add_patch(
        plots.plt.Rectangle(
            (0.0, 0.0), 0.0038, 1.0, transform=ax.transAxes, color=accent, clip_on=False, zorder=5
        )
    )

    ax.text(
        0.011,
        0.79,
        run.run_id,
        fontsize=13.5 if len(run.run_id) <= 44 else 11.5,
        fontweight="bold",
        color=PALETTE.ink,
        va="center",
    )
    ax.text(0.011, 0.45, label, fontsize=10.5, fontweight="bold", color=accent, va="center")
    if status is not None:
        meta = (
            f"pid {status.pid} - {status.device or 'device n/a'}"
            + (f" - {status.num_envs} envs" if status.num_envs else "")
            + (f" - {status.vram_used_mb:,.0f} MiB VRAM" if status.vram_used_mb else "")
            + f" - commit {str(status.git_commit)[:8]}"
        )
        ax.text(0.011, 0.14, meta, fontsize=8.5, color=PALETTE.ink_secondary, va="center")

    iteration = status.iteration if status else (int(table.x[-1]) if len(table) else 0)
    steps = status.total_timesteps if status else int(table.last("total_timesteps") or 0)
    wall = status.wall_clock_s if status else (table.last("wall_clock_s") or 0.0)
    rate = status.steps_per_s if status else 0.0
    best_name = (status.best_metric_name if status else "") or "ep_return_mean"
    best_value = status.best_metric_value if status else table.last(best_name)
    best_iter = status.best_iteration if status else None

    fields: Sequence[tuple[str, str]] = (
        ("iteration", fmt_int(iteration)),
        ("env steps", fmt_int(steps)),
        ("wall clock", fmt_duration(wall)),
        ("throughput", f"{rate:,.0f} steps/s" if rate else "-"),
        (
            f"best {best_name}",
            fmt_metric(best_value) + (f"  @ iter {fmt_int(best_iter)}" if best_iter is not None else ""),
        ),
        ("heartbeat age", fmt_duration(age) if age is not None else "-"),
    )
    left, right = 0.315, 0.998
    span = (right - left) / len(fields)
    for index, (name, value) in enumerate(fields):
        x = left + index * span
        ax.text(x, 0.72, name, fontsize=8.0, color=PALETTE.ink_muted, va="center")
        ax.text(x, 0.34, value, fontsize=11.0, fontweight="bold", color=PALETTE.ink, va="center")


def _footer(figure: Figure, run: RunDir, table: MetricTable, panel_count: int) -> None:
    """Draw the provenance line at the bottom of the composite.

    Args:
        figure: The figure being composed.
        run: The run directory.
        table: The metrics table.
        panel_count: How many panels were drawn.
    """
    text = (
        f"{str(run.root).replace(os.sep, '/')}   -   {len(table)} iterations logged, "
        f"{panel_count} panels   -   {_artifact_note(run)}   -   rendered {utc_now_iso()}"
    )
    figure.text(0.006, 0.004, text, fontsize=7.5, color=PALETTE.ink_muted, ha="left", va="bottom")


def _draw_panel(ax: Axes, panel: Panel, table: MetricTable) -> None:
    """Draw one panel, converting any failure into a visible message.

    Args:
        ax: Target axes.
        panel: The panel.
        table: Source table.
    """
    try:
        panel.draw(ax, table)
    except Exception as exc:  # deliberately broad: one bad panel must never cost the whole figure
        ax.clear()
        ax.set_title(panel.title)
        ax.text(
            0.5,
            0.5,
            f"panel failed:\n{exc!r}"[:160],
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8,
            color=PALETTE.critical,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)


def render_dashboard(
    run: RunDir | str | os.PathLike[str],
    panels: bool = False,
    dpi: int = 100,
    width_in: float = WIDTH_IN,
    max_extra: int = 12,
) -> Path:
    """Render the composite dashboard, and optionally every panel standalone.

    Degenerate inputs are handled rather than avoided: a run with zero rows renders a header and
    a grid of "not logged yet" panels, one row renders single markers, an all-NaN column renders
    its own message, and a crashed run renders red. None of them raise.

    Args:
        run: The run directory, or a path to it.
        panels: Also write ``figures/<panel>.png`` for each panel.
        dpi: Output resolution. The default gives a 1600 px wide composite.
        width_in: Figure width in inches.
        max_extra: Cap on discovered extra panels.

    Returns:
        The path of ``metrics/graphs/_dashboard.png``.
    """
    require_matplotlib()
    run_dir = run if isinstance(run, RunDir) else RunDir.open(run)
    run_dir.graphs_dir.mkdir(parents=True, exist_ok=True)

    table = MetricTable.from_rows(run_dir.read_metrics())
    status = run_dir.read_status()
    panel_list = build_panels(table, max_extra=max_extra)
    rows = max(1, math.ceil(len(panel_list) / NCOLS))

    height = (
        TOP_PAD_IN + HEADER_IN + HEADER_GAP_IN + rows * PANEL_H_IN + (rows - 1) * PANEL_GAP_IN + FOOTER_IN
    )
    panel_w = (width_in - PAD_LEFT_IN - PAD_RIGHT_IN - (NCOLS - 1) * COL_GAP_IN) / NCOLS

    with style_context():
        figure: Figure = plots.plt.figure(figsize=(width_in, height), dpi=dpi)
        grid = figure.add_gridspec(
            nrows=rows,
            ncols=NCOLS,
            left=PAD_LEFT_IN / width_in,
            right=1.0 - PAD_RIGHT_IN / width_in,
            bottom=FOOTER_IN / height,
            top=1.0 - (TOP_PAD_IN + HEADER_IN + HEADER_GAP_IN) / height,
            wspace=COL_GAP_IN / panel_w,
            hspace=PANEL_GAP_IN / PANEL_H_IN,
        )
        header_ax = figure.add_axes(
            (
                PAD_LEFT_IN / width_in,
                1.0 - (TOP_PAD_IN + HEADER_IN) / height,
                (width_in - PAD_LEFT_IN - PAD_RIGHT_IN) / width_in,
                HEADER_IN / height,
            )
        )
        _header(header_ax, run_dir, status, table)
        for index, panel in enumerate(panel_list):
            ax = figure.add_subplot(grid[index // NCOLS, index % NCOLS])
            _draw_panel(ax, panel, table)
        _footer(figure, run_dir, table, len(panel_list))

        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=dpi, facecolor=figure.get_facecolor())
        plots.plt.close(figure)

    atomic_write_bytes(run_dir.dashboard_figure, buffer.getvalue())
    if panels:
        for panel in panel_list:
            _render_panel_atomic(run_dir, panel, table, dpi=max(dpi, 110))
    return run_dir.dashboard_figure


def _render_panel_atomic(run_dir: RunDir, panel: Panel, table: MetricTable, dpi: int) -> Path:
    """Render one standalone panel PNG atomically.

    Args:
        run_dir: The run directory.
        panel: The panel to render.
        table: Source table.
        dpi: Output resolution.

    Returns:
        The path written.
    """
    require_matplotlib()
    with style_context():
        figure: Figure = plots.plt.figure(figsize=(7.4, 3.9), dpi=dpi)
        grid = figure.add_gridspec(1, 1, left=0.105, right=0.955, top=0.885, bottom=0.135)
        _draw_panel(figure.add_subplot(grid[0, 0]), panel, table)
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=dpi, facecolor=figure.get_facecolor())
        plots.plt.close(figure)
    return atomic_write_bytes(run_dir.panel_path(panel.slug), buffer.getvalue())


def watch(
    run: RunDir | str | os.PathLike[str],
    interval: float = 20.0,
    panels: bool = False,
    dpi: int = 100,
    max_iterations: int | None = None,
    on_render: Callable[[Path, int], None] | None = None,
    exit_when_done: bool = True,
    grace_renders: int = 1,
) -> Path | None:
    """Re-render the composite whenever ``metrics.jsonl`` grows, until the run ends.

    Safe to start before training does: a missing run directory is polled for, not an error. Safe
    to keep running after training ends: once ``status.json`` reports ``finished`` or ``crashed``
    one final render is made and the loop returns.

    Args:
        run: The run directory, or a path to it.
        interval: Seconds between polls.
        panels: Also write standalone panel PNGs on every render.
        dpi: Output resolution.
        max_iterations: Stop after this many polls. Used by the tests; None means no limit.
        on_render: Called with ``(figure_path, render_count)`` after each successful render.
        exit_when_done: Return once the run reports a terminal state.
        grace_renders: How many extra renders to make after the terminal state is seen, so the
            final iteration's metrics are never missing from the last figure.

    Returns:
        The figure path if anything was rendered, else None.
    """
    run_dir = run if isinstance(run, RunDir) else RunDir.open(run)
    fingerprint: tuple[int, float] | None = None
    rendered: Path | None = None
    renders = 0
    polls = 0
    terminal_seen = 0

    while True:
        polls += 1
        current = run_dir.metrics_fingerprint()
        status = run_dir.read_status()
        terminal = status is not None and status.state in ("finished", "crashed")
        changed = current != fingerprint

        if run_dir.exists() and (changed or terminal):
            fingerprint = current
            try:
                rendered = render_dashboard(run_dir, panels=panels, dpi=dpi)
                renders += 1
                if on_render is not None:
                    on_render(rendered, renders)
            except Exception as exc:  # deliberately broad: a transient read must not kill the watcher
                print(f"[dashboard] render failed, will retry: {exc!r}", flush=True)

        if terminal:
            terminal_seen += 1
            if exit_when_done and terminal_seen > grace_renders:
                return rendered
        if max_iterations is not None and polls >= max_iterations:
            return rendered
        time.sleep(max(0.05, interval))


def summarise(run: RunDir | str | os.PathLike[str]) -> str:
    """Return a one-screen text summary of a run, for logs and for ``--once`` output.

    Args:
        run: The run directory, or a path to it.

    Returns:
        A multi-line human-readable report.
    """
    run_dir = run if isinstance(run, RunDir) else RunDir.open(run)
    table = MetricTable.from_rows(run_dir.read_metrics())
    status = run_dir.read_status()
    age = status_age_s(status)
    label, _ = describe_state(status, age)
    index = run_dir.read_index()
    lines = [
        f"run        : {run_dir.run_id}",
        f"path       : {str(run_dir.root).replace(os.sep, '/')}",
        f"state      : {label}",
        f"iterations : {len(table)} logged, last iteration {fmt_int(status.iteration if status else 0)}",
        f"env steps  : {fmt_int(status.total_timesteps if status else table.last('total_timesteps'))}",
        f"wall clock : {fmt_duration(status.wall_clock_s if status else table.last('wall_clock_s'))}",
    ]
    if status is not None and status.best_metric_value is not None:
        lines.append(
            f"best       : {status.best_metric_name} = {fmt_metric(status.best_metric_value)} "
            f"at iteration {fmt_int(status.best_iteration)}"
        )
    for kind in ("latest", "best"):
        entry: Any = index.get(kind)
        if isinstance(entry, dict):
            lines.append(
                f"{kind:<11}: {entry.get('file')} iter {fmt_int(entry.get('iteration'))} "
                f"{entry.get('metric_name', '')}={fmt_metric(entry.get('metric_value'))} "
                f"sha256 {str(entry.get('sha256', ''))[:12]}"
            )
    return "\n".join(lines)
