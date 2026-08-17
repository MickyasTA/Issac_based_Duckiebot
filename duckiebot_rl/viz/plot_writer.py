"""Live PNG graphs of every scalar, in the house standard's style, refreshed off the hot path.

Two writers share one API, exactly as the reference implementation does:

``TBPlotWriter``
    Writes real TensorBoard event files into ``metrics/runs/<session>/`` **and** the PNGs. Needs
    torch, which is imported lazily so this module stays importable without it.
``PngPlotWriter``
    Identical ``add_scalar`` / ``add_text`` / ``render`` / ``close`` signatures, PNGs only, no
    torch. This is the one an interpreter without torch uses.

Both write into the same directory::

    metrics/runs/<session>/    TensorBoard events           (TBPlotWriter only)
    metrics/graphs/<tag>.png   one PNG per scalar tag       (both)
    metrics/graphs/_overview.png   the 3-column grid        (both)
    metrics/graphs/_series.json    every series, persisted  (both)

Three properties make this useful rather than decorative, and none of them is optional.

**Continuity across processes.** ``_series.json`` is written on every render and reloaded in the
constructor. A second curriculum stage, or a resume after a crash, therefore *extends* the same
curves instead of starting new ones. That is what lets the house standard keep one run folder per
training continuum rather than one per stage, and it is the property to check first if a graph
ever looks like it restarted.

**Rendering never touches the training thread.** A daemon thread re-renders every ``refresh_sec``
whenever something changed, and every exception inside it is swallowed. A broken plot must cost a
figure, never a run that has been going for two days.

**Writes are atomic.** Each PNG is rendered to bytes and swapped into place, so an image viewer
polling ``_overview.png`` never loads a half-written file. This is the one place this module
deliberately improves on the reference, which calls ``savefig`` straight onto the target path.

The style is the reference's, matched call for call: raw series in ``#9ecae1`` at linewidth 1.0,
an EMA with ``alpha=0.1`` over it in ``#08519c`` at linewidth 2.0, the tag verbatim as the title,
and a ``last=`` badge in the bottom-right corner on a ``#fff7bc`` rounded patch.
"""

from __future__ import annotations

import contextlib
import json
import math
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from duckiebot_rl.viz.run_dir import (
    OVERVIEW_NAME,
    SERIES_NAME,
    atomic_write_bytes,
    atomic_write_text,
    read_text_tolerant,
    tag_filename,
)

__all__ = [
    "DEFAULT_REFRESH_S",
    "EMA_ALPHA",
    "OVERVIEW_TITLE",
    "RAW_COLOR",
    "SMOOTH_COLOR",
    "PngPlotWriter",
    "ScalarPlotWriter",
    "TBPlotWriter",
    "ema",
    "make_plot_writer",
]

RAW_COLOR = "#9ecae1"
"""Colour of the unsmoothed series. Pale enough to read as texture under the trend line."""

SMOOTH_COLOR = "#08519c"
"""Colour of the EMA. Same hue family as :data:`RAW_COLOR`, four steps darker."""

BADGE_FACE = "#fff7bc"
"""Fill of the ``last=`` annotation patch."""

EMA_ALPHA = 0.1
"""Smoothing factor of the trend line. Fixed by the house standard, not a tuning knob."""

DEFAULT_REFRESH_S = 20.0
"""Seconds between background renders."""

MIN_REFRESH_S = 2.0
"""Floor on the refresh period. Below this the renderer starts competing with training for the
GIL on a run that logs every iteration."""

OVERVIEW_TITLE = "duckiebot - training metrics (live)"
"""Suptitle of ``_overview.png``."""

OVERVIEW_COLS = 3
"""Columns in the overview grid."""

_MIN_POINTS_FOR_EMA = 3
"""Below this a smoothed line says nothing the raw line does not, so it is not drawn."""


def ema(values: list[float], alpha: float = EMA_ALPHA) -> list[float]:
    """Exponential moving average, as a plain list, seeded on the first sample.

    Args:
        values: The raw series.
        alpha: Smoothing factor in (0, 1]; larger follows the raw series more closely.

    Returns:
        A list of the same length; empty when ``values`` is empty.
    """
    if not values:
        return []
    out = [values[0]]
    for value in values[1:]:
        out.append(alpha * value + (1.0 - alpha) * out[-1])
    return out


class ScalarPlotWriter:
    """Accumulates scalar series and renders them as PNGs on a background thread.

    Subclasses add whatever else they want to do with a scalar; this class owns the series, the
    persistence, the thread and the drawing.

    Attributes:
        graph_dir: Directory the PNGs and ``_series.json`` are written into.
    """

    def __init__(self, graph_dir: str | Path, refresh_sec: float = DEFAULT_REFRESH_S) -> None:
        """Load any persisted series and start the background renderer.

        Args:
            graph_dir: Directory for the PNGs; created if absent.
            refresh_sec: Seconds between background renders, floored at :data:`MIN_REFRESH_S`.

        Raises:
            ImportError: If matplotlib is not installed, with the exact pip command to fix it.
        """
        from duckiebot_rl.viz.plots import require_matplotlib

        require_matplotlib()

        self.graph_dir = Path(graph_dir)
        self.graph_dir.mkdir(parents=True, exist_ok=True)
        self._series_path = self.graph_dir / SERIES_NAME
        self._series: dict[str, tuple[list[float], list[float]]] = {}
        self._load_series()
        self._lock = threading.Lock()
        # pyplot keeps global state, so two concurrent renders would interleave figures. close()
        # renders on the caller's thread while the daemon may be mid-render, so they take turns.
        self._render_lock = threading.Lock()
        self._dirty = False
        self._closed = False
        self._stop = threading.Event()
        self._refresh = max(MIN_REFRESH_S, float(refresh_sec))
        self._also_render: Callable[[], None] | None = None
        self._thread = threading.Thread(target=self._loop, name="viz-plot-writer", daemon=True)
        self._thread.start()

    # ----------------------------------------------------------------------- continuity

    def _load_series(self) -> None:
        """Reload ``_series.json`` so a resumed or subsequent stage extends the same curves.

        A missing, truncated or hand-mangled file is treated as "no history": losing continuity is
        a cosmetic regression, refusing to start training over it would not be.
        """
        text = read_text_tolerant(self._series_path)
        if not text:
            return
        try:
            payload = json.loads(text)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return
        if not isinstance(payload, dict):
            return
        for tag, pair in payload.items():
            if not isinstance(pair, list | tuple) or len(pair) != 2:
                continue
            steps, values = pair
            if not isinstance(steps, list) or not isinstance(values, list):
                continue
            width = min(len(steps), len(values))
            restored_steps = [float(step) for step in steps[:width]]
            restored_values = [float(value) for value in values[:width]]
            if restored_values:
                self._series[str(tag)] = (restored_steps, restored_values)

    def _persist_series(self, data: Mapping[str, tuple[list[float], list[float]]]) -> None:
        """Write ``_series.json`` atomically.

        ``fsync`` is off here on purpose. The file is a convenience for the *next* process, it is
        rewritten every render, and on a long run it grows to tens of megabytes; forcing that to
        the platter every twenty seconds would buy nothing and cost real I/O.

        Args:
            data: The snapshot to persist.
        """
        with contextlib.suppress(OSError):
            atomic_write_text(self._series_path, json.dumps(data, separators=(",", ":")), fsync=False)

    # -------------------------------------------------------------------------- recording

    def add_scalar(
        self, tag: str, scalar_value: Any, global_step: int | None = None, *_: Any, **__: Any
    ) -> None:
        """Record one scalar sample.

        Args:
            tag: Scalar tag, for example ``episode/reward``. It becomes the plot title verbatim
                and, mangled by :func:`~duckiebot_rl.viz.run_dir.tag_filename`, the file name.
            scalar_value: The value. Anything that is not a finite float is dropped silently,
                because a NaN in the series poisons the whole EMA tail after it.
            global_step: The x coordinate; defaults to the number of samples already recorded.
        """
        try:
            value = float(scalar_value)
        except (TypeError, ValueError):
            return
        if not math.isfinite(value):
            return
        with self._lock:
            steps, values = self._series.setdefault(str(tag), ([], []))
            steps.append(float(global_step) if global_step is not None else float(len(values)))
            values.append(value)
            self._dirty = True

    def add_text(self, tag: str, text: str, global_step: int | None = None) -> None:
        """Record a text entry. The PNG layer has nowhere to put one, so this is a no-op here.

        Args:
            tag: Text tag.
            text: The payload.
            global_step: Step it belongs to.
        """

    def set_extra_render(self, callback: Callable[[], None] | None) -> None:
        """Attach a second renderer to run on the same background thread.

        This is how the composite ``_dashboard.png`` is refreshed at the same cadence as the
        house graphs without a second timer, a second thread or a second failure mode.

        Args:
            callback: Called after each render, or None to detach. Its exceptions are swallowed
                exactly like the plotting ones.
        """
        self._also_render = callback

    @property
    def tags(self) -> list[str]:
        """Every tag recorded so far, sorted.

        Returns:
            The tag list.
        """
        with self._lock:
            return sorted(self._series)

    def point_count(self, tag: str) -> int:
        """How many points a tag currently holds, history included.

        Args:
            tag: Scalar tag.

        Returns:
            The number of samples, 0 for an unknown tag.
        """
        with self._lock:
            entry = self._series.get(str(tag))
            return 0 if entry is None else len(entry[1])

    # --------------------------------------------------------------------------- thread

    def _loop(self) -> None:
        """Re-render on a timer for as long as the writer is open."""
        while not self._stop.wait(self._refresh):
            if self._dirty:
                # deliberately broad: a plotting bug must cost a figure, never a two-day run
                with contextlib.suppress(Exception):
                    self.render()

    def _snapshot(self) -> dict[str, tuple[list[float], list[float]]]:
        """Copy the series under the lock and clear the dirty flag.

        Returns:
            Tag to ``(steps, values)``, skipping empty series.
        """
        with self._lock:
            self._dirty = False
            return {
                tag: (list(steps), list(values)) for tag, (steps, values) in self._series.items() if values
            }

    # -------------------------------------------------------------------------- drawing

    def render(self) -> list[Path]:
        """Render one PNG per tag plus ``_overview.png``, and persist ``_series.json``.

        Returns:
            The paths written, empty when nothing has been recorded yet.
        """
        with self._render_lock:
            return self._render_locked()

    def _render_locked(self) -> list[Path]:
        """Do the drawing. The caller holds :attr:`_render_lock`.

        Returns:
            The paths written.
        """
        from duckiebot_rl.viz.plots import plt

        data = self._snapshot()
        if not data:
            return []
        self._persist_series(data)

        written: list[Path] = []
        for tag, (steps, values) in data.items():
            figure = plt.figure(figsize=(6.4, 3.6), dpi=110)
            try:
                axes = figure.subplots()
                axes.plot(steps, values, color=RAW_COLOR, lw=1.0, label="raw")
                if len(values) >= _MIN_POINTS_FOR_EMA:
                    axes.plot(steps, ema(values), color=SMOOTH_COLOR, lw=2.0, label="smoothed")
                axes.set_title(tag, fontsize=10)
                axes.set_xlabel("episode/step")
                axes.set_ylabel("value")
                axes.grid(True, alpha=0.3)
                axes.legend(fontsize=7, loc="best")
                axes.annotate(
                    f"last={values[-1]:.4g}",
                    xy=(0.98, 0.02),
                    xycoords="axes fraction",
                    ha="right",
                    va="bottom",
                    fontsize=8,
                    bbox={"boxstyle": "round", "fc": BADGE_FACE, "ec": "none", "alpha": 0.8},
                )
                figure.tight_layout()
                written.append(self._save(figure, self.graph_dir / f"{tag_filename(tag)}.png", dpi=110))
            finally:
                plt.close(figure)

        written.append(self._render_overview(data))
        if self._also_render is not None:
            # deliberately broad: same rule as the plotting above
            with contextlib.suppress(Exception):
                self._also_render()
        return written

    def _render_overview(self, data: Mapping[str, tuple[list[float], list[float]]]) -> Path:
        """Draw the 3-column grid of every tag.

        Args:
            data: The snapshot to draw.

        Returns:
            The path of ``_overview.png``.
        """
        from duckiebot_rl.viz.plots import plt

        tags = sorted(data)
        count = len(tags)
        cols = min(OVERVIEW_COLS, count)
        rows = (count + cols - 1) // cols
        figure, axes = plt.subplots(rows, cols, figsize=(cols * 5.0, rows * 3.0), dpi=100, squeeze=False)
        try:
            for index, tag in enumerate(tags):
                steps, values = data[tag]
                axis = axes[index // cols][index % cols]
                axis.plot(steps, values, color=RAW_COLOR, lw=0.9)
                if len(values) >= _MIN_POINTS_FOR_EMA:
                    axis.plot(steps, ema(values), color=SMOOTH_COLOR, lw=1.8)
                axis.set_title(tag, fontsize=8)
                axis.grid(True, alpha=0.3)
                axis.tick_params(labelsize=6)
            for index in range(count, rows * cols):
                axes[index // cols][index % cols].axis("off")
            figure.suptitle(OVERVIEW_TITLE, fontsize=12)
            figure.tight_layout(rect=(0, 0, 1, 0.97))
            return self._save(figure, self.graph_dir / OVERVIEW_NAME, dpi=100)
        finally:
            plt.close(figure)

    @staticmethod
    def _save(figure: Any, destination: Path, dpi: int) -> Path:
        """Serialise a figure to PNG bytes and swap them into place atomically.

        Args:
            figure: The matplotlib figure.
            destination: Final path.
            dpi: Output resolution.

        Returns:
            The path written.
        """
        import io

        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=dpi)
        return atomic_write_bytes(destination, buffer.getvalue(), fsync=False)

    # ---------------------------------------------------------------------------- close

    def close(self, render: bool = True) -> None:
        """Stop the background thread and render one last time.

        Calling it twice is harmless.

        Args:
            render: Draw a final set of PNGs before stopping. Pass False to shut down without
                drawing, which is what a caller that does not want figures (a test, or a run
                torn down early) should do; rendering every tag is not free.
        """
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        if render:
            # deliberately broad: a failed final render must not mask an otherwise clean exit
            with contextlib.suppress(Exception):
                self.render()
        self._thread.join(timeout=self._refresh + 5.0)

    def __enter__(self) -> ScalarPlotWriter:
        """Enter the context manager.

        Returns:
            ``self``.
        """
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class PngPlotWriter(ScalarPlotWriter):
    """The torch-free writer: PNG graphs only, identical API to :class:`TBPlotWriter`."""


class TBPlotWriter(ScalarPlotWriter):
    """A ``SummaryWriter`` that also exports the house standard's live PNG graphs.

    Attributes:
        graph_dir: Where the PNGs go.
    """

    def __init__(
        self,
        log_dir: str | Path | None = None,
        graph_dir: str | Path | None = None,
        refresh_sec: float = DEFAULT_REFRESH_S,
        **kwargs: Any,
    ) -> None:
        """Open a TensorBoard writer and start the PNG renderer.

        Args:
            log_dir: Where TensorBoard event files go, normally ``metrics/runs/<session>/``.
            graph_dir: Where the PNGs go; defaults to ``<log_dir>/graphs``.
            refresh_sec: Seconds between background renders.
            **kwargs: Forwarded to ``SummaryWriter``.

        Raises:
            ImportError: If torch, or matplotlib, is not installed in this interpreter.
        """
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise ImportError(
                "TBPlotWriter needs torch for the TensorBoard event files. Use PngPlotWriter, "
                "which has the same add_scalar/render/close API and writes only the PNGs."
            ) from exc
        resolved = Path(graph_dir) if graph_dir is not None else Path(log_dir or ".") / "graphs"
        super().__init__(resolved, refresh_sec=refresh_sec)
        self._tb = SummaryWriter(log_dir=None if log_dir is None else str(log_dir), **kwargs)

    def add_scalar(
        self, tag: str, scalar_value: Any, global_step: int | None = None, *args: Any, **kwargs: Any
    ) -> None:
        """Write the scalar to TensorBoard and record it for the PNGs.

        Args:
            tag: Scalar tag.
            scalar_value: The value.
            global_step: The x coordinate.
            *args: Forwarded to ``SummaryWriter.add_scalar``.
            **kwargs: Forwarded to ``SummaryWriter.add_scalar``.
        """
        with contextlib.suppress(TypeError, ValueError, RuntimeError, OSError):
            self._tb.add_scalar(tag, scalar_value, global_step, *args, **kwargs)
        super().add_scalar(tag, scalar_value, global_step)

    def add_text(self, tag: str, text: str, global_step: int | None = None) -> None:
        """Write a text entry to TensorBoard.

        Args:
            tag: Text tag.
            text: The payload.
            global_step: Step it belongs to.
        """
        with contextlib.suppress(TypeError, ValueError, RuntimeError, OSError):
            self._tb.add_text(tag, text, global_step)

    def flush(self) -> None:
        """Flush pending TensorBoard events to disk."""
        with contextlib.suppress(RuntimeError, OSError):
            self._tb.flush()

    def close(self, render: bool = True) -> None:
        """Close the TensorBoard writer after the final PNG render.

        Args:
            render: Draw a final set of PNGs before stopping. The event file is flushed and
                closed either way.
        """
        already = self._closed
        super().close(render=render)
        if not already:
            with contextlib.suppress(RuntimeError, OSError):
                self._tb.close()


def make_plot_writer(
    graph_dir: str | Path,
    tb_log_dir: str | Path | None = None,
    refresh_sec: float = DEFAULT_REFRESH_S,
) -> ScalarPlotWriter:
    """Return the best writer this interpreter can support.

    A torch interpreter gets TensorBoard events as well as the PNGs; a torch-free one gets the
    PNGs. Both have the same API, so no caller has to know which it got.

    Args:
        graph_dir: Where the PNGs go, normally ``metrics/graphs``.
        tb_log_dir: Where TensorBoard events go, normally ``metrics/runs/<session>``. Pass None
            to force the PNG-only writer.
        refresh_sec: Seconds between background renders.

    Returns:
        A :class:`TBPlotWriter` or a :class:`PngPlotWriter`.

    Raises:
        ImportError: If matplotlib is missing, since neither writer can draw without it.
    """
    if tb_log_dir is not None:
        try:
            return TBPlotWriter(log_dir=tb_log_dir, graph_dir=graph_dir, refresh_sec=refresh_sec)
        except ImportError:
            pass
    return PngPlotWriter(graph_dir, refresh_sec=refresh_sec)
