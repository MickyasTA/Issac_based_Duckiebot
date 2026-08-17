"""The matplotlib layer: one palette, one style, one panel vocabulary.

Everything visual lives here so that :mod:`duckiebot_rl.viz.dashboard` is only a composer. The
rules this module follows, in order of how often they get broken:

* **Agg, explicitly.** Set at import time. The dashboard runs headless on Windows, and letting
  matplotlib pick TkAgg there means a stray window and, in a service, a crash.
* **One palette, defined once** (:data:`PALETTE`, :data:`SERIES`). The categorical order is the
  validated colorblind-safe order: blue, orange, aqua, yellow, magenta, green, violet, red.
  Panels with a single series all use slot 1, so the figure stays calm and a colour never means
  two different things in two panels.
* **Raw plus smoothing, never smoothing alone.** The raw series is drawn faintly underneath an
  EMA. A dashboard that only shows the smoothed curve hides exactly the variance spike you opened
  it to find.
* **Text wears ink, marks wear colour.** Endpoint values are printed in secondary ink next to a
  dot in the series colour, so identity is never carried by colour alone. Legends appear whenever
  a panel has two or more series.
* **Nothing raises.** Zero rows, one row, an all-NaN column, a key that only appears halfway
  through a run, a crashed run: each renders a readable panel that says so.

Units are on every axis label. The x axis is the PPO iteration by default; total environment
steps and wall clock are in the header, where they belong.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

_MPL_HINT = (
    "matplotlib is required for duckiebot_rl.viz plotting but is not installed in this "
    "interpreter. Install it with:\n"
    '    python -m pip install "duckiebot-rl[viz]"\n'
    "or directly:\n"
    "    python -m pip install matplotlib>=3.8"
)

try:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator, PercentFormatter

    _MPL_IMPORT_ERROR: ImportError | None = None
except ImportError as _exc:  # pragma: no cover - exercised only on a matplotlib-free interpreter
    matplotlib = None  # type: ignore[assignment]
    plt = None  # type: ignore[assignment]
    PercentFormatter = None  # type: ignore[assignment]
    NullLocator = None  # type: ignore[assignment]
    _MPL_IMPORT_ERROR = _exc

__all__ = [
    "PALETTE",
    "PANELS",
    "SERIES",
    "MetricTable",
    "Palette",
    "Panel",
    "build_panels",
    "ema",
    "fmt_duration",
    "fmt_int",
    "fmt_metric",
    "matplotlib_available",
    "require_matplotlib",
    "style_context",
]


# ------------------------------------------------------------------------------------- palette


@dataclass(frozen=True)
class Palette:
    """The one place a colour is chosen.

    Attributes:
        surface: Chart surface.
        plane: Page plane behind the panels.
        ink: Primary text.
        ink_secondary: Secondary text, used for values and axis titles.
        ink_muted: Axis tick labels and reference lines.
        grid: Hairline gridlines.
        axis: Baselines and spines.
        good: Status colour for a healthy, finished run.
        warning: Status colour for a stale heartbeat.
        serious: Status colour reserved for degraded states.
        critical: Status colour for a crashed run.
        band_alpha: Opacity of a +/-1 std band.
        raw_alpha: Opacity of the unsmoothed series drawn under its EMA.
    """

    surface: str = "#fcfcfb"
    plane: str = "#f9f9f7"
    ink: str = "#0b0b0b"
    ink_secondary: str = "#52514e"
    ink_muted: str = "#898781"
    grid: str = "#e1e0d9"
    axis: str = "#c3c2b7"
    good: str = "#0ca30c"
    warning: str = "#fab219"
    serious: str = "#ec835a"
    critical: str = "#d03b3b"
    band_alpha: float = 0.16
    raw_alpha: float = 0.30


PALETTE = Palette()

SERIES: tuple[str, ...] = (
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
)
"""Categorical slots, assigned in this fixed order and never cycled past eight."""

STYLE_RC: dict[str, Any] = {
    "figure.facecolor": PALETTE.plane,
    "figure.edgecolor": PALETTE.plane,
    "savefig.facecolor": PALETTE.plane,
    "axes.facecolor": PALETTE.surface,
    "axes.edgecolor": PALETTE.axis,
    "axes.labelcolor": PALETTE.ink_secondary,
    "axes.titlecolor": PALETTE.ink,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 7.0,
    "axes.labelsize": 8.5,
    "grid.color": PALETTE.grid,
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",
    "xtick.color": PALETTE.ink_muted,
    "ytick.color": PALETTE.ink_muted,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "legend.fontsize": 8.0,
    "legend.labelcolor": PALETTE.ink_secondary,
    "legend.handlelength": 1.6,
    "legend.borderaxespad": 0.2,
    "lines.linewidth": 2.0,
    "lines.solid_capstyle": "round",
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
    "font.size": 9.0,
    "text.color": PALETTE.ink,
}


def matplotlib_available() -> bool:
    """Whether matplotlib could be imported.

    Returns:
        True when plotting is possible in this interpreter.
    """
    return _MPL_IMPORT_ERROR is None


def require_matplotlib() -> None:
    """Raise a actionable error when matplotlib is missing.

    Raises:
        ImportError: With the exact pip command to fix it.
    """
    if _MPL_IMPORT_ERROR is not None:
        raise ImportError(_MPL_HINT) from _MPL_IMPORT_ERROR


def style_context() -> Any:
    """Return a matplotlib ``rc_context`` carrying the dashboard style.

    Returns:
        A context manager. Using a context rather than mutating global rcParams keeps the style
        from leaking into a caller's own figures.
    """
    require_matplotlib()
    return plt.rc_context(STYLE_RC)


# -------------------------------------------------------------------------------- formatting


def fmt_int(value: float | int | None) -> str:
    """Format an integer with thousands separators.

    Args:
        value: The number, or None.

    Returns:
        For example ``"1,234,567"``, or ``"-"`` when the value is missing or non-finite.
    """
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "-"
    return f"{int(value):,}"


def fmt_metric(value: float | int | None, digits: int = 4) -> str:
    """Format a metric value compactly without losing its magnitude.

    Args:
        value: The number, or None.
        digits: Significant digits.

    Returns:
        A short decimal or scientific string, or ``"-"``.
    """
    if value is None:
        return "-"
    number = float(value)
    if not math.isfinite(number):
        return "nan"
    if number != 0 and (abs(number) >= 1e5 or abs(number) < 1e-3):
        return f"{number:.{max(1, digits - 1)}e}"
    return f"{number:.{digits}g}"


def fmt_duration(seconds: float | None) -> str:
    """Format a wall-clock duration as ``H:MM:SS`` with days when needed.

    Args:
        seconds: Duration in seconds, or None.

    Returns:
        For example ``"2d 03:14:15"``, or ``"-"``.
    """
    if seconds is None or not math.isfinite(float(seconds)) or seconds < 0:
        return "-"
    total = int(seconds)
    days, rest = divmod(total, 86400)
    hours, rest = divmod(rest, 3600)
    minutes, secs = divmod(rest, 60)
    if days:
        return f"{days}d {hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours:d}:{minutes:02d}:{secs:02d}"


def prettify(key: str) -> str:
    """Turn a metric key into a human title.

    Args:
        key: A ``snake_case`` metric key.

    Returns:
        For example ``"grad norm"``.
    """
    return key.replace("_", " ").strip()


# ------------------------------------------------------------------------------- metric table


def ema(values: np.ndarray, alpha: float = 0.15) -> np.ndarray:
    """Exponential moving average that steps over NaNs instead of poisoning the tail.

    Args:
        values: 1-D array, possibly containing NaNs.
        alpha: Smoothing factor in (0, 1]; larger follows the raw series more closely.

    Returns:
        An array of the same shape. Positions whose input was NaN stay NaN, so a gap in the raw
        series stays a visible gap in the smoothed one.
    """
    out = np.full(values.shape, np.nan, dtype=float)
    state: float | None = None
    for index, raw in enumerate(values):
        if not np.isfinite(raw):
            continue
        state = float(raw) if state is None else (1.0 - alpha) * state + alpha * float(raw)
        out[index] = state
    return out


@dataclass
class MetricTable:
    """Columns of ``metrics.jsonl``, coerced to float arrays with NaN for anything missing.

    A key that only appears halfway through a run therefore reads as NaN before it appears, which
    plots as a gap rather than as a spike to zero.

    Attributes:
        rows: The raw decoded rows.
        columns: Numeric columns by key.
        x_key: Which column is used as the x axis.
    """

    rows: list[dict[str, Any]] = field(default_factory=list)
    columns: dict[str, np.ndarray] = field(default_factory=dict)
    x_key: str = "iteration"

    @classmethod
    def from_rows(cls, rows: Iterable[Mapping[str, Any]], x_key: str = "iteration") -> MetricTable:
        """Build a table from decoded ``metrics.jsonl`` rows.

        Args:
            rows: The rows, in file order.
            x_key: Column to use for the x axis.

        Returns:
            A :class:`MetricTable`; an empty one when ``rows`` is empty.
        """
        materialised = [dict(row) for row in rows]
        keys: list[str] = []
        seen: set[str] = set()
        for row in materialised:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        count = len(materialised)
        columns: dict[str, np.ndarray] = {}
        for key in keys:
            column = np.full(count, np.nan, dtype=float)
            numeric = False
            for index, row in enumerate(materialised):
                value = row.get(key)
                if isinstance(value, bool | int | float):
                    column[index] = float(value)
                    numeric = True
            if numeric:
                columns[key] = column
        return cls(rows=materialised, columns=columns, x_key=x_key)

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def x(self) -> np.ndarray:
        """The x axis values.

        Returns:
            The ``x_key`` column when it is present and finite, otherwise ``1..n``, so a caller
            that forgot to log ``iteration`` still gets a readable plot.
        """
        column = self.columns.get(self.x_key)
        if column is None or not np.any(np.isfinite(column)):
            return np.arange(1, len(self.rows) + 1, dtype=float)
        filled = column.copy()
        gaps = ~np.isfinite(filled)
        if np.any(gaps):
            filled[gaps] = np.arange(1, len(filled) + 1, dtype=float)[gaps]
        return filled

    def column(self, key: str) -> np.ndarray:
        """Return one column, NaN-filled when the key was never logged.

        Args:
            key: Metric key.

        Returns:
            A float array as long as the table.
        """
        column = self.columns.get(key)
        if column is None:
            return np.full(len(self.rows), np.nan, dtype=float)
        return column

    def has(self, key: str) -> bool:
        """Whether a column exists and holds at least one finite value.

        Args:
            key: Metric key.

        Returns:
            True when the key is plottable.
        """
        column = self.columns.get(key)
        return column is not None and bool(np.any(np.isfinite(column)))

    def last(self, key: str) -> float | None:
        """The most recent finite value of a column.

        Args:
            key: Metric key.

        Returns:
            The value, or None if the column has none.
        """
        column = self.column(key)
        finite = np.flatnonzero(np.isfinite(column))
        return float(column[finite[-1]]) if finite.size else None

    def extra_keys(self, consumed: Iterable[str]) -> list[str]:
        """Numeric keys that no fixed panel plots, in first-seen order.

        Args:
            consumed: Keys already claimed by the fixed panels or used as bookkeeping.

        Returns:
            The leftovers, so the dashboard can append a panel per key instead of dropping data
            the training loop went to the trouble of logging.
        """
        claimed = set(consumed)
        return [key for key in self.columns if key not in claimed]


# ------------------------------------------------------------------------------ drawing atoms


def _empty(ax: Axes, message: str) -> None:
    """Fill an axes with an explanatory message instead of an empty grid.

    Args:
        ax: Target axes.
        message: What to say, for example ``"not logged yet"``.
    """
    ax.text(
        0.5,
        0.5,
        message,
        transform=ax.transAxes,
        ha="center",
        va="center",
        color=PALETTE.ink_muted,
        fontsize=9,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def _endpoint(ax: Axes, x: np.ndarray, y: np.ndarray, color: str) -> None:
    """Mark and label the last finite point of a series.

    The dot carries the series identity in colour; the value is printed in secondary ink, so the
    text never depends on colour to be understood.

    Args:
        ax: Target axes.
        x: X values.
        y: Y values.
        color: The series colour.
    """
    finite = np.flatnonzero(np.isfinite(y))
    if finite.size == 0:
        return
    last = int(finite[-1])
    ax.plot(
        [x[last]],
        [y[last]],
        marker="o",
        markersize=4.5,
        color=color,
        markeredgecolor=PALETTE.surface,
        markeredgewidth=1.6,
        linestyle="none",
        zorder=6,
    )
    ax.annotate(
        fmt_metric(float(y[last]), digits=3),
        xy=(x[last], y[last]),
        xytext=(5, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=7.5,
        color=PALETTE.ink_secondary,
        zorder=7,
        annotation_clip=False,
    )


def draw_line(
    ax: Axes,
    table: MetricTable,
    key: str,
    color: str,
    label: str | None = None,
    smooth: float = 0.15,
    endpoint: bool = True,
) -> bool:
    """Draw one metric as a faint raw line under an EMA, plus its endpoint label.

    Args:
        ax: Target axes.
        table: Source table.
        key: Metric key to draw.
        color: Series colour.
        label: Legend label; defaults to a prettified ``key``.
        smooth: EMA alpha. Series shorter than five points are drawn raw only, because smoothing
            three points says nothing.
        endpoint: Mark and label the final value.

    Returns:
        True if anything was drawn.
    """
    y = table.column(key)
    if not np.any(np.isfinite(y)):
        return False
    x = table.x
    name = label if label is not None else prettify(key)
    finite_count = int(np.count_nonzero(np.isfinite(y)))

    if finite_count == 1:
        index = int(np.flatnonzero(np.isfinite(y))[0])
        ax.plot(
            [x[index]],
            [y[index]],
            marker="o",
            markersize=6.0,
            color=color,
            markeredgecolor=PALETTE.surface,
            markeredgewidth=1.6,
            linestyle="none",
            label=name,
        )
    elif finite_count < 5:
        ax.plot(x, y, color=color, linewidth=2.0, label=name)
    else:
        ax.plot(x, y, color=color, linewidth=1.0, alpha=PALETTE.raw_alpha, zorder=2)
        ax.plot(x, ema(y, smooth), color=color, linewidth=2.0, label=name, zorder=4)
    if endpoint:
        _endpoint(ax, x, y, color)
    return True


def draw_band(ax: Axes, table: MetricTable, mean_key: str, std_key: str, color: str) -> None:
    """Shade the +/-1 standard-deviation band around a mean series.

    Args:
        ax: Target axes.
        table: Source table.
        mean_key: Key of the mean.
        std_key: Key of the standard deviation.
        color: Band colour, matching its mean line.
    """
    mean = table.column(mean_key)
    std = np.abs(table.column(std_key))
    valid = np.isfinite(mean) & np.isfinite(std)
    if not np.any(valid):
        return
    ax.fill_between(
        table.x,
        np.where(valid, mean - std, np.nan),
        np.where(valid, mean + std, np.nan),
        color=color,
        alpha=PALETTE.band_alpha,
        linewidth=0,
        zorder=1,
        label="+/-1 std",
    )


def _reference(ax: Axes, value: float, label: str, color: str | None = None) -> None:
    """Draw a horizontal threshold line with a small label.

    A dashed rule here is deliberate and is not the "dashed gridline" anti-pattern: this line is a
    threshold, and dashing is what says so.

    Args:
        ax: Target axes.
        value: Y position.
        label: Text drawn at the left end.
        color: Line colour; defaults to muted ink.
    """
    ax.axhline(value, color=color or PALETTE.ink_muted, linewidth=1.0, linestyle=(0, (4, 3)), zorder=3)
    ax.annotate(
        label,
        xy=(0.985, value),
        xycoords=("axes fraction", "data"),
        xytext=(0, 3),
        textcoords="offset points",
        fontsize=7.0,
        color=PALETTE.ink_muted,
        va="bottom",
        ha="right",
    )


def _finish(
    ax: Axes,
    title: str,
    ylabel: str,
    legend: bool = False,
    xlabel: str = "iteration",
    ylabel_color: str | None = None,
) -> None:
    """Apply the shared axes chrome: title, labels, grid, headroom for endpoint labels.

    Args:
        ax: Target axes.
        title: Panel title.
        ylabel: Y axis label, including units.
        legend: Show a legend. Pass True whenever the panel has two or more series.
        xlabel: X axis label.
        ylabel_color: Colour of the y label, for the one panel whose axes are colour-keyed.
    """
    ax.set_title(title)
    ax.set_ylabel(ylabel, color=ylabel_color or PALETTE.ink_secondary)
    ax.set_xlabel(xlabel)
    ax.grid(True, axis="both")
    ax.margins(x=0.02)
    left, right = ax.get_xlim()
    if math.isfinite(left) and math.isfinite(right) and right > left:
        ax.set_xlim(left, right + 0.13 * (right - left))
    if legend:
        ax.legend(loc="best", ncols=2)


def _thin_yticks(ax: Axes, keep: int = 7) -> None:
    """Drop every other y tick until at most ``keep`` remain.

    A symlog axis emits a decade tick per decade, which on a loss panel two inches tall collides
    into an unreadable stack. Thinning after the limits are final is the one reliable fix.

    Args:
        ax: Target axes.
        keep: Maximum number of tick labels to leave.
    """
    ticks = [tick for tick in ax.get_yticks() if ax.get_ylim()[0] <= tick <= ax.get_ylim()[1]]
    while len(ticks) > keep:
        ticks = ticks[::2]
    if ticks:
        ax.set_yticks(ticks)


def _split_stacked(ax: Axes) -> tuple[Axes, Axes]:
    """Replace one grid cell with two stacked axes sharing an x axis.

    This is the honest alternative to a second y scale for two unrelated quantities: two small
    charts, one measure each, instead of one chart implying a correlation that is not in the data.

    Args:
        ax: The axes occupying the cell. It is removed.

    Returns:
        ``(top, bottom)``.
    """
    figure = ax.get_figure()
    spec = ax.get_subplotspec()
    ax.remove()
    sub = spec.subgridspec(2, 1, hspace=0.42)
    top = figure.add_subplot(sub[0])
    bottom = figure.add_subplot(sub[1])
    return top, bottom


# -------------------------------------------------------------------------------- the panels


def panel_ep_return(ax: Axes, table: MetricTable) -> None:
    """Episode return, mean line over a +/-1 std band.

    Args:
        ax: Target axes.
        table: Source table.
    """
    if not table.has("ep_return_mean"):
        _empty(ax, "ep_return_mean not logged yet")
        ax.set_title("episode return")
        return
    draw_band(ax, table, "ep_return_mean", "ep_return_std", SERIES[0])
    draw_line(ax, table, "ep_return_mean", SERIES[0], label="mean")
    _finish(ax, "episode return", "return (reward units)", legend=table.has("ep_return_std"))


def panel_ep_len(ax: Axes, table: MetricTable) -> None:
    """Mean episode length in environment steps.

    Args:
        ax: Target axes.
        table: Source table.
    """
    if not draw_line(ax, table, "ep_len_mean", SERIES[0], label="mean"):
        _empty(ax, "ep_len_mean not logged yet")
        ax.set_title("episode length")
        return
    _finish(ax, "episode length", "steps per episode")


def panel_lane_dev(ax: Axes, table: MetricTable) -> None:
    """Lane deviation, RMS and worst case.

    Args:
        ax: Target axes.
        table: Source table.
    """
    drew_rms = draw_line(ax, table, "lane_dev_rms_m", SERIES[0], label="RMS")
    drew_max = draw_line(ax, table, "lane_dev_max_m", SERIES[1], label="max")
    if not (drew_rms or drew_max):
        _empty(ax, "lane deviation not logged yet")
        ax.set_title("lane deviation")
        return
    ax.set_ylim(bottom=0.0)
    _finish(ax, "lane deviation", "distance from lane centre (m)", legend=drew_rms and drew_max)


def panel_success(ax: Axes, table: MetricTable) -> None:
    """Success rate, drawn on a fixed 0 to 100 percent scale.

    Args:
        ax: Target axes.
        table: Source table.
    """
    if not table.has("success_rate"):
        _empty(ax, "success_rate not logged yet")
        ax.set_title("success rate")
        return
    column = table.column("success_rate")
    fraction = float(np.nanmax(np.abs(column))) <= 1.5
    draw_line(ax, table, "success_rate", SERIES[0], label="success")
    if fraction:
        ax.set_ylim(-0.02, 1.02)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    _finish(ax, "success rate", "episodes reaching the goal (%)")


def panel_ppo_health(ax: Axes, table: MetricTable) -> None:
    """Approximate KL and clip fraction, with the target-KL threshold drawn.

    These two share a panel on twin axes because they are read together: a KL that walks past its
    target while the clip fraction climbs is the signature of too large a step size. The left
    spine, its ticks and its endpoint dot are the KL colour; the right spine is the clip
    fraction's, so which scale belongs to which series is never ambiguous.

    Args:
        ax: Target axes.
        table: Source table.
    """
    has_kl = table.has("approx_kl")
    has_clip = table.has("clipfrac")
    if not (has_kl or has_clip):
        _empty(ax, "approx_kl / clipfrac not logged yet")
        ax.set_title("PPO health")
        return

    kl_label_color: str | None = None
    if has_kl:
        draw_line(ax, table, "approx_kl", SERIES[0], label="approx KL")
        target = table.last("target_kl")
        _reference(ax, float(target if target is not None else 0.015), "target KL")
        ax.set_ylim(bottom=0.0)
        ax.tick_params(axis="y", colors=SERIES[0])
        ax.spines["left"].set_color(SERIES[0])
        kl_label_color = SERIES[0]

    if has_clip:
        twin = ax.twinx()
        twin.grid(False)
        twin.set_facecolor("none")
        draw_line(twin, table, "clipfrac", SERIES[1], label="clip fraction")
        twin.set_ylim(bottom=0.0)
        twin.set_ylabel("clipped samples (fraction)", color=SERIES[1])
        twin.tick_params(axis="y", colors=SERIES[1])
        twin.spines["right"].set_visible(True)
        twin.spines["right"].set_color(SERIES[1])
        twin.spines["top"].set_visible(False)
        left, right = twin.get_xlim()
        if math.isfinite(left) and math.isfinite(right) and right > left:
            twin.set_xlim(left, right + 0.13 * (right - left))

    _finish(ax, "PPO health", "approximate KL (nats)", ylabel_color=kl_label_color)
    if has_kl and has_clip:
        ax.legend(
            handles=[
                plt.Line2D([], [], color=SERIES[0], linewidth=2.0, label="approx KL (left)"),
                plt.Line2D([], [], color=SERIES[1], linewidth=2.0, label="clip fraction (right)"),
            ],
            loc="upper left",
            ncols=1,
        )


def panel_explained_variance(ax: Axes, table: MetricTable) -> None:
    """Explained variance of the critic, with the zero reference drawn.

    Args:
        ax: Target axes.
        table: Source table.
    """
    if not draw_line(ax, table, "explained_variance", SERIES[0], label="explained variance"):
        _empty(ax, "explained_variance not logged yet")
        ax.set_title("critic explained variance")
        return
    _reference(ax, 0.0, "0 = no better than predicting the mean")
    column = table.column("explained_variance")
    low = float(np.nanmin(column))
    ax.set_ylim(min(-0.15, max(-1.6, low - 0.1)), 1.05)
    _finish(ax, "critic explained variance", "1 - Var(residual) / Var(returns)")


def panel_losses(ax: Axes, table: MetricTable) -> None:
    """Policy and value loss on one symmetric-log axis.

    Both are losses, so they share a scale rather than a second y axis; symlog keeps a value loss
    two orders of magnitude larger than the policy loss from flattening the policy curve, and
    handles the negative policy-surrogate values a linear log axis could not.

    Args:
        ax: Target axes.
        table: Source table.
    """
    drew_policy = draw_line(ax, table, "policy_loss", SERIES[0], label="policy")
    drew_value = draw_line(ax, table, "value_loss", SERIES[1], label="value")
    if not (drew_policy or drew_value):
        _empty(ax, "losses not logged yet")
        ax.set_title("losses")
        return
    values = np.concatenate([table.column("policy_loss"), table.column("value_loss")])
    finite = np.abs(values[np.isfinite(values)])
    finite = finite[finite > 0]
    label = "loss"
    if finite.size and float(finite.max() / finite.min()) > 50.0:
        ax.set_yscale("symlog", linthresh=max(float(np.percentile(finite, 5)), 1e-8))
        ax.yaxis.set_minor_locator(NullLocator())
        label = "loss (symlog)"
    _finish(ax, "losses", label, legend=drew_policy and drew_value)
    _thin_yticks(ax, keep=7)


def panel_entropy_lr(ax: Axes, table: MetricTable) -> None:
    """Policy entropy and learning rate, as two stacked charts sharing one cell.

    Args:
        ax: Target axes; it is split into two.
        table: Source table.
    """
    top, bottom = _split_stacked(ax)

    if draw_line(top, table, "entropy", SERIES[0], label="entropy"):
        _finish(top, "policy entropy", "nats", xlabel="")
        top.tick_params(axis="x", labelbottom=False)
        top.set_xlabel("")
    else:
        _empty(top, "entropy not logged yet")
        top.set_title("policy entropy")

    if draw_line(bottom, table, "learning_rate", SERIES[0], label="learning rate"):
        _finish(bottom, "learning rate", "step size")
        bottom.ticklabel_format(style="sci", scilimits=(-3, 3), axis="y", useMathText=True)
        bottom.yaxis.get_offset_text().set_fontsize(7.0)
        bottom.yaxis.get_offset_text().set_color(PALETTE.ink_muted)
    else:
        _empty(bottom, "learning_rate not logged yet")
        bottom.set_title("learning rate")


def panel_curriculum(ax: Axes, table: MetricTable) -> None:
    """The two domain-randomisation curriculum scalars.

    Args:
        ax: Target axes.
        table: Source table.
    """
    drew_vis = draw_line(ax, table, "alpha_vis", SERIES[0], label="alpha_vis")
    drew_dyn = draw_line(ax, table, "alpha_dyn", SERIES[1], label="alpha_dyn")
    if not (drew_vis or drew_dyn):
        _empty(ax, "curriculum alphas not logged yet")
        ax.set_title("DR curriculum")
        return
    ax.set_ylim(-0.02, 1.02)
    _finish(ax, "DR curriculum", "randomisation scale (0 = nominal, 1 = full)", legend=drew_vis and drew_dyn)


def panel_generic(ax: Axes, table: MetricTable, key: str) -> None:
    """Fallback panel for a metric key discovered at read time.

    Args:
        ax: Target axes.
        table: Source table.
        key: The discovered key.
    """
    if not draw_line(ax, table, key, SERIES[0], label=prettify(key)):
        _empty(ax, f"{key} not logged yet")
        ax.set_title(prettify(key))
        return
    _finish(ax, prettify(key), prettify(key))


@dataclass(frozen=True)
class Panel:
    """One dashboard panel.

    Attributes:
        slug: File-name stem for the standalone PNG.
        title: Human title, used in error messages.
        draw: Callable that renders the panel onto an axes.
    """

    slug: str
    title: str
    draw: Callable[[Axes, MetricTable], None]


PANELS: tuple[Panel, ...] = (
    Panel("ep_return", "episode return", panel_ep_return),
    Panel("ep_len", "episode length", panel_ep_len),
    Panel("lane_deviation", "lane deviation", panel_lane_dev),
    Panel("success_rate", "success rate", panel_success),
    Panel("ppo_health", "PPO health", panel_ppo_health),
    Panel("explained_variance", "critic explained variance", panel_explained_variance),
    Panel("losses", "losses", panel_losses),
    Panel("entropy_lr", "entropy and learning rate", panel_entropy_lr),
    Panel("dr_curriculum", "DR curriculum", panel_curriculum),
)
"""The fixed panels, in reading order."""

CONSUMED_KEYS: frozenset[str] = frozenset(
    {
        "iteration",
        "total_timesteps",
        "wall_clock_s",
        "ep_return_mean",
        "ep_return_std",
        "ep_len_mean",
        "lane_dev_rms_m",
        "lane_dev_max_m",
        "success_rate",
        "approx_kl",
        "clipfrac",
        "target_kl",
        "explained_variance",
        "policy_loss",
        "value_loss",
        "entropy",
        "learning_rate",
        "alpha_vis",
        "alpha_dyn",
    }
)
"""Keys the fixed panels already show. Everything else earns an extra panel."""


def build_panels(table: MetricTable, max_extra: int = 12) -> list[Panel]:
    """Return the fixed panels plus one discovered panel per unclaimed metric key.

    Args:
        table: The metrics read from disk.
        max_extra: Cap on discovered panels, so a run that logs a hundred diagnostics still
            produces a figure a human can open.

    Returns:
        The panel list in render order.
    """
    panels = list(PANELS)
    for key in table.extra_keys(CONSUMED_KEYS)[:max_extra]:
        slug = "".join(char if char.isalnum() or char in "._-" else "-" for char in key)
        panels.append(
            Panel(
                slug=f"extra_{slug}",
                title=prettify(key),
                draw=lambda ax, tbl, _key=key: panel_generic(ax, tbl, _key),
            )
        )
    return panels


def render_panel_standalone(
    panel: Panel,
    table: MetricTable,
    path: str,
    size: Sequence[float] = (7.4, 3.8),
    dpi: int = 130,
) -> str:
    """Render one panel into its own PNG.

    Args:
        panel: The panel to draw.
        table: Source table.
        path: Destination PNG path.
        size: Figure size in inches.
        dpi: Output resolution.

    Returns:
        The path written.
    """
    require_matplotlib()
    with style_context():
        figure: Figure = plt.figure(figsize=tuple(size), dpi=dpi)
        axes = figure.add_subplot(1, 1, 1)
        try:
            panel.draw(axes, table)
        except Exception as exc:  # deliberately broad: one broken panel must not lose the figure
            figure.clf()
            axes = figure.add_subplot(1, 1, 1)
            _empty(axes, f"panel failed: {exc!r}"[:120])
            axes.set_title(panel.title)
        figure.tight_layout(pad=1.1)
        figure.savefig(path, dpi=dpi, facecolor=figure.get_facecolor())
        plt.close(figure)
    return path
