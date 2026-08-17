"""Render the live training dashboard for a run directory.

Typical use, in a second terminal, while training runs::

    d:/Personal/personal/wheeled_quadruped_robot/.venv/Scripts/python.exe scripts/dashboard.py ^
        --run runs/20260817T104500Z_lanefollow_seed0 --watch

Then keep ``runs/<run_id>/figures/latest.png`` open in any image viewer that reloads on change.
The figure is written atomically, so the viewer never catches a half-written PNG.

Nothing about this process touches the GPU: it reads ``metrics.jsonl`` and ``status.json`` and
draws with the matplotlib Agg backend. It costs no VRAM and can run from any interpreter that has
matplotlib, including the MuJoCo venv.

``--watch`` is safe to start before training does (it polls for the run directory) and safe to
leave running after training ends (it renders once more, then exits).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from duckiebot_rl.viz import dashboard as _dashboard  # noqa: E402
from duckiebot_rl.viz.plots import matplotlib_available, require_matplotlib  # noqa: E402
from duckiebot_rl.viz.run_dir import RunDir, find_latest_run  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="dashboard.py",
        description="Render runs/<run_id>/figures/latest.png from a run's metrics and heartbeat.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--run",
        type=Path,
        default=None,
        help="run directory to read. Defaults to the newest directory under --runs-root.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=_REPO_ROOT / "runs",
        help="where to look for the newest run when --run is omitted (default: ./runs)",
    )
    parser.add_argument("--watch", action="store_true", help="re-render whenever metrics.jsonl grows")
    parser.add_argument("--once", action="store_true", help="render exactly once and exit (the default)")
    parser.add_argument("--interval", type=float, default=20.0, help="seconds between polls in --watch")
    parser.add_argument("--dpi", type=int, default=100, help="output resolution; 100 gives a 1600 px width")
    parser.add_argument(
        "--panels",
        action="store_true",
        help="also write figures/<panel>.png for every panel, not just the composite",
    )
    parser.add_argument(
        "--no-summary", action="store_true", help="do not print the text summary after rendering"
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="in --watch, do not exit when the run reports finished or crashed",
    )
    return parser


def resolve_run(args: argparse.Namespace) -> RunDir | None:
    """Work out which run directory to read.

    Args:
        args: Parsed arguments.

    Returns:
        The run directory, or None when ``--run`` was omitted and no run exists yet.
    """
    if args.run is not None:
        return RunDir.open(args.run)
    latest = find_latest_run(args.runs_root)
    return None if latest is None else RunDir.open(latest)


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code. 0 on success, 1 for a missing run or a missing matplotlib, 130 on
        Ctrl+C in watch mode.
    """
    args = build_parser().parse_args(argv)
    if not matplotlib_available():
        try:
            require_matplotlib()
        except ImportError as exc:
            print(str(exc), file=sys.stderr)
        return 1

    run = resolve_run(args)
    if run is None:
        print(
            f"no run directory found under {args.runs_root}. Pass --run <dir>, or start training "
            "first; --watch also accepts a directory that does not exist yet.",
            file=sys.stderr,
        )
        return 1

    if args.watch:
        print(f"[dashboard] watching {run.root} every {args.interval:g}s; Ctrl+C to stop", flush=True)
        if not run.exists():
            print("[dashboard] run directory does not exist yet, polling for it", flush=True)

        def report(path: Path, count: int) -> None:
            size = path.stat().st_size if path.exists() else 0
            print(f"[dashboard] render {count}: {path} ({size:,} bytes)", flush=True)

        try:
            _dashboard.watch(
                run,
                interval=args.interval,
                panels=args.panels,
                dpi=args.dpi,
                on_render=report,
                exit_when_done=not args.keep_going,
            )
        except KeyboardInterrupt:
            print("[dashboard] stopped", flush=True)
            return 130
        if not args.no_summary:
            print(_dashboard.summarise(run))
        return 0

    if not run.exists():
        print(f"run directory does not exist: {run.root}", file=sys.stderr)
        return 1
    figure = _dashboard.render_dashboard(run, panels=args.panels, dpi=args.dpi)
    size = figure.stat().st_size if figure.exists() else 0
    print(f"wrote {figure} ({size:,} bytes)")
    if not args.no_summary:
        print(_dashboard.summarise(run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
