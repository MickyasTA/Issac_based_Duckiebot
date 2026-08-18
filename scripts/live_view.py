"""Watch the newest policy of a running training job drive, live.

Training runs headless. This process attaches to its run directory, loads the newest checkpoint
into a network it builds exactly once, drives an episode, and hot-reloads the moment the trainer
publishes a newer checkpoint. Nothing restarts: not the simulator, not the network, not the window.

    # follow the newest run under runs/, forever, on the CPU MuJoCo backend
    d:/Personal/personal/mujoco_venv/Scripts/python.exe scripts/live_view.py --loop

    # follow one specific run, record video and observation snapshots into it
    d:/Personal/personal/mujoco_venv/Scripts/python.exe scripts/live_view.py ^
        --run runs/20260817T104500Z_lanefollow_seed0 --loop --record --map loop_big

The default backend is MuJoCo because it costs zero VRAM. Headless Isaac training already uses
6.4 to 7.6 GiB of the 8 GiB on this machine, and a second Isaac Kit process costs another 2.5 to
3.5 GiB of baseline, so ``--backend isaac`` refuses to start unless ``--allow-isaac-vram`` says
the caller understands that arithmetic.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:  # pragma: no cover - convenience for a fresh clone
    sys.path.insert(0, str(_REPO_ROOT))

from duckiebot_rl.viz.backends import IsaacVramRefusal, isaac_vram_message, make_backend  # noqa: E402
from duckiebot_rl.viz.policy_host import ArchitectureMismatch, PolicyHost  # noqa: E402
from duckiebot_rl.viz.render import annotate_frame, write_rollout  # noqa: E402
from duckiebot_rl.viz.run_dir import RUNS_ROOT_DIRNAME, find_latest_run  # noqa: E402
from duckiebot_rl.viz.watcher import CheckpointInfo, CheckpointWatcher  # noqa: E402

DEFAULT_RUNS_ROOT = _REPO_ROOT / RUNS_ROOT_DIRNAME
_STOP = False


def _install_sigint_handler() -> None:
    """Make Ctrl-C set a flag so the current episode finishes tidily instead of dying mid-step."""

    def handler(_signum: int, _frame: Any) -> None:
        global _STOP
        _STOP = True
        print("\n[live_view] interrupt received; finishing the current step and shutting down")

    signal.signal(signal.SIGINT, handler)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="live_view",
        description="Watch the newest checkpoint of a training run drive, hot-reloading as it trains.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--run",
        type=Path,
        default=None,
        help="run directory to watch; defaults to the newest under --runs-root",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=DEFAULT_RUNS_ROOT,
        help=f"where run directories live (default: {DEFAULT_RUNS_ROOT.as_posix()})",
    )
    parser.add_argument(
        "--backend",
        choices=("mujoco", "isaac"),
        default="mujoco",
        help="simulator backend (default: mujoco, CPU, zero VRAM)",
    )
    parser.add_argument(
        "--which",
        choices=("latest", "best"),
        default="latest",
        help="which checkpoint slot to follow (default: latest)",
    )
    parser.add_argument("--episodes", type=int, default=1, help="episodes to run (default: 1)")
    parser.add_argument(
        "--loop",
        action="store_true",
        help="run episodes forever, hot-reloading whenever a new checkpoint appears",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="write video/latest_rollout.{mp4,gif} and obs/latest_obs.png into the run directory",
    )
    parser.add_argument("--map", default="loop_small", help="city map to drive on (default: loop_small)")
    parser.add_argument(
        "--allow-isaac-vram",
        action="store_true",
        help="acknowledge that a second Isaac process needs 2.5-3.5 GiB on top of training's "
        "6.4-7.6 GiB of 8 GiB; REQUIRED for --backend isaac",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="sample from the policy instead of using the deterministic mean action",
    )
    parser.add_argument("--seed", type=int, default=0, help="base episode seed (default: 0)")
    parser.add_argument("--device", default="cpu", help="torch device for the policy (default: cpu)")
    parser.add_argument(
        "--episode-seconds",
        type=float,
        default=30.0,
        help="episode truncation horizon in simulated seconds (default: 30)",
    )
    parser.add_argument("--fps", type=int, default=20, help="frames per second for recordings")
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="seconds between checkpoint polls (default: 2.0)",
    )
    parser.add_argument(
        "--wait-for-checkpoint",
        type=float,
        default=120.0,
        help="seconds to wait for the first checkpoint before giving up (default: 120)",
    )
    parser.add_argument(
        "--window",
        dest="window",
        action="store_true",
        default=None,
        help="force the interactive MuJoCo window on",
    )
    parser.add_argument(
        "--no-window",
        dest="window",
        action="store_false",
        help="run headless; required over SSH and in automated checks",
    )
    parser.add_argument(
        "--asset-dir",
        type=Path,
        default=None,
        help="directory for generated tile textures (default: a temporary directory)",
    )
    parser.add_argument(
        "--dynamics-dr",
        action="store_true",
        help="apply the S7.3 dynamics randomisation while viewing (sim-to-sim condition C6)",
    )
    parser.add_argument(
        "--photometric-dr",
        action="store_true",
        help="apply the shared photometric randomisation to the observation",
    )
    parser.add_argument(
        "--allow-unindexed",
        action="store_true",
        help="follow a run directory that has no checkpoints/index.json, using a weaker "
        "size-stability check instead of hash verification",
    )
    return parser


def resolve_run_dir(args: argparse.Namespace) -> Path:
    """Determine which run directory to watch.

    Args:
        args: Parsed arguments.

    Returns:
        The run directory.

    Raises:
        SystemExit: If no run directory can be found, with a message naming where it looked.
    """
    if args.run is not None:
        run_dir = Path(args.run)
        if not run_dir.is_dir():
            raise SystemExit(f"[live_view] --run {run_dir} is not a directory")
        return run_dir
    newest = find_latest_run(args.runs_root)
    if newest is None:
        raise SystemExit(
            f"[live_view] no run directories under {Path(args.runs_root).as_posix()}.\n"
            "  start a training run first, or pass --run <dir> explicitly."
        )
    print(f"[live_view] no --run given; following the newest run {newest.as_posix()}")
    return newest


def wait_for_first_checkpoint(watcher: CheckpointWatcher, timeout: float) -> CheckpointInfo:
    """Block until the first verified checkpoint appears.

    Args:
        watcher: The watcher to poll.
        timeout: Seconds to wait.

    Returns:
        The first checkpoint.

    Raises:
        SystemExit: On timeout, quoting the watcher's own reason for failing.
    """
    print(
        f"[live_view] watching {watcher.run_dir.as_posix()} for a {watcher.which!r} checkpoint "
        f"(up to {timeout:.0f}s)"
    )
    found = watcher.wait_for_new(timeout=timeout, stop=lambda: _STOP)
    if found is None:
        raise SystemExit(
            f"[live_view] no verified {watcher.which!r} checkpoint after {timeout:.0f}s.\n"
            f"  last reason: {watcher.last_error or 'none reported'}\n"
            "  the trainer publishes one via TrainLogger.save_checkpoint(), which calls "
            "RunDir.record_checkpoint(); pass --allow-unindexed to follow a run directory "
            "that never writes checkpoints/index.json."
        )
    return found


def apply_reload(
    host: PolicyHost,
    info: CheckpointInfo,
    probe_obs: dict[str, np.ndarray] | None,
) -> bool:
    """Hot-reload a checkpoint and print exactly what changed.

    The same observation is fed to the policy before and after the swap, so the log shows that the
    weights genuinely took effect rather than merely that a file was read.

    Args:
        host: The policy host.
        info: The checkpoint to load.
        probe_obs: An observation to evaluate before and after, or None to skip the comparison.

    Returns:
        True if the reload succeeded, False if it was refused.
    """
    before = None
    if probe_obs is not None and host.loaded:
        try:
            before = host.act(probe_obs, stochastic=False)
        except (KeyError, ValueError, RuntimeError):
            before = None
    try:
        state = host.load_from_info(info)
    except ArchitectureMismatch as exc:
        print(f"[live_view] REFUSED {info.path.name}: {exc}")
        print("[live_view] keeping the previously loaded policy")
        return False

    print(f"[live_view] RELOAD  {info.describe()}")
    print(f"[live_view]         {state.describe()}")
    if probe_obs is not None:
        try:
            after = host.act(probe_obs, stochastic=False)
        except (KeyError, ValueError, RuntimeError):
            after = None
        if before is not None and after is not None:
            delta = float(np.max(np.abs(after - before)))
            verdict = "CHANGED" if delta > 0.0 else "identical"
            print(
                f"[live_view]         action on the same observation: "
                f"{np.array2string(before, precision=5)} -> {np.array2string(after, precision=5)} "
                f"(max|delta|={delta:.6f}, {verdict})"
            )
    return True


def _window_wanted(args: argparse.Namespace) -> bool:
    """Resolve the tri-state window flag into a decision.

    ``--window`` sets it True, ``--no-window`` False, and the default is None, which means
    "headless unless asked". Headless is the safe default: it is what a recording run and an
    automated check both need.

    Args:
        args: The parsed command line.

    Returns:
        True when an interactive window was explicitly requested.
    """
    return bool(getattr(args, "window", None))


def _launch_isaac(headless: bool = True) -> Any:
    """Start Isaac Kit and return the app handle.

    Isaac Sim can only be configured before ``SimulationApp`` starts, which is why this runs
    before the environment is constructed rather than inside the backend.

    Args:
        headless: Run without the Isaac UI. Rendering still happens, because the viewer needs
            camera frames; only the editor window is suppressed.

    Returns:
        The ``SimulationApp`` handle, to be closed on the way out.

    Raises:
        ImportError: If Isaac Lab is not importable in this interpreter.
        RuntimeError: If Kit fails to start. The usual cause on Windows is exhausted commit
            during compute-pipeline compilation, which shows up as ``bad allocation`` inside
            ``vkCreateComputePipelines``; free host memory or raise the pagefile.
    """
    from isaaclab.app import AppLauncher

    print(f"[live_view] launching Isaac Sim (headless={headless}, cameras on); this takes a minute")
    launcher = AppLauncher(headless=headless, enable_cameras=True)
    print("[live_view] Isaac Sim up")
    return launcher.app


def _open_window(backend: Any, requested: bool | None) -> Any:
    """Try to open the interactive MuJoCo window.

    Args:
        backend: The rollout backend; only the MuJoCo one has a window.
        requested: True to force on, False to force off, None to try and degrade quietly.

    Returns:
        The passive viewer handle, or None when running headless.
        See :func:`_window_wanted` for how the tri-state flag is resolved.
    """
    if requested is False or not hasattr(backend, "env"):
        return None
    try:
        import mujoco.viewer

        handle = mujoco.viewer.launch_passive(backend.env.model, backend.env.data)
    except Exception as exc:  # any windowing failure degrades to headless, none is fatal
        message = f"[live_view] no interactive window ({type(exc).__name__}: {exc}); running headless"
        if requested is True:
            raise SystemExit(message) from exc
        print(message)
        return None
    print("[live_view] interactive MuJoCo window open; close it or press Ctrl-C to stop")
    return handle


def _write_live_frame(directory: Path, frame: Any) -> None:
    """Atomically publish the newest annotated frame as ``live_frame.png``.

    A watch page polls this file at about 1 Hz, which turns the headless viewer into a live
    on-screen Isaac view without a second Kit process. Failures are swallowed: a dropped live
    frame must never end an episode.

    Args:
        directory: The run's ``obs/`` directory.
        frame: The annotated RGB frame.
    """
    try:
        import imageio.v2 as iio

        # A distinct stem keeps the extension ".png" so imageio can infer the encoder; a
        # ".png.tmp" suffix makes imwrite raise (and the raise would be swallowed below).
        tmp = directory / "live_frame_next.png"
        iio.imwrite(tmp, frame)
        tmp.replace(directory / "live_frame.png")
    except Exception:
        pass


def run_episode(
    backend: Any,
    host: PolicyHost,
    watcher: CheckpointWatcher,
    episode: int,
    seed: int,
    record: bool,
    poll_every: int,
    window: Any,
    live_frame_dir: Path | None = None,
) -> dict[str, Any]:
    """Drive one episode, polling for new checkpoints as it goes.

    Args:
        backend: The rollout backend.
        host: The policy host.
        watcher: The checkpoint watcher.
        episode: Episode index, for logging.
        seed: Episode seed.
        record: Capture frames for the video.
        live_frame_dir: Directory to stream ``live_frame.png`` into every other step, or None.
        poll_every: Control steps between checkpoint polls.
        window: Passive viewer handle, or None.

    Returns:
        A summary dict with the episode's return, length, reloads and final observation.
    """
    obs = backend.reset(seed=seed)
    frames: list[np.ndarray] = []
    total_reward = 0.0
    reloads = 0
    steps = 0
    deviations: list[float] = []
    reason = "running"

    while not _STOP:
        if steps % poll_every == 0:
            found = watcher.poll()
            if found is not None and apply_reload(host, found, obs):
                reloads += 1

        action = host.act(obs)
        obs, reward, terminated, truncated, info = backend.step(action)
        total_reward += reward
        steps += 1
        deviations.append(abs(float(info.get("d", 0.0))))

        if record:
            frames.append(
                annotate_frame(
                    backend.render_frame(),
                    [
                        f"EP {episode}  STEP {steps}  R {total_reward:+.1f}",
                        f"ITER {host.state.iteration if host.state else -1}  "
                        f"ACT {action[0]:+.2f} {action[1]:+.2f}",
                    ],
                )
            )
            if live_frame_dir is not None and steps % 2 == 0:
                _write_live_frame(live_frame_dir, frames[-1])
        if window is not None:
            window.sync()
            if not window.is_running():
                reason = "window closed"
                break

        if terminated or truncated:
            reason = str(info.get("reason", "")) or ("truncated" if truncated else "terminated")
            break

    rms = float(np.sqrt(np.mean(np.square(deviations)))) if deviations else float("nan")
    print(
        f"[live_view] episode {episode}: steps={steps} return={total_reward:+.2f} "
        f"lane_dev_rms={rms:.4f} m reason={reason} reloads={reloads} "
        f"iteration={host.state.iteration if host.state else -1}"
    )
    return {
        "frames": frames,
        "obs": obs,
        "return": total_reward,
        "steps": steps,
        "reloads": reloads,
        "reason": reason,
        "lane_dev_rms": rms,
    }


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument list, or None to read ``sys.argv``.

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args(argv)
    _install_sigint_handler()

    # Refuse an unaffordable backend before doing anything slow. Waiting up to two minutes for a
    # checkpoint and loading a policy, only to then say the backend was never going to start, is
    # the kind of ordering that makes a tool feel broken.
    if args.backend == "isaac" and not args.allow_isaac_vram:
        print(f"[live_view] {isaac_vram_message()}", file=sys.stderr)
        return 3

    run_dir = resolve_run_dir(args)
    watcher = CheckpointWatcher(
        run_dir,
        which=args.which,
        poll_interval=args.poll_interval,
        require_index=not args.allow_unindexed,
    )
    first = wait_for_first_checkpoint(watcher, args.wait_for_checkpoint)

    host = PolicyHost(device=args.device, stochastic=args.stochastic, seed=args.seed)
    try:
        host.load_from_info(first)
    except ArchitectureMismatch as exc:
        print(f"[live_view] cannot start: {exc}", file=sys.stderr)
        return 2
    print(f"[live_view] START   {first.describe()}")
    print(f"[live_view]         {host.describe()}")

    # Isaac Sim can only be configured before SimulationApp starts, so Kit has to come up BEFORE
    # the environment is built. Keeping it here rather than at import time means the MuJoCo path
    # never pays for it, and a fresh clone with no Isaac still runs the default backend.
    isaac_app = None
    if args.backend == "isaac":
        try:
            isaac_app = _launch_isaac(headless=not _window_wanted(args))
        except (ImportError, RuntimeError) as exc:
            print(f"[live_view] cannot launch Isaac Sim: {exc}", file=sys.stderr)
            return 4

    try:
        backend = make_backend(
            backend=args.backend,
            map_name=args.map,
            episode_length_s=args.episode_seconds,
            seed=args.seed,
            allow_isaac_vram=args.allow_isaac_vram,
            asset_dir=args.asset_dir,
            device=args.device,
            dynamics_dr=args.dynamics_dr,
            photometric_dr=args.photometric_dr,
        )
    except IsaacVramRefusal as exc:
        print(f"[live_view] {exc}", file=sys.stderr)
        return 3
    except (RuntimeError, ImportError, ValueError) as exc:
        print(f"[live_view] cannot build the {args.backend} backend: {exc}", file=sys.stderr)
        return 4

    print(
        f"[live_view] backend={args.backend} map={args.map} control_dt={backend.control_dt:.4f}s "
        f"action={'sampled' if args.stochastic else 'deterministic mean'}"
    )
    # The interactive window is the MuJoCo passive viewer; on the Isaac backend the view comes
    # from the recorded frames and the live_frame stream (Kit's editor GUI hard-crashes when a
    # plain script steps the env underneath it, measured on this machine at uptime 21 s).
    window = _open_window(backend, args.window) if args.backend == "mujoco" else None
    poll_every = max(round(args.poll_interval / max(backend.control_dt, 1e-6)), 1)

    episode = 0
    started = time.monotonic()
    try:
        while not _STOP and (args.loop or episode < args.episodes):
            episode += 1
            result = run_episode(
                backend=backend,
                host=host,
                watcher=watcher,
                episode=episode,
                seed=args.seed + episode,
                record=args.record,
                poll_every=poll_every,
                window=window,
                live_frame_dir=(run_dir / "obs") if args.record else None,
            )
            if result["reason"] == "window closed":
                break
            if args.record:
                iteration = host.state.iteration if host.state else -1
                artifacts = write_rollout(
                    run_dir,
                    result["frames"],
                    fps=args.fps,
                    stacked_obs=result["obs"].get("rgb"),
                    obs_title=f"ITERATION {iteration}  EPISODE {episode}",
                )
                print(f"[live_view] {artifacts.message}")
    finally:
        if window is not None:
            with_close = getattr(window, "close", None)
            if callable(with_close):
                with_close()
        backend.close()
        if isaac_app is not None:
            # Kit's close() can hang on Windows after a rendering session, so bound the wait and
            # report rather than leaving the process wedged forever.
            closer = threading.Thread(target=isaac_app.close, daemon=True)
            closer.start()
            closer.join(timeout=30.0)
            if closer.is_alive():
                print("[live_view] note: Isaac Sim did not shut down within 30s; exiting anyway")

    elapsed = time.monotonic() - started
    print(f"[live_view] done: {episode} episode(s), {host.reload_count} policy load(s), {elapsed:.1f}s")

    if isaac_app is not None:
        # Kit's teardown segfaults during interpreter shutdown on Windows, AFTER every episode has
        # run and every artefact has been written, turning a completely successful run into exit
        # 139. Nothing useful happens between here and process exit, so leave immediately with a
        # truthful status. Streams are flushed first because os._exit skips buffers.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
