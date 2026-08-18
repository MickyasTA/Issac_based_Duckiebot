"""Training entry point: from-scratch PPO on the Isaac Lab lane-following task (SPEC v2 S5, S6).

Run it with the Isaac Sim interpreter, which on this machine is::

    & d:/Personal/personal/wheeled_quadruped_robot/.venv/Scripts/python.exe `
        scripts/train.py --task Duckiebot-LaneFollow-v0 --num_envs 256 --headless --enable_cameras

There is deliberately **no rendering flag**. Critic item G: ``--rendering_mode`` on the command
line outranks ``RenderCfg.rendering_mode`` (``simulation_context.py:741-745``), so exposing it
would let an ablation silently run a different renderer than its config records, and the number
that came out would be unfalsifiable. The renderer is chosen once, in
:class:`duckiebot_rl.envs.env_cfg.RenderingSettings`, and nowhere else.

The two hard launch rules of Isaac Lab are obeyed literally below:

1. ``AppLauncher.add_app_launcher_args(parser)`` comes AFTER this script's own arguments.
2. No ``isaaclab`` / ``isaacsim`` / ``omni`` import happens before ``AppLauncher(...)`` is
   constructed, other than ``from isaaclab.app import AppLauncher`` itself. Everything else is
   imported inside :func:`main`. The two ``duckiebot_rl.viz`` imports at module scope are
   standard-library-only by construction, which is the whole point of that layer, and
   ``duckiebot_rl.envs.viz_env`` imports nothing heavier than numpy at module scope (its Kit
   imports all live inside function bodies) while ``duckiebot_rl.envs`` itself is PEP 562 lazy.
   The parser needs the mesh vocabulary at import time, which is why that one is not deferred.

Robot visuals
-------------

``--robot-mesh`` chooses what the robot is DRAWN as, on every one of the N environments, and
nothing else: :func:`duckiebot_rl.envs.viz_env.attach_robot_visuals` only adds visual-only USD
references under links that already exist and hides the environment's own visual scopes. No
collider, rigid body, joint, mass or camera prim is touched, so the physics and the observation
contract are identical for ``db21j``, ``db17`` and ``primitive``. The robot's own geometry also
falls entirely inside the policy camera's 0.05 m near plane, which
``tests/unit/test_robot_mesh.py`` proves vertex by vertex; what a mesh swap CAN still change is
indirect lighting and the robot's own shadow, so a campaign that cares compares
``scripts/check_obs.py --robot-mesh primitive`` against ``--robot-mesh db21j`` before it starts.

Where a run goes
----------------

No path is spelled out in this file. Every artifact is written by
:class:`duckiebot_rl.viz.TrainLogger`, which owns the house-standard run directory::

    training_results/<UTCstamp>_<name>_seed<N>/
        model_best.pth  model_latest.pth  model_episode_<N>.pth  model_final.pth
        train.log  config.yaml  status.json  metrics.jsonl
        checkpoints/index.json
        metrics/episodes_<session>.csv     one row per EPISODE, the resume source of truth
        metrics/summary_<session>.json
        metrics/runs/<session>/            TensorBoard events
        metrics/graphs/                    one PNG per scalar tag, _overview.png, _series.json

The older ad-hoc ``checkpoints/<timestamp>`` plus ``logs/<timestamp>`` pair is gone, and with it
``--log-dir`` and ``--checkpoint-dir``: a run is one directory, and ``--run-root`` moves it.

``--resume`` takes that run directory (or a ``.pth`` inside it, or the word ``latest``) and keeps
writing into it: same run id, same episode CSV, same ``metrics/graphs/_series.json``, so the
curves **extend** instead of restarting. The episode counter and the global step come back from
the CSV through :meth:`TrainLogger.resume_state`, which is what makes the CSV the source of truth.

The rollout loop and the buffer contract
----------------------------------------

``RolloutBuffer.capture_terminal`` defaults its step index to ``buffer.current_step``, which is
the slot ``add`` is about to write. The loop below therefore has exactly one legal order::

    out = ppo.act(obs)                 # obs is what the policy sees at step t
    env.step(out["clipped_action"])    # _reset_idx fires INSIDE this call and captures terminals
    buffer.add(obs, ..., reward, terminated, truncated)

Swapping the last two lines shifts every captured terminal by one rollout slot, which silently
corrupts the truncation bootstrap and shows up only as a critic that never explains the variance.

What a checkpoint contains (SPEC v2 S6.9)
-----------------------------------------

Model, optimiser, learning rate, iteration, global step, the three running normalisers, the RNG
states of torch/numpy/python, the config hash, the git commit, and - mandatory, checked on load -
the curriculum state: ``alpha_vis``, ``alpha_dyn``, the ADR buffers and the hard-example mining
table. Without those last fields a resume silently restarts domain randomization at alpha 0 and
the run quietly becomes a different experiment.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from duckiebot_rl.envs.viz_env import DEFAULT_ROBOT_MESH, ROBOT_MESH_CHOICES  # noqa: E402
from duckiebot_rl.viz.metrics_logger import EpisodeRecord  # noqa: E402
from duckiebot_rl.viz.run_dir import DEFAULT_RESULTS_ROOT, RunDir, find_latest_run  # noqa: E402

_TERMINATION_CONDITIONS = ("off_drivable", "obstacle", "rollover", "stall", "spin")
"""The five S5.5 terminating conditions, in the order :class:`TerminationFlags` declares them."""

_FLAT_STAT_KEYS = frozenset(
    {
        "policy_loss",
        "value_loss",
        "entropy",
        "bounds_loss",
        "approx_kl",
        "clipfrac",
        "explained_variance",
        "grad_norm",
        "learning_rate",
        "mean_sigma",
    }
)
"""PPO diagnostics that the metrics row carries unprefixed, because the dashboard's fixed panels
are built from exactly these names. The rest of ``PPO.update``'s stats keep a ``train/`` prefix."""

_ADR_ALPHA_KEYS = ("curriculum/alpha_vis", "curriculum/alpha_dyn")
"""``TwoScalarADR.metrics`` reports the two alphas under the same tags that the contract's flat
``alpha_vis`` and ``alpha_dyn`` keys are grouped into, so one of the two copies has to go."""

_HOST_SAMPLE_PERIOD_S = 30.0
"""How often the nvidia-smi and commit-limit probes may run. Each is a subprocess, so they are
sampled on a timer and the previous reading is carried forward in between."""

_BANNER = "=" * 100


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser, with the AppLauncher arguments appended last.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        description="Train the Duckiebot lane-following policy with the from-scratch PPO of S6.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    run = parser.add_argument_group("run")
    run.add_argument("--task", default="Duckiebot-LaneFollow-v0", help="task name, recorded in logs")
    run.add_argument("--num_envs", type=int, default=256, help="parallel environments N")
    run.add_argument("--seed", type=int, default=0, help="master seed for torch, numpy and random")
    run.add_argument("--max_iterations", type=int, default=20000, help="PPO iterations to run")
    run.add_argument("--total-steps", type=int, default=None, help="stop after this many env steps")
    run.add_argument(
        "--resume",
        default=None,
        help="run directory to continue, a .pth inside one, or 'latest' for the newest run",
    )
    run.add_argument("--run-name", default=None, help="name component of the run id")
    run.add_argument("--run-root", default=None, help=f"results root; defaults to ./{DEFAULT_RESULTS_ROOT}")
    run.add_argument("--session", default="train", help="session id in episodes_<session>.csv")
    run.add_argument(
        "--graph-refresh",
        type=float,
        default=60.0,
        help="seconds between live graph renders; they run inside this process, so on a "
        "commit-limited box a longer period is cheaper. scripts/dashboard.py renders for free.",
    )

    task = parser.add_argument_group("task")
    task.add_argument("--vec-only", action="store_true", help="drop the camera (S6.2 vec-only mode)")
    task.add_argument("--no-visual-dr", action="store_true", help="disable the S4.3 photometric DR")
    task.add_argument("--no-dynamics-dr", action="store_true", help="disable the S7.3 dynamics DR")
    task.add_argument("--obstacles", action="store_true", help="spawn the S5.1 obstacle field")
    task.add_argument("--obstacle-stage", type=int, default=3, help="S7.4 task-curriculum stage")
    task.add_argument("--city-root", default=None, help="directory holding the generated city USD")
    task.add_argument("--num-variants", type=int, default=64, help="training city layouts")
    task.add_argument(
        "--robot-mesh",
        dest="robot_mesh",
        choices=ROBOT_MESH_CHOICES,
        default=DEFAULT_ROBOT_MESH,
        help="robot VISUAL only, applied to all N environments: db21j is the Duckiematrix DB21, "
        "db17 the per-part classic meshes, primitive the environment's own boxes. Physics, "
        "collisions, actuation and the policy camera are identical for all three; a mesh that "
        "is not on disk downgrades to the next one instead of failing the run",
    )

    schedule = parser.add_argument_group("schedule")
    schedule.add_argument("--save-interval", type=int, default=100, help="iterations between saves")
    schedule.add_argument("--archive-every", type=int, default=500, help="episodes between archive copies")
    schedule.add_argument("--eval-interval", type=int, default=250, help="iterations between evals")
    schedule.add_argument("--eval-steps", type=int, default=1200, help="control steps per eval")
    schedule.add_argument("--no-eval", action="store_true", help="skip the S6.10 evaluation loop")

    # MUST be last: add_app_launcher_args inspects the parser it is given.
    AppLauncher.add_app_launcher_args(parser)
    return parser


def reject_rendering_override(args: argparse.Namespace) -> None:
    """Refuse to run if the renderer was chosen on the command line (critic item G).

    ``AppLauncher.add_app_launcher_args`` always registers ``--rendering_mode``, so this script
    cannot remove the flag. What it can do is refuse to proceed when it was used. The precedence
    is real: ``AppLauncher._set_rendering_mode_settings`` writes the CLI value into
    ``/isaaclab/rendering/rendering_mode``, and ``simulation_context.py:741-745`` prefers that
    setting over ``RenderCfg.rendering_mode``. Left unguarded, ``--rendering_mode quality`` would
    run an ablation on a renderer its own config claims it did not use.

    Left unset, the launcher writes an empty string, which is falsy, and the environment config
    wins - which is the intended path.

    Args:
        args: Parsed arguments.

    Raises:
        SystemExit: If ``--rendering_mode`` was passed explicitly.
    """
    if getattr(args, "rendering_mode", None):
        raise SystemExit(
            "refusing to run: --rendering_mode was set on the command line, and it silently "
            "outranks RenderCfg (simulation_context.py:741-745). The renderer is a property of "
            "the experiment, not of the invocation: change RenderingSettings in "
            "duckiebot_rl/envs/env_cfg.py instead, so the config records what actually ran."
        )


def results_root(args: argparse.Namespace) -> Path:
    """Return the results root every run directory is created under.

    Args:
        args: Parsed arguments.

    Returns:
        ``--run-root`` when given, otherwise ``<repo>/training_results``.
    """
    return Path(args.run_root).expanduser() if args.run_root else _REPO_ROOT / DEFAULT_RESULTS_ROOT


def run_name(args: argparse.Namespace) -> str:
    """Return the name component of the run id.

    Args:
        args: Parsed arguments.

    Returns:
        ``--run-name`` when given, otherwise a name that says which experiment this is.
    """
    if args.run_name:
        return str(args.run_name)
    if args.obstacles:
        return "obstacles"
    return "veconly" if args.vec_only else "lanefollow"


def resolve_resume(target: str, root: Path) -> tuple[RunDir, Path]:
    """Resolve ``--resume`` to the run directory to continue and the checkpoint to load.

    Three forms are accepted, because all three are what a tired operator actually types: the run
    directory itself, a ``.pth`` inside it, and the literal ``latest``.

    Args:
        target: The ``--resume`` value.
        root: Where runs live, used to resolve ``latest``.

    Returns:
        ``(run, checkpoint_path)``.

    Raises:
        SystemExit: If nothing resumable is there. Failing loudly beats silently starting a fresh
            run that the operator believes is a continuation.
    """
    if target == "latest":
        found = find_latest_run(root)
        if found is None:
            raise SystemExit(f"--resume latest: no run directory under {root.as_posix()}")
        path = found
    else:
        path = Path(target).expanduser()

    if path.is_dir():
        run = RunDir.open(path)
        checkpoint = run.latest_checkpoint
    elif path.is_file():
        run = RunDir.open(path.parent)
        checkpoint = path
    else:
        raise SystemExit(f"--resume: {path.as_posix()} does not exist")

    if not checkpoint.is_file():
        raise SystemExit(f"--resume: {checkpoint.as_posix()} does not exist")
    return run, checkpoint


def _seed_everything(seed: int) -> None:
    """Seed torch, numpy and the stdlib generator.

    The stdlib ``random`` module is not optional: ``MultiUsdFileCfg`` picks each environment's
    city stage with it (``sim/spawners/wrappers/wrappers.py:8,111``), so an unseeded stdlib
    generator makes the layout assignment irreproducible across runs (critic item D).

    Args:
        seed: Master seed.
    """
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - the legacy global stream is what Isaac Lab itself seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_settings(args: argparse.Namespace) -> Any:
    """Translate the command line into :class:`LaneFollowSettings`.

    Args:
        args: Parsed arguments.

    Returns:
        The populated settings object.
    """
    from duckiebot_rl.envs.env_cfg import CitySettings, LaneFollowSettings, ObstacleSettings

    return LaneFollowSettings(
        num_envs=args.num_envs,
        device=args.device,
        seed=args.seed,
        city=CitySettings(root=args.city_root, num_variants=args.num_variants),
        obstacles=ObstacleSettings(enabled=args.obstacles, stage=args.obstacle_stage),
        use_image=not args.vec_only,
        visual_dr=not args.no_visual_dr,
        dynamics_dr=not args.no_dynamics_dr,
    )


def build_learner(settings: Any, args: argparse.Namespace) -> tuple[Any, Any]:
    """Construct the PPO learner and its rollout buffer from the S6.6 hyperparameters.

    Args:
        settings: The environment settings, whose space widths the network must match.
        args: Parsed arguments.

    Returns:
        ``(learner, buffer)``.
    """
    from duckiebot_rl.ppo import ActorCritic, NetworkConfig, PPOConfig, RolloutBuffer, configure_precision
    from duckiebot_rl.ppo.ppo import PPO

    strict = configure_precision()
    network = NetworkConfig(
        use_image=settings.use_image,
        obs_height=settings.spaces.obs_height,
        obs_width=settings.spaces.obs_width,
        obs_channels=settings.spaces.obs_channels,
        vec_dim=settings.spaces.vec_dim,
        priv_dim=settings.spaces.priv_dim,
        act_dim=settings.spaces.act_dim,
    )
    cfg = PPOConfig(
        num_envs=settings.num_envs,
        seed=args.seed,
        device=settings.device,
        network=network,
        total_timesteps=args.total_steps or PPOConfig.total_timesteps,
    )
    print(f"[train] precision: {'strict fp32 (TF32 off)' if strict else 'fp32 with TF32 kernels'}")
    learner = PPO(ActorCritic(network), cfg, device=settings.device)
    # The two kernel-level accelerations are gated on the device and on strict mode, so what the
    # config REQUESTED and what the learner APPLIED can differ. Print what was applied: a run that
    # cannot say which kernels it used is a run whose timings cannot be compared with another's.
    print(
        f"[train] accelerations applied: channels_last={learner.channels_last} "
        f"fused_adam={learner.fused_optimizer} checkpoint_encoder={cfg.checkpoint_encoder}"
    )
    buffer = RolloutBuffer(
        num_steps=cfg.num_steps,
        num_envs=cfg.num_envs,
        vec_dim=network.vec_dim,
        priv_dim=network.priv_dim,
        act_dim=network.act_dim,
        obs_shape=settings.spaces.rgb_shape if settings.use_image else None,
        device=settings.device,
    )
    return learner, buffer


def run_config(args: argparse.Namespace, settings: Any, learner_cfg: Any = None) -> dict[str, Any]:
    """Assemble the fully resolved config that lands verbatim in ``config.yaml``.

    Args:
        args: Parsed arguments.
        settings: The environment settings.
        learner_cfg: The :class:`PPOConfig`, once it exists. It is omitted on the first write,
            which happens before the learner is built so that a crash during scene creation
            still leaves a readable run directory behind.

    Returns:
        A JSON-safe mapping.
    """
    skip = {"experience", "kit_args"}
    cli = {key: value for key, value in sorted(vars(args).items()) if key not in skip}
    config: dict[str, Any] = {"cli": cli, "env": settings.summary()}
    if learner_cfg is not None:
        ppo = {key: value for key, value in vars(learner_cfg).items() if key != "network"}
        ppo["batch_size"] = learner_cfg.batch_size
        ppo["minibatch_size"] = learner_cfg.minibatch_size
        ppo["network"] = vars(learner_cfg.network)
        config["ppo"] = ppo
    return config


class HostSampler:
    """Samples the two host resources that actually end multi-day runs on this machine.

    VRAM is the headline 8 GiB constraint and the Windows commit limit is the documented killer
    of long Isaac sessions here, so both belong in the metrics row next to the losses.

    Both probes are subprocesses and they are SLOW: ``nvidia-smi`` plus a ``Get-CimInstance``
    PowerShell launch were measured at 1179 ms together, which is 4 to 5% of wall time at the
    N=64 iteration cadence when they run inline on the training thread. They therefore run on a
    daemon thread of their own and :meth:`sample` only ever reads the last completed reading, so
    building a metrics row costs a dict copy under a lock and never blocks the learner. The
    readings stay on the same :data:`_HOST_SAMPLE_PERIOD_S` cadence they always had; what changed
    is which thread pays for them.

    Monitoring must never be able to end a training run, so the worker swallows every probe
    failure and simply leaves the affected key absent.

    Attributes:
        period_s: Seconds between probes.
    """

    def __init__(self, period_s: float = _HOST_SAMPLE_PERIOD_S, start: bool = True) -> None:
        """Create a sampler and, by default, start its background thread.

        Args:
            period_s: Seconds between probes.
            start: Start the worker thread immediately. Tests pass False and drive
                :meth:`probe_once` directly.
        """
        self.period_s = float(period_s)
        self._values: dict[str, float] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        if start:
            self.start()

    def start(self) -> None:
        """Start the probing thread if it is not already running."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="host-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Ask the probing thread to finish. It is a daemon, so this is a courtesy, not a duty."""
        self._stop.set()

    def _loop(self) -> None:
        """Probe, publish, wait, repeat, until :meth:`stop` is called."""
        while not self._stop.is_set():
            values = self.probe_once()
            with self._lock:
                self._values = values
            self._stop.wait(self.period_s)

    def sample(self) -> dict[str, float]:
        """Return the most recent completed host readings without blocking.

        Returns:
            ``{"vram_nvsmi_mb": ..., "gpu_temp_c": ..., "free_commit_gb": ...}``, with any probe
            that failed simply absent, and empty until the first probe has completed.
        """
        with self._lock:
            return dict(self._values)

    def probe_once(self) -> dict[str, float]:
        """Run both probes now, on the calling thread, and return what they produced.

        Returns:
            The readings that succeeded; failures are omitted rather than raised.
        """
        values: dict[str, float] = {}
        gpu = self._run(
            ["nvidia-smi", "--query-gpu=memory.used,temperature.gpu", "--format=csv,noheader,nounits"]
        )
        if gpu:
            parts = gpu.splitlines()[0].split(",")
            try:
                values["vram_nvsmi_mb"] = float(parts[0])
                values["gpu_temp_c"] = float(parts[1])
            except (IndexError, ValueError):
                pass
        query = "(Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory"
        commit = self._run(["powershell", "-NoProfile", "-Command", query])
        if commit:
            with contextlib.suppress(IndexError, ValueError):
                values["free_commit_gb"] = float(commit.split()[0]) / (1024.0 * 1024.0)
        return values

    @staticmethod
    def _run(command: list[str]) -> str:
        """Run one probe and return its stdout, or an empty string if it did not work.

        Args:
            command: Argument vector.

        Returns:
            Captured stdout, stripped.
        """
        try:
            done = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
        except (OSError, subprocess.SubprocessError):
            return ""
        return done.stdout.strip() if done.returncode == 0 else ""


class EpisodeTracker:
    """Turns the vectorised step stream into one :class:`EpisodeRecord` per finished episode.

    The environment reports episode metrics as means over an iteration, which is the right shape
    for a scalar plot and the wrong shape for the house standard's per-episode CSV. This tracker
    keeps the four per-env accumulators that CSV needs and empties them on the step an env
    finishes, so every episode is recorded as it ends carrying its own ``global_step``.

    Nothing in the hot path synchronises with the device. The naive shape of this class asks
    ``done.any()`` on every control step and then, on every step where an episode did end, runs a
    ``nonzero`` and five ``.tolist()`` transfers; the campaign profile counted that among the 23.4
    host synchronisations per step, and its cost grows with ``N`` because the fraction of steps
    containing a reset does. Instead :meth:`update` writes the per-env accumulators into
    preallocated ``(T, N)`` planes and a ``(T, N)`` "an episode finished here" mask, all on the
    device, and :meth:`drain` converts the whole rollout in two transfers at the end of the
    iteration. The training loop only ever consumes the episodes after the rollout has finished,
    so deferring costs nothing, and the records - their contents and their order, step-major then
    env-major - are exactly the ones the per-step version produced.

    Lateral deviation is read from the privileged critic vector, whose first ``vec_dim`` entries
    are the actor vector and whose next entry is ``d`` in metres (S5.2). It is sampled from the
    vector the policy acted on at the start of the step rather than after it, because ``env.step``
    has already respawned the finished envs by the time it returns and their terminal ``d`` is
    gone. One step of a 450-step episode is not worth reaching into environment internals for.

    One episode per environment is deliberately **not** written. ``_reset_idx`` staggers
    ``episode_length_buf`` with a random start so the horizon truncations do not all land on the
    same iteration, which means the episode in flight when the tracker starts began before the
    tracker was watching: its return and its lane statistics cover only the tail of it. The CSV
    is the source of truth for resume and for the S8.4 tables, so it gets whole episodes only,
    and the first one each env finishes after a start, a resume or an evaluation is dropped.

    Attributes:
        num_envs: Parallel environment count.
        step_dt: Control period in seconds, which turns a step count into a duration.
        d_index: Index of ``d`` inside ``vec_priv``.
    """

    def __init__(self, num_envs: int, device: Any, step_dt: float, d_index: int, horizon: int = 32) -> None:
        """Allocate the per-env accumulators and the per-rollout staging planes.

        Args:
            num_envs: Parallel environment count.
            device: Torch device the accumulators live on.
            step_dt: Control period in seconds.
            d_index: Index of ``d`` inside ``vec_priv``, that is ``spaces.vec_dim``.
            horizon: Expected number of :meth:`update` calls between drains, that is the rollout
                length ``T``. The planes double if a caller exceeds it, so this is only a hint.
        """
        import torch

        self.num_envs = int(num_envs)
        self.step_dt = float(step_dt)
        self.d_index = int(d_index)
        self._torch = torch
        self._device = device
        self._return = torch.zeros(self.num_envs, dtype=torch.float32, device=device)
        self._steps = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self._d_sq = torch.zeros(self.num_envs, dtype=torch.float32, device=device)
        self._d_max = torch.zeros(self.num_envs, dtype=torch.float32, device=device)
        self._partial = torch.ones(self.num_envs, dtype=torch.bool, device=device)
        self._dropped = torch.zeros((), dtype=torch.long, device=device)
        self._cursor = 0
        self._context: list[tuple[int, float, str]] = []
        self._horizon = 0
        self._allocate(max(1, int(horizon)))

    def _allocate(self, horizon: int) -> None:
        """Preallocate the staging planes for ``horizon`` control steps.

        Args:
            horizon: Number of rollout steps the planes must hold.
        """
        torch = self._torch
        shape = (horizon, self.num_envs)
        self._horizon = horizon
        self._p_finished = torch.zeros(*shape, dtype=torch.bool, device=self._device)
        self._p_return = torch.zeros(*shape, dtype=torch.float32, device=self._device)
        self._p_steps = torch.zeros(*shape, dtype=torch.long, device=self._device)
        self._p_d_sq = torch.zeros(*shape, dtype=torch.float32, device=self._device)
        self._p_d_max = torch.zeros(*shape, dtype=torch.float32, device=self._device)
        self._p_success = torch.zeros(*shape, dtype=torch.bool, device=self._device)

    def _planes(self) -> tuple[Any, ...]:
        """Return the staging planes in a fixed order.

        Returns:
            The six ``(T, N)`` staging tensors.
        """
        return (
            self._p_finished,
            self._p_return,
            self._p_steps,
            self._p_d_sq,
            self._p_d_max,
            self._p_success,
        )

    def _grow(self) -> None:
        """Double the staging planes, preserving whatever has already been staged."""
        previous, cursor, old_horizon = self._planes(), self._cursor, self._horizon
        self._allocate(old_horizon * 2)
        for old, new in zip(previous, self._planes(), strict=True):
            new[:old_horizon] = old
        self._cursor = cursor

    @property
    def dropped(self) -> int:
        """How many partial episodes have been discarded since the tracker was built.

        Reading this synchronises with the device, so the training loop reads it once, at the end,
        rather than accumulating it on the host every step.

        Returns:
            The cumulative count of dropped partial episodes.
        """
        return int(self._dropped.item())

    def reset(self) -> None:
        """Drop every episode in flight and every record not yet drained.

        Called after an evaluation, which steps the same environments and then resets them all:
        the episodes that were in flight before it are not episodes the training stream produced,
        and the ones that start after it are staggered again.
        """
        self._return.zero_()
        self._steps.zero_()
        self._d_sq.zero_()
        self._d_max.zero_()
        self._partial.fill_(True)
        self._cursor = 0
        self._context.clear()

    def update(
        self,
        vec_priv: Any,
        reward: Any,
        terminated: Any,
        truncated: Any,
        global_step: int,
        reason: str,
    ) -> None:
        """Fold one control step into the accumulators and stage any episodes that ended on it.

        Args:
            vec_priv: ``(N, priv_dim)`` privileged vector the policy acted on this step.
            reward: ``(N,)`` reward the step returned.
            terminated: ``(N,)`` bool, a failure condition fired.
            truncated: ``(N,)`` bool, the time limit was reached.
            global_step: Total env steps consumed once this step is counted.
            reason: Why the terminating envs terminated; see :func:`termination_reason`. It is
                recorded for every step and read back only for the steps that produced an episode.
        """
        if self._cursor >= self._horizon:
            self._grow()

        deviation = vec_priv[:, self.d_index].detach().abs().to(self._d_sq.dtype)
        self._return += reward.detach().to(self._return.dtype)
        self._steps += 1
        self._d_sq += deviation * deviation
        self._torch.maximum(self._d_max, deviation, out=self._d_max)

        done = terminated | truncated
        # An episode counts only if the tracker watched it from step one.
        finished = done & ~self._partial
        self._dropped += (done & self._partial).sum()
        self._partial &= ~done

        step = self._cursor
        self._p_finished[step] = finished
        self._p_return[step] = self._return
        self._p_steps[step] = self._steps
        self._p_d_sq[step] = self._d_sq
        self._p_d_max[step] = self._d_max
        self._p_success[step] = truncated
        self._context.append((int(global_step), time.time(), reason))
        self._cursor = step + 1

        # The accumulators of every finished env are cleared, dropped ones included. masked_fill_
        # rather than an index write, because building that index means a nonzero and a
        # synchronisation for a result the host never looks at.
        self._return.masked_fill_(done, 0.0)
        self._steps.masked_fill_(done, 0)
        self._d_sq.masked_fill_(done, 0.0)
        self._d_max.masked_fill_(done, 0.0)

    def drain(self) -> list[EpisodeRecord]:
        """Return every episode that finished since the last drain and rewind the staging planes.

        Returns:
            The finished episodes in step order, then environment order. Empty when the rollout
            produced none, or when the only episodes that ended were the partial first ones.
        """
        count, self._cursor = self._cursor, 0
        context, self._context = self._context, []
        if count == 0:
            return []

        rows = self._p_finished[:count].nonzero(as_tuple=False)
        if rows.numel() == 0:
            return []
        step_index, env_index = rows[:, 0], rows[:, 1]
        lengths = self._p_steps[step_index, env_index].to(self._p_return.dtype)
        payload = self._torch.stack(
            (
                self._p_return[step_index, env_index],
                lengths,
                self._torch.sqrt(self._p_d_sq[step_index, env_index] / lengths.clamp(min=1.0)),
                self._p_d_max[step_index, env_index],
                self._p_success[step_index, env_index].to(self._p_return.dtype),
            ),
            dim=1,
        ).tolist()
        staged_at = step_index.tolist()

        records: list[EpisodeRecord] = []
        for (score, length, rms, peak, success), where in zip(payload, staged_at, strict=True):
            global_step, stamp, reason = context[where]
            reached_horizon = success > 0.5
            records.append(
                EpisodeRecord(
                    score=float(score),
                    steps=int(length),
                    duration=float(length) * self.step_dt,
                    global_step=global_step,
                    timestamp=stamp,
                    lane_dev_rms=float(rms),
                    lane_dev_max=float(peak),
                    success=reached_horizon,
                    termination_reason="truncated" if reached_horizon else reason,
                )
            )
        return records


def termination_reason(env: Any) -> str:
    """Name the S5.5 condition that fired on the step just taken.

    ``TerminationFlags`` carries a per-env mask for each of the five conditions, but the
    environment keeps only the per-step counts (``_termination_counts``, which it also publishes
    through ``drain_episode_log`` as ``terminations/*``). That is enough to name the reason
    exactly whenever a single condition fired anywhere on the step, which is the overwhelmingly
    common case, and to name the candidates honestly when several did.

    Args:
        env: The environment.

    Returns:
        A condition name, several joined by ``+`` when the step is ambiguous, or ``terminated``
        when the environment publishes no counts at all.
    """
    counts = getattr(env, "_termination_counts", None) or {}
    fired = [name for name in _TERMINATION_CONDITIONS if counts.get(name)]
    if len(fired) == 1:
        return fired[0]
    return "+".join(fired) if fired else "terminated"


def episode_metrics(records: list[EpisodeRecord]) -> dict[str, float]:
    """Aggregate the episodes that finished in one iteration into the flat house-standard keys.

    Args:
        records: Episodes that finished during one iteration.

    Returns:
        ``ep_return_mean``, ``ep_return_std``, ``ep_len_mean``, ``success_rate``,
        ``lane_dev_rms_m``, ``lane_dev_max_m`` and ``episodes_this_iter``. With no episodes the
        six statistics are NaN, which ``metrics.jsonl`` writes as ``null`` and the plot writer
        drops, rather than a zero that would read as a real measurement.
    """
    if not records:
        nan = float("nan")
        return {
            "ep_return_mean": nan,
            "ep_return_std": nan,
            "ep_len_mean": nan,
            "success_rate": nan,
            "lane_dev_rms_m": nan,
            "lane_dev_max_m": nan,
            "episodes_this_iter": 0.0,
        }
    returns = [record.score for record in records]
    mean = sum(returns) / len(returns)
    variance = sum((value - mean) ** 2 for value in returns) / len(returns)
    rms = [record.lane_dev_rms for record in records if math.isfinite(record.lane_dev_rms)]
    peaks = [record.lane_dev_max for record in records if math.isfinite(record.lane_dev_max)]
    return {
        "ep_return_mean": mean,
        "ep_return_std": math.sqrt(variance),
        "ep_len_mean": sum(record.steps for record in records) / len(records),
        "success_rate": sum(1.0 for record in records if record.success) / len(records),
        "lane_dev_rms_m": sum(rms) / len(rms) if rms else float("nan"),
        "lane_dev_max_m": max(peaks) if peaks else float("nan"),
        "episodes_this_iter": float(len(records)),
    }


def vram_used_mb() -> float:
    """Return the CUDA memory this process has reserved, in MiB.

    Reserved rather than allocated: the caching allocator's reservation is what the 8 GiB budget
    is spent against, and it is what ``nvidia-smi`` reports.

    Returns:
        MiB reserved, or NaN when there is no CUDA device.
    """
    import torch

    if not torch.cuda.is_available():
        return float("nan")
    return float(torch.cuda.memory_reserved()) / (1024.0 * 1024.0)


def run_evaluation(
    env: Any,
    learner: Any,
    steps: int,
    heartbeat: Callable[[], None] | None = None,
    record: Any = None,
    iteration: int = 0,
) -> dict[str, float]:
    """Run the S6.10 deterministic evaluation on the training envs.

    Training is frozen, the policy acts on its mean, the DR alphas are held where they are and
    the per-step photometric DR is switched off. The rollout is discarded for learning: it exists
    to produce the model-selection metric, which is the mean lane-frame consecutive distance in
    tiles, tie-broken on collisions.

    When ``record`` is given, env 0's Isaac-rendered camera frames are captured along the way and
    written as ``videos/latest_rollout.{mp4,gif}`` plus ``obs/latest_obs.png`` in the run
    directory. This is the Isaac view of the trained policy DURING training, at zero extra VRAM:
    the frames are the very tiles the training camera already renders, so no second Kit process
    is needed. The clone is mandatory; ``data.output["rgb"]`` is a view into the live buffer.

    Args:
        env: The environment.
        learner: The PPO learner.
        steps: Control steps to run.
        heartbeat: Called every 50 steps so ``status.json`` does not go stale while an evaluation
            longer than the dashboard's freshness window is in progress.
        record: The run's ``RunDir`` (or None to skip recording).
        iteration: Training iteration, stamped onto the recorded artifacts.

    Returns:
        A dict of ``eval/*`` metrics; empty if no episode completed.
    """
    import torch

    photometric_was_on = env.settings.visual_dr
    env.settings.visual_dr = False
    env.drain_episode_log()
    frames: list[Any] = []
    camera = getattr(env, "_camera", None) if record is not None else None
    max_frames = 450  # one full episode at 15 Hz; caps the memory and the encode time
    try:
        with torch.no_grad():
            for step in range(steps):
                out = learner.act(
                    env.stacked_obs if env.settings.use_image else None,
                    env.vec,
                    env.vec_priv,
                    deterministic=True,
                )
                env.step(out["clipped_action"])
                if camera is not None and len(frames) < max_frames:
                    rgb = camera.data.output.get("rgb")
                    if rgb is not None:
                        frames.append(rgb[0, ..., :3].clone())
                if heartbeat is not None and step % 50 == 0:
                    heartbeat()
        metrics = env.drain_episode_log()
    finally:
        env.settings.visual_dr = photometric_was_on

    if record is not None and frames:
        # Encode on CPU after the eval so the capture itself stays off the hot path. A failure
        # to encode must never take training down.
        try:
            import numpy as np

            from duckiebot_rl.viz.render import write_rollout

            stack = torch.stack(frames).cpu().numpy()
            if stack.dtype != np.uint8:
                stack = (np.clip(stack, 0.0, 1.0) * 255.0).astype(np.uint8)
            obs_stack = None
            if env.settings.use_image:
                obs_stack = env.stacked_obs[0].detach().cpu().numpy()
            result = write_rollout(
                record.root,
                list(stack),
                fps=15,
                stacked_obs=obs_stack,
                obs_title=f"EVAL ITER {iteration}",
            )
            print(f"[train] eval recording: {result.frames} Isaac frames -> videos/latest_rollout.*")
        except Exception as exc:
            print(f"[train] eval recording skipped: {exc!r}")
    return {f"eval/{key.split('/', 1)[-1]}": value for key, value in metrics.items()}


def save_run_checkpoint(
    log: Any,
    writer: Callable[..., Path],
    learner: Any,
    iteration: int,
    global_step: int,
    curriculum_state: dict[str, Any],
    settings: Any,
    selection: float | None,
) -> None:
    """Hand one checkpoint to the logger, which owns ``model_best.pth`` and the index.

    Until the first evaluation produces a selection metric there is nothing to select on, so
    ``model_best.pth`` simply tracks the latest weights. That keeps the promise the layout makes
    to every reader of it - ``scripts/live_view.py --which best`` works from the first save - and
    the moment a real ``eval/distance_tiles`` arrives, ordinary model selection takes over and
    never hands the name back.

    Args:
        log: The open :class:`TrainLogger`.
        writer: ``duckiebot_rl.ppo.checkpoint.save_checkpoint``.
        learner: The PPO learner.
        iteration: Completed iteration count.
        global_step: Total env steps consumed.
        curriculum_state: The mandatory S6.9 curriculum payload.
        settings: Environment settings, stored as the checkpoint's fingerprint.
        selection: ``eval/distance_tiles`` when an evaluation just ran, else None.
    """
    unselected = log.status.best_metric_value is None
    log.save_checkpoint(
        lambda path: writer(
            path,
            learner,
            iteration,
            global_step,
            curriculum_state,
            config=learner.cfg,
            env_fingerprint=settings.summary(),
            extra={"best_metric": log.status.best_metric_value, "selection_metric": selection},
        ),
        metric=selection,
        is_best=True if selection is None and unselected else None,
        iteration=iteration,
    )


def main() -> int:
    """Open the run directory, train inside it, and return a process exit code.

    Returns:
        0 on a clean finish.
    """
    from duckiebot_rl.viz import TrainLogger

    args, _unknown = _PARSER.parse_known_args()
    _seed_everything(args.seed)
    settings = build_settings(args)
    root = results_root(args)

    resume_run, resume_checkpoint, previous = None, None, None
    if args.resume:
        resume_run, resume_checkpoint = resolve_resume(args.resume, root)
        # Read the old heartbeat BEFORE opening the logger. TrainLogger writes a fresh
        # status.json in its constructor, so a read afterwards returns zeros and the run would
        # silently forget which checkpoint was best and how long it had already been training.
        previous = resume_run.read_status()
        # The heartbeat's wall clock is this-session-only (``finish`` stamps its own elapsed
        # time), so the cumulative figure comes from the last metrics row, which is exactly the
        # cumulative value the previous session wrote. Three sessions in, only this still adds up.
        rows = resume_run.read_metrics()
        if previous is not None and rows:
            logged = rows[-1].get("wall_clock_s")
            if isinstance(logged, int | float) and logged > previous.wall_clock_s:
                previous.wall_clock_s = float(logged)

    log = TrainLogger.create(
        resume_run.root.parent if resume_run is not None else root,
        name=run_name(args),
        seed=args.seed,
        run_id=resume_run.run_id if resume_run is not None else None,
        config=run_config(args, settings),
        num_envs=args.num_envs,
        device=str(args.device),
        best_metric="eval/distance_tiles",
        best_mode="max",
        archive_every=args.archive_every,
        session=args.session,
        graph_refresh_s=args.graph_refresh,
    )
    print(f"\n{_BANNER}\n[train] RUN DIRECTORY  {log.run.root.as_posix()}\n{_BANNER}\n")
    log.write_log(f"launch: {' '.join(sys.argv[1:])}")

    try:
        code = train(args, settings, log, resume_checkpoint, previous)
    except BaseException:
        log.finish(state="crashed")
        print(f"\n{_BANNER}\n[train] CRASHED. RUN DIRECTORY  {log.run.root.as_posix()}\n{_BANNER}\n")
        raise
    log.finish(state="finished")
    print(f"\n{_BANNER}\n[train] RUN DIRECTORY  {log.run.root.as_posix()}\n{_BANNER}\n")
    return code


def train(
    args: argparse.Namespace,
    settings: Any,
    log: Any,
    resume_checkpoint: Path | None,
    previous: Any = None,
) -> int:
    """Run the PPO loop against an already-open :class:`TrainLogger`.

    Split out of :func:`main` so that every exit path, a crash inside scene creation included,
    goes through exactly one ``log.finish`` call and one printed run directory.

    Args:
        args: Parsed arguments.
        settings: The environment settings.
        log: The open logger, which owns the run directory.
        resume_checkpoint: Checkpoint to restore, or None for a fresh run.
        previous: The heartbeat this run directory carried before the logger reopened it, read in
            :func:`main`; it is where the best metric and the accumulated wall clock come from.

    Returns:
        0 on a clean finish.
    """
    from duckiebot_rl.dr.curriculum import TwoScalarADR
    from duckiebot_rl.envs.env_cfg import expected_carb_settings, lane_follow_env_cfg
    from duckiebot_rl.envs.lane_follow_env import DuckiebotLaneFollowEnv
    from duckiebot_rl.envs.viz_env import attach_robot_visuals
    from duckiebot_rl.ppo import load_checkpoint, save_checkpoint

    host = HostSampler()
    print(f"[train] building the scene: {settings.num_envs} envs, {settings.city.num_variants} layouts")
    cfg = lane_follow_env_cfg(settings)
    env = DuckiebotLaneFollowEnv(cfg)
    # Visual only, and only after the scene exists: the robots are addressed through the stage,
    # as /World/envs/env_<i>/Robot. One line is printed per distinct outcome, not per environment.
    applied = attach_robot_visuals(env, robot_mesh=args.robot_mesh, num_envs=settings.num_envs)
    print(f"[train] robot visuals: requested {args.robot_mesh!r}, applied {applied!r}")
    # SPEC v2 S4.4 acceptance item 3: the antialiasing setter swallows its own exceptions, so
    # the only way to know the renderer is configured is to read carb back after launch.
    observed = env.carb_settings_readback()
    for path, want in expected_carb_settings(settings.rendering).items():
        got = observed.get(path)
        flag = "ok " if got == want else "MISMATCH"
        print(f"[train] carb {flag} {path} = {got!r} (expected {want!r})")

    learner, buffer = build_learner(settings, args)
    env.attach_rollout_buffer(buffer)
    log.run.write_config(run_config(args, settings, learner.cfg))

    adr = TwoScalarADR()
    env.attach_curriculum(adr)
    env.set_curriculum_alphas(adr.alpha_vis, adr.alpha_dyn)

    iteration, global_step, wall_clock_offset = 0, 0, 0.0
    if resume_checkpoint is not None:
        payload = load_checkpoint(resume_checkpoint, learner=learner)
        iteration = int(payload["iteration"])
        global_step = int(payload["global_step"])
        curriculum = payload["curriculum"]
        if "adr" in curriculum:
            adr.load_state_dict(curriculum["adr"])
        env.load_env_state_dict(curriculum)
        env.set_curriculum_alphas(adr.alpha_vis, adr.alpha_dyn)

        # The episode CSV, not the checkpoint, is the house standard's source of truth for the
        # counters: it is appended as episodes finish, so it is never behind the last save.
        resumed = log.resume_state()
        if resumed.global_step > global_step:
            print(
                f"[train] the episode CSV is ahead of the checkpoint: step {resumed.global_step:,} "
                f"vs {global_step:,}; continuing from the CSV so the curves stay monotone"
            )
            global_step = resumed.global_step
        # Model selection and the wall clock continue from the heartbeat this directory carried
        # before the logger reopened it; the checkpoint's own record is the fallback.
        if previous is not None:
            log.status.best_metric_value = previous.best_metric_value
            log.status.best_iteration = previous.best_iteration
            wall_clock_offset = float(previous.wall_clock_s or 0.0)
        if log.status.best_metric_value is None:
            recorded = payload.get("extra", {}).get("best_metric")
            log.status.best_metric_value = None if recorded is None else float(recorded)
        log.status.iteration, log.status.total_timesteps = iteration, global_step
        message = (
            f"resumed from {Path(resume_checkpoint).as_posix()} at iteration {iteration}, "
            f"step {global_step:,}, episode {resumed.episode} ({resumed.rows} CSV rows), "
            f"alpha_vis {adr.alpha_vis:.3f}, alpha_dyn {adr.alpha_dyn:.3f}, "
            f"best {log.status.best_metric_value}, wall clock {wall_clock_offset:.0f} s"
        )
        print(f"[train] {message}")
        log.write_log(message)

    # A resume fully resets every env: PhysX state is not checkpointable (S6.9), so the
    # environment stream restores statistically while the learner restores exactly.
    env.reset()
    tracker = EpisodeTracker(
        env.num_envs,
        settings.device,
        env.step_dt,
        settings.spaces.vec_dim,
        horizon=buffer.num_steps,
    )
    steps_per_iteration = buffer.num_steps * env.num_envs
    opening = (
        f"{steps_per_iteration} env-steps per iteration, {learner.cfg.minibatch_size} minibatch, "
        f"{env.num_envs} envs, save every {args.save_interval} it, eval every {args.eval_interval} it"
    )
    print(f"[train] {opening}")
    log.write_log(opening)
    started = time.perf_counter()

    while iteration < args.max_iterations:
        if args.total_steps is not None and global_step >= args.total_steps:
            break
        rollout_started = time.perf_counter()
        for step in range(buffer.num_steps):
            image = env.stacked_obs if settings.use_image else None
            vec, vec_priv = env.vec.clone(), env.vec_priv.clone()
            out = learner.act(image, vec, vec_priv)
            _obs, reward, terminated, truncated, _extras = env.step(out["clipped_action"])
            buffer.add(
                vec=vec,
                vec_priv=vec_priv,
                action=out["action"],
                log_prob=out["log_prob"],
                value=out["value"],
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                mu=out["mu"],
                log_std=out["log_std"],
                image=image,
            )
            tracker.update(
                vec_priv,
                reward,
                terminated,
                truncated,
                global_step + (step + 1) * env.num_envs,
                termination_reason(env),
            )
        # One transfer for the whole rollout, instead of one host synchronisation per step.
        finished = tracker.drain()
        rollout_seconds = time.perf_counter() - rollout_started

        update_started = time.perf_counter()
        num_terminals = learner.compute_returns(
            buffer,
            env.stacked_obs if settings.use_image else None,
            env.vec_priv,
        )
        stats = learner.update(buffer)
        buffer.reset()
        update_seconds = time.perf_counter() - update_started

        for name, values in env.drain_curriculum_records().items():
            adr.record(name, values)
        actions = adr.update()
        env.set_curriculum_alphas(adr.alpha_vis, adr.alpha_dyn)

        iteration += 1
        global_step += steps_per_iteration
        # Episodes are appended before the iteration row that summarises them, so a kill between
        # the two loses one iteration row and never an episode.
        log.log_episodes(finished)
        episode = env.drain_episode_log()

        is_eval = (not args.no_eval) and iteration % args.eval_interval == 0
        evaluation: dict[str, float] = {}
        if is_eval:
            log.status.iteration, log.status.total_timesteps = iteration, global_step
            evaluation = run_evaluation(
                env, learner, args.eval_steps, heartbeat=log.heartbeat, record=log.run, iteration=iteration
            )
            distance = evaluation.get("eval/distance_tiles", float("nan"))
            print(f"[train] eval at it {iteration}: distance_tiles {distance:.2f}")

        iteration_seconds = rollout_seconds + update_seconds
        row: dict[str, Any] = {
            "iteration": iteration,
            "total_timesteps": global_step,
            "wall_clock_s": wall_clock_offset + (time.perf_counter() - started),
            **episode_metrics(finished),
            "policy_loss": stats["policy_loss"],
            "value_loss": stats["value_loss"],
            "entropy": stats["entropy"],
            "bounds_loss": stats.get("bounds_loss", float("nan")),
            "approx_kl": stats["approx_kl"],
            "clipfrac": stats["clipfrac"],
            "explained_variance": stats["explained_variance"],
            "grad_norm": stats["grad_norm"],
            "learning_rate": stats["learning_rate"],
            "target_kl": learner.cfg.kl_target_upper,
            "mean_sigma": stats.get("mean_sigma", float("nan")),
            "alpha_vis": adr.alpha_vis,
            "alpha_dyn": adr.alpha_dyn,
            "steps_per_s": steps_per_iteration / max(1e-9, iteration_seconds),
            "rollout_steps_per_s": steps_per_iteration / max(1e-9, rollout_seconds),
            "vram_used_mb": vram_used_mb(),
            **host.sample(),
            # The diagnostics the table above does not name keep the train/ prefix; the ones it
            # names are already flat, and logging both would draw every PPO curve twice.
            **{f"train/{key}": value for key, value in stats.items() if key not in _FLAT_STAT_KEYS},
            **episode,
            **env.reward_term_means(),
            # Already tagged curriculum/*; the two alphas are dropped because the flat
            # alpha_vis / alpha_dyn above land on the same tags and would draw each point twice.
            **{key: value for key, value in adr.metrics().items() if key not in _ADR_ALPHA_KEYS},
            **evaluation,
            "time/rollout_s": rollout_seconds,
            "time/update_s": update_seconds,
            "buffer/terminals_captured": float(num_terminals),
        }
        log.log_iteration(row)
        if log.tensorboard is not None:
            for name, action in actions.items():
                log.tensorboard.add_text(f"curriculum/{name}_action", action, global_step)

        line = (
            f"it {iteration:6d}  step {global_step:>12,}  "
            f"kl {stats['approx_kl']:.4f}  ev {stats['explained_variance']:+.3f}  "
            f"lr {stats['learning_rate']:.2e}  "
            f"fps {row['steps_per_s']:8.0f}  eps {len(finished):4d}  "
            f"a_vis {adr.alpha_vis:.2f}  a_dyn {adr.alpha_dyn:.2f}"
        )
        print(f"[train] {line}")
        log.write_log(line)

        if is_eval or iteration % args.save_interval == 0 or iteration >= args.max_iterations:
            save_run_checkpoint(
                log,
                save_checkpoint,
                learner,
                iteration,
                global_step,
                {**env.env_state_dict(), "adr": adr.state_dict()},
                settings,
                evaluation.get("eval/distance_tiles") if is_eval else None,
            )
        if is_eval:
            env.reset()
            tracker.reset()

    save_run_checkpoint(
        log,
        save_checkpoint,
        learner,
        iteration,
        global_step,
        {**env.env_state_dict(), "adr": adr.state_dict()},
        settings,
        None,
    )
    env.close()
    closing = (
        f"finished at iteration {iteration}, {global_step:,} env steps, "
        f"{log.resume_state().episode} episodes logged, {tracker.dropped} partial episodes dropped"
    )
    print(f"[train] {closing}")
    log.write_log(closing)
    return 0


_PARSER = build_parser()

if __name__ == "__main__":
    _args, _hydra = _PARSER.parse_known_args()
    reject_rendering_override(_args)
    if _args.enable_cameras is False and not _args.vec_only:
        # The camera env cannot run without it: TiledCamera raises at sensor init if the carb
        # setting /isaaclab/cameras_enabled is False, and that setting comes from the .kit
        # experience file AppLauncher selects from this flag.
        _args.enable_cameras = True
    _app_launcher = AppLauncher(_args)
    _simulation_app = _app_launcher.app

    _code = main()
    _simulation_app.close()
    raise SystemExit(_code)
