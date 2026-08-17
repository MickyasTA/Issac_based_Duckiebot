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
   imported inside :func:`main`.

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
import sys
import time
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


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
    run.add_argument("--resume", default=None, help="checkpoint to resume from")
    run.add_argument("--run-name", default=None, help="log subdirectory; defaults to a timestamp")
    run.add_argument("--log-dir", default="logs", help="TensorBoard root")
    run.add_argument("--checkpoint-dir", default="checkpoints", help="checkpoint root")

    task = parser.add_argument_group("task")
    task.add_argument("--vec-only", action="store_true", help="drop the camera (S6.2 vec-only mode)")
    task.add_argument("--no-visual-dr", action="store_true", help="disable the S4.3 photometric DR")
    task.add_argument("--no-dynamics-dr", action="store_true", help="disable the S7.3 dynamics DR")
    task.add_argument("--obstacles", action="store_true", help="spawn the S5.1 obstacle field")
    task.add_argument("--obstacle-stage", type=int, default=3, help="S7.4 task-curriculum stage")
    task.add_argument("--city-root", default=None, help="directory holding the generated city USD")
    task.add_argument("--num-variants", type=int, default=64, help="training city layouts")

    schedule = parser.add_argument_group("schedule")
    schedule.add_argument("--save-interval", type=int, default=250, help="iterations between saves")
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


def _resolve_run_dir(args: argparse.Namespace) -> tuple[Path, Path]:
    """Return the TensorBoard and checkpoint directories for this run.

    Args:
        args: Parsed arguments.

    Returns:
        ``(log_dir, checkpoint_dir)``, both created.
    """
    name = args.run_name or time.strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path(args.log_dir) / name
    checkpoint_dir = Path(args.checkpoint_dir) / name
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return log_dir, checkpoint_dir


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


def run_evaluation(env: Any, learner: Any, steps: int) -> dict[str, float]:
    """Run the S6.10 deterministic evaluation on the training envs.

    Training is frozen, the policy acts on its mean, the DR alphas are held where they are and
    the per-step photometric DR is switched off. The rollout is discarded: it exists only to
    produce the model-selection metric, which is the mean lane-frame consecutive distance in
    tiles, tie-broken on collisions.

    Args:
        env: The environment.
        learner: The PPO learner.
        steps: Control steps to run.

    Returns:
        A dict of ``eval/*`` metrics; empty if no episode completed.
    """
    import torch

    photometric_was_on = env.settings.visual_dr
    env.settings.visual_dr = False
    env.drain_episode_log()
    try:
        with torch.no_grad():
            for _ in range(steps):
                out = learner.act(
                    env.stacked_obs if env.settings.use_image else None,
                    env.vec,
                    env.vec_priv,
                    deterministic=True,
                )
                env.step(out["clipped_action"])
        metrics = env.drain_episode_log()
    finally:
        env.settings.visual_dr = photometric_was_on
    return {f"eval/{key.split('/', 1)[-1]}": value for key, value in metrics.items()}


def main() -> int:
    """Build the environment, train, and return a process exit code.

    Returns:
        0 on a clean finish.
    """
    args, _unknown = _PARSER.parse_known_args()
    from torch.utils.tensorboard import SummaryWriter

    from duckiebot_rl.dr.curriculum import TwoScalarADR
    from duckiebot_rl.envs.env_cfg import lane_follow_env_cfg
    from duckiebot_rl.envs.lane_follow_env import DuckiebotLaneFollowEnv
    from duckiebot_rl.ppo import load_checkpoint, save_checkpoint

    _seed_everything(args.seed)
    log_dir, checkpoint_dir = _resolve_run_dir(args)
    settings = build_settings(args)

    print(f"[train] building the scene: {settings.num_envs} envs, {settings.city.num_variants} layouts")
    cfg = lane_follow_env_cfg(settings)
    env = DuckiebotLaneFollowEnv(cfg)
    # SPEC v2 S4.4 acceptance item 3: the antialiasing setter swallows its own exceptions, so
    # the only way to know the renderer is configured is to read carb back after launch.
    from duckiebot_rl.envs.env_cfg import expected_carb_settings

    observed = env.carb_settings_readback()
    for path, want in expected_carb_settings(settings.rendering).items():
        got = observed.get(path)
        flag = "ok " if got == want else "MISMATCH"
        print(f"[train] carb {flag} {path} = {got!r} (expected {want!r})")

    learner, buffer = build_learner(settings, args)
    env.attach_rollout_buffer(buffer)

    adr = TwoScalarADR()
    env.attach_curriculum(adr)
    env.set_curriculum_alphas(adr.alpha_vis, adr.alpha_dyn)

    iteration, global_step, best_metric = 0, 0, float("-inf")
    if args.resume:
        payload = load_checkpoint(args.resume, learner=learner)
        iteration = int(payload["iteration"])
        global_step = int(payload["global_step"])
        curriculum = payload["curriculum"]
        if "adr" in curriculum:
            adr.load_state_dict(curriculum["adr"])
        env.load_env_state_dict(curriculum)
        best_metric = float(payload.get("extra", {}).get("best_metric", float("-inf")))
        print(f"[train] resumed from {args.resume} at iteration {iteration}, step {global_step}")

    writer = SummaryWriter(log_dir.as_posix())
    for key, value in settings.summary().items():
        writer.add_text(f"config/{key}", str(value), 0)

    # A resume fully resets every env: PhysX state is not checkpointable (S6.9), so the
    # environment stream restores statistically while the learner restores exactly.
    env.reset()
    steps_per_iteration = buffer.num_steps * env.num_envs
    print(f"[train] {steps_per_iteration} env-steps per iteration, {learner.cfg.minibatch_size} minibatch")

    while iteration < args.max_iterations:
        if args.total_steps is not None and global_step >= args.total_steps:
            break
        rollout_started = time.perf_counter()
        for _ in range(buffer.num_steps):
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
        episode = env.drain_episode_log()
        scalars: dict[str, float] = {
            **{f"train/{k}": v for k, v in stats.items()},
            **episode,
            **env.reward_term_means(),
            **{f"curriculum/{k}": v for k, v in adr.metrics().items()},
            "time/rollout_s": rollout_seconds,
            "time/update_s": update_seconds,
            "time/env_steps_per_s": steps_per_iteration / max(1e-9, rollout_seconds),
            "buffer/terminals_captured": float(num_terminals),
        }
        for key, value in scalars.items():
            writer.add_scalar(key, value, global_step)
        for name, action in actions.items():
            writer.add_text(f"curriculum/{name}_action", action, global_step)

        print(
            f"[train] it {iteration:6d}  step {global_step:>12,}  "
            f"kl {stats['approx_kl']:.4f}  ev {stats['explained_variance']:+.3f}  "
            f"lr {stats['learning_rate']:.2e}  "
            f"fps {steps_per_iteration / max(1e-9, rollout_seconds):8.0f}  "
            f"a_vis {adr.alpha_vis:.2f}  a_dyn {adr.alpha_dyn:.2f}"
        )

        curriculum_state = {**env.env_state_dict(), "adr": adr.state_dict()}
        is_eval = (not args.no_eval) and iteration % args.eval_interval == 0
        if is_eval:
            metrics = run_evaluation(env, learner, args.eval_steps)
            for key, value in metrics.items():
                writer.add_scalar(key, value, global_step)
            selection = metrics.get("eval/distance_tiles", float("-inf"))
            print(f"[train] eval at it {iteration}: distance_tiles {selection:.2f}")
            if selection > best_metric:
                best_metric = selection
                save_checkpoint(
                    checkpoint_dir / "best.pt",
                    learner,
                    iteration,
                    global_step,
                    curriculum_state,
                    config=learner.cfg,
                    env_fingerprint=settings.summary(),
                    extra={"best_metric": best_metric, "eval": metrics},
                )
            env.reset()

        if iteration % args.save_interval == 0 or iteration >= args.max_iterations:
            save_checkpoint(
                checkpoint_dir / "last.pt",
                learner,
                iteration,
                global_step,
                curriculum_state,
                config=learner.cfg,
                env_fingerprint=settings.summary(),
                extra={"best_metric": best_metric},
            )

    save_checkpoint(
        checkpoint_dir / "last.pt",
        learner,
        iteration,
        global_step,
        {**env.env_state_dict(), "adr": adr.state_dict()},
        config=learner.cfg,
        env_fingerprint=settings.summary(),
        extra={"best_metric": best_metric},
    )
    writer.close()
    env.close()
    print(f"[train] finished at iteration {iteration}, {global_step:,} env steps")
    print(f"[train] checkpoints in {checkpoint_dir.as_posix()}, TensorBoard in {log_dir.as_posix()}")
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
