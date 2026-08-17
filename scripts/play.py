"""Load a checkpoint, drive the policy, and report episode metrics (SPEC v2 S5, S6.10, S8.4).

Run it with the Isaac Sim interpreter::

    & d:/Personal/personal/wheeled_quadruped_robot/.venv/Scripts/python.exe `
        scripts/play.py --checkpoint training_results/<run>/model_best.pth --num_envs 16 --enable_cameras

    # headless, held-out layouts, recording the first environment to MP4
    & $ISAAC scripts/play.py --checkpoint training_results/<run>/model_best.pth --headless --enable_cameras `
        --eval-maps --video --video-length 900

Three uses, one script:

* **Watch it drive.** Drop ``--headless`` and keep ``--enable_cameras``; the viewer opens on the
  first environment.
* **Record.** ``--video`` writes an MP4 of the policy's own observation stream - the stacked
  frame it actually acted on, upscaled - rather than a free camera, so what you see is what the
  network saw, principal-point jitter and all.
* **Measure.** Every run prints the four AI-DO-style metrics of S6.8: lane-frame consecutive
  distance, survival time, time-integrated ``|d|`` and time-integrated out-of-lane, plus the
  per-condition termination histogram.

The policy is deterministic by default (``a = mu``), which is the S6.10 model-selection setting.
``--stochastic`` samples instead, which is what you want when you are looking for the failure
modes rather than the headline number.

Domain randomization is OFF by default here. The evaluation conditions of S8.4 are defined by
which DR is enabled, so it has to be an explicit flag rather than an inherited default: a
"C0 nominal" number accidentally measured with the training DR still on is not a C0 number.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
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
        description="Run a trained Duckiebot lane-following policy and report episode metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="checkpoint .pt written by train.py")
    parser.add_argument("--num_envs", type=int, default=16, help="parallel environments")
    parser.add_argument("--seed", type=int, default=0, help="master seed")
    parser.add_argument("--steps", type=int, default=1350, help="control steps to run (3 episodes)")
    parser.add_argument("--stochastic", action="store_true", help="sample instead of using the mean")
    parser.add_argument("--vec-only", action="store_true", help="load a vec-only (S6.2) policy")
    parser.add_argument("--visual-dr", action="store_true", help="enable the S4.3 photometric DR")
    parser.add_argument("--dynamics-dr", action="store_true", help="enable the S7.3 dynamics DR")
    parser.add_argument("--obstacles", action="store_true", help="spawn the S5.1 obstacle field")
    parser.add_argument("--city-root", default=None, help="directory holding the generated city USD")
    parser.add_argument("--num-variants", type=int, default=64, help="city layouts to load")
    parser.add_argument("--eval-maps", action="store_true", help="use the 4 held-out layouts instead")
    parser.add_argument("--video", action="store_true", help="write an MP4 of the policy's own view")
    parser.add_argument("--video-length", type=int, default=450, help="frames to record")
    parser.add_argument("--out-dir", default="outputs/play", help="where the report and video land")

    # MUST be last: add_app_launcher_args inspects the parser it is given.
    AppLauncher.add_app_launcher_args(parser)
    return parser


def build_settings(args: argparse.Namespace, alphas: tuple[float, float]) -> Any:
    """Translate the command line into :class:`LaneFollowSettings`.

    Args:
        args: Parsed arguments.
        alphas: ``(alpha_vis, alpha_dyn)`` restored from the checkpoint. They are applied only
            when the matching DR flag is given, so ``--visual-dr`` reproduces the DR level the
            policy was last trained at rather than a fresh alpha of 0.

    Returns:
        The populated settings object.
    """
    from duckiebot_rl.envs.env_cfg import CitySettings, LaneFollowSettings, ObstacleSettings

    variants = 4 if args.eval_maps else args.num_variants
    return LaneFollowSettings(
        num_envs=args.num_envs,
        device=args.device,
        seed=args.seed,
        city=CitySettings(root=args.city_root, num_variants=variants),
        obstacles=ObstacleSettings(enabled=args.obstacles),
        use_image=not args.vec_only,
        visual_dr=args.visual_dr,
        dynamics_dr=args.dynamics_dr,
        dr_alpha_vis=alphas[0] if args.visual_dr else 0.0,
        dr_alpha_dyn=alphas[1] if args.dynamics_dr else 0.0,
    )


def write_video(frames: list[Any], path: Path, fps: float) -> bool:
    """Write the recorded observation frames to an MP4 if a codec is available.

    Args:
        frames: List of ``(H, W, 3)`` uint8 numpy arrays.
        path: Destination file.
        fps: Frame rate.

    Returns:
        True if the file was written, False if no encoder is installed. Returning a flag rather
        than raising keeps a missing optional codec from destroying a completed evaluation.
    """
    if not frames:
        return False
    try:
        import cv2
    except ImportError:
        print("[play] opencv is not installed, so no video was written (pip install duckiebot-rl[cv])")
        return False
    height, width = frames[0].shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(path.as_posix(), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    try:
        for frame in frames:
            writer.write(frame[:, :, ::-1])  # RGB -> BGR, the classic deployment bug, done once
    finally:
        writer.release()
    return True


def summarise(values: list[float]) -> dict[str, float]:
    """Return count, mean, median and standard deviation of a metric.

    Args:
        values: Per-episode samples.

    Returns:
        A dict of summary statistics; zeros when nothing completed.
    """
    if not values:
        return {"count": 0.0, "mean": 0.0, "median": 0.0, "std": 0.0}
    return {
        "count": float(len(values)),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def main() -> int:
    """Run the policy and write the report.

    Returns:
        0 on success, 1 if the checkpoint could not be matched to the requested spaces.
    """
    args, _unknown = _PARSER.parse_known_args()
    import numpy as np
    import torch

    from duckiebot_rl.envs.env_cfg import lane_follow_env_cfg
    from duckiebot_rl.envs.lane_follow_env import DuckiebotLaneFollowEnv
    from duckiebot_rl.ppo import ActorCritic, NetworkConfig, PPOConfig, load_checkpoint
    from duckiebot_rl.ppo.ppo import PPO

    payload = load_checkpoint(args.checkpoint, learner=None, restore_rng=False)
    curriculum = payload.get("curriculum", {})
    alphas = (float(curriculum.get("alpha_vis", 0.0)), float(curriculum.get("alpha_dyn", 0.0)))
    print(
        f"[play] checkpoint iteration {payload['iteration']}, step {payload['global_step']:,}, "
        f"alpha_vis {alphas[0]:.2f}, alpha_dyn {alphas[1]:.2f}, commit {payload.get('git_commit')}"
    )

    settings = build_settings(args, alphas)
    network = NetworkConfig(
        use_image=settings.use_image,
        obs_height=settings.spaces.obs_height,
        obs_width=settings.spaces.obs_width,
        obs_channels=settings.spaces.obs_channels,
        vec_dim=settings.spaces.vec_dim,
        priv_dim=settings.spaces.priv_dim,
        act_dim=settings.spaces.act_dim,
    )
    ppo_cfg = PPOConfig(num_envs=args.num_envs, device=args.device, network=network)
    learner = PPO(ActorCritic(network), ppo_cfg)
    try:
        learner.load_state_dict(payload["learner"])
    except (KeyError, RuntimeError) as exc:
        print(f"[play] the checkpoint does not match the requested observation spaces: {exc}")
        return 1

    cfg = lane_follow_env_cfg(settings)
    env = DuckiebotLaneFollowEnv(cfg, render_mode="rgb_array" if args.video else None)
    env.reset()
    env.drain_episode_log()

    frames: list[Any] = []
    with torch.no_grad():
        for step in range(args.steps):
            out = learner.act(
                env.stacked_obs if settings.use_image else None,
                env.vec,
                env.vec_priv,
                deterministic=not args.stochastic,
            )
            env.step(out["clipped_action"])
            if args.video and settings.use_image and step < args.video_length:
                # Channels 0-2 are the newest frame of the (t, t-2, t-4) stack: the image the
                # policy acted on this step, not a free camera looking at the robot.
                newest = env.stacked_obs[0, :, :, :3].detach().cpu().numpy().astype(np.uint8)
                frames.append(np.repeat(np.repeat(newest, 4, axis=0), 4, axis=1))

    metrics = env.drain_episode_log()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "checkpoint": str(Path(args.checkpoint).resolve().as_posix()),
        "iteration": payload["iteration"],
        "global_step": payload["global_step"],
        "git_commit": payload.get("git_commit"),
        "deterministic": not args.stochastic,
        "settings": settings.summary(),
        "control_steps": args.steps,
        "metrics": metrics,
    }
    report_path = out_dir / "play_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("[play] episode metrics (means over completed episodes):")
    for key in sorted(metrics):
        print(f"  {key:<44s} {metrics[key]:10.4f}")
    if not metrics:
        print("  no episode completed within the requested number of control steps")

    if args.video:
        video_path = out_dir / "policy_view.mp4"
        if write_video(frames, video_path, settings.rates.control_hz):
            print(f"[play] wrote {len(frames)} frames to {video_path.as_posix()}")

    print(f"[play] report written to {report_path.as_posix()}")
    env.close()
    return 0


_PARSER = build_parser()

if __name__ == "__main__":
    _args, _hydra = _PARSER.parse_known_args()
    if _args.enable_cameras is False and not _args.vec_only:
        _args.enable_cameras = True
    _app_launcher = AppLauncher(_args)
    _simulation_app = _app_launcher.app

    _code = main()
    _simulation_app.close()
    raise SystemExit(_code)
