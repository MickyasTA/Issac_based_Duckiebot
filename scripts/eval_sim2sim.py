"""Command line entry point for the MuJoCo sim-to-sim harness (SPEC v2 S8, owner ``[sim2sim]``).

Run this with the **tools venv**, which is the only interpreter on this machine that has MuJoCo::

    d:/Personal/personal/mujoco_venv/Scripts/python.exe scripts/eval_sim2sim.py check-env

Subcommands:

``check-env``
    Report what this interpreter can run and print the exact pip commands for anything missing.
``build``
    Generate ``duckiebot.xml`` and a track scene, compile both in MuJoCo, and write them to disk.
``sysid``
    Run the two-stage Isaac-to-MuJoCo matching against a reference file, or against a synthetic
    self-test reference when no file is given, and write the parameter file plus residual report.
``eval``
    Run conditions C5 and C6 of the S8.4 matrix and write the JSON report.

Every subcommand prints its provenance: which module each robot constant came from, whether the
tile markings came from the shared city generator, and whether the physics rates match the Isaac
reference. A number without that context is not reportable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from duckiebot_rl.sim2sim import _resolve  # noqa: E402
from duckiebot_rl.sim2sim import evaluate as _evaluate  # noqa: E402
from duckiebot_rl.sim2sim import mjcf as _mjcf  # noqa: E402
from duckiebot_rl.sim2sim import sysid as _sysid  # noqa: E402
from duckiebot_rl.sim2sim import track as _track  # noqa: E402
from duckiebot_rl.sim2sim.env import MjEnvCfg  # noqa: E402


def cmd_check_env(_args: argparse.Namespace) -> int:
    """Print the interpreter capability report.

    Args:
        _args: unused.

    Returns:
        0 if MuJoCo and numpy are present, 1 otherwise.
    """
    report = _resolve.environment_report()
    print(_resolve.format_environment_report(report))
    robot, source = _resolve.resolve_robot_params()
    sim, rate_source = _resolve.resolve_sim_params()
    print(f"robot parameters : {source}")
    print(
        f"physics rates    : {rate_source} -> dt {sim.physics_dt:.6f} s, "
        f"decimation {sim.decimation}, control {sim.control_hz:.2f} Hz"
    )
    print(
        f"ground clearance : {robot.ground_clearance * 1e3:.1f} mm, "
        f"caster r {robot.caster_radius * 1e3:.1f} mm"
    )
    ok = bool(report["packages"]["mujoco"]) and bool(report["packages"]["numpy"])
    return 0 if ok else 1


def cmd_build(args: argparse.Namespace) -> int:
    """Generate and compile the robot MJCF and a track scene.

    Args:
        args: parsed arguments carrying ``out`` and ``map``.

    Returns:
        0 on success.
    """
    import mujoco

    out = Path(args.out)
    cfg = _mjcf.MjcfCfg.from_shared()
    robot_path = _mjcf.write_robot_xml(out / "duckiebot.xml", cfg)
    model = mujoco.MjModel.from_xml_path(str(robot_path))
    print(f"robot  : {robot_path}")
    print(
        f"         {model.nbody - 1} bodies, {model.nv} dof, {model.nu} actuators, "
        f"{model.ngeom} geoms, provenance: {cfg.params_source}"
    )

    source = args.map if args.map else _track.LOOP_5X5
    scene = _track.build_track(source, cfg=_mjcf.MjcfCfg.from_shared(), asset_dir=str(out))
    scene_path = scene.write(out / "track.xml")
    scene_model = mujoco.MjModel.from_xml_path(str(scene_path))
    issues = scene.lane.check_against_map()
    print(f"track  : {scene_path}")
    print(
        f"         map {scene.map.name} {scene.map.nrows}x{scene.map.ncols}, "
        f"{len(scene.lane.segments)} lane segments, {scene_model.ngeom} geoms"
    )
    print(
        f"         textures: {type(scene.texture_provider).__name__} "
        f"(valid for vision: {scene.texture_provider.valid_for_vision})"
    )
    print(f"         loop length through segment 0: {scene.lane.cycle_length(0):.3f} m")
    if issues:
        print("         map consistency warnings:")
        for issue in issues:
            print(f"           {issue}")
    return 0


def cmd_sysid(args: argparse.Namespace) -> int:
    """Run the two-stage physics matching.

    Args:
        args: parsed arguments carrying ``reference``, ``out``, ``outer`` and ``max_iter``.

    Returns:
        0 when the S8.2 acceptance criteria hold, 2 otherwise.
    """
    cfg = _mjcf.MjcfCfg.from_shared(include_camera=False)
    if args.reference:
        trajectories, metadata = _sysid.load_reference(args.reference)
        print(
            f"reference: {args.reference} ({metadata.get('source')}, "
            f"{len(trajectories)} programs, control {metadata.get('control_hz')} Hz)"
        )
        if metadata.get("physics_dt") and abs(float(metadata["physics_dt"]) - cfg.sim.physics_dt) > 1e-12:
            print(
                f"  WARNING: the reference was produced at physics_dt "
                f"{metadata['physics_dt']} but MuJoCo runs {cfg.sim.physics_dt}. That is critic "
                f"item J's integration-rate confound."
            )
        result, _model = _sysid.run_sysid(
            trajectories,
            control_hz=cfg.sim.control_hz,
            cfg=cfg,
            outer_passes=args.outer,
            max_iter=args.max_iter,
            verbose=args.verbose,
        )
    else:
        print(
            "no --reference given: running the synthetic self-test (perturb a known model, "
            "then try to recover it). This validates the procedure, not the Isaac gap."
        )
        print(f"perturbation: {_sysid.SELFTEST_PERTURBATION}")
        result, _model = _sysid.run_selftest(
            cfg,
            outer_passes=args.outer,
            max_iter=args.max_iter,
            verbose=args.verbose,
        )
    print(result.report())
    if args.out:
        print(f"written: {result.save(args.out)}")
    return 0 if result.accepted else 2


def cmd_eval(args: argparse.Namespace) -> int:
    """Run the MuJoCo half of the S8.4 evaluation matrix.

    Args:
        args: parsed arguments.

    Returns:
        0 on success.
    """
    policy = _evaluate.PolicySpec(
        kind="torchscript" if args.policy else args.scripted,
        path=args.policy or "",
        device=args.device,
    )
    base = MjEnvCfg(
        map=args.map if args.map else _track.LOOP_5X5,
        asset_dir=args.asset_dir,
        obs_mode=args.obs_mode,
        episode_length_s=args.episode_seconds,
    )
    report = _evaluate.run_matrix(
        policy,
        base_cfg=base,
        conditions=args.conditions,
        seeds=tuple(args.seeds),
        episodes_per_seed=args.episodes,
        max_seconds=args.episode_seconds,
        workers=args.workers,
    )
    print(report.report())
    if args.out:
        print(f"written: {report.save(args.out)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="eval_sim2sim.py",
        description="MuJoCo sim-to-sim harness for the Duckiebot lane-following policy.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check-env", help="report what this interpreter can run").set_defaults(func=cmd_check_env)

    build = sub.add_parser("build", help="generate and compile the MJCF and a track scene")
    build.add_argument("--out", default="build/sim2sim", help="output directory")
    build.add_argument("--map", default="", help="MapFormat1 YAML path; omit for the built-in loop")
    build.set_defaults(func=cmd_build)

    sysid = sub.add_parser("sysid", help="run the two-stage Isaac-to-MuJoCo physics matching")
    sysid.add_argument("--reference", default="", help="reference JSON from the Isaac side")
    sysid.add_argument("--out", default="", help="write the result JSON here")
    sysid.add_argument("--outer", type=int, default=3, help="stage1/stage2 alternations")
    sysid.add_argument("--max-iter", dest="max_iter", type=int, default=50, help="LM iteration cap")
    sysid.add_argument("--verbose", action="store_true", help="print the LM iteration log")
    sysid.set_defaults(func=cmd_sysid)

    evaluate = sub.add_parser("eval", help="run conditions C5 and C6")
    evaluate.add_argument("--policy", default="", help="TorchScript policy artifact")
    evaluate.add_argument(
        "--scripted",
        default="zero",
        choices=("zero", "constant"),
        help="built-in policy used when --policy is not given (smoke tests only)",
    )
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--map", default="", help="MapFormat1 YAML path")
    evaluate.add_argument("--asset-dir", dest="asset_dir", default="build/sim2sim")
    evaluate.add_argument("--obs-mode", dest="obs_mode", default="vec", choices=("rgb_vec", "vec", "none"))
    evaluate.add_argument("--conditions", nargs="+", default=["C5", "C6"])
    evaluate.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    evaluate.add_argument("--episodes", type=int, default=200, help="episodes per seed")
    evaluate.add_argument("--episode-seconds", dest="episode_seconds", type=float, default=45.0)
    evaluate.add_argument("--workers", type=int, default=None)
    evaluate.add_argument("--out", default="", help="write the report JSON here")
    evaluate.set_defaults(func=cmd_eval)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch.

    Args:
        argv: argument list; None uses ``sys.argv[1:]``.

    Returns:
        The subcommand's exit code.
    """
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
