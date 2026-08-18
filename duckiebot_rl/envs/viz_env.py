"""``make_viz_env``: a single-environment, rendering Isaac adapter for the live viewer.

:mod:`duckiebot_rl.viz.backends` drives every simulator through one small contract, so the viewer
does not care whether it is looking at MuJoCo or Isaac Sim. This module is the Isaac side of that
contract: a thin adapter over :class:`~duckiebot_rl.envs.lane_follow_env.DuckiebotLaneFollowEnv`
that presents the scalar, single-environment interface the viewer expects.

The adapter exists because the training environment is deliberately vectorized and GPU-resident:
``step`` takes a ``(num_envs, act_dim)`` tensor and returns tensors, while the viewer works one
episode at a time with numpy. Rather than complicate the training env with a special case, the
squeeze lives here.

Booting Kit is the caller's job. Isaac Sim can only be configured before ``SimulationApp`` starts,
so ``scripts/live_view.py`` launches the app (``AppLauncher``) and only then asks for the backend.
Calling :func:`make_viz_env` without a live Kit raises with an actionable message instead of
crashing somewhere inside USD.

Memory note, learned the hard way on the reference machine: the RTX rendering pipeline needs a
large host allocation while it compiles compute pipelines. With under roughly 6 GB of free
Windows commit it either dies with ``bad allocation`` inside ``vkCreateComputePipelines`` or
thrashes indefinitely. That is host memory, not VRAM, and it happens before any environment is
created, so lowering ``num_envs`` does not help. See ``docs/live_view.md``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

__all__ = ["IsaacVizEnv", "make_viz_env"]


def _require_kit() -> None:
    """Fail early and clearly when Kit is not running.

    Raises:
        RuntimeError: If ``SimulationApp`` has not been started yet.
    """
    # omni.kit.app is only importable once Kit has bootstrapped, and get_app().is_running() is
    # the accessor SimulationApp.is_running itself uses (simulation_app.py:269, :861). Do NOT
    # probe SimulationApp._instance: no such attribute exists, so the check always reports "not
    # running" even with a perfectly healthy Kit.
    try:
        import omni.kit.app
    except ImportError as exc:  # pragma: no cover - needs Isaac Sim
        raise RuntimeError(
            "Isaac Kit is not running, so 'omni.kit.app' is not importable. The app has to be "
            "launched BEFORE the environment is built, because Isaac Sim can only be configured "
            "pre-start. Run the viewer through scripts/live_view.py, which calls AppLauncher "
            "first, and make sure you are using the Isaac venv:\n"
            "  d:/Personal/personal/wheeled_quadruped_robot/.venv/Scripts/python.exe"
        ) from exc
    app = omni.kit.app.get_app()
    if app is None or not app.is_running():  # pragma: no cover - needs Isaac Sim
        raise RuntimeError(
            "Isaac Kit was imported but is not running. Launch it with AppLauncher before "
            "building the environment."
        )


def _variant_index(name: str) -> int:
    """Parse a ``city_<N>`` or ``eval_<N>`` name into a variant index.

    Args:
        name: The map name the viewer asked for.

    Returns:
        The parsed index, or 0 when the name carries no parsable index. Zero is the correct
        default: a single-environment scene is env 0, which ``index % len`` maps to variant 0.
    """
    tail = str(name).rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else 0


class IsaacVizEnv:
    """Single-environment, numpy-facing view of the vectorized Isaac lane-following env.

    Attributes:
        env: The wrapped :class:`DuckiebotLaneFollowEnv`.
        requested_map: The map name the viewer asked for, recorded for reporting.
    """

    requested_map: str = ""

    def __init__(self, env: Any) -> None:
        """Wrap a constructed environment.

        Args:
            env: A ``DuckiebotLaneFollowEnv`` built with ``num_envs=1``.
        """
        self.env = env
        self._chase: Any = None
        self._chase_failed = False

    @property
    def control_dt(self) -> float:
        """Seconds of simulated time per :meth:`step`.

        Returns:
            The environment's control period, which is ``physics_dt * decimation``.
        """
        return float(self.env.step_dt)

    def _to_numpy(self, obs: Any) -> dict[str, np.ndarray]:
        """Convert one environment's observation to numpy, dropping the batch axis.

        Args:
            obs: The observation dict returned by the vectorized environment.

        Returns:
            The same keys, as numpy arrays with the leading env axis removed.
        """
        source = obs.get("policy", obs) if isinstance(obs, dict) else obs
        out: dict[str, np.ndarray] = {}
        for key, value in source.items():
            array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
            out[key] = array[0] if array.ndim > 0 and array.shape[0] == 1 else array
        return out

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        """Reset the environment.

        Args:
            seed: Episode seed, forwarded to the environment.

        Returns:
            The first observation, batch axis removed.
        """
        obs, _ = self.env.reset(seed=seed)
        return self._to_numpy(obs)

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Advance one control step.

        Args:
            action: A ``(act_dim,)`` action for the single environment.

        Returns:
            ``(obs, reward, terminated, truncated, info)`` with scalars rather than tensors.
        """
        import torch

        tensor = torch.as_tensor(np.asarray(action, dtype=np.float32), device=self.env.device)
        obs, reward, terminated, truncated, info = self.env.step(tensor.reshape(1, -1))
        extras: dict[str, Any] = dict(info) if isinstance(info, dict) else {}
        # The viewer reports lane deviation from info["d"], and DirectRLEnv does not populate
        # extras["log"] the way the manager-based workflow does, so surface the signed lane
        # offset the env already tracks. Without this the RMS is reported as a flat 0.0000.
        offset = getattr(self.env, "_d", None)
        if offset is not None:
            extras.setdefault("d", float(offset[0].item()))
        return (
            self._to_numpy(obs),
            float(reward[0].item()),
            bool(terminated[0].item()),
            bool(truncated[0].item()),
            extras,
        )

    def _pov_frame(self) -> np.ndarray | None:
        """Return the robot's onboard camera frame.

        Returns:
            An ``(H, W, 3)`` uint8 frame, or None when no camera is present.
        """
        camera = self.env.scene.sensors.get("camera") if hasattr(self.env, "scene") else None
        if camera is None:
            return None
        rgb = camera.data.output.get("rgb")
        if rgb is None:
            return None
        frame = rgb[0].detach().cpu().numpy()
        if frame.dtype != np.uint8:
            frame = (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
        return frame[..., :3]

    def _ensure_chase_camera(self) -> Any:
        """Create the third-person chase camera on first use.

        A plain ``isaacsim.sensors.camera.Camera`` alongside the Isaac Lab scene: its render
        product updates on the same ``sim.render()`` the env already performs each control step,
        so the extra cost is one more 640x360 render, not a second pipeline.

        Returns:
            The camera, or None when it could not be created (recorded once, then silent).
        """
        if self._chase is not None or self._chase_failed:
            return self._chase
        try:
            from isaacsim.core.utils.extensions import enable_extension

            # the camera sensor ships as an extension and is not on sys.path until enabled
            enable_extension("isaacsim.sensors.camera")
            from isaacsim.sensors.camera import Camera

            self._chase = Camera(
                prim_path="/World/chase_cam",
                resolution=(640, 360),
                position=np.array([0.0, 0.0, 1.0]),
            )
            self._chase.initialize()
        except Exception as exc:
            print(f"[viz_env] chase camera unavailable, falling back to onboard view: {exc!r}")
            self._chase_failed = True
            self._chase = None
        return self._chase

    def _update_chase_pose(self) -> None:
        """Place the chase camera behind and above the robot, looking down at it."""
        import torch  # noqa: F401 - ensures torch is initialised before reading device tensors

        data = self.env._robot.data
        pos = data.root_pos_w[0].detach().cpu().numpy()
        w, x, y, z = (float(v) for v in data.root_quat_w[0].detach().cpu())
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        # High and steep enough to clear the city's perimeter walls; a low chase camera ends
        # up inside them and streams a gray frame (measured, not guessed).
        back, height, pitch = 0.72, 0.55, np.deg2rad(33.0)
        cam_pos = np.array([pos[0] - back * np.cos(yaw), pos[1] - back * np.sin(yaw), pos[2] + height])
        # world camera axes: +X forward, +Z up. q = qz(yaw) * qy(pitch) tilts the forward axis
        # down toward the robot. wxyz order.
        cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
        cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
        # Hamilton product qz(yaw) x qy(pitch), wxyz: [cy cp, -sy sp, cy sp, sy cp]. The y
        # component is cy*sp; writing cp*sp only matches at yaw 0 and mis-rolls the camera in
        # every turn (measured as the ground plane swallowing the frame).
        quat = np.array([cy * cp, -sy * sp, cy * sp, sy * cp])
        self._chase.set_world_pose(cam_pos, quat, camera_axes="world")

    def render_frame(self) -> np.ndarray | None:
        """Return the current view for recording and live streaming.

        The third-person chase view showing the robot in the 3D city, falling back to the
        onboard camera when the chase camera is unavailable.

        Returns:
            An ``(H, W, 3)`` uint8 frame, or None when no camera works at all.
        """
        chase = self._ensure_chase_camera()
        if chase is None:
            return self._pov_frame()
        try:
            self._update_chase_pose()
            rgba = chase.get_rgba()
            if rgba is None or getattr(rgba, "size", 0) == 0:
                return self._pov_frame()
            frame = np.asarray(rgba)
            if frame.dtype != np.uint8:
                frame = (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
            return frame[..., :3]
        except Exception as exc:
            print(f"[viz_env] chase render failed, using onboard view: {exc!r}")
            self._chase_failed = True
            self._chase = None
            return self._pov_frame()

    def close(self) -> None:
        """Close the wrapped environment."""
        self.env.close()


_VISUAL_COLORS: dict[str, tuple[float, float, float]] = {
    "chassis": (0.72, 0.07, 0.05),
    "computer": (0.10, 0.12, 0.16),
    "camera": (0.06, 0.06, 0.07),
    "left_wheel": (0.05, 0.05, 0.05),
    "right_wheel": (0.05, 0.05, 0.05),
}
"""Duckiebot colors: red chassis, dark electronics, black tires. The split OBJs
carry no material, so the converter leaves them default gray, which on a gray road makes the
robot near-invisible (measured, chase frames)."""


def _bind_color(stage: Any, prim: Any, rgb: tuple[float, float, float]) -> None:
    """Bind a simple colored material to a visual subtree.

    Args:
        stage: The USD stage.
        prim: The Xform holding the referenced mesh.
        rgb: Diffuse color.
    """
    from pxr import Gf, Sdf, UsdShade

    path = prim.GetPath().AppendChild("mat")
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path.AppendChild("prev"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        material, bindingStrength=UsdShade.Tokens.strongerThanDescendants
    )


_REAL_VISUALS: tuple[tuple[str, str, tuple[float, float, float]], ...] = (
    ("base_link", "chassis", (0.0, 0.0, 0.0)),
    ("base_link", "computer", (0.0, 0.0, 0.0)),
    ("base_link", "camera", (0.0, 0.0, 0.0)),
    ("left_wheel_link", "left_wheel", (0.0, -0.05, 0.0)),
    ("right_wheel_link", "right_wheel", (0.0, 0.05, 0.0)),
)
"""Real-mesh visual attachments: (link, mesh stem, translation). The wheel translations undo the
joint offsets because the meshes are authored at their assembled pose (0.100 m baseline)."""


def _find_visual_mesh_dir() -> Any:
    """Locate the fetched real-robot meshes, if the user has them locally.

    The meshes derive from Duckietown's CAD via gym-duckietown and are NOT redistributable, so
    they are never committed: ``scripts/fetch_visual_mesh.py`` places them under ``_refs/`` and
    the research prototypes are the on-disk fallback. No meshes means primitive visuals, which
    is always correct for training; this path only upgrades the view.

    Returns:
        The directory holding ``chassis.obj`` and friends, or None.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    for candidate in (
        root / "_refs" / "visual_mesh",
        root / "_research" / "prototypes" / "db2" / "meshes",
    ):
        if (candidate / "chassis.obj").is_file():
            return candidate
    return None


def _attach_real_visuals(env: Any) -> None:
    """Swap the primitive robot visuals for the real Duckiebot meshes, viewer-side only.

    Physics, collisions and the camera are untouched: the meshes are added as pure visual
    references and the primitive visual scopes are hidden. Every failure falls back to the
    primitive look, because the view must never take the session down.

    Args:
        env: The constructed ``DuckiebotLaneFollowEnv``.
    """
    mesh_dir = _find_visual_mesh_dir()
    if mesh_dir is None:
        return
    try:
        import asyncio

        import omni.kit.asset_converter as asset_converter
        import omni.usd
        from isaacsim.core.utils.extensions import enable_extension
        from pxr import Gf, Sdf, UsdGeom

        enable_extension("omni.kit.asset_converter")

        usd_dir = mesh_dir.parent / "visual_usd"
        usd_dir.mkdir(parents=True, exist_ok=True)

        async def _convert(src: Any, dst: Any) -> bool:
            task = asset_converter.get_instance().create_converter_task(str(src), str(dst), None)
            return await task.wait_until_finished()

        loop = asyncio.get_event_loop()
        converted: dict[str, Any] = {}
        for _, stem, _ in _REAL_VISUALS:
            src, dst = mesh_dir / f"{stem}.obj", usd_dir / f"{stem}.usd"
            if not dst.is_file() and not loop.run_until_complete(_convert(src, dst)):
                print(f"[viz_env] mesh convert failed for {stem}; keeping primitive visuals")
                return
            converted[stem] = dst

        from pxr import Usd

        expected_m = {
            "chassis": 0.200,
            "computer": 0.095,
            "camera": 0.040,
            "left_wheel": 0.066,
            "right_wheel": 0.066,
        }
        stage = omni.usd.get_context().get_stage()
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
        robot = "/World/envs/env_0/Robot"

        # The latest-generation robot: duckietown-world's duckiebot3 glTF (DB21, tall camera
        # mast, back plate), fetched from behind Git LFS by scripts/fetch_visual_mesh.py. When
        # present it replaces the whole per-part DB17 path below.
        db21 = mesh_dir.parent / "visual_mesh" / "db21" / "main.gltf"
        if not db21.is_file():
            from pathlib import Path as _P

            db21 = _P(__file__).resolve().parents[2] / "_refs" / "visual_mesh" / "db21" / "main.gltf"
        if db21.is_file():
            dst = db21.with_suffix(".usd")
            if not dst.is_file() and not loop.run_until_complete(_convert(db21, dst)):
                print("[viz_env] db21 convert failed; falling back to per-part meshes")
            else:
                xform = UsdGeom.Xform.Define(stage, f"{robot}/base_link/real_db21")
                xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
                orient_op = xform.AddRotateXYZOp()
                orient_op.Set(Gf.Vec3f(0.0))
                scale_op = xform.AddScaleOp()
                scale_op.Set(Gf.Vec3f(1.0))
                xform.GetPrim().GetReferences().AddReference(str(dst))
                bbox_cache.Clear()
                size = bbox_cache.ComputeWorldBound(xform.GetPrim()).ComputeAlignedRange().GetSize()
                dims = f"{size[0]:.4f} x {size[1]:.4f} x {size[2]:.4f}"
                print(f"[viz_env] db21: bbox {dims} m")
                # glTF is Y-up; if the mast landed along Y instead of Z the model is lying down.
                if size[1] > 1.5 * max(size[2], 1e-9):
                    orient_op.Set(Gf.Vec3f(90.0, 0.0, 0.0))
                    print("[viz_env] db21: applied +90 deg X (Y-up correction)")
                longest = max(size[0], size[1], size[2], 1e-9)
                factor = 0.20 / longest
                if not 0.5 < factor < 2.0:
                    scale_op.Set(Gf.Vec3f(float(factor)))
                    print(f"[viz_env] db21: scale {factor:.4g}")
                for link in ("base_link", "left_wheel_link", "right_wheel_link", "caster_link"):
                    visuals = stage.GetPrimAtPath(f"{robot}/{link}/visuals")
                    if visuals and visuals.IsValid():
                        UsdGeom.Imageable(visuals).MakeInvisible()
                marker = stage.GetPrimAtPath(f"{robot}/base_link/marker_visual")
                if marker and marker.IsValid():
                    UsdGeom.Imageable(marker).MakeInvisible()
                print("[viz_env] DB21 (duckiebot3) visual attached")
                return

        for link, stem, offset in _REAL_VISUALS:
            xform = UsdGeom.Xform.Define(stage, f"{robot}/{link}/real_{stem}")
            xform.AddTranslateOp().Set(Gf.Vec3d(*offset))
            scale_op = xform.AddScaleOp()
            scale_op.Set(Gf.Vec3f(1.0))
            xform.GetPrim().GetReferences().AddReference(str(converted[stem]))
            # OBJ carries no unit metadata, so the converter guesses; measure and normalise to
            # the robot's known dimensions instead of trusting the guess.
            bbox_cache.Clear()
            size = bbox_cache.ComputeWorldBound(xform.GetPrim()).ComputeAlignedRange().GetSize()
            longest = max(size[0], size[1], size[2], 1e-9)
            factor = expected_m[stem] / longest
            dims = f"{size[0]:.4f} x {size[1]:.4f} x {size[2]:.4f}"
            print(f"[viz_env] {stem}: bbox {dims} m -> scale {factor:.4g}")
            if not 0.5 < factor < 2.0:
                scale_op.Set(Gf.Vec3f(float(factor)))
            _bind_color(stage, xform.GetPrim(), _VISUAL_COLORS.get(stem, (0.08, 0.08, 0.09)))
        for link in ("base_link", "left_wheel_link", "right_wheel_link", "caster_link"):
            visuals = stage.GetPrimAtPath(f"{robot}/{link}/visuals")
            if visuals and visuals.IsValid():
                UsdGeom.Imageable(visuals).MakeInvisible()
        # the yellow marker sphere lives outside visuals/ on some builds; hide it too if present
        marker = stage.GetPrimAtPath(f"{robot}/base_link/marker_visual")
        if marker and marker.IsValid():
            UsdGeom.Imageable(marker).MakeInvisible()
        _ = Sdf  # imported for side effect parity across Kit versions
        print(f"[viz_env] real Duckiebot meshes attached from {mesh_dir}")
    except Exception as exc:
        print(f"[viz_env] real visuals unavailable, keeping primitives: {exc!r}")


def make_viz_env(
    map: str = "loop_small",
    num_envs: int = 1,
    device: str = "cuda:0",
    render: bool = True,
    dynamics_dr: bool = False,
    photometric_dr: bool = False,
    episode_length_s: float | None = None,
    seed: int = 0,
    **overrides: Any,
) -> IsaacVizEnv:
    """Build the single-environment Isaac backend the live viewer drives.

    Args:
        map: Accepted for the viewer's backend contract, which is shared with MuJoCo. The Isaac
            scene does NOT take a named map: ``CitySettings`` generates ``num_variants`` layouts
            and ``MultiUsdFileCfg`` assigns them per environment by ``index % len`` (SPEC v2 S7.1,
            critic item D). A single-environment viewer is therefore always env 0, which always
            gets variant 0. The name is recorded on the returned adapter as
            :attr:`IsaacVizEnv.requested_map` so the viewer can report it, and ``city_<N>`` or
            ``eval_<N>`` selects that variant index when it can be parsed.
        num_envs: Environment count. The viewer runs one episode at a time, so anything above 1
            is wasted work and is clamped to 1.
        device: Torch device for the simulation.
        render: Whether cameras and rendering are enabled. False gives physics only, which is
            useful when the host cannot afford the RTX pipeline (see the module docstring).
        dynamics_dr: Apply the S7.3 dynamics randomisation, which is sim-to-sim condition C6.
        photometric_dr: Apply the visual randomisation. Named for the viewer's shared flag; it
            maps onto ``LaneFollowSettings.visual_dr``.
        episode_length_s: Truncation horizon in simulated seconds, or None to keep the default.
        seed: Environment seed.
        **overrides: Extra fields forwarded to
            :func:`~duckiebot_rl.envs.env_cfg.lane_follow_env_cfg`.

    Returns:
        An :class:`IsaacVizEnv` satisfying the viewer's backend contract.

    Raises:
        RuntimeError: If Isaac Sim is unavailable or Kit has not been launched.
    """
    _require_kit()

    from duckiebot_rl.envs.env_cfg import CitySettings, LaneFollowSettings, lane_follow_env_cfg
    from duckiebot_rl.envs.lane_follow_env import DuckiebotLaneFollowEnv

    city = CitySettings(num_variants=max(1, _variant_index(map) + 1))
    settings = LaneFollowSettings(
        num_envs=1 if num_envs is None else max(1, int(num_envs)),
        device=str(device),
        seed=int(seed),
        city=city,
        visual_dr=bool(photometric_dr),
        dynamics_dr=bool(dynamics_dr),
    )
    if episode_length_s is not None:
        settings = dataclasses.replace(
            settings, rates=dataclasses.replace(settings.rates, episode_length_s=float(episode_length_s))
        )
    cfg = lane_follow_env_cfg(settings, **overrides)
    env = DuckiebotLaneFollowEnv(cfg, render_mode="rgb_array" if render else None)
    _attach_real_visuals(env)
    adapter = IsaacVizEnv(env)
    adapter.requested_map = str(map)
    return adapter
