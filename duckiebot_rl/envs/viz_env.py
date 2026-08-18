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
import re
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "DB21J_SOURCES",
    "DB21_OBJ_EXPECTED_SIZE_M",
    "DB21_OBJ_ROTATE_XYZ",
    "DEFAULT_ROBOT_MESH",
    "ROBOT_MESH_CHOICES",
    "IsaacVizEnv",
    "attach_robot_visuals",
    "duckiematrix_point_to_base_link",
    "make_viz_env",
    "resolve_city_selection",
    "resolve_db21j_source",
    "resolve_robot_mesh",
    "robot_prim_path",
    "stage_stem",
]


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


#: Stage-stem families the generator emits, and the zero padding each one uses.
_STEM_WIDTHS: dict[str, int] = {"city": 3, "eval": 2}

#: ``<family>_<digits>``, the shape of every generated stage stem.
_STEM_RE = re.compile(r"^(city|eval)_(\d+)$")


def stage_stem(name: str) -> str:
    """Normalise a requested map name to the stem ``scripts/build_city.py`` writes.

    The generator zero-pads: training layouts are ``city_000 .. city_063`` and the held-out ones
    ``eval_00 .. eval_03``. A human types ``--map city_7``, so the padding is restored here
    rather than making the caller count zeros. A name that is not a ``city_``/``eval_`` name is
    returned unchanged, which is what lets a hand-built stage be named directly.

    Args:
        name: The map name the viewer asked for.

    Returns:
        The stage stem, with no directory and no suffix.
    """
    match = _STEM_RE.match(str(name).strip())
    if match is None:
        return str(name).strip()
    family, digits = match.group(1), match.group(2)
    return f"{family}_{int(digits):0{_STEM_WIDTHS[family]}d}"


def _selection_from_path(name: str) -> tuple[str, str] | None:
    """Interpret a map name that is a path to a generated map YAML or USD stage.

    ``scripts/build_city.py --out DIR`` writes ``DIR/maps/<stem>.yaml`` and
    ``DIR/usd/<stem>.usda``, so either file identifies both the build root and the layout. This
    is how a layout from a non-default build root, such as a hard set under ``build/city_hard``,
    is selected without inventing a second command-line flag.

    Args:
        name: The map name the viewer asked for.

    Returns:
        ``(root, stem)`` if ``name`` points at an existing map YAML or USD stage, else ``None``.
    """
    path = Path(str(name).strip())
    if path.suffix.lower() not in (".yaml", ".yml", ".usda", ".usd") or not path.is_file():
        return None
    parent = path.parent
    root = parent.parent if parent.name in ("maps", "usd") else parent
    return root.as_posix(), path.stem


def _builtin_variant_slot(name: str, city_root: str | None) -> str | None:
    """Find which generated variant, if any, carries a built-in map's layout.

    ``variant_maps`` puts the hand-authored built-ins at variants 0 to 4 in
    ``BUILTIN_MAP_NAMES`` order, so ``zigzag`` normally lives at ``city_002``. That is a property
    of the nominal generator, not a guarantee: a build made with a non-nominal difficulty drops
    the built-ins entirely, and ``city_002`` there is an unrelated random loop. The claim is
    therefore checked rather than assumed, by comparing layout signatures with the map YAML that
    was actually written.

    Args:
        name: A name from ``BUILTIN_MAP_NAMES``.
        city_root: Build root to look in, or ``None`` for the default search roots.

    Returns:
        The matching stage stem, or ``None`` if this build has no such variant.
    """
    from duckiebot_rl.city.maps import BUILTIN_MAP_NAMES, builtin_map, load_map
    from duckiebot_rl.envs.env_cfg import CitySettings, resolve_city_assets

    if name not in BUILTIN_MAP_NAMES:
        return None
    stem = f"city_{BUILTIN_MAP_NAMES.index(name):03d}"
    try:
        _, map_paths, _ = resolve_city_assets(
            CitySettings(root=city_root, num_variants=1, variant_names=(stem,))
        )
        on_disk = load_map(map_paths[0])
    except (FileNotFoundError, OSError):
        return None
    return stem if on_disk.layout_signature() == builtin_map(name).layout_signature() else None


def resolve_city_selection(name: str) -> Any:
    """Build the ``CitySettings`` that puts the requested layout in front of the viewer.

    The bug this exists to fix: the viewer used to keep the generated ``city_000 .. city_<N>``
    list and merely grow it, but ``MultiUsdFileCfg`` assigns assets to environments by
    ``index % len`` and a viewer scene has exactly one environment. Env 0 therefore always got
    ``usd_paths[0]``, which is always ``city_000``, so ``--map city_37`` silently showed
    ``city_000``. The fix is to make the requested layout the only entry in the list.

    Args:
        name: The map name the viewer asked for: ``city_<N>``, ``eval_<N>``, a bare stage stem,
            or a path to a generated map YAML or USD stage.

    Returns:
        A ``CitySettings`` selecting exactly that layout, or ``CitySettings(num_variants=1)``
        when the name matches nothing on disk. Falling back is deliberate: the viewer is a
        debugging tool, and refusing to start over a mistyped map name is worse than showing
        ``city_000`` and saying so on stdout.
    """
    from duckiebot_rl.envs.env_cfg import CitySettings, resolve_city_assets

    from_path = _selection_from_path(name)
    root = from_path[0] if from_path is not None else None
    stem = from_path[1] if from_path is not None else stage_stem(name)

    wanted = CitySettings(root=root, num_variants=1, variant_names=(stem,))
    try:
        usd_paths, _, _ = resolve_city_assets(wanted)
    except FileNotFoundError:
        slot = _builtin_variant_slot(stem, root)
        if slot is None:
            print(
                f"[viz_env] no generated stage matches --map {name!r}; falling back to city_000. "
                f"Build the stages with: python scripts/build_city.py --all --out build/city"
            )
            return CitySettings(root=root, num_variants=1)
        wanted = CitySettings(root=root, num_variants=1, variant_names=(slot,))
        usd_paths, _, _ = resolve_city_assets(wanted)
        print(f"[viz_env] built-in map {name!r} is variant {slot!r} in this build")

    where = wanted.root or "the default city search roots"
    print(f"[viz_env] map {name!r} -> stage {wanted.variant_names[0]!r} under {where}")
    print(f"[viz_env] env 0 stage file: {usd_paths[0]}")
    return wanted


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
            # The sensor's default near plane clips everything within ~1 m, which is the whole
            # robot (0.88 m away) and the near ground: the bottom third of every frame came out
            # as the uniform gray background (measured). Pull the near plane in.
            self._chase.set_clipping_range(0.02, 1.0e6)
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


def _define_color_material(stage: Any, path: Any, rgb: tuple[float, float, float]) -> Any:
    """Author one flat ``UsdPreviewSurface`` material.

    Args:
        stage: The USD stage to author on.
        path: ``Sdf.Path`` the material is defined at.
        rgb: Diffuse color.

    Returns:
        The ``UsdShade.Material``.
    """
    from pxr import Gf, Sdf, UsdShade

    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path.AppendChild("prev"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _bind_material(prim: Any, material: Any, strength: str | None = None) -> None:
    """Bind an existing material to a prim.

    Args:
        prim: The prim to bind on.
        material: The ``UsdShade.Material`` to bind.
        strength: Binding strength token, or None for ``strongerThanDescendants``. A body colour
            bound at the root of a subtree that also carries per-part bindings has to be
            ``weakerThanDescendants``, or it would win over the parts.
    """
    from pxr import UsdShade

    token = UsdShade.Tokens.strongerThanDescendants if strength is None else strength
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, bindingStrength=token)


def _bind_color(stage: Any, prim: Any, rgb: tuple[float, float, float]) -> None:
    """Bind a simple colored material to a visual subtree.

    Args:
        stage: The USD stage.
        prim: The Xform holding the referenced mesh.
        rgb: Diffuse color.
    """
    _bind_material(prim, _define_color_material(stage, prim.GetPath().AppendChild("mat"), rgb))


_REAL_VISUALS: tuple[tuple[str, str, tuple[float, float, float]], ...] = (
    ("base_link", "chassis", (0.0, 0.0, 0.0)),
    ("base_link", "computer", (0.0, 0.0, 0.0)),
    ("base_link", "camera", (0.0, 0.0, 0.0)),
    ("left_wheel_link", "left_wheel", (0.0, -0.05, 0.0)),
    ("right_wheel_link", "right_wheel", (0.0, 0.05, 0.0)),
)
"""Real-mesh visual attachments: (link, mesh stem, translation). The wheel translations undo the
joint offsets because the meshes are authored at their assembled pose (0.100 m baseline)."""


ROBOT_MESH_CHOICES: tuple[str, ...] = ("db21j", "db17", "primitive")
"""Robot visual-mesh selectors. Viewer AND trainer side; never physics.

``db21j``
    The latest-generation DB21. The preferred source is ``_refs/visual_mesh/db21j/main.obj``,
    the geometry extracted from Duckietown's own Duckiematrix engine; the older
    ``_refs/visual_mesh/db21/main.gltf`` is the fallback and is DB18-era, not DB21. See
    :data:`DB21J_SOURCES`.
``db17``
    The older per-part DB17 OBJs (``chassis``, ``computer``, ``camera``, two wheels), assembled
    and colored here because the OBJs carry no material.
``primitive``
    No attachment at all: the boxes and cylinders the environment itself builds.

Physics, collisions, actuation and the policy camera are identical under all three. Only what is
drawn changes.
"""

DEFAULT_ROBOT_MESH = "db21j"
"""Default selector: the latest-generation robot when its mesh has been fetched."""

_ROBOT_PRIM = "/World/envs/env_0/Robot"
"""Prim path of environment 0's robot, the one the chase camera follows."""


def robot_prim_path(index: int = 0) -> str:
    """Return the USD prim path of one environment's robot.

    Isaac Lab clones the scene under ``/World/envs/env_<i>``, so every robot is addressable from
    the stage alone; that is what lets one attachment pass cover a whole training scene.

    Args:
        index: Environment index.

    Returns:
        The prim path, for example ``/World/envs/env_7/Robot``.
    """
    return f"/World/envs/env_{int(index)}/Robot"


_FETCH_HINT = (
    "  fetch it with the Isaac venv python:\n"
    "    d:/Personal/personal/wheeled_quadruped_robot/.venv/Scripts/python.exe "
    "scripts/fetch_visual_mesh.py"
)
"""How to obtain the meshes, which are not redistributable and so are never committed."""


def _find_visual_mesh_dir() -> Any:
    """Locate the fetched per-part DB17 meshes, if the user has them locally.

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


def _find_db21j_obj() -> Any:
    """Locate the Duckiematrix DB21 extraction, if the user has it locally.

    This is the real latest-generation robot: ``scripts/fetch_visual_mesh.py`` pulls Duckietown's
    Duckiematrix engine build and exports the ``DB21`` GameObject hierarchy out of it, writing
    ``_refs/visual_mesh/db21j/main.obj``. Metres, Y-up, group hierarchy preserved, no materials.

    Returns:
        The path to ``main.obj``, or None.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    candidate = root / "_refs" / "visual_mesh" / "db21j" / "main.obj"
    return candidate if candidate.is_file() else None


def _find_db21j_gltf() -> Any:
    """Locate the older duckietown-world ``duckiebot3`` glTF, if the user has it locally.

    Named ``duckiebot3`` upstream, but the file is export_DB18's asset: it is the DB18-era robot,
    NOT a DB21. It stays as the second-choice source because it is one self-contained file that
    many users already have, and it still looks far more like a Duckiebot than five boxes do.

    Deliberately independent of :func:`_find_visual_mesh_dir` and :func:`_find_db21j_obj`.

    Returns:
        The path to ``main.gltf``, or None.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    for candidate in (
        root / "_refs" / "visual_mesh" / "db21" / "main.gltf",
        root / "_research" / "prototypes" / "db2" / "visual_mesh" / "db21" / "main.gltf",
    ):
        if candidate.is_file():
            return candidate
    return None


DB21J_SOURCES: tuple[tuple[str, str], ...] = (
    ("duckiematrix-obj", "_refs/visual_mesh/db21j/main.obj"),
    ("db18-gltf", "_refs/visual_mesh/db21/main.gltf"),
)
"""The ``db21j`` sources in preference order, as ``(kind, documented location)``.

The Duckiematrix OBJ wins because it is the only genuine DB21 of the two: duckietown-world's
``duckiebot3/main.gltf`` is export_DB18's model sitting under a misleading directory name.
"""


def resolve_db21j_source() -> Any:
    """Pick which ``db21j`` source file to use.

    Returns:
        ``(kind, path)`` with ``kind`` one of the keys of :data:`DB21J_SOURCES`, or None when
        neither source is on disk.
    """
    obj = _find_db21j_obj()
    if obj is not None:
        return ("duckiematrix-obj", obj)
    gltf = _find_db21j_gltf()
    if gltf is not None:
        return ("db18-gltf", gltf)
    return None


def resolve_robot_mesh(robot_mesh: str) -> str:
    """Validate a mesh selector and downgrade it to what is actually on disk.

    Nothing is imported from Isaac here, so the answer is the same inside and outside Kit and
    the decision can be logged before a single USD prim is touched.

    Args:
        robot_mesh: One of :data:`ROBOT_MESH_CHOICES`.

    Returns:
        The selector to act on: ``robot_mesh`` itself, or a fallback when the requested meshes
        are missing.

    Raises:
        ValueError: If ``robot_mesh`` is not one of :data:`ROBOT_MESH_CHOICES`.
    """
    choice = str(robot_mesh).lower().strip()
    if choice not in ROBOT_MESH_CHOICES:
        raise ValueError(
            f"unknown robot_mesh {robot_mesh!r}; expected one of {', '.join(ROBOT_MESH_CHOICES)}"
        )
    if choice == "primitive":
        return choice
    if choice == "db21j":
        if resolve_db21j_source() is not None:
            return "db21j"
        wanted = " or ".join(location for _kind, location in DB21J_SOURCES)
        print(f"[viz_env] robot mesh 'db21j' requested but neither {wanted} is present.\n" + _FETCH_HINT)
        choice = "db17"
        print("[viz_env] falling back to robot mesh 'db17' (the per-part meshes)")
    if _find_visual_mesh_dir() is None:
        print(
            "[viz_env] robot mesh 'db17' requested but no chassis.obj was found under "
            "_refs/visual_mesh or _research/prototypes/db2/meshes.\n" + _FETCH_HINT
        )
        print("[viz_env] falling back to robot mesh 'primitive'")
        return "primitive"
    return "db17"


def _hide_primitive_visuals(stage: Any, robot_prim: str = _ROBOT_PRIM) -> None:
    """Hide one robot's own primitive visuals so that only the attached mesh is drawn.

    Args:
        stage: The USD stage.
        robot_prim: Prim path of the robot, from :func:`robot_prim_path`.
    """
    from pxr import UsdGeom

    for link in ("base_link", "left_wheel_link", "right_wheel_link", "caster_link"):
        visuals = stage.GetPrimAtPath(f"{robot_prim}/{link}/visuals")
        if visuals and visuals.IsValid():
            UsdGeom.Imageable(visuals).MakeInvisible()
    # the yellow marker sphere lives outside visuals/ on some builds; hide it too if present
    marker = stage.GetPrimAtPath(f"{robot_prim}/base_link/marker_visual")
    if marker and marker.IsValid():
        UsdGeom.Imageable(marker).MakeInvisible()


# ------------------------------------------------------------- the Duckiematrix DB21 source

DB21_OBJ_ROTATE_XYZ: tuple[float, float, float] = (90.0, 0.0, 180.0)
"""Deterministic orientation for the Duckiematrix OBJ, as a USD ``RotateXYZ`` triple [deg].

The extraction is in Unity's frame: metres, +Y up, +Z the robot's LEFT, and the camera mast and
front bumper on the -X side. USD applies X, then Y, then Z, so ``(90, 0, 180)`` maps
``(x, y, z) -> (-x, z, y)``: the mast lands on +Z, the front on +X and the left wheel on +Y,
which is the Isaac convention.

A +90 about X alone is NOT enough. It would leave the robot facing -X with its left wheel on -Y,
which is this model yawed by 180 deg, so it would appear to drive backwards. Measured from the
source, not assumed: ``front_bumper`` and ``camera_mount`` span negative X, ``left_motor``
positive Z.

Nothing here is inferred at runtime. The lying-down heuristic used for the glTF could not fire
on this source anyway, its 0.134 m width and 0.122 m height being only 1.10x apart against a
1.5x threshold, which is exactly why this source gets a fixed transform instead.
"""

DB21_OBJ_EXPECTED_SIZE_M: tuple[float, float, float] = (0.2151, 0.1340, 0.1216)
"""Expected oriented bounding box, as (length, width, height) [m], measured from the source.

The 281,367 vertices span 0.215083 in X, 0.121565 in Y and 0.134002 in Z in the OBJ's own frame;
after :data:`DB21_OBJ_ROTATE_XYZ` the height is that 0.1216 m Y extent and the width the
0.1340 m Z extent. Printed beside the measured box at attach time, so an asset converter that
silently reoriented the source shows up as a number rather than as a robot lying on its side.
"""

_DB21_SIZE_TOL_M = 0.010
"""How far the measured oriented box may fall from :data:`DB21_OBJ_EXPECTED_SIZE_M` [m]."""

_DB21_WHEEL_HALF_WIDTH_M = 0.050
"""Meshes reaching past this on the lateral axis are wheels [m].

The geometric backstop behind the name rules, and the reason black wheels are guaranteed rather
than hoped for: whatever the asset converter does to the OBJ's group names, the wheels are the
only parts of this model that reach past 0.050 m off centre. Measured per group: the six wheel
meshes reach 0.062 to 0.067 m and the widest non-wheel part, the top plate, reaches 0.0447 m.
"""

_DB21_BODY_COLOR: tuple[float, float, float] = (0.16, 0.19, 0.26)
"""Dark blue-gray body colour. The extraction carries no materials at all, so without this the
robot renders default gray and is near-invisible against asphalt (measured, chase frames)."""

_DB21_PART_COLORS: tuple[tuple[tuple[str, ...], str, tuple[float, float, float]], ...] = (
    (("led_left_off", "led_right_off"), "led_off", (0.12, 0.12, 0.13)),
    (("led",), "led_on", (0.95, 0.88, 0.62)),
    (("wheel", "motor", "caster", "tyre", "tire"), "tyre", (0.045, 0.045, 0.050)),
    (("fish_eye", "camera"), "optics", (0.05, 0.05, 0.06)),
    (("jetson", "devkit"), "computer", (0.10, 0.11, 0.12)),
    (("bumper",), "bumper", (0.08, 0.08, 0.09)),
    (("screen",), "screen", (0.05, 0.09, 0.20)),
    (("tag",), "tag", (0.86, 0.86, 0.86)),
)
"""Flat colours keyed by substrings of the lowercased prim path, first match winning.

The OBJ preserves the Unity group hierarchy (``DB21/footprint/base/bottom_plate/...``), so when
the asset converter keeps those names the parts colour individually. When it does not, every
mesh falls through to :data:`_DB21_BODY_COLOR` except the wheels, which the geometric rule
catches regardless. The off-state LEDs come first because their names also contain ``led``.
"""


def duckiematrix_point_to_base_link(point: Any, base_link_height_m: float, ground_y_m: float) -> Any:
    """Map a point of the Duckiematrix extraction into the robot's ``base_link`` frame.

    The pure-python twin of what :data:`DB21_OBJ_ROTATE_XYZ` and the attach-time translation do
    in USD, so that the transform can be checked against the measured source without Kit.

    Args:
        point: ``(x, y, z)`` in the OBJ's own frame.
        base_link_height_m: Height of ``base_link`` above the ground, which is the wheel radius.
        ground_y_m: The source's lowest Y, the height at which its wheels touch the ground.

    Returns:
        ``(x, y, z)`` in ``base_link``, with +X forward, +Y left and +Z up.
    """
    x, y, z = (float(value) for value in point)
    return (-x, z, y - float(ground_y_m) - float(base_link_height_m))


def _lateral_axis(minimum: Any, maximum: Any) -> int:
    """Pick which axis of a measured box is the robot's left-right axis.

    The Duckiebot is symmetric across its long axis and across nothing else: the extraction spans
    -0.067 to +0.067 laterally, 0 to +0.122 vertically and -0.084 to +0.131 lengthwise. Choosing
    the most nearly centred axis therefore identifies the lateral one however the asset converter
    laid the source out, which comparing sizes could not: 0.134 m and 0.122 m are 10% apart.

    Args:
        minimum: Box minimum, indexable by axis.
        maximum: Box maximum, indexable by axis.

    Returns:
        0, 1 or 2.
    """

    def offset(axis: int) -> float:
        """Return how far off centre the box is on one axis, relative to its extent.

        Args:
            axis: Axis index.

        Returns:
            The normalised centre offset, 0.0 for a perfectly symmetric axis.
        """
        low, high = float(minimum[axis]), float(maximum[axis])
        return abs(low + high) / max(high - low, 1e-9)

    return min(range(3), key=offset)


def _db21_material_name(prim_path: str) -> str | None:
    """Return which named part colour a prim's path asks for, if any.

    Args:
        prim_path: The prim's path.

    Returns:
        A name from :data:`_DB21_PART_COLORS`, or None when no rule matches.
    """
    lowered = str(prim_path).lower()
    for keywords, name, _rgb in _DB21_PART_COLORS:
        if any(keyword in lowered for keyword in keywords):
            return name
    return None


def _color_db21_meshes(stage: Any, root: Any) -> dict[str, int]:
    """Bind flat materials to every mesh of a converted DB21, by group name and by geometry.

    Args:
        stage: The stage the wrapper layer is authored on.
        root: The ``/Root`` prim the geometry is referenced under.

    Returns:
        How many meshes each material name was bound to.
    """
    from pxr import Usd, UsdGeom

    looks = UsdGeom.Scope.Define(stage, root.GetPath().AppendChild("Looks")).GetPrim()
    palette = {name: rgb for _keywords, name, rgb in _DB21_PART_COLORS}
    palette["body"] = _DB21_BODY_COLOR
    materials = {
        name: _define_color_material(stage, looks.GetPath().AppendChild(name), rgb)
        for name, rgb in palette.items()
    }
    # The body colour is the floor: weaker than its descendants so the per-part bindings win,
    # and it still covers anything the loop below never reaches.
    _bind_material(root, materials["body"], strength="weakerThanDescendants")

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    whole = cache.ComputeWorldBound(root).ComputeAlignedRange()
    lateral = _lateral_axis(whole.GetMin(), whole.GetMax())
    counts: dict[str, int] = {}
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        name = _db21_material_name(prim.GetPath().pathString)
        if name is None:
            cache.Clear()
            box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            reach = max(abs(box.GetMin()[lateral]), abs(box.GetMax()[lateral]))
            name = "tyre" if reach >= _DB21_WHEEL_HALF_WIDTH_M else "body"
        _bind_material(prim, materials[name])
        counts[name] = counts.get(name, 0) + 1
    return counts


def _build_colored_db21(main_usd: Any, out_usd: Any) -> Any:
    """Author a colored, referenceable wrapper layer over the geometry-only conversion.

    The colours have to live in a layer of their own rather than on each environment's copy,
    because a training scene marks its attachments ``instanceable`` so that USD and Hydra share
    one prototype, and opinions authored inside an instance are ignored. Building the wrapper
    once and referencing it everywhere buys both the colours and the sharing.

    Args:
        main_usd: The converter's geometry-only output.
        out_usd: Where to write the wrapper.

    Returns:
        The wrapper path, or None when it could not be authored.
    """
    from pxr import Usd, UsdGeom

    if out_usd.is_file():
        return out_usd
    try:
        stage = Usd.Stage.CreateNew(str(out_usd))
        root = UsdGeom.Xform.Define(stage, "/Root").GetPrim()
        stage.SetDefaultPrim(root)
        root.GetReferences().AddReference(str(main_usd))
        counts = _color_db21_meshes(stage, root)
        stage.GetRootLayer().Save()
    except Exception as exc:
        print(f"[viz_env] db21j: could not author the colored layer ({exc!r}); geometry only")
        return None
    summary = ", ".join(f"{name} x{count}" for name, count in sorted(counts.items()))
    print(f"[viz_env] db21j: bound materials over {sum(counts.values())} meshes: {summary or 'none'}")
    if not counts.get("tyre"):
        print("[viz_env] db21j: no mesh classified as a wheel; check the conversion")
    return out_usd


def _prepare_db21_obj(convert: Any, obj: Any) -> Any:
    """Convert the Duckiematrix OBJ and wrap it in the colored layer, once per machine.

    Args:
        convert: ``(src, dst) -> bool`` asset conversion callable.
        obj: Path to ``main.obj``.

    Returns:
        The USD to reference from every robot, or None when the conversion failed.
    """
    geometry = obj.with_suffix(".usd")
    if not geometry.is_file() and not convert(obj, geometry):
        print("[viz_env] db21j: OBJ conversion failed")
        return None
    return _build_colored_db21(geometry, obj.with_name("main_colored.usd")) or geometry


def _attach_db21_obj(stage: Any, bbox_cache: Any, prepared: Any, robot_prim: str, verbose: bool) -> str:
    """Reference the prepared Duckiematrix DB21 under one robot's ``base_link``.

    Args:
        stage: The USD stage.
        bbox_cache: A ``UsdGeom.BBoxCache`` used to measure the referenced mesh.
        prepared: The USD produced by :func:`_prepare_db21_obj`.
        robot_prim: Prim path of the robot to attach to.
        verbose: Print the measured bounding box. True for the first robot only.

    Returns:
        A short outcome message, which the caller counts and prints once per distinct value.
    """
    from pxr import Gf, UsdGeom

    from duckiebot_rl.assets.params import DUCKIEBOT

    if not stage.GetPrimAtPath(f"{robot_prim}/base_link").IsValid():
        return f"no base_link under {robot_prim}; nothing attached"
    if stage.GetPrimAtPath(f"{robot_prim}/base_link/real_db21").IsValid():
        # The robot spawns with copy_from_source=False, so env_1..N are internal references to
        # env_0 and COMPOSE its attachment live. Authoring the ops again on a clone raises
        # ("xformOp:translate already exists in xformOpOrder"), which used to take the whole
        # attachment down to primitives on any multi-env scene.
        return "DB21 (Duckiematrix extraction) attached"
    xform = UsdGeom.Xform.Define(stage, f"{robot_prim}/base_link/real_db21")
    translate_op = xform.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(0.0, 0.0, 0.0))
    xform.AddRotateXYZOp().Set(Gf.Vec3f(*DB21_OBJ_ROTATE_XYZ))
    scale_op = xform.AddScaleOp()
    scale_op.Set(Gf.Vec3f(1.0))
    xform.GetPrim().GetReferences().AddReference(str(prepared))

    bbox_cache.Clear()
    size = bbox_cache.ComputeWorldBound(xform.GetPrim()).ComputeAlignedRange().GetSize()
    measured = tuple(float(size[axis]) for axis in range(3))
    # OBJ carries no unit metadata, so the converter guesses. The measure-and-rescale guard stays
    # even though this source is authored in metres and therefore lands on a factor of ~1.0.
    factor = DB21_OBJ_EXPECTED_SIZE_M[0] / max(measured[0], measured[1], measured[2], 1e-9)
    rescaled = not 0.5 < factor < 2.0
    if rescaled:
        scale_op.Set(Gf.Vec3f(float(factor)))
        bbox_cache.Clear()
        size = bbox_cache.ComputeWorldBound(xform.GetPrim()).ComputeAlignedRange().GetSize()
        measured = tuple(float(size[axis]) for axis in range(3))
    # Put the wheels on the ground: base_link rides one wheel radius above it and the source is
    # authored standing on its own contact patch.
    bbox_cache.Clear()
    local = bbox_cache.ComputeLocalBound(xform.GetPrim()).ComputeAlignedRange()
    translate_op.Set(Gf.Vec3d(0.0, 0.0, -DUCKIEBOT.base_link_height_m - float(local.GetMin()[2])))
    if verbose:
        expected = " x ".join(f"{value:.4f}" for value in DB21_OBJ_EXPECTED_SIZE_M)
        got = " x ".join(f"{value:.4f}" for value in measured)
        print(f"[viz_env] db21j: oriented bbox {got} m (expected length x width x height {expected})")
        applied = "" if rescaled else " (in range, not applied)"
        print(f"[viz_env] db21j: scale factor {factor:.4g}{applied}")
    off = max(abs(measured[axis] - DB21_OBJ_EXPECTED_SIZE_M[axis]) for axis in range(3))
    if off > _DB21_SIZE_TOL_M:
        return (
            f"oriented bbox is {off * 1000.0:.1f} mm off the expected box: the asset converter "
            f"reoriented the source, so DB21_OBJ_ROTATE_XYZ no longer applies"
        )
    _hide_primitive_visuals(stage, robot_prim)
    return "DB21 (Duckiematrix extraction) attached"


def _attach_db21j_gltf(
    stage: Any, bbox_cache: Any, convert: Any, gltf: Any, robot_prim: str, verbose: bool
) -> str:
    """Attach the whole-robot DB18-era glTF under one robot's ``base_link``.

    Args:
        stage: The USD stage.
        bbox_cache: A ``UsdGeom.BBoxCache`` used to measure the referenced mesh.
        convert: ``(src, dst) -> bool`` asset conversion callable.
        gltf: Path to ``main.gltf``.
        robot_prim: Prim path of the robot to attach to.
        verbose: Print the measured bounding box. True for the first robot only.

    Returns:
        A short outcome message, empty when the caller should fall back.
    """
    from pxr import Gf, UsdGeom

    dst = gltf.with_suffix(".usd")
    if not dst.is_file() and not convert(gltf, dst):
        print("[viz_env] db21j convert failed")
        return ""
    if not stage.GetPrimAtPath(f"{robot_prim}/base_link").IsValid():
        return f"no base_link under {robot_prim}; nothing attached"
    if stage.GetPrimAtPath(f"{robot_prim}/base_link/real_db21").IsValid():
        # Clone of env_0 (copy_from_source=False): the attachment composes in already.
        return "DB18-era glTF attached (duckietown-world 'duckiebot3', not a real DB21)"
    xform = UsdGeom.Xform.Define(stage, f"{robot_prim}/base_link/real_db21")
    xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
    orient_op = xform.AddRotateXYZOp()
    orient_op.Set(Gf.Vec3f(0.0))
    scale_op = xform.AddScaleOp()
    scale_op.Set(Gf.Vec3f(1.0))
    xform.GetPrim().GetReferences().AddReference(str(dst))
    bbox_cache.Clear()
    size = bbox_cache.ComputeWorldBound(xform.GetPrim()).ComputeAlignedRange().GetSize()
    if verbose:
        dims = f"{size[0]:.4f} x {size[1]:.4f} x {size[2]:.4f}"
        print(f"[viz_env] db21j (db18-era glTF): bbox {dims} m")
    # glTF is Y-up; if the mast landed along Y instead of Z the model is lying down.
    if size[1] > 1.5 * max(size[2], 1e-9):
        orient_op.Set(Gf.Vec3f(90.0, 0.0, 0.0))
        if verbose:
            print("[viz_env] db21j: applied +90 deg X (Y-up correction)")
    longest = max(size[0], size[1], size[2], 1e-9)
    factor = 0.20 / longest
    if not 0.5 < factor < 2.0:
        scale_op.Set(Gf.Vec3f(float(factor)))
        if verbose:
            print(f"[viz_env] db21j: scale {factor:.4g}")
    _hide_primitive_visuals(stage, robot_prim)
    return "DB18-era glTF attached (duckietown-world 'duckiebot3', not a real DB21)"


def _attach_db17(
    stage: Any, bbox_cache: Any, convert: Any, mesh_dir: Any, robot_prim: str, verbose: bool
) -> str:
    """Attach the per-part DB17 OBJs to one robot, one reference per link.

    Args:
        stage: The USD stage.
        bbox_cache: A ``UsdGeom.BBoxCache`` used to measure each referenced mesh.
        convert: ``(src, dst) -> bool`` asset conversion callable.
        mesh_dir: Directory holding ``chassis.obj`` and friends.
        robot_prim: Prim path of the robot to attach to.
        verbose: Print the measured bounding boxes. True for the first robot only.

    Returns:
        A short outcome message, empty when the caller should fall back.
    """
    from pxr import Gf, UsdGeom

    usd_dir = mesh_dir.parent / "visual_usd"
    usd_dir.mkdir(parents=True, exist_ok=True)
    converted: dict[str, Any] = {}
    for _link, stem, _offset in _REAL_VISUALS:
        src, dst = mesh_dir / f"{stem}.obj", usd_dir / f"{stem}.usd"
        if not dst.is_file() and not convert(src, dst):
            print(f"[viz_env] mesh convert failed for {stem}; keeping primitive visuals")
            return ""
        converted[stem] = dst

    expected_m = {
        "chassis": 0.200,
        "computer": 0.095,
        "camera": 0.040,
        "left_wheel": 0.066,
        "right_wheel": 0.066,
    }
    for link, stem, offset in _REAL_VISUALS:
        if not stage.GetPrimAtPath(f"{robot_prim}/{link}").IsValid():
            return f"no {link} under {robot_prim}; nothing attached"
        if stage.GetPrimAtPath(f"{robot_prim}/{link}/real_{stem}").IsValid():
            # Clone of env_0 (copy_from_source=False): this part composes in already.
            continue
        xform = UsdGeom.Xform.Define(stage, f"{robot_prim}/{link}/real_{stem}")
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
        if verbose:
            dims = f"{size[0]:.4f} x {size[1]:.4f} x {size[2]:.4f}"
            print(f"[viz_env] {stem}: bbox {dims} m -> scale {factor:.4g}")
        if not 0.5 < factor < 2.0:
            scale_op.Set(Gf.Vec3f(float(factor)))
        _bind_color(stage, xform.GetPrim(), _VISUAL_COLORS.get(stem, (0.08, 0.08, 0.09)))
    _hide_primitive_visuals(stage, robot_prim)
    return f"DB17 per-part meshes attached from {mesh_dir}"


def _make_converter() -> Any:
    """Return a ``(src, dst) -> bool`` mesh-to-USD conversion callable.

    Returns:
        A callable that converts and blocks. When Kit's asset converter is unavailable the
        callable reports why and returns False, which still lets cached ``.usd`` files attach.
    """
    try:
        # The extension has to be enabled BEFORE the module import: Kit only puts
        # omni.kit.asset_converter on sys.path once the extension is loaded.
        import asyncio

        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("omni.kit.asset_converter")
        import omni.kit.asset_converter as asset_converter

        async def _convert_async(src: Any, dst: Any) -> bool:
            """Run one conversion task to completion.

            Args:
                src: Source mesh path.
                dst: Destination ``.usd`` path.

            Returns:
                True on success.
            """
            task = asset_converter.get_instance().create_converter_task(str(src), str(dst), None)
            return await task.wait_until_finished()

        loop = asyncio.get_event_loop()

        def convert(src: Any, dst: Any) -> bool:
            """Convert one source mesh to USD, blocking until Kit finishes.

            Args:
                src: Source mesh path.
                dst: Destination ``.usd`` path.

            Returns:
                True on success.
            """
            return bool(loop.run_until_complete(_convert_async(src, dst)))

        return convert
    except Exception as exc:  # cached .usd files can still attach without a converter
        converter_error = repr(exc)

        def unavailable(src: Any, dst: Any) -> bool:
            """Report that no converter is available; cached .usd files still attach.

            Args:
                src: Source mesh path.
                dst: Destination ``.usd`` path that would have been written.

            Returns:
                False, always.
            """
            print(f"[viz_env] cannot convert {src} -> {dst}: no asset converter ({converter_error})")
            return False

        return unavailable


def _env_count(env: Any) -> int:
    """Read how many environments the scene holds.

    Args:
        env: The constructed environment.

    Returns:
        The environment count, at least 1.
    """
    for owner, attribute in ((env, "num_envs"), (getattr(env, "scene", None), "num_envs")):
        value = getattr(owner, attribute, None)
        if isinstance(value, int) and value > 0:
            return value
    return 1


def _report(outcomes: dict[str, int]) -> None:
    """Print each distinct attachment outcome once, with how many robots it covered.

    64 environments produce 64 identical messages, which buries the one line that differs.

    Args:
        outcomes: Outcome message to robot count.
    """
    for message, count in outcomes.items():
        suffix = "" if count == 1 else f" (x{count} envs)"
        print(f"[viz_env] {message}{suffix}")


def attach_robot_visuals(env: Any, robot_mesh: str = DEFAULT_ROBOT_MESH, num_envs: int | None = None) -> str:
    """Swap the primitive robot visuals for a real Duckiebot mesh on every environment.

    Purely additive USD: the meshes are referenced as visual-only prims under links that already
    exist, and the environment's own visual scopes are hidden. No collider, rigid body, joint,
    mass or camera is touched, so physics, actuation and the policy's observation contract are
    exactly what they were. Every failure falls back to the primitive look, because the view must
    never take a training run or a viewer session down.

    Above one environment the attachments are marked ``instanceable`` so that USD shares a single
    prototype instead of holding 64 copies of a 281k-vertex mesh, which is the difference between
    a rounding error and gigabytes on an 8 GB card.

    Args:
        env: The constructed environment. Used only to count environments; the robots themselves
            are addressed through the USD stage.
        robot_mesh: One of :data:`ROBOT_MESH_CHOICES`.
        num_envs: Environment count, or None to read it from ``env``.

    Returns:
        The selector that was actually applied, which is ``"primitive"`` when nothing attached.

    Raises:
        ValueError: If ``robot_mesh`` is not one of :data:`ROBOT_MESH_CHOICES`. The selector is a
            programming or command-line error, unlike a missing mesh file, which downgrades.
    """
    choice = resolve_robot_mesh(robot_mesh)
    if choice == "primitive":
        print("[viz_env] robot mesh 'primitive': keeping the environment's own visuals")
        return "primitive"
    count = _env_count(env) if num_envs is None else max(1, int(num_envs))
    try:
        import omni.usd
        from pxr import Usd, UsdGeom

        convert = _make_converter()
        stage = omni.usd.get_context().get_stage()
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
        outcomes: dict[str, int] = {}

        source = resolve_db21j_source() if choice == "db21j" else None
        kind = None if source is None else source[0]
        prepared = _prepare_db21_obj(convert, source[1]) if kind == "duckiematrix-obj" else None
        if kind == "duckiematrix-obj" and prepared is None:
            kind = None

        for index in range(count):
            prim = robot_prim_path(index)
            verbose = index == 0
            if kind == "duckiematrix-obj":
                message = _attach_db21_obj(stage, bbox_cache, prepared, prim, verbose)
            elif kind == "db18-gltf":
                message = _attach_db21j_gltf(stage, bbox_cache, convert, source[1], prim, verbose)
            else:
                message = ""
            if not message:
                mesh_dir = _find_visual_mesh_dir()
                if mesh_dir is None:
                    _report({"no per-part meshes on disk; keeping primitive visuals": count})
                    return "primitive"
                message = _attach_db17(stage, bbox_cache, convert, mesh_dir, prim, verbose)
                if not message:
                    _report({"mesh conversion failed; keeping primitive visuals": count})
                    return "primitive"
                choice = "db17"
            if count > 1:
                attached = stage.GetPrimAtPath(f"{prim}/base_link/real_db21")
                if attached and attached.IsValid():
                    attached.SetInstanceable(True)
            outcomes[message] = outcomes.get(message, 0) + 1
        if count > 1 and choice == "db21j":
            print(f"[viz_env] db21j: {count} attachments marked instanceable (one shared prototype)")
        _report(outcomes)
        return choice
    except Exception as exc:
        print(f"[viz_env] real visuals unavailable, keeping primitives: {exc!r}")
        return "primitive"


def _attach_real_visuals(env: Any, robot_mesh: str = DEFAULT_ROBOT_MESH) -> None:
    """Attach the chosen robot visual to every environment the viewer built.

    The environment count is read from ``env`` itself (see ``_env_count``), so a 64-env
    parallel grid dresses all 64 robots, not just env 0.

    Args:
        env: The constructed ``DuckiebotLaneFollowEnv``.
        robot_mesh: One of :data:`ROBOT_MESH_CHOICES`.

    Raises:
        ValueError: If ``robot_mesh`` is not one of :data:`ROBOT_MESH_CHOICES`.
    """
    attach_robot_visuals(env, robot_mesh=robot_mesh)


def make_viz_env(
    map: str = "loop_small",
    num_envs: int = 1,
    device: str = "cuda:0",
    render: bool = True,
    dynamics_dr: bool = False,
    photometric_dr: bool = False,
    episode_length_s: float | None = None,
    seed: int = 0,
    robot_mesh: str = DEFAULT_ROBOT_MESH,
    **overrides: Any,
) -> IsaacVizEnv:
    """Build the single-environment Isaac backend the live viewer drives.

    Args:
        map: Which generated layout to show. ``city_<N>`` and ``eval_<N>`` name a training or
            held-out layout and are zero-padded for you; a bare stem names a stage directly; and
            a path to a generated ``maps/<stem>.yaml`` or ``usd/<stem>.usda`` also selects the
            build root that file lives in, which is how a layout from a non-default build such
            as ``build/city_hard`` is viewed. See :func:`resolve_city_selection`. The name is
            also recorded on the returned adapter as :attr:`IsaacVizEnv.requested_map`.
        num_envs: Environment count. 1 for the classic single-robot viewer; the parallel grid
            passes the full count and every robot gets the same visual treatment.
        device: Torch device for the simulation.
        render: Whether cameras and rendering are enabled. False gives physics only, which is
            useful when the host cannot afford the RTX pipeline (see the module docstring).
        dynamics_dr: Apply the S7.3 dynamics randomisation, which is sim-to-sim condition C6.
        photometric_dr: Apply the visual randomisation. Named for the viewer's shared flag; it
            maps onto ``LaneFollowSettings.visual_dr``.
        episode_length_s: Truncation horizon in simulated seconds, or None to keep the default.
        seed: Environment seed.
        robot_mesh: Which robot visual to draw, one of :data:`ROBOT_MESH_CHOICES`. Viewer-side
            only: physics, collisions and the camera are identical for all three, so this can
            never change what a policy sees or how it drives.
        **overrides: Extra fields forwarded to
            :func:`~duckiebot_rl.envs.env_cfg.lane_follow_env_cfg`.

    Returns:
        An :class:`IsaacVizEnv` satisfying the viewer's backend contract.

    Raises:
        RuntimeError: If Isaac Sim is unavailable or Kit has not been launched.
        ValueError: If ``robot_mesh`` is not one of :data:`ROBOT_MESH_CHOICES`. Checked before
            Kit is touched, so a typo costs nothing.
    """
    if str(robot_mesh).lower().strip() not in ROBOT_MESH_CHOICES:
        raise ValueError(
            f"unknown robot_mesh {robot_mesh!r}; expected one of {', '.join(ROBOT_MESH_CHOICES)}"
        )
    _require_kit()

    from duckiebot_rl.envs.env_cfg import LaneFollowSettings, lane_follow_env_cfg
    from duckiebot_rl.envs.lane_follow_env import DuckiebotLaneFollowEnv

    city = resolve_city_selection(map)
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
    if settings.num_envs == 1:
        # The stagger exists to spread 64 training resets across the horizon; inherited by a
        # single-environment viewer it randomises where episode 1 truncates, so a requested
        # 15-second showcase episode can end after a fraction of a second (observed: 5 steps).
        # The parallel grid keeps it, for the same reason training does.
        settings = dataclasses.replace(
            settings, rates=dataclasses.replace(settings.rates, stagger_initial_episode_length=False)
        )
    cfg = lane_follow_env_cfg(settings, **overrides)
    env = DuckiebotLaneFollowEnv(cfg, render_mode="rgb_array" if render else None)
    _attach_real_visuals(env, robot_mesh=robot_mesh)
    adapter = IsaacVizEnv(env)
    adapter.requested_map = str(map)
    return adapter
