"""Simulator backends for the live viewer, and the VRAM refusal that protects a training run.

Two backends, and the asymmetry between them is deliberate.

MuJoCo, the default
-------------------
:class:`MujocoBackend` wraps the existing :class:`duckiebot_rl.sim2sim.env.MjDuckiebotEnv` in
``obs_mode="rgb_vec"``, so the viewer drives the *same* environment, the *same* action path and
the *same* S4.3 observation chain that the sim-to-sim evaluation uses. Nothing about the task is
re-implemented here; this module only adds a chase camera, because the environment's own renderer
is the 192x128 robot camera and a 192x128 robot's-eye view is not a thing a human can watch a
policy drive in. It runs on the CPU and costs zero VRAM.

Isaac, opt-in only
------------------
Headless Isaac training on the target machine is budgeted at 6.4 to 7.6 GiB of 8 GiB. A second
Isaac Kit process costs another 2.5 to 3.5 GiB of baseline before it renders anything, so
6.4 + 2.5 = 8.9 GiB against an 8 GiB card: starting an Isaac viewer next to an Isaac trainer does
not run slowly, it out-of-memories, and it takes the training run down with it. So
:func:`make_backend` refuses ``backend="isaac"`` unless the caller passes ``allow_isaac_vram``,
and the refusal shows the arithmetic instead of saying "not supported".

``duckiebot_rl/envs`` is owned by another module and is being written concurrently. The Isaac
backend therefore lives behind exactly one lazy import, in :func:`_import_isaac_env_factory`, with
a documented factory contract. It starts working the moment that module lands and fails with an
actionable message until then. Nothing else in the viewer depends on it.

One robot or a grid of them
---------------------------
``num_envs=1`` is the chase-camera viewer: the factory's own single-environment adapter is
returned untouched and the viewer drives it with scalars, exactly as before. ``num_envs > 1`` is
the parallel grid: :class:`ParallelIsaacBackend` wraps that adapter, drives the vectorized
environment underneath it in one batched ``step`` per control tick, and leaves the whole picture
to the Kit viewport, which is the point. See :class:`ParallelIsaacBackend` for why the grid needs
a different layout selection than a single view does.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = [
    "BACKEND_NAMES",
    "DEFAULT_ROBOT_MESH",
    "ROBOT_MESH_NAMES",
    "IsaacVramRefusal",
    "MujocoBackend",
    "ParallelIsaacBackend",
    "RolloutBackend",
    "isaac_vram_message",
    "make_backend",
    "parallel_refusal_message",
    "resolve_map",
]

BACKEND_NAMES: tuple[str, ...] = ("mujoco", "isaac")
"""Backend names accepted by :func:`make_backend`."""

ROBOT_MESH_NAMES: tuple[str, ...] = ("db21j", "db17", "primitive")
"""Robot visual meshes the Isaac backend can draw, viewer-side only.

Spelled out here rather than imported from :mod:`duckiebot_rl.envs.viz_env` so that the viewer
still imports, parses a command line and runs the MuJoCo backend on a machine with no Isaac and
no ``duckiebot_rl.envs`` at all, which is the property the whole module is built around.
``tests/unit/test_live_view.py`` asserts the two tuples stay identical.
"""

DEFAULT_ROBOT_MESH = "db21j"
"""Default robot visual: the latest-generation DB21, when its glTF has been fetched."""

ISAAC_TRAIN_VRAM_GIB = (6.4, 7.6)
"""Measured headless-training VRAM envelope on the target RTX 3080 Laptop, 8 GiB."""

ISAAC_KIT_BASELINE_GIB = (2.5, 3.5)
"""Baseline VRAM a second Isaac Kit process costs before it renders anything."""

_ISAAC_FACTORY_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("duckiebot_rl.envs", "make_viz_env"),
    ("duckiebot_rl.envs", "make_env"),
    ("duckiebot_rl.envs.lane_following", "make_viz_env"),
    ("duckiebot_rl.envs.lane_following", "make_env"),
)
"""Where the Isaac backend looks for an environment factory, in order.

Contract for whoever owns ``duckiebot_rl/envs``: expose a callable taking keyword arguments
``map``, ``num_envs``, ``device``, ``render`` and ``robot_mesh`` and returning an object with
``reset(seed=None)``, ``step(action)``, ``render_frame()``, ``close()`` and a ``control_dt``
property, matching :class:`RolloutBackend`. Any one of the four names above is picked up
automatically.
"""


class IsaacVramRefusal(RuntimeError):
    """Raised when an Isaac viewer is requested without acknowledging its VRAM cost."""


@runtime_checkable
class RolloutBackend(Protocol):
    """The surface the viewer needs from a simulator.

    Deliberately tiny: everything the viewer does is reset, step, render, close.
    """

    @property
    def control_dt(self) -> float:
        """Seconds of simulated time per control step."""

    def reset(self, seed: int | None = None) -> Mapping[str, np.ndarray]:
        """Start a new episode and return the first observation.

        Args:
            seed: Episode seed.

        Returns:
            The observation mapping.
        """

    def step(
        self, action: np.ndarray
    ) -> tuple[Mapping[str, np.ndarray], float, bool, bool, Mapping[str, Any]]:
        """Advance one control step.

        Args:
            action: Action in the ``[-1, 1]`` box.

        Returns:
            ``(obs, reward, terminated, truncated, info)``.
        """

    def render_frame(self) -> np.ndarray:
        """Return one ``(H, W, 3)`` uint8 RGB frame for the viewer and the video."""

    def close(self) -> None:
        """Release simulator resources."""


def isaac_vram_message(reason: str = "") -> str:
    """Return the refusal text explaining why a second Isaac process will not fit.

    Args:
        reason: Optional extra sentence appended to the explanation.

    Returns:
        A multi-line message showing the VRAM arithmetic and naming the alternative.
    """
    train_lo, train_hi = ISAAC_TRAIN_VRAM_GIB
    kit_lo, kit_hi = ISAAC_KIT_BASELINE_GIB
    text = (
        "refusing to start an Isaac viewer without --allow-isaac-vram.\n"
        f"  headless Isaac training on this machine uses {train_lo}-{train_hi} GiB of 8 GiB VRAM\n"
        f"  a second Isaac Kit process costs another {kit_lo}-{kit_hi} GiB of baseline\n"
        f"  {train_lo} + {kit_lo} = {train_lo + kit_lo:.1f} GiB against an 8 GiB card, so the two "
        "together OOM and take the training run with them\n"
        "  use --backend mujoco instead: it is the default, runs on the CPU, costs zero VRAM and "
        "drives the same task through the same observation chain\n"
        "  if training is NOT running right now, pass --allow-isaac-vram to acknowledge the cost"
    )
    return f"{text}\n  {reason}" if reason else text


def resolve_map(source: Any) -> Any:
    """Turn a ``--map`` argument into something the track builder accepts.

    :func:`duckiebot_rl.sim2sim.track.load_map` takes a ``MapSpec``, a mapping, or a path to a
    MapFormat1 YAML file. It does NOT take a built-in map name, so passing ``"loop_small"``
    straight through makes it try to open a file called ``loop_small`` and fail with a bare
    ``FileNotFoundError``. Built-in names are resolved here through the ``[city]`` module, which
    owns them.

    Args:
        source: A built-in map name, a path to a map file, a mapping, or a ``MapSpec``.

    Returns:
        A value :func:`~duckiebot_rl.sim2sim.track.load_map` understands.

    Raises:
        ValueError: If a string is neither a built-in name nor an existing file, listing the
            built-in names so the caller can see what was available.
    """
    if not isinstance(source, str):
        return source
    try:
        from duckiebot_rl.city.maps import BUILTIN_MAP_NAMES, builtin_map
    except ImportError:  # pragma: no cover - the [city] module is a hard dependency in practice
        return source
    if source in BUILTIN_MAP_NAMES:
        return builtin_map(source).to_dict()
    if Path(source).is_file():
        return source
    raise ValueError(
        f"unknown map {source!r}: it is neither a built-in map name nor an existing file.\n"
        f"  built-in names: {', '.join(BUILTIN_MAP_NAMES)}"
    )


class MujocoBackend:
    """The default viewer backend: the existing MuJoCo sim-to-sim environment plus a chase camera.

    Args:
        map_name: Map accepted by :func:`duckiebot_rl.sim2sim.track.load_map`, normally one of
            :data:`duckiebot_rl.city.maps.BUILTIN_MAP_NAMES`.
        episode_length_s: Truncation horizon in seconds of simulated time.
        seed: Environment seed.
        asset_dir: Directory the track builder writes tile textures into. None uses a temporary
            directory that is removed by :meth:`close`.
        chase_size: ``(height, width)`` of the chase render, in pixels.
        chase_distance: Chase camera distance from the robot, in metres.
        chase_elevation: Chase camera elevation in degrees; negative looks down.
        chase_azimuth: Chase camera azimuth in degrees, relative to the world frame.
        inset: Draw the robot's own 192x128 camera as an inset in the chase frame.
        dynamics_dr: Apply the S7.3 dynamics randomisation, the sim-to-sim C6 condition.
        photometric_dr: Apply the shared photometric randomisation inside preprocess.
        dr_alpha: Curriculum scalar in ``[0, 1]`` scaling every randomisation range.

    Attributes:
        env: The wrapped :class:`~duckiebot_rl.sim2sim.env.MjDuckiebotEnv`.
        name: Backend name, ``"mujoco"``.
    """

    name = "mujoco"

    def __init__(
        self,
        map_name: str = "loop_small",
        episode_length_s: float = 30.0,
        seed: int = 0,
        asset_dir: str | os.PathLike[str] | None = None,
        chase_size: tuple[int, int] = (480, 720),
        chase_distance: float = 1.1,
        chase_elevation: float = -28.0,
        chase_azimuth: float = 135.0,
        inset: bool = True,
        dynamics_dr: bool = False,
        photometric_dr: bool = False,
        dr_alpha: float = 1.0,
    ) -> None:
        import mujoco

        from duckiebot_rl.sim2sim.env import MjDuckiebotEnv, MjEnvCfg

        self._mj = mujoco
        self._owns_asset_dir = asset_dir is None
        self._asset_dir = (
            Path(asset_dir) if asset_dir is not None else Path(tempfile.mkdtemp(prefix="dbviz_"))
        )
        self._asset_dir.mkdir(parents=True, exist_ok=True)

        self.env = MjDuckiebotEnv(
            MjEnvCfg(
                map=resolve_map(map_name),
                asset_dir=str(self._asset_dir),
                episode_length_s=float(episode_length_s),
                obs_mode="rgb_vec",
                dynamics_dr=bool(dynamics_dr),
                photometric_dr=bool(photometric_dr),
                dr_alpha=float(dr_alpha),
                seed=int(seed),
            )
        )
        self.inset = bool(inset)
        height, width = int(chase_size[0]), int(chase_size[1])
        # The generated MJCF sizes its offscreen framebuffer for the 192x128 robot camera, which
        # is all the sim-to-sim harness ever renders. A chase view is much larger, and MuJoCo
        # refuses to build a renderer wider than the buffer. Growing the buffer here, after the
        # env has already built its own 192x128 renderer, leaves the observation path untouched:
        # the policy still sees exactly the frames the S4.3 chain produces.
        self.env.model.vis.global_.offwidth = max(self.env.model.vis.global_.offwidth, width)
        self.env.model.vis.global_.offheight = max(self.env.model.vis.global_.offheight, height)
        self._chase = mujoco.Renderer(self.env.model, height, width)
        self._camera = mujoco.MjvCamera()
        self._camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        # The env exposes no public accessor for its base body id; reading the private attribute
        # is preferable to a second mj_name2id lookup that could silently disagree with it.
        self._camera.trackbodyid = self.env._base_body
        self._camera.distance = float(chase_distance)
        self._camera.elevation = float(chase_elevation)
        self._camera.azimuth = float(chase_azimuth)
        self._last_obs: dict[str, np.ndarray] = {}

    @property
    def control_dt(self) -> float:
        """Seconds of simulated time per control step."""
        return float(self.env.control_dt)

    @property
    def last_obs(self) -> dict[str, np.ndarray]:
        """The most recent observation, kept so the viewer can snapshot what the policy saw."""
        return self._last_obs

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        """Start a new episode.

        Args:
            seed: Episode seed; None keeps the environment's own stream running.

        Returns:
            The first observation.
        """
        obs, _info = self.env.reset(seed=seed)
        self._last_obs = dict(obs)
        return self._last_obs

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Advance one control step.

        Args:
            action: Action in the ``[-1, 1]`` box; it is clipped here.

        Returns:
            ``(obs, reward, terminated, truncated, info)``.
        """
        clipped = np.clip(np.asarray(action, dtype=np.float32).reshape(2), -1.0, 1.0)
        obs, reward, terminated, truncated, info = self.env.step(clipped)
        self._last_obs = dict(obs)
        return self._last_obs, float(reward), bool(terminated), bool(truncated), dict(info)

    def render_chase(self) -> np.ndarray:
        """Render the third-person chase view.

        Returns:
            An ``(H, W, 3)`` uint8 RGB frame.
        """
        self._chase.update_scene(self.env.data, camera=self._camera)
        return np.asarray(self._chase.render())

    def render_onboard(self) -> np.ndarray:
        """Render the robot's own camera at the canonical 192x128 render resolution.

        Returns:
            A ``(128, 192, 3)`` uint8 RGB frame, before the S4.3 preprocessing tail.
        """
        return np.asarray(self.env.render_frame())

    def render_frame(self) -> np.ndarray:
        """Return the viewer frame: the chase view, with the robot camera inset when enabled.

        Returns:
            An ``(H, W, 3)`` uint8 RGB frame.
        """
        frame = np.ascontiguousarray(self.render_chase().astype(np.uint8))
        if not self.inset:
            return frame
        onboard = self.render_onboard().astype(np.uint8)
        scale = 2
        tile = np.repeat(np.repeat(onboard, scale, axis=0), scale, axis=1)
        tile_h, tile_w = tile.shape[:2]
        margin = 12
        top = frame.shape[0] - tile_h - margin
        left = frame.shape[1] - tile_w - margin
        if top < 1 or left < 1:
            return frame
        frame[top - 2 : top + tile_h + 2, left - 2 : left + tile_w + 2] = (240, 240, 245)
        frame[top : top + tile_h, left : left + tile_w] = tile
        return frame

    def close(self) -> None:
        """Release the renderers, the environment and any temporary texture directory."""
        for renderer in (getattr(self, "_chase", None),):
            if renderer is not None:
                with _suppress_close():
                    renderer.close()
        with _suppress_close():
            self.env.close()
        if self._owns_asset_dir:
            shutil.rmtree(self._asset_dir, ignore_errors=True)


class _suppress_close:
    """Context manager that swallows teardown errors so ``close()`` is always safe to call."""

    def __enter__(self) -> None:
        """Enter the context. Does nothing."""
        return None

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        """Swallow any exception raised inside the context.

        Args:
            exc_type: Exception type, if any.
            exc: Exception instance, if any.
            tb: Traceback, if any.

        Returns:
            True, so the exception is suppressed.
        """
        return True


def _import_isaac_env_factory() -> Any:
    """Import the Isaac environment factory owned by ``duckiebot_rl/envs``.

    This is THE single lazy import of the concurrently developed module. Keeping it in one
    function means the rest of the viewer imports, tests and runs with no Isaac and no
    ``duckiebot_rl.envs`` present at all.

    Returns:
        The first factory callable found among :data:`_ISAAC_FACTORY_CANDIDATES`.

    Raises:
        RuntimeError: If none is importable, naming every location that was tried and the
            keyword contract the factory has to satisfy.
    """
    tried: list[str] = []
    for module_name, attr in _ISAAC_FACTORY_CANDIDATES:
        tried.append(f"{module_name}:{attr}")
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        factory = getattr(module, attr, None)
        if callable(factory):
            return factory
    raise RuntimeError(
        "the Isaac viewer backend needs an environment factory from duckiebot_rl/envs, which is "
        "not importable yet.\n  tried: "
        + ", ".join(tried)
        + "\n  the factory must accept the keywords map, num_envs, device and render, and return "
        "an object with reset(seed=None), step(action), render_frame(), close() and a control_dt "
        "property.\n  until that module lands, use --backend mujoco, which is the default and "
        "drives the same task through the same observation chain."
    )


def parallel_refusal_message(backend: str, num_envs: int) -> str:
    """Return the refusal text for a parallel view the requested backend cannot give.

    Args:
        backend: The backend name that was asked for.
        num_envs: The environment count that was asked for.

    Returns:
        A message naming what parallel mode needs and what to run instead.
    """
    return (
        f"--num-envs {num_envs} is an Isaac-only feature, and --backend {backend} was requested.\n"
        "  the parallel grid is one Isaac scene holding num_envs cities side by side, each with "
        "its own robot, all driven by one policy in one batched forward pass\n"
        "  the MuJoCo backend is a single CPU simulation with one map and one robot; running "
        f"{num_envs} of them would mean {num_envs} processes, not a grid\n"
        "  use --backend isaac --allow-isaac-vram --window --num-envs "
        f"{num_envs}, or drop --num-envs to watch one MuJoCo episode"
    )


class ParallelIsaacBackend:
    """Drive every environment of a multi-city Isaac scene from one policy, for the grid view.

    The single-environment adapter this wraps exists to hide the vectorized environment: it
    reshapes one action to ``(1, act_dim)`` and returns ``reward[0]`` as a float. That is exactly
    the wrong shape for a grid, so the grid does not use it. It reaches through the adapter's
    documented ``env`` attribute and drives the vectorized environment in its native batched form:
    one ``(N, act_dim)`` tensor in, ``(N, ...)`` observations out, no per-environment Python loop
    anywhere in the step path.

    Nothing is copied to the host per step. ``reward``, ``terminated`` and ``truncated`` are
    handed back as the environment's own tensors, so the caller decides when to pay for a
    synchronisation; the status line does, once every few seconds.

    Episode bookkeeping is deliberately absent. The environment resets its own finished
    environments inside ``step``, which is what makes the grid a continuous picture rather than N
    episodes that all have to end before anything restarts.

    Args:
        adapter: The object the environment factory returned. Must expose ``env`` (the vectorized
            environment), ``control_dt`` and ``close``.
        num_envs: How many environments the scene was built with.

    Attributes:
        adapter: The wrapped factory object.
        num_envs: Environment count.
        name: Backend name, ``"isaac-parallel"``.

    Raises:
        RuntimeError: If the adapter exposes no vectorized ``env``.
    """

    name = "isaac-parallel"

    def __init__(self, adapter: Any, num_envs: int) -> None:
        env = getattr(adapter, "env", None)
        if env is None:
            raise RuntimeError(
                "the parallel grid needs the vectorized environment underneath the viewer "
                "adapter, which is the adapter's documented 'env' attribute, and this adapter "
                f"({type(adapter).__name__}) has none. Run with --num-envs 1 for the "
                "single-environment view."
            )
        self.adapter = adapter
        self.env = env
        self.num_envs = int(num_envs)

    @property
    def control_dt(self) -> float:
        """Seconds of simulated time per :meth:`step`."""
        return float(self.adapter.control_dt)

    @staticmethod
    def _policy_obs(obs: Any) -> Any:
        """Unwrap the actor's half of an Isaac Lab observation dict, batch axis intact.

        Args:
            obs: What the environment returned, normally ``{"policy": {...}}``.

        Returns:
            The mapping the policy consumes, with its leading environment axis untouched.
        """
        if isinstance(obs, Mapping) and "policy" in obs:
            return obs["policy"]
        return obs

    def reset(self, seed: int | None = None) -> Any:
        """Reset every environment once.

        The grid is reset exactly once, at the start. After that the environment resets finished
        environments itself, inside :meth:`step`.

        Args:
            seed: Seed forwarded to the environment.

        Returns:
            The batched observation mapping, entries shaped ``(num_envs, ...)``.
        """
        obs, _info = self.env.reset(seed=seed)
        return self._policy_obs(obs)

    def step(self, actions: Any) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        """Advance every environment one control step with one batched action.

        Args:
            actions: An ``(num_envs, act_dim)`` array or tensor in the ``[-1, 1]`` box; it is
                clipped here, the same way the single-environment backend clips.

        Returns:
            ``(obs, reward, terminated, truncated, info)``. The three middle entries are the
            environment's own ``(num_envs,)`` tensors, not host scalars.
        """
        import torch

        if isinstance(actions, torch.Tensor):
            tensor = actions.detach().to(device=self.env.device, dtype=torch.float32)
        else:
            tensor = torch.as_tensor(np.asarray(actions, dtype=np.float32), device=self.env.device)
        tensor = tensor.reshape(self.num_envs, -1).clamp(-1.0, 1.0)
        obs, reward, terminated, truncated, info = self.env.step(tensor)
        return (
            self._policy_obs(obs),
            reward,
            terminated,
            truncated,
            dict(info) if isinstance(info, Mapping) else {},
        )

    def render_frame(self) -> np.ndarray | None:
        """Return nothing: the grid's picture is the Kit viewport, not a captured frame.

        A chase camera follows one robot, and there are ``num_envs`` of them. Recording a grid is
        the training evaluation recorder's job, not the live viewer's.

        Returns:
            None, always.
        """
        return None

    def close(self) -> None:
        """Close the wrapped adapter, and hence the environment."""
        with _suppress_close():
            self.adapter.close()


def _parallel_factory_kwargs(factory: Any, map_name: str, num_envs: int) -> dict[str, Any]:
    """Work out what the environment factory has to be told to build a VARIED grid.

    Left alone, the viewer factory pins the scene's layout list to the single stage ``--map``
    named, because a one-environment scene is env 0 and ``MultiUsdFileCfg`` assigns assets by
    ``index % len``. That is exactly right for one robot and exactly wrong for a grid: N
    environments over a one-entry list is N copies of the same city.

    Two ways to widen it, in order:

    1. If the factory advertises an ``allow_multi`` parameter, it owns the decision; pass
       ``allow_multi=True`` and let it choose the layouts.
    2. Otherwise widen the selection here, through ``city``, which is a documented
       ``LaneFollowSettings`` field the factory forwards to the config builder. The build root
       ``--map`` resolved to is kept, so ``--map build/city_hard/maps/city_007.yaml --num-envs 64``
       still shows the hard build; only the variant list grows, to ``city_000 .. city_{N-1}``.

    Args:
        factory: The environment factory callable.
        map_name: The ``--map`` argument, used only to pick the build root.
        num_envs: How many distinct layouts the grid wants.

    Returns:
        Keyword arguments to add to the factory call. Empty when neither route is available, in
        which case the grid still runs and simply shows one layout N times.
    """
    try:
        parameters = inspect.signature(factory).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtins have no signature
        parameters = {}
    if "allow_multi" in parameters:
        return {"allow_multi": True}

    try:
        from duckiebot_rl.envs.viz_env import resolve_city_selection
    except ImportError:  # pragma: no cover - a factory from somewhere else entirely
        return {}
    try:
        selection = resolve_city_selection(map_name)
        widened = dataclasses.replace(selection, num_variants=num_envs, variant_names=None)
    except (TypeError, ValueError, FileNotFoundError, OSError) as exc:
        print(f"[live_view] keeping the factory's own layout selection for the grid: {exc!r}")
        return {}
    print(
        f"[live_view] grid layouts: city_000 .. city_{num_envs - 1:03d} from "
        f"{getattr(widened, 'root', None) or 'the default city search roots'}"
    )
    return {"city": widened}


def make_backend(
    backend: str = "mujoco",
    map_name: str = "loop_small",
    episode_length_s: float = 30.0,
    seed: int = 0,
    allow_isaac_vram: bool = False,
    asset_dir: str | os.PathLike[str] | None = None,
    device: str = "cuda",
    robot_mesh: str = DEFAULT_ROBOT_MESH,
    num_envs: int = 1,
    **kwargs: Any,
) -> RolloutBackend:
    """Build a rollout backend by name.

    Args:
        backend: ``"mujoco"`` or ``"isaac"``.
        map_name: Map to drive on.
        episode_length_s: Truncation horizon in seconds of simulated time.
        seed: Environment seed.
        allow_isaac_vram: Required to build the Isaac backend. See :func:`isaac_vram_message`.
        asset_dir: Texture directory for the MuJoCo backend.
        device: Torch/Isaac device for the Isaac backend.
        robot_mesh: Which robot visual the Isaac scene draws, one of :data:`ROBOT_MESH_NAMES`.
            Named explicitly rather than left to ``**kwargs`` because the MuJoCo backend has no
            such concept and would reject the keyword.
        num_envs: How many environments the scene holds. 1 returns the single-environment
            adapter unchanged; anything above 1 builds the parallel grid and returns a
            :class:`ParallelIsaacBackend`, which is Isaac-only.
        **kwargs: Extra keyword arguments forwarded to the backend constructor.

    Returns:
        The backend.

    Raises:
        ValueError: If ``backend`` is not a known name, or if a parallel grid is asked of the
            MuJoCo backend.
        IsaacVramRefusal: If the Isaac backend is requested without ``allow_isaac_vram``.
        RuntimeError: If the Isaac backend is requested but ``duckiebot_rl/envs`` is absent.
    """
    name = str(backend).lower().strip()
    count = max(1, int(num_envs))
    if name == "mujoco":
        if count > 1:
            raise ValueError(parallel_refusal_message(name, count))
        return MujocoBackend(
            map_name=map_name,
            episode_length_s=episode_length_s,
            seed=seed,
            asset_dir=asset_dir,
            **kwargs,
        )
    if name == "isaac":
        if not allow_isaac_vram:
            raise IsaacVramRefusal(isaac_vram_message())
        factory = _import_isaac_env_factory()
        extra = _parallel_factory_kwargs(factory, map_name, count) if count > 1 else {}
        # episode_length_s and seed must be forwarded too: without them the Isaac env falls back
        # to its own default horizon, which truncated a viewer episode after 5 steps.
        adapter = factory(
            map=map_name,
            num_envs=count,
            device=device,
            render=True,
            episode_length_s=episode_length_s,
            seed=seed,
            robot_mesh=robot_mesh,
            **extra,
            **kwargs,
        )
        return adapter if count == 1 else ParallelIsaacBackend(adapter, num_envs=count)
    raise ValueError(f"unknown backend {backend!r}; expected one of {', '.join(BACKEND_NAMES)}")
