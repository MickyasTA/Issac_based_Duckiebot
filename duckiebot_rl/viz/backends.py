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
"""

from __future__ import annotations

import importlib
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = [
    "BACKEND_NAMES",
    "IsaacVramRefusal",
    "MujocoBackend",
    "RolloutBackend",
    "isaac_vram_message",
    "make_backend",
    "resolve_map",
]

BACKEND_NAMES: tuple[str, ...] = ("mujoco", "isaac")
"""Backend names accepted by :func:`make_backend`."""

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
``map``, ``num_envs``, ``device`` and ``render`` and returning an object with ``reset(seed=None)``,
``step(action)``, ``render_frame()``, ``close()`` and a ``control_dt`` property, matching
:class:`RolloutBackend`. Any one of the four names above is picked up automatically.
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


def make_backend(
    backend: str = "mujoco",
    map_name: str = "loop_small",
    episode_length_s: float = 30.0,
    seed: int = 0,
    allow_isaac_vram: bool = False,
    asset_dir: str | os.PathLike[str] | None = None,
    device: str = "cuda",
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
        **kwargs: Extra keyword arguments forwarded to the backend constructor.

    Returns:
        The backend.

    Raises:
        ValueError: If ``backend`` is not a known name.
        IsaacVramRefusal: If the Isaac backend is requested without ``allow_isaac_vram``.
        RuntimeError: If the Isaac backend is requested but ``duckiebot_rl/envs`` is absent.
    """
    name = str(backend).lower().strip()
    if name == "mujoco":
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
        # episode_length_s and seed must be forwarded too: without them the Isaac env falls back
        # to its own default horizon, which truncated a viewer episode after 5 steps.
        return factory(
            map=map_name,
            num_envs=1,
            device=device,
            render=True,
            episode_length_s=episode_length_s,
            seed=seed,
            **kwargs,
        )
    raise ValueError(f"unknown backend {backend!r}; expected one of {', '.join(BACKEND_NAMES)}")
