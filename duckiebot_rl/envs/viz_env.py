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

from typing import Any

import numpy as np

__all__ = ["IsaacVizEnv", "make_viz_env"]


def _require_kit() -> None:
    """Fail early and clearly when Kit is not running.

    Raises:
        RuntimeError: If ``SimulationApp`` has not been started yet.
    """
    try:
        import isaacsim  # noqa: F401
        from isaacsim.simulation_app import SimulationApp
    except ImportError as exc:  # pragma: no cover - needs Isaac Sim
        raise RuntimeError(
            "the Isaac viewer backend needs Isaac Sim, which is not importable in this "
            "interpreter. Use the Isaac venv:\n"
            "  d:/Personal/personal/wheeled_quadruped_robot/.venv/Scripts/python.exe"
        ) from exc
    if getattr(SimulationApp, "_instance", None) is None:  # pragma: no cover - needs Isaac Sim
        raise RuntimeError(
            "Isaac Kit is not running. The app has to be launched BEFORE the environment is "
            "built, because Isaac Sim can only be configured pre-start. Run the viewer through "
            "scripts/live_view.py, which calls AppLauncher first."
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
        return (
            self._to_numpy(obs),
            float(reward[0].item()),
            bool(terminated[0].item()),
            bool(truncated[0].item()),
            dict(info) if isinstance(info, dict) else {},
        )

    def render_frame(self) -> np.ndarray | None:
        """Return the current RGB frame for recording.

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

    def close(self) -> None:
        """Close the wrapped environment."""
        self.env.close()


def make_viz_env(
    map: str = "loop_small",
    num_envs: int = 1,
    device: str = "cuda:0",
    render: bool = True,
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
        city=city,
    )
    cfg = lane_follow_env_cfg(settings, **overrides)
    env = DuckiebotLaneFollowEnv(cfg, render_mode="rgb_array" if render else None)
    adapter = IsaacVizEnv(env)
    adapter.requested_map = str(map)
    return adapter
