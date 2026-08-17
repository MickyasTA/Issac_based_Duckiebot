"""The Isaac Lab lane-following environment and everything it is assembled from (SPEC v2 S5).

Owner ``[env]`` (SPEC v2 S10, engineer E3).

Module map
----------

==========================  ====================================================  =============
module                      purpose                                               needs Isaac?
==========================  ====================================================  =============
:mod:`.camera_math`         the shared camera model, S2 quaternions, S4.4 optics   no
:mod:`.rewards`             every S5.4 reward term, vectorised over N envs         no
:mod:`.terminations`        every S5.5 termination and the truncation              no
:mod:`.action_path`         the S5.3 action path in torch                          no
:mod:`.obstacles`           the obstacle motion model and safety-circle query      no [1]
:mod:`.env_cfg`             the S5 numbers, and the DirectRLEnvCfg builder          no [1]
:mod:`.lane_follow_env`     the ``DirectRLEnv`` subclass                            YES
==========================  ====================================================  =============

[1] The module imports without Isaac; only its config-builder function needs it, and that
    function imports ``isaaclab`` inside its own body.

Imports are lazy (PEP 562), so ``from duckiebot_rl.envs import rewards`` on a CPU-only test
runner never touches ``lane_follow_env`` and therefore never touches Isaac Lab. Reaching for
:class:`~duckiebot_rl.envs.lane_follow_env.DuckiebotLaneFollowEnv` outside the Isaac Sim
interpreter raises the real, actionable ``ImportError`` from that module rather than a confusing
``AttributeError`` from this one.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - resolved by type checkers, never at run time
    from . import action_path, camera_math, env_cfg, obstacles, rewards, terminations  # noqa: F401
    from .action_path import TorchActionPath  # noqa: F401
    from .camera_math import (  # noqa: F401
        GOLDEN_QUAT_PITCH_0,
        GOLDEN_QUAT_PITCH_NOMINAL,
        focal_px,
        horizon_row,
        pinhole_camera_kwargs,
        quat_cam_ros,
        quat_cam_ros_rpy,
        quat_cam_ros_torch,
    )
    from .env_cfg import (  # noqa: F401
        CitySettings,
        LaneFollowSettings,
        ObstacleSettings,
        RateSettings,
        RenderingSettings,
        SpaceSettings,
        VramBudget,
        lane_follow_env_cfg,
        vram_budget,
    )
    from .lane_follow_env import DuckiebotLaneFollowEnv  # noqa: F401
    from .obstacles import DEFAULT_OBSTACLE_LAYOUT, ObstacleField, ObstacleSpec  # noqa: F401
    from .rewards import RewardTerms, RewardWeights, compute_reward  # noqa: F401
    from .terminations import TerminationFlags, TerminationState  # noqa: F401

#: Public name -> submodule that defines it.
_EXPORTS: dict[str, str] = {
    "GOLDEN_QUAT_PITCH_0": "camera_math",
    "GOLDEN_QUAT_PITCH_NOMINAL": "camera_math",
    "focal_px": "camera_math",
    "horizon_row": "camera_math",
    "pinhole_camera_kwargs": "camera_math",
    "quat_cam_ros": "camera_math",
    "quat_cam_ros_rpy": "camera_math",
    "quat_cam_ros_torch": "camera_math",
    "RewardTerms": "rewards",
    "RewardWeights": "rewards",
    "compute_reward": "rewards",
    "TerminationFlags": "terminations",
    "TerminationState": "terminations",
    "TorchActionPath": "action_path",
    "DEFAULT_OBSTACLE_LAYOUT": "obstacles",
    "ObstacleField": "obstacles",
    "ObstacleSpec": "obstacles",
    "CitySettings": "env_cfg",
    "LaneFollowSettings": "env_cfg",
    "ObstacleSettings": "env_cfg",
    "RateSettings": "env_cfg",
    "RenderingSettings": "env_cfg",
    "SpaceSettings": "env_cfg",
    "VramBudget": "env_cfg",
    "lane_follow_env_cfg": "env_cfg",
    "vram_budget": "env_cfg",
    "DuckiebotLaneFollowEnv": "lane_follow_env",
}

_SUBMODULES = (
    "action_path",
    "camera_math",
    "env_cfg",
    "lane_follow_env",
    "obstacles",
    "rewards",
    "terminations",
)

__all__ = sorted({*_EXPORTS, *_SUBMODULES})


def __getattr__(name: str) -> Any:
    """Import submodules and their public names on first access (PEP 562).

    Args:
        name: Attribute being looked up on the package.

    Returns:
        The submodule or the exported object.

    Raises:
        AttributeError: If ``name`` is not part of the public API.
    """
    if name in _SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    if name in _EXPORTS:
        module = importlib.import_module(f".{_EXPORTS[name]}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return the public API for tab completion."""
    return list(__all__)
