"""Domain randomization and the canonical observation preprocessing chain.

Owner: [dr] (SPEC v2 S10). This package is the sim-to-real heart of the project and is shared by
three consumers - Isaac Lab training, the MuJoCo sim-to-sim harness and the deployed ROS node -
so it is pure torch/numpy with **no Isaac, gym or RL-library imports anywhere**.

Modules:
    preprocess: THE observation operator chain of SPEC v2 S4.3 (crop, matched blur, exact box
        downsample, quantize, frame ring, stacking, principal-point jitter), with a numpy twin
        that is numerically equivalent to the torch one.
    visual: per-step, per-env photometric DR (S7.2 V10-V19) plus the scene-side range book.
    dynamics: the S7.3 dynamics DR sampler (D1-D18), returning per-env parameter tensors.
    delay: the per-env variable-delay ring buffer behind D8 (actuation) and D9 (observation).
    curriculum: the two-scalar auto-DR (alpha_vis, alpha_dyn) and the ``Range`` rule that gates
        every axis in ``visual`` and ``dynamics``.

Changes to ``preprocess`` require sign-off from [ppo] + [sim2sim] + [deploy] (S10).

Torch-free import path
----------------------
``preprocess`` and ``delay`` must import in a venv WITHOUT torch (the MuJoCo venv shipped
without it, and the Jetson ROS node should not need it), so they are imported eagerly here and
guard their own torch import. ``visual``, ``dynamics`` and ``curriculum`` genuinely require
torch and are therefore exposed lazily through PEP 562 ``__getattr__``: importing this package
never pulls torch in by itself, and asking for a torch-only symbol raises the real ImportError.
"""

from __future__ import annotations

import importlib
from typing import Any

from duckiebot_rl.dr.delay import DelayBuffer
from duckiebot_rl.dr.preprocess import (
    CROP_TOP,
    FRAME_STACK_OFFSETS,
    KERNEL5,
    OBS_CHANNELS,
    OBS_H,
    OBS_W,
    RENDER_H,
    RENDER_W,
    FrameStack,
    preprocess_frame,
    preprocess_frame_np,
    tail,
    tail_np,
)

# Symbols that live in a torch-only submodule, resolved on first attribute access.
_LAZY: dict[str, str] = {
    "CurriculumCfg": "curriculum",
    "HardExampleMiner": "curriculum",
    "HardExampleMinerCfg": "curriculum",
    "Range": "curriculum",
    "RangeBook": "curriculum",
    "TwoScalarADR": "curriculum",
    "sample_book": "curriculum",
    "DynamicsDRCfg": "dynamics",
    "DynamicsParams": "dynamics",
    "DynamicsRandomizer": "dynamics",
    "default_dynamics_ranges": "dynamics",
    "quantize_encoder": "dynamics",
    "VisualDR": "visual",
    "VisualDRCfg": "visual",
    "VisualParams": "visual",
    "default_scene_ranges": "visual",
    "default_visual_ranges": "visual",
    "sample_camera_mount": "visual",
    "sample_frame_repeat": "visual",
    "sample_scene_params": "visual",
}


def __getattr__(name: str) -> Any:
    """Resolve a torch-only symbol on first access (PEP 562).

    Args:
        name: Attribute name.

    Returns:
        The requested object.

    Raises:
        AttributeError: If the name is not exported by this package.
    """
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"duckiebot_rl.dr.{module}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return the full export list, including the lazy symbols.

    Returns:
        Sorted attribute names.
    """
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "CROP_TOP",
    "FRAME_STACK_OFFSETS",
    "KERNEL5",
    "OBS_CHANNELS",
    "OBS_H",
    "OBS_W",
    "RENDER_H",
    "RENDER_W",
    "CurriculumCfg",
    "DelayBuffer",
    "DynamicsDRCfg",
    "DynamicsParams",
    "DynamicsRandomizer",
    "FrameStack",
    "HardExampleMiner",
    "HardExampleMinerCfg",
    "Range",
    "RangeBook",
    "TwoScalarADR",
    "VisualDR",
    "VisualDRCfg",
    "VisualParams",
    "default_dynamics_ranges",
    "default_scene_ranges",
    "default_visual_ranges",
    "preprocess_frame",
    "preprocess_frame_np",
    "quantize_encoder",
    "sample_book",
    "sample_camera_mount",
    "sample_frame_repeat",
    "sample_scene_params",
    "tail",
    "tail_np",
]
