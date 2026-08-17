"""Run logging and the live training dashboard.

Three layers, each usable on its own:

``run_dir``
    The run-directory contract implemented once: ids, atomic writers, tolerant readers, the
    ``status.json`` heartbeat, the ``metrics.jsonl`` appender and the checkpoint index. Pure
    standard library, so it imports in the MuJoCo venv and in CI.
``logger``
    :class:`~duckiebot_rl.viz.logger.TrainLogger`, the four-call API a training script uses.
``plots`` and ``dashboard``
    The matplotlib layer and the composite figure. Importing these needs matplotlib, which is an
    optional extra (``pip install "duckiebot-rl[viz]"``); the names below are re-exported lazily
    so that ``import duckiebot_rl.viz`` keeps working without it.

The integration contract for ``scripts/train.py`` is documented in ``docs/live_training.md``.

The live viewer sits on top of the same run directory and is documented in ``docs/live_view.md``:

``watcher``
    :class:`~duckiebot_rl.viz.watcher.CheckpointWatcher` reads the ``checkpoints/index.json`` that
    :meth:`~duckiebot_rl.viz.run_dir.RunDir.record_checkpoint` writes, and reports each new
    checkpoint once its content matches the recorded SHA-256.
``policy_host``
    :class:`~duckiebot_rl.viz.policy_host.PolicyHost` owns one
    :class:`~duckiebot_rl.ppo.networks.ActorCritic` and swaps weights into it in place, so a new
    checkpoint costs no rebuild of the network, the simulator or the window.
``render`` and ``backends``
    Video and GIF encoding, the "what the policy sees" observation snapshot, and the MuJoCo
    (default, CPU, zero VRAM) and opt-in Isaac simulator backends.

The viewer adds NOTHING to the trainer's obligations: it consumes what ``TrainLogger`` already
writes. It also never writes into ``checkpoints/``; ``--record`` writes only ``video/`` and
``obs/``. These names are lazy too, because they need torch and a simulator.
"""

from __future__ import annotations

from typing import Any

from duckiebot_rl.viz.logger import TrainLogger
from duckiebot_rl.viz.run_dir import (
    REQUIRED_METRIC_KEYS,
    SCHEMA_VERSION,
    CheckpointEntry,
    RunDir,
    RunStatus,
    find_latest_run,
    make_run_id,
)

__all__ = [
    "REQUIRED_METRIC_KEYS",
    "SCHEMA_VERSION",
    "ArchitectureMismatch",
    "CheckpointEntry",
    "CheckpointInfo",
    "CheckpointWatcher",
    "IsaacVramRefusal",
    "MujocoBackend",
    "PolicyHost",
    "RunDir",
    "RunStatus",
    "TrainLogger",
    "find_latest_run",
    "make_backend",
    "make_run_id",
    "render_dashboard",
    "watch",
    "write_obs_snapshot",
    "write_rollout",
]

_LAZY: dict[str, str] = {
    "render_dashboard": "duckiebot_rl.viz.dashboard",
    "watch": "duckiebot_rl.viz.dashboard",
    "summarise": "duckiebot_rl.viz.dashboard",
    # Live viewer. Lazy because these need torch, and the simulator backends need mujoco.
    "CheckpointInfo": "duckiebot_rl.viz.watcher",
    "CheckpointWatcher": "duckiebot_rl.viz.watcher",
    "ArchitectureMismatch": "duckiebot_rl.viz.policy_host",
    "HostState": "duckiebot_rl.viz.policy_host",
    "PolicyHost": "duckiebot_rl.viz.policy_host",
    "RolloutArtifacts": "duckiebot_rl.viz.render",
    "write_obs_snapshot": "duckiebot_rl.viz.render",
    "write_rollout": "duckiebot_rl.viz.render",
    "IsaacVramRefusal": "duckiebot_rl.viz.backends",
    "MujocoBackend": "duckiebot_rl.viz.backends",
    "make_backend": "duckiebot_rl.viz.backends",
}


def __getattr__(name: str) -> Any:
    """Import the matplotlib-, torch- and simulator-backed helpers only when asked for.

    Args:
        name: Attribute being looked up.

    Returns:
        The resolved attribute.

    Raises:
        AttributeError: If ``name`` is not exported by this package.
    """
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_name), name)
