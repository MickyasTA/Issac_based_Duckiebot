"""Atomic checkpoint save/load with full RNG and curriculum state (SPEC v2 S6.9).

Contents of one ``.pt`` file, per the spec:

* model (both towers plus ``log_std``), optimiser, current learning rate, iteration, global step;
* running-normaliser state for ``vec``, ``vec_priv`` and the value target;
* curriculum state: ``alpha_vis``, ``alpha_dyn``, the ADR success buffers and the hard-example
  mining table. ``alpha_vis`` and ``alpha_dyn`` are MANDATORY and are checked on load, because a
  resume that silently restarted domain randomisation at alpha 0 would be invisible in the logs
  and would quietly destroy the sim-to-real result;
* RNG state for torch CPU, torch CUDA (all devices), numpy and the python stdlib ``random``;
* config hash, git commit, environment fingerprint and spec version.

Atomicity: the payload is written to ``<path>.tmp``, flushed and ``fsync``-ed, then moved into
place with :func:`os.replace`, which is atomic on Windows and POSIX alike. A crash mid-write
therefore never corrupts the previous checkpoint.

Resume guarantee, stated honestly: learner state restores EXACTLY (CPU-bitwise, asserted by
``tests/unit/test_checkpoint_resume.py``); the environment stream restores only statistically,
because PhysX state is not checkpointable and mid-episode randomisation is resampled.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from duckiebot_rl.ppo.ppo import PPO

CHECKPOINT_FORMAT_VERSION = 1
SPEC_VERSION = "SPEC_V2"
REQUIRED_CURRICULUM_KEYS: tuple[str, ...] = ("alpha_vis", "alpha_dyn")


def config_hash(config: Any) -> str:
    """Return a stable short hash of a config object.

    Args:
        config: A dataclass instance, a mapping, or anything JSON-serialisable.

    Returns:
        The first 16 hex characters of the SHA-256 of the canonical JSON encoding.
    """
    if is_dataclass(config) and not isinstance(config, type):
        payload: Any = asdict(config)
    elif isinstance(config, Mapping):
        payload = dict(config)
    else:
        payload = config
    encoded = json.dumps(payload, sort_keys=True, default=repr).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def git_commit(repo_root: str | os.PathLike[str] | None = None) -> str:
    """Return the current git commit hash, or ``"unknown"`` outside a repository.

    Args:
        repo_root: Directory to run ``git`` in; defaults to this file's repository.

    Returns:
        A 40-character hex commit hash, or ``"unknown"``.
    """
    cwd = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else "unknown"


def collect_rng_state() -> dict[str, Any]:
    """Snapshot every random number generator the learner touches.

    Returns:
        Dict with keys ``torch``, ``torch_cuda``, ``numpy`` and ``python``.
    """
    return {
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        # The LEGACY global numpy generator is snapshotted on purpose: it is the stream that
        # numpy-based env code, seeding.py and third-party callers actually consume, so a
        # np.random.Generator would checkpoint a stream nothing uses.
        "numpy": np.random.get_state(),  # noqa: NPY002
        "python": random.getstate(),
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore generators snapshotted by :func:`collect_rng_state`.

    Every generator state is forced onto the CPU first: ``torch.load(..., map_location="cuda")``
    moves the whole payload to the GPU, and both ``torch.set_rng_state`` and
    ``torch.cuda.set_rng_state_all`` require CPU ByteTensors and raise otherwise.

    CUDA state is restored only when the number of visible devices matches the snapshot, so that
    moving a checkpoint between machines degrades to a warning-free no-op on the CUDA stream
    rather than raising.

    Args:
        state: The mapping returned by :func:`collect_rng_state`.

    Raises:
        KeyError: If a mandatory generator is missing from the snapshot.
    """
    for key in ("torch", "numpy", "python"):
        if key not in state:
            raise KeyError(f"RNG state is missing mandatory key {key!r}")
    torch.set_rng_state(torch.as_tensor(state["torch"], dtype=torch.uint8).cpu())
    np.random.set_state(state["numpy"])  # noqa: NPY002 - legacy global stream, see collect_rng_state
    random.setstate(tuple(state["python"]) if isinstance(state["python"], list) else state["python"])
    cuda_state = state.get("torch_cuda") or []
    if cuda_state and torch.cuda.is_available() and len(cuda_state) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all([torch.as_tensor(s, dtype=torch.uint8).cpu() for s in cuda_state])


def _validate_curriculum(curriculum_state: Mapping[str, Any] | None) -> dict[str, Any]:
    """Check that the curriculum payload carries the mandatory alpha scalars.

    Args:
        curriculum_state: Mapping of curriculum fields.

    Returns:
        A plain dict copy of the payload.

    Raises:
        ValueError: If the payload is None or is missing ``alpha_vis`` or ``alpha_dyn``.
    """
    if curriculum_state is None:
        raise ValueError(
            "curriculum_state is mandatory (SPEC v2 S6.9): without alpha_vis and alpha_dyn a "
            "resume silently restarts domain randomisation at alpha 0"
        )
    missing = [key for key in REQUIRED_CURRICULUM_KEYS if key not in curriculum_state]
    if missing:
        raise ValueError(f"curriculum_state is missing mandatory key(s): {', '.join(missing)}")
    return dict(curriculum_state)


def _running_norm_summary(learner: PPO) -> dict[str, dict[str, Any]]:
    """Return a flat, framework-agnostic view of the learner's running normalisers.

    The keys are the names the deployment exporter looks for. Values are plain CPU tensors so a
    consumer needs neither this package nor a GPU to read them.

    Args:
        learner: The learner whose normalisers are summarised.

    Returns:
        Mapping from quantity name to ``{"mean", "var", "std", "count"}``.
    """
    summary: dict[str, dict[str, Any]] = {}
    for name, norm in (
        ("vec", learner.vec_norm),
        ("vec_priv", learner.priv_norm),
        ("value", learner.value_norm),
    ):
        summary[name] = {
            "mean": norm.mean.detach().cpu().clone(),
            "var": norm.var.detach().cpu().clone(),
            "std": norm.std().detach().cpu().clone(),
            "count": float(norm.count),
        }
    return summary


def save_checkpoint(
    path: str | os.PathLike[str],
    learner: PPO,
    iteration: int,
    global_step: int,
    curriculum_state: Mapping[str, Any] | None,
    config: Any = None,
    env_fingerprint: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write a complete, resumable checkpoint atomically.

    Args:
        path: Destination ``.pt`` path. Parent directories are created.
        learner: The PPO learner whose state is saved.
        iteration: Completed PPO iteration count.
        global_step: Total environment steps consumed.
        curriculum_state: Curriculum payload; must contain ``alpha_vis`` and ``alpha_dyn``.
        config: Optional config object; stored verbatim plus as a short hash.
        env_fingerprint: Optional environment fingerprint from ``seeding.py``.
        extra: Optional additional fields merged into the payload under ``extra``.

    Returns:
        The path that was written.

    Raises:
        ValueError: If the curriculum payload is missing or incomplete.
    """
    curriculum = _validate_curriculum(curriculum_state)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    config_dict = asdict(config) if is_dataclass(config) and not isinstance(config, type) else config

    payload: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "spec_version": SPEC_VERSION,
        "iteration": int(iteration),
        "global_step": int(global_step),
        "learner": learner.state_dict(),
        # Flat mirror of the three normalisers. The learner reloads from "learner"; this block
        # exists so that duckiebot_rl/deploy/export_onnx.py can bake the policy-side vec
        # statistics into the exported graph without importing the learner.
        "running_norm": _running_norm_summary(learner),
        "curriculum": curriculum,
        "rng": collect_rng_state(),
        "config": config_dict,
        "config_hash": config_hash(config) if config is not None else None,
        "seed": config_dict.get("seed") if isinstance(config_dict, Mapping) else None,
        "git_commit": git_commit(),
        "env_fingerprint": dict(env_fingerprint) if env_fingerprint is not None else None,
        "extra": dict(extra) if extra is not None else {},
    }

    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    tmp_path.replace(destination)
    return destination


def load_checkpoint(
    path: str | os.PathLike[str],
    learner: PPO | None = None,
    map_location: torch.device | str = "cpu",
    restore_rng: bool = True,
    require_curriculum: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint and optionally restore a learner and the RNG streams in place.

    ``weights_only=False`` is required because the payload deliberately contains numpy and stdlib
    ``random`` states, which are not tensors. Only load checkpoints you produced.

    Args:
        path: Checkpoint path.
        learner: If given, its model, optimiser, learning rate and normalisers are restored.
        map_location: Device to map tensors onto while loading.
        restore_rng: Restore the torch/numpy/python generators from the snapshot.
        require_curriculum: Enforce the presence of ``alpha_vis`` and ``alpha_dyn``.

    Returns:
        The full payload dict, so the caller can read ``iteration``, ``global_step``,
        ``curriculum``, ``config`` and ``extra``.

    Raises:
        ValueError: If the format version is unknown, or if the curriculum payload is incomplete
            while ``require_curriculum`` is True.
    """
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    version = payload.get("format_version")
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"unsupported checkpoint format_version {version!r}, expected {CHECKPOINT_FORMAT_VERSION}"
        )
    if require_curriculum:
        _validate_curriculum(payload.get("curriculum"))
    if learner is not None:
        learner.load_state_dict(payload["learner"])
    if restore_rng and "rng" in payload:
        restore_rng_state(payload["rng"])
    return payload
