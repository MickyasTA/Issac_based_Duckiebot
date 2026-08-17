"""Per-env variable-delay ring buffer (SPEC v2 S5.3 step 2, S7.3 axes D8 and D9).

Why this is re-implemented instead of using ``isaaclab.actuators.DelayedPDActuatorCfg``
--------------------------------------------------------------------------------------
The critic (01_CRITIC_NOTES.md, item on ``DelayedPDActuatorCfg``) is right that Isaac Lab 2.3.2
ships a delayed actuator with ``min_delay``/``max_delay``. SPEC v2 S1 row 20 records the decision
to reimplement anyway, for four reasons that the Isaac class cannot satisfy:

1. **Three consumers, one class.** The identical delay must run inside Isaac, inside the MuJoCo
   sim-to-sim harness (a *separate* venv that has no ``isaaclab`` and, before M0, no torch at
   all) and inside the deployed ROS node on the Jetson. ``DelayedPDActuatorCfg`` cannot be
   imported in either of the latter two, and a sim-to-sim study whose delay model differs
   between the two simulators measures its own bug.
2. **Granularity.** ``DelayedPDActuatorCfg`` delays by whole *physics* steps inside the actuator
   model. D8 is specified in *control* steps (1/15 s) plus a sub-step fraction U(0, 0.9), i.e.
   the delay is not an integer number of buffer slots; we linearly interpolate between the two
   neighbouring slots.
3. **Observations, not just actuators.** D9 delays the camera *stream* before frame stacking
   (S4.3 step 9). No actuator class can do that; the same buffer serves both so the two delay
   paths cannot drift apart.
4. **Testability.** This class is pure array code; the whole delay contract is unit tested on
   CPU with no simulator, which is a hard requirement of the project.

Backends
--------
The buffer runs on torch tensors *or* numpy arrays with the same code path and the same
semantics. ``backend="numpy"`` exists because ``d:/Personal/personal/mujoco_venv`` shipped
without torch (critic finding) and because the ROS node on a Jetson Nano should not need torch
to reproduce the training-time delay.

Semantics
---------
After ``push(x_t)``, ``tap(k)`` returns ``x_{t-k}`` (so ``tap(0)`` is the value just pushed) and
``get()`` returns ``(1-f) * x_{t-k} + f * x_{t-k-1}`` for the per-env integer delay ``k`` and
fraction ``f``. Slots are pre-filled by :meth:`reset`, so no read ever returns uninitialised
memory.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:  # pragma: no cover - exercised implicitly by the backend selection
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - the mujoco_venv / Jetson path
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False

__all__ = ["DelayBuffer"]

Array = Any
"""Either a ``torch.Tensor`` or a ``numpy.ndarray``; the class is agnostic."""


class DelayBuffer:
    """A per-env ring buffer implementing an independently variable delay per env.

    Args:
        num_envs: Number of parallel envs (the second buffer axis).
        feature_shape: Shape of one env's payload, e.g. ``(2,)`` for wheel targets or
            ``(48, 96, 3)`` for an observation frame. ``()`` means a scalar per env.
        max_delay: Largest integer delay (in pushes) that may be requested. The ring holds
            ``max_delay + 2`` slots so that sub-step interpolation can still read ``k + 1``.
        dtype: Backend dtype of the payload. Defaults to float32 for the active backend.
        device: Torch device (ignored by the numpy backend).
        backend: ``"torch"`` or ``"numpy"``.
        interpolate: If False, :meth:`get` ignores the sub-step fraction and returns ``tap(k)``.
            Automatically forced off for integer dtypes (an interpolated uint8 frame would be a
            silent blend of two camera frames).

    Raises:
        ValueError: If ``max_delay`` is negative or the backend name is unknown.
        RuntimeError: If the torch backend is requested but torch is not importable.
    """

    def __init__(
        self,
        num_envs: int,
        feature_shape: tuple[int, ...] = (),
        max_delay: int = 3,
        *,
        dtype: Any = None,
        device: Any = None,
        backend: str = "torch",
        interpolate: bool = True,
    ) -> None:
        if max_delay < 0:
            raise ValueError(f"max_delay must be >= 0, got {max_delay}")
        if backend not in ("torch", "numpy"):
            raise ValueError(f"unknown backend {backend!r} (expected 'torch' or 'numpy')")
        if backend == "torch" and not _HAS_TORCH:
            raise RuntimeError("backend='torch' requested but torch is not importable")

        self.num_envs = int(num_envs)
        self.feature_shape = tuple(feature_shape)
        self.max_delay = int(max_delay)
        self.depth = self.max_delay + 2
        self.backend = backend
        self._is_np = backend == "numpy"
        self.device = device

        if self._is_np:
            self.dtype = np.float32 if dtype is None else np.dtype(dtype)
            self._buf: Array = np.zeros((self.depth, self.num_envs, *self.feature_shape), self.dtype)
            self._env_idx: Array = np.arange(self.num_envs)
            self._delay_steps: Array = np.zeros(self.num_envs, dtype=np.int64)
            self._delay_frac: Array = np.zeros(self.num_envs, dtype=np.float32)
            is_float = np.issubdtype(np.dtype(self.dtype), np.floating)
        else:
            self.dtype = torch.float32 if dtype is None else dtype
            self._buf = torch.zeros(
                (self.depth, self.num_envs, *self.feature_shape), dtype=self.dtype, device=device
            )
            self._env_idx = torch.arange(self.num_envs, device=device)
            self._delay_steps = torch.zeros(self.num_envs, dtype=torch.long, device=device)
            self._delay_frac = torch.zeros(self.num_envs, dtype=torch.float32, device=device)
            is_float = torch.is_floating_point(self._buf)

        self.interpolate = bool(interpolate) and is_float
        self._ptr = 0

    # ------------------------------------------------------------------ properties

    @property
    def delay_steps(self) -> Array:
        """Per-env integer delay, shape ``(num_envs,)``."""
        return self._delay_steps

    @property
    def delay_frac(self) -> Array:
        """Per-env sub-step delay fraction in [0, 1), shape ``(num_envs,)``."""
        return self._delay_frac

    # ------------------------------------------------------------------ configuration

    def set_delay(self, steps: Array | int, frac: Array | float | None = None) -> None:
        """Set the per-env delay.

        Args:
            steps: Integer delay per env (scalar broadcast allowed). Clamped to
                ``[0, max_delay]``.
            frac: Sub-step fraction in [0, 1) per env (scalar broadcast allowed). ``None``
                leaves the current fractions untouched.

        Raises:
            ValueError: If a delay is negative or exceeds ``max_delay``.
        """
        s = self._as_array(steps, integer=True)
        if self.num_envs > 0:
            lo, hi = int(s.min()), int(s.max())
            if lo < 0 or hi > self.max_delay:
                raise ValueError(f"delay steps must be in [0, {self.max_delay}], got [{lo}, {hi}]")
        self._delay_steps = s
        if frac is not None:
            f = self._as_array(frac, integer=False)
            self._delay_frac = self._clip01(f)

    # ------------------------------------------------------------------ core ops

    def reset(self, env_ids: Array | None = None, value: Array | None = None) -> None:
        """Refill every slot of the ring for the given envs.

        Called on episode reset so that the first steps of a new episode never see frames or
        wheel targets from the previous episode.

        Args:
            env_ids: Envs to reset; ``None`` means all envs.
            value: Payload to write into every slot, shape ``(len(env_ids), *feature_shape)``
                or broadcastable to it. ``None`` writes zeros.
        """
        if env_ids is None:
            if value is None:
                self._buf[:] = 0
            else:
                self._buf[:] = self._to_buf_dtype(value)
            return
        idx = self._as_array(env_ids, integer=True)
        if value is None:
            self._buf[:, idx] = 0
        else:
            self._buf[:, idx] = self._to_buf_dtype(value)

    def push(self, x: Array) -> None:
        """Append one time step.

        Args:
            x: Payload of shape ``(num_envs, *feature_shape)``.

        Raises:
            ValueError: If the payload shape does not match the buffer.
        """
        expected = (self.num_envs, *self.feature_shape)
        if tuple(x.shape) != expected:
            raise ValueError(f"push expected shape {expected}, got {tuple(x.shape)}")
        self._ptr = (self._ptr + 1) % self.depth
        self._buf[self._ptr] = self._to_buf_dtype(x)

    def tap(self, k: Array | int) -> Array:
        """Read the value pushed ``k`` steps ago.

        Args:
            k: Integer delay per env, or a python int broadcast to all envs. Values are
                clamped to ``[0, depth - 1]``.

        Returns:
            An array of shape ``(num_envs, *feature_shape)``.
        """
        kk = self._as_array(k, integer=True)
        kk = self._clip_int(kk, 0, self.depth - 1)
        idx = (self._ptr - kk) % self.depth
        return self._buf[idx, self._env_idx]

    def get(self) -> Array:
        """Read the delayed value using the configured per-env delay.

        Returns:
            ``tap(k)`` for integer payloads, or the sub-step interpolation
            ``(1 - f) * tap(k) + f * tap(k + 1)`` for float payloads.
        """
        a = self.tap(self._delay_steps)
        if not self.interpolate:
            return a
        b = self.tap(self._delay_steps + 1)
        f = self._delay_frac
        for _ in self.feature_shape:
            f = f[..., None]
        return a + (b - a) * self._to_buf_dtype(f)

    def step(self, x: Array) -> Array:
        """Push one payload and return the delayed read.

        Args:
            x: Payload of shape ``(num_envs, *feature_shape)``.

        Returns:
            The delayed payload for this step.
        """
        self.push(x)
        return self.get()

    # ------------------------------------------------------------------ serialization

    def state_dict(self) -> dict[str, Any]:
        """Serialize the buffer (used by checkpoint/resume tests and the deploy node).

        Returns:
            A dict with the ring contents, the write pointer and the per-env delays.
        """
        return {
            "buffer": self._buf,
            "ptr": self._ptr,
            "delay_steps": self._delay_steps,
            "delay_frac": self._delay_frac,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a state produced by :meth:`state_dict`.

        Args:
            state: The dict returned by :meth:`state_dict`.
        """
        self._buf[...] = self._to_buf_dtype(state["buffer"])
        self._ptr = int(state["ptr"])
        self._delay_steps = self._as_array(state["delay_steps"], integer=True)
        self._delay_frac = self._as_array(state["delay_frac"], integer=False)

    # ------------------------------------------------------------------ backend helpers

    def _as_array(self, x: Array | float | int, *, integer: bool) -> Array:
        """Coerce a scalar or array to a per-env backend array.

        Args:
            x: Scalar or array-like of shape ``(num_envs,)``.
            integer: Whether the result must be an integer array.

        Returns:
            A backend array of shape ``(num_envs,)``.
        """
        if self._is_np:
            dt = np.int64 if integer else np.float32
            arr = np.asarray(x)
            if arr.ndim == 0:
                arr = np.full(self.num_envs, arr)
            return arr.astype(dt, copy=False)
        dt = torch.long if integer else torch.float32
        if not torch.is_tensor(x):
            arr = torch.as_tensor(x, device=self.device)
        else:
            arr = x.to(self.device) if self.device is not None else x
        if arr.ndim == 0:
            arr = arr.expand(self.num_envs)
        return arr.to(dt)

    def _to_buf_dtype(self, x: Array) -> Array:
        """Cast a payload to the buffer dtype/device.

        Args:
            x: Payload array.

        Returns:
            The payload in the buffer's dtype (and device, for torch).
        """
        if self._is_np:
            return np.asarray(x, dtype=self.dtype)
        t = x if torch.is_tensor(x) else torch.as_tensor(x, device=self.device)
        return t.to(dtype=self._buf.dtype, device=self._buf.device)

    def _clip01(self, x: Array) -> Array:
        """Clamp to [0, 1).

        Args:
            x: Array to clamp.

        Returns:
            The clamped array.
        """
        hi = float(np.nextafter(np.float32(1.0), np.float32(0.0)))
        return np.clip(x, 0.0, hi) if self._is_np else x.clamp(0.0, hi)

    def _clip_int(self, x: Array, lo: int, hi: int) -> Array:
        """Clamp an integer array.

        Args:
            x: Array to clamp.
            lo: Inclusive lower bound.
            hi: Inclusive upper bound.

        Returns:
            The clamped array.
        """
        return np.clip(x, lo, hi) if self._is_np else x.clamp(lo, hi)
