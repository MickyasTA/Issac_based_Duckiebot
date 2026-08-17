"""THE canonical observation preprocessing chain (SPEC v2 S4.3).

This module is the single source of truth for turning a rendered/captured RGB frame into the
policy observation. Exactly the same operator chain runs in three places:

* Isaac Lab training (torch, GPU, batched over N envs, every control step),
* the MuJoCo sim-to-sim harness (torch or numpy, CPU),
* the deployed ROS node on the Duckiebot (numpy, CPU, no torch required).

The critic's first-order finding was that a sim/real *resize-kernel* mismatch silently changes
what a 24 mm dashed yellow line looks like. SPEC v2 S4.3 fixes it by rendering at 2x the
observation resolution and using one fixed low-pass + exact box downsample everywhere. That is
implemented here once; nobody re-implements a resize.

Operator chain (S4.3 steps 2-10; step numbers are quoted from the spec)::

    2.  float  : f = frame.float() / 255, NHWC uint8 -> NCHW float32
    3.  DR     : photometric domain randomization (duckiebot_rl.dr.visual) - TRAIN ONLY
    4.  DR     : principal-point jitter V11, integer shift at RENDER res - TRAIN ONLY
    5.  blur   : fixed separable 5-tap Gaussian (sigma 0.6 px), replicate padding - ALWAYS
    6.  box    : exact 2x2 average pool 192x128 -> 96x64                        - ALWAYS
    7.  crop   : drop the top CROP_TOP=16 rows -> 96x48                         - ALWAYS
    8.  uint8  : round(f*255).clamp(0,255)                                      - ALWAYS
    9.  ring   : push into the per-env frame ring; D9 observation latency shifts the read index
    10. stack  : channel-concat frames (t, t-2, t-4) -> (N, 48, 96, 9) uint8

Steps 5-8 are the *tail*: they are bit-for-bit identical on every path, which is what makes the
sim-to-real observation comparable at all. Steps 3 and 4 are training-only. The robot adds, in
front of the tail, JPEG decode -> BGR2RGB -> Gaussian pre-blur (sigma
:data:`ROBOT_PREBLUR_SIGMA` at 640x480) -> ``cv2.remap`` to the canonical 192x128 camera.

Principal-point jitter (V11) lives here and *not* in USD: Isaac Sim silently ignores camera
aperture offsets (``sensors.py`` hardcodes both offsets to 0.0 and warns; internal ticket
OM-42611), so a USD-side cx/cy randomization is a no-op. It is a torch/numpy pixel shift.

Numpy parity
------------
Every torch function has a ``*_np`` twin with identical semantics; ``tests/unit/test_preprocess``
asserts they agree to <= 1e-6 in float and <= 1 LSB in uint8. The numpy path exists because the
MuJoCo venv shipped without torch (critic finding) and because the Jetson ROS node should not
need torch to reproduce training-time preprocessing.

Frame-ring depth
----------------
S4.3 step 9 says "per-env frame ring (depth 5)". With the D9 observation latency of up to 2
control steps *and* the stack offset of 4, the oldest slot actually read is ``t - 6``, so a
depth-5 ring would alias. :class:`FrameStack` therefore sizes its ring from
``max_obs_delay + max(offsets) + 1`` (= 7 at the spec values) and the "depth 5" figure in S4.3 is
treated as an arithmetic slip in the spec text, not as a design decision.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

from duckiebot_rl.dr.delay import DelayBuffer

try:  # pragma: no cover - exercised implicitly by the backend selection
    import torch
    import torch.nn.functional as F

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - the mujoco_venv / Jetson path
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _HAS_TORCH = False

__all__ = [
    "BOX",
    "CROP_TOP",
    "FRAME_STACK_OFFSETS",
    "KERNEL5",
    "MAX_OBS_DELAY",
    "OBS_CHANNELS",
    "OBS_H",
    "OBS_W",
    "RENDER_H",
    "RENDER_W",
    "ROBOT_PREBLUR_SIGMA",
    "FrameStack",
    "blur5",
    "blur5_np",
    "box_downsample",
    "box_downsample_np",
    "crop_rows",
    "crop_rows_np",
    "gaussian_kernel1d",
    "preprocess_frame",
    "preprocess_frame_np",
    "quantize_uint8",
    "quantize_uint8_np",
    "shift_principal_point",
    "shift_principal_point_np",
    "tail",
    "tail_cv2",
    "tail_np",
]

# --------------------------------------------------------------------------------------------
# Constants (SPEC v2 S4.1 and S4.3). Changing any of these changes the trained observation and
# requires sign-off from [ppo] + [sim2sim] + [deploy] (S10).
# --------------------------------------------------------------------------------------------

RENDER_W: int = 192
"""Canonical render width in px (2x supersample of the observation)."""

RENDER_H: int = 128
"""Canonical render height in px."""

OBS_W: int = 96
"""Observation width in px after the 2x2 box downsample."""

OBS_H: int = 48
"""Observation height in px after the top crop."""

CROP_TOP: int = 16
"""Rows removed from the top, AFTER the downsample (S4.3 step 7)."""

BOX: int = 2
"""Box-downsample factor. RENDER_W / BOX == OBS_W and RENDER_H / BOX == OBS_H + CROP_TOP."""

KERNEL5: tuple[float, float, float, float, float] = (0.00256, 0.16555, 0.66378, 0.16555, 0.00256)
"""Fixed separable low-pass applied at render resolution in both axes (sigma 0.6 px).

This is the *matched* anti-alias kernel: it is not a domain-randomization axis and it is never
skipped, on any path. It is what makes the Isaac rasteriser (1 spp + FXAA) and the robot's
``cv2.remap`` produce comparable high-frequency content before the box downsample.
"""

FRAME_STACK_OFFSETS: tuple[int, ...] = (0, 2, 4)
"""Control-step offsets of the stacked frames: (t, t-2, t-4)."""

MAX_OBS_DELAY: int = 2
"""Largest D9 observation latency, in control steps (S7.3: U{0, 1, 2})."""

OBS_CHANNELS: int = 3 * len(FRAME_STACK_OFFSETS)
"""Channel count of the stacked observation (9)."""

ROBOT_PREBLUR_SIGMA: float = 1.0
"""Gaussian sigma (px, at 640x480) applied on the robot BEFORE the rectification remap.

S4.3 robot path: the remap decimates by ~3.3x, which no low-pass in ``cv2.remap`` compensates.
This pre-blur makes the robot's total pre-decimation low-pass approximate the sim path.
"""


def _check_render_shape(shape: tuple[int, ...]) -> None:
    """Validate an incoming NHWC frame batch shape.

    Args:
        shape: The shape tuple of the frame batch.

    Raises:
        ValueError: If the batch is not ``(N, RENDER_H, RENDER_W, 3)``.
    """
    if len(shape) != 4 or shape[1] != RENDER_H or shape[2] != RENDER_W or shape[3] != 3:
        raise ValueError(f"expected NHWC frames of shape (N, {RENDER_H}, {RENDER_W}, 3), got {tuple(shape)}")


def gaussian_kernel1d(sigma: float, radius: int = 2) -> tuple[float, ...]:
    """Build a normalized 1-D Gaussian kernel.

    Provided so that the robot-side pre-blur (:data:`ROBOT_PREBLUR_SIGMA`) and any documentation
    of :data:`KERNEL5` use one definition instead of a hand-typed table.

    Args:
        sigma: Standard deviation in pixels. Must be > 0.
        radius: Kernel half-width; the kernel has ``2 * radius + 1`` taps.

    Returns:
        The kernel taps, summing to 1.

    Raises:
        ValueError: If ``sigma <= 0`` or ``radius < 1``.
    """
    if sigma <= 0.0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    if radius < 1:
        raise ValueError(f"radius must be >= 1, got {radius}")
    taps = [math.exp(-0.5 * (i / sigma) ** 2) for i in range(-radius, radius + 1)]
    total = sum(taps)
    return tuple(t / total for t in taps)


# --------------------------------------------------------------------------------------------
# Torch implementation (training path)
# --------------------------------------------------------------------------------------------


def blur5(x: Any, kernel: tuple[float, ...] = KERNEL5) -> Any:
    """Apply the fixed separable low-pass (S4.3 step 5), torch.

    Args:
        x: NCHW float32 tensor.
        kernel: Separable 1-D kernel; defaults to :data:`KERNEL5`.

    Returns:
        The blurred NCHW tensor, same shape and dtype.
    """
    n_ch = x.shape[1]
    k = torch.tensor(kernel, dtype=x.dtype, device=x.device)
    r = (len(kernel) - 1) // 2
    kw = k.view(1, 1, 1, -1).expand(n_ch, 1, 1, len(kernel))
    kh = k.view(1, 1, -1, 1).expand(n_ch, 1, len(kernel), 1)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), kw, groups=n_ch)
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), kh, groups=n_ch)
    return x


def box_downsample(x: Any, factor: int = BOX) -> Any:
    """Exact box (average) downsample (S4.3 step 6), torch.

    At an integer factor this is numerically identical to ``cv2.INTER_AREA``.

    Args:
        x: NCHW float32 tensor whose H and W are divisible by ``factor``.
        factor: Downsample factor.

    Returns:
        The downsampled NCHW tensor.

    Raises:
        ValueError: If H or W is not divisible by ``factor``.
    """
    if x.shape[-2] % factor or x.shape[-1] % factor:
        raise ValueError(f"shape {tuple(x.shape)} not divisible by box factor {factor}")
    return F.avg_pool2d(x, kernel_size=factor, stride=factor)


def crop_rows(x: Any, top: int = CROP_TOP, height: int = OBS_H) -> Any:
    """Crop rows from the top (S4.3 step 7), torch.

    Args:
        x: NCHW tensor.
        top: First kept row.
        height: Number of kept rows.

    Returns:
        The cropped NCHW tensor.

    Raises:
        ValueError: If the crop window leaves the image.
    """
    if top < 0 or top + height > x.shape[-2]:
        raise ValueError(f"crop [{top}, {top + height}) outside height {x.shape[-2]}")
    return x[..., top : top + height, :]


def quantize_uint8(x: Any) -> Any:
    """Quantize a float image in [0, 1] to uint8 (S4.3 step 8), torch.

    Args:
        x: NCHW float tensor.

    Returns:
        An NHWC uint8 tensor (the layout the rollout buffer stores).
    """
    q = torch.round(x * 255.0).clamp_(0.0, 255.0).to(torch.uint8)
    return q.permute(0, 2, 3, 1).contiguous()


def shift_principal_point(x: Any, dx: Any, dy: Any) -> Any:
    """Integer principal-point jitter with replicate padding (V11, S4.3 step 4), torch.

    A positive ``dx`` moves the image content toward +u (right), which is equivalent to moving
    the principal point by ``-dx``. Padding replicates the border, so no black bars appear.
    Fully batched: every env may use a different shift, with no python loop.

    Args:
        x: NCHW float tensor at RENDER resolution.
        dx: Per-env integer column shift, shape ``(N,)``.
        dy: Per-env integer row shift, shape ``(N,)``.

    Returns:
        The shifted NCHW tensor.
    """
    n, c, h, w = x.shape
    cols = (torch.arange(w, device=x.device).view(1, w) - dx.to(x.device).view(n, 1)).clamp(0, w - 1)
    x = x.gather(3, cols.view(n, 1, 1, w).expand(n, c, h, w))
    rows = (torch.arange(h, device=x.device).view(1, h) - dy.to(x.device).view(n, 1)).clamp(0, h - 1)
    return x.gather(2, rows.view(n, 1, h, 1).expand(n, c, h, w))


def tail(x: Any) -> Any:
    """Run the always-on tail (S4.3 steps 5-8), torch.

    Args:
        x: NCHW float32 tensor in [0, 1] at RENDER resolution.

    Returns:
        NHWC uint8 tensor of shape ``(N, OBS_H, OBS_W, 3)``.
    """
    return quantize_uint8(crop_rows(box_downsample(blur5(x))))


def preprocess_frame(
    frame: Any,
    *,
    photometric: Callable[[Any], Any] | None = None,
    pp_shift: tuple[Any, Any] | None = None,
) -> Any:
    """Run S4.3 steps 2-8 on a batch of rendered frames, torch.

    Args:
        frame: NHWC uint8 tensor ``(N, RENDER_H, RENDER_W, 3)``. The caller is responsible for
            cloning Isaac's live camera buffer BEFORE calling this (S4.3 step 1).
        photometric: Optional step-3 photometric DR operator on the NCHW float tensor. Training
            only; eval and deploy pass ``None``.
        pp_shift: Optional ``(dx, dy)`` per-env integer shifts for V11. Training only.

    Returns:
        NHWC uint8 tensor ``(N, OBS_H, OBS_W, 3)``.
    """
    _check_render_shape(tuple(frame.shape))
    x = frame.to(torch.float32).permute(0, 3, 1, 2) / 255.0
    if photometric is not None:
        x = photometric(x)
    if pp_shift is not None:
        x = shift_principal_point(x, pp_shift[0], pp_shift[1])
    return tail(x)


# --------------------------------------------------------------------------------------------
# Numpy twin (MuJoCo venv without torch, and the Jetson ROS node)
# --------------------------------------------------------------------------------------------


def blur5_np(x: np.ndarray, kernel: tuple[float, ...] = KERNEL5) -> np.ndarray:
    """Numpy twin of :func:`blur5`.

    Args:
        x: NCHW float32 array.
        kernel: Separable 1-D kernel.

    Returns:
        The blurred NCHW float32 array.
    """
    k = np.asarray(kernel, dtype=np.float32)
    r = (len(kernel) - 1) // 2
    pad_w = np.pad(x, ((0, 0), (0, 0), (0, 0), (r, r)), mode="edge")
    out = np.zeros_like(x)
    for i in range(len(kernel)):
        out += k[i] * pad_w[..., i : i + x.shape[-1]]
    pad_h = np.pad(out, ((0, 0), (0, 0), (r, r), (0, 0)), mode="edge")
    out2 = np.zeros_like(x)
    for i in range(len(kernel)):
        out2 += k[i] * pad_h[..., i : i + x.shape[-2], :]
    return out2


def box_downsample_np(x: np.ndarray, factor: int = BOX) -> np.ndarray:
    """Numpy twin of :func:`box_downsample` (exact, factor 2 only).

    Args:
        x: NCHW float32 array whose H and W are divisible by ``factor``.
        factor: Downsample factor.

    Returns:
        The downsampled NCHW float32 array.

    Raises:
        ValueError: If H or W is not divisible by ``factor``.
    """
    if x.shape[-2] % factor or x.shape[-1] % factor:
        raise ValueError(f"shape {x.shape} not divisible by box factor {factor}")
    if factor != 2:
        n, c, h, w = x.shape
        blocks = x.reshape(n, c, h // factor, factor, w // factor, factor)
        return blocks.sum(axis=(3, 5), dtype=np.float32) / np.float32(factor * factor)
    # Explicit row-major accumulation, matching the order torch's avg_pool2d uses, so the two
    # backends agree bit-for-bit rather than merely to float tolerance.
    acc = x[..., 0::2, 0::2] + x[..., 0::2, 1::2]
    acc = acc + x[..., 1::2, 0::2]
    acc = acc + x[..., 1::2, 1::2]
    return (acc * np.float32(0.25)).astype(np.float32, copy=False)


def crop_rows_np(x: np.ndarray, top: int = CROP_TOP, height: int = OBS_H) -> np.ndarray:
    """Numpy twin of :func:`crop_rows`.

    Args:
        x: NCHW array.
        top: First kept row.
        height: Number of kept rows.

    Returns:
        The cropped NCHW array.

    Raises:
        ValueError: If the crop window leaves the image.
    """
    if top < 0 or top + height > x.shape[-2]:
        raise ValueError(f"crop [{top}, {top + height}) outside height {x.shape[-2]}")
    return x[..., top : top + height, :]


def quantize_uint8_np(x: np.ndarray) -> np.ndarray:
    """Numpy twin of :func:`quantize_uint8`.

    ``np.rint`` and ``torch.round`` both round half to even, which is what keeps the two paths
    within 1 LSB.

    Args:
        x: NCHW float array in [0, 1].

    Returns:
        NHWC uint8 array.
    """
    q = np.clip(np.rint(x * np.float32(255.0)), 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(np.transpose(q, (0, 2, 3, 1)))


def shift_principal_point_np(x: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Numpy twin of :func:`shift_principal_point`.

    Args:
        x: NCHW float array at RENDER resolution.
        dx: Per-env integer column shift, shape ``(N,)``.
        dy: Per-env integer row shift, shape ``(N,)``.

    Returns:
        The shifted NCHW array.
    """
    n, c, h, w = x.shape
    cols = np.clip(np.arange(w)[None, :] - np.asarray(dx).reshape(n, 1), 0, w - 1)
    x = np.take_along_axis(x, np.broadcast_to(cols[:, None, None, :], (n, c, h, w)), axis=3)
    rows = np.clip(np.arange(h)[None, :] - np.asarray(dy).reshape(n, 1), 0, h - 1)
    return np.take_along_axis(x, np.broadcast_to(rows[:, None, :, None], (n, c, h, w)), axis=2)


def tail_np(x: np.ndarray) -> np.ndarray:
    """Numpy twin of :func:`tail`.

    Args:
        x: NCHW float32 array in [0, 1] at RENDER resolution.

    Returns:
        NHWC uint8 array ``(N, OBS_H, OBS_W, 3)``.
    """
    return quantize_uint8_np(crop_rows_np(box_downsample_np(blur5_np(x))))


def preprocess_frame_np(
    frame: np.ndarray,
    *,
    photometric: Callable[[np.ndarray], np.ndarray] | None = None,
    pp_shift: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Numpy twin of :func:`preprocess_frame`.

    Args:
        frame: NHWC uint8 array ``(N, RENDER_H, RENDER_W, 3)``.
        photometric: Optional step-3 operator on the NCHW float array.
        pp_shift: Optional ``(dx, dy)`` per-env integer shifts (V11).

    Returns:
        NHWC uint8 array ``(N, OBS_H, OBS_W, 3)``.
    """
    _check_render_shape(tuple(frame.shape))
    x = np.transpose(frame.astype(np.float32), (0, 3, 1, 2)) / np.float32(255.0)
    if photometric is not None:
        x = photometric(x)
    if pp_shift is not None:
        x = shift_principal_point_np(x, pp_shift[0], pp_shift[1])
    return tail_np(x)


def tail_cv2(frame: np.ndarray) -> np.ndarray:
    """Reference implementation of the tail using OpenCV, for the parity test only.

    Production code never calls this: the robot runs :func:`preprocess_frame_np` so that the
    deployed tail is the *same* code as the trained tail. This function exists to prove, in
    ``tests/unit/test_preprocess.py``, that our fixed blur + box pool really does equal
    ``cv2.GaussianBlur`` + ``cv2.INTER_AREA`` at an integer 2x factor (the resize-kernel-parity
    claim of S4.3).

    Args:
        frame: A single HWC uint8 frame ``(RENDER_H, RENDER_W, 3)``.

    Returns:
        An HWC uint8 array ``(OBS_H, OBS_W, 3)``.

    Raises:
        ImportError: If OpenCV is not installed.
    """
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("tail_cv2 requires opencv-python-headless") from exc
    k = np.asarray(KERNEL5, dtype=np.float64).reshape(-1, 1)
    f = frame.astype(np.float32) / np.float32(255.0)
    f = cv2.sepFilter2D(f, -1, k, k, borderType=cv2.BORDER_REPLICATE)
    f = cv2.resize(f, (OBS_W, RENDER_H // BOX), interpolation=cv2.INTER_AREA)
    f = f[CROP_TOP : CROP_TOP + OBS_H, :, :]
    return np.clip(np.rint(f * np.float32(255.0)), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------------------------
# Frame ring + stacking (S4.3 steps 9-10)
# --------------------------------------------------------------------------------------------


class FrameStack:
    """Per-env frame ring implementing D9 observation latency and the (t, t-2, t-4) stack.

    The ring stores *preprocessed uint8 frames* (the output of :func:`preprocess_frame`) and the
    stacked observation is produced by channel-concatenating three taps. Delaying the camera
    stream *before* stacking is the resolution SPEC v2 S1 row 34 gives for the critic's D9/stack
    ambiguity: the whole stack slides, the offsets between its frames never change.

    Args:
        num_envs: Number of parallel envs.
        obs_hw: ``(H, W)`` of one preprocessed frame.
        channels: Channels per frame (3).
        offsets: Stack offsets in control steps, newest first.
        max_obs_delay: Largest D9 latency in control steps.
        device: Torch device (torch backend only).
        backend: ``"torch"`` or ``"numpy"``.

    Raises:
        ValueError: If ``offsets`` is empty or contains a negative value.
    """

    def __init__(
        self,
        num_envs: int,
        *,
        obs_hw: tuple[int, int] = (OBS_H, OBS_W),
        channels: int = 3,
        offsets: tuple[int, ...] = FRAME_STACK_OFFSETS,
        max_obs_delay: int = MAX_OBS_DELAY,
        device: Any = None,
        backend: str = "torch",
    ) -> None:
        if not offsets:
            raise ValueError("offsets must be non-empty")
        if min(offsets) < 0:
            raise ValueError(f"offsets must be >= 0, got {offsets}")
        self.num_envs = int(num_envs)
        self.obs_hw = (int(obs_hw[0]), int(obs_hw[1]))
        self.channels = int(channels)
        self.offsets = tuple(int(o) for o in offsets)
        self.max_obs_delay = int(max_obs_delay)
        self.backend = backend
        self._is_np = backend == "numpy"
        dtype = np.uint8 if self._is_np else torch.uint8
        self._ring = DelayBuffer(
            num_envs,
            (self.obs_hw[0], self.obs_hw[1], self.channels),
            max_delay=self.max_obs_delay + max(self.offsets),
            dtype=dtype,
            device=device,
            backend=backend,
            interpolate=False,
        )
        self._delay = self._ring.delay_steps

    @property
    def stacked_shape(self) -> tuple[int, int, int, int]:
        """Shape of the stacked observation, ``(N, H, W, 3 * len(offsets))``."""
        return (self.num_envs, self.obs_hw[0], self.obs_hw[1], self.channels * len(self.offsets))

    @property
    def obs_delay(self) -> Any:
        """Per-env D9 observation latency in control steps, shape ``(N,)``."""
        return self._ring.delay_steps

    def set_obs_delay(self, delays: Any) -> None:
        """Set the per-env D9 observation latency.

        Args:
            delays: Integer latency per env (or a scalar), in ``[0, max_obs_delay]``.
        """
        self._ring.set_delay(delays)
        self._delay = self._ring.delay_steps

    def reset(self, env_ids: Any = None, frame: Any = None) -> None:
        """Refill the ring for the given envs (called on episode reset).

        Args:
            env_ids: Envs to reset; ``None`` means all.
            frame: Preprocessed uint8 frame(s) to fill every slot with, shape
                ``(len(env_ids), H, W, C)``. ``None`` fills zeros.
        """
        self._ring.reset(env_ids, frame)

    def push(self, frame: Any, repeat_mask: Any = None) -> None:
        """Push one preprocessed frame batch (S4.3 step 9).

        Args:
            frame: uint8 batch ``(N, H, W, C)``.
            repeat_mask: Optional bool mask ``(N,)``; where True the previous frame is pushed
                again instead of ``frame``. This is DR axis V19 (frame repeat, p ~ U(0, 0.10)):
                time still advances, the sensor just did not deliver a new image.
        """
        if repeat_mask is not None:
            prev = self._ring.tap(0)
            if self._is_np:
                m = np.asarray(repeat_mask).reshape(-1, 1, 1, 1)
                frame = np.where(m, prev, frame)
            else:
                m = repeat_mask.to(torch.bool).view(-1, 1, 1, 1)
                frame = torch.where(m, prev, frame)
        self._ring.push(frame)

    def get(self) -> Any:
        """Return the stacked observation (S4.3 step 10).

        Returns:
            uint8 array/tensor ``(N, H, W, 3 * len(offsets))``, newest frame in channels 0-2.
        """
        taps = [self._ring.tap(self._delay + o) for o in self.offsets]
        return np.concatenate(taps, axis=-1) if self._is_np else torch.cat(taps, dim=-1)

    def step(self, frame: Any, repeat_mask: Any = None) -> Any:
        """Push a frame and return the stacked observation.

        Args:
            frame: uint8 batch ``(N, H, W, C)``.
            repeat_mask: Optional V19 frame-repeat mask.

        Returns:
            The stacked observation.
        """
        self.push(frame, repeat_mask)
        return self.get()

    def state_dict(self) -> dict[str, Any]:
        """Serialize the ring (checkpoint/resume and terminal-observation capture).

        Returns:
            The underlying ring state.
        """
        return self._ring.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a ring state produced by :meth:`state_dict`.

        Args:
            state: The dict returned by :meth:`state_dict`.
        """
        self._ring.load_state_dict(state)
        self._delay = self._ring.delay_steps
