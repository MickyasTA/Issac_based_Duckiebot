"""Export a trained lane-following policy to ONNX with the preprocessing baked in.

This module implements SPEC v2 S9.1. One checkpoint produces two ONNX artifacts:

* **Target A** (Duckiebot DB21-J4, Jetson Nano 4GB, JetPack 4.6.x, TensorRT 8.2.1.8):
  opset 13, static batch 1, legacy tracing exporter (``dynamo=False``).
* **Target B** (Duckiebot DB26-J, Jetson Orin Nano Super, JetPack 6.x, TensorRT 10.x):
  opset 18, still exported with the legacy tracing exporter by default because it is the
  path that is actually verified here; ``--dynamo`` opts into the dynamo exporter and falls
  back to the legacy one with a warning if it fails.

Nothing in this file has ever run on a physical robot. There is no Duckiebot in this project.
The artifacts and the parity numbers are produced and checked entirely offline; the TensorRT
engine build commands are documented in ``docs/setup_windows.md`` and in the sidecar JSON, and
are deliberately not executed.

What "preprocessing baked in" means here:

* ``input_stage="obs"`` (default, the S9.1 contract): the graph input is the stacked uint8
  observation ``(1, 48, 96, 9)`` in NHWC. The graph itself performs the uint8 to float cast,
  the ``/255`` scaling, the observation-vector running-mean/std normalisation with the
  statistics frozen as constants, the policy forward pass, and the scaling of the raw
  Gaussian mean into physical units ``(v [m/s], omega [rad/s])``. The frame ring and the
  three-frame stack stay outside the graph because they are stateful.
* ``input_stage="render"``: the graph input is instead the stacked *canonical render*
  ``(1, 128, 192, 9)`` and steps 5 to 8 of the S4.3 operator chain (fixed 5-tap blur,
  exact 2x2 box downsample, 16-row crop, uint8 quantisation) run inside the graph. This
  guarantees byte-identical resampling on the robot without trusting the robot-side OpenCV
  build, at the cost of a slightly larger graph. ``duckiebot_rl.deploy.parity`` proves the
  baked chain equals the shared implementation.

The actor contract (what a checkpoint must provide):

    actor(rgb_uint8_nhwc: Tensor[N, 48, 96, 9], vec_normalised: Tensor[N, 8]) -> Tensor[N, 2]

that is, the actor takes the *unnormalised uint8* image stack (it performs its own
``permute -> float -> /255`` as specified in S6.2) plus the *already normalised* observation
vector, and returns the raw Gaussian mean. If it returns a tuple or list, element 0 is used.

Example:
    >>> policy = DeployablePolicy(actor)                      # doctest: +SKIP
    >>> artifacts = export_dual_targets(policy, "out/")       # doctest: +SKIP
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

__all__ = [
    "ACTION_DIM",
    "BOX",
    "CROP_TOP",
    "KERNEL5",
    "OBS_CHANNELS",
    "OBS_H",
    "OBS_W",
    "RENDER_H",
    "RENDER_W",
    "SPEC_VERSION",
    "VEC_DIM",
    "BakedPreprocess",
    "DeployablePolicy",
    "ExportedArtifact",
    "build_policy_from_checkpoint",
    "export_dual_targets",
    "export_onnx",
    "load_shared_preprocess",
    "main",
    "save_torchscript",
    "sha256_file",
]

# --------------------------------------------------------------------------------------------
# Constants (SPEC v2 S4.3 and S5.3). These are duplicated here on purpose so that the deploy
# package has zero import-time dependency on the training packages: the ROS node and the export
# CLI must work in a bare python with numpy and onnxruntime only. `duckiebot_rl.deploy.parity`
# asserts that these values equal the shared implementation whenever it is importable, so the
# duplication can never silently drift.
# --------------------------------------------------------------------------------------------

SPEC_VERSION = "v2"

RENDER_W = 192
"""Canonical render width in pixels (S4.1)."""
RENDER_H = 128
"""Canonical render height in pixels (S4.1)."""
OBS_W = 96
"""Observation width after the exact 2x2 box downsample (S4.3)."""
OBS_H = 48
"""Observation height after the 16-row top crop (S4.3)."""
CROP_TOP = 16
"""Rows removed from the top AFTER downsampling (S4.3)."""
BOX = 2
"""Box-downsample factor; exactly equals cv2.INTER_AREA at an integer 2x ratio (S4.3)."""
KERNEL5: tuple[float, float, float, float, float] = (0.00256, 0.16555, 0.66378, 0.16555, 0.00256)
"""Separable Gaussian, sigma 0.6 px, applied at render resolution in both axes (S4.3)."""
STACK_LEN = 3
"""Number of stacked frames (t, t-2, t-4)."""
STACK_STRIDE = 2
"""Control steps between stacked frames."""
OBS_CHANNELS = STACK_LEN * 3
"""Channel count of the stacked RGB observation."""
VEC_DIM = 8
"""Policy-side observation vector width (S5.2); the critic's 14-wide vector never deploys."""
ACTION_DIM = 2
"""Raw action dimension (a_v, a_omega) in [-1, 1]."""

V_MAX_MPS = 0.6
"""Commanded forward speed at a_v = +1 (S5.3). The ROS node caps this again for first runs."""
OMEGA_MAX_RPS = 4.0
"""Commanded yaw rate at a_omega = +1 (S5.3)."""
NORM_CLIP = 5.0
"""Running-normaliser clip applied to the observation vector (S5.2)."""

CANONICAL_FX = 65.98
"""Canonical pinhole focal length in pixels at 192x128 (S4.1)."""
CANONICAL_CX = 96.0
"""Canonical principal point x (S4.1)."""
CANONICAL_CY = 64.0
"""Canonical principal point y (S4.1)."""

OPSET_TARGET_A = 13
"""Jetson Nano / TensorRT 8.2 target opset."""
OPSET_TARGET_B = 18
"""Orin Nano / TensorRT 10 target opset."""

SHARED_PREPROCESS_MODULES: tuple[str, ...] = (
    "duckiebot_rl.dr.preprocess",
    "duckiebot_rl.preprocess",
)
"""Import candidates for the shared preprocessing module, most specific first."""

InputStage = Literal["obs", "render"]
InputDType = Literal["uint8", "float32"]


# --------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------


def load_shared_preprocess() -> ModuleType | None:
    """Import the shared preprocessing module if one of the known paths exists.

    The training-side preprocessing lives in a module owned by another engineer, and this
    package must remain importable without it (the robot image has no torch training stack).

    Returns:
        The imported module, or ``None`` when no candidate is importable.
    """
    for name in SHARED_PREPROCESS_MODULES:
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    return None


def sha256_file(path: str | Path) -> str:
    """Compute the SHA-256 of a file.

    Args:
        path: File to hash.

    Returns:
        Lowercase hex digest.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo_root: str | Path | None = None) -> str | None:
    """Return the current git commit hash, or ``None`` when git is unavailable.

    Args:
        repo_root: Directory to run git in. Defaults to this file's repository root.

    Returns:
        The 40-character commit hash, or ``None``.
    """
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = out.stdout.strip()
    return commit if out.returncode == 0 and len(commit) == 40 else None


def resolve_callable(spec: str) -> Any:
    """Resolve a ``"package.module:attribute"`` string to the attribute.

    Args:
        spec: Entry-point style string, for example ``"duckiebot_rl.ppo.networks:build_actor"``.

    Returns:
        The resolved attribute.

    Raises:
        ValueError: If the string is not of the form ``module:attribute``.
        ImportError: If the module cannot be imported or lacks the attribute.
    """
    if ":" not in spec:
        raise ValueError(f"expected 'module:attribute', got {spec!r}")
    module_name, _, attr = spec.partition(":")
    module = importlib.import_module(module_name)
    if not hasattr(module, attr):
        raise ImportError(f"module {module_name!r} has no attribute {attr!r}")
    return getattr(module, attr)


# --------------------------------------------------------------------------------------------
# Baked preprocessing
# --------------------------------------------------------------------------------------------


class BakedPreprocess(nn.Module):
    """Steps 5 to 8 of the S4.3 operator chain as an exportable module.

    The chain is, in order: cast to float and divide by 255, a fixed separable 5-tap Gaussian
    (replicate padded, width first then height), an exact 2x2 average pool, a top crop of
    ``crop_top`` rows, and re-quantisation to uint8. It deliberately contains no domain
    randomisation: DR is training only and never ships.

    Input and output are both NHWC uint8 so that the wrapped actor keeps its documented
    ``uint8 NHWC`` contract.

    Attributes:
        channels: Channel count of the stacked frames (9 for the 3-frame RGB stack).
        pad: Replicate padding applied on each side for the 5-tap kernel.
        crop_top: Number of rows removed from the top after downsampling.
        obs_h: Output height in pixels.
        obs_w: Output width in pixels.
    """

    def __init__(
        self,
        channels: int = OBS_CHANNELS,
        kernel: Sequence[float] = KERNEL5,
        crop_top: int = CROP_TOP,
        obs_h: int = OBS_H,
        obs_w: int = OBS_W,
        box: int = BOX,
    ) -> None:
        """Initialise the baked preprocessing module.

        Args:
            channels: Number of image channels (9 for the stacked observation).
            kernel: Separable blur taps; must have odd length.
            crop_top: Rows removed from the top after downsampling.
            obs_h: Expected output height.
            obs_w: Expected output width.
            box: Box-pool factor.

        Raises:
            ValueError: If the kernel length is even or the geometry is inconsistent.
        """
        super().__init__()
        if len(kernel) % 2 == 0:
            raise ValueError(f"kernel must have odd length, got {len(kernel)}")
        if obs_w * box != RENDER_W:
            raise ValueError(f"obs_w * box must equal the render width {RENDER_W}")
        if crop_top + obs_h > RENDER_H // box:
            raise ValueError("crop_top + obs_h exceeds the downsampled render height")
        self.channels = int(channels)
        self.pad = (len(kernel) - 1) // 2
        self.crop_top = int(crop_top)
        self.obs_h = int(obs_h)
        self.obs_w = int(obs_w)
        self.box = int(box)
        taps = torch.tensor(tuple(kernel), dtype=torch.float32)
        self.register_buffer("kernel_w", taps.view(1, 1, 1, -1).repeat(self.channels, 1, 1, 1))
        self.register_buffer("kernel_h", taps.view(1, 1, -1, 1).repeat(self.channels, 1, 1, 1))

    def forward(self, render_nhwc: Tensor) -> Tensor:
        """Run the S4.3 tail on a stacked canonical render.

        Args:
            render_nhwc: ``(N, 128, 192, 9)`` uint8 tensor, RGB channel order.

        Returns:
            ``(N, 48, 96, 9)`` uint8 tensor, the observation the policy consumes.
        """
        x = render_nhwc.permute(0, 3, 1, 2).float().div(255.0)
        x = F.pad(x, (self.pad, self.pad, 0, 0), mode="replicate")
        x = F.conv2d(x, self.kernel_w, groups=self.channels)
        x = F.pad(x, (0, 0, self.pad, self.pad), mode="replicate")
        x = F.conv2d(x, self.kernel_h, groups=self.channels)
        x = F.avg_pool2d(x, kernel_size=self.box, stride=self.box)
        x = x[:, :, self.crop_top : self.crop_top + self.obs_h, :]
        x = torch.round(x * 255.0).clamp(0.0, 255.0)
        return x.permute(0, 2, 3, 1).to(torch.uint8)


# --------------------------------------------------------------------------------------------
# Deployable policy
# --------------------------------------------------------------------------------------------


class DeployablePolicy(nn.Module):
    """Inference-only wrapper around the trained actor (SPEC v2 S9.1).

    The wrapper owns everything the learner did outside the network and the robot would
    otherwise have to re-derive: the frozen observation-vector normalisation, the optional
    baked image preprocessing, the deterministic action selection (``a = mu``, no sampling)
    and the conversion of the raw action box into physical units.

    The module has two outputs:

    * ``action``: ``(N, 2)`` float32, ``[v (m/s), omega (rad/s)]``, ready for a
      ``Twist2DStamped`` message.
    * ``mu``: ``(N, 2)`` float32, the raw unclipped Gaussian mean in the ``[-1, 1]`` action
      box. The MuJoCo sim-to-sim harness consumes this one, because it re-runs the full
      S5.3 action path itself.

    Attributes:
        input_stage: ``"obs"`` for the S9.1 contract, ``"render"`` to bake steps 5 to 8.
        input_dtype: ``"uint8"`` (default) or ``"float32"``. TensorRT 8.2 accepts uint8
            network inputs only in limited configurations, so float32 in the 0 to 255 range
            is offered as an escape hatch that keeps bit-identical results (the graph rounds
            and casts immediately).
    """

    def __init__(
        self,
        actor: nn.Module,
        vec_mean: Tensor | Sequence[float] | None = None,
        vec_std: Tensor | Sequence[float] | None = None,
        *,
        vec_dim: int = VEC_DIM,
        v_max: float = V_MAX_MPS,
        omega_max: float = OMEGA_MAX_RPS,
        norm_clip: float = NORM_CLIP,
        input_stage: InputStage = "obs",
        input_dtype: InputDType = "uint8",
    ) -> None:
        """Wrap an actor for deployment.

        Args:
            actor: Module honouring the actor contract documented in the module docstring.
            vec_mean: Frozen running mean of the observation vector; zeros when ``None``.
            vec_std: Frozen running standard deviation; ones when ``None``.
            vec_dim: Width of the observation vector.
            v_max: Forward speed commanded at ``a_v = +1``.
            omega_max: Yaw rate commanded at ``a_omega = +1``.
            norm_clip: Symmetric clip applied after vector normalisation.
            input_stage: ``"obs"`` or ``"render"``.
            input_dtype: ``"uint8"`` or ``"float32"``.

        Raises:
            ValueError: On an unknown stage or dtype, or a shape mismatch in the statistics.
        """
        super().__init__()
        if input_stage not in ("obs", "render"):
            raise ValueError(f"input_stage must be 'obs' or 'render', got {input_stage!r}")
        if input_dtype not in ("uint8", "float32"):
            raise ValueError(f"input_dtype must be 'uint8' or 'float32', got {input_dtype!r}")
        self.actor = actor
        self.vec_dim = int(vec_dim)
        self.v_max = float(v_max)
        self.omega_max = float(omega_max)
        self.norm_clip = float(norm_clip)
        self.input_stage: InputStage = input_stage
        self.input_dtype: InputDType = input_dtype
        self.preprocess = BakedPreprocess() if input_stage == "render" else None

        mean = (
            torch.zeros(self.vec_dim) if vec_mean is None else torch.as_tensor(vec_mean, dtype=torch.float32)
        )
        std = torch.ones(self.vec_dim) if vec_std is None else torch.as_tensor(vec_std, dtype=torch.float32)
        mean = mean.reshape(-1).float()
        std = std.reshape(-1).float()
        if mean.numel() != self.vec_dim or std.numel() != self.vec_dim:
            raise ValueError(
                f"vec_mean/vec_std must have {self.vec_dim} elements, got {mean.numel()} and {std.numel()}"
            )
        if bool(torch.any(std <= 0)):
            raise ValueError("vec_std must be strictly positive")
        self.register_buffer("vec_mean", mean.view(1, -1))
        self.register_buffer("vec_std", std.view(1, -1))

    @property
    def image_shape(self) -> tuple[int, int, int]:
        """Return the ``(H, W, C)`` of the image input for the configured stage."""
        if self.input_stage == "render":
            return (RENDER_H, RENDER_W, OBS_CHANNELS)
        return (OBS_H, OBS_W, OBS_CHANNELS)

    def example_inputs(self, batch: int = 1, seed: int | None = 0) -> tuple[Tensor, Tensor]:
        """Build a deterministic example input pair for tracing and parity checks.

        Args:
            batch: Batch size.
            seed: RNG seed; ``None`` uses the ambient RNG state.

        Returns:
            Tuple ``(image, vec)`` matching the configured stage and dtype.
        """
        generator = None
        if seed is not None:
            generator = torch.Generator().manual_seed(int(seed))
        height, width, channels = self.image_shape
        img = torch.randint(0, 256, (batch, height, width, channels), dtype=torch.uint8, generator=generator)
        if self.input_dtype == "float32":
            img = img.float()
        vec = torch.randn(batch, self.vec_dim, generator=generator)
        return img, vec

    def forward(self, image: Tensor, vec: Tensor) -> tuple[Tensor, Tensor]:
        """Map one observation to a physical command.

        Args:
            image: ``(N, H, W, 9)`` image stack. uint8, or float32 holding integral 0 to 255
                values when ``input_dtype == "float32"``. ``H, W`` are ``(48, 96)`` for the
                ``"obs"`` stage and ``(128, 192)`` for the ``"render"`` stage.
            vec: ``(N, 8)`` float32 raw (unnormalised) observation vector.

        Returns:
            Tuple ``(action, mu)`` where ``action`` is ``[v, omega]`` in physical units and
            ``mu`` is the raw Gaussian mean inside the ``[-1, 1]`` action box.
        """
        img = image
        if self.input_dtype == "float32":
            img = torch.round(img).clamp(0.0, 255.0).to(torch.uint8)
        if self.preprocess is not None:
            img = self.preprocess(img)

        vec_n = (vec - self.vec_mean) / self.vec_std
        vec_n = vec_n.clamp(-self.norm_clip, self.norm_clip)

        out = self.actor(img, vec_n)
        mu = out[0] if isinstance(out, (tuple, list)) else out

        clipped = mu.clamp(-1.0, 1.0)
        v = 0.5 * self.v_max * (clipped[:, 0:1] + 1.0)
        omega = self.omega_max * clipped[:, 1:2]
        action = torch.cat([v, omega], dim=1)
        return action, mu


# --------------------------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ExportedArtifact:
    """One exported artifact plus its sidecar.

    Attributes:
        target: Human-readable target name, for example ``"jetson-nano-trt82"``.
        onnx_path: Path of the written ``.onnx`` file.
        sidecar_path: Path of the written ``.json`` sidecar.
        opset: ONNX opset version used.
        dynamo: Whether the dynamo exporter produced the file.
        sha256: SHA-256 of the ``.onnx`` file.
        metadata: The full sidecar dictionary.
    """

    target: str
    onnx_path: Path
    sidecar_path: Path
    opset: int
    dynamo: bool
    sha256: str
    metadata: dict[str, Any] = field(default_factory=dict)


def preprocess_metadata(policy: DeployablePolicy) -> dict[str, Any]:
    """Describe the preprocessing contract for the sidecar JSON.

    Args:
        policy: The policy being exported.

    Returns:
        A JSON-serialisable description of every constant the robot must reproduce.
    """
    height, width, channels = policy.image_shape
    baked = policy.input_stage == "render"
    return {
        "spec_section": "S4.3",
        "input_stage": policy.input_stage,
        "graph_input_shape_nhwc": [1, height, width, channels],
        "graph_input_dtype": policy.input_dtype,
        "color_order": "RGB",
        "canonical_render": {
            "width": RENDER_W,
            "height": RENDER_H,
            "fx": CANONICAL_FX,
            "fy": CANONICAL_FX,
            "cx": CANONICAL_CX,
            "cy": CANONICAL_CY,
            "hfov_deg": 111.0,
            "vfov_deg": 88.3,
        },
        "observation": {"width": OBS_W, "height": OBS_H, "channels": channels},
        "blur_kernel": list(KERNEL5),
        "blur_padding": "replicate",
        "box_downsample": BOX,
        "crop_top_rows_after_downsample": CROP_TOP,
        "frame_stack": {"length": STACK_LEN, "stride_control_steps": STACK_STRIDE},
        "baked_into_graph": {
            "blur": baked,
            "box_downsample": baked,
            "crop": baked,
            "uint8_quantise": baked,
            "divide_by_255": True,
            "vec_running_norm": True,
            "action_scaling": True,
            "frame_ring_and_stack": False,
        },
        "robot_side_steps_before_the_graph": [
            "JPEG decode from camera_node/image/compressed",
            "BGR to RGB (the classic bug; asserted by the deploy fixture tests)",
            "Gaussian pre-blur sigma 1.0 px at 640x480",
            "cv2.remap to the canonical 192x128 pinhole using K_canon",
        ]
        + ([] if baked else ["fixed 5-tap blur", "exact 2x2 box downsample", "16-row crop", "uint8 quantise"])
        + ["frame ring push", "stack (t, t-2, t-4)"],
    }


def action_metadata(policy: DeployablePolicy) -> dict[str, Any]:
    """Describe the action outputs for the sidecar JSON.

    Args:
        policy: The policy being exported.

    Returns:
        A JSON-serialisable description of both output tensors.
    """
    return {
        "outputs": ["action", "mu"],
        "action": {
            "units": ["m/s", "rad/s"],
            "names": ["v", "omega"],
            "formula": "v = 0.5 * v_max * (clip(mu_v, -1, 1) + 1); omega = omega_max * clip(mu_omega, -1, 1)",
            "v_max": policy.v_max,
            "omega_max": policy.omega_max,
            "ros_message": "duckietown_msgs/Twist2DStamped on car_cmd_switch_node/cmd at 15 Hz",
        },
        "mu": {
            "units": ["dimensionless", "dimensionless"],
            "note": "raw unclipped Gaussian mean in the [-1, 1] action box; the sim-to-sim "
            "harness consumes this and re-runs the S5.3 action path itself",
        },
        "deterministic": True,
        "control_rate_hz": 15.0,
    }


@contextmanager
def _tolerant_console() -> Iterator[None]:
    """Stop unicode console output from killing an otherwise successful export.

    The dynamo exporter prints status lines containing emoji. On Windows the default console
    encoding is cp1252, so those prints raise ``UnicodeEncodeError`` *after* the graph has been
    built, which looks exactly like an export failure and silently demotes the opset-18 target
    to the legacy exporter. This project is Windows only, so the encoding is switched to
    replacement characters for the duration of the export and restored afterwards.

    Yields:
        None.
    """
    streams = [stream for stream in (sys.stdout, sys.stderr) if hasattr(stream, "reconfigure")]
    previous = [(stream, getattr(stream, "errors", None)) for stream in streams]
    for stream in streams:
        with suppress(ValueError, OSError):  # exotic stream wrappers
            stream.reconfigure(errors="replace")  # type: ignore[union-attr]
    try:
        yield
    finally:
        for stream, errors in previous:
            if errors is None:
                continue
            with suppress(ValueError, OSError):
                stream.reconfigure(errors=errors)  # type: ignore[union-attr]


def _dynamic_axes(dynamic_batch: bool) -> dict[str, dict[int, str]] | None:
    """Build the ``dynamic_axes`` mapping for the legacy exporter.

    Args:
        dynamic_batch: Whether the batch axis should be dynamic.

    Returns:
        The mapping, or ``None`` for a fully static graph.
    """
    if not dynamic_batch:
        return None
    return {
        "image": {0: "batch"},
        "vec": {0: "batch"},
        "action": {0: "batch"},
        "mu": {0: "batch"},
    }


def export_onnx(
    policy: DeployablePolicy,
    out_path: str | Path,
    *,
    opset: int = OPSET_TARGET_A,
    dynamo: bool = False,
    batch: int = 1,
    dynamic_batch: bool = False,
) -> Path:
    """Export one ONNX artifact.

    Args:
        policy: The wrapped policy. It is switched to eval mode.
        out_path: Destination ``.onnx`` path; parent directories are created.
        opset: ONNX opset version. 13 for TensorRT 8.2, 18 for TensorRT 10.
        dynamo: Use the dynamo exporter. Falls back to the legacy tracing exporter with a
            warning if the dynamo path raises, because the legacy path is the verified one.
        batch: Batch size of the tracing example (1 for both deployment targets).
        dynamic_batch: Mark the batch axis dynamic. TensorRT 8.2 static-batch engines do not
            need this and it is off by default.

    Returns:
        The path written.

    Raises:
        RuntimeError: If the exporter produced no file.
    """
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    policy.eval()
    example = policy.example_inputs(batch=batch)

    with torch.no_grad():
        if dynamo:
            try:
                with _tolerant_console():
                    torch.onnx.export(
                        policy,
                        example,
                        str(path),
                        input_names=["image", "vec"],
                        output_names=["action", "mu"],
                        opset_version=opset,
                        dynamo=True,
                    )
            except Exception as exc:  # exporter raises many unrelated types
                warnings.warn(
                    f"dynamo ONNX export failed ({type(exc).__name__}: {exc}); "
                    "falling back to the legacy tracing exporter",
                    RuntimeWarning,
                    stacklevel=2,
                )
                dynamo = False
        if not dynamo:
            torch.onnx.export(
                policy,
                example,
                str(path),
                input_names=["image", "vec"],
                output_names=["action", "mu"],
                opset_version=opset,
                dynamic_axes=_dynamic_axes(dynamic_batch),
                do_constant_folding=True,
                dynamo=False,
            )

    if not path.is_file():
        raise RuntimeError(f"ONNX export produced no file at {path}")
    return path


def save_torchscript(policy: DeployablePolicy, out_path: str | Path, batch: int = 1) -> Path:
    """Trace the policy to TorchScript.

    Per SPEC S8.3 item 5 this traced module is the single inference artifact that drives the
    Isaac evaluation, the MuJoCo evaluation and the ONNX parity test, so that no evaluation
    number can come from a differently wired forward pass.

    Args:
        policy: The wrapped policy.
        out_path: Destination ``.pt`` path.
        batch: Batch size of the tracing example.

    Returns:
        The path written.
    """
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    policy.eval()
    example = policy.example_inputs(batch=batch)
    with torch.no_grad():
        traced = torch.jit.trace(policy, example, check_trace=False)
    torch.jit.save(traced, str(path))
    return path


def build_sidecar(
    policy: DeployablePolicy,
    onnx_path: str | Path,
    *,
    target: str,
    opset: int,
    dynamo: bool,
    trt_note: str,
    checkpoint: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the sidecar metadata dictionary for one artifact.

    Args:
        policy: The exported policy.
        onnx_path: Path of the exported file (must exist; it is hashed).
        target: Target name.
        opset: Opset used.
        dynamo: Whether the dynamo exporter was used.
        trt_note: The documented, deliberately unexecuted TensorRT build command.
        checkpoint: Provenance fields carried over from the training checkpoint.
        extra: Any additional fields to merge at the top level.

    Returns:
        A JSON-serialisable dictionary.
    """
    path = Path(onnx_path)
    height, width, channels = policy.image_shape
    meta: dict[str, Any] = {
        "artifact": {
            "file": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        },
        "target": target,
        "opset": opset,
        "exporter": "dynamo" if dynamo else "torchscript-trace",
        "tensorrt_build_command": trt_note,
        "hardware_validation": (
            "NONE. No physical robot exists in this project. This artifact is verified only by "
            "offline onnxruntime-vs-torch parity and TorchScript equality."
        ),
        "io": {
            "inputs": [
                {
                    "name": "image",
                    "dtype": policy.input_dtype,
                    "shape": [1, height, width, channels],
                    "layout": "NHWC",
                },
                {"name": "vec", "dtype": "float32", "shape": [1, policy.vec_dim]},
            ],
            "outputs": [
                {"name": "action", "dtype": "float32", "shape": [1, ACTION_DIM]},
                {"name": "mu", "dtype": "float32", "shape": [1, ACTION_DIM]},
            ],
        },
        "preprocess": preprocess_metadata(policy),
        "action": action_metadata(policy),
        "normalisation": {
            "vec_mean": policy.vec_mean.detach().cpu().reshape(-1).tolist(),
            "vec_std": policy.vec_std.detach().cpu().reshape(-1).tolist(),
            "clip": policy.norm_clip,
            "baked_as_constants": True,
        },
        "provenance": {
            "spec_version": SPEC_VERSION,
            "repo_commit": git_commit(),
            "exported_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "torch_version": torch.__version__,
            "python_version": sys.version.split()[0],
        },
    }
    try:
        import onnx  # optional at runtime, only needed for the version string

        meta["provenance"]["onnx_version"] = onnx.__version__
    except ImportError:
        meta["provenance"]["onnx_version"] = None
    if checkpoint:
        meta["checkpoint"] = checkpoint
    if extra:
        meta.update(extra)
    return meta


def write_sidecar(metadata: dict[str, Any], onnx_path: str | Path) -> Path:
    """Write the sidecar JSON next to its ONNX file.

    Args:
        metadata: Dictionary from :func:`build_sidecar`.
        onnx_path: Path of the ONNX file.

    Returns:
        The sidecar path.
    """
    path = Path(onnx_path).with_suffix(".json")
    path.write_text(json.dumps(metadata, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path


TRT_NOTE_A = (
    "DOCUMENTED, NOT RUN: nvpmodel -m 0 && jetson_clocks && "
    "trtexec --onnx=policy_opset13.onnx --fp16 --workspace=512 --saveEngine=policy_nano.plan"
)
TRT_NOTE_B = "DOCUMENTED, NOT RUN: trtexec --onnx=policy_opset18.onnx --fp16 --saveEngine=policy_orin.plan"


def export_dual_targets(
    policy: DeployablePolicy,
    out_dir: str | Path,
    *,
    stem: str = "policy",
    dynamo_target_b: bool = False,
    checkpoint: dict[str, Any] | None = None,
    torchscript: bool = True,
) -> list[ExportedArtifact]:
    """Export both deployment targets plus the TorchScript artifact.

    Args:
        policy: The wrapped policy.
        out_dir: Output directory; created if needed.
        stem: Filename stem, giving ``<stem>_opset13.onnx`` and ``<stem>_opset18.onnx``.
        dynamo_target_b: Use the dynamo exporter for the opset-18 target.
        checkpoint: Provenance fields from the training checkpoint.
        torchscript: Also write ``<stem>_traced.pt``.

    Returns:
        One :class:`ExportedArtifact` per ONNX target, in target order A then B.
    """
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    targets = (
        ("jetson-nano-trt82", OPSET_TARGET_A, False, TRT_NOTE_A),
        ("orin-nano-trt10", OPSET_TARGET_B, dynamo_target_b, TRT_NOTE_B),
    )
    artifacts: list[ExportedArtifact] = []
    for target, opset, dynamo, trt_note in targets:
        onnx_path = directory / f"{stem}_opset{opset}.onnx"
        export_onnx(policy, onnx_path, opset=opset, dynamo=dynamo)
        metadata = build_sidecar(
            policy,
            onnx_path,
            target=target,
            opset=opset,
            dynamo=dynamo,
            trt_note=trt_note,
            checkpoint=checkpoint,
        )
        sidecar = write_sidecar(metadata, onnx_path)
        artifacts.append(
            ExportedArtifact(
                target=target,
                onnx_path=onnx_path,
                sidecar_path=sidecar,
                opset=opset,
                dynamo=bool(metadata["exporter"] == "dynamo"),
                sha256=str(metadata["artifact"]["sha256"]),
                metadata=metadata,
            )
        )
    if torchscript:
        save_torchscript(policy, directory / f"{stem}_traced.pt")
    return artifacts


# --------------------------------------------------------------------------------------------
# Checkpoint loading
# --------------------------------------------------------------------------------------------

_STATE_DICT_KEYS = ("actor", "model", "model_state_dict", "state_dict", "policy")
_NORM_KEYS = ("running_norm", "normalizers", "obs_norm")
_VEC_NORM_KEYS = ("vec", "obs_vec", "vec_obs")


def _extract_vec_norm(checkpoint: dict[str, Any], vec_dim: int) -> tuple[Tensor | None, Tensor | None]:
    """Pull the observation-vector running mean and standard deviation out of a checkpoint.

    The learner's checkpoint schema (S6.9) stores a ``running_norm`` mapping with one entry per
    normalised quantity. Only the policy-side ``vec`` entry deploys; the critic's ``vec_priv``
    never leaves training.

    Args:
        checkpoint: Loaded checkpoint dictionary.
        vec_dim: Expected vector width.

    Returns:
        Tuple ``(mean, std)``; both ``None`` when the checkpoint carries no statistics.
    """
    for norm_key in _NORM_KEYS:
        norms = checkpoint.get(norm_key)
        if not isinstance(norms, dict):
            continue
        for vec_key in _VEC_NORM_KEYS:
            entry = norms.get(vec_key)
            if not isinstance(entry, dict):
                continue
            mean = entry.get("mean")
            var = entry.get("var", entry.get("variance"))
            std = entry.get("std")
            if mean is None:
                continue
            mean_t = torch.as_tensor(mean, dtype=torch.float32).reshape(-1)
            if std is not None:
                std_t = torch.as_tensor(std, dtype=torch.float32).reshape(-1)
            elif var is not None:
                std_t = torch.sqrt(torch.as_tensor(var, dtype=torch.float32).reshape(-1) + 1e-8)
            else:
                continue
            if mean_t.numel() == vec_dim and std_t.numel() == vec_dim:
                return mean_t, std_t.clamp_min(1e-6)
    return None, None


def build_policy_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    actor_factory: str = "duckiebot_rl.ppo.networks:build_actor",
    vec_dim: int = VEC_DIM,
    input_stage: InputStage = "obs",
    input_dtype: InputDType = "uint8",
    v_max: float = V_MAX_MPS,
    omega_max: float = OMEGA_MAX_RPS,
    map_location: str = "cpu",
) -> tuple[DeployablePolicy, dict[str, Any]]:
    """Load a training checkpoint and wrap its actor for deployment.

    The actor class belongs to the PPO package, so it is resolved through an entry-point
    string rather than guessed. The factory is called as ``factory(checkpoint)`` first and,
    if that raises ``TypeError``, as ``factory()`` followed by ``load_state_dict``. Both
    conventions are common and both are supported explicitly.

    Args:
        checkpoint_path: Path to the ``.pt`` written by ``scripts/train.py``.
        actor_factory: ``"module:callable"`` returning the actor module.
        vec_dim: Observation-vector width.
        input_stage: ``"obs"`` or ``"render"``.
        input_dtype: ``"uint8"`` or ``"float32"``.
        v_max: Forward-speed scale.
        omega_max: Yaw-rate scale.
        map_location: torch.load device mapping.

    Returns:
        Tuple ``(policy, provenance)`` where ``provenance`` holds the checkpoint fields worth
        recording in the sidecar.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
        KeyError: If no recognisable actor state dict is present.
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise KeyError(f"expected a dict checkpoint, got {type(checkpoint).__name__}")

    factory = resolve_callable(actor_factory)
    try:
        actor = factory(checkpoint)
    except TypeError:
        actor = factory()
        state: dict[str, Any] | None = None
        for key in _STATE_DICT_KEYS:
            value = checkpoint.get(key)
            if isinstance(value, dict):
                state = value
                break
        if state is None:
            raise KeyError(f"no actor state dict found under any of {_STATE_DICT_KEYS} in {path}") from None
        actor.load_state_dict(state, strict=False)
    actor.eval()

    mean, std = _extract_vec_norm(checkpoint, vec_dim)
    policy = DeployablePolicy(
        actor,
        vec_mean=mean,
        vec_std=std,
        vec_dim=vec_dim,
        v_max=v_max,
        omega_max=omega_max,
        input_stage=input_stage,
        input_dtype=input_dtype,
    )
    provenance = {
        "path": str(path),
        "sha256": sha256_file(path),
        "iteration": checkpoint.get("iteration"),
        "global_step": checkpoint.get("global_step"),
        "seed": checkpoint.get("seed"),
        "train_commit": checkpoint.get("git_commit"),
        "config_hash": checkpoint.get("config_hash"),
        "spec_version": checkpoint.get("spec_version", SPEC_VERSION),
        "vec_norm_found": mean is not None,
    }
    return policy, provenance


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser shared with ``scripts/export_policy.py``.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="duckiebot-export-policy",
        description=(
            "Export a trained Duckiebot lane-following policy to ONNX for both deployment "
            "targets, with preprocessing baked in. Offline only: nothing here touches hardware."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, type=Path, help="training checkpoint (.pt)")
    parser.add_argument("--out-dir", type=Path, default=Path("exports"), help="output directory")
    parser.add_argument("--stem", type=str, default="policy", help="output filename stem")
    parser.add_argument(
        "--actor-factory",
        type=str,
        default="duckiebot_rl.ppo.networks:build_actor",
        help="'module:callable' that returns the actor module",
    )
    parser.add_argument("--vec-dim", type=int, default=VEC_DIM, help="observation vector width")
    parser.add_argument(
        "--input-stage",
        choices=("obs", "render"),
        default="obs",
        help="'obs' takes the stacked 48x96x9 observation; 'render' bakes S4.3 steps 5-8",
    )
    parser.add_argument(
        "--input-dtype",
        choices=("uint8", "float32"),
        default="uint8",
        help="graph image input dtype; float32 (values 0-255) avoids TensorRT 8.2 uint8-IO limits",
    )
    parser.add_argument("--v-max", type=float, default=V_MAX_MPS, help="forward speed at a_v = +1")
    parser.add_argument("--omega-max", type=float, default=OMEGA_MAX_RPS, help="yaw rate at a_omega = +1")
    parser.add_argument(
        "--dynamo",
        action="store_true",
        help="use the dynamo exporter for the opset-18 target (falls back to tracing on failure)",
    )
    parser.add_argument("--no-torchscript", action="store_true", help="skip the TorchScript trace")
    parser.add_argument(
        "--parity-samples",
        type=int,
        default=1000,
        help="random samples for the onnxruntime parity gate (0 disables it)",
    )
    parser.add_argument(
        "--parity-atol",
        type=float,
        default=1e-5,
        help="max allowed absolute action difference between torch and onnxruntime",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the export CLI.

    Args:
        argv: Argument list; ``sys.argv[1:]`` when ``None``.

    Returns:
        Process exit code: 0 on success, 1 when the parity gate fails.
    """
    args = build_arg_parser().parse_args(argv)
    policy, provenance = build_policy_from_checkpoint(
        args.checkpoint,
        actor_factory=args.actor_factory,
        vec_dim=args.vec_dim,
        input_stage=args.input_stage,
        input_dtype=args.input_dtype,
        v_max=args.v_max,
        omega_max=args.omega_max,
    )
    artifacts = export_dual_targets(
        policy,
        args.out_dir,
        stem=args.stem,
        dynamo_target_b=args.dynamo,
        checkpoint=provenance,
        torchscript=not args.no_torchscript,
    )
    for artifact in artifacts:
        print(f"[export] {artifact.target}: {artifact.onnx_path} (opset {artifact.opset})")
        print(f"[export]   sha256 {artifact.sha256}")
        print(f"[export]   sidecar {artifact.sidecar_path}")

    if args.parity_samples <= 0:
        print("[export] parity gate disabled by --parity-samples 0")
        return 0

    from duckiebot_rl.deploy.parity import onnx_parity  # avoids a circular import

    failed = False
    for artifact in artifacts:
        report = onnx_parity(policy, artifact.onnx_path, n_random=args.parity_samples, atol=args.parity_atol)
        print(f"[parity] {artifact.target}: {report.summary()}")
        failed = failed or not report.passed
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
