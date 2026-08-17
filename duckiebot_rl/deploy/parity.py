"""Offline parity gates for the exported policy (SPEC v2 S9.1 and S8.3).

Three independent things must agree before any deployment artifact is trusted, and none of
them require hardware:

1. **ONNX Runtime vs torch.** The exported graph must reproduce the torch forward pass to
   ``max |delta action| < 1e-5`` over 1000 random observations plus the golden fixture frames,
   for BOTH opset targets. This is the M11 gate.
2. **TorchScript vs torch.** The traced module is the artifact that drives the Isaac and
   MuJoCo evaluations (S8.3 item 5), so an evaluation number can never come from a different
   forward pass than the one that was exported.
3. **Baked preprocessing vs the shared implementation.** When ``input_stage="render"``, the
   graph contains its own copy of S4.3 steps 5 to 8. That copy must be byte-identical to the
   shared training-side implementation (``duckiebot_rl.dr.preprocess`` or
   ``duckiebot_rl.preprocess``) and to the numpy implementation the ROS node uses, otherwise
   the sim-to-real gap silently contains a resampling artifact.

Run everything with::

    python -m duckiebot_rl.deploy.parity --onnx exports/policy_opset13.onnx --checkpoint ckpt.pt

or call the functions directly from tests. Every check returns a :class:`ParityReport` rather
than raising, so a caller can report all failures at once.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from duckiebot_rl.deploy.export_onnx import (
    BOX,
    CROP_TOP,
    KERNEL5,
    OBS_H,
    OBS_W,
    RENDER_H,
    RENDER_W,
    SHARED_PREPROCESS_MODULES,
    BakedPreprocess,
    DeployablePolicy,
    load_shared_preprocess,
    resolve_callable,
)

__all__ = [
    "SHARED_PIPELINE_CANDIDATES",
    "ParityReport",
    "check_shared_constants",
    "main",
    "numpy_tail_parity",
    "onnx_parity",
    "resolve_shared_pipeline",
    "shared_preprocess_parity",
    "torchscript_parity",
]

SHARED_PIPELINE_CANDIDATES: tuple[str, ...] = (
    "tail",
    "render_to_obs",
    "preprocess_render",
    "preprocess_frames",
    "obs_from_render",
    "antialias_downsample_crop",
    "blur_downsample_crop",
)
"""Function names tried inside the shared preprocessing module, in priority order.

The shared module is owned by another engineer. Rather than guessing at import time, this
list is tried and, failing that, the ``DUCKIEBOT_RL_PREPROCESS_FN`` environment variable
(``"module:callable"``) is honoured. A missing shared implementation is reported as a skipped
check, never as a pass.
"""

_CALL_CONVENTIONS: tuple[str, ...] = ("nhwc_uint8", "nchw_float01")
"""Input conventions tried against the shared callable.

The deploy side speaks NHWC uint8 because that is the graph's input contract; the training
side speaks NCHW float32 in [0, 1] because that is what sits between the DR stages. Both are
tried so that the parity gate does not depend on which convention the shared module settled on.
"""

ENV_PREPROCESS_FN = "DUCKIEBOT_RL_PREPROCESS_FN"


@dataclass
class ParityReport:
    """Result of one parity check.

    Attributes:
        name: Human-readable check name.
        passed: Whether the check met its tolerance.
        skipped: Whether the check could not run (missing optional dependency or module).
        reason: Why it was skipped, or extra detail on failure.
        n_samples: Number of samples compared.
        tolerance: Tolerance applied.
        deltas: Maximum absolute difference per compared tensor.
        details: Any extra JSON-serialisable diagnostics.
    """

    name: str
    passed: bool = False
    skipped: bool = False
    reason: str = ""
    n_samples: int = 0
    tolerance: float = 0.0
    deltas: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Render a single-line summary suitable for CI logs.

        Returns:
            The summary string.
        """
        if self.skipped:
            return f"SKIP  {self.name}: {self.reason}"
        status = "PASS" if self.passed else "FAIL"
        deltas = ", ".join(f"{k}={v:.3e}" for k, v in sorted(self.deltas.items()))
        detail = f" ({self.reason})" if self.reason else ""
        return f"{status}  {self.name}: n={self.n_samples} atol={self.tolerance:g} {deltas}{detail}"


def _as_numpy(tensor: Tensor) -> np.ndarray:
    """Detach a tensor to a contiguous numpy array.

    Args:
        tensor: Any torch tensor.

    Returns:
        The equivalent numpy array.
    """
    return np.ascontiguousarray(tensor.detach().cpu().numpy())


# --------------------------------------------------------------------------------------------
# 1. ONNX Runtime vs torch
# --------------------------------------------------------------------------------------------


def onnx_parity(
    policy: DeployablePolicy,
    onnx_path: str | Path,
    *,
    n_random: int = 1000,
    golden: Sequence[tuple[Tensor, Tensor]] | None = None,
    atol: float = 1e-5,
    seed: int = 12345,
) -> ParityReport:
    """Compare the exported ONNX graph against the torch forward pass.

    Args:
        policy: The wrapped policy that was exported.
        onnx_path: Path to the ``.onnx`` file.
        n_random: Number of uniformly random observations to compare.
        golden: Optional fixture pairs ``(image, vec)`` to compare in addition. These are the
            recorded frames from the deploy fixtures; they matter because random uint8 noise
            never exercises the saturated and near-constant image regions a real camera
            produces.
        atol: Maximum tolerated absolute difference on any output element.
        seed: RNG seed for the random samples.

    Returns:
        A :class:`ParityReport`. Skipped with a clear reason when onnxruntime is unavailable.
    """
    name = f"onnx_parity[{Path(onnx_path).name}]"
    try:
        import onnxruntime as ort  # optional dependency, checked at call time
    except ImportError as exc:
        return ParityReport(
            name=name,
            skipped=True,
            reason=f"onnxruntime is not installed ({exc}); install with pip install 'duckiebot-rl[export]'",
        )

    path = Path(onnx_path)
    if not path.is_file():
        return ParityReport(name=name, skipped=True, reason=f"missing artifact {path}")

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]

    policy.eval()
    generator = torch.Generator().manual_seed(int(seed))
    height, width, channels = policy.image_shape
    np_dtype = np.uint8 if policy.input_dtype == "uint8" else np.float32

    samples: list[tuple[Tensor, Tensor]] = []
    for _ in range(max(0, int(n_random))):
        img = torch.randint(0, 256, (1, height, width, channels), dtype=torch.uint8, generator=generator)
        if policy.input_dtype == "float32":
            img = img.float()
        vec = torch.randn(1, policy.vec_dim, generator=generator) * 2.0
        samples.append((img, vec))
    for img, vec in golden or ():
        image = img if img.dim() == 4 else img.unsqueeze(0)
        vector = vec if vec.dim() == 2 else vec.unsqueeze(0)
        if policy.input_dtype == "float32":
            image = image.float()
        else:
            image = image.to(torch.uint8)
        samples.append((image, vector.float()))

    if not samples:
        return ParityReport(name=name, skipped=True, reason="no samples requested")

    max_delta = {"action": 0.0, "mu": 0.0}
    with torch.no_grad():
        for img, vec in samples:
            torch_action, torch_mu = policy(img, vec)
            feeds = {
                input_names[0]: _as_numpy(img).astype(np_dtype, copy=False),
                input_names[1]: _as_numpy(vec).astype(np.float32, copy=False),
            }
            outputs = session.run(output_names, feeds)
            got = dict(zip(output_names, outputs, strict=True))
            action_key = "action" if "action" in got else output_names[0]
            mu_key = "mu" if "mu" in got else output_names[min(1, len(output_names) - 1)]
            max_delta["action"] = max(
                max_delta["action"], float(np.max(np.abs(got[action_key] - _as_numpy(torch_action))))
            )
            max_delta["mu"] = max(max_delta["mu"], float(np.max(np.abs(got[mu_key] - _as_numpy(torch_mu)))))

    passed = all(value <= atol for value in max_delta.values())
    return ParityReport(
        name=name,
        passed=passed,
        n_samples=len(samples),
        tolerance=atol,
        deltas=max_delta,
        details={
            "providers": session.get_providers(),
            "n_random": int(n_random),
            "n_golden": len(golden or ()),
            "input_stage": policy.input_stage,
            "input_dtype": policy.input_dtype,
        },
    )


# --------------------------------------------------------------------------------------------
# 2. TorchScript vs torch
# --------------------------------------------------------------------------------------------


def torchscript_parity(
    policy: DeployablePolicy,
    script_path: str | Path,
    *,
    n_random: int = 256,
    atol: float = 1e-6,
    seed: int = 999,
) -> ParityReport:
    """Compare the traced TorchScript module against the eager policy.

    Args:
        policy: The wrapped policy.
        script_path: Path to the ``.pt`` written by ``save_torchscript``.
        n_random: Number of random observations to compare.
        atol: Maximum tolerated absolute difference.
        seed: RNG seed.

    Returns:
        A :class:`ParityReport`.
    """
    name = f"torchscript_parity[{Path(script_path).name}]"
    path = Path(script_path)
    if not path.is_file():
        return ParityReport(name=name, skipped=True, reason=f"missing artifact {path}")

    traced = torch.jit.load(str(path), map_location="cpu")
    traced.eval()
    policy.eval()
    generator = torch.Generator().manual_seed(int(seed))
    height, width, channels = policy.image_shape
    max_delta = {"action": 0.0, "mu": 0.0}
    with torch.no_grad():
        for _ in range(int(n_random)):
            img = torch.randint(0, 256, (1, height, width, channels), dtype=torch.uint8, generator=generator)
            if policy.input_dtype == "float32":
                img = img.float()
            vec = torch.randn(1, policy.vec_dim, generator=generator) * 2.0
            eager_action, eager_mu = policy(img, vec)
            traced_action, traced_mu = traced(img, vec)
            max_delta["action"] = max(max_delta["action"], float((traced_action - eager_action).abs().max()))
            max_delta["mu"] = max(max_delta["mu"], float((traced_mu - eager_mu).abs().max()))

    return ParityReport(
        name=name,
        passed=all(value <= atol for value in max_delta.values()),
        n_samples=int(n_random),
        tolerance=atol,
        deltas=max_delta,
    )


# --------------------------------------------------------------------------------------------
# 3. Baked preprocessing vs the shared implementation
# --------------------------------------------------------------------------------------------


def check_shared_constants() -> ParityReport:
    """Assert that the deploy-side S4.3 constants equal the shared module's constants.

    The deploy package duplicates the preprocessing constants so that it stays importable
    without the training stack. This check is what stops that duplication from drifting.

    Returns:
        A :class:`ParityReport`; skipped when no shared module is importable.
    """
    name = "shared_constants"
    module = load_shared_preprocess()
    if module is None:
        return ParityReport(
            name=name,
            skipped=True,
            reason=(
                "no shared preprocess module importable (tried " + ", ".join(SHARED_PREPROCESS_MODULES) + ")"
            ),
        )
    expected: dict[str, Any] = {
        "RENDER_W": RENDER_W,
        "RENDER_H": RENDER_H,
        "OBS_W": OBS_W,
        "OBS_H": OBS_H,
        "CROP_TOP": CROP_TOP,
        "BOX": BOX,
        "KERNEL5": list(KERNEL5),
    }
    mismatches: list[str] = []
    compared = 0
    for key, want in expected.items():
        if not hasattr(module, key):
            continue
        compared += 1
        got = getattr(module, key)
        got_value = list(got) if isinstance(got, (list, tuple)) else got
        if isinstance(want, list):
            same = (
                isinstance(got_value, list)
                and len(got_value) == len(want)
                and all(abs(float(a) - float(b)) <= 1e-12 for a, b in zip(got_value, want, strict=True))
            )
        else:
            same = got_value == want
        if not same:
            mismatches.append(f"{key}: deploy={want!r} shared={got_value!r}")
    if compared == 0:
        return ParityReport(
            name=name,
            skipped=True,
            reason=f"module {module.__name__} exposes none of {sorted(expected)}",
        )
    return ParityReport(
        name=name,
        passed=not mismatches,
        n_samples=compared,
        reason="; ".join(mismatches),
        details={"module": module.__name__, "compared": compared},
    )


def resolve_shared_pipeline() -> tuple[Any | None, str]:
    """Find the shared render-to-observation callable.

    Returns:
        Tuple ``(callable, description)``. The callable is ``None`` when nothing was found,
        in which case the description explains what was tried.
    """
    override = os.environ.get(ENV_PREPROCESS_FN, "").strip()
    if override:
        try:
            return resolve_callable(override), f"{ENV_PREPROCESS_FN}={override}"
        except (ImportError, ValueError) as exc:
            return None, f"{ENV_PREPROCESS_FN}={override} could not be resolved: {exc}"
    module = load_shared_preprocess()
    if module is None:
        return None, "no shared preprocess module importable"
    for candidate in SHARED_PIPELINE_CANDIDATES:
        fn = getattr(module, candidate, None)
        if callable(fn):
            return fn, f"{module.__name__}:{candidate}"
    return None, (
        f"module {module.__name__} exposes none of {list(SHARED_PIPELINE_CANDIDATES)}; "
        f"set {ENV_PREPROCESS_FN}='module:callable' to point at it"
    )


def _to_nhwc_uint8(out: Any) -> Tensor | None:
    """Coerce a shared-pipeline output to ``(N, H, W, C)`` uint8.

    Args:
        out: Whatever the shared callable returned.

    Returns:
        The coerced tensor, or ``None`` when the shape is unrecognised.
    """
    tensor = out[0] if isinstance(out, (tuple, list)) else out
    if not isinstance(tensor, Tensor) or tensor.dim() != 4:
        return None
    if tensor.shape[1] == OBS_H and tensor.shape[2] == OBS_W:
        nhwc = tensor
    elif tensor.shape[2] == OBS_H and tensor.shape[3] == OBS_W:
        nhwc = tensor.permute(0, 2, 3, 1)
    else:
        return None
    if nhwc.dtype == torch.uint8:
        return nhwc
    scale = 255.0 if float(nhwc.max()) <= 1.0 + 1e-6 else 1.0
    return torch.round(nhwc.float() * scale).clamp(0, 255).to(torch.uint8)


def _call_shared(fn: Any, frames_nhwc_u8: Tensor) -> tuple[Tensor | None, str, list[str]]:
    """Call the shared pipeline under each supported input convention until one works.

    Args:
        fn: The resolved shared callable.
        frames_nhwc_u8: ``(N, 128, 192, 9)`` uint8 render stack.

    Returns:
        Tuple ``(observation, convention, failures)``. ``observation`` is ``None`` when no
        convention produced a recognisable output, in which case ``failures`` explains each
        attempt.
    """
    failures: list[str] = []
    for convention in _CALL_CONVENTIONS:
        if convention == "nhwc_uint8":
            argument = frames_nhwc_u8.clone()
        else:
            argument = frames_nhwc_u8.permute(0, 3, 1, 2).float().div(255.0)
        try:
            raw = fn(argument)
        except (TypeError, RuntimeError, ValueError, IndexError, AttributeError) as exc:
            failures.append(f"{convention}: {type(exc).__name__}: {exc}")
            continue
        out = _to_nhwc_uint8(raw)
        if out is None:
            failures.append(f"{convention}: unrecognised output {type(raw).__name__}")
            continue
        return out, convention, failures
    return None, "", failures


def shared_preprocess_parity(*, n_frames: int = 8, max_lsb: int = 0, seed: int = 7) -> ParityReport:
    """Compare :class:`BakedPreprocess` against the shared preprocessing implementation.

    Args:
        n_frames: Number of random render frames to push through both paths.
        max_lsb: Maximum tolerated absolute difference in uint8 least-significant bits.
            Zero means exact byte equality, which is what the two torch implementations
            should produce because they are the same op sequence.
        seed: RNG seed.

    Returns:
        A :class:`ParityReport`; skipped when the shared callable cannot be resolved.
    """
    name = "baked_vs_shared_preprocess"
    fn, description = resolve_shared_pipeline()
    if fn is None:
        return ParityReport(name=name, skipped=True, reason=description)

    baked = BakedPreprocess().eval()
    generator = torch.Generator().manual_seed(int(seed))
    frames = torch.randint(
        0, 256, (int(n_frames), RENDER_H, RENDER_W, 9), dtype=torch.uint8, generator=generator
    )
    with torch.no_grad():
        ours = baked(frames)
        theirs, convention, failures = _call_shared(fn, frames)
    if theirs is None:
        return ParityReport(
            name=name,
            skipped=True,
            reason=f"{description} accepted none of {list(_CALL_CONVENTIONS)}: {'; '.join(failures)}",
        )
    delta = (ours.int() - theirs.int()).abs()
    max_abs = float(delta.max())
    frac_within_1 = float((delta <= 1).float().mean())
    return ParityReport(
        name=name,
        passed=max_abs <= max_lsb,
        n_samples=int(n_frames),
        tolerance=float(max_lsb),
        deltas={"uint8_lsb": max_abs},
        details={
            "shared": description,
            "convention": convention,
            "fraction_within_1_lsb": frac_within_1,
        },
    )


def numpy_tail_parity(*, n_frames: int = 8, max_lsb: int = 1, seed: int = 11) -> ParityReport:
    """Compare :class:`BakedPreprocess` against the numpy tail used by the ROS node.

    The robot never imports torch, so the ROS node carries a numpy implementation of the same
    operator sequence. Float summation order differs between the two, so one least-significant
    bit of difference is tolerated by default, matching the S4.3 parity rule.

    Args:
        n_frames: Number of random render frames.
        max_lsb: Maximum tolerated absolute uint8 difference.
        seed: RNG seed.

    Returns:
        A :class:`ParityReport`.
    """
    from duckiebot_rl.deploy.ros_node import preprocess_tail_numpy  # keeps import cheap

    name = "baked_vs_numpy_tail"
    baked = BakedPreprocess().eval()
    generator = torch.Generator().manual_seed(int(seed))
    frames = torch.randint(
        0, 256, (int(n_frames), RENDER_H, RENDER_W, 9), dtype=torch.uint8, generator=generator
    )
    with torch.no_grad():
        ours = baked(frames).numpy().astype(np.int32)
    theirs = np.stack([preprocess_tail_numpy(_as_numpy(frame)) for frame in frames]).astype(np.int32)
    delta = np.abs(ours - theirs)
    max_abs = float(delta.max())
    frac_exact = float((delta == 0).mean())
    return ParityReport(
        name=name,
        passed=max_abs <= max_lsb,
        n_samples=int(n_frames),
        tolerance=float(max_lsb),
        deltas={"uint8_lsb": max_abs},
        details={"fraction_exact": frac_exact},
    )


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the parity CLI parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="duckiebot-parity",
        description="Offline parity gates for the exported Duckiebot policy (no hardware involved).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--onnx", type=Path, nargs="*", default=(), help="ONNX artifacts to check")
    parser.add_argument("--torchscript", type=Path, default=None, help="traced .pt to check")
    parser.add_argument("--checkpoint", type=Path, default=None, help="checkpoint to rebuild the policy from")
    parser.add_argument(
        "--actor-factory",
        type=str,
        default="duckiebot_rl.ppo.networks:build_actor",
        help="'module:callable' returning the actor module",
    )
    parser.add_argument("--samples", type=int, default=1000, help="random samples per artifact")
    parser.add_argument("--atol", type=float, default=1e-5, help="ONNX parity tolerance")
    parser.add_argument(
        "--skip-is-failure",
        action="store_true",
        help="treat skipped checks as failures (used by the M11 gate, where nothing may be skipped)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the parity CLI.

    Args:
        argv: Argument list; ``sys.argv[1:]`` when ``None``.

    Returns:
        Process exit code: 0 when every executed check passed.
    """
    args = build_arg_parser().parse_args(argv)
    reports: list[ParityReport] = [check_shared_constants(), shared_preprocess_parity(), numpy_tail_parity()]

    if args.onnx or args.torchscript:
        if args.checkpoint is None:
            reports.append(
                ParityReport(
                    name="artifact_parity",
                    skipped=True,
                    reason="--checkpoint is required to rebuild the reference torch policy",
                )
            )
        else:
            from duckiebot_rl.deploy.export_onnx import (  # avoids a circular import
                build_policy_from_checkpoint,
            )

            policy, _ = build_policy_from_checkpoint(args.checkpoint, actor_factory=args.actor_factory)
            for onnx_path in args.onnx:
                reports.append(onnx_parity(policy, onnx_path, n_random=args.samples, atol=args.atol))
            if args.torchscript is not None:
                reports.append(torchscript_parity(policy, args.torchscript))

    for report in reports:
        print(report.summary())
    bad = [r for r in reports if not r.passed and (not r.skipped or args.skip_is_failure)]
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
