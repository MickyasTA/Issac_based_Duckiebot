"""CLI: export a trained policy to ONNX for both deployment targets, then gate it on parity.

This is a thin wrapper around :mod:`duckiebot_rl.deploy.export_onnx` so that the repository has
one obvious entry point (SPEC v2 S10, milestone M11). It adds nothing but ``sys.path`` handling
for running straight out of a fresh clone without installing the package.

Usage (Windows, from the repository root)::

    d:/Personal/personal/wheeled_quadruped_robot/.venv/Scripts/python.exe scripts/export_policy.py ^
        --checkpoint checkpoints/lane_follow_seed0_best.pt ^
        --out-dir exports/lane_follow_seed0

That writes, into the output directory:

    policy_opset13.onnx   + policy_opset13.json    Jetson Nano, TensorRT 8.2, static batch 1
    policy_opset18.onnx   + policy_opset18.json    Orin Nano, TensorRT 10
    policy_traced.pt                               TorchScript, the sim-to-sim inference artifact

and then runs the onnxruntime parity gate on both graphs. A non-zero exit code means the gate
failed, which is a release blocker: no artifact ships without it.

No TensorRT engine is built here and no robot is contacted. The ``trtexec`` commands are
recorded in the sidecar JSONs as documentation.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from duckiebot_rl.deploy.export_onnx import main  # noqa: E402 - after the sys.path shim

if __name__ == "__main__":
    raise SystemExit(main())
