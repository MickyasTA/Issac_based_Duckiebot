"""Deployment path: ONNX export, offline parity gates and the ROS node skeleton.

Everything in this package is offline. No hardware exists in this project, so nothing here is
validated on a robot; the artifacts are proven only against each other and against the shared
preprocessing implementation.

Attribute access is lazy on purpose. ``duckiebot_rl.deploy.ros_node`` must stay importable in a
torch-free environment (the robot image has numpy, OpenCV and onnxruntime only), so importing
this package must not drag torch in as a side effect.

Modules:
    export_onnx: :class:`~duckiebot_rl.deploy.export_onnx.DeployablePolicy` and the dual-target
        ONNX export (opset 13 for TensorRT 8.2, opset 18 for TensorRT 10). Needs torch.
    parity: onnxruntime-vs-torch, TorchScript-vs-torch and preprocessing parity gates.
        Needs torch.
    ros_node: the ROS 1 node skeleton plus the numpy preprocessing tail. No torch.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BakedPreprocess",
    "DeployablePolicy",
    "ExportedArtifact",
    "ParityReport",
    "build_policy_from_checkpoint",
    "export_dual_targets",
    "export_onnx",
    "onnx_parity",
]

_LAZY: dict[str, str] = {
    "BakedPreprocess": "duckiebot_rl.deploy.export_onnx",
    "DeployablePolicy": "duckiebot_rl.deploy.export_onnx",
    "ExportedArtifact": "duckiebot_rl.deploy.export_onnx",
    "build_policy_from_checkpoint": "duckiebot_rl.deploy.export_onnx",
    "export_dual_targets": "duckiebot_rl.deploy.export_onnx",
    "export_onnx": "duckiebot_rl.deploy.export_onnx",
    "ParityReport": "duckiebot_rl.deploy.parity",
    "onnx_parity": "duckiebot_rl.deploy.parity",
}


def __getattr__(name: str) -> Any:
    """Resolve public names lazily so that importing this package never pulls torch in.

    Args:
        name: Attribute name.

    Returns:
        The resolved attribute.

    Raises:
        AttributeError: If the name is not part of the public surface.
    """
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib  # only needed on the lazy path

    return getattr(importlib.import_module(module_name), name)


def __dir__() -> list[str]:
    """List the public surface for tab completion.

    Returns:
        Sorted public names.
    """
    return sorted(__all__)
