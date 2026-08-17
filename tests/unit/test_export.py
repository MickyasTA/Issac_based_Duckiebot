"""Unit tests for the deployment path: ONNX export, sidecars and offline parity.

These tests build a tiny dummy actor that honours the documented actor contract, export it to
both deployment targets and prove that onnxruntime reproduces the torch forward pass. They run
on CPU in a few seconds and need no Isaac, no GPU and no robot.

If onnxruntime is missing the ONNX tests skip with an explicit install hint; everything that
does not need it (preprocessing parity, action scaling, sidecar contents, checkpoint loading)
still runs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from duckiebot_rl.deploy import ros_node
from duckiebot_rl.deploy.export_onnx import (
    ACTION_DIM,
    OBS_CHANNELS,
    OBS_H,
    OBS_W,
    OPSET_TARGET_A,
    OPSET_TARGET_B,
    RENDER_H,
    RENDER_W,
    VEC_DIM,
    BakedPreprocess,
    DeployablePolicy,
    build_policy_from_checkpoint,
    export_dual_targets,
    export_onnx,
    save_torchscript,
)
from duckiebot_rl.deploy.parity import (
    check_shared_constants,
    numpy_tail_parity,
    onnx_parity,
    shared_preprocess_parity,
    torchscript_parity,
)

ORT_HINT = "onnxruntime is not installed; install it with: pip install 'duckiebot-rl[export]'"


class DummyActor(nn.Module):
    """Minimal actor honouring the deployment contract.

    It mirrors the real encoder's interface, not its capacity: uint8 NHWC image stack in,
    global average pooling, concatenation with the normalised observation vector, and a linear
    head producing the raw Gaussian mean.
    """

    def __init__(self, hidden: int = 8, vec_dim: int = VEC_DIM) -> None:
        """Build the dummy actor with deterministic weights.

        Args:
            hidden: Channel count of the single convolution.
            vec_dim: Observation vector width.
        """
        super().__init__()
        torch.manual_seed(0)
        self.conv = nn.Conv2d(OBS_CHANNELS, hidden, kernel_size=3, stride=2, padding=1)
        self.head = nn.Linear(hidden + vec_dim, ACTION_DIM)

    def forward(self, rgb: Tensor, vec: Tensor) -> Tensor:
        """Map an observation to the raw Gaussian mean.

        Args:
            rgb: ``(N, 48, 96, 9)`` uint8 image stack.
            vec: ``(N, 8)`` normalised observation vector.

        Returns:
            ``(N, 2)`` raw action mean.
        """
        x = rgb.permute(0, 3, 1, 2).float().div(255.0)
        x = torch.relu(self.conv(x))
        x = x.mean(dim=(2, 3))
        return self.head(torch.cat([x, vec], dim=1))


@pytest.fixture
def policy() -> DeployablePolicy:
    """A deployable policy in the default S9.1 configuration."""
    torch.manual_seed(0)
    return DeployablePolicy(
        DummyActor(),
        vec_mean=torch.linspace(-1.0, 1.0, VEC_DIM),
        vec_std=torch.full((VEC_DIM,), 2.0),
    ).eval()


@pytest.fixture
def render_policy() -> DeployablePolicy:
    """A deployable policy that bakes S4.3 steps 5 to 8 into the graph."""
    torch.manual_seed(0)
    return DeployablePolicy(DummyActor(), input_stage="render").eval()


# --------------------------------------------------------------------------------------------
# Constants and preprocessing
# --------------------------------------------------------------------------------------------


def test_deploy_constants_match_ros_node():
    """The torch-free ROS copy of the S4.3 constants must equal the exporter's copy."""
    from duckiebot_rl.deploy import export_onnx as ex

    for name in ("RENDER_W", "RENDER_H", "OBS_W", "OBS_H", "CROP_TOP", "BOX", "STACK_LEN", "STACK_STRIDE"):
        assert getattr(ex, name) == getattr(ros_node, name), name
    assert tuple(ex.KERNEL5) == tuple(ros_node.KERNEL5)
    assert ex.CANONICAL_FX == ros_node.CANONICAL_FX
    assert abs(float(np.sum(ros_node.KERNEL5)) - 1.0) < 1e-6


def test_baked_preprocess_shape_and_dtype():
    """The baked chain turns a canonical render stack into the observation tile."""
    baked = BakedPreprocess().eval()
    frames = torch.randint(0, 256, (2, RENDER_H, RENDER_W, OBS_CHANNELS), dtype=torch.uint8)
    with torch.no_grad():
        out = baked(frames)
    assert out.shape == (2, OBS_H, OBS_W, OBS_CHANNELS)
    assert out.dtype == torch.uint8


def test_baked_preprocess_matches_numpy_tail():
    """The torch and numpy implementations of S4.3 steps 5-8 agree to within 1 LSB."""
    report = numpy_tail_parity(n_frames=6, max_lsb=1)
    assert not report.skipped, report.summary()
    assert report.passed, report.summary()
    assert report.details["fraction_exact"] > 0.99, report.summary()


def test_baked_preprocess_matches_the_shared_training_implementation():
    """The graph's copy of S4.3 steps 5-8 must equal the training-side implementation exactly.

    The deploy package deliberately duplicates the operator chain so that it stays importable
    without the training stack. This is the check that stops the duplicate from drifting: any
    difference here is a sim-to-real gap that no amount of domain randomisation covers.
    """
    constants = check_shared_constants()
    if constants.skipped:
        pytest.skip(constants.reason)
    assert constants.passed, constants.summary()

    report = shared_preprocess_parity(n_frames=6, max_lsb=0)
    if report.skipped:
        pytest.skip(report.reason)
    assert report.passed, report.summary()


def test_baked_preprocess_is_a_box_filter_on_a_constant_image():
    """A constant image survives blur, downsample and crop unchanged."""
    baked = BakedPreprocess().eval()
    frames = torch.full((1, RENDER_H, RENDER_W, OBS_CHANNELS), 137, dtype=torch.uint8)
    with torch.no_grad():
        out = baked(frames)
    assert int(out.min()) == 137
    assert int(out.max()) == 137


# --------------------------------------------------------------------------------------------
# Policy semantics
# --------------------------------------------------------------------------------------------


def test_action_scaling_bounds():
    """The physical action stays inside [0, v_max] x [-omega_max, omega_max]."""

    class ConstantActor(nn.Module):
        def __init__(self, value: float) -> None:
            super().__init__()
            self.value = value

        def forward(self, rgb: Tensor, vec: Tensor) -> Tensor:
            return torch.full((rgb.shape[0], ACTION_DIM), self.value, dtype=torch.float32)

    image = torch.zeros(1, OBS_H, OBS_W, OBS_CHANNELS, dtype=torch.uint8)
    vec = torch.zeros(1, VEC_DIM)

    high = DeployablePolicy(ConstantActor(5.0)).eval()
    low = DeployablePolicy(ConstantActor(-5.0)).eval()
    with torch.no_grad():
        action_high, mu_high = high(image, vec)
        action_low, _ = low(image, vec)

    assert torch.allclose(action_high, torch.tensor([[0.6, 4.0]]), atol=1e-6)
    assert torch.allclose(action_low, torch.tensor([[0.0, -4.0]]), atol=1e-6)
    # mu is the UNCLIPPED mean: the bounds loss keeps it near the box, it is not squashed.
    assert torch.allclose(mu_high, torch.full((1, ACTION_DIM), 5.0))


def test_vec_normalisation_is_baked_and_clipped():
    """The frozen running-norm statistics are applied inside the graph, then clipped."""

    class EchoActor(nn.Module):
        def forward(self, rgb: Tensor, vec: Tensor) -> Tensor:
            return vec[:, :ACTION_DIM]

    mean = torch.arange(VEC_DIM, dtype=torch.float32)
    std = torch.full((VEC_DIM,), 2.0)
    wrapped = DeployablePolicy(EchoActor(), vec_mean=mean, vec_std=std).eval()
    image = torch.zeros(1, OBS_H, OBS_W, OBS_CHANNELS, dtype=torch.uint8)

    raw = torch.tensor([[4.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    with torch.no_grad():
        _, mu = wrapped(image, raw)
    assert torch.allclose(mu, torch.tensor([[2.0, 2.0]]), atol=1e-6)

    extreme = torch.full((1, VEC_DIM), 1000.0)
    with torch.no_grad():
        _, mu_clipped = wrapped(image, extreme)
    assert torch.allclose(mu_clipped, torch.full((1, ACTION_DIM), 5.0), atol=1e-6)


def test_policy_rejects_invalid_configuration():
    """Configuration errors fail loudly at construction, not silently at inference."""
    actor = DummyActor()
    with pytest.raises(ValueError, match="input_stage"):
        DeployablePolicy(actor, input_stage="raw")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="input_dtype"):
        DeployablePolicy(actor, input_dtype="float16")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="elements"):
        DeployablePolicy(actor, vec_mean=torch.zeros(3), vec_std=torch.ones(3))
    with pytest.raises(ValueError, match="positive"):
        DeployablePolicy(actor, vec_mean=torch.zeros(VEC_DIM), vec_std=torch.zeros(VEC_DIM))


def test_example_inputs_match_the_declared_stage(policy: DeployablePolicy, render_policy: DeployablePolicy):
    """Example inputs follow the configured stage, which is what tracing relies on."""
    image, vec = policy.example_inputs(batch=1)
    assert tuple(image.shape) == (1, OBS_H, OBS_W, OBS_CHANNELS)
    assert tuple(vec.shape) == (1, VEC_DIM)
    render_image, _ = render_policy.example_inputs(batch=1)
    assert tuple(render_image.shape) == (1, RENDER_H, RENDER_W, OBS_CHANNELS)


# --------------------------------------------------------------------------------------------
# ONNX export and parity
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("opset", [OPSET_TARGET_A, OPSET_TARGET_B])
def test_onnx_parity_both_targets(policy: DeployablePolicy, tmp_path: Path, opset: int):
    """Onnxruntime must reproduce torch to 1e-5 for both deployment targets (M11 gate)."""
    pytest.importorskip("onnxruntime", reason=ORT_HINT)
    path = export_onnx(policy, tmp_path / f"policy_opset{opset}.onnx", opset=opset)
    assert path.is_file() and path.stat().st_size > 0

    report = onnx_parity(policy, path, n_random=64, atol=1e-5)
    assert not report.skipped, report.summary()
    assert report.passed, report.summary()


def test_onnx_parity_render_stage(render_policy: DeployablePolicy, tmp_path: Path):
    """The baked preprocessing survives the export: blur, box, crop and quantise included."""
    pytest.importorskip("onnxruntime", reason=ORT_HINT)
    path = export_onnx(render_policy, tmp_path / "policy_render.onnx", opset=OPSET_TARGET_A)
    report = onnx_parity(render_policy, path, n_random=32, atol=1e-5)
    assert not report.skipped, report.summary()
    assert report.passed, report.summary()


def test_onnx_parity_float32_input(tmp_path: Path):
    """The float32 input escape hatch for TensorRT 8.2 stays bit-identical to uint8."""
    pytest.importorskip("onnxruntime", reason=ORT_HINT)
    torch.manual_seed(0)
    actor = DummyActor()
    uint8_policy = DeployablePolicy(actor, input_dtype="uint8").eval()
    float_policy = DeployablePolicy(actor, input_dtype="float32").eval()

    image = torch.randint(0, 256, (1, OBS_H, OBS_W, OBS_CHANNELS), dtype=torch.uint8)
    vec = torch.randn(1, VEC_DIM)
    with torch.no_grad():
        action_u8, _ = uint8_policy(image, vec)
        action_f32, _ = float_policy(image.float(), vec)
    assert torch.allclose(action_u8, action_f32, atol=0.0)

    path = export_onnx(float_policy, tmp_path / "policy_f32.onnx", opset=OPSET_TARGET_A)
    report = onnx_parity(float_policy, path, n_random=32, atol=1e-5)
    assert not report.skipped, report.summary()
    assert report.passed, report.summary()


def test_onnx_parity_detects_a_mismatch(policy: DeployablePolicy, tmp_path: Path):
    """The gate must FAIL when the torch model and the artifact disagree.

    A parity check that cannot fail is worthless, so this exports one policy and compares it
    against a different one.
    """
    pytest.importorskip("onnxruntime", reason=ORT_HINT)
    path = export_onnx(policy, tmp_path / "policy.onnx", opset=OPSET_TARGET_A)
    torch.manual_seed(1234)
    other = DeployablePolicy(DummyActor(hidden=8)).eval()
    with torch.no_grad():
        for parameter in other.parameters():
            parameter.add_(0.5)
    report = onnx_parity(other, path, n_random=8, atol=1e-5)
    assert not report.skipped
    assert not report.passed, "parity gate did not fire on a deliberately mismatched model"


def test_torchscript_parity(policy: DeployablePolicy, tmp_path: Path):
    """The traced artifact that drives the evaluations equals the eager policy."""
    path = save_torchscript(policy, tmp_path / "policy_traced.pt")
    report = torchscript_parity(policy, path, n_random=16, atol=1e-6)
    assert not report.skipped, report.summary()
    assert report.passed, report.summary()


def test_export_dual_targets_writes_artifacts_and_sidecars(policy: DeployablePolicy, tmp_path: Path):
    """Both targets, both sidecars and the TorchScript trace land in the output directory."""
    artifacts = export_dual_targets(policy, tmp_path, stem="policy")
    assert [a.opset for a in artifacts] == [OPSET_TARGET_A, OPSET_TARGET_B]
    assert (tmp_path / "policy_traced.pt").is_file()

    for artifact in artifacts:
        assert artifact.onnx_path.is_file()
        assert artifact.sidecar_path.is_file()
        meta = json.loads(artifact.sidecar_path.read_text(encoding="utf-8"))

        assert meta["artifact"]["sha256"] == artifact.sha256
        assert len(meta["artifact"]["sha256"]) == 64
        assert meta["opset"] == artifact.opset
        assert "trtexec" in meta["tensorrt_build_command"]
        assert "DOCUMENTED, NOT RUN" in meta["tensorrt_build_command"]
        assert "NONE" in meta["hardware_validation"]

        preprocess = meta["preprocess"]
        assert preprocess["blur_kernel"] == list(ros_node.KERNEL5)
        assert preprocess["box_downsample"] == ros_node.BOX
        assert preprocess["crop_top_rows_after_downsample"] == ros_node.CROP_TOP
        assert preprocess["color_order"] == "RGB"
        assert preprocess["baked_into_graph"]["frame_ring_and_stack"] is False
        assert preprocess["canonical_render"]["fx"] == ros_node.CANONICAL_FX

        action = meta["action"]
        assert action["action"]["units"] == ["m/s", "rad/s"]
        assert action["control_rate_hz"] == 15.0
        assert meta["normalisation"]["vec_std"] == [2.0] * VEC_DIM
        assert meta["provenance"]["spec_version"] == "v2"


def test_sidecar_is_json_serialisable_and_stable(policy: DeployablePolicy, tmp_path: Path):
    """Exporting twice gives the same graph bytes, so sidecar hashes are comparable."""
    first = export_onnx(policy, tmp_path / "a.onnx", opset=OPSET_TARGET_A)
    second = export_onnx(policy, tmp_path / "b.onnx", opset=OPSET_TARGET_A)
    assert first.read_bytes() == second.read_bytes()


# --------------------------------------------------------------------------------------------
# Checkpoint loading
# --------------------------------------------------------------------------------------------


_FACTORY_MODULE = '''
"""Throwaway actor factory used by the checkpoint-loading test."""

import torch
from torch import Tensor, nn


class TinyActor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(8, 2)

    def forward(self, rgb: Tensor, vec: Tensor) -> Tensor:
        return self.head(vec)


def build_actor() -> TinyActor:
    return TinyActor()
'''


def test_build_policy_from_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A checkpoint plus an entry-point factory rebuilds a fully configured policy."""
    module_dir = tmp_path / "factory_pkg"
    module_dir.mkdir()
    (module_dir / "dummy_factory.py").write_text(_FACTORY_MODULE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(module_dir))
    sys.modules.pop("dummy_factory", None)

    import dummy_factory  # created above on purpose

    actor = dummy_factory.build_actor()
    checkpoint = {
        "model": actor.state_dict(),
        "running_norm": {"vec": {"mean": [0.5] * VEC_DIM, "var": [4.0] * VEC_DIM}},
        "iteration": 4200,
        "global_step": 137_000_000,
        "seed": 3,
        "git_commit": "0" * 40,
        "spec_version": "v2",
    }
    checkpoint_path = tmp_path / "ckpt.pt"
    torch.save(checkpoint, checkpoint_path)

    loaded, provenance = build_policy_from_checkpoint(
        checkpoint_path, actor_factory="dummy_factory:build_actor"
    )
    assert provenance["iteration"] == 4200
    assert provenance["global_step"] == 137_000_000
    assert provenance["vec_norm_found"] is True
    assert len(provenance["sha256"]) == 64
    assert torch.allclose(loaded.vec_std, torch.full((1, VEC_DIM), 2.0), atol=1e-4)
    assert torch.allclose(loaded.vec_mean, torch.full((1, VEC_DIM), 0.5))

    image = torch.zeros(1, OBS_H, OBS_W, OBS_CHANNELS, dtype=torch.uint8)
    with torch.no_grad():
        action, _ = loaded(image, torch.zeros(1, VEC_DIM))
    assert action.shape == (1, ACTION_DIM)
    assert 0.0 <= float(action[0, 0]) <= 0.6


def test_build_policy_from_checkpoint_missing_file(tmp_path: Path):
    """A missing checkpoint fails with a clear FileNotFoundError, not a torch traceback."""
    with pytest.raises(FileNotFoundError):
        build_policy_from_checkpoint(tmp_path / "nope.pt", actor_factory="torch.nn:Identity")


# --------------------------------------------------------------------------------------------
# ROS-side helpers that are pure enough to unit test without ROS
# --------------------------------------------------------------------------------------------


def test_ros_node_imports_without_ros_or_torch():
    """The node module must import in a bare environment; only the class needs ROS."""
    assert hasattr(ros_node, "LaneFollowingNode")
    assert hasattr(ros_node, "preprocess_tail_numpy")
    assert "torch" not in ros_node.__dict__


def test_bgr_to_rgb_is_a_real_swap():
    """The single most common sim-to-real bug gets its own assertion."""
    bgr = np.zeros((2, 2, 3), dtype=np.uint8)
    bgr[..., 0] = 10  # blue
    bgr[..., 2] = 30  # red
    rgb = ros_node.bgr_to_rgb(bgr)
    assert int(rgb[0, 0, 0]) == 30
    assert int(rgb[0, 0, 2]) == 10
    with pytest.raises(ValueError):
        ros_node.bgr_to_rgb(np.zeros((2, 2), dtype=np.uint8))


def test_frame_ring_stacking_semantics():
    """The ring reproduces the trainer's (t, t-2, t-4) stack and its reset behaviour."""
    ring = ros_node.FrameRing()
    first = np.full((OBS_H, OBS_W, 3), 1, dtype=np.uint8)
    ring.push(first)
    obs = ring.observation()
    assert obs.shape == (1, OBS_H, OBS_W, OBS_CHANNELS)
    assert np.all(obs == 1), "the first observation of an episode is three copies of one frame"

    for value in (2, 3, 4, 5):
        ring.push(np.full((OBS_H, OBS_W, 3), value, dtype=np.uint8))
    obs = ring.observation()
    assert [int(obs[0, 0, 0, i]) for i in (0, 3, 6)] == [5, 3, 1]

    with pytest.raises(RuntimeError):
        ros_node.FrameRing().observation()
    with pytest.raises(ValueError):
        ros_node.FrameRing(depth=3, stack_len=3, stride=2)


def test_canonical_camera_matrix_matches_the_spec():
    """K_canon is the one camera every pipeline targets (S4.1)."""
    assert ros_node.K_CANON[0, 0] == pytest.approx(65.98)
    assert ros_node.K_CANON[1, 1] == pytest.approx(65.98)
    assert ros_node.K_CANON[0, 2] == pytest.approx(RENDER_W / 2)
    assert ros_node.K_CANON[1, 2] == pytest.approx(RENDER_H / 2)


def test_preprocess_tail_rejects_wrong_geometry():
    """A frame that is not the canonical render is rejected instead of silently resized."""
    with pytest.raises(ValueError, match="render"):
        ros_node.preprocess_tail_numpy(np.zeros((64, 64, 3), dtype=np.uint8))
