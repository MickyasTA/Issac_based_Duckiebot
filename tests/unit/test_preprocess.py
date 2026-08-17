"""Unit tests for the canonical preprocessing chain (SPEC v2 S4.3).

Covers: torch/numpy numerical parity, output shapes and dtypes, the cv2 resize-kernel parity
claim, frame stacking at the ring boundaries, the D9 observation delay, V19 frame repeat and
determinism under a seed.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from duckiebot_rl.dr import preprocess as pp
from duckiebot_rl.dr.preprocess import FrameStack

N_ENVS = 4


def _frames(n: int = N_ENVS, seed: int = 0) -> np.ndarray:
    """Random render-resolution frames with hard edges (worst case for a resize kernel)."""
    rng = np.random.default_rng(seed)
    f = rng.integers(0, 256, size=(n, pp.RENDER_H, pp.RENDER_W, 3), dtype=np.uint8)
    # Paint a 24 mm-tape-like thin bright stripe and a black road patch: the exact structures
    # the sim/real resize-kernel mismatch would destroy.
    f[:, :, 90:92, :] = 255
    f[:, 60:90, 20:60, :] = 0
    f[:, 30:31, :, 0] = 255
    return f


# ---------------------------------------------------------------------------- shapes / dtypes


def test_output_shape_and_dtype_torch():
    out = pp.preprocess_frame(torch.from_numpy(_frames()))
    assert out.shape == (N_ENVS, pp.OBS_H, pp.OBS_W, 3)
    assert out.dtype == torch.uint8
    assert int(out.min()) >= 0 and int(out.max()) <= 255


def test_output_shape_and_dtype_numpy():
    out = pp.preprocess_frame_np(_frames())
    assert out.shape == (N_ENVS, pp.OBS_H, pp.OBS_W, 3)
    assert out.dtype == np.uint8


def test_spec_constants_are_consistent():
    assert pp.RENDER_W // pp.BOX == pp.OBS_W
    assert pp.RENDER_H // pp.BOX == pp.OBS_H + pp.CROP_TOP
    assert abs(sum(pp.KERNEL5) - 1.0) < 1e-12
    assert 3 * len(pp.FRAME_STACK_OFFSETS) == pp.OBS_CHANNELS


def test_bad_input_shape_raises():
    with pytest.raises(ValueError, match="expected NHWC"):
        pp.preprocess_frame_np(np.zeros((2, 64, 64, 3), dtype=np.uint8))


# ---------------------------------------------------------------------------- torch/numpy parity


def test_blur_parity():
    x = _frames().astype(np.float32).transpose(0, 3, 1, 2) / 255.0
    a = pp.blur5(torch.from_numpy(x)).numpy()
    b = pp.blur5_np(x)
    assert np.max(np.abs(a - b)) < 1e-6


def test_box_downsample_parity_is_bitwise():
    x = _frames().astype(np.float32).transpose(0, 3, 1, 2) / 255.0
    a = pp.box_downsample(torch.from_numpy(x)).numpy()
    b = pp.box_downsample_np(x)
    assert np.array_equal(a, b)


def test_shift_principal_point_parity():
    x = _frames().astype(np.float32).transpose(0, 3, 1, 2) / 255.0
    dx = np.array([-2, 0, 1, 2], dtype=np.int64)
    dy = np.array([2, -1, 0, -2], dtype=np.int64)
    a = pp.shift_principal_point(torch.from_numpy(x), torch.from_numpy(dx), torch.from_numpy(dy))
    b = pp.shift_principal_point_np(x, dx, dy)
    assert np.array_equal(a.numpy(), b)


def test_full_chain_parity_torch_vs_numpy():
    frames = _frames(seed=3)
    a = pp.preprocess_frame(torch.from_numpy(frames)).numpy().astype(np.int16)
    b = pp.preprocess_frame_np(frames).astype(np.int16)
    diff = np.abs(a - b)
    assert diff.max() <= 1
    assert (diff == 0).mean() >= 0.999


def test_float_tail_parity_is_tight():
    x = _frames(seed=5).astype(np.float32).transpose(0, 3, 1, 2) / 255.0
    a = pp.crop_rows(pp.box_downsample(pp.blur5(torch.from_numpy(x)))).numpy()
    b = pp.crop_rows_np(pp.box_downsample_np(pp.blur5_np(x)))
    assert np.max(np.abs(a - b)) < 1e-6


# ---------------------------------------------------------------------------- cv2 parity (S4.3)


def test_cv2_parity_of_the_tail():
    pytest.importorskip("cv2")
    frames = _frames(seed=7)
    ours = pp.preprocess_frame_np(frames).astype(np.int16)
    for i in range(frames.shape[0]):
        ref = pp.tail_cv2(frames[i]).astype(np.int16)
        diff = np.abs(ours[i] - ref)
        assert diff.max() <= 2
        assert (diff <= 1).mean() >= 0.999


def test_box_downsample_equals_inter_area_at_2x():
    cv2 = pytest.importorskip("cv2")
    x = _frames(seed=11)[0].astype(np.float32) / 255.0
    ours = pp.box_downsample_np(x.transpose(2, 0, 1)[None])[0].transpose(1, 2, 0)
    ref = cv2.resize(x, (pp.OBS_W, pp.RENDER_H // pp.BOX), interpolation=cv2.INTER_AREA)
    assert np.max(np.abs(ours - ref)) < 1e-6


# ---------------------------------------------------------------------------- principal point


def test_principal_point_shift_moves_content_and_replicates_border():
    x = np.zeros((1, 3, pp.RENDER_H, pp.RENDER_W), dtype=np.float32)
    x[:, :, :, 10] = 1.0
    x[:, :, :, 0] = 0.5
    out = pp.shift_principal_point_np(x, np.array([3]), np.array([0]))
    assert np.allclose(out[:, :, :, 13], 1.0)
    assert np.allclose(out[:, :, :, 10], 0.0)
    # Columns 0..2 replicate the original column 0 rather than showing black bars.
    assert np.allclose(out[:, :, :, 0:4], 0.5)


def test_zero_shift_is_identity():
    x = _frames().astype(np.float32).transpose(0, 3, 1, 2) / 255.0
    z = np.zeros(N_ENVS, dtype=np.int64)
    assert np.array_equal(pp.shift_principal_point_np(x, z, z), x)


# ---------------------------------------------------------------------------- frame stacking


def _const_frame(value: int, n: int = N_ENVS, backend: str = "torch") -> object:
    shape = (n, pp.OBS_H, pp.OBS_W, 3)
    if backend == "numpy":
        return np.full(shape, value, dtype=np.uint8)
    return torch.full(shape, value, dtype=torch.uint8)


@pytest.mark.parametrize("backend", ["torch", "numpy"])
def test_frame_stack_shape_and_reset(backend):
    fs = FrameStack(N_ENVS, backend=backend)
    fs.reset(frame=_const_frame(7, backend=backend))
    obs = fs.get()
    assert tuple(obs.shape) == fs.stacked_shape == (N_ENVS, pp.OBS_H, pp.OBS_W, 9)
    assert int(np.asarray(obs).min()) == 7 and int(np.asarray(obs).max()) == 7


@pytest.mark.parametrize("backend", ["torch", "numpy"])
def test_frame_stack_indexes_t_tm2_tm4(backend):
    fs = FrameStack(N_ENVS, backend=backend)
    fs.reset(frame=_const_frame(0, backend=backend))
    for v in range(1, 11):
        fs.push(_const_frame(v, backend=backend))
    obs = np.asarray(fs.get())
    # Newest frame first: value 10 at t, 8 at t-2, 6 at t-4.
    assert obs[0, 0, 0, 0] == 10
    assert obs[0, 0, 0, 3] == 8
    assert obs[0, 0, 0, 6] == 6


def test_frame_stack_boundary_after_reset_has_no_leakage():
    fs = FrameStack(N_ENVS)
    for v in range(1, 8):
        fs.push(_const_frame(v))
    # Reset only env 0 with a distinctive frame: its whole stack must be that frame, while the
    # other envs keep their history. This is the terminal-observation boundary case of S6.4.
    fs.reset(torch.tensor([0]), _const_frame(99, n=1))
    obs = fs.get().numpy()
    assert np.all(obs[0] == 99)
    assert obs[1, 0, 0, 0] == 7 and obs[1, 0, 0, 3] == 5 and obs[1, 0, 0, 6] == 3


def test_frame_stack_wraparound_is_correct_over_many_pushes():
    fs = FrameStack(2)
    fs.reset(frame=_const_frame(0, n=2))
    for v in range(1, 201):
        fs.push(_const_frame(v % 251, n=2))
    obs = fs.get().numpy()
    assert obs[0, 0, 0, 0] == 200 % 251
    assert obs[0, 0, 0, 3] == 198 % 251
    assert obs[0, 0, 0, 6] == 196 % 251


def test_observation_delay_shifts_the_whole_stack():
    fs = FrameStack(3)
    fs.reset(frame=_const_frame(0, n=3))
    fs.set_obs_delay(torch.tensor([0, 1, 2]))
    for v in range(1, 21):
        fs.push(_const_frame(v, n=3))
    obs = fs.get().numpy()
    for env, d in enumerate((0, 1, 2)):
        assert obs[env, 0, 0, 0] == 20 - d
        assert obs[env, 0, 0, 3] == 18 - d
        assert obs[env, 0, 0, 6] == 16 - d


def test_frame_repeat_mask_repeats_previous_frame():
    fs = FrameStack(2)
    fs.reset(frame=_const_frame(0, n=2))
    fs.push(_const_frame(5, n=2))
    fs.push(_const_frame(9, n=2), repeat_mask=torch.tensor([True, False]))
    obs = fs.get().numpy()
    assert obs[0, 0, 0, 0] == 5  # env 0 repeated the previous frame
    assert obs[1, 0, 0, 0] == 9  # env 1 got the new one


def test_frame_stack_ring_is_deep_enough_for_max_delay_and_offset():
    fs = FrameStack(1)
    assert fs._ring.depth >= pp.MAX_OBS_DELAY + max(pp.FRAME_STACK_OFFSETS) + 1


def test_frame_stack_torch_and_numpy_agree():
    ft = FrameStack(2, backend="torch")
    fn = FrameStack(2, backend="numpy")
    ft.reset(frame=_const_frame(0, n=2))
    fn.reset(frame=_const_frame(0, n=2, backend="numpy"))
    ft.set_obs_delay(torch.tensor([0, 2]))
    fn.set_obs_delay(np.array([0, 2]))
    for v in range(1, 30):
        ft.push(_const_frame(v, n=2))
        fn.push(_const_frame(v, n=2, backend="numpy"))
    assert np.array_equal(ft.get().numpy(), fn.get())


# ---------------------------------------------------------------------------- determinism


def test_determinism_under_seed_with_photometric_dr():
    from duckiebot_rl.dr.visual import VisualDR

    frames = torch.from_numpy(_frames(seed=13))
    outs = []
    for _ in range(2):
        g = torch.Generator().manual_seed(1234)
        dr = VisualDR(N_ENVS, device=None, generator=g)
        params = dr.sample(alpha=1.0)
        outs.append(pp.preprocess_frame(frames, photometric=dr.operator(params)))
    assert torch.equal(outs[0], outs[1])


def test_different_seeds_give_different_dr():
    from duckiebot_rl.dr.visual import VisualDR

    frames = torch.from_numpy(_frames(seed=13))
    outs = []
    for seed in (1, 2):
        g = torch.Generator().manual_seed(seed)
        dr = VisualDR(N_ENVS, device=None, generator=g)
        outs.append(pp.preprocess_frame(frames, photometric=dr.operator(dr.sample(alpha=1.0))))
    assert not torch.equal(outs[0], outs[1])


def test_gaussian_kernel1d():
    k = pp.gaussian_kernel1d(0.6, 2)
    assert len(k) == 5
    assert abs(sum(k) - 1.0) < 1e-12
    assert k[0] == pytest.approx(k[4])
    # The hardcoded KERNEL5 is exactly this kernel rounded to 5 decimals.
    for a, b in zip(k, pp.KERNEL5, strict=True):
        assert abs(a - b) < 1e-4


# ---------------------------------------------------------------------------- torch-free path

# Executed in a subprocess whose import system refuses to load torch, proving that the
# preprocessing chain and the delay ring really do run in the MuJoCo venv and on the Jetson,
# where torch is absent (critic finding on mujoco_venv).
TORCH_FREE_SCRIPT = """
import sys
class Block:
    def find_module(self, name, path=None):
        return self if name == 'torch' or name.startswith('torch.') else None
    def load_module(self, name):
        raise ImportError('torch is blocked for this test')
sys.meta_path.insert(0, Block())
import numpy as np
from duckiebot_rl.dr import preprocess as pp
from duckiebot_rl.dr.delay import DelayBuffer
assert 'torch' not in sys.modules
f = np.zeros((2, pp.RENDER_H, pp.RENDER_W, 3), dtype=np.uint8)
assert pp.preprocess_frame_np(f).shape == (2, pp.OBS_H, pp.OBS_W, 3)
fs = pp.FrameStack(2, backend='numpy')
fs.reset(frame=np.full((2, pp.OBS_H, pp.OBS_W, 3), 3, dtype=np.uint8))
assert fs.get().shape == (2, pp.OBS_H, pp.OBS_W, 9)
b = DelayBuffer(2, (2,), max_delay=2, backend='numpy')
b.reset()
b.set_delay(2, 0.0)
for t in range(1, 6):
    out = b.step(np.full((2, 2), float(t), dtype=np.float32))
assert out[0, 0] == 3.0
assert 'torch' not in sys.modules
print('TORCH_FREE_OK')
"""


def test_preprocess_and_delay_import_and_run_without_torch():
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=str(root))
    proc = subprocess.run(
        [sys.executable, "-c", TORCH_FREE_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "TORCH_FREE_OK" in proc.stdout


# ---------------------------------------------------------------------------- GPU (opt-in)


@pytest.mark.gpu
def test_chain_runs_on_cuda_and_is_deterministic():
    from duckiebot_rl.dr.visual import VisualDR

    assert torch.cuda.is_available()
    dev = "cuda"
    n = 8
    frames = torch.randint(0, 256, (n, pp.RENDER_H, pp.RENDER_W, 3), dtype=torch.uint8, device=dev)
    outs = []
    for _ in range(2):
        g = torch.Generator(device=dev).manual_seed(7)
        dr = VisualDR(n, device=dev, generator=g)
        stack = FrameStack(n, device=dev)
        stack.reset(frame=torch.zeros((n, pp.OBS_H, pp.OBS_W, 3), dtype=torch.uint8, device=dev))
        stack.set_obs_delay(torch.tensor([0, 1, 2, 0, 1, 2, 0, 1], device=dev))
        obs = pp.preprocess_frame(frames, photometric=dr.operator(dr.sample(1.0)))
        outs.append(stack.step(obs))
    assert outs[0].device.type == "cuda"
    assert outs[0].shape == (n, pp.OBS_H, pp.OBS_W, 9)
    assert torch.equal(outs[0], outs[1])
