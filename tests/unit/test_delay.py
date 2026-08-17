"""Unit tests for the per-env variable-delay ring buffer (SPEC v2 S7.3 D8/D9).

Covers: a delay of k steps genuinely delays by k, per-env delays are independent, the ring wraps
around correctly over many pushes, sub-step interpolation is exact, the torch and numpy backends
agree, and reset/serialization behave.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from duckiebot_rl.dr.delay import DelayBuffer

BACKENDS = ["torch", "numpy"]


def _buf(
    num_envs: int = 3,
    feature: tuple[int, ...] = (2,),
    max_delay: int = 4,
    backend: str = "torch",
) -> DelayBuffer:
    return DelayBuffer(num_envs, feature, max_delay=max_delay, backend=backend)


def _payload(t: float, num_envs: int = 3, backend: str = "torch") -> object:
    data = [[t, -t] for _ in range(num_envs)]
    return np.array(data, dtype=np.float32) if backend == "numpy" else torch.tensor(data)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("k", [0, 1, 2, 3, 4])
def test_a_delay_of_k_steps_delays_by_exactly_k(backend, k):
    b = _buf(backend=backend)
    b.reset(value=_payload(0.0, backend=backend))
    b.set_delay(k, 0.0)
    for t in range(1, 20):
        out = np.asarray(b.step(_payload(float(t), backend=backend)))
        expected = float(max(t - k, 0))
        assert out[0, 0] == pytest.approx(expected)
        assert out[0, 1] == pytest.approx(-expected)


@pytest.mark.parametrize("backend", BACKENDS)
def test_per_env_variable_delays_are_independent(backend):
    b = _buf(num_envs=5, max_delay=4, backend=backend)
    b.reset(value=_payload(0.0, 5, backend=backend))
    delays = np.array([0, 1, 2, 3, 4])
    b.set_delay(delays if backend == "numpy" else torch.from_numpy(delays), 0.0)
    for t in range(1, 30):
        out = np.asarray(b.step(_payload(float(t), 5, backend=backend)))
        for env, k in enumerate(delays):
            assert out[env, 0] == pytest.approx(float(max(t - k, 0)))


@pytest.mark.parametrize("backend", BACKENDS)
def test_ring_wraparound_over_many_pushes(backend):
    b = _buf(max_delay=3, backend=backend)
    b.reset(value=_payload(0.0, backend=backend))
    b.set_delay(3, 0.0)
    last = None
    for t in range(1, 1000):
        last = np.asarray(b.step(_payload(float(t), backend=backend)))
    assert last[0, 0] == pytest.approx(996.0)
    # depth = max_delay + 2, so 999 pushes wrapped the 5-slot ring ~200 times.
    assert b.depth == 5


@pytest.mark.parametrize("backend", BACKENDS)
def test_substep_interpolation(backend):
    b = _buf(backend=backend)
    b.reset(value=_payload(0.0, backend=backend))
    b.set_delay(1, 0.25)
    for t in range(1, 10):
        out = np.asarray(b.step(_payload(float(t), backend=backend)))
    # delay 1 + 0.25 of a step: 0.75 * x(t-1) + 0.25 * x(t-2)
    assert out[0, 0] == pytest.approx(0.75 * 8.0 + 0.25 * 7.0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_tap_reads_the_requested_history(backend):
    b = _buf(backend=backend)
    b.reset(value=_payload(0.0, backend=backend))
    for t in range(1, 10):
        b.push(_payload(float(t), backend=backend))
    for k in range(5):
        assert np.asarray(b.tap(k))[0, 0] == pytest.approx(float(9 - k))


def test_torch_and_numpy_backends_agree():
    bt = _buf(num_envs=4, max_delay=3, backend="torch")
    bn = _buf(num_envs=4, max_delay=3, backend="numpy")
    bt.reset(value=_payload(0.0, 4))
    bn.reset(value=_payload(0.0, 4, backend="numpy"))
    d = np.array([0, 1, 2, 3])
    f = np.array([0.0, 0.3, 0.6, 0.9], dtype=np.float32)
    bt.set_delay(torch.from_numpy(d), torch.from_numpy(f))
    bn.set_delay(d, f)
    for t in range(1, 50):
        a = bt.step(_payload(float(t), 4)).numpy()
        c = bn.step(_payload(float(t), 4, backend="numpy"))
        assert np.allclose(a, c, atol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_reset_clears_history_per_env(backend):
    b = _buf(num_envs=2, max_delay=2, backend=backend)
    b.set_delay(2, 0.0)
    for t in range(1, 10):
        b.push(_payload(float(t), 2, backend=backend))
    ids = np.array([0]) if backend == "numpy" else torch.tensor([0])
    fresh = _payload(-5.0, 1, backend=backend)
    b.reset(ids, fresh)
    out = np.asarray(b.get())
    assert out[0, 0] == pytest.approx(-5.0)
    assert out[1, 0] == pytest.approx(7.0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_integer_payloads_are_not_interpolated(backend):
    dtype = np.uint8 if backend == "numpy" else torch.uint8
    b = DelayBuffer(2, (1,), max_delay=2, dtype=dtype, backend=backend)
    assert b.interpolate is False
    b.reset()
    for t in range(1, 10):
        payload = np.full((2, 1), t, dtype=np.uint8)
        b.push(payload if backend == "numpy" else torch.from_numpy(payload))
    b.set_delay(1, 0.9)
    assert int(np.asarray(b.get())[0, 0]) == 8


def test_scalar_feature_shape():
    b = DelayBuffer(3, (), max_delay=2)
    b.reset()
    b.set_delay(2, 0.0)
    for t in range(1, 6):
        out = b.step(torch.full((3,), float(t)))
    assert out.shape == (3,)
    assert float(out[0]) == pytest.approx(3.0)


def test_invalid_configuration_raises():
    with pytest.raises(ValueError, match="max_delay"):
        DelayBuffer(2, (1,), max_delay=-1)
    with pytest.raises(ValueError, match="unknown backend"):
        DelayBuffer(2, (1,), backend="jax")
    b = _buf()
    with pytest.raises(ValueError, match="delay steps must be"):
        b.set_delay(99)
    with pytest.raises(ValueError, match="push expected shape"):
        b.push(torch.zeros((3, 5)))


def test_state_dict_round_trip():
    a = _buf(max_delay=3)
    a.reset(value=_payload(0.0))
    a.set_delay(2, 0.5)
    for t in range(1, 12):
        a.push(_payload(float(t)))
    b = _buf(max_delay=3)
    b.load_state_dict(a.state_dict())
    assert torch.equal(a.get(), b.get())
    assert torch.equal(a.step(_payload(99.0)), b.step(_payload(99.0)))
