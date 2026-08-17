"""Running mean/std normaliser: correctness, chunk invariance, round-trip, state_dict."""

from __future__ import annotations

import pytest
import torch

from duckiebot_rl.ppo.running_norm import RunningMeanStd


def test_statistics_converge_to_the_population_values() -> None:
    """After one large batch the running mean and variance match the batch statistics."""
    torch.manual_seed(0)
    data = torch.randn(20_000, 4) * torch.tensor([1.0, 3.0, 0.5, 10.0]) + torch.tensor([-2.0, 0.0, 5.0, 1.0])
    norm = RunningMeanStd((4,), epsilon=1e-8)
    norm.update(data)
    torch.testing.assert_close(norm.mean, data.mean(0), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(norm.var, data.var(0, unbiased=False), rtol=1e-3, atol=1e-3)


def test_chunked_updates_equal_a_single_update() -> None:
    """The parallel variance update is exact, so batching order cannot change the result."""
    torch.manual_seed(1)
    data = torch.randn(3000, 3) * 4.0 + 1.5
    single = RunningMeanStd((3,), epsilon=1e-8)
    single.update(data)
    chunked = RunningMeanStd((3,), epsilon=1e-8)
    for chunk in data.split(137):
        chunked.update(chunk)
    torch.testing.assert_close(chunked.mean, single.mean, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(chunked.var, single.var, rtol=1e-4, atol=1e-4)


def test_normalize_produces_zero_mean_unit_variance() -> None:
    """Normalising the data the statistics were fitted on gives a standard distribution."""
    torch.manual_seed(2)
    data = torch.randn(5000, 2) * 7.0 - 3.0
    norm = RunningMeanStd((2,), epsilon=1e-8)
    norm.update(data)
    out = norm.normalize(data)
    torch.testing.assert_close(out.mean(0), torch.zeros(2), rtol=0, atol=1e-3)
    torch.testing.assert_close(out.std(0), torch.ones(2), rtol=0, atol=1e-2)


def test_normalize_denormalize_round_trip() -> None:
    """``denormalize`` inverts ``normalize`` when no clipping is applied."""
    torch.manual_seed(3)
    norm = RunningMeanStd((5,))
    norm.update(torch.randn(500, 5) * 2.0 + 1.0)
    x = torch.randn(17, 5)
    torch.testing.assert_close(norm.denormalize(norm.normalize(x)), x, rtol=1e-5, atol=1e-5)


def test_clipping_bounds_the_output() -> None:
    """SPEC v2 S5.2 clips normalised vector observations to +/- 5."""
    norm = RunningMeanStd((1,), epsilon=1e-8)
    norm.update(torch.randn(1000, 1))
    out = norm.normalize(torch.full((10, 1), 500.0), clip=5.0)
    assert float(out.max()) == pytest.approx(5.0)
    assert float(out.min()) == pytest.approx(5.0)


def test_scalar_shape_tracks_the_value_targets() -> None:
    """A ``()`` shaped normaliser reduces every dimension, which is what value targets need."""
    norm = RunningMeanStd((), epsilon=1e-8)
    data = torch.randn(32, 256) * 3.0 + 10.0
    norm.update(data)
    assert norm.mean.shape == ()
    assert float(norm.mean) == pytest.approx(float(data.mean()), abs=1e-3)


def test_empty_update_is_a_no_op() -> None:
    """An update with zero rows leaves the statistics untouched."""
    norm = RunningMeanStd((2,))
    before = (norm.mean.clone(), norm.var.clone(), norm.count.clone())
    norm.update(torch.zeros(0, 2))
    torch.testing.assert_close(norm.mean, before[0], rtol=0, atol=0)
    torch.testing.assert_close(norm.count, before[2], rtol=0, atol=0)


def test_shape_mismatch_raises() -> None:
    """A wrong trailing shape raises instead of silently broadcasting."""
    norm = RunningMeanStd((4,))
    with pytest.raises(ValueError, match="expected trailing shape"):
        norm.update(torch.zeros(10, 3))


def test_state_dict_round_trip_restores_the_statistics() -> None:
    """Buffers travel through ``state_dict``, which is how they reach the checkpoint."""
    torch.manual_seed(4)
    source = RunningMeanStd((3,))
    source.update(torch.randn(200, 3) * 5.0)
    target = RunningMeanStd((3,))
    target.load_state_dict(source.state_dict())
    torch.testing.assert_close(target.mean, source.mean, rtol=0, atol=0)
    torch.testing.assert_close(target.var, source.var, rtol=0, atol=0)
    torch.testing.assert_close(target.count, source.count, rtol=0, atol=0)
