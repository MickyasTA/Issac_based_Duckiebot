"""Diagonal Gaussian head checked against ``torch.distributions`` ground truth."""

from __future__ import annotations

import math

import pytest
import torch
from torch.distributions import Normal, kl_divergence

from duckiebot_rl.ppo.distributions import (
    DEFAULT_LOG_STD_MAX,
    DEFAULT_LOG_STD_MIN,
    DiagGaussianHead,
    approx_kl_k3,
    diag_gaussian_entropy,
    diag_gaussian_kl,
    diag_gaussian_log_prob,
    orthogonal_init_,
    tanh_log_det,
)


def _random_params(batch: int = 64, act_dim: int = 3) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draw a random sample, mean and log-std triple.

    Args:
        batch: Batch size.
        act_dim: Action dimensionality.

    Returns:
        Tuple ``(x, mu, log_std)``.
    """
    torch.manual_seed(7)
    mu = torch.randn(batch, act_dim) * 2.0
    log_std = torch.randn(batch, act_dim) * 0.6 - 0.3
    x = mu + log_std.exp() * torch.randn(batch, act_dim)
    return x, mu, log_std


def test_log_prob_matches_torch_distributions() -> None:
    """The hand-written log density equals ``Normal.log_prob`` summed over the action dim."""
    x, mu, log_std = _random_params()
    reference = Normal(mu, log_std.exp()).log_prob(x).sum(-1)
    torch.testing.assert_close(diag_gaussian_log_prob(x, mu, log_std), reference, rtol=0, atol=1e-5)


def test_entropy_matches_torch_distributions() -> None:
    """The closed-form entropy equals ``Normal.entropy`` summed over the action dim."""
    _, mu, log_std = _random_params()
    reference = Normal(mu, log_std.exp()).entropy().sum(-1)
    torch.testing.assert_close(diag_gaussian_entropy(log_std), reference, rtol=0, atol=1e-5)


def test_analytic_kl_matches_torch_distributions() -> None:
    """The analytic diagonal KL equals ``kl_divergence`` and is direction-correct."""
    torch.manual_seed(11)
    mu_p, mu_q = torch.randn(32, 4), torch.randn(32, 4)
    log_std_p, log_std_q = torch.randn(32, 4) * 0.3, torch.randn(32, 4) * 0.3
    reference = kl_divergence(Normal(mu_p, log_std_p.exp()), Normal(mu_q, log_std_q.exp())).sum(-1)
    ours = diag_gaussian_kl(mu_p, log_std_p, mu_q, log_std_q)
    torch.testing.assert_close(ours, reference, rtol=0, atol=1e-5)
    assert bool((ours >= 0).all())
    # Asymmetric: swapping the arguments must change the value.
    swapped = diag_gaussian_kl(mu_q, log_std_q, mu_p, log_std_p)
    assert not torch.allclose(ours, swapped, atol=1e-3)


def test_analytic_kl_is_zero_for_identical_distributions() -> None:
    """KL(p || p) is exactly zero."""
    _, mu, log_std = _random_params()
    torch.testing.assert_close(
        diag_gaussian_kl(mu, log_std, mu, log_std), torch.zeros(mu.shape[0]), rtol=0, atol=1e-6
    )


def test_k3_estimator_is_non_negative_and_unbiased() -> None:
    """The k3 estimator is non-negative everywhere and tracks the true KL on a large sample."""
    torch.manual_seed(0)
    q = Normal(torch.tensor(0.1), torch.tensor(1.1))
    p = Normal(torch.tensor(0.0), torch.tensor(1.0))
    samples = q.sample((400_000,))
    log_ratio = p.log_prob(samples) - q.log_prob(samples)
    k3 = approx_kl_k3(log_ratio)
    assert bool((k3 >= 0).all())
    true_kl = kl_divergence(q, p).item()
    assert k3.mean().item() == pytest.approx(true_kl, rel=0.05)
    # k1 is the sign-indefinite estimator k3 replaces.
    assert bool(((-log_ratio) < 0).any())


def test_tanh_log_det_is_finite_where_the_naive_form_overflows() -> None:
    """The stable identity stays finite at ``|x| = 20`` where ``log(1 - tanh^2 x)`` is ``-inf``."""
    x = torch.tensor([-20.0, -5.0, 0.0, 5.0, 20.0], dtype=torch.float64)
    stable = tanh_log_det(x)
    naive = torch.log(1.0 - torch.tanh(x) ** 2)
    assert bool(torch.isfinite(stable).all())
    assert bool(torch.isinf(naive[0])) and bool(torch.isinf(naive[-1]))
    torch.testing.assert_close(stable[1:4], naive[1:4], rtol=0, atol=1e-6)
    assert stable[2].item() == pytest.approx(0.0, abs=1e-12)


def test_head_initialises_log_std_to_log_half() -> None:
    """SPEC v2 S6.2: sigma starts at 0.5, not at torch's default 1.0."""
    head = DiagGaussianHead(feat_dim=16, act_dim=2)
    torch.testing.assert_close(
        head.log_std_param.detach(), torch.full((1, 2), math.log(0.5)), rtol=0, atol=1e-7
    )
    assert head.log_std_param.requires_grad
    assert head.log_std_param.shape == (1, 2)


def test_log_std_is_clamped_in_both_directions() -> None:
    """A runaway ``log_std`` parameter is clamped to ``[-5, 2]`` before use."""
    head = DiagGaussianHead(feat_dim=4, act_dim=2)
    with torch.no_grad():
        head.log_std_param.copy_(torch.tensor([[-40.0, 40.0]]))
    clamped = head.clamped_log_std()
    torch.testing.assert_close(
        clamped, torch.tensor([[DEFAULT_LOG_STD_MIN, DEFAULT_LOG_STD_MAX]]), rtol=0, atol=0
    )
    out = head(torch.zeros(3, 4))
    assert bool(torch.isfinite(out.log_prob).all())
    assert bool(torch.isfinite(out.entropy).all())


def test_evaluating_a_stored_action_reproduces_its_log_prob_exactly() -> None:
    """The update path must return the same log-prob the rollout path produced. Ratio == 1."""
    torch.manual_seed(3)
    head = DiagGaussianHead(feat_dim=8, act_dim=2)
    features = torch.randn(50, 8)
    sampled = head(features)
    evaluated = head(features, action=sampled.action)
    torch.testing.assert_close(evaluated.log_prob, sampled.log_prob, rtol=0, atol=0)
    ratio = (evaluated.log_prob - sampled.log_prob).exp()
    torch.testing.assert_close(ratio, torch.ones_like(ratio), rtol=0, atol=0)


def test_deterministic_mode_returns_the_mean() -> None:
    """Evaluation uses ``a = mu``, with no sampling noise."""
    torch.manual_seed(5)
    head = DiagGaussianHead(feat_dim=8, act_dim=2)
    features = torch.randn(10, 8)
    out = head(features, deterministic=True)
    torch.testing.assert_close(out.action, out.mu, rtol=0, atol=0)


def test_sampling_is_reproducible_under_a_seed() -> None:
    """Two identically seeded draws from the same head agree bitwise."""
    head = DiagGaussianHead(feat_dim=8, act_dim=2)
    features = torch.randn(16, 8)
    torch.manual_seed(99)
    first = head(features).action
    torch.manual_seed(99)
    second = head(features).action
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_squashed_mode_applies_the_jacobian_to_log_prob_and_entropy() -> None:
    """With ``squash=True`` the correction is subtracted from log-prob and added to entropy."""
    torch.manual_seed(13)
    raw = DiagGaussianHead(feat_dim=8, act_dim=2, squash=False)
    squashed = DiagGaussianHead(feat_dim=8, act_dim=2, squash=True)
    squashed.load_state_dict(raw.state_dict())
    features = torch.randn(12, 8)
    action = torch.randn(12, 2)
    a = raw(features, action=action)
    b = squashed(features, action=action)
    correction = tanh_log_det(action).sum(-1)
    torch.testing.assert_close(b.log_prob, a.log_prob - correction, rtol=0, atol=1e-6)
    torch.testing.assert_close(b.entropy, a.entropy + correction, rtol=0, atol=1e-6)


def test_orthogonal_init_produces_an_orthonormal_scaled_matrix() -> None:
    """``orthogonal_init_`` gives ``W W^T = gain^2 I`` for a wide layer, and a zero bias."""
    layer = torch.nn.Linear(64, 8)
    orthogonal_init_(layer, gain=math.sqrt(2.0))
    gram = layer.weight @ layer.weight.T
    torch.testing.assert_close(gram, 2.0 * torch.eye(8), rtol=0, atol=1e-4)
    torch.testing.assert_close(layer.bias, torch.zeros(8), rtol=0, atol=0)


def test_invalid_construction_arguments_raise() -> None:
    """Bad sigma or clamp windows fail at construction, not at the first NaN."""
    with pytest.raises(ValueError, match="sigma_init must be positive"):
        DiagGaussianHead(feat_dim=4, act_dim=2, sigma_init=0.0)
    with pytest.raises(ValueError, match="log_std_min < log_std_max"):
        DiagGaussianHead(feat_dim=4, act_dim=2, log_std_min=1.0, log_std_max=-1.0)
    with pytest.raises(ValueError, match="outside the clamp window"):
        DiagGaussianHead(feat_dim=4, act_dim=2, sigma_init=100.0)
