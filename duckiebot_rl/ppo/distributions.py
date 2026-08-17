"""Diagonal Gaussian policy distribution with a state-independent log standard deviation.

SPEC v2 S6.2 fixes the policy distribution for this project:

* raw ``Normal`` (no tanh squash by default); the environment clips the action to ``[-1, 1]`` and
  the buffer stores the UNCLIPPED sample, so the PPO importance ratio stays exact;
* ``log_std`` is a single ``nn.Parameter`` of shape ``(1, act_dim)`` initialised to ``log(0.5)``
  and clamped to ``[-5, 2]`` (sigma in ``[6.7e-3, 7.39]``);
* the mean head is orthogonally initialised with gain 0.01 so the initial policy is nearly
  observation-independent and centred on zero.

The log-probability, entropy and KL are written out by hand instead of going through
``torch.distributions``: it avoids per-call distribution-object construction in the hot loop, it
keeps every op traceable for the ONNX/TorchScript export path, and it lets us clamp ``log_std``
exactly once. ``tests/unit/test_distributions.py`` pins all three against ``torch.distributions``
ground truth.

No categorical head is provided: SPEC v2 S6.2 and the M4 CI gate (Pendulum-v1) specify a
continuous action space only, so a discrete head would be untested dead code.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_TWO: float = math.log(2.0)
LOG_TWO_PI: float = math.log(2.0 * math.pi)
HALF_LOG_TWO_PI_E: float = 0.5 * (LOG_TWO_PI + 1.0)

DEFAULT_SIGMA_INIT: float = 0.5
DEFAULT_LOG_STD_MIN: float = -5.0
DEFAULT_LOG_STD_MAX: float = 2.0
DEFAULT_MEAN_HEAD_GAIN: float = 0.01


def orthogonal_init_(
    layer: nn.Module,
    gain: float = math.sqrt(2.0),
    bias_const: float = 0.0,
) -> nn.Module:
    """Orthogonally initialise ``layer.weight`` and set ``layer.bias`` to a constant, in place.

    ``nn.init.orthogonal_`` on a ``Conv2d`` weight of shape ``(out, in, kh, kw)`` treats it as an
    ``(out, in * kh * kw)`` matrix, which is the standard CleanRL / rsl_rl behaviour.

    Args:
        layer: Module exposing a ``weight`` tensor and an optional ``bias`` tensor.
        gain: Multiplicative gain applied to the orthogonal matrix. SPEC v2 S6.2 uses ``sqrt(2)``
            for hidden and conv layers, ``0.01`` for the policy mean head and ``1.0`` for the
            value head.
        bias_const: Constant written into the bias.

    Returns:
        The same layer, initialised in place, so that calls can be chained.
    """
    nn.init.orthogonal_(layer.weight, gain)
    if getattr(layer, "bias", None) is not None:
        nn.init.constant_(layer.bias, bias_const)
    return layer


def tanh_log_det(x: torch.Tensor) -> torch.Tensor:
    """Per-component ``log|d tanh(x) / dx|``, numerically stable for ``|x| >> 1``.

    The naive ``log(1 - tanh(x) ** 2)`` returns ``-inf`` at ``|x| ~ 20`` even in float64. The
    identity ``log(1 - tanh^2 x) = 2 * (log 2 - x - softplus(-2x))`` is exact and finite.

    Args:
        x: Pre-squash sample, any shape.

    Returns:
        Tensor of the same shape as ``x``.
    """
    return 2.0 * (LOG_TWO - x - F.softplus(-2.0 * x))


def diag_gaussian_log_prob(
    x: torch.Tensor,
    mu: torch.Tensor,
    log_std: torch.Tensor,
) -> torch.Tensor:
    """Log-density of a diagonal Gaussian, summed over the last (action) dimension.

    Args:
        x: Samples, shape ``(..., A)``.
        mu: Means, broadcastable to ``x``.
        log_std: Log standard deviations, broadcastable to ``x``. Assumed already clamped.

    Returns:
        Log probabilities of shape ``x.shape[:-1]``.
    """
    z = (x - mu) * torch.exp(-log_std)
    return (-0.5 * z * z - log_std - 0.5 * LOG_TWO_PI).sum(dim=-1)


def diag_gaussian_entropy(log_std: torch.Tensor) -> torch.Tensor:
    """Differential entropy of a diagonal Gaussian, summed over the last dimension.

    Args:
        log_std: Log standard deviations, shape ``(..., A)``. Assumed already clamped.

    Returns:
        Entropies of shape ``log_std.shape[:-1]``.
    """
    return (log_std + HALF_LOG_TWO_PI_E).sum(dim=-1)


def diag_gaussian_kl(
    mu_p: torch.Tensor,
    log_std_p: torch.Tensor,
    mu_q: torch.Tensor,
    log_std_q: torch.Tensor,
) -> torch.Tensor:
    """Exact analytic ``KL(p || q)`` between two diagonal Gaussians, summed over the last dim.

    SPEC v2 S6.5 drives the KL-adaptive learning rate from this quantity, with ``p`` the OLD
    policy (whose ``mu`` and ``log_std`` are snapshotted in the rollout buffer) and ``q`` the
    current one.

    Args:
        mu_p: Mean of ``p``, shape ``(..., A)``.
        log_std_p: Log std of ``p``, broadcastable to ``mu_p``.
        mu_q: Mean of ``q``, broadcastable to ``mu_p``.
        log_std_q: Log std of ``q``, broadcastable to ``mu_p``.

    Returns:
        Non-negative KL values of shape ``mu_p.shape[:-1]``.
    """
    var_ratio = torch.exp(2.0 * (log_std_p - log_std_q))
    mean_term = (mu_p - mu_q).pow(2) * torch.exp(-2.0 * log_std_q)
    return ((log_std_q - log_std_p) + 0.5 * (var_ratio + mean_term - 1.0)).sum(dim=-1)


def approx_kl_k3(log_ratio: torch.Tensor) -> torch.Tensor:
    """Schulman k3 estimator of ``KL(old || new)`` from ``log_ratio = log pi_new - log pi_old``.

    ``k3 = (r - 1) - log r`` is unbiased, provably non-negative and has the lowest variance of the
    three standard estimators (k1 is unbiased but sign-indefinite, k2 carries a ~17% bias).
    ``torch.expm1`` is used for ``r - 1`` because it is accurate when ``log_ratio`` is near zero,
    which is exactly the regime of the epoch-0 ratio assert.

    Args:
        log_ratio: Elementwise log importance ratio, any shape.

    Returns:
        Non-negative per-element KL estimates of the same shape.
    """
    return torch.expm1(log_ratio) - log_ratio


class GaussianOutput(NamedTuple):
    """Bundle returned by :meth:`DiagGaussianHead.forward`.

    Attributes:
        action: Unclipped raw sample (or the pre-squash sample when ``squash`` is enabled).
        log_prob: Summed log density of ``action``, shape ``(B,)``.
        entropy: Summed differential entropy, shape ``(B,)``.
        mu: Distribution mean, shape ``(B, A)``.
        log_std: Clamped log standard deviation broadcast to ``(B, A)``.
    """

    action: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    mu: torch.Tensor
    log_std: torch.Tensor


class DiagGaussianHead(nn.Module):
    """Linear mean head plus a state-independent, clamped ``log_std`` parameter.

    Args:
        feat_dim: Width of the incoming feature vector.
        act_dim: Action dimensionality.
        sigma_init: Initial standard deviation; ``log_std`` is filled with ``log(sigma_init)``.
        log_std_min: Lower clamp on ``log_std``.
        log_std_max: Upper clamp on ``log_std``.
        squash: If True, apply a tanh squash and the exact Jacobian correction. Off by default
            (SPEC v2 S6.2 ships raw Normal plus env-side clipping); kept behind the flag as a
            documented ablation. The buffer must always store the PRE-squash sample.
        mean_head_gain: Orthogonal-init gain for the mean layer.

    Raises:
        ValueError: If ``sigma_init`` is not positive, if the clamp window is empty, or if
            ``log(sigma_init)`` falls outside the clamp window.
    """

    def __init__(
        self,
        feat_dim: int,
        act_dim: int,
        sigma_init: float = DEFAULT_SIGMA_INIT,
        log_std_min: float = DEFAULT_LOG_STD_MIN,
        log_std_max: float = DEFAULT_LOG_STD_MAX,
        squash: bool = False,
        mean_head_gain: float = DEFAULT_MEAN_HEAD_GAIN,
    ) -> None:
        super().__init__()
        if sigma_init <= 0.0:
            raise ValueError(f"sigma_init must be positive, got {sigma_init}")
        if not log_std_min < log_std_max:
            raise ValueError(f"need log_std_min < log_std_max, got ({log_std_min}, {log_std_max})")
        log_sigma_init = math.log(sigma_init)
        if not log_std_min <= log_sigma_init <= log_std_max:
            raise ValueError(f"log(sigma_init) = {log_sigma_init:.4f} is outside the clamp window")

        self.act_dim = act_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.squash = squash

        self.mean = orthogonal_init_(nn.Linear(feat_dim, act_dim), gain=mean_head_gain)
        self.log_std_param = nn.Parameter(torch.full((1, act_dim), log_sigma_init))

    def clamped_log_std(self) -> torch.Tensor:
        """Return the ``log_std`` parameter clamped to the configured window, shape ``(1, A)``."""
        return self.log_std_param.clamp(self.log_std_min, self.log_std_max)

    def forward(
        self,
        features: torch.Tensor,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> GaussianOutput:
        """Sample an action, or evaluate a stored one.

        Args:
            features: Feature batch of shape ``(B, feat_dim)``.
            action: If given, evaluate this pre-existing (unclipped, pre-squash) sample instead of
                drawing a new one. This is the PPO update path and is what makes the epoch-0
                importance ratio exactly 1.
            deterministic: If True and ``action`` is None, return the mean instead of a sample.

        Returns:
            A :class:`GaussianOutput`. When ``squash`` is enabled the returned ``action`` is still
            the pre-squash sample; call ``torch.tanh`` on it to obtain the environment action.
        """
        mu = self.mean(features)
        log_std = self.clamped_log_std().expand_as(mu)
        std = log_std.exp()

        if action is None:
            action = mu if deterministic else mu + std * torch.randn_like(mu)

        log_prob = diag_gaussian_log_prob(action, mu, log_std)
        entropy = diag_gaussian_entropy(log_std)
        if self.squash:
            correction = tanh_log_det(action).sum(dim=-1)
            log_prob = log_prob - correction
            entropy = entropy + correction
        return GaussianOutput(
            action=action,
            log_prob=log_prob,
            entropy=entropy,
            mu=mu,
            log_std=log_std,
        )

    def extra_repr(self) -> str:
        """Return a one-line description of the clamp window for ``print(model)``."""
        return (
            f"act_dim={self.act_dim}, "
            f"log_std_clamp=[{self.log_std_min}, {self.log_std_max}], squash={self.squash}"
        )
