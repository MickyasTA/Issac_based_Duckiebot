"""GAE correctness: hand-computed fixtures, truncation-vs-termination, and a naive reference.

SPEC v2 S6.7 guard 2. This is the single most important test in the repository: a policy trained
with a broken truncation bootstrap still learns something, so the bug is invisible in the reward
curve and only shows up as a critic that believes the world ends every 30 seconds.
"""

from __future__ import annotations

import pytest
import torch

from duckiebot_rl.ppo.gae import compute_gae, compute_gae_reference

GAMMA = 0.9
LAM = 0.8


def _col(values: list[float]) -> torch.Tensor:
    """Return a ``(T, 1)`` float32 column tensor.

    Args:
        values: Per-step values.

    Returns:
        Column tensor of shape ``(len(values), 1)``.
    """
    return torch.tensor(values, dtype=torch.float32).unsqueeze(1)


def test_hand_computed_no_dones() -> None:
    """A four-step rollout with no flags matches the advantages computed by hand."""
    rewards = _col([1.0, 2.0, 3.0, 4.0])
    values = _col([0.5, 1.0, 1.5, 2.0])
    zeros = torch.zeros(4, 1, dtype=torch.bool)
    last_values = torch.tensor([2.5], dtype=torch.float32)

    advantages, returns = compute_gae(
        rewards, values, zeros, zeros, last_values, torch.zeros(4, 1), gamma=GAMMA, lam=LAM
    )

    # delta3 = 4 + 0.9*2.5 - 2.0 = 4.25                    -> A3 = 4.25
    # delta2 = 3 + 0.9*2.0 - 1.5 = 3.3                     -> A2 = 3.3  + 0.72*4.25   = 6.36
    # delta1 = 2 + 0.9*1.5 - 1.0 = 2.35                    -> A1 = 2.35 + 0.72*6.36   = 6.9292
    # delta0 = 1 + 0.9*1.0 - 0.5 = 1.4                     -> A0 = 1.4  + 0.72*6.9292 = 6.389024
    expected = _col([6.389024, 6.9292, 6.36, 4.25])
    torch.testing.assert_close(advantages, expected, rtol=0, atol=1e-5)
    torch.testing.assert_close(returns, expected + values, rtol=0, atol=1e-5)


def test_truncated_bootstraps_and_terminated_does_not() -> None:
    """The defining test: truncation adds ``gamma * V(terminal)``, termination adds nothing."""
    rewards = _col([1.0, 2.0, 3.0, 4.0])
    values = _col([0.5, 1.0, 1.5, 2.0])
    terminated = torch.tensor([[False], [True], [False], [False]])
    truncated = torch.tensor([[False], [False], [True], [False]])
    term_values = _col([0.0, 0.0, 7.0, 0.0])
    last_values = torch.tensor([2.5], dtype=torch.float32)

    advantages, _ = compute_gae(
        rewards, values, terminated, truncated, last_values, term_values, gamma=GAMMA, lam=LAM
    )

    # t=3 plain          : delta = 4 + 0.9*2.5 - 2.0 = 4.25
    # t=2 TRUNCATED      : bootstraps V(terminal)=7.0, trace cut
    #                      delta = 3 + 0.9*7.0 - 1.5 = 7.8
    # t=1 TERMINATED     : NO bootstrap at all, trace cut
    #                      delta = 2 + 0        - 1.0 = 1.0
    # t=0 plain          : delta = 1 + 0.9*1.0 - 0.5 = 1.4 -> A0 = 1.4 + 0.72*1.0 = 2.12
    expected = _col([2.12, 1.0, 7.8, 4.25])
    torch.testing.assert_close(advantages, expected, rtol=0, atol=1e-5)


def test_terminated_removes_the_bootstrap_that_truncated_keeps() -> None:
    """Flipping one flag on the same data changes the advantage by exactly ``gamma * V_next``."""
    rewards = torch.zeros(1, 1)
    values = torch.zeros(1, 1)
    last_values = torch.tensor([0.0])
    term_values = _col([4.0])
    flag_on = torch.tensor([[True]])
    flag_off = torch.tensor([[False]])

    truncated_adv, _ = compute_gae(
        rewards, values, flag_off, flag_on, last_values, term_values, gamma=GAMMA, lam=LAM
    )
    terminated_adv, _ = compute_gae(
        rewards, values, flag_on, flag_off, last_values, term_values, gamma=GAMMA, lam=LAM
    )
    assert truncated_adv.item() == pytest.approx(GAMMA * 4.0, abs=1e-6)
    assert terminated_adv.item() == pytest.approx(0.0, abs=1e-6)


def test_truncation_at_last_step_uses_term_value_not_last_values() -> None:
    """A truncation at ``t = T - 1`` must ignore ``last_values`` entirely.

    This is the path that a naive implementation gets wrong, because ``t == T - 1`` and
    ``truncated`` are handled by two different branches.
    """
    rewards = torch.zeros(2, 1)
    values = torch.zeros(2, 1)
    terminated = torch.zeros(2, 1, dtype=torch.bool)
    truncated = torch.tensor([[False], [True]])
    term_values = _col([0.0, 5.0])
    last_values = torch.tensor([99.0])  # poison: must never be read at t = T - 1

    advantages, _ = compute_gae(
        rewards, values, terminated, truncated, last_values, term_values, gamma=1.0, lam=1.0
    )
    torch.testing.assert_close(advantages, _col([5.0, 5.0]), rtol=0, atol=1e-6)


def test_terminated_wins_when_both_flags_are_set() -> None:
    """If an environment reports terminated and truncated together, no bootstrap survives."""
    rewards = _col([1.0])
    values = _col([0.25])
    both = torch.tensor([[True]])
    advantages, _ = compute_gae(
        rewards, values, both, both, torch.tensor([3.0]), _col([9.0]), gamma=GAMMA, lam=LAM
    )
    assert advantages.item() == pytest.approx(1.0 - 0.25, abs=1e-6)


def test_undiscounted_full_trace_returns_the_reward_to_go() -> None:
    """With ``gamma = lam = 1`` and no flags, ``returns[t]`` is the exact reward-to-go."""
    torch.manual_seed(0)
    rewards = torch.randn(9, 4)
    values = torch.randn(9, 4)
    zeros = torch.zeros(9, 4, dtype=torch.bool)
    last_values = torch.randn(4)

    _, returns = compute_gae(
        rewards, values, zeros, zeros, last_values, torch.zeros(9, 4), gamma=1.0, lam=1.0
    )
    expected = torch.flip(torch.cumsum(torch.flip(rewards, [0]), dim=0), [0]) + last_values
    torch.testing.assert_close(returns, expected, rtol=0, atol=1e-4)


def test_matches_naive_quadratic_reference_on_random_flags() -> None:
    """The backward recursion agrees with the textbook O(T^2) forward sum."""
    torch.manual_seed(1234)
    num_steps, num_envs = 8, 3
    rewards = torch.randn(num_steps, num_envs)
    values = torch.randn(num_steps, num_envs)
    term_values = torch.randn(num_steps, num_envs)
    last_values = torch.randn(num_envs)
    terminated = torch.rand(num_steps, num_envs) < 0.12
    truncated = (torch.rand(num_steps, num_envs) < 0.12) & ~terminated
    assert bool(terminated.any()) and bool(truncated.any())

    fast = compute_gae(rewards, values, terminated, truncated, last_values, term_values, gamma=GAMMA, lam=LAM)
    slow = compute_gae_reference(
        rewards, values, terminated, truncated, last_values, term_values, gamma=GAMMA, lam=LAM
    )
    torch.testing.assert_close(fast[0], slow[0], rtol=0, atol=1e-5)
    torch.testing.assert_close(fast[1], slow[1], rtol=0, atol=1e-5)


def test_rsl_rl_approximation_bootstraps_from_the_stored_value() -> None:
    """The ablation flag reproduces the rsl_rl behaviour: bootstrap ``V(s_t)`` at a truncation."""
    rewards = torch.zeros(1, 1)
    values = _col([2.0])
    truncated = torch.tensor([[True]])
    terminated = torch.zeros(1, 1, dtype=torch.bool)
    exact, _ = compute_gae(
        rewards, values, terminated, truncated, torch.tensor([0.0]), _col([6.0]), gamma=GAMMA, lam=LAM
    )
    approx, _ = compute_gae(
        rewards,
        values,
        terminated,
        truncated,
        torch.tensor([0.0]),
        _col([6.0]),
        gamma=GAMMA,
        lam=LAM,
        rsl_rl_approx=True,
    )
    assert exact.item() == pytest.approx(GAMMA * 6.0 - 2.0, abs=1e-6)
    assert approx.item() == pytest.approx(GAMMA * 2.0 - 2.0, abs=1e-6)


def test_missing_term_values_with_truncations_raises() -> None:
    """Forgetting the terminal cache is a loud error, never a silent wrong bootstrap."""
    with pytest.raises(ValueError, match="term_values is required"):
        compute_gae(
            torch.zeros(2, 1),
            torch.zeros(2, 1),
            torch.zeros(2, 1, dtype=torch.bool),
            torch.tensor([[False], [True]]),
            torch.zeros(1),
            None,
        )


def test_shape_validation() -> None:
    """Mismatched shapes raise instead of broadcasting into a wrong answer."""
    with pytest.raises(ValueError, match="last_values must have shape"):
        compute_gae(
            torch.zeros(4, 2),
            torch.zeros(4, 2),
            torch.zeros(4, 2, dtype=torch.bool),
            torch.zeros(4, 2, dtype=torch.bool),
            torch.zeros(3),
            torch.zeros(4, 2),
        )
    with pytest.raises(ValueError, match="values must have shape"):
        compute_gae(
            torch.zeros(4, 2),
            torch.zeros(5, 2),
            torch.zeros(4, 2, dtype=torch.bool),
            torch.zeros(4, 2, dtype=torch.bool),
            torch.zeros(2),
            torch.zeros(4, 2),
        )
