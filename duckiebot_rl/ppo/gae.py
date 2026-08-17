"""Generalised Advantage Estimation with exact terminated-vs-truncated handling (SPEC v2 S6.4).

This is the module the whole project's correctness hinges on, so the semantics are spelled out.

Index convention. ``rewards[t]``, ``terminated[t]`` and ``truncated[t]`` are the reward and flags
produced by the transition OUT OF the observation stored at index ``t``. ``values[t]`` is
``V(obs[t])``.

Two flags, two jobs.

* ``terminated[t] = 1`` means the MDP genuinely ended (collision, off-drivable, rollover, stall,
  spin). There is no successor state, so the bootstrap term is ZEROED. The lambda-trace is also
  cut.
* ``truncated[t] = 1`` means the episode was cut by the 450-step time limit, which is an artifact
  of the training harness and not of the MDP. The value of the true next state MUST still be
  bootstrapped, otherwise every time-limited episode teaches the critic that the world ends at
  30 seconds. The lambda-trace is cut here too, because the next stored observation belongs to a
  different episode.

Where the bootstrap value comes from. At a truncation the observation stored at ``t + 1`` is the
POST-RESET observation of the next episode, so ``values[t + 1]`` is the wrong number. The
environment therefore captures the true terminal observation before resetting (SPEC v2 S6.4:
build the delayed stacked observation and ``vec_priv`` from pre-reset state, then call
``super()._reset_idx``), the learner evaluates the critic on that cache, and the result arrives
here as ``term_values[t]``.

The final step. For ``t = T - 1`` there is no ``values[t + 1]`` at all. The learner runs one
extra no-grad critic pass on the current (``T + 1``-th) observation that the environment has
already produced, and passes it as ``last_values``.

Both flags are used together: the discount chain ``(1 - done)`` uses ``terminated | truncated``,
while the bootstrap mask ``(1 - terminated)`` uses ``terminated`` alone. Collapsing these two into
one flag is the single most common from-scratch PPO bug.
"""

from __future__ import annotations

import torch


def _as_float(flag: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Convert a bool or numeric flag tensor to the dtype and device of ``reference``.

    Args:
        flag: Flag tensor of shape ``(T, N)``.
        reference: Tensor whose dtype and device should be matched.

    Returns:
        Float tensor of the same shape as ``flag``.
    """
    return flag.to(dtype=reference.dtype, device=reference.device)


@torch.no_grad()
def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    last_values: torch.Tensor,
    term_values: torch.Tensor | None = None,
    gamma: float = 0.99,
    lam: float = 0.95,
    rsl_rl_approx: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE advantages and value targets for one rollout.

    Args:
        rewards: ``(T, N)`` float rewards.
        values: ``(T, N)`` float ``V(obs[t])``, in the same (real) scale as ``rewards``.
        terminated: ``(T, N)`` bool or float; 1 where the MDP truly ended.
        truncated: ``(T, N)`` bool or float; 1 where the time limit fired.
        last_values: ``(N,)`` float ``V(obs[T])`` for the observation after the last stored step.
        term_values: ``(T, N)`` float ``V(true terminal obs)``, meaningful only where
            ``truncated == 1``; ignored elsewhere. Required unless ``rsl_rl_approx`` is set.
        gamma: Discount factor.
        lam: GAE trace-decay parameter.
        rsl_rl_approx: Ablation switch. When True, ``term_values`` is replaced by ``values``
            itself, which reproduces the rsl_rl / Isaac-ecosystem approximation exactly
            (bootstrap ``V(s_t)`` instead of ``V(s_terminal)`` at a truncation).

    Returns:
        Tuple ``(advantages, returns)``, both ``(T, N)``. ``returns = advantages + values`` and is
        the regression target for the critic.

    Raises:
        ValueError: If shapes are inconsistent, or if ``term_values`` is missing while truncations
            are present and ``rsl_rl_approx`` is False.
    """
    if rewards.ndim != 2:
        raise ValueError(f"rewards must be (T, N), got shape {tuple(rewards.shape)}")
    num_steps, num_envs = rewards.shape
    for name, tensor in (
        ("values", values),
        ("terminated", terminated),
        ("truncated", truncated),
    ):
        if tuple(tensor.shape) != (num_steps, num_envs):
            raise ValueError(f"{name} must have shape {(num_steps, num_envs)}, got {tuple(tensor.shape)}")
    if tuple(last_values.shape) != (num_envs,):
        raise ValueError(f"last_values must have shape {(num_envs,)}, got {tuple(last_values.shape)}")

    terminated_f = _as_float(terminated, values)
    truncated_f = _as_float(truncated, values)

    if rsl_rl_approx:
        term_values_f = values
    elif term_values is None:
        if bool(truncated_f.any()):
            raise ValueError(
                "term_values is required whenever truncations are present; pass the captured "
                "terminal-observation values, or set rsl_rl_approx=True to use the approximation"
            )
        term_values_f = values
    else:
        if tuple(term_values.shape) != (num_steps, num_envs):
            raise ValueError(
                f"term_values must have shape {(num_steps, num_envs)}, got {tuple(term_values.shape)}"
            )
        term_values_f = term_values.to(dtype=values.dtype, device=values.device)

    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(num_envs, dtype=values.dtype, device=values.device)
    done_f = torch.clamp(terminated_f + truncated_f, max=1.0)

    for t in range(num_steps - 1, -1, -1):
        next_values = last_values if t == num_steps - 1 else values[t + 1]
        # At a truncation the stored next observation belongs to the NEXT episode, so its value is
        # meaningless. Substitute the captured terminal value.
        next_values = torch.where(truncated_f[t] > 0.0, term_values_f[t], next_values)
        # Bootstrap only survives a truncation, never a true termination.
        not_terminal = 1.0 - terminated_f[t]
        # The lambda trace is cut by EITHER flag: the next stored step is a different episode.
        not_done = 1.0 - done_f[t]

        delta = rewards[t] + gamma * next_values * not_terminal - values[t]
        last_gae = delta + gamma * lam * not_done * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


@torch.no_grad()
def compute_gae_reference(
    rewards: torch.Tensor,
    values: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    last_values: torch.Tensor,
    term_values: torch.Tensor | None = None,
    gamma: float = 0.99,
    lam: float = 0.95,
    rsl_rl_approx: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Naive O(T^2) forward-sum reference implementation of :func:`compute_gae`.

    ``A_t = sum_l (gamma * lam)^l * delta_{t+l}``, truncated at the first done flag. This is the
    textbook definition written without any recursion, so a disagreement with the fast backward
    recursion localises the bug immediately. SPEC v2 S6.7 guard 2 requires the unit tests to check
    one against the other.

    Args and Returns are identical to :func:`compute_gae`.

    Returns:
        Tuple ``(advantages, returns)``, both ``(T, N)``.
    """
    num_steps, num_envs = rewards.shape
    terminated_f = _as_float(terminated, values)
    truncated_f = _as_float(truncated, values)
    if rsl_rl_approx or term_values is None:
        term_values_f = values
    else:
        term_values_f = term_values.to(dtype=values.dtype, device=values.device)

    advantages = torch.zeros_like(rewards)
    for t in range(num_steps):
        for n in range(num_envs):
            advantage = torch.zeros((), dtype=values.dtype, device=values.device)
            discount = 1.0
            for step in range(t, num_steps):
                if truncated_f[step, n] > 0.0:
                    next_value = term_values_f[step, n]
                elif step == num_steps - 1:
                    next_value = last_values[n]
                else:
                    next_value = values[step + 1, n]
                not_terminal = 1.0 - terminated_f[step, n]
                delta = rewards[step, n] + gamma * next_value * not_terminal - values[step, n]
                advantage = advantage + discount * delta
                if float(terminated_f[step, n] + truncated_f[step, n]) > 0.0:
                    break
                discount *= gamma * lam
            advantages[t, n] = advantage
    return advantages, advantages + values
