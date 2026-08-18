"""Equivalence proofs and regression guards for the learner-side performance work.

Every optimisation in ``duckiebot_rl/ppo`` and ``scripts/train.py`` from the throughput campaign
is paired here with a test that pins the property that made it safe, so the cost cannot silently
come back and the semantics cannot silently drift:

* the GAE recursion was hoisted out of its per-step tensor algebra: pinned BIT-EXACT against a
  transcription of the original per-step form, on adversarial flag patterns;
* the actor and the critic now share one uint8-to-NCHW-float conversion: pinned bit-exact against
  a per-tower conversion, and pinned to actually happen (the conversion is counted);
* ``compute_returns`` evaluates the captured terminals and the last-step bootstrap in ONE critic
  pass: pinned against the two-pass form;
* the update's eleven diagnostics are pinned to a stated key order and to exactly ONE host
  synchronisation for the whole update (SPEC v2 S6.7 guard 5), which is the property a future
  "just log this one extra scalar" change is most likely to break;
* ``PPO.act`` no longer walks the module tree on every control step: pinned to leave the tree in
  the same mode it always did;
* encoder activation checkpointing is pinned to produce bit-identical gradients, which is what
  makes it a pure memory-for-time trade;
* ``EpisodeTracker`` performs no host synchronisation per control step: pinned by counting the
  synchronising calls it makes, and its records pinned against the per-step implementation it
  replaced.
"""

from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import torch

from duckiebot_rl.ppo import ActorCritic, NetworkConfig, PPOConfig, RolloutBuffer
from duckiebot_rl.ppo.gae import compute_gae
from duckiebot_rl.ppo.ppo import PPO, UPDATE_STAT_KEYS, accelerator_flags

_REPO_ROOT = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------------ helpers


def _tiny_network() -> NetworkConfig:
    """Return a small but structurally complete image network.

    Returns:
        A :class:`NetworkConfig` with the real architecture at a cheap resolution.
    """
    return NetworkConfig(
        obs_height=8,
        obs_width=16,
        obs_channels=9,
        vec_dim=4,
        priv_dim=6,
        act_dim=2,
        encoder_channels=(4, 8),
        encoder_out=16,
        hidden_dim=16,
    )


def _tiny_config(**overrides: Any) -> PPOConfig:
    """Return a PPO config sized for a fast CPU test.

    Args:
        **overrides: Fields to override on the returned config.

    Returns:
        A :class:`PPOConfig` on CPU.
    """
    fields: dict[str, Any] = {
        "num_envs": 4,
        "num_steps": 4,
        "num_minibatches": 2,
        "update_epochs": 2,
        "device": "cpu",
        "network": _tiny_network(),
    }
    fields.update(overrides)
    return PPOConfig(**fields)


def _fill_buffer(buffer: RolloutBuffer, seed: int = 0) -> None:
    """Fill every slot of a buffer with reproducible pseudo-data.

    Args:
        buffer: The buffer to fill.
        seed: Seed for the generator.
    """
    generator = torch.Generator().manual_seed(seed)
    envs = buffer.num_envs
    for step in range(buffer.num_steps):
        terminated = torch.rand(envs, generator=generator) < 0.15
        truncated = (torch.rand(envs, generator=generator) < 0.15) & ~terminated
        buffer.add(
            vec=torch.randn(envs, buffer.vec_dim, generator=generator),
            vec_priv=torch.randn(envs, buffer.priv_dim, generator=generator),
            action=torch.randn(envs, buffer.act_dim, generator=generator),
            log_prob=torch.randn(envs, generator=generator),
            value=torch.randn(envs, generator=generator),
            reward=torch.randn(envs, generator=generator),
            terminated=terminated,
            truncated=truncated,
            mu=torch.randn(envs, buffer.act_dim, generator=generator),
            log_std=torch.full((envs, buffer.act_dim), -0.7),
            image=torch.randint(0, 255, (envs, *buffer.obs_shape), dtype=torch.uint8, generator=generator)
            if buffer.obs_shape is not None
            else None,
        )
        if truncated.any():
            ids = truncated.nonzero(as_tuple=False).flatten()
            buffer.capture_terminal(
                env_ids=ids,
                vec_priv=torch.randn(ids.numel(), buffer.priv_dim, generator=generator),
                image=torch.randint(
                    0, 255, (ids.numel(), *buffer.obs_shape), dtype=torch.uint8, generator=generator
                )
                if buffer.obs_shape is not None
                else None,
                step=step,
            )


class SyncCounter:
    """Counts the calls PROJECT code makes that would force a CPU/GPU synchronisation.

    ``Tensor.item``, ``Tensor.tolist``, ``Tensor.cpu``, ``Tensor.numpy``, ``Tensor.nonzero`` and
    the ``float``/``int``/``bool`` conversions are the calls that make the host wait for the
    device. On CPU they cost nothing, which is exactly why a counting guard is needed: the
    sync-freedom of the hot loop is invisible to a CPU test unless it is counted rather than
    timed.

    Calls made from inside ``torch`` itself are deliberately NOT counted. ``torch.optim.Adam``
    reads its step counter with ``.item()`` once per parameter per step, but that counter is a
    CPU tensor whenever the optimiser is neither capturable nor fused, so on the real device it is
    not a synchronisation at all - counting it would measure torch's implementation rather than
    this project's hot path.

    Attributes:
        counts: Per-method call counts observed inside the ``with`` block.
    """

    METHODS = ("item", "tolist", "cpu", "numpy", "nonzero", "__float__", "__int__", "__bool__")

    def __init__(self) -> None:
        """Create an inactive counter."""
        self.counts: dict[str, int] = {}
        self._originals: dict[str, Any] = {}

    @staticmethod
    def _caller_is_project_code() -> bool:
        """Report whether the immediate caller lives outside the torch package.

        Returns:
            True when the calling frame belongs to project or test code.
        """
        frame = sys._getframe(2)
        module = frame.f_globals.get("__name__", "")
        return not (module == "torch" or module.startswith("torch."))

    def __enter__(self) -> SyncCounter:
        """Patch the synchronising tensor methods.

        Returns:
            This counter.
        """
        self.counts = dict.fromkeys(self.METHODS, 0)
        for name in self.METHODS:
            original = getattr(torch.Tensor, name)
            self._originals[name] = original

            def make(method_name: str, func: Any) -> Any:
                def wrapper(tensor: Any, *args: Any, **kwargs: Any) -> Any:
                    if self._caller_is_project_code():
                        self.counts[method_name] += 1
                    return func(tensor, *args, **kwargs)

                return wrapper

            setattr(torch.Tensor, name, make(name, original))
        return self

    def __exit__(self, *exc: Any) -> None:
        """Restore the original tensor methods."""
        for name, original in self._originals.items():
            setattr(torch.Tensor, name, original)
        self._originals.clear()

    @property
    def total(self) -> int:
        """Total number of synchronising calls observed in project code.

        Returns:
            The sum over every counted method.
        """
        return sum(self.counts.values())


# ------------------------------------------------------------------ GAE


def _compute_gae_per_step(
    rewards: torch.Tensor,
    values: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    last_values: torch.Tensor,
    term_values: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The pre-optimisation GAE loop, transcribed verbatim as the equivalence reference.

    Args:
        rewards: ``(T, N)`` rewards.
        values: ``(T, N)`` value estimates.
        terminated: ``(T, N)`` termination flags.
        truncated: ``(T, N)`` truncation flags.
        last_values: ``(N,)`` bootstrap for the final step.
        term_values: ``(T, N)`` captured terminal values.
        gamma: Discount factor.
        lam: Trace-decay parameter.

    Returns:
        Tuple ``(advantages, returns)``.
    """
    terminated_f = terminated.to(values.dtype)
    truncated_f = truncated.to(values.dtype)
    num_steps, num_envs = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(num_envs, dtype=values.dtype)
    done_f = torch.clamp(terminated_f + truncated_f, max=1.0)
    for t in range(num_steps - 1, -1, -1):
        next_values = last_values if t == num_steps - 1 else values[t + 1]
        next_values = torch.where(truncated_f[t] > 0.0, term_values[t], next_values)
        not_terminal = 1.0 - terminated_f[t]
        not_done = 1.0 - done_f[t]
        delta = rewards[t] + gamma * next_values * not_terminal - values[t]
        last_gae = delta + gamma * lam * not_done * last_gae
        advantages[t] = last_gae
    return advantages, advantages + values


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_gae_vectorised_algebra_is_bit_exact_against_the_per_step_form(seed: int) -> None:
    """The hoisted GAE algebra must reproduce the per-step loop bit for bit, not merely closely."""
    generator = torch.Generator().manual_seed(seed)
    steps, envs = 32, 12
    rewards = torch.randn(steps, envs, generator=generator)
    values = torch.randn(steps, envs, generator=generator)
    terminated = torch.rand(steps, envs, generator=generator) < 0.2
    truncated = (torch.rand(steps, envs, generator=generator) < 0.2) & ~terminated
    last_values = torch.randn(envs, generator=generator)
    term_values = torch.randn(steps, envs, generator=generator)

    fast_adv, fast_ret = compute_gae(
        rewards=rewards,
        values=values,
        terminated=terminated,
        truncated=truncated,
        last_values=last_values,
        term_values=term_values,
        gamma=0.99,
        lam=0.95,
    )
    slow_adv, slow_ret = _compute_gae_per_step(
        rewards, values, terminated, truncated, last_values, term_values, 0.99, 0.95
    )
    assert torch.equal(fast_adv, slow_adv)
    assert torch.equal(fast_ret, slow_ret)


def test_gae_is_bit_exact_when_every_step_is_a_done_and_when_none_is() -> None:
    """The two degenerate flag patterns, where the trace is always or never cut."""
    generator = torch.Generator().manual_seed(11)
    steps, envs = 8, 5
    rewards = torch.randn(steps, envs, generator=generator)
    values = torch.randn(steps, envs, generator=generator)
    last_values = torch.randn(envs, generator=generator)
    term_values = torch.randn(steps, envs, generator=generator)
    for terminated, truncated in (
        (torch.ones(steps, envs, dtype=torch.bool), torch.zeros(steps, envs, dtype=torch.bool)),
        (torch.zeros(steps, envs, dtype=torch.bool), torch.ones(steps, envs, dtype=torch.bool)),
        (torch.zeros(steps, envs, dtype=torch.bool), torch.zeros(steps, envs, dtype=torch.bool)),
    ):
        fast = compute_gae(
            rewards=rewards,
            values=values,
            terminated=terminated,
            truncated=truncated,
            last_values=last_values,
            term_values=term_values,
            gamma=0.99,
            lam=0.95,
        )
        slow = _compute_gae_per_step(
            rewards, values, terminated, truncated, last_values, term_values, 0.99, 0.95
        )
        assert torch.equal(fast[0], slow[0])
        assert torch.equal(fast[1], slow[1])


def test_gae_handles_a_single_step_rollout() -> None:
    """``T = 1`` exercises the ``next_values[:-1]`` empty-slice path of the hoisted form."""
    rewards = torch.tensor([[1.0, 2.0]])
    values = torch.tensor([[0.5, 0.25]])
    flags = torch.zeros(1, 2, dtype=torch.bool)
    fast = compute_gae(
        rewards=rewards,
        values=values,
        terminated=flags,
        truncated=flags,
        last_values=torch.tensor([3.0, 4.0]),
        term_values=torch.zeros(1, 2),
        gamma=0.99,
        lam=0.95,
    )
    slow = _compute_gae_per_step(
        rewards, values, flags, flags, torch.tensor([3.0, 4.0]), torch.zeros(1, 2), 0.99, 0.95
    )
    assert torch.equal(fast[0], slow[0])


# ------------------------------------------------- shared image conversion


def test_shared_image_conversion_is_bit_exact_and_actually_shared() -> None:
    """One conversion feeds both towers, and it produces exactly what two conversions produced."""
    torch.manual_seed(0)
    agent = ActorCritic(_tiny_network())
    agent.eval()
    image = torch.randint(0, 255, (6, 8, 16, 9), dtype=torch.uint8)
    vec = torch.randn(6, 4)
    priv = torch.randn(6, 6)

    calls: list[int] = []
    original = type(agent.actor.encoder).prepare

    def counting_prepare(self: Any, img: torch.Tensor) -> torch.Tensor:
        calls.append(1)
        return original(self, img)

    with torch.no_grad():
        # Reference: each tower converts for itself, as the code did before the change.
        reference_action = agent.get_action(image, vec)
        reference_value = agent.get_value(image, priv)

        type(agent.actor.encoder).prepare = counting_prepare
        try:
            out = agent.get_action_and_value(image, vec, priv, action=reference_action.action)
        finally:
            type(agent.actor.encoder).prepare = original

    assert len(calls) == 1, "the two towers must share a single uint8-to-float conversion"
    assert torch.equal(out.value, reference_value)
    assert torch.equal(out.mu, reference_action.mu)
    assert torch.equal(out.log_prob, reference_action.log_prob)


def test_prepare_matches_the_documented_conversion_exactly() -> None:
    """``prepare`` is the one place NHWC uint8 becomes NCHW float in [0, 1] (SPEC v2 S5.2)."""
    agent = ActorCritic(_tiny_network())
    image = torch.randint(0, 255, (3, 8, 16, 9), dtype=torch.uint8)
    prepared = agent.prepare_image(image)
    assert prepared is not None
    assert torch.equal(prepared, image.permute(0, 3, 1, 2).float() / 255.0)


def test_prepare_image_is_none_in_vec_only_mode() -> None:
    """Vec-only towers have no encoder, so there is nothing to prepare."""
    cfg = NetworkConfig(use_image=False, vec_dim=4, priv_dim=6, act_dim=2, hidden_dim=8)
    assert ActorCritic(cfg).prepare_image(None) is None


def test_tower_still_converts_for_itself_when_no_prepared_batch_is_supplied() -> None:
    """The single-tower entry points stay usable on their own (the export path uses them)."""
    torch.manual_seed(3)
    agent = ActorCritic(_tiny_network())
    image = torch.randint(0, 255, (2, 8, 16, 9), dtype=torch.uint8)
    vec = torch.randn(2, 4)
    with torch.no_grad():
        assert torch.equal(
            agent.actor(image, vec), agent.actor(image, vec, prepared=agent.prepare_image(image))
        )


# ------------------------------------------------------ encoder checkpointing


def test_encoder_checkpointing_produces_bit_identical_gradients() -> None:
    """Recomputing the trunk is exact, which is what makes it a pure memory-for-time trade."""
    torch.manual_seed(5)
    image = torch.randint(0, 255, (6, 8, 16, 9), dtype=torch.uint8)
    vec, priv, action = torch.randn(6, 4), torch.randn(6, 6), torch.randn(6, 2)

    def grads(checkpointed: bool) -> list[torch.Tensor]:
        torch.manual_seed(99)
        agent = ActorCritic(_tiny_network())
        agent.set_encoder_checkpointing(checkpointed)
        out = agent.get_action_and_value(image, vec, priv, action=action)
        (out.log_prob.mean() + out.value.pow(2).mean()).backward()
        return [p.grad.clone() for p in agent.parameters() if p.grad is not None]

    plain, checkpointed = grads(False), grads(True)
    assert len(plain) == len(checkpointed) > 0
    for a, b in zip(plain, checkpointed, strict=True):
        assert torch.equal(a, b)


def test_encoder_checkpointing_is_inert_under_no_grad() -> None:
    """The rollout path has no activations to trade, so the flag must not change its output."""
    torch.manual_seed(7)
    agent = ActorCritic(_tiny_network())
    agent.eval()
    image = torch.randint(0, 255, (4, 8, 16, 9), dtype=torch.uint8)
    priv = torch.randn(4, 6)
    with torch.no_grad():
        plain = agent.get_value(image, priv)
        agent.set_encoder_checkpointing(True)
        checkpointed = agent.get_value(image, priv)
    assert torch.equal(plain, checkpointed)


# -------------------------------------------------------------- compute_returns


def test_fused_terminal_and_bootstrap_pass_matches_two_separate_passes() -> None:
    """Batching the terminals with the last observation returns exactly the same values."""
    torch.manual_seed(13)
    cfg = _tiny_config()
    net = cfg.network

    def build() -> tuple[PPO, RolloutBuffer]:
        torch.manual_seed(21)
        learner = PPO(ActorCritic(net), cfg, device="cpu")
        buffer = RolloutBuffer(
            num_steps=cfg.num_steps,
            num_envs=cfg.num_envs,
            vec_dim=net.vec_dim,
            priv_dim=net.priv_dim,
            act_dim=net.act_dim,
            obs_shape=net.obs_shape,
            device="cpu",
        )
        _fill_buffer(buffer, seed=4)
        return learner, buffer

    last_image = torch.randint(0, 255, (cfg.num_envs, *net.obs_shape), dtype=torch.uint8)
    last_priv = torch.randn(cfg.num_envs, net.priv_dim)

    learner, buffer = build()
    count = learner.compute_returns(buffer, last_image, last_priv)
    fused_adv, fused_ret, fused_term = (
        buffer.advantages.clone(),
        buffer.returns.clone(),
        buffer.term_values.clone(),
    )

    # Reference: the two-pass form, evaluated exactly as the previous implementation did.
    reference, ref_buffer = build()
    ref_count = ref_buffer.compute_terminal_values(reference.predict_values)
    ref_last = reference.predict_values(last_image, last_priv)
    from duckiebot_rl.ppo.gae import compute_gae as gae

    ref_adv, ref_ret = gae(
        rewards=ref_buffer.rewards,
        values=ref_buffer.values,
        terminated=ref_buffer.terminated,
        truncated=ref_buffer.truncated,
        last_values=ref_last,
        term_values=ref_buffer.term_values,
        gamma=cfg.gamma,
        lam=cfg.gae_lambda,
        rsl_rl_approx=cfg.rsl_rl_gae_approx,
    )

    # Not bit-exact and it cannot be: evaluating K + N rows in one batch blocks the kernels
    # differently from evaluating K rows then N rows, so a handful of values move by one float32
    # ULP. The tolerance below is two orders of magnitude tighter than that, so an actual
    # regression - a wrong row, a wrong slot, a swapped slice - still fails the test loudly.
    assert count == ref_count > 0
    assert torch.allclose(fused_term, ref_buffer.term_values, rtol=1e-6, atol=1e-6)
    assert torch.allclose(fused_adv, ref_adv, rtol=1e-6, atol=1e-6)
    assert torch.allclose(fused_ret, ref_ret, rtol=1e-6, atol=1e-6)
    # The slots that received a terminal value must be exactly the same slots.
    assert torch.equal(fused_term != 0.0, ref_buffer.term_values != 0.0)


def test_compute_returns_handles_a_rollout_with_no_captured_terminals() -> None:
    """With an empty terminal cache the fused batch is just the bootstrap observation."""
    torch.manual_seed(17)
    cfg = _tiny_config()
    net = cfg.network
    learner = PPO(ActorCritic(net), cfg, device="cpu")
    buffer = RolloutBuffer(
        num_steps=cfg.num_steps,
        num_envs=cfg.num_envs,
        vec_dim=net.vec_dim,
        priv_dim=net.priv_dim,
        act_dim=net.act_dim,
        obs_shape=net.obs_shape,
        device="cpu",
    )
    envs = cfg.num_envs
    for _ in range(cfg.num_steps):
        buffer.add(
            vec=torch.randn(envs, net.vec_dim),
            vec_priv=torch.randn(envs, net.priv_dim),
            action=torch.randn(envs, net.act_dim),
            log_prob=torch.randn(envs),
            value=torch.randn(envs),
            reward=torch.randn(envs),
            terminated=torch.zeros(envs, dtype=torch.bool),
            truncated=torch.zeros(envs, dtype=torch.bool),
            mu=torch.randn(envs, net.act_dim),
            log_std=torch.zeros(envs, net.act_dim),
            image=torch.randint(0, 255, (envs, *net.obs_shape), dtype=torch.uint8),
        )
    count = learner.compute_returns(
        buffer,
        torch.randint(0, 255, (envs, *net.obs_shape), dtype=torch.uint8),
        torch.randn(envs, net.priv_dim),
    )
    assert count == 0
    assert torch.all(buffer.term_values == 0.0)
    assert torch.isfinite(buffer.advantages).all()


def test_compute_returns_rejects_a_missing_bootstrap_image() -> None:
    """Silently dropping the image would evaluate the critic on a different observation space."""
    torch.manual_seed(19)
    cfg = _tiny_config()
    net = cfg.network
    learner = PPO(ActorCritic(net), cfg, device="cpu")
    buffer = RolloutBuffer(
        num_steps=cfg.num_steps,
        num_envs=cfg.num_envs,
        vec_dim=net.vec_dim,
        priv_dim=net.priv_dim,
        act_dim=net.act_dim,
        obs_shape=net.obs_shape,
        device="cpu",
    )
    _fill_buffer(buffer, seed=2)
    with pytest.raises(ValueError, match="last_image is required"):
        learner.compute_returns(buffer, None, torch.randn(cfg.num_envs, net.priv_dim))


# ---------------------------------------------------------------- update stats


def test_update_stat_keys_are_the_sorted_reported_diagnostics() -> None:
    """The accumulator's column order is what the closing dict-zip relies on."""
    assert list(UPDATE_STAT_KEYS) == sorted(UPDATE_STAT_KEYS)
    torch.manual_seed(23)
    cfg = _tiny_config()
    net = cfg.network
    learner = PPO(ActorCritic(net), cfg, device="cpu")
    buffer = RolloutBuffer(
        num_steps=cfg.num_steps,
        num_envs=cfg.num_envs,
        vec_dim=net.vec_dim,
        priv_dim=net.priv_dim,
        act_dim=net.act_dim,
        obs_shape=net.obs_shape,
        device="cpu",
    )
    _fill_buffer(buffer, seed=6)
    learner.compute_returns(
        buffer,
        torch.randint(0, 255, (cfg.num_envs, *net.obs_shape), dtype=torch.uint8),
        torch.randn(cfg.num_envs, net.priv_dim),
    )
    cfg.ratio_assert = False
    stats = learner.update(buffer)
    for key in UPDATE_STAT_KEYS:
        assert key in stats, key
        assert isinstance(stats[key], float)


def test_update_synchronises_with_the_host_a_bounded_number_of_times() -> None:
    """SPEC v2 S6.7 guard 5: the diagnostics cost ONE synchronisation for the whole update.

    The budget below is the ratio assert (1), the per-minibatch KL controller (one per gradient
    step, which ``kl_adapt_per_minibatch`` explicitly asks for) and the single closing transfer
    of every diagnostic. Accumulating the eleven statistics per gradient step must add nothing.
    """
    torch.manual_seed(29)
    cfg = _tiny_config()
    net = cfg.network
    learner = PPO(ActorCritic(net), cfg, device="cpu")
    buffer = RolloutBuffer(
        num_steps=cfg.num_steps,
        num_envs=cfg.num_envs,
        vec_dim=net.vec_dim,
        priv_dim=net.priv_dim,
        act_dim=net.act_dim,
        obs_shape=net.obs_shape,
        device="cpu",
    )
    _fill_buffer(buffer, seed=8)
    learner.compute_returns(
        buffer,
        torch.randint(0, 255, (cfg.num_envs, *net.obs_shape), dtype=torch.uint8),
        torch.randn(cfg.num_envs, net.priv_dim),
    )
    # The buffer carries synthetic log-probs, so the epoch-0 ratio guard cannot hold here; it is
    # exercised on its own in tests/unit/test_ppo_loss.py. Its single .item() is still counted in
    # the budget below so that re-enabling it cannot silently blow the budget.
    cfg.ratio_assert = False
    grad_steps = cfg.update_epochs * cfg.num_minibatches
    with SyncCounter() as counter:
        learner.update(buffer)
    # ratio assert (item) + one KL item per gradient step + the closing tolist/cpu pair.
    assert counter.counts["item"] <= 1 + grad_steps
    assert counter.counts["tolist"] <= 1
    assert counter.counts["cpu"] <= 1


def test_update_reports_exactly_the_declared_keys_and_the_documented_tail() -> None:
    """``UPDATE_STAT_KEYS`` plus the four tail statistics is the whole reported surface.

    The closing ``dict(zip(keys, merged, strict=True))`` silently mis-pairs every diagnostic if a
    statistic is appended without being declared, so the two lists are pinned against each other.
    """
    torch.manual_seed(31)
    cfg = _tiny_config()
    cfg.ratio_assert = False
    net = cfg.network
    learner = PPO(ActorCritic(net), cfg, device="cpu")
    buffer = RolloutBuffer(
        num_steps=cfg.num_steps,
        num_envs=cfg.num_envs,
        vec_dim=net.vec_dim,
        priv_dim=net.priv_dim,
        act_dim=net.act_dim,
        obs_shape=net.obs_shape,
        device="cpu",
    )
    _fill_buffer(buffer, seed=12)
    learner.compute_returns(
        buffer,
        torch.randint(0, 255, (cfg.num_envs, *net.obs_shape), dtype=torch.uint8),
        torch.randn(cfg.num_envs, net.priv_dim),
    )
    stats = learner.update(buffer)
    tail = {
        "explained_variance",
        "advantage_mean",
        "advantage_std",
        "value_target_mean",
        "learning_rate",
    }
    assert set(stats) == set(UPDATE_STAT_KEYS) | tail
    assert learner.num_updates == 1


# ------------------------------------------------------------ agent mode switch


def test_act_leaves_the_module_tree_in_eval_mode_without_walking_it_every_call() -> None:
    """The early-out must still put every submodule in eval mode the first time."""
    torch.manual_seed(37)
    cfg = _tiny_config()
    net = cfg.network
    learner = PPO(ActorCritic(net), cfg, device="cpu")
    image = torch.randint(0, 255, (cfg.num_envs, *net.obs_shape), dtype=torch.uint8)
    vec, priv = torch.randn(cfg.num_envs, net.vec_dim), torch.randn(cfg.num_envs, net.priv_dim)

    learner.act(image, vec, priv)
    assert all(not module.training for module in learner.agent.modules())

    walks: list[bool] = []
    original = type(learner.agent).train

    def counting_train(self: Any, mode: bool = True) -> Any:
        walks.append(mode)
        return original(self, mode)

    type(learner.agent).train = counting_train
    try:
        for _ in range(5):
            learner.act(image, vec, priv)
    finally:
        type(learner.agent).train = original
    assert walks == [], "act must not re-walk the module tree once the agent is already in eval"


def test_update_puts_the_agent_back_in_train_mode() -> None:
    """The early-out must not strand the agent in eval mode across an update."""
    torch.manual_seed(41)
    cfg = _tiny_config()
    cfg.ratio_assert = False
    net = cfg.network
    learner = PPO(ActorCritic(net), cfg, device="cpu")
    buffer = RolloutBuffer(
        num_steps=cfg.num_steps,
        num_envs=cfg.num_envs,
        vec_dim=net.vec_dim,
        priv_dim=net.priv_dim,
        act_dim=net.act_dim,
        obs_shape=net.obs_shape,
        device="cpu",
    )
    _fill_buffer(buffer, seed=10)
    learner.act(
        torch.randint(0, 255, (cfg.num_envs, *net.obs_shape), dtype=torch.uint8),
        torch.randn(cfg.num_envs, net.vec_dim),
        torch.randn(cfg.num_envs, net.priv_dim),
    )
    assert not learner.agent.training
    learner.compute_returns(
        buffer,
        torch.randint(0, 255, (cfg.num_envs, *net.obs_shape), dtype=torch.uint8),
        torch.randn(cfg.num_envs, net.priv_dim),
    )
    learner.update(buffer)
    assert all(module.training for module in learner.agent.modules())


# ------------------------------------------------------------ accelerator gates


def test_accelerator_flags_are_off_on_cpu() -> None:
    """Neither channels-last convolution nor fused Adam exists on the CPU path."""
    cfg = _tiny_config()
    assert accelerator_flags(cfg, torch.device("cpu")) == (False, False)


def test_accelerator_flags_are_off_in_strict_fp32_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strict mode is the bit-reproducible reference, so both kernel changes stand down."""
    cfg = _tiny_config()
    monkeypatch.setenv("DUCKIEBOT_RL_STRICT_FP32", "1")
    assert accelerator_flags(cfg, torch.device("cuda")) == (False, False)


def test_accelerator_flags_follow_the_config_on_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Outside strict mode the config decides, so a run can turn either lever off."""
    monkeypatch.setenv("DUCKIEBOT_RL_STRICT_FP32", "0")
    assert accelerator_flags(_tiny_config(), torch.device("cuda")) == (True, True)
    assert accelerator_flags(_tiny_config(channels_last=False), torch.device("cuda")) == (False, True)
    assert accelerator_flags(_tiny_config(fused_optimizer=False), torch.device("cuda")) == (True, False)


def test_learner_records_which_accelerations_it_applied() -> None:
    """A run must be able to say what path it took; silent kernel changes are the failure mode."""
    learner = PPO(ActorCritic(_tiny_network()), _tiny_config(), device="cpu")
    assert learner.channels_last is False
    assert learner.fused_optimizer is False


def test_checkpoint_encoder_config_reaches_the_encoders() -> None:
    """The flag is plumbed, not merely declared."""
    learner = PPO(ActorCritic(_tiny_network()), _tiny_config(checkpoint_encoder=True), device="cpu")
    assert learner.agent.actor.encoder is not None
    assert learner.agent.actor.encoder.checkpoint_trunk is True
    assert learner.agent.critic.encoder is not None
    assert learner.agent.critic.encoder.checkpoint_trunk is True


# ----------------------------------------------------------------- train.py


@pytest.fixture(scope="module")
def train_module() -> Any:
    """Import ``scripts/train.py`` with a stubbed ``isaaclab.app``.

    The script imports ``AppLauncher`` at module scope, which is the documented Isaac Lab launch
    rule and not something to work around in production code; a stub lets the pure-Python helpers
    below be tested without Isaac Sim installed.

    Returns:
        The imported module.
    """
    if "isaaclab.app" not in sys.modules:

        class _StubAppLauncher:
            """Stands in for ``isaaclab.app.AppLauncher`` while importing the script."""

            @staticmethod
            def add_app_launcher_args(parser: Any) -> None:
                """Add the launcher flags the script's own code reads back.

                Args:
                    parser: The argument parser being built at module scope.
                """
                parser.add_argument("--headless", action="store_true")
                parser.add_argument("--enable_cameras", action="store_true", default=None)
                parser.add_argument("--device", default="cuda:0")

        package = types.ModuleType("isaaclab")
        package.__path__ = []  # type: ignore[attr-defined]
        app = types.ModuleType("isaaclab.app")
        app.AppLauncher = _StubAppLauncher  # type: ignore[attr-defined]
        sys.modules.setdefault("isaaclab", package)
        sys.modules["isaaclab.app"] = app
    spec = importlib.util.spec_from_file_location(
        "duckiebot_train_under_test", _REPO_ROOT / "scripts" / "train.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _ReferenceTracker:
    """The pre-optimisation :class:`EpisodeTracker`, transcribed as the equivalence reference."""

    def __init__(self, num_envs: int, step_dt: float, d_index: int, record: Any) -> None:
        """Allocate the per-env accumulators.

        Args:
            num_envs: Parallel environment count.
            step_dt: Control period in seconds.
            d_index: Index of ``d`` inside ``vec_priv``.
            record: The ``EpisodeRecord`` class to build.
        """
        self.num_envs, self.step_dt, self.d_index, self.record = num_envs, step_dt, d_index, record
        self.dropped = 0
        self._return = torch.zeros(num_envs)
        self._steps = torch.zeros(num_envs, dtype=torch.long)
        self._d_sq = torch.zeros(num_envs)
        self._d_max = torch.zeros(num_envs)
        self._partial = torch.ones(num_envs, dtype=torch.bool)

    def update(
        self,
        vec_priv: torch.Tensor,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        global_step: int,
        reason: str,
    ) -> list[Any]:
        """Fold one step in and return the episodes that ended on it.

        Args:
            vec_priv: Privileged vector observation.
            reward: Step reward.
            terminated: Termination flags.
            truncated: Truncation flags.
            global_step: Env steps consumed.
            reason: Termination reason for this step.

        Returns:
            The episodes that finished on this step.
        """
        deviation = vec_priv[:, self.d_index].detach().abs()
        self._return += reward
        self._steps += 1
        self._d_sq += deviation * deviation
        self._d_max = torch.maximum(self._d_max, deviation)
        done = terminated | truncated
        if not bool(done.any()):
            return []
        whole = done & ~self._partial
        self.dropped += int((done & self._partial).sum().item())
        self._partial &= ~done
        records: list[Any] = []
        ids = whole.nonzero(as_tuple=False).flatten()
        if ids.numel():
            lengths = self._steps[ids]
            rms = torch.sqrt(self._d_sq[ids] / lengths.to(torch.float32).clamp(min=1.0))
            records = [
                self.record(
                    score=float(score),
                    steps=int(length),
                    duration=float(length) * self.step_dt,
                    global_step=global_step,
                    timestamp=0.0,
                    lane_dev_rms=float(value),
                    lane_dev_max=float(peak),
                    success=bool(success),
                    termination_reason="truncated" if success else reason,
                )
                for score, length, value, peak, success in zip(
                    self._return[ids].tolist(),
                    lengths.tolist(),
                    rms.tolist(),
                    self._d_max[ids].tolist(),
                    truncated[ids].tolist(),
                    strict=True,
                )
            ]
        cleared = done.nonzero(as_tuple=False).flatten()
        self._return[cleared] = 0.0
        self._steps[cleared] = 0
        self._d_sq[cleared] = 0.0
        self._d_max[cleared] = 0.0
        return records


def _episode_stream(seed: int, steps: int, envs: int, priv_dim: int) -> list[tuple[torch.Tensor, ...]]:
    """Build a reproducible stream of per-step tensors with plenty of episode endings.

    Args:
        seed: Generator seed.
        steps: Number of control steps.
        envs: Parallel environment count.
        priv_dim: Width of the privileged vector.

    Returns:
        A list of ``(vec_priv, reward, terminated, truncated)`` tuples.
    """
    generator = torch.Generator().manual_seed(seed)
    stream = []
    for _ in range(steps):
        terminated = torch.rand(envs, generator=generator) < 0.18
        truncated = (torch.rand(envs, generator=generator) < 0.18) & ~terminated
        stream.append(
            (
                torch.randn(envs, priv_dim, generator=generator),
                torch.randn(envs, generator=generator),
                terminated,
                truncated,
            )
        )
    return stream


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_episode_tracker_matches_the_per_step_implementation(train_module: Any, seed: int) -> None:
    """Deferring the drain must not change a single field of a single record, nor their order."""
    envs, steps, priv_dim, d_index, step_dt = 6, 20, 8, 4, 0.05
    stream = _episode_stream(seed, steps, envs, priv_dim)

    tracker = train_module.EpisodeTracker(envs, "cpu", step_dt, d_index, horizon=steps)
    reference = _ReferenceTracker(envs, step_dt, d_index, train_module.EpisodeRecord)

    expected: list[Any] = []
    for index, (priv, reward, terminated, truncated) in enumerate(stream):
        global_step = (index + 1) * envs
        reason = f"reason_{index}"
        tracker.update(priv, reward, terminated, truncated, global_step, reason)
        expected.extend(reference.update(priv, reward, terminated, truncated, global_step, reason))
    produced = tracker.drain()

    assert len(produced) == len(expected)
    assert tracker.dropped == reference.dropped
    for got, want in zip(produced, expected, strict=True):
        assert got.score == pytest.approx(want.score)
        assert got.steps == want.steps
        assert got.duration == pytest.approx(want.duration)
        assert got.global_step == want.global_step
        assert got.lane_dev_rms == pytest.approx(want.lane_dev_rms)
        assert got.lane_dev_max == pytest.approx(want.lane_dev_max)
        assert got.success == want.success
        assert got.termination_reason == want.termination_reason


def test_episode_tracker_update_never_synchronises_with_the_host(train_module: Any) -> None:
    """The whole point of the rewrite: the per-step path must reach the host zero times."""
    envs, steps, priv_dim = 8, 12, 8
    tracker = train_module.EpisodeTracker(envs, "cpu", 0.05, 4, horizon=steps)
    stream = _episode_stream(3, steps, envs, priv_dim)
    with SyncCounter() as counter:
        for index, (priv, reward, terminated, truncated) in enumerate(stream):
            tracker.update(priv, reward, terminated, truncated, index * envs, "off_drivable")
    assert counter.total == 0, f"EpisodeTracker.update synchronised: {counter.counts}"


def test_episode_tracker_grows_its_planes_when_the_rollout_runs_long(train_module: Any) -> None:
    """The horizon is a hint, so exceeding it must keep every staged episode."""
    envs, steps, priv_dim = 4, 17, 8
    stream = _episode_stream(5, steps, envs, priv_dim)
    small = train_module.EpisodeTracker(envs, "cpu", 0.05, 4, horizon=2)
    exact = train_module.EpisodeTracker(envs, "cpu", 0.05, 4, horizon=steps)
    for index, (priv, reward, terminated, truncated) in enumerate(stream):
        for tracker in (small, exact):
            tracker.update(priv, reward, terminated, truncated, index * envs, "stall")
    grown, reference = small.drain(), exact.drain()
    assert len(grown) == len(reference) > 0
    for got, want in zip(grown, reference, strict=True):
        assert got.score == pytest.approx(want.score)
        assert got.steps == want.steps
        assert got.global_step == want.global_step


def test_episode_tracker_reset_drops_everything_in_flight(train_module: Any) -> None:
    """After an evaluation nothing staged belongs to the training stream any more."""
    envs, priv_dim = 4, 8
    tracker = train_module.EpisodeTracker(envs, "cpu", 0.05, 4, horizon=8)
    for index, (priv, reward, terminated, truncated) in enumerate(_episode_stream(9, 6, envs, priv_dim)):
        tracker.update(priv, reward, terminated, truncated, index * envs, "spin")
    tracker.reset()
    assert tracker.drain() == []


def test_episode_tracker_drain_is_empty_before_any_step(train_module: Any) -> None:
    """A drain on an untouched tracker is a no-op, not an index error."""
    assert train_module.EpisodeTracker(4, "cpu", 0.05, 4).drain() == []


def test_host_sampler_reads_are_non_blocking_and_carry_the_last_probe(train_module: Any) -> None:
    """The metrics row must read a cached value, never launch a subprocess on the train thread."""
    sampler = train_module.HostSampler(period_s=3600.0, start=False)
    assert sampler.sample() == {}

    calls: list[list[str]] = []

    def fake_run(command: list[str]) -> str:
        calls.append(command)
        if command[0] == "nvidia-smi":
            return "3421, 68"
        return "9876543"

    sampler._run = staticmethod(fake_run)  # type: ignore[method-assign]
    values = sampler.probe_once()
    assert values["vram_nvsmi_mb"] == pytest.approx(3421.0)
    assert values["gpu_temp_c"] == pytest.approx(68.0)
    assert values["free_commit_gb"] == pytest.approx(9876543 / (1024.0 * 1024.0))
    assert len(calls) == 2

    # sample() must not probe: it publishes whatever the worker last stored.
    before = len(calls)
    sampler._values = values
    assert sampler.sample() == values
    assert len(calls) == before


def test_host_sampler_survives_failing_probes(train_module: Any) -> None:
    """Monitoring must never be able to end a training run."""
    sampler = train_module.HostSampler(period_s=3600.0, start=False)
    sampler._run = staticmethod(lambda command: "")  # type: ignore[method-assign]
    assert sampler.probe_once() == {}
    sampler._run = staticmethod(lambda command: "not-a-number, nope")  # type: ignore[method-assign]
    assert "vram_nvsmi_mb" not in sampler.probe_once()


def test_host_sampler_worker_thread_is_a_daemon(train_module: Any) -> None:
    """A monitoring thread must never hold the process open at the end of a run."""
    sampler = train_module.HostSampler(period_s=3600.0, start=False)
    sampler._run = staticmethod(lambda command: "")  # type: ignore[method-assign]
    sampler.start()
    try:
        assert sampler._thread is not None
        assert sampler._thread.daemon is True
    finally:
        sampler.stop()


def test_episode_metrics_still_reports_nan_for_an_empty_iteration(train_module: Any) -> None:
    """The drained list can legitimately be empty; that must stay a NaN row, not a zero row."""
    metrics = train_module.episode_metrics([])
    assert metrics["episodes_this_iter"] == 0.0
    assert math.isnan(metrics["ep_return_mean"])
