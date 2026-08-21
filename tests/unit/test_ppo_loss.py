"""PPO update: the ratio guard, the clipped surrogate, the KL controller and the diagnostics.

The centrepiece is the SPEC v2 S6.7 guard 1 test and its M4 acceptance criterion: the guard must
FIRE on a deliberately introduced clipped-action-storage bug. A guard that has never been seen to
fail is not a guard.
"""

from __future__ import annotations

import math

import pytest
import torch

from duckiebot_rl.ppo.buffer import RolloutBuffer
from duckiebot_rl.ppo.config import NetworkConfig, PPOConfig
from duckiebot_rl.ppo.networks import ActorCritic
from duckiebot_rl.ppo.ppo import PPO, RatioAssertionError

OBS_DIM = 5
ACT_DIM = 2
NUM_ENVS = 4
NUM_STEPS = 8


def _config(**overrides: object) -> PPOConfig:
    """Build a small vec-only PPO configuration for fast CPU tests.

    Args:
        **overrides: Fields to override on :class:`PPOConfig`.

    Returns:
        A validated configuration.
    """
    network = NetworkConfig(
        use_image=False,
        vec_dim=OBS_DIM,
        priv_dim=OBS_DIM,
        act_dim=ACT_DIM,
        hidden_dim=32,
        sigma_init=float(overrides.pop("sigma_init", 0.5)),  # type: ignore[arg-type]
    )
    base = {
        "num_envs": NUM_ENVS,
        "num_steps": NUM_STEPS,
        "num_minibatches": 2,
        "update_epochs": 2,
        "device": "cpu",
        "network": network,
    }
    base.update(overrides)
    return PPOConfig(**base)  # type: ignore[arg-type]


def _learner(cfg: PPOConfig, seed: int = 0) -> PPO:
    """Build a seeded learner for a configuration.

    Args:
        cfg: The configuration.
        seed: Torch seed applied before constructing the network.

    Returns:
        A :class:`PPO` learner on CPU.
    """
    torch.manual_seed(seed)
    return PPO(ActorCritic(cfg.network), cfg, device="cpu")


def _buffer(cfg: PPOConfig) -> RolloutBuffer:
    """Allocate a matching vec-only rollout buffer.

    Args:
        cfg: The configuration.

    Returns:
        A fresh :class:`RolloutBuffer`.
    """
    return RolloutBuffer(
        num_steps=cfg.num_steps,
        num_envs=cfg.num_envs,
        vec_dim=OBS_DIM,
        priv_dim=OBS_DIM,
        act_dim=ACT_DIM,
        obs_shape=None,
        device="cpu",
        terminal_capacity=16,
    )


def _fill(
    learner: PPO,
    buffer: RolloutBuffer,
    store_clipped_action: bool = False,
    truncate_at: int | None = None,
) -> torch.Tensor:
    """Collect a synthetic rollout through the real ``act`` path.

    Args:
        learner: The learner producing the actions.
        buffer: Destination buffer.
        store_clipped_action: If True, store the CLIPPED action, which is the deliberate bug the
            epoch-0 ratio guard exists to catch.
        truncate_at: Optional step index at which every environment reports a truncation.

    Returns:
        The ``T + 1``-th observation, for the last-step bootstrap.
    """
    buffer.reset()
    obs = torch.randn(buffer.num_envs, OBS_DIM)
    for step in range(buffer.num_steps):
        out = learner.act(None, obs)
        truncated = torch.full((buffer.num_envs,), step == truncate_at, dtype=torch.bool)
        terminated = torch.zeros(buffer.num_envs, dtype=torch.bool)
        if bool(truncated.any()):
            buffer.capture_terminal(
                env_ids=torch.arange(buffer.num_envs),
                vec_priv=torch.randn(buffer.num_envs, OBS_DIM),
            )
        buffer.add(
            vec=obs,
            vec_priv=obs,
            action=out["clipped_action"] if store_clipped_action else out["action"],
            log_prob=out["log_prob"],
            value=out["value"],
            reward=torch.randn(buffer.num_envs),
            terminated=terminated,
            truncated=truncated,
            mu=out["mu"],
            log_std=out["log_std"],
        )
        obs = torch.randn(buffer.num_envs, OBS_DIM)
    return obs


def test_update_runs_and_reports_every_documented_diagnostic() -> None:
    """One update returns the full SPEC v2 S6.8 diagnostic set with sane values."""
    cfg = _config()
    learner = _learner(cfg)
    buffer = _buffer(cfg)
    last_obs = _fill(learner, buffer, truncate_at=3)
    num_terminals = learner.compute_returns(buffer, None, last_obs)
    assert num_terminals == NUM_ENVS

    stats = learner.update(buffer)
    expected = {
        "policy_loss",
        "value_loss",
        "entropy",
        "bounds_loss",
        "approx_kl",
        "analytic_kl",
        "clipfrac",
        "ratio_mean",
        "grad_norm",
        "mean_sigma",
        "mean_abs_mu",
        "explained_variance",
        "learning_rate",
        "advantage_mean",
        "advantage_std",
        "value_target_mean",
    }
    assert expected <= set(stats)
    assert all(isinstance(value, float) for value in stats.values())
    assert stats["approx_kl"] >= 0.0, "the k3 estimator is provably non-negative"
    assert 0.0 <= stats["clipfrac"] <= 1.0
    assert stats["mean_sigma"] == pytest.approx(0.5, rel=0.2)
    assert learner.num_updates == 1


def test_ratio_guard_fires_on_a_clipped_action_storage_bug() -> None:
    """M4 acceptance: the guard must be seen to fail on the classic bug.

    With ``sigma_init = 1.5`` a large fraction of raw samples fall outside ``[-1, 1]``, so storing
    the clipped action makes the stored log-prob disagree with the recomputed one and the epoch-0
    importance ratio moves away from 1.
    """
    cfg = _config(sigma_init=1.5)
    learner = _learner(cfg, seed=1)
    buffer = _buffer(cfg)
    last_obs = _fill(learner, buffer, store_clipped_action=True)
    learner.compute_returns(buffer, None, last_obs)
    with pytest.raises(RatioAssertionError, match="importance ratio deviates"):
        learner.update(buffer)


def test_ratio_guard_passes_when_the_unclipped_action_is_stored() -> None:
    """The same configuration with the correct action storage passes the guard."""
    cfg = _config(sigma_init=1.5)
    learner = _learner(cfg, seed=1)
    buffer = _buffer(cfg)
    last_obs = _fill(learner, buffer, store_clipped_action=False)
    learner.compute_returns(buffer, None, last_obs)
    stats = learner.update(buffer)
    assert stats["ratio_mean"] == pytest.approx(1.0, abs=0.2)


def test_ratio_guard_still_holds_on_the_second_update() -> None:
    """The vector normalisers must not move between acting and updating.

    If they were refreshed mid-update, the observations fed to the network during update ``k + 1``
    would differ from the ones the policy acted on, and this second guard would fire.
    """
    cfg = _config()
    learner = _learner(cfg, seed=2)
    buffer = _buffer(cfg)
    for _ in range(3):
        last_obs = _fill(learner, buffer)
        learner.compute_returns(buffer, None, last_obs)
        learner.update(buffer)
    assert learner.vec_norm.count.item() > 1.0, "the normaliser should have been updated by now"


def test_disabling_the_guard_lets_the_bug_through() -> None:
    """``ratio_assert=False`` is a real switch, so the guard is proven to be load-bearing."""
    cfg = _config(sigma_init=1.5, ratio_assert=False)
    learner = _learner(cfg, seed=1)
    buffer = _buffer(cfg)
    last_obs = _fill(learner, buffer, store_clipped_action=True)
    learner.compute_returns(buffer, None, last_obs)
    stats = learner.update(buffer)
    assert stats["ratio_mean"] != pytest.approx(1.0, abs=1e-4)


def test_advantages_are_normalised_once_at_batch_level() -> None:
    """Batch-level normalisation is reflected in the reported pre-normalisation moments."""
    cfg = _config()
    learner = _learner(cfg)
    buffer = _buffer(cfg)
    last_obs = _fill(learner, buffer)
    learner.compute_returns(buffer, None, last_obs)
    raw = buffer.advantages.reshape(-1)
    stats = learner.update(buffer)
    assert stats["advantage_mean"] == pytest.approx(float(raw.mean()), abs=1e-5)
    assert stats["advantage_std"] == pytest.approx(float(raw.std()), abs=1e-5)


def test_kl_controller_lowers_the_learning_rate_above_the_upper_target() -> None:
    """A KL above 0.02 divides the learning rate by 1.5, clamped at 1e-5."""
    cfg = _config()
    learner = _learner(cfg)
    learner.set_learning_rate(1e-3)
    learner._adapt_learning_rate(0.05)
    assert learner.learning_rate == pytest.approx(1e-3 / 1.5)
    for _ in range(50):
        learner._adapt_learning_rate(0.05)
    assert learner.learning_rate == pytest.approx(cfg.lr_min)


def test_kl_controller_raises_the_learning_rate_below_the_lower_target() -> None:
    """A KL below 0.005 multiplies the learning rate by 1.5, clamped at lr_max = 1e-3.

    The ceiling is part of the property. The controller raises lr while KL is LOW, and a dying
    vision encoder keeps KL low (the policy barely changes when it cannot see), so a generous
    ceiling turns encoder trouble into a self-reinforcing death spiral: lr climbs, Adam's
    per-parameter step grows with lr no matter what the clipped gradient norm is, the encoder
    dies harder, KL falls further. Measured killing the actor's conv trunk twice (iteration
    ~600 of one run, iteration 12 of a fresh one) before the cap was lowered from 1e-2.
    """
    cfg = _config()
    learner = _learner(cfg)
    learner.set_learning_rate(3e-4)
    learner._adapt_learning_rate(0.001)
    assert learner.learning_rate == pytest.approx(3e-4 * 1.5)
    for _ in range(50):
        learner._adapt_learning_rate(0.001)
    assert learner.learning_rate == pytest.approx(cfg.lr_max)
    assert cfg.lr_max == pytest.approx(1e-3)


def test_kl_controller_holds_inside_the_dead_band_and_ignores_nan() -> None:
    """Between the two targets nothing changes, and a non-finite KL is ignored."""
    cfg = _config()
    learner = _learner(cfg)
    learner.set_learning_rate(3e-4)
    learner._adapt_learning_rate(0.01)
    assert learner.learning_rate == pytest.approx(3e-4)
    learner._adapt_learning_rate(float("nan"))
    assert learner.learning_rate == pytest.approx(3e-4)


def test_optimizer_learning_rate_tracks_the_controller() -> None:
    """The controller writes through to the Adam parameter group, not just to a field."""
    cfg = _config()
    learner = _learner(cfg)
    learner.set_learning_rate(7e-4)
    assert learner.optimizer.param_groups[0]["lr"] == pytest.approx(7e-4)


def test_gradient_norm_is_clipped_and_the_pre_clip_norm_is_reported() -> None:
    """SPEC v2 S6.5 logs the PRE-clip norm; large rewards must show a large reported norm."""
    cfg = _config(max_grad_norm=1e-6, learning_rate=1e-5)
    learner = _learner(cfg)
    buffer = _buffer(cfg)
    last_obs = _fill(learner, buffer)
    buffer.rewards.mul_(1000.0)
    learner.compute_returns(buffer, None, last_obs)
    before = [p.detach().clone() for p in learner.agent.parameters()]
    stats = learner.update(buffer)
    assert stats["grad_norm"] > 1e-6, "the reported norm must be the pre-clip value"
    moved = max(float((a - b).abs().max()) for a, b in zip(before, learner.agent.parameters(), strict=True))
    assert moved < 1e-3, "with a 1e-6 clip and a 1e-5 lr the parameters should barely move"


def test_explained_variance_is_high_for_a_perfect_critic() -> None:
    """Forcing values equal to returns drives explained variance to 1."""
    cfg = _config()
    learner = _learner(cfg)
    buffer = _buffer(cfg)
    last_obs = _fill(learner, buffer)
    learner.compute_returns(buffer, None, last_obs)
    buffer.values.copy_(buffer.returns)
    stats = learner.update(buffer)
    assert stats["explained_variance"] == pytest.approx(1.0, abs=1e-4)


def test_value_clipping_ablation_flag_runs() -> None:
    """``clip_vloss=True`` is a supported ablation path, not dead code."""
    cfg = _config(clip_vloss=True)
    learner = _learner(cfg)
    buffer = _buffer(cfg)
    last_obs = _fill(learner, buffer)
    learner.compute_returns(buffer, None, last_obs)
    stats = learner.update(buffer)
    assert math.isfinite(stats["value_loss"])


def test_deterministic_act_returns_the_mean_and_clips_for_the_environment() -> None:
    """The evaluation path uses ``a = mu`` and hands the env a clipped copy."""
    cfg = _config(sigma_init=1.5)
    learner = _learner(cfg)
    obs = torch.randn(16, OBS_DIM) * 5.0
    out = learner.act(None, obs, deterministic=True)
    torch.testing.assert_close(out["action"], out["mu"], rtol=0, atol=0)
    torch.testing.assert_close(out["clipped_action"], out["action"].clamp(-1.0, 1.0), rtol=0, atol=0)


def test_config_rejects_an_indivisible_minibatch_split() -> None:
    """SPEC v2 S6.6 asserts ``batch % num_minibatches == 0`` at config time."""
    with pytest.raises(ValueError, match="divisible by num_minibatches"):
        _config(num_minibatches=7)


def test_default_config_matches_the_spec_table() -> None:
    """Pin the SPEC v2 S6.6 hyperparameter table so a silent edit fails CI."""
    cfg = PPOConfig()
    assert (cfg.num_envs, cfg.num_steps, cfg.batch_size) == (256, 32, 8192)
    assert (cfg.num_minibatches, cfg.minibatch_size) == (16, 512)
    assert cfg.update_epochs == 4
    assert cfg.gradient_steps_per_update == 64
    assert (cfg.gamma, cfg.gae_lambda) == (0.99, 0.95)
    assert cfg.clip_coef == 0.2
    assert cfg.clip_vloss is False
    assert cfg.ent_coef == 0.0
    assert cfg.vf_coef == 1.0
    assert cfg.bounds_coef == 1e-4
    assert cfg.max_grad_norm == 1.0
    assert (cfg.learning_rate, cfg.adam_eps) == (3e-4, 1e-5)
    assert (cfg.kl_target_lower, cfg.kl_target_upper, cfg.lr_factor) == (0.005, 0.02, 1.5)
    assert (cfg.lr_min, cfg.lr_max) == (1e-5, 1e-3)  # lr_max lowered 2026-08-21, see PPOConfig
    assert cfg.network.sigma_init == 0.5
    assert (cfg.network.log_std_min, cfg.network.log_std_max) == (-5.0, 2.0)
    assert cfg.network.obs_shape == (48, 96, 9)
    assert (cfg.network.vec_dim, cfg.network.priv_dim, cfg.network.act_dim) == (8, 14, 2)
