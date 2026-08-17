"""Checkpoint save/load: bitwise-identical resume, mandatory curriculum, RNG and atomicity.

SPEC v2 S6.9 and milestone M4. The guarantee under test is the honest one: LEARNER state restores
exactly on CPU. Environment state does not and is not claimed to.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from duckiebot_rl.ppo.buffer import RolloutBuffer
from duckiebot_rl.ppo.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    collect_rng_state,
    config_hash,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from duckiebot_rl.ppo.config import NetworkConfig, PPOConfig
from duckiebot_rl.ppo.networks import ActorCritic
from duckiebot_rl.ppo.ppo import PPO

OBS_DIM = 4
ACT_DIM = 2
CURRICULUM = {
    "alpha_vis": 0.35,
    "alpha_dyn": 0.20,
    "adr_success_buffer": [1.0, 0.0, 1.0],
    "hard_example_table": {"tile_7": 0.9},
}


def _config() -> PPOConfig:
    """Return a small vec-only configuration for CPU checkpoint tests."""
    return PPOConfig(
        num_envs=4,
        num_steps=8,
        num_minibatches=2,
        update_epochs=2,
        device="cpu",
        network=NetworkConfig(
            use_image=False,
            vec_dim=OBS_DIM,
            priv_dim=OBS_DIM,
            act_dim=ACT_DIM,
            hidden_dim=32,
        ),
    )


def _learner(cfg: PPOConfig, seed: int) -> PPO:
    """Build a seeded learner.

    Args:
        cfg: Configuration.
        seed: Torch seed applied before network construction.

    Returns:
        A CPU learner.
    """
    torch.manual_seed(seed)
    return PPO(ActorCritic(cfg.network), cfg, device="cpu")


def _filled_buffer(cfg: PPOConfig, learner: PPO) -> tuple[RolloutBuffer, torch.Tensor]:
    """Collect a deterministic synthetic rollout.

    Args:
        cfg: Configuration.
        learner: Learner used to produce the actions.

    Returns:
        Tuple ``(buffer, last_obs)`` with advantages and returns already computed.
    """
    buffer = RolloutBuffer(
        num_steps=cfg.num_steps,
        num_envs=cfg.num_envs,
        vec_dim=OBS_DIM,
        priv_dim=OBS_DIM,
        act_dim=ACT_DIM,
        obs_shape=None,
        device="cpu",
    )
    torch.manual_seed(1234)
    obs = torch.randn(cfg.num_envs, OBS_DIM)
    for step in range(cfg.num_steps):
        out = learner.act(None, obs)
        truncated = torch.full((cfg.num_envs,), step == 4, dtype=torch.bool)
        if bool(truncated.any()):
            buffer.capture_terminal(
                env_ids=torch.arange(cfg.num_envs),
                vec_priv=torch.randn(cfg.num_envs, OBS_DIM),
            )
        buffer.add(
            vec=obs,
            vec_priv=obs,
            action=out["action"],
            log_prob=out["log_prob"],
            value=out["value"],
            reward=torch.randn(cfg.num_envs),
            terminated=torch.zeros(cfg.num_envs, dtype=torch.bool),
            truncated=truncated,
            mu=out["mu"],
            log_std=out["log_std"],
        )
        obs = torch.randn(cfg.num_envs, OBS_DIM)
    learner.compute_returns(buffer, None, obs)
    return buffer, obs


def test_resume_then_update_is_bitwise_identical(tmp_path) -> None:
    """The M4 acceptance test: save, update, reload, update again, compare every parameter.

    A single mismatch means some piece of learner state (optimiser moments, normaliser
    statistics, the learning rate, or an RNG stream feeding the minibatch permutation) is not
    being carried through the checkpoint.
    """
    cfg = _config()
    original = _learner(cfg, seed=7)

    # Warm the optimiser moments and the normalisers so the checkpoint carries non-trivial state,
    # then collect a FRESH on-policy rollout: reusing the warm-up rollout would legitimately trip
    # the epoch-0 ratio guard, because the policy that produced it no longer exists.
    warmup, _ = _filled_buffer(cfg, original)
    original.update(warmup)
    buffer, _ = _filled_buffer(cfg, original)

    path = tmp_path / "ckpt.pt"
    save_checkpoint(
        path,
        learner=original,
        iteration=12,
        global_step=98_304,
        curriculum_state=CURRICULUM,
        config=cfg,
        env_fingerprint={"spec": "v2"},
    )

    stats_a = original.update(buffer)
    params_a = {name: p.detach().clone() for name, p in original.agent.named_parameters()}

    restored = _learner(cfg, seed=999)  # deliberately different init
    payload = load_checkpoint(path, learner=restored)
    assert payload["iteration"] == 12
    assert payload["global_step"] == 98_304
    assert payload["curriculum"]["alpha_vis"] == pytest.approx(0.35)
    assert payload["spec_version"] == "SPEC_V2"
    assert payload["config_hash"] == config_hash(cfg)

    stats_b = restored.update(buffer)
    for name, param in restored.agent.named_parameters():
        torch.testing.assert_close(param.detach(), params_a[name], rtol=0, atol=0)
    for key, value in stats_a.items():
        assert stats_b[key] == value, f"diagnostic {key} diverged: {stats_b[key]} != {value}"


def test_checkpoint_restores_optimizer_moments_and_normalisers(tmp_path) -> None:
    """Adam moments, the learning rate and all three normalisers survive the round trip."""
    cfg = _config()
    original = _learner(cfg, seed=3)
    buffer, _ = _filled_buffer(cfg, original)
    original.update(buffer)
    original.set_learning_rate(4.2e-4)

    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, original, iteration=1, global_step=32, curriculum_state=CURRICULUM)

    restored = _learner(cfg, seed=11)
    load_checkpoint(path, learner=restored)

    assert restored.learning_rate == pytest.approx(4.2e-4)
    assert restored.optimizer.param_groups[0]["lr"] == pytest.approx(4.2e-4)
    assert restored.num_updates == original.num_updates
    for norm_name in ("vec_norm", "priv_norm", "value_norm"):
        source = getattr(original, norm_name)
        target = getattr(restored, norm_name)
        torch.testing.assert_close(target.mean, source.mean, rtol=0, atol=0)
        torch.testing.assert_close(target.var, source.var, rtol=0, atol=0)
        torch.testing.assert_close(target.count, source.count, rtol=0, atol=0)

    source_state = original.optimizer.state_dict()["state"]
    target_state = restored.optimizer.state_dict()["state"]
    assert set(source_state) == set(target_state)
    assert source_state, "the optimiser should hold Adam moments after an update"
    for key, entry in source_state.items():
        torch.testing.assert_close(target_state[key]["exp_avg"], entry["exp_avg"], rtol=0, atol=0)
        torch.testing.assert_close(target_state[key]["exp_avg_sq"], entry["exp_avg_sq"], rtol=0, atol=0)


def test_curriculum_state_is_mandatory_on_save_and_load(tmp_path) -> None:
    """Without alpha_vis and alpha_dyn a resume would silently restart DR at alpha 0."""
    cfg = _config()
    learner = _learner(cfg, seed=0)
    with pytest.raises(ValueError, match="curriculum_state is mandatory"):
        save_checkpoint(tmp_path / "a.pt", learner, 0, 0, curriculum_state=None)
    with pytest.raises(ValueError, match="missing mandatory key"):
        save_checkpoint(tmp_path / "b.pt", learner, 0, 0, curriculum_state={"alpha_vis": 0.1})

    path = tmp_path / "c.pt"
    save_checkpoint(path, learner, 0, 0, curriculum_state=CURRICULUM)
    payload = torch.load(path, weights_only=False)
    del payload["curriculum"]["alpha_dyn"]
    torch.save(payload, path)
    with pytest.raises(ValueError, match="missing mandatory key"):
        load_checkpoint(path)
    # The escape hatch exists but must be explicit.
    assert load_checkpoint(path, require_curriculum=False)["iteration"] == 0


def _draw_from_every_stream() -> tuple[torch.Tensor, np.ndarray, list[float]]:
    """Draw one sample from each generator the checkpoint snapshots.

    Returns:
        Tuple ``(torch_sample, numpy_sample, python_samples)``.
    """
    return (
        torch.rand(4),
        np.random.rand(4),  # noqa: NPY002 - the legacy global stream is what gets checkpointed
        [random.random() for _ in range(4)],
    )


def test_rng_snapshot_round_trip_reproduces_every_stream() -> None:
    """torch, numpy and the python stdlib generators all restore from the snapshot."""
    torch.manual_seed(5)
    np.random.seed(5)  # noqa: NPY002
    random.seed(5)
    state = collect_rng_state()
    expected = _draw_from_every_stream()

    torch.manual_seed(999)
    np.random.seed(999)  # noqa: NPY002
    random.seed(999)
    restore_rng_state(state)
    actual = _draw_from_every_stream()

    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2] == expected[2]


def test_rng_state_restores_even_when_the_payload_was_mapped_to_a_device() -> None:
    """``torch.load(map_location=...)`` moves the RNG tensors; they must be forced back to CPU.

    ``torch.set_rng_state`` and ``torch.cuda.set_rng_state_all`` both reject non-CPU tensors, so
    a checkpoint loaded with ``map_location="cuda"`` used to raise ``TypeError`` on resume.
    """
    torch.manual_seed(17)
    state = collect_rng_state()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    moved = dict(state)
    moved["torch"] = state["torch"].to(device)
    moved["torch_cuda"] = [s.to(device) for s in state["torch_cuda"]]
    expected = torch.rand(4)
    restore_rng_state(moved)
    torch.testing.assert_close(torch.rand(4), expected, rtol=0, atol=0)


@pytest.mark.gpu
def test_checkpoint_round_trip_on_cuda(tmp_path) -> None:
    """A GPU learner saves and resumes, including the CUDA RNG stream."""
    if not torch.cuda.is_available():  # pragma: no cover - the marker normally guards this
        pytest.skip("no CUDA device")
    cfg = _config()
    cfg.device = "cuda"
    torch.manual_seed(0)
    learner = PPO(ActorCritic(cfg.network), cfg, device="cuda")
    path = tmp_path / "cuda.pt"
    save_checkpoint(path, learner, 0, 0, curriculum_state=CURRICULUM, config=cfg)
    payload = load_checkpoint(path, learner=learner, map_location="cuda")
    assert payload["iteration"] == 0
    assert learner.vec_norm.mean.device.type == "cuda"


def test_rng_snapshot_rejects_an_incomplete_payload() -> None:
    """A truncated RNG payload raises rather than silently reseeding."""
    with pytest.raises(KeyError, match="torch"):
        restore_rng_state({"numpy": None, "python": None})


def test_save_is_atomic_and_leaves_no_temporary_file(tmp_path) -> None:
    """The tmp-then-replace dance must not litter, and must overwrite in place."""
    cfg = _config()
    learner = _learner(cfg, seed=0)
    path = tmp_path / "nested" / "ckpt.pt"
    save_checkpoint(path, learner, 0, 0, curriculum_state=CURRICULUM)
    save_checkpoint(path, learner, 1, 8192, curriculum_state=CURRICULUM)
    assert path.exists()
    assert list(tmp_path.rglob("*.tmp")) == []
    assert load_checkpoint(path)["iteration"] == 1


def test_unknown_format_version_is_refused(tmp_path) -> None:
    """A checkpoint from a future format is rejected, not silently misread."""
    cfg = _config()
    learner = _learner(cfg, seed=0)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, learner, 0, 0, curriculum_state=CURRICULUM)
    payload = torch.load(path, weights_only=False)
    payload["format_version"] = CHECKPOINT_FORMAT_VERSION + 1
    torch.save(payload, path)
    with pytest.raises(ValueError, match="unsupported checkpoint format_version"):
        load_checkpoint(path)


def test_checkpoint_exposes_a_flat_running_norm_block_for_the_exporter(tmp_path) -> None:
    """``duckiebot_rl/deploy/export_onnx.py`` reads ``running_norm['vec']`` without this package."""
    cfg = _config()
    learner = _learner(cfg, seed=0)
    buffer, _ = _filled_buffer(cfg, learner)
    learner.update(buffer)

    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, learner, 3, 96, curriculum_state=CURRICULUM, config=cfg)
    payload = torch.load(path, weights_only=False)

    assert set(payload["running_norm"]) == {"vec", "vec_priv", "value"}
    entry = payload["running_norm"]["vec"]
    assert entry["mean"].numel() == OBS_DIM
    assert entry["var"].numel() == OBS_DIM
    torch.testing.assert_close(entry["mean"], learner.vec_norm.mean.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(entry["std"], learner.vec_norm.std().cpu(), rtol=0, atol=0)
    assert payload["seed"] == cfg.seed
    # The exporter also needs the architecture, which build_actor reads back out of "config".
    assert payload["config"]["network"]["act_dim"] == ACT_DIM


def test_config_hash_is_stable_and_sensitive() -> None:
    """The hash is deterministic across calls and changes when a hyperparameter changes."""
    a = _config()
    b = _config()
    assert config_hash(a) == config_hash(b)
    c = _config()
    c.clip_coef = 0.15
    assert config_hash(c) != config_hash(a)
    assert len(config_hash(a)) == 16


def test_learner_state_dict_rejects_a_missing_field() -> None:
    """Loading a truncated learner payload fails loudly."""
    cfg = _config()
    learner = _learner(cfg, seed=0)
    state = learner.state_dict()
    del state["value_norm"]
    with pytest.raises(KeyError, match="value_norm"):
        learner.load_state_dict(state)
