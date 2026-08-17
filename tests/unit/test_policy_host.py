"""PolicyHost: hot-reload changes the action, and an architecture mismatch is refused cleanly.

CPU only. The image-mode test deliberately uses a 16x16 observation and a one-stage encoder so the
full convolutional path is exercised in milliseconds rather than the production 48x96x9 stack.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch

from duckiebot_rl.ppo.config import NetworkConfig
from duckiebot_rl.ppo.networks import ActorCritic
from duckiebot_rl.viz.policy_host import (
    ArchitectureMismatch,
    HostState,
    PolicyHost,
    network_config_from_checkpoint,
)
from duckiebot_rl.viz.run_dir import RunDir
from duckiebot_rl.viz.watcher import CheckpointWatcher

VEC_CFG = NetworkConfig(use_image=False, vec_dim=4, priv_dim=4, act_dim=2, hidden_dim=16)
IMG_CFG = NetworkConfig(
    obs_height=16,
    obs_width=16,
    obs_channels=9,
    vec_dim=4,
    priv_dim=6,
    act_dim=2,
    encoder_channels=(4,),
    encoder_out=8,
    hidden_dim=8,
)


def make_payload(
    cfg: NetworkConfig,
    seed: int,
    iteration: int = 1,
    vec_mean: np.ndarray | None = None,
    vec_std: np.ndarray | None = None,
    include_config: bool = True,
) -> dict:
    """Build a checkpoint payload shaped like the one save_checkpoint writes."""
    torch.manual_seed(seed)
    agent = ActorCritic(cfg)
    mean = np.zeros(cfg.vec_dim, dtype=np.float32) if vec_mean is None else np.asarray(vec_mean)
    std = np.ones(cfg.vec_dim, dtype=np.float32) if vec_std is None else np.asarray(vec_std)
    payload = {
        "format_version": 1,
        "iteration": iteration,
        "global_step": iteration * 100,
        "learner": {
            "model": agent.state_dict(),
            "vec_norm": {
                "mean": torch.as_tensor(mean, dtype=torch.float32),
                "var": torch.as_tensor(std, dtype=torch.float32) ** 2,
                "count": 1000.0,
            },
        },
        "running_norm": {
            "vec": {
                "mean": torch.as_tensor(mean, dtype=torch.float32),
                "std": torch.as_tensor(std, dtype=torch.float32),
                "count": 1000.0,
            }
        },
        "curriculum": {"alpha_vis": 0.5, "alpha_dyn": 0.25},
    }
    if include_config:
        payload["config"] = {"network": dataclasses.asdict(cfg)}
    return payload


def vec_obs(cfg: NetworkConfig, value: float = 0.3) -> dict[str, np.ndarray]:
    """Return a deterministic vector-only observation."""
    return {"vec": np.full(cfg.vec_dim, value, dtype=np.float32)}


def img_obs(cfg: NetworkConfig) -> dict[str, np.ndarray]:
    """Return a deterministic image observation."""
    rng = np.random.default_rng(0)
    return {
        "vec": np.full(cfg.vec_dim, 0.3, dtype=np.float32),
        "rgb": rng.integers(0, 256, (cfg.obs_height, cfg.obs_width, cfg.obs_channels), dtype=np.uint8),
    }


# --------------------------------------------------------------------------------- hot reloading


def test_hot_reload_changes_the_action_without_rebuilding_the_network():
    host = PolicyHost(network_cfg=VEC_CFG)
    obs = vec_obs(VEC_CFG)

    host.load_checkpoint(make_payload(VEC_CFG, seed=1, iteration=10))
    network_identity = id(host.agent)
    first = host.act(obs)

    host.load_checkpoint(make_payload(VEC_CFG, seed=2, iteration=20))

    assert id(host.agent) == network_identity, "the network must be reused, not reconstructed"
    second = host.act(obs)
    assert not np.allclose(first, second), "different weights must produce a different action"
    assert host.reload_count == 2
    assert host.state is not None
    assert host.state.iteration == 20


def test_reloading_the_same_weights_reproduces_the_action_exactly():
    host = PolicyHost(network_cfg=VEC_CFG)
    obs = vec_obs(VEC_CFG)
    host.load_checkpoint(make_payload(VEC_CFG, seed=7))
    first = host.act(obs)
    host.load_checkpoint(make_payload(VEC_CFG, seed=1))
    host.load_checkpoint(make_payload(VEC_CFG, seed=7))
    np.testing.assert_allclose(host.act(obs), first, rtol=0, atol=0)


def test_from_checkpoint_infers_the_architecture(tmp_path: Path):
    path = tmp_path / "ckpt.pt"
    torch.save(make_payload(IMG_CFG, seed=3, iteration=42), path)

    host = PolicyHost.from_checkpoint(path)

    assert host.network_cfg.obs_height == IMG_CFG.obs_height
    assert host.network_cfg.encoder_out == IMG_CFG.encoder_out
    assert host.state is not None
    assert host.state.iteration == 42
    assert host.state.alpha_vis == pytest.approx(0.5)
    assert host.state.alpha_dyn == pytest.approx(0.25)
    action = host.act(img_obs(IMG_CFG))
    assert action.shape == (IMG_CFG.act_dim,)
    assert action.dtype == np.float32


def test_load_from_info_uses_the_watcher_metadata(tmp_path: Path):
    run = RunDir.open(tmp_path / "run", create=True)
    path = run.latest_checkpoint
    torch.save(make_payload(VEC_CFG, seed=5, iteration=11), path)
    run.record_checkpoint(path, iteration=11, metric_name="ep_return_mean", metric_value=3.5)

    info = CheckpointWatcher(run.root).poll()
    assert info is not None

    host = PolicyHost(network_cfg=VEC_CFG)
    state = host.load_from_info(info)

    assert isinstance(state, HostState)
    assert state.iteration == 11
    assert state.metric_name == "ep_return_mean"
    assert state.metric_value == pytest.approx(3.5)
    assert state.sha256 == info.sha256
    assert "ep_return_mean=3.5" in state.describe()


# ---------------------------------------------------------------------------- architecture guard


def test_architecture_mismatch_is_refused_with_a_readable_message():
    host = PolicyHost(network_cfg=VEC_CFG)
    host.load_checkpoint(make_payload(VEC_CFG, seed=1))
    obs = vec_obs(VEC_CFG)
    before = host.act(obs)

    wider = dataclasses.replace(VEC_CFG, hidden_dim=32)
    with pytest.raises(ArchitectureMismatch) as excinfo:
        host.load_checkpoint(make_payload(wider, seed=1))

    message = str(excinfo.value)
    assert "hidden_dim" in message
    assert "live=16" in message
    assert "checkpoint=32" in message
    np.testing.assert_allclose(host.act(obs), before, rtol=0, atol=0)
    assert host.reload_count == 1


def test_vec_width_mismatch_is_refused():
    host = PolicyHost(network_cfg=VEC_CFG)
    host.load_checkpoint(make_payload(VEC_CFG, seed=1))
    other = NetworkConfig(use_image=False, vec_dim=6, priv_dim=6, act_dim=2, hidden_dim=16)

    with pytest.raises(ArchitectureMismatch, match="vec_dim"):
        host.load_checkpoint(make_payload(other, seed=1))


def test_image_mode_mismatch_is_refused():
    host = PolicyHost(network_cfg=VEC_CFG)
    host.load_checkpoint(make_payload(VEC_CFG, seed=1))

    with pytest.raises(ArchitectureMismatch, match="use_image"):
        host.load_checkpoint(make_payload(IMG_CFG, seed=1))


def test_state_dict_mismatch_is_caught_without_a_config_block():
    host = PolicyHost(network_cfg=VEC_CFG)
    host.load_checkpoint(make_payload(VEC_CFG, seed=1))
    wider = dataclasses.replace(VEC_CFG, hidden_dim=32)

    with pytest.raises(ArchitectureMismatch) as excinfo:
        host.load_checkpoint(make_payload(wider, seed=1, include_config=False))

    assert "state dict does not fit" in str(excinfo.value)


def test_a_checkpoint_without_weights_is_refused():
    host = PolicyHost(network_cfg=VEC_CFG)
    with pytest.raises(ArchitectureMismatch, match="no actor-critic state dict"):
        host.load_checkpoint({"iteration": 1})


def test_unconfigured_host_needs_an_architecture_somewhere():
    host = PolicyHost()
    with pytest.raises(ArchitectureMismatch, match=r"config\.network"):
        host.load_checkpoint(make_payload(VEC_CFG, seed=1, include_config=False))


def test_network_config_from_checkpoint_returns_none_without_a_config():
    assert network_config_from_checkpoint({"learner": {}}) is None


# ------------------------------------------------------------------------------- act() behaviour


def test_act_is_deterministic_by_default_and_equals_the_gaussian_mean():
    host = PolicyHost(network_cfg=VEC_CFG)
    host.load_checkpoint(make_payload(VEC_CFG, seed=4))
    obs = vec_obs(VEC_CFG)

    first, second = host.act(obs), host.act(obs)
    np.testing.assert_allclose(first, second, rtol=0, atol=0)

    with torch.no_grad():
        vec = torch.as_tensor(obs["vec"], dtype=torch.float32).unsqueeze(0)
        expected = host.agent.get_action(None, vec, deterministic=True).mu.squeeze(0).numpy()
    np.testing.assert_allclose(first, expected, rtol=1e-6, atol=1e-6)


def test_stochastic_mode_samples_around_the_mean():
    host = PolicyHost(network_cfg=VEC_CFG, stochastic=True, seed=0)
    host.load_checkpoint(make_payload(VEC_CFG, seed=4))
    obs = vec_obs(VEC_CFG)

    samples = np.stack([host.act(obs) for _ in range(8)])
    assert samples.std(axis=0).max() > 0.0, "stochastic mode must not return a constant"

    mean_action = host.act(obs, stochastic=False)
    np.testing.assert_allclose(host.act(obs, stochastic=False), mean_action, rtol=0, atol=0)


def test_vector_normalisation_from_the_checkpoint_is_applied():
    obs = vec_obs(VEC_CFG, value=2.0)

    plain = PolicyHost(network_cfg=VEC_CFG)
    plain.load_checkpoint(make_payload(VEC_CFG, seed=6))

    shifted = PolicyHost(network_cfg=VEC_CFG)
    shifted.load_checkpoint(
        make_payload(
            VEC_CFG,
            seed=6,
            vec_mean=np.full(VEC_CFG.vec_dim, 5.0, dtype=np.float32),
            vec_std=np.full(VEC_CFG.vec_dim, 2.0, dtype=np.float32),
        )
    )

    assert plain.state is not None and plain.state.normalized_vec
    assert shifted.state is not None and shifted.state.normalized_vec
    assert not np.allclose(plain.act(obs), shifted.act(obs))


def test_normalisation_can_be_disabled_and_is_reported():
    host = PolicyHost(network_cfg=VEC_CFG, normalize_vec=False)
    host.load_checkpoint(make_payload(VEC_CFG, seed=6))
    assert host.state is not None
    assert host.state.normalized_vec is False
    assert "vec-RAW" in host.state.describe()


def test_act_before_loading_raises():
    host = PolicyHost(network_cfg=VEC_CFG)
    with pytest.raises(RuntimeError, match="before any checkpoint"):
        host.act(vec_obs(VEC_CFG))


def test_wrong_observation_width_is_reported_clearly():
    host = PolicyHost(network_cfg=VEC_CFG)
    host.load_checkpoint(make_payload(VEC_CFG, seed=1))
    with pytest.raises(ValueError, match="network expects 4"):
        host.act({"vec": np.zeros(9, dtype=np.float32)})


def test_missing_image_is_reported_clearly():
    host = PolicyHost(network_cfg=IMG_CFG)
    host.load_checkpoint(make_payload(IMG_CFG, seed=1))
    with pytest.raises(KeyError, match="rgb"):
        host.act({"vec": np.zeros(IMG_CFG.vec_dim, dtype=np.float32)})


def test_describe_before_loading():
    assert PolicyHost(network_cfg=VEC_CFG).describe() == "no checkpoint loaded"
