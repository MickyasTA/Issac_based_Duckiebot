"""Impoola encoder and actor/critic towers: shapes, parameter counts, init, determinism."""

from __future__ import annotations

import math

import pytest
import torch

from duckiebot_rl.ppo.config import NetworkConfig
from duckiebot_rl.ppo.networks import (
    ActorCritic,
    ActorMean,
    ConvSequence,
    ImpoolaEncoder,
    ResidualBlock,
    build_actor,
    count_parameters,
    parameter_report,
)

# Exact counts for the SPEC v2 S6.2 architecture at input 9 x 48 x 96, channels (16, 32, 32),
# encoder_out 256, hidden 256, vec_dim 8, priv_dim 14, act_dim 2. Derivation of the encoder:
#   stage 1: conv 9->16   1,312 + 2 res blocks 2 x 4,640  =  10,592
#   stage 2: conv 16->32  4,640 + 2 res blocks 2 x 18,496 =  41,632
#   stage 3: conv 32->32  9,248 + 2 res blocks 2 x 18,496 =  46,240
#   fc 32->256                                            =   8,448
#   total                                                 = 106,912
EXPECTED_COUNTS = {
    "actor_encoder": 106_912,
    "actor_mlp": 133_632,
    "policy_head": 516,
    "critic_encoder": 106_912,
    "critic_mlp": 135_168,
    "critic_head": 257,
    "actor_total": 241_060,
    "critic_total": 242_337,
    "total": 483_397,
}


def _obs(batch: int = 4) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build one batch of NHWC uint8 pixels plus both vector observations.

    Args:
        batch: Batch size.

    Returns:
        Tuple ``(image, vec, vec_priv)``.
    """
    image = torch.randint(0, 256, (batch, 48, 96, 9), dtype=torch.uint8)
    return image, torch.randn(batch, 8), torch.randn(batch, 14)


def test_encoder_pools_to_the_expected_spatial_map_and_feature_width() -> None:
    """Three stride-2 pools take 48 x 96 to 6 x 12, then global average pooling to 256 features."""
    encoder = ImpoolaEncoder(in_channels=9)
    image, _, _ = _obs(3)
    nchw = image.permute(0, 3, 1, 2).float() / 255.0
    assert tuple(encoder.stages(nchw).shape) == (3, 32, 6, 12)
    assert tuple(encoder(image).shape) == (3, 256)


def test_encoder_normalises_pixels_internally() -> None:
    """``/255`` happens inside the encoder, so uint8 and the same values as float agree."""
    torch.manual_seed(0)
    encoder = ImpoolaEncoder(in_channels=9).eval()
    image, _, _ = _obs(2)
    torch.testing.assert_close(encoder(image), encoder(image.float()), rtol=0, atol=1e-6)


def test_encoder_rejects_nchw_and_wrong_channel_counts() -> None:
    """A silently transposed observation is the classic vision-RL bug; it must raise."""
    encoder = ImpoolaEncoder(in_channels=9)
    with pytest.raises(ValueError, match="channels last"):
        encoder(torch.zeros(2, 9, 48, 96, dtype=torch.uint8))
    with pytest.raises(ValueError, match="4 dims"):
        encoder(torch.zeros(48, 96, 9, dtype=torch.uint8))


def test_residual_block_preserves_shape_and_is_a_true_skip() -> None:
    """A residual block is the identity when both convolutions are zeroed."""
    block = ResidualBlock(channels=8, gain=math.sqrt(2.0))
    with torch.no_grad():
        for conv in (block.conv0, block.conv1):
            conv.weight.zero_()
            conv.bias.zero_()
    x = torch.randn(2, 8, 6, 6)
    torch.testing.assert_close(block(x), x, rtol=0, atol=0)


def test_conv_sequence_halves_both_spatial_dimensions() -> None:
    """``MaxPool2d(3, stride=2, padding=1)`` halves an even input exactly."""
    stage = ConvSequence(4, 16, gain=math.sqrt(2.0))
    assert tuple(stage(torch.randn(2, 4, 48, 96)).shape) == (2, 16, 24, 48)


def test_exact_parameter_counts() -> None:
    """Pin the architecture: any silent width change moves one of these numbers."""
    agent = ActorCritic(NetworkConfig())
    assert parameter_report(agent) == EXPECTED_COUNTS
    assert count_parameters(agent) == EXPECTED_COUNTS["total"]


def test_towers_are_fully_separate() -> None:
    """SPEC v2 S6.2: no tensor is shared between the actor and critic towers."""
    agent = ActorCritic(NetworkConfig())
    actor_ids = {id(p) for p in agent.actor.parameters()}
    critic_ids = {id(p) for p in agent.critic.parameters()}
    assert actor_ids.isdisjoint(critic_ids)
    assert EXPECTED_COUNTS["actor_encoder"] == EXPECTED_COUNTS["critic_encoder"]
    # The asymmetric critic is wider at the fusion layer by exactly (priv_dim - vec_dim) * hidden.
    assert EXPECTED_COUNTS["critic_mlp"] - EXPECTED_COUNTS["actor_mlp"] == (14 - 8) * 256


def test_forward_shapes() -> None:
    """Both towers produce the documented shapes."""
    agent = ActorCritic(NetworkConfig())
    image, vec, vec_priv = _obs(5)
    out = agent.get_action_and_value(image, vec, vec_priv)
    assert tuple(out.action.shape) == (5, 2)
    assert tuple(out.mu.shape) == (5, 2)
    assert tuple(out.log_std.shape) == (5, 2)
    assert tuple(out.log_prob.shape) == (5,)
    assert tuple(out.entropy.shape) == (5,)
    assert tuple(out.value.shape) == (5,)
    assert tuple(agent.get_value(image, vec_priv).shape) == (5,)


def test_construction_is_deterministic_under_a_seed() -> None:
    """Two identically seeded agents have bitwise identical parameters."""
    torch.manual_seed(42)
    first = ActorCritic(NetworkConfig())
    torch.manual_seed(42)
    second = ActorCritic(NetworkConfig())
    pairs = zip(first.named_parameters(), second.named_parameters(), strict=True)
    for (name_a, pa), (name_b, pb) in pairs:
        assert name_a == name_b
        torch.testing.assert_close(pa, pb, rtol=0, atol=0)


def test_forward_is_deterministic_for_a_stored_action() -> None:
    """Evaluating a stored action twice gives bitwise identical log-probs and values."""
    torch.manual_seed(3)
    agent = ActorCritic(NetworkConfig()).eval()
    image, vec, vec_priv = _obs(4)
    action = torch.randn(4, 2)
    first = agent.get_action_and_value(image, vec, vec_priv, action=action)
    second = agent.get_action_and_value(image, vec, vec_priv, action=action)
    torch.testing.assert_close(first.log_prob, second.log_prob, rtol=0, atol=0)
    torch.testing.assert_close(first.value, second.value, rtol=0, atol=0)


def test_orthogonal_init_gains() -> None:
    """Hidden layers get gain sqrt(2), the policy mean head 0.01, the value head 1.0."""
    agent = ActorCritic(NetworkConfig())

    def gain_of(weight: torch.Tensor) -> float:
        """Recover the orthogonal gain from ``W W^T = gain^2 I`` for a wide matrix.

        Args:
            weight: A 2-D weight of shape ``(out, in)`` with ``out <= in``.

        Returns:
            The recovered gain.
        """
        gram = weight @ weight.T
        return float(torch.sqrt(torch.diagonal(gram).mean()))

    hidden = agent.actor.mlp[0].weight
    assert gain_of(hidden) == pytest.approx(math.sqrt(2.0), rel=1e-3)
    off_diagonal = (hidden @ hidden.T - 2.0 * torch.eye(hidden.shape[0])).abs().max()
    assert float(off_diagonal) < 1e-4

    assert gain_of(agent.policy_head.mean.weight) == pytest.approx(0.01, rel=1e-3)
    assert gain_of(agent.critic.head.weight) == pytest.approx(1.0, rel=1e-3)

    for name, param in agent.named_parameters():
        if name.endswith("bias"):
            assert float(param.abs().max()) == 0.0, f"{name} should be zero-initialised"


def test_small_policy_head_gain_gives_a_near_zero_initial_action() -> None:
    """The 0.01 head gain means the untrained robot starts by doing roughly nothing."""
    torch.manual_seed(0)
    agent = ActorCritic(NetworkConfig()).eval()
    image, vec, vec_priv = _obs(32)
    out = agent.get_action_and_value(image, vec, vec_priv, deterministic=True)
    assert float(out.mu.abs().max()) < 0.2


def test_vec_only_mode_drops_the_encoder() -> None:
    """Vec-only mode is a first-class path: no encoder, no image argument, same heads."""
    cfg = NetworkConfig(use_image=False, vec_dim=3, priv_dim=3, act_dim=1, hidden_dim=64)
    agent = ActorCritic(cfg)
    assert agent.actor.encoder is None and agent.critic.encoder is None
    report = parameter_report(agent)
    assert report["actor_encoder"] == 0 and report["critic_encoder"] == 0
    vec = torch.randn(6, 3)
    out = agent.get_action_and_value(None, vec)
    assert tuple(out.action.shape) == (6, 1)
    assert tuple(out.value.shape) == (6,)


def test_image_tower_rejects_a_missing_image() -> None:
    """Forgetting the pixels in image mode raises rather than silently degrading."""
    agent = ActorCritic(NetworkConfig())
    with pytest.raises(ValueError, match="use_image=True but image is None"):
        agent.get_action_and_value(None, torch.randn(2, 8), torch.randn(2, 14))


def test_asymmetric_critic_requires_explicit_priv_when_widths_differ() -> None:
    """Reusing ``vec`` for a 14-dim critic would be a silent shape bug; it raises instead."""
    agent = ActorCritic(NetworkConfig())
    image, vec, _ = _obs(2)
    with pytest.raises(ValueError, match="vec_priv is required"):
        agent.get_action_and_value(image, vec)


def test_build_actor_returns_the_deterministic_mean_of_the_trained_agent() -> None:
    """The deployment entry point is a view over the trained actor, never a re-implementation."""
    torch.manual_seed(21)
    agent = ActorCritic(NetworkConfig()).eval()
    actor = ActorMean(agent).eval()
    image, vec, vec_priv = _obs(3)
    expected = agent.get_action_and_value(image, vec, vec_priv, deterministic=True).mu
    torch.testing.assert_close(actor(image, vec), expected, rtol=0, atol=0)


def test_build_actor_rebuilds_the_architecture_from_a_checkpoint_payload() -> None:
    """``build_actor`` recovers the trained shape from the config the checkpoint records."""
    cfg = NetworkConfig(use_image=False, vec_dim=3, priv_dim=3, act_dim=1, hidden_dim=64)
    torch.manual_seed(31)
    trained = ActorCritic(cfg)
    payload = {
        "config": {"network": {**cfg.__dict__}},
        "learner": {"model": trained.state_dict()},
    }
    actor = build_actor(payload)
    assert isinstance(actor, ActorMean)
    assert not actor.training
    vec = torch.randn(4, 3)
    expected = trained.get_action(None, vec, deterministic=True).mu
    torch.testing.assert_close(actor(None, vec), expected, rtol=0, atol=0)


def test_build_actor_without_a_checkpoint_uses_the_default_architecture() -> None:
    """Called with no arguments it yields an untrained actor of the production shape."""
    actor = build_actor()
    image, vec, _ = _obs(2)
    assert tuple(actor(image, vec).shape) == (2, 2)


def test_build_actor_rejects_a_payload_without_weights() -> None:
    """A checkpoint with no model state dict fails loudly rather than exporting noise."""
    with pytest.raises(KeyError, match="no model state dict"):
        build_actor({"config": {}})


def test_network_config_validation() -> None:
    """Bad shapes are rejected at config time."""
    with pytest.raises(ValueError, match="divisible by 8"):
        NetworkConfig(obs_height=47)
    with pytest.raises(ValueError, match="priv_dim"):
        NetworkConfig(vec_dim=14, priv_dim=8)
