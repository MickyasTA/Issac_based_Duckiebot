"""Impoola-style encoder and the separate actor / critic towers (SPEC v2 S6.2).

Architecture, verbatim from the spec:

* Encoder: three ConvSequences with channels ``(16, 32, 32)``. Each is
  ``Conv3x3(pad 1) -> MaxPool3x3/stride2/pad1 -> ResBlock -> ResBlock`` where a ResBlock is
  ``ReLU -> Conv3x3 -> ReLU -> Conv3x3 -> +skip``. Then ``ReLU``, GLOBAL AVERAGE POOL over the
  remaining ``6 x 12`` spatial map, ``Linear(32 -> 256)``, ``ReLU``.
* Towers: fully separate actor and critic, each with its own encoder. Each concatenates the
  256-d image feature with the (already normalised) vector observation, then
  ``Linear -> ELU -> Linear -> ELU -> head``.
* The critic is asymmetric: it consumes ``vec_priv`` (14 dims) where the actor consumes ``vec``
  (8 dims).
* Orthogonal init: gain ``sqrt(2)`` on conv and hidden layers, ``0.01`` on the policy mean head,
  ``1.0`` on the value head, all biases zero.

Global average pooling replaces the Impala flatten. It removes the post-flatten linear layer that
holds roughly 84% of Impala's parameters, and its translation insensitivity is the inductive bias
we want under camera-pose domain randomisation. Trumpp et al. (2025, arXiv:2503.05546) report
that the pooled variant "outperforms larger and more complex models ... particularly in terms of
generalization"; no stronger numeric claim is made here. GAP saves PARAMETERS, not activations;
the activation budget is what sets the minibatch size in S5.6.

Vec-only mode (``NetworkConfig.use_image = False``) drops the encoder entirely and feeds the
vector observation straight into the fusion MLP. It is used by milestones M3 and M5 and by the
Pendulum-v1 CI gate, so the Gaussian head, the KL controller and the full loss path are exercised
without a GPU or a renderer.
"""

from __future__ import annotations

import dataclasses
from typing import Any, NamedTuple

import torch
import torch.nn as nn

from duckiebot_rl.ppo.config import NetworkConfig
from duckiebot_rl.ppo.distributions import DiagGaussianHead, GaussianOutput, orthogonal_init_

layer_init = orthogonal_init_


class ResidualBlock(nn.Module):
    """Pre-activation residual block: ``ReLU -> Conv3x3 -> ReLU -> Conv3x3 -> +skip``.

    Args:
        channels: Input and output channel count (unchanged by the block).
        gain: Orthogonal-init gain for both convolutions.
    """

    def __init__(self, channels: int, gain: float) -> None:
        super().__init__()
        self.conv0 = layer_init(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1), gain)
        self.conv1 = layer_init(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1), gain)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block.

        Args:
            x: NCHW feature map.

        Returns:
            NCHW feature map of the same shape.
        """
        y = self.conv0(self.relu(x))
        y = self.conv1(self.relu(y))
        return x + y


class ConvSequence(nn.Module):
    """One Impala/Impoola stage: ``Conv3x3 -> MaxPool3x3/2 -> ResBlock -> ResBlock``.

    Args:
        in_channels: Input channel count.
        out_channels: Output channel count.
        gain: Orthogonal-init gain for the convolution.
    """

    def __init__(self, in_channels: int, out_channels: int, gain: float) -> None:
        super().__init__()
        self.conv = layer_init(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1), gain)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res0 = ResidualBlock(out_channels, gain)
        self.res1 = ResidualBlock(out_channels, gain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the stage, halving both spatial dimensions.

        Args:
            x: NCHW feature map.

        Returns:
            NCHW feature map with ``out_channels`` channels and halved spatial size.
        """
        return self.res1(self.res0(self.pool(self.conv(x))))


class ImpoolaEncoder(nn.Module):
    """Impala-CNN with global average pooling in place of the flatten.

    Args:
        in_channels: Stacked observation channel count (9 for three RGB frames).
        channels: Per-stage output channels.
        out_dim: Width of the final linear projection.
        gain: Orthogonal-init gain for every convolution and the projection.

    Attributes:
        out_dim: Width of the produced feature vector.
    """

    def __init__(
        self,
        in_channels: int,
        channels: tuple[int, ...] = (16, 32, 32),
        out_dim: int = 256,
        gain: float = 2.0**0.5,
    ) -> None:
        super().__init__()
        stages: list[nn.Module] = []
        current = in_channels
        for channel in channels:
            stages.append(ConvSequence(current, channel, gain))
            current = channel
        self.stages = nn.Sequential(*stages)
        self.relu = nn.ReLU(inplace=False)
        self.fc = layer_init(nn.Linear(current, out_dim), gain)
        self.out_dim = out_dim
        self.in_channels = in_channels

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Encode a batch of stacked frames.

        Args:
            image: NHWC batch, ``(B, H, W, C)``, dtype uint8 (raw observation) or float32
                (already in ``[0, 255]``). The conversion to NCHW float and the ``/255`` scaling
                happen here and nowhere else, per SPEC v2 S5.2.

        Returns:
            Feature batch of shape ``(B, out_dim)``.

        Raises:
            ValueError: If the input is not 4-dimensional NHWC with the expected channel count.
        """
        if image.ndim != 4:
            raise ValueError(f"expected NHWC image with 4 dims, got shape {tuple(image.shape)}")
        if image.shape[-1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} channels last (NHWC), got shape {tuple(image.shape)}"
            )
        x = image.permute(0, 3, 1, 2).float() / 255.0
        features = self.relu(self.stages(x))
        pooled = features.mean(dim=(2, 3))
        return self.relu(self.fc(pooled))


class Tower(nn.Module):
    """One complete branch: optional image encoder, vector fusion MLP, optional output head.

    Args:
        cfg: Network configuration.
        vector_dim: Width of the vector observation this tower consumes (``vec_dim`` for the
            actor, ``priv_dim`` for the asymmetric critic).
        head_dim: Output width of the final linear head, or None to return the hidden features
            (the actor returns features, which the Gaussian head then consumes).
        head_gain: Orthogonal-init gain for the head.

    Attributes:
        feature_dim: Width of the tower's hidden representation.
    """

    def __init__(
        self,
        cfg: NetworkConfig,
        vector_dim: int,
        head_dim: int | None,
        head_gain: float = 1.0,
    ) -> None:
        super().__init__()
        self.encoder: ImpoolaEncoder | None = None
        fusion_in = vector_dim
        if cfg.use_image:
            self.encoder = ImpoolaEncoder(
                in_channels=cfg.obs_channels,
                channels=tuple(cfg.encoder_channels),
                out_dim=cfg.encoder_out,
                gain=cfg.hidden_gain,
            )
            fusion_in += cfg.encoder_out

        self.mlp = nn.Sequential(
            layer_init(nn.Linear(fusion_in, cfg.hidden_dim), cfg.hidden_gain),
            nn.ELU(),
            layer_init(nn.Linear(cfg.hidden_dim, cfg.hidden_dim), cfg.hidden_gain),
            nn.ELU(),
        )
        self.feature_dim = cfg.hidden_dim
        self.head = (
            layer_init(nn.Linear(cfg.hidden_dim, head_dim), head_gain) if head_dim is not None else None
        )

    def forward(self, image: torch.Tensor | None, vector: torch.Tensor) -> torch.Tensor:
        """Run the tower.

        Args:
            image: NHWC image batch, or None in vec-only mode.
            vector: Normalised vector observation of shape ``(B, vector_dim)``.

        Returns:
            ``(B, head_dim)`` if the tower has a head, else ``(B, hidden_dim)``.

        Raises:
            ValueError: If an image is required but not supplied.
        """
        if self.encoder is not None:
            if image is None:
                raise ValueError("this tower was built with use_image=True but image is None")
            fused = torch.cat([self.encoder(image), vector], dim=-1)
        else:
            fused = vector
        features = self.mlp(fused)
        return features if self.head is None else self.head(features)


class ActorCriticOutput(NamedTuple):
    """Bundle returned by :meth:`ActorCritic.get_action_and_value`.

    Attributes:
        action: Unclipped raw Gaussian sample, shape ``(B, A)``.
        log_prob: Summed log density of ``action``, shape ``(B,)``.
        entropy: Summed policy entropy, shape ``(B,)``.
        value: Raw critic output, shape ``(B,)``. In normalised space when value-target
            normalisation is enabled; the learner denormalises it.
        mu: Policy mean, shape ``(B, A)``.
        log_std: Clamped policy log std broadcast to ``(B, A)``.
    """

    action: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor
    mu: torch.Tensor
    log_std: torch.Tensor


class ActorCritic(nn.Module):
    """Separate actor and critic towers plus a shared-nothing diagonal Gaussian head.

    The critic is asymmetric: it reads ``vec_priv``, which contains privileged lane-frame and
    obstacle state that the deployed policy never sees. The actor reads ``vec`` only, so the
    exported artifact is deployable as-is.

    Args:
        cfg: Network configuration.
    """

    def __init__(self, cfg: NetworkConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.actor = Tower(cfg, vector_dim=cfg.vec_dim, head_dim=None)
        self.critic = Tower(cfg, vector_dim=cfg.priv_dim, head_dim=1, head_gain=cfg.value_head_gain)
        self.policy_head = DiagGaussianHead(
            feat_dim=self.actor.feature_dim,
            act_dim=cfg.act_dim,
            sigma_init=cfg.sigma_init,
            log_std_min=cfg.log_std_min,
            log_std_max=cfg.log_std_max,
            squash=cfg.squash,
            mean_head_gain=cfg.policy_head_gain,
        )

    def actor_features(self, image: torch.Tensor | None, vec: torch.Tensor) -> torch.Tensor:
        """Return the actor tower's hidden features, shape ``(B, hidden_dim)``.

        Args:
            image: NHWC image batch, or None in vec-only mode.
            vec: Normalised actor vector observation.

        Returns:
            Hidden feature batch.
        """
        return self.actor(image, vec)

    def get_value(self, image: torch.Tensor | None, vec_priv: torch.Tensor) -> torch.Tensor:
        """Return the raw critic output, shape ``(B,)``.

        Args:
            image: NHWC image batch, or None in vec-only mode.
            vec_priv: Normalised privileged vector observation.

        Returns:
            Raw (possibly normalised-space) value estimates.
        """
        return self.critic(image, vec_priv).squeeze(-1)

    def get_action(
        self,
        image: torch.Tensor | None,
        vec: torch.Tensor,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> GaussianOutput:
        """Sample or evaluate an action without touching the critic.

        Args:
            image: NHWC image batch, or None in vec-only mode.
            vec: Normalised actor vector observation.
            action: Stored unclipped sample to evaluate, or None to draw a new one.
            deterministic: Return the mean instead of sampling.

        Returns:
            A :class:`GaussianOutput`.
        """
        return self.policy_head(self.actor_features(image, vec), action=action, deterministic=deterministic)

    def get_action_and_value(
        self,
        image: torch.Tensor | None,
        vec: torch.Tensor,
        vec_priv: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> ActorCriticOutput:
        """Run both towers.

        Args:
            image: NHWC image batch, or None in vec-only mode.
            vec: Normalised actor vector observation, shape ``(B, vec_dim)``.
            vec_priv: Normalised privileged observation, shape ``(B, priv_dim)``. If None, ``vec``
                is reused, which is only valid when ``priv_dim == vec_dim`` (vec-only tasks such
                as the Pendulum CI gate).
            action: Stored unclipped sample to evaluate, or None to draw a new one.
            deterministic: Return the mean instead of sampling.

        Returns:
            An :class:`ActorCriticOutput`.

        Raises:
            ValueError: If ``vec_priv`` is omitted while ``priv_dim != vec_dim``.
        """
        if vec_priv is None:
            if self.cfg.priv_dim != self.cfg.vec_dim:
                raise ValueError(
                    f"vec_priv is required when priv_dim ({self.cfg.priv_dim}) != vec_dim "
                    f"({self.cfg.vec_dim})"
                )
            vec_priv = vec
        gaussian = self.get_action(image, vec, action=action, deterministic=deterministic)
        value = self.get_value(image, vec_priv)
        return ActorCriticOutput(
            action=gaussian.action,
            log_prob=gaussian.log_prob,
            entropy=gaussian.entropy,
            value=value,
            mu=gaussian.mu,
            log_std=gaussian.log_std,
        )


class ActorMean(nn.Module):
    """Deployment view of a trained agent: ``(image, vec) -> mu``.

    The exported artifact is deterministic and contains no sampling, no critic and no privileged
    input. It is deliberately a thin view over the trained :class:`ActorCritic` rather than a
    re-implementation, so the exported graph cannot drift from the trained one.

    Args:
        agent: A trained actor-critic. Only its actor tower and policy mean head are used.
    """

    def __init__(self, agent: ActorCritic) -> None:
        super().__init__()
        self.agent = agent

    def forward(self, image: torch.Tensor | None, vec: torch.Tensor) -> torch.Tensor:
        """Return the policy mean.

        Args:
            image: ``(N, H, W, C)`` NHWC uint8 or float observation, or None in vec-only mode.
            vec: ``(N, vec_dim)`` ALREADY-NORMALISED vector observation. The deployment wrapper
                bakes the running-normaliser constants in ahead of this call.

        Returns:
            ``(N, act_dim)`` policy mean, unclipped.
        """
        return self.agent.policy_head.mean(self.agent.actor(image, vec))


def build_actor(
    checkpoint: dict[str, Any] | None = None,
    network_cfg: NetworkConfig | None = None,
) -> ActorMean:
    """Entry point used by the deployment exporter to rebuild a trained actor.

    Referenced by ``duckiebot_rl/deploy/export_onnx.py`` as
    ``"duckiebot_rl.ppo.networks:build_actor"``. It accepts the full payload written by
    :func:`duckiebot_rl.ppo.checkpoint.save_checkpoint` and recovers the architecture from the
    config the run was trained with, so an exported policy can never be built against a different
    network shape than the weights it loads.

    Args:
        checkpoint: A loaded checkpoint payload, or None to build an untrained actor with the
            default architecture.
        network_cfg: Explicit architecture, overriding whatever the checkpoint records.

    Returns:
        An :class:`ActorMean` in eval mode.

    Raises:
        KeyError: If a checkpoint is supplied but carries no recognisable model state dict.
    """
    cfg = network_cfg
    if cfg is None and isinstance(checkpoint, dict):
        raw_config = checkpoint.get("config")
        if isinstance(raw_config, dict) and isinstance(raw_config.get("network"), dict):
            valid = {field.name for field in dataclasses.fields(NetworkConfig)}
            cfg = NetworkConfig(**{k: v for k, v in raw_config["network"].items() if k in valid})
    agent = ActorCritic(cfg if cfg is not None else NetworkConfig())

    if isinstance(checkpoint, dict):
        state: dict[str, Any] | None = None
        learner = checkpoint.get("learner")
        if isinstance(learner, dict) and isinstance(learner.get("model"), dict):
            state = learner["model"]
        elif isinstance(checkpoint.get("model"), dict):
            state = checkpoint["model"]
        if state is None:
            raise KeyError("checkpoint carries no model state dict under 'learner.model' or 'model'")
        agent.load_state_dict(state)

    return ActorMean(agent).eval()


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    """Count parameters in a module.

    Args:
        module: Any torch module.
        trainable_only: Count only parameters with ``requires_grad``.

    Returns:
        Total parameter count.
    """
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


def parameter_report(agent: ActorCritic) -> dict[str, int]:
    """Return exact per-component parameter counts for an :class:`ActorCritic`.

    Used by ``tests/unit/test_networks.py`` to pin the architecture: any silent change to channel
    widths, the fusion dimension or the head shapes moves one of these numbers.

    Args:
        agent: The agent to inspect.

    Returns:
        Dict with keys ``actor_encoder``, ``actor_mlp``, ``critic_encoder``, ``critic_mlp``,
        ``critic_head``, ``policy_head``, ``actor_total``, ``critic_total`` and ``total``.
        Encoder entries are 0 in vec-only mode.
    """
    actor_encoder = count_parameters(agent.actor.encoder) if agent.actor.encoder is not None else 0
    critic_encoder = count_parameters(agent.critic.encoder) if agent.critic.encoder is not None else 0
    actor_mlp = count_parameters(agent.actor.mlp)
    critic_mlp = count_parameters(agent.critic.mlp)
    critic_head = count_parameters(agent.critic.head) if agent.critic.head is not None else 0
    policy_head = count_parameters(agent.policy_head)
    report = {
        "actor_encoder": actor_encoder,
        "actor_mlp": actor_mlp,
        "policy_head": policy_head,
        "critic_encoder": critic_encoder,
        "critic_mlp": critic_mlp,
        "critic_head": critic_head,
    }
    report["actor_total"] = actor_encoder + actor_mlp + policy_head
    report["critic_total"] = critic_encoder + critic_mlp + critic_head
    report["total"] = count_parameters(agent)
    return report
