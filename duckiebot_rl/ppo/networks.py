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
from torch.utils.checkpoint import checkpoint_sequential

from duckiebot_rl.ppo.config import NetworkConfig
from duckiebot_rl.ppo.distributions import DiagGaussianHead, GaussianOutput, orthogonal_init_

layer_init = orthogonal_init_

LEAKY_SLOPE = 0.01
"""Negative slope of every encoder activation (2026-08-21, the second dead-encoder incident).

Plain ReLU gave the actor's conv trunk a one-way death: once a KL-starved learning-rate spiral
(the adaptive controller raises lr toward lr_max whenever KL is low, and a dying encoder makes
KL low) pushed every pre-activation negative, output and gradient were exactly zero forever.
Measured twice: at iteration ~600 of the survival run and at iteration 12 of a fresh run, with
pre-ReLU magnitudes around 1.7e4. A leaky slope keeps a gradient alive at any excursion, so the
same event becomes a recoverable dip instead of brain death. 0.01 is the standard slope; at
healthy activations the network is numerically indistinguishable from ReLU. The paired fix caps
``PPOConfig.lr_max`` at 1e-3, closing the spiral's throttle. Archived checkpoints predate this
and were all either healthy-ReLU (compatible in practice: healthy activations rarely cross
zero) or dead (worthless); the S6.2 activation note in the spec is amended by this constant.
"""


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
        self.relu = nn.LeakyReLU(LEAKY_SLOPE, inplace=False)

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
        self.relu = nn.LeakyReLU(LEAKY_SLOPE, inplace=False)
        self.fc = layer_init(nn.Linear(current, out_dim), gain)
        self.out_dim = out_dim
        self.in_channels = in_channels
        self.checkpoint_trunk = False

    def prepare(self, image: torch.Tensor) -> torch.Tensor:
        """Convert an NHWC observation to the NCHW float tensor the convolutions consume.

        Split out of :meth:`forward` so that the actor and the critic, which own separate
        encoders but read the SAME image, pay for the permute, the widening and the ``/255``
        exactly once per batch instead of twice (see
        :meth:`ActorCritic.get_action_and_value`). The returned tensor is byte-for-byte the one
        :meth:`forward` would have built, so sharing it changes no number anywhere.

        Args:
            image: NHWC batch, ``(B, H, W, C)``, dtype uint8 (raw observation) or float32
                (already in ``[0, 255]``). The conversion to NCHW float and the ``/255`` scaling
                happen here and nowhere else, per SPEC v2 S5.2.

        Returns:
            ``(B, C, H, W)`` float tensor scaled to ``[0, 1]``.

        Raises:
            ValueError: If the input is not 4-dimensional NHWC with the expected channel count.
        """
        if image.ndim != 4:
            raise ValueError(f"expected NHWC image with 4 dims, got shape {tuple(image.shape)}")
        if image.shape[-1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} channels last (NHWC), got shape {tuple(image.shape)}"
            )
        return image.permute(0, 3, 1, 2).float() / 255.0

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Run the convolutional trunk on an already-prepared NCHW float batch.

        When :attr:`checkpoint_trunk` is set and gradients are being recorded, the three
        ConvSequences are wrapped in ``checkpoint_sequential`` so their activations are recomputed
        during the backward pass instead of being kept. The trunk is deterministic - no dropout,
        no normalisation layer, no RNG - so the recomputed activations are bit-identical to the
        discarded ones and the gradients are unchanged; this is a pure memory-for-time trade, not
        an approximation. Under ``torch.no_grad`` (the rollout path) the flag is ignored, because
        there are no activations to keep.

        Args:
            x: ``(B, C, H, W)`` float tensor as produced by :meth:`prepare`.

        Returns:
            Feature batch of shape ``(B, out_dim)``.
        """
        if self.checkpoint_trunk and torch.is_grad_enabled():
            trunk = checkpoint_sequential(self.stages, len(self.stages), x, use_reentrant=False)
        else:
            trunk = self.stages(x)
        pooled = self.relu(trunk).mean(dim=(2, 3))
        return self.relu(self.fc(pooled))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Encode a batch of stacked frames.

        Args:
            image: NHWC batch, ``(B, H, W, C)``, dtype uint8 (raw observation) or float32
                (already in ``[0, 255]``).

        Returns:
            Feature batch of shape ``(B, out_dim)``.

        Raises:
            ValueError: If the input is not 4-dimensional NHWC with the expected channel count.
        """
        return self.encode(self.prepare(image))


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

    def forward(
        self,
        image: torch.Tensor | None,
        vector: torch.Tensor,
        prepared: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the tower.

        Args:
            image: NHWC image batch, or None in vec-only mode.
            vector: Normalised vector observation of shape ``(B, vector_dim)``.
            prepared: The NCHW float batch that :meth:`ImpoolaEncoder.prepare` would have built
                from ``image``, when the caller has already built it for the sibling tower. It is
                the same tensor by construction, so passing it is a pure saving; ``image`` is
                ignored when it is given.

        Returns:
            ``(B, head_dim)`` if the tower has a head, else ``(B, hidden_dim)``.

        Raises:
            ValueError: If an image is required but neither ``image`` nor ``prepared`` is given.
        """
        if self.encoder is not None:
            if prepared is None:
                if image is None:
                    raise ValueError("this tower was built with use_image=True but image is None")
                prepared = self.encoder.prepare(image)
            fused = torch.cat([self.encoder.encode(prepared), vector], dim=-1)
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

    def set_encoder_checkpointing(self, enabled: bool) -> None:
        """Turn activation checkpointing of both encoder trunks on or off.

        Args:
            enabled: True to recompute the trunk during the backward pass instead of storing its
                activations. See :meth:`ImpoolaEncoder.encode` for why this is exact rather than
                approximate, and ``PPOConfig.checkpoint_encoder`` for the measured trade.
        """
        for tower in (self.actor, self.critic):
            if tower.encoder is not None:
                tower.encoder.checkpoint_trunk = bool(enabled)

    def prepare_image(self, image: torch.Tensor | None) -> torch.Tensor | None:
        """Build the NCHW float batch both towers consume, once.

        The actor and the critic own SEPARATE encoders (S6.2) but read the SAME pixels, so the
        NHWC-to-NCHW permute, the uint8 widening and the ``/255`` were being paid twice per
        forward pass. :meth:`ImpoolaEncoder.prepare` does not touch any parameter, so the tensor
        the actor's encoder builds is bit-for-bit the one the critic's encoder would have built;
        sharing it removes two kernels and one full-size float activation per pass and changes no
        number.

        Args:
            image: NHWC image batch, or None in vec-only mode.

        Returns:
            The prepared ``(B, C, H, W)`` float batch, or None when there is no image encoder.
        """
        encoder = self.actor.encoder if self.actor.encoder is not None else self.critic.encoder
        if encoder is None or image is None:
            return None
        return encoder.prepare(image)

    def actor_features(
        self,
        image: torch.Tensor | None,
        vec: torch.Tensor,
        prepared: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the actor tower's hidden features, shape ``(B, hidden_dim)``.

        Args:
            image: NHWC image batch, or None in vec-only mode.
            vec: Normalised actor vector observation.
            prepared: Output of :meth:`prepare_image`, when it has already been built.

        Returns:
            Hidden feature batch.
        """
        return self.actor(image, vec, prepared=prepared)

    def get_value(
        self,
        image: torch.Tensor | None,
        vec_priv: torch.Tensor,
        prepared: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the raw critic output, shape ``(B,)``.

        Args:
            image: NHWC image batch, or None in vec-only mode.
            vec_priv: Normalised privileged vector observation.
            prepared: Output of :meth:`prepare_image`, when it has already been built.

        Returns:
            Raw (possibly normalised-space) value estimates.
        """
        return self.critic(image, vec_priv, prepared=prepared).squeeze(-1)

    def get_action(
        self,
        image: torch.Tensor | None,
        vec: torch.Tensor,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
        prepared: torch.Tensor | None = None,
    ) -> GaussianOutput:
        """Sample or evaluate an action without touching the critic.

        Args:
            image: NHWC image batch, or None in vec-only mode.
            vec: Normalised actor vector observation.
            action: Stored unclipped sample to evaluate, or None to draw a new one.
            deterministic: Return the mean instead of sampling.
            prepared: Output of :meth:`prepare_image`, when it has already been built.

        Returns:
            A :class:`GaussianOutput`.
        """
        features = self.actor_features(image, vec, prepared=prepared)
        return self.policy_head(features, action=action, deterministic=deterministic)

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
        # One conversion, two towers. See :meth:`prepare_image`.
        prepared = self.prepare_image(image)
        gaussian = self.get_action(image, vec, action=action, deterministic=deterministic, prepared=prepared)
        value = self.get_value(image, vec_priv, prepared=prepared)
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


@torch.no_grad()
def encoder_liveness(
    encoder: ImpoolaEncoder,
    images: torch.Tensor,
    sample: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(live_frac, out_std)``: is this encoder's output alive and image-dependent?

    Why this exists: the actor encoder of run ``20260819T031011Z_lanefollow_seed0_survival_seed0``
    died silently between iterations 600 and 700. Its trunk pre-ReLU activations went all-negative
    (|mean| ~1.7e4), so ``relu(trunk)`` was exactly zero at every position, the pooled vector was
    zero, ``relu(fc(0))`` was zero for all 256 units, and from then on the policy returned
    bit-identical actions for real, blank and random images. That state is a one-way cliff: with
    every ReLU shut, the image gradient is exactly zero forever, and nothing in the training
    diagnostics showed it. This probe makes the cliff visible in the iteration it happens.

    Two scalars, both computed on the device so the caller can fold them into an existing host
    transfer (SPEC v2 S6.7 guard 5):

    * ``live_frac``: fraction of output units that are strictly positive for at least one sampled
      frame. The dead-zero pathway reads exactly 0.0; a healthy fresh encoder reads well above 0.
    * ``out_std``: mean over units of the across-sample standard deviation of the output. This
      catches the subtler failure the first scalar cannot: a constant nonzero output, which is
      just as blind (the observed signature was "same action for any image"). Reads 0.0 whenever
      the output does not depend on the input.

    Alarm rule for dashboards and babysitting: either scalar at 0.0 on an image run means the
    vision pathway is dead or blind, and the run is burning GPU-hours on a vec-only policy.

    Args:
        encoder: The encoder to probe (in practice the ACTOR's; the critic's was still alive in
            the postmortem and would have masked the failure).
        images: ``(B, H, W, C)`` NHWC uint8 or float batch, e.g. the rollout buffer's flattened
            frames. A strided sample is probed so the cost stays one small forward pass.
        sample: Maximum frames to probe; the batch is strided to spread the sample across it.

    Returns:
        ``(live_frac, out_std)``, each a 0-dim tensor on the encoder's device.
    """
    stride = max(1, int(images.shape[0]) // max(1, int(sample)))
    features = encoder(images[::stride][: max(1, int(sample))])
    live_frac = (features > 0.0).any(dim=0).float().mean()
    # Population std: defined (0.0) even for a single-frame probe, where the unbiased form is NaN.
    out_std = features.std(dim=0, correction=0).mean()
    return live_frac, out_std


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
