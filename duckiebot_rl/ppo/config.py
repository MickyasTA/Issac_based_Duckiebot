"""Configuration dataclasses for the from-scratch PPO learner.

Defaults are the SPEC v2 S6.6 hyperparameter table verbatim. Anything not in that table is
derived from S6.2 (network), S6.5 (losses / precision) or S5.2 (observation spaces).

Precision policy (SPEC v2 S6.5, resolving critic item G): fp32 end to end. bf16/fp16 autocast is
FORBIDDEN. TF32 matmul/conv is allowed for throughput, which loosens the epoch-0 ratio assert to
``atol = 5e-3``; setting ``DUCKIEBOT_RL_STRICT_FP32=1`` disables TF32 and tightens the assert to
``1e-5``. CI and the M4 gate run strict mode.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

STRICT_FP32_ENV_VAR = "DUCKIEBOT_RL_STRICT_FP32"
RATIO_ATOL_TF32 = 5e-3
RATIO_ATOL_STRICT = 1e-5


def strict_fp32_enabled() -> bool:
    """Return True when ``DUCKIEBOT_RL_STRICT_FP32`` is set to a truthy value.

    Returns:
        True if strict fp32 mode is requested, else False.
    """
    return os.environ.get(STRICT_FP32_ENV_VAR, "0").strip().lower() in {"1", "true", "yes", "on"}


def ratio_assert_atol() -> float:
    """Return the tolerance for the epoch-0 ``ratio == 1`` guard (SPEC v2 S6.7 guard 1).

    Returns:
        ``1e-5`` in strict fp32 mode, else ``5e-3`` because TF32 kernels may differ between the
        rollout pass and the update pass.
    """
    return RATIO_ATOL_STRICT if strict_fp32_enabled() else RATIO_ATOL_TF32


def configure_precision() -> bool:
    """Apply the SPEC v2 S6.5 precision policy to the global torch backends.

    Enables TF32 for matmul and cuDNN unless strict fp32 mode is requested, in which case TF32 is
    disabled so that the rollout and update passes select bit-identical kernels.

    Returns:
        True if strict fp32 mode is active.
    """
    strict = strict_fp32_enabled()
    torch.backends.cuda.matmul.allow_tf32 = not strict
    torch.backends.cudnn.allow_tf32 = not strict
    return strict


@dataclass
class NetworkConfig:
    """Shapes and architecture of the actor and critic towers (SPEC v2 S5.2, S6.2).

    Attributes:
        use_image: If False the image encoder is dropped entirely and both towers consume only
            the vector observation. This is the "vec-only mode" of S6.2, used by M3/M5 and by the
            Pendulum-v1 CI gate, and it is a first-class code path, not a test-only shim.
        obs_height: Observation height in pixels after the S4.3 crop.
        obs_width: Observation width in pixels.
        obs_channels: Stacked channel count (3 frames x RGB).
        vec_dim: Actor vector-observation width.
        priv_dim: Critic privileged vector-observation width (asymmetric critic).
        act_dim: Action dimensionality.
        encoder_channels: Per-ConvSequence output channels of the Impoola encoder.
        encoder_out: Width of the encoder output linear layer.
        hidden_dim: Width of the two ELU fusion layers in each tower.
        sigma_init: Initial policy standard deviation.
        log_std_min: Lower clamp on ``log_std``.
        log_std_max: Upper clamp on ``log_std``.
        squash: Enable the tanh-squashed policy ablation.
        hidden_gain: Orthogonal-init gain for conv and hidden linear layers.
        policy_head_gain: Orthogonal-init gain for the policy mean head.
        value_head_gain: Orthogonal-init gain for the value head.
    """

    use_image: bool = True
    obs_height: int = 48
    obs_width: int = 96
    obs_channels: int = 9
    vec_dim: int = 8
    priv_dim: int = 14
    act_dim: int = 2
    encoder_channels: tuple[int, ...] = (16, 32, 32)
    encoder_out: int = 256
    hidden_dim: int = 256
    sigma_init: float = 0.5
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    squash: bool = False
    hidden_gain: float = math.sqrt(2.0)
    policy_head_gain: float = 0.01
    value_head_gain: float = 1.0

    def __post_init__(self) -> None:
        """Validate the shape fields.

        Raises:
            ValueError: If any dimension is non-positive, or if the image dimensions are not
                divisible by 8 (three stride-2 max-pools) when the image encoder is enabled.
        """
        for name in ("vec_dim", "priv_dim", "act_dim", "encoder_out", "hidden_dim"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.priv_dim < self.vec_dim:
            raise ValueError(
                f"priv_dim ({self.priv_dim}) must be >= vec_dim ({self.vec_dim}): the privileged "
                "critic observation is the actor observation plus extra fields"
            )
        if self.use_image:
            if min(self.obs_height, self.obs_width, self.obs_channels) <= 0:
                raise ValueError("image observation dimensions must all be positive")
            if self.obs_height % 8 or self.obs_width % 8:
                raise ValueError(
                    f"obs_height and obs_width must be divisible by 8 for {len(self.encoder_channels)} "
                    f"stride-2 pools, got ({self.obs_height}, {self.obs_width})"
                )
            if len(self.encoder_channels) == 0:
                raise ValueError("encoder_channels must not be empty when use_image is True")

    @property
    def obs_shape(self) -> tuple[int, int, int]:
        """Return the NHWC image observation shape ``(H, W, C)``."""
        return (self.obs_height, self.obs_width, self.obs_channels)


@dataclass
class PPOConfig:
    """PPO hyperparameters (SPEC v2 S6.6 table).

    Attributes:
        num_envs: Parallel environment count ``N``.
        num_steps: Rollout length ``T``; batch is ``N * T``.
        num_minibatches: Minibatch count per epoch. VRAM-derived in S5.6.
        update_epochs: Passes over each rollout.
        gamma: Discount factor.
        gae_lambda: GAE trace-decay parameter.
        clip_coef: PPO clipping epsilon.
        clip_vloss: Value-loss clipping. False per S6.5 (plain MSE); kept as an ablation flag.
        vf_coef: Value-loss weight.
        ent_coef: Entropy bonus weight. 0.0 per S6.5; a state-independent Gaussian ``log_std``
            has unbounded entropy, so a bonus inflates sigma monotonically.
        bounds_coef: Weight of ``mean(relu(|mu| - 1) ** 2)``, which keeps the mean inside the box.
        max_grad_norm: Global gradient-norm clip.
        learning_rate: Initial Adam learning rate.
        adam_eps: Adam epsilon.
        kl_adaptive: Enable the KL-adaptive learning-rate controller.
        kl_target_upper: Above this analytic KL the learning rate is divided by ``lr_factor``.
        kl_target_lower: Below this analytic KL the learning rate is multiplied by ``lr_factor``.
        lr_factor: Multiplicative step of the controller.
        lr_min: Lower bound on the learning rate.
        lr_max: Upper bound on the learning rate. 1e-3, an order below its original value:
            the KL-adaptive controller RAISES lr while KL is under target, and a dying encoder
            keeps KL small, so a high ceiling turns encoder trouble into a death spiral (lr
            climbs toward the ceiling, Adam's per-parameter step grows with lr regardless of
            the clipped gradient norm, the encoder dies harder, KL falls further). Measured
            killing the actor's vision twice before the cap; PPO with Adam essentially never
            profits from lr above 1e-3 anyway.
        kl_adapt_per_minibatch: If True the controller fires once per minibatch (rsl_rl / Isaac
            ecosystem behaviour); if False it fires once per update from the batch-mean KL.
        norm_adv: Normalise advantages once at batch level (never also per minibatch).
        normalize_value_targets: Fit a running mean/std to the value targets and train the critic
            in normalised space.
        normalize_vec: Normalise ``vec`` and ``vec_priv`` with running statistics.
        vec_clip: Symmetric clip applied after vector normalisation.
        rsl_rl_gae_approx: Ablation flag. When True, truncation bootstraps from ``V(s_t)`` instead
            of the captured terminal observation, recovering the rsl_rl approximation exactly.
        ratio_assert: Enable the epoch-0 minibatch-0 ``ratio == 1`` guard.
        channels_last: Store the convolution weights in ``torch.channels_last`` on CUDA. The
            observation arrives NHWC and ``ImpoolaEncoder.prepare`` permutes it, so the tensor
            reaching every convolution is ALREADY channels-last-strided and cuDNN already selects
            channels-last kernels for it; what this flag removes is the per-call
            ``weight.contiguous(channels_last)`` copy cuDNN otherwise performs on every one of the
            30 convolutions. Measured on this machine (RTX 3080 Laptop, clean process, TF32):
            no-grad two-tower forward at N=64 17.59 -> 10.36 ms, grad step at minibatch 128
            50.15 -> 48.47 ms. The forward is bit-identical (measured: first-step loss delta 0.0);
            the BACKWARD selects different kernels, so after 8 optimiser steps the parameters
            differ by up to 4.4e-5 absolute, which is inside the TF32 noise the precision policy
            already sanctions. Ignored on CPU (where channels-last convolution is a different and
            slower code path) and forced off in strict fp32 mode, which exists to provide a
            bit-reproducible reference.
        fused_optimizer: Use the fused multi-tensor CUDA Adam kernel. Same update rule, one kernel
            instead of a foreach chain. Measured: grad step at minibatch 128 50.15 -> 39.99 ms
            (-20%), the single largest learner-side saving available. Parameters after 8 steps
            differ from the unfused path by up to 1.1e-5 absolute (float reassociation only).
            Ignored on CPU and forced off in strict fp32 mode.
        checkpoint_encoder: Recompute the encoder trunk during the backward pass instead of
            storing its activations. Provably exact rather than merely close: the trunk is
            deterministic (no dropout, no batch norm, no RNG), so the recomputed activations are
            bit-identical and therefore so are the gradients. Measured at minibatch 512: peak
            allocation 1437.6 -> 735.7 MiB (-49%) for 144.9 -> 176.6 ms (+22%). Off by default;
            it is the lever to reach for if the N=256 update turns out to be limited by VRAM
            residency rather than by arithmetic.
        total_timesteps: Environment-step budget for a full run.
        seed: Master seed recorded in checkpoints.
        device: Torch device string.
    """

    num_envs: int = 256
    num_steps: int = 32
    num_minibatches: int = 16
    update_epochs: int = 4

    gamma: float = 0.99
    gae_lambda: float = 0.95

    clip_coef: float = 0.2
    clip_vloss: bool = False
    vf_coef: float = 1.0
    ent_coef: float = 0.0
    bounds_coef: float = 1e-4
    max_grad_norm: float = 1.0

    learning_rate: float = 3e-4
    adam_eps: float = 1e-5
    kl_adaptive: bool = True
    kl_target_upper: float = 0.02
    kl_target_lower: float = 0.005
    lr_factor: float = 1.5
    lr_min: float = 1e-5
    lr_max: float = 1e-3
    kl_adapt_per_minibatch: bool = True

    norm_adv: bool = True
    normalize_value_targets: bool = True
    normalize_vec: bool = True
    vec_clip: float = 5.0

    rsl_rl_gae_approx: bool = False
    ratio_assert: bool = True

    channels_last: bool = True
    fused_optimizer: bool = True
    checkpoint_encoder: bool = False

    total_timesteps: int = 150_000_000
    seed: int = 0
    device: str = "cuda"

    network: NetworkConfig = field(default_factory=NetworkConfig)

    def __post_init__(self) -> None:
        """Validate the hyperparameters.

        Raises:
            ValueError: If the batch is not divisible by ``num_minibatches``, if any count is
                non-positive, or if the learning-rate / KL bounds are inconsistent.
        """
        for name in ("num_envs", "num_steps", "num_minibatches", "update_epochs"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.batch_size % self.num_minibatches != 0:
            raise ValueError(
                f"batch_size ({self.batch_size}) must be divisible by num_minibatches "
                f"({self.num_minibatches})"
            )
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError(f"gamma must be in (0, 1], got {self.gamma}")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError(f"gae_lambda must be in [0, 1], got {self.gae_lambda}")
        if self.clip_coef <= 0.0:
            raise ValueError(f"clip_coef must be positive, got {self.clip_coef}")
        if not self.lr_min <= self.learning_rate <= self.lr_max:
            raise ValueError(f"learning_rate {self.learning_rate} outside [{self.lr_min}, {self.lr_max}]")
        if not 0.0 < self.kl_target_lower < self.kl_target_upper:
            raise ValueError(
                f"need 0 < kl_target_lower < kl_target_upper, got "
                f"({self.kl_target_lower}, {self.kl_target_upper})"
            )
        if self.lr_factor <= 1.0:
            raise ValueError(f"lr_factor must be > 1, got {self.lr_factor}")

    @property
    def batch_size(self) -> int:
        """Return ``num_envs * num_steps``, the number of transitions per update."""
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self) -> int:
        """Return the number of transitions per gradient step."""
        return self.batch_size // self.num_minibatches

    @property
    def gradient_steps_per_update(self) -> int:
        """Return ``update_epochs * num_minibatches``."""
        return self.update_epochs * self.num_minibatches

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable nested dict of every field."""
        return asdict(self)
