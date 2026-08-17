"""From-scratch PPO for the Duckiebot lane-following task (SPEC v2, section S6).

Nothing in this subpackage imports Isaac Sim, Isaac Lab, or any RL library. Every module runs on
CPU so that the whole learner is unit-testable in CI.
"""

from __future__ import annotations

from duckiebot_rl.ppo.buffer import RolloutBuffer, TerminalCache
from duckiebot_rl.ppo.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    REQUIRED_CURRICULUM_KEYS,
    collect_rng_state,
    config_hash,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from duckiebot_rl.ppo.config import NetworkConfig, PPOConfig, configure_precision, ratio_assert_atol
from duckiebot_rl.ppo.distributions import (
    DiagGaussianHead,
    GaussianOutput,
    diag_gaussian_entropy,
    diag_gaussian_kl,
    diag_gaussian_log_prob,
    orthogonal_init_,
    tanh_log_det,
)
from duckiebot_rl.ppo.gae import compute_gae, compute_gae_reference
from duckiebot_rl.ppo.networks import (
    ActorCritic,
    ActorMean,
    ConvSequence,
    ImpoolaEncoder,
    ResidualBlock,
    Tower,
    build_actor,
    count_parameters,
    parameter_report,
)
from duckiebot_rl.ppo.ppo import PPO
from duckiebot_rl.ppo.running_norm import RunningMeanStd

__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "PPO",
    "REQUIRED_CURRICULUM_KEYS",
    "ActorCritic",
    "ActorMean",
    "ConvSequence",
    "DiagGaussianHead",
    "GaussianOutput",
    "ImpoolaEncoder",
    "NetworkConfig",
    "PPOConfig",
    "ResidualBlock",
    "RolloutBuffer",
    "RunningMeanStd",
    "TerminalCache",
    "Tower",
    "build_actor",
    "collect_rng_state",
    "compute_gae",
    "compute_gae_reference",
    "config_hash",
    "configure_precision",
    "count_parameters",
    "diag_gaussian_entropy",
    "diag_gaussian_kl",
    "diag_gaussian_log_prob",
    "load_checkpoint",
    "orthogonal_init_",
    "parameter_report",
    "ratio_assert_atol",
    "restore_rng_state",
    "save_checkpoint",
    "tanh_log_det",
]
