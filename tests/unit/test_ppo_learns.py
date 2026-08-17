"""Credibility gate: the from-scratch PPO must actually learn a continuous-control task.

SPEC v2 S6.7 / milestone M4. The task is Pendulum-v1, not CartPole, because CartPole is discrete
and would leave the entire production path untested. Pendulum exercises:

* the diagonal Gaussian head with a state-independent, clamped ``log_std``;
* raw-Normal sampling with env-side clipping and UNCLIPPED storage, hence the epoch-0
  ``ratio == 1`` guard on real data;
* the KL-adaptive learning-rate controller driven by the exact analytic Gaussian KL;
* the bounds loss, the value-target normaliser and the vector observation normaliser;
* the truncation bootstrap: Pendulum never terminates, it only truncates at 200 steps, so every
  single episode boundary in this test goes through the terminal-capture path. A learner that
  bootstrapped truncations as terminations would be visibly worse here.

The environments are driven by a small explicit vectoriser rather than
``gymnasium.vector.SyncVectorEnv`` so that the test is immune to the gymnasium 1.x next-step
autoreset convention, and so that the terminal observation is handed to the buffer exactly the way
the Isaac Lab environment does it (capture BEFORE reset).

Budget: CPU only, roughly 45 s per seed and well under three minutes for the whole module. The
module carries the ``slow`` marker because ``ci.yml`` runs it as its own job
(``pytest tests/unit/test_ppo_learns.py --runslow``) rather than inside the fast unit sweep.
"""

from __future__ import annotations

import time

import gymnasium as gym
import numpy as np
import pytest
import torch

from duckiebot_rl.ppo.buffer import RolloutBuffer
from duckiebot_rl.ppo.config import NetworkConfig, PPOConfig
from duckiebot_rl.ppo.networks import ActorCritic
from duckiebot_rl.ppo.ppo import PPO

ENV_ID = "Pendulum-v1"
ACTION_SCALE = 2.0
OBS_DIM = 3
ACT_DIM = 1
NUM_ENVS = 16
NUM_STEPS = 128
TOTAL_STEPS = 245_760  # 120 iterations of 16 envs x 128 steps
RETURN_THRESHOLD = -350.0
FINAL_WINDOW = 50
INITIAL_WINDOW = 20
SEEDS = (0, 1)


class ExplicitVectorEnv:
    """Minimal synchronous vectoriser that exposes the true terminal observation.

    ``gymnasium.vector`` autoresets on the caller's behalf and, since 1.0, does so on the NEXT
    step, which silently injects a dummy transition into the rollout. This class instead reports
    the pre-reset observation alongside the post-reset one, which is the same contract the Isaac
    Lab environment offers through its ``_reset_idx`` terminal capture.

    Args:
        env_id: Gymnasium environment id.
        num_envs: Number of independent copies.
        seed: Base seed; env ``i`` is seeded with ``seed + i``.

    Attributes:
        num_envs: Number of environments.
        episode_returns: Undiscounted return accumulated in the current episode, per env.
    """

    def __init__(self, env_id: str, num_envs: int, seed: int) -> None:
        self.envs = [gym.make(env_id) for _ in range(num_envs)]
        self.num_envs = num_envs
        self._seed = seed
        self.episode_returns = np.zeros(num_envs, dtype=np.float64)
        self.completed_returns: list[float] = []

    def reset(self) -> np.ndarray:
        """Reset every environment.

        Returns:
            ``(num_envs, obs_dim)`` float32 observations.
        """
        obs = [env.reset(seed=self._seed + i)[0] for i, env in enumerate(self.envs)]
        self.episode_returns[:] = 0.0
        return np.asarray(obs, dtype=np.float32)

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Step every environment, resetting the finished ones immediately.

        Args:
            actions: ``(num_envs, act_dim)`` actions already scaled to the env action space.

        Returns:
            Tuple ``(next_obs, final_obs, rewards, terminated, truncated)``. ``next_obs`` is the
            post-reset observation where an episode ended; ``final_obs`` holds the true terminal
            observation at those indices and is meaningless elsewhere.
        """
        next_obs = np.zeros((self.num_envs, OBS_DIM), dtype=np.float32)
        final_obs = np.zeros((self.num_envs, OBS_DIM), dtype=np.float32)
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)

        for i, env in enumerate(self.envs):
            obs, reward, term, trunc, _ = env.step(actions[i])
            rewards[i] = reward
            terminated[i] = term
            truncated[i] = trunc
            final_obs[i] = obs
            self.episode_returns[i] += float(reward)
            if term or trunc:
                self.completed_returns.append(float(self.episode_returns[i]))
                self.episode_returns[i] = 0.0
                obs, _ = env.reset()
            next_obs[i] = obs
        return next_obs, final_obs, rewards, terminated, truncated

    def close(self) -> None:
        """Close every underlying environment."""
        for env in self.envs:
            env.close()


def _make_config(seed: int) -> PPOConfig:
    """Build the Pendulum PPO configuration.

    The structural settings (raw Normal, batch-level advantage normalisation, no value clipping,
    zero entropy bonus, KL-adaptive learning rate, bounds loss) are the production ones; only the
    sizes and the discount are retuned for a 200-step toy task.

    Args:
        seed: Master seed.

    Returns:
        A validated :class:`PPOConfig`.
    """
    return PPOConfig(
        num_envs=NUM_ENVS,
        num_steps=NUM_STEPS,
        num_minibatches=8,
        update_epochs=10,
        gamma=0.9,
        gae_lambda=0.95,
        clip_coef=0.2,
        vf_coef=1.0,
        ent_coef=0.0,
        learning_rate=1e-3,
        kl_adaptive=True,
        total_timesteps=TOTAL_STEPS,
        seed=seed,
        device="cpu",
        network=NetworkConfig(
            use_image=False,
            vec_dim=OBS_DIM,
            priv_dim=OBS_DIM,
            act_dim=ACT_DIM,
            hidden_dim=64,
        ),
    )


def train_pendulum(seed: int, total_steps: int = TOTAL_STEPS, verbose: bool = False) -> dict[str, float]:
    """Train PPO on Pendulum-v1 and report the final performance.

    Args:
        seed: Master seed for torch, numpy and the environments.
        total_steps: Environment-step budget.
        verbose: Print a progress line per iteration.

    Returns:
        Dict with ``final_return`` (mean of the last 50 completed episodes), ``initial_return``
        (mean of the first 20), ``explained_variance`` (median over the last five iterations, so a
        single learning-rate excursion cannot decide the gate), ``mean_sigma``, ``episodes``,
        ``truncations`` and ``seconds``.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - seeds the legacy global stream the checkpoint snapshots
    cfg = _make_config(seed)
    agent = ActorCritic(cfg.network)
    learner = PPO(agent, cfg, device="cpu")
    buffer = RolloutBuffer(
        num_steps=cfg.num_steps,
        num_envs=cfg.num_envs,
        vec_dim=OBS_DIM,
        priv_dim=OBS_DIM,
        act_dim=ACT_DIM,
        obs_shape=None,
        device="cpu",
        terminal_capacity=64,
    )

    envs = ExplicitVectorEnv(ENV_ID, cfg.num_envs, seed)
    obs = torch.from_numpy(envs.reset())
    truncations = 0
    diagnostics: dict[str, float] = {}
    explained_history: list[float] = []
    start = time.perf_counter()

    num_iterations = total_steps // (cfg.num_envs * cfg.num_steps)
    for iteration in range(num_iterations):
        buffer.reset()
        for _ in range(cfg.num_steps):
            step_out = learner.act(None, obs)
            env_action = (step_out["clipped_action"] * ACTION_SCALE).numpy()
            next_obs, final_obs, rewards, terminated, truncated = envs.step(env_action)

            done = np.logical_or(terminated, truncated)
            if done.any():
                env_ids = torch.from_numpy(np.nonzero(done)[0]).long()
                # Capture BEFORE the observation is replaced by the reset one, exactly as
                # DuckiebotLaneFollowEnv._reset_idx does (SPEC v2 S6.4).
                buffer.capture_terminal(
                    env_ids=env_ids,
                    vec_priv=torch.from_numpy(final_obs[done]),
                )
                truncations += int(truncated.sum())

            buffer.add(
                vec=obs,
                vec_priv=obs,
                action=step_out["action"],
                log_prob=step_out["log_prob"],
                value=step_out["value"],
                reward=torch.from_numpy(rewards),
                terminated=torch.from_numpy(terminated),
                truncated=torch.from_numpy(truncated),
                mu=step_out["mu"],
                log_std=step_out["log_std"],
            )
            obs = torch.from_numpy(next_obs)

        learner.compute_returns(buffer, None, obs)
        diagnostics = learner.update(buffer)
        explained_history.append(diagnostics["explained_variance"])

        if verbose:
            recent = envs.completed_returns[-FINAL_WINDOW:]
            print(
                f"iter {iteration:3d} steps {(iteration + 1) * cfg.batch_size:7d} "
                f"return {np.mean(recent) if recent else float('nan'):8.1f} "
                f"ev {diagnostics['explained_variance']:6.3f} "
                f"kl {diagnostics['analytic_kl']:.4f} "
                f"lr {diagnostics['learning_rate']:.2e} "
                f"sigma {diagnostics['mean_sigma']:.3f}"
            )

    envs.close()
    recent = envs.completed_returns[-FINAL_WINDOW:]
    early = envs.completed_returns[:INITIAL_WINDOW]
    return {
        "final_return": float(np.mean(recent)) if recent else float("nan"),
        "initial_return": float(np.mean(early)) if early else float("nan"),
        "explained_variance": float(np.median(explained_history[-5:])) if explained_history else -1.0,
        "mean_sigma": diagnostics.get("mean_sigma", float("nan")),
        "episodes": float(len(envs.completed_returns)),
        "truncations": float(truncations),
        "seconds": time.perf_counter() - start,
    }


@pytest.mark.slow
@pytest.mark.parametrize("seed", SEEDS)
def test_ppo_learns_pendulum(seed: int) -> None:
    """PPO reaches the return threshold on Pendulum-v1 from every seed.

    Args:
        seed: Master seed supplied by the parametrisation.
    """
    result = train_pendulum(seed)
    assert result["episodes"] > 100, "the run produced too few episodes to judge"
    assert result["truncations"] > 100, "Pendulum should truncate constantly; terminal capture idle"
    assert result["final_return"] > RETURN_THRESHOLD, (
        f"seed {seed}: mean return over the last 20 episodes was {result['final_return']:.1f}, "
        f"expected > {RETURN_THRESHOLD}. Initial was {result['initial_return']:.1f}."
    )
    assert result["final_return"] > result["initial_return"] + 300.0, "no measurable improvement"
    assert result["explained_variance"] > 0.5, (
        f"critic is not fitting: explained_variance {result['explained_variance']:.3f}"
    )
    assert result["mean_sigma"] < 0.5, "sigma never contracted, the policy did not commit"


if __name__ == "__main__":  # pragma: no cover - manual tuning entry point
    for _seed in SEEDS:
        print(_seed, train_pendulum(_seed, verbose=True))
