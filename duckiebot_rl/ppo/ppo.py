"""The PPO learner: losses, KL-adaptive learning rate, minibatch loop, diagnostics.

SPEC v2 S6.5 fixes every choice made here:

* clipped surrogate implemented as ``max`` of the two negatives; the ratio comes from
  ``logratio.exp()``, never from a probability division;
* plain MSE value loss, no value clipping (Andrychowicz et al. swept the clip threshold and found
  ``None`` won); the clipping path is kept behind ``clip_vloss`` purely as a reportable ablation;
* entropy coefficient 0.0, because a state-independent Gaussian ``log_std`` has unbounded entropy
  and any bonus inflates sigma monotonically; exploration comes from ``sigma_init = 0.5``;
* bounds loss ``1e-4 * mean(relu(|mu| - 1) ** 2)`` keeps the mean inside the action box, which is
  the known failure mode of raw-Normal-plus-clipping;
* global gradient-norm clip 1.0, with the PRE-clip norm logged;
* Adam(3e-4, eps 1e-5) with a KL-adaptive learning rate driven by the EXACT analytic diagonal
  Gaussian KL against the ``(mu, log_std)`` snapshot stored in the buffer: above 0.02 divide by
  1.5, below 0.005 multiply by 1.5, clamped to ``[1e-5, 1e-2]``. No annealing, no KL early stop.
* fp32 end to end; see :func:`duckiebot_rl.ppo.config.configure_precision`.

Normaliser update timing is load-bearing and asymmetric:

* the VECTOR normalisers are refreshed at the END of :meth:`PPO.update`, so the statistics that
  produced the rollout are byte-identical to the ones used inside the update. If they moved in
  between, the observations fed to the network during the update would differ from the ones the
  policy acted on and the epoch-0 importance ratio would not be 1;
* the VALUE normaliser is refreshed at the START of :meth:`PPO.update`, so the critic is trained
  against targets in exactly the scale that the next rollout will denormalise with.

Diagnostics are accumulated as device tensors and synchronised to the host exactly once at the end
of the update (SPEC v2 S6.7 guard 5). The two deliberate exceptions are the epoch-0 ratio assert
(once per update, guard 1) and the KL-adaptive learning-rate controller when
``kl_adapt_per_minibatch`` is enabled.
"""

from __future__ import annotations

import math
from collections import defaultdict

import torch
import torch.nn as nn

from duckiebot_rl.ppo.buffer import RolloutBuffer
from duckiebot_rl.ppo.config import PPOConfig, ratio_assert_atol
from duckiebot_rl.ppo.distributions import approx_kl_k3, diag_gaussian_kl
from duckiebot_rl.ppo.gae import compute_gae
from duckiebot_rl.ppo.networks import ActorCritic
from duckiebot_rl.ppo.running_norm import RunningMeanStd


class RatioAssertionError(AssertionError):
    """Raised when the epoch-0 minibatch-0 importance ratio is not 1.

    SPEC v2 S6.7 guard 1. A failure means one of: the clipped (rather than raw) action was stored,
    ``log_prob`` was summed over the wrong axis, the observation fed to the update differs from the
    one the policy acted on (stale normaliser, frame-stack reassembly, mutated buffer), a
    stochastic layer such as dropout is active, or autocast is on.
    """


class PPO:
    """Proximal Policy Optimisation learner.

    Args:
        agent: The actor-critic module. Moved to ``device``.
        cfg: Hyperparameters.
        device: Torch device; defaults to ``cfg.device``.

    Attributes:
        agent: The actor-critic module.
        cfg: Hyperparameters.
        optimizer: Adam over every agent parameter.
        learning_rate: Current learning rate, mutated by the KL controller.
        vec_norm: Running normaliser for the actor vector observation.
        priv_norm: Running normaliser for the privileged vector observation.
        value_norm: Running normaliser for the value targets.
        num_updates: Number of completed calls to :meth:`update`.
    """

    def __init__(self, agent: ActorCritic, cfg: PPOConfig, device: torch.device | str | None = None) -> None:
        self.cfg = cfg
        self.device = torch.device(device if device is not None else cfg.device)
        self.agent = agent.to(self.device)
        self.learning_rate = cfg.learning_rate
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=cfg.learning_rate, eps=cfg.adam_eps)
        net = cfg.network
        self.vec_norm = RunningMeanStd((net.vec_dim,), device=self.device)
        self.priv_norm = RunningMeanStd((net.priv_dim,), device=self.device)
        self.value_norm = RunningMeanStd((), device=self.device)
        self.num_updates = 0

    # ---------------------------------------------------------------- helpers

    def set_learning_rate(self, learning_rate: float) -> None:
        """Set the optimiser learning rate, clamped to the configured bounds.

        Args:
            learning_rate: Requested learning rate.
        """
        self.learning_rate = float(min(max(learning_rate, self.cfg.lr_min), self.cfg.lr_max))
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate

    def normalize_vec(self, vec: torch.Tensor) -> torch.Tensor:
        """Normalise the actor vector observation with the frozen running statistics.

        Args:
            vec: ``(B, vec_dim)`` raw vector observation.

        Returns:
            Normalised and clipped tensor of the same shape.
        """
        if not self.cfg.normalize_vec:
            return vec
        return self.vec_norm.normalize(vec, clip=self.cfg.vec_clip)

    def normalize_priv(self, vec_priv: torch.Tensor) -> torch.Tensor:
        """Normalise the privileged vector observation with the frozen running statistics.

        Args:
            vec_priv: ``(B, priv_dim)`` raw privileged observation.

        Returns:
            Normalised and clipped tensor of the same shape.
        """
        if not self.cfg.normalize_vec:
            return vec_priv
        return self.priv_norm.normalize(vec_priv, clip=self.cfg.vec_clip)

    def _real_value(self, raw_value: torch.Tensor) -> torch.Tensor:
        """Map a raw critic output back to reward scale.

        Args:
            raw_value: Critic output.

        Returns:
            Value in the same scale as the rewards.
        """
        return self.value_norm.denormalize(raw_value) if self.cfg.normalize_value_targets else raw_value

    # ------------------------------------------------------------ rollout API

    @torch.no_grad()
    def act(
        self,
        image: torch.Tensor | None,
        vec: torch.Tensor,
        vec_priv: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Sample an action and estimate its value for one environment step.

        Args:
            image: ``(N, H, W, C)`` stacked uint8 observation, or None in vec-only mode.
            vec: ``(N, vec_dim)`` RAW actor vector observation (normalisation happens here).
            vec_priv: ``(N, priv_dim)`` RAW privileged observation; defaults to ``vec`` when the
                two widths agree.
            deterministic: Return the policy mean instead of a sample (evaluation path).

        Returns:
            Dict with keys ``action`` (unclipped, store this in the buffer), ``clipped_action``
            (what the environment should execute), ``log_prob``, ``value`` (real scale), ``mu``
            and ``log_std``.
        """
        self.agent.eval()
        priv = vec if vec_priv is None else vec_priv
        out = self.agent.get_action_and_value(
            image,
            self.normalize_vec(vec),
            self.normalize_priv(priv),
            deterministic=deterministic,
        )
        return {
            "action": out.action,
            "clipped_action": out.action.clamp(-1.0, 1.0),
            "log_prob": out.log_prob,
            "value": self._real_value(out.value),
            "mu": out.mu,
            "log_std": out.log_std,
        }

    @torch.no_grad()
    def predict_values(self, image: torch.Tensor | None, vec_priv: torch.Tensor) -> torch.Tensor:
        """Evaluate the critic in real (reward) scale.

        Used for the last-step bootstrap ``V(obs[T])`` and for the captured terminal observations.

        Args:
            image: ``(B, H, W, C)`` stacked uint8 observation, or None in vec-only mode.
            vec_priv: ``(B, priv_dim)`` RAW privileged observation.

        Returns:
            ``(B,)`` values in reward scale.
        """
        self.agent.eval()
        raw = self.agent.get_value(image, self.normalize_priv(vec_priv))
        return self._real_value(raw)

    @torch.no_grad()
    def compute_returns(
        self,
        buffer: RolloutBuffer,
        last_image: torch.Tensor | None,
        last_vec_priv: torch.Tensor,
    ) -> int:
        """Run the terminal-value pass, the last-step bootstrap and GAE, then fill the buffer.

        Args:
            buffer: The filled rollout buffer.
            last_image: The ``T + 1``-th stacked observation the environment has already produced,
                or None in vec-only mode.
            last_vec_priv: The ``T + 1``-th raw privileged observation.

        Returns:
            Number of captured terminal observations that were evaluated.
        """
        num_terminals = buffer.compute_terminal_values(self.predict_values)
        last_values = self.predict_values(last_image, last_vec_priv)
        advantages, returns = compute_gae(
            rewards=buffer.rewards,
            values=buffer.values,
            terminated=buffer.terminated,
            truncated=buffer.truncated,
            last_values=last_values,
            term_values=buffer.term_values,
            gamma=self.cfg.gamma,
            lam=self.cfg.gae_lambda,
            rsl_rl_approx=self.cfg.rsl_rl_gae_approx,
        )
        buffer.set_advantages(advantages, returns)
        return num_terminals

    # -------------------------------------------------------------- update

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        """Run ``update_epochs`` passes of minibatch PPO over one rollout.

        :meth:`compute_returns` must have been called first.

        Args:
            buffer: The filled rollout buffer with advantages and returns populated.

        Returns:
            Dict of scalar diagnostics: ``policy_loss``, ``value_loss``, ``entropy``,
            ``bounds_loss``, ``approx_kl`` (k3), ``analytic_kl``, ``clipfrac``, ``ratio_mean``,
            ``grad_norm`` (pre-clip), ``mean_sigma``, ``mean_abs_mu``, ``explained_variance``,
            ``learning_rate``, ``advantage_mean``, ``advantage_std``, ``value_target_mean``.

        Raises:
            RatioAssertionError: If the epoch-0 minibatch-0 importance ratio is not 1 within the
                precision-dependent tolerance.
        """
        cfg = self.cfg
        flat = buffer.flat()
        images = flat.get("image")

        # Value normaliser first, so the critic trains in the scale the next rollout will use.
        if cfg.normalize_value_targets:
            self.value_norm.update(flat["returns"])
            value_targets = self.value_norm.normalize(flat["returns"])
            old_values_norm = self.value_norm.normalize(flat["values"])
        else:
            value_targets = flat["returns"]
            old_values_norm = flat["values"]

        advantages = flat["advantages"]
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        if cfg.norm_adv:
            # Batch level, exactly once. Never also per minibatch.
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)

        vec_norm = self.normalize_vec(flat["vec"])
        priv_norm = self.normalize_priv(flat["vec_priv"])

        batch = buffer.batch_size
        minibatch_size = batch // cfg.num_minibatches
        atol = ratio_assert_atol()
        stats: dict[str, list[torch.Tensor]] = defaultdict(list)
        self.agent.train()

        for epoch in range(cfg.update_epochs):
            perm = torch.randperm(batch, device=self.device)
            for mb_index in range(cfg.num_minibatches):
                idx = perm[mb_index * minibatch_size : (mb_index + 1) * minibatch_size]
                mb_image = images[idx] if images is not None else None
                out = self.agent.get_action_and_value(
                    mb_image,
                    vec_norm[idx],
                    priv_norm[idx],
                    action=flat["actions"][idx],
                )

                log_ratio = out.log_prob - flat["log_probs"][idx]
                ratio = log_ratio.exp()

                if epoch == 0 and mb_index == 0 and cfg.ratio_assert:
                    max_dev = (ratio.detach() - 1.0).abs().max().item()
                    if not math.isfinite(max_dev) or max_dev > atol:
                        raise RatioAssertionError(
                            f"epoch-0 minibatch-0 importance ratio deviates by {max_dev:.3e} "
                            f"(atol {atol:.1e}). See SPEC v2 S6.7 guard 1 for the candidate causes."
                        )

                mb_adv = advantages[idx]
                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * ratio.clamp(1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef),
                ).mean()

                if cfg.clip_vloss:
                    mb_old_v = old_values_norm[idx]
                    v_clipped = mb_old_v + (out.value - mb_old_v).clamp(-cfg.clip_coef, cfg.clip_coef)
                    value_loss = torch.max(
                        (out.value - value_targets[idx]).pow(2),
                        (v_clipped - value_targets[idx]).pow(2),
                    ).mean()
                else:
                    value_loss = (out.value - value_targets[idx]).pow(2).mean()

                entropy = out.entropy.mean()
                bounds_loss = torch.relu(out.mu.abs() - 1.0).pow(2).mean()
                loss = (
                    pg_loss
                    + cfg.vf_coef * value_loss
                    - cfg.ent_coef * entropy
                    + cfg.bounds_coef * bounds_loss
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.agent.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    analytic_kl = diag_gaussian_kl(
                        flat["mu"][idx], flat["log_std"][idx], out.mu, out.log_std
                    ).mean()
                    stats["policy_loss"].append(pg_loss.detach())
                    stats["value_loss"].append(value_loss.detach())
                    stats["entropy"].append(entropy.detach())
                    stats["bounds_loss"].append(bounds_loss.detach())
                    stats["approx_kl"].append(approx_kl_k3(log_ratio.detach()).mean())
                    stats["analytic_kl"].append(analytic_kl)
                    stats["clipfrac"].append(((ratio.detach() - 1.0).abs() > cfg.clip_coef).float().mean())
                    stats["ratio_mean"].append(ratio.detach().mean())
                    stats["grad_norm"].append(grad_norm.detach())
                    stats["mean_sigma"].append(out.log_std.detach().exp().mean())
                    stats["mean_abs_mu"].append(out.mu.detach().abs().mean())

                if cfg.kl_adaptive and cfg.kl_adapt_per_minibatch:
                    self._adapt_learning_rate(float(analytic_kl.item()))

        with torch.no_grad():
            keys = sorted(stats)
            means = torch.stack([torch.stack(stats[key]).mean() for key in keys])
            values_real = flat["values"]
            returns_real = flat["returns"]
            var_returns = returns_real.var()
            explained = torch.where(
                var_returns > 0,
                1.0 - (returns_real - values_real).var() / (var_returns + 1e-8),
                torch.zeros_like(var_returns),
            )
            tail = torch.stack([explained, adv_mean, adv_std, returns_real.mean()])
            # One host synchronisation for every diagnostic in the update.
            merged = torch.cat([means, tail]).cpu().tolist()

        out_stats = dict(zip(keys, merged[: len(keys)], strict=True))
        out_stats["explained_variance"] = merged[len(keys)]
        out_stats["advantage_mean"] = merged[len(keys) + 1]
        out_stats["advantage_std"] = merged[len(keys) + 2]
        out_stats["value_target_mean"] = merged[len(keys) + 3]

        if self.cfg.kl_adaptive and not self.cfg.kl_adapt_per_minibatch:
            self._adapt_learning_rate(out_stats["analytic_kl"])
        out_stats["learning_rate"] = self.learning_rate

        # Vector normalisers last: the statistics must not move between acting and updating.
        if cfg.normalize_vec:
            self.vec_norm.update(flat["vec"])
            self.priv_norm.update(flat["vec_priv"])
        self.num_updates += 1
        return out_stats

    def _adapt_learning_rate(self, kl: float) -> None:
        """Apply one step of the KL-adaptive learning-rate controller.

        Args:
            kl: Mean exact analytic ``KL(old || new)`` for the sample this step is based on.
        """
        cfg = self.cfg
        if not math.isfinite(kl):
            return
        if kl > cfg.kl_target_upper:
            self.set_learning_rate(self.learning_rate / cfg.lr_factor)
        elif kl < cfg.kl_target_lower:
            self.set_learning_rate(self.learning_rate * cfg.lr_factor)

    # ---------------------------------------------------------- checkpointing

    def state_dict(self) -> dict[str, object]:
        """Return every piece of learner state that a resume must restore.

        Returns:
            Dict with the model, optimiser, learning rate, update counter and the three running
            normalisers.
        """
        return {
            "model": self.agent.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "learning_rate": self.learning_rate,
            "num_updates": self.num_updates,
            "vec_norm": self.vec_norm.state_dict(),
            "priv_norm": self.priv_norm.state_dict(),
            "value_norm": self.value_norm.state_dict(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore learner state produced by :meth:`state_dict`.

        Args:
            state: The dict returned by :meth:`state_dict`.

        Raises:
            KeyError: If a mandatory field is missing.
        """
        for key in ("model", "optimizer", "learning_rate", "vec_norm", "priv_norm", "value_norm"):
            if key not in state:
                raise KeyError(f"learner state is missing mandatory key {key!r}")
        self.agent.load_state_dict(state["model"])  # type: ignore[arg-type]
        self.optimizer.load_state_dict(state["optimizer"])  # type: ignore[arg-type]
        self.vec_norm.load_state_dict(state["vec_norm"])  # type: ignore[arg-type]
        self.priv_norm.load_state_dict(state["priv_norm"])  # type: ignore[arg-type]
        self.value_norm.load_state_dict(state["value_norm"])  # type: ignore[arg-type]
        self.num_updates = int(state.get("num_updates", 0))  # type: ignore[arg-type]
        self.set_learning_rate(float(state["learning_rate"]))  # type: ignore[arg-type]
