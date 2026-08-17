"""Preallocated GPU rollout buffer with terminal-observation capture (SPEC v2 S6.3).

Layout is time-major ``(T, N, ...)``, all tensors preallocated once on the training device so the
rollout loop performs no allocation and no host synchronisation.

The image field stores the FULL STACKED observation as uint8, not single frames. SPEC v2 decision
4 makes this explicit: reassembling a ``(t, t-2, t-4)`` stack at minibatch time is impossible for
``t < 4`` of each rollout, and any clamping there makes the reconstructed observation differ from
the one the policy actually acted on, which breaks the epoch-0 importance ratio. At the v2
resolution the stacked store costs ``32 * 256 * 48 * 96 * 9`` bytes = 324 MiB, which the S5.6
budget carries.

Terminal capture. When an episode ends the environment must hand the learner the TRUE terminal
observation (built from the pre-reset frame ring, with the same D9 observation delay) and the
pre-reset ``vec_priv``. Those live in a sparse :class:`TerminalCache` rather than in a dense
``(T, N, ...)`` tensor, because a second dense image field would cost another 324 MiB to hold a
few hundred entries. At update time one no-grad critic pass over the cache produces ``term_value``
which GAE consumes at exactly the ``(t, env)`` slots that were captured.

Ordering contract for the rollout loop, which the capture indexing depends on:

.. code-block:: text

    action, logp, value = ppo.act(obs)          # obs is what the policy sees at step t
    next_obs, reward, terminated, truncated = env.step(action)
        # inside env.step, _reset_idx fires and calls buffer.capture_terminal(...)
        # with the default step index, which is the slot about to be written
    buffer.add(obs, ..., reward, terminated, truncated)

That is, ``capture_terminal`` is called BEFORE the matching ``add``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import torch


class TerminalCache:
    """Sparse store of true terminal observations awaiting a critic evaluation.

    Args:
        obs_shape: NHWC image observation shape ``(H, W, C)``, or None in vec-only mode.
        priv_dim: Width of the privileged vector observation.
        device: Torch device.
        capacity: Initial number of slots. Grows by doubling when exceeded, which is rare
            (an iteration sees roughly ``T * N / episode_length`` resets).
        image_dtype: Dtype of the stored image, uint8 in production.
    """

    def __init__(
        self,
        obs_shape: tuple[int, int, int] | None,
        priv_dim: int,
        device: torch.device | str = "cpu",
        capacity: int = 256,
        image_dtype: torch.dtype = torch.uint8,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.obs_shape = obs_shape
        self.priv_dim = priv_dim
        self.device = torch.device(device)
        self.image_dtype = image_dtype
        self._capacity = capacity
        self.count = 0

        self.step_index = torch.zeros(capacity, dtype=torch.long, device=self.device)
        self.env_index = torch.zeros(capacity, dtype=torch.long, device=self.device)
        self.vec_priv = torch.zeros(capacity, priv_dim, dtype=torch.float32, device=self.device)
        self.image: torch.Tensor | None = (
            torch.zeros(capacity, *obs_shape, dtype=image_dtype, device=self.device)
            if obs_shape is not None
            else None
        )

    @property
    def capacity(self) -> int:
        """Return the current number of allocated slots."""
        return self._capacity

    def clear(self) -> None:
        """Drop every cached entry without freeing the allocation."""
        self.count = 0

    def _grow(self, required: int) -> None:
        """Double the allocation until it holds ``required`` entries.

        Args:
            required: Number of entries that must fit.
        """
        new_capacity = self._capacity
        while new_capacity < required:
            new_capacity *= 2
        extra = new_capacity - self._capacity
        pad_long = torch.zeros(extra, dtype=torch.long, device=self.device)
        self.step_index = torch.cat([self.step_index, pad_long])
        self.env_index = torch.cat([self.env_index, pad_long.clone()])
        pad_priv = torch.zeros(extra, self.priv_dim, dtype=torch.float32, device=self.device)
        self.vec_priv = torch.cat([self.vec_priv, pad_priv])
        if self.image is not None and self.obs_shape is not None:
            pad_image = torch.zeros(extra, *self.obs_shape, dtype=self.image_dtype, device=self.device)
            self.image = torch.cat([self.image, pad_image])
        self._capacity = new_capacity

    @torch.no_grad()
    def add(
        self,
        step: int,
        env_ids: torch.Tensor,
        vec_priv: torch.Tensor,
        image: torch.Tensor | None = None,
    ) -> None:
        """Record the terminal observations of a batch of resetting environments.

        Args:
            step: Rollout time index the terminal transition belongs to.
            env_ids: ``(K,)`` long tensor of environment indices that are resetting.
            vec_priv: ``(K, priv_dim)`` pre-reset privileged observation.
            image: ``(K, H, W, C)`` pre-reset stacked observation, or None in vec-only mode.

        Raises:
            ValueError: If shapes disagree, or if an image is expected but not supplied.
        """
        num = int(env_ids.numel())
        if num == 0:
            return
        if vec_priv.shape[0] != num or vec_priv.shape[1] != self.priv_dim:
            raise ValueError(
                f"vec_priv must have shape ({num}, {self.priv_dim}), got {tuple(vec_priv.shape)}"
            )
        if self.image is not None:
            if image is None:
                raise ValueError("this cache stores images but image is None")
            if tuple(image.shape) != (num, *self.obs_shape):  # type: ignore[misc]
                raise ValueError(
                    f"image must have shape {(num, *self.obs_shape)}, got {tuple(image.shape)}"  # type: ignore[misc]
                )
        if self.count + num > self._capacity:
            self._grow(self.count + num)

        lo, hi = self.count, self.count + num
        self.step_index[lo:hi] = step
        self.env_index[lo:hi] = env_ids.to(device=self.device, dtype=torch.long)
        # copy_ (never aliasing assignment): the caller's tensor may be a live camera buffer.
        self.vec_priv[lo:hi].copy_(vec_priv)
        if self.image is not None and image is not None:
            self.image[lo:hi].copy_(image)
        self.count = hi

    def entries(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Return views over the filled portion of the cache.

        Returns:
            Tuple ``(step_index, env_index, vec_priv, image)``, each sliced to ``count`` rows.
            ``image`` is None in vec-only mode.
        """
        n = self.count
        return (
            self.step_index[:n],
            self.env_index[:n],
            self.vec_priv[:n],
            self.image[:n] if self.image is not None else None,
        )


class RolloutBuffer:
    """Time-major on-device rollout storage for one PPO iteration.

    Args:
        num_steps: Rollout length ``T``.
        num_envs: Parallel environment count ``N``.
        vec_dim: Actor vector-observation width.
        priv_dim: Critic privileged vector-observation width.
        act_dim: Action dimensionality.
        obs_shape: NHWC image observation shape ``(H, W, C)``, or None in vec-only mode.
        device: Torch device holding every tensor.
        image_dtype: Image storage dtype; uint8 in production.
        terminal_capacity: Initial size of the terminal cache.

    Attributes:
        pos: Index of the next slot to be written.
        full: True once ``num_steps`` transitions have been written since the last reset.
    """

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        vec_dim: int,
        priv_dim: int,
        act_dim: int,
        obs_shape: tuple[int, int, int] | None = None,
        device: torch.device | str = "cpu",
        image_dtype: torch.dtype = torch.uint8,
        terminal_capacity: int = 256,
    ) -> None:
        if num_steps <= 0 or num_envs <= 0:
            raise ValueError(f"num_steps and num_envs must be positive, got ({num_steps}, {num_envs})")
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.vec_dim = vec_dim
        self.priv_dim = priv_dim
        self.act_dim = act_dim
        self.obs_shape = obs_shape
        self.device = torch.device(device)
        self.image_dtype = image_dtype

        shape_tn = (num_steps, num_envs)
        f32 = torch.float32
        self.images: torch.Tensor | None = (
            torch.zeros(*shape_tn, *obs_shape, dtype=image_dtype, device=self.device)
            if obs_shape is not None
            else None
        )
        self.vec = torch.zeros(*shape_tn, vec_dim, dtype=f32, device=self.device)
        self.vec_priv = torch.zeros(*shape_tn, priv_dim, dtype=f32, device=self.device)
        self.actions = torch.zeros(*shape_tn, act_dim, dtype=f32, device=self.device)
        self.mu = torch.zeros(*shape_tn, act_dim, dtype=f32, device=self.device)
        self.log_std = torch.zeros(*shape_tn, act_dim, dtype=f32, device=self.device)
        self.log_probs = torch.zeros(*shape_tn, dtype=f32, device=self.device)
        self.values = torch.zeros(*shape_tn, dtype=f32, device=self.device)
        self.rewards = torch.zeros(*shape_tn, dtype=f32, device=self.device)
        self.terminated = torch.zeros(*shape_tn, dtype=torch.bool, device=self.device)
        self.truncated = torch.zeros(*shape_tn, dtype=torch.bool, device=self.device)
        self.term_values = torch.zeros(*shape_tn, dtype=f32, device=self.device)
        self.advantages = torch.zeros(*shape_tn, dtype=f32, device=self.device)
        self.returns = torch.zeros(*shape_tn, dtype=f32, device=self.device)

        self.terminal_cache = TerminalCache(
            obs_shape=obs_shape,
            priv_dim=priv_dim,
            device=self.device,
            capacity=terminal_capacity,
            image_dtype=image_dtype,
        )
        self.pos = 0
        self.full = False

    @property
    def batch_size(self) -> int:
        """Return ``num_steps * num_envs``."""
        return self.num_steps * self.num_envs

    @property
    def current_step(self) -> int:
        """Return the slot index that the next :meth:`add` will write.

        This is also the step index a terminal capture belongs to, because the environment's reset
        happens inside the ``env.step`` call whose result is written into that slot.
        """
        return self.pos

    def reset(self) -> None:
        """Rewind the write pointer and drop the terminal cache and the derived tensors."""
        self.pos = 0
        self.full = False
        self.terminal_cache.clear()
        self.term_values.zero_()
        self.advantages.zero_()
        self.returns.zero_()

    @torch.no_grad()
    def add(
        self,
        vec: torch.Tensor,
        vec_priv: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        mu: torch.Tensor,
        log_std: torch.Tensor,
        image: torch.Tensor | None = None,
    ) -> None:
        """Write one transition for all ``N`` environments.

        Every field is written with ``copy_`` rather than by assignment: the image argument may be
        a view of the live tiled-camera output buffer, and assigning it would alias a tensor that
        the renderer overwrites on the next step (SPEC v2 S6.7 guard 3).

        The write pointer wraps: writing ``num_steps + 1`` transitions overwrites slot 0 and keeps
        ``full`` True. Production code calls :meth:`reset` after each update, so wraparound only
        appears if the caller loses track of the rollout length.

        Args:
            vec: ``(N, vec_dim)`` raw actor vector observation.
            vec_priv: ``(N, priv_dim)`` raw privileged vector observation.
            action: ``(N, act_dim)`` UNCLIPPED Gaussian sample.
            log_prob: ``(N,)`` log density of ``action`` under the acting policy.
            value: ``(N,)`` critic estimate in real (reward) scale.
            reward: ``(N,)`` reward produced by this transition.
            terminated: ``(N,)`` bool; the MDP truly ended.
            truncated: ``(N,)`` bool; the time limit fired.
            mu: ``(N, act_dim)`` acting-policy mean, snapshotted for the analytic KL.
            log_std: ``(N, act_dim)`` acting-policy log std, snapshotted for the analytic KL.
            image: ``(N, H, W, C)`` STACKED observation, or None in vec-only mode.

        Raises:
            ValueError: If an image is expected but not supplied, or vice versa.
        """
        t = self.pos
        if self.images is not None:
            if image is None:
                raise ValueError("this buffer stores images but image is None")
            self.images[t].copy_(image)
        elif image is not None:
            raise ValueError("this buffer is in vec-only mode but an image was supplied")

        self.vec[t].copy_(vec)
        self.vec_priv[t].copy_(vec_priv)
        self.actions[t].copy_(action)
        self.log_probs[t].copy_(log_prob)
        self.values[t].copy_(value)
        self.rewards[t].copy_(reward)
        self.terminated[t].copy_(terminated)
        self.truncated[t].copy_(truncated)
        self.mu[t].copy_(mu)
        self.log_std[t].copy_(log_std)

        self.pos += 1
        if self.pos >= self.num_steps:
            self.pos = 0
            self.full = True

    @torch.no_grad()
    def capture_terminal(
        self,
        env_ids: torch.Tensor,
        vec_priv: torch.Tensor,
        image: torch.Tensor | None = None,
        step: int | None = None,
    ) -> None:
        """Record true terminal observations for environments that are about to reset.

        Args:
            env_ids: ``(K,)`` indices of the resetting environments.
            vec_priv: ``(K, priv_dim)`` privileged observation computed from PRE-reset state.
            image: ``(K, H, W, C)`` stacked observation built from the pre-reset frame ring, with
                the same D9 observation delay ``_get_observations`` would apply.
            step: Rollout index this terminal belongs to. Defaults to :attr:`current_step`, which
                is correct when the capture happens inside the ``env.step`` whose result is
                written next.
        """
        self.terminal_cache.add(
            step=self.current_step if step is None else step,
            env_ids=env_ids,
            vec_priv=vec_priv,
            image=image,
        )

    @torch.no_grad()
    def compute_terminal_values(
        self,
        value_fn: Callable[[torch.Tensor | None, torch.Tensor], torch.Tensor],
    ) -> int:
        """Evaluate the critic on the terminal cache and scatter the results into ``term_values``.

        Args:
            value_fn: Callable ``(image, vec_priv) -> (K,)`` returning REAL-scale values. In
                practice this is ``PPO.predict_values``.

        Returns:
            Number of cached terminals that were evaluated.
        """
        self.term_values.zero_()
        step_index, env_index, vec_priv, image = self.terminal_cache.entries()
        if step_index.numel() == 0:
            return 0
        values = value_fn(image, vec_priv).to(dtype=self.term_values.dtype)
        self.term_values[step_index, env_index] = values
        return int(step_index.numel())

    @torch.no_grad()
    def set_advantages(self, advantages: torch.Tensor, returns: torch.Tensor) -> None:
        """Store the GAE outputs.

        Args:
            advantages: ``(T, N)`` advantages.
            returns: ``(T, N)`` value targets.
        """
        self.advantages.copy_(advantages)
        self.returns.copy_(returns)

    def flat(self) -> dict[str, torch.Tensor]:
        """Return every stored field flattened to ``(T * N, ...)``.

        ``reshape`` is used rather than ``view`` so the call is safe for any storage layout.

        Returns:
            Dict of flattened tensors. The ``image`` key is absent in vec-only mode.
        """
        batch = self.batch_size
        out = {
            "vec": self.vec.reshape(batch, self.vec_dim),
            "vec_priv": self.vec_priv.reshape(batch, self.priv_dim),
            "actions": self.actions.reshape(batch, self.act_dim),
            "mu": self.mu.reshape(batch, self.act_dim),
            "log_std": self.log_std.reshape(batch, self.act_dim),
            "log_probs": self.log_probs.reshape(batch),
            "values": self.values.reshape(batch),
            "rewards": self.rewards.reshape(batch),
            "advantages": self.advantages.reshape(batch),
            "returns": self.returns.reshape(batch),
            "terminated": self.terminated.reshape(batch),
            "truncated": self.truncated.reshape(batch),
        }
        if self.images is not None and self.obs_shape is not None:
            out["image"] = self.images.reshape(batch, *self.obs_shape)
        return out

    def minibatches(
        self,
        num_minibatches: int,
        generator: torch.Generator | None = None,
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Yield ``num_minibatches`` disjoint random minibatches covering the whole batch once.

        The permutation is drawn with ``torch.randperm`` on the buffer's device so the shuffle
        never forces a host synchronisation (SPEC v2 S6.7 guard 5).

        Args:
            num_minibatches: Number of minibatches per epoch.
            generator: Optional torch generator for reproducible shuffling.

        Yields:
            Dicts of index-selected tensors with the same keys as :meth:`flat`.

        Raises:
            ValueError: If the batch is not divisible by ``num_minibatches``.
        """
        batch = self.batch_size
        if batch % num_minibatches != 0:
            raise ValueError(f"batch_size {batch} is not divisible by num_minibatches {num_minibatches}")
        size = batch // num_minibatches
        flat = self.flat()
        perm = torch.randperm(batch, device=self.device, generator=generator)
        for start in range(0, batch, size):
            idx = perm[start : start + size]
            yield {key: tensor[idx] for key, tensor in flat.items()}
