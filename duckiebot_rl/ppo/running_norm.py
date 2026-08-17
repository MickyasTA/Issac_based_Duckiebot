"""Running mean/std normaliser for vector observations and value targets (SPEC v2 S6.6).

Three instances live in the learner: one for ``vec`` (8 dims), one for ``vec_priv`` (14 dims) and
one scalar instance for the value targets. Pixels are never normalised here, only divided by 255
inside the encoder.

Update timing matters and is deliberate (see :mod:`duckiebot_rl.ppo.ppo`):

* the vector normalisers are refreshed at the END of an update, so the statistics used to build
  the rollout are byte-identical to the ones used in the update, which is what keeps the epoch-0
  importance ratio exactly 1;
* the value normaliser is refreshed at the START of an update, so the critic is trained against
  targets in the same scale that the next rollout will denormalise with.

The variance update is the numerically stable parallel (Chan et al.) form, so a long run does not
accumulate the catastrophic cancellation of a naive sum-of-squares.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RunningMeanStd(nn.Module):
    """Track a running mean and variance over the leading dimensions of a tensor.

    Registered as buffers so the state travels with ``state_dict`` into the checkpoint.

    Args:
        shape: Trailing shape of the quantity being tracked. ``()`` for a scalar such as the
            value target, ``(8,)`` for the actor vector observation.
        epsilon: Initial pseudo-count. Keeps the first normalisation finite.
        device: Torch device.
        dtype: Torch dtype; must be a floating type.
    """

    def __init__(
        self,
        shape: tuple[int, ...] = (),
        epsilon: float = 1e-4,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.shape = tuple(shape)
        self.register_buffer("mean", torch.zeros(self.shape, dtype=dtype, device=device))
        self.register_buffer("var", torch.ones(self.shape, dtype=dtype, device=device))
        self.register_buffer("count", torch.tensor(float(epsilon), dtype=dtype, device=device))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """Fold a batch into the running statistics.

        Args:
            x: Tensor whose trailing dims equal ``shape``; all leading dims are reduced.

        Raises:
            ValueError: If the trailing dims of ``x`` do not match ``shape``.
        """
        ndim = len(self.shape)
        if ndim and tuple(x.shape[-ndim:]) != self.shape:
            raise ValueError(f"expected trailing shape {self.shape}, got {tuple(x.shape)}")
        flat = x.reshape(-1, *self.shape).to(dtype=self.mean.dtype)
        batch_count = flat.shape[0]
        if batch_count == 0:
            return
        batch_mean = flat.mean(dim=0)
        # Population variance (unbiased=False) so that the parallel update below is exact.
        batch_var = flat.var(dim=0, unbiased=False)

        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * (batch_count / total)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * (self.count * batch_count / total)
        self.mean.copy_(new_mean)
        self.var.copy_(m2 / total)
        self.count.copy_(total)

    def std(self, eps: float = 1e-8) -> torch.Tensor:
        """Return the running standard deviation, floored for numerical safety.

        Args:
            eps: Additive floor inside the square root.

        Returns:
            Tensor of shape ``shape``.
        """
        return torch.sqrt(self.var + eps)

    def normalize(self, x: torch.Tensor, clip: float | None = None) -> torch.Tensor:
        """Standardise ``x`` with the current statistics.

        Args:
            x: Tensor whose trailing dims equal ``shape``.
            clip: Symmetric clip applied after standardisation, or None for no clipping.

        Returns:
            Standardised tensor of the same shape as ``x``.
        """
        out = (x - self.mean) / self.std()
        if clip is not None:
            out = out.clamp(-clip, clip)
        return out

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Invert :meth:`normalize` (without un-clipping).

        Args:
            x: Standardised tensor.

        Returns:
            Tensor mapped back to the original scale.
        """
        return x * self.std() + self.mean

    def extra_repr(self) -> str:
        """Return a one-line summary for ``print(model)``."""
        return f"shape={self.shape}, count={float(self.count):.1f}"
