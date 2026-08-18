"""Per-episode diagnostics accumulated on the device, drained to the host once per iteration.

The problem this solves
-----------------------

``DuckiebotLaneFollowEnv._log_finished_episodes`` runs inside ``_reset_idx``, i.e. on every
control step in which any environment terminates. It records six S6.8 episode metrics, the
hard-example miner's spawn slot and tracking error, and the two ADR probe channels. Written the
obvious way - ``self._episode_log[key].extend(values.cpu().tolist())`` - that is ten
device-to-host round trips per reset.

The M-phase profile put a number on it: 41.2 ms per reset at ``N=64``, 0.375 resets per control
step, 15.4 ms amortised into a 340.7 ms step (profile rank 6). Worse, the reset fraction is
proportional to ``N``: at ``N=256`` nearly every step pays it, which is part of why throughput
was measured to be flat in ``N``.

Nothing reads these values until ``drain_episode_log`` and ``drain_curriculum_records``, which
the training loop calls once per iteration. So the fix is not to make the transfer cheaper but
to stop doing it in the hot path: keep the chunks as device tensors, concatenate at drain time,
and pay one transfer per channel per iteration instead of ten per reset.

Why a class rather than three inline lists
------------------------------------------

``lane_follow_env.py`` imports Isaac Lab at module scope and therefore cannot be imported on a
CPU runner at all. Anything left inline there is untestable. Here, the whole accumulate-and-
drain contract - including the promise that draining is bit-for-bit what the per-reset transfers
produced, and the promise that appending performs no host sync - is exercised by
``tests/unit/test_episode_log.py``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

__all__ = ["DeviceLog"]


class DeviceLog:
    """A set of named channels of ``(M,)`` device tensors, transferred to the host on drain.

    A channel accumulates one tensor per :meth:`append` call. Draining concatenates a channel and
    moves it to the host in a single transfer, so the number of host syncs a drain costs is the
    number of non-empty channels, not the number of appends.

    Every tensor handed to :meth:`append` must be one the caller no longer mutates. In the
    environment they always are: each comes from advanced indexing (``self._ep_return[ids]``),
    which copies, so the per-env buffers that ``_reset_idx`` zeroes immediately afterwards are
    not the tensors held here.
    """

    def __init__(self) -> None:
        self._chunks: dict[str, list[torch.Tensor]] = {}

    @property
    def pending(self) -> bool:
        """Whether any channel holds anything.

        Host-side metadata only: this reads python list lengths, never tensor contents, so it
        costs no device synchronisation and is safe to test on every control step.

        Returns:
            True when at least one channel has at least one chunk.
        """
        return any(self._chunks.values())

    def append(self, key: str, values: torch.Tensor) -> None:
        """Add one chunk to a channel.

        Empty chunks are dropped rather than stored, which keeps the concatenation at drain time
        free of zero-length tensors and makes :attr:`pending` mean what it says.

        Args:
            key: Channel name.
            values: A ``(M,)`` device tensor. Detached here, so the caller need not.
        """
        if values.numel() == 0:
            return
        self._chunks.setdefault(key, []).append(values.detach())

    def extend(self, entries: Mapping[str, torch.Tensor]) -> None:
        """Add one chunk to each of several channels.

        Args:
            entries: ``{channel_name: (M,) tensor}``.
        """
        for key, values in entries.items():
            self.append(key, values)

    def drain(self) -> dict[str, list[Any]]:
        """Return every channel as host lists and clear the log.

        One ``torch.cat`` and one ``.cpu().tolist()`` per non-empty channel. The result is
        element-for-element identical to having transferred each chunk separately and extended a
        python list, because concatenation is a copy and ``tolist`` preserves dtype: floats stay
        floats, ``torch.long`` channels come back as ints.

        Returns:
            ``{channel_name: [value, ...]}`` in append order, omitting empty channels.
        """
        out = {key: torch.cat(chunks).cpu().tolist() for key, chunks in self._chunks.items() if chunks}
        self._chunks = {}
        return out

    def drain_means(self) -> dict[str, float]:
        """Return the mean of every channel and clear the log.

        Args:
            None.

        Returns:
            ``{channel_name: mean}``, omitting empty channels. The mean is taken on the host over
            the drained python floats, which is what the S6.8 diagnostics reported before the
            accumulation moved to the device; averaging on the device instead would round
            differently and silently change a logged number.
        """
        return {key: sum(values) / len(values) for key, values in self.drain().items()}
