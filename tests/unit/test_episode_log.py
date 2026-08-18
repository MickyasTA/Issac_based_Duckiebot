"""Equivalence and regression tests for the device-side episode log (profile rank 6).

Two things are pinned here, and the second is the one that matters long term:

1. **Equivalence.** Draining a channel produces exactly the values the old per-reset
   ``.cpu().tolist()`` bursts produced, in the same order and with the same dtypes, and
   :meth:`DeviceLog.drain_means` produces exactly the float the old ``sum(values)/len(values)``
   produced.
2. **The cost cannot come back.** ``append`` and ``pending`` must perform ZERO host
   synchronisation, so that putting a ``.item()``, a ``.cpu()``, a ``.tolist()`` or a truthiness
   test back into ``_log_finished_episodes`` fails a test rather than quietly costing 15 ms of
   every control step again.

The counter itself is the ``count_host_syncs`` fixture in the root ``conftest.py``.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from duckiebot_rl.envs.episode_log import DeviceLog


def test_the_sync_counter_actually_counts(count_host_syncs: Callable) -> None:
    """A guard that cannot fail is not a guard; prove the instrument works before trusting it."""
    with count_host_syncs() as syncs:
        assert syncs() == 0
        torch.zeros(3).tolist()
        assert syncs() == 1
        float(torch.zeros(()))
        assert syncs() >= 2


def test_the_sync_counter_restores_torch_after_a_failure(count_host_syncs: Callable) -> None:
    """A leaked patch would silently slow, and subtly alter, every later test in the session."""
    original = torch.Tensor.tolist
    with pytest.raises(RuntimeError), count_host_syncs():
        raise RuntimeError("boom")
    assert torch.Tensor.tolist is original


# ---------------------------------------------------------------------------------------------
# Equivalence with the per-reset transfers it replaced
# ---------------------------------------------------------------------------------------------


def _per_reset_reference(chunks: list[torch.Tensor]) -> list[float]:
    """Return what the old code produced: one transfer per chunk, extended into a list.

    Args:
        chunks: The per-reset tensors, in reset order.

    Returns:
        The concatenated host values.
    """
    out: list[float] = []
    for chunk in chunks:
        out.extend(chunk.detach().cpu().tolist())
    return out


def test_drain_matches_the_per_reset_transfers_element_for_element() -> None:
    """Concatenating on the device then transferring once changes no value and no order."""
    generator = torch.Generator().manual_seed(11)
    chunks = [torch.rand(k, generator=generator) for k in (3, 1, 7, 2, 5)]

    log = DeviceLog()
    for chunk in chunks:
        log.append("episode/return", chunk)

    assert log.drain()["episode/return"] == _per_reset_reference(chunks)


def test_drain_means_matches_the_old_host_side_average_exactly() -> None:
    """The S6.8 numbers are logged; averaging on the device instead would round differently."""
    generator = torch.Generator().manual_seed(3)
    chunks = [torch.rand(k, generator=generator) for k in (4, 6, 1)]
    reference = _per_reset_reference(chunks)

    log = DeviceLog()
    for chunk in chunks:
        log.append("episode/length_s", chunk)

    assert log.drain_means()["episode/length_s"] == sum(reference) / len(reference)


def test_drain_preserves_integer_dtype_for_the_spawn_slots() -> None:
    """The hard-example miner indexes a table with these; floats would be a silent failure."""
    log = DeviceLog()
    log.append("slot", torch.tensor([4, 9, 1], dtype=torch.long))
    drained = log.drain()["slot"]
    assert drained == [4, 9, 1]
    assert all(isinstance(v, int) for v in drained)


def test_channels_are_independent() -> None:
    """The six S6.8 metrics share one log; they must not bleed into one another."""
    log = DeviceLog()
    log.extend({"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([10.0])})
    log.extend({"a": torch.tensor([3.0]), "b": torch.tensor([20.0, 30.0])})
    assert log.drain() == {"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]}


def test_drain_clears_so_an_iteration_never_double_counts() -> None:
    """``drain_episode_log`` runs once per iteration and must not re-report old episodes."""
    log = DeviceLog()
    log.append("x", torch.tensor([1.0]))
    assert log.drain() == {"x": [1.0]}
    assert log.drain() == {}
    assert log.pending is False


def test_empty_chunks_are_dropped_not_stored() -> None:
    """The two ADR probe channels are masked subsets and are usually empty on a given reset."""
    log = DeviceLog()
    log.extend({"vis": torch.zeros(0), "dyn": torch.tensor([2.0])})
    assert log.pending is True
    assert log.drain() == {"dyn": [2.0]}


def test_drain_of_an_empty_log_is_an_empty_dict_not_a_nan() -> None:
    """An iteration in which nothing terminated must report nothing, not ``0/0``."""
    assert DeviceLog().drain_means() == {}


def test_append_detaches_so_a_graph_is_never_retained() -> None:
    """These are diagnostics; holding autograd graphs alive for a whole iteration would leak."""
    log = DeviceLog()
    log.append("x", torch.ones(2, requires_grad=True) * 2.0)
    assert log.drain()["x"] == [2.0, 2.0]


def test_appended_chunks_are_not_aliases_of_the_callers_buffer() -> None:
    """``_reset_idx`` zeroes the per-env buffers right after logging them.

    The environment appends ``self._ep_return[ids]``, which is advanced indexing and therefore a
    copy. This test states that dependency explicitly, so a future refactor to a view (a slice, a
    ``narrow``) fails here instead of silently logging zeros.
    """
    buffer = torch.tensor([5.0, 6.0, 7.0])
    ids = torch.tensor([0, 2])

    log = DeviceLog()
    log.append("episode/return", buffer[ids])
    buffer[ids] = 0.0

    assert log.drain()["episode/return"] == [5.0, 7.0]


# ---------------------------------------------------------------------------------------------
# The regression guard: the hot path must not synchronise
# ---------------------------------------------------------------------------------------------


def test_append_performs_no_host_sync(count_host_syncs: Callable) -> None:
    """This is the whole point of profile rank 6, and it runs inside every ``_reset_idx``."""
    log = DeviceLog()
    with count_host_syncs() as syncs:
        for _ in range(32):
            log.extend(
                {
                    "episode/return": torch.rand(7),
                    "episode/length_s": torch.rand(7),
                    "slot": torch.randint(0, 40, (7,)),
                }
            )
        assert syncs() == 0


def test_pending_performs_no_host_sync(count_host_syncs: Callable) -> None:
    """``drain_episode_log`` tests it, but so could a per-step caller; keep it metadata-only."""
    log = DeviceLog()
    with count_host_syncs() as syncs:
        assert log.pending is False
        log.append("x", torch.rand(4))
        assert log.pending is True
        assert syncs() == 0


def test_drain_costs_one_transfer_per_nonempty_channel_regardless_of_append_count(
    count_host_syncs: Callable,
) -> None:
    """Twenty resets across three channels must cost three transfers, not sixty.

    ``cpu()`` and ``tolist()`` are both counted, so the expected total is two per channel.
    """
    log = DeviceLog()
    for _ in range(20):
        log.extend({"a": torch.rand(3), "b": torch.rand(3), "c": torch.rand(3)})
    with count_host_syncs() as syncs:
        log.drain()
        assert syncs() == 2 * 3


@pytest.mark.parametrize("channels", [1, 4])
def test_drain_sync_count_scales_with_channels_not_with_episodes(
    channels: int, count_host_syncs: Callable
) -> None:
    """The cost of the diagnostics must not grow with how many episodes finished."""
    counts = []
    for episodes in (1, 50):
        log = DeviceLog()
        for _ in range(episodes):
            log.extend({f"c{i}": torch.rand(2) for i in range(channels)})
        with count_host_syncs() as syncs:
            log.drain()
            counts.append(syncs())
    assert counts[0] == counts[1] == 2 * channels
