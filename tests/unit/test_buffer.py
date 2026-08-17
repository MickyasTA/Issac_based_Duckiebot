"""Rollout buffer: layout, aliasing safety, wraparound, terminal capture, minibatching."""

from __future__ import annotations

import pytest
import torch

from duckiebot_rl.ppo.buffer import RolloutBuffer, TerminalCache

OBS_SHAPE = (48, 96, 9)
VEC_DIM = 8
PRIV_DIM = 14
ACT_DIM = 2


def _buffer(num_steps: int = 4, num_envs: int = 3, image: bool = True) -> RolloutBuffer:
    """Build a small buffer for testing.

    Args:
        num_steps: Rollout length.
        num_envs: Environment count.
        image: Whether to allocate the image field.

    Returns:
        A fresh :class:`RolloutBuffer`.
    """
    return RolloutBuffer(
        num_steps=num_steps,
        num_envs=num_envs,
        vec_dim=VEC_DIM,
        priv_dim=PRIV_DIM,
        act_dim=ACT_DIM,
        obs_shape=(8, 16, 9) if image else None,
        device="cpu",
        terminal_capacity=2,
    )


def _step_payload(num_envs: int, value: float, image_shape: tuple[int, int, int] | None) -> dict:
    """Build one ``add`` payload where every field is filled with ``value``.

    Args:
        num_envs: Environment count.
        value: Fill value.
        image_shape: NHWC image shape, or None.

    Returns:
        Kwargs for :meth:`RolloutBuffer.add`.
    """
    payload = {
        "vec": torch.full((num_envs, VEC_DIM), value),
        "vec_priv": torch.full((num_envs, PRIV_DIM), value),
        "action": torch.full((num_envs, ACT_DIM), value),
        "log_prob": torch.full((num_envs,), value),
        "value": torch.full((num_envs,), value),
        "reward": torch.full((num_envs,), value),
        "terminated": torch.zeros(num_envs, dtype=torch.bool),
        "truncated": torch.zeros(num_envs, dtype=torch.bool),
        "mu": torch.full((num_envs, ACT_DIM), value),
        "log_std": torch.full((num_envs, ACT_DIM), value),
    }
    if image_shape is not None:
        payload["image"] = torch.full((num_envs, *image_shape), int(value), dtype=torch.uint8)
    return payload


def test_shapes_dtypes_and_time_major_layout() -> None:
    """Every field is preallocated time-major with the dtype the spec requires."""
    buffer = RolloutBuffer(
        num_steps=32,
        num_envs=256,
        vec_dim=VEC_DIM,
        priv_dim=PRIV_DIM,
        act_dim=ACT_DIM,
        obs_shape=OBS_SHAPE,
        device="cpu",
    )
    assert buffer.images is not None
    assert buffer.images.shape == (32, 256, *OBS_SHAPE)
    assert buffer.images.dtype == torch.uint8
    assert buffer.vec.shape == (32, 256, VEC_DIM)
    assert buffer.vec_priv.shape == (32, 256, PRIV_DIM)
    assert buffer.actions.shape == buffer.mu.shape == buffer.log_std.shape == (32, 256, ACT_DIM)
    for field in (buffer.log_probs, buffer.values, buffer.rewards, buffer.term_values):
        assert field.shape == (32, 256)
        assert field.dtype == torch.float32
    assert buffer.terminated.dtype == torch.bool
    assert buffer.truncated.dtype == torch.bool
    assert buffer.batch_size == 32 * 256
    # SPEC v2 S5.6: the stacked uint8 image store is 324 MiB at N=256, T=32.
    assert buffer.images.numel() == 324 * 1024 * 1024


def test_add_writes_the_expected_slot_and_advances_the_pointer() -> None:
    """``add`` writes at ``current_step`` and moves on; ``full`` flips at the end."""
    buffer = _buffer()
    assert buffer.current_step == 0 and not buffer.full
    for step in range(4):
        buffer.add(**_step_payload(3, float(step + 1), (8, 16, 9)))
        expected_pos = 0 if step == 3 else step + 1
        assert buffer.pos == expected_pos
    assert buffer.full
    torch.testing.assert_close(buffer.vec[2], torch.full((3, VEC_DIM), 3.0), rtol=0, atol=0)
    assert buffer.images is not None
    assert int(buffer.images[3].max()) == 4


def test_add_copies_and_does_not_alias_the_caller_tensor() -> None:
    """SPEC v2 S6.7 guard 3: writes use ``copy_`` so a live camera buffer cannot leak in."""
    buffer = _buffer()
    payload = _step_payload(3, 1.0, (8, 16, 9))
    live_image = payload["image"]
    live_vec = payload["vec"]
    buffer.add(**payload)
    live_image.fill_(200)
    live_vec.fill_(-77.0)
    assert buffer.images is not None
    assert int(buffer.images[0].max()) == 1
    torch.testing.assert_close(buffer.vec[0], torch.ones(3, VEC_DIM), rtol=0, atol=0)


def test_wraparound_overwrites_the_oldest_slot() -> None:
    """Writing past ``num_steps`` wraps to slot 0 and keeps ``full`` set."""
    buffer = _buffer(num_steps=4, num_envs=3)
    for step in range(5):
        buffer.add(**_step_payload(3, float(step + 1), (8, 16, 9)))
    assert buffer.full
    assert buffer.pos == 1
    torch.testing.assert_close(buffer.vec[0], torch.full((3, VEC_DIM), 5.0), rtol=0, atol=0)
    torch.testing.assert_close(buffer.vec[1], torch.full((3, VEC_DIM), 2.0), rtol=0, atol=0)


def test_reset_rewinds_the_pointer_and_clears_derived_state() -> None:
    """``reset`` clears the terminal cache and the derived tensors but keeps the allocation."""
    buffer = _buffer()
    buffer.add(**_step_payload(3, 1.0, (8, 16, 9)))
    buffer.capture_terminal(
        env_ids=torch.tensor([0]),
        vec_priv=torch.ones(1, PRIV_DIM),
        image=torch.ones(1, 8, 16, 9, dtype=torch.uint8),
    )
    buffer.term_values.fill_(3.0)
    buffer.reset()
    assert buffer.pos == 0 and not buffer.full
    assert buffer.terminal_cache.count == 0
    assert float(buffer.term_values.abs().max()) == 0.0


def test_terminal_capture_defaults_to_the_slot_about_to_be_written() -> None:
    """Capture happens inside ``env.step``, before the matching ``add``, hence index ``pos``."""
    buffer = _buffer(num_steps=4, num_envs=3)
    buffer.add(**_step_payload(3, 1.0, (8, 16, 9)))  # step 0 done, pos == 1
    buffer.capture_terminal(
        env_ids=torch.tensor([2]),
        vec_priv=torch.full((1, PRIV_DIM), 5.0),
        image=torch.full((1, 8, 16, 9), 5, dtype=torch.uint8),
    )
    step_index, env_index, vec_priv, image = buffer.terminal_cache.entries()
    assert step_index.tolist() == [1]
    assert env_index.tolist() == [2]
    torch.testing.assert_close(vec_priv, torch.full((1, PRIV_DIM), 5.0), rtol=0, atol=0)
    assert image is not None and int(image.max()) == 5


def test_terminal_cache_grows_beyond_its_initial_capacity() -> None:
    """More resets than the initial capacity grow the cache instead of dropping entries."""
    cache = TerminalCache(obs_shape=(4, 4, 9), priv_dim=PRIV_DIM, capacity=2)
    for step in range(5):
        cache.add(
            step=step,
            env_ids=torch.tensor([step]),
            vec_priv=torch.full((1, PRIV_DIM), float(step)),
            image=torch.full((1, 4, 4, 9), step, dtype=torch.uint8),
        )
    assert cache.count == 5
    assert cache.capacity >= 5
    step_index, env_index, vec_priv, _ = cache.entries()
    assert step_index.tolist() == [0, 1, 2, 3, 4]
    assert env_index.tolist() == [0, 1, 2, 3, 4]
    torch.testing.assert_close(vec_priv[:, 0], torch.arange(5, dtype=torch.float32), rtol=0, atol=0)


def test_compute_terminal_values_scatters_into_the_right_slots() -> None:
    """The critic pass over the cache lands at exactly the captured ``(t, env)`` positions."""
    buffer = _buffer(num_steps=4, num_envs=3)
    buffer.capture_terminal(
        env_ids=torch.tensor([1, 2]),
        vec_priv=torch.stack([torch.full((PRIV_DIM,), 2.0), torch.full((PRIV_DIM,), 3.0)]),
        image=torch.zeros(2, 8, 16, 9, dtype=torch.uint8),
        step=2,
    )

    def value_fn(image: torch.Tensor | None, vec_priv: torch.Tensor) -> torch.Tensor:
        """Return the first privileged column, so the scatter is checkable by eye."""
        assert image is not None
        return vec_priv[:, 0] * 10.0

    count = buffer.compute_terminal_values(value_fn)
    assert count == 2
    assert buffer.term_values[2, 1].item() == pytest.approx(20.0)
    assert buffer.term_values[2, 2].item() == pytest.approx(30.0)
    assert float(buffer.term_values.sum()) == pytest.approx(50.0)


def test_flat_flattens_time_major_in_row_major_order() -> None:
    """``flat`` maps ``(t, n)`` to ``t * num_envs + n``, which the minibatch indices rely on."""
    buffer = _buffer(num_steps=3, num_envs=2)
    for step in range(3):
        payload = _step_payload(2, 0.0, (8, 16, 9))
        payload["vec"] = torch.tensor([[float(step * 2 + 0)] * VEC_DIM, [float(step * 2 + 1)] * VEC_DIM])
        buffer.add(**payload)
    flat = buffer.flat()
    assert flat["vec"].shape == (6, VEC_DIM)
    torch.testing.assert_close(flat["vec"][:, 0], torch.arange(6, dtype=torch.float32), rtol=0, atol=0)
    assert flat["image"].shape == (6, 8, 16, 9)


def test_minibatches_partition_the_batch_exactly_once_per_epoch() -> None:
    """Every transition appears in exactly one minibatch, and the shapes are uniform."""
    buffer = _buffer(num_steps=4, num_envs=4)
    for step in range(4):
        payload = _step_payload(4, 0.0, (8, 16, 9))
        payload["vec"] = (
            torch.arange(step * 4, step * 4 + 4, dtype=torch.float32).unsqueeze(1).repeat(1, VEC_DIM)
        )
        buffer.add(**payload)
    seen: list[float] = []
    count = 0
    for minibatch in buffer.minibatches(num_minibatches=4):
        assert minibatch["vec"].shape == (4, VEC_DIM)
        assert minibatch["image"].shape == (4, 8, 16, 9)
        seen.extend(minibatch["vec"][:, 0].tolist())
        count += 1
    assert count == 4
    assert sorted(seen) == list(range(16))


def test_minibatches_reject_an_indivisible_split() -> None:
    """SPEC v2 S6.6 requires ``batch % num_minibatches == 0``; violating it raises."""
    buffer = _buffer(num_steps=4, num_envs=3)
    with pytest.raises(ValueError, match="not divisible"):
        next(iter(buffer.minibatches(num_minibatches=5)))


def test_vec_only_mode_allocates_no_image_field() -> None:
    """With ``obs_shape=None`` the buffer holds no pixels and rejects image payloads."""
    buffer = _buffer(image=False)
    assert buffer.images is None
    payload = _step_payload(3, 1.0, None)
    buffer.add(**payload)
    assert "image" not in buffer.flat()
    with pytest.raises(ValueError, match="vec-only mode"):
        buffer.add(**_step_payload(3, 1.0, (8, 16, 9)))


def test_image_buffer_rejects_a_missing_image() -> None:
    """An image buffer fed a vec-only payload fails loudly."""
    buffer = _buffer(image=True)
    with pytest.raises(ValueError, match="image is None"):
        buffer.add(**_step_payload(3, 1.0, None))
