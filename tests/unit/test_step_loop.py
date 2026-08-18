"""Equivalence and regression tests for the hoisted decimation window (profile rank 1).

``duckiebot_rl.envs.lane_follow_env`` cannot be imported without Kit, so the *policy* of the
override lives in the Isaac-free :mod:`duckiebot_rl.envs.step_loop` and is tested here. What
these tests pin down is the whole equivalence claim that the hoist rests on:

* the physics steps are the same in number and in position,
* the renders are the same in number and in position,
* the *only* difference is how many times the constant actuation is written and how many times
  the lazy buffers are timestamped,
* and turning the switch off reproduces ``DirectRLEnv.step``'s loop op for op.
"""

from __future__ import annotations

import pytest

from duckiebot_rl.assets.params import DUCKIEBOT
from duckiebot_rl.envs.env_cfg import LaneFollowSettings, PerfSettings, RateSettings
from duckiebot_rl.envs.step_loop import (
    APPLY,
    PHYSICS,
    RENDER,
    UPDATE,
    WRITE,
    Op,
    baseline_window_ops,
    render_substeps,
    run_window,
    window_ops,
)

DECIMATION = DUCKIEBOT.decimation
"""16 on this robot; the campaign profile that motivated the hoist ran at exactly this value."""


# ---------------------------------------------------------------------------------------------
# The reference loop, written out a second time and independently
# ---------------------------------------------------------------------------------------------


def _direct_rl_env_loop(
    start: int, decimation: int, render_interval: int, is_rendering: bool
) -> tuple[Op, ...]:
    """Return the base-class window, transcribed straight from ``direct_rl_env.py:369-384``.

    Deliberately a second, independent transcription rather than a call into ``step_loop``: a
    test that reuses the code under test to build its own expectation proves nothing.

    Args:
        start: ``_sim_step_counter`` before the window.
        decimation: Physics steps per control step.
        render_interval: Physics steps per render.
        is_rendering: ``sim.has_gui() or sim.has_rtx_sensors()``.

    Returns:
        The operation sequence.
    """
    ops = []
    counter = start
    for _ in range(decimation):
        counter += 1
        ops.append(APPLY)
        ops.append(WRITE)
        ops.append(PHYSICS)
        if counter % render_interval == 0 and is_rendering:
            ops.append(RENDER)
        ops.append(UPDATE)
    return tuple(ops)


# ---------------------------------------------------------------------------------------------
# The hoist changes nothing that physics or the renderer can see
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("start", [0, 1, 15, 16, 17, 4096, 1_000_003])
@pytest.mark.parametrize("decimation", [1, 2, 16])
def test_baseline_plan_matches_direct_rl_env_verbatim(start: int, decimation: int) -> None:
    """The unhoisted plan is the base class's loop, op for op, at every phase offset."""
    assert baseline_window_ops(start, decimation, decimation, True) == _direct_rl_env_loop(
        start, decimation, decimation, True
    )


@pytest.mark.parametrize("start", [0, 3, 16, 33, 255])
def test_hoist_preserves_physics_and_render_positions(start: int) -> None:
    """Dropping the writes and updates from either plan leaves the identical skeleton.

    This is the equivalence claim in its sharpest form: the sequence of things that advance the
    simulation or produce an observation is invariant under the hoist.
    """
    skeleton = (PHYSICS, RENDER)
    hoisted = window_ops(start, DECIMATION, DECIMATION, True)
    baseline = baseline_window_ops(start, DECIMATION, DECIMATION, True)
    assert tuple(op for op in hoisted if op in skeleton) == tuple(op for op in baseline if op in skeleton)


@pytest.mark.parametrize("is_rendering", [True, False])
@pytest.mark.parametrize("start", [0, 5, 16, 31])
def test_hoist_preserves_render_count_including_the_no_sensor_case(start: int, is_rendering: bool) -> None:
    """One render per control step with an RTX sensor, none without, hoisted or not."""
    hoisted = window_ops(start, DECIMATION, DECIMATION, is_rendering)
    baseline = baseline_window_ops(start, DECIMATION, DECIMATION, is_rendering)
    assert hoisted.count(RENDER) == baseline.count(RENDER) == (1 if is_rendering else 0)


def test_hoist_collapses_exactly_the_repeated_work() -> None:
    """16 writes and 16 updates become 1 each; the physics count is untouched."""
    hoisted = window_ops(0, DECIMATION, DECIMATION, True)
    baseline = baseline_window_ops(0, DECIMATION, DECIMATION, True)

    assert baseline.count(WRITE) == baseline.count(APPLY) == baseline.count(UPDATE) == DECIMATION
    assert hoisted.count(WRITE) == hoisted.count(APPLY) == hoisted.count(UPDATE) == 1
    assert hoisted.count(PHYSICS) == baseline.count(PHYSICS) == DECIMATION


def test_hoisted_write_precedes_every_physics_step() -> None:
    """The single actuation write must land before the first ``sim.step``, not after it."""
    ops = window_ops(0, DECIMATION, DECIMATION, True)
    assert ops[:2] == (APPLY, WRITE)
    assert ops.index(WRITE) < ops.index(PHYSICS)


def test_hoisted_update_trails_the_last_physics_step_and_the_render() -> None:
    """The single ``scene.update`` must be the last op, so the snapshot reads post-window state.

    ``_get_dones`` takes the whole per-step physics snapshot immediately after this window. If
    the trailing update ran before the last ``sim.step``, every root pose the reward and the
    terminations see would be one substep stale.
    """
    ops = window_ops(0, DECIMATION, DECIMATION, True)
    assert ops[-1] == UPDATE
    assert len(ops) - 1 > max(i for i, op in enumerate(ops) if op in (PHYSICS, RENDER))


@pytest.mark.parametrize("start", [0, 7, 16, 100])
def test_both_flags_off_is_the_baseline(start: int) -> None:
    """The fallback path is the same function with different flags, not a second loop."""
    assert window_ops(
        start, DECIMATION, DECIMATION, True, hoist_writes=False, hoist_updates=False
    ) == baseline_window_ops(start, DECIMATION, DECIMATION, True)


def test_writes_hoisted_updates_not_is_a_valid_intermediate() -> None:
    """The two flags are independent, which is what makes an A/B of them possible."""
    ops = window_ops(0, DECIMATION, DECIMATION, True, hoist_writes=True, hoist_updates=False)
    assert ops.count(WRITE) == 1
    assert ops.count(UPDATE) == DECIMATION
    assert ops.count(PHYSICS) == DECIMATION


# ---------------------------------------------------------------------------------------------
# render_substeps
# ---------------------------------------------------------------------------------------------


def test_render_substeps_fires_once_at_the_end_when_interval_equals_decimation() -> None:
    """``RateSettings`` forces ``render_interval == decimation``; that means the last substep."""
    assert render_substeps(0, DECIMATION, DECIMATION, True) == (DECIMATION,)


def test_render_substeps_tracks_the_running_counter_not_the_window() -> None:
    """The counter is global, so a window starting mid-interval renders mid-window."""
    assert render_substeps(3, 16, 16, True) == (13,)
    assert render_substeps(15, 16, 16, True) == (1,)
    assert render_substeps(0, 16, 4, True) == (4, 8, 12, 16)


def test_render_substeps_empty_without_a_renderer() -> None:
    """Vec-only mode with no GUI renders nothing at all."""
    assert render_substeps(0, DECIMATION, DECIMATION, False) == ()


@pytest.mark.parametrize("bad", [0, -1])
def test_render_substeps_rejects_nonpositive_rates(bad: int) -> None:
    """A zero decimation would silently produce an empty window."""
    with pytest.raises(ValueError):
        render_substeps(0, bad, DECIMATION, True)
    with pytest.raises(ValueError):
        render_substeps(0, DECIMATION, bad, True)


# ---------------------------------------------------------------------------------------------
# run_window: the executor the environment actually calls
# ---------------------------------------------------------------------------------------------


class _Recorder:
    """Stands in for the five Isaac calls, recording the order they are made in."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.counter = 0
        self.update_dts: list[float] = []

    def table(self, update_dt: float) -> dict:
        """Return an op table shaped exactly like the environment's.

        Args:
            update_dt: The ``dt`` the environment would pass to ``scene.update``.

        Returns:
            The op-to-callable mapping :func:`run_window` consumes.
        """

        def physics() -> None:
            self.counter += 1
            self.calls.append(PHYSICS)

        def update() -> None:
            self.update_dts.append(update_dt)
            self.calls.append(UPDATE)

        return {
            APPLY: lambda: self.calls.append(APPLY),
            WRITE: lambda: self.calls.append(WRITE),
            PHYSICS: physics,
            RENDER: lambda: self.calls.append(RENDER),
            UPDATE: update,
        }


@pytest.mark.parametrize("hoist", [True, False])
def test_run_window_executes_the_plan_in_order(hoist: bool) -> None:
    """The executed order is the planned order, hoisted or not."""
    ops = window_ops(0, DECIMATION, DECIMATION, True, hoist_writes=hoist, hoist_updates=hoist)
    rec = _Recorder()
    run_window(ops, rec.table(1.0))
    assert tuple(rec.calls) == ops
    assert rec.counter == DECIMATION


def test_run_window_advances_the_sim_step_counter_exactly_decimation_times() -> None:
    """The counter drives the render schedule of the NEXT window, so it must not drift."""
    rec = _Recorder()
    start = 0
    for _ in range(4):
        ops = window_ops(start, DECIMATION, DECIMATION, True)
        run_window(ops, rec.table(1.0))
        start = rec.counter
    assert rec.counter == 4 * DECIMATION
    assert rec.calls.count(RENDER) == 4


def test_run_window_raises_on_a_missing_op() -> None:
    """A plan the caller cannot execute must fail loudly, not skip a physics step."""
    with pytest.raises(KeyError):
        run_window(window_ops(0, 2, 2, True), {APPLY: lambda: None})


# ---------------------------------------------------------------------------------------------
# The environment's dt bookkeeping, reproduced without Kit
# ---------------------------------------------------------------------------------------------


def test_hoisted_update_carries_the_whole_window_dt() -> None:
    """One update must advance sensor timestamps by ``decimation * physics_dt``, not by one dt.

    ``SensorBase.update`` accumulates ``dt`` and compares it against ``update_period``; a
    sixteenth of the elapsed time would make a rate-limited sensor fire sixteen times too
    slowly. This reproduces ``_run_physics_window``'s expression exactly.
    """
    rates = RateSettings()
    physics_dt = rates.sim_dt_s

    for hoist in (True, False):
        ops = window_ops(
            0, rates.decimation, rates.render_interval, True, hoist_writes=hoist, hoist_updates=hoist
        )
        update_dt = physics_dt * (rates.decimation if ops.count(UPDATE) == 1 else 1)
        rec = _Recorder()
        run_window(ops, rec.table(update_dt))
        assert sum(rec.update_dts) == pytest.approx(rates.control_dt_s)


# ---------------------------------------------------------------------------------------------
# The switches are recorded, so a run can never be silently confounded
# ---------------------------------------------------------------------------------------------


def test_perf_defaults_hoist_but_do_not_touch_the_rng() -> None:
    """The two semantics-preserving switches are on; the stream-reordering one is off."""
    perf = PerfSettings()
    assert perf.hoist_actuation_writes is True
    assert perf.hoist_scene_updates is True
    assert perf.fused_visual_dr_draws is False


def test_perf_switches_appear_in_the_run_summary() -> None:
    """A throughput switch that is invisible in the checkpoint is a silent confound."""
    summary = LaneFollowSettings(perf=PerfSettings(fused_visual_dr_draws=True)).summary()
    assert summary["perf_hoist_actuation_writes"] is True
    assert summary["perf_hoist_scene_updates"] is True
    assert summary["perf_fused_visual_dr_draws"] is True
