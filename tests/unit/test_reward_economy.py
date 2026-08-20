"""The reward economy: driving must out-pay creeping, creeping parking, and parking dying.

Two regressions reached live runs, one per direction of the same level error.

2026-08-18, the suicide equilibrium: once the lane matcher started reporting honest ``d``, an
untrained policy paid -2.0 to -4.4 per step while wandering, against a one-off terminal penalty
of -10. Terminating 400 steps early therefore SAVED roughly 800 reward, and PPO found that
optimum in under a million steps: mean episode length fell from 72 to 44 steps while returns
rose from -316 to -88, and the eval metric pinned flat. The fix was the survival income.

2026-08-19, the parking annuity: paid unconditionally, that same survival income (5.0/step) made
NOT driving the best risk-free policy in the MDP. Measured through ``compute_reward``: parking
earned +5.18/step until the stall guard fired, a 0.04 m/s creep earned +6.40/step FOREVER (any
single step at or above 0.03 m/s resets the 2 s stall counter, so near-stillness is
termination-proof), while full-speed driving 8 cm off centre, outside the progress gate, earned
+5.05/step. Between iterations 600 and 700 of
``20260819T031011Z_lanefollow_seed0_survival_seed0`` every env stopped driving (reward/stall
0.0 to 1.0, returns 233 to 1656, eval distance 1.62 to 0.02 tiles), and the training metrics
called the result 57% "success" because nothing they measured required covering distance. The
fix is the :func:`~duckiebot_rl.envs.rewards.r_survival` motion gate, and the archetype tests
below pin the repaired income ordering DRIVE > CREEP > PARK > DIE with margins, through the
real stall guard.
"""

from __future__ import annotations

import math

import torch

from duckiebot_rl.assets.params import DUCKIEBOT
from duckiebot_rl.envs.rewards import RewardWeights, compute_reward
from duckiebot_rl.envs.terminations import TerminationState

# Two reference states, both grounded in data rather than imagination.
#
# MEASURED_DYING is the trajectory the collapsed run actually chose: lane_dev_rms 0.37 over the
# episode means a mix of moderate excursions, ordinary steering, healthy speed, roughly -2 per
# step under the old economy. This is the state the level argument must fix: it has to be worth
# surviving OUTRIGHT, because PPO demonstrably found death faster than it found recovery.
#
# ADVERSARIAL_STACK piles every penalty to its ceiling at once: departure at its cap, heading
# fully wrong, oncoming lane, full action flip. Sizing the survival constant to make THIS state
# positive would take about +12, which would press perfect driving (progress 6 + survival)
# against the +/-20 reward clip and compress the signal that separates good driving from great.
# The stack is instead covered by the gradient property: from there, every step of recovery pays
# immediately, so continuing toward the lane strictly beats freezing, and death forfeits the
# recoverable future.
MEASURED_DYING = {
    "d": torch.tensor([0.15]),
    "psi": torch.tensor([0.6]),
    "ds": torch.tensor([0.0]),
    "action": torch.tensor([[0.6, 0.2]]),
    "prev_action": torch.tensor([[0.5, 0.1]]),
    "body_speed": torch.tensor([0.3]),
    "lane_width": torch.tensor([0.2046]),
}

ADVERSARIAL_STACK = {
    "d": torch.tensor([0.37]),
    "psi": torch.tensor([2.0]),
    "ds": torch.tensor([0.0]),
    "action": torch.tensor([[1.0, 1.0]]),
    "prev_action": torch.tensor([[-1.0, -1.0]]),
    "body_speed": torch.tensor([0.3]),
    "lane_width": torch.tensor([0.2046]),
}

TRUNCATION_STEPS = 450


def _reward(state: dict, terminated: bool) -> float:
    """One step of ``state``, with or without a termination."""
    out, _ = compute_reward(**state, terminated=torch.tensor([terminated]), weights=RewardWeights())
    return float(out)


def test_the_measured_dying_state_is_worth_surviving_outright() -> None:
    """The state the collapsed run averaged while choosing death now nets positive.

    Old economy: about -2 per step here, -10 to die, so dying 400 steps early saved ~800 and
    the run cashed that in (episode length 72 to 44, eval flat at 1.8 tiles for 900
    iterations). With the survival term this state is positive, so there is nothing to cash.
    """
    live = _reward(MEASURED_DYING, terminated=False)
    assert live > 0.0, f"the measured dying state pays {live:.2f}; the equilibrium is back"


def test_surviving_beats_dying_at_every_horizon() -> None:
    """``live * k > die`` for every remaining horizon, which is "suicide never profits"."""
    live = _reward(MEASURED_DYING, terminated=False)
    die = _reward(MEASURED_DYING, terminated=True)
    for k in range(1, TRUNCATION_STEPS + 1):
        assert live * k > die, f"dying now ({die:.2f}) beats surviving {k} steps ({live * k:.2f})"


def test_recovery_pays_at_every_step_even_from_the_adversarial_stack() -> None:
    """From the all-penalties-at-cap state, each unit of recovery strictly raises the reward.

    The stack itself is allowed to be net negative (see the header), so what is pinned is the
    escape incentive: reward strictly increases as ``d`` comes home, and dropping the action
    flip alone is an immediate gain. A state that punishes recovery as much as freezing is the
    only way the stack could become a trap, and this is the assertion that forbids it.
    """
    previous = _reward(ADVERSARIAL_STACK, terminated=False)
    calmer = dict(ADVERSARIAL_STACK)
    calmer.update(prev_action=torch.tensor([[1.0, 1.0]]))  # stop flipping: instant gain
    assert _reward(calmer, terminated=False) > previous + 0.5
    for d in (0.30, 0.234, 0.15, 0.10, 0.05, 0.0):
        step = dict(ADVERSARIAL_STACK)
        step.update(d=torch.tensor([d]), psi=torch.tensor([2.0 * d / 0.37]))
        now = _reward(step, terminated=False)
        assert now > previous, f"recovery to d={d} did not pay ({now:.2f} <= {previous:.2f})"
        previous = now


def test_good_driving_still_dominates_bad_driving_by_the_same_margin() -> None:
    """At driving speeds the survival term is a constant shift: no driving style is reordered.

    Since 2026-08-20 the term is speed-gated, so "constant shift" holds exactly at and above
    ``survival_speed_ref`` (0.3 m/s), which both compared states satisfy (0.6 and 0.3). That is
    the design: the gate must separate motion from stillness while leaving every preference
    BETWEEN moving trajectories untouched, and this assertion is what pins the second half.
    Below the reference speed the shift is deliberately not constant; the archetype tests own
    that regime.
    """
    good = dict(MEASURED_DYING)
    good.update(
        d=torch.tensor([0.0]),
        psi=torch.tensor([0.0]),
        ds=torch.tensor([0.041]),
        action=torch.tensor([[1.0, 0.0]]),
        prev_action=torch.tensor([[1.0, 0.0]]),
        body_speed=torch.tensor([0.6]),
    )
    with_term, _ = compute_reward(**good, terminated=None, weights=RewardWeights())
    without_term, _ = compute_reward(**good, terminated=None, weights=RewardWeights(survival=0.0))
    bad_with, _ = compute_reward(**MEASURED_DYING, terminated=None, weights=RewardWeights())
    bad_without, _ = compute_reward(**MEASURED_DYING, terminated=None, weights=RewardWeights(survival=0.0))
    margin_with = float(with_term - bad_with)
    margin_without = float(without_term - bad_without)
    assert abs(margin_with - margin_without) < 1e-5, "the shift changed a driving preference"
    assert margin_with > 4.0, "good driving must clearly dominate the measured dying state"


def test_legacy_recipe_still_reproduces_the_v1_economy() -> None:
    """``RewardWeights.legacy()`` keeps survival at zero so old numbers stay reproducible."""
    assert RewardWeights.legacy().survival == 0.0
    assert RewardWeights().survival == 5.0


# =============================================================================================
# The 2026-08-19 parking annuity: four archetypes through the real reward and the real guard
# =============================================================================================
#
# Each archetype is a stationary behaviour, evaluated per step by ``compute_reward`` and rolled
# over a 450-step training window with the REAL ``TerminationState`` deciding when the stall
# guard ends an episode. Lateral state is the lane centre for all four, which is the BEST case
# for the degenerate behaviours (the collapsed run parked ~3.5 cm from centre at spawn), so the
# margins below are lower bounds on the real separation.

_DT = DUCKIEBOT.control_dt_s
_LANE_W = 0.2046
_WINDOW = 450


def _step(d: float, psi: float, v: float, ds: float, terminated: bool = False) -> float:
    """One ``compute_reward`` step of a steady behaviour, default weights, no action churn."""
    out, _ = compute_reward(
        d=torch.tensor([d]),
        psi=torch.tensor([psi]),
        ds=torch.tensor([ds]),
        action=torch.tensor([[0.0, 0.0]]),
        prev_action=torch.tensor([[0.0, 0.0]]),
        body_speed=torch.tensor([v]),
        lane_width=torch.tensor([_LANE_W]),
        terminated=torch.tensor([terminated]),
        weights=RewardWeights(),
        control_dt=_DT,
    )
    return float(out)


def _psi_target(d: float) -> float:
    """The corrective heading the reward asks for at offset ``d`` (S5.4)."""
    return -max(-1.0, min(1.0, d / 0.05)) * math.pi / 4.0


def _stall_episode_steps(v: float, horizon: int = _WINDOW) -> int | None:
    """Steps until the real stall guard ends an episode at constant speed ``v``, else None."""
    state = TerminationState(1)
    state.reset(spawn_xy=torch.zeros(1, 2))
    x = 0.0
    for i in range(1, horizon + 1):
        x += v * _DT
        stall, spin = state.update(
            torch.tensor([v]), torch.tensor([0.0]), torch.tensor([x]), torch.tensor([0.0])
        )
        if bool(stall[0]) or bool(spin[0]):
            return i
    return None


# Per-step incomes of the four live archetypes. PARK sits at exactly 0 m/s; CREEP at 0.04 m/s,
# just above the 0.03 stall threshold; DRIVE at 0.5 m/s on the centreline inside the progress
# gate; SLOPPY_DRIVE at full speed 8 cm off centre with the CORRECT corrective heading, which
# puts it outside the 5.68 cm progress gate: the execution-error case that the unconditional
# survival term inverted (it paid 5.05 against creeping's 6.40).
PARK_STEP = _step(0.0, 0.0, 0.0, 0.0)
CREEP_STEP = _step(0.0, 0.0, 0.04, 0.04 * _DT)
DRIVE_STEP = _step(0.0, 0.0, 0.5, 0.5 * _DT)
SLOPPY_STEP = _step(0.08, _psi_target(0.08), 0.6, 0.6 * _DT)


def test_the_default_weights_still_carry_the_measured_gate() -> None:
    """The archetype margins below were sized at these values; a resize must re-derive them."""
    weights = RewardWeights()
    assert weights.survival == 5.0
    assert weights.survival_speed_ref == 0.3


def test_park_and_creep_evade_or_meet_the_real_stall_guard_as_measured() -> None:
    """The guard prices exact stillness only: parking dies at step 31, creeping NEVER dies.

    This is the loophole the collapsed run exploited and the reason a faster or harsher stall
    guard cannot fix the economy: 0.04 m/s resets the consecutive-stall counter on every step,
    so the guard never fires and the creep annuity runs to the horizon. The fix must therefore
    live in the income (the motion gate), not in the guard, and CREEP > PARK in the window test
    below is deliberate: near-stillness remains alive and merely underpaid, so the guard keeps
    its S5.5 meaning and no jitter threshold is added for a policy to straddle.
    """
    assert _stall_episode_steps(0.0) == 31  # floor(2.0 s / dt) = 30 stalled steps, fires on 31
    assert _stall_episode_steps(0.04) is None


def test_per_step_income_orders_drive_over_creep_over_park_with_margins() -> None:
    """DRIVE > SLOPPY_DRIVE > CREEP > PARK per live step, each by a real margin.

    Measured values with the default weights: 11.00, 5.05, 2.07, 0.50. The first inequality
    failing means centre-line driving lost its progress stream; the second is the one the
    2026-08-19 run inverted (out-of-gate driving must beat the best risk-free crawl, or
    execution error collapses into creeping); the third is what makes creeping merely
    underpaid rather than punished into the parking trap.
    """
    assert DRIVE_STEP > SLOPPY_STEP + 2.0, f"drive {DRIVE_STEP:.2f} vs sloppy {SLOPPY_STEP:.2f}"
    assert SLOPPY_STEP > CREEP_STEP + 2.0, f"sloppy {SLOPPY_STEP:.2f} vs creep {CREEP_STEP:.2f}"
    assert CREEP_STEP > PARK_STEP + 1.0, f"creep {CREEP_STEP:.2f} vs park {PARK_STEP:.2f}"


def test_window_returns_order_drive_creep_park_die_through_the_real_guard() -> None:
    """Over one 450-step window: DRIVE > CREEP > PARK > DIE, restarts included.

    PARK is scored honestly: repeated 31-step episodes (the real guard's timing), each paying
    the terminal -10, restarted until the window is spent. Under the pre-gate economy this
    ordering was CREEP (2880) > PARK (~2250) > DRIVE-as-achieved (~230 per life), which is the
    collapse. Measured now: 4950 > 930 > 85 > -9.5.
    """
    park_life = _stall_episode_steps(0.0)
    assert park_life is not None
    park_episode = (park_life - 1) * PARK_STEP + _step(0.0, 0.0, 0.0, 0.0, terminated=True)
    lives = _WINDOW // park_life
    park_window = lives * park_episode + (_WINDOW - lives * park_life) * PARK_STEP
    creep_window = _WINDOW * CREEP_STEP
    drive_window = _WINDOW * DRIVE_STEP
    die_now = _step(0.0, 0.0, 0.0, 0.0, terminated=True)

    assert drive_window > 4.0 * creep_window, f"{drive_window:.0f} vs {creep_window:.0f}"
    assert creep_window > 5.0 * park_window, f"{creep_window:.0f} vs {park_window:.0f}"
    assert park_window > 0.0 > die_now, f"{park_window:.0f} vs {die_now:.1f}"


def test_the_motion_gate_saturates_at_driving_speed_and_reorders_nothing_above_it() -> None:
    """At or above 0.3 m/s the gated income equals the old constant bit for bit.

    This is the property that let every pre-existing anti-suicide pin in this file pass
    unchanged: both fixture states move at 0.3 m/s or faster, so their scores did not move.
    """
    for v in (0.3, 0.45, 0.6):
        gated = _step(0.0, 0.0, v, v * _DT)
        out, _ = compute_reward(
            d=torch.tensor([0.0]),
            psi=torch.tensor([0.0]),
            ds=torch.tensor([v * _DT]),
            action=torch.tensor([[0.0, 0.0]]),
            prev_action=torch.tensor([[0.0, 0.0]]),
            body_speed=torch.tensor([v]),
            lane_width=torch.tensor([_LANE_W]),
            terminated=torch.tensor([False]),
            weights=RewardWeights(survival_speed_ref=0.0),  # the gate disabled: the old constant
            control_dt=_DT,
        )
        assert gated == float(out), f"gate moved a driving score at v={v}"


def test_accelerating_out_of_a_slow_bad_state_pays_at_every_step() -> None:
    """The residual risk of the gate, pinned: slow-and-off-lane must escape by SPEEDING UP.

    Gating survival on motion makes a slow bad state (v=0.05, d=0.15, psi=0.6) pay about
    -3.1/step, a regime the old constant hid at +1.1. The state is not a trap for the same
    reason the adversarial stack is not: every unit of recovery pays immediately, and here the
    recovery axis is the throttle itself, at ``survival / speed_ref`` = 16.7 reward per m/s.
    If this monotonicity ever breaks, low-speed despair (choosing the -10 over accelerating)
    becomes rational and the documented fallback is a floored gate
    ``survival * (0.3 + 0.7 * min(v / 0.3, 1))``; see the fix report of 2026-08-20.
    """
    previous = _step(0.15, 0.6, 0.05, 0.0)
    for v in (0.10, 0.15, 0.20, 0.30):
        now = _step(0.15, 0.6, v, 0.0)
        assert now > previous, f"accelerating to {v} m/s did not pay ({now:.2f} <= {previous:.2f})"
        previous = now
    assert _step(0.15, 0.6, 0.3, 0.0) > 0.0, "the measured dying state must stay worth surviving"
