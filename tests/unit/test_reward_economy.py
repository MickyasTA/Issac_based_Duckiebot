"""The reward economy: dying must never pay better than driving, badly or otherwise.

The regression this file pins reached a live run on 2026-08-18. Once the lane matcher started
reporting honest ``d``, an untrained policy paid -2.0 to -4.4 per step while wandering, against
a one-off terminal penalty of -10. Terminating 400 steps early therefore SAVED roughly 800
reward, and PPO found that optimum in under a million steps: mean episode length fell from 72
to 44 steps while returns rose from -316 to -88, and the eval metric pinned flat. No reward
gradient was wrong; the LEVEL was, which is a property none of the per-term tests looked at.
"""

from __future__ import annotations

import torch

from duckiebot_rl.envs.rewards import RewardWeights, compute_reward

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
    """The survival term is a constant shift: it must not reorder driving styles."""
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
