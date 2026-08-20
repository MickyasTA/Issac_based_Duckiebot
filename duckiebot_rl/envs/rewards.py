"""The SPEC v2 S5.4 reward, one documented function per term, vectorised over ``N`` envs.

Pure torch: no Isaac import anywhere, so every term is unit-testable on CPU against a
hand-computed fixture. The environment supplies ``d``, ``psi`` and ``ds`` from
:class:`duckiebot_rl.city.lane_graph.BatchedLaneGraph`; nothing here re-derives lane geometry.

Sign conventions (SPEC v2 S2, resolving critic item H)
------------------------------------------------------

* ``d > 0`` means the robot is LEFT of the right-lane centreline, i.e. toward the yellow centre
  tape. ``d < 0`` means toward the white outer edge.
* ``psi > 0`` means the heading is rotated counter-clockwise (left) of the lane tangent.
* With those, ``psi_target = -clip(d / 0.05, -1, 1) * 45 deg`` steers back toward the centreline
  for both signs, which is the property the whole heading term rests on.

The lane width is a parameter, never a literal (critic item H)
--------------------------------------------------------------

DR axis V9 randomises the clear lane width over ``U(0.17, 0.28)`` m and the centreline offset by
+/-15 mm, so the v1 constants ``0.115`` and ``0.065`` were wrong on essentially every randomised
episode: they encode half of a 0.23 m lane and half of the robot. Both are arguments here.
``lane_width`` is the per-env value from ``BatchedLaneGraph.lane_width[variant_idx]``, which is
derived from the variant's own texture bucket, and ``robot_width`` defaults to the single source
of truth in :mod:`duckiebot_rl.assets.params`.

Reward hacking this design is meant to make unprofitable
--------------------------------------------------------

Four degenerate policies that reached (or nearly reached) a live run, and the term that prices
each one:

* **Spinning in place.** ``r_prog`` is gated on ``ds > 0`` and ``ds`` is the displacement
  projected on the lane tangent, so a stationary spin earns nothing from it; ``r_heading``
  averages to roughly its minimum over a full turn, ``r_smooth`` pays for the oscillating
  action, and the stall indicator subtracts 0.5 per step outright.
* **Hugging one line.** ``r_lat`` is ``-(1 - 0.001 ** (|d| / (w/2)))``, which is ~-0.999 at the
  lane edge against ~0 at the centre, and the progress gate refuses credit once ``d`` exceeds
  ``(w - W_R) / 2 + 0.02`` so that riding the oncoming lane pays nothing at all.
* **Driving backwards.** ``ds < 0`` fails the same gate, so reversing along the lane collects no
  progress reward while still paying the heading penalty for a ~180 deg heading error.
* **Parking or creeping.** The survival income is scaled by :func:`r_survival`, which is
  ``clamp(|v| / survival_speed_ref, 0, 1)``, so a stationary robot collects almost none of it
  and the stall indicator subtracts 0.5 on top, while a barely-above-stall creep collects a
  sliver against full-speed driving's whole stream. This one is not hypothetical: with the
  survival term UNGATED, the 2026-08-19 run measured parking at +5.18/step risk-free and a
  0.04 m/s creep at +6.40/step forever (the creep resets the 2 s stall guard every step), while
  out-of-gate full-speed driving paid +5.05/step. PPO found the inversion between iterations
  600 and 700 and every env stopped driving within ~50 iterations.

``tests/unit/test_rewards.py`` rolls the first three policies out against a real lane graph and
asserts each scores strictly worse than clean lane following; ``tests/unit/test_reward_economy.py``
pins the fourth with archetype rollouts through the REAL stall guard, asserting the full income
ordering DRIVE > CREEP > PARK > DIE with margins. The claims above are checked rather than merely
written down.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from duckiebot_rl.assets.params import DUCKIEBOT

__all__ = [
    "LANE_DEPARTURE_CAP",
    "PROGRESS_GATE_MARGIN_M",
    "PSI_TARGET_LOOKAHEAD_M",
    "PSI_TARGET_MAX_RAD",
    "REWARD_CLIP",
    "RewardTerms",
    "RewardWeights",
    "clear_half_lane",
    "compute_reward",
    "leaky_cos",
    "psi_target",
    "r_heading",
    "r_lane_departure",
    "r_lateral",
    "r_progress",
    "r_proximity",
    "r_smooth",
    "r_survival",
    "stall_indicator",
    "terminal_reward",
    "wrong_lane_indicator",
]

PSI_TARGET_LOOKAHEAD_M: float = 0.05
"""Lateral error at which the heading target saturates, in metres (S5.4 ``clip(d / 0.05, ...)``)."""

PSI_TARGET_MAX_RAD: float = math.pi / 4.0
"""Largest corrective heading the target ever asks for: 45 degrees."""

PROGRESS_GATE_MARGIN_M: float = 0.02
"""Slack added to the half-clear-lane progress gate, in metres (S5.4)."""

REWARD_CLIP: float = 20.0
"""Symmetric clip applied to the total reward including the terminal penalty (S5.4)."""

LANE_DEPARTURE_CAP: float = 3.0
"""Ceiling of :func:`r_lane_departure`, in half-lane widths beyond the clear lane.

The point of the term is to have a live gradient everywhere the robot can physically be, so the
cap is placed OUTSIDE the reachable set rather than at the lane edge. Measured on ``city_000``
(``tile_size`` 0.570 m, ``w_ep`` 0.2046 m) the widest on-road excursion from a lane centreline
is about 0.25 m on a curve tile, which is 2.44 half-lane widths; beyond 3.0 the off-drivable
termination has already fired. Contrast :func:`r_lateral`, whose bound is reached AT the lane
edge and which is therefore numerically flat over the entire out-of-lane region.
"""


@dataclass(frozen=True)
class RewardWeights:
    """The SPEC v2 S5.4 weights.

    Kept as a frozen dataclass rather than module constants so that an ablation config can build
    a modified copy without mutating global state, and so that the corner-cutting escalation the
    spec pre-commits to (raise ``lateral`` from 0.5 to 1.0 if the eval median time-integrated
    ``|d|`` exceeds 0.04 m*s per episode-second) is a one-line, logged config change rather than
    an edit to this file.

    Attributes:
        heading: Weight of :func:`r_heading`.
        progress: Weight of :func:`r_progress`.
        lateral: Weight of :func:`r_lateral`.
        smooth: Weight of :func:`r_smooth`.
        proximity: Weight of :func:`r_proximity`.
        stall: Weight SUBTRACTED for each stalled step.
        terminal: Penalty added on a true termination (collision or off-drivable). Never applied
            on truncation, which is bootstrapped instead (S6.4).
        departure: Weight of :func:`r_lane_departure`, the lane-discipline fix (see below).
        wrong_lane: Weight SUBTRACTED while :func:`wrong_lane_indicator` fires.
        survival: Income ADDED on every live step, scaled by the :func:`r_survival` motion gate,
            the fix for the suicide equilibrium the continuity-constrained matcher exposed
            (2026-08-18). Under the free matcher an untrained policy paid about -0.71 per step,
            because re-homing onto the nearest lane forgave excursions; with honest ``d`` the
            same wandering costs -2.0 to -4.4 per step, while dying costs -10 once. Terminating
            400 steps early then SAVES roughly 800 reward, and the run learned exactly that:
            mean episode length fell from 72 to 44 steps, returns "improved" from -316 to -88 by
            dying sooner, and eval distance pinned at 1.8 tiles for 900 iterations. 5.0 exceeds
            the worst SUSTAINED wandering cost that was measured (4.4), so every recoverable
            state is worth surviving, while the momentary worst states (departure at its cap on
            the way off the road) stay net negative, which is the correct residual pressure.

            Why the term is motion-gated (2026-08-20). Paid as an unconditional constant, this
            term created the next equilibrium: a risk-free annuity for not driving. Measured
            through this very function on the 2026-08-19 run's parameters: PARK earned +5.18 per
            step until the 2 s stall guard fired (episode +160.5, restartable forever), a
            0.04 m/s CREEP earned +6.40 per step with NO termination ever (one step at or above
            0.03 m/s resets the stall counter, so any epsilon above it is guard-proof), while
            full-speed driving at ``d`` = 8 cm, outside the progress gate, earned +5.05. The
            best risk-free policy in the MDP was creeping, the run converged to exactly that
            between iterations 600 and 700 (reward/stall 0.0 to 1.0, returns 233 to 1656, eval
            distance 1.62 to 0.02 tiles), and the actor's vision pathway died with nothing left
            to see. Scaling the income by ``clamp(|v| / survival_speed_ref, 0, 1)`` repairs the
            ordering with margins, measured at the archetypes: DRIVE 11.0-12.0/step at or above
            0.3 m/s (bit-identical to the ungated economy there, so no driving preference is
            reordered), CREEP 2.07/step, PARK 0.5/step, DIE -10 once. Even out-of-gate driving
            (5.05) now beats creeping, so the dominance survives execution error.
        survival_speed_ref: Body speed, in m/s, at which the survival income saturates to its
            full weight; half of ``DUCKIEBOT.v_cmd_max_m_s``. Values at or below zero disable
            the gate and reproduce the unconditional pre-2026-08-20 term for ablations (that is
            the economy that trained the parked policy of
            ``20260819T031011Z_lanefollow_seed0_survival_seed0``; see
            ``tests/unit/test_reward_economy.py`` before reaching for it).
        two_sided_progress_gate: Gate :func:`r_progress` on ``|d|`` rather than on signed ``d``.

    The lane-discipline fix (2026-08-18) and how to undo it
    -------------------------------------------------------

    The v1 weights above let a policy leave its lane and keep almost all of its income. Measured
    on ``city_000`` by sweeping the robot sideways across a lane and evaluating this reward at
    each offset (the on-centre step reward is +7.00):

    * On a CURVE tile the robot can sit 0.25 m from the centreline - 2.44 half-lane widths,
      squarely across the white edge line - and still be on drivable tiles, still collect the
      FULL 6.0-weighted progress term, and pay a total of only +4.25 per step. That is 61 % of
      the on-centre reward for being completely out of the lane.
    * The reason is two independent holes. ``r_progress`` gated on signed ``d < gate``, which
      locks the yellow/oncoming side only and never fires for the white/outer side; and
      ``r_lateral`` saturates AT the lane edge, its gradient falling from -6.8e-02 /m at one
      half-lane width to -6.8e-08 /m at three, so nothing pulls the robot back once it is out.
    * The training run of 2026-08-18 shows the consequence: over iterations 0 to 281 the mean
      episode ``out_of_lane_integral_ms`` ROSE from 0.058 to 0.173 while the return rose from
      -6.8 to +1776. More training was buying lane indiscipline, so a longer run alone would
      have made the reported symptom worse rather than better.

    :meth:`legacy` returns the exact pre-fix weight set, so an ablation can reproduce the old
    behaviour and the two can be compared on the same gates.
    """

    heading: float = 1.0
    progress: float = 6.0
    lateral: float = 0.5
    smooth: float = 0.10
    proximity: float = 1.0
    stall: float = 0.5
    terminal: float = -10.0
    departure: float = 2.0
    wrong_lane: float = 2.0
    survival: float = 5.0
    survival_speed_ref: float = 0.3
    two_sided_progress_gate: bool = True

    @classmethod
    def legacy(cls) -> RewardWeights:
        """Return the pre-2026-08-18 weights, with the lane-discipline terms disabled.

        Both new terms are zero-weighted and the progress gate reverts to its one-sided form, so
        :func:`compute_reward` reproduces the v1 reward exactly. Use it to re-run an ablation
        against the checkpoints trained before the fix.

        Returns:
            A weight set that scores identically to the reward used up to iteration 281 of
            ``20260818T034543Z_lanefollow_seed0_main_seed0``.
        """
        return cls(departure=0.0, wrong_lane=0.0, two_sided_progress_gate=False, survival=0.0)


@dataclass
class RewardTerms:
    """The per-step terms and indicators, each ``(N,)`` and UNWEIGHTED.

    Attributes:
        heading: :func:`r_heading`.
        progress: :func:`r_progress`.
        lateral: :func:`r_lateral`.
        smooth: :func:`r_smooth`.
        proximity: :func:`r_proximity`.
        stall: :func:`stall_indicator`, 1.0 where the robot is stalled.
        departure: :func:`r_lane_departure`.
        wrong_lane: :func:`wrong_lane_indicator`, 1.0 where the robot is in the oncoming lane.
        survival: :func:`r_survival`, the motion gate on the survival income in ``[0, 1]``.
            Logged since 2026-08-20 because it is no longer a constant: its fleet mean is the
            cheapest parking alarm there is (1.0 while driving, near 0 while the fleet idles).
        total: The weighted sum, before the terminal penalty and before the clip.
    """

    heading: torch.Tensor
    progress: torch.Tensor
    lateral: torch.Tensor
    smooth: torch.Tensor
    proximity: torch.Tensor
    stall: torch.Tensor
    departure: torch.Tensor
    wrong_lane: torch.Tensor
    survival: torch.Tensor
    total: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        """Return the terms keyed by name, for the S6.8 per-term reward logging.

        Returns:
            ``{term_name: tensor}`` for all seven fields plus ``total``.
        """
        return dict(self.__dict__)


# =============================================================================================
# Individual terms
# =============================================================================================


def leaky_cos(x: torch.Tensor) -> torch.Tensor:
    """Return ``cos(x)`` inside one half-period and a gentle linear ramp outside it.

    SPEC v2 S5.4 defines ``leaky_cos(x) = cos(x) if |x| < pi else -1 - 0.05 * (|x| - pi)``. The
    leak matters: a plain cosine is periodic, so a heading error of ``2 * pi / scale`` would
    score as well as no error at all and a policy could be rewarded for facing backwards. The
    linear tail makes the term monotone in ``|x|`` for every reachable error.

    Args:
        x: Scaled heading error, any shape.

    Returns:
        A tensor of the same shape.
    """
    absolute = x.abs()
    return torch.where(absolute < math.pi, torch.cos(x), -1.0 - 0.05 * (absolute - math.pi))


def psi_target(d: torch.Tensor) -> torch.Tensor:
    """Return the heading the robot should hold given its lateral error.

    ``psi_target = -clip(d / 0.05, -1, 1) * 45 deg``. With the S2 sign convention (``d > 0`` is
    left of the centreline, toward yellow) a positive ``d`` asks for a negative, i.e. rightward,
    heading, which points back at the centreline. The clip saturates the request at 45 degrees
    once the robot is more than 5 cm off, so a large excursion does not demand an impossible
    heading and then punish the policy for not achieving it.

    Args:
        d: ``(N,)`` signed lateral error in metres.

    Returns:
        ``(N,)`` target heading error in radians.
    """
    return -torch.clamp(d / PSI_TARGET_LOOKAHEAD_M, -1.0, 1.0) * PSI_TARGET_MAX_RAD


def r_heading(d: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    """Return the two-scale heading reward (SPEC v2 S5.4).

    ``0.5 * (leaky_cos(pi * e / 10 deg) + leaky_cos(pi * e / 50 deg))`` where
    ``e = psi - psi_target(d)``. The narrow 10 degree lobe gives a sharp peak that rewards
    precise tracking; the wide 50 degree lobe keeps a usable gradient out to large errors so the
    policy is not learning from a flat landscape early on. Adapted from the AI-DO 5/6
    lane-following entry of Kalapos, Gor, Moni and Harmati (ISMCR 2020, arXiv:2012.07461;
    ACTA IMEKO 10(3):7-14, 2021).

    Args:
        d: ``(N,)`` signed lateral error in metres.
        psi: ``(N,)`` heading error in radians, positive counter-clockwise of the lane tangent.

    Returns:
        ``(N,)`` reward in ``(-inf, 1]``; exactly 1.0 when the heading matches the target.
    """
    error = psi - psi_target(d)
    narrow = leaky_cos(error * (math.pi / math.radians(10.0)))
    wide = leaky_cos(error * (math.pi / math.radians(50.0)))
    return 0.5 * (narrow + wide)


def r_progress(
    ds: torch.Tensor,
    d: torch.Tensor,
    lane_width: torch.Tensor | float,
    robot_width: float = DUCKIEBOT.robot_width_m,
    v_max: float = DUCKIEBOT.v_cmd_max_m_s,
    control_dt: float = DUCKIEBOT.control_dt_s,
    two_sided: bool = True,
    gate_scale: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    """Return the gated lane-frame progress reward (SPEC v2 S5.4).

    ``ds / (v_max * dt_c)`` when the robot is moving FORWARD along the lane AND is inside its
    own lane, and 0 otherwise. Normalising by the distance a full-speed step covers makes the
    term at most 1.0 per step regardless of the control rate, so the S5.4 weight of 6.0 means
    the same thing if the decimation ever changes.

    The gate is the direction lock and the lane lock in one expression:

    * ``ds > 0`` refuses credit for driving backwards. ``ds`` is the world displacement
      projected on the lane tangent (:func:`duckiebot_rl.city.lane_graph.progress_delta`), so it
      is negative against the lane direction and continuous across tile boundaries.
    * ``|d| < (lane_width - robot_width) / 2 + margin`` refuses credit once the robot has left
      its own lane. Both widths are parameters, not the v1 literals 0.115 and 0.065, because V9
      randomises the lane width per variant.

    Why the gate is now two-sided (2026-08-18)
    -------------------------------------------

    It used to test signed ``d < gate``, on the reasoning that the yellow side is the oncoming
    lane while the white side is the shoulder, which the off-drivable termination already
    guards. Measurement says the shoulder is NOT guarded on curve tiles: a curve carries enough
    drivable area that the robot can sit 0.25 m from its lane centreline - 2.44 half-lane widths
    - with all four off-drivable test points still on the road, and the one-sided gate paid it
    the full 1.0 progress the whole way out. Straights do behave as the old comment assumed
    (on-road only to 0.98 half-lane widths), which is why the hole stayed invisible: it opens
    exactly on the curves, which is exactly where a lane-follower drifts wide.

    Args:
        ds: ``(N,)`` lane-frame forward progress this control step, in metres.
        d: ``(N,)`` signed lateral error in metres.
        lane_width: ``(N,)`` or scalar clear lane width in metres, the episode's ``w_ep``.
        robot_width: Robot width in metres.
        v_max: Commanded speed cap in m/s.
        control_dt: Control period in seconds.
        two_sided: Gate on ``|d|``. False restores the pre-fix one-sided gate.
        gate_scale: Multiplier on the gate width, a manual ablation knob (2026-08-19). Nothing
            anneals it: the planned env-side curriculum was evaluated and NOT wired
            (2026-08-20 decision, below). Both training and evaluation run at 1.0 by default.

            Why the knob was added, with the measurement that motivated it: the spec gate is
            ``(0.2046 - 0.13) / 2 + 0.02 = 0.057 m``. A fresh pixels policy wobbles outside a
            5.7 cm band nearly always, so the one meaningful positive term (weight 6.0) almost
            never pays and the visual task was thought to get no dense signal: after 1M honest
            steps the run sat at lane_dev_rms 0.35 m flat, 100% off-drivable endings, eval
            pinned near 1.8 tiles, which is "drive straight to the first curve and fall off".

            Why no anneal was wired: the autopsy of the 2026-08-19 run showed the sparsity
            diagnosis was incomplete. At gate_scale 1.0 that run's healthy phase still improved
            eval distance from 0 to 1.62-1.9 tiles by iteration 500-600, so density at 1.0 was
            sufficient to learn; what killed the run was the unconditional survival income
            making not-driving the best risk-free policy (see :func:`r_survival`). With
            survival motion-gated, a moving policy earns a dense speed-proportional stream even
            with zero progress income, which is the early-training density the anneal was meant
            to provide, minus a second moving part. The archetype economy in
            ``tests/unit/test_reward_economy.py`` is pinned at gate_scale 1.0; an anneal would
            silently re-weigh those margins mid-run. The knob stays for ablations; if it is
            ever driven by a schedule, note that even at 3.0 (0.170 m) the opposing lane's
            centreline at 0.234 m stays OUTSIDE the band, the continuity-constrained matcher
            keeps ``d`` truthful, and the wrong-lane flag still fires, so the corner-cutting
            exploit cannot return; evaluation must stay at 1.0.

    Returns:
        ``(N,)`` reward in ``[0, ~1]``.
    """
    gate = clear_half_lane(lane_width, robot_width, like=ds) * gate_scale
    offset = d.abs() if two_sided else d
    allowed = (ds > 0.0) & (offset < gate)
    return torch.where(allowed, ds / (v_max * control_dt), torch.zeros_like(ds))


def clear_half_lane(
    lane_width: torch.Tensor | float,
    robot_width: float = DUCKIEBOT.robot_width_m,
    like: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return ``(lane_width - robot_width) / 2 + margin``: the offset where the robot reaches a lane line.

    Single source of the expression so :func:`r_progress` and :func:`r_lane_departure` cannot
    drift apart: the offset at which progress stops paying is by construction the same offset at
    which the departure penalty starts charging, and the reward therefore has no dead band
    between them.

    Args:
        lane_width: ``(N,)`` or scalar clear lane width in metres.
        robot_width: Robot width in metres.
        like: Tensor supplying dtype and device; defaults to those of ``lane_width``.

    Returns:
        ``(N,)`` or scalar half-width in metres.
    """
    if like is None:
        width = torch.as_tensor(lane_width)
    else:
        width = torch.as_tensor(lane_width, dtype=like.dtype, device=like.device)
    return 0.5 * (width - robot_width) + PROGRESS_GATE_MARGIN_M


def r_lane_departure(
    d: torch.Tensor,
    lane_width: torch.Tensor | float,
    robot_width: float = DUCKIEBOT.robot_width_m,
    cap: float = LANE_DEPARTURE_CAP,
) -> torch.Tensor:
    """Return a LINEAR penalty for how far outside its own lane the robot is.

    ``-min(clip(|d| - clear_half_lane, 0) / (w_ep / 2), cap)``: zero inside the lane, then one
    unit of penalty per half-lane width of excursion, capped at ``cap`` half-lane widths.

    This is the term that gives the out-of-lane region a gradient. :func:`r_lateral` cannot: its
    exponential reaches -0.999 at exactly one half-lane width, so measured on ``city_000`` its
    slope is -6.8e-02 /m at the lane edge and -6.8e-08 /m at three half-lane widths. A policy
    already out of its lane sees a numerically flat landscape and has nothing to descend. The
    two terms are complementary and both are kept: the bounded exponential still does the job it
    was designed for, which is not to dominate the return during the early flailing phase, while
    this term supplies the restoring force outside the lane where the exponential has none.

    Linear rather than quadratic on purpose. The excursion is bounded by the road, so a
    quadratic buys nothing at the far end but does flatten the gradient near the lane edge,
    which is the region the policy has to learn first.

    Args:
        d: ``(N,)`` signed lateral error in metres.
        lane_width: ``(N,)`` or scalar clear lane width in metres.
        robot_width: Robot width in metres.
        cap: Ceiling in half-lane widths; see :data:`LANE_DEPARTURE_CAP`.

    Returns:
        ``(N,)`` penalty in ``[-cap, 0]``.
    """
    width = torch.as_tensor(lane_width, dtype=d.dtype, device=d.device)
    over = torch.clamp(d.abs() - clear_half_lane(width, robot_width, like=d), min=0.0)
    return -torch.clamp(over / (0.5 * width), max=cap)


def wrong_lane_indicator(psi: torch.Tensor) -> torch.Tensor:
    """Return 1.0 where the robot is matched to a lane it is travelling AGAINST.

    Why this exists, and why it is not redundant with ``d``
    -------------------------------------------------------

    :class:`duckiebot_rl.city.lane_graph.BatchedLaneGraph` matches a pose to the nearest lane
    over ALL lanes, with only a 0.002 m^2 heading tie-break. That is correct for a robot in its
    own lane, but it means a robot that has fully crossed the yellow line is re-matched to the
    ONCOMING lane, and every quantity derived from that match silently changes meaning: measured
    on ``city_000``, sweeping across the centreline flips ``seg_id`` from 2 to 3 and collapses
    ``|d|`` from 0.103 m back to 0.003 m. So ``|d|`` reports a robot squarely in the wrong lane
    as being beautifully centred, ``r_lateral`` charges it almost nothing, and
    ``episode/out_of_lane_integral_ms`` - which is ``clip(|d| - w/2, 0)`` integrated - stops
    accumulating entirely. The metric is blind to exactly the failure it is named after.

    ``psi`` is the one quantity the re-match does NOT hide: it flips to +/-180 degrees, because
    the oncoming lane's tangent points the other way. Testing ``|psi| > 90 degrees`` therefore
    detects the crossing that ``d`` conceals, and costs one comparison.

    The current training maps are Hamiltonian loops with no intersections, so no legitimate
    manoeuvre holds ``|psi| > 90 deg``. On a map with 3- or 4-way tiles a mid-turn pose can
    momentarily match a crossing lane; treat this as an instantaneous penalty only, never as a
    termination, unless it is first backed by a sustained counter.

    Args:
        psi: ``(N,)`` heading error in radians against the matched lane.

    Returns:
        ``(N,)`` float, 1.0 where the robot faces against its matched lane.
    """
    return (psi.abs() > 0.5 * math.pi).to(psi.dtype)


def r_lateral(d: torch.Tensor, lane_width: torch.Tensor | float) -> torch.Tensor:
    """Return the bounded lateral-error penalty (SPEC v2 S5.4).

    ``-(1 - 0.001 ** (|d| / (w_ep / 2)))``, which is 0 on the centreline and asymptotically
    -1 far outside the lane. The exponential shape is what keeps the term bounded: a quadratic
    penalty would dominate the return during the early flailing phase and a policy that has not
    yet learned to move would be optimised toward standing still on the centreline.

    Args:
        d: ``(N,)`` signed lateral error in metres.
        lane_width: ``(N,)`` or scalar clear lane width in metres.

    Returns:
        ``(N,)`` penalty in ``(-1, 0]``.
    """
    width = torch.as_tensor(lane_width, dtype=d.dtype, device=d.device)
    normalised = d.abs() / (0.5 * width)
    return -(1.0 - torch.pow(torch.full_like(normalised, 0.001), normalised))


def r_smooth(action: torch.Tensor, prev_action: torch.Tensor) -> torch.Tensor:
    """Return the action-smoothness penalty (SPEC v2 S5.4).

    ``-||a - a_prev||^2`` over the 2-D action. Both arguments are the CLIPPED actions the
    environment executed, not the unclipped Gaussian samples the buffer stores: penalising the
    unclipped tail would charge the policy for exploration it never actuated.

    Args:
        action: ``(N, 2)`` action executed this step.
        prev_action: ``(N, 2)`` action executed last step.

    Returns:
        ``(N,)`` penalty, at most 0.
    """
    return -torch.sum((action - prev_action) ** 2, dim=-1)


def r_proximity(gap: torch.Tensor, prev_gap: torch.Tensor) -> torch.Tensor:
    """Return the obstacle-recovery reward (SPEC v2 S5.4).

    ``clip(-(p_prev - p) * 50, 0, 1.5)`` where ``p = min(gap, 0)`` is the safety-circle OVERLAP,
    so the term is nonzero only while an overlap exists and only while it is shrinking. Feeding
    the raw signed clearance instead would pay up to 1.5 per step for moving away from an
    obstacle anywhere on the map, which is comparable to the 6.0-weighted progress term and
    turns a collision-recovery term into a general keep-away bonus. The MuJoCo twin
    (``duckiebot_rl/sim2sim/env.py``) applies the same ``min(gap, 0)``; the two must agree or
    the C0-vs-C5 return delta stops being a transfer measurement.

    An infinite gap (an obstacle-free map) yields exactly 0.

    Args:
        gap: ``(N,)`` signed clearance to the nearest safety circle this step, in metres.
        prev_gap: ``(N,)`` the same quantity last step.

    Returns:
        ``(N,)`` reward in ``[0, 1.5]``.
    """
    zero = torch.zeros_like(gap)
    overlap = torch.minimum(gap, zero)
    overlap_prev = torch.minimum(prev_gap, zero)
    value = torch.clamp(-(overlap_prev - overlap) * 50.0, 0.0, 1.5)
    finite = torch.isfinite(gap) & torch.isfinite(prev_gap)
    return torch.where(finite, value, zero)


def stall_indicator(body_speed: torch.Tensor, stall_speed: float = 0.03) -> torch.Tensor:
    """Return 1.0 where the robot is stalled (SPEC v2 S5.4).

    The threshold is on ``|v_body|``, not on the signed speed, so reversing at 0.2 m/s does not
    count as a stall and is penalised by the progress gate instead.

    Args:
        body_speed: ``(N,)`` body-frame forward speed in m/s.
        stall_speed: Speed below which the robot counts as stalled.

    Returns:
        ``(N,)`` float indicator.
    """
    return (body_speed.abs() < stall_speed).to(body_speed.dtype)


def r_survival(body_speed: torch.Tensor, speed_ref: float = 0.3) -> torch.Tensor:
    """Return the motion gate on the survival income: ``clamp(|v| / speed_ref, 0, 1)``.

    The survival term exists so that death forfeits future reward instead of harvesting it
    (2026-08-18). Paid unconditionally it overshot: a robot that simply did not drive kept the
    income at zero risk, and the 2026-08-19 run cashed that in, converging to a sub-stall-guard
    creep in ~50 iterations once one env found it (see :class:`RewardWeights.survival`). This
    gate makes the income proportional to motion below ``speed_ref`` and exactly the old
    constant at or above it, so:

    * living trajectories at driving speed are scored bit-identically to the ungated term, and
      no preference between driving styles moves;
    * a parked robot earns ~0 survival and still pays the stall indicator, so the parking
      annuity is gone;
    * a creeping robot earns income linear in its speed, which measured at the archetypes puts
      creeping (2.07/step at 0.04 m/s) far below even out-of-gate full-speed driving
      (5.05/step);
    * from a slow bad state, accelerating recovers income at ``survival / speed_ref`` per m/s
      (16.7 with the default weights), so the escape gradient points toward moving, and the
      anti-suicide property is carried by that gradient exactly as the adversarial-stack pin in
      ``tests/unit/test_reward_economy.py`` demands.

    The magnitude is used, like :func:`stall_indicator`, so reversing still counts as motion:
    the progress gate already refuses to pay it, and a recoverable reversing state must not read
    as worth abandoning.

    Args:
        body_speed: ``(N,)`` body-frame forward speed in m/s.
        speed_ref: Speed at which the gate saturates at 1.0. At or below zero the gate is
            disabled and the pre-2026-08-20 unconditional term is reproduced (ablations only).

    Returns:
        ``(N,)`` gate in ``[0, 1]``.
    """
    if speed_ref <= 0.0:
        return torch.ones_like(body_speed)
    return torch.clamp(body_speed.abs() / speed_ref, max=1.0)


def terminal_reward(
    terminated: torch.Tensor,
    weights: RewardWeights = RewardWeights(),
) -> torch.Tensor:
    """Return the terminal penalty, applied on termination only (SPEC v2 S5.4).

    ``R_terminal = -10`` on collision or off-drivable, and ``0`` on truncation. Truncation is a
    harness artifact and is bootstrapped from the captured terminal observation instead
    (S6.4); charging it a penalty would teach the critic that the world ends at 30 seconds.

    Args:
        terminated: ``(N,)`` bool; True where the MDP truly ended.
        weights: Weight set carrying the ``terminal`` scalar.

    Returns:
        ``(N,)`` float penalty.
    """
    return terminated.to(torch.float32) * weights.terminal


# =============================================================================================
# The assembled reward
# =============================================================================================


def compute_reward(
    d: torch.Tensor,
    psi: torch.Tensor,
    ds: torch.Tensor,
    action: torch.Tensor,
    prev_action: torch.Tensor,
    body_speed: torch.Tensor,
    lane_width: torch.Tensor | float,
    gap: torch.Tensor | None = None,
    prev_gap: torch.Tensor | None = None,
    terminated: torch.Tensor | None = None,
    weights: RewardWeights = RewardWeights(),
    progress_gate_scale: float | torch.Tensor = 1.0,
    robot_width: float = DUCKIEBOT.robot_width_m,
    v_max: float = DUCKIEBOT.v_cmd_max_m_s,
    control_dt: float = DUCKIEBOT.control_dt_s,
    stall_speed: float = 0.03,
    clip: float = REWARD_CLIP,
) -> tuple[torch.Tensor, RewardTerms]:
    """Evaluate the full SPEC v2 S5.4 reward for one control step.

    Order of operations matters and is fixed by the spec: the six per-step terms are weighted
    and summed, the terminal penalty is ADDED, and only then is the result clipped to
    ``[-20, +20]``. Clipping before the penalty would let a -10 collision be absorbed by a
    large positive step reward, which is exactly the case the clip exists to bound.

    Args:
        d: ``(N,)`` signed lateral error in metres.
        psi: ``(N,)`` heading error in radians.
        ds: ``(N,)`` lane-frame forward progress this step, in metres.
        action: ``(N, 2)`` clipped action executed this step.
        prev_action: ``(N, 2)`` clipped action executed last step.
        body_speed: ``(N,)`` body-frame forward speed in m/s.
        lane_width: ``(N,)`` or scalar clear lane width, the episode's ``w_ep``.
        gap: ``(N,)`` signed clearance to the nearest safety circle, or None on an
            obstacle-free map (equivalent to ``+inf``).
        prev_gap: ``(N,)`` the same quantity last step, or None.
        terminated: ``(N,)`` bool termination flag, or None to skip the terminal penalty.
        progress_gate_scale: Ablation multiplier on the progress gate width; see
            :func:`r_progress`. No schedule drives it; training and evaluation both run at the
            default 1.0.
        weights: The S5.4 weight set.
        robot_width: Robot width in metres.
        v_max: Commanded speed cap in m/s.
        control_dt: Control period in seconds.
        stall_speed: Stall threshold in m/s.
        clip: Symmetric clip on the final reward.

    Returns:
        ``(reward, terms)`` where ``reward`` is ``(N,)`` after the terminal penalty and the clip,
        and ``terms`` holds the UNWEIGHTED per-step terms for the S6.8 diagnostics.
    """
    heading = r_heading(d, psi)
    progress = r_progress(
        ds,
        d,
        lane_width,
        robot_width,
        v_max,
        control_dt,
        weights.two_sided_progress_gate,
        gate_scale=progress_gate_scale,
    )
    lateral = r_lateral(d, lane_width)
    smooth = r_smooth(action, prev_action)
    if gap is None or prev_gap is None:
        proximity = torch.zeros_like(d)
    else:
        proximity = r_proximity(gap, prev_gap)
    stall = stall_indicator(body_speed, stall_speed)
    departure = r_lane_departure(d, lane_width, robot_width)
    wrong_lane = wrong_lane_indicator(psi)
    survival = r_survival(body_speed, weights.survival_speed_ref)

    total = (
        weights.heading * heading
        + weights.progress * progress
        + weights.lateral * lateral
        + weights.smooth * smooth
        + weights.proximity * proximity
        - weights.stall * stall
        + weights.departure * departure
        - weights.wrong_lane * wrong_lane
        + weights.survival * survival
    )
    terms = RewardTerms(
        heading=heading,
        progress=progress,
        lateral=lateral,
        smooth=smooth,
        proximity=proximity,
        stall=stall,
        departure=departure,
        wrong_lane=wrong_lane,
        survival=survival,
        total=total,
    )
    reward = total if terminated is None else total + terminal_reward(terminated, weights)
    return torch.clamp(reward, -clip, clip), terms
