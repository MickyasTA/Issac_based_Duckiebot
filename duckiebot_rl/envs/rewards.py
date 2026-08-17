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

Three degenerate policies the research flagged, and the term that prices each one:

* **Spinning in place.** ``r_prog`` is gated on ``ds > 0`` and ``ds`` is the displacement
  projected on the lane tangent, so a stationary spin earns nothing from it; ``r_heading``
  averages to roughly its minimum over a full turn, ``r_smooth`` pays for the oscillating
  action, and the stall indicator subtracts 0.5 per step outright.
* **Hugging one line.** ``r_lat`` is ``-(1 - 0.001 ** (|d| / (w/2)))``, which is ~-0.999 at the
  lane edge against ~0 at the centre, and the progress gate refuses credit once ``d`` exceeds
  ``(w - W_R) / 2 + 0.02`` so that riding the oncoming lane pays nothing at all.
* **Driving backwards.** ``ds < 0`` fails the same gate, so reversing along the lane collects no
  progress reward while still paying the heading penalty for a ~180 deg heading error.

``tests/unit/test_rewards.py`` rolls each of those three policies out against a real lane graph
and asserts it scores strictly worse than clean lane following, so the claims above are checked
rather than merely written down.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from duckiebot_rl.assets.params import DUCKIEBOT

__all__ = [
    "PROGRESS_GATE_MARGIN_M",
    "PSI_TARGET_LOOKAHEAD_M",
    "PSI_TARGET_MAX_RAD",
    "REWARD_CLIP",
    "RewardTerms",
    "RewardWeights",
    "compute_reward",
    "leaky_cos",
    "psi_target",
    "r_heading",
    "r_lateral",
    "r_progress",
    "r_proximity",
    "r_smooth",
    "stall_indicator",
    "terminal_reward",
]

PSI_TARGET_LOOKAHEAD_M: float = 0.05
"""Lateral error at which the heading target saturates, in metres (S5.4 ``clip(d / 0.05, ...)``)."""

PSI_TARGET_MAX_RAD: float = math.pi / 4.0
"""Largest corrective heading the target ever asks for: 45 degrees."""

PROGRESS_GATE_MARGIN_M: float = 0.02
"""Slack added to the half-clear-lane progress gate, in metres (S5.4)."""

REWARD_CLIP: float = 20.0
"""Symmetric clip applied to the total reward including the terminal penalty (S5.4)."""


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
    """

    heading: float = 1.0
    progress: float = 6.0
    lateral: float = 0.5
    smooth: float = 0.10
    proximity: float = 1.0
    stall: float = 0.5
    terminal: float = -10.0


@dataclass
class RewardTerms:
    """The six per-step terms plus the stall indicator, each ``(N,)`` and UNWEIGHTED.

    Attributes:
        heading: :func:`r_heading`.
        progress: :func:`r_progress`.
        lateral: :func:`r_lateral`.
        smooth: :func:`r_smooth`.
        proximity: :func:`r_proximity`.
        stall: :func:`stall_indicator`, 1.0 where the robot is stalled.
        total: The weighted sum, before the terminal penalty and before the clip.
    """

    heading: torch.Tensor
    progress: torch.Tensor
    lateral: torch.Tensor
    smooth: torch.Tensor
    proximity: torch.Tensor
    stall: torch.Tensor
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
) -> torch.Tensor:
    """Return the gated lane-frame progress reward (SPEC v2 S5.4).

    ``ds / (v_max * dt_c)`` when the robot is moving FORWARD along the lane AND is inside its
    own lane, and 0 otherwise. Normalising by the distance a full-speed step covers makes the
    term at most 1.0 per step regardless of the control rate, so the S5.4 weight of 6.0 means
    the same thing if the decimation ever changes.

    The gate is the direction lock and the oncoming-lane lock in one expression:

    * ``ds > 0`` refuses credit for driving backwards. ``ds`` is the world displacement
      projected on the lane tangent (:func:`duckiebot_rl.city.lane_graph.progress_delta`), so it
      is negative against the lane direction and continuous across tile boundaries.
    * ``d < (lane_width - robot_width) / 2 + margin`` refuses credit once the robot has crossed
      far enough left to be in the oncoming lane. Both widths are parameters, not the v1
      literals 0.115 and 0.065, because V9 randomises the lane width per variant.

    Args:
        ds: ``(N,)`` lane-frame forward progress this control step, in metres.
        d: ``(N,)`` signed lateral error in metres.
        lane_width: ``(N,)`` or scalar clear lane width in metres, the episode's ``w_ep``.
        robot_width: Robot width in metres.
        v_max: Commanded speed cap in m/s.
        control_dt: Control period in seconds.

    Returns:
        ``(N,)`` reward in ``[0, ~1]``.
    """
    width = torch.as_tensor(lane_width, dtype=ds.dtype, device=ds.device)
    gate = 0.5 * (width - robot_width) + PROGRESS_GATE_MARGIN_M
    allowed = (ds > 0.0) & (d < gate)
    return torch.where(allowed, ds / (v_max * control_dt), torch.zeros_like(ds))


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
    progress = r_progress(ds, d, lane_width, robot_width, v_max, control_dt)
    lateral = r_lateral(d, lane_width)
    smooth = r_smooth(action, prev_action)
    if gap is None or prev_gap is None:
        proximity = torch.zeros_like(d)
    else:
        proximity = r_proximity(gap, prev_gap)
    stall = stall_indicator(body_speed, stall_speed)

    total = (
        weights.heading * heading
        + weights.progress * progress
        + weights.lateral * lateral
        + weights.smooth * smooth
        + weights.proximity * proximity
        - weights.stall * stall
    )
    terms = RewardTerms(
        heading=heading,
        progress=progress,
        lateral=lateral,
        smooth=smooth,
        proximity=proximity,
        stall=stall,
        total=total,
    )
    reward = total if terminated is None else total + terminal_reward(terminated, weights)
    return torch.clamp(reward, -clip, clip), terms
