"""The decimation window of ``DirectRLEnv.step``, expressed as an Isaac-free plan.

Why this module exists
----------------------

:class:`~duckiebot_rl.envs.lane_follow_env.DuckiebotLaneFollowEnv` overrides
``DirectRLEnv.step`` for one reason only: the base class runs the whole actuation write path
once per PHYSICS step, and this robot's actuation is constant across the decimation window by
design. See ``_apply_action``: the deployed ``car_cmd_switch_node`` publishes at 15 Hz and the
motor controller holds the last command, which is exactly why ``decimation`` is 16 and the
target is not re-derived at 240 Hz.

The measured cost of that repetition, at ``N=64`` on the campaign machine, was

* ``scene.write_data_to_sim`` 1.75 ms x 16 = 39.1 ms per control step,
* ``scene.update`` 0.76 ms x 16 = 17.0 ms per control step,

against a 340.7 ms control step: 16.5% of wall time spent writing sixteen identical buffers and
advancing sixteen lazy-buffer timestamps, only the last of which is ever read.

``lane_follow_env.py`` cannot be imported without Kit, so the policy of which operation runs on
which substep lives here instead, as a pure function of integers. That is what makes the hoist
testable on CPU: :func:`window_ops` is compared op-for-op against :func:`baseline_window_ops`,
which is ``isaaclab.envs.direct_rl_env.DirectRLEnv.step``'s inner loop transcribed verbatim.

The equivalence argument, in full
---------------------------------

Hoisting is only sound because of four facts about *this* environment, each of which is
re-checked at run time by ``DuckiebotLaneFollowEnv._hoistable`` rather than assumed:

1. **The wheel targets are constant across the window.** ``_apply_action`` writes
   ``self._wheel_targets``, which ``_pre_physics_step`` computes once per control step. Sixteen
   identical ``set_joint_velocity_target`` calls produce one identical
   ``_data.joint_vel_target``.
2. **The actuator is implicit** (``ImplicitActuatorCfg``, see
   ``duckiebot_rl.assets.robot_cfg.duckiebot_articulation_cfg``). ``ImplicitActuator.compute``
   is a pass-through: it returns the control action unchanged and only derives the diagnostic
   ``computed_effort`` / ``applied_effort`` from the live joint state. An explicit actuator (a
   ``DCMotor``, an ``IdealPDActuator``) would compute a state-dependent torque per substep and
   the hoist would be wrong, so ``_hoistable`` refuses to hoist unless every actuator reports
   ``is_implicit_model``.
3. **Nothing reads the per-substep diagnostics.** ``data.computed_torque``,
   ``data.applied_torque``, ``data.joint_acc`` and ``data.body_acc_w`` have no consumer in this
   repository: reward, terminations, the observation and the privileged vector all read root
   pose and root velocities, which are direct PhysX reads refreshed by the single trailing
   ``scene.update``. The hoist leaves those four fields describing the window rather than its
   last substep.
4. **No external wrench is in flight.** ``Articulation.write_data_to_sim`` re-applies external
   forces every substep precisely because PhysX clears them on every ``simulate()``, and it
   resets the instantaneous wrench composer afterwards. Nothing in this repository calls
   ``set_external_force_and_torque`` (the D14 push is a ``write_root_velocity_to_sim``, which is
   a state write, not a force), but a future caller that did would be silently robbed of fifteen
   sixteenths of its impulse. ``_hoistable`` inspects the composers and falls back to the
   verbatim base-class loop the moment either is active.

The joint effort target is a fifth, weaker fact: ``set_dof_actuation_forces`` is fed
``_data.joint_effort_target``, which is all-zero for this robot because only
``set_joint_velocity_target`` is ever called. Writing zeros once and writing zeros sixteen times
are the same instruction stream whether or not PhysX clears DOF forces per step, so this one
needs no guard.

Rendering is untouched. The render still fires after exactly the same ``sim.step`` as before,
because :func:`window_ops` derives the render substeps from the same
``_sim_step_counter % render_interval == 0`` rule the base class uses, and the rendered frame is
an observation. Observations are not negotiable (SPEC v2 S5.2).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Literal

__all__ = [
    "APPLY",
    "PHYSICS",
    "RENDER",
    "UPDATE",
    "WRITE",
    "Op",
    "baseline_window_ops",
    "render_substeps",
    "run_window",
    "window_ops",
]

Op = Literal["apply", "write", "physics", "render", "update"]
"""One operation of a decimation window.

* ``"apply"`` is ``DirectRLEnv._apply_action()``
* ``"write"`` is ``InteractiveScene.write_data_to_sim()``
* ``"physics"`` is ``SimulationContext.step(render=False)``
* ``"render"`` is ``SimulationContext.render()``
* ``"update"`` is ``InteractiveScene.update(dt)``
"""

APPLY: Op = "apply"
WRITE: Op = "write"
PHYSICS: Op = "physics"
RENDER: Op = "render"
UPDATE: Op = "update"


def render_substeps(
    start_counter: int, decimation: int, render_interval: int, is_rendering: bool
) -> tuple[int, ...]:
    """Return the 1-based substep indices on which the base class renders.

    Transcribes ``direct_rl_env.py``'s rule: the substep counter is incremented BEFORE the
    physics step, and a render follows the physics step whenever the incremented counter is a
    multiple of ``render_interval``.

    Args:
        start_counter: Value of ``DirectRLEnv._sim_step_counter`` before the window.
        decimation: Physics steps per control step.
        render_interval: Physics steps per render.
        is_rendering: ``sim.has_gui() or sim.has_rtx_sensors()``; renders are skipped when False.

    Returns:
        Ascending 1-based substep indices, empty when ``is_rendering`` is False.

    Raises:
        ValueError: If ``decimation`` or ``render_interval`` is not positive.
    """
    if decimation <= 0:
        raise ValueError(f"decimation must be positive, got {decimation}")
    if render_interval <= 0:
        raise ValueError(f"render_interval must be positive, got {render_interval}")
    if not is_rendering:
        return ()
    return tuple(k for k in range(1, decimation + 1) if (start_counter + k) % render_interval == 0)


def baseline_window_ops(
    start_counter: int, decimation: int, render_interval: int, is_rendering: bool
) -> tuple[Op, ...]:
    """Return the base class's decimation window, verbatim.

    This is ``isaaclab.envs.direct_rl_env.DirectRLEnv.step``'s inner loop transcribed as data.
    It is the reference the hoisted plan is tested against, and it is the plan actually executed
    whenever ``DuckiebotLaneFollowEnv._hoistable`` says the hoist is unsafe.

    Args:
        start_counter: Value of ``DirectRLEnv._sim_step_counter`` before the window.
        decimation: Physics steps per control step.
        render_interval: Physics steps per render.
        is_rendering: ``sim.has_gui() or sim.has_rtx_sensors()``.

    Returns:
        The operation sequence of one control step's physics window.
    """
    return window_ops(
        start_counter,
        decimation,
        render_interval,
        is_rendering,
        hoist_writes=False,
        hoist_updates=False,
    )


def window_ops(
    start_counter: int,
    decimation: int,
    render_interval: int,
    is_rendering: bool,
    *,
    hoist_writes: bool = True,
    hoist_updates: bool = True,
) -> tuple[Op, ...]:
    """Return the decimation window with the constant-actuation writes hoisted.

    With both flags set this emits one ``apply``/``write`` pair before the physics steps and one
    trailing ``update``, instead of ``decimation`` of each. The physics steps and the renders are
    bit-for-bit the base class's: same count, same interleaving, same substeps.

    Setting both flags False reproduces the base-class loop exactly, which is what makes the
    fallback path in ``DuckiebotLaneFollowEnv.step`` a single call with different flags rather
    than a second copy of the loop.

    Args:
        start_counter: Value of ``DirectRLEnv._sim_step_counter`` before the window.
        decimation: Physics steps per control step.
        render_interval: Physics steps per render.
        is_rendering: ``sim.has_gui() or sim.has_rtx_sensors()``.
        hoist_writes: Emit ``apply``/``write`` once before the window instead of per substep.
        hoist_updates: Emit ``update`` once after the window instead of per substep.

    Returns:
        The operation sequence of one control step's physics window.
    """
    renders = set(render_substeps(start_counter, decimation, render_interval, is_rendering))
    ops: list[Op] = []
    if hoist_writes:
        ops += [APPLY, WRITE]
    for k in range(1, decimation + 1):
        if not hoist_writes:
            ops += [APPLY, WRITE]
        ops.append(PHYSICS)
        if k in renders:
            ops.append(RENDER)
        if not hoist_updates:
            ops.append(UPDATE)
    if hoist_updates:
        ops.append(UPDATE)
    return tuple(ops)


def run_window(ops: Sequence[Op], table: Mapping[Op, Callable[[], None]]) -> None:
    """Execute a plan from :func:`window_ops`.

    The dispatch lives here, next to the planner, so that the executed order can be tested
    against the planned order on a CPU runner with no Kit: the environment's real
    ``_run_physics_window`` differs from the test's only in what the five callables do.

    Args:
        ops: The plan, from :func:`window_ops` or :func:`baseline_window_ops`.
        table: One callable per operation kind.

    Raises:
        KeyError: If ``table`` is missing an operation the plan uses.
    """
    for op in ops:
        table[op]()
