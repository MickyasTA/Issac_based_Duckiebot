"""``DuckiebotLaneFollowEnv``: the Isaac Lab environment of SPEC v2 S5.

This is the only module in the repository that imports Isaac Lab at module scope, which is why
:mod:`duckiebot_rl.envs` imports it lazily. Everything numeric it depends on lives elsewhere and
is unit-tested on CPU: the reward in :mod:`duckiebot_rl.envs.rewards`, the terminations in
:mod:`duckiebot_rl.envs.terminations`, the action path in :mod:`duckiebot_rl.envs.action_path`,
the camera model in :mod:`duckiebot_rl.envs.camera_math`, the observation chain in
:mod:`duckiebot_rl.dr.preprocess`, the lane geometry in :mod:`duckiebot_rl.city.lane_graph`.
What is left here is the wiring, the per-step order of operations, and the reset.

The order of operations, and why it is what it is
-------------------------------------------------

``DirectRLEnv.step`` runs, verbatim:

.. code-block:: text

    _pre_physics_step(action)
    for _ in range(decimation):   _apply_action(); write_data_to_sim(); sim.step(); [sim.render()]
    episode_length_buf += 1
    reset_terminated, reset_time_outs = _get_dones()
    reward_buf = _get_rewards()
    if any reset:  _reset_idx(ids);  [sim.render() x num_rerenders_on_reset]
    [interval events]
    obs_buf = _get_observations()

Three consequences this environment is built around:

1. ``_get_dones`` is the FIRST hook after physics, so it is where the per-step physics snapshot
   is taken: root pose, body velocities, the lane query, the progress delta, the obstacle
   geometry, and the preprocessed camera frame pushed into the frame ring. ``_get_rewards`` and
   ``_get_observations`` then read that one snapshot instead of re-querying, which keeps a
   single lane query per step rather than three.
2. ``_reset_idx`` runs BEFORE ``_get_observations``, so the observation a reset env returns is
   the new episode's first observation, not the terminal one. That is exactly why the terminal
   capture has to happen inside ``_reset_idx``.
3. The extra renders happen AFTER ``_reset_idx``, so the frame ring of a reset env cannot be
   refilled inside ``_reset_idx``: at that moment the camera still holds the pre-reset frame.
   The refill is therefore deferred to ``_get_observations`` through a pending mask, which is
   also what makes the S6.7 guard-4 pixel-change test meaningful.

Terminal capture (SPEC v2 S6.4 and critic item G), in the exact order
---------------------------------------------------------------------

Critic item G lists three ways this is silently got wrong, and all three are addressed here:

* **The terminal stack is 3 frames, not 1.** ``self._stacked_obs`` is the full ``(N, 48, 96, 9)``
  stack built in ``_get_dones`` from the frame ring with the D9 delay applied, i.e. byte for byte
  what ``_get_observations`` would have returned had the episode continued. Cloning one frame of
  ``camera.data.output["rgb"]`` would hand the critic a third of the observation it was trained
  on and the bootstrap would be quietly biased.
* **The critic is asymmetric**, so ``vec_priv`` (14 dims) is captured from PRE-reset physics
  state as well. ``self._vec_priv`` is likewise computed in the ``_get_dones`` snapshot.
* **Capture happens before ``super()._reset_idx(env_ids)``**, which is the call that resets the
  scene, applies reset events and lets this subclass write the new root state. After it, the
  pre-reset state is gone.

The capture is written into the learner's :class:`duckiebot_rl.ppo.buffer.TerminalCache` through
:meth:`attach_rollout_buffer`. If no buffer is attached the environment still runs correctly (for
``scripts/play.py``, for a scripted probe, for the M3 random-policy smoke test); it simply has
nowhere to put the terminals.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

try:
    from isaaclab.envs import DirectRLEnv
except ImportError as exc:  # pragma: no cover - exercised only outside Isaac
    raise ImportError(
        "Isaac Lab is not importable from this interpreter, so DuckiebotLaneFollowEnv cannot be "
        "defined. This module only runs inside the Isaac Sim python environment, and the "
        "AppLauncher must have been constructed before it is imported. On this machine the "
        "interpreter is d:/Personal/personal/wheeled_quadruped_robot/.venv/Scripts/python.exe. "
        "Everything numeric this environment uses (rewards, terminations, the action path, the "
        "camera model) lives in Isaac-free modules and can be imported without Kit."
    ) from exc

from duckiebot_rl.city.lane_graph import BatchedLaneGraph, LaneGraph, progress_delta
from duckiebot_rl.city.maps import load_map
from duckiebot_rl.city.spec import geometry_buckets
from duckiebot_rl.dr.curriculum import HardExampleMiner, HardExampleMinerCfg, TwoScalarADR
from duckiebot_rl.dr.dynamics import DynamicsRandomizer, quantize_encoder
from duckiebot_rl.dr.preprocess import FrameStack, preprocess_frame
from duckiebot_rl.dr.visual import VisualDR, VisualDRCfg, sample_camera_mount, sample_frame_repeat
from duckiebot_rl.envs.action_path import TorchActionPath
from duckiebot_rl.envs.camera_math import quat_cam_ros_torch
from duckiebot_rl.envs.episode_log import DeviceLog
from duckiebot_rl.envs.obstacles import NO_OBSTACLE_DISTANCE, ObstacleField, lane_frame_to_world
from duckiebot_rl.envs.rewards import RewardWeights, compute_reward, wrong_lane_indicator
from duckiebot_rl.envs.step_loop import APPLY, PHYSICS, RENDER, UPDATE, WRITE, run_window, window_ops
from duckiebot_rl.envs.terminations import TerminationFlags, TerminationState

__all__ = ["DuckiebotLaneFollowEnv"]

_CURVATURE_LOOKAHEAD_M = 0.3
"""Lookahead distance of the ``curvature`` entry of the privileged observation (SPEC v2 S5.2)."""

_HARD_MINING_TOURNAMENT = 8
"""Candidate segments compared when a spawn is biased toward a high-error segment (S7.4)."""


# =============================================================================================
# Small torch quaternion helpers
# =============================================================================================


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return the Hamilton product of two batches of ``(w, x, y, z)`` quaternions.

    Args:
        a: ``(N, 4)`` left quaternions.
        b: ``(N, 4)`` right quaternions.

    Returns:
        ``(N, 4)`` products.
    """
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def _quat_rotate(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate a batch of vectors by a batch of ``(w, x, y, z)`` quaternions.

    Args:
        quat: ``(N, 4)`` unit quaternions.
        vec: ``(N, 3)`` vectors.

    Returns:
        ``(N, 3)`` rotated vectors.
    """
    w = quat[:, 0:1]
    xyz = quat[:, 1:]
    t = 2.0 * torch.cross(xyz, vec, dim=-1)
    return vec + w * t + torch.cross(xyz, t, dim=-1)


def _quat_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
    """Return the ``(N, 4)`` quaternion of a rotation about ``+z``.

    Args:
        yaw: ``(N,)`` yaw angles in radians.

    Returns:
        ``(N, 4)`` quaternions in ``(w, x, y, z)`` order.
    """
    half = 0.5 * yaw
    zero = torch.zeros_like(half)
    return torch.stack([torch.cos(half), zero, zero, torch.sin(half)], dim=-1)


def _roll_pitch_yaw(quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decompose ``(w, x, y, z)`` quaternions into intrinsic ZYX Euler angles.

    Written here rather than pulled from ``isaaclab.utils.math`` so that the sign convention of
    the rollover termination is visible next to the termination that uses it.

    Args:
        quat: ``(N, 4)`` unit quaternions.

    Returns:
        ``(roll, pitch, yaw)``, each ``(N,)`` in radians.
    """
    w, x, y, z = quat.unbind(-1)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


# =============================================================================================
# The environment
# =============================================================================================


class DuckiebotLaneFollowEnv(DirectRLEnv):
    """Vision-based Duckiebot lane following on a procedurally generated miniature Duckietown.

    Args:
        cfg: The config produced by :func:`duckiebot_rl.envs.env_cfg.lane_follow_env_cfg`.
        render_mode: Gymnasium render mode; ``"rgb_array"`` enables the video recorder wrapper.
        **kwargs: Forwarded to ``DirectRLEnv``.
    """

    def __init__(self, cfg: Any, render_mode: str | None = None, **kwargs: Any) -> None:
        self.settings = cfg.settings
        super().__init__(cfg, render_mode, **kwargs)

        params = self.settings.params
        spaces = self.settings.spaces
        n = self.num_envs
        device = self.device

        self._generator = torch.Generator(device=device)
        self._generator.manual_seed(int(self.settings.seed) + 0x5EED)

        # --- lane geometry ---------------------------------------------------------------
        self._lane = self._build_lane_graph()
        self._variant_idx = torch.arange(n, device=device, dtype=torch.long) % self._lane.num_variants
        self._lane_width = self._lane.lane_width.to(device)[self._variant_idx]
        self._segment_count = self._lane.seg_valid.sum(dim=1).to(device)

        # --- domain randomization --------------------------------------------------------
        self._dynamics_dr = DynamicsRandomizer(n, device=device, generator=self._generator)
        self._visual_dr = VisualDR(
            n,
            VisualDRCfg(fused_draws=self.settings.perf.fused_visual_dr_draws),
            device=device,
            generator=self._generator,
        )
        self._alpha_vis = float(self.settings.dr_alpha_vis)
        self._alpha_dyn = float(self.settings.dr_alpha_dyn)
        self._miner = HardExampleMiner(
            HardExampleMinerCfg(num_tiles=self._lane.num_variants * self._lane.max_segments)
        )
        # ADR boundary probes. `Range.sample` takes a per-env boundary mask and draws that env
        # from the EDGE of the live range instead of its interior, which is what makes an ADR
        # measurement about the boundary rather than about the population mean. The masks are
        # assigned per episode so a probe episode is a probe from its first control step.
        self._adr: TwoScalarADR | None = None
        self._probe_vis = torch.zeros(n, dtype=torch.bool, device=device)
        self._probe_dyn = torch.zeros(n, dtype=torch.bool, device=device)
        self._curriculum_records = DeviceLog()

        # --- action path -----------------------------------------------------------------
        self._action_path = TorchActionPath(n, self.step_dt, params, device=device, generator=self._generator)
        self._wheel_ids, _ = self._robot.find_joints(params.wheel_joint_regex, preserve_order=True)
        self._wheel_targets = torch.zeros(n, 2, device=device)
        self._raw_actions = torch.zeros(n, spaces.act_dim, device=device)
        self._action = torch.zeros(n, spaces.act_dim, device=device)
        self._prev_action = torch.zeros(n, spaces.act_dim, device=device)
        self._prev_action2 = torch.zeros(n, spaces.act_dim, device=device)

        # --- observation ------------------------------------------------------------------
        self._frames: FrameStack | None = None
        if self.settings.use_image:
            self._frames = FrameStack(
                n, obs_hw=(spaces.obs_height, spaces.obs_width), device=device, backend="torch"
            )
        self._stacked_obs = torch.zeros(n, *spaces.rgb_shape, dtype=torch.uint8, device=device)
        self._vec = torch.zeros(n, spaces.vec_dim, device=device)
        self._vec_priv = torch.zeros(n, spaces.priv_dim, device=device)
        self._pending_refill = torch.zeros(n, dtype=torch.bool, device=device)
        # Host-side mirror of "is `_pending_refill` non-empty". The mask itself is a device
        # tensor, so testing it costs a sync on EVERY control step even though it is False on
        # the 62.5% of steps at N=64 that contain no reset (profile rank 9). The flag is set in
        # `_reset_idx` and cleared in `_get_observations`, which are the only two writers.
        self._pending_refill_any = False

        # --- obstacles ---------------------------------------------------------------------
        self._obstacle_field = ObstacleField(n, self._lane, device=device, generator=self._generator)
        self._gap = torch.full((n,), float("inf"), device=device)
        self._prev_gap = torch.full((n,), float("inf"), device=device)

        # --- terminations and per-step scratch ----------------------------------------------
        self._terminations = TerminationState(n, self.step_dt, device=device)
        self._weights = RewardWeights()
        self._prev_xy = torch.zeros(n, 2, device=device)
        self._d = torch.zeros(n, device=device)
        self._psi = torch.zeros(n, device=device)
        self._ds = torch.zeros(n, device=device)
        self._seg_id = torch.zeros(n, dtype=torch.long, device=device)
        self._arc_s = torch.zeros(n, device=device)
        # route position of the previous step's lane match, NaN until the first post-reset
        # query. Feeding it back constrains matching to route-continuous segments, which is
        # what keeps d truthful when the robot leaves its lane; see BatchedLaneGraph.query.
        self._route_pos = torch.full((n,), float("nan"), device=device)
        self._body_speed = torch.zeros(n, device=device)
        self._reward_terms: dict[str, torch.Tensor] = {}
        self._flags: TerminationFlags | None = None
        self._counts_cache: dict[str, int] | None = None
        self._in_step = False
        self._buffer: Any = None

        # --- episode statistics for the S6.8 diagnostics --------------------------------------
        self._ep_distance = torch.zeros(n, device=device)
        self._ep_abs_d_integral = torch.zeros(n, device=device)
        self._ep_out_of_lane_integral = torch.zeros(n, device=device)
        self._ep_wrong_lane_s = torch.zeros(n, device=device)
        self._ep_return = torch.zeros(n, device=device)
        # Accumulated as DEVICE tensors, one chunk per reset, and drained to the host once per
        # training iteration; see `duckiebot_rl.envs.episode_log`. Holding python floats here
        # instead meant a `.cpu().tolist()` burst inside every `_reset_idx`, which at N=64 fires
        # on 37.5% of control steps: profile rank 6 measured 41.2 ms per reset, 15.4 ms
        # amortised per control step, and the fraction of steps that pay it grows with N.
        self._episode_log = DeviceLog()
        self._spawn_log = DeviceLog()

        # --- D14 external push -----------------------------------------------------------------
        self._push_timer = torch.zeros(n, device=device)

        # Spawn bookkeeping for the S7.4 hard-example miner: which (variant, segment) slot each
        # env started its current episode on.
        self._spawn_slot = torch.zeros(n, dtype=torch.long, device=device)
        self._error_table: torch.Tensor | None = None
        self._staggered = not self.settings.rates.stagger_initial_episode_length

    # =========================================================================================
    # Construction helpers
    # =========================================================================================

    def _build_lane_graph(self) -> BatchedLaneGraph:
        """Load the city maps that the spawned USD stages were generated from.

        The maps come from the same build directory as the stages (``scripts/build_city.py``
        writes ``<out>/maps`` and ``<out>/usd`` in one pass), and each map's marking geometry
        comes from its own texture bucket. Reading the YAML rather than regenerating the maps in
        memory is what guarantees the lane graph describes the city that is actually on screen:
        a mismatched ``--seed`` would otherwise produce a plausible-looking graph over the wrong
        layout and every reward would be quietly wrong.

        Returns:
            The batched lane graph, one variant per city stage, in spawn order.
        """
        city_cfg = self.settings.city
        buckets = geometry_buckets(count=city_cfg.geometry_buckets, seed=city_cfg.variant_seed)
        graphs: list[LaneGraph] = []
        for path in self.cfg.map_paths:
            city = load_map(path)
            bucket = int(city.meta.get("geometry_bucket", 0)) % len(buckets)
            graphs.append(LaneGraph(city, buckets[bucket]))
        return BatchedLaneGraph(graphs, device=self.device)

    def _setup_scene(self) -> None:
        """Cache handles to the entities ``InteractiveScene`` built from the scene config.

        The scene graph is declared as a full ``InteractiveSceneCfg`` subclass in
        :func:`duckiebot_rl.envs.env_cfg.lane_follow_env_cfg`, so the scene is already cloned and
        its collisions already filtered by the time this runs. Critic item E: with
        ``replicate_physics=False`` and ``filter_collisions=True`` the filter call is made
        automatically at ``interactive_scene.py:214-215``, and a manual call afterwards hits the
        ``/World/collisions`` early-out and does nothing.
        """
        self._robot = self.scene["robot"]
        self._camera = self.scene["camera"] if self.settings.use_image else None
        self._obstacles = self.scene.rigid_object_collections.get("obstacles")

    # =========================================================================================
    # Action path (SPEC v2 S5.3)
    # =========================================================================================

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Run the S5.3 action path once per control step and advance the obstacle field.

        Args:
            actions: ``(N, 2)`` raw policy action. Stored unclipped for the buffer; the clip to
                ``[-1, 1]`` happens inside the action path, which is the only place it happens.
        """
        self._in_step = True
        self._raw_actions = actions.clone()
        self._prev_action2 = self._prev_action
        self._prev_action = self._action
        self._action = torch.clamp(actions, -1.0, 1.0)

        wheel_velocity = self._robot.data.joint_vel[:, self._wheel_ids]
        self._wheel_targets = self._action_path(
            self._raw_actions,
            wheel_velocity,
            self._dynamics_dr.params,
            apply_dr=self.settings.dynamics_dr,
        )

        self._advance_obstacles()
        self._apply_external_push()

    def _apply_action(self) -> None:
        """Write the wheel-velocity targets. Called once per PHYSICS step, not per control step.

        The target is constant across the decimation window on purpose: the deployed
        ``car_cmd_switch_node`` publishes at 15 Hz and the motor controller holds the last
        command, so re-deriving it at 240 Hz would model a robot that does not exist.
        """
        self._robot.set_joint_velocity_target(self._wheel_targets, joint_ids=self._wheel_ids)

    # =========================================================================================
    # The decimation window (profile rank 1)
    # =========================================================================================

    def _hoistable(self) -> bool:
        """Return whether the constant-actuation writes may be hoisted out of the window.

        The four preconditions are stated and argued in :mod:`duckiebot_rl.envs.step_loop`. Two
        of them are properties of this class and cannot change at run time (the wheel target is
        computed once per control step; nothing reads the per-substep torque diagnostics). The
        other two are properties of the assets, so they are checked here, every step, cheaply:

        * every actuator on the robot must be an implicit one, and
        * no articulation or rigid-object collection in the scene may have an external wrench in
          flight, because ``write_data_to_sim`` re-applies external wrenches precisely BECAUSE
          PhysX clears them on every ``simulate()``.

        Both checks are python attribute reads on host-side objects. There is no device work and
        no host sync, which is what lets them run unconditionally rather than being a start-up
        assertion that a later change could invalidate.

        Returns:
            True when :func:`~duckiebot_rl.envs.step_loop.window_ops` may hoist.
        """
        if not self.settings.perf.hoist_actuation_writes:
            return False
        for actuator in self._robot.actuators.values():
            if not getattr(actuator, "is_implicit_model", False):
                return False
        entities = (
            *self.scene.articulations.values(),
            *self.scene.rigid_objects.values(),
            *self.scene.rigid_object_collections.values(),
        )
        for entity in entities:
            for name in ("_instantaneous_wrench_composer", "_permanent_wrench_composer"):
                composer = getattr(entity, name, None)
                if composer is not None and getattr(composer, "active", False):
                    return False
        return True

    def _run_physics_window(self) -> None:
        """Run one control step's worth of physics, writing the constant actuation once.

        This replaces the inner loop of ``DirectRLEnv.step`` and nothing else. The physics steps
        and the renders are identical in count and in interleaving to the base class's, because
        :func:`~duckiebot_rl.envs.step_loop.window_ops` derives them from the same
        ``_sim_step_counter % render_interval`` rule; only the repeated actuation write and the
        repeated lazy-buffer timestamp bump are collapsed.

        When :meth:`_hoistable` says no, the emitted plan IS the base class's loop, op for op.
        """
        decimation = int(self.cfg.decimation)
        start = int(self._sim_step_counter)
        hoist = self._hoistable()
        ops = window_ops(
            start,
            decimation,
            int(self.cfg.sim.render_interval),
            self.sim.has_gui() or self.sim.has_rtx_sensors(),
            hoist_writes=hoist,
            hoist_updates=hoist and self.settings.perf.hoist_scene_updates,
        )
        # One trailing `scene.update` must advance the lazy buffers by the WHOLE window, not by
        # one substep: `SensorBase.update` accumulates `dt` into the timestamp it compares
        # against `update_period`, so feeding it a sixteenth of the elapsed time would make a
        # rate-limited sensor fire sixteen times too slowly.
        update_dt = self.physics_dt * (decimation if ops.count(UPDATE) == 1 else 1)

        def _physics() -> None:
            self._sim_step_counter += 1
            self.sim.step(render=False)

        run_window(
            ops,
            {
                APPLY: self._apply_action,
                WRITE: self.scene.write_data_to_sim,
                PHYSICS: _physics,
                RENDER: self.sim.render,
                UPDATE: lambda: self.scene.update(dt=update_dt),
            },
        )

    def step(self, action: torch.Tensor) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Advance one control step; ``DirectRLEnv.step`` with the physics window replaced.

        Everything outside the decimation window is the base class's code verbatim, in the base
        class's order, because that order is what this environment's whole snapshot design is
        built around (see the module docstring). The one substantive difference is
        :meth:`_run_physics_window`.

        Args:
            action: ``(N, 2)`` raw policy action.

        Returns:
            ``(obs, reward, terminated, truncated, extras)``, exactly as ``DirectRLEnv.step``.
        """
        action = action.to(self.device)
        if self.cfg.action_noise_model:
            action = self._action_noise_model(action)

        self._pre_physics_step(action)
        self._run_physics_window()

        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
            if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()

        if self.cfg.events and "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.obs_buf = self._get_observations()
        if self.cfg.observation_noise_model:
            self.obs_buf["policy"] = self._observation_noise_model(self.obs_buf["policy"])

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _advance_obstacles(self) -> None:
        """Advance the kinematic obstacle field and push the new poses into PhysX."""
        if self._obstacles is None:
            return
        self._obstacle_field.step(self.step_dt, self._variant_idx)
        poses = self._obstacle_field.world_poses(self.scene.env_origins)
        self._obstacles.write_object_pose_to_sim(poses)

    def _apply_external_push(self) -> None:
        """Apply the D14 interval push by adding a velocity impulse to the root state."""
        if not self.settings.dynamics_dr:
            return
        params = self._dynamics_dr.params
        self._push_timer += self.step_dt
        due = self._push_timer >= params.push_interval_s
        ids = due.nonzero(as_tuple=False).squeeze(-1)
        if ids.numel() == 0:
            return
        self._push_timer[ids] = 0.0
        d_v, d_yaw = self._dynamics_dr.sample_push(ids, self._alpha_dyn)
        velocity = self._robot.data.root_vel_w[ids].clone()
        yaw = _roll_pitch_yaw(self._robot.data.root_quat_w[ids])[2]
        velocity[:, 0] += d_v * torch.cos(yaw)
        velocity[:, 1] += d_v * torch.sin(yaw)
        velocity[:, 5] += d_yaw
        self._robot.write_root_velocity_to_sim(velocity, env_ids=ids)

    # =========================================================================================
    # Per-step snapshot, terminations and reward
    # =========================================================================================

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Take the post-physics snapshot, then evaluate the S5.5 conditions.

        This is the first hook after the decimation loop, so it owns the single lane query, the
        single obstacle query and the single camera read of the control step. Everything
        downstream (``_get_rewards``, ``_get_observations``, the terminal capture) reads the
        cached result.

        Returns:
            ``(terminated, truncated)``, each ``(N,)`` bool.
        """
        self._snapshot()
        flags = self._terminations.evaluate(
            self._lane,
            self._variant_idx,
            self._root_xy[:, 0],
            self._root_xy[:, 1],
            self._yaw,
            self._roll,
            self._pitch,
            self._body_speed,
            self._yaw_rate,
            self._gap,
            self.episode_length_buf,
            self.max_episode_length,
            self.settings.params,
        )
        # Deliberately NOT `flags.counts()`. That call is a host sync, `_get_dones` runs on
        # every control step, and the counts are read only when an episode actually ended
        # (`scripts/train.py:termination_reason`) or once per iteration (`drain_episode_log`).
        # Keeping the masks and materialising the dict on demand is profile rank 5: it removed
        # 7 of the 23.4 synchronizing calls per control step without changing a single number.
        self._flags = flags
        self._counts_cache = None
        return flags.terminated, flags.truncated

    @property
    def _termination_counts(self) -> dict[str, int]:
        """``{condition_name: count}`` for the last completed step, materialised on demand.

        Private-by-name because ``scripts/train.py`` reads it through ``getattr`` and because it
        was a plain attribute before the sync census; the name is part of the interface with the
        learner side and is kept.

        Returns:
            The counts of the most recent ``_get_dones``, or an empty dict before the first one.
        """
        if self._counts_cache is None:
            self._counts_cache = {} if self._flags is None else self._flags.counts()
        return self._counts_cache

    def _snapshot(self) -> None:
        """Read physics once, build the lane query, the obstacle query and the observation."""
        data = self._robot.data
        origins = self.scene.env_origins
        self._root_xy = data.root_pos_w[:, :2] - origins[:, :2]
        self._roll, self._pitch, self._yaw = _roll_pitch_yaw(data.root_quat_w)
        self._body_speed = data.root_lin_vel_b[:, 0]
        self._yaw_rate = data.root_ang_vel_b[:, 2]

        query = self._lane.query(
            self._variant_idx,
            self._root_xy[:, 0],
            self._root_xy[:, 1],
            self._yaw,
            prev_route_pos=self._route_pos,
        )
        self._d = query.d
        self._psi = query.psi
        self._seg_id = query.seg_id
        self._arc_s = query.s
        self._route_pos = self._lane.route_progress(self._variant_idx, query.seg_id, query.s)
        self._ds = progress_delta(
            self._prev_xy[:, 0],
            self._prev_xy[:, 1],
            self._root_xy[:, 0],
            self._root_xy[:, 1],
            query.tangent_x,
            query.tangent_y,
        )
        self._prev_xy = self._root_xy.clone()
        self._curvature = self._lane.curvature_at_lookahead(
            self._variant_idx, self._seg_id, self._arc_s, _CURVATURE_LOOKAHEAD_M
        )

        self._prev_gap = self._gap
        if self._obstacles is not None:
            world_vel = data.root_lin_vel_w
            distance, gap, closing = self._obstacle_field.nearest(
                self._root_xy[:, 0],
                self._root_xy[:, 1],
                world_vel[:, 0],
                world_vel[:, 1],
                self.settings.obstacles.margin_m,
            )
        else:
            distance = torch.full_like(self._d, NO_OBSTACLE_DISTANCE)
            gap = torch.full_like(self._d, float("inf"))
            closing = torch.zeros_like(self._d)
        self._gap = gap
        self._obstacle_distance = distance
        self._obstacle_closing = closing

        self._build_vectors()
        if self._frames is not None:
            self._push_camera_frame()
        self._accumulate_episode_stats()

    def _build_vectors(self) -> None:
        """Build the 8-dim actor vector and the 14-dim privileged critic vector (S5.2)."""
        wheel_speed = self._robot.data.joint_vel[:, self._wheel_ids]
        if self.settings.dynamics_dr:
            wheel_speed = quantize_encoder(
                wheel_speed,
                self.step_dt,
                self._dynamics_dr.params.encoder_dropout_p,
                generator=self._generator,
            )
        else:
            wheel_speed = quantize_encoder(
                wheel_speed,
                self.step_dt,
                torch.zeros(self.num_envs, device=self.device),
                tick_noise=0.0,
                generator=self._generator,
            )
        self._vec = torch.cat(
            [
                self._prev_action,
                self._prev_action2,
                wheel_speed,
                self._yaw_rate.unsqueeze(-1),
                self._body_speed.unsqueeze(-1),
            ],
            dim=-1,
        )
        self._vec_priv = torch.cat(
            [
                self._vec,
                self._d.unsqueeze(-1),
                self._psi.unsqueeze(-1),
                self._curvature.unsqueeze(-1),
                self._obstacle_distance.unsqueeze(-1),
                self._obstacle_closing.unsqueeze(-1),
                (self._body_speed * torch.cos(self._psi)).unsqueeze(-1),
            ],
            dim=-1,
        )

    def _push_camera_frame(self) -> None:
        """Run S4.3 steps 1-9 on the current render and push the result into the frame ring."""
        assert self._camera is not None and self._frames is not None
        # Step 1: the clone is MANDATORY. `data.output["rgb"]` is a view into the live rgba
        # buffer that the renderer overwrites on the next sim.render().
        frame = self._camera.data.output["rgb"].clone()

        photometric = None
        pp_shift = None
        if self.settings.visual_dr:
            speed_frac = torch.clamp(self._body_speed.abs() / self.settings.params.v_cmd_max_m_s, 0.0, 1.0)
            params = self._visual_dr.sample(self._alpha_vis, speed_frac=speed_frac, boundary=self._probe_vis)
            photometric = self._visual_dr.operator(params)
            pp_shift = self._principal_point_shift()
            repeat = sample_frame_repeat(params.frame_repeat_p, generator=self._generator)
        else:
            repeat = None

        processed = preprocess_frame(frame, photometric=photometric, pp_shift=pp_shift)
        self._frames.push(processed, repeat_mask=repeat)
        self._stacked_obs = self._frames.get()

    def _principal_point_shift(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw the V11 integer principal-point jitter, in render pixels.

        USD aperture offsets are hardcoded to 0.0 and any nonzero value is ignored with a warning
        (critic item C), so the jitter is a torch-side shift at S4.3 step 4 and the authored
        camera keeps an exactly centred principal point.

        Returns:
            ``(dx, dy)``, each ``(N,)`` long in ``[-2, 2]``.
        """
        shape = (self.num_envs,)
        dx = torch.randint(-2, 3, shape, generator=self._generator, device=self.device)
        dy = torch.randint(-2, 3, shape, generator=self._generator, device=self.device)
        return dx, dy

    def _accumulate_episode_stats(self) -> None:
        """Fold this step into the AI-DO-style online metrics of SPEC v2 S6.8.

        ``out_of_lane_integral`` is corrected for the lane re-match (2026-08-18). The lane graph
        matches a pose to the NEAREST lane, so a robot that has fully crossed the yellow line is
        re-matched to the oncoming lane and its ``|d|`` collapses back toward zero: measured on
        ``city_000``, ``|d|`` falls from 0.103 m to 0.003 m as the robot continues across. The
        raw ``clip(|d| - w/2, 0)`` integral therefore stops accumulating exactly when the robot
        is most thoroughly out of its lane, and reports a sustained wrong-lane run as zero.

        :func:`duckiebot_rl.envs.rewards.wrong_lane_indicator` detects that case from ``psi``,
        which the re-match does not hide. While it fires, the excursion is charged at no less
        than a half-lane width, so the integral is monotone through a full crossing instead of
        collapsing at the far side of it.
        """
        half_lane = 0.5 * self._lane_width
        wrong_lane = wrong_lane_indicator(self._psi)
        excursion = torch.clamp(self._d.abs() - half_lane, min=0.0)
        excursion = torch.where(wrong_lane > 0.0, torch.maximum(excursion, half_lane), excursion)
        self._ep_distance += torch.clamp(self._ds, min=0.0)
        self._ep_abs_d_integral += self._d.abs() * self.step_dt
        self._ep_out_of_lane_integral += excursion * self.step_dt
        self._ep_wrong_lane_s += wrong_lane * self.step_dt

    def _get_rewards(self) -> torch.Tensor:
        """Evaluate the SPEC v2 S5.4 reward from the cached snapshot.

        ``DirectRLEnv.step`` calls ``_get_dones`` first, so ``self.reset_terminated`` is already
        populated and the terminal penalty can be applied inside the same expression that clips
        the total, which is the order S5.4 specifies.

        Returns:
            ``(N,)`` reward.
        """
        reward, terms = compute_reward(
            d=self._d,
            psi=self._psi,
            ds=self._ds,
            action=self._action,
            prev_action=self._prev_action,
            body_speed=self._body_speed,
            lane_width=self._lane_width,
            gap=self._gap if self._obstacles is not None else None,
            prev_gap=self._prev_gap if self._obstacles is not None else None,
            terminated=self.reset_terminated,
            weights=self._weights,
            robot_width=self.settings.params.robot_width_m,
            v_max=self.settings.params.v_cmd_max_m_s,
            control_dt=self.step_dt,
        )
        self._reward_terms = {k: v.detach() for k, v in terms.as_dict().items()}
        self._ep_return += reward
        return reward

    # =========================================================================================
    # Observations
    # =========================================================================================

    def _get_observations(self) -> dict[str, Any]:
        """Return the actor observation, refilling the frame ring of any env that just reset.

        The refill cannot happen inside ``_reset_idx``: the extra renders that produce the first
        post-reset frame run AFTER it (``direct_rl_env.py:398-402``). Deferring to here is what
        makes ``num_rerenders_on_reset = 1`` actually deliver a fresh frame, and it is the code
        path the S6.7 guard-4 teleport test exercises.

        Returns:
            ``{"policy": {"rgb": ..., "vec": ...}}``, with the ``rgb`` key absent in vec-only
            mode. The critic's view is served separately by :meth:`_get_states`.
        """
        if self._frames is not None and self._pending_refill_any:
            assert self._camera is not None
            ids = self._pending_refill.nonzero(as_tuple=False).squeeze(-1)
            frame = self._camera.data.output["rgb"][ids].clone()
            self._frames.reset(ids, preprocess_frame(frame))
            self._stacked_obs = self._frames.get()
            self._pending_refill[:] = False
            self._pending_refill_any = False

        self._in_step = False
        policy: dict[str, torch.Tensor] = {"vec": self._vec}
        if self.settings.use_image:
            policy["rgb"] = self._stacked_obs
        return {"policy": policy}

    def _get_states(self) -> dict[str, Any]:
        """Return the asymmetric critic's observation (SPEC v2 S5.2).

        Returns:
            ``{"rgb": ..., "vec_priv": ...}``, with the ``rgb`` key absent in vec-only mode.
        """
        state: dict[str, torch.Tensor] = {"vec_priv": self._vec_priv}
        if self.settings.use_image:
            state["rgb"] = self._stacked_obs
        return state

    @property
    def vec(self) -> torch.Tensor:
        """``(N, 8)`` raw actor vector observation of the last completed control step.

        ``DirectRLEnv.step`` returns only ``obs["policy"]``; the learner needs the privileged
        vector as well, and it needs both without paying for a second dict construction.
        """
        return self._vec

    @property
    def vec_priv(self) -> torch.Tensor:
        """``(N, 14)`` raw privileged critic observation of the last completed control step."""
        return self._vec_priv

    @property
    def stacked_obs(self) -> torch.Tensor:
        """``(N, 48, 96, 9)`` stacked uint8 image observation, or a zero tensor in vec-only mode."""
        return self._stacked_obs

    # -----------------------------------------------------------------------------------------
    # Read-only handles for diagnostics (scripts/check_obs.py, scripts/live_view.py).
    #
    # They exist so that a diagnostic does not have to reach into `_robot`, `_camera`, `_d`,
    # `_psi` or `_variant_idx`. A tool that reads privates is a tool that breaks silently the
    # next time this class is refactored, and these five are exactly the handles every
    # observation-side check needs.
    # -----------------------------------------------------------------------------------------

    @property
    def robot(self) -> Any:
        """The robot ``Articulation``, i.e. ``scene["robot"]``."""
        return self._robot

    @property
    def onboard_camera(self) -> Any:
        """The onboard ``TiledCamera``, or None in vec-only mode.

        Named ``onboard_camera`` rather than ``camera`` because ``DirectRLEnv`` already carries a
        ``cfg.viewer`` notion of "the camera" and a bare name would read as that one.
        """
        return self._camera

    @property
    def lane_offset(self) -> torch.Tensor:
        """``(N,)`` signed lateral offset of the last snapshot, metres, positive toward yellow."""
        return self._d

    @property
    def lane_heading_error(self) -> torch.Tensor:
        """``(N,)`` heading error of the last snapshot, radians, positive left of the tangent."""
        return self._psi

    @property
    def variant_index(self) -> torch.Tensor:
        """``(N,)`` long tensor giving the city map variant each environment was spawned on."""
        return self._variant_idx

    # =========================================================================================
    # Reset (SPEC v2 S6.4 order, critic item G)
    # =========================================================================================

    def attach_rollout_buffer(self, buffer: Any) -> None:
        """Give the environment somewhere to put captured terminal observations.

        Args:
            buffer: A :class:`duckiebot_rl.ppo.buffer.RolloutBuffer`, or None to detach.
        """
        self._buffer = buffer

    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        """Reset the given envs, capturing the true terminal observation first.

        The five numbered steps below are SPEC v2 S6.4 verbatim, and the ordering is the whole
        point: after ``super()._reset_idx`` the pre-reset physics state and the pre-reset frame
        ring are both gone, and a capture taken afterwards silently returns the NEW episode's
        first observation instead of the terminal one.

        Args:
            env_ids: Indices of the environments to reset.
        """
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).reshape(-1)
        if ids.numel() == 0:
            return

        # 1-3. Capture the terminal stack and the pre-reset privileged vector, then hand both to
        #      the learner. Only inside a step(): the initial reset() has no transition to
        #      attribute a terminal to, and _stacked_obs still holds zeros there.
        if self._buffer is not None and self._in_step:
            image = self._stacked_obs[ids].clone() if self.settings.use_image else None
            self._buffer.capture_terminal(env_ids=ids, vec_priv=self._vec_priv[ids].clone(), image=image)

        self._log_finished_episodes(ids)

        # 4. Scene reset, reset events, base-class bookkeeping.
        super()._reset_idx(ids)

        # 5. Resample per-episode DR and re-place the robot, the obstacles and the camera.
        if self._adr is not None:
            masks = self._adr.assign_probes(int(ids.numel()), generator=self._generator, device=self.device)
            self._probe_vis[ids] = masks["vis"]
            self._probe_dyn[ids] = masks["dyn"]
        if self.settings.dynamics_dr:
            self._dynamics_dr.resample(ids, self._alpha_dyn, boundary=self._probe_dyn[ids])
        self._action_path.reset(ids)
        self._action_path.set_delay(self._dynamics_dr.params.delay_steps, self._dynamics_dr.params.delay_frac)
        if self._frames is not None:
            self._frames.set_obs_delay(self._dynamics_dr.params.obs_delay_steps)

        spawn_seg, spawn_s, spawn_xy = self._respawn_robot(ids)
        if self._obstacles is not None:
            self._obstacle_field.reset(
                ids,
                self._variant_idx[ids],
                spawn_seg,
                spawn_s,
                stage=self.settings.obstacles.stage,
                density=self.settings.obstacles.density,
            )
            poses = self._obstacle_field.world_poses(self.scene.env_origins)
            self._obstacles.write_object_pose_to_sim(poses)
        if self.settings.visual_dr and self._camera is not None:
            self._randomize_camera_mount(ids)

        self._terminations.reset(ids, spawn_xy)
        self._action[ids] = 0.0
        self._prev_action[ids] = 0.0
        self._prev_action2[ids] = 0.0
        self._push_timer[ids] = 0.0
        self._gap[ids] = float("inf")
        self._prev_gap[ids] = float("inf")
        self._ep_distance[ids] = 0.0
        self._route_pos[ids] = float("nan")  # first post-reset match is a free global search
        self._ep_abs_d_integral[ids] = 0.0
        self._ep_out_of_lane_integral[ids] = 0.0
        self._ep_wrong_lane_s[ids] = 0.0
        self._ep_return[ids] = 0.0
        # The frame ring is refilled in _get_observations, after the post-reset render.
        self._pending_refill[ids] = True
        self._pending_refill_any = True

        # Stagger the very first all-env reset. Doing it here rather than in __init__ is what
        # makes it survive: super()._reset_idx zeroes episode_length_buf, so a stagger applied
        # before the first reset() would be wiped and every env would then terminate on the same
        # control step, turning the S5.2 rerender factor into a periodic spike.
        if not self._staggered and int(ids.numel()) == self.num_envs:
            self.episode_length_buf = torch.randint(
                0,
                self.max_episode_length,
                (self.num_envs,),
                device=self.device,
                dtype=self.episode_length_buf.dtype,
            )
            self._staggered = True

    def _respawn_robot(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Place the robot at a fresh D16 spawn pose on a drivable lane.

        Args:
            ids: ``(M,)`` env indices being reset.

        Returns:
            ``(segment, arc_length, xy)`` of the spawn point, in env-local coordinates.
        """
        count = int(ids.numel())
        variant = self._variant_idx[ids]
        segment = self._sample_spawn_segments(variant)
        seg_length = self._lane.seg_length[variant, segment]
        arc = seg_length * torch.rand(count, generator=self._generator, device=self.device)

        if self.settings.dynamics_dr:
            lateral, heading = self._dynamics_dr.spawn_pose(ids, self._alpha_dyn)
        else:
            lateral = torch.zeros(count, device=self.device)
            heading = torch.zeros(count, device=self.device)

        x, y, tangent_yaw = lane_frame_to_world(self._lane, variant, segment, arc, lateral)
        yaw = tangent_yaw + heading

        root_state = self._robot.data.default_root_state[ids].clone()
        origins = self.scene.env_origins[ids]
        root_state[:, 0] = x + origins[:, 0]
        root_state[:, 1] = y + origins[:, 1]
        root_state[:, 2] = self.settings.params.base_link_height_m + origins[:, 2]
        root_state[:, 3:7] = _quat_from_yaw(yaw)
        root_state[:, 7:] = 0.0
        self._robot.write_root_state_to_sim(root_state, env_ids=ids)

        zeros = torch.zeros(count, len(self._wheel_ids), device=self.device)
        self._robot.write_joint_state_to_sim(zeros, zeros, joint_ids=self._wheel_ids, env_ids=ids)

        xy = torch.stack([x, y], dim=-1)
        self._prev_xy[ids] = xy
        self._spawn_slot[ids] = variant * self._lane.max_segments + segment
        return segment, arc, xy

    def _sample_spawn_segments(self, variant: torch.Tensor) -> torch.Tensor:
        """Draw spawn segments, biasing a fraction of them toward high-error segments.

        SPEC v2 S7.4 asks for "bias 25% of spawns to the worst-decile tiles by tracking error".
        The draw is a tournament: a biased spawn compares ``_HARD_MINING_TOURNAMENT`` uniform
        candidates FROM THE ENV'S OWN VARIANT and keeps the one with the highest recorded error.
        Restricting to the env's variant is not optional - a spawn segment only exists inside the
        city stage that env was given, and a global worst-decile draw would place the robot on a
        lane belonging to a different map.

        Args:
            variant: ``(M,)`` map variant of each resetting env.

        Returns:
            ``(M,)`` segment indices, each valid within its own variant.
        """
        count = int(variant.numel())
        counts = self._segment_count[variant].to(torch.float32)
        uniform = (torch.rand(count, generator=self._generator, device=self.device) * counts).to(torch.long)
        uniform = torch.clamp(uniform, max=(counts.to(torch.long) - 1).clamp(min=0))

        table = self._error_table
        if table is None:
            table = torch.tensor(self._miner.error_table, dtype=torch.float32, device=self.device)
            self._error_table = table
        if table.numel() == 0 or not bool((table > 0.0).any()):
            return uniform

        candidates = (
            torch.rand(count, _HARD_MINING_TOURNAMENT, generator=self._generator, device=self.device)
            * counts.unsqueeze(-1)
        ).to(torch.long)
        candidates = torch.minimum(candidates, (counts.to(torch.long) - 1).clamp(min=0).unsqueeze(-1))
        slots = variant.unsqueeze(-1) * self._lane.max_segments + candidates
        best = table[slots].argmax(dim=1)
        hard = candidates[torch.arange(count, device=self.device), best]

        take_hard = torch.rand(count, generator=self._generator, device=self.device) < (
            self._miner.cfg.hard_fraction
        )
        return torch.where(take_hard, hard, uniform)

    def _randomize_camera_mount(self, ids: torch.Tensor) -> None:
        """Apply the V10 camera-mount DR by writing the camera's world pose (SPEC v2 S7.2).

        The camera prim is a child of ``base_link``, so writing its world pose while the robot
        sits at its freshly written spawn pose is exactly equivalent to writing a new local
        offset, and it stays correct for the rest of the episode as the robot drives. The pose is
        computed from the root state this reset just wrote rather than read back from the sensor,
        because ``update_latest_camera_pose`` is False and ``camera.data.pos_w`` is stale by
        design (S5.1).

        Args:
            ids: ``(M,)`` env indices being reset.
        """
        assert self._camera is not None
        count = int(ids.numel())
        mount = sample_camera_mount(
            count,
            self._alpha_vis,
            generator=self._generator,
            device=self.device,
            boundary=self._probe_vis[ids],
        )
        offset = torch.stack(
            [mount["forward_m"], torch.zeros(count, device=self.device), mount["base_z_m"]], dim=-1
        )
        root_pos = self._robot.data.root_pos_w[ids]
        root_quat = self._robot.data.root_quat_w[ids]
        position = root_pos + _quat_rotate(root_quat, offset)

        optical = quat_cam_ros_torch(mount["pitch_down_rad"], mount["yaw_rad"], mount["roll_rad"])
        self._camera.set_world_poses(position, _quat_mul(root_quat, optical), env_ids=ids, convention="ros")

    # =========================================================================================
    # Diagnostics
    # =========================================================================================

    def _log_finished_episodes(self, ids: torch.Tensor) -> None:
        """Record the S6.8 episode metrics of the envs that are about to reset.

        ``DirectRLEnv`` does not populate ``extras["log"]`` (only the manager-based workflow
        does), so the four AI-DO-style metrics are accumulated here and drained once per
        training iteration by :meth:`drain_episode_log`.

        Args:
            ids: ``(M,)`` env indices being reset.
        """
        if ids.numel() == 0:
            return
        tile_metres = self._lane.pitch[self._variant_idx[ids]]
        distance = self._ep_distance[ids]
        duration = self.episode_length_buf[ids].to(torch.float32) * self.step_dt
        entries = {
            "episode/return": self._ep_return[ids],
            "episode/length_s": duration,
            "episode/distance_m": distance,
            "episode/distance_tiles": distance / tile_metres,
            "episode/abs_d_integral_ms": self._ep_abs_d_integral[ids],
            "episode/out_of_lane_integral_ms": self._ep_out_of_lane_integral[ids],
            "episode/wrong_lane_s": self._ep_wrong_lane_s[ids],
        }
        # Everything below stays on the device. Each of these is a fresh tensor (advanced
        # indexing copies), so none of them aliases a per-env buffer that `_reset_idx` is about
        # to zero, and the drain reads exactly the values this reset produced.
        self._episode_log.extend(entries)
        self._spawn_log.extend({"slot": self._spawn_slot[ids], "error": self._ep_abs_d_integral[ids]})

        # The ADR success metric is the S7.4 one: mean lane-frame consecutive distance in tiles.
        tiles = entries["episode/distance_tiles"]
        self._curriculum_records.extend(
            {"vis": tiles[self._probe_vis[ids]], "dyn": tiles[self._probe_dyn[ids]]}
        )

    def drain_episode_log(self) -> dict[str, float]:
        """Return and clear the accumulated per-episode metrics.

        Also folds the finished episodes' tracking errors into the hard-example mining table, so
        that a caller which never asks for diagnostics still cannot end up with a stale table.

        Returns:
            ``{metric_name: mean_over_completed_episodes}``, empty when nothing finished.
        """
        if self._spawn_log.pending:
            spawn = self._spawn_log.drain()
            self._miner.update(spawn["slot"], spawn["error"])
            self._error_table = None
        out = self._episode_log.drain_means()
        out.update({f"terminations/{k}": float(v) for k, v in self._termination_counts.items()})
        return out

    def reward_term_means(self) -> dict[str, float]:
        """Return the mean of each unweighted S5.4 reward term over the last control step.

        Returns:
            ``{term_name: mean}``.
        """
        return {f"reward/{k}": float(v.mean()) for k, v in self._reward_terms.items()}

    # =========================================================================================
    # Curriculum and checkpointing (SPEC v2 S6.9, S7.4)
    # =========================================================================================

    def attach_curriculum(self, adr: TwoScalarADR | None) -> None:
        """Let the environment draw ADR boundary probes on reset.

        The learner owns the :class:`~duckiebot_rl.dr.curriculum.TwoScalarADR` because it is the
        learner that checkpoints it (SPEC v2 S6.9 makes ``alpha_vis``, ``alpha_dyn`` and the ADR
        buffers mandatory checkpoint fields). The environment only needs it to assign probe
        masks at reset and to tag finished episodes, which :meth:`drain_curriculum_records`
        hands back.

        Args:
            adr: The ADR controller, or None to detach.
        """
        self._adr = adr

    def drain_curriculum_records(self) -> dict[str, list[float]]:
        """Return and clear the finished probe episodes, keyed by scalar.

        Returns:
            ``{"vis": [distance_tiles, ...], "dyn": [...]}``, ready for ``TwoScalarADR.record``.
        """
        drained = self._curriculum_records.drain()
        return {name: drained.get(name, []) for name in ("vis", "dyn")}

    def set_curriculum_alphas(self, alpha_vis: float, alpha_dyn: float) -> None:
        """Set the two auto-DR scalars.

        Args:
            alpha_vis: Visual curriculum scalar in ``[0, 1]``.
            alpha_dyn: Dynamics curriculum scalar in ``[0, 1]``.

        Raises:
            ValueError: If either scalar is outside ``[0, 1]``.
        """
        for name, value in (("alpha_vis", alpha_vis), ("alpha_dyn", alpha_dyn)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        self._alpha_vis = float(alpha_vis)
        self._alpha_dyn = float(alpha_dyn)

    @property
    def hard_example_miner(self) -> HardExampleMiner:
        """The hard-example mining table; part of the S6.9 checkpoint contract."""
        return self._miner

    @property
    def lane_graph(self) -> BatchedLaneGraph:
        """The batched lane graph, exposed for evaluation scripts and debug overlays."""
        return self._lane

    def env_state_dict(self) -> dict[str, Any]:
        """Serialise the environment-side stream state for a checkpoint.

        SPEC v2 S6.9 is explicit that the learner restores exactly while the environment stream
        restores only statistically: PhysX state is not checkpointable and a resume fully resets
        every env. What IS restored is everything that would otherwise silently rewind a
        schedule: the two curriculum scalars and the hard-example mining table.

        Returns:
            A JSON/torch-serialisable mapping.
        """
        return {
            "alpha_vis": self._alpha_vis,
            "alpha_dyn": self._alpha_dyn,
            "hard_example_table": self._miner.state_dict(),
        }

    def load_env_state_dict(self, state: dict[str, Any]) -> None:
        """Restore the state produced by :meth:`env_state_dict`.

        Args:
            state: The mapping returned by :meth:`env_state_dict`.

        Raises:
            KeyError: If a mandatory field is missing.
        """
        for key in ("alpha_vis", "alpha_dyn"):
            if key not in state:
                raise KeyError(
                    f"environment state is missing mandatory field {key!r}; without it a resume "
                    "silently restarts domain randomization at alpha 0 (SPEC v2 S6.9)"
                )
        self.set_curriculum_alphas(float(state["alpha_vis"]), float(state["alpha_dyn"]))
        table = state.get("hard_example_table")
        if table is not None:
            self._miner.load_state_dict(table)

    def carb_settings_readback(self) -> dict[str, Any]:
        """Read back the render settings that S4.4 acceptance item 3 requires to be verified.

        The Isaac Lab antialiasing setter swallows exceptions, so a failed write leaves the run
        on the preset's DLSS with no error anywhere. This returns what carb actually holds.

        Returns:
            ``{setting_path: value}`` for every path in
            :func:`duckiebot_rl.envs.env_cfg.expected_carb_settings`.
        """
        import carb

        from duckiebot_rl.envs.env_cfg import expected_carb_settings

        interface = carb.settings.get_settings()
        return {path: interface.get(path) for path in expected_carb_settings(self.settings.rendering)}
