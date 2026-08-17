"""Dynamic and static obstacles for the lane-following task (SPEC v2 S5.1, S5.5, S7.4).

Two halves, deliberately separated:

* :func:`obstacle_collection_cfg` builds the Isaac ``RigidObjectCollectionCfg``. It imports
  ``isaaclab`` lazily so this module stays importable on a CPU-only test runner.
* :class:`ObstacleField` is the pure-torch motion model, geometry and safety-circle query. It
  never touches Isaac, so the NPC drivers and the ``nearest`` query are unit-testable.

Why every mover is kinematic
----------------------------

Critic item H: NPC bodies driven by batched pose writes **must** be
``kinematic_enabled=True``. Teleporting a dynamic rigid body every 66 ms next to a 1.10 kg robot
makes PhysX resolve an arbitrarily large penetration in one substep, and the depenetration
impulse launches the ego across the map. Kinematic bodies push the ego and are not pushed back,
which is also the honest model of "an NPC that does not react to you" that S7.4 asks for.
:func:`obstacle_collection_cfg` sets the flag from the layout and
:func:`assert_movers_are_kinematic` re-checks it on an already-built config, because this is the
kind of setting that survives a refactor by being silently dropped.

The safety circle is the primary termination mechanism
------------------------------------------------------

S5.5 condition 2 terminates on ``dist(robot centre, obstacle centre) < 0.12 + r_obs``, a purely
geometric test evaluated here by :meth:`ObstacleField.nearest`. The physical colliders exist as
a backstop for the case where the geometric test is somehow bypassed, not as the mechanism.
That split is what critic item H asked to be decided and written down.

Motion models (S7.4)
--------------------

* **NPC Duckiebot**: pure pursuit of the lane centreline at ``U(0.05, 0.15)`` m/s, spawned
  ``U(0.30, 0.40)`` m ahead of the ego along the lane, non-reactive. It advances its own
  lane-frame arc length and hops to the straightest successor segment at a tile boundary, so it
  goes around corners without any collision avoidance.
* **Pedestrian duckie**: crosses the road, oscillating in the lane-frame lateral coordinate at
  ``U(0.10, 0.35)`` m/s. The step is ``speed * dt`` with the real control period, which is a
  documented divergence from the upstream frame-rate-dependent implementation.
* **Cone and parked bot**: static, placed at a fixed lane-frame offset.

Lane-frame forward map
----------------------

``BatchedLaneGraph`` answers world -> lane queries. Driving an NPC needs the inverse, lane ->
world, which :func:`lane_frame_to_world` computes from the graph's own segment tables
(``seg_p0``, ``seg_t0``, ``seg_center``, ``seg_radius``, ``seg_theta0``, ``seg_sign``). It reads
those tensors rather than re-deriving any geometry, so there is still exactly one description of
where a lane is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

__all__ = [
    "DEFAULT_OBSTACLE_LAYOUT",
    "NO_OBSTACLE_DISTANCE",
    "ObstacleField",
    "ObstacleSpec",
    "assert_movers_are_kinematic",
    "lane_frame_to_world",
    "obstacle_collection_cfg",
]

NO_OBSTACLE_DISTANCE: float = 10.0
"""Distance reported for "no obstacle anywhere", in metres.

Not 0.0. The privileged critic observation carries ``dist_nearest_obstacle``, and mapping the
emptiest possible scene onto 0.0 would encode it as the most dangerous state the critic can see.
The MuJoCo harness uses the same sentinel (``duckiebot_rl.sim2sim.NO_OBSTACLE_DISTANCE``), which
is what keeps the two ``vec_priv`` vectors comparable.
"""


@dataclass(frozen=True)
class ObstacleSpec:
    """One slot class in the obstacle layout.

    Attributes:
        name: Prim-name stem; slot ``i`` becomes ``f"{name}_{i}"``.
        count: Number of slots of this class per environment.
        motion: One of ``"npc"`` (lane-following mover), ``"pedestrian"`` (crossing mover),
            ``"static"`` (never moves).
        radius_m: The ``r_obs`` of the S5.5 safety circle, in metres. For a box this is the
            plan-view half-diagonal, which is the smallest circle that contains it.
        size_m: Full extents of a box obstacle, or None for the round shapes.
        height_m: Cylinder height, used only when ``shape`` is ``"cylinder"``.
        shape: ``"box"``, ``"sphere"`` or ``"cylinder"``.
        stage: Curriculum stage at which this class first appears (S7.4 task curriculum:
            stage 2 introduces one leading NPC, stage 3 the scenario-sampled rest).
    """

    name: str
    count: int
    motion: str
    radius_m: float
    size_m: tuple[float, float, float] | None = None
    height_m: float = 0.0
    shape: str = "sphere"
    stage: int = 3

    @property
    def is_mobile(self) -> bool:
        """Return True when slots of this class are driven by pose writes."""
        return self.motion in ("npc", "pedestrian")


DEFAULT_OBSTACLE_LAYOUT: tuple[ObstacleSpec, ...] = (
    ObstacleSpec(
        name="npc_bot",
        count=2,
        motion="npc",
        # Plan-view half-diagonal of the 0.18 x 0.13 footprint: hypot(0.09, 0.065) = 0.1110.
        radius_m=math.hypot(0.090, 0.065),
        size_m=(0.180, 0.130, 0.100),
        shape="box",
        stage=2,
    ),
    ObstacleSpec(name="duckie", count=4, motion="pedestrian", radius_m=0.040, shape="sphere", stage=3),
    ObstacleSpec(
        name="cone", count=4, motion="static", radius_m=0.030, height_m=0.070, shape="cylinder", stage=3
    ),
)
"""The SPEC v2 S5.1 obstacle inventory: 2 NPC bots, 4 duckies, 4 cones per env."""


# =============================================================================================
# Lane-frame forward map
# =============================================================================================


def lane_frame_to_world(
    lane: Any,
    variant_idx: torch.Tensor,
    seg_id: torch.Tensor,
    arc_s: torch.Tensor,
    lateral: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map lane-frame coordinates back to world position and heading.

    The inverse of :meth:`duckiebot_rl.city.lane_graph.BatchedLaneGraph.query`, computed from
    that class's own segment tables so the two directions cannot disagree. Straight segments
    advance along ``t0``; arcs advance in angle by ``arc_s / radius`` in the segment's sweep
    direction. The lateral offset uses the same left-positive convention as ``LaneQuery.d``.

    Args:
        lane: A ``BatchedLaneGraph``.
        variant_idx: ``(B,)`` long tensor of variant indices.
        seg_id: ``(B,)`` long tensor of segment indices within each variant.
        arc_s: ``(B,)`` arc length along the segment, in metres.
        lateral: ``(B,)`` lateral offset, positive to the LEFT of the direction of travel.

    Returns:
        ``(x, y, yaw)``, each ``(B,)``.
    """
    vidx = variant_idx.reshape(-1)
    sid = seg_id.reshape(-1)
    is_arc = lane.seg_is_arc[vidx, sid]
    p0 = lane.seg_p0[vidx, sid]
    t0 = lane.seg_t0[vidx, sid]
    center = lane.seg_center[vidx, sid]
    radius = lane.seg_radius[vidx, sid]
    theta0 = lane.seg_theta0[vidx, sid]
    sign = lane.seg_sign[vidx, sid]

    # Straight branch.
    x_line = p0[:, 0] + arc_s * t0[:, 0]
    y_line = p0[:, 1] + arc_s * t0[:, 1]
    tan_x_line, tan_y_line = t0[:, 0], t0[:, 1]

    # Arc branch.
    theta = theta0 + sign * (arc_s / torch.clamp(radius, min=1e-9))
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    x_arc = center[:, 0] + radius * cos_t
    y_arc = center[:, 1] + radius * sin_t
    tan_x_arc = -sign * sin_t
    tan_y_arc = sign * cos_t

    x_c = torch.where(is_arc, x_arc, x_line)
    y_c = torch.where(is_arc, y_arc, y_line)
    tan_x = torch.where(is_arc, tan_x_arc, tan_x_line)
    tan_y = torch.where(is_arc, tan_y_arc, tan_y_line)

    # Left normal of the tangent is (-ty, tx); LaneQuery.d = tx * off_y - ty * off_x uses the
    # same basis, so adding `lateral` along it round-trips through query() exactly.
    x = x_c - tan_y * lateral
    y = y_c + tan_x * lateral
    return x, y, torch.atan2(tan_y, tan_x)


# =============================================================================================
# The Isaac config half
# =============================================================================================


def obstacle_collection_cfg(
    layout: tuple[ObstacleSpec, ...] = DEFAULT_OBSTACLE_LAYOUT,
    prim_root: str = "{ENV_REGEX_NS}/Obstacle",
    parking_z_m: float = -5.0,
) -> Any:
    """Build the ``RigidObjectCollectionCfg`` holding every obstacle slot of one environment.

    One collection rather than N rigid objects: the collection batches all slots into a single
    PhysX view, so the whole field is repositioned with one ``write_object_pose_to_sim`` call
    per control step instead of one per obstacle (S5.1, "driven by batched pose writes").

    Every mover carries ``kinematic_enabled=True``; see the module docstring for why that is not
    optional. Static slots are kinematic too: they never move, and a kinematic static obstacle
    cannot be nudged out of position by a glancing contact, which would silently desynchronise
    the geometric safety circle from the collider.

    Args:
        layout: The obstacle inventory.
        prim_root: Prim path PREFIX, not a parent directory: slot ``foo`` is spawned at
            ``f"{prim_root}_foo"``, flat under the environment prim. A grouping prim such as
            ``{ENV_REGEX_NS}/Obstacles/foo`` does not work, because the spawner requires the
            parent to exist already (``sim/utils/prims.py:706``: "Unable to find source prim
            path") and nothing in the scene builder creates an intermediate Xform.
        parking_z_m: Initial z of every slot, well below the ground plane. Inactive slots stay
            parked there, out of the camera frustum and out of contact with anything.

    Returns:
        An ``isaaclab.assets.RigidObjectCollectionCfg``.

    Raises:
        ImportError: If Isaac Lab is not importable in this interpreter.
        ValueError: If a spec names an unsupported shape.
    """
    try:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg
    except ImportError as exc:  # pragma: no cover - exercised only inside Isaac
        raise ImportError(
            "Isaac Lab is not importable from this interpreter, so the obstacle collection "
            "config cannot be built. ObstacleField (the motion model and the safety-circle "
            "query) does not need Isaac and can be used on its own."
        ) from exc

    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        kinematic_enabled=True,
        disable_gravity=True,
        max_depenetration_velocity=1.0,
    )
    collision_props = sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.005)
    objects: dict[str, Any] = {}
    for spec in layout:
        for index in range(spec.count):
            name = f"{spec.name}_{index}"
            if spec.shape == "box":
                if spec.size_m is None:
                    raise ValueError(f"obstacle {spec.name!r} is a box but declares no size_m")
                spawn: Any = sim_utils.CuboidCfg(
                    size=spec.size_m,
                    rigid_props=rigid_props,
                    collision_props=collision_props,
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.15, 0.15)),
                )
                height = 0.5 * spec.size_m[2]
            elif spec.shape == "sphere":
                spawn = sim_utils.SphereCfg(
                    radius=spec.radius_m,
                    rigid_props=rigid_props,
                    collision_props=collision_props,
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.80, 0.10)),
                )
                height = spec.radius_m
            elif spec.shape == "cylinder":
                spawn = sim_utils.CylinderCfg(
                    radius=spec.radius_m,
                    height=spec.height_m,
                    rigid_props=rigid_props,
                    collision_props=collision_props,
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.35, 0.05)),
                )
                height = 0.5 * spec.height_m
            else:
                raise ValueError(f"unsupported obstacle shape {spec.shape!r} for {spec.name!r}")
            objects[name] = RigidObjectCfg(
                prim_path=f"{prim_root}_{name}",
                spawn=spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, parking_z_m + height)),
            )
    return RigidObjectCollectionCfg(rigid_objects=objects)


def assert_movers_are_kinematic(cfg: Any) -> None:
    """Re-check that every slot of a built collection config is kinematic (critic item H).

    Args:
        cfg: A ``RigidObjectCollectionCfg``.

    Raises:
        ValueError: If any slot has no rigid properties or is not kinematic.
    """
    for name, obj in cfg.rigid_objects.items():
        props = getattr(obj.spawn, "rigid_props", None)
        if props is None or not getattr(props, "kinematic_enabled", False):
            raise ValueError(
                f"obstacle {name!r} is not kinematic; teleporting a dynamic rigid body next to a "
                "1.10 kg robot produces a depenetration impulse that launches the ego"
            )


# =============================================================================================
# The pure-torch half
# =============================================================================================


class ObstacleField:
    """Per-env obstacle state, motion models and the S5.5 safety-circle query.

    Slot layout is flat and fixed: the slots of :data:`DEFAULT_OBSTACLE_LAYOUT` are concatenated
    in declaration order, so slot index ``k`` always means the same obstacle class and always
    matches the ``k``-th key of :func:`obstacle_collection_cfg`. That correspondence is what lets
    a single ``(N, K, 7)`` tensor be written straight into the collection.

    Args:
        num_envs: Number of parallel envs.
        lane: A ``BatchedLaneGraph`` supplying segment tables for placement and NPC driving.
        layout: The obstacle inventory.
        device: Torch device.
        generator: Torch generator, for reproducible placement.
        parking_z_m: z of a parked (inactive) slot.
    """

    def __init__(
        self,
        num_envs: int,
        lane: Any,
        layout: tuple[ObstacleSpec, ...] = DEFAULT_OBSTACLE_LAYOUT,
        device: torch.device | str = "cpu",
        generator: torch.Generator | None = None,
        parking_z_m: float = -5.0,
    ) -> None:
        self.num_envs = int(num_envs)
        self.lane = lane
        self.layout = layout
        self.device = torch.device(device)
        self.generator = generator
        self.parking_z_m = float(parking_z_m)

        self.names: list[str] = []
        radii: list[float] = []
        motions: list[str] = []
        stages: list[int] = []
        heights: list[float] = []
        for spec in layout:
            for index in range(spec.count):
                self.names.append(f"{spec.name}_{index}")
                radii.append(spec.radius_m)
                motions.append(spec.motion)
                stages.append(spec.stage)
                heights.append(
                    0.5 * spec.size_m[2]
                    if spec.shape == "box" and spec.size_m is not None
                    else (0.5 * spec.height_m if spec.shape == "cylinder" else spec.radius_m)
                )
        self.num_obstacles = len(self.names)

        f32 = torch.float32
        self.radius = torch.tensor(radii, dtype=f32, device=self.device)
        self.spawn_height = torch.tensor(heights, dtype=f32, device=self.device)
        self.stage = torch.tensor(stages, dtype=torch.long, device=self.device)
        self.is_npc = torch.tensor([m == "npc" for m in motions], dtype=torch.bool, device=self.device)
        self.is_pedestrian = torch.tensor(
            [m == "pedestrian" for m in motions], dtype=torch.bool, device=self.device
        )
        self.is_mobile = self.is_npc | self.is_pedestrian

        shape = (self.num_envs, self.num_obstacles)
        self.active = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.seg_id = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.arc_s = torch.zeros(shape, dtype=f32, device=self.device)
        self.lateral = torch.zeros(shape, dtype=f32, device=self.device)
        self.speed = torch.zeros(shape, dtype=f32, device=self.device)
        self.cross_sign = torch.ones(shape, dtype=f32, device=self.device)
        self.cross_amplitude = torch.full(shape, 0.12, dtype=f32, device=self.device)
        self.pos = torch.zeros(*shape, 2, dtype=f32, device=self.device)
        self.yaw = torch.zeros(shape, dtype=f32, device=self.device)
        self.vel = torch.zeros(*shape, 2, dtype=f32, device=self.device)

    # ------------------------------------------------------------------------------ sampling
    def _uniform(self, shape: tuple[int, ...], low: float, high: float) -> torch.Tensor:
        """Draw uniform noise on this field's device with this field's generator.

        Args:
            shape: Output shape.
            low: Inclusive lower bound.
            high: Exclusive upper bound.

        Returns:
            The sampled tensor.
        """
        u = torch.rand(shape, generator=self.generator, device=self.device, dtype=torch.float32)
        return low + (high - low) * u

    def reset(
        self,
        env_ids: torch.Tensor,
        variant_idx: torch.Tensor,
        robot_seg: torch.Tensor,
        robot_s: torch.Tensor,
        stage: int = 3,
        density: float = 1.0,
    ) -> None:
        """Place the obstacle field for the given envs at episode start.

        Args:
            env_ids: ``(M,)`` long tensor of resetting env indices.
            variant_idx: ``(M,)`` map variant of each resetting env.
            robot_seg: ``(M,)`` lane segment the robot spawned on.
            robot_s: ``(M,)`` arc length of the robot's spawn point along that segment.
            stage: Curriculum stage (S7.4). Slots whose ``ObstacleSpec.stage`` exceeds this are
                forced inactive, so stage 0/1 training sees an obstacle-free map and stage 2
                sees exactly one leading NPC.
            density: Probability in ``[0, 1]`` that an eligible non-leading slot is activated.
                The leading NPC (slot 0) is always active from stage 2 onward.
        """
        if env_ids.numel() == 0:
            return
        ids = env_ids.to(device=self.device, dtype=torch.long)
        count = int(ids.numel())
        k = self.num_obstacles
        vidx = variant_idx.to(device=self.device, dtype=torch.long).reshape(-1)
        seg = robot_seg.to(device=self.device, dtype=torch.long).reshape(-1)
        s0 = robot_s.to(device=self.device, dtype=torch.float32).reshape(-1)

        eligible = (self.stage <= stage).view(1, k).expand(count, k)
        draw = self._uniform((count, k), 0.0, 1.0) < density
        active = eligible & draw
        # The leading NPC is deterministic once its stage is reached: S7.4 stage 2 is "one
        # leading NPC", not "one leading NPC with probability `density`".
        if self.num_obstacles > 0 and int(self.stage[0]) <= stage:
            active[:, 0] = True

        # Everything is placed on the ego's own segment, which keeps obstacles inside the camera
        # frustum and inside the env's 3.6 m placement box without a rejection loop.
        seg_all = seg.view(count, 1).expand(count, k).clone()
        lead_offset = self._uniform((count, k), 0.30, 0.40)
        far_offset = self._uniform((count, k), 0.60, 2.40)
        offset = torch.where(self.is_npc.view(1, k), lead_offset, far_offset)
        arc = s0.view(count, 1) + offset

        lateral = torch.where(
            self.is_pedestrian.view(1, k),
            self._uniform((count, k), -0.12, 0.12),
            self._uniform((count, k), -0.05, 0.05),
        )
        speed = torch.where(
            self.is_npc.view(1, k),
            self._uniform((count, k), 0.05, 0.15),
            torch.where(
                self.is_pedestrian.view(1, k),
                self._uniform((count, k), 0.10, 0.35),
                torch.zeros((count, k), device=self.device),
            ),
        )

        self.active[ids] = active
        self.seg_id[ids] = seg_all
        self.arc_s[ids] = arc
        self.lateral[ids] = lateral
        self.speed[ids] = speed
        self.cross_sign[ids] = torch.where(
            self._uniform((count, k), -1.0, 1.0) < 0.0,
            -torch.ones((count, k), device=self.device),
            torch.ones((count, k), device=self.device),
        )
        self.cross_amplitude[ids] = self._uniform((count, k), 0.10, 0.20)
        self._refresh_world(ids, vidx)
        self.vel[ids] = 0.0

    # ------------------------------------------------------------------------------- driving
    def _refresh_world(self, env_ids: torch.Tensor, variant_idx: torch.Tensor) -> None:
        """Recompute world pose from lane-frame state for the given envs.

        Args:
            env_ids: ``(M,)`` env indices.
            variant_idx: ``(M,)`` variant index of each env.
        """
        count = int(env_ids.numel())
        k = self.num_obstacles
        vidx = variant_idx.view(count, 1).expand(count, k).reshape(-1)
        x, y, yaw = lane_frame_to_world(
            self.lane,
            vidx,
            self.seg_id[env_ids].reshape(-1),
            self.arc_s[env_ids].reshape(-1),
            self.lateral[env_ids].reshape(-1),
        )
        self.pos[env_ids] = torch.stack([x, y], dim=-1).view(count, k, 2)
        self.yaw[env_ids] = yaw.view(count, k)

    def step(self, dt: float, variant_idx: torch.Tensor) -> None:
        """Advance every mover by one control step.

        NPCs advance their arc length and hop to the straightest successor segment when they run
        off the end of the current one, which is the pure-pursuit behaviour of S7.4 expressed in
        lane coordinates. Pedestrians oscillate laterally and reverse at the amplitude, crossing
        the road at a constant real speed. Both use ``speed * dt`` with the true control period,
        the documented divergence from the upstream frame-rate-dependent implementation.

        Args:
            dt: Control period in seconds.
            variant_idx: ``(N,)`` map variant of every env.
        """
        if self.num_obstacles == 0:
            return
        n, k = self.num_envs, self.num_obstacles
        vidx = variant_idx.to(device=self.device, dtype=torch.long).reshape(-1)
        moving = self.active & self.is_mobile.view(1, k)

        # NPC: advance along the lane, hopping segments at the boundary.
        advance = torch.where(
            self.active & self.is_npc.view(1, k), self.speed * dt, torch.zeros_like(self.speed)
        )
        self.arc_s = self.arc_s + advance
        flat_v = vidx.view(n, 1).expand(n, k).reshape(-1)
        flat_seg = self.seg_id.reshape(-1)
        seg_len = self.lane.seg_length[flat_v, flat_seg].view(n, k)
        overflow = self.arc_s > seg_len
        if bool(overflow.any()):
            successor = self.lane.seg_primary[flat_v, flat_seg].view(n, k)
            self.arc_s = torch.where(overflow, self.arc_s - seg_len, self.arc_s)
            self.seg_id = torch.where(overflow, successor, self.seg_id)

        # Pedestrian: oscillate laterally, reversing at the amplitude.
        crossing = self.active & self.is_pedestrian.view(1, k)
        lateral_step = torch.where(
            crossing, self.cross_sign * self.speed * dt, torch.zeros_like(self.lateral)
        )
        new_lateral = self.lateral + lateral_step
        bounced = crossing & (new_lateral.abs() > self.cross_amplitude)
        self.cross_sign = torch.where(bounced, -self.cross_sign, self.cross_sign)
        self.lateral = torch.where(
            bounced, torch.clamp(new_lateral, -self.cross_amplitude, self.cross_amplitude), new_lateral
        )

        previous = self.pos.clone()
        all_ids = torch.arange(n, device=self.device)
        self._refresh_world(all_ids, vidx)
        # Only movers get a velocity; a static obstacle whose world pose is recomputed must not
        # report motion, or the critic's rel_speed_nearest_obstacle picks up numerical noise.
        self.vel = torch.where(moving.unsqueeze(-1), (self.pos - previous) / dt, torch.zeros_like(self.vel))

    # -------------------------------------------------------------------------------- query
    def nearest(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        vx: torch.Tensor,
        vy: torch.Tensor,
        margin_m: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return distance, safety-circle gap and closing speed to the nearest obstacle.

        Args:
            x: ``(N,)`` robot world x.
            y: ``(N,)`` robot world y.
            vx: ``(N,)`` robot world velocity x.
            vy: ``(N,)`` robot world velocity y.
            margin_m: The S5.5 geometric margin added to each obstacle radius, 0.12 m.

        Returns:
            ``(distance, gap, closing)``. ``distance`` is centre-to-centre and falls back to
            :data:`NO_OBSTACLE_DISTANCE` when the env has no active obstacle; ``gap`` is
            ``distance - (margin + r_obs)`` and is ``+inf`` in that case, so
            :func:`duckiebot_rl.envs.terminations.obstacle_contact` and
            :func:`duckiebot_rl.envs.rewards.r_proximity` both no-op; ``closing`` is the
            relative velocity projected on the bearing, positive when the gap is shrinking.
        """
        n = self.num_envs
        if self.num_obstacles == 0:
            full = torch.full((n,), NO_OBSTACLE_DISTANCE, dtype=torch.float32, device=self.device)
            return full, torch.full_like(full, float("inf")), torch.zeros_like(full)

        dx = self.pos[..., 0] - x.view(n, 1)
        dy = self.pos[..., 1] - y.view(n, 1)
        distance = torch.sqrt(dx * dx + dy * dy)
        gap = distance - (margin_m + self.radius.view(1, -1))
        masked_gap = torch.where(self.active, gap, torch.full_like(gap, float("inf")))
        best_gap, best = masked_gap.min(dim=1)

        rows = torch.arange(n, device=self.device)
        any_active = self.active.any(dim=1)
        best_distance = torch.where(
            any_active,
            distance[rows, best],
            torch.full((n,), NO_OBSTACLE_DISTANCE, device=self.device),
        )
        safe_distance = torch.clamp(distance[rows, best], min=1e-9)
        rel_vx = vx - self.vel[rows, best, 0]
        rel_vy = vy - self.vel[rows, best, 1]
        closing = (rel_vx * dx[rows, best] + rel_vy * dy[rows, best]) / safe_distance
        closing = torch.where(any_active, closing, torch.zeros_like(closing))
        return best_distance, best_gap, closing

    def world_poses(self, env_origins: torch.Tensor) -> torch.Tensor:
        """Return the ``(N, K, 7)`` pose tensor for ``write_object_pose_to_sim``.

        Inactive slots are parked below the ground plane rather than deleted, because the
        collection's PhysX view is allocated once at startup and its slot count is fixed.

        Args:
            env_origins: ``(N, 3)`` world origin of each environment, from ``scene.env_origins``.

        Returns:
            ``(N, K, 7)`` of ``(x, y, z, qw, qx, qy, qz)`` in WORLD coordinates.
        """
        n, k = self.num_envs, self.num_obstacles
        z_active = self.spawn_height.view(1, k).expand(n, k)
        z_parked = torch.full_like(z_active, self.parking_z_m)
        z = torch.where(self.active, z_active, z_parked)
        pos = torch.stack(
            [
                self.pos[..., 0] + env_origins[:, 0].view(n, 1),
                self.pos[..., 1] + env_origins[:, 1].view(n, 1),
                z + env_origins[:, 2].view(n, 1),
            ],
            dim=-1,
        )
        half = 0.5 * self.yaw
        quat = torch.stack(
            [torch.cos(half), torch.zeros_like(half), torch.zeros_like(half), torch.sin(half)], dim=-1
        )
        return torch.cat([pos, quat], dim=-1)

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Serialise the field for a checkpoint.

        Returns:
            ``{field_name: tensor}`` for every mutable per-env tensor.
        """
        keys = ("active", "seg_id", "arc_s", "lateral", "speed", "cross_sign", "cross_amplitude")
        out = {key: getattr(self, key).clone() for key in keys}
        out["pos"] = self.pos.clone()
        out["yaw"] = self.yaw.clone()
        out["vel"] = self.vel.clone()
        return out

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Restore a field serialised by :meth:`state_dict`.

        Args:
            state: The mapping returned by :meth:`state_dict`.

        Raises:
            KeyError: If a field is missing.
        """
        for key, current in (
            ("active", self.active),
            ("seg_id", self.seg_id),
            ("arc_s", self.arc_s),
            ("lateral", self.lateral),
            ("speed", self.speed),
            ("cross_sign", self.cross_sign),
            ("cross_amplitude", self.cross_amplitude),
            ("pos", self.pos),
            ("yaw", self.yaw),
            ("vel", self.vel),
        ):
            if key not in state:
                raise KeyError(f"obstacle state is missing field {key!r}")
            current.copy_(state[key].to(device=current.device, dtype=current.dtype))
