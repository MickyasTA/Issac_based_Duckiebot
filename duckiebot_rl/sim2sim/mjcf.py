"""Generate the Duckiebot MJCF from the shared parameter source of truth (SPEC v2 S8.1).

Nothing here hardcodes a robot dimension. Every number arrives through
:func:`duckiebot_rl.sim2sim._resolve.resolve_robot_params`, which reads
:mod:`duckiebot_rl.assets.params` (the same object the Isaac Lab ``ArticulationCfg`` and the URDF
generator read). If the two simulators ever disagree it will be because of physics, not because
somebody retyped 0.0318 into a second file.

Two findings from the research phase are locked in here and are asserted by
``tests/unit/test_mjcf.py``. Both were measured, not guessed:

1. **Wheel collision geoms are SPHERES.** A cylinder produces a two-point line contact whose
   torsional couple fights differential-drive yaw. Measured on this model at the SPEC v2
   parameters, ``dt = 1/240``, on the (10, 30) rad/s arc: a sphere tracks the kinematic prediction
   to 0.04% while a cylinder is 15.9% short of it, and turns at 79% of the sphere's yaw rate. (The
   research-phase figure of -74% was measured at the v1 parameters and does **not** reproduce here;
   the ablation still separates the two models by a factor of 400 in error, which is what
   ``test_cylinder_wheel_contact_destroys_yaw`` asserts.) A sphere of the wheel radius centred on
   the axle has exactly one contact point at exactly the right height, which is also what the PhysX
   side converges to.
2. **The integrator is ``implicitfast``**, because PhysX integrates its implicit velocity drive
   implicitly and matching structure is the job. :meth:`SimParams.validate` refuses ``Euler``.

   Correction to the v1 justification, measured against this model: the v1 claim that ``Euler``
   costs -87% forward speed and +133% yaw rate on a tight arc **does not reproduce** at the SPEC v2
   parameters. At ``dt = 1/240`` the two integrators agree to 0.07%, across the whole S7.3 armature
   randomization range and up to a servo gain of 0.4. The v1 blow-up was a consequence of the
   unset/undersized joint armature that S1 item 26 corrects: explicit integration of the ``kv`` term
   is stable while ``dt * kv / I_effective`` is below about 2, which was roughly 8 at the v1 numbers
   and is roughly 0.9 at the v2 numbers. The margin is under 3x and system identification may move
   both the armature and the gain, so the lock stays; the *reason* for it is now structural rather
   than empirical. ``tests/unit/test_mj_kinematics.py`` carries the measurement.

Actuator mapping to the Isaac implicit drive (research report table row 9):

============================  ========================================  ==============
MuJoCo                        Isaac Lab                                 Identity
============================  ========================================  ==============
``<velocity kv="d">``         ``ImplicitActuatorCfg(stiffness=0,``      ``kv = d``
                              ``damping=d)``
``forcerange="-e e"``         ``effort_limit_sim = e``                  equal
``ctrlrange="-w w"``          ``velocity_limit_sim = w``                equal
``<joint armature=... />``    ``ImplicitActuatorCfg(armature=...)``     equal
``<joint frictionloss=.../>`` ``ImplicitActuatorCfg(friction=...)``     equal
============================  ========================================  ==============

The MJCF joint's own ``damping`` attribute is therefore **zero** by default: the 0.05 N.m.s/rad of
S2 is the servo gain ``kv``, not a second passive damper. ``RobotParams.passive_joint_damping``
exists only so system identification stage 2 can introduce a passive term deliberately.

Ground contact lives on a single plane, exactly as in Isaac (S3.3: cities carry zero physics
colliders, physics is one authored plane at ``/World/ground``). :mod:`duckiebot_rl.sim2sim.track`
therefore emits every tile, wall and sign as a *visual-only* geom.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from math import radians
from pathlib import Path

from ._resolve import (
    ObsParams,
    RobotParams,
    SimParams,
    mj_camera_xyaxes,
    resolve_robot_params,
    resolve_sim_params,
)

__all__ = [
    "CAMERA_NAME",
    "GROUND_GEOM_NAME",
    "IMU_SITE_NAME",
    "MjcfCfg",
    "build_robot_body",
    "build_robot_xml",
    "build_scene_xml",
    "write_robot_xml",
]

CAMERA_NAME = "db_cam"
GROUND_GEOM_NAME = "ground"
IMU_SITE_NAME = "imu"

_RGBA_CHASSIS = (0.10, 0.10, 0.12, 1.0)
_RGBA_WHEEL = (0.12, 0.12, 0.13, 1.0)
_RGBA_CASTER = (0.72, 0.72, 0.74, 1.0)
_RGBA_CAMERA = (0.02, 0.02, 0.02, 1.0)
_RGBA_MARKER = (0.95, 0.78, 0.06, 1.0)
_RGBA_GROUND = (0.37, 0.51, 0.16, 1.0)
_COLLISION_GROUP = 3
_VISUAL_GROUP = 2


def _fmt(value: float) -> str:
    """Format one float for an MJCF attribute so it round-trips exactly.

    ``repr`` emits the shortest decimal string that parses back to the identical double. Truncating
    to a fixed precision instead would silently change ``physics_dt`` from 1/240 to 0.00416666667,
    and the whole point of resolving critic item J is that the two simulators integrate at the *same*
    rate, not at nearly the same rate.
    """
    return repr(float(value))


def _vec(values: Iterable[float]) -> str:
    """Format a vector for an MJCF attribute."""
    return " ".join(_fmt(v) for v in values)


@dataclass
class MjcfCfg:
    """Everything the MJCF writer needs, assembled from the shared parameter modules.

    Attributes:
        robot: dimensional and actuator constants, resolved from ``duckiebot_rl.assets.params``.
        sim: integrator, rate and contact-softness settings.
        obs: render resolution, used for the ``<visual><global>`` offscreen buffer size.
        params_source: provenance string recorded in an XML comment for traceability.
        camera_pitch_down_deg: override for the nominal mount pitch (V10 domain randomization).
        camera_pos: override for the nominal mount position, in ``base_link`` frame.
        include_camera: emit the ``<camera>`` element (False for physics-only sysid models).
        include_visuals: emit the non-colliding decorative geoms.
        shadows: enable shadow casting on the scene light (SPEC v2 S8.1 keeps shadows on).
        texturedir: value of ``<compiler texturedir>``; the track builder points it at its own
            texture output directory.
        model_name: MJCF ``<mujoco model>`` name.
    """

    robot: RobotParams = field(default_factory=lambda: resolve_robot_params()[0])
    sim: SimParams = field(default_factory=lambda: resolve_sim_params()[0])
    obs: ObsParams = field(default_factory=ObsParams)
    params_source: str = ""
    camera_pitch_down_deg: float | None = None
    camera_pos: tuple[float, float, float] | None = None
    include_camera: bool = True
    include_visuals: bool = True
    shadows: bool = True
    texturedir: str = "."
    model_name: str = "duckiebot"

    @classmethod
    def from_shared(cls, **overrides: object) -> MjcfCfg:
        """Build a config from the shared parameter modules, recording their provenance.

        Args:
            **overrides: any :class:`MjcfCfg` field to override.

        Returns:
            The configured :class:`MjcfCfg`.
        """
        robot, robot_src = resolve_robot_params()
        sim, sim_src = resolve_sim_params()
        base = cls(robot=robot, sim=sim, params_source=f"robot: {robot_src}; rates: {sim_src}")
        return replace(base, **overrides) if overrides else base

    def __post_init__(self) -> None:
        """Validate the resolved parameters as soon as a config is built.

        Raises:
            ValueError: propagated from :meth:`RobotParams.validate` or :meth:`SimParams.validate`.
        """
        self.robot.validate()
        self.sim.validate()

    @property
    def pitch_down_rad(self) -> float:
        """Camera pitch actually written into the MJCF, in radians."""
        deg = self.robot.camera_pitch_down_deg
        if self.camera_pitch_down_deg is not None:
            deg = self.camera_pitch_down_deg
        return radians(deg)

    @property
    def mount_pos(self) -> tuple[float, float, float]:
        """Camera mount position actually written into the MJCF, in ``base_link`` frame."""
        return self.camera_pos if self.camera_pos is not None else self.robot.camera_pos


# ------------------------------------------------------------------------------- XML fragments
def _sub(parent: ET.Element, tag: str, **attrs: str) -> ET.Element:
    """Append a child element with string attributes."""
    return ET.SubElement(parent, tag, {k: v for k, v in attrs.items() if v is not None})


def _compiler(cfg: MjcfCfg) -> ET.Element:
    """Return the ``<compiler>`` element. Angles are radians everywhere in this package."""
    return ET.Element(
        "compiler",
        {"angle": "radian", "autolimits": "true", "texturedir": cfg.texturedir},
    )


def _option(cfg: MjcfCfg) -> ET.Element:
    """Return the ``<option>`` element carrying the locked integrator and contact settings."""
    sim = cfg.sim
    return ET.Element(
        "option",
        {
            "timestep": _fmt(sim.physics_dt),
            "integrator": sim.integrator,
            "cone": sim.cone,
            "impratio": _fmt(sim.impratio),
            "solver": sim.solver,
            "iterations": str(int(sim.iterations)),
            "ls_iterations": str(int(sim.ls_iterations)),
            "gravity": _vec(sim.gravity),
        },
    )


def _statistic() -> ET.Element:
    """Pin ``extent`` to 1 m so ``<visual><map znear zfar>`` are read directly in metres.

    MuJoCo expresses ``znear`` and ``zfar`` as fractions of the model extent, and the compiler
    infers the extent from the scene unless it is set. Pinning it makes the near and far clip
    numerically identical to the Isaac ``clipping_range``, which is what S4.1 requires.
    """
    return ET.Element("statistic", {"extent": "1", "center": "0 0 0.1"})


def _visual(cfg: MjcfCfg) -> ET.Element:
    """Return the ``<visual>`` element: offscreen buffer, clip planes and shadow quality."""
    near, far = cfg.robot.clipping_range
    element = ET.Element("visual")
    _sub(
        element,
        "global",
        offwidth=str(int(cfg.obs.render_w)),
        offheight=str(int(cfg.obs.render_h)),
        fovy=_fmt(cfg.robot.camera_fovy_deg),
    )
    _sub(element, "map", znear=_fmt(near), zfar=_fmt(far))
    _sub(element, "quality", shadowsize="2048" if cfg.shadows else "0", offsamples="4")
    _sub(
        element,
        "headlight",
        ambient="0.35 0.35 0.35",
        diffuse="0.55 0.55 0.55",
        specular="0.1 0.1 0.1",
    )
    return element


def _defaults(cfg: MjcfCfg) -> ET.Element:
    """Return the ``<default>`` tree holding the wheel, caster, collision and visual classes."""
    robot, sim = cfg.robot, cfg.sim
    root = ET.Element("default")
    db = _sub(root, "default", **{"class": "db"})
    _sub(
        db,
        "geom",
        condim="3",
        friction=_vec((1.0, 0.005, 0.0001)),
        solref=_vec(sim.solref_default),
        solimp=_vec(sim.solimp_default),
    )

    wheel = _sub(db, "default", **{"class": "db/wheel"})
    _sub(
        wheel,
        "joint",
        type="hinge",
        axis="0 1 0",
        damping=_fmt(robot.passive_joint_damping),
        armature=_fmt(robot.joint_armature),
        frictionloss=_fmt(robot.joint_friction),
        limited="false",
    )

    wheel_col = _sub(wheel, "default", **{"class": "db/wheel_col"})
    _sub(
        wheel_col,
        "geom",
        type="sphere",
        size=_fmt(robot.wheel_radius),
        group=str(_COLLISION_GROUP),
        condim="3",
        priority="2",
        friction=_vec((robot.wheel_friction, 0.005, 0.0001)),
        solref=_vec(sim.solref_wheel),
        solimp=_vec(sim.solimp_wheel),
        rgba="1 0 0 0.25",
    )

    wheel_vis = _sub(wheel, "default", **{"class": "db/wheel_vis"})
    _sub(
        wheel_vis,
        "geom",
        type="cylinder",
        size=_vec((robot.wheel_radius, 0.5 * robot.wheel_width)),
        euler=_vec((radians(90.0), 0.0, 0.0)),
        group=str(_VISUAL_GROUP),
        contype="0",
        conaffinity="0",
        density="0",
        rgba=_vec(_RGBA_WHEEL),
    )

    caster = _sub(db, "default", **{"class": "db/caster"})
    _sub(
        caster,
        "geom",
        type="sphere",
        size=_fmt(robot.caster_radius),
        group=str(_COLLISION_GROUP),
        condim="1",
        priority="1",
        friction=_vec((robot.caster_friction, 0.0, 0.0)),
        solref=_vec(sim.solref_default),
        rgba=_vec(_RGBA_CASTER),
    )

    collision = _sub(db, "default", **{"class": "db/collision"})
    _sub(collision, "geom", group=str(_COLLISION_GROUP), rgba="1 0 0 0.25")

    visual = _sub(db, "default", **{"class": "db/visual"})
    _sub(
        visual,
        "geom",
        group=str(_VISUAL_GROUP),
        contype="0",
        conaffinity="0",
        density="0",
    )
    return root


def build_robot_body(cfg: MjcfCfg, pos: Sequence[float] = (0.0, 0.0, 0.0), yaw: float = 0.0) -> ET.Element:
    """Return the Duckiebot ``<body>`` element, ready to append to a ``<worldbody>``.

    The body frame is ``base_link``: origin at the wheel-axle midpoint, x forward, y left, z up.
    ``pos[2]`` is ignored and replaced by :attr:`RobotParams.base_height` so the robot always starts
    exactly seated on its wheels; a buried sphere against a stiff contact launches the chassis.

    Args:
        cfg: the MJCF configuration.
        pos: world ``(x, y, z)``; only x and y are used.
        yaw: initial heading in radians.

    Returns:
        The ``<body>`` element, containing a free joint, both wheel bodies, the camera and the
        collision and visual geoms.
    """
    robot = cfg.robot
    body = ET.Element(
        "body",
        {
            "name": robot.base_link_name,
            "pos": _vec((pos[0], pos[1], robot.base_height)),
            "euler": _vec((0.0, 0.0, yaw)),
            "childclass": "db",
        },
    )
    _sub(body, "freejoint", name="root")
    _sub(
        body,
        "inertial",
        pos=_vec(robot.base_com),
        mass=_fmt(robot.base_mass),
        diaginertia=_vec(robot.base_inertia_diag),
    )
    _sub(body, "site", name=IMU_SITE_NAME, pos="0 0 0", size="0.005", group="4", rgba="1 0 0 0.3")

    _sub(
        body,
        "geom",
        name=f"{robot.base_link_name}_collision",
        **{"class": "db/collision"},
        type="box",
        size=_vec(0.5 * s for s in robot.chassis_size),
        pos=_vec(robot.chassis_center),
    )
    _sub(
        body,
        "geom",
        name="caster_collision",
        **{"class": "db/caster"},
        pos=_vec(robot.caster_center),
    )

    if cfg.include_visuals:
        _sub(
            body,
            "geom",
            name=f"{robot.base_link_name}_visual",
            **{"class": "db/visual"},
            type="box",
            size=_vec(0.5 * s for s in robot.chassis_size),
            pos=_vec(robot.chassis_center),
            rgba=_vec(_RGBA_CHASSIS),
        )
        _sub(
            body,
            "geom",
            name="camera_block_visual",
            **{"class": "db/visual"},
            type="box",
            size=_vec(0.5 * s for s in robot.camera_block_size),
            pos=_vec(cfg.mount_pos),
            rgba=_vec(_RGBA_CAMERA),
        )
        _sub(
            body,
            "geom",
            name="marker_visual",
            **{"class": "db/visual"},
            type="sphere",
            size=_fmt(robot.marker_radius),
            pos=_vec(robot.marker_center),
            rgba=_vec(_RGBA_MARKER),
        )
        _sub(
            body,
            "geom",
            name="caster_visual",
            **{"class": "db/visual"},
            type="sphere",
            size=_fmt(robot.caster_radius),
            pos=_vec(robot.caster_center),
            rgba=_vec(_RGBA_CASTER),
        )

    if cfg.include_camera:
        _sub(
            body,
            "camera",
            name=CAMERA_NAME,
            mode="fixed",
            pos=_vec(cfg.mount_pos),
            xyaxes=_vec(mj_camera_xyaxes(cfg.pitch_down_rad)),
            fovy=_fmt(robot.camera_fovy_deg),
            resolution=f"{int(cfg.obs.render_w)} {int(cfg.obs.render_h)}",
        )

    half = 0.5 * robot.wheel_separation
    wheels = (
        (robot.left_wheel_link_name, robot.left_wheel_joint_name, +half),
        (robot.right_wheel_link_name, robot.right_wheel_joint_name, -half),
    )
    for link_name, joint_name, y_offset in wheels:
        wheel = _sub(body, "body", name=link_name, pos=_vec((0.0, y_offset, 0.0)))
        _sub(wheel, "joint", name=joint_name, **{"class": "db/wheel"})
        _sub(
            wheel,
            "inertial",
            pos="0 0 0",
            mass=_fmt(robot.wheel_mass),
            diaginertia=_vec(robot.wheel_inertia_diag),
        )
        _sub(wheel, "geom", name=f"{link_name}_collision", **{"class": "db/wheel_col"})
        if cfg.include_visuals:
            _sub(wheel, "geom", name=f"{link_name}_visual", **{"class": "db/wheel_vis"})
    return body


def _actuators(cfg: MjcfCfg) -> ET.Element:
    """Return the ``<actuator>`` element with one velocity servo per wheel."""
    robot = cfg.robot
    element = ET.Element("actuator")
    pairs = (
        ("act_left_wheel", robot.left_wheel_joint_name),
        ("act_right_wheel", robot.right_wheel_joint_name),
    )
    for name, joint in pairs:
        _sub(
            element,
            "velocity",
            name=name,
            joint=joint,
            kv=_fmt(robot.joint_damping),
            ctrlrange=_vec((-robot.velocity_limit, robot.velocity_limit)),
            forcerange=_vec((-robot.effort_limit, robot.effort_limit)),
        )
    return element


def _sensors(cfg: MjcfCfg) -> ET.Element:
    """Return the ``<sensor>`` element supplying everything the vec observation needs."""
    robot = cfg.robot
    element = ET.Element("sensor")
    base = robot.base_link_name
    _sub(element, "framepos", name="base_pos", objtype="body", objname=base)
    _sub(element, "framequat", name="base_quat", objtype="body", objname=base)
    _sub(element, "framelinvel", name="base_linvel", objtype="body", objname=base)
    _sub(element, "frameangvel", name="base_angvel", objtype="body", objname=base)
    _sub(element, "velocimeter", name="body_vel", site=IMU_SITE_NAME)
    _sub(element, "gyro", name="imu_gyro", site=IMU_SITE_NAME)
    _sub(element, "accelerometer", name="imu_acc", site=IMU_SITE_NAME)
    _sub(element, "jointvel", name="wheel_vel_left", joint=robot.left_wheel_joint_name)
    _sub(element, "jointvel", name="wheel_vel_right", joint=robot.right_wheel_joint_name)
    _sub(element, "jointpos", name="wheel_pos_left", joint=robot.left_wheel_joint_name)
    _sub(element, "jointpos", name="wheel_pos_right", joint=robot.right_wheel_joint_name)
    _sub(element, "actuatorfrc", name="wheel_torque_left", actuator="act_left_wheel")
    _sub(element, "actuatorfrc", name="wheel_torque_right", actuator="act_right_wheel")
    return element


def ground_geom(cfg: MjcfCfg, half_extent: float = 5.0, material: str | None = None) -> ET.Element:
    """Return the single collidable ground plane.

    This is the MuJoCo counterpart of the one authored ``ground.usda`` plane in Isaac (S3.3). It is
    the *only* geom in a track scene that collides, which is why tiles and walls elsewhere in the
    package are emitted with ``contype=0 conaffinity=0``.

    Args:
        cfg: the MJCF configuration (supplies the contact softness defaults).
        half_extent: half-size of the plane in metres.
        material: optional material name; when None a flat grass-coloured rgba is used.

    Returns:
        The ``<geom>`` element.
    """
    attrs = {
        "name": GROUND_GEOM_NAME,
        "type": "plane",
        "size": _vec((half_extent, half_extent, 0.05)),
        "condim": "3",
        "friction": _vec((1.0, 0.005, 0.0001)),
        "solref": _vec(cfg.sim.solref_default),
        "solimp": _vec(cfg.sim.solimp_default),
    }
    if material is None:
        attrs["rgba"] = _vec(_RGBA_GROUND)
    else:
        attrs["material"] = material
    return ET.Element("geom", attrs)


def _lights(cfg: MjcfCfg) -> list[ET.Element]:
    """Return the default key and fill lights."""
    key = ET.Element(
        "light",
        {
            "name": "sun",
            "directional": "true",
            "diffuse": "0.7 0.7 0.7",
            "specular": "0.1 0.1 0.1",
            "pos": "0 0 3",
            "dir": "0.2 0.2 -1",
            "castshadow": "true" if cfg.shadows else "false",
        },
    )
    fill = ET.Element(
        "light",
        {
            "name": "fill",
            "directional": "true",
            "diffuse": "0.25 0.25 0.25",
            "pos": "-2 -2 2",
            "dir": "0.4 0.4 -1",
            "castshadow": "false",
        },
    )
    return [key, fill]


def build_scene_xml(
    cfg: MjcfCfg,
    world_children: Sequence[ET.Element] = (),
    assets: Sequence[ET.Element] = (),
    include_skybox: bool = True,
) -> str:
    """Assemble a complete MJCF document around the robot.

    Args:
        cfg: the MJCF configuration.
        world_children: extra elements appended to ``<worldbody>`` (ground, tiles, walls,
            obstacles). The caller owns the ground plane so a track can bind a material to it.
        assets: extra elements appended to ``<asset>`` (textures and materials).
        include_skybox: emit the built-in gradient skybox.

    Returns:
        The MJCF document as a UTF-8 string, indented and with a provenance comment.
    """
    root = ET.Element("mujoco", {"model": cfg.model_name})
    if cfg.params_source:
        root.append(ET.Comment(f" generated by duckiebot_rl.sim2sim.mjcf; {cfg.params_source} "))
    root.append(_compiler(cfg))
    root.append(_option(cfg))
    root.append(_statistic())
    root.append(_visual(cfg))
    root.append(_defaults(cfg))

    asset = ET.SubElement(root, "asset")
    if include_skybox:
        _sub(
            asset,
            "texture",
            name="skybox",
            type="skybox",
            builtin="gradient",
            rgb1="0.45 0.82 1.0",
            rgb2="0.85 0.93 1.0",
            width="512",
            height="512",
        )
    for element in assets:
        asset.append(element)

    world = ET.SubElement(root, "worldbody")
    for light in _lights(cfg):
        world.append(light)
    for child in world_children:
        world.append(child)

    root.append(_actuators(cfg))
    root.append(_sensors(cfg))

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode") + "\n"


def build_robot_xml(
    cfg: MjcfCfg | None = None,
    pos: Sequence[float] = (0.0, 0.0, 0.0),
    yaw: float = 0.0,
    ground: bool = True,
    ground_half_extent: float = 5.0,
) -> str:
    """Return a standalone MJCF containing the robot on a flat plane.

    This is the model used by system identification (:mod:`duckiebot_rl.sim2sim.sysid`) and by the
    differential-drive parity test, where a track would only add contact noise.

    Args:
        cfg: the MJCF configuration; a fresh :meth:`MjcfCfg.from_shared` is used when None.
        pos: initial world ``(x, y)`` of ``base_link``.
        yaw: initial heading in radians.
        ground: include the collidable ground plane.
        ground_half_extent: half-size of that plane in metres.

    Returns:
        The MJCF document as a string.
    """
    cfg = cfg if cfg is not None else MjcfCfg.from_shared()
    children: list[ET.Element] = []
    if ground:
        children.append(ground_geom(cfg, ground_half_extent))
    children.append(build_robot_body(cfg, pos=pos, yaw=yaw))
    return build_scene_xml(cfg, world_children=children)


def write_robot_xml(path: str | Path, cfg: MjcfCfg | None = None, **kwargs: object) -> Path:
    """Write :func:`build_robot_xml` to disk.

    Args:
        path: destination file path.
        cfg: the MJCF configuration, or None to resolve a fresh one.
        **kwargs: forwarded to :func:`build_robot_xml`.

    Returns:
        The resolved destination path.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(build_robot_xml(cfg, **kwargs), encoding="utf-8")  # type: ignore[arg-type]
    return destination


if __name__ == "__main__":  # pragma: no cover - manual smoke entry point
    print(build_robot_xml())
