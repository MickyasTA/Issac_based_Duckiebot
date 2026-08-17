"""Generate the clean-room Duckiebot URDF from :mod:`duckiebot_rl.assets.params`.

SPEC v2 S3.2. Every visual and collision shape is an authored primitive: ``box``, ``cylinder`` or
``sphere``. There is no ``<mesh>`` element anywhere, by construction and by test
(``tests/unit/test_urdf.py``), because the Duckietown asset licence grants no redistribution
right. Dimensions are facts and are reproduced from published figures; the geometry that realizes
them here is original.

Link tree produced (3 links, 2 joints, matching the M1 acceptance of "3 bodies / 2 DOF"):

.. code-block:: text

    base_link                       mass 1.000 kg, CoM (-0.015, 0, +0.015)
      collision  box    0.180 x 0.130 x 0.075 at (-0.015, 0, +0.0267)   underside at 21 mm
      collision  sphere r 0.0165           at (-0.085, 0, -0.0153)      caster, frictionless
      visual     box    chassis, camera housing, marker sphere, caster ball
      |
      +-- left_wheel_joint   continuous, origin (0, +0.050, 0), axis (0, 1, 0)
      |     left_wheel_link  mass 0.050 kg
      |       collision  sphere   r 0.0318 at the joint origin
      |       visual     cylinder r 0.0318, length 0.027
      |
      +-- right_wheel_joint  mirrored at (0, -0.050, 0)
            right_wheel_link

Three design decisions worth stating explicitly, because each reverses a v1 choice:

* **The caster is geometry on** ``base_link``, not a link. SPEC v2 S2 says it is "merged into base
  (no joint, no inertial tag)". A URDF link with no joint is not valid URDF anyway, and relying on
  the importer's fixed-joint merging to reach 3 bodies is a bet on importer behaviour. Authoring
  it as a second collision shape on ``base_link`` reaches the required body count directly.
* **Wheel colliders are spheres, never cylinders.** The MuJoCo study behind the spec measured a
  74% loss of yaw response with cylinder wheel contact. The visual stays a cylinder.
* **There is no** ``camera_link``. The v1 URDF encoded a 0.066 m / 19.15 deg mount that the same
  document's text rejected, and a second camera prim was then spawned at the correct pose. The
  camera is spawned at runtime by ``TiledCameraCfg.OffsetCfg`` from
  :attr:`~duckiebot_rl.assets.params.DuckiebotParams.camera_pos_base_frame_m` and
  :attr:`~duckiebot_rl.assets.params.DuckiebotParams.camera_pitch_down_deg`. The cube this module
  emits at that pose is a visual housing only: it carries no frame, no collider and no inertia.

Inertia tensors are computed analytically by :func:`box_inertia`, :func:`cylinder_inertia` and
:func:`sphere_inertia`; nothing is hand-tuned or copied.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from duckiebot_rl.assets.params import DUCKIEBOT, DuckiebotParams

__all__ = [
    "ROBOT_NAME",
    "URDF_FILENAME",
    "InertiaTensor",
    "box_inertia",
    "build_urdf_tree",
    "cylinder_inertia",
    "generate_urdf",
    "sphere_inertia",
    "write_urdf",
]

URDF_FILENAME = "duckiebot.urdf"
"""Canonical filename of the generated robot description."""

ROBOT_NAME = "duckiebot"
"""The ``<robot name=...>`` attribute; also the prim name after USD import."""

# Visual colours, RGBA in [0, 1]. Deliberately plain: the simulator randomizes the robot's own
# albedo anyway, and the robot is never in its own camera view.
_COLOR_CHASSIS = (0.15, 0.15, 0.17, 1.0)
_COLOR_WHEEL = (0.08, 0.08, 0.08, 1.0)
_COLOR_CAMERA = (0.05, 0.05, 0.05, 1.0)
_COLOR_MARKER = (1.00, 0.85, 0.10, 1.0)
_COLOR_CASTER = (0.75, 0.75, 0.75, 1.0)

_MATERIALS: dict[str, tuple[float, float, float, float]] = {
    "duckiebot_chassis_gray": _COLOR_CHASSIS,
    "duckiebot_wheel_black": _COLOR_WHEEL,
    "duckiebot_camera_black": _COLOR_CAMERA,
    "duckiebot_marker_yellow": _COLOR_MARKER,
    "duckiebot_caster_gray": _COLOR_CASTER,
}

# Rotation that takes the URDF cylinder's local +z axis onto the robot's +y axle direction.
_HALF_PI = 1.5707963267948966

InertiaTensor = tuple[float, float, float, float, float, float]
"""Rotational inertia as ``(ixx, ixy, ixz, iyy, iyz, izz)`` in kg.m^2, URDF attribute order."""


def _fmt(value: float) -> str:
    """Format a float for a URDF attribute.

    Uses 9 significant digits, which round-trips every constant in
    :mod:`duckiebot_rl.assets.params` exactly while keeping the file readable.

    Args:
        value: The number to format.

    Returns:
        The formatted string, with ``-0`` normalised to ``0``.
    """
    text = f"{value:.9g}"
    return "0" if text == "-0" else text


def _fmt_xyz(vec: tuple[float, float, float]) -> str:
    """Format a 3-vector as a URDF ``xyz`` attribute.

    Args:
        vec: The ``(x, y, z)`` values.

    Returns:
        Space-separated formatted components.
    """
    return " ".join(_fmt(v) for v in vec)


def box_inertia(mass: float, size: tuple[float, float, float]) -> InertiaTensor:
    """Rotational inertia of a solid rectangular box about its centre of mass.

    ``Ixx = m (b^2 + c^2) / 12`` and cyclic permutations, where ``a, b, c`` are the full extents
    along x, y and z. Products of inertia vanish for an axis-aligned box.

    Args:
        mass: Body mass in kg. Must be strictly positive.
        size: Full extents ``(a, b, c)`` along x, y, z in metres. All strictly positive.

    Returns:
        The tensor as ``(ixx, ixy, ixz, iyy, iyz, izz)`` in kg.m^2.

    Raises:
        ValueError: If the mass or any extent is not strictly positive.
    """
    if mass <= 0.0:
        raise ValueError(f"box mass must be positive, got {mass}")
    if min(size) <= 0.0:
        raise ValueError(f"box extents must all be positive, got {size}")
    a, b, c = size
    return (
        mass * (b * b + c * c) / 12.0,
        0.0,
        0.0,
        mass * (a * a + c * c) / 12.0,
        0.0,
        mass * (a * a + b * b) / 12.0,
    )


def cylinder_inertia(mass: float, radius: float, length: float, axis: str = "z") -> InertiaTensor:
    """Rotational inertia of a solid cylinder about its centre of mass.

    Axial moment ``I_axial = m r^2 / 2``; transverse moments ``I_t = m (3 r^2 + h^2) / 12``.

    Args:
        mass: Body mass in kg. Must be strictly positive.
        radius: Cylinder radius in metres. Must be strictly positive.
        length: Cylinder length along its axis in metres. Must be strictly positive.
        axis: Which body axis the cylinder spins about: ``"x"``, ``"y"`` or ``"z"``.

    Returns:
        The tensor as ``(ixx, ixy, ixz, iyy, iyz, izz)`` in kg.m^2.

    Raises:
        ValueError: If any argument is non-positive or ``axis`` is not one of x, y, z.
    """
    if mass <= 0.0:
        raise ValueError(f"cylinder mass must be positive, got {mass}")
    if radius <= 0.0 or length <= 0.0:
        raise ValueError(f"cylinder radius and length must be positive, got {radius}, {length}")
    if axis not in ("x", "y", "z"):
        raise ValueError(f"cylinder axis must be one of x, y, z; got {axis!r}")
    i_axial = 0.5 * mass * radius * radius
    i_transverse = mass * (3.0 * radius * radius + length * length) / 12.0
    diag = {
        "x": (i_axial, i_transverse, i_transverse),
        "y": (i_transverse, i_axial, i_transverse),
        "z": (i_transverse, i_transverse, i_axial),
    }[axis]
    return (diag[0], 0.0, 0.0, diag[1], 0.0, diag[2])


def sphere_inertia(mass: float, radius: float) -> InertiaTensor:
    """Rotational inertia of a solid sphere about its centre of mass.

    ``I = 2 m r^2 / 5`` on every axis.

    Args:
        mass: Body mass in kg. Must be strictly positive.
        radius: Sphere radius in metres. Must be strictly positive.

    Returns:
        The tensor as ``(ixx, ixy, ixz, iyy, iyz, izz)`` in kg.m^2.

    Raises:
        ValueError: If the mass or the radius is not strictly positive.
    """
    if mass <= 0.0:
        raise ValueError(f"sphere mass must be positive, got {mass}")
    if radius <= 0.0:
        raise ValueError(f"sphere radius must be positive, got {radius}")
    i = 0.4 * mass * radius * radius
    return (i, 0.0, 0.0, i, 0.0, i)


def _add_origin(
    parent: ET.Element,
    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> ET.Element:
    """Append an ``<origin>`` element.

    Args:
        parent: Element to append to.
        xyz: Translation in metres.
        rpy: Fixed-axis roll, pitch, yaw in radians.

    Returns:
        The created element.
    """
    return ET.SubElement(parent, "origin", {"xyz": _fmt_xyz(xyz), "rpy": _fmt_xyz(rpy)})


def _add_inertial(
    link: ET.Element,
    mass: float,
    inertia: InertiaTensor,
    com: tuple[float, float, float],
) -> None:
    """Append a complete ``<inertial>`` block to a link.

    Args:
        link: The ``<link>`` element.
        mass: Link mass in kg.
        inertia: Tensor about the centre of mass, in the link frame.
        com: Centre-of-mass position in the link frame, in metres.
    """
    inertial = ET.SubElement(link, "inertial")
    _add_origin(inertial, com)
    ET.SubElement(inertial, "mass", {"value": _fmt(mass)})
    ixx, ixy, ixz, iyy, iyz, izz = inertia
    ET.SubElement(
        inertial,
        "inertia",
        {
            "ixx": _fmt(ixx),
            "ixy": _fmt(ixy),
            "ixz": _fmt(ixz),
            "iyy": _fmt(iyy),
            "iyz": _fmt(iyz),
            "izz": _fmt(izz),
        },
    )


def _add_shape(
    link: ET.Element,
    kind: str,
    name: str,
    geometry: tuple[str, dict[str, str]],
    xyz: tuple[float, float, float],
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
    material: str | None = None,
) -> ET.Element:
    """Append a ``<visual>`` or ``<collision>`` element carrying one primitive.

    Args:
        link: The ``<link>`` element.
        kind: Either ``"visual"`` or ``"collision"``.
        name: Element name attribute, used by the tests and by USD prim naming.
        geometry: ``(tag, attributes)`` for the primitive, e.g. ``("sphere", {"radius": "0.03"})``.
        xyz: Shape origin in the link frame, in metres.
        rpy: Shape orientation as fixed-axis roll, pitch, yaw in radians.
        material: Name of a top-level ``<material>`` to reference. Visuals only.

    Returns:
        The created element.

    Raises:
        ValueError: If ``kind`` is not ``"visual"`` or ``"collision"``, or if the geometry tag is
            not one of the three allowed primitives (the clean-room guard: a ``mesh`` tag can
            never be produced by this function).
    """
    if kind not in ("visual", "collision"):
        raise ValueError(f"shape kind must be visual or collision, got {kind!r}")
    tag, attrs = geometry
    if tag not in ("box", "cylinder", "sphere"):
        raise ValueError(
            f"only box, cylinder and sphere primitives may be authored, got {tag!r}. "
            "Mesh geometry is forbidden by the clean-room policy (SPEC v2 S3.1/S3.4)."
        )
    element = ET.SubElement(link, kind, {"name": name})
    _add_origin(element, xyz, rpy)
    geom = ET.SubElement(element, "geometry")
    ET.SubElement(geom, tag, attrs)
    if material is not None:
        ET.SubElement(element, "material", {"name": material})
    return element


def _box_geometry(size: tuple[float, float, float]) -> tuple[str, dict[str, str]]:
    """Build a ``<box>`` geometry descriptor.

    Args:
        size: Full extents in metres.

    Returns:
        A ``(tag, attributes)`` pair for :func:`_add_shape`.
    """
    return ("box", {"size": _fmt_xyz(size)})


def _sphere_geometry(radius: float) -> tuple[str, dict[str, str]]:
    """Build a ``<sphere>`` geometry descriptor.

    Args:
        radius: Sphere radius in metres.

    Returns:
        A ``(tag, attributes)`` pair for :func:`_add_shape`.
    """
    return ("sphere", {"radius": _fmt(radius)})


def _cylinder_geometry(radius: float, length: float) -> tuple[str, dict[str, str]]:
    """Build a ``<cylinder>`` geometry descriptor.

    Args:
        radius: Cylinder radius in metres.
        length: Cylinder length along its local z axis, in metres.

    Returns:
        A ``(tag, attributes)`` pair for :func:`_add_shape`.
    """
    return ("cylinder", {"radius": _fmt(radius), "length": _fmt(length)})


def _add_base_link(robot: ET.Element, params: DuckiebotParams) -> None:
    """Append ``base_link`` with its inertial, collision and visual content.

    Args:
        robot: The ``<robot>`` root element.
        params: The parameter set to author from.
    """
    link = ET.SubElement(robot, "link", {"name": params.base_link_name})
    _add_inertial(
        link,
        params.base_mass_kg,
        box_inertia(params.base_mass_kg, params.chassis_size_m),
        params.base_com_base_frame_m,
    )

    link.append(ET.Comment(" collision: chassis box, underside at the 21 mm ground clearance "))
    _add_shape(
        link,
        "collision",
        "chassis_collision",
        _box_geometry(params.chassis_size_m),
        params.chassis_center_base_frame_m,
    )
    link.append(
        ET.Comment(
            " collision: rear caster ball; contact point lies exactly on z = 0. "
            "A frictionless physics material is bound to this shape by tools/patch_usd.py. "
        )
    )
    _add_shape(
        link,
        "collision",
        "caster_collision",
        _sphere_geometry(params.caster_radius_m),
        params.caster_center_base_frame_m,
    )

    _add_shape(
        link,
        "visual",
        "chassis_visual",
        _box_geometry(params.chassis_size_m),
        params.chassis_center_base_frame_m,
        material="duckiebot_chassis_gray",
    )
    link.append(
        ET.Comment(
            " visual only: the camera housing. It defines no frame. The optical pose comes from "
            "TiledCameraCfg.OffsetCfg, which reads params.camera_pos_base_frame_m and "
            "params.camera_pitch_down_deg. There is deliberately no camera_link. "
        )
    )
    _add_shape(
        link,
        "visual",
        "camera_housing_visual",
        _box_geometry(params.camera_block_size_m),
        params.camera_pos_base_frame_m,
        material="duckiebot_camera_black",
    )
    _add_shape(
        link,
        "visual",
        "marker_visual",
        _sphere_geometry(params.duckie_marker_radius_m),
        params.duckie_marker_center_base_frame_m,
        material="duckiebot_marker_yellow",
    )
    _add_shape(
        link,
        "visual",
        "caster_visual",
        _sphere_geometry(params.caster_radius_m),
        params.caster_center_base_frame_m,
        material="duckiebot_caster_gray",
    )


def _add_wheel(robot: ET.Element, params: DuckiebotParams, side: str) -> None:
    """Append one wheel link and its continuous joint.

    Args:
        robot: The ``<robot>`` root element.
        params: The parameter set to author from.
        side: Either ``"left"`` or ``"right"``.

    Raises:
        ValueError: If ``side`` is not ``"left"`` or ``"right"``.
    """
    if side == "left":
        link_name = params.left_wheel_link_name
        joint_name = params.left_wheel_joint_name
        origin = params.left_wheel_origin_m
    elif side == "right":
        link_name = params.right_wheel_link_name
        joint_name = params.right_wheel_joint_name
        origin = params.right_wheel_origin_m
    else:
        raise ValueError(f"side must be left or right, got {side!r}")

    link = ET.SubElement(robot, "link", {"name": link_name})
    _add_inertial(
        link,
        params.wheel_mass_kg,
        cylinder_inertia(params.wheel_mass_kg, params.wheel_radius_m, params.wheel_width_m, axis="y"),
        (0.0, 0.0, 0.0),
    )
    link.append(
        ET.Comment(
            " collision: SPHERE, never a cylinder. A cylindrical wheel collider costs about 74% of "
            "the robot's yaw response because the contact degenerates to a line. "
        )
    )
    _add_shape(
        link,
        "collision",
        f"{side}_wheel_collision",
        _sphere_geometry(params.wheel_radius_m),
        (0.0, 0.0, 0.0),
    )
    _add_shape(
        link,
        "visual",
        f"{side}_wheel_visual",
        _cylinder_geometry(params.wheel_radius_m, params.wheel_width_m),
        (0.0, 0.0, 0.0),
        rpy=(_HALF_PI, 0.0, 0.0),
        material="duckiebot_wheel_black",
    )

    joint = ET.SubElement(robot, "joint", {"name": joint_name, "type": "continuous"})
    ET.SubElement(joint, "parent", {"link": params.base_link_name})
    ET.SubElement(joint, "child", {"link": link_name})
    _add_origin(joint, origin)
    ET.SubElement(joint, "axis", {"xyz": "0 1 0"})
    ET.SubElement(
        joint,
        "limit",
        {
            "effort": _fmt(params.wheel_effort_limit_nm),
            "velocity": _fmt(params.wheel_velocity_limit_rad_s),
        },
    )
    ET.SubElement(
        joint,
        "dynamics",
        {"damping": _fmt(params.joint_damping), "friction": _fmt(params.joint_friction_nm)},
    )


def build_urdf_tree(params: DuckiebotParams = DUCKIEBOT) -> ET.Element:
    """Build the URDF document as an ElementTree element.

    Args:
        params: The parameter set to author from. Defaults to the shared
            :data:`~duckiebot_rl.assets.params.DUCKIEBOT` singleton.

    Returns:
        The ``<robot>`` root element, with materials, three links and two joints.
    """
    robot = ET.Element("robot", {"name": ROBOT_NAME})
    robot.append(
        ET.Comment(
            " Generated by duckiebot_rl.assets.urdf from duckiebot_rl.assets.params. "
            "Do not edit by hand: rerun scripts/build_robot_asset.py. "
        )
    )
    robot.append(
        ET.Comment(
            " Clean room: every shape below is an authored primitive (box, cylinder, sphere). "
            "No mesh file is referenced, and none may ever be added. "
        )
    )
    for name, rgba in _MATERIALS.items():
        material = ET.SubElement(robot, "material", {"name": name})
        ET.SubElement(material, "color", {"rgba": " ".join(_fmt(c) for c in rgba)})
    _add_base_link(robot, params)
    _add_wheel(robot, params, "left")
    _add_wheel(robot, params, "right")
    return robot


def generate_urdf(params: DuckiebotParams = DUCKIEBOT) -> str:
    """Render the URDF to a formatted XML string.

    Args:
        params: The parameter set to author from.

    Returns:
        The complete URDF document, including the XML declaration and a trailing newline.
    """
    root = build_urdf_tree(params)
    ET.indent(root, space="  ")
    body = ET.tostring(root, encoding="unicode")
    return f'<?xml version="1.0"?>\n{body}\n'


def write_urdf(path: str | Path, params: DuckiebotParams = DUCKIEBOT) -> Path:
    """Write the URDF to disk, creating parent directories as needed.

    Args:
        path: Destination file path.
        params: The parameter set to author from.

    Returns:
        The resolved path that was written.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(generate_urdf(params), encoding="utf-8", newline="\n")
    return destination.resolve()
