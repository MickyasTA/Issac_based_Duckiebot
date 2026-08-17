"""Unit tests for the generated Duckiebot URDF (SPEC v2 S3.2, clean-room gate S3.4).

These tests run on CPU with no GPU, no Isaac Sim and no third-party dependency beyond pytest.
They parse the URDF that :mod:`duckiebot_rl.assets.urdf` produces and check four families of
property:

1. structure: 3 links, 2 joints, one root, a valid tree;
2. physics: every inertia tensor is diagonal, positive definite and satisfies the triangle
   inequality on its principal moments, and the masses sum to the S2 figure;
3. the clean-room gate: no ``<mesh>`` element anywhere and no geometry tag outside
   ``box`` / ``cylinder`` / ``sphere``;
4. the four numbers the v1 critique found self-contradictory: caster radius and contact height,
   chassis ground clearance, wheel effort limit, and the absence of a ``camera_link``.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from duckiebot_rl.assets.params import DUCKIEBOT  # noqa: E402
from duckiebot_rl.assets.urdf import (  # noqa: E402
    ROBOT_NAME,
    URDF_FILENAME,
    box_inertia,
    build_urdf_tree,
    cylinder_inertia,
    generate_urdf,
    sphere_inertia,
    write_urdf,
)

# Geometry primitives the clean-room policy permits. Anything else, and in particular "mesh",
# is a licence violation as well as a spec violation.
ALLOWED_GEOMETRY_TAGS = frozenset({"box", "cylinder", "sphere"})
BANNED_GEOMETRY_TAGS = frozenset({"mesh"})

# Relative tolerance for comparing generated attribute values against recomputed floats. The
# generator writes 9 significant digits, whose worst-case relative round-off is just under 5e-9
# (a half unit in the last place of a mantissa leading with 1), so 1e-8 is the round-trip floor.
REL_TOL = 1e-8


@pytest.fixture(scope="module")
def urdf_text() -> str:
    """The generated URDF document.

    Returns:
        The URDF as a string.
    """
    return generate_urdf(DUCKIEBOT)


@pytest.fixture(scope="module")
def root(urdf_text: str) -> ET.Element:
    """The parsed URDF root element.

    Args:
        urdf_text: The generated document.

    Returns:
        The ``<robot>`` element.
    """
    return ET.fromstring(urdf_text)


def _origin_xyz(element: ET.Element) -> tuple[float, float, float]:
    """Read an element's ``<origin xyz=...>``.

    Args:
        element: A ``<visual>``, ``<collision>``, ``<inertial>`` or ``<joint>`` element.

    Returns:
        The translation as a 3-tuple of floats.
    """
    origin = element.find("origin")
    assert origin is not None, "every shape and joint must carry an explicit <origin>"
    values = [float(v) for v in origin.attrib["xyz"].split()]
    assert len(values) == 3
    return (values[0], values[1], values[2])


def _find_named(link: ET.Element, tag: str, name: str) -> ET.Element:
    """Find a named ``<visual>`` or ``<collision>`` child of a link.

    Args:
        link: The ``<link>`` element.
        tag: Either ``"visual"`` or ``"collision"``.
        name: The value of the ``name`` attribute.

    Returns:
        The matching element.
    """
    for child in link.findall(tag):
        if child.attrib.get("name") == name:
            return child
    raise AssertionError(f"link {link.attrib['name']!r} has no {tag} named {name!r}")


def _links(root: ET.Element) -> dict[str, ET.Element]:
    """Index the links by name.

    Args:
        root: The ``<robot>`` element.

    Returns:
        Mapping from link name to element.
    """
    return {link.attrib["name"]: link for link in root.findall("link")}


def _joints(root: ET.Element) -> dict[str, ET.Element]:
    """Index the joints by name.

    Args:
        root: The ``<robot>`` element.

    Returns:
        Mapping from joint name to element.
    """
    return {joint.attrib["name"]: joint for joint in root.findall("joint")}


# ---------------------------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------------------------


def test_robot_name_and_xml_declaration(urdf_text: str, root: ET.Element) -> None:
    """The document is well formed and names the robot as the rest of the pipeline expects."""
    assert urdf_text.startswith('<?xml version="1.0"?>\n')
    assert urdf_text.endswith("\n")
    assert root.tag == "robot"
    assert root.attrib["name"] == ROBOT_NAME


def test_link_and_joint_counts(root: ET.Element) -> None:
    """Exactly 3 links and 2 joints: the M1 acceptance criterion of 3 bodies / 2 DOF."""
    links = _links(root)
    joints = _joints(root)
    assert len(links) == 3, f"expected 3 links, got {sorted(links)}"
    assert len(joints) == 2, f"expected 2 joints, got {sorted(joints)}"
    assert set(links) == {
        DUCKIEBOT.base_link_name,
        DUCKIEBOT.left_wheel_link_name,
        DUCKIEBOT.right_wheel_link_name,
    }
    assert set(joints) == {DUCKIEBOT.left_wheel_joint_name, DUCKIEBOT.right_wheel_joint_name}


def test_kinematic_tree_is_valid_and_has_one_root(root: ET.Element) -> None:
    """Every joint connects declared links, and exactly one link is nobody's child."""
    links = _links(root)
    children: list[str] = []
    for joint in root.findall("joint"):
        parent_link = joint.find("parent")
        child_link = joint.find("child")
        assert parent_link is not None and child_link is not None
        parent_name = parent_link.attrib["link"]
        child_name = child_link.attrib["link"]
        assert parent_name in links, f"joint references unknown parent {parent_name!r}"
        assert child_name in links, f"joint references unknown child {child_name!r}"
        children.append(child_name)
    assert len(children) == len(set(children)), "a link has two parents; that is not a tree"
    roots = set(links) - set(children)
    assert roots == {DUCKIEBOT.base_link_name}


def test_no_camera_link_exists(root: ET.Element) -> None:
    """There is no ``camera_link`` (critic item F).

    The v1 URDF carried a ``camera_link`` at 0.066 m / 19.15 deg, which is exactly the mount the
    surrounding text rejected, and the environment then spawned a second camera prim at the
    correct pose. The camera pose now has one source: the params module.
    """
    names = set(_links(root)) | set(_joints(root))
    assert not any("camera" in name for name in names), f"a camera frame leaked back in: {names}"


def test_camera_housing_is_visual_only(root: ET.Element) -> None:
    """The camera cube is decoration: a visual with no collider and no inertia of its own."""
    base = _links(root)[DUCKIEBOT.base_link_name]
    housing = _find_named(base, "visual", "camera_housing_visual")
    assert _origin_xyz(housing) == pytest.approx(DUCKIEBOT.camera_pos_base_frame_m, rel=REL_TOL)
    collision_names = {c.attrib["name"] for c in base.findall("collision")}
    assert collision_names == {"chassis_collision", "caster_collision"}


# ---------------------------------------------------------------------------------------------
# Clean-room gate
# ---------------------------------------------------------------------------------------------


def test_no_mesh_element_anywhere(root: ET.Element, urdf_text: str) -> None:
    """The clean-room gate: not a single ``<mesh>`` element, and no ``filename`` attribute.

    A ``<mesh>`` in this file would mean a geometry file had to ship with the repository, and the
    only Duckiebot meshes in existence are non-redistributable.
    """
    for element in root.iter():
        assert element.tag not in BANNED_GEOMETRY_TAGS, f"forbidden element <{element.tag}>"
        assert "filename" not in element.attrib, f"<{element.tag}> references an external file"
    assert "<mesh" not in urdf_text
    assert "filename" not in urdf_text
    for extension in (".obj", ".stl", ".dae", ".glb", ".gltf", ".mtl", ".fbx", ".usd"):
        assert extension not in urdf_text, f"the URDF references a {extension} asset"


def test_every_geometry_is_an_allowed_primitive(root: ET.Element) -> None:
    """Every ``<geometry>`` holds exactly one box, cylinder or sphere."""
    geometries = list(root.iter("geometry"))
    assert geometries, "the robot has no geometry at all"
    for geometry in geometries:
        shapes = list(geometry)
        assert len(shapes) == 1, f"<geometry> must hold exactly one shape, got {len(shapes)}"
        assert shapes[0].tag in ALLOWED_GEOMETRY_TAGS, f"forbidden shape <{shapes[0].tag}>"


def test_no_third_party_asset_provenance_string(urdf_text: str) -> None:
    """The S3.4 gate greps asset files for the upstream project name; it must not appear here."""
    assert "duckietown" not in urdf_text.lower()


def test_shape_builder_refuses_mesh_geometry() -> None:
    """The generator itself cannot emit a mesh, even if a future edit asks it to."""
    from duckiebot_rl.assets.urdf import _add_shape

    link = ET.Element("link", {"name": "test_link"})
    with pytest.raises(ValueError, match="clean-room"):
        _add_shape(
            link,
            "visual",
            "bad_visual",
            ("mesh", {"filename": "package://duckiebot/meshes/chassis.obj"}),
            (0.0, 0.0, 0.0),
        )


# ---------------------------------------------------------------------------------------------
# Mass and inertia
# ---------------------------------------------------------------------------------------------


def _read_inertia(link: ET.Element) -> tuple[float, float, float, float, float, float]:
    """Read a link's inertia attributes.

    Args:
        link: The ``<link>`` element.

    Returns:
        ``(ixx, ixy, ixz, iyy, iyz, izz)`` in kg.m^2.
    """
    inertial = link.find("inertial")
    assert inertial is not None, f"link {link.attrib['name']!r} has no <inertial>"
    inertia = inertial.find("inertia")
    assert inertia is not None
    return (
        float(inertia.attrib["ixx"]),
        float(inertia.attrib["ixy"]),
        float(inertia.attrib["ixz"]),
        float(inertia.attrib["iyy"]),
        float(inertia.attrib["iyz"]),
        float(inertia.attrib["izz"]),
    )


def _read_mass(link: ET.Element) -> float:
    """Read a link's mass.

    Args:
        link: The ``<link>`` element.

    Returns:
        Mass in kg.
    """
    inertial = link.find("inertial")
    assert inertial is not None
    mass = inertial.find("mass")
    assert mass is not None
    return float(mass.attrib["value"])


def test_total_mass_matches_params(root: ET.Element) -> None:
    """Per-link masses and their sum match the parameter module."""
    links = _links(root)
    assert _read_mass(links[DUCKIEBOT.base_link_name]) == pytest.approx(DUCKIEBOT.base_mass_kg)
    assert _read_mass(links[DUCKIEBOT.left_wheel_link_name]) == pytest.approx(DUCKIEBOT.wheel_mass_kg)
    assert _read_mass(links[DUCKIEBOT.right_wheel_link_name]) == pytest.approx(DUCKIEBOT.wheel_mass_kg)
    total = sum(_read_mass(link) for link in links.values())
    assert total == pytest.approx(DUCKIEBOT.total_mass_kg)
    assert total == pytest.approx(1.10)


@pytest.mark.parametrize(
    "link_name",
    ["base_link", "left_wheel_link", "right_wheel_link"],
)
def test_inertia_tensor_is_physically_valid(root: ET.Element, link_name: str) -> None:
    """Each inertia tensor is diagonal, positive definite and obeys the triangle inequality.

    Diagonality is asserted first because every primitive here is axis aligned, which makes the
    diagonal entries the principal moments. Positive definiteness then follows from Sylvester's
    criterion on the leading principal minors, and the triangle inequality
    (``Ixx + Iyy >= Izz`` and its permutations) is the condition for a real rigid body to exist
    with those principal moments.
    """
    ixx, ixy, ixz, iyy, iyz, izz = _read_inertia(_links(root)[link_name])

    assert (ixy, ixz, iyz) == (0.0, 0.0, 0.0), "axis-aligned primitives must have zero products"

    minor_1 = ixx
    minor_2 = ixx * iyy - ixy * ixy
    minor_3 = ixx * (iyy * izz - iyz * iyz) - ixy * (ixy * izz - iyz * ixz) + ixz * (ixy * iyz - iyy * ixz)
    assert minor_1 > 0.0, f"{link_name}: leading minor 1 is {minor_1}"
    assert minor_2 > 0.0, f"{link_name}: leading minor 2 is {minor_2}"
    assert minor_3 > 0.0, f"{link_name}: determinant is {minor_3}"

    assert ixx + iyy >= izz, f"{link_name}: Ixx + Iyy < Izz"
    assert iyy + izz >= ixx, f"{link_name}: Iyy + Izz < Ixx"
    assert izz + ixx >= iyy, f"{link_name}: Izz + Ixx < Iyy"


def test_base_inertia_matches_the_box_formula(root: ET.Element) -> None:
    """The chassis tensor is the analytic solid-box result, and matches the S3.2 table."""
    ixx, ixy, ixz, iyy, iyz, izz = _read_inertia(_links(root)[DUCKIEBOT.base_link_name])
    expected = box_inertia(DUCKIEBOT.base_mass_kg, DUCKIEBOT.chassis_size_m)
    assert (ixx, ixy, ixz, iyy, iyz, izz) == pytest.approx(expected, rel=REL_TOL)
    # The values SPEC v2 S3.2 prints, to the 4 significant figures it prints them at.
    assert ixx == pytest.approx(1.877e-3, rel=5e-4)
    assert iyy == pytest.approx(3.169e-3, rel=5e-4)
    assert izz == pytest.approx(4.108e-3, rel=5e-4)


def test_wheel_inertia_matches_the_cylinder_formula(root: ET.Element) -> None:
    """The wheel tensor is the analytic solid-cylinder result about the y (axle) axis."""
    for link_name in (DUCKIEBOT.left_wheel_link_name, DUCKIEBOT.right_wheel_link_name):
        ixx, ixy, ixz, iyy, iyz, izz = _read_inertia(_links(root)[link_name])
        expected = cylinder_inertia(
            DUCKIEBOT.wheel_mass_kg, DUCKIEBOT.wheel_radius_m, DUCKIEBOT.wheel_width_m, axis="y"
        )
        assert (ixx, ixy, ixz, iyy, iyz, izz) == pytest.approx(expected, rel=REL_TOL)
        # The axle moment is the largest: that is what identifies y as the spin axis.
        assert iyy > ixx and iyy > izz
        assert iyy == pytest.approx(2.528e-5, rel=5e-4)
        assert ixx == pytest.approx(1.568e-5, rel=5e-4)
        assert ixx == pytest.approx(izz)


def test_inertial_origins_sit_at_the_declared_centres_of_mass(root: ET.Element) -> None:
    """The chassis inertial frame sits at its CoM offset; the wheels' sit at the axle."""
    links = _links(root)
    base_inertial = links[DUCKIEBOT.base_link_name].find("inertial")
    assert base_inertial is not None
    assert _origin_xyz(base_inertial) == pytest.approx(DUCKIEBOT.base_com_base_frame_m, rel=REL_TOL)
    for link_name in (DUCKIEBOT.left_wheel_link_name, DUCKIEBOT.right_wheel_link_name):
        wheel_inertial = links[link_name].find("inertial")
        assert wheel_inertial is not None
        assert _origin_xyz(wheel_inertial) == (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------------------------
# Joints
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("side", ["left", "right"])
def test_joint_limits_and_dynamics_match_params(root: ET.Element, side: str) -> None:
    """Joint limits carry the corrected 0.15 N.m effort and 35 rad/s velocity (critic item F).

    Isaac Lab overwrites both from the actuator config at play, but any raw-USD consumer, the
    MuJoCo twin and every reviewer read these numbers, so they must not be a fiction.
    """
    joint_name = f"{side}_wheel_joint"
    joint = _joints(root)[joint_name]
    assert joint.attrib["type"] == "continuous"

    limit = joint.find("limit")
    assert limit is not None, "a continuous wheel joint still needs effort and velocity limits"
    assert float(limit.attrib["effort"]) == pytest.approx(DUCKIEBOT.wheel_effort_limit_nm)
    assert float(limit.attrib["velocity"]) == pytest.approx(DUCKIEBOT.wheel_velocity_limit_rad_s)
    assert float(limit.attrib["effort"]) == pytest.approx(0.15)
    assert float(limit.attrib["effort"]) < 0.3, "the v1 value of 2.0 N.m must never return"

    dynamics = joint.find("dynamics")
    assert dynamics is not None, "damping and friction must be numeric so MuJoCo sysid can fit them"
    assert float(dynamics.attrib["damping"]) == pytest.approx(DUCKIEBOT.joint_damping)
    assert float(dynamics.attrib["friction"]) == pytest.approx(DUCKIEBOT.joint_friction_nm)
    assert float(dynamics.attrib["damping"]) > 0.0
    assert float(dynamics.attrib["friction"]) > 0.0


def test_joint_origins_and_axis(root: ET.Element) -> None:
    """The wheels straddle the baseline symmetrically and spin about +y."""
    joints = _joints(root)
    left = joints[DUCKIEBOT.left_wheel_joint_name]
    right = joints[DUCKIEBOT.right_wheel_joint_name]
    assert _origin_xyz(left) == pytest.approx(DUCKIEBOT.left_wheel_origin_m, rel=REL_TOL)
    assert _origin_xyz(right) == pytest.approx(DUCKIEBOT.right_wheel_origin_m, rel=REL_TOL)
    separation = _origin_xyz(left)[1] - _origin_xyz(right)[1]
    assert separation == pytest.approx(DUCKIEBOT.wheel_baseline_m, rel=REL_TOL)
    for joint in (left, right):
        axis = joint.find("axis")
        assert axis is not None
        assert [float(v) for v in axis.attrib["xyz"].split()] == [0.0, 1.0, 0.0]


# ---------------------------------------------------------------------------------------------
# The four corrected numbers
# ---------------------------------------------------------------------------------------------


def test_caster_collision_touches_the_ground_exactly(root: ET.Element) -> None:
    """The caster is a sphere of r 0.0165 whose contact point lies on z = 0 (critic item F)."""
    caster = _find_named(_links(root)[DUCKIEBOT.base_link_name], "collision", "caster_collision")
    shape = caster.find("geometry/sphere")
    assert shape is not None, "the caster collider must be a sphere"
    radius = float(shape.attrib["radius"])
    assert radius == pytest.approx(DUCKIEBOT.caster_radius_m)
    assert radius == pytest.approx(0.0165)
    centre = _origin_xyz(caster)
    assert centre == pytest.approx(DUCKIEBOT.caster_center_base_frame_m, rel=REL_TOL)
    contact_height = DUCKIEBOT.base_link_height_m + centre[2] - radius
    assert contact_height == pytest.approx(0.0, abs=1e-9)


def test_chassis_collision_bottom_is_at_21_mm(root: ET.Element) -> None:
    """The chassis box underside sits at the stated 21 mm clearance (critic item F).

    At the v1 centre of ``z = +0.012`` it sat at 6.3 mm, which the D15 tile tilt (3.1 mm over the
    0.18 m box) and D14 pushes would have driven into the ground plane.
    """
    chassis = _find_named(_links(root)[DUCKIEBOT.base_link_name], "collision", "chassis_collision")
    shape = chassis.find("geometry/box")
    assert shape is not None
    size = [float(v) for v in shape.attrib["size"].split()]
    assert size == pytest.approx(DUCKIEBOT.chassis_size_m, rel=REL_TOL)
    centre = _origin_xyz(chassis)
    assert centre == pytest.approx(DUCKIEBOT.chassis_center_base_frame_m, rel=REL_TOL)
    bottom = DUCKIEBOT.base_link_height_m + centre[2] - 0.5 * size[2]
    assert bottom == pytest.approx(DUCKIEBOT.ground_clearance_m, abs=1e-9)
    assert bottom == pytest.approx(0.021, abs=1e-9)
    assert bottom > DUCKIEBOT.caster_radius_m, "the caster must reach the ground before the chassis"


@pytest.mark.parametrize("side", ["left", "right"])
def test_wheel_collider_is_a_sphere_not_a_cylinder(root: ET.Element, side: str) -> None:
    """Wheel colliders are spheres; a cylinder collider costs ~74% of the yaw response."""
    link = _links(root)[f"{side}_wheel_link"]
    collisions = link.findall("collision")
    assert len(collisions) == 1
    assert collisions[0].find("geometry/cylinder") is None
    sphere = collisions[0].find("geometry/sphere")
    assert sphere is not None
    assert float(sphere.attrib["radius"]) == pytest.approx(DUCKIEBOT.wheel_radius_m)
    assert _origin_xyz(collisions[0]) == (0.0, 0.0, 0.0), "the collider sits at the joint origin"

    visual = _find_named(link, "visual", f"{side}_wheel_visual")
    cylinder = visual.find("geometry/cylinder")
    assert cylinder is not None, "the wheel VISUAL is still a cylinder"
    assert float(cylinder.attrib["radius"]) == pytest.approx(DUCKIEBOT.wheel_radius_m)
    assert float(cylinder.attrib["length"]) == pytest.approx(DUCKIEBOT.wheel_width_m)


def test_wheel_visual_cylinder_is_rotated_onto_the_axle(root: ET.Element) -> None:
    """A URDF cylinder runs along its local z, so the visual must be rotated 90 deg about x."""
    for side in ("left", "right"):
        visual = _find_named(_links(root)[f"{side}_wheel_link"], "visual", f"{side}_wheel_visual")
        origin = visual.find("origin")
        assert origin is not None
        roll, pitch, yaw = (float(v) for v in origin.attrib["rpy"].split())
        assert roll == pytest.approx(1.5707963267948966, abs=1e-8)
        assert (pitch, yaw) == (0.0, 0.0)


# ---------------------------------------------------------------------------------------------
# Materials, determinism and the on-disk artifact
# ---------------------------------------------------------------------------------------------


def test_every_visual_material_is_declared(root: ET.Element) -> None:
    """Visual material references resolve to a top-level ``<material>`` with a colour."""
    declared = {m.attrib["name"] for m in root.findall("material")}
    assert declared, "no materials were declared"
    for material in root.findall("material"):
        assert material.find("color") is not None
    for visual in root.iter("visual"):
        reference = visual.find("material")
        assert reference is not None, f"visual {visual.attrib['name']!r} has no material"
        assert reference.attrib["name"] in declared


def test_generation_is_deterministic() -> None:
    """Two calls produce byte-identical documents; the asset is reproducible."""
    assert generate_urdf(DUCKIEBOT) == generate_urdf(DUCKIEBOT)
    first = ET.tostring(build_urdf_tree(DUCKIEBOT), encoding="unicode")
    second = ET.tostring(build_urdf_tree(DUCKIEBOT), encoding="unicode")
    assert first == second


def test_write_urdf_round_trips(tmp_path: Path) -> None:
    """``write_urdf`` creates parents, writes LF newlines and reproduces the document exactly."""
    destination = tmp_path / "nested" / "dir" / URDF_FILENAME
    written = write_urdf(destination, DUCKIEBOT)
    assert written.exists()
    raw = written.read_bytes()
    assert b"\r\n" not in raw, "the URDF must use LF endings so hashes match across platforms"
    assert raw.decode("utf-8") == generate_urdf(DUCKIEBOT)
    ET.fromstring(raw.decode("utf-8"))


def test_committed_urdf_is_not_stale() -> None:
    """If ``assets/duckiebot/duckiebot.urdf`` exists, it matches the current parameters."""
    committed = _REPO_ROOT / "assets" / "duckiebot" / URDF_FILENAME
    if not committed.exists():
        pytest.skip("run scripts/build_robot_asset.py to generate the URDF")
    assert committed.read_text(encoding="utf-8") == generate_urdf(DUCKIEBOT), (
        "the committed URDF is out of date; run python scripts/build_robot_asset.py"
    )


# ---------------------------------------------------------------------------------------------
# The inertia helpers themselves
# ---------------------------------------------------------------------------------------------


def test_box_inertia_against_a_hand_computed_case() -> None:
    """A unit cube of unit mass has ``I = 1/6`` on every axis."""
    ixx, ixy, ixz, iyy, iyz, izz = box_inertia(1.0, (1.0, 1.0, 1.0))
    assert (ixx, iyy, izz) == pytest.approx((1.0 / 6.0,) * 3)
    assert (ixy, ixz, iyz) == (0.0, 0.0, 0.0)
    # A thin plate: the out-of-plane moment equals the sum of the in-plane moments exactly.
    pixx, _, _, piyy, _, pizz = box_inertia(2.0, (0.4, 0.3, 1e-9))
    assert pizz == pytest.approx(pixx + piyy, rel=1e-6)


def test_cylinder_inertia_axis_selection() -> None:
    """The axial moment lands on whichever axis is named, and equals ``m r^2 / 2``."""
    mass, radius, length = 3.0, 0.2, 0.5
    axial = 0.5 * mass * radius * radius
    transverse = mass * (3.0 * radius * radius + length * length) / 12.0
    for axis, index in (("x", 0), ("y", 3), ("z", 5)):
        tensor = cylinder_inertia(mass, radius, length, axis=axis)
        assert tensor[index] == pytest.approx(axial)
        others = [tensor[i] for i in (0, 3, 5) if i != index]
        assert others == pytest.approx([transverse, transverse])
    with pytest.raises(ValueError, match="axis"):
        cylinder_inertia(1.0, 1.0, 1.0, axis="w")


def test_sphere_inertia_is_isotropic() -> None:
    """A solid sphere has ``I = 2 m r^2 / 5`` on every axis."""
    ixx, ixy, ixz, iyy, iyz, izz = sphere_inertia(2.0, 0.5)
    assert (ixx, iyy, izz) == pytest.approx((0.2,) * 3)
    assert (ixy, ixz, iyz) == (0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    ("func", "args"),
    [
        (box_inertia, (0.0, (1.0, 1.0, 1.0))),
        (box_inertia, (1.0, (1.0, -1.0, 1.0))),
        (cylinder_inertia, (-1.0, 1.0, 1.0)),
        (cylinder_inertia, (1.0, 0.0, 1.0)),
        (sphere_inertia, (1.0, 0.0)),
        (sphere_inertia, (-1.0, 1.0)),
    ],
)
def test_inertia_helpers_reject_degenerate_bodies(func: object, args: tuple[object, ...]) -> None:
    """Zero or negative mass and extents raise instead of silently producing a bad tensor."""
    with pytest.raises(ValueError):
        func(*args)  # type: ignore[operator]
