"""Tests for the [assets] USD toolchain: ``tools/verify_usd.py`` and ``tools/patch_usd.py``.

Three layers, in increasing order of what they need installed:

1. **Pure assertion logic** (the bulk of this file). ``check_scene`` and ``plan_patch`` operate on
   :class:`verify_usd.RobotScene`, a plain dataclass tree, so every M1 acceptance criterion is
   tested here on a CPU runner with no Isaac Sim, no Kit and no USD. Each criterion gets a
   conforming scene that must pass and at least one mutation that must fail it, and the
   mutations are the real failure modes (a caster that floats a millimetre, a wheel that came in
   as a cylinder, inertia whose spin axis got permuted), not sentinel garbage.
2. **Tolerance boundaries.** A tolerance nobody tests is a tolerance that silently widens, so the
   tests pin both sides of each one: an error just inside it passes, an error just outside fails.
3. **Real USD round trip**, skipped automatically when ``pxr`` cannot be imported. It authors a
   synthetic robot stage with the same shape as the Isaac Sim URDF importer's output, including
   the ``instanceable`` collision scopes that hide colliders from a naive traversal, then runs
   the extractor, the patch and the verification over it for real.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "tools")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import import_urdf_headless  # noqa: E402
import patch_usd  # noqa: E402
import verify_usd  # noqa: E402
from patch_usd import PatchPlanError, apply_patch, plan_patch  # noqa: E402
from verify_usd import (  # noqa: E402
    Collider,
    Joint,
    PhysicsMaterial,
    RigidBody,
    RobotScene,
    StageOpenError,
    check_scene,
    extract_scene,
    format_report,
)

from duckiebot_rl.assets.params import DUCKIEBOT  # noqa: E402
from duckiebot_rl.assets.robot_cfg import DEFAULT_USD_PATH, physics_material_spec  # noqa: E402

WHEEL_MATERIAL_PATH = "/duckiebot/PhysicsMaterials/duckiebot_wheel_material"
CASTER_MATERIAL_PATH = "/duckiebot/PhysicsMaterials/duckiebot_caster_material"


def _pxr_is_available() -> bool:
    """Whether a USD runtime can be made importable in this interpreter.

    Returns:
        True when :func:`verify_usd.ensure_pxr` succeeds.
    """
    try:
        verify_usd.ensure_pxr()
    except StageOpenError:
        return False
    return True


_HAS_PXR = _pxr_is_available()
needs_usd = pytest.mark.skipif(
    not _HAS_PXR, reason="no USD runtime: run with the Isaac Sim python or pip install usd-core"
)


# =============================================================================================
# Synthetic scenes
# =============================================================================================


def conforming_scene() -> RobotScene:
    """Build the scene an imported and patched Duckiebot USD must produce.

    Every number comes from :data:`duckiebot_rl.assets.params.DUCKIEBOT`, so this is the
    acceptance target restated in the extractor's vocabulary rather than a second set of
    constants that could drift from the first.

    Returns:
        A scene on which every check passes.
    """
    p = DUCKIEBOT
    spec = physics_material_spec(p)
    wheel_spec = spec["duckiebot_wheel_material"]
    caster_spec = spec["duckiebot_caster_material"]
    return RobotScene(
        source="synthetic conforming scene",
        root_prim_path="/duckiebot",
        meters_per_unit=1.0,
        up_axis="Z",
        bodies=[
            RigidBody(
                prim_path="/duckiebot/base_link",
                name=p.base_link_name,
                mass_kg=p.base_mass_kg,
                com_root_m=p.base_com_base_frame_m,
                diagonal_inertia_kg_m2=p.base_inertia_about_com,
                principal_axes_wxyz=(1.0, 0.0, 0.0, 0.0),
                translate_root_m=(0.0, 0.0, 0.0),
            ),
            RigidBody(
                prim_path="/duckiebot/left_wheel_link",
                name=p.left_wheel_link_name,
                mass_kg=p.wheel_mass_kg,
                com_root_m=(0.0, 0.0, 0.0),
                diagonal_inertia_kg_m2=p.wheel_inertia_about_com,
                principal_axes_wxyz=(1.0, 0.0, 0.0, 0.0),
                translate_root_m=p.left_wheel_origin_m,
            ),
            RigidBody(
                prim_path="/duckiebot/right_wheel_link",
                name=p.right_wheel_link_name,
                mass_kg=p.wheel_mass_kg,
                com_root_m=(0.0, 0.0, 0.0),
                diagonal_inertia_kg_m2=p.wheel_inertia_about_com,
                principal_axes_wxyz=(1.0, 0.0, 0.0, 0.0),
                translate_root_m=p.right_wheel_origin_m,
            ),
        ],
        joints=[
            Joint(
                prim_path="/duckiebot/joints/left_wheel_joint",
                name=p.left_wheel_joint_name,
                kind="revolute",
                axis="Y",
                body0="/duckiebot/base_link",
                body1="/duckiebot/left_wheel_link",
                local_pos0_m=p.left_wheel_origin_m,
                local_pos1_m=(0.0, 0.0, 0.0),
            ),
            Joint(
                prim_path="/duckiebot/joints/right_wheel_joint",
                name=p.right_wheel_joint_name,
                kind="revolute",
                axis="Y",
                body0="/duckiebot/base_link",
                body1="/duckiebot/right_wheel_link",
                local_pos0_m=p.right_wheel_origin_m,
                local_pos1_m=(0.0, 0.0, 0.0),
            ),
        ],
        colliders=[
            Collider(
                prim_path="/duckiebot/base_link/collisions/chassis_collision/box",
                body=p.base_link_name,
                kind="box",
                center_root_m=p.chassis_center_base_frame_m,
                half_extents_m=tuple(0.5 * s for s in p.chassis_size_m),
            ),
            Collider(
                prim_path="/duckiebot/base_link/collisions/caster_collision/sphere",
                body=p.base_link_name,
                kind="sphere",
                center_root_m=p.caster_center_base_frame_m,
                radius_m=p.caster_radius_m,
                material_path=CASTER_MATERIAL_PATH,
            ),
            Collider(
                prim_path="/duckiebot/left_wheel_link/collisions/left_wheel_collision/sphere",
                body=p.left_wheel_link_name,
                kind="sphere",
                center_root_m=p.left_wheel_origin_m,
                radius_m=p.wheel_radius_m,
                material_path=WHEEL_MATERIAL_PATH,
            ),
            Collider(
                prim_path="/duckiebot/right_wheel_link/collisions/right_wheel_collision/sphere",
                body=p.right_wheel_link_name,
                kind="sphere",
                center_root_m=p.right_wheel_origin_m,
                radius_m=p.wheel_radius_m,
                material_path=WHEEL_MATERIAL_PATH,
            ),
        ],
        materials={
            WHEEL_MATERIAL_PATH: PhysicsMaterial(
                prim_path=WHEEL_MATERIAL_PATH,
                static_friction=wheel_spec["static_friction"],
                dynamic_friction=wheel_spec["dynamic_friction"],
                restitution=0.0,
                friction_combine_mode=wheel_spec["friction_combine_mode"],
                restitution_combine_mode=wheel_spec["restitution_combine_mode"],
                improve_patch_friction=wheel_spec["improve_patch_friction"],
            ),
            CASTER_MATERIAL_PATH: PhysicsMaterial(
                prim_path=CASTER_MATERIAL_PATH,
                static_friction=caster_spec["static_friction"],
                dynamic_friction=caster_spec["dynamic_friction"],
                restitution=0.0,
                friction_combine_mode=caster_spec["friction_combine_mode"],
                restitution_combine_mode=caster_spec["restitution_combine_mode"],
                improve_patch_friction=caster_spec["improve_patch_friction"],
            ),
        },
        articulation_root_paths=["/duckiebot/base_link"],
        mesh_prim_paths=[],
    )


def _replace(scene: RobotScene, **changes: Any) -> RobotScene:
    """Return a copy of a scene with some top-level fields replaced.

    Args:
        scene: The scene to copy.
        **changes: Fields to override.

    Returns:
        The modified copy.
    """
    data = {
        "source": scene.source,
        "root_prim_path": scene.root_prim_path,
        "meters_per_unit": scene.meters_per_unit,
        "up_axis": scene.up_axis,
        "bodies": list(scene.bodies),
        "joints": list(scene.joints),
        "colliders": list(scene.colliders),
        "materials": dict(scene.materials),
        "articulation_root_paths": list(scene.articulation_root_paths),
        "mesh_prim_paths": list(scene.mesh_prim_paths),
    }
    data.update(changes)
    return RobotScene(**data)


def _with_collider(scene: RobotScene, index: int, **changes: Any) -> RobotScene:
    """Return a copy of a scene with one collider's fields replaced.

    Args:
        scene: The scene to copy.
        index: Index into ``scene.colliders``.
        **changes: Collider fields to override.

    Returns:
        The modified copy.
    """
    colliders = list(scene.colliders)
    current = colliders[index]
    data = {
        "prim_path": current.prim_path,
        "body": current.body,
        "kind": current.kind,
        "center_root_m": current.center_root_m,
        "radius_m": current.radius_m,
        "half_extents_m": current.half_extents_m,
        "height_m": current.height_m,
        "axis": current.axis,
        "material_path": current.material_path,
        "enabled": current.enabled,
    }
    data.update(changes)
    colliders[index] = Collider(**data)
    return _replace(scene, colliders=colliders)


def _with_body(scene: RobotScene, index: int, **changes: Any) -> RobotScene:
    """Return a copy of a scene with one rigid body's fields replaced.

    Args:
        scene: The scene to copy.
        index: Index into ``scene.bodies``.
        **changes: Body fields to override.

    Returns:
        The modified copy.
    """
    bodies = list(scene.bodies)
    current = bodies[index]
    data = {
        "prim_path": current.prim_path,
        "name": current.name,
        "mass_kg": current.mass_kg,
        "com_root_m": current.com_root_m,
        "diagonal_inertia_kg_m2": current.diagonal_inertia_kg_m2,
        "principal_axes_wxyz": current.principal_axes_wxyz,
        "translate_root_m": current.translate_root_m,
    }
    data.update(changes)
    bodies[index] = RigidBody(**data)
    return _replace(scene, bodies=bodies)


def _result(results: list[verify_usd.CheckResult], check_id: str) -> verify_usd.CheckResult:
    """Look one check up by id.

    Args:
        results: Output of :func:`check_scene`.
        check_id: The identifier to find.

    Returns:
        The matching result.
    """
    matches = [r for r in results if r.check_id == check_id]
    assert matches, f"no check named {check_id} in {[r.check_id for r in results]}"
    return matches[0]


def _failed_ids(scene: RobotScene) -> set[str]:
    """Run the checks and return the ids that failed.

    Args:
        scene: The scene to check.

    Returns:
        Set of failing check identifiers.
    """
    return {r.check_id for r in check_scene(scene) if not r.ok}


# Index constants for readability in the mutation tests.
CHASSIS, CASTER, LEFT_WHEEL, RIGHT_WHEEL = 0, 1, 2, 3
BASE_BODY, LEFT_BODY, RIGHT_BODY = 0, 1, 2


# =============================================================================================
# The conforming scene, and the shape of the check list
# =============================================================================================


def test_the_conforming_scene_passes_every_check() -> None:
    """The scene built from params is exactly what M1 accepts."""
    results = check_scene(conforming_scene())
    failures = [(r.check_id, r.detail) for r in results if not r.ok]
    assert failures == []


def test_every_m1_acceptance_quantity_has_its_own_check() -> None:
    """The four quantities S11 M1 names are individually identifiable in the report."""
    ids = {r.check_id for r in check_scene(conforming_scene())}
    assert {"M1.bodies", "M1.dof", "M1.masses", "M1.caster", "M1.chassis"} <= ids


def test_check_scene_is_pure() -> None:
    """Checking a scene must not mutate it: the tools run it twice, before and after patching."""
    scene = conforming_scene()
    snapshot = scene.to_dict()
    check_scene(scene)
    assert scene.to_dict() == snapshot


def test_scene_round_trips_through_json_form() -> None:
    """``to_dict``/``from_dict`` is what ``--json`` writes and what the tests feed back in."""
    scene = conforming_scene()
    restored = RobotScene.from_dict(scene.to_dict())
    assert restored == scene
    assert [r.ok for r in check_scene(restored)] == [r.ok for r in check_scene(scene)]


# =============================================================================================
# M1.bodies and M1.dof
# =============================================================================================


def test_a_missing_wheel_body_fails_the_body_count() -> None:
    """Two bodies is not three, however plausible the rest of the asset looks."""
    scene = conforming_scene()
    scene = _replace(scene, bodies=scene.bodies[:2])
    assert "M1.bodies" in _failed_ids(scene)


def test_an_extra_body_fails_the_body_count() -> None:
    """A caster that imported as its own link instead of merging is four bodies, not three."""
    scene = conforming_scene()
    extra = RigidBody(
        prim_path="/duckiebot/caster_link",
        name="caster_link",
        mass_kg=0.0,
        com_root_m=(0.0, 0.0, 0.0),
        diagonal_inertia_kg_m2=(0.0, 0.0, 0.0),
        principal_axes_wxyz=(1.0, 0.0, 0.0, 0.0),
        translate_root_m=DUCKIEBOT.caster_center_base_frame_m,
    )
    assert "M1.bodies" in _failed_ids(_replace(scene, bodies=[*scene.bodies, extra]))


def test_a_renamed_link_fails_the_body_check() -> None:
    """The link names are the contract the actuator regex and the DR code index by."""
    scene = _with_body(conforming_scene(), LEFT_BODY, name="wheel_left")
    assert "M1.bodies" in _failed_ids(scene)


def test_a_third_dof_fails_the_dof_count() -> None:
    """A caster that imported as a joint would add a DOF the action path does not drive."""
    scene = conforming_scene()
    extra = Joint(
        prim_path="/duckiebot/joints/caster_joint",
        name="caster_joint",
        kind="revolute",
        axis="Z",
        body0="/duckiebot/base_link",
        body1="/duckiebot/caster_link",
        local_pos0_m=DUCKIEBOT.caster_center_base_frame_m,
        local_pos1_m=(0.0, 0.0, 0.0),
    )
    assert "M1.dof" in _failed_ids(_replace(scene, joints=[*scene.joints, extra]))


def test_a_disabled_joint_is_not_counted_as_a_dof() -> None:
    """``physics:jointEnabled = false`` removes the DOF, so the count must notice."""
    scene = conforming_scene()
    joints = list(scene.joints)
    joints[0] = Joint(
        prim_path=joints[0].prim_path,
        name=joints[0].name,
        kind=joints[0].kind,
        axis=joints[0].axis,
        body0=joints[0].body0,
        body1=joints[0].body1,
        local_pos0_m=joints[0].local_pos0_m,
        local_pos1_m=joints[0].local_pos1_m,
        enabled=False,
    )
    assert "M1.dof" in _failed_ids(_replace(scene, joints=joints))


def test_a_wheel_joint_about_the_wrong_axis_fails() -> None:
    """Wheels spin about the body y axis; anything else is a differential drive that will not."""
    scene = conforming_scene()
    joints = list(scene.joints)
    joints[0] = Joint(
        prim_path=joints[0].prim_path,
        name=joints[0].name,
        kind="revolute",
        axis="X",
        body0=joints[0].body0,
        body1=joints[0].body1,
        local_pos0_m=joints[0].local_pos0_m,
        local_pos1_m=joints[0].local_pos1_m,
    )
    assert "M1.dof" in _failed_ids(_replace(scene, joints=joints))


def test_a_fixed_joint_to_the_world_fails_the_dof_check() -> None:
    """``set_fix_base(True)`` pins the robot to the stage; the drive test would report nothing."""
    scene = conforming_scene()
    anchor = Joint(
        prim_path="/duckiebot/joints/root_joint",
        name="root_joint",
        kind="fixed",
        axis=None,
        body0=None,
        body1="/duckiebot/base_link",
        local_pos0_m=(0.0, 0.0, 0.0),
        local_pos1_m=(0.0, 0.0, 0.0),
    )
    assert "M1.dof" in _failed_ids(_replace(scene, joints=[*scene.joints, anchor]))


# =============================================================================================
# M1.masses and the inertia check
# =============================================================================================


def test_a_wrong_base_mass_fails() -> None:
    """1.2 kg is a plausible-looking chassis mass and still the wrong one."""
    scene = _with_body(conforming_scene(), BASE_BODY, mass_kg=1.2)
    assert "M1.masses" in _failed_ids(scene)


def test_a_wrong_wheel_mass_fails() -> None:
    """A wheel that came in at the default density instead of the URDF mass."""
    scene = _with_body(conforming_scene(), LEFT_BODY, mass_kg=0.08)
    assert "M1.masses" in _failed_ids(scene)


def test_the_mass_tolerance_accepts_float32_rounding_but_not_a_real_error() -> None:
    """0.05 kg arrives from USD as 0.05000000074505806; 0.0501 kg is a different wheel."""
    float32_wheel = 0.05000000074505806
    scene = _with_body(conforming_scene(), LEFT_BODY, mass_kg=float32_wheel)
    assert "M1.masses" not in _failed_ids(scene)
    just_outside = DUCKIEBOT.wheel_mass_kg + 10.0 * verify_usd.MASS_TOL_KG
    assert "M1.masses" in _failed_ids(_with_body(conforming_scene(), LEFT_BODY, mass_kg=just_outside))


def test_inertia_computed_from_a_density_instead_of_the_urdf_fails() -> None:
    """``set_import_inertia_tensor(False)`` would size the chassis inertia from the box volume."""
    box_volume_inertia = tuple(4.0 * v for v in DUCKIEBOT.base_inertia_about_com)
    scene = _with_body(conforming_scene(), BASE_BODY, diagonal_inertia_kg_m2=box_volume_inertia)
    assert "S3.2.inertia" in _failed_ids(scene)


def test_a_permuted_wheel_inertia_fails_because_the_spin_axis_moved() -> None:
    """Swapping Iyy and Izz makes the wheel spin about the wrong axis, at 1.6x the inertia."""
    ixx, iyy, izz = DUCKIEBOT.wheel_inertia_about_com
    scene = _with_body(conforming_scene(), LEFT_BODY, diagonal_inertia_kg_m2=(ixx, izz, iyy))
    assert "S3.2.inertia" in _failed_ids(scene)


def test_a_permuted_inertia_with_a_compensating_rotation_is_the_same_body() -> None:
    """USD may store any diagonal plus a rotation into principal axes: the tensor is the check."""
    ixx, iyy, izz = DUCKIEBOT.wheel_inertia_about_com
    quarter = math.sqrt(0.5)
    # +90 deg about x maps the stored y axis onto z and z onto -y, so the permuted diagonal
    # (Ixx, Izz, Iyy) with this rotation reconstructs exactly the original tensor.
    scene = _with_body(
        conforming_scene(),
        LEFT_BODY,
        diagonal_inertia_kg_m2=(ixx, izz, iyy),
        principal_axes_wxyz=(quarter, quarter, 0.0, 0.0),
    )
    assert "S3.2.inertia" not in _failed_ids(scene)


def test_a_shifted_centre_of_mass_fails() -> None:
    """The CoM offset is a DR nominal that the whole dynamics story is built on."""
    scene = _with_body(conforming_scene(), BASE_BODY, com_root_m=(0.0, 0.0, 0.0))
    assert "S3.2.inertia" in _failed_ids(scene)


# =============================================================================================
# M1.caster
# =============================================================================================


def test_the_caster_radius_is_checked() -> None:
    """0.021 m was the v1 prose value; the asset must carry 0.0165 m."""
    scene = _with_collider(conforming_scene(), CASTER, radius_m=0.021)
    assert "M1.caster" in _failed_ids(scene)


def test_a_caster_that_floats_by_one_millimetre_fails() -> None:
    """A floating caster turns a stable tripod into a robot that rocks on two wheels."""
    x, y, z = DUCKIEBOT.caster_center_base_frame_m
    scene = _with_collider(conforming_scene(), CASTER, center_root_m=(x, y, z + 0.001))
    result = _result(check_scene(scene), "M1.caster")
    assert not result.ok
    assert "floats" in result.detail


def test_a_caster_that_penetrates_the_ground_fails_with_that_wording() -> None:
    """The message has to distinguish the two directions: they have different causes."""
    x, y, z = DUCKIEBOT.caster_center_base_frame_m
    scene = _with_collider(conforming_scene(), CASTER, center_root_m=(x, y, z - 0.002))
    result = _result(check_scene(scene), "M1.caster")
    assert not result.ok
    assert "penetrates the ground" in result.detail


def test_the_caster_height_tolerance_is_one_micrometre() -> None:
    """Half a micrometre of float noise passes; two micrometres of modelling error does not."""
    x, y, z = DUCKIEBOT.caster_center_base_frame_m
    inside = _with_collider(conforming_scene(), CASTER, center_root_m=(x, y, z + 5.0e-7))
    outside = _with_collider(conforming_scene(), CASTER, center_root_m=(x, y, z + 2.0e-6))
    assert "M1.caster" not in _failed_ids(inside)
    assert "M1.caster" in _failed_ids(outside)


def test_a_disabled_caster_collider_fails() -> None:
    """``physics:collisionEnabled = false`` is a caster that is not there at all."""
    scene = _with_collider(conforming_scene(), CASTER, enabled=False)
    assert "M1.caster" in _failed_ids(scene)


def test_two_spheres_on_the_base_link_are_ambiguous_and_fail() -> None:
    """If the base carried two spheres, no rule could say which one is the caster."""
    scene = conforming_scene()
    duplicate = Collider(
        prim_path="/duckiebot/base_link/collisions/marker_collision/sphere",
        body=DUCKIEBOT.base_link_name,
        kind="sphere",
        center_root_m=DUCKIEBOT.duckie_marker_center_base_frame_m,
        radius_m=DUCKIEBOT.duckie_marker_radius_m,
    )
    assert "M1.caster" in _failed_ids(_replace(scene, colliders=[*scene.colliders, duplicate]))


# =============================================================================================
# M1.chassis
# =============================================================================================


def test_the_v1_chassis_height_error_fails_the_clearance_check() -> None:
    """v1 put the box centre at +0.012, giving 6.3 mm of clearance instead of 21 mm."""
    x, y, _ = DUCKIEBOT.chassis_center_base_frame_m
    scene = _with_collider(conforming_scene(), CHASSIS, center_root_m=(x, y, 0.012))
    result = _result(check_scene(scene), "M1.chassis")
    assert not result.ok
    assert "0.021" in result.detail


def test_a_wrong_chassis_size_fails() -> None:
    """The box extents feed the clearance arithmetic and the DR bounds alike."""
    scene = _with_collider(conforming_scene(), CHASSIS, half_extents_m=(0.09, 0.065, 0.05))
    assert "M1.chassis" in _failed_ids(scene)


def test_a_chassis_that_is_a_sphere_fails_the_chassis_check() -> None:
    """A base link with no box collider at all has no underside to measure."""
    scene = _with_collider(conforming_scene(), CHASSIS, kind="sphere", radius_m=0.09)
    assert "M1.chassis" in _failed_ids(scene)


# =============================================================================================
# S3.2 wheels, collider census, meshes, stage, articulation root
# =============================================================================================


def test_a_cylindrical_wheel_collider_fails_two_checks() -> None:
    """The exact regression S3.2 exists to prevent: -74% yaw response, silently."""
    scene = _with_collider(conforming_scene(), LEFT_WHEEL, kind="cylinder", height_m=0.027, axis="Y")
    failed = _failed_ids(scene)
    assert "S3.2.wheels" in failed
    assert "S3.2.colliders" in failed


def test_a_wheel_sphere_of_the_wrong_radius_fails() -> None:
    """Wheel radius is the scale factor of the entire drive model."""
    scene = _with_collider(conforming_scene(), RIGHT_WHEEL, radius_m=0.035)
    assert "S3.2.wheels" in _failed_ids(scene)


def test_a_wheel_collider_off_its_axle_fails() -> None:
    """A wheel whose collider is not on the joint origin is a wheel that scuffs."""
    x, y, z = DUCKIEBOT.left_wheel_origin_m
    scene = _with_collider(conforming_scene(), LEFT_WHEEL, center_root_m=(x, y + 0.005, z))
    assert "S3.2.wheels" in _failed_ids(scene)


def test_an_extra_collider_fails_the_census() -> None:
    """``set_collision_from_visuals(True)`` would turn the camera block into a collider."""
    scene = conforming_scene()
    extra = Collider(
        prim_path="/duckiebot/base_link/collisions/camera_housing_visual/box",
        body=DUCKIEBOT.base_link_name,
        kind="box",
        center_root_m=DUCKIEBOT.camera_pos_base_frame_m,
        half_extents_m=(0.015, 0.015, 0.015),
    )
    assert "S3.2.colliders" in _failed_ids(_replace(scene, colliders=[*scene.colliders, extra]))


def test_a_mesh_prim_fails_the_clean_room_check() -> None:
    """Clean-room rule 3: the robot may contain no Mesh prim at all, visual or collision."""
    scene = _replace(conforming_scene(), mesh_prim_paths=["/duckiebot/base_link/visuals/mesh_0"])
    assert "S3.4.no_mesh" in _failed_ids(scene)


def test_centimetre_stage_units_fail() -> None:
    """A stage authored in centimetres would scale every length in this file by 100."""
    scene = _replace(conforming_scene(), meters_per_unit=0.01)
    assert "S3.2.stage" in _failed_ids(scene)


def test_y_up_fails() -> None:
    """REP-103 is z up; a y-up import silently lies the robot on its side."""
    assert "S3.2.stage" in _failed_ids(_replace(conforming_scene(), up_axis="Y"))


def test_a_missing_default_prim_fails() -> None:
    """``UsdFileCfg`` references the default prim; without one the spawn fails at training time."""
    assert "S3.2.stage" in _failed_ids(_replace(conforming_scene(), root_prim_path=""))


@pytest.mark.parametrize("roots", [[], ["/duckiebot/base_link", "/duckiebot/left_wheel_link"]])
def test_the_articulation_root_must_be_unique(roots: list[str]) -> None:
    """Zero roots is not an articulation; two is an ambiguity PhysX resolves by guessing.

    Args:
        roots: The articulation root paths to place on the scene.
    """
    scene = _replace(conforming_scene(), articulation_root_paths=roots)
    assert "S5.1.articulation_root" in _failed_ids(scene)


# =============================================================================================
# S3.2 materials
# =============================================================================================


def test_an_unbound_caster_fails_the_material_check() -> None:
    """This is the failure the patch step exists to prevent, and it is invisible in simulation."""
    scene = _with_collider(conforming_scene(), CASTER, material_path=None)
    result = _result(check_scene(scene), "S3.2.materials")
    assert not result.ok
    assert "caster" in result.detail


def test_a_caster_bound_to_the_wheel_material_fails() -> None:
    """A mis-resolved selector that binds the wrong material is worse than binding none."""
    scene = _with_collider(conforming_scene(), CASTER, material_path=WHEEL_MATERIAL_PATH)
    assert "S3.2.materials" in _failed_ids(scene)


def test_a_dangling_material_binding_fails() -> None:
    """A binding relationship whose target prim does not exist resolves to no material."""
    scene = _with_collider(conforming_scene(), CASTER, material_path="/duckiebot/Looks/nope")
    assert "S3.2.materials" in _failed_ids(scene)


def test_the_caster_friction_must_be_exactly_zero() -> None:
    """PhysX's 0.5 default is what an unbound caster gets, and it fights every turn."""
    scene = conforming_scene()
    materials = dict(scene.materials)
    materials[CASTER_MATERIAL_PATH] = PhysicsMaterial(
        prim_path=CASTER_MATERIAL_PATH,
        static_friction=0.5,
        dynamic_friction=0.5,
        friction_combine_mode="min",
    )
    assert "S3.2.materials" in _failed_ids(_replace(scene, materials=materials))


def test_the_caster_combine_mode_must_be_min() -> None:
    """With ``average`` the frictionless caster becomes mu 0.25 against a mu 0.5 ground."""
    scene = conforming_scene()
    materials = dict(scene.materials)
    materials[CASTER_MATERIAL_PATH] = PhysicsMaterial(
        prim_path=CASTER_MATERIAL_PATH,
        static_friction=0.0,
        dynamic_friction=0.0,
        friction_combine_mode="average",
    )
    assert "S3.2.materials" in _failed_ids(_replace(scene, materials=materials))


def test_the_wheel_material_must_keep_improve_patch_friction_on() -> None:
    """Without it a rolling sphere's contact patch flickers and the traction model is noise."""
    scene = conforming_scene()
    materials = dict(scene.materials)
    wheel = materials[WHEEL_MATERIAL_PATH]
    materials[WHEEL_MATERIAL_PATH] = PhysicsMaterial(
        prim_path=wheel.prim_path,
        static_friction=wheel.static_friction,
        dynamic_friction=wheel.dynamic_friction,
        friction_combine_mode=wheel.friction_combine_mode,
        improve_patch_friction=False,
    )
    assert "S3.2.materials" in _failed_ids(_replace(scene, materials=materials))


def test_the_material_check_reads_the_shared_spec_not_its_own_numbers() -> None:
    """verify_usd must agree with robot_cfg by construction, not by a copied literal."""
    spec = physics_material_spec()
    assert spec["duckiebot_wheel_material"]["static_friction"] == DUCKIEBOT.wheel_friction_static
    assert spec["duckiebot_caster_material"]["static_friction"] == DUCKIEBOT.caster_friction
    assert spec["duckiebot_caster_material"]["friction_combine_mode"] == "min"


# =============================================================================================
# Reporting and the command-line contract
# =============================================================================================


def test_the_report_names_every_failing_check() -> None:
    """The report is what a human reads when M1 fails; it must not hide anything."""
    scene = _with_collider(conforming_scene(), CASTER, radius_m=0.021)
    results = check_scene(scene)
    report = format_report(scene, results)
    assert "FAILED" in report
    assert "M1.caster" in report
    assert "0.0165" in report


def test_the_report_of_a_clean_asset_says_so() -> None:
    """A passing run has to be unmistakable in a build log."""
    scene = conforming_scene()
    report = format_report(scene, check_scene(scene))
    assert "PASSED all" in report
    assert "FAIL" not in report


def test_main_exits_with_2_when_the_asset_is_missing() -> None:
    """Exit code 2 means "could not look", which is not the same as "looked and it is wrong"."""
    assert verify_usd.main(["definitely/not/a/real/asset.usd"]) == 2


# =============================================================================================
# patch_usd planning: pure, no USD
# =============================================================================================


def test_the_plan_binds_both_wheels_and_the_caster_and_nothing_else() -> None:
    """Three bindings, and the chassis box is reported as deliberately unbound."""
    plan = plan_patch(conforming_scene())
    assert plan.replacements == []
    assert len(plan.bindings) == 3
    by_material = {b.material_name for b in plan.bindings}
    assert by_material == {"duckiebot_wheel_material", "duckiebot_caster_material"}
    assert plan.unbound == ["/duckiebot/base_link/collisions/chassis_collision/box"]


def test_the_plan_selects_the_caster_by_geometry_not_by_prim_name() -> None:
    """The selector must survive an importer that names collider prims differently."""
    scene = _with_collider(conforming_scene(), CASTER, prim_path="/duckiebot/base_link/collisions/mesh_0")
    caster_bindings = [
        b for b in plan_patch(scene).bindings if b.material_name == "duckiebot_caster_material"
    ]
    assert [b.prim_path for b in caster_bindings] == ["/duckiebot/base_link/collisions/mesh_0"]


def test_the_plan_replaces_a_cylinder_wheel_collider_and_still_binds_it() -> None:
    """The swap is planned first and the binding is resolved against the post-swap geometry."""
    scene = _with_collider(
        conforming_scene(),
        LEFT_WHEEL,
        kind="cylinder",
        radius_m=DUCKIEBOT.wheel_radius_m,
        height_m=DUCKIEBOT.wheel_width_m,
        axis="Y",
        material_path=None,
    )
    plan = plan_patch(scene)
    assert [r.from_kind for r in plan.replacements] == ["cylinder"]
    assert plan.replacements[0].radius_m == DUCKIEBOT.wheel_radius_m
    assert len(plan.bindings) == 3


def test_a_capsule_wheel_collider_is_also_replaced() -> None:
    """``set_replace_cylinders_with_capsules(True)`` is one flag away from being on."""
    scene = _with_collider(conforming_scene(), LEFT_WHEEL, kind="capsule", height_m=0.027, axis="Y")
    assert [r.from_kind for r in plan_patch(scene).replacements] == ["capsule"]


def test_a_round_collider_off_the_wheels_raises_rather_than_being_replaced() -> None:
    """S3.2 sanctions replacing a wheel collider, nothing else; guessing here would be wrong."""
    scene = _with_collider(conforming_scene(), CHASSIS, kind="cylinder", radius_m=0.09)
    with pytest.raises(PatchPlanError, match="only sanctions replacing"):
        plan_patch(scene)


def test_a_selector_that_matches_nothing_raises_instead_of_doing_nothing() -> None:
    """The whole point: a silent no-op leaves the caster on PhysX's 0.5/0.5 default forever."""
    scene = conforming_scene()
    scene = _replace(scene, colliders=[c for c in scene.colliders if c.kind != "sphere"])
    with pytest.raises(PatchPlanError, match="selects 0 colliders"):
        plan_patch(scene)


def test_a_selector_that_matches_too_many_raises() -> None:
    """Two caster-sized spheres on the base link is an asset nobody should patch blindly."""
    scene = conforming_scene()
    duplicate = Collider(
        prim_path="/duckiebot/base_link/collisions/second_caster/sphere",
        body=DUCKIEBOT.base_link_name,
        kind="sphere",
        center_root_m=DUCKIEBOT.caster_center_base_frame_m,
        radius_m=DUCKIEBOT.caster_radius_m,
    )
    with pytest.raises(PatchPlanError, match="selects 2 colliders"):
        plan_patch(_replace(scene, colliders=[*scene.colliders, duplicate]))


def test_a_missing_wheel_collider_raises() -> None:
    """One wheel collider instead of two is not something to patch half of."""
    scene = conforming_scene()
    scene = _replace(scene, colliders=scene.colliders[:3])
    with pytest.raises(PatchPlanError, match="selects 1 colliders"):
        plan_patch(scene)


def test_the_plan_is_idempotent_on_an_already_patched_asset() -> None:
    """Re-running the toolchain must produce the same plan, not an accumulating one."""
    scene = conforming_scene()
    first = plan_patch(scene)
    second = plan_patch(scene)
    assert first == second


# =============================================================================================
# Real USD: extractor, patch and verification against an authored stage
# =============================================================================================


def _author_synthetic_robot(usd_path: Path, wheel_kind: str = "sphere") -> None:
    """Author a robot stage shaped like the Isaac Sim URDF importer's output.

    The important structural details, all copied from a real import of
    ``assets/duckiebot/duckiebot.urdf``: collider geometry lives under a root-level ``/colliders``
    scope, each link pulls its colliders in through an ``instanceable`` reference, the chassis box
    is a unit ``Cube`` scaled by ``xformOp:scale``, and the collision APIs sit on the leaf shapes.

    Args:
        usd_path: Where to write the ``.usda``.
        wheel_kind: ``sphere`` for a correct asset, ``cylinder`` to exercise the patch swap.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    p = DUCKIEBOT
    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    def xform(
        path: str,
        translate: tuple[float, float, float] = (0.0, 0.0, 0.0),
        scale: tuple[float, float, float] | None = None,
    ) -> Any:
        """Define an Xform with an optional translate and scale."""
        prim = UsdGeom.Xform.Define(stage, path)
        prim.AddTranslateOp().Set(Gf.Vec3d(*translate))
        if scale is not None:
            prim.AddScaleOp().Set(Gf.Vec3d(*scale))
        return prim

    def rigid_body(
        path: str,
        translate: tuple[float, float, float],
        mass: float,
        com: tuple[float, float, float],
        inertia: tuple[float, float, float],
    ) -> Any:
        """Define a link Xform carrying the rigid body and mass APIs."""
        prim = xform(path, translate).GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.CreateMassAttr().Set(mass)
        mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(*com))
        mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*inertia))
        mass_api.CreatePrincipalAxesAttr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        return prim

    robot = xform("/duckiebot").GetPrim()
    stage.SetDefaultPrim(robot)
    base = rigid_body(
        "/duckiebot/base_link",
        (0.0, 0.0, 0.0),
        p.base_mass_kg,
        p.base_com_base_frame_m,
        p.base_inertia_about_com,
    )
    UsdPhysics.ArticulationRootAPI.Apply(base)
    for link, origin in (
        (p.left_wheel_link_name, p.left_wheel_origin_m),
        (p.right_wheel_link_name, p.right_wheel_origin_m),
    ):
        rigid_body(
            f"/duckiebot/{link}",
            origin,
            p.wheel_mass_kg,
            (0.0, 0.0, 0.0),
            p.wheel_inertia_about_com,
        )

    # Collider source scope, referenced instanceably into each link, exactly as the importer does.
    UsdGeom.Scope.Define(stage, "/colliders")
    chassis = xform("/colliders/base_link/chassis_collision", p.chassis_center_base_frame_m, p.chassis_size_m)
    del chassis
    box = UsdGeom.Cube.Define(stage, "/colliders/base_link/chassis_collision/box")
    box.CreateSizeAttr().Set(1.0)
    UsdPhysics.CollisionAPI.Apply(box.GetPrim())
    xform("/colliders/base_link/caster_collision", p.caster_center_base_frame_m)
    caster = UsdGeom.Sphere.Define(stage, "/colliders/base_link/caster_collision/sphere")
    caster.CreateRadiusAttr().Set(p.caster_radius_m)
    UsdPhysics.CollisionAPI.Apply(caster.GetPrim())
    for link in (p.left_wheel_link_name, p.right_wheel_link_name):
        xform(f"/colliders/{link}/{link}_collision")
        if wheel_kind == "sphere":
            shape = UsdGeom.Sphere.Define(stage, f"/colliders/{link}/{link}_collision/sphere")
            shape.CreateRadiusAttr().Set(p.wheel_radius_m)
        else:
            shape = UsdGeom.Cylinder.Define(stage, f"/colliders/{link}/{link}_collision/cylinder")
            shape.CreateRadiusAttr().Set(p.wheel_radius_m)
            shape.CreateHeightAttr().Set(p.wheel_width_m)
            shape.CreateAxisAttr().Set(UsdGeom.Tokens.y)
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
    for link in (p.base_link_name, p.left_wheel_link_name, p.right_wheel_link_name):
        collisions = stage.DefinePrim(f"/duckiebot/{link}/collisions", "Xform")
        collisions.GetReferences().AddInternalReference(f"/colliders/{link}")
        collisions.SetInstanceable(True)

    UsdGeom.Scope.Define(stage, "/duckiebot/joints")
    for joint_name, link, origin in (
        (p.left_wheel_joint_name, p.left_wheel_link_name, p.left_wheel_origin_m),
        (p.right_wheel_joint_name, p.right_wheel_link_name, p.right_wheel_origin_m),
    ):
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"/duckiebot/joints/{joint_name}")
        joint.CreateAxisAttr().Set(UsdGeom.Tokens.y)
        joint.CreateBody0Rel().SetTargets(["/duckiebot/base_link"])
        joint.CreateBody1Rel().SetTargets([f"/duckiebot/{link}"])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*origin))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    stage.GetRootLayer().Save()


@needs_usd
def test_the_extractor_sees_colliders_hidden_behind_instance_proxies(tmp_path: Path) -> None:
    """A plain ``stage.Traverse()`` reports zero colliders on an imported robot; this must not."""
    usd_path = tmp_path / "robot.usda"
    _author_synthetic_robot(usd_path)
    scene = extract_scene(verify_usd.open_stage(usd_path), source=str(usd_path))
    assert len(scene.colliders) == 4
    assert sorted(c.kind for c in scene.colliders) == ["box", "sphere", "sphere", "sphere"]
    assert {c.body for c in scene.colliders} == {
        DUCKIEBOT.base_link_name,
        DUCKIEBOT.left_wheel_link_name,
        DUCKIEBOT.right_wheel_link_name,
    }


@needs_usd
def test_the_extractor_folds_the_xform_scale_into_the_box_extents(tmp_path: Path) -> None:
    """The importer writes a unit Cube plus a scale; reading ``size`` alone gives 0.5 m half."""
    usd_path = tmp_path / "robot.usda"
    _author_synthetic_robot(usd_path)
    scene = extract_scene(verify_usd.open_stage(usd_path), source=str(usd_path))
    box = next(c for c in scene.colliders if c.kind == "box")
    expected = tuple(0.5 * s for s in DUCKIEBOT.chassis_size_m)
    # 1e-7 m: USD stores the transform in float32, and the error this guards against
    # (reading the unit Cube's size and ignoring the scale) is 0.41 m, not a nanometre.
    assert box.half_extents_m == pytest.approx(expected, abs=1e-7)


@needs_usd
def test_an_unpatched_stage_fails_only_the_material_check(tmp_path: Path) -> None:
    """Import alone gets the geometry right and the physics materials wrong. That is the point."""
    usd_path = tmp_path / "robot.usda"
    _author_synthetic_robot(usd_path)
    _scene, results = verify_usd.verify_usd_file(usd_path)
    assert {r.check_id for r in results if not r.ok} == {"S3.2.materials"}


@needs_usd
def test_patching_then_verifying_passes_every_check(tmp_path: Path) -> None:
    """The full S3.2 chain on a real stage: extract, plan, apply, re-open, verify."""
    usd_path = tmp_path / "robot.usda"
    _author_synthetic_robot(usd_path)
    report = patch_usd.patch_usd_file(usd_path)
    assert report.saved
    assert len(report.bound) == 3
    _scene, results = verify_usd.verify_usd_file(usd_path)
    assert [(r.check_id, r.detail) for r in results if not r.ok] == []


@needs_usd
def test_patching_replaces_a_real_cylinder_wheel_collider(tmp_path: Path) -> None:
    """The swap path, executed for real: cylinder in, sphere of radius 0.0318 m out."""
    usd_path = tmp_path / "robot.usda"
    _author_synthetic_robot(usd_path, wheel_kind="cylinder")
    before = extract_scene(verify_usd.open_stage(usd_path), source=str(usd_path))
    assert sorted(c.kind for c in before.colliders) == ["box", "cylinder", "cylinder", "sphere"]

    patch_usd.patch_usd_file(usd_path)

    scene, results = verify_usd.verify_usd_file(usd_path)
    assert sorted(c.kind for c in scene.colliders) == ["box", "sphere", "sphere", "sphere"]
    assert [(r.check_id, r.detail) for r in results if not r.ok] == []


@needs_usd
def test_patching_twice_is_idempotent(tmp_path: Path) -> None:
    """A rebuild must not accumulate materials, bindings or replacement prims."""
    usd_path = tmp_path / "robot.usda"
    _author_synthetic_robot(usd_path)
    patch_usd.patch_usd_file(usd_path)
    first = extract_scene(verify_usd.open_stage(usd_path), source=str(usd_path)).to_dict()
    patch_usd.patch_usd_file(usd_path)
    second = extract_scene(verify_usd.open_stage(usd_path), source=str(usd_path)).to_dict()
    assert first == second


@needs_usd
def test_a_dry_run_changes_nothing_on_disk(tmp_path: Path) -> None:
    """``--dry-run`` has to be safe to point at a good asset."""
    usd_path = tmp_path / "robot.usda"
    _author_synthetic_robot(usd_path)
    original = usd_path.read_text(encoding="utf-8")
    report = patch_usd.patch_usd_file(usd_path, save=False)
    assert not report.saved
    assert usd_path.read_text(encoding="utf-8") == original


@needs_usd
def test_apply_patch_reports_the_de_instancing_it_had_to_do(tmp_path: Path) -> None:
    """Instance proxies are read-only; the report must say which prims were de-instanced."""
    usd_path = tmp_path / "robot.usda"
    _author_synthetic_robot(usd_path)
    stage = verify_usd.open_stage(usd_path)
    scene = extract_scene(stage, source=str(usd_path))
    report = apply_patch(stage, plan_patch(scene), save=False)
    assert sorted(report.de_instanced) == [
        "/duckiebot/base_link/collisions",
        "/duckiebot/left_wheel_link/collisions",
        "/duckiebot/right_wheel_link/collisions",
    ]


# =============================================================================================
# The URDF import configuration, checked without Isaac Sim
# =============================================================================================


class _RecordingImportConfig:
    """Stand-in for ``isaacsim.asset.importer.urdf._urdf.ImportConfig``.

    It accepts exactly the setters named in :data:`import_urdf_headless.IMPORT_SETTINGS`, minus
    any the test asks it to withhold, and records what it was called with.

    Attributes:
        calls: ``setter name -> positional arguments``, in call order.
    """

    def __init__(self, missing: str | None = None) -> None:
        """Build the double.

        Args:
            missing: Setter to leave undefined, to simulate an older or newer importer build.
        """
        self.calls: dict[str, tuple[object, ...]] = {}
        for name in import_urdf_headless.IMPORT_SETTINGS:
            if name == missing:
                continue
            setattr(self, name, self._recorder(name))

    def _recorder(self, name: str) -> object:
        """Return a callable that records one setter's arguments.

        Args:
            name: Setter name.

        Returns:
            The recording callable.
        """

        def record(*args: object) -> None:
            """Record a call."""
            self.calls[name] = args

        return record


def test_the_import_configuration_is_applied_in_full() -> None:
    """Every entry of IMPORT_SETTINGS reaches the ImportConfig, with its arguments intact."""
    config = _RecordingImportConfig()
    applied = import_urdf_headless.apply_import_settings(config)
    assert config.calls == import_urdf_headless.IMPORT_SETTINGS
    assert len(applied) == len(import_urdf_headless.IMPORT_SETTINGS)


def test_a_missing_setter_stops_the_import_and_names_it() -> None:
    """This is exactly how Isaac Lab's UrdfConverter fails here, only with a usable message."""
    config = _RecordingImportConfig(missing="set_import_inertia_tensor")
    with pytest.raises(import_urdf_headless.ImportError_) as excinfo:
        import_urdf_headless.apply_import_settings(config)
    assert "set_import_inertia_tensor" in str(excinfo.value)
    assert "set_density" in str(excinfo.value)  # the message lists what this build does have


def test_colliders_never_come_from_the_visual_geometry() -> None:
    """The wheel visual is a cylinder: importing colliders from visuals would recreate the bug."""
    assert import_urdf_headless.IMPORT_SETTINGS["set_collision_from_visuals"] == (False,)
    assert import_urdf_headless.IMPORT_SETTINGS["set_replace_cylinders_with_capsules"] == (False,)


def test_mass_properties_come_from_the_urdf_and_not_from_a_density() -> None:
    """Density 0 plus import_inertia_tensor is what makes the M1 mass and inertia checks pass."""
    assert import_urdf_headless.IMPORT_SETTINGS["set_density"] == (0.0,)
    assert import_urdf_headless.IMPORT_SETTINGS["set_import_inertia_tensor"] == (True,)


def test_the_robot_imports_free_floating_and_z_up() -> None:
    """A fixed base cannot drive, and a y-up import lies the robot on its side."""
    assert import_urdf_headless.IMPORT_SETTINGS["set_fix_base"] == (False,)
    assert import_urdf_headless.IMPORT_SETTINGS["set_up_vector"] == (0.0, 0.0, 1.0)
    assert import_urdf_headless.IMPORT_SETTINGS["set_merge_fixed_joints"] == (True,)
    assert import_urdf_headless.IMPORT_SETTINGS["set_distance_scale"] == (1.0,)


def test_the_binary_staging_directory_is_not_the_published_asset() -> None:
    """The importer's binary output must not land where the clean-room gate will find it."""
    output = Path(DEFAULT_USD_PATH)
    staged = import_urdf_headless.staging_path_for(output)
    assert output.suffix == ".usda"
    assert staged.suffix == ".usd"
    assert staged.parent.name == "_import"
    assert staged != output


def test_a_stale_urdf_stops_the_build_before_kit_boots(tmp_path: Path) -> None:
    """Importing a stale URDF wastes a Kit boot and produces a USD that explains nothing."""
    stale = tmp_path / "duckiebot.urdf"
    stale.write_text("<robot name='duckiebot'/>", encoding="utf-8")
    with pytest.raises(import_urdf_headless.ImportError_, match="stale"):
        import_urdf_headless.check_urdf_is_current(stale)
    missing = tmp_path / "nope.urdf"
    with pytest.raises(import_urdf_headless.ImportError_, match="build_robot_asset"):
        import_urdf_headless.check_urdf_is_current(missing)


def test_the_committed_urdf_is_accepted_as_current() -> None:
    """The positive case, so the staleness check cannot pass by always raising."""
    import_urdf_headless.check_urdf_is_current(Path(import_urdf_headless.DEFAULT_URDF_PATH))
