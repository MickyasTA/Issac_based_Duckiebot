r"""Structural tests for the generated MJCF (SPEC v2 S8.1, owner ``[sim2sim]``).

Interpreter: these tests need **only** ``mujoco`` and ``numpy``, so they run today in the tools venv
``d:/Personal/personal/mujoco_venv/Scripts/python.exe``, which is the only interpreter on this
machine with MuJoCo. They deliberately do not import torch, Pillow, pyyaml or cv2, none of which the
tools venv has yet (see SPEC v2 M0 and :func:`duckiebot_rl.sim2sim.environment_report`).

Run with::

    d:/Personal/personal/mujoco_venv/Scripts/python.exe -m pytest tests/unit/test_mjcf.py \\
        --run-mujoco -q
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

#: The project's opt-in marker (conftest.py, pyproject.toml). Without it these tests could not be
#: SELECTED by `pytest -m mujoco --run-mujoco`, which is the mechanism CI uses to run them on a
#: runner that has the mujoco wheel installed; they would only ever run when a human remembered to
#: point the tools venv at this file. The importorskip below stays as a belt-and-braces guard for
#: anyone who runs the file directly.
pytestmark = pytest.mark.mujoco

mujoco = pytest.importorskip("mujoco", reason="run these with the tools venv (mujoco_venv)")

from duckiebot_rl.sim2sim import _resolve  # noqa: E402
from duckiebot_rl.sim2sim import mjcf as _mjcf  # noqa: E402

TOL = 1e-9


@pytest.fixture(scope="module")
def cfg() -> _mjcf.MjcfCfg:
    """The MJCF configuration resolved from the shared parameter modules."""
    return _mjcf.MjcfCfg.from_shared()


@pytest.fixture(scope="module")
def model(cfg: _mjcf.MjcfCfg):
    """A compiled robot-on-a-plane model."""
    return mujoco.MjModel.from_xml_string(_mjcf.build_robot_xml(cfg))


def _geom(model, name: str) -> int:
    """Return a geom id by name, failing the test if it is missing."""
    index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    assert index >= 0, f"model has no geom named {name!r}"
    return int(index)


def _body(model, name: str) -> int:
    """Return a body id by name, failing the test if it is missing."""
    index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    assert index >= 0, f"model has no body named {name!r}"
    return int(index)


# ------------------------------------------------------------------------------------- parsing
def test_generated_mjcf_parses(model) -> None:
    """The generated XML compiles in MuJoCo without warnings-as-errors."""
    assert model.nq > 0 and model.nv > 0


def test_parameters_come_from_the_shared_module(cfg: _mjcf.MjcfCfg) -> None:
    """The MJCF must be driven by ``duckiebot_rl.assets.params``, not by a second copy.

    If this fails with the SPEC v2 fallback in the provenance string, the shared module either does
    not exist or does not expose the whole adapter, and the two simulators are free to drift.
    """
    assert "duckiebot_rl.assets.params" in cfg.params_source, (
        f"robot parameters came from {cfg.params_source!r}; the whole point of generating the MJCF "
        f"is that it reads the same source of truth as the Isaac articulation config"
    )


# -------------------------------------------------------------------------------- structure
def test_body_joint_and_actuator_counts(model, cfg: _mjcf.MjcfCfg) -> None:
    """Three bodies, two wheel hinges plus the free joint, two velocity actuators."""
    robot = cfg.robot
    assert model.nbody == 4, "world plus base_link plus two wheels"
    for name in (robot.base_link_name, robot.left_wheel_link_name, robot.right_wheel_link_name):
        _body(model, name)
    assert model.njnt == 3, "one free joint plus two wheel hinges"
    assert model.nv == 8, "6 free-joint dof plus 2 wheel dof"
    assert model.nu == 2
    for name in (robot.left_wheel_joint_name, robot.right_wheel_joint_name):
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert joint >= 0, f"missing wheel joint {name!r}"
        assert model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_HINGE
        np.testing.assert_allclose(model.jnt_axis[joint], [0.0, 1.0, 0.0], atol=TOL)


def test_wheel_collision_geoms_are_spheres(model, cfg: _mjcf.MjcfCfg) -> None:
    """Wheel contact must be a sphere.

    A cylinder makes a two-point line contact whose torsional couple fights differential-drive yaw.
    Measured here at the SPEC v2 parameters, dt 1/240, on the (10, 30) rad/s arc: sphere 0.04% of
    yaw error against the kinematic prediction, cylinder 15.9%. This is the single most important
    structural property of the model, so it is asserted rather than trusted, and
    ``test_mj_kinematics.py::test_cylinder_wheel_contact_destroys_yaw`` re-measures the ablation.
    """
    robot = cfg.robot
    for link in (robot.left_wheel_link_name, robot.right_wheel_link_name):
        geom = _geom(model, f"{link}_collision")
        assert model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_SPHERE, (
            f"{link} collision geom is {mujoco.mjtGeom(model.geom_type[geom]).name}, not a sphere; "
            f"cylinder wheel contact measured 15.9% of differential-drive yaw error against "
            f"the sphere's 0.04%"
        )
        assert model.geom_size[geom, 0] == pytest.approx(robot.wheel_radius, abs=TOL)
        assert model.geom_contype[geom] != 0, "the wheel collision geom must actually collide"


def test_visual_geoms_do_not_collide(model) -> None:
    """Decorative geoms are visual only, so they cannot perturb the physics."""
    for name in ("left_wheel_link_visual", "camera_block_visual", "marker_visual", "caster_visual"):
        geom = _geom(model, name)
        assert model.geom_contype[geom] == 0 and model.geom_conaffinity[geom] == 0


def test_caster_is_a_frictionless_sphere_tangent_to_the_ground(model, cfg: _mjcf.MjcfCfg) -> None:
    """The caster is a frictionless sphere whose contact point sits exactly on z = 0."""
    robot = cfg.robot
    geom = _geom(model, "caster_collision")
    assert model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_SPHERE
    assert model.geom_size[geom, 0] == pytest.approx(robot.caster_radius, abs=TOL)
    assert model.geom_condim[geom] == 1, "condim 1 is what makes the caster frictionless"
    contact_z = robot.base_height + model.geom_pos[geom, 2] - model.geom_size[geom, 0]
    assert contact_z == pytest.approx(0.0, abs=1e-9)


def test_chassis_ground_clearance(model, cfg: _mjcf.MjcfCfg) -> None:
    """The chassis box bottom sits 21 mm above the ground (SPEC v2 S1 item 24)."""
    robot = cfg.robot
    geom = _geom(model, f"{robot.base_link_name}_collision")
    assert model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX
    bottom = robot.base_height + model.geom_pos[geom, 2] - model.geom_size[geom, 2]
    assert bottom == pytest.approx(0.021, abs=1e-6)


def test_masses_and_inertias_match_the_shared_params(model, cfg: _mjcf.MjcfCfg) -> None:
    """Body masses and principal moments come from the shared parameter object."""
    robot = cfg.robot
    assert model.body_mass[_body(model, robot.base_link_name)] == pytest.approx(robot.base_mass, abs=1e-12)
    for link in (robot.left_wheel_link_name, robot.right_wheel_link_name):
        assert model.body_mass[_body(model, link)] == pytest.approx(robot.wheel_mass, abs=1e-12)
    np.testing.assert_allclose(
        model.body_inertia[_body(model, robot.base_link_name)], robot.base_inertia_diag, rtol=1e-9
    )
    np.testing.assert_allclose(
        model.body_ipos[_body(model, robot.base_link_name)], robot.base_com, atol=1e-12
    )
    np.testing.assert_allclose(
        model.body_inertia[_body(model, robot.left_wheel_link_name)],
        robot.wheel_inertia_diag,
        rtol=1e-9,
    )
    total = float(model.body_mass.sum())
    assert total == pytest.approx(robot.base_mass + 2 * robot.wheel_mass, abs=1e-12)


def test_wheel_track_matches_the_baseline(model, cfg: _mjcf.MjcfCfg) -> None:
    """The two wheel bodies straddle the base frame by exactly the wheel baseline."""
    robot = cfg.robot
    left = model.body_pos[_body(model, robot.left_wheel_link_name), 1]
    right = model.body_pos[_body(model, robot.right_wheel_link_name), 1]
    assert float(left - right) == pytest.approx(robot.wheel_separation, abs=TOL)


# --------------------------------------------------------------------------------- integrator
def test_integrator_is_implicitfast(model, cfg: _mjcf.MjcfCfg) -> None:
    """The integrator is locked, for the structural reason.

    PhysX integrates its implicit velocity drive implicitly, so ``implicitfast`` is the matching
    choice and matching structure is the entire job of this package. The v1 empirical justification
    (-87% forward speed and +133% yaw rate under ``Euler``) does **not** reproduce at the SPEC v2
    armature: ``test_mj_kinematics.py::test_integrator_choice_is_locked_and_its_margin_measured``
    measures the separation at 0.00002% and asserts it stays small. Do not restate the v1 numbers
    here; they were measured on a model that no longer exists.
    """
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
    assert model.opt.impratio == pytest.approx(cfg.sim.impratio)
    assert model.opt.timestep == pytest.approx(cfg.sim.physics_dt, rel=1e-12)


def test_euler_integrator_is_rejected_by_config() -> None:
    """``SimParams.validate`` refuses an explicit integrator outright."""
    with pytest.raises(ValueError, match="implicitfast"):
        _resolve.SimParams(integrator="Euler").validate()


def test_rates_match_the_isaac_reference(cfg: _mjcf.MjcfCfg) -> None:
    """Both simulators run 1/240 with decimation 16, resolving critic item J."""
    assert cfg.sim.physics_dt == pytest.approx(1.0 / 240.0)
    assert cfg.sim.decimation == 16
    assert cfg.sim.control_hz == pytest.approx(15.0)
    ratio = (1.0 / cfg.sim.control_hz) / cfg.sim.physics_dt
    assert ratio == pytest.approx(round(ratio)), "S8.3 item 3: integer decimation in both sims"


# ----------------------------------------------------------------------------------- actuators
def test_actuators_map_onto_the_isaac_implicit_drive(model, cfg: _mjcf.MjcfCfg) -> None:
    """``kv`` equals the Isaac drive damping, and the limits equal the Isaac effort and velocity."""
    robot = cfg.robot
    for actuator in range(model.nu):
        assert model.actuator_gainprm[actuator, 0] == pytest.approx(robot.joint_damping, abs=TOL)
        assert model.actuator_biasprm[actuator, 2] == pytest.approx(-robot.joint_damping, abs=TOL)
        np.testing.assert_allclose(
            model.actuator_forcerange[actuator], [-robot.effort_limit, robot.effort_limit], atol=TOL
        )
        np.testing.assert_allclose(
            model.actuator_ctrlrange[actuator],
            [-robot.velocity_limit, robot.velocity_limit],
            atol=TOL,
        )


def test_joint_armature_friction_and_damping(model, cfg: _mjcf.MjcfCfg) -> None:
    """Armature and joint friction are non-zero so system identification stage 2 has a target.

    The MJCF joint's own ``damping`` is zero on purpose: the S2 value of 0.05 N.m.s/rad is the servo
    gain ``kv``, not a second passive damper. Duplicating it would double the drive damping.
    """
    robot = cfg.robot
    for name in (robot.left_wheel_joint_name, robot.right_wheel_joint_name):
        dof = model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
        assert model.dof_armature[dof] == pytest.approx(robot.joint_armature, abs=TOL)
        assert model.dof_frictionloss[dof] == pytest.approx(robot.joint_friction, abs=TOL)
        assert model.dof_damping[dof] == pytest.approx(robot.passive_joint_damping, abs=TOL)
        assert model.dof_armature[dof] > 0.0 and model.dof_frictionloss[dof] > 0.0


# -------------------------------------------------------------------------------------- camera
def test_camera_pose_matches_the_golden_quaternion(cfg: _mjcf.MjcfCfg) -> None:
    """The MuJoCo ``xyaxes`` derive from the single shared ROS-convention quaternion helper."""
    zero = _resolve.quat_cam_ros(0.0)
    np.testing.assert_allclose(zero, (0.5, -0.5, 0.5, -0.5), atol=1e-12)
    nominal = _resolve.quat_cam_ros(math.radians(25.3))
    np.testing.assert_allclose(nominal, (0.37837, -0.59736, 0.59736, -0.37837), atol=1e-5)

    pitch = math.radians(cfg.robot.camera_pitch_down_deg)
    xyaxes = _resolve.mj_camera_xyaxes(pitch)
    np.testing.assert_allclose(xyaxes[:3], (0.0, -1.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(xyaxes[3:], (math.sin(pitch), 0.0, math.cos(pitch)), atol=1e-9)
    forward = _resolve.mj_camera_forward(pitch)
    np.testing.assert_allclose(forward, (math.cos(pitch), 0.0, -math.sin(pitch)), atol=1e-9)


def test_camera_is_authored_in_the_model(model, cfg: _mjcf.MjcfCfg) -> None:
    """The camera exists at the S2 mount pose with the S4.1 vertical field of view."""
    camera = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, _mjcf.CAMERA_NAME)
    assert camera >= 0
    np.testing.assert_allclose(model.cam_pos[camera], cfg.robot.camera_pos, atol=1e-9)
    assert model.cam_fovy[camera] == pytest.approx(cfg.robot.camera_fovy_deg, abs=1e-6)
    expected_fovy = math.degrees(2.0 * math.atan(0.5 * cfg.obs.render_h / cfg.robot.camera_focal_px))
    assert model.cam_fovy[camera] == pytest.approx(expected_fovy, abs=0.02), (
        "the MuJoCo fovy must equal the canonical pinhole, or the two simulators see different "
        "amounts of road"
    )


# ------------------------------------------------------------------------------------ contacts
def test_robot_rests_on_exactly_three_tangent_contacts(model) -> None:
    """At the reset pose the two wheels and the caster all touch, with zero penetration."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert data.ncon == 3, f"expected 3 contacts at rest, got {data.ncon}"
    names = set()
    for index in range(data.ncon):
        contact = data.contact[index]
        assert abs(contact.dist) < 1e-9, "the robot must start seated, not buried or floating"
        for geom in (contact.geom1, contact.geom2):
            names.add(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom)))
    assert _mjcf.GROUND_GEOM_NAME in names
    assert "caster_collision" in names


def test_track_scene_has_exactly_one_collidable_surface() -> None:
    """Tiles and walls are visual only; physics is one plane, exactly as in the Isaac scene."""
    import tempfile

    from duckiebot_rl.sim2sim import track as _track

    with tempfile.TemporaryDirectory() as tmp:
        scene = _track.build_track(_track.LOOP_5X5, asset_dir=tmp)
        model = mujoco.MjModel.from_xml_string(scene.xml, {})
    world_geoms = [
        index
        for index in range(model.ngeom)
        if model.geom_bodyid[index] == 0 and model.geom_contype[index] != 0
    ]
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in world_geoms]
    assert names == [_mjcf.GROUND_GEOM_NAME], (
        f"the only collidable world geom must be the ground plane, found {names}; tile-edge "
        f"contacts are a divergence source Isaac does not have"
    )


def test_track_textures_load_through_both_mujoco_entry_points() -> None:
    """``texturedir`` must be absolute, because the two compile paths resolve it differently.

    The regression this pins cost the whole S8 harness a day. ``build_track`` used to set
    ``<compiler texturedir>`` to the asset directory as given. ``MjModel.from_xml_path``
    resolves a RELATIVE texturedir against the directory holding the XML, and the scene XML is
    written into that same asset directory, so the compiler looked under
    ``<asset_dir>/<asset_dir>/tile_0.png`` and failed with "Error opening file" while six
    perfectly valid PNGs sat next to the XML. The obvious repair, ``texturedir="."``, fixes that
    path and breaks the other one: ``MjModel.from_xml_string`` has no file context and resolves
    against the process CWD, which is where ``MjDuckiebotEnv`` compiles every scene it runs.

    Only an absolute path is correct for both, so both are exercised here.
    """
    import re
    import tempfile
    from pathlib import Path

    from duckiebot_rl.sim2sim import track as _track

    with tempfile.TemporaryDirectory() as tmp:
        scene = _track.build_track(_track.LOOP_5X5, asset_dir=tmp)
        texturedir = re.search(r'texturedir="([^"]*)"', scene.xml).group(1)
        assert Path(texturedir).is_absolute(), (
            f"texturedir {texturedir!r} must be absolute; a relative value resolves against "
            "different bases in from_xml_path and from_xml_string"
        )

        from_string = mujoco.MjModel.from_xml_string(scene.xml)

        xml_path = Path(tmp) / "track.xml"
        xml_path.write_text(scene.xml, encoding="utf-8")
        from_path = mujoco.MjModel.from_xml_path(str(xml_path))

    assert from_string.ntex == from_path.ntex > 0, "both entry points must load the tile textures"
