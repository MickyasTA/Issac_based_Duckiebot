"""Consistency tests for :mod:`duckiebot_rl.assets.params` (SPEC v2 S2).

The v1 architecture document stated four mutually contradictory sets of numbers, and nothing in
the build caught it: the caster radius appeared as both 0.021 m and 0.0318 m, the chassis
clearance was written as 21 mm while the geometry gave 6.3 mm, the wheel effort limit was 13x a
DG01D 48:1 stall torque, and the camera frame carried a mount pose the surrounding text rejected.

These tests are the machine that stops that happening again. They fall into three groups:

1. the closure identities (a number recomputed from other numbers must match the number stated),
   tested both by asserting on the nominal singleton and by asserting that a deliberately broken
   copy is rejected at construction;
2. the shape of the parameter set: frozen, fully documented, every field carrying a unit and a
   provenance tag, every domain-randomization range well ordered and bracketing its nominal;
3. agreement between the parameters and the plain-dictionary views that
   :mod:`duckiebot_rl.assets.robot_cfg` hands to Isaac Lab, to the USD patch script and to the
   MuJoCo twin.

Everything here runs on CPU with no Isaac Sim and no GPU.
"""

from __future__ import annotations

import ast
import dataclasses
import math
import re
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from duckiebot_rl.assets import params as params_module  # noqa: E402
from duckiebot_rl.assets.params import (  # noqa: E402
    DUCKIEBOT,
    DuckiebotParams,
    ParameterConsistencyError,
)
from duckiebot_rl.assets.robot_cfg import (  # noqa: E402
    camera_mount_spec,
    is_isaaclab_available,
    physics_material_spec,
    spawn_property_spec,
    wheel_actuator_spec,
)

PROVENANCE_TAGS = ("[S]", "[C]", "[M]", "[E]", "[v2]")
"""The provenance vocabulary defined in the params module docstring."""

NAMING_FIELDS = frozenset(
    {
        "base_link_name",
        "left_wheel_link_name",
        "right_wheel_link_name",
        "left_wheel_joint_name",
        "right_wheel_joint_name",
        "wheel_joint_regex",
    }
)
"""Fields that name things rather than measure them, so they carry no unit or provenance tag."""


def _broken(**overrides: object) -> None:
    """Construct a parameter set with overrides, expecting it to be rejected.

    Args:
        **overrides: Field values to replace on a copy of the nominal set.

    Raises:
        ParameterConsistencyError: Which is what the caller asserts on.
    """
    dataclasses.replace(DUCKIEBOT, **overrides)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------------------------
# Standing geometry: caster, wheels, ground clearance
# ---------------------------------------------------------------------------------------------


def test_caster_wheel_and_clearance_are_mutually_consistent() -> None:
    """The three numbers the v1 critique found contradicting each other now close exactly.

    A level chassis rests on two wheels of radius ``r_w`` and one rear ball of radius ``r_c``.
    That forces ``base_link`` to sit at ``r_w`` and the caster centre to sit at ``r_c`` above the
    ground, i.e. at ``r_c - r_w`` in base frame. Any other triple makes the robot rock.
    """
    assert DUCKIEBOT.base_link_height_m == pytest.approx(DUCKIEBOT.wheel_radius_m)
    assert DUCKIEBOT.caster_radius_m == pytest.approx(0.0165)
    assert DUCKIEBOT.caster_radius_m < DUCKIEBOT.wheel_radius_m
    expected_centre_z = DUCKIEBOT.caster_radius_m - DUCKIEBOT.base_link_height_m
    assert DUCKIEBOT.caster_center_base_frame_m[2] == pytest.approx(expected_centre_z, abs=1e-9)
    assert DUCKIEBOT.caster_contact_height_m == pytest.approx(0.0, abs=1e-9)


def test_ground_clearance_matches_the_chassis_box() -> None:
    """The stated 21 mm clearance is what the collision box actually produces."""
    assert DUCKIEBOT.chassis_bottom_height_m == pytest.approx(DUCKIEBOT.ground_clearance_m, abs=1e-9)
    assert DUCKIEBOT.chassis_bottom_height_m == pytest.approx(0.021, abs=1e-9)
    # It must exceed the worst-case D15 tile tilt of 1 deg over the 0.18 m box length.
    tilt_rise = 0.5 * DUCKIEBOT.chassis_size_m[0] * math.tan(math.radians(1.0))
    assert DUCKIEBOT.chassis_bottom_height_m > tilt_rise
    # And the caster, not the chassis, must be the rear ground contact.
    assert DUCKIEBOT.chassis_bottom_height_m > DUCKIEBOT.caster_radius_m


def test_robot_width_bounds_the_physical_envelope() -> None:
    """``W_R``, which parameterizes the wrong-lane gate, really is the widest dimension."""
    wheel_envelope = DUCKIEBOT.wheel_baseline_m + DUCKIEBOT.wheel_width_m
    assert DUCKIEBOT.robot_width_m >= wheel_envelope
    assert DUCKIEBOT.robot_width_m >= DUCKIEBOT.chassis_size_m[1]
    assert DUCKIEBOT.robot_width_m == pytest.approx(0.131)
    # The S5.5 wrong-lane gate is (w_lane - W_R) / 2 + 0.02. At the narrowest randomized lane
    # (0.170 m) that must still leave the robot room to exist.
    narrowest_lane_m = 0.170
    assert DUCKIEBOT.robot_width_m < narrowest_lane_m


@pytest.mark.parametrize(
    ("field", "value", "why"),
    [
        ("caster_radius_m", 0.021, "the v1 prose value: the contact point would float 4.5 mm up"),
        ("caster_radius_m", 0.0318, "the v1 URDF value: a rear ball as tall as the wheels"),
        ("chassis_center_base_frame_m", (-0.015, 0.0, 0.012), "the v1 6.3 mm clearance"),
        ("base_link_height_m", 0.0400, "a base frame that does not sit on the wheels"),
        ("ground_clearance_m", 0.0063, "the clearance the v1 geometry actually produced"),
    ],
)
def test_geometry_contradictions_are_rejected(field: str, value: object, why: str) -> None:
    """Each historical contradiction now fails loudly at construction time."""
    with pytest.raises(ParameterConsistencyError):
        _broken(**{field: value})


# ---------------------------------------------------------------------------------------------
# Mass and inertia
# ---------------------------------------------------------------------------------------------


def test_masses_sum_to_the_spec_figure() -> None:
    """1.00 kg base plus two 0.05 kg wheels is the 1.10 kg assembled mass."""
    assert DUCKIEBOT.base_mass_kg == pytest.approx(1.000)
    assert DUCKIEBOT.wheel_mass_kg == pytest.approx(0.050)
    assert DUCKIEBOT.total_mass_kg == pytest.approx(1.100)


def test_inertia_properties_match_the_spec_table() -> None:
    """The analytic tensors reproduce the values printed in SPEC v2 S3.2."""
    ixx, iyy, izz = DUCKIEBOT.base_inertia_about_com
    assert (ixx, iyy, izz) == pytest.approx((1.877e-3, 3.169e-3, 4.108e-3), rel=5e-4)
    wxx, wyy, wzz = DUCKIEBOT.wheel_inertia_about_com
    assert (wxx, wyy, wzz) == pytest.approx((1.568e-5, 2.528e-5, 1.568e-5), rel=5e-4)
    # The chassis is longest in x, so its largest moment must be about z.
    assert izz > iyy > ixx
    # The wheel spins about y, so y carries the largest moment.
    assert wyy > wxx


def test_centre_of_mass_sits_inside_the_chassis_and_low() -> None:
    """The CoM is inside the box and below its geometric centre, as a battery-heavy robot is."""
    com = DUCKIEBOT.base_com_base_frame_m
    centre = DUCKIEBOT.chassis_center_base_frame_m
    half = tuple(0.5 * s for s in DUCKIEBOT.chassis_size_m)
    for i in range(3):
        assert abs(com[i] - centre[i]) <= half[i]
    assert com[2] < centre[2], "the centre of mass should be below the box centre"
    assert com[1] == 0.0, "the robot is laterally symmetric"


def test_centre_of_mass_outside_the_chassis_is_rejected() -> None:
    """A CoM outside the body is a modelling error, not a randomization."""
    with pytest.raises(ParameterConsistencyError):
        _broken(base_com_base_frame_m=(0.5, 0.0, 0.015))


# ---------------------------------------------------------------------------------------------
# Actuation
# ---------------------------------------------------------------------------------------------


def test_effort_limit_is_physically_achievable() -> None:
    """0.15 N.m gives 0.87 g of tractive acceleration; v1's 2.0 N.m gave 11.7 g."""
    assert DUCKIEBOT.wheel_effort_limit_nm == pytest.approx(0.15)
    assert DUCKIEBOT.max_tractive_force_n == pytest.approx(9.43, abs=0.02)
    assert DUCKIEBOT.max_tractive_accel_g < 1.0
    # The nominal sits inside the D17 randomization clamp, and that clamp stays plausible for a
    # DG01D 48:1 whose stall torque is quoted between 0.08 and 0.18 N.m.
    lo, hi = DUCKIEBOT.dr_effort_limit_nm
    assert lo <= DUCKIEBOT.wheel_effort_limit_nm <= hi
    assert hi < 0.5


def test_v1_effort_limit_is_rejected() -> None:
    """The exact v1 number must not be settable."""
    with pytest.raises(ParameterConsistencyError):
        _broken(wheel_effort_limit_nm=2.0, dr_effort_limit_nm=(0.06, 4.0))


def test_speed_limits_are_mutually_consistent() -> None:
    """Commanded speed sits under the open-loop top speed, which is ``k * r``."""
    assert DUCKIEBOT.top_speed_m_s == pytest.approx(
        DUCKIEBOT.motor_constant_k_rad_s_per_duty * DUCKIEBOT.wheel_radius_m, abs=1e-3
    )
    assert DUCKIEBOT.v_cmd_max_m_s < DUCKIEBOT.top_speed_m_s
    assert DUCKIEBOT.omega_cmd_max_rad_s < DUCKIEBOT.omega_robot_clamp_rad_s
    assert DUCKIEBOT.nominal_max_wheel_speed_rad_s < DUCKIEBOT.wheel_velocity_limit_rad_s
    headroom = DUCKIEBOT.wheel_velocity_limit_rad_s / DUCKIEBOT.nominal_max_wheel_speed_rad_s
    assert headroom > 1.2, "the velocity limit must not clip the nominal command envelope"


def test_joint_dynamics_have_real_values_for_sysid_to_fit() -> None:
    """Damping, armature and friction are numeric and positive (critic item 63).

    The MuJoCo sysid stage 2 fits exactly these three. In v1 they were 0.0 on the Isaac side, so
    the fit had nothing to match against.
    """
    assert DUCKIEBOT.joint_stiffness == 0.0
    assert DUCKIEBOT.joint_damping > 0.0
    assert DUCKIEBOT.joint_armature_kg_m2 > 0.0
    assert DUCKIEBOT.joint_friction_nm > 0.0
    assert DUCKIEBOT.joint_armature_kg_m2 == pytest.approx(2.0e-4)
    # Armature is the rotor inertia reflected through the 48:1 gearbox, so it must be small
    # against the wheel's own spin inertia but not negligible.
    wheel_spin_inertia = DUCKIEBOT.wheel_inertia_about_com[1]
    assert 1.0 < DUCKIEBOT.joint_armature_kg_m2 / wheel_spin_inertia < 100.0


@pytest.mark.parametrize("field", ["joint_damping", "joint_armature_kg_m2", "joint_friction_nm"])
def test_zeroing_a_sysid_target_is_rejected(field: str) -> None:
    """Setting any sysid target back to zero reproduces the v1 defect and must fail."""
    with pytest.raises(ParameterConsistencyError):
        _broken(**{field: 0.0})


def test_control_rate_and_delay_are_consistent() -> None:
    """240 Hz physics with decimation 16 is the 15 Hz deploy rate; 150 ms is 2.25 steps."""
    assert DUCKIEBOT.sim_dt_s == pytest.approx(1.0 / 240.0)
    assert DUCKIEBOT.decimation == 16
    assert DUCKIEBOT.control_hz == pytest.approx(15.0)
    assert DUCKIEBOT.control_dt_s == pytest.approx(1.0 / 15.0)
    assert DUCKIEBOT.actuation_delay_steps == pytest.approx(2.25)
    lo, hi = DUCKIEBOT.dr_delay_control_steps
    assert lo <= DUCKIEBOT.actuation_delay_steps <= hi


def test_pwm_deadband_ordering() -> None:
    """The H-bridge release duty is below the first duty that actually moves the wheel."""
    assert 0.0 < DUCKIEBOT.pwm_release_duty < DUCKIEBOT.pwm_first_nonzero_duty < 1.0
    assert DUCKIEBOT.brake_dw_max_rad_s_per_step > 0.0


# ---------------------------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------------------------


def test_camera_mount_height_closes() -> None:
    """The stated 0.101 m optical height is base height plus mount z, not an independent number."""
    assert DUCKIEBOT.base_link_height_m + DUCKIEBOT.camera_pos_base_frame_m[2] == pytest.approx(
        DUCKIEBOT.camera_height_m, abs=1e-9
    )
    assert DUCKIEBOT.camera_height_m == pytest.approx(0.101)
    assert DUCKIEBOT.camera_pos_base_frame_m[0] == pytest.approx(0.078)
    assert DUCKIEBOT.camera_pitch_down_deg == pytest.approx(25.3)
    # The v1 defect was not an out-of-range value; 0.066 m / 0.108 m is a legal V10 sample. It was
    # that two sources disagreed. Any pair that fails to close is now rejected, which is what
    # having a single source means in practice.
    with pytest.raises(ParameterConsistencyError):
        _broken(camera_height_m=0.108)
    with pytest.raises(ParameterConsistencyError):
        _broken(camera_pos_base_frame_m=(0.066, 0.0, 0.0762))


def test_canonical_camera_is_a_square_pixel_pinhole() -> None:
    """One focal length in both axes, consistent with the stated FOVs and the USD apertures."""
    f_from_hfov = 96.0 / math.tan(math.radians(DUCKIEBOT.camera_hfov_deg / 2.0))
    assert DUCKIEBOT.camera_focal_px == pytest.approx(f_from_hfov, abs=0.01)
    vfov = 2.0 * math.degrees(math.atan(64.0 / DUCKIEBOT.camera_focal_px))
    assert DUCKIEBOT.camera_vfov_deg == pytest.approx(vfov, abs=0.02)
    assert round(DUCKIEBOT.camera_vfov_deg, 1) == 88.3, "S2 quotes the rounded value 88.3"

    f_from_usd = (
        DUCKIEBOT.camera_focal_length_mm * DUCKIEBOT.render_width_px / DUCKIEBOT.camera_horizontal_aperture_mm
    )
    assert f_from_usd == pytest.approx(DUCKIEBOT.camera_focal_px, abs=0.01)
    aspect = DUCKIEBOT.render_height_px / DUCKIEBOT.render_width_px
    assert DUCKIEBOT.camera_vertical_aperture_mm == pytest.approx(
        DUCKIEBOT.camera_horizontal_aperture_mm * aspect, abs=1e-3
    )


def test_canonical_intrinsic_matrix_has_a_centred_principal_point() -> None:
    """``K_canon`` is what the robot rectifies to, so it must be exactly the image centre."""
    k = DUCKIEBOT.canonical_intrinsic_matrix
    assert k[0][0] == k[1][1] == DUCKIEBOT.camera_focal_px
    assert k[0][2] == DUCKIEBOT.render_width_px / 2.0 == 96.0
    assert k[1][2] == DUCKIEBOT.render_height_px / 2.0 == 64.0
    assert k[0][1] == 0.0 and k[1][0] == 0.0
    assert k[2] == (0.0, 0.0, 1.0)


def test_clipping_range_hides_the_neighbouring_city() -> None:
    """Far clip below the 8.0 m env spacing, near clip below the nearest visible ground point."""
    near, far = DUCKIEBOT.camera_clipping_range_m
    assert (near, far) == (0.05, 6.0)
    assert far < 8.0
    assert near < 0.11


def test_render_resolution_is_the_budgeted_one() -> None:
    """192 x 128 is what the S4.2 angular table and the S5.6 VRAM budget were computed at."""
    assert (DUCKIEBOT.render_width_px, DUCKIEBOT.render_height_px) == (192, 128)
    with pytest.raises(ParameterConsistencyError):
        _broken(render_width_px=256)


# ---------------------------------------------------------------------------------------------
# Shape of the parameter set
# ---------------------------------------------------------------------------------------------


def test_parameters_are_frozen() -> None:
    """The shared singleton cannot be mutated; domain randomization must not write here."""
    assert dataclasses.is_dataclass(DuckiebotParams)
    with pytest.raises(dataclasses.FrozenInstanceError):
        DUCKIEBOT.wheel_radius_m = 0.05  # type: ignore[misc]


def test_as_dict_covers_every_field() -> None:
    """``as_dict`` is the serialization the deploy sidecar and the docs read."""
    dumped = DUCKIEBOT.as_dict()
    assert set(dumped) == {f.name for f in dataclasses.fields(DuckiebotParams)}
    assert dumped["wheel_radius_m"] == DUCKIEBOT.wheel_radius_m
    assert "total_mass_kg" not in dumped, "derived properties are recomputed, never stored"


@pytest.mark.parametrize(
    "field",
    sorted(f.name for f in dataclasses.fields(DuckiebotParams) if f.name.startswith("dr_")),
)
def test_domain_randomization_ranges_are_well_ordered(field: str) -> None:
    """Every ``dr_*`` range is a ``(low, high)`` pair with ``low < high``."""
    value = getattr(DUCKIEBOT, field)
    assert isinstance(value, tuple) and len(value) == 2, f"{field} must be a (low, high) pair"
    low, high = value
    assert low < high, f"{field} is not ordered: {value}"


@pytest.mark.parametrize(
    ("nominal_field", "range_field"),
    [
        ("base_mass_kg", "dr_base_mass_kg"),
        ("wheel_baseline_m", "dr_baseline_m"),
        ("wheel_effort_limit_nm", "dr_effort_limit_nm"),
        ("motor_gain_nominal", "dr_motor_gain"),
        ("motor_trim_nominal", "dr_motor_trim"),
        ("joint_friction_nm", "dr_joint_friction_nm"),
        ("camera_height_m", "dr_camera_height_m"),
        ("camera_pitch_down_deg", "dr_camera_pitch_down_deg"),
        ("wheel_friction_static", "dr_tire_friction_static"),
    ],
)
def test_nominal_values_lie_inside_their_randomization_clamps(nominal_field: str, range_field: str) -> None:
    """A nominal outside its own clamp would make the DR-off and DR-on runs incomparable."""
    nominal = getattr(DUCKIEBOT, nominal_field)
    low, high = getattr(DUCKIEBOT, range_field)
    assert low <= nominal <= high, f"{nominal_field}={nominal} outside {range_field}={low, high}"


def _field_docstrings() -> dict[str, str]:
    """Extract the attribute docstring of every dataclass field by parsing the module source.

    Attribute docstrings are not retained at runtime, so this reads ``params.py`` with :mod:`ast`.

    Returns:
        Mapping from field name to its docstring.
    """
    source = Path(params_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    class_def = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DuckiebotParams"
    )
    docs: dict[str, str] = {}
    body = class_def.body
    for index, node in enumerate(body[:-1]):
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        following = body[index + 1]
        if (
            isinstance(following, ast.Expr)
            and isinstance(following.value, ast.Constant)
            and isinstance(following.value.value, str)
        ):
            docs[node.target.id] = following.value.value
    return docs


def test_every_field_is_documented() -> None:
    """No field may be added without an attribute docstring; that is the provenance record."""
    docs = _field_docstrings()
    declared = {f.name for f in dataclasses.fields(DuckiebotParams)}
    missing = sorted(declared - set(docs))
    assert not missing, f"fields with no attribute docstring: {missing}"


def test_every_measured_field_carries_a_provenance_tag() -> None:
    """Physical constants say where they came from; estimates are thereby visible as estimates."""
    docs = _field_docstrings()
    untagged = sorted(
        name
        for name, doc in docs.items()
        if name not in NAMING_FIELDS
        and not name.startswith("dr_")
        and not any(tag in doc for tag in PROVENANCE_TAGS)
    )
    assert not untagged, f"fields with no [S]/[C]/[M]/[E]/[v2] provenance tag: {untagged}"


def test_every_measured_field_states_its_unit() -> None:
    """Units live in square brackets so a reader never has to guess metres versus millimetres."""
    docs = _field_docstrings()
    unitless = sorted(
        name for name, doc in docs.items() if name not in NAMING_FIELDS and "[" not in doc.split("\n")[0]
    )
    assert not unitless, f"fields whose first docstring line states no unit: {unitless}"


NOT_RANDOMIZED: dict[str, str] = {
    "camera_block_size_m": "visual only: no collider, no inertia, not in the camera's own view",
    "duckie_marker_radius_m": "visual only",
    "duckie_marker_center_base_frame_m": "visual only",
    "chassis_size_m": "fixed geometry; D4 randomizes the mass, CoM and inertia instead",
    "caster_radius_m": "fixed geometry; it is frictionless, so only its height matters",
    "caster_center_base_frame_m": "fixed geometry, pinned to the ground-contact identity",
    "caster_friction": "fixed at zero by design; a swivelling ball caster has no traction",
    "wheel_mass_kg": "D4 randomizes the base body only; the wheels are 4.5% of the total mass",
    "joint_damping": "held fixed so the S8.2 sysid fit of armature and friction stays identifiable",
}
"""Estimated fields that are deliberately not randomized, each with the reason it is exempt."""

# A domain-randomization axis reference: D1 to D18 (dynamics) or V1 to V19 (visual).
_DR_AXIS_PATTERN = re.compile(r"\b[DV]\d{1,2}\b")


def test_every_estimate_is_a_randomization_axis() -> None:
    """The ``[E]`` contract: an estimate that is not randomized is an unexamined assumption.

    An estimated field must name its DR axis or its distribution in its docstring, unless it
    appears in :data:`NOT_RANDOMIZED` with a stated reason. That list is the complete inventory of
    guesses this project has decided to live with, which is exactly the thing a reviewer wants.
    """
    docs = _field_docstrings()
    offenders = []
    for name, doc in docs.items():
        if "[E]" not in doc or name in NOT_RANDOMIZED:
            continue
        if not (_DR_AXIS_PATTERN.search(doc) or "U(" in doc):
            offenders.append(name)
    assert not offenders, f"estimated fields with no randomization: {sorted(offenders)}"


def test_the_not_randomized_exemption_list_is_not_stale() -> None:
    """Every exemption still refers to a real, still-estimated field."""
    docs = _field_docstrings()
    unknown = sorted(set(NOT_RANDOMIZED) - set(docs))
    assert not unknown, f"exemptions for fields that no longer exist: {unknown}"
    not_estimates = sorted(name for name in NOT_RANDOMIZED if "[E]" not in docs[name])
    assert not not_estimates, f"exemptions for fields that are no longer estimates: {not_estimates}"


# ---------------------------------------------------------------------------------------------
# Agreement with the Isaac Lab config views
# ---------------------------------------------------------------------------------------------


def test_robot_cfg_imports_without_isaac_lab() -> None:
    """The import guard works: the module loads and reports availability without raising."""
    assert isinstance(is_isaaclab_available(), bool)


def test_actuator_spec_mirrors_params_exactly() -> None:
    """The ImplicitActuatorCfg numbers are the params numbers, not a second copy."""
    spec = wheel_actuator_spec()
    assert spec["joint_names_expr"] == [DUCKIEBOT.wheel_joint_regex]
    assert spec["effort_limit_sim"] == DUCKIEBOT.wheel_effort_limit_nm
    assert spec["velocity_limit_sim"] == DUCKIEBOT.wheel_velocity_limit_rad_s
    assert spec["stiffness"] == DUCKIEBOT.joint_stiffness == 0.0
    assert spec["damping"] == DUCKIEBOT.joint_damping
    assert spec["armature"] == DUCKIEBOT.joint_armature_kg_m2
    assert spec["friction"] == DUCKIEBOT.joint_friction_nm


def test_spawn_contact_offset_stays_under_the_ground_clearance() -> None:
    """A PhysX contact offset near the 21 mm clearance would fabricate chassis-ground contacts."""
    collision = spawn_property_spec()["collision_props"]
    assert 0.0 < collision["contact_offset"] < DUCKIEBOT.ground_clearance_m / 2.0
    assert collision["rest_offset"] == 0.0
    # It must also exceed the distance a contact point travels in one physics substep, or fast
    # wheels can tunnel through the ground plane.
    substep_travel_m = DUCKIEBOT.top_speed_m_s * DUCKIEBOT.sim_dt_s
    assert collision["contact_offset"] > substep_travel_m


def test_spawn_angular_velocity_cap_clears_the_wheel_limit() -> None:
    """Isaac Lab takes ``max_angular_velocity`` in deg/s; it must not clip the 35 rad/s wheels."""
    rigid = spawn_property_spec()["rigid_props"]
    cap_rad_s = math.radians(rigid["max_angular_velocity"])
    assert cap_rad_s > DUCKIEBOT.wheel_velocity_limit_rad_s
    assert rigid["disable_gravity"] is False


def test_physics_materials_encode_the_frictionless_caster() -> None:
    """Wheels grip with combine max; the caster is frictionless with combine min."""
    materials = physics_material_spec()
    wheel = materials["duckiebot_wheel_material"]
    caster = materials["duckiebot_caster_material"]
    assert wheel["static_friction"] == DUCKIEBOT.wheel_friction_static
    assert wheel["dynamic_friction"] == DUCKIEBOT.wheel_friction_dynamic
    assert wheel["dynamic_friction"] <= wheel["static_friction"]
    assert wheel["friction_combine_mode"] == "max"
    assert caster["static_friction"] == caster["dynamic_friction"] == 0.0
    assert caster["friction_combine_mode"] == "min"
    assert set(wheel["bind_to"]) == {
        DUCKIEBOT.left_wheel_link_name,
        DUCKIEBOT.right_wheel_link_name,
    }


def test_camera_mount_spec_is_the_single_source_of_the_mount_pose() -> None:
    """The env's TiledCameraCfg reads the params values, in the ROS convention, unmodified."""
    spec = camera_mount_spec()
    assert spec["offset_pos"] == DUCKIEBOT.camera_pos_base_frame_m
    assert spec["pitch_down_deg"] == DUCKIEBOT.camera_pitch_down_deg
    assert spec["convention"] == "ros"
    assert (spec["width"], spec["height"]) == (192, 128)
    assert spec["focal_length"] == DUCKIEBOT.camera_focal_length_mm
    assert spec["clipping_range"] == DUCKIEBOT.camera_clipping_range_m
    assert DUCKIEBOT.base_link_name in spec["parent_prim_path"]
