"""Single source of truth for every Duckiebot physical constant (SPEC v2, S2 and S3.2).

This module is deliberately dependency-free (stdlib only) so that it can be imported by:

* the URDF generator (:mod:`duckiebot_rl.assets.urdf`),
* the Isaac Lab articulation config (:mod:`duckiebot_rl.assets.robot_cfg`),
* the MuJoCo MJCF generator in ``sim2sim/`` (which runs in a venv without torch or Isaac),
* the deployment sidecar writer, and
* the documentation build.

Every field carries its unit and a provenance tag in its attribute docstring:

============  ==========================================================================
Tag           Meaning
============  ==========================================================================
``[S]``       Manufacturer spec sheet / published Duckietown dimensional fact.
``[C]``       Official Duckietown code default (dt-core / duckietown_msgs).
``[M]``       Measured or decomposed from primary artifacts.
``[E]``       Estimate. Every ``[E]`` value MUST also be a domain-randomization axis.
``[v2]``      Introduced or corrected by SPEC v2 relative to the v1 architecture doc.
============  ==========================================================================

Frames are REP-103 (x forward, y left, z up). The ``base_link`` origin sits at the
wheel-axle midpoint, :attr:`DuckiebotParams.base_link_height_m` above the ground when level.

Clean-room note: none of these numbers are copied from gym-duckietown source or assets.
Dimensions are facts and are not copyrightable expression; the geometry that realizes them in
:mod:`duckiebot_rl.assets.urdf` is authored from primitives only.

The four contradictions the v1 critique flagged are resolved here and are re-checked at import
time by :meth:`DuckiebotParams.__post_init__`:

1. Caster radius (0.021 in prose vs 0.0318 in the URDF) becomes 0.0165 m with the sphere centre
   at ``z = -0.0153`` in base frame, so the contact point is exactly on the ground plane.
2. Chassis ground clearance (6.3 mm actual vs 21 mm stated) is fixed by moving the collision-box
   centre to ``z = +0.0267``, putting its underside at exactly 21 mm.
3. Wheel effort limit (2.0 N.m, roughly 13x a DG01D 48:1 stall torque) becomes 0.15 N.m.
4. Camera mount (the URDF ``camera_link`` held the gym-duckietown values the text rejected):
   there is no ``camera_link`` at all; the mount pose lives here and only here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

__all__ = ["DUCKIEBOT", "DuckiebotParams", "ParameterConsistencyError"]


class ParameterConsistencyError(ValueError):
    """Raised when a :class:`DuckiebotParams` instance is internally contradictory."""


# Absolute tolerance for the geometric closure checks, in metres. One micrometre: far below any
# manufacturing tolerance, far above float64 round-off at these magnitudes.
_GEOM_TOL_M = 1.0e-6
# Absolute tolerance for the angular / intrinsic closure checks, in degrees.
_ANGLE_TOL_DEG = 0.02


@dataclass(frozen=True)
class DuckiebotParams:
    """Every physical constant of the clean-room Duckiebot, frozen and self-checking.

    The dataclass is frozen so no consumer can mutate the shared singleton :data:`DUCKIEBOT`.
    Domain randomization never mutates these values; it samples around them, and the sampled
    values live in the environment's per-env state tensors.

    Fields named ``dr_*`` are the SPEC v2 S7.3 (dynamics, ``D*``) and S7.2 (visual, ``V*``) clamp
    endpoints, stored as ``(low, high)`` tuples in the same unit as the nominal field.

    Raises:
        ParameterConsistencyError: If the geometry, mass, actuation or camera numbers contradict
            one another (for example if the caster sphere would not touch the ground plane).
    """

    # ---------------------------------------------------------------------------------------
    # Wheels and drivetrain geometry
    # ---------------------------------------------------------------------------------------
    wheel_radius_m: float = 0.0318
    """Wheel rolling radius [m]. [C]/[S] DR axis D1 (x U(0.95, 1.05), L/R asym U(-6%, +6%))."""

    wheel_baseline_m: float = 0.100
    """Distance between the two wheel contact patches, i.e. the track [m]. [C] DR axis D2."""

    wheel_width_m: float = 0.027
    """Visual wheel cylinder length along the axle [m]. [M] The collider is a sphere, not this."""

    base_link_height_m: float = 0.0318
    """Height of ``base_link`` (the axle midpoint) above the ground when level [m]. [S]

    Equal to :attr:`wheel_radius_m` by construction: the robot rides on its wheels.
    """

    # ---------------------------------------------------------------------------------------
    # Chassis body
    # ---------------------------------------------------------------------------------------
    chassis_size_m: tuple[float, float, float] = (0.180, 0.130, 0.075)
    """Chassis box full extents (length x width x height) [m]. [S]/[E]"""

    chassis_center_base_frame_m: tuple[float, float, float] = (-0.015, 0.0, 0.0267)
    """Chassis box centre in ``base_link`` frame [m]. [v2]

    ``z = +0.0267`` places the underside at :attr:`ground_clearance_m` (21 mm). The v1 value of
    ``+0.012`` gave 6.3 mm and produced phantom chassis-ground contacts under the D15 tile tilt
    and the D14 push randomization.
    """

    ground_clearance_m: float = 0.021
    """Clearance between the chassis underside and the ground when level [m]. [S]"""

    robot_width_m: float = 0.131
    """Overall robot width used by the lane-departure gates, symbol ``W_R`` [m]. [M]

    Slightly wider than both the chassis (0.130) and the wheel envelope
    (:attr:`wheel_baseline_m` + :attr:`wheel_width_m` = 0.127) because it includes the tyre
    sidewall bulge. Consumed by the S5.4 reward and the S5.5 wrong-lane termination.
    """

    # ---------------------------------------------------------------------------------------
    # Caster (merged into base_link: no joint, no inertial tag, frictionless material)
    # ---------------------------------------------------------------------------------------
    caster_radius_m: float = 0.0165
    """Rear caster ball radius [m]. [E] Resolves the v1 0.021-vs-0.0318 contradiction."""

    caster_center_base_frame_m: tuple[float, float, float] = (-0.085, 0.0, -0.0153)
    """Caster sphere centre in ``base_link`` frame [m]. [E]

    ``base_link_height_m + z == caster_radius_m`` exactly, so the contact point is at ``z = 0``
    and the chassis sits level on three points without pre-loading the wheels.
    """

    caster_friction: float = 0.0
    """Static and dynamic friction coefficient of the caster physics material [-]. [E]

    Frictionless with combine mode ``min``: a real ball caster swivels freely, and a
    high-friction rear contact on a differential drive fights every turn.
    """

    wheel_friction_static: float = 1.0
    """Wheel physics-material static friction [-]. [E] DR axis D3, mu_s U(0.40, 1.40)."""

    wheel_friction_dynamic: float = 1.0
    """Wheel physics-material dynamic friction [-]. [E] DR axis D3, mu_d = mu_s x U(0.7, 1.0)."""

    # ---------------------------------------------------------------------------------------
    # Decorative primitives (visual only: no collider, no inertia)
    # ---------------------------------------------------------------------------------------
    camera_block_size_m: tuple[float, float, float] = (0.030, 0.030, 0.030)
    """Visual-only camera housing cube full extents [m]. [E] Authored primitive, never a mesh."""

    duckie_marker_radius_m: float = 0.025
    """Visual-only yellow marker sphere radius on the chassis roof [m]. [E]"""

    duckie_marker_center_base_frame_m: tuple[float, float, float] = (-0.030, 0.0, 0.090)
    """Yellow marker sphere centre in ``base_link`` frame [m]. [E]"""

    # ---------------------------------------------------------------------------------------
    # Mass and inertia
    # ---------------------------------------------------------------------------------------
    base_mass_kg: float = 1.000
    """Mass of the chassis body: battery, Jetson, hat, shell [kg]. [E] DR D4 U(0.85, 1.40)."""

    wheel_mass_kg: float = 0.050
    """Mass of one wheel link [kg]. [E]"""

    base_com_base_frame_m: tuple[float, float, float] = (-0.015, 0.0, 0.015)
    """Chassis centre of mass in ``base_link`` frame [m]. [E]

    Below the box centre because the battery sits low. DR D4 perturbs it by +/-20 mm in xy and
    +/-10 mm in z.
    """

    # ---------------------------------------------------------------------------------------
    # Actuation limits and joint dynamics (the URDF and the ActuatorCfg both read these)
    # ---------------------------------------------------------------------------------------
    wheel_effort_limit_nm: float = 0.15
    """Per-wheel torque limit [N.m]. [M]/[E] DR axis D17 U(0.06, 0.25).

    A DG01D 48:1 brushed gearmotor stalls near 0.08-0.18 N.m. At :attr:`wheel_radius_m` this is
    4.7 N per wheel and 9.4 N total, i.e. 0.87 g of tractive acceleration on the 1.10 kg robot.
    v1's 2.0 N.m implied 11.7 g and invalidated every downstream acceleration and braking number.
    """

    wheel_velocity_limit_rad_s: float = 35.0
    """Per-wheel angular velocity limit [rad/s]. [C]/[E]

    :attr:`motor_constant_k_rad_s_per_duty` gives 27 rad/s at duty 1, and the nominal command
    envelope needs :attr:`nominal_max_wheel_speed_rad_s` = 25.2 rad/s, so this leaves 39% headroom.
    It does NOT cover the extreme corner of the D1/D2/D5 randomization (small radius, wide
    baseline, gain 1.40, trim +0.12 asks for about 45 rad/s); that corner saturates, which is the
    intended behaviour because a real motor saturates there too.
    """

    joint_stiffness: float = 0.0
    """Implicit-drive position gain [N.m/rad]. [v2] Zero: the wheels are velocity controlled."""

    joint_damping: float = 0.05
    """Implicit-drive velocity gain [N.m.s/rad]. [E] A sysid stage-2 fit target, NOT a DR axis.

    Deliberately the one estimated dynamics quantity that is not randomized. Damping, armature and
    joint friction are near-degenerate against each other in an open-loop spin-down fit, so the
    S8.2 procedure holds damping fixed and fits the other two. Randomizing all three would make
    that fit unidentifiable rather than robust.
    """

    joint_armature_kg_m2: float = 2.0e-4
    """Reflected rotor inertia at the wheel [kg.m^2]. [E] DR x U(0.5, 2.0).

    Rotor inertia multiplied by the 48:1 gear ratio squared. Non-zero so the MuJoCo sysid
    stage-2 fit has a real Isaac-side counterpart to match; v1 left it at 0.0.
    """

    joint_friction_nm: float = 0.010
    """Joint Coulomb friction [N.m]. [E] DR U(0.005, 0.03). Also a sysid stage-2 fit target."""

    # ---------------------------------------------------------------------------------------
    # Motor / command model (S5.3 action path)
    # ---------------------------------------------------------------------------------------
    motor_constant_k_rad_s_per_duty: float = 27.0
    """Wheel speed produced per unit PWM duty [rad/s per duty]. [C]"""

    motor_gain_nominal: float = 1.0
    """Nominal multiplicative motor gain [-]. [C] DR D5 U(0.60, 1.40)."""

    motor_trim_nominal: float = 0.0
    """Nominal left/right motor trim [-]. [C] DR D5 U(-0.12, +0.12)."""

    top_speed_m_s: float = 0.859
    """Open-loop model top speed at duty 1 [m/s]. [C]/[M] Equals k * r."""

    v_cmd_max_m_s: float = 0.6
    """Maximum commanded forward speed [m/s]. [v2] Action ``a_v`` maps to ``[0, 0.6]``."""

    omega_cmd_max_rad_s: float = 4.0
    """Maximum commanded yaw rate [rad/s]. [v2] Action ``a_om`` maps to ``[-4, +4]``."""

    omega_robot_clamp_rad_s: float = 8.0
    """Yaw-rate clamp enforced by the real robot's kinematics node [rad/s]. [C]"""

    pwm_release_duty: float = 0.01
    """Duty below which the driver releases the H-bridge [-]. [C] Below this the wheel COASTS."""

    pwm_first_nonzero_duty: float = 0.235
    """Smallest duty that actually turns the wheel on hardware [-]. [C] DR D7 U(0, 0.15)."""

    brake_dw_max_rad_s_per_step: float = 12.0
    """Deceleration slew cap per control step [rad/s]. [E] Scaled by D18 beta U(0.4, 1.0).

    Hardware coasts at duty 0 (back-EMF braking only), so an implicit velocity drive commanded to
    zero would brake far harder than the real robot can. v1 called ``a_v = -1`` a full brake.
    """

    actuation_delay_s: float = 0.150
    """Modelled command-to-motion latency [s]. [C]/[E] DR D8 U(1, 3) control steps + sub-step."""

    encoder_ticks_per_rev: int = 135
    """Wheel encoder resolution [ticks/rev]. [S]/[C] The vec observation is quantized to this."""

    # ---------------------------------------------------------------------------------------
    # Control rates (S5.2; kept here because the delay model needs the control period)
    # ---------------------------------------------------------------------------------------
    sim_dt_s: float = 1.0 / 240.0
    """Physics timestep [s]. [v2] Identical in Isaac and MuJoCo so integration is not a confound."""

    decimation: int = 16
    """Physics steps per control step [-]. [v2] 240 / 16 = 15 Hz, the deployment rate."""

    # ---------------------------------------------------------------------------------------
    # Camera mount: the ONLY source. There is no camera_link in the URDF.
    # ---------------------------------------------------------------------------------------
    camera_pos_base_frame_m: tuple[float, float, float] = (0.078, 0.0, 0.0692)
    """Camera optical centre in ``base_link`` frame [m]. [M]/[E] DR axis V10."""

    camera_height_m: float = 0.101
    """Camera optical centre height above ground when level [m]. [M]/[E] DR axis V10.

    Randomized over U(0.090, 0.120) m; the environment applies it as a base-frame z of
    ``height - base_link_height_m``.
    """

    camera_pitch_down_deg: float = 25.3
    """Camera downward pitch, stored as a POSITIVE scalar [deg]. [M]/[E] DR axis V10 U(15, 28).

    Only ``duckiebot_rl.camera_math.quat_cam_ros`` may turn this into a rotation. Golden values
    that helper must return, as ``(w, x, y, z)``: pitch 0 gives ``(0.5, -0.5, 0.5, -0.5)``;
    pitch 25.3 deg gives ``(0.37837, -0.59736, 0.59736, -0.37837)``.
    """

    # ---------------------------------------------------------------------------------------
    # Canonical trained camera (S4.1): one square-pixel pinhole for Isaac, MuJoCo and the robot
    # ---------------------------------------------------------------------------------------
    render_width_px: int = 192
    """Canonical render width [px]. [v2] A 2x supersample of the 96 px observation width."""

    render_height_px: int = 128
    """Canonical render height [px]. [v2]"""

    camera_focal_px: float = 65.98
    """Canonical focal length, identical in x and y because pixels are square [px]. [v2]"""

    camera_hfov_deg: float = 111.0
    """Canonical horizontal field of view [deg]. [v2] Matches the dt-core rectified hFOV."""

    camera_vfov_deg: float = 88.26
    """Canonical vertical field of view [deg]. [v2] S2 quotes this rounded to 88.3."""

    camera_focal_length_mm: float = 7.201
    """USD ``focalLength`` for ``PinholeCameraCfg`` [mm]. [v2]

    Authored directly. ``from_intrinsic_matrix`` is FORBIDDEN here: Isaac Sim 5.1 averages fx/fy
    and forces the aperture ratio to the render aspect.
    """

    camera_horizontal_aperture_mm: float = 20.955
    """USD ``horizontalAperture`` [mm]. [v2]"""

    camera_vertical_aperture_mm: float = 13.970
    """USD ``verticalAperture`` [mm]. [v2]

    Equals ``horizontal_aperture * H / W``, so the rendered geometry is invariant to whether
    Isaac Sim 5.1 honours this value or recomputes it from the render aspect.
    """

    camera_clipping_range_m: tuple[float, float] = (0.05, 6.0)
    """Near and far clip planes [m]. [v2]

    Far 6.0 m with 0.30 m walls stops a robot seeing the neighbouring city across the 8.0 m
    ``env_spacing``. Near 0.05 m is below the 0.11 m ray to the nearest visible ground point.
    """

    # ---------------------------------------------------------------------------------------
    # Raw physical camera: used ONLY by the C8 fisheye eval and the robot rectification docs
    # ---------------------------------------------------------------------------------------
    raw_resolution_px: tuple[int, int] = (640, 480)
    """Native IMX219 capture resolution used by the Duckiebot camera node [px]. [S]"""

    raw_fps: int = 30
    """Native capture rate [Hz]. [S]"""

    raw_intrinsics_fx_fy_cx_cy: tuple[float, float, float, float] = (305.57, 308.83, 303.08, 231.88)
    """Example measured raw intrinsics (fx, fy, cx, cy) [px]. [M] Per-robot in reality."""

    raw_distortion_plumb_bob: tuple[float, float, float, float, float] = (
        -0.2,
        0.0305,
        5.86e-4,
        -6.70e-4,
        0.0,
    )
    """Example plumb-bob distortion coefficients (k1, k2, p1, p2, k3) [-]. [M]"""

    # ---------------------------------------------------------------------------------------
    # Domain-randomization clamp endpoints referenced by the fields above (S7.2 / S7.3)
    # ---------------------------------------------------------------------------------------
    dr_wheel_radius_scale: tuple[float, float] = (0.95, 1.05)
    """D1 wheel-radius multiplier range [-]."""

    dr_wheel_radius_asymmetry: tuple[float, float] = (-0.06, 0.06)
    """D1 left/right wheel-radius asymmetry range [-]."""

    dr_baseline_m: tuple[float, float] = (0.090, 0.110)
    """D2 wheel baseline range [m]."""

    dr_tire_friction_static: tuple[float, float] = (0.40, 1.40)
    """D3 static friction range [-]."""

    dr_tire_friction_dynamic_scale: tuple[float, float] = (0.7, 1.0)
    """D3 mu_d / mu_s ratio range [-]. Enforces kinetic <= static."""

    dr_base_mass_kg: tuple[float, float] = (0.85, 1.40)
    """D4 chassis mass range [kg]."""

    dr_com_xy_m: tuple[float, float] = (-0.020, 0.020)
    """D4 centre-of-mass xy perturbation range [m]."""

    dr_com_z_m: tuple[float, float] = (-0.010, 0.010)
    """D4 centre-of-mass z perturbation range [m]."""

    dr_motor_gain: tuple[float, float] = (0.60, 1.40)
    """D5 motor gain range [-]."""

    dr_motor_trim: tuple[float, float] = (-0.12, 0.12)
    """D5 motor trim range [-]."""

    dr_battery_sag: tuple[float, float] = (0.90, 1.04)
    """D6 battery-sag multiplier range [-]."""

    dr_deadband_duty: tuple[float, float] = (0.0, 0.15)
    """D7 PWM dead-band range [duty]. Sampled independently per wheel."""

    dr_delay_control_steps: tuple[int, int] = (1, 3)
    """D8 actuation delay range [control steps], inclusive."""

    dr_delay_substep: tuple[float, float] = (0.0, 0.9)
    """D8 sub-step interpolation fraction range [-]."""

    dr_motor_lag_alpha: tuple[float, float] = (0.4, 0.9)
    """D8 first-order motor-lag coefficient range [-]."""

    dr_obs_latency_steps: tuple[int, int] = (0, 2)
    """D9 camera-stream observation latency range [control steps], inclusive."""

    dr_effort_limit_nm: tuple[float, float] = (0.06, 0.25)
    """D17 per-wheel effort-limit range [N.m]. Fixed per env at startup, never per episode."""

    dr_brake_authority_beta: tuple[float, float] = (0.4, 1.0)
    """D18 brake-authority range [-]."""

    dr_armature_scale: tuple[float, float] = (0.5, 2.0)
    """Armature multiplier range [-]."""

    dr_joint_friction_nm: tuple[float, float] = (0.005, 0.03)
    """Joint Coulomb friction range [N.m]."""

    dr_camera_height_m: tuple[float, float] = (0.090, 0.120)
    """V10 camera height above ground range [m]."""

    dr_camera_pitch_down_deg: tuple[float, float] = (15.0, 28.0)
    """V10 camera downward pitch range [deg]."""

    dr_camera_forward_m: tuple[float, float] = (0.058, 0.085)
    """V10 camera forward offset range [m], in ``base_link`` frame."""

    dr_camera_yaw_roll_deg: tuple[float, float] = (-3.0, 3.0)
    """V10 camera yaw and roll perturbation range [deg]."""

    # ---------------------------------------------------------------------------------------
    # Naming: kept here so the URDF, the actuator regex and the tests cannot drift apart
    # ---------------------------------------------------------------------------------------
    base_link_name: str = "base_link"
    """Name of the chassis link in the URDF and in the imported USD."""

    left_wheel_link_name: str = "left_wheel_link"
    """Name of the left wheel link."""

    right_wheel_link_name: str = "right_wheel_link"
    """Name of the right wheel link."""

    left_wheel_joint_name: str = "left_wheel_joint"
    """Name of the left wheel joint."""

    right_wheel_joint_name: str = "right_wheel_joint"
    """Name of the right wheel joint."""

    wheel_joint_regex: str = ".*_wheel_joint"
    """Regex matching both wheel joints, for ``ImplicitActuatorCfg.joint_names_expr``."""

    # =======================================================================================
    # Derived quantities
    # =======================================================================================

    @property
    def total_mass_kg(self) -> float:
        """Assembled robot mass [kg]: chassis plus both wheels."""
        return self.base_mass_kg + 2.0 * self.wheel_mass_kg

    @property
    def half_baseline_m(self) -> float:
        """Lateral offset of each wheel joint from ``base_link`` [m]."""
        return 0.5 * self.wheel_baseline_m

    @property
    def left_wheel_origin_m(self) -> tuple[float, float, float]:
        """Left wheel joint origin in ``base_link`` frame [m]."""
        return (0.0, self.half_baseline_m, 0.0)

    @property
    def right_wheel_origin_m(self) -> tuple[float, float, float]:
        """Right wheel joint origin in ``base_link`` frame [m]."""
        return (0.0, -self.half_baseline_m, 0.0)

    @property
    def chassis_bottom_height_m(self) -> float:
        """Height of the chassis-box underside above the ground when level [m]."""
        return self.base_link_height_m + self.chassis_center_base_frame_m[2] - 0.5 * self.chassis_size_m[2]

    @property
    def caster_contact_height_m(self) -> float:
        """Height of the caster's lowest point above the ground when level [m]. Must be 0."""
        return self.base_link_height_m + self.caster_center_base_frame_m[2] - self.caster_radius_m

    @property
    def control_dt_s(self) -> float:
        """Control period [s]: ``sim_dt_s * decimation``."""
        return self.sim_dt_s * self.decimation

    @property
    def control_hz(self) -> float:
        """Control frequency [Hz]."""
        return 1.0 / self.control_dt_s

    @property
    def actuation_delay_steps(self) -> float:
        """Modelled actuation delay expressed in control steps [-]."""
        return self.actuation_delay_s / self.control_dt_s

    @property
    def nominal_max_wheel_speed_rad_s(self) -> float:
        """Largest wheel speed the nominal command envelope can ask for [rad/s].

        Differential-drive inverse kinematics at the action-space corner
        ``v = v_cmd_max, |omega| = omega_cmd_max``, evaluated with nominal radius and baseline.

        Returns:
            Wheel angular speed in rad/s.
        """
        v_wheel = self.v_cmd_max_m_s + 0.5 * self.omega_cmd_max_rad_s * self.wheel_baseline_m
        return v_wheel / self.wheel_radius_m

    @property
    def max_tractive_force_n(self) -> float:
        """Total tractive force at the effort limit [N]: ``2 * tau / r``."""
        return 2.0 * self.wheel_effort_limit_nm / self.wheel_radius_m

    @property
    def max_tractive_accel_g(self) -> float:
        """Tractive acceleration at the effort limit, in units of g [-]."""
        return self.max_tractive_force_n / (self.total_mass_kg * 9.80665)

    @property
    def base_inertia_about_com(self) -> tuple[float, float, float]:
        """Chassis principal moments about its own centre of mass [kg.m^2].

        Solid-box formulas applied to :attr:`chassis_size_m` at :attr:`base_mass_kg`. Products of
        inertia are zero because the box is axis aligned.

        Returns:
            ``(Ixx, Iyy, Izz)`` in kg.m^2.
        """
        sx, sy, sz = self.chassis_size_m
        m = self.base_mass_kg
        return (
            m * (sy * sy + sz * sz) / 12.0,
            m * (sx * sx + sz * sz) / 12.0,
            m * (sx * sx + sy * sy) / 12.0,
        )

    @property
    def wheel_inertia_about_com(self) -> tuple[float, float, float]:
        """Wheel principal moments about its centre of mass [kg.m^2].

        Solid cylinder of radius :attr:`wheel_radius_m` and length :attr:`wheel_width_m`, spinning
        about the body y axis (the axle).

        Returns:
            ``(Ixx, Iyy, Izz)`` in kg.m^2, with ``Iyy`` the axial (spin) moment.
        """
        m = self.wheel_mass_kg
        r = self.wheel_radius_m
        h = self.wheel_width_m
        i_axial = 0.5 * m * r * r
        i_transverse = m * (3.0 * r * r + h * h) / 12.0
        return (i_transverse, i_axial, i_transverse)

    @property
    def canonical_intrinsic_matrix(self) -> tuple[tuple[float, float, float], ...]:
        """The canonical pinhole camera matrix ``K_canon`` [px].

        The robot rectifies directly to this matrix via ``cv2.initUndistortRectifyMap``; the
        simulator authors a camera that produces it. Sim and robot geometry are therefore
        identical by construction rather than by calibration.

        Returns:
            A 3x3 row-major tuple of tuples.
        """
        f = self.camera_focal_px
        return (
            (f, 0.0, self.render_width_px / 2.0),
            (0.0, f, self.render_height_px / 2.0),
            (0.0, 0.0, 1.0),
        )

    # =======================================================================================
    # Validation
    # =======================================================================================

    def __post_init__(self) -> None:
        """Re-check every cross-field identity that the v1 critique found broken.

        Raises:
            ParameterConsistencyError: On the first identity that does not hold.
        """
        self._check_geometry()
        self._check_mass_and_inertia()
        self._check_actuation()
        self._check_camera()

    def _fail(self, message: str) -> None:
        """Raise a consistency error.

        Args:
            message: Human-readable description of the violated identity.

        Raises:
            ParameterConsistencyError: Always.
        """
        raise ParameterConsistencyError(message)

    def _check_geometry(self) -> None:
        """Validate the standing geometry: level chassis, caster contact, ground clearance."""
        if abs(self.base_link_height_m - self.wheel_radius_m) > _GEOM_TOL_M:
            self._fail(
                f"base_link height {self.base_link_height_m} m must equal the wheel radius "
                f"{self.wheel_radius_m} m: the robot rides on its wheels."
            )
        if abs(self.caster_contact_height_m) > _GEOM_TOL_M:
            self._fail(
                f"caster lowest point is at z = {self.caster_contact_height_m:+.6f} m, not 0. The "
                "centre height must equal the caster radius or the chassis will not sit level."
            )
        if not 0.0 < self.caster_radius_m < self.wheel_radius_m:
            self._fail(
                f"caster radius {self.caster_radius_m} m must be positive and smaller than the "
                f"wheel radius {self.wheel_radius_m} m: a level chassis cannot rest on a larger "
                "rear ball."
            )
        if abs(self.chassis_bottom_height_m - self.ground_clearance_m) > _GEOM_TOL_M:
            self._fail(
                f"chassis underside is at {self.chassis_bottom_height_m:.6f} m but "
                f"ground_clearance_m says {self.ground_clearance_m} m."
            )
        if self.ground_clearance_m <= 0.5 * self.caster_radius_m:
            self._fail("ground clearance is implausibly small relative to the caster radius.")
        if self.wheel_width_m >= 2.0 * self.wheel_radius_m:
            self._fail("wheel width exceeds the wheel diameter; that cylinder is not a wheel.")
        wheel_envelope = self.wheel_baseline_m + self.wheel_width_m
        widest = max(wheel_envelope, self.chassis_size_m[1])
        if not widest - _GEOM_TOL_M <= self.robot_width_m <= widest + 0.010:
            self._fail(
                f"robot_width_m {self.robot_width_m} m must bound the widest part "
                f"({widest:.3f} m: chassis {self.chassis_size_m[1]} m, wheels "
                f"{wheel_envelope:.3f} m) by no more than 10 mm."
            )
        cx, _, cz = self.chassis_center_base_frame_m
        sx, _, sz = self.chassis_size_m
        caster_x = self.caster_center_base_frame_m[0]
        if caster_x >= 0.0:
            self._fail("the caster must sit behind the wheel axle, otherwise the robot tips back.")
        if not cx - 0.5 * sx - self.caster_radius_m <= caster_x <= cx:
            self._fail(
                f"the caster centre x = {caster_x} m must lie under the rear half of the chassis "
                f"box (which spans {cx - 0.5 * sx:.4f} to {cx + 0.5 * sx:.4f} m)."
            )
        cam_x, _, cam_z = self.camera_pos_base_frame_m
        if cam_x < cx + 0.5 * sx - 0.5 * self.camera_block_size_m[0]:
            self._fail("the camera must be mounted at or ahead of the chassis front face.")
        if cam_z <= cz + 0.5 * sz - self.camera_block_size_m[2]:
            self._fail("the camera must sit at the top of the chassis, not inside it.")

    def _check_mass_and_inertia(self) -> None:
        """Validate masses and the analytic inertia tensors."""
        if abs(self.total_mass_kg - 1.10) > 1e-9:
            self._fail(f"assembled mass {self.total_mass_kg} kg does not match the S2 figure of 1.10 kg.")
        if self.base_mass_kg <= 0.0 or self.wheel_mass_kg <= 0.0:
            self._fail("masses must be strictly positive.")
        lo, hi = self.dr_base_mass_kg
        if not lo <= self.base_mass_kg <= hi:
            self._fail(f"nominal base mass {self.base_mass_kg} kg lies outside its DR clamp ({lo}, {hi}).")
        for name, inertia in (
            ("chassis", self.base_inertia_about_com),
            ("wheel", self.wheel_inertia_about_com),
        ):
            ixx, iyy, izz = inertia
            if min(ixx, iyy, izz) <= 0.0:
                self._fail(f"{name} inertia is not positive definite: {inertia}.")
            if ixx + iyy < izz or iyy + izz < ixx or izz + ixx < iyy:
                self._fail(f"{name} principal moments violate the triangle inequality: {inertia}.")
        com_x, com_y, com_z = self.base_com_base_frame_m
        cx, cy, cz = self.chassis_center_base_frame_m
        sx, sy, sz = self.chassis_size_m
        inside = abs(com_x - cx) <= 0.5 * sx and abs(com_y - cy) <= 0.5 * sy and abs(com_z - cz) <= 0.5 * sz
        if not inside:
            self._fail(f"the centre of mass {self.base_com_base_frame_m} lies outside the chassis box.")

    def _check_actuation(self) -> None:
        """Validate the motor, limit and delay numbers against each other."""
        lo, hi = self.dr_effort_limit_nm
        if not lo <= self.wheel_effort_limit_nm <= hi:
            self._fail(f"effort limit {self.wheel_effort_limit_nm} N.m is outside its DR clamp ({lo}, {hi}).")
        if self.max_tractive_accel_g > 3.0:
            self._fail(
                f"the effort limit implies {self.max_tractive_accel_g:.2f} g of tractive "
                "acceleration; a DG01D 48:1 gearmotor cannot do that (the v1 2.0 N.m failure)."
            )
        k_speed = self.motor_constant_k_rad_s_per_duty
        if self.wheel_velocity_limit_rad_s < self.nominal_max_wheel_speed_rad_s:
            self._fail(
                f"velocity limit {self.wheel_velocity_limit_rad_s} rad/s is below the nominal "
                f"command envelope {self.nominal_max_wheel_speed_rad_s:.1f} rad/s; the policy could "
                "not reach its own commanded speed even with DR off."
            )
        if self.wheel_velocity_limit_rad_s < k_speed:
            self._fail(
                f"velocity limit {self.wheel_velocity_limit_rad_s} rad/s is below the open-loop "
                f"duty-1 speed {k_speed} rad/s."
            )
        if abs(self.top_speed_m_s - k_speed * self.wheel_radius_m) > 1e-3:
            self._fail(
                f"top speed {self.top_speed_m_s} m/s disagrees with k * r = "
                f"{k_speed * self.wheel_radius_m:.4f} m/s."
            )
        if self.v_cmd_max_m_s >= self.top_speed_m_s:
            self._fail("the commanded speed cap must stay below the open-loop top speed.")
        if self.omega_cmd_max_rad_s >= self.omega_robot_clamp_rad_s:
            self._fail("the commanded yaw rate must stay inside the robot's own kinematics clamp.")
        if not 0.0 < self.pwm_release_duty < self.pwm_first_nonzero_duty < 1.0:
            self._fail("the PWM release duty must be below the first duty that moves the wheel.")
        if self.joint_stiffness != 0.0:
            self._fail("the wheel drives are velocity controlled; stiffness must be exactly 0.")
        if self.joint_damping <= 0.0 or self.joint_armature_kg_m2 <= 0.0 or self.joint_friction_nm <= 0.0:
            self._fail(
                "damping, armature and joint friction must all be strictly positive so the MuJoCo "
                "sysid stage-2 fit has an Isaac-side counterpart; v1 left them at 0."
            )
        d_lo, d_hi = self.dr_delay_control_steps
        if not d_lo <= self.actuation_delay_steps <= d_hi:
            self._fail(
                f"the modelled delay of {self.actuation_delay_steps:.2f} control steps is outside "
                f"the D8 clamp ({d_lo}, {d_hi})."
            )
        if abs(self.control_hz - 15.0) > 1e-9:
            self._fail(f"control rate {self.control_hz} Hz must be the 15 Hz deployment rate.")
        if self.encoder_ticks_per_rev <= 0:
            self._fail("encoder resolution must be positive.")

    def _check_camera(self) -> None:
        """Validate the canonical pinhole against its USD authoring parameters."""
        mount_height = self.base_link_height_m + self.camera_pos_base_frame_m[2]
        if abs(mount_height - self.camera_height_m) > _GEOM_TOL_M:
            self._fail(
                f"camera height {self.camera_height_m} m disagrees with base height + mount z = "
                f"{mount_height:.6f} m."
            )
        h_lo, h_hi = self.dr_camera_height_m
        if not h_lo <= self.camera_height_m <= h_hi:
            self._fail(f"nominal camera height is outside its V10 clamp ({h_lo}, {h_hi}).")
        p_lo, p_hi = self.dr_camera_pitch_down_deg
        if not p_lo <= self.camera_pitch_down_deg <= p_hi:
            self._fail(f"nominal camera pitch is outside its V10 clamp ({p_lo}, {p_hi}).")
        f_lo, f_hi = self.dr_camera_forward_m
        if not f_lo <= self.camera_pos_base_frame_m[0] <= f_hi:
            self._fail(f"nominal camera forward offset is outside its V10 clamp ({f_lo}, {f_hi}).")
        if (self.render_width_px, self.render_height_px) != (192, 128):
            self._fail(
                "the canonical render resolution must stay 192 x 128; the S4.2 angular-scale table "
                "and the S5.6 VRAM budget are both computed at that resolution."
            )
        f_from_hfov = (0.5 * self.render_width_px) / math.tan(math.radians(0.5 * self.camera_hfov_deg))
        if abs(f_from_hfov - self.camera_focal_px) > 0.01:
            self._fail(
                f"focal length {self.camera_focal_px} px disagrees with the stated hFOV "
                f"({f_from_hfov:.3f} px)."
            )
        vfov = 2.0 * math.degrees(math.atan((0.5 * self.render_height_px) / self.camera_focal_px))
        if abs(vfov - self.camera_vfov_deg) > _ANGLE_TOL_DEG:
            self._fail(
                f"vFOV {self.camera_vfov_deg} deg disagrees with the square-pixel value {vfov:.3f} deg."
            )
        f_from_usd = self.camera_focal_length_mm * self.render_width_px / self.camera_horizontal_aperture_mm
        if abs(f_from_usd - self.camera_focal_px) > 0.01:
            self._fail(
                f"USD focalLength / horizontalAperture give {f_from_usd:.3f} px, not "
                f"{self.camera_focal_px} px."
            )
        expected_v_aperture = (
            self.camera_horizontal_aperture_mm * self.render_height_px / self.render_width_px
        )
        if abs(expected_v_aperture - self.camera_vertical_aperture_mm) > 1e-3:
            self._fail(
                "the aperture ratio must equal the render aspect, otherwise the geometry depends "
                "on whether Isaac Sim recomputes verticalAperture."
            )
        near, far = self.camera_clipping_range_m
        if not 0.0 < near < far:
            self._fail(f"invalid clipping range {self.camera_clipping_range_m}.")
        if far > 7.5:
            self._fail(
                "the far clip must stay below env_spacing (8.0 m) so neighbouring cities stay out of view."
            )

    # =======================================================================================
    # Reporting
    # =======================================================================================

    def as_dict(self) -> dict[str, object]:
        """Return every stored field as a plain dictionary.

        Derived properties are not included; recompute them from the returned values.

        Returns:
            Mapping of field name to value, in declaration order.
        """
        return {f.name: getattr(self, f.name) for f in fields(self)}


DUCKIEBOT: DuckiebotParams = DuckiebotParams()
"""The one shared, validated Duckiebot parameter set. Import this; do not build your own."""
