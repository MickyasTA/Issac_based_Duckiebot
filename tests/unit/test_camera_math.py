"""The analytic camera model of SPEC v2 S2 / S4.1 / S4.4 (owner ``[env]``).

Every number asserted here is quoted somewhere in the specification, so this file is the place a
silent change to the camera geometry becomes a red test rather than a sim-to-real gap discovered
months later.

The load-bearing one is :func:`test_pixels_are_square`. The whole S4 redesign exists because
Isaac Sim 5.1 averages anisotropic ``fx``/``fy`` and forces the aperture ratio to the render
aspect; the chosen aperture pair makes the two focal lengths identical, which is what makes the
rendered geometry invariant to whether Isaac honours the authored ``vertical_aperture`` or
recomputes it. If ``fx != fy`` the invariance is gone and nothing downstream notices.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from duckiebot_rl import camera_math as shim  # noqa: E402
from duckiebot_rl.assets.params import DUCKIEBOT  # noqa: E402
from duckiebot_rl.envs import camera_math as cm  # noqa: E402

NOMINAL_PITCH_RAD = math.radians(DUCKIEBOT.camera_pitch_down_deg)


# ------------------------------------------------------------------------------- intrinsics


def test_pixels_are_square():
    """``fx == fy`` exactly. This is the entire point of the S4.1 canonical camera."""
    fx, fy = cm.focal_px()
    assert fx == pytest.approx(fy, rel=0, abs=1e-12)
    assert fx == pytest.approx(DUCKIEBOT.camera_focal_px, abs=5e-3)


def test_pixel_pitch_is_identical_on_both_axes():
    """The aperture pair is what makes the pixels square; assert the ratio, not just the result."""
    horizontal = DUCKIEBOT.camera_horizontal_aperture_mm / DUCKIEBOT.render_width_px
    vertical = DUCKIEBOT.camera_vertical_aperture_mm / DUCKIEBOT.render_height_px
    assert horizontal == pytest.approx(vertical, rel=1e-4)


def test_field_of_view_matches_the_spec():
    """111.0 deg horizontal (the dt-core rectified FOV) and 88.26 deg vertical."""
    assert cm.hfov_deg() == pytest.approx(111.0, abs=0.01)
    assert cm.vfov_deg() == pytest.approx(DUCKIEBOT.camera_vfov_deg, abs=0.01)
    assert cm.mj_camera_fovy_deg() == pytest.approx(cm.vfov_deg(), abs=1e-12)


def test_fov_follows_from_the_authored_numbers():
    """FOV is recomputed from focal length and aperture, not read from a stored constant."""
    kwargs = cm.pinhole_camera_kwargs()
    expected_h = math.degrees(2.0 * math.atan(kwargs["horizontal_aperture"] / (2.0 * kwargs["focal_length"])))
    assert cm.hfov_deg() == pytest.approx(expected_h, abs=1e-9)


def test_pinhole_kwargs_are_authored_directly():
    """The Isaac spawn args are the S4.1 numbers, including the 6 m far clip."""
    kwargs = cm.pinhole_camera_kwargs()
    assert kwargs["focal_length"] == pytest.approx(7.201)
    assert kwargs["horizontal_aperture"] == pytest.approx(20.955)
    assert kwargs["vertical_aperture"] == pytest.approx(13.970)
    assert kwargs["clipping_range"] == (0.05, 6.0)


def test_principal_point_is_the_image_centre():
    """USD aperture offsets are ignored by Isaac, so cx/cy must be the exact centre."""
    cx, cy = cm.principal_point()
    assert (cx, cy) == (96.0, 64.0)


# ------------------------------------------------------------------------------- quaternions


def test_golden_quaternion_at_zero_pitch():
    """SPEC v2 S2: pitch 0 gives (0.5, -0.5, 0.5, -0.5)."""
    quat = cm.quat_cam_ros(0.0)
    for got, want in zip(quat, cm.GOLDEN_QUAT_PITCH_0, strict=True):
        assert got == pytest.approx(want, abs=1e-12)


def test_golden_quaternion_at_nominal_pitch():
    """SPEC v2 S2: pitch 25.3 deg gives (0.37837, -0.59736, 0.59736, -0.37837)."""
    quat = cm.quat_cam_ros(NOMINAL_PITCH_RAD)
    for got, want in zip(quat, cm.GOLDEN_QUAT_PITCH_NOMINAL, strict=True):
        assert got == pytest.approx(want, abs=1e-5)


def test_quaternion_is_unit_length_across_the_dr_range():
    """V10 sweeps pitch over U(15, 28) deg; the closed form must stay normalised throughout."""
    for degrees in (0.0, 5.0, 15.0, 25.3, 28.0, 45.0, -20.0):
        quat = cm.quat_cam_ros(math.radians(degrees))
        assert sum(component**2 for component in quat) == pytest.approx(1.0, abs=1e-12)


def test_quaternion_columns_are_the_optical_axes():
    """The rotation's columns are right, down and forward, which is what fixes every sign."""
    pitch = NOMINAL_PITCH_RAD
    right, down, forward = cm.quat_columns(cm.quat_cam_ros(pitch))
    assert right == pytest.approx((0.0, -1.0, 0.0), abs=1e-12)
    assert down == pytest.approx((-math.sin(pitch), 0.0, -math.cos(pitch)), abs=1e-9)
    assert forward == pytest.approx((math.cos(pitch), 0.0, -math.sin(pitch)), abs=1e-9)


def test_pitch_outside_the_representable_range_raises():
    """A camera pointing straight down is never a valid mount and must not silently produce NaN."""
    with pytest.raises(ValueError, match="outside"):
        cm.quat_cam_ros(math.pi / 2.0)
    with pytest.raises(ValueError, match="outside"):
        cm.quat_cam_ros(-math.pi)


def test_rpy_helper_reduces_to_the_pitch_only_helper():
    """Zero yaw and roll must reproduce quat_cam_ros bit for bit, not merely closely."""
    for degrees in (0.0, 15.0, 25.3, 28.0):
        pitch = math.radians(degrees)
        assert cm.quat_cam_ros_rpy(pitch) == cm.quat_cam_ros(pitch)


def test_torch_and_scalar_quaternions_agree():
    """The batched torch twin is a second transcription of one formula; pin them together."""
    torch = pytest.importorskip("torch")
    pitches = [math.radians(v) for v in (15.0, 20.0, 25.3, 28.0)]
    yaws = [math.radians(v) for v in (-3.0, 0.0, 1.5, 3.0)]
    rolls = [math.radians(v) for v in (3.0, -1.0, 0.0, -3.0)]
    batched = cm.quat_cam_ros_torch(
        torch.tensor(pitches, dtype=torch.float64),
        torch.tensor(yaws, dtype=torch.float64),
        torch.tensor(rolls, dtype=torch.float64),
    )
    for i, (pitch, yaw, roll) in enumerate(zip(pitches, yaws, rolls, strict=True)):
        expected = cm.quat_cam_ros_rpy(pitch, yaw, roll)
        for j in range(4):
            assert float(batched[i, j]) == pytest.approx(expected[j], abs=1e-9)


def test_mujoco_axes_derive_from_the_same_quaternion():
    """The MuJoCo xyaxes and the Isaac offset cannot disagree, because both come from one helper."""
    pitch = NOMINAL_PITCH_RAD
    xyaxes = cm.mj_camera_xyaxes(pitch)
    assert xyaxes[:3] == pytest.approx((0.0, -1.0, 0.0), abs=1e-12)
    assert xyaxes[3:] == pytest.approx((math.sin(pitch), 0.0, math.cos(pitch)), abs=1e-9)
    assert cm.mj_camera_forward(pitch) == pytest.approx((math.cos(pitch), 0.0, -math.sin(pitch)), abs=1e-9)


# ---------------------------------------------------------------- horizon and ground geometry


@pytest.mark.parametrize(
    ("pitch_deg", "expected_row"),
    [(25.3, 32.8), (15.0, 46.3), (28.0, 28.9)],
)
def test_horizon_row_matches_the_s4_4_acceptance_values(pitch_deg, expected_row):
    """SPEC v2 S4.4 item 1 pins three rows; M6 compares the render against these within 1.5 px."""
    row = cm.horizon_row(math.radians(pitch_deg))
    assert row == pytest.approx(expected_row, abs=0.05)


def test_horizon_row_is_the_analytic_closed_form():
    """``row = cy - f * tan(pitch)``, with no hidden fudge factor."""
    pitch = NOMINAL_PITCH_RAD
    _cx, cy = cm.principal_point()
    fx, _fy = cm.focal_px()
    assert cm.horizon_row(pitch) == pytest.approx(cy - fx * math.tan(pitch), abs=1e-9)


def test_horizon_row_agrees_with_the_projection_of_a_far_ground_point():
    """A ground point 500 m away must land on the analytic horizon; two derivations, one answer."""
    _u, v = cm.project_ground_point(500.0, 0.0)
    assert v == pytest.approx(cm.horizon_row(), abs=0.02)


def test_horizon_sits_just_inside_the_cropped_observation():
    """At the nominal mount the horizon is at obs row 0.4: the crop keeps a sliver of background."""
    assert cm.horizon_row_obs() == pytest.approx(0.4, abs=0.05)


def test_row_conversion_round_trips():
    """The downsample-then-crop mapping is invertible, half-pixel offset included."""
    for render_row in (0.0, 32.81, 64.0, 127.0):
        assert cm.obs_row_to_render_row(cm.render_row_to_obs_row(render_row)) == pytest.approx(
            render_row, abs=1e-12
        )


def test_nearest_visible_ground_point():
    """SPEC v2 S4.4: 0.038 m ahead of the camera glass at the nominal mount."""
    assert cm.ground_distance_for_row(128.0) == pytest.approx(0.038, abs=0.001)


def test_ground_distance_and_row_are_inverses():
    """Round-trip the ground map so a sign error in either direction shows up."""
    for distance in (0.05, 0.2, 0.6, 1.5, 3.0):
        row = cm.row_for_ground_distance(distance)
        assert cm.ground_distance_for_row(row) == pytest.approx(distance, rel=1e-6)


def test_rows_at_or_above_the_horizon_see_no_ground():
    """A ray at or above the horizon never meets a flat floor."""
    assert cm.ground_distance_for_row(cm.horizon_row()) == math.inf
    assert cm.ground_distance_for_row(0.0) == math.inf


def test_observation_spans_the_documented_vertical_angles():
    """SPEC v2 S4.4: the crop spans +0.6 deg above the horizon down to 69.4 deg below."""
    assert cm.elevation_deg_for_row(32.0) == pytest.approx(0.57, abs=0.05)
    assert cm.elevation_deg_for_row(128.0) == pytest.approx(-69.4, abs=0.1)


def test_at_the_pitch_dr_edge_the_crop_admits_the_background():
    """At the 15 deg DR edge the top row looks 11 deg above horizontal, so walls enter the frame."""
    assert cm.elevation_deg_for_row(32.0, math.radians(15.0)) == pytest.approx(11.0, abs=0.5)


# ------------------------------------------------------------------------------- projection


def test_lateral_offset_projects_to_the_analytic_column():
    """SPEC v2 S4.4 item 2: the column of a known lateral offset at a known distance.

    Column displacement from the centre is ``-f * lateral / z_optical``, where ``z_optical``
    already includes the pitch. Any fx/fy averaging or aperture forcing moves this column.
    """
    pitch = NOMINAL_PITCH_RAD
    height = DUCKIEBOT.camera_height_m
    forward, lateral = 1.0, 0.12
    fx, _fy = cm.focal_px()
    cx, _cy = cm.principal_point()
    z_optical = forward * math.cos(pitch) + height * math.sin(pitch)
    u, _v = cm.project_ground_point(forward, lateral, pitch, height)
    assert u == pytest.approx(cx - fx * lateral / z_optical, abs=1e-9)


def test_a_point_to_the_left_lands_left_of_centre():
    """Sign check on the lateral axis: robot-left must be image-left."""
    cx, _cy = cm.principal_point()
    left, _ = cm.project_ground_point(1.0, +0.15)
    right, _ = cm.project_ground_point(1.0, -0.15)
    assert left < cx < right


def test_closer_ground_points_are_lower_in_the_image():
    """Sign check on the vertical axis: nearer ground must map to a larger row."""
    rows = [cm.project_ground_point(distance, 0.0)[1] for distance in (0.1, 0.5, 1.0, 3.0)]
    assert rows == sorted(rows, reverse=True)


def test_a_point_behind_the_camera_raises():
    """A negative depth must fail loudly rather than fold to a mirrored pixel."""
    with pytest.raises(ValueError, match="in front of the camera"):
        cm.project_camera_point(0.0, 0.0, -0.1)


# ------------------------------------------------------------------------------ shared module


def test_the_shared_shim_re_exports_the_same_objects():
    """``duckiebot_rl/camera_math.py`` is the path CONTRIBUTING and sim2sim/_resolve expect."""
    for name in cm.__all__:
        if name in ("CROP_TOP", "OBS_H", "OBS_W", "RENDER_H", "RENDER_W"):
            continue
        assert getattr(shim, name) is getattr(cm, name), name


def test_the_sim2sim_resolver_now_uses_the_shared_helper():
    """The MuJoCo harness must pick up this module instead of its local fallback."""
    from duckiebot_rl.sim2sim import _resolve

    module = _resolve.resolve_camera_math()
    assert module is not None
    assert module.quat_cam_ros(0.0) == cm.GOLDEN_QUAT_PITCH_0
    assert _resolve.quat_cam_ros(NOMINAL_PITCH_RAD) == pytest.approx(cm.GOLDEN_QUAT_PITCH_NOMINAL, abs=1e-5)
