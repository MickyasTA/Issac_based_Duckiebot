"""The one camera model shared by Isaac, MuJoCo and the deployed robot (SPEC v2 S2, S4.1, S4.4).

Owner ``[env]`` with sign-off from ``[ppo]`` + ``[sim2sim]`` + ``[deploy]`` (SPEC v2 S10 lists
``camera_math.py`` as shared load-bearing code, alongside ``dr/preprocess.py`` and
``dr/delay.py``). A one-line change here moves the sim-to-real gap in three consumers at once.

Two rules this module exists to enforce
---------------------------------------

**1. The intrinsics are authored, never inferred.** Isaac Sim 5.1 cannot represent anisotropic
pixels: ``PinholeCameraCfg.from_intrinsic_matrix`` averages ``fx`` and ``fy``
(``isaaclab/utils/sensors.py:36-37``, "Camera non square pixels are not supported by Omniverse.
The average of f_x and f_y are used.") and then *forces* the aperture ratio to the render aspect
(lines 53-56), so an anisotropic request silently becomes a different camera. SPEC v2 S4.1
therefore picks a square-pixel pinhole up front and authors ``focal_length``,
``horizontal_aperture`` and ``vertical_aperture`` directly. :func:`pinhole_camera_kwargs` is the
only place those three numbers are assembled, and :func:`focal_px` proves the result is square:
``fx == fy`` exactly, because ``horizontal_aperture / width == vertical_aperture / height``.

**2. Camera pitch is a positive scalar until the very last moment.** ``pitch_down`` is stored as
a positive "nose down" angle everywhere in the repository. :func:`quat_cam_ros` is the only
function allowed to turn it into a rotation, which is what keeps the Isaac ``OffsetCfg``, the
MuJoCo ``xyaxes`` and the deployment extrinsics from disagreeing by a sign (critic item J).

Frames
------

Two frames, and the whole module is the map between them.

* ``base_link``: REP-103, x forward, y left, z up. The camera mount pose lives here.
* ROS *optical*: x right, y down, z forward. Image coordinates are ``u`` right, ``v`` down from
  the top-left corner, and the projection is the textbook ``u = cx + f * X / Z``.

The rotation taking optical axes into ``base_link`` for a camera pitched ``p`` down is

.. code-block:: text

    R(p) = Ry(p) @ M0,     M0 = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]

whose columns are exactly ``right = (0, -1, 0)``, ``down = (-sin p, 0, -cos p)`` and
``forward = (cos p, 0, -sin p)``. Its trace is ``-sin p``, which collapses the usual quaternion
trace formula to the two-square-root closed form in :func:`quat_cam_ros`.

Analytic predictions this module owns (SPEC v2 S4.4 acceptance items 1 and 2)
----------------------------------------------------------------------------

* :func:`horizon_row` - the ground/background boundary row of a flat-ground render. The M6 gate
  compares the rendered boundary against this number within +/-1.5 px, which is the only test
  that can catch a silent ``fx``/``fy`` average or a forced aperture ratio.
* :func:`project_ground_point` / :func:`ground_distance_for_row` - where a known ground point
  lands, and how far ahead the nearest visible ground point is.
* :func:`elevation_deg_for_row` - the vertical angle subtended by any render row, which is what
  makes "the crop keeps 0.6 deg of sky at nominal pitch, and 11 deg at the DR edge" checkable.

Dependencies: the standard library only. This module must import in the MuJoCo venv and on a
Jetson, so it may never import torch, numpy, Isaac or OpenCV.
"""

from __future__ import annotations

import math
from typing import Any

from duckiebot_rl.assets.params import DUCKIEBOT, DuckiebotParams
from duckiebot_rl.dr.preprocess import CROP_TOP, OBS_H, OBS_W, RENDER_H, RENDER_W

__all__ = [
    "CROP_TOP",
    "GOLDEN_QUAT_PITCH_0",
    "GOLDEN_QUAT_PITCH_NOMINAL",
    "OBS_H",
    "OBS_W",
    "RENDER_H",
    "RENDER_W",
    "Quaternion",
    "elevation_deg_for_row",
    "focal_px",
    "ground_distance_for_row",
    "hfov_deg",
    "horizon_row",
    "horizon_row_obs",
    "mj_camera_forward",
    "mj_camera_fovy_deg",
    "mj_camera_xyaxes",
    "obs_row_to_render_row",
    "pinhole_camera_kwargs",
    "principal_point",
    "project_camera_point",
    "project_ground_point",
    "quat_cam_ros",
    "quat_cam_ros_rpy",
    "quat_cam_ros_torch",
    "quat_columns",
    "render_row_to_obs_row",
    "row_for_ground_distance",
    "vfov_deg",
]

Quaternion = tuple[float, float, float, float]
"""A unit quaternion in Isaac Lab / USD order: ``(w, x, y, z)``."""

GOLDEN_QUAT_PITCH_0: Quaternion = (0.5, -0.5, 0.5, -0.5)
"""SPEC v2 S2 golden value: the ROS optical frame of an unpitched camera, in ``base_link``."""

GOLDEN_QUAT_PITCH_NOMINAL: Quaternion = (0.37837, -0.59736, 0.59736, -0.37837)
"""SPEC v2 S2 golden value at the nominal 25.3 deg down pitch, quoted to 5 decimal places."""


# =============================================================================================
# Rotation
# =============================================================================================


def quat_cam_ros(pitch_down_rad: float) -> Quaternion:
    """Return the ROS optical frame of a pitched-down camera, expressed in ``base_link``.

    This is the single sign-bearing function of the whole camera pipeline (SPEC v2 S2). Every
    consumer - the Isaac ``CameraCfg.OffsetCfg(convention="ros")``, the MuJoCo ``xyaxes``, the
    deployment extrinsics sidecar - derives its orientation from here and from nowhere else.

    Args:
        pitch_down_rad: Downward pitch in radians, a POSITIVE scalar meaning "nose down".

    Returns:
        The unit quaternion ``(w, x, y, z)``.

    Raises:
        ValueError: If the pitch is outside the representable open range ``(-pi/2, +pi/2)``.
            At exactly +/-90 deg the closed form loses a square root's worth of precision and,
            more importantly, a camera pointing straight down is never a valid mount pose.
    """
    if not -math.pi / 2.0 < pitch_down_rad < math.pi / 2.0:
        raise ValueError(f"pitch_down_rad {pitch_down_rad!r} is outside (-pi/2, +pi/2)")
    sin_p = math.sin(pitch_down_rad)
    lo = math.sqrt(1.0 - sin_p) / 2.0
    hi = math.sqrt(1.0 + sin_p) / 2.0
    return (lo, -hi, hi, -lo)


def quat_columns(quat: Quaternion) -> tuple[tuple[float, float, float], ...]:
    """Return the three columns of the rotation matrix of a ``(w, x, y, z)`` quaternion.

    For a quaternion produced by :func:`quat_cam_ros` the columns are the optical ``right``,
    ``down`` and ``forward`` axes written in ``base_link`` coordinates.

    Args:
        quat: Unit quaternion ``(w, x, y, z)``.

    Returns:
        ``(col0, col1, col2)``, each a 3-tuple.
    """
    w, x, y, z = quat
    col0 = (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y))
    col1 = (2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x))
    col2 = (2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y))
    return col0, col1, col2


def _quat_multiply(a: Quaternion, b: Quaternion) -> Quaternion:
    """Return the Hamilton product ``a * b`` of two ``(w, x, y, z)`` quaternions.

    Args:
        a: Left quaternion.
        b: Right quaternion.

    Returns:
        The product, normalised to unit length.
    """
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    out = (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )
    norm = math.sqrt(sum(component * component for component in out))
    return (out[0] / norm, out[1] / norm, out[2] / norm, out[3] / norm)


def quat_cam_ros_rpy(pitch_down_rad: float, yaw_rad: float = 0.0, roll_rad: float = 0.0) -> Quaternion:
    """Return the optical-frame quaternion of a mount with the full V10 orientation jitter.

    DR axis V10 perturbs the mount by yaw and roll of +/-3 deg on top of the pitch. The mount
    rotation in ``base_link`` is composed in the fixed order ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``
    and then post-multiplied by the pitch-only optical quaternion of :func:`quat_cam_ros`, so
    ``yaw = roll = 0`` reproduces that function bit for bit and no second sign convention is
    introduced.

    Args:
        pitch_down_rad: Downward pitch in radians, positive is nose down.
        yaw_rad: Mount yaw about ``base_link`` +z, positive turns the camera left.
        roll_rad: Mount roll about the camera's forward axis, positive rolls the image left.

    Returns:
        The unit quaternion ``(w, x, y, z)``.
    """
    base = quat_cam_ros(pitch_down_rad)
    if yaw_rad == 0.0 and roll_rad == 0.0:
        return base
    half_yaw, half_roll = 0.5 * yaw_rad, 0.5 * roll_rad
    q_yaw: Quaternion = (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))
    # Roll is about the optical z axis (the viewing direction), so it multiplies on the right.
    q_roll: Quaternion = (math.cos(half_roll), 0.0, 0.0, math.sin(half_roll))
    return _quat_multiply(_quat_multiply(q_yaw, base), q_roll)


def quat_cam_ros_torch(
    pitch_down_rad: Any,
    yaw_rad: Any = None,
    roll_rad: Any = None,
) -> Any:
    """Batched torch twin of :func:`quat_cam_ros_rpy`, for the per-env V10 mount jitter.

    The Isaac environment resamples the camera mount for every resetting env at once, and
    calling the scalar helper in a python loop would force one host synchronisation per reset
    (SPEC v2 S6.7 guard 5). This is the same closed form, written once more in torch; the two
    are pinned to each other by ``tests/unit/test_camera_math.py``, which evaluates both over a
    sweep of pitch, yaw and roll and asserts they agree to 1e-6. Torch is imported inside the
    function so this module still loads in a venv without it.

    Args:
        pitch_down_rad: ``(N,)`` tensor of downward pitches in radians.
        yaw_rad: Optional ``(N,)`` mount yaw in radians.
        roll_rad: Optional ``(N,)`` mount roll in radians.

    Returns:
        An ``(N, 4)`` tensor of ``(w, x, y, z)`` quaternions.
    """
    import torch

    pitch = torch.as_tensor(pitch_down_rad)
    sin_p = torch.sin(pitch)
    lo = torch.sqrt(torch.clamp(1.0 - sin_p, min=0.0)) / 2.0
    hi = torch.sqrt(torch.clamp(1.0 + sin_p, min=0.0)) / 2.0
    quat = torch.stack([lo, -hi, hi, -lo], dim=-1)
    if yaw_rad is None and roll_rad is None:
        return quat

    def _multiply(a: Any, b: Any) -> Any:
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

    zero = torch.zeros_like(pitch)
    if yaw_rad is not None:
        half = 0.5 * torch.as_tensor(yaw_rad)
        quat = _multiply(torch.stack([torch.cos(half), zero, zero, torch.sin(half)], dim=-1), quat)
    if roll_rad is not None:
        half = 0.5 * torch.as_tensor(roll_rad)
        quat = _multiply(quat, torch.stack([torch.cos(half), zero, zero, torch.sin(half)], dim=-1))
    return quat


def mj_camera_xyaxes(pitch_down_rad: float) -> tuple[float, ...]:
    """Return the MuJoCo ``xyaxes`` attribute for the same mount pose.

    MuJoCo cameras look along ``-z`` of their own frame with ``+y`` up, so ``xyaxes`` is
    ``[right ; up]`` in parent-body coordinates. Both vectors come from :func:`quat_cam_ros`:
    the ROS optical x axis *is* MuJoCo's camera x, and MuJoCo's camera y is the negated ROS
    optical y (down becomes up).

    Args:
        pitch_down_rad: Downward pitch in radians.

    Returns:
        ``(rx, ry, rz, ux, uy, uz)``.
    """
    right, down, _forward = quat_columns(quat_cam_ros(pitch_down_rad))
    return tuple(right) + tuple(-v for v in down)


def mj_camera_forward(pitch_down_rad: float) -> tuple[float, float, float]:
    """Return the viewing direction (ROS optical +z) in parent-body coordinates.

    Args:
        pitch_down_rad: Downward pitch in radians.

    Returns:
        The unit forward vector.
    """
    _right, _down, forward = quat_columns(quat_cam_ros(pitch_down_rad))
    return forward


# =============================================================================================
# Intrinsics
# =============================================================================================


def focal_px(params: DuckiebotParams = DUCKIEBOT) -> tuple[float, float]:
    """Return the focal length in pixels along x and y, which are equal by construction.

    Isaac authors a camera through a physical film back: ``fx = width * focal_length /
    horizontal_aperture`` and ``fy = height * focal_length / vertical_aperture``. SPEC v2 S4.1
    picks the aperture pair so that ``horizontal_aperture / width == vertical_aperture / height``
    (20.955 / 192 == 13.970 / 128 == 0.10914 mm per pixel), which makes the pixels square and
    makes the geometry invariant to whether Isaac 5.1 honours the authored
    ``vertical_aperture`` or recomputes it from the render aspect. Both branches give the same
    camera; that invariance is the entire point of the S4 redesign.

    Args:
        params: Parameter set to read. Defaults to the shared singleton.

    Returns:
        ``(fx, fy)`` in pixels.
    """
    fx = params.render_width_px * params.camera_focal_length_mm / params.camera_horizontal_aperture_mm
    fy = params.render_height_px * params.camera_focal_length_mm / params.camera_vertical_aperture_mm
    return fx, fy


def principal_point(params: DuckiebotParams = DUCKIEBOT) -> tuple[float, float]:
    """Return the principal point in render pixels.

    Isaac hardcodes both USD aperture offsets to 0.0 (``isaaclab/utils/sensors.py:57-58``) and
    the spawner warns and ignores any nonzero value (internal ticket OM-42611), so the principal
    point is the exact image centre and DR axis V11 shifts it in torch instead
    (:func:`duckiebot_rl.dr.preprocess.shift_principal_point`).

    Args:
        params: Parameter set to read.

    Returns:
        ``(cx, cy)`` in pixels.
    """
    return 0.5 * params.render_width_px, 0.5 * params.render_height_px


def hfov_deg(params: DuckiebotParams = DUCKIEBOT) -> float:
    """Return the horizontal field of view in degrees.

    Args:
        params: Parameter set to read.

    Returns:
        ``2 * atan(horizontal_aperture / (2 * focal_length))`` in degrees; 111.0 at the S4.1
        canonical numbers, which is the dt-core rectified horizontal FOV.
    """
    half = params.camera_horizontal_aperture_mm / (2.0 * params.camera_focal_length_mm)
    return math.degrees(2.0 * math.atan(half))


def vfov_deg(params: DuckiebotParams = DUCKIEBOT) -> float:
    """Return the vertical field of view in degrees.

    Args:
        params: Parameter set to read.

    Returns:
        ``2 * atan(vertical_aperture / (2 * focal_length))`` in degrees; 88.26 at the S4.1
        canonical numbers, inside the ~90 deg valid vertical FOV of an alpha=0 rectification.
    """
    half = params.camera_vertical_aperture_mm / (2.0 * params.camera_focal_length_mm)
    return math.degrees(2.0 * math.atan(half))


def mj_camera_fovy_deg(params: DuckiebotParams = DUCKIEBOT) -> float:
    """Return the MuJoCo ``fovy`` that reproduces this camera.

    MuJoCo parameterises a camera by its vertical field of view, so the sim-to-sim harness needs
    exactly :func:`vfov_deg`. It is given its own name because ``fovy`` is what the MJCF
    attribute is called and a reader should not have to know the two are the same number.

    Args:
        params: Parameter set to read.

    Returns:
        The vertical field of view in degrees.
    """
    return vfov_deg(params)


def pinhole_camera_kwargs(params: DuckiebotParams = DUCKIEBOT) -> dict[str, object]:
    """Return the keyword arguments for Isaac's ``PinholeCameraCfg``, authored directly.

    Never call ``PinholeCameraCfg.from_intrinsic_matrix``: it averages ``fx`` and ``fy`` and
    forces the aperture ratio to the render aspect (critic item C). This function is the
    authored alternative and is the only place the four numbers appear together.

    Args:
        params: Parameter set to read.

    Returns:
        A dict with ``focal_length``, ``horizontal_aperture``, ``vertical_aperture`` and
        ``clipping_range``, all in Isaac's "tenth of a world unit" convention, which for a
        metre-based stage means centimetres.
    """
    return {
        "focal_length": params.camera_focal_length_mm,
        "horizontal_aperture": params.camera_horizontal_aperture_mm,
        "vertical_aperture": params.camera_vertical_aperture_mm,
        "clipping_range": params.camera_clipping_range_m,
    }


# =============================================================================================
# Projection and the analytic horizon (SPEC v2 S4.4 acceptance items 1 and 2)
# =============================================================================================


def project_camera_point(
    x: float, y: float, z: float, params: DuckiebotParams = DUCKIEBOT
) -> tuple[float, float]:
    """Project a point given in ROS optical coordinates to render pixels.

    Args:
        x: Optical x in metres, positive to the right of the viewing direction.
        y: Optical y in metres, positive downward.
        z: Optical z in metres, the depth along the viewing direction; must be positive.
        params: Parameter set to read.

    Returns:
        ``(u, v)`` in render pixels, ``u`` rightward and ``v`` downward from the top-left corner.

    Raises:
        ValueError: If ``z`` is not strictly positive, i.e. the point is behind the camera.
    """
    if z <= 0.0:
        raise ValueError(f"optical z must be > 0 (the point must be in front of the camera), got {z!r}")
    fx, fy = focal_px(params)
    cx, cy = principal_point(params)
    return cx + fx * x / z, cy + fy * y / z


def project_ground_point(
    forward_m: float,
    lateral_m: float,
    pitch_down_rad: float | None = None,
    camera_height_m: float | None = None,
    params: DuckiebotParams = DUCKIEBOT,
) -> tuple[float, float]:
    """Project a point on the flat ground plane, measured from below the camera.

    This is SPEC v2 S4.4 acceptance item 2 in closed form: "a vertical pole of known lateral
    offset at known distance projects at the analytically predicted column within 1 px", which
    is what proves no ``fx``/``fy`` averaging or aperture forcing happened at render time.

    Args:
        forward_m: Distance ahead of the camera glass, in metres, along ``base_link`` +x.
        lateral_m: Offset to the robot's LEFT, in metres, along ``base_link`` +y.
        pitch_down_rad: Downward pitch. Defaults to the nominal S2 mount pitch.
        camera_height_m: Camera height above the ground. Defaults to the nominal S2 height.
        params: Parameter set to read.

    Returns:
        ``(u, v)`` in render pixels.

    Raises:
        ValueError: If the point falls behind the image plane.
    """
    pitch = math.radians(params.camera_pitch_down_deg) if pitch_down_rad is None else pitch_down_rad
    height = params.camera_height_m if camera_height_m is None else camera_height_m
    cos_p, sin_p = math.cos(pitch), math.sin(pitch)
    # (forward, lateral, -height) in base_link, rotated into the optical frame by R(p) transpose.
    x_opt = -lateral_m
    y_opt = height * cos_p - forward_m * sin_p
    z_opt = forward_m * cos_p + height * sin_p
    return project_camera_point(x_opt, y_opt, z_opt, params)


def horizon_row(pitch_down_rad: float | None = None, params: DuckiebotParams = DUCKIEBOT) -> float:
    """Return the render row of the horizon on a flat ground plane.

    A ray aimed at the horizon has optical coordinates ``(0, -sin p, cos p)``, so the row is
    ``cy - f * tan(p)`` exactly. SPEC v2 S4.4 item 1 pins three values: 32.8 at the nominal
    25.3 deg, 46.3 at the 15 deg DR edge and 28.9 at the 28 deg DR edge, each within +/-1.5 px
    of the rendered ground/background boundary (FXAA smear allowed).

    Args:
        pitch_down_rad: Downward pitch. Defaults to the nominal S2 mount pitch.
        params: Parameter set to read.

    Returns:
        The row index from the top of the RENDER image, as a float.
    """
    pitch = math.radians(params.camera_pitch_down_deg) if pitch_down_rad is None else pitch_down_rad
    _cx, cy = principal_point(params)
    _fx, fy = focal_px(params)
    return cy - fy * math.tan(pitch)


def render_row_to_obs_row(render_row: float, crop_top: int = CROP_TOP) -> float:
    """Convert a render-resolution row to the observation row after downsample and crop.

    The S4.3 tail is an exact 2x2 box average followed by ``rows[crop_top:crop_top + OBS_H]``,
    so observation row ``k`` covers render rows ``[2 * (k + crop_top), 2 * (k + crop_top) + 2)``.
    The mapping is therefore ``obs = render / 2 - crop_top``, and it is deliberately a float:
    rounding here would hide the half-pixel offset that the M6 comparison has to account for.

    Args:
        render_row: Row index at render resolution.
        crop_top: Rows removed from the top AFTER downsampling.

    Returns:
        The (possibly out-of-range) observation row.
    """
    return 0.5 * render_row - crop_top


def obs_row_to_render_row(obs_row: float, crop_top: int = CROP_TOP) -> float:
    """Invert :func:`render_row_to_obs_row`.

    Args:
        obs_row: Row index in the cropped observation.
        crop_top: Rows removed from the top AFTER downsampling.

    Returns:
        The corresponding render row.
    """
    return 2.0 * (obs_row + crop_top)


def horizon_row_obs(pitch_down_rad: float | None = None, params: DuckiebotParams = DUCKIEBOT) -> float:
    """Return the horizon row in OBSERVATION coordinates.

    At the nominal mount this is ``32.81 / 2 - 16 = 0.40``: the horizon sits just below the top
    row of the observation, so the crop keeps a sliver of background rather than removing it.
    That is intended (SPEC v2 S4.4): background invariance has to be learned from the DR, not
    granted by the crop, and at the 15 deg pitch DR edge the top row reaches 11 deg above
    horizontal and walls enter the frame outright.

    Args:
        pitch_down_rad: Downward pitch. Defaults to the nominal S2 mount pitch.
        params: Parameter set to read.

    Returns:
        The horizon row measured in the cropped observation. Negative means above the crop.
    """
    return render_row_to_obs_row(horizon_row(pitch_down_rad, params))


def elevation_deg_for_row(
    render_row: float,
    pitch_down_rad: float | None = None,
    params: DuckiebotParams = DUCKIEBOT,
) -> float:
    """Return the elevation angle of a render row above the horizontal plane.

    Args:
        render_row: Row index from the top of the render image.
        pitch_down_rad: Downward pitch. Defaults to the nominal S2 mount pitch.
        params: Parameter set to read.

    Returns:
        Degrees above horizontal; negative means below. At the nominal mount the top of the
        cropped observation (render row 32) sits at +0.57 deg and the bottom (render row 128)
        at -69.4 deg, which is the S4.4 vertical span.
    """
    pitch = math.radians(params.camera_pitch_down_deg) if pitch_down_rad is None else pitch_down_rad
    _cx, cy = principal_point(params)
    _fx, fy = focal_px(params)
    cos_p, sin_p = math.cos(pitch), math.sin(pitch)
    ratio = (render_row - cy) / fy
    # Ray direction in base_link: R(p) @ (ratio, 1) restricted to the vertical plane.
    forward = cos_p - sin_p * ratio
    up = -(cos_p * ratio + sin_p)
    return math.degrees(math.atan2(up, forward))


def ground_distance_for_row(
    render_row: float,
    pitch_down_rad: float | None = None,
    camera_height_m: float | None = None,
    params: DuckiebotParams = DUCKIEBOT,
) -> float:
    """Return how far ahead the ground point imaged at a given render row lies.

    Args:
        render_row: Row index from the top of the render image.
        pitch_down_rad: Downward pitch. Defaults to the nominal S2 mount pitch.
        camera_height_m: Camera height above the ground. Defaults to the nominal S2 height.
        params: Parameter set to read.

    Returns:
        Distance ahead of the camera in metres. At the nominal mount the bottom edge (row 128)
        gives 0.038 m, the S4.4 "nearest visible ground point", which clears the chassis front
        edge by 0.04 m. Rows at or above the horizon return ``inf``.
    """
    pitch = math.radians(params.camera_pitch_down_deg) if pitch_down_rad is None else pitch_down_rad
    height = params.camera_height_m if camera_height_m is None else camera_height_m
    _cx, cy = principal_point(params)
    _fx, fy = focal_px(params)
    cos_p, sin_p = math.cos(pitch), math.sin(pitch)
    offset = render_row - cy
    denominator = offset * cos_p + fy * sin_p
    # The tolerance is relative to the focal length, not an absolute zero test: at exactly the
    # horizon row the denominator is a few ulps of `fy` rather than 0.0, and a bare `<= 0.0`
    # would return a finite distance of order 1e15 m instead of infinity.
    if denominator <= 1e-9 * fy:
        return math.inf
    return height * (fy * cos_p - offset * sin_p) / denominator


def row_for_ground_distance(
    forward_m: float,
    pitch_down_rad: float | None = None,
    camera_height_m: float | None = None,
    params: DuckiebotParams = DUCKIEBOT,
) -> float:
    """Invert :func:`ground_distance_for_row`.

    Args:
        forward_m: Distance ahead of the camera in metres; must be positive.
        pitch_down_rad: Downward pitch. Defaults to the nominal S2 mount pitch.
        camera_height_m: Camera height above the ground. Defaults to the nominal S2 height.
        params: Parameter set to read.

    Returns:
        The render row imaging that ground point.

    Raises:
        ValueError: If ``forward_m`` is not positive.
    """
    if forward_m <= 0.0:
        raise ValueError(f"forward_m must be > 0, got {forward_m!r}")
    _u, v = project_ground_point(forward_m, 0.0, pitch_down_rad, camera_height_m, params)
    return v
