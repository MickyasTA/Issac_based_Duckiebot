"""Import shim for the shared camera model. The implementation lives in ``envs/camera_math.py``.

``CONTRIBUTING.md`` and SPEC v2 S10 both name ``duckiebot_rl/camera_math.py`` as one of the three
shared, sign-off-gated modules, and :func:`duckiebot_rl.sim2sim._resolve.resolve_camera_math`
imports exactly that path to decide whether the sim-to-sim harness can drop its local fallback.
The implementation belongs to the ``[env]`` package next to the environment that authors the
Isaac camera, so this module re-exports it rather than holding a second copy: a duplicated
focal length or a duplicated quaternion would be precisely the divergence the sign-off rule
exists to prevent.

Nothing here may grow logic. If you are about to add a function, add it to
:mod:`duckiebot_rl.envs.camera_math` and extend ``__all__`` below.
"""

from __future__ import annotations

from duckiebot_rl.envs.camera_math import (
    GOLDEN_QUAT_PITCH_0,
    GOLDEN_QUAT_PITCH_NOMINAL,
    Quaternion,
    elevation_deg_for_row,
    focal_px,
    ground_distance_for_row,
    hfov_deg,
    horizon_row,
    horizon_row_obs,
    mj_camera_forward,
    mj_camera_fovy_deg,
    mj_camera_xyaxes,
    obs_row_to_render_row,
    pinhole_camera_kwargs,
    principal_point,
    project_camera_point,
    project_ground_point,
    quat_cam_ros,
    quat_cam_ros_rpy,
    quat_cam_ros_torch,
    quat_columns,
    render_row_to_obs_row,
    row_for_ground_distance,
    vfov_deg,
)

__all__ = [
    "GOLDEN_QUAT_PITCH_0",
    "GOLDEN_QUAT_PITCH_NOMINAL",
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
