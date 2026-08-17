r"""MuJoCo sim-to-sim transfer target and evaluation harness (SPEC v2 S8, owner ``[sim2sim]``).

This package is the second, independent simulator the trained policy has to survive. Its job is to
be *the same task* under *different physics and a different renderer*, so that the C1-vs-C5 delta
in the S8.4 evaluation matrix measures robustness rather than a modelling slip.

Modules
-------
``mjcf``
    Generates ``duckiebot.xml`` from :mod:`duckiebot_rl.assets.params`. Wheel collision geoms are
    spheres and the integrator is ``implicitfast``; both are measured requirements, see the module
    docstring.
``track``
    Builds a MuJoCo scene from a MapFormat1 map plus, when available, the tile textures the
    ``[city]`` module generates. Also derives the lane graph used for ``d``, ``psi`` and progress.
    Both come from one ``duckiebot_rl.city.spec.TileSpec``, resolved by
    :func:`resolve_city_params`, and the texture is rotated into each tile's own orientation, so
    the reward reference line and the painted lane are the same line.
``env``
    Single-robot environment with the same control rate, the same action path and the same
    observation preprocessing as the Isaac Lab environment.
``sysid``
    Two-stage Isaac-to-MuJoCo physics matching (closed-form kinematics, then Levenberg-Marquardt on
    armature, joint friction and damping), producing a parameter file and a residual report.
``evaluate``
    The S8.4 evaluation harness: N episodes per condition per seed, medians with confidence
    intervals, a wall-clock budget and process-level parallelism.

Which interpreter runs this
---------------------------
Everything in this package is designed for the **tools venv**,
``d:/Personal/personal/mujoco_venv/Scripts/python.exe``, because that is the only interpreter on
this machine with ``mujoco``. The Isaac venv (``.../wheeled_quadruped_robot/.venv``) has torch and
cv2 but no ``mujoco``, and installing Isaac Sim's dependency tree next to MuJoCo is not worth the
risk.

The tools venv is only partially provisioned right now. Measured on this machine:

======================  =========  =========================================================
Package                 State      Needed by
======================  =========  =========================================================
``mujoco`` 3.11.0       present    everything
``numpy`` 2.4.6         present    everything
``pytest`` 9.1.1        present    the unit tests
``torch``               MISSING    TorchScript policies in :mod:`evaluate` (``--policy``)
``Pillow``              MISSING    nothing here; the shared tile generator is pure numpy
``pyyaml``              MISSING    loading a map from a ``.yaml`` file through ``[city]``
``opencv-python``       MISSING    optional rectification parity fixtures
======================  =========  =========================================================

What that means in practice, verified on this machine rather than assumed:

* ``tests/unit/test_mjcf.py`` and ``tests/unit/test_mj_kinematics.py`` need nothing beyond
  ``mujoco`` and ``numpy``. They run today.
* The **whole vision path runs today too**. The ``[dr]``-owned preprocess module ships a numpy
  implementation of the S4.3 chain and a numpy ``FrameStack``, and the ``[city]`` tile generator is
  pure numpy, so ``MjEnvCfg(obs_mode="rgb_vec")`` produces a real ``(48, 96, 9)`` uint8 observation
  from real lane markings without torch. This was the single biggest risk in critic item J and it
  turned out not to bite.
* What still needs torch is loading a **trained** policy: ``evaluate --policy some.pt`` traces
  through ``torch.jit``. Until torch is installed the harness runs only the built-in scripted
  policies, which is enough for smoke tests and wall-clock calibration but not for a C-number.
* Loading a map from disk needs pyyaml, either through ``duckiebot_rl.city.maps`` or through this
  package's own fallback loader. Maps passed as plain dicts work without it.

So the SPEC v2 M0 installs are still required before any reportable evaluation::

    d:/Personal/personal/mujoco_venv/Scripts/python.exe -m pip install torch \\
        --index-url https://download.pytorch.org/whl/cpu
    d:/Personal/personal/mujoco_venv/Scripts/python.exe -m pip install \\
        opencv-python-headless Pillow pyyaml onnxruntime usd-core

Call :func:`environment_report` or run ``python scripts/eval_sim2sim.py check-env`` to see the
current state of any interpreter, including which cross-team modules have landed.
"""

from __future__ import annotations

from ._resolve import (
    CityParams,
    ObsParams,
    RobotParams,
    SharedModuleUnavailable,
    SimParams,
    SpecFallbackWarning,
    environment_report,
    format_environment_report,
    mj_camera_xyaxes,
    quat_cam_ros,
    resolve_city_params,
    resolve_robot_params,
    resolve_sim_params,
)

__all__ = [
    "CityParams",
    "ObsParams",
    "RobotParams",
    "SharedModuleUnavailable",
    "SimParams",
    "SpecFallbackWarning",
    "environment_report",
    "format_environment_report",
    "mj_camera_xyaxes",
    "quat_cam_ros",
    "resolve_city_params",
    "resolve_robot_params",
    "resolve_sim_params",
]

__spec_version__ = "SPEC v2 (03_SPEC_V2.md), section S8"
