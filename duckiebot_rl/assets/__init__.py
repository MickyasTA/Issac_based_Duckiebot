"""Clean-room Duckiebot robot description (SPEC v2 S3.2).

Three modules, one direction of dependency:

.. code-block:: text

    params.py  ->  urdf.py       (writes assets/duckiebot/duckiebot.urdf)
               ->  robot_cfg.py  (Isaac Lab ArticulationCfg, lazily imported)

:mod:`duckiebot_rl.assets.params` is the single source of truth for every physical constant. The
URDF generator, the Isaac Lab config, the MuJoCo MJCF generator and the documentation all read it;
none of them hold a second copy of any number.

Only :mod:`duckiebot_rl.assets.params` and :mod:`duckiebot_rl.assets.urdf` are re-exported here.
:mod:`duckiebot_rl.assets.robot_cfg` is deliberately NOT imported eagerly: it is importable
without Isaac Lab, but keeping it out of this ``__init__`` means a CPU-only tool that just wants
the parameters never even touches the guarded import path.

Licensing: every shape in this package is an authored primitive. No mesh, texture or material
from any third-party robot asset pack enters this repository. See SPEC v2 S3.1 and the gate at
``scripts/check_clean_room.py``.
"""

from __future__ import annotations

from duckiebot_rl.assets.params import DUCKIEBOT, DuckiebotParams, ParameterConsistencyError
from duckiebot_rl.assets.urdf import (
    ROBOT_NAME,
    URDF_FILENAME,
    box_inertia,
    cylinder_inertia,
    generate_urdf,
    sphere_inertia,
    write_urdf,
)

__all__ = [
    "DUCKIEBOT",
    "ROBOT_NAME",
    "URDF_FILENAME",
    "DuckiebotParams",
    "ParameterConsistencyError",
    "box_inertia",
    "cylinder_inertia",
    "generate_urdf",
    "sphere_inertia",
    "write_urdf",
]
