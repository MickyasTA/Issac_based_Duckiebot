"""Isaac Lab articulation configuration for the clean-room Duckiebot (SPEC v2 S2, S3.2, S5.1).

Import guard
------------
Isaac Sim and Isaac Lab are never pip dependencies of this repository, and CI runs on CPU-only
runners without them. This module therefore imports nothing from ``isaaclab`` at module scope:

* the config builders import ``isaaclab`` lazily, inside the function body;
* :func:`is_isaaclab_available` reports whether they can succeed;
* the numeric content that CI needs to check is exposed as plain dictionaries
  (:func:`wheel_actuator_spec`, :func:`spawn_property_spec`, :func:`physics_material_spec`,
  :func:`camera_mount_spec`) that need no Isaac import at all;
* ``DUCKIEBOT_CFG`` is served through a module-level ``__getattr__`` so that
  ``from duckiebot_rl.assets.robot_cfg import DUCKIEBOT_CFG`` still works inside Isaac while
  merely importing this module stays free on a CPU runner.

Where the numbers come from
---------------------------
All of them come from :mod:`duckiebot_rl.assets.params`. Nothing is duplicated here. Four choices
deserve a note because each one reverses a v1 decision:

* ``effort_limit_sim = 0.15`` N.m and ``velocity_limit_sim = 35`` rad/s. Isaac Lab overwrites the
  URDF's joint limits from the actuator config at every articulation init, so these fields, not
  the URDF, are what the simulation actually runs. The v1 value of 2.0 N.m was roughly 13x a
  DG01D 48:1 stall torque.
* ``armature`` and ``friction`` are numeric and non-zero. The MuJoCo sysid stage-2 fit tunes
  exactly these two plus ``damping``; with them unset there would be no Isaac-side counterpart to
  fit against.
* Motor variation is NOT applied here. ``randomize_actuator_gains`` is not used anywhere in this
  project (it forces a CPU sync on every reset). Every motor-side domain-randomization axis
  (D1, D2, D5, D6, D7, D8, D12, D18) lives in the S5.3 action path instead. The one exception is
  D17, the per-env effort limit, which is written once at startup.
* ``contact_offset`` is tightened to 5 mm. The PhysX default of 20 mm is comparable to the
  robot's 21 mm ground clearance and would generate chassis-ground contact pairs on flat ground,
  which the M1 acceptance test explicitly forbids.
* ``joint_drive_props.drive_type = "force"``. This one is not a preference, it is the difference
  between a robot that drives and a robot that does not.

The imported USD carries ``drive:angular:physics:type = "acceleration"``
(``assets/usd/duckiebot.usda``, both wheel joints), which is what the URDF importer writes.
Isaac Lab overwrites stiffness, damping, effort limit, velocity limit and armature from the
actuator config at every articulation init, but it never touches the drive TYPE. Under an
acceleration drive PhysX multiplies the drive gains by the joint's effective inertia, so the
0.05 N.m.s/rad damping against an 18.87 rad/s target produces
``2.25e-4 kg.m2 x 0.94 = 2.1e-4 N.m`` instead of the intended 0.15 N.m. That is 48x BELOW the
0.010 N.m joint friction of S2, so the wheels never break static friction and the robot sits
still at full commanded speed while ``data.applied_torque`` cheerfully reports the effort limit
(that field is Isaac Lab's own analytic estimate, not a PhysX readback).

Measured on this machine with a 2 s open-loop full-speed command: acceleration drive gives
0.000 m travelled and a wheel speed of 6e-9 rad/s; force drive gives 1.139 m, a wheel speed of
18.59 rad/s, a body speed of 0.592 m/s and 1.09 mm of lateral drift, i.e. 0.96 mm/m against the
M1 acceptance bound of 20 mm/m.

The startup warning about a "non-existent path" under ``visuals`` is COSMETIC
------------------------------------------------------------------------------
Every launch that spawns this robot logs exactly one pair of lines that looks alarming and is
not::

    [Warning] [omni.fabric.plugin] getAttributeCount called on non-existent path
      /World/envs/env_<N-1>/Robot/base_link/visuals/chassis_visual
    [Warning] [omni.fabric.plugin] getTypes called on non-existent path  ...same...

**What it is.** ``assets/usd/duckiebot.usda`` gives every link a ``visuals`` Xform carrying
``instanceable = true`` and a reference to a flattened prototype (the URDF importer authors it
that way, and it is the reason 256 robots cost one copy of the geometry instead of 256). The
children of an instanceable prim are therefore *instance proxies*: ``chassis_visual`` and its
siblings are composed and renderable through the instance, but the real prims live under
``/__Prototype_<k>`` and no prim is authored at the proxy path. USD's Fabric mirror is populated
from the real scene index, so a Fabric lookup keyed on the proxy path finds nothing and logs
this. It is a lookup miss on a path that is not supposed to be in Fabric, not a missing robot
part.

**What was measured** (Isaac Sim 5.1, Isaac Lab 2.3.2, RTX 3080 Laptop, headless + cameras):

* The census over ``--num_envs`` 2, 4, 6, 8 gives ``env_1``, ``env_3``, ``env_5``, ``env_7``:
  always the LAST environment, never any other, and always exactly ONE pair of lines per
  process, independent of ``N``. The prim it names varies between runs (``chassis_visual`` at
  N = 6 and 8, ``right_wheel_visual`` at N = 2 and 4), which is the signature of a single
  one-shot probe rather than of a per-env or per-prim traversal.
* It is emitted after ``_setup_scene`` returns and before ``DirectRLEnv.__init__`` finishes,
  i.e. inside ``sim.reset()``, the call that starts physics and populates Fabric for the first
  time.
* Clearing ``instanceable`` on the three ``visuals`` prims immediately before ``sim.reset()``
  makes the warning disappear entirely, which pins the cause on the instancing and on nothing
  else. It also costs 5.23 s instead of 4.05 s of environment construction at N = 4 (+29%) and
  leaves the observation unchanged (mean 119.83 against 119.77), so it is a worse trade at every
  N and is deliberately NOT done.
* The last environment is not degraded. Its ``chassis_visual`` resolves
  (``prim.IsInstanceProxy()`` is True, the world bound is non-empty, and its env-local centre and
  size match ``env_0`` to the last printed digit); the Fabric path IS present when queried
  through ``usdrt`` after ``sim.reset()``; and a third-person RTX render of ``env_0`` and
  ``env_3`` placed at an identical env-local pose on an identical city stage shows the same
  complete robot -- chassis box, marker sphere, camera housing and caster -- in both.
* ``scripts/check_obs.py`` at N = 6 puts the warned environment's camera squarely in family with
  the others (mean 126.8/121.2/98.8 against a 74..127 spread, 781 yellow-tape pixels, no black
  frame, frame ring advancing).

**Therefore:** ignore it. If it ever needs re-checking, the reproduction is
``scripts/check_obs.py --num_envs <N>`` and the thing to verify is that the named environment is
still ``env_{N-1}`` and that its row of the contact sheet still looks like the others. A warning
that starts naming a MIDDLE environment, or that fires more than once, would mean something
genuinely changed and is worth chasing.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from duckiebot_rl.assets.params import DUCKIEBOT, DuckiebotParams

if TYPE_CHECKING:  # pragma: no cover - typing only, never executed at runtime
    from isaaclab.assets import ArticulationCfg

__all__ = [
    "DEFAULT_PRIM_PATH",
    "DEFAULT_USD_PATH",
    "ISAACLAB_HINT",
    "RELATIVE_USD_PATH",
    "camera_mount_spec",
    "duckiebot_articulation_cfg",
    "get_duckiebot_cfg",
    "is_isaaclab_available",
    "physics_material_spec",
    "spawn_property_spec",
    "wheel_actuator_spec",
]

RELATIVE_USD_PATH = "assets/usd/duckiebot.usda"
"""Path of the imported robot USD relative to the repository root.

Text ``.usda``, not binary ``.usd``, because SPEC v2 S3.4 rule 1 bans ``.usd``, ``.usdc`` and
``.usdz`` repository-wide and ``scripts/check_clean_room.py`` enforces that on every push: the
Isaac Sim URDF importer writes binary USDC by default, so ``tools/import_urdf_headless.py``
flattens its output into one text layer here and deletes the binary staging directory. A robot
built the other way turns the licensing gate red the moment anyone runs the build.
"""

DEFAULT_USD_PATH = (Path(__file__).resolve().parents[2] / RELATIVE_USD_PATH).as_posix()
"""Absolute path of the imported robot USD, with forward slashes.

Absolute on purpose. Isaac Lab hands a relative ``usd_path`` to the USD asset resolver, which
resolves it against the process working directory, and the supported launch (``& $ISAAC
scripts/train.py ...``) runs from wherever the shell happens to be. A relative path fails there
as an opaque stage-open error that names USD rather than this constant.

``assets/usd/`` is gitignored (SPEC v2 S3.2): the USD is a build artifact, rebuilt from the URDF
by ``tools/import_urdf_headless.py``, which runs ``tools/patch_usd.py`` and ``tools/verify_usd.py``
in the same command. Only the text ``.urdf`` and text ``.usda`` sources are ever committed.
"""

DEFAULT_PRIM_PATH = "{ENV_REGEX_NS}/Robot"
"""Default scene prim path for the robot, matching the S5.1 scene graph."""

ISAACLAB_HINT = (
    "Isaac Lab is not importable from this interpreter. This module's config builders only run "
    "inside the Isaac Sim python environment. On this machine that is "
    "d:/Personal/personal/wheeled_quadruped_robot/.venv/Scripts/python.exe, and the AppLauncher "
    "must have been constructed before any isaaclab.assets import. CPU-only tooling should use "
    "wheel_actuator_spec(), spawn_property_spec(), physics_material_spec() and "
    "camera_mount_spec() instead, which return plain dictionaries."
)

# PhysX solver iteration counts. Two bodies on three contact points is a trivial articulation;
# 8 position iterations is generous and 0 velocity iterations is the Isaac Lab wheeled default.
_SOLVER_POSITION_ITERATIONS = 8
_SOLVER_VELOCITY_ITERATIONS = 0
# Contact generation distance in metres. See the module docstring: must stay well under the 21 mm
# ground clearance, and above the 4.2 mm a wheel contact point travels in one 1/240 s substep at
# the 1.0 m/s worst case.
_CONTACT_OFFSET_M = 0.005
_REST_OFFSET_M = 0.0


_LAST_IMPORT_ERROR: Exception | None = None
"""The exception raised by the most recent failed ``isaaclab.assets`` import.

Inside the Isaac venv the usual failure is not absence: it is that the ``AppLauncher`` has not
been constructed yet, or that Kit or carb failed to initialise. Keeping the original exception
lets :func:`_require_isaaclab` chain it so the real cause survives in the traceback instead of
being replaced by a message about a missing package.
"""


def is_isaaclab_available() -> bool:
    """Report whether the Isaac Lab config classes can be imported in this interpreter.

    Returns:
        ``True`` if ``isaaclab.assets`` imports cleanly, ``False`` otherwise. Never raises.
    """
    global _LAST_IMPORT_ERROR
    try:  # pragma: no cover - the branch taken depends entirely on the interpreter
        import isaaclab.assets  # noqa: F401
    except Exception as error:
        _LAST_IMPORT_ERROR = error
        return False
    _LAST_IMPORT_ERROR = None
    return True


def _require_isaaclab() -> None:
    """Raise a useful error if Isaac Lab is missing.

    Raises:
        ImportError: If ``isaaclab`` cannot be imported, with :data:`ISAACLAB_HINT` as the
            message and the original import failure chained as its cause.
    """
    if not is_isaaclab_available():
        raise ImportError(ISAACLAB_HINT) from _LAST_IMPORT_ERROR


def wheel_actuator_spec(params: DuckiebotParams = DUCKIEBOT) -> dict[str, Any]:
    """Return the wheel actuator numbers as a plain dictionary.

    The keys are exactly the ``ImplicitActuatorCfg`` field names, so
    :func:`duckiebot_articulation_cfg` can splat this straight into the constructor and CI can
    assert on the same values without importing Isaac Lab.

    Args:
        params: Parameter set to read. Defaults to the shared singleton.

    Returns:
        Mapping of ``ImplicitActuatorCfg`` field name to value.
    """
    return {
        "joint_names_expr": [params.wheel_joint_regex],
        "effort_limit_sim": params.wheel_effort_limit_nm,
        "velocity_limit_sim": params.wheel_velocity_limit_rad_s,
        "stiffness": params.joint_stiffness,
        "damping": params.joint_damping,
        "armature": params.joint_armature_kg_m2,
        "friction": params.joint_friction_nm,
    }


def spawn_property_spec(params: DuckiebotParams = DUCKIEBOT) -> dict[str, dict[str, Any]]:
    """Return the USD spawn property blocks as plain dictionaries.

    Args:
        params: Parameter set to read. Defaults to the shared singleton.

    Returns:
        Mapping with keys ``rigid_props``, ``articulation_props``, ``collision_props`` and
        ``joint_drive_props``, each holding the field names of the corresponding Isaac Lab
        schema config.
    """
    return {
        "rigid_props": {
            "disable_gravity": False,
            "retain_accelerations": False,
            "linear_damping": 0.0,
            "angular_damping": 0.0,
            "max_linear_velocity": 5.0,
            # deg/s in Isaac Lab. 4000 deg/s = 69.8 rad/s, twice the wheel velocity limit.
            "max_angular_velocity": 4000.0,
            "max_depenetration_velocity": 1.0,
            "enable_gyroscopic_forces": True,
            "solver_position_iteration_count": _SOLVER_POSITION_ITERATIONS,
            "solver_velocity_iteration_count": _SOLVER_VELOCITY_ITERATIONS,
        },
        "articulation_props": {
            "articulation_enabled": True,
            "enabled_self_collisions": False,
            "fix_root_link": False,
            "solver_position_iteration_count": _SOLVER_POSITION_ITERATIONS,
            "solver_velocity_iteration_count": _SOLVER_VELOCITY_ITERATIONS,
            # A 1.1 kg robot that stops moving must not be put to sleep: the policy keeps
            # commanding it and PhysX would otherwise ignore the drive targets for a few steps.
            "sleep_threshold": 0.0,
            "stabilization_threshold": 0.0,
        },
        "collision_props": {
            "collision_enabled": True,
            "contact_offset": _CONTACT_OFFSET_M,
            "rest_offset": _REST_OFFSET_M,
        },
        "joint_drive_props": {
            # MUST be "force". See the module docstring: the URDF importer writes an
            # acceleration drive, and with a 2.25e-4 kg.m2 wheel the resulting torque is three
            # orders of magnitude below the 0.010 N.m joint friction, so the wheels never turn.
            "drive_type": "force",
        },
    }


def physics_material_spec(params: DuckiebotParams = DUCKIEBOT) -> dict[str, dict[str, Any]]:
    """Return the two rigid-body physics materials the robot needs, with their binding selectors.

    ``tools/patch_usd.py`` binds these after URDF import (SPEC v2 S3.2). They are returned as
    data rather than as ``RigidBodyMaterialCfg`` objects for two reasons: the patch script must
    run without constructing an Isaac Lab config, and ``improve_patch_friction`` is not a field of
    ``RigidBodyMaterialCfg`` in Isaac Lab 2.3.2, so the patch script sets that PhysX attribute
    directly on the material prim.

    Binding contract
    ----------------
    Three keys define which colliders a material lands on, and ``tools/patch_usd.py`` is the only
    implementation of them:

    * ``bind_to`` is a list of LINK names, never collision-element names. Every collider owned by
      those links is a candidate.
    * ``restrict_to_shape`` is ``None``, or a mapping that narrows the candidates by ``kind`` and,
      for a sphere, ``radius_m`` and ``center_base_frame_m`` within ``position_tol_m``. The caster
      needs this because ``base_link`` owns two colliders that must not share a material: the
      chassis box and the caster ball.
    * ``expect_matches`` is the exact number of colliders the selector must resolve to. The patch
      script raises rather than binding a different number, which is what makes a mis-resolved
      selector a loud build failure instead of a caster silently left on the PhysX 0.5/0.5
      default and a robot that just turns badly forever.

    Geometry, not prim names, is the selector for the caster on purpose. The Isaac Sim 5.1
    importer does derive collider prim names from the URDF ``<collision name>`` attribute
    (``/colliders/base_link/caster_collision/sphere`` for this asset), but that is importer
    behaviour rather than a documented contract, and it is not shared with any other consumer.
    Radius and centre come from :mod:`duckiebot_rl.assets.params`, which every consumer reads.

    Combine modes are a pair, not a property of one material. PhysX resolves a contact with the
    stricter of the two modes involved, in the order ``max`` beats ``min`` beats ``average``. The
    caster's ``min`` therefore only wins while the ground plane leaves its combine mode unset, as
    ``assets/usd/ground.usda`` does today. If the ground is ever given ``max`` to pair with the
    wheels, the caster becomes mu = 1.0 with no error anywhere: the ground material and this one
    have to be changed together.

    Args:
        params: Parameter set to read. Defaults to the shared singleton.

    Returns:
        Mapping from material name to its properties and binding selector.
    """
    return {
        "duckiebot_wheel_material": {
            "static_friction": params.wheel_friction_static,
            "dynamic_friction": params.wheel_friction_dynamic,
            "restitution": 0.0,
            "friction_combine_mode": "max",
            "restitution_combine_mode": "min",
            "improve_patch_friction": True,
            "bind_to": [params.left_wheel_link_name, params.right_wheel_link_name],
            "restrict_to_shape": None,
            "expect_matches": 2,
        },
        "duckiebot_caster_material": {
            "static_friction": params.caster_friction,
            "dynamic_friction": params.caster_friction,
            "restitution": 0.0,
            "friction_combine_mode": "min",
            "restitution_combine_mode": "min",
            "improve_patch_friction": False,
            "bind_to": [params.base_link_name],
            "restrict_to_shape": {
                "kind": "sphere",
                "radius_m": params.caster_radius_m,
                "center_base_frame_m": params.caster_center_base_frame_m,
                "position_tol_m": 1.0e-4,
            },
            "expect_matches": 1,
        },
    }


def camera_mount_spec(params: DuckiebotParams = DUCKIEBOT) -> dict[str, Any]:
    """Return the camera mount pose as data, for the environment's ``TiledCameraCfg``.

    This is the single source of the mount pose (SPEC v2 S2, resolving critic item F). The URDF
    has no ``camera_link``; the rotation is produced from ``pitch_down_deg`` by
    ``duckiebot_rl.camera_math.quat_cam_ros`` and by nothing else, in all three consumers
    (Isaac, MuJoCo, deployment).

    Args:
        params: Parameter set to read. Defaults to the shared singleton.

    Returns:
        Mapping with the parent prim path, the offset position, the pitch scalar, the offset
        convention and the canonical render resolution.
    """
    return {
        "parent_prim_path": f"{DEFAULT_PRIM_PATH}/{params.base_link_name}/front_cam",
        "offset_pos": params.camera_pos_base_frame_m,
        "pitch_down_deg": params.camera_pitch_down_deg,
        "convention": "ros",
        "width": params.render_width_px,
        "height": params.render_height_px,
        "focal_length": params.camera_focal_length_mm,
        "horizontal_aperture": params.camera_horizontal_aperture_mm,
        "vertical_aperture": params.camera_vertical_aperture_mm,
        "clipping_range": params.camera_clipping_range_m,
    }


def duckiebot_articulation_cfg(
    usd_path: str = DEFAULT_USD_PATH,
    prim_path: str = DEFAULT_PRIM_PATH,
    params: DuckiebotParams = DUCKIEBOT,
) -> ArticulationCfg:
    """Build the Isaac Lab :class:`ArticulationCfg` for the Duckiebot.

    Args:
        usd_path: Path to the imported and patched robot USD.
        prim_path: Scene prim path, normally containing the ``{ENV_REGEX_NS}`` placeholder.
        params: Parameter set to read. Defaults to the shared singleton.

    Returns:
        A fully populated ``ArticulationCfg`` with one ``ImplicitActuatorCfg`` covering both
        wheel joints.

    Raises:
        ImportError: If Isaac Lab is not importable in this interpreter.
    """
    _require_isaaclab()

    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg

    spawn_props = spawn_property_spec(params)
    spawn = sim_utils.UsdFileCfg(
        usd_path=usd_path,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(**spawn_props["rigid_props"]),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(**spawn_props["articulation_props"]),
        collision_props=sim_utils.CollisionPropertiesCfg(**spawn_props["collision_props"]),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(**spawn_props["joint_drive_props"]),
        # Each env references the same source layer instead of flattening a private copy of it.
        # At N = 256 with replicate_physics=False that is the difference between a stage build
        # measured in seconds and one measured in minutes (the M6 gate allows 5 minutes total).
        copy_from_source=False,
    )
    init_state = ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, params.base_link_height_m),
        rot=(1.0, 0.0, 0.0, 0.0),
        lin_vel=(0.0, 0.0, 0.0),
        ang_vel=(0.0, 0.0, 0.0),
        joint_pos={params.wheel_joint_regex: 0.0},
        joint_vel={params.wheel_joint_regex: 0.0},
    )
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=spawn,
        init_state=init_state,
        soft_joint_pos_limit_factor=1.0,
        actuators={"wheels": ImplicitActuatorCfg(**wheel_actuator_spec(params))},
    )


_DUCKIEBOT_CFG_SINGLETON: ArticulationCfg | None = None
"""Memoized default config. See :func:`get_duckiebot_cfg` for why it has to be memoized."""


def get_duckiebot_cfg() -> ArticulationCfg:
    """Return the shared default :class:`ArticulationCfg`, building it on first use.

    Isaac Lab's own assets expose a module-level constant (``ANYMAL_C_CFG``, ``CARTPOLE_CFG``),
    and consumers follow that idiom by mutating it in place or by keeping a reference across
    modules. This module cannot build the config at import time, because that would require Isaac
    Lab in every CPU-only test run, so it memoizes instead: every caller gets the same object, and
    an in-place edit sticks. Callers who want a private copy should use
    :func:`duckiebot_articulation_cfg`, which builds a fresh one on every call.

    Returns:
        The shared config instance.

    Raises:
        ImportError: If Isaac Lab is not importable in this interpreter.
    """
    global _DUCKIEBOT_CFG_SINGLETON
    if _DUCKIEBOT_CFG_SINGLETON is None:
        _DUCKIEBOT_CFG_SINGLETON = duckiebot_articulation_cfg()
    return _DUCKIEBOT_CFG_SINGLETON


# ANN401: a module-level __getattr__ is typed (str) -> Any by the language, not by choice.
def __getattr__(name: str) -> Any:
    """Serve ``DUCKIEBOT_CFG`` lazily so importing this module never needs Isaac Lab.

    Args:
        name: Attribute being looked up.

    Returns:
        The shared :class:`ArticulationCfg` when ``name`` is ``"DUCKIEBOT_CFG"``. It is the same
        object on every access, so ``DUCKIEBOT_CFG.prim_path = ...`` behaves the way the Isaac Lab
        idiom promises.

    Raises:
        AttributeError: For any other name.
        ImportError: If ``DUCKIEBOT_CFG`` is requested without Isaac Lab available.
    """
    if name == "DUCKIEBOT_CFG":
        return get_duckiebot_cfg()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
