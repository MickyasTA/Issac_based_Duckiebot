"""Tests for :mod:`duckiebot_rl.assets.robot_cfg`, the module that produces the ArticulationCfg.

``tests/unit/test_params.py`` covers the plain-dictionary views of this module. What it cannot
cover on a CPU runner is the module's actual product, the ``ArticulationCfg``, and the two
contracts around it that fail silently rather than loudly:

* the USD path handed to ``UsdFileCfg`` must be absolute, because Isaac Lab resolves a relative
  one against the process working directory and the supported launch does not run from the
  repository root;
* the physics-material binding selectors must name links and shapes that ``tools/patch_usd.py``
  can actually resolve, because a selector that resolves to nothing leaves the caster on the
  PhysX 0.5/0.5 default and no simulation ever reports it.

The ``ArticulationCfg`` construction itself is marked ``isaac`` and runs with ``--run-isaac``
inside the Isaac Sim python. It is the only place that proves the plain dictionaries splat
cleanly into the real Isaac Lab config classes, which is what breaks on an Isaac Lab bump.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

# The repository root goes on the path before the package imports: conftest.py does this for the
# pytest run, but this module is also executed directly as the Isaac Lab probe (see __main__).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from duckiebot_rl.assets.params import DUCKIEBOT  # noqa: E402
from duckiebot_rl.assets.robot_cfg import (  # noqa: E402
    DEFAULT_PRIM_PATH,
    DEFAULT_USD_PATH,
    ISAACLAB_HINT,
    RELATIVE_USD_PATH,
    duckiebot_articulation_cfg,
    is_isaaclab_available,
    physics_material_spec,
    spawn_property_spec,
    wheel_actuator_spec,
)

ISAAC_PROBE_OK = "ISAAC_PROBE_OK"
"""Sentinel the ``__main__`` probe prints when the built config matches the dictionaries."""

needs_isaac = pytest.mark.isaac


# =============================================================================================
# The USD path contract
# =============================================================================================


def test_the_default_usd_path_is_absolute() -> None:
    """A relative usd_path resolves against the CWD, which is not the repository root."""
    assert Path(DEFAULT_USD_PATH).is_absolute()


def test_the_default_usd_path_uses_forward_slashes() -> None:
    """USD asset paths are forward-slashed on every platform, including Windows."""
    assert "\\" not in DEFAULT_USD_PATH


def test_the_default_usd_path_points_into_this_repository() -> None:
    """The absolute path must be this checkout's asset, not a machine-specific string."""
    assert Path(DEFAULT_USD_PATH) == (_REPO_ROOT / RELATIVE_USD_PATH)


def test_the_relative_usd_path_is_the_gitignored_build_directory() -> None:
    """SPEC v2 S3.2: the robot USD is a build artifact under assets/usd/, never committed."""
    assert RELATIVE_USD_PATH.startswith("assets/usd/")
    gitignore = (_REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "assets/usd/" in gitignore


def test_the_prim_path_carries_the_env_namespace_placeholder() -> None:
    """Isaac Lab clones the scene by expanding this token; a literal path spawns one robot."""
    assert "{ENV_REGEX_NS}" in DEFAULT_PRIM_PATH


# =============================================================================================
# The physics-material binding contract that tools/patch_usd.py implements
# =============================================================================================


def test_every_material_binds_to_link_names_not_collision_element_names() -> None:
    """``bind_to`` is a list of links. Mixing in a ``<collision name>`` has no resolution rule."""
    link_names = {
        DUCKIEBOT.base_link_name,
        DUCKIEBOT.left_wheel_link_name,
        DUCKIEBOT.right_wheel_link_name,
    }
    for name, spec in physics_material_spec().items():
        assert set(spec["bind_to"]) <= link_names, f"{name} binds to something that is not a link"


def test_the_wheel_material_binds_to_both_wheels_and_expects_two_matches() -> None:
    """One collider per wheel: the count is what makes a mis-resolved selector an error."""
    wheel = physics_material_spec()["duckiebot_wheel_material"]
    assert set(wheel["bind_to"]) == {
        DUCKIEBOT.left_wheel_link_name,
        DUCKIEBOT.right_wheel_link_name,
    }
    assert wheel["expect_matches"] == 2
    assert wheel["restrict_to_shape"] is None


def test_the_caster_material_is_narrowed_to_the_caster_sphere_by_geometry() -> None:
    """base_link owns two colliders; only the ball may become frictionless, never the chassis."""
    caster = physics_material_spec()["duckiebot_caster_material"]
    assert caster["bind_to"] == [DUCKIEBOT.base_link_name]
    assert caster["expect_matches"] == 1
    shape = caster["restrict_to_shape"]
    assert shape["kind"] == "sphere"
    assert shape["radius_m"] == DUCKIEBOT.caster_radius_m
    assert shape["center_base_frame_m"] == DUCKIEBOT.caster_center_base_frame_m
    assert 0.0 < shape["position_tol_m"] < 0.001


def test_the_caster_selector_cannot_match_the_chassis_box() -> None:
    """The chassis half-height is 37.5 mm; the caster radius is 16.5 mm. Nothing can alias."""
    shape = physics_material_spec()["duckiebot_caster_material"]["restrict_to_shape"]
    chassis_center = DUCKIEBOT.chassis_center_base_frame_m
    separation = max(abs(a - b) for a, b in zip(shape["center_base_frame_m"], chassis_center, strict=True))
    assert separation > shape["position_tol_m"]


def test_the_two_materials_cover_every_collider_exactly_once() -> None:
    """Three of the four colliders are claimed; the fourth is the chassis, deliberately unclaimed."""
    total = sum(spec["expect_matches"] for spec in physics_material_spec().values())
    assert total == 3


# =============================================================================================
# The Isaac Lab boundary
# =============================================================================================


@pytest.mark.skipif(
    is_isaaclab_available(), reason="Isaac Lab is importable here; run the isaac test instead"
)
def test_building_the_cfg_without_isaac_lab_raises_a_useful_error() -> None:
    """The error has to point at the interpreter and the AppLauncher, not at a missing file."""
    with pytest.raises(ImportError) as excinfo:
        duckiebot_articulation_cfg()
    assert str(excinfo.value) == ISAACLAB_HINT
    assert "AppLauncher" in ISAACLAB_HINT


@needs_isaac
def test_the_articulation_cfg_builds_inside_isaac_lab() -> None:
    """Run the Isaac Lab probe in a child process and require its success sentinel.

    The probe cannot run inside pytest: booting Kit makes the interpreter exit through Kit's own
    teardown, which returns status 139 on this install no matter how the tests went, so the
    pytest exit code would stop meaning anything. The child does the Isaac work and prints
    :data:`ISAAC_PROBE_OK`; this test owns the verdict.
    """
    child = subprocess.run(
        [sys.executable, str(Path(__file__).resolve())],
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert ISAAC_PROBE_OK in child.stdout, (
        f"the Isaac Lab probe did not succeed.\nstdout:\n{child.stdout}\nstderr:\n{child.stderr}"
    )


def isaac_cfg_problems() -> list[str]:
    """Compare the built ``ArticulationCfg`` against the plain dictionaries, inside Isaac Lab.

    Isaac Lab must already be importable, which means an ``AppLauncher`` has been constructed.

    Returns:
        One description per mismatch. Empty when the config is exactly what the dictionaries say.
    """
    from duckiebot_rl.assets.robot_cfg import get_duckiebot_cfg

    problems: list[str] = []

    def compare(obj: object, expected: dict[str, object], label: str) -> None:
        """Record a problem for every field of ``expected`` the object does not match."""
        for field, value in expected.items():
            actual = getattr(obj, field, "<missing>")
            if actual != value:
                problems.append(f"{label}.{field} is {actual!r}, expected {value!r}")

    cfg = get_duckiebot_cfg()
    compare(cfg.actuators["wheels"], wheel_actuator_spec(), "actuators.wheels")
    spawn_props = spawn_property_spec()
    compare(cfg.spawn.rigid_props, spawn_props["rigid_props"], "spawn.rigid_props")
    compare(cfg.spawn.articulation_props, spawn_props["articulation_props"], "spawn.articulation_props")
    compare(cfg.spawn.collision_props, spawn_props["collision_props"], "spawn.collision_props")

    if cfg.spawn.usd_path != DEFAULT_USD_PATH:
        problems.append(f"spawn.usd_path is {cfg.spawn.usd_path!r}, expected {DEFAULT_USD_PATH!r}")
    if cfg.prim_path != DEFAULT_PRIM_PATH:
        problems.append(f"prim_path is {cfg.prim_path!r}, expected {DEFAULT_PRIM_PATH!r}")
    if cfg.init_state.pos != (0.0, 0.0, DUCKIEBOT.base_link_height_m):
        problems.append(f"init_state.pos is {cfg.init_state.pos!r}, expected the wheel radius")
    if cfg.spawn.copy_from_source is not False:
        problems.append("spawn.copy_from_source must stay False: 256 envs share one source layer")

    # The shared instance must be one object (the Isaac Lab idiom mutates it in place), while the
    # builder must hand out independent copies.
    if get_duckiebot_cfg() is not cfg:
        problems.append("get_duckiebot_cfg() returned a different object on the second call")
    if duckiebot_articulation_cfg() is duckiebot_articulation_cfg():
        problems.append("duckiebot_articulation_cfg() must build a fresh config on every call")
    return problems


def isaac_spawn_problems() -> list[str]:
    """Spawn the built asset in a real PhysX scene and read its structure back.

    This is the only check anywhere that proves the file ``tools/import_urdf_headless.py``
    produces actually loads: ``tools/verify_usd.py`` reads the USD, this reads what PhysX made of
    it. It is off unless ``DUCKIEBOT_RL_ISAAC_SPAWN=1`` is set, because it needs a GPU-backed
    PhysX scene and takes minutes when the machine is busy, while the rest of the probe is a
    seven-second import check. It is also skipped, with a recorded note, when the asset has not
    been built, because ``assets/usd/`` is gitignored.

    Returns:
        One description per mismatch, empty when the spawned articulation is the M1 robot.
    """
    if os.environ.get("DUCKIEBOT_RL_ISAAC_SPAWN", "0") in ("", "0", "false", "False"):
        print("ISAAC_PROBE_NOTE spawn skipped: set DUCKIEBOT_RL_ISAAC_SPAWN=1 to run it", flush=True)
        return []
    if not Path(DEFAULT_USD_PATH).is_file():
        print(f"ISAAC_PROBE_NOTE spawn skipped: {DEFAULT_USD_PATH} is not built", flush=True)
        return []

    from isaaclab.assets import Articulation
    from isaaclab.sim import SimulationCfg, SimulationContext

    from duckiebot_rl.assets.robot_cfg import duckiebot_articulation_cfg as build_cfg

    problems: list[str] = []
    sim = SimulationContext(SimulationCfg(dt=DUCKIEBOT.sim_dt_s, device="cpu"))
    robot = Articulation(build_cfg(prim_path="/World/Robot"))
    sim.reset()

    expected_bodies = [
        DUCKIEBOT.base_link_name,
        DUCKIEBOT.left_wheel_link_name,
        DUCKIEBOT.right_wheel_link_name,
    ]
    if sorted(robot.body_names) != sorted(expected_bodies):
        problems.append(f"PhysX reports bodies {robot.body_names}, expected {expected_bodies}")
    expected_joints = [DUCKIEBOT.left_wheel_joint_name, DUCKIEBOT.right_wheel_joint_name]
    if sorted(robot.joint_names) != sorted(expected_joints):
        problems.append(f"PhysX reports joints {robot.joint_names}, expected {expected_joints}")

    masses = robot.root_physx_view.get_masses()[0].tolist()
    by_name = dict(zip(robot.body_names, masses, strict=True))
    expected_masses = {
        DUCKIEBOT.base_link_name: DUCKIEBOT.base_mass_kg,
        DUCKIEBOT.left_wheel_link_name: DUCKIEBOT.wheel_mass_kg,
        DUCKIEBOT.right_wheel_link_name: DUCKIEBOT.wheel_mass_kg,
    }
    for name, expected in expected_masses.items():
        actual = by_name.get(name)
        if actual is None or abs(actual - expected) > 1e-5:
            problems.append(f"PhysX mass of {name} is {actual}, expected {expected}")
    return problems


def _run_isaac_probe() -> int:
    """Boot Isaac Lab headless, check the config, print the verdict.

    Returns:
        ``0`` when the config matches the dictionaries, ``1`` otherwise. The exit status is not
        reliable once Kit has booted, so the printed sentinel is the real result.
    """
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=True)
    problems = isaac_cfg_problems() + isaac_spawn_problems()
    if problems:
        for problem in problems:
            print(f"ISAAC_PROBE_PROBLEM {problem}", flush=True)
        print("ISAAC_PROBE_FAILED", flush=True)
    else:
        print(ISAAC_PROBE_OK, flush=True)
    launcher.app.close()
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(_run_isaac_probe())
