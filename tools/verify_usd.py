"""Verify the imported and patched Duckiebot USD against the SPEC v2 M1 acceptance quantities.

This is the M1 gate of SPEC v2 S11:

    verify_usd.py exit 0 (3 bodies / 2 DOF; masses 1.00/0.05/0.05; caster sphere r 0.0165 with
    center height = radius; chassis box bottom 0.021)

plus the S3.2 and S3.4 structural requirements that nothing else in the repository checks: the
wheel colliders must be spheres and never cylinders, the caster must carry the frictionless
physics material, and the robot must contain no ``Mesh`` prim at all.

Layout, and why it is split this way
------------------------------------
Two layers, on purpose:

* :func:`extract_scene` needs USD. It walks a composed stage (including instance proxies, which
  is where the URDF importer puts every collider) and flattens it into :class:`RobotScene`, a
  plain dataclass tree of floats and strings that round-trips through JSON.
* :func:`check_scene` is pure Python. It takes a :class:`RobotScene` and the parameter set and
  returns one :class:`CheckResult` per acceptance criterion. It imports nothing but the standard
  library and :mod:`duckiebot_rl.assets.params`.

``tests/unit/test_verify_usd.py`` therefore tests every assertion on synthetic scenes on a CPU
runner with no Isaac Sim and no USD installed, and additionally re-tests the extractor against a
hand-authored ``.usda`` whenever USD happens to be importable.

Usage (from the repository root):

.. code-block:: text

    python tools/verify_usd.py                          # verifies the default asset path
    python tools/verify_usd.py assets/usd/duckiebot.usd
    python tools/verify_usd.py --json report.json       # machine readable, still exits non-zero

Exit codes: ``0`` every check passed, ``1`` at least one check failed, ``2`` the stage could not
be opened (missing file, no USD runtime, no default prim).

Where USD comes from on this machine
------------------------------------
``pxr`` is not a pip dependency of this repository and is not importable out of the box from
either venv. :func:`ensure_pxr` finds the complete USD build that Isaac Sim 5.1 ships inside
``isaacsim/extscache/omni.usd.libs-*`` and makes it importable without booting Kit, which turns
verification into a sub-second operation. If usd-core is installed instead (the tools venv), that
is used as is.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import sysconfig
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from duckiebot_rl.assets.params import DUCKIEBOT, DuckiebotParams  # noqa: E402
from duckiebot_rl.assets.robot_cfg import DEFAULT_USD_PATH, physics_material_spec  # noqa: E402

__all__ = [
    "CheckResult",
    "Collider",
    "Joint",
    "PhysicsMaterial",
    "RigidBody",
    "RobotScene",
    "StageOpenError",
    "check_scene",
    "ensure_pxr",
    "extract_scene",
    "format_report",
    "main",
    "open_stage",
    "verify_usd_file",
]

# ---------------------------------------------------------------------------------------------
# Tolerances. The USD attributes the importer writes are float32, so 0.05 kg arrives as
# 0.05000000074505806: every tolerance below is comfortably above float32 round-off at these
# magnitudes and far below any quantity a real modelling error would move.
# ---------------------------------------------------------------------------------------------
LENGTH_TOL_M = 1.0e-6
"""Absolute tolerance on any length or coordinate, in metres (one micrometre)."""

MASS_TOL_KG = 1.0e-6
"""Absolute tolerance on a mass, in kilograms."""

INERTIA_REL_TOL = 1.0e-5
"""Relative tolerance on an inertia tensor entry, scaled by the largest expected entry."""

FRICTION_TOL = 1.0e-6
"""Absolute tolerance on a friction coefficient. The values are exact 0.0 and 1.0 in float32."""

PHYSICS_BINDING_PURPOSE = "physics"
"""Material-binding purpose token PhysX reads: the relationship is ``material:binding:physics``."""

PHYSX_FRICTION_COMBINE_ATTR = "physxMaterial:frictionCombineMode"
"""PhysX friction combine mode attribute, from the PhysxMaterialAPI schema."""

PHYSX_RESTITUTION_COMBINE_ATTR = "physxMaterial:restitutionCombineMode"
"""PhysX restitution combine mode attribute."""

PHYSX_IMPROVE_PATCH_ATTR = "physxMaterial:improvePatchFriction"
"""PhysX patch-friction flag: needed for a stable contact patch under a rolling sphere."""

_ROUND_COLLIDER_KINDS = frozenset({"cylinder", "capsule", "cone"})
"""Collider shapes S3.2 forbids: their contact patch degenerates to a line on flat ground."""


class StageOpenError(RuntimeError):
    """Raised when the USD stage cannot be opened or contains no robot."""


# =============================================================================================
# Scene description: the flattened, USD-free view of the asset
# =============================================================================================


def _as_tuple3(values: Any) -> tuple[float, float, float]:
    """Coerce any 3-element sequence to a tuple of floats.

    Args:
        values: Any indexable of length 3.

    Returns:
        ``(x, y, z)`` as floats.
    """
    return (float(values[0]), float(values[1]), float(values[2]))


@dataclass(frozen=True)
class RigidBody:
    """One PhysX rigid body found on the stage.

    Attributes:
        prim_path: Absolute prim path.
        name: Prim name, which the URDF importer sets to the URDF link name.
        mass_kg: ``physics:mass``.
        com_root_m: ``physics:centerOfMass``, in the body's own frame.
        diagonal_inertia_kg_m2: ``physics:diagonalInertia``.
        principal_axes_wxyz: ``physics:principalAxes`` as ``(w, x, y, z)``.
        translate_root_m: Body origin expressed in the robot root frame.
    """

    prim_path: str
    name: str
    mass_kg: float
    com_root_m: tuple[float, float, float]
    diagonal_inertia_kg_m2: tuple[float, float, float]
    principal_axes_wxyz: tuple[float, float, float, float]
    translate_root_m: tuple[float, float, float]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RigidBody:
        """Rebuild a body from its :func:`dataclasses.asdict` form.

        Args:
            data: Mapping produced by ``asdict``.

        Returns:
            The reconstructed body.
        """
        axes = data["principal_axes_wxyz"]
        return cls(
            prim_path=str(data["prim_path"]),
            name=str(data["name"]),
            mass_kg=float(data["mass_kg"]),
            com_root_m=_as_tuple3(data["com_root_m"]),
            diagonal_inertia_kg_m2=_as_tuple3(data["diagonal_inertia_kg_m2"]),
            principal_axes_wxyz=(float(axes[0]), float(axes[1]), float(axes[2]), float(axes[3])),
            translate_root_m=_as_tuple3(data["translate_root_m"]),
        )


@dataclass(frozen=True)
class Collider:
    """One prim carrying ``UsdPhysics.CollisionAPI``.

    All geometry is resolved to metres in the robot root frame, with the prim's own transform
    scale already folded into the radius or half extents. ``body`` is the name of the nearest
    ancestor rigid body, which is the link the shape actually belongs to.

    Attributes:
        prim_path: Absolute prim path (an instance-proxy path when the importer instanced it).
        body: Name of the owning rigid body.
        kind: ``sphere``, ``box``, ``cylinder``, ``capsule``, ``cone``, ``mesh`` or ``other``.
        center_root_m: Shape centre in the robot root frame.
        radius_m: Radius for sphere, cylinder, capsule and cone; ``None`` otherwise.
        half_extents_m: Box half extents; ``None`` otherwise.
        height_m: Axial length for cylinder, capsule and cone; ``None`` otherwise.
        axis: Cylinder, capsule or cone axis token (``X``, ``Y`` or ``Z``); ``None`` otherwise.
        material_path: Prim path of the bound physics material, or ``None`` when unbound.
        enabled: Value of ``physics:collisionEnabled``.
    """

    prim_path: str
    body: str
    kind: str
    center_root_m: tuple[float, float, float]
    radius_m: float | None = None
    half_extents_m: tuple[float, float, float] | None = None
    height_m: float | None = None
    axis: str | None = None
    material_path: str | None = None
    enabled: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Collider:
        """Rebuild a collider from its :func:`dataclasses.asdict` form.

        Args:
            data: Mapping produced by ``asdict``.

        Returns:
            The reconstructed collider.
        """
        half = data.get("half_extents_m")
        radius = data.get("radius_m")
        height = data.get("height_m")
        return cls(
            prim_path=str(data["prim_path"]),
            body=str(data["body"]),
            kind=str(data["kind"]),
            center_root_m=_as_tuple3(data["center_root_m"]),
            radius_m=None if radius is None else float(radius),
            half_extents_m=None if half is None else _as_tuple3(half),
            height_m=None if height is None else float(height),
            axis=data.get("axis"),
            material_path=data.get("material_path"),
            enabled=bool(data.get("enabled", True)),
        )


@dataclass(frozen=True)
class Joint:
    """One physics joint.

    Attributes:
        prim_path: Absolute prim path.
        name: Prim name, which the importer sets to the URDF joint name.
        kind: ``revolute``, ``prismatic``, ``fixed``, ``spherical`` or ``generic``.
        axis: ``physics:axis`` token, or ``None`` for joints that have none.
        body0: Prim path of the parent body, or ``None`` when the joint anchors to the world.
        body1: Prim path of the child body, or ``None``.
        local_pos0_m: Anchor in the parent body frame.
        local_pos1_m: Anchor in the child body frame.
        enabled: Value of ``physics:jointEnabled``.
        excluded_from_articulation: Value of ``physics:excludeFromArticulation``.
    """

    prim_path: str
    name: str
    kind: str
    axis: str | None
    body0: str | None
    body1: str | None
    local_pos0_m: tuple[float, float, float]
    local_pos1_m: tuple[float, float, float]
    enabled: bool = True
    excluded_from_articulation: bool = False

    @property
    def is_articulated_dof(self) -> bool:
        """Whether this joint contributes a degree of freedom to the articulation."""
        return self.kind in ("revolute", "prismatic") and self.enabled and not self.excluded_from_articulation

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Joint:
        """Rebuild a joint from its :func:`dataclasses.asdict` form.

        Args:
            data: Mapping produced by ``asdict``.

        Returns:
            The reconstructed joint.
        """
        return cls(
            prim_path=str(data["prim_path"]),
            name=str(data["name"]),
            kind=str(data["kind"]),
            axis=data.get("axis"),
            body0=data.get("body0"),
            body1=data.get("body1"),
            local_pos0_m=_as_tuple3(data["local_pos0_m"]),
            local_pos1_m=_as_tuple3(data["local_pos1_m"]),
            enabled=bool(data.get("enabled", True)),
            excluded_from_articulation=bool(data.get("excluded_from_articulation", False)),
        )


@dataclass(frozen=True)
class PhysicsMaterial:
    """One ``UsdPhysics.MaterialAPI`` material, with its PhysX extensions.

    Attributes:
        prim_path: Absolute prim path.
        static_friction: ``physics:staticFriction``.
        dynamic_friction: ``physics:dynamicFriction``.
        restitution: ``physics:restitution``.
        friction_combine_mode: ``physxMaterial:frictionCombineMode``, or ``None`` if unauthored
            (PhysX then uses its ``average`` default).
        restitution_combine_mode: ``physxMaterial:restitutionCombineMode`` or ``None``.
        improve_patch_friction: ``physxMaterial:improvePatchFriction`` or ``None``.
    """

    prim_path: str
    static_friction: float
    dynamic_friction: float
    restitution: float = 0.0
    friction_combine_mode: str | None = None
    restitution_combine_mode: str | None = None
    improve_patch_friction: bool | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PhysicsMaterial:
        """Rebuild a material from its :func:`dataclasses.asdict` form.

        Args:
            data: Mapping produced by ``asdict``.

        Returns:
            The reconstructed material.
        """
        improve = data.get("improve_patch_friction")
        return cls(
            prim_path=str(data["prim_path"]),
            static_friction=float(data["static_friction"]),
            dynamic_friction=float(data["dynamic_friction"]),
            restitution=float(data.get("restitution", 0.0)),
            friction_combine_mode=data.get("friction_combine_mode"),
            restitution_combine_mode=data.get("restitution_combine_mode"),
            improve_patch_friction=None if improve is None else bool(improve),
        )


@dataclass(frozen=True)
class RobotScene:
    """Everything the M1 checks need to know about a robot USD, with no USD types left in it.

    Attributes:
        source: Where the scene came from (a file path, or a description for synthetic scenes).
        root_prim_path: Path of the default prim, which is the robot root.
        meters_per_unit: Stage ``metersPerUnit``.
        up_axis: Stage ``upAxis``.
        bodies: Every rigid body, in traversal order.
        joints: Every physics joint.
        colliders: Every prim with a collision API.
        materials: Every physics material referenced by a collider, keyed by prim path.
        articulation_root_paths: Prims carrying ``UsdPhysics.ArticulationRootAPI``.
        mesh_prim_paths: Every ``UsdGeom.Mesh`` prim. Must be empty (SPEC v2 S3.4 rule 3).
    """

    source: str
    root_prim_path: str
    meters_per_unit: float
    up_axis: str
    bodies: list[RigidBody] = field(default_factory=list)
    joints: list[Joint] = field(default_factory=list)
    colliders: list[Collider] = field(default_factory=list)
    materials: dict[str, PhysicsMaterial] = field(default_factory=dict)
    articulation_root_paths: list[str] = field(default_factory=list)
    mesh_prim_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable copy of the scene.

        Returns:
            Nested plain dictionaries and lists.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RobotScene:
        """Rebuild a scene from :meth:`to_dict` output.

        Args:
            data: Mapping produced by :meth:`to_dict`.

        Returns:
            The reconstructed scene.
        """
        return cls(
            source=str(data["source"]),
            root_prim_path=str(data["root_prim_path"]),
            meters_per_unit=float(data["meters_per_unit"]),
            up_axis=str(data["up_axis"]),
            bodies=[RigidBody.from_dict(b) for b in data.get("bodies", [])],
            joints=[Joint.from_dict(j) for j in data.get("joints", [])],
            colliders=[Collider.from_dict(c) for c in data.get("colliders", [])],
            materials={str(k): PhysicsMaterial.from_dict(v) for k, v in data.get("materials", {}).items()},
            articulation_root_paths=[str(p) for p in data.get("articulation_root_paths", [])],
            mesh_prim_paths=[str(p) for p in data.get("mesh_prim_paths", [])],
        )

    def body(self, name: str) -> RigidBody | None:
        """Look a rigid body up by name.

        Args:
            name: Prim name, that is the URDF link name.

        Returns:
            The body, or ``None`` when the stage has no such body.
        """
        for candidate in self.bodies:
            if candidate.name == name:
                return candidate
        return None

    def colliders_of(self, body_name: str) -> list[Collider]:
        """Return every collider owned by one rigid body.

        Args:
            body_name: Name of the owning body.

        Returns:
            The colliders, in traversal order.
        """
        return [c for c in self.colliders if c.body == body_name]

    def material_of(self, collider: Collider) -> PhysicsMaterial | None:
        """Return the physics material bound to a collider.

        Args:
            collider: The collider to resolve.

        Returns:
            The material, or ``None`` when the collider is unbound or the binding dangles.
        """
        if collider.material_path is None:
            return None
        return self.materials.get(collider.material_path)


@dataclass(frozen=True)
class CheckResult:
    """The outcome of one acceptance check.

    Attributes:
        check_id: Stable identifier, for example ``M1.masses``.
        ok: Whether the check passed.
        summary: One line, always present, describing what was checked.
        detail: Explanation of the failure. Empty when the check passed.
    """

    check_id: str
    ok: bool
    summary: str
    detail: str = ""


# =============================================================================================
# Pure assertion logic: no USD, no Isaac, no numpy
# =============================================================================================


def _close(actual: float, expected: float, tol: float) -> bool:
    """Absolute-tolerance float comparison that treats NaN as a failure.

    Args:
        actual: Measured value.
        expected: Target value.
        tol: Absolute tolerance.

    Returns:
        True when the values agree.
    """
    if math.isnan(actual) or math.isnan(expected):
        return False
    return abs(actual - expected) <= tol


def _close3(actual: tuple[float, float, float], expected: tuple[float, float, float], tol: float) -> bool:
    """Component-wise absolute-tolerance comparison of two 3-vectors.

    Args:
        actual: Measured vector.
        expected: Target vector.
        tol: Absolute tolerance per component.

    Returns:
        True when every component agrees.
    """
    return all(_close(a, e, tol) for a, e in zip(actual, expected, strict=True))


def _quat_to_matrix(
    quat_wxyz: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float], ...]:
    """Convert a ``(w, x, y, z)`` quaternion to a 3x3 rotation matrix.

    Args:
        quat_wxyz: Unit quaternion, scalar first.

    Returns:
        Row-major rotation matrix.
    """
    w, x, y, z = quat_wxyz
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0.0:
        return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )


def _inertia_tensor(
    diagonal: tuple[float, float, float], principal_axes_wxyz: tuple[float, float, float, float]
) -> tuple[tuple[float, float, float], ...]:
    """Rebuild the full inertia tensor ``R diag(I) R^T`` from USD's split representation.

    USD stores inertia as a diagonal plus the rotation into principal axes. Comparing the full
    tensor is the only comparison that is invariant to how the importer chose to split it: a
    permuted diagonal with a compensating rotation is the same physical body, a permuted diagonal
    without one is a wheel that spins about the wrong axis.

    Args:
        diagonal: ``physics:diagonalInertia``.
        principal_axes_wxyz: ``physics:principalAxes``.

    Returns:
        Row-major 3x3 inertia tensor in the body frame.
    """
    rot = _quat_to_matrix(principal_axes_wxyz)
    scaled = tuple(tuple(rot[i][k] * diagonal[k] for k in range(3)) for i in range(3))
    return tuple(tuple(sum(scaled[i][k] * rot[j][k] for k in range(3)) for j in range(3)) for i in range(3))


def _tensors_close(
    actual: tuple[tuple[float, float, float], ...],
    expected: tuple[tuple[float, float, float], ...],
    rel_tol: float,
) -> bool:
    """Compare two 3x3 tensors with a relative tolerance scaled by the largest expected entry.

    Args:
        actual: Measured tensor.
        expected: Target tensor.
        rel_tol: Relative tolerance.

    Returns:
        True when every entry agrees.
    """
    scale = max(abs(v) for row in expected for v in row)
    tol = rel_tol * max(scale, 1.0e-12)
    return all(_close(actual[i][j], expected[i][j], tol) for i in range(3) for j in range(3))


def _height_above_ground(params: DuckiebotParams, z_root_m: float) -> float:
    """Convert a z coordinate in the robot root frame into a height above the ground plane.

    The URDF places ``base_link`` at the wheel-axle midpoint, which stands
    :attr:`DuckiebotParams.base_link_height_m` above the ground when the robot is level, and the
    importer places the robot root at the stage origin with an identity transform. The wheel
    collider check confirms that convention from the asset itself rather than assuming it: a
    wheel sphere centred on the axle has its centre exactly one wheel radius above the ground.

    Args:
        params: Parameter set.
        z_root_m: Height in the robot root frame.

    Returns:
        Height above the ground plane, in metres.
    """
    return params.base_link_height_m + z_root_m


def _check_stage_units(scene: RobotScene) -> CheckResult:
    """Check stage metadata: metres, Z up, and a default prim to reference.

    Args:
        scene: The extracted scene.

    Returns:
        One check result.
    """
    problems = []
    if not _close(scene.meters_per_unit, 1.0, 1.0e-9):
        problems.append(f"metersPerUnit is {scene.meters_per_unit}, expected 1.0")
    if scene.up_axis != "Z":
        problems.append(f"upAxis is {scene.up_axis!r}, expected 'Z'")
    if not scene.root_prim_path:
        problems.append("the stage has no default prim, so UsdFileCfg cannot reference it")
    return CheckResult(
        "S3.2.stage",
        not problems,
        "stage is metres, Z up, with a default prim",
        "; ".join(problems),
    )


def _check_bodies(scene: RobotScene, params: DuckiebotParams) -> CheckResult:
    """Check the M1 body count and the link names.

    Args:
        scene: The extracted scene.
        params: Parameter set.

    Returns:
        One check result.
    """
    expected = [params.base_link_name, params.left_wheel_link_name, params.right_wheel_link_name]
    found = sorted(b.name for b in scene.bodies)
    problems = []
    if len(scene.bodies) != 3:
        problems.append(f"found {len(scene.bodies)} rigid bodies, expected 3")
    if found != sorted(expected):
        problems.append(f"body names {found} != {sorted(expected)}")
    return CheckResult(
        "M1.bodies", not problems, "3 rigid bodies: base_link and two wheels", "; ".join(problems)
    )


def _check_dof(scene: RobotScene, params: DuckiebotParams) -> CheckResult:
    """Check the M1 degree-of-freedom count and that both DOFs are the wheel joints.

    Args:
        scene: The extracted scene.
        params: Parameter set.

    Returns:
        One check result.
    """
    expected = sorted([params.left_wheel_joint_name, params.right_wheel_joint_name])
    dofs = [j for j in scene.joints if j.is_articulated_dof]
    problems = []
    if len(dofs) != 2:
        problems.append(
            f"found {len(dofs)} articulated DOFs, expected 2: {[(j.name, j.kind) for j in scene.joints]}"
        )
    if sorted(j.name for j in dofs) != expected:
        problems.append(f"DOF names {sorted(j.name for j in dofs)} != {expected}")
    for joint in dofs:
        if joint.kind != "revolute":
            problems.append(f"{joint.name} is {joint.kind}, expected revolute")
        if joint.axis != "Y":
            problems.append(f"{joint.name} axis is {joint.axis}, expected Y")
    fixed_to_world = [j for j in scene.joints if j.kind == "fixed" and (not j.body0 or not j.body1)]
    if fixed_to_world:
        problems.append(
            f"the root is pinned to the world by {[j.name for j in fixed_to_world]}; "
            "the robot must import with fix_base off"
        )
    return CheckResult(
        "M1.dof", not problems, "2 DOF: both wheel joints, revolute about Y", "; ".join(problems)
    )


def _check_masses(scene: RobotScene, params: DuckiebotParams) -> CheckResult:
    """Check the M1 masses: 1.00 kg base and 0.05 kg per wheel.

    Args:
        scene: The extracted scene.
        params: Parameter set.

    Returns:
        One check result.
    """
    expected = {
        params.base_link_name: params.base_mass_kg,
        params.left_wheel_link_name: params.wheel_mass_kg,
        params.right_wheel_link_name: params.wheel_mass_kg,
    }
    problems = []
    for name, target in expected.items():
        body = scene.body(name)
        if body is None:
            problems.append(f"{name} is missing")
            continue
        if not _close(body.mass_kg, target, MASS_TOL_KG):
            problems.append(f"{name} mass {body.mass_kg:.9f} kg != {target:.3f} kg")
    total = sum(b.mass_kg for b in scene.bodies)
    if not _close(total, params.total_mass_kg, 3.0 * MASS_TOL_KG):
        problems.append(f"total mass {total:.9f} kg != {params.total_mass_kg:.3f} kg")
    return CheckResult(
        "M1.masses",
        not problems,
        "masses 1.00 / 0.05 / 0.05 kg, total 1.10 kg",
        "; ".join(problems),
    )


def _check_inertia(scene: RobotScene, params: DuckiebotParams) -> CheckResult:
    """Check that the URDF inertia tensors and centres of mass survived the import.

    The importer can either honour the URDF ``<inertial>`` block or recompute inertia from the
    collision geometry and a density. Those two give very different numbers here (the chassis box
    is far larger than the mass distribution it stands for), so this check is what pins the
    import configuration down to ``set_import_inertia_tensor(True)`` with ``set_density(0.0)``.

    Args:
        scene: The extracted scene.
        params: Parameter set.

    Returns:
        One check result.
    """
    expected_diag = {
        params.base_link_name: params.base_inertia_about_com,
        params.left_wheel_link_name: params.wheel_inertia_about_com,
        params.right_wheel_link_name: params.wheel_inertia_about_com,
    }
    expected_com = {
        params.base_link_name: params.base_com_base_frame_m,
        params.left_wheel_link_name: (0.0, 0.0, 0.0),
        params.right_wheel_link_name: (0.0, 0.0, 0.0),
    }
    problems = []
    for name, diag in expected_diag.items():
        body = scene.body(name)
        if body is None:
            problems.append(f"{name} is missing")
            continue
        target = tuple(tuple(diag[i] if i == j else 0.0 for j in range(3)) for i in range(3))
        actual = _inertia_tensor(body.diagonal_inertia_kg_m2, body.principal_axes_wxyz)
        if not _tensors_close(actual, target, INERTIA_REL_TOL):
            problems.append(
                f"{name} inertia {body.diagonal_inertia_kg_m2} about axes "
                f"{body.principal_axes_wxyz} does not match the URDF tensor diag{diag}"
            )
        if not _close3(body.com_root_m, expected_com[name], LENGTH_TOL_M):
            problems.append(f"{name} CoM {body.com_root_m} != {expected_com[name]}")
    return CheckResult(
        "S3.2.inertia",
        not problems,
        "inertia tensors and centres of mass came from the URDF, not from a density",
        "; ".join(problems),
    )


def _check_caster(scene: RobotScene, params: DuckiebotParams) -> CheckResult:
    """Check the M1 caster: one sphere of radius 0.0165 m whose centre height equals its radius.

    Args:
        scene: The extracted scene.
        params: Parameter set.

    Returns:
        One check result.
    """
    summary = "caster is one sphere of radius 0.0165 m touching the ground"
    spheres = [c for c in scene.colliders_of(params.base_link_name) if c.kind == "sphere"]
    if len(spheres) != 1:
        return CheckResult(
            "M1.caster",
            False,
            summary,
            f"found {len(spheres)} sphere colliders on {params.base_link_name}, expected 1",
        )
    caster = spheres[0]
    radius = caster.radius_m or 0.0
    problems = []
    if not _close(radius, params.caster_radius_m, LENGTH_TOL_M):
        problems.append(f"radius {radius:.6f} m != {params.caster_radius_m:.4f} m")
    if not _close3(caster.center_root_m, params.caster_center_base_frame_m, LENGTH_TOL_M):
        problems.append(f"centre {caster.center_root_m} != {params.caster_center_base_frame_m} in base frame")
    height = _height_above_ground(params, caster.center_root_m[2])
    if not _close(height, radius, LENGTH_TOL_M):
        problems.append(
            f"centre height above ground {height:.6f} m != radius {radius:.6f} m, so the caster "
            f"{'floats' if height > radius else 'penetrates the ground'} by "
            f"{abs(height - radius) * 1000.0:.3f} mm"
        )
    if not caster.enabled:
        problems.append("physics:collisionEnabled is false")
    return CheckResult("M1.caster", not problems, summary, "; ".join(problems))


def _check_chassis(scene: RobotScene, params: DuckiebotParams) -> CheckResult:
    """Check the M1 chassis box: correct size, and underside exactly at the 21 mm clearance.

    Args:
        scene: The extracted scene.
        params: Parameter set.

    Returns:
        One check result.
    """
    summary = "chassis box underside sits at the 21 mm ground clearance"
    boxes = [c for c in scene.colliders_of(params.base_link_name) if c.kind == "box"]
    if len(boxes) != 1:
        return CheckResult(
            "M1.chassis",
            False,
            summary,
            f"found {len(boxes)} box colliders on {params.base_link_name}, expected 1",
        )
    box = boxes[0]
    problems = []
    half = box.half_extents_m or (0.0, 0.0, 0.0)
    expected_half = tuple(0.5 * s for s in params.chassis_size_m)
    if not _close3(half, expected_half, LENGTH_TOL_M):
        problems.append(f"half extents {half} != {expected_half}")
    if not _close3(box.center_root_m, params.chassis_center_base_frame_m, LENGTH_TOL_M):
        problems.append(f"centre {box.center_root_m} != {params.chassis_center_base_frame_m} in base frame")
    bottom = _height_above_ground(params, box.center_root_m[2]) - half[2]
    if not _close(bottom, params.ground_clearance_m, LENGTH_TOL_M):
        problems.append(f"underside at {bottom:.6f} m != {params.ground_clearance_m:.4f} m ground clearance")
    if not box.enabled:
        problems.append("physics:collisionEnabled is false")
    return CheckResult("M1.chassis", not problems, summary, "; ".join(problems))


def _check_wheel_colliders(scene: RobotScene, params: DuckiebotParams) -> CheckResult:
    """Check that each wheel carries exactly one sphere collider on the axle.

    S3.2 is emphatic that the wheel collider is a sphere and never a cylinder: a cylindrical
    wheel contact degenerates to a line and the MuJoCo study measured a 74% loss of yaw response.

    Args:
        scene: The extracted scene.
        params: Parameter set.

    Returns:
        One check result.
    """
    problems = []
    for link, origin in (
        (params.left_wheel_link_name, params.left_wheel_origin_m),
        (params.right_wheel_link_name, params.right_wheel_origin_m),
    ):
        colliders = scene.colliders_of(link)
        if len(colliders) != 1:
            problems.append(f"{link} has {len(colliders)} colliders, expected exactly 1")
            continue
        wheel = colliders[0]
        if wheel.kind != "sphere":
            problems.append(f"{link} collider is a {wheel.kind}, expected a sphere")
            continue
        radius = wheel.radius_m or 0.0
        if not _close(radius, params.wheel_radius_m, LENGTH_TOL_M):
            problems.append(f"{link} radius {radius:.6f} m != {params.wheel_radius_m:.4f} m")
        if not _close3(wheel.center_root_m, origin, LENGTH_TOL_M):
            problems.append(f"{link} centre {wheel.center_root_m} != axle origin {origin}")
        height = _height_above_ground(params, wheel.center_root_m[2])
        if not _close(height, radius, LENGTH_TOL_M):
            problems.append(
                f"{link} centre height {height:.6f} m != radius {radius:.6f} m, so the robot does "
                "not stand on its wheels"
            )
        if not wheel.enabled:
            problems.append(f"{link} has physics:collisionEnabled false")
    return CheckResult(
        "S3.2.wheels",
        not problems,
        "each wheel is one sphere of radius 0.0318 m centred on its axle",
        "; ".join(problems),
    )


def _check_collider_census(scene: RobotScene) -> CheckResult:
    """Check that the asset has exactly the four intended colliders and no round ones.

    Args:
        scene: The extracted scene.

    Returns:
        One check result.
    """
    problems = []
    round_colliders = [c for c in scene.colliders if c.kind in _ROUND_COLLIDER_KINDS]
    if round_colliders:
        problems.append(
            "cylinder-like colliders survive the patch step: "
            f"{[(c.prim_path, c.kind) for c in round_colliders]}"
        )
    if len(scene.colliders) != 4:
        problems.append(
            f"found {len(scene.colliders)} colliders, expected 4 (chassis box, caster sphere, "
            f"two wheel spheres): {[(c.prim_path, c.kind) for c in scene.colliders]}"
        )
    return CheckResult(
        "S3.2.colliders",
        not problems,
        "exactly 4 colliders, none of them a cylinder, capsule or cone",
        "; ".join(problems),
    )


def _check_no_meshes(scene: RobotScene) -> CheckResult:
    """Check the clean-room requirement that the robot contains no ``Mesh`` prim at all.

    Args:
        scene: The extracted scene.

    Returns:
        One check result.
    """
    return CheckResult(
        "S3.4.no_mesh",
        not scene.mesh_prim_paths,
        "no Mesh prim anywhere in the robot (clean-room rule 3)",
        f"Mesh prims found: {scene.mesh_prim_paths}" if scene.mesh_prim_paths else "",
    )


def _check_articulation_root(scene: RobotScene) -> CheckResult:
    """Check that the stage carries exactly one articulation root.

    Args:
        scene: The extracted scene.

    Returns:
        One check result.
    """
    count = len(scene.articulation_root_paths)
    return CheckResult(
        "S5.1.articulation_root",
        count == 1,
        "exactly one UsdPhysics.ArticulationRootAPI prim",
        "" if count == 1 else f"found {count}: {scene.articulation_root_paths}",
    )


def _material_problems(material: PhysicsMaterial | None, spec: dict[str, Any], label: str) -> list[str]:
    """Compare one bound material against its :func:`physics_material_spec` entry.

    Args:
        material: The material the collider resolves to, or ``None`` when unbound.
        spec: The expected entry from :func:`physics_material_spec`.
        label: Human-readable name of the collider being checked.

    Returns:
        A list of problem descriptions, empty when the material is correct.
    """
    if material is None:
        return [f"{label} has no physics material bound, so PhysX uses its 0.5/0.5 default"]
    problems = []
    if not _close(material.static_friction, spec["static_friction"], FRICTION_TOL):
        problems.append(f"{label} static friction {material.static_friction} != {spec['static_friction']}")
    if not _close(material.dynamic_friction, spec["dynamic_friction"], FRICTION_TOL):
        problems.append(f"{label} dynamic friction {material.dynamic_friction} != {spec['dynamic_friction']}")
    if material.friction_combine_mode != spec["friction_combine_mode"]:
        problems.append(
            f"{label} friction combine mode {material.friction_combine_mode!r} != "
            f"{spec['friction_combine_mode']!r}"
        )
    if bool(material.improve_patch_friction) != bool(spec["improve_patch_friction"]):
        problems.append(
            f"{label} improvePatchFriction {material.improve_patch_friction} != "
            f"{spec['improve_patch_friction']}"
        )
    return problems


def _check_materials(scene: RobotScene, params: DuckiebotParams) -> CheckResult:
    """Check the two physics materials S3.2 requires the patch step to bind.

    Args:
        scene: The extracted scene.
        params: Parameter set.

    Returns:
        One check result.
    """
    spec = physics_material_spec(params)
    problems = []

    caster_spheres = [c for c in scene.colliders_of(params.base_link_name) if c.kind == "sphere"]
    if not caster_spheres:
        problems.append("no caster sphere to carry the frictionless material")
    for caster in caster_spheres:
        problems.extend(
            _material_problems(scene.material_of(caster), spec["duckiebot_caster_material"], "caster")
        )
    for link in (params.left_wheel_link_name, params.right_wheel_link_name):
        for wheel in scene.colliders_of(link):
            problems.extend(
                _material_problems(scene.material_of(wheel), spec["duckiebot_wheel_material"], link)
            )
    return CheckResult(
        "S3.2.materials",
        not problems,
        "wheels grip at mu 1.0 (combine max), caster is frictionless (combine min)",
        "; ".join(problems),
    )


def check_scene(scene: RobotScene, params: DuckiebotParams = DUCKIEBOT) -> list[CheckResult]:
    """Run every acceptance check against a scene description.

    This function is pure: it touches no file, imports no simulator and has no side effects, so
    the unit tests drive it with synthetic scenes on a CPU-only runner.

    Args:
        scene: Flattened description of the robot USD.
        params: Parameter set holding the expected values.

    Returns:
        One :class:`CheckResult` per criterion, in report order. The asset is acceptable when
        every result has ``ok`` set.
    """
    return [
        _check_stage_units(scene),
        _check_bodies(scene, params),
        _check_dof(scene, params),
        _check_masses(scene, params),
        _check_inertia(scene, params),
        _check_caster(scene, params),
        _check_chassis(scene, params),
        _check_wheel_colliders(scene, params),
        _check_collider_census(scene),
        _check_materials(scene, params),
        _check_no_meshes(scene),
        _check_articulation_root(scene),
    ]


def format_report(scene: RobotScene, results: list[CheckResult]) -> str:
    """Render the check results as a fixed-width report.

    Args:
        scene: The scene that was checked.
        results: Output of :func:`check_scene`.

    Returns:
        The report text, without a trailing newline.
    """
    width = max(len(r.check_id) for r in results)
    lines = [
        f"verify_usd: {scene.source}",
        f"  root prim {scene.root_prim_path}, {len(scene.bodies)} bodies, "
        f"{len([j for j in scene.joints if j.is_articulated_dof])} DOF, "
        f"{len(scene.colliders)} colliders, {len(scene.materials)} physics materials",
        "-" * 78,
    ]
    for result in results:
        status = "PASS" if result.ok else "FAIL"
        lines.append(f"  [{status}] {result.check_id.ljust(width)}  {result.summary}")
        if not result.ok and result.detail:
            lines.extend(f"         {' ' * width}  -> {part}" for part in result.detail.split("; "))
    failed = [r.check_id for r in results if not r.ok]
    lines.append("-" * 78)
    if failed:
        lines.append(f"FAILED {len(failed)} of {len(results)} checks: {', '.join(failed)}")
    else:
        lines.append(f"PASSED all {len(results)} checks (SPEC v2 M1 asset acceptance)")
    return "\n".join(lines)


# =============================================================================================
# USD side: make pxr importable, open a stage, flatten it
# =============================================================================================


def ensure_pxr() -> str:
    """Make ``pxr`` importable in this interpreter and report where it came from.

    Three sources are tried in order:

    1. an already-importable ``pxr`` (usd-core in the tools venv, or a live Kit process);
    2. the directory named by ``DUCKIEBOT_USD_LIBS_DIR``;
    3. the USD build Isaac Sim ships in ``isaacsim/extscache/omni.usd.libs-*``, which is a
       complete standalone USD and needs no Kit boot.

    For source 3 the extension's ``bin`` directory is added both as a DLL search directory (for
    the Python extension modules) and to ``PATH`` (for USD's own plugin loader, which uses the
    process search path and would otherwise fail to load ``usd_usd.dll`` lazily, with a
    misleading "Cannot determine file format" error on the first layer you open).

    Returns:
        A one-line description of the USD runtime that is now importable.

    Raises:
        StageOpenError: If no USD runtime can be found.
    """
    try:
        import pxr

        # pxr is a namespace package, so ``__file__`` is None and only ``__path__`` locates it.
        return f"pxr already importable from {list(getattr(pxr, '__path__', []))}"
    except ImportError:
        pass

    candidates: list[Path] = []
    override = os.environ.get("DUCKIEBOT_USD_LIBS_DIR")
    if override:
        candidates.append(Path(override))
    for key in ("purelib", "platlib"):
        site_dir = sysconfig.get_paths().get(key)
        if site_dir:
            extscache = Path(site_dir) / "isaacsim" / "extscache"
            candidates.extend(sorted(extscache.glob("omni.usd.libs-*")))

    for root in candidates:
        if not (root / "pxr").is_dir():
            continue
        bin_dir = root / "bin"
        if bin_dir.is_dir():
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
        entry = str(root)
        if entry not in sys.path:
            sys.path.insert(0, entry)
        try:
            import pxr

            return f"pxr loaded from the Isaac Sim USD build at {entry}"
        except ImportError:
            sys.path.remove(entry)

    raise StageOpenError(
        "No USD runtime found. Either run this with the Isaac Sim python "
        "(d:/Personal/personal/wheeled_quadruped_robot/.venv/Scripts/python.exe), or "
        "pip install usd-core into the tools venv, or point DUCKIEBOT_USD_LIBS_DIR at a "
        "directory containing a pxr package."
    )


def open_stage(usd_path: str | Path) -> Any:
    """Open a USD stage with every payload loaded.

    Args:
        usd_path: Path to the ``.usd`` or ``.usda`` file.

    Returns:
        The opened ``Usd.Stage``.

    Raises:
        StageOpenError: If the file is missing or USD refuses to open it.
    """
    ensure_pxr()
    from pxr import Usd

    path = Path(usd_path)
    if not path.is_file():
        raise StageOpenError(f"{path} does not exist. Build it with: python tools/import_urdf_headless.py")
    stage = Usd.Stage.Open(str(path), Usd.Stage.LoadAll)
    if stage is None:
        raise StageOpenError(f"USD could not open {path}")
    return stage


def _matrix_scale(matrix: Any) -> tuple[float, float, float]:
    """Extract the per-axis scale from a 4x4 transform matrix.

    Args:
        matrix: A ``Gf.Matrix4d``.

    Returns:
        ``(sx, sy, sz)``, the lengths of the three basis rows.
    """
    rows = [[matrix[i][j] for j in range(3)] for i in range(3)]
    scales = [math.sqrt(sum(v * v for v in row)) for row in rows]
    return (scales[0], scales[1], scales[2])


def _shape_of(prim: Any, local_to_root: Any) -> dict[str, Any]:
    """Describe the geometry of one collider prim in the robot root frame.

    The URDF importer authors a unit ``Cube`` with the box extents in ``xformOp:scale``, so the
    scale has to be folded in here rather than read off the geometry attributes.

    Args:
        prim: The prim carrying the collision API.
        local_to_root: Its 4x4 transform into the robot root frame.

    Returns:
        Keyword arguments for :class:`Collider`: ``kind`` plus whichever of ``radius_m``,
        ``half_extents_m``, ``height_m`` and ``axis`` apply.
    """
    type_name = str(prim.GetTypeName())
    scale = _matrix_scale(local_to_root)
    mean_scale = sum(scale) / 3.0

    def attr(name: str, default: float) -> float:
        """Read a float attribute, falling back to the schema default."""
        value = prim.GetAttribute(name).Get() if prim.HasAttribute(name) else None
        return float(default if value is None else value)

    if type_name == "Sphere":
        return {"kind": "sphere", "radius_m": attr("radius", 1.0) * mean_scale}
    if type_name == "Cube":
        size = attr("size", 2.0)
        return {"kind": "box", "half_extents_m": tuple(0.5 * size * s for s in scale)}
    if type_name in ("Cylinder", "Capsule", "Cone"):
        axis = prim.GetAttribute("axis").Get() if prim.HasAttribute("axis") else "Z"
        return {
            "kind": type_name.lower(),
            "radius_m": attr("radius", 1.0) * mean_scale,
            "height_m": attr("height", 2.0) * mean_scale,
            "axis": str(axis),
        }
    if type_name == "Mesh":
        return {"kind": "mesh"}
    return {"kind": "other"}


def _bound_physics_material_path(prim: Any, usd_shade: Any, usd_physics: Any) -> str | None:
    """Resolve the physics-purpose material binding of a collider prim.

    ``ComputeBoundMaterial`` falls back to the all-purpose binding when no physics-purpose one is
    found, which on an imported robot would resolve to the OmniPBR *visual* material of an
    ancestor. The result is therefore only accepted when the material prim actually carries
    ``UsdPhysics.MaterialAPI``; anything else counts as unbound, which is what PhysX does too.

    Args:
        prim: The collider prim.
        usd_shade: The ``UsdShade`` module (passed in so this helper needs no import of its own).
        usd_physics: The ``UsdPhysics`` module.

    Returns:
        The bound physics material's prim path, or ``None`` when nothing is bound.
    """
    binding_api = usd_shade.MaterialBindingAPI(prim)
    material, _ = binding_api.ComputeBoundMaterial(PHYSICS_BINDING_PURPOSE)
    if not material:
        return None
    material_prim = material.GetPrim()
    if not material_prim.IsValid() or not material_prim.HasAPI(usd_physics.MaterialAPI):
        return None
    return str(material_prim.GetPath())


def _read_physics_material(prim: Any, usd_physics: Any) -> PhysicsMaterial:
    """Read the friction properties of a physics material prim.

    The three PhysX extensions (friction combine mode, restitution combine mode and
    ``improvePatchFriction``) are read as raw attributes rather than through ``PhysxSchema``: that
    schema library ships in a different Isaac Sim extension than USD itself and does not exist at
    all in a plain usd-core install, and the attribute names are stable schema names either way.

    Args:
        prim: The material prim.
        usd_physics: The ``UsdPhysics`` module.

    Returns:
        The material description.
    """
    api = usd_physics.MaterialAPI(prim)

    def value(attr: Any, default: float) -> float:
        """Read a typed attribute with a default."""
        raw = attr.Get() if attr else None
        return float(default if raw is None else raw)

    def raw(name: str) -> Any:
        """Read a raw attribute by name, or None when it is unauthored."""
        return prim.GetAttribute(name).Get() if prim.HasAttribute(name) else None

    friction_mode = raw(PHYSX_FRICTION_COMBINE_ATTR)
    restitution_mode = raw(PHYSX_RESTITUTION_COMBINE_ATTR)
    improve_patch = raw(PHYSX_IMPROVE_PATCH_ATTR)
    return PhysicsMaterial(
        prim_path=str(prim.GetPath()),
        static_friction=value(api.GetStaticFrictionAttr(), 0.0),
        dynamic_friction=value(api.GetDynamicFrictionAttr(), 0.0),
        restitution=value(api.GetRestitutionAttr(), 0.0),
        friction_combine_mode=None if friction_mode is None else str(friction_mode),
        restitution_combine_mode=None if restitution_mode is None else str(restitution_mode),
        improve_patch_friction=None if improve_patch is None else bool(improve_patch),
    )


_JOINT_KINDS: dict[str, str] = {
    "PhysicsRevoluteJoint": "revolute",
    "PhysicsPrismaticJoint": "prismatic",
    "PhysicsFixedJoint": "fixed",
    "PhysicsSphericalJoint": "spherical",
    "PhysicsDistanceJoint": "distance",
    "PhysicsJoint": "generic",
}
"""Prim type name to the ``kind`` recorded on :class:`Joint`."""


def _read_joint(prim: Any, kind: str) -> Joint:
    """Read one physics joint prim.

    Args:
        prim: The joint prim.
        kind: Its resolved kind, from :data:`_JOINT_KINDS`.

    Returns:
        The joint description.
    """

    def target(name: str) -> str | None:
        """First target of a relationship, or None."""
        rel = prim.GetRelationship(name)
        targets = rel.GetTargets() if rel else []
        return str(targets[0]) if targets else None

    def vec(name: str) -> tuple[float, float, float]:
        """A point3f attribute, defaulting to the origin."""
        value = prim.GetAttribute(name).Get() if prim.HasAttribute(name) else None
        return (0.0, 0.0, 0.0) if value is None else _as_tuple3(value)

    def flag(name: str, default: bool) -> bool:
        """A bool attribute with a default."""
        value = prim.GetAttribute(name).Get() if prim.HasAttribute(name) else None
        return default if value is None else bool(value)

    axis = prim.GetAttribute("physics:axis").Get() if prim.HasAttribute("physics:axis") else None
    return Joint(
        prim_path=str(prim.GetPath()),
        name=str(prim.GetName()),
        kind=kind,
        axis=None if axis is None else str(axis),
        body0=target("physics:body0"),
        body1=target("physics:body1"),
        local_pos0_m=vec("physics:localPos0"),
        local_pos1_m=vec("physics:localPos1"),
        enabled=flag("physics:jointEnabled", True),
        excluded_from_articulation=flag("physics:excludeFromArticulation", False),
    )


def extract_scene(stage: Any, source: str = "", params: DuckiebotParams = DUCKIEBOT) -> RobotScene:
    """Flatten a composed USD stage into a :class:`RobotScene`.

    Instance proxies are traversed explicitly: the Isaac Sim URDF importer puts every collider
    behind an ``instanceable = true`` reference, and a plain ``stage.Traverse()`` walks straight
    past them, which makes a perfectly good asset look as if it had no colliders at all.

    Args:
        stage: An opened ``Usd.Stage``.
        source: Description recorded in the scene, normally the file path.
        params: Parameter set. Unused today; accepted so callers can pass a variant set through.

    Returns:
        The flattened scene.

    Raises:
        StageOpenError: If the stage has no default prim.
    """
    del params  # the extractor reads the asset, it never assumes the expected values
    ensure_pxr()
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

    root = stage.GetDefaultPrim()
    if not root or not root.IsValid():
        raise StageOpenError(
            f"{source or stage} has no default prim, so UsdFileCfg cannot reference the robot"
        )
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    root_inverse = xform_cache.GetLocalToWorldTransform(root).GetInverse()

    scene = RobotScene(
        source=source or str(stage.GetRootLayer().identifier),
        root_prim_path=str(root.GetPath()),
        meters_per_unit=float(UsdGeom.GetStageMetersPerUnit(stage)),
        up_axis=str(UsdGeom.GetStageUpAxis(stage)),
    )
    pseudo_root = stage.GetPseudoRoot()

    def body_name_for(prim: Any) -> str:
        """Name of the nearest ancestor rigid body, or an empty string."""
        current = prim
        while current and current.IsValid() and current != pseudo_root:
            if current.HasAPI(UsdPhysics.RigidBodyAPI):
                return str(current.GetName())
            current = current.GetParent()
        return ""

    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        type_name = str(prim.GetTypeName())

        if type_name == "Mesh":
            scene.mesh_prim_paths.append(path)
        if type_name in _JOINT_KINDS:
            scene.joints.append(_read_joint(prim, _JOINT_KINDS[type_name]))
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            scene.articulation_root_paths.append(path)

        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            mass_api = UsdPhysics.MassAPI(prim)
            quat = mass_api.GetPrincipalAxesAttr().Get() or Gf.Quatf(1.0, 0.0, 0.0, 0.0)
            imaginary = quat.GetImaginary()
            local = xform_cache.GetLocalToWorldTransform(prim) * root_inverse
            scene.bodies.append(
                RigidBody(
                    prim_path=path,
                    name=str(prim.GetName()),
                    mass_kg=float(mass_api.GetMassAttr().Get() or 0.0),
                    com_root_m=_as_tuple3(mass_api.GetCenterOfMassAttr().Get() or (0.0, 0.0, 0.0)),
                    diagonal_inertia_kg_m2=_as_tuple3(
                        mass_api.GetDiagonalInertiaAttr().Get() or (0.0, 0.0, 0.0)
                    ),
                    principal_axes_wxyz=(
                        float(quat.GetReal()),
                        float(imaginary[0]),
                        float(imaginary[1]),
                        float(imaginary[2]),
                    ),
                    translate_root_m=_as_tuple3(local.ExtractTranslation()),
                )
            )

        if prim.HasAPI(UsdPhysics.CollisionAPI):
            local = xform_cache.GetLocalToWorldTransform(prim) * root_inverse
            enabled_attr = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
            material_path = _bound_physics_material_path(prim, UsdShade, UsdPhysics)
            scene.colliders.append(
                Collider(
                    prim_path=path,
                    body=body_name_for(prim),
                    center_root_m=_as_tuple3(local.ExtractTranslation()),
                    material_path=material_path,
                    enabled=True if enabled_attr is None else bool(enabled_attr),
                    **_shape_of(prim, local),
                )
            )
            if material_path and material_path not in scene.materials:
                material_prim = stage.GetPrimAtPath(material_path)
                if material_prim and material_prim.IsValid():
                    scene.materials[material_path] = _read_physics_material(material_prim, UsdPhysics)
    return scene


def verify_usd_file(
    usd_path: str | Path, params: DuckiebotParams = DUCKIEBOT
) -> tuple[RobotScene, list[CheckResult]]:
    """Open, flatten and check one robot USD.

    Args:
        usd_path: Path to the asset.
        params: Parameter set holding the expected values.

    Returns:
        The scene and its check results.

    Raises:
        StageOpenError: If the stage cannot be opened or has no default prim.
    """
    stage = open_stage(usd_path)
    scene = extract_scene(stage, source=str(usd_path), params=params)
    return scene, check_scene(scene, params)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]``.

    Returns:
        The parsed namespace with ``usd``, ``json`` and ``quiet`` attributes.
    """
    parser = argparse.ArgumentParser(
        prog="verify_usd.py",
        description="Check an imported Duckiebot USD against the SPEC v2 M1 acceptance criteria.",
    )
    parser.add_argument(
        "usd",
        nargs="?",
        default=DEFAULT_USD_PATH,
        help=f"path to the robot USD (default: {DEFAULT_USD_PATH})",
    )
    parser.add_argument("--json", type=Path, default=None, help="also write a JSON report here")
    parser.add_argument("--quiet", action="store_true", help="print only the final verdict line")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code: ``0`` all checks passed, ``1`` a check failed, ``2`` the stage could
        not be opened.
    """
    args = parse_args(argv)
    try:
        scene, results = verify_usd_file(args.usd)
    except StageOpenError as error:
        print(f"verify_usd: {error}", file=sys.stderr)
        return 2

    report = format_report(scene, results)
    print(report.splitlines()[-1] if args.quiet else report)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {"scene": scene.to_dict(), "checks": [asdict(r) for r in results]}
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
