"""Patch an imported Duckiebot USD: swap round wheel colliders, bind the physics materials.

SPEC v2 S3.2 allows this script exactly two jobs, and it does exactly those two:

1. replace a wheel collider with a sphere of radius 0.0318 m if the URDF importer emitted a
   cylinder or a capsule (a cylindrical wheel contact degenerates to a line: the MuJoCo study
   behind S3.2 measured a 74% loss of yaw response);
2. bind the two rigid-body physics materials named by
   :func:`duckiebot_rl.assets.robot_cfg.physics_material_spec`: wheels at mu_s = mu_d = 1.0 with
   combine mode ``max`` and ``improvePatchFriction`` on, caster at mu = 0 with combine ``min``.

Drive gains, joint limits, armature, solver iteration counts and contact offsets are deliberately
NOT written here. Those all come from ``ArticulationCfg`` and the spawn config at play time
(SPEC v2 S3.2, resolving critic item E), and writing them into the USD as well would create a
second source of truth that silently disagrees with the first.

What the Isaac Sim 5.1 importer actually produces, and what that forces
----------------------------------------------------------------------
The importer writes a small layer stack: ``duckiebot.usd`` selects variants and payloads
``configuration/duckiebot_physics.usd``, which sublayers ``configuration/duckiebot_base.usd``.
Collider geometry is authored once under a root-level ``/colliders`` scope and pulled into each
link as ``/duckiebot/<link>/collisions``, an Xform with ``instanceable = true``.

Two consequences, both handled here:

* an instance proxy is read-only, so every collider that has to be touched has its instanceable
  ancestor turned into an ordinary prim first. The cost is nil (the four colliders are a Cube and
  three Spheres) and it keeps the whole patch inside the root layer;
* all edits are authored into the root layer of the file passed on the command line, so the
  importer's own output under ``configuration/`` stays pristine and re-running the patch is
  idempotent.

Usage (from the repository root):

.. code-block:: text

    python tools/patch_usd.py                          # patches the default asset path
    python tools/patch_usd.py assets/usd/duckiebot.usd
    python tools/patch_usd.py --dry-run                # print the plan, write nothing

Exit codes: ``0`` patched (or already correct), ``1`` the asset cannot be patched as specified,
``2`` the stage could not be opened.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TOOLS_DIR = Path(__file__).resolve().parent
for _path in (str(_REPO_ROOT), str(_TOOLS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from verify_usd import (  # noqa: E402
    PHYSICS_BINDING_PURPOSE,
    PHYSX_FRICTION_COMBINE_ATTR,
    PHYSX_IMPROVE_PATCH_ATTR,
    PHYSX_RESTITUTION_COMBINE_ATTR,
    Collider,
    RobotScene,
    StageOpenError,
    ensure_pxr,
    extract_scene,
    open_stage,
)

from duckiebot_rl.assets.params import DUCKIEBOT, DuckiebotParams  # noqa: E402
from duckiebot_rl.assets.robot_cfg import DEFAULT_USD_PATH, physics_material_spec  # noqa: E402

__all__ = [
    "MATERIAL_SCOPE_NAME",
    "BindMaterial",
    "PatchPlan",
    "PatchPlanError",
    "PatchReport",
    "ReplaceCollider",
    "apply_patch",
    "main",
    "patch_usd_file",
    "plan_patch",
]

MATERIAL_SCOPE_NAME = "PhysicsMaterials"
"""Scope created under the robot root to hold the two rigid-body physics materials."""

_ROUND_KINDS = frozenset({"cylinder", "capsule", "cone"})
"""Collider kinds that must become spheres on a wheel."""


class PatchPlanError(RuntimeError):
    """Raised when the asset cannot be patched exactly as SPEC v2 S3.2 describes.

    This is deliberately loud. The failure mode this whole script exists to prevent is a silent
    no-op: a selector that matches nothing leaves the caster on the PhysX default material
    (0.5/0.5) and nothing downstream ever reports it, the robot just turns badly forever.
    """


@dataclass(frozen=True)
class ReplaceCollider:
    """Planned replacement of one round collider by a sphere.

    Attributes:
        prim_path: The collider prim to deactivate.
        from_kind: What it is today (``cylinder``, ``capsule`` or ``cone``).
        radius_m: Radius of the sphere that replaces it.
        body: Name of the owning link.
    """

    prim_path: str
    from_kind: str
    radius_m: float
    body: str

    def describe(self) -> str:
        """Return a one-line description of the action.

        Returns:
            Human-readable text for the report.
        """
        return (
            f"replace {self.from_kind} collider {self.prim_path} on {self.body} with a sphere of "
            f"radius {self.radius_m:.4f} m"
        )


@dataclass(frozen=True)
class BindMaterial:
    """Planned binding of one physics material to one collider.

    Attributes:
        prim_path: The collider prim to bind.
        material_name: Key in :func:`physics_material_spec`, also the material prim name.
        body: Name of the owning link.
    """

    prim_path: str
    material_name: str
    body: str

    def describe(self) -> str:
        """Return a one-line description of the action.

        Returns:
            Human-readable text for the report.
        """
        return f"bind {self.material_name} to {self.prim_path} on {self.body}"


@dataclass(frozen=True)
class PatchPlan:
    """Everything the patch will do, computed without touching USD.

    Attributes:
        replacements: Collider swaps, applied before the bindings.
        bindings: Material bindings, computed against the post-swap geometry.
        materials: The material definitions to author, keyed by material name.
        unbound: Colliders no material claims. The chassis box belongs here: S3.2 gives the patch
            step two materials, wheels and caster, and the chassis is 21 mm off the ground and
            never contacts it. Reported rather than silently dropped.
    """

    replacements: list[ReplaceCollider] = field(default_factory=list)
    bindings: list[BindMaterial] = field(default_factory=list)
    materials: dict[str, dict[str, Any]] = field(default_factory=dict)
    unbound: list[str] = field(default_factory=list)

    def describe(self) -> str:
        """Render the plan as text.

        Returns:
            One line per action, plus one line per deliberately unbound collider.
        """
        lines = [action.describe() for action in (*self.replacements, *self.bindings)]
        lines.extend(
            f"leave {path} on the PhysX default material (no S3.2 material claims it)"
            for path in self.unbound
        )
        return "\n".join(f"  - {line}" for line in lines) if lines else "  - nothing to do"


@dataclass
class PatchReport:
    """What the patch actually did.

    Attributes:
        usd_path: The file that was patched.
        plan: The plan that was applied.
        created_materials: Prim paths of the materials that were authored.
        de_instanced: Prim paths whose ``instanceable`` metadata was cleared.
        replaced: Prim paths that were deactivated and replaced by a sphere.
        bound: ``(collider path, material path)`` pairs that were bound.
        saved: Whether the stage was written back to disk.
    """

    usd_path: str
    plan: PatchPlan
    created_materials: list[str] = field(default_factory=list)
    de_instanced: list[str] = field(default_factory=list)
    replaced: list[str] = field(default_factory=list)
    bound: list[tuple[str, str]] = field(default_factory=list)
    saved: bool = False

    def format(self) -> str:
        """Render the report as text.

        Returns:
            The report, without a trailing newline.
        """
        lines = [f"patch_usd: {self.usd_path}", "plan:", self.plan.describe(), "applied:"]
        lines.extend(f"  - de-instanced {path}" for path in self.de_instanced)
        lines.extend(f"  - authored material {path}" for path in self.created_materials)
        lines.extend(f"  - replaced {path} with a sphere" for path in self.replaced)
        lines.extend(f"  - bound {material} to {collider}" for collider, material in self.bound)
        lines.append(f"saved: {self.saved}")
        return "\n".join(lines)


# =============================================================================================
# Planning: pure, unit-testable, no USD
# =============================================================================================


def _select_colliders(scene: RobotScene, spec: dict[str, Any], material_name: str) -> list[Collider]:
    """Resolve one material's selector against the scene.

    The selector contract is defined by :func:`physics_material_spec` and implemented here, and
    only here:

    * ``bind_to`` lists link names. Every collider owned by those links is a candidate.
    * ``restrict_to_shape``, when present, narrows the candidates to the colliders whose shape
      matches, by ``kind`` plus (for a sphere) ``radius_m`` and the centre position, within
      ``position_tol_m``.
    * ``expect_matches`` is the exact number of colliders the selector must resolve to.

    Args:
        scene: The scene being patched.
        spec: One entry of :func:`physics_material_spec`.
        material_name: Name of the material, for error messages.

    Returns:
        The selected colliders, in scene order.

    Raises:
        PatchPlanError: If the selector does not resolve to exactly ``expect_matches`` colliders.
    """
    candidates: list[Collider] = []
    for link in spec["bind_to"]:
        candidates.extend(scene.colliders_of(link))

    shape = spec.get("restrict_to_shape")
    if shape is not None:
        tol = float(shape.get("position_tol_m", 1.0e-4))
        wanted_kind = str(shape["kind"])
        wanted_radius = shape.get("radius_m")
        wanted_center = shape.get("center_base_frame_m")
        narrowed = []
        for collider in candidates:
            if collider.kind != wanted_kind:
                continue
            if wanted_radius is not None and (
                collider.radius_m is None or abs(collider.radius_m - float(wanted_radius)) > tol
            ):
                continue
            if wanted_center is not None and any(
                abs(a - float(b)) > tol for a, b in zip(collider.center_root_m, wanted_center, strict=True)
            ):
                continue
            narrowed.append(collider)
        candidates = narrowed

    expected = int(spec["expect_matches"])
    if len(candidates) != expected:
        raise PatchPlanError(
            f"material {material_name!r} selects {len(candidates)} colliders but must select "
            f"exactly {expected}. Selector: bind_to={spec['bind_to']}, "
            f"restrict_to_shape={shape}. Colliders on those links: "
            f"{[(c.prim_path, c.kind, c.radius_m, c.center_root_m) for c in candidates]}"
        )
    return candidates


def _scene_after_replacements(scene: RobotScene, replacements: list[ReplaceCollider]) -> RobotScene:
    """Return a copy of the scene with the planned collider swaps already applied.

    The bindings are planned against this view, so a wheel whose cylinder is about to become a
    sphere is selected by exactly the same rule as a wheel that was a sphere all along.

    Args:
        scene: The scene as extracted.
        replacements: Planned swaps.

    Returns:
        A new scene with the swapped colliders rewritten in place.
    """
    if not replacements:
        return scene
    by_path = {r.prim_path: r for r in replacements}
    colliders = []
    for collider in scene.colliders:
        swap = by_path.get(collider.prim_path)
        if swap is None:
            colliders.append(collider)
            continue
        colliders.append(
            Collider(
                prim_path=collider.prim_path,
                body=collider.body,
                kind="sphere",
                center_root_m=collider.center_root_m,
                radius_m=swap.radius_m,
                material_path=collider.material_path,
                enabled=collider.enabled,
            )
        )
    return RobotScene(
        source=scene.source,
        root_prim_path=scene.root_prim_path,
        meters_per_unit=scene.meters_per_unit,
        up_axis=scene.up_axis,
        bodies=list(scene.bodies),
        joints=list(scene.joints),
        colliders=colliders,
        materials=dict(scene.materials),
        articulation_root_paths=list(scene.articulation_root_paths),
        mesh_prim_paths=list(scene.mesh_prim_paths),
    )


def plan_patch(scene: RobotScene, params: DuckiebotParams = DUCKIEBOT) -> PatchPlan:
    """Compute the patch plan for one scene.

    Args:
        scene: The scene as extracted from the imported USD.
        params: Parameter set holding the target values.

    Returns:
        The plan. An already-correct asset yields a plan whose bindings are all no-ops when
        applied, never an empty selector.

    Raises:
        PatchPlanError: If a round collider sits on something other than a wheel, or if a
            material selector does not resolve to exactly the number of colliders it declares.
    """
    wheel_links = (params.left_wheel_link_name, params.right_wheel_link_name)
    replacements = []
    for collider in scene.colliders:
        if collider.kind not in _ROUND_KINDS:
            continue
        if collider.body not in wheel_links:
            raise PatchPlanError(
                f"{collider.prim_path} is a {collider.kind} collider on {collider.body!r}. "
                "S3.2 only sanctions replacing a round collider on a wheel link; a round collider "
                "anywhere else means the URDF changed and this script must be revisited."
            )
        replacements.append(
            ReplaceCollider(
                prim_path=collider.prim_path,
                from_kind=collider.kind,
                radius_m=params.wheel_radius_m,
                body=collider.body,
            )
        )

    patched_scene = _scene_after_replacements(scene, replacements)
    spec = physics_material_spec(params)
    bindings = []
    for material_name, material_spec in spec.items():
        for collider in _select_colliders(patched_scene, material_spec, material_name):
            bindings.append(
                BindMaterial(prim_path=collider.prim_path, material_name=material_name, body=collider.body)
            )

    bound_paths = [b.prim_path for b in bindings]
    duplicated = {p for p in bound_paths if bound_paths.count(p) > 1}
    if duplicated:
        raise PatchPlanError(
            f"colliders {sorted(duplicated)} are selected by more than one physics material; "
            "the last binding would silently win"
        )
    unbound = sorted(c.prim_path for c in patched_scene.colliders if c.prim_path not in bound_paths)
    return PatchPlan(replacements=replacements, bindings=bindings, materials=spec, unbound=unbound)


# =============================================================================================
# Application: the USD side
# =============================================================================================


def _de_instance_ancestors(stage: Any, prim_path: str, report: PatchReport) -> None:
    """Turn every instanceable ancestor of a prim into an ordinary prim.

    Args:
        stage: The stage being patched.
        prim_path: Path of the prim that must become editable.
        report: Report to record the affected paths in.
    """
    from pxr import Sdf

    path = Sdf.Path(prim_path)
    for ancestor_path in path.GetAncestorsRange():
        if ancestor_path.IsAbsoluteRootPath():
            continue
        prim = stage.GetPrimAtPath(ancestor_path)
        if prim and prim.IsValid() and prim.IsInstanceable():
            prim.SetInstanceable(False)
            report.de_instanced.append(str(ancestor_path))


def _author_material(stage: Any, root_path: str, name: str, spec: dict[str, Any]) -> str:
    """Author one rigid-body physics material and return its prim path.

    ``UsdPhysics.MaterialAPI`` carries the frictions and restitution. The three PhysX extensions
    are authored as raw attributes with their schema names, because ``PhysxSchema`` lives in a
    different Isaac Sim extension than USD and does not exist in a plain usd-core install; the
    PhysX USD parser reads the attributes, not the Python binding that created them.

    Args:
        stage: The stage being patched.
        root_path: Path of the robot root prim.
        name: Material name, used as the prim name.
        spec: The :func:`physics_material_spec` entry.

    Returns:
        The material prim path.
    """
    from pxr import Sdf, UsdPhysics, UsdShade

    scope_path = f"{root_path}/{MATERIAL_SCOPE_NAME}"
    stage.DefinePrim(scope_path, "Scope")
    material_path = f"{scope_path}/{name}"
    material = UsdShade.Material.Define(stage, material_path)
    prim = material.GetPrim()

    physics_material = UsdPhysics.MaterialAPI.Apply(prim)
    physics_material.CreateStaticFrictionAttr().Set(float(spec["static_friction"]))
    physics_material.CreateDynamicFrictionAttr().Set(float(spec["dynamic_friction"]))
    physics_material.CreateRestitutionAttr().Set(float(spec["restitution"]))

    prim.CreateAttribute(PHYSX_FRICTION_COMBINE_ATTR, Sdf.ValueTypeNames.Token).Set(
        str(spec["friction_combine_mode"])
    )
    prim.CreateAttribute(PHYSX_RESTITUTION_COMBINE_ATTR, Sdf.ValueTypeNames.Token).Set(
        str(spec["restitution_combine_mode"])
    )
    prim.CreateAttribute(PHYSX_IMPROVE_PATCH_ATTR, Sdf.ValueTypeNames.Bool).Set(
        bool(spec["improve_patch_friction"])
    )
    return material_path


def _replace_collider(stage: Any, action: ReplaceCollider, report: PatchReport) -> str:
    """Deactivate one round collider and define a sphere in its place.

    The pose lives on the parent Xform that the importer creates per collision element, so the
    replacement sphere is authored as a sibling with an identity local transform and inherits
    exactly the pose the URDF asked for. ``verify_usd`` re-measures the result, so a mistake here
    cannot pass unnoticed.

    Args:
        stage: The stage being patched.
        action: The planned replacement.
        report: Report to record the replaced path in.

    Returns:
        The path of the new sphere prim.

    Raises:
        PatchPlanError: If the collider prim is missing from the stage.
    """
    from pxr import UsdGeom, UsdPhysics

    prim = stage.GetPrimAtPath(action.prim_path)
    if not prim or not prim.IsValid():
        raise PatchPlanError(f"{action.prim_path} vanished between planning and patching")
    parent = prim.GetParent()
    name = "sphere" if not parent.GetChild("sphere") else f"{prim.GetName()}_sphere"
    sphere_path = f"{parent.GetPath()}/{name}"

    prim.SetActive(False)
    report.replaced.append(action.prim_path)

    sphere = UsdGeom.Sphere.Define(stage, sphere_path)
    sphere.CreateRadiusAttr().Set(float(action.radius_m))
    radius = float(action.radius_m)
    sphere.CreateExtentAttr().Set([(-radius, -radius, -radius), (radius, radius, radius)])
    UsdGeom.Imageable(sphere).CreatePurposeAttr().Set(UsdGeom.Tokens.guide)
    UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
    return sphere_path


def apply_patch(stage: Any, plan: PatchPlan, save: bool = True) -> PatchReport:
    """Apply a plan to an open stage.

    Args:
        stage: The stage to patch. Its root layer is the edit target, so the importer's output
            under ``configuration/`` is never modified.
        plan: The plan from :func:`plan_patch`.
        save: Whether to write the root layer back to disk.

    Returns:
        What was done.

    Raises:
        PatchPlanError: If a planned prim is missing from the stage.
    """
    ensure_pxr()
    from pxr import UsdShade

    root_path = str(stage.GetDefaultPrim().GetPath())
    report = PatchReport(usd_path=str(stage.GetRootLayer().identifier), plan=plan)

    material_paths = {}
    for name, spec in plan.materials.items():
        material_paths[name] = _author_material(stage, root_path, name, spec)
        report.created_materials.append(material_paths[name])

    replaced_paths = {}
    for action in plan.replacements:
        _de_instance_ancestors(stage, action.prim_path, report)
        replaced_paths[action.prim_path] = _replace_collider(stage, action, report)

    for binding in plan.bindings:
        target_path = replaced_paths.get(binding.prim_path, binding.prim_path)
        _de_instance_ancestors(stage, target_path, report)
        prim = stage.GetPrimAtPath(target_path)
        if not prim or not prim.IsValid():
            raise PatchPlanError(f"{target_path} vanished between planning and patching")
        material = UsdShade.Material.Get(stage, material_paths[binding.material_name])
        binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
        binding_api.Bind(
            material,
            bindingStrength=UsdShade.Tokens.weakerThanDescendants,
            materialPurpose=PHYSICS_BINDING_PURPOSE,
        )
        report.bound.append((target_path, material_paths[binding.material_name]))

    if save:
        stage.GetRootLayer().Save()
        report.saved = True
    return report


def patch_usd_file(
    usd_path: str | Path, params: DuckiebotParams = DUCKIEBOT, save: bool = True
) -> PatchReport:
    """Open, plan and patch one robot USD.

    Args:
        usd_path: Path to the imported asset.
        params: Parameter set holding the target values.
        save: Whether to write the changes back to disk.

    Returns:
        What was done.

    Raises:
        StageOpenError: If the stage cannot be opened.
        PatchPlanError: If the asset cannot be patched as specified.
    """
    stage = open_stage(usd_path)
    scene = extract_scene(stage, source=str(usd_path), params=params)
    plan = plan_patch(scene, params)
    return apply_patch(stage, plan, save=save)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]``.

    Returns:
        The parsed namespace with ``usd``, ``dry_run`` and ``quiet`` attributes.
    """
    parser = argparse.ArgumentParser(
        prog="patch_usd.py",
        description=(
            "Swap round wheel colliders for spheres and bind the wheel and caster physics "
            "materials on an imported Duckiebot USD (SPEC v2 S3.2)."
        ),
    )
    parser.add_argument(
        "usd",
        nargs="?",
        default=DEFAULT_USD_PATH,
        help=f"path to the robot USD (default: {DEFAULT_USD_PATH})",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the plan and write nothing")
    parser.add_argument("--quiet", action="store_true", help="print only the final summary line")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code: ``0`` patched, ``1`` the asset cannot be patched as specified, ``2``
        the stage could not be opened.
    """
    args = parse_args(argv)
    try:
        report = patch_usd_file(args.usd, save=not args.dry_run)
    except StageOpenError as error:
        print(f"patch_usd: {error}", file=sys.stderr)
        return 2
    except PatchPlanError as error:
        print(f"patch_usd: {error}", file=sys.stderr)
        return 1

    if args.quiet:
        print(
            f"patch_usd: {len(report.bound)} bindings, {len(report.replaced)} collider "
            f"replacements, saved={report.saved}"
        )
    else:
        print(report.format())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
