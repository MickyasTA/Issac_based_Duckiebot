"""Import the clean-room Duckiebot URDF into USD headlessly, then patch and verify it.

This is the missing step between ``scripts/build_robot_asset.py`` (which writes the URDF) and
everything that spawns a robot: it produces ``assets/usd/duckiebot.usda``, the file
:data:`duckiebot_rl.assets.robot_cfg.DEFAULT_USD_PATH` points at. One command does the whole
S3.2 chain:

.. code-block:: text

    python tools/import_urdf_headless.py
      1. check the committed URDF is not stale with respect to duckiebot_rl.assets.params
      2. boot Isaac Sim headless and import the URDF with the raw isaacsim commands
      3. run tools/patch_usd.py  (collider swap + physics materials)
      4. flatten the importer's layer stack into one text .usda and delete the binary staging
      5. run tools/verify_usd.py (the M1 acceptance assertions) on that final artifact

Run it with the Isaac Sim python:
``d:/Personal/personal/wheeled_quadruped_robot/.venv/Scripts/python.exe``.

Why the raw importer commands and not ``isaaclab.sim.UrdfConverter``
--------------------------------------------------------------------
``isaaclab/sim/converters/urdf_converter.py`` calls
``import_config.set_merge_fixed_ignore_inertia(...)``. That setter does not exist in the URDF
importer this install ships (``isaacsim.asset.importer.urdf`` v2.4.30, whose ``ImportConfig``
exposes 18 setters, none of them that one), so ``UrdfConverter`` raises ``AttributeError`` before
it imports anything. Rather than monkeypatching a private Isaac Lab method, this script drives
the four public ``omni.kit.commands`` the importer registers, which is the same path Isaac Lab
itself takes underneath and is stable across both versions:

``URDFCreateImportConfig`` -> ``URDFParseFile`` -> ``URDFImportRobot``.

:data:`IMPORT_SETTINGS` is the entire import configuration, as data, with a reason attached to
every entry. Two of those entries do real work rather than restating a default:
``set_collision_from_visuals(False)``, because the wheel VISUAL is a cylinder and only the
``<collision>`` spheres may become colliders, and ``set_import_inertia_tensor(True)`` with
``set_density(0.0)``, because the chassis inertia must come from the URDF and not from the
oversized collision box.

Why the output is flattened into one text .usda
------------------------------------------------
The importer writes binary USDC (the file starts ``PXR-USDC``) plus a ``configuration/``
directory of four more binary layers. SPEC v2 S3.4 rule 1 bans ``.usd``, ``.usdc`` and ``.usdz``
repository-wide and ``scripts/check_clean_room.py`` enforces it, so a build that left those files
in ``assets/usd/`` would turn the licensing gate red (5 violations, verified). The import
therefore goes to a staging directory, the patch is applied there, and the composed result is
flattened into a single text layer at the output path, after which the staging directory is
deleted. The flattened asset is about 20 KB of readable USD with no ``Mesh`` prim in it, which is
also what makes clean-room rule 3 checkable on the robot at all.

Why the import runs in a child process
--------------------------------------
``SimulationApp.close()`` never returns: Isaac Sim launches Kit with ``--/app/fastShutdown=True``
and the shutdown path calls ``shutdown_and_release_framework()``, which terminates the process
where it stands, with exit status 0 whatever happened. Anything written after ``close()`` in the
same process, including the patch step, the verification and a non-zero exit code, is simply
never reached. The import therefore runs as ``python tools/import_urdf_headless.py
--import-only``, and the parent process (which never boots Kit) decides success from the
``IMPORT_OK`` sentinel the child prints and from the file on disk, then patches and verifies.

Exit codes: ``0`` imported, patched and verified; ``1`` verification failed; ``2`` the patch could
not be applied; ``3`` the import itself failed; ``4`` the URDF on disk is stale.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TOOLS_DIR = Path(__file__).resolve().parent
for _path in (str(_REPO_ROOT), str(_TOOLS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from patch_usd import PatchPlanError, patch_usd_file  # noqa: E402
from verify_usd import (  # noqa: E402
    StageOpenError,
    ensure_pxr,
    format_report,
    verify_usd_file,
)

from duckiebot_rl.assets.params import DUCKIEBOT, DuckiebotParams  # noqa: E402
from duckiebot_rl.assets.robot_cfg import DEFAULT_USD_PATH  # noqa: E402
from duckiebot_rl.assets.urdf import URDF_FILENAME, generate_urdf  # noqa: E402

__all__ = [
    "DEFAULT_URDF_PATH",
    "IMPORT_OK_SENTINEL",
    "IMPORT_SETTINGS",
    "URDF_IMPORTER_EXTENSION",
    "ImportError_",
    "apply_import_settings",
    "check_urdf_is_current",
    "import_urdf",
    "main",
    "run_import_subprocess",
]

IMPORT_OK_SENTINEL = "IMPORT_OK"
"""Line the ``--import-only`` child prints once the USD is on disk.

Kit's fast shutdown makes the child's exit status meaningless, so this sentinel plus the presence
of the output file is what the parent process trusts.
"""

DEFAULT_URDF_PATH = (_REPO_ROOT / "assets" / "duckiebot" / URDF_FILENAME).as_posix()
"""The URDF written by ``scripts/build_robot_asset.py``."""

URDF_IMPORTER_EXTENSION = "isaacsim.asset.importer.urdf"
"""Kit extension that registers the four URDF import commands."""

IMPORT_SETTINGS: dict[str, tuple[Any, ...]] = {
    # Units and framing.
    "set_distance_scale": (1.0,),
    "set_up_vector": (0.0, 0.0, 1.0),
    "set_make_default_prim": (True,),
    # The environment owns the physics scene; an asset that carries its own fights it.
    "set_create_physics_scene": (False,),
    # Mass properties: take the URDF's inertia tensors verbatim, never infer from geometry.
    # density 0.0 means "do not compute mass from volume".
    "set_density": (0.0,),
    "set_import_inertia_tensor": (True,),
    # Colliders come from the <collision> elements only. The wheel VISUAL is a cylinder and must
    # never become a collider (SPEC v2 S3.2: a cylindrical wheel contact costs ~74% of yaw
    # response), and the camera block and duckie marker are decoration with no collider at all.
    "set_collision_from_visuals": (False,),
    "set_convex_decomp": (False,),
    # There are no meshes and no capsules in this asset; both of these keep it that way.
    "set_replace_cylinders_with_capsules": (False,),
    # Structure: merge any fixed joint into its parent (3 bodies / 2 DOF), free-floating base.
    "set_merge_fixed_joints": (True,),
    "set_fix_base": (False,),
    "set_self_collision": (False,),
    "set_parse_mimic": (False,),
}
"""The complete URDF import configuration, as ``setter name -> positional arguments``.

Kept as data so that it can be reviewed in one place, printed into the build log and asserted on
by a unit test with a recording double, without Isaac Sim.
"""


class ImportError_(RuntimeError):
    """Raised when the headless URDF import cannot be completed.

    Named with a trailing underscore so it cannot shadow the builtin ``ImportError``.
    """


def apply_import_settings(
    import_config: Any, settings: dict[str, tuple[Any, ...]] | None = None
) -> list[str]:
    """Apply :data:`IMPORT_SETTINGS` to an ``ImportConfig``.

    Every setter is required. If one is missing the import stops with a message naming it, which
    is exactly the failure Isaac Lab's ``UrdfConverter`` hits on this install and reports as a
    bare ``AttributeError`` from inside a private method.

    Args:
        import_config: The ``isaacsim.asset.importer.urdf._urdf.ImportConfig`` to configure.
        settings: Setter name to positional arguments. Defaults to :data:`IMPORT_SETTINGS`.

    Returns:
        The applied settings, one ``"name(args)"`` string per entry, in application order.

    Raises:
        ImportError_: If the ``ImportConfig`` of this Isaac Sim build lacks a required setter.
    """
    applied = []
    for name, args in (settings or IMPORT_SETTINGS).items():
        setter = getattr(import_config, name, None)
        if setter is None:
            available = sorted(n for n in dir(import_config) if n.startswith("set_"))
            raise ImportError_(
                f"ImportConfig has no {name}(). This Isaac Sim build exposes {available}. "
                "The import configuration in tools/import_urdf_headless.py must be updated for "
                "this version before the asset can be built."
            )
        setter(*args)
        applied.append(f"{name}({', '.join(repr(a) for a in args)})")
    return applied


def check_urdf_is_current(urdf_path: Path, params: DuckiebotParams = DUCKIEBOT) -> None:
    """Fail if the URDF on disk does not match the parameters it is generated from.

    Importing a stale URDF produces a USD that passes nothing and explains nothing, so this runs
    before Kit boots rather than after five minutes of import.

    Args:
        urdf_path: Path to the committed URDF.
        params: Parameter set the URDF must match.

    Raises:
        ImportError_: If the file is missing or out of date.
    """
    if not urdf_path.is_file():
        raise ImportError_(f"{urdf_path} does not exist. Write it with: python scripts/build_robot_asset.py")
    if urdf_path.read_text(encoding="utf-8") != generate_urdf(params):
        raise ImportError_(
            f"{urdf_path} is stale with respect to duckiebot_rl.assets.params. "
            "Regenerate it with: python scripts/build_robot_asset.py"
        )


def import_urdf(
    urdf_path: str | Path,
    output_path: str | Path,
    clean: bool = True,
    quiet: bool = False,
) -> str:
    """Boot Isaac Sim headless and convert the URDF into a USD asset.

    Args:
        urdf_path: Path to the URDF to import.
        output_path: Path of the ``.usd`` to write. The importer also writes a ``configuration/``
            directory of sublayers next to it.
        clean: Whether to delete a previous build first, so the result never depends on what was
            already on disk.
        quiet: Whether to suppress the progress lines.

    Returns:
        The prim path of the imported robot on the temporary stage, for example ``/duckiebot``.

    Raises:
        ImportError_: If Isaac Sim, the importer extension, or the import itself fails.
    """
    urdf = Path(urdf_path).resolve()
    output = Path(output_path).resolve()

    def log(message: str) -> None:
        """Print a progress line unless running quiet."""
        if not quiet:
            print(f"import_urdf_headless: {message}", flush=True)

    if clean and output.parent.exists():
        for stale in (output, output.parent / "configuration"):
            if stale.is_dir():
                shutil.rmtree(stale)
            elif stale.is_file():
                stale.unlink()
        log(f"cleaned {output.parent}")
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        from isaacsim import SimulationApp
    except Exception as error:  # pragma: no cover - depends entirely on the interpreter
        raise ImportError_(
            "isaacsim is not importable. Run this with the Isaac Sim python: "
            "d:/Personal/personal/wheeled_quadruped_robot/.venv/Scripts/python.exe"
        ) from error

    log("booting Isaac Sim headless (this takes about 15 s)")
    app = SimulationApp({"headless": True})
    try:
        from isaacsim.core.utils.extensions import enable_extension

        if not enable_extension(URDF_IMPORTER_EXTENSION):
            raise ImportError_(f"could not enable the {URDF_IMPORTER_EXTENSION} extension")
        import omni.kit.commands

        status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
        if not status:
            raise ImportError_("URDFCreateImportConfig failed")
        for line in apply_import_settings(import_config):
            log(f"  {line}")

        status, robot_model = omni.kit.commands.execute(
            "URDFParseFile", urdf_path=str(urdf), import_config=import_config
        )
        if not status or robot_model is None:
            raise ImportError_(f"URDFParseFile failed on {urdf}")
        log(f"parsed links {list(robot_model.links.keys())} joints {list(robot_model.joints.keys())}")

        status, prim_path = omni.kit.commands.execute(
            "URDFImportRobot",
            urdf_path=str(urdf),
            urdf_robot=robot_model,
            import_config=import_config,
            dest_path=output.as_posix(),
            get_articulation_root=False,
        )
        if not status or not prim_path:
            raise ImportError_(f"URDFImportRobot failed writing {output}")
        if not output.is_file():
            raise ImportError_(f"the importer reported success but {output} does not exist")
        log(f"wrote {output} with robot prim {prim_path}")
        # Printed before close() because close() terminates the process (see the module
        # docstring): nothing after this line is guaranteed to run.
        print(f"{IMPORT_OK_SENTINEL} {prim_path}", flush=True)
    finally:
        app.close()
    return str(prim_path)


def run_import_subprocess(
    urdf_path: str | Path, output_path: str | Path, clean: bool = True, quiet: bool = False
) -> str:
    """Run the Kit import in a child process and wait for it.

    Args:
        urdf_path: Path to the URDF to import.
        output_path: Path of the ``.usd`` to write.
        clean: Whether the child should delete a previous build first.
        quiet: Whether to suppress the child's progress lines. The Kit boot log is always shown,
            because when the import fails it is the only place the reason appears.

    Returns:
        The robot prim path the child reported.

    Raises:
        ImportError_: If the child did not report success or did not write the file.
    """
    output = Path(output_path).resolve()
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--import-only",
        "--urdf",
        str(Path(urdf_path).resolve()),
        "--output",
        str(output),
    ]
    if not clean:
        command.append("--keep-existing")
    if quiet:
        command.append("--quiet")

    prim_path = ""
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=None, text=True, bufsize=1) as child:
        assert child.stdout is not None
        for line in child.stdout:
            print(line, end="", flush=True)
            if line.startswith(IMPORT_OK_SENTINEL):
                prim_path = line.split(maxsplit=1)[1].strip()
        child.wait()

    if not prim_path:
        raise ImportError_(
            "the headless import did not report success. The Kit log above holds the reason; "
            f"the child command was: {' '.join(command)}"
        )
    if not output.is_file():
        raise ImportError_(f"the import reported success but {output} does not exist")
    return prim_path


def staging_path_for(output_path: str | Path) -> Path:
    """Return the binary staging path the importer writes before flattening.

    Args:
        output_path: The final ``.usda`` the build produces.

    Returns:
        Path of the staging ``.usd``, in a ``_import`` directory beside the output.
    """
    output = Path(output_path).resolve()
    return output.parent / "_import" / f"{output.stem}.usd"


def flatten_to_text_usda(source_path: str | Path, output_path: str | Path) -> int:
    """Compose the importer's layer stack and write it out as one text layer.

    Args:
        source_path: The staged ``.usd`` the importer wrote.
        output_path: Where to write the flattened text asset.

    Returns:
        Size of the written file in bytes.

    Raises:
        ImportError_: If the flattened layer could not be written.
    """
    ensure_pxr()
    from pxr import Usd

    stage = Usd.Stage.Open(str(source_path), Usd.Stage.LoadAll)
    if stage is None:
        raise ImportError_(f"could not open {source_path} to flatten it")
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not stage.Flatten().Export(output.as_posix()):
        raise ImportError_(f"USD refused to export the flattened stage to {output}")
    return output.stat().st_size


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]``.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(
        prog="import_urdf_headless.py",
        description=(
            "Import the clean-room Duckiebot URDF into USD with Isaac Sim headless, then patch "
            "and verify it (SPEC v2 S3.2, M1 acceptance)."
        ),
    )
    parser.add_argument(
        "--urdf", type=Path, default=Path(DEFAULT_URDF_PATH), help=f"default: {DEFAULT_URDF_PATH}"
    )
    parser.add_argument(
        "--output", type=Path, default=Path(DEFAULT_USD_PATH), help=f"default: {DEFAULT_USD_PATH}"
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="do not delete a previous build before importing",
    )
    parser.add_argument("--no-patch", action="store_true", help="skip the S3.2 patch step")
    parser.add_argument("--no-verify", action="store_true", help="skip the M1 verification step")
    parser.add_argument(
        "--keep-import-layers",
        action="store_true",
        help=(
            "write the importer's binary layer stack straight to --output instead of flattening "
            "it into one text .usda. For debugging the importer only: the result violates the "
            "clean-room gate (rule 1 bans .usd and .usdc)."
        ),
    )
    parser.add_argument("--quiet", action="store_true", help="suppress progress lines")
    parser.add_argument(
        "--import-only",
        action="store_true",
        help=(
            "boot Kit and import, then stop. This is how the script re-invokes itself: Kit's "
            "shutdown terminates the process, so the patch and verify steps cannot follow it "
            "in the same interpreter."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point: check, import, patch, verify.

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code: ``0`` success, ``1`` verification failed, ``2`` the patch failed,
        ``3`` the import failed, ``4`` the URDF is stale or missing.
    """
    args = parse_args(argv)
    try:
        check_urdf_is_current(args.urdf)
    except ImportError_ as error:
        print(f"import_urdf_headless: {error}", file=sys.stderr)
        return 4

    if args.import_only:
        try:
            import_urdf(args.urdf, args.output, clean=not args.keep_existing, quiet=args.quiet)
        except ImportError_ as error:
            print(f"import_urdf_headless: {error}", file=sys.stderr)
            return 3
        return 0  # not reached in practice: Kit's shutdown ends the process first

    staged = Path(args.output) if args.keep_import_layers else staging_path_for(args.output)
    try:
        run_import_subprocess(args.urdf, staged, clean=not args.keep_existing, quiet=args.quiet)
    except ImportError_ as error:
        print(f"import_urdf_headless: {error}", file=sys.stderr)
        return 3

    if not args.no_patch:
        try:
            report = patch_usd_file(staged)
        except (PatchPlanError, StageOpenError) as error:
            print(f"import_urdf_headless: patch failed: {error}", file=sys.stderr)
            return 2
        if not args.quiet:
            print(report.format())

    if not args.keep_import_layers:
        try:
            size = flatten_to_text_usda(staged, args.output)
        except ImportError_ as error:
            print(f"import_urdf_headless: {error}", file=sys.stderr)
            return 3
        shutil.rmtree(staged.parent, ignore_errors=True)
        if not args.quiet:
            print(
                f"import_urdf_headless: flattened into {args.output} ({size} bytes of text USD); "
                f"removed the binary staging directory {staged.parent}"
            )

    if args.no_verify:
        return 0
    try:
        scene, results = verify_usd_file(args.output)
    except StageOpenError as error:
        print(f"import_urdf_headless: verification failed: {error}", file=sys.stderr)
        return 1
    print(format_report(scene, results))
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
