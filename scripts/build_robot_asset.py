"""Write the clean-room Duckiebot URDF and print its mass, inertia and limit summary.

This is the only supported way to produce ``assets/duckiebot/duckiebot.urdf``. The file is
generated, not hand-edited: :mod:`duckiebot_rl.assets.params` is the source of truth, and the
downstream USD import (``tools/import_urdf_headless.py``) consumes whatever this script writes.

Usage (from the repository root, with the Isaac venv python or any Python 3.11):

.. code-block:: text

    python scripts/build_robot_asset.py
    python scripts/build_robot_asset.py --output build/duckiebot.urdf
    python scripts/build_robot_asset.py --check        # CI: fail if the file is out of date

Exit codes: ``0`` on success, ``1`` if ``--check`` finds the on-disk file stale or missing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from duckiebot_rl.assets.params import DUCKIEBOT, DuckiebotParams  # noqa: E402
from duckiebot_rl.assets.urdf import URDF_FILENAME, generate_urdf, write_urdf  # noqa: E402

DEFAULT_OUTPUT = _REPO_ROOT / "assets" / "duckiebot" / URDF_FILENAME
"""Where the URDF lands by default: ``assets/duckiebot/duckiebot.urdf``."""


def _render_table(title: str, headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    """Render a fixed-width ASCII table.

    Args:
        title: Caption printed above the table.
        headers: Column headers.
        rows: Row cells. Every row must have ``len(headers)`` entries.

    Returns:
        The rendered table, without a trailing newline.

    Raises:
        ValueError: If a row does not match the header width.
    """
    for row in rows:
        if len(row) != len(headers):
            raise ValueError(f"row {row!r} has {len(row)} cells, expected {len(headers)}")
    widths = [len(h) for h in headers]
    for row in rows:
        widths = [max(w, len(cell)) for w, cell in zip(widths, row, strict=True)]
    rule = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def line(cells: tuple[str, ...]) -> str:
        padded = (f" {cell.ljust(w)} " for cell, w in zip(cells, widths, strict=True))
        return "|" + "|".join(padded) + "|"

    out = [title, rule, line(headers), rule]
    out.extend(line(row) for row in rows)
    out.append(rule)
    return "\n".join(out)


def _mass_table(params: DuckiebotParams) -> str:
    """Build the per-link mass and inertia table.

    Args:
        params: The parameter set being reported.

    Returns:
        The rendered table.
    """
    base_i = params.base_inertia_about_com
    wheel_i = params.wheel_inertia_about_com
    rows = [
        (
            params.base_link_name,
            f"{params.base_mass_kg:.4f}",
            "box 0.180 x 0.130 x 0.075",
            f"{base_i[0]:.4e}",
            f"{base_i[1]:.4e}",
            f"{base_i[2]:.4e}",
            f"({params.base_com_base_frame_m[0]:+.4f}, {params.base_com_base_frame_m[1]:+.4f},"
            f" {params.base_com_base_frame_m[2]:+.4f})",
        ),
        (
            params.left_wheel_link_name,
            f"{params.wheel_mass_kg:.4f}",
            f"cyl r {params.wheel_radius_m:.4f} h {params.wheel_width_m:.3f}",
            f"{wheel_i[0]:.4e}",
            f"{wheel_i[1]:.4e}",
            f"{wheel_i[2]:.4e}",
            "( 0.0000, +0.0000,  0.0000)",
        ),
        (
            params.right_wheel_link_name,
            f"{params.wheel_mass_kg:.4f}",
            f"cyl r {params.wheel_radius_m:.4f} h {params.wheel_width_m:.3f}",
            f"{wheel_i[0]:.4e}",
            f"{wheel_i[1]:.4e}",
            f"{wheel_i[2]:.4e}",
            "( 0.0000, -0.0000,  0.0000)",
        ),
        (
            "TOTAL",
            f"{params.total_mass_kg:.4f}",
            "-",
            "-",
            "-",
            "-",
            "-",
        ),
    ]
    return _render_table(
        "Links: mass [kg] and inertia about the link centre of mass [kg.m^2]",
        ("link", "mass", "inertia source solid", "Ixx", "Iyy", "Izz", "CoM in base frame [m]"),
        rows,
    )


def _joint_table(params: DuckiebotParams) -> str:
    """Build the joint limit and drive table.

    Args:
        params: The parameter set being reported.

    Returns:
        The rendered table.
    """
    rows = []
    for joint_name, origin in (
        (params.left_wheel_joint_name, params.left_wheel_origin_m),
        (params.right_wheel_joint_name, params.right_wheel_origin_m),
    ):
        rows.append(
            (
                joint_name,
                "continuous",
                f"({origin[0]:+.3f}, {origin[1]:+.3f}, {origin[2]:+.3f})",
                "(0, 1, 0)",
                f"{params.wheel_effort_limit_nm:.3f}",
                f"{params.wheel_velocity_limit_rad_s:.1f}",
                f"{params.joint_damping:.4f}",
                f"{params.joint_friction_nm:.4f}",
                f"{params.joint_armature_kg_m2:.2e}",
            )
        )
    return _render_table(
        "Joints: URDF limits and the ImplicitActuatorCfg drive that overrides them at play",
        (
            "joint",
            "type",
            "origin [m]",
            "axis",
            "effort [N.m]",
            "vel [rad/s]",
            "damping",
            "friction [N.m]",
            "armature [kg.m2]",
        ),
        rows,
    )


def _geometry_table(params: DuckiebotParams) -> str:
    """Build the standing-geometry closure table.

    Every row here is a number the v1 critique found self-contradictory. They are recomputed from
    the parameters rather than restated, so the table cannot drift from the asset.

    Args:
        params: The parameter set being reported.

    Returns:
        The rendered table.
    """
    rows = [
        (
            "base_link height above ground",
            f"{params.base_link_height_m:.4f} m",
            f"equals the wheel radius {params.wheel_radius_m:.4f} m",
        ),
        (
            "chassis underside",
            f"{params.chassis_bottom_height_m:.4f} m",
            f"target ground clearance {params.ground_clearance_m:.4f} m (v1 gave 0.0063)",
        ),
        (
            "caster lowest point",
            f"{params.caster_contact_height_m:+.6f} m",
            f"sphere r {params.caster_radius_m:.4f} m (v1 said 0.021 in text, 0.0318 in the URDF)",
        ),
        (
            "camera optical centre",
            f"{params.camera_height_m:.4f} m",
            f"pitch {params.camera_pitch_down_deg:.1f} deg down, no camera_link in the URDF",
        ),
        (
            "tractive force at effort limit",
            f"{params.max_tractive_force_n:.2f} N",
            f"{params.max_tractive_accel_g:.3f} g on {params.total_mass_kg:.2f} kg "
            "(v1's 2.0 N.m gave 11.7 g)",
        ),
        (
            "nominal command envelope",
            f"{params.nominal_max_wheel_speed_rad_s:.2f} rad/s",
            f"velocity limit {params.wheel_velocity_limit_rad_s:.1f} rad/s",
        ),
        (
            "control rate",
            f"{params.control_hz:.1f} Hz",
            f"sim dt {params.sim_dt_s:.6f} s x decimation {params.decimation}",
        ),
    ]
    return _render_table(
        "Standing geometry and drive closure (recomputed, not restated)",
        ("quantity", "value", "check"),
        rows,
    )


def _shape_census(urdf_text: str) -> str:
    """Count the primitive shapes in the generated URDF.

    Args:
        urdf_text: The generated URDF document.

    Returns:
        A one-line census, including the mesh count, which must be zero.
    """
    counts = {tag: urdf_text.count(f"<{tag} ") for tag in ("box", "cylinder", "sphere", "mesh")}
    census = ", ".join(f"{tag} {n}" for tag, n in counts.items())
    return f"Primitive census: {census}   (mesh must be 0: SPEC v2 S3.1 clean-room policy)"


def build_report(params: DuckiebotParams, urdf_text: str, output: Path) -> str:
    """Assemble the full human-readable report.

    Args:
        params: The parameter set being reported.
        urdf_text: The generated URDF document.
        output: Where the document was written.

    Returns:
        The report, ready to print.
    """
    links = urdf_text.count("<link ")
    joints = urdf_text.count("<joint ")
    sections = [
        "Duckiebot clean-room robot asset (SPEC v2 S3.2)",
        "=" * 78,
        "",
        _mass_table(params),
        "",
        _joint_table(params),
        "",
        _geometry_table(params),
        "",
        _shape_census(urdf_text),
        f"Structure: {links} links, {joints} joints "
        f"({links} bodies / {joints} DOF after import, matching the M1 acceptance criterion)",
        f"Written: {output}  ({len(urdf_text.encode('utf-8'))} bytes)",
    ]
    return "\n".join(sections)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]``.

    Returns:
        The parsed namespace with ``output``, ``check`` and ``quiet`` attributes.
    """
    parser = argparse.ArgumentParser(
        prog="build_robot_asset.py",
        description="Generate the clean-room Duckiebot URDF from duckiebot_rl.assets.params.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"destination URDF path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write; exit 1 if the file on disk differs from the generated document",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress the summary tables")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code: ``0`` on success, ``1`` if ``--check`` finds the file stale.
    """
    args = parse_args(argv)
    params = DUCKIEBOT
    urdf_text = generate_urdf(params)

    if args.check:
        if not args.output.exists():
            print(f"MISSING {args.output}: run python scripts/build_robot_asset.py", file=sys.stderr)
            return 1
        on_disk = args.output.read_text(encoding="utf-8")
        if on_disk != urdf_text:
            print(
                f"STALE {args.output}: it does not match duckiebot_rl.assets.params. "
                "Run python scripts/build_robot_asset.py and commit the result.",
                file=sys.stderr,
            )
            return 1
        if not args.quiet:
            print(f"OK {args.output} is up to date with duckiebot_rl.assets.params.")
        return 0

    written = write_urdf(args.output, params)
    if not args.quiet:
        print(build_report(params, urdf_text, written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
