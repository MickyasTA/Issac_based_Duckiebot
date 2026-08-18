"""Clean-room asset gate (SPEC v2 S3.4).

The Duckietown asset licenses are non-commercial and grant no redistribution right, so no
Duckietown mesh, texture or scene file may ever enter this Apache-2.0 repository, converted or
not. A prose policy is not enough: the v1 plan had one and the prototypes were contaminated
anyway. This script is the machine gate that runs in CI, in pre-commit and in the M0 and M2
acceptance checks.

Five rules, each of which fails the build:

0. **Generated output is inspected.** Every directory this project's generators write into
   (:data:`GENERATED_OUTPUT_DIRS`) must be fully covered by the content rules below, and the
   run must have opened at least ``--min-files`` files. This rule exists because the gate once
   shipped with ``"build"`` in :data:`EXCLUDED_DIRS` while ``scripts/build_city.py`` defaulted
   to ``--out build/city``: 68 ``.usda`` and 117 ``.png`` files sat on disk, the gate inspected
   exactly one file (the URDF), and it reported PASS. :data:`EXCLUDED_DIRS` is now checked at
   import time and may never shadow a generator target.
1. **Banned geometry formats.** ``.obj .glb .gltf .mtl .stl .dae .fbx .usd .usdc .usdz`` are
   opaque containers: geometry inside them cannot be reviewed in a diff. USD must be text
   ``.usda``. Meshes must not exist at all; every shape is a primitive or is generated. A
   ``.png`` that does not start with the PNG signature is a container in disguise and fails the
   same rule.
2. **No Duckietown provenance strings inside assets.** ``duckietown`` may appear in prose
   (README, NOTICE, docs, source comments describing interoperability) but never inside an
   asset file, where it would indicate a converted upstream artifact.
3. **Vertex budget on .usda.** A city ``.usda`` declares its tile count in layer metadata
   (``customLayerData = { int tiles = N }``, authored by
   :func:`duckiebot_rl.city.usd_builder.build_city_usda`) and may then hold at most
   ``4 * N + 64`` mesh points, which is what a merged quad-per-tile layout needs. A ``.usda``
   that declares no tile count was not produced by this generator and gets the 64-point slack
   and nothing more. A robot ``.usda`` may hold no ``Mesh`` prim at all: robot geometry is
   ``Cube``, ``Cylinder``, ``Sphere`` or ``Capsule``.
4. **PNG manifest.** Every PNG in the tree must be listed in a ``MANIFEST.yaml`` with either
   the generator script that produced it or its third-party (BSD-2 AprilTag) source. Manifest
   keys are resolved as paths (against the scan root, against the manifest's own directory and
   against this repository's root), never matched as bare filenames.

Exit code 0 means clean. Anything else is a licensing problem, not a style problem.

Usage::

    python scripts/check_clean_room.py [--root .] [--min-files N] [--verbose]
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
"""Root of this repository, one of the bases a manifest key may be relative to."""

BANNED_SUFFIXES: frozenset[str] = frozenset(
    {".obj", ".glb", ".gltf", ".mtl", ".stl", ".dae", ".fbx", ".usd", ".usdc", ".usdz", ".blend"}
)
"""Binary or opaque geometry container formats, banned repository-wide."""

EXCLUDED_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".tox",
        ".nox",
        "node_modules",
        "site-packages",
        "_research",  # research dumps: gitignored, holds the contaminated prototypes
        "_refs",  # vendor reference downloads: gitignored
        ".tmp",  # local scratch for one-off probe scripts and frame strips: gitignored
        "_remeasure_tmp",  # redirected run roots from throughput measurements: training output
        "checkpoints",  # training output: weights and metrics, never geometry
        "runs",
        "training_results",  # house-standard run root: models, CSVs and generated graph PNGs
        "logs",
        "wandb",
        "videos",
        "dist",  # packaging output: a repack of files this gate already inspected
    }
)
"""Directories skipped entirely.

``_research`` and ``_refs`` are gitignored by design: they hold the contaminated prototypes that
this project reads for dimensional reference and never copies. The rest are tool caches,
packaging output or training output, none of which can hold an asset.

Generated *asset* output is deliberately not in this set. ``build`` used to be, which made the
whole gate vacuous; :data:`GENERATED_OUTPUT_DIRS` and :func:`_check_configuration` now make that
mistake impossible to reintroduce.
"""

GENERATED_OUTPUT_DIRS: tuple[str, ...] = ("assets", "build/city", "assets/city", "assets/usd")
"""Directories this project's generators write assets into, relative to the scan root.

``build/city`` is the default ``--out`` of ``scripts/build_city.py`` and
``tests/unit/test_clean_room.py`` asserts that the two stay in step. Rule 0 requires that every
inspectable file inside these directories is actually opened by a content rule.
"""

ASSET_TEXT_SUFFIXES: frozenset[str] = frozenset({".usda", ".urdf", ".xml", ".mjcf", ".sdf"})
"""Text asset formats that are scanned for upstream provenance strings."""

INSPECTABLE_SUFFIXES: frozenset[str] = ASSET_TEXT_SUFFIXES | frozenset({".png"})
"""Suffixes a content rule must open. Rule 0 counts these when checking generator coverage."""

PROSE_ALLOWLIST: tuple[str, ...] = (
    "README.md",
    "NOTICE",
    "THIRD_PARTY_LICENSES.md",
    "CONTRIBUTING.md",
    "CITATION.cff",
    "CHANGELOG.md",
    "MODEL_CARD.md",
    "docs/",
    "scripts/check_clean_room.py",
    "tests/",
)
"""Paths where the word "duckietown" is legitimate prose: acknowledgement, citation, the gate
itself and its self-tests. Asset files are never in this list.
"""

BANNED_STRING = "duckietown"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
MESH_POINTS_SLACK = 64
"""Extra vertices allowed per city file beyond ``4 * tiles`` (walls, stop lines, sign cards)."""

DEFAULT_MIN_INSPECTED = 1
"""Command-line floor on the number of files the gate must open.

One, because a bare checkout of this repository contains exactly one committed asset file
(``assets/duckiebot/duckiebot.urdf``); every other asset is generated and gitignored, so a
larger constant would fail the pre-commit hook on a clean clone. The floor with real teeth is
rule 0's coverage check, which is derived from what is on disk rather than from a constant: once
``scripts/build_city.py`` has run, every one of its 186 asset files must be inspected or the
gate fails. CI runs the generator first and then passes an explicit ``--min-files``, so the CI
leg cannot go green on an empty scan either.
"""

_MESH_PRIM_RE = re.compile(r'\bdef\s+Mesh\b|\bover\s+Mesh\b|"Mesh"')
_POINTS_RE = re.compile(r"point3f\[\]\s+points\s*=\s*\[(.*?)\]", re.DOTALL)
_TILE_HINT_RE = re.compile(r"\bint\s+tiles\s*=\s*(\d+)\b")
"""Tile-count hint, anchored to USD's typed-integer syntax inside ``customLayerData``.

The anchor is load-bearing. The previous pattern was ``tiles?\\s*[:=]\\s*(\\d+)``, which matched
the substring ``tile = 0`` inside the generator's own ``double lane_center_offset_tile = 0.2``:
every real city file was read as having zero tiles and was then reported as a licensing
violation. ``int`` cannot precede a float-valued key, so no ``double``, ``float`` or ``string``
metadatum can be parsed as a tile count again.
"""

_PRIM_START_RE = re.compile(r"^(?:def|over|class)\s", re.MULTILINE)


def _check_configuration() -> None:
    """Fail loudly at import if the exclusion list would hide generated assets.

    Raises:
        RuntimeError: If any component of a :data:`GENERATED_OUTPUT_DIRS` entry appears in
            :data:`EXCLUDED_DIRS`, which would make the gate skip the very files it exists to
            inspect.
    """
    for entry in GENERATED_OUTPUT_DIRS:
        for part in entry.split("/"):
            if part in EXCLUDED_DIRS:
                raise RuntimeError(
                    f"clean-room gate misconfigured: EXCLUDED_DIRS contains {part!r}, which "
                    f"hides the generator output directory {entry!r}. The gate would report "
                    f"PASS without reading a single generated asset."
                )


_check_configuration()


@dataclass
class GateResult:
    """Outcome of the clean-room gate.

    Attributes:
        violations: One human-readable line per violation.
        inspected: Absolute, resolved paths of every file a content rule opened.
        checked_rules: Names of the rules that ran.
    """

    violations: list[str] = field(default_factory=list)
    inspected: set[Path] = field(default_factory=set)
    checked_rules: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Whether the tree is clean."""
        return not self.violations

    @property
    def scanned(self) -> int:
        """Number of distinct files opened by a content rule."""
        return len(self.inspected)

    def note_inspected(self, path: Path) -> None:
        """Record that a content rule opened ``path``.

        Args:
            path: File that was read.
        """
        self.inspected.add(path.resolve())

    def add(self, rule: str, path: Path, message: str) -> None:
        """Record one violation.

        Args:
            rule: Rule identifier, for example ``"R1"``.
            path: Offending file.
            message: What is wrong and why it matters.
        """
        self.violations.append(f"[{rule}] {path.as_posix()}: {message}")


def iter_files(root: Path) -> Iterable[Path]:
    """Walk the tree, skipping excluded directories.

    Args:
        root: Repository root.

    Yields:
        Every file worth inspecting.
    """
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in EXCLUDED_DIRS for part in path.relative_to(root).parts[:-1]):
            continue
        yield path


def is_prose_path(relative: Path) -> bool:
    """Whether a path is allowed to mention the upstream project in prose.

    Args:
        relative: Path relative to the repository root.

    Returns:
        True when the path is on the prose allowlist.
    """
    text = relative.as_posix()
    return any(text == entry or text.startswith(entry) for entry in PROSE_ALLOWLIST)


def check_banned_formats(root: Path, result: GateResult) -> None:
    """Rule 1: no binary or opaque geometry containers, and no container in PNG clothing.

    Args:
        root: Repository root.
        result: Result accumulator.
    """
    result.checked_rules.append("R1 banned geometry formats")
    for path in iter_files(root):
        suffix = path.suffix.lower()
        if suffix in BANNED_SUFFIXES:
            result.add(
                "R1",
                path.relative_to(root),
                f"'{path.suffix}' is a banned geometry container; author primitives or generate text .usda",
            )
            continue
        if suffix != ".png":
            continue
        try:
            with path.open("rb") as handle:
                head = handle.read(len(PNG_MAGIC))
        except OSError:
            continue
        result.note_inspected(path)
        if head != PNG_MAGIC:
            result.add(
                "R1",
                path.relative_to(root),
                "has a .png suffix but not the PNG signature; a renamed container is still a "
                "container and its contents cannot be reviewed in a diff",
            )


def check_provenance_strings(root: Path, result: GateResult) -> None:
    """Rule 2: no upstream provenance strings inside asset files.

    Args:
        root: Repository root.
        result: Result accumulator.
    """
    result.checked_rules.append("R2 upstream provenance strings in assets")
    for path in iter_files(root):
        relative = path.relative_to(root)
        in_assets = bool(relative.parts) and relative.parts[0] == "assets"
        is_asset_text = path.suffix.lower() in ASSET_TEXT_SUFFIXES
        if not (in_assets or is_asset_text) or is_prose_path(relative):
            continue
        if path.suffix.lower() == ".png":
            continue  # binary payload; rule 1 checks its signature and rule 4 its provenance
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        result.note_inspected(path)
        if BANNED_STRING in text.lower():
            result.add(
                "R2",
                relative,
                f"contains the string '{BANNED_STRING}'; asset files must carry no upstream "
                "provenance (the acknowledgement belongs in README/NOTICE only)",
            )


def _count_tiles(text: str) -> int | None:
    """Read the tile count a city ``.usda`` declares in its layer metadata.

    Only the layer header is searched, so a hint cannot be smuggled in from inside a prim body.

    Args:
        text: File contents.

    Returns:
        The declared tile count, or ``None`` when the file declares none.
    """
    prim_start = _PRIM_START_RE.search(text)
    header = text[: prim_start.start()] if prim_start else text
    match = _TILE_HINT_RE.search(header)
    return int(match.group(1)) if match else None


def check_usda_geometry(root: Path, result: GateResult) -> None:
    """Rule 3: vertex budget for city ``.usda``; zero Mesh prims for robot ``.usda``.

    Args:
        root: Repository root.
        result: Result accumulator.
    """
    result.checked_rules.append("R3 usda mesh budget")
    for path in iter_files(root):
        if path.suffix.lower() != ".usda":
            continue
        relative = path.relative_to(root)
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        result.note_inspected(path)
        is_robot = "robot" in relative.as_posix().lower() or "duckiebot" in relative.name.lower()
        has_mesh = bool(_MESH_PRIM_RE.search(text))
        if is_robot and has_mesh:
            result.add(
                "R3",
                relative,
                "robot USD must contain no Mesh prim; use Cube/Cylinder/Sphere/Capsule primitives",
            )
            continue
        if not has_mesh:
            continue
        points = sum(block.count("(") for block in _POINTS_RE.findall(text))
        tiles = _count_tiles(text)
        budget = MESH_POINTS_SLACK if tiles is None else 4 * tiles + MESH_POINTS_SLACK
        declared = (
            f"declares no 'int tiles = N' in its layer metadata, so it gets the "
            f"{MESH_POINTS_SLACK}-point slack and nothing more"
            if tiles is None
            else f"declares {tiles} tiles"
        )
        if points > budget:
            result.add(
                "R3",
                relative,
                f"{points} mesh points exceed the quad-per-tile budget of {budget}; the file "
                f"{declared}. A merged tile mesh needs 4 points per tile, anything more "
                "suggests an imported mesh",
            )


def _parse_manifest_keys(manifest_path: Path) -> set[str]:
    """Read the path-like strings out of one ``MANIFEST.yaml``.

    Args:
        manifest_path: Path to the manifest.

    Returns:
        Every string in the manifest that could name an asset.
    """
    text = manifest_path.read_text(encoding="utf-8")
    try:
        import yaml  # optional; the manifest is readable without it
    except ImportError:
        # Flat line scan: the manifest is a mapping of path -> provenance, one key per line.
        entries: set[str] = set()
        for line in text.splitlines():
            stripped = line.strip().lstrip("- ").strip()
            if ".png" in stripped.lower():
                entries.add(stripped.split(":")[0].strip().strip("\"'"))
        return entries

    entries = set()

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                entries.add(str(key))
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)
        elif isinstance(node, str):
            entries.add(node)

    walk(yaml.safe_load(text) or {})
    return entries


def _manifested_paths(root: Path, manifests: Sequence[Path]) -> set[Path]:
    """Resolve every manifest key into the absolute path it names.

    A key is resolved against three bases: the scan root, the manifest's own directory, and this
    repository's root. That covers a manifest written for an in-repo build (keys relative to the
    repository), for a build into ``assets/`` (keys relative to the tree root) and for an
    out-of-tree build (keys relative to the output directory), without ever falling back to
    matching a bare filename, which would let any image anywhere inherit another image's
    provenance.

    Args:
        root: Tree being scanned.
        manifests: Manifest files found in the tree.

    Returns:
        Absolute resolved paths that the manifests account for.
    """
    resolved: set[Path] = set()
    for manifest in manifests:
        bases = {root, manifest.parent, REPO_ROOT}
        for key in _parse_manifest_keys(manifest):
            if not key or key.startswith("/") or "\\" in key or ":" in key:
                continue
            for base in bases:
                try:
                    resolved.add((base / key).resolve())
                except (OSError, ValueError):  # pragma: no cover - exotic path input
                    continue
    return resolved


def check_png_manifest(root: Path, result: GateResult) -> None:
    """Rule 4: every PNG in the tree is accounted for in a manifest.

    Args:
        root: Repository root.
        result: Result accumulator.
    """
    result.checked_rules.append("R4 png manifest")
    pngs = [p for p in iter_files(root) if p.suffix.lower() == ".png"]
    if not pngs:
        return
    manifests = [p for p in iter_files(root) if p.name == "MANIFEST.yaml"]
    if not manifests:
        result.add(
            "R4",
            Path("MANIFEST.yaml"),
            f"{len(pngs)} PNG file(s) exist in the tree but no MANIFEST.yaml was found; every "
            "image must name its generator script or its BSD-2 AprilTag source",
        )
        return
    accounted = _manifested_paths(root, manifests)
    for png in pngs:
        result.note_inspected(png)
        if png.resolve() not in accounted:
            result.add(
                "R4",
                png.relative_to(root),
                "not listed in any MANIFEST.yaml under a key that resolves to it; unmanifested "
                "images cannot be shown to be generated rather than copied",
            )


def check_generated_output_coverage(root: Path, result: GateResult) -> None:
    """Rule 0a: the gate must have opened every asset inside every generator output directory.

    This is the anti-vacuity rule. It walks :data:`GENERATED_OUTPUT_DIRS` *without* applying
    :data:`EXCLUDED_DIRS`, so an exclusion that hides generated assets from the content rules
    cannot hide them from this check as well. Run it last, after the content rules have
    populated :attr:`GateResult.inspected`.

    Args:
        root: Repository root.
        result: Result accumulator.
    """
    result.checked_rules.append("R0 generated output coverage")
    for entry in GENERATED_OUTPUT_DIRS:
        directory = root / entry
        if not directory.is_dir():
            continue
        present = {
            p.resolve()
            for p in directory.rglob("*")
            if p.is_file() and p.suffix.lower() in INSPECTABLE_SUFFIXES
        }
        missed = present - result.inspected
        if missed:
            example = sorted(missed)[0]
            result.add(
                "R0",
                Path(entry),
                f"{len(missed)} of {len(present)} generated asset file(s) were never inspected "
                f"(for example {example.name}); the gate cannot certify files it never reads",
            )


def check_minimum_inspection(result: GateResult, min_files: int) -> None:
    """Rule 0b: explicit floor on how many files the gate opened.

    Args:
        result: Result accumulator.
        min_files: Minimum number of distinct files a content rule must have opened.
    """
    if min_files <= 0:
        return
    result.checked_rules.append(f"R0 minimum inspection floor ({min_files})")
    if result.scanned < min_files:
        result.add(
            "R0",
            Path(),
            f"inspected only {result.scanned} file(s), below the required floor of {min_files}; "
            "a gate that reads nothing certifies nothing",
        )


def run_gate(root: Path, min_files: int = 0) -> GateResult:
    """Run every rule against a tree.

    Args:
        root: Repository root to scan.
        min_files: Minimum number of files the content rules must open; ``0`` disables the
            explicit floor and leaves rule 0's derived coverage check as the anti-vacuity guard.

    Returns:
        The accumulated :class:`GateResult`.
    """
    result = GateResult()
    check_banned_formats(root, result)
    check_provenance_strings(root, result)
    check_usda_geometry(root, result)
    check_png_manifest(root, result)
    check_generated_output_coverage(root, result)
    check_minimum_inspection(result, min_files)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Run the clean-room gate as a command.

    Args:
        argv: Argument list; ``sys.argv[1:]`` when ``None``.

    Returns:
        0 when the tree is clean, 1 otherwise.
    """
    parser = argparse.ArgumentParser(
        description="Clean-room asset gate: fail the build on any redistributable-asset risk.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="tree to scan")
    parser.add_argument(
        "--min-files",
        type=int,
        default=DEFAULT_MIN_INSPECTED,
        help="fail if fewer than this many files were opened by a content rule",
    )
    parser.add_argument("--verbose", action="store_true", help="list the rules that ran")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    result = run_gate(root, min_files=args.min_files)

    if args.verbose:
        print(f"clean-room gate over {root}")
        for rule in result.checked_rules:
            print(f"  rule: {rule}")
        print(f"  files inspected for content: {result.scanned}")

    if result.ok:
        print(
            f"clean-room gate: PASS ({result.scanned} file(s) inspected; no banned formats, "
            "no upstream assets, manifest complete)"
        )
        return 0
    print(f"clean-room gate: FAIL ({len(result.violations)} violation(s))", file=sys.stderr)
    for violation in result.violations:
        print(f"  {violation}", file=sys.stderr)
    print(
        "\nThese are licensing violations, not style issues. Duckietown assets are "
        "non-commercial with no redistribution grant and cannot ship in an Apache-2.0 repo.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
