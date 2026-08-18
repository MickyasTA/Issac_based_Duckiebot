"""Generate the procedural Duckietown city: maps, tile textures and text USD stages.

This is the ``[city]`` half of SPEC v2 milestone M2. It writes, under ``--out``:

.. code-block:: text

    maps/<name>.yaml              one MapFormat1-shaped YAML per layout
    textures/bucket_NN/*.png      one marking-geometry bucket per SPEC v2 S3.3 (16 by default)
    textures/signs/*.png          procedural sign faces (visual distractors only)
    usd/<name>.usda               one text USD stage per layout
    usd/ground.usda               the single physics ground plane
    MANIFEST.yaml                 every PNG mapped to its generator, seed and sha256

Everything is deterministic in ``--seed``: re-running the script reproduces byte-identical PNGs
and YAML, which is what lets ``scripts/check_clean_room.py`` verify the manifest.

Examples:
---------
.. code-block:: text

    python scripts/build_city.py --list
    python scripts/build_city.py --map loop_small
    python scripts/build_city.py --builtin --out build/city
    python scripts/build_city.py --all --out assets/city
    python scripts/build_city.py --all --no-usd          # works without any USD runtime
    python scripts/build_city.py --all --difficulty hard --out build/city_hard

``--difficulty`` picks the trajectory-complexity profile of the procedural layouts, and
``nominal`` is the historical generator: rebuilding with the default reproduces the existing
layouts byte for byte. A harder or easier set therefore has to go to a different ``--out``, so
that a training run already referencing ``build/city`` keeps the layouts it was started on.

Run it with the Isaac venv python, the MuJoCo tools venv python, or any interpreter that has
numpy and PyYAML; USD is needed only when stages are being written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import yaml  # noqa: E402

from duckiebot_rl.city import maps as city_maps  # noqa: E402
from duckiebot_rl.city import spec as city_spec  # noqa: E402
from duckiebot_rl.city import tiles as city_tiles  # noqa: E402

__all__ = ["build", "main", "resolve_maps"]


def resolve_maps(args: argparse.Namespace) -> list[city_maps.CityMap]:
    """Turn the CLI selection flags into the list of maps to build.

    Args:
        args: Parsed arguments.

    Returns:
        The selected maps, in build order.

    Raises:
        SystemExit: If no selection flag was given, or a named map is unknown.
    """
    selected: list[city_maps.CityMap] = []
    if args.all:
        selected.extend(
            city_maps.variant_maps(
                count=args.variants,
                seed=args.seed,
                geometry_buckets=args.buckets,
                difficulty=args.difficulty,
            )
        )
        # The eval layouts take the same profile as the training set: held-out maps are only a
        # fair test of a training distribution when they are drawn from it.
        selected.extend(
            city_maps.eval_maps(
                count=args.eval_maps,
                train_count=args.variants,
                train_seed=args.seed,
                difficulty=args.difficulty,
            )
        )
    if args.builtin:
        selected.extend(city_maps.builtin_map(name) for name in city_maps.BUILTIN_MAP_NAMES)
    for name in args.map or ():
        path = Path(name)
        if path.suffix in (".yaml", ".yml") and path.is_file():
            selected.append(city_maps.load_map(path))
            continue
        try:
            selected.append(city_maps.builtin_map(name))
        except KeyError as exc:
            raise SystemExit(
                f"error: {exc}. Use --list to see the built-in maps, or pass a path to a map YAML file."
            ) from exc
    if not selected:
        raise SystemExit("error: nothing selected. Pass --map NAME, --builtin, --all, or --list.")
    return selected


def _sha256(path: Path) -> str:
    """Return the hex sha256 of a file."""
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def manifest_key_base(outdir: Path, manifest_path: Path) -> Path:
    """Directory that manifest entry keys are written relative to.

    Rule 4 of ``scripts/check_clean_room.py`` resolves a manifest key as a path, against the
    root of the tree it scans, against the manifest's own directory and against the repository
    root. Keys were previously written relative to ``--out``, which none of those bases can
    resolve when the manifest and the output tree are not the same directory: building with
    ``--out assets/city --manifest assets/MANIFEST.yaml`` produced the key
    ``textures/bucket_00/threeway.png`` for a file the gate knows as
    ``assets/city/textures/bucket_00/threeway.png``, and every generated image came out
    unmanifested.

    Three cases, in order:

    1. the output tree is inside this repository, so the gate will be run at the repository
       root: key relative to the repository root;
    2. an out-of-tree build whose manifest lands in ``<root>/assets/MANIFEST.yaml``: the gate
       will be run at ``<root>``, so key relative to ``<root>``;
    3. any other out-of-tree build: key relative to the manifest's own directory, which the gate
       always tries.

    Args:
        outdir: Resolved output directory.
        manifest_path: Resolved path the manifest will be written to.

    Returns:
        The base directory for manifest keys.
    """
    if outdir.is_relative_to(_REPO_ROOT):
        return _REPO_ROOT
    if manifest_path.parent.name == "assets":
        return manifest_path.parent.parent
    return manifest_path.parent


def _manifest_key(path: Path, key_base: Path) -> str:
    """Key one manifest entry by a path the clean-room gate can resolve.

    Args:
        path: File that was written.
        key_base: Base directory from :func:`manifest_key_base`.

    Returns:
        A forward-slashed relative path, falling back to the file name if ``path`` somehow
        escapes ``key_base``.
    """
    resolved = path.resolve()
    try:
        return resolved.relative_to(key_base).as_posix()
    except ValueError:  # pragma: no cover - only reachable via a hand-built odd layout
        return resolved.name


def _write_textures(
    outdir: Path,
    key_base: Path,
    buckets: Sequence[int],
    n_buckets: int,
    seed: int,
    alpha: float,
    res: int | None,
    supersample: int,
    manifest: dict[str, Any],
) -> None:
    """Render every needed marking-geometry bucket plus the shared sign faces.

    Args:
        outdir: Root output directory.
        key_base: Base directory for manifest keys, from :func:`manifest_key_base`.
        buckets: Bucket indices actually referenced by the selected maps.
        n_buckets: Total number of buckets in the quantisation.
        seed: Base seed.
        alpha: Curriculum scalar for the geometry and palette sampling.
        res: Override for the per-kind texture resolution, or ``None``.
        supersample: Supersampling factor.
        manifest: Manifest dict, extended in place.
    """
    geometry = city_spec.geometry_buckets(count=n_buckets, seed=seed, alpha=alpha)
    palettes = city_spec.palette_buckets(count=n_buckets, seed=seed + 1, alpha=alpha)
    override = dict.fromkeys(city_tiles.TILE_KINDS, res) if res else None
    for bucket in sorted(set(buckets)):
        bucket_dir = outdir / "textures" / f"bucket_{bucket:02d}"
        style = city_tiles.TileStyle(palette=palettes[bucket], noise=0.004, mottle=0.05)
        written = city_tiles.save_tile_set(
            bucket_dir,
            spec=geometry[bucket],
            style=style,
            res=override,
            supersample=supersample,
            seed=seed + 1000 * bucket,
        )
        for kind, path in written.items():
            manifest["entries"][_manifest_key(path, key_base)] = {
                "generator": "duckiebot_rl.city.tiles.render_tile",
                "kind": kind,
                "bucket": bucket,
                "seed": seed + 1000 * bucket,
                "tile_pitch_mm": round(geometry[bucket].tile_pitch_mm, 4),
                "clear_lane_mm": round(geometry[bucket].clear_lane_mm, 4),
                "sha256": _sha256(path),
            }

    sign_dir = outdir / "textures" / "signs"
    for kind, path in city_tiles.save_sign_set(sign_dir, seed=seed + 7).items():
        manifest["entries"][_manifest_key(path, key_base)] = {
            "generator": "duckiebot_rl.city.tiles.render_sign",
            "kind": kind,
            "seed": seed + 7,
            "sha256": _sha256(path),
        }


def build(args: argparse.Namespace) -> int:
    """Run the generator.

    Args:
        args: Parsed arguments.

    Returns:
        Process exit code: ``0`` on success, ``2`` when USD stages were requested but no USD
        runtime is available.
    """
    started = time.perf_counter()
    selected = resolve_maps(args)
    outdir = Path(args.out).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest).resolve() if args.manifest else outdir / "MANIFEST.yaml"
    key_base = manifest_key_base(outdir, manifest_path)

    manifest: dict[str, Any] = {
        "generator": "scripts/build_city.py",
        "package": "duckiebot_rl.city",
        "command": " ".join(sys.argv),
        "seed": args.seed,
        "alpha": args.alpha,
        "geometry_buckets": args.buckets,
        "difficulty": args.difficulty,
        "entries": {},
    }

    for city in selected:
        path = city_maps.save_map(city, outdir / "maps" / f"{city.name}.yaml")
        warnings = city.topology_warnings()
        if warnings and args.verbose:
            for line in warnings:
                print(f"  topology note: {line}")
        manifest["entries"][_manifest_key(path, key_base)] = {
            "generator": "duckiebot_rl.city.maps",
            "tiles": f"{city.n_rows}x{city.n_cols}",
            "drivable": len(city.drivable_cells()),
            "closed_loop": city.is_closed_loop(),
            "complexity_score": city_maps.loop_complexity(city).score,
            "sha256": _sha256(path),
        }
    scores = sorted(city_maps.loop_complexity(c).score for c in selected)
    middle = scores[len(scores) // 2]
    print(f"maps:     {len(selected)} written to {outdir / 'maps'}")
    print(
        f"maps:     difficulty {args.difficulty}, complexity score min {scores[0]} "
        f"median {middle} max {scores[-1]}"
    )

    if args.maps_only:
        _finish(manifest, started, manifest_path)
        return 0

    buckets = [int(c.meta.get("geometry_bucket", 0)) % args.buckets for c in selected]
    _write_textures(
        outdir,
        key_base,
        buckets,
        args.buckets,
        args.seed,
        args.alpha,
        args.res,
        args.supersample,
        manifest,
    )
    n_bucket_dirs = len(set(buckets))
    print(
        f"textures: {n_bucket_dirs} geometry bucket(s) + {len(city_tiles.SIGN_KINDS)} sign faces "
        f"in {outdir / 'textures'}"
    )

    if args.no_usd or args.textures_only:
        _finish(manifest, started, manifest_path)
        return 0

    from duckiebot_rl.city import usd_builder  # imported late: USD is optional

    try:
        usd = usd_builder.ensure_usd()
    except usd_builder.UsdUnavailableError as exc:
        print(f"\nUSD stages were NOT written.\n{exc}", file=sys.stderr)
        _finish(manifest, started, manifest_path)
        return 2
    print(f"usd:      runtime {usd.source} {'.'.join(str(v) for v in usd.version)}")

    geometry = city_spec.geometry_buckets(count=args.buckets, seed=args.seed, alpha=args.alpha)
    palettes = city_spec.palette_buckets(count=args.buckets, seed=args.seed + 1, alpha=args.alpha)
    usd_dir = outdir / "usd"
    total_points = 0
    for city in selected:
        bucket = int(city.meta.get("geometry_bucket", 0)) % args.buckets
        variant = int(city.meta.get("variant_index", 0))
        tint, brightness, roughness = city_spec.variant_material_scalars(
            variant, seed=args.seed, alpha=args.alpha
        )
        target = usd_dir / f"{city.name}.usda"
        if target.exists():
            target.unlink()
        usd_builder.build_city_usda(
            city,
            target,
            texture_dir=f"../textures/bucket_{bucket:02d}",
            sign_texture_dir="../textures/signs",
            spec=geometry[bucket],
            palette=palettes[bucket],
            max_signs=args.signs,
            n_distractors=args.distractors,
            tint=tint,
            albedo_brightness=brightness,
            roughness=roughness,
        )
        total_points += usd_builder.city_vertex_count(city, args.signs)
    ground = usd_dir / "ground.usda"
    if ground.exists():
        ground.unlink()
    usd_builder.build_ground_usda(ground)
    print(f"usd:      {len(selected)} city stage(s) + ground.usda in {usd_dir}")
    print(f"usd:      {total_points} mesh points total (budget 4*tiles + 64 per file)")

    _finish(manifest, started, manifest_path)
    return 0


def _finish(manifest: dict[str, Any], started: float, manifest_path: Path) -> None:
    """Write the manifest and print the wall-clock summary.

    The manifest is written with explicit LF newlines. ``Path.write_text`` translates to CRLF on
    Windows, which changes the file's sha256 and makes a manifest generated on Windows disagree
    with the same build on the Linux CI leg.

    Args:
        manifest: Manifest contents.
        started: ``time.perf_counter()`` value at the start of the build.
        manifest_path: Where to write the manifest. ``build()`` defaults it to
            ``<out>/MANIFEST.yaml``; rule 4 of ``scripts/check_clean_room.py`` finds it wherever
            it lands, as long as the entry keys come from :func:`manifest_key_base`.
    """
    manifest["entry_count"] = len(manifest["entries"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(json.loads(json.dumps(manifest)), sort_keys=False, width=200)
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    print(f"manifest: {manifest['entry_count']} entries in {manifest_path}")
    print(f"done in {time.perf_counter() - started:.2f} s")


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and run the build.

    Args:
        argv: Command line, excluding the program name; defaults to :data:`sys.argv`.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(
        prog="build_city.py",
        description="Generate the clean-room procedural Duckietown city (maps, textures, USD).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    select = parser.add_argument_group("selection")
    select.add_argument(
        "--map",
        action="append",
        metavar="NAME_OR_YAML",
        help="a built-in map name or a path to a map YAML; repeatable",
    )
    select.add_argument("--builtin", action="store_true", help="build every built-in map")
    select.add_argument(
        "--all",
        action="store_true",
        help="build the full training variant set plus the held-out eval maps",
    )
    select.add_argument("--list", action="store_true", help="list the built-in maps and exit")
    select.add_argument("--variants", type=int, default=64, help="training variants for --all")
    select.add_argument("--eval-maps", type=int, default=4, help="held-out eval maps for --all")
    select.add_argument(
        "--difficulty",
        choices=city_maps.DIFFICULTY_NAMES,
        default="nominal",
        help="trajectory-complexity profile of the procedural layouts for --all. 'nominal' is "
        "the historical generator and reproduces build/city byte for byte; 'hard' drops the "
        "gentle built-ins, uses the 8x8 36-tile loops and keeps the twistiest of 24 candidate "
        "layouts per slot. Send a non-nominal build to its own --out",
    )

    output = parser.add_argument_group("output")
    output.add_argument("--out", default="build/city", help="output directory")
    output.add_argument("--maps-only", action="store_true", help="write map YAML only")
    output.add_argument("--textures-only", action="store_true", help="skip the USD stages")
    output.add_argument("--no-usd", action="store_true", help="alias of --textures-only")
    output.add_argument(
        "--manifest",
        default=None,
        help="where to write MANIFEST.yaml; defaults to <out>/MANIFEST.yaml. Entry keys are "
        "keyed off this location so that scripts/check_clean_room.py rule 4 can resolve "
        "them wherever the manifest lands",
    )
    output.add_argument("-v", "--verbose", action="store_true", help="print topology notes")

    look = parser.add_argument_group("appearance")
    look.add_argument("--seed", type=int, default=0, help="master seed; everything derives from it")
    look.add_argument(
        "--buckets",
        type=int,
        default=city_spec.GEOMETRY_BUCKET_COUNT,
        help="marking-geometry texture buckets",
    )
    look.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="domain-randomisation curriculum scalar in [0, 1]; 0 gives the nominal spec",
    )
    look.add_argument("--res", type=int, default=None, help="override the texture resolution")
    look.add_argument("--supersample", type=int, default=2, help="texture supersampling factor")
    look.add_argument("--signs", type=int, default=8, help="roadside sign distractors per city")
    look.add_argument("--distractors", type=int, default=4, help="off-road props per city")

    args = parser.parse_args(argv)
    if args.list:
        print("built-in maps:")
        for name in city_maps.BUILTIN_MAP_NAMES:
            city = city_maps.builtin_map(name)
            print(
                f"  {name:20s} {city.n_rows}x{city.n_cols} grid, "
                f"{len(city.drivable_cells()):3d} drivable, "
                f"{'closed loop' if city.is_closed_loop() else 'has intersections'}, "
                f"{len(city.objects)} objects"
            )
        print("\n--all additionally generates city_000.. and eval_00.. procedurally.")
        return 0
    if args.buckets <= 0:
        parser.error("--buckets must be > 0")
    if not 0.0 <= args.alpha <= 1.0:
        parser.error("--alpha must be in [0, 1]")
    return build(args)


if __name__ == "__main__":
    raise SystemExit(main())
