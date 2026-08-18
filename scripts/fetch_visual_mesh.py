"""Fetch the real Duckiebot visual meshes into ``_refs/`` (gitignored, never committed).

Duckietown's 3D models are licensed for personal, educational and research use with no
redistribution grant, so this repository ships no copy of them: training and physics use the
clean-room primitive robot, and the meshes fetched here upgrade the LOOK of the viewer only.
Each user downloads them from the upstream source under Duckietown's own terms.

Two models:

* **DB21 (latest generation)**: ``duckiebot3/main.gltf`` from ``duckietown/duckietown-world``.
  The file sits behind Git LFS, so the raw endpoint returns a 130-byte pointer; the actual
  bytes come from ``media.githubusercontent.com``. Self-contained glTF (embedded buffer and
  textures), authored in centimetres; the viewer measures and rescales it at attach time.
* **DB17 (classic)**: the per-link OBJ split of gym-duckietown's ``duckiebot.obj``, used as the
  fallback when the DB21 file is absent.

Usage::

    python scripts/fetch_visual_mesh.py [--force]
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEST = REPO_ROOT / "_refs" / "visual_mesh"

DB21_URL = (
    "https://media.githubusercontent.com/media/duckietown/duckietown-world/ente/"
    "src/duckietown_world/data/gd2/meshes/duckiebot3/main.gltf"
)
"""The DB21-generation model. If the branch layout moves, find the file with the GitHub tree
API (``git/trees/ente?recursive=1``) and look for ``duckiebot3/main.gltf``."""

MIN_REAL_BYTES = 100_000
"""A Git LFS pointer is ~130 bytes; real geometry is megabytes. Anything below this is a miss."""


def fetch(url: str, dest: Path, force: bool) -> bool:
    """Download one file, refusing to accept an LFS pointer as geometry.

    Args:
        url: Source URL.
        dest: Destination path.
        force: Redownload even if the destination exists.

    Returns:
        True when the destination holds real geometry.
    """
    if dest.is_file() and dest.stat().st_size >= MIN_REAL_BYTES and not force:
        print(f"already present: {dest} ({dest.stat().st_size:,} bytes)")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"fetching {url}")
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read()
    except OSError as exc:
        print(f"  download failed: {exc}")
        return False
    if len(data) < MIN_REAL_BYTES:
        head = data[:60].decode("utf-8", "replace")
        print(f"  got {len(data)} bytes, which is a pointer, not geometry (starts {head!r})")
        return False
    dest.write_bytes(data)
    print(f"  wrote {dest} ({len(data):,} bytes)")
    return True


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument list, or None for ``sys.argv``.

    Returns:
        Process exit code: 0 when the DB21 model is in place.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="redownload even if present")
    args = parser.parse_args(argv)

    ok = fetch(DB21_URL, DEST / "db21" / "main.gltf", args.force)
    if ok:
        print("done. The Isaac viewer picks the DB21 model up automatically on next launch.")
    else:
        print("DB21 fetch failed; the viewer will fall back to per-part or primitive visuals.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
