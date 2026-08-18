"""Fetch the real Duckiebot visual meshes into ``_refs/`` (gitignored, never committed).

Duckietown's 3D models are licensed for personal, educational and research use with no
redistribution grant, so this repository ships no copy of them: training and physics use the
clean-room primitive robot, and the meshes fetched here upgrade the LOOK of the viewer and of the
training scene only. Each user downloads them from the upstream source under Duckietown's own
terms.

Three sources, in the order the code tries them
-----------------------------------------------

1. **DB21, the real latest-generation robot** (``_refs/visual_mesh/db21j/main.obj``). There is no
   standalone download of it: the model ships inside Duckietown's own **Duckiematrix** simulator,
   as Unity assets. This script downloads the engine build, extracts the ``DB21`` GameObject
   hierarchy with `UnityPy <https://pypi.org/project/UnityPy/>`_, bakes each part's world
   transform and writes one grouped OBJ. 281,367 vertices, 184,838 faces, metres, Y-up, no
   materials. Roughly 123 MB downloaded and a few minutes of work, once per machine.
2. **The DB18-era glTF** (``_refs/visual_mesh/db21/main.gltf``), from
   ``duckietown/duckietown-world``. Upstream calls the directory ``duckiebot3``, but the file is
   export_DB18's asset: it is NOT a DB21, and the viewer labels it honestly. It stays because it
   is one self-contained file behind a plain HTTP GET, which is a useful fallback when the
   Duckiematrix path cannot run. The file sits behind Git LFS, so the raw endpoint returns a
   130-byte pointer; the actual bytes come from ``media.githubusercontent.com``.
3. **DB17 (classic)**: the per-link OBJ split of gym-duckietown's ``duckiebot.obj``. Not fetched
   here; it is the on-disk fallback under ``_research/prototypes/db2/meshes``.

Usage::

    python scripts/fetch_visual_mesh.py                 # everything that is missing
    python scripts/fetch_visual_mesh.py --force         # redownload and re-extract
    python scripts/fetch_visual_mesh.py --skip-duckiematrix   # glTF only, no 123 MB download
    python scripts/fetch_visual_mesh.py --keep-engine   # leave the extracted engine on disk

``UnityPy`` is installed on demand into the interpreter running this script, which is the Isaac
venv in practice; it is also declared as the ``mesh`` extra (``pip install -e .[mesh]``) so the
dependency is recorded rather than conjured. Nothing else in the project imports it.

Reproducing the extraction by hand
----------------------------------
:func:`export_db21` is the tidied form of the throwaway scripts the model was first found with
(``_refs/db21_hunt/dashboard_lead/scripts/``, gitignored like the rest of ``_refs``). The
algorithm is theirs, and the vertex and face counts this script verifies are the counts those
scripts produced, so a mismatch means the upstream build changed, not that the port drifted.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEST = REPO_ROOT / "_refs" / "visual_mesh"

DB18_GLTF_URL = (
    "https://media.githubusercontent.com/media/duckietown/duckietown-world/ente/"
    "src/duckietown_world/data/gd2/meshes/duckiebot3/main.gltf"
)
"""The DB18-era model, mislabelled ``duckiebot3`` upstream. If the branch layout moves, find the
file with the GitHub tree API (``git/trees/ente?recursive=1``)."""

DUCKIEMATRIX_URL = (
    "https://duckietown-public-storage.s3.amazonaws.com/assets/duckiematrix/releases/"
    "duckiematrix-0.11.4-windows.zip"
)
"""Duckietown's public S3 release of the Duckiematrix engine, which carries the DB21 model.

The Windows build is used because its ``Duckiematrix_Data`` layout is the one this extraction was
developed and verified against. The Unity assets inside are platform-independent; only the
executable next to them is not, and it is never run.
"""

DUCKIEMATRIX_CACHE = REPO_ROOT / "_refs" / "duckiematrix"
"""Where the engine zip and its extraction are cached, so ``--force`` is the only redownload."""

ENGINE_DATA_DIR = "Duckiematrix_Data"
"""The Unity data directory inside the release archive. UnityPy is pointed at this."""

_ENGINE_SKIP_PREFIXES = (
    f"{ENGINE_DATA_DIR}/il2cpp_data/",
    f"{ENGINE_DATA_DIR}/Plugins/",
)
"""Subtrees of the data directory that hold no assets: compiled IL2CPP metadata and native
plugins, about 40 MB of the archive that UnityPy would never open."""

DB21_GAMEOBJECT = "DB21"
"""Name of the root GameObject to export. Its children are the plates, bumpers, camera mast,
Jetson devkit, wheels and caster, and the group names in the OBJ are their paths."""

_SKIP_MESH_NODE = "FakeCamera"
"""Node whose geometry is dropped, along with every descendant's.

``FakeCamera`` is the render-target placeholder Duckiematrix uses for the robot's simulated
camera: a flat quad hanging in front of the mast, which is scenery for the simulator's own UI and
not part of the robot.
"""

_SKIP_MESH_PATH = "Curve_002"
"""Child of the root whose geometry is dropped. It is a duplicate of the wheel spline, parented
directly to the root and spanning the whole model. Only the root-level one is dropped: the wheels
themselves carry nodes of the same name (``left_motor/wheel/Curve_002``) and those ARE the tyres.
"""

EXPECTED_VERTICES = 281_367
"""Vertices the 0.11.4 build produces. Verified after every export."""

EXPECTED_FACES = 184_838
"""Faces the 0.11.4 build produces."""

EXPECTED_SIZE_M: tuple[float, float, float] = (0.215083, 0.121565, 0.134002)
"""Bounding box of the export in the OBJ's own frame [m]: length along X, height along Y (the
source is Y-up) and width along Z. ``duckiebot_rl.envs.viz_env`` rotates this into Isaac's frame;
its ``DB21_OBJ_EXPECTED_SIZE_M`` is the same box, reordered."""

_SIZE_TOL_M = 0.002
"""How far a fresh export's box may fall from :data:`EXPECTED_SIZE_M` before it is called a
different model [m]."""

MIN_REAL_BYTES = 100_000
"""A Git LFS pointer is ~130 bytes; real geometry is megabytes. Anything below this is a miss."""

_UNITYPY_REQUIREMENT = "UnityPy>=1.10"
"""What is installed on demand, and what the ``mesh`` extra declares."""


# ------------------------------------------------------------------------------- downloading


def fetch(url: str, dest: Path, force: bool) -> bool:
    """Download one small file, refusing to accept an LFS pointer as geometry.

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


def download_stream(url: str, dest: Path, force: bool) -> bool:
    """Download a large file straight to disk, in chunks.

    :func:`fetch` reads the whole body into memory, which is fine for a 2 MB glTF and wrong for a
    123 MB engine build on a machine whose commit limit is the documented reason Isaac sessions
    die here. The download goes to a ``.part`` file and is renamed only once it completes, so an
    interrupted run leaves no half file that the next run would trust.

    Args:
        url: Source URL.
        dest: Destination path.
        force: Redownload even if the destination exists.

    Returns:
        True when ``dest`` holds the downloaded file.
    """
    if dest.is_file() and dest.stat().st_size >= MIN_REAL_BYTES and not force:
        print(f"already present: {dest} ({dest.stat().st_size:,} bytes)")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    print(f"fetching {url}")
    try:
        with urllib.request.urlopen(url, timeout=300) as response, part.open("wb") as handle:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)
        print()
    except OSError as exc:
        print(f"\n  download failed: {exc}")
        part.unlink(missing_ok=True)
        return False
    part.replace(dest)
    print(f"  wrote {dest} ({dest.stat().st_size:,} bytes)")
    return True


def find_cached_zip() -> Path | None:
    """Return an engine archive that is already on this machine, if there is one.

    Downloading 123 MB twice because the file sits under a different ``_refs`` subdirectory is
    pure waste, so the hunt's own download location is checked too.

    Returns:
        Path to an existing archive, or None.
    """
    name = DUCKIEMATRIX_URL.rsplit("/", 1)[-1]
    for candidate in (
        DUCKIEMATRIX_CACHE / name,
        REPO_ROOT / "_refs" / "db21_hunt" / "dashboard_lead" / name,
    ):
        if candidate.is_file() and candidate.stat().st_size >= MIN_REAL_BYTES:
            return candidate
    return None


def engine_members(names: list[str]) -> list[str]:
    """Select the archive entries worth extracting.

    Only the Unity data directory is needed, and not all of it: the IL2CPP metadata and the native
    plugins are about a third of the archive and contain no assets. The executable and the
    ``GameAssembly.dll`` next to it are never extracted at all, which is also why this script can
    honestly say it downloads a game engine and never runs one.

    Args:
        names: Every entry name in the archive.

    Returns:
        The entries to extract, in archive order.
    """
    wanted = []
    for name in names:
        if not name.startswith(f"{ENGINE_DATA_DIR}/") or name.endswith("/"):
            continue
        if any(name.startswith(prefix) for prefix in _ENGINE_SKIP_PREFIXES):
            continue
        wanted.append(name)
    return wanted


def extract_engine(archive: Path, out_dir: Path, force: bool) -> Path | None:
    """Extract the engine's Unity data directory.

    Args:
        archive: The downloaded zip.
        out_dir: Directory to extract into.
        force: Re-extract even when the data directory is already there.

    Returns:
        The extracted ``Duckiematrix_Data`` directory, or None on failure.
    """
    data_dir = out_dir / ENGINE_DATA_DIR
    if data_dir.is_dir() and not force:
        print(f"already extracted: {data_dir}")
        return data_dir
    if force and data_dir.is_dir():
        shutil.rmtree(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive) as zf:
            members = engine_members(zf.namelist())
            if not members:
                print(f"  {archive.name} has no {ENGINE_DATA_DIR}/ entries; is it the engine build?")
                return None
            print(f"extracting {len(members)} asset files from {archive.name}")
            zf.extractall(out_dir, members=members)
    except (OSError, zipfile.BadZipFile) as exc:
        print(f"  extraction failed: {exc}")
        return None
    return data_dir if data_dir.is_dir() else None


# --------------------------------------------------------------------------- the DB21 export


def ensure_unitypy(install: bool = True) -> Any:
    """Import UnityPy, installing it into the running interpreter if it is missing.

    Args:
        install: Attempt the pip install. False turns a missing dependency into a clean None.

    Returns:
        The imported module, or None when it is unavailable.
    """
    import importlib

    try:
        return importlib.import_module("UnityPy")
    except ImportError:
        pass
    if not install:
        print(f"  UnityPy is not installed; install it with: pip install {_UNITYPY_REQUIREMENT!r}")
        return None
    print(f"installing {_UNITYPY_REQUIREMENT} into {sys.executable}")
    # The interpreter's own pip and a constant requirement: nothing here comes from the user.
    result = subprocess.run([sys.executable, "-m", "pip", "install", _UNITYPY_REQUIREMENT], check=False)
    if result.returncode != 0:
        print(f"  pip install failed with code {result.returncode}")
        return None
    importlib.invalidate_caches()
    try:
        return importlib.import_module("UnityPy")
    except ImportError as exc:
        print(f"  UnityPy still not importable after install: {exc}")
        return None


def _quat_to_mat(quat: Any) -> list[list[float]]:
    """Convert a Unity quaternion to a 3x3 rotation matrix.

    Args:
        quat: An object with ``x``, ``y``, ``z`` and ``w`` attributes.

    Returns:
        The rotation matrix as three rows of three floats.
    """
    x, y, z, w = float(quat.x), float(quat.y), float(quat.z), float(quat.w)
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def _mat_mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    """Multiply two 3x3 matrices.

    Args:
        a: Left matrix.
        b: Right matrix.

    Returns:
        ``a @ b``.
    """
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _apply(mat: list[list[float]], translation: tuple[float, float, float], point: Any) -> tuple[float, ...]:
    """Apply a 3x3 matrix and a translation to a point.

    Args:
        mat: Rotation-and-scale matrix.
        translation: Translation to add afterwards.
        point: ``(x, y, z)``.

    Returns:
        The transformed point.
    """
    return tuple(
        mat[i][0] * point[0] + mat[i][1] * point[1] + mat[i][2] * point[2] + translation[i] for i in range(3)
    )


def _read_component(component: Any) -> Any:
    """Dereference one component pointer of a GameObject.

    UnityPy has spelled this both ways across releases: the entry is either a PPtr with ``read``
    or a wrapper carrying ``.component``. Both are accepted so the export does not break on a
    dependency bump it has no other reason to care about.

    Args:
        component: The entry from ``GameObject.m_Components``.

    Returns:
        The read component, or None when it cannot be dereferenced.
    """
    try:
        return component.read() if hasattr(component, "read") else component.component.read()
    except Exception:
        return None


def _transform_of(game_object: Any) -> Any:
    """Return a GameObject's Transform component.

    Args:
        game_object: The GameObject.

    Returns:
        The Transform, or None when it has none (which cannot happen in a valid scene).
    """
    for component in game_object.m_Components:
        read = _read_component(component)
        if read is not None and "Transform" in type(read).__name__:
            return read
    return None


class _ObjWriter:
    """Accumulates the flattened OBJ: baked vertices, faces, and one group per mesh node.

    Attributes:
        vertices: World-space vertices, in the OBJ's own handedness.
        lines: ``g`` and ``f`` lines, in emission order.
        parts: ``(node path, mesh name, vertex count)`` per exported mesh, for the report.
    """

    def __init__(self) -> None:
        """Start an empty document."""
        self.vertices: list[tuple[float, float, float]] = []
        self.lines: list[str] = []
        self.parts: list[tuple[str, str, int]] = []

    def add_mesh(self, path: str, mesh: Any, mat: list[list[float]], translation: tuple[float, ...]) -> None:
        """Bake one mesh into the document under its node's world transform.

        UnityPy's own OBJ export negates X (it converts Unity's left-handed frame to a
        right-handed one at the very last step). The negation is undone before the world transform
        is applied, so the transform runs in the frame the transform was authored in, and reapplied
        on the way out. Getting that order wrong mirrors every child about its parent's origin,
        which looks plausible on a symmetric part and wrong on the camera mast.

        Args:
            path: The node's path under the root, used as the OBJ group name.
            mesh: The Unity Mesh object.
            mat: The node's world rotation-and-scale matrix.
            translation: The node's world translation.
        """
        try:
            text = mesh.export()
        except Exception as exc:
            print(f"  mesh export failed under {path}: {exc!r}")
            return
        base = len(self.vertices)
        self.lines.append("g " + path.replace(" ", "_"))
        count = 0
        for line in text.splitlines():
            if line.startswith("v "):
                fields = line.split()
                unity = (-float(fields[1]), float(fields[2]), float(fields[3]))
                world = _apply(mat, translation, unity)
                self.vertices.append((-world[0], world[1], world[2]))
                count += 1
            elif line.startswith("f "):
                indices = [base + int(token.split("/")[0]) for token in line.split()[1:]]
                self.lines.append("f " + " ".join(str(index) for index in indices))
        self.parts.append((path, str(getattr(mesh, "m_Name", "")), count))

    def write(self, dest: Path, header: str) -> None:
        """Write the document.

        Args:
            dest: Destination ``.obj`` path.
            header: Comment line placed at the top of the file.
        """
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w", encoding="utf-8") as handle:
            handle.write(f"# {header}\n")
            for x, y, z in self.vertices:
                handle.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
            for line in self.lines:
                handle.write(line + "\n")


def skips_mesh(path: str) -> bool:
    """Return whether the node at ``path`` contributes geometry.

    A node is skipped for its own geometry only; the walk still descends through it, because a
    dropped parent does not make its children scenery. That distinction is why this is a function
    of the path rather than an early return in the walk.

    Args:
        path: The node's path under the root, root name included.

    Returns:
        True when the node's meshes must not be baked into the export.
    """
    name = path.rsplit("/", 1)[-1]
    if name == _SKIP_MESH_NODE or f"/{_SKIP_MESH_NODE}" in path:
        return True
    return path == f"{DB21_GAMEOBJECT}/{_SKIP_MESH_PATH}"


def _walk(
    writer: _ObjWriter, transform: Any, mat: list[list[float]], trans: tuple[float, ...], prefix: str
) -> None:
    """Recursively bake a Transform subtree into the OBJ document.

    Args:
        writer: The document being built.
        transform: The Transform to visit.
        mat: The parent's world rotation-and-scale matrix.
        trans: The parent's world translation.
        prefix: Path of the parent, ending in ``/`` (empty at the root).
    """
    game_object = transform.m_GameObject.read()
    name = str(game_object.m_Name)
    path = prefix + name
    local = _mat_mul(
        _quat_to_mat(transform.m_LocalRotation),
        [
            [float(transform.m_LocalScale.x), 0.0, 0.0],
            [0.0, float(transform.m_LocalScale.y), 0.0],
            [0.0, 0.0, float(transform.m_LocalScale.z)],
        ],
    )
    world_mat = _mat_mul(mat, local)
    position = transform.m_LocalPosition
    world_trans = _apply(mat, trans, (float(position.x), float(position.y), float(position.z)))

    if not skips_mesh(path):
        for component in game_object.m_Components:
            read = _read_component(component)
            if read is None or type(read).__name__ != "MeshFilter":
                continue
            try:
                mesh = read.m_Mesh.read()
            except Exception as exc:
                print(f"  mesh read failed under {path}: {exc!r}")
                continue
            writer.add_mesh(path, mesh, world_mat, world_trans)

    for child in transform.m_Children:
        try:
            _walk(writer, child.read(), world_mat, world_trans, path + "/")
        except Exception as exc:
            print(f"  child failed under {path}: {exc!r}")


def export_db21(unity_py: Any, data_dir: Path, dest: Path) -> bool:
    """Extract the DB21 hierarchy from the engine's Unity assets and write it as one OBJ.

    Args:
        unity_py: The imported ``UnityPy`` module.
        data_dir: The engine's ``Duckiematrix_Data`` directory.
        dest: Destination ``main.obj``.

    Returns:
        True when the OBJ was written and matches the expected model.
    """
    print(f"loading Unity assets from {data_dir}")
    env = unity_py.load(str(data_dir))
    root = None
    for obj in env.objects:
        if obj.type.name != "GameObject":
            continue
        try:
            game_object = obj.read()
        except Exception:
            continue
        if str(game_object.m_Name) == DB21_GAMEOBJECT:
            root = game_object
            break
    if root is None:
        print(f"  no GameObject named {DB21_GAMEOBJECT!r} in this build")
        return False

    transform = _transform_of(root)
    if transform is None:
        print(f"  {DB21_GAMEOBJECT} has no Transform component")
        return False

    writer = _ObjWriter()
    identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    _walk(writer, transform, identity, (0.0, 0.0, 0.0), "")
    if not writer.vertices:
        print("  the hierarchy carried no meshes")
        return False
    writer.write(dest, f"{DB21_GAMEOBJECT} extracted from Duckiematrix 0.11.4 (Unity) via UnityPy")
    print(f"  wrote {dest} ({dest.stat().st_size:,} bytes), {len(writer.parts)} parts")
    drop_derived_usd(dest)
    return verify_obj(dest)


def drop_derived_usd(obj: Path) -> None:
    """Delete the USD files a previous OBJ was converted into.

    ``duckiebot_rl.envs.viz_env`` converts the OBJ to USD once per machine and caches the result
    next to it, then wraps that in a coloured layer, and both caches are keyed on nothing but the
    file name. Writing a new OBJ over the old one without clearing them would leave the viewer and
    the trainer drawing the previous model for ever, which is the kind of bug that is only found
    by noticing that a supposedly fixed mesh looks exactly as wrong as before.

    Args:
        obj: The OBJ that was just written.
    """
    for stale in (obj.with_suffix(".usd"), obj.with_name("main_colored.usd")):
        if stale.is_file():
            stale.unlink()
            print(f"  removed the stale conversion {stale.name}; it will be rebuilt on next launch")


# ------------------------------------------------------------------------------ verification


def measure_obj(path: Path) -> tuple[int, int, tuple[float, float, float]]:
    """Count an OBJ's vertices and faces and measure its bounding box.

    Args:
        path: The ``.obj`` file.

    Returns:
        ``(vertices, faces, (size_x, size_y, size_z))``. The size is ``(0, 0, 0)`` for a file with
        no vertices.
    """
    low = [float("inf")] * 3
    high = [float("-inf")] * 3
    vertices = faces = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("v "):
                vertices += 1
                fields = line.split()
                for axis in range(3):
                    value = float(fields[axis + 1])
                    low[axis] = min(low[axis], value)
                    high[axis] = max(high[axis], value)
            elif line.startswith("f "):
                faces += 1
    if vertices == 0:
        return 0, faces, (0.0, 0.0, 0.0)
    size = tuple(high[axis] - low[axis] for axis in range(3))
    return vertices, faces, (size[0], size[1], size[2])


def verify_obj(path: Path) -> bool:
    """Check that an extracted OBJ is the model this project measured its transforms against.

    ``duckiebot_rl.envs.viz_env`` orients the DB21 with a hard-coded rotation and rejects anything
    whose oriented box misses the expected one. That check runs inside Kit, minutes into a boot;
    this one runs here, in milliseconds, and says the same thing.

    Args:
        path: The written ``.obj``.

    Returns:
        True when the counts and the bounding box match.
    """
    vertices, faces, size = measure_obj(path)
    dims = " x ".join(f"{value:.6f}" for value in size)
    print(f"  {vertices:,} vertices, {faces:,} faces, bbox {dims} m")
    off = max(abs(size[axis] - EXPECTED_SIZE_M[axis]) for axis in range(3))
    if vertices == EXPECTED_VERTICES and faces == EXPECTED_FACES and off <= _SIZE_TOL_M:
        return True
    expected = " x ".join(f"{value:.6f}" for value in EXPECTED_SIZE_M)
    print(
        f"  WARNING: expected {EXPECTED_VERTICES:,} vertices, {EXPECTED_FACES:,} faces and "
        f"bbox {expected} m. The upstream build has changed: check the orientation constants in "
        f"duckiebot_rl/envs/viz_env.py (DB21_OBJ_ROTATE_XYZ, DB21_OBJ_EXPECTED_SIZE_M) before "
        f"trusting what it draws."
    )
    return False


# -------------------------------------------------------------------------------- the paths


def fetch_duckiematrix(dest: Path, force: bool, install: bool = True, keep_engine: bool = False) -> bool:
    """Put the real DB21 OBJ in place, downloading and extracting only if it is not there.

    Args:
        dest: Destination ``main.obj``.
        force: Redownload, re-extract and re-export even when the OBJ is already there.
        install: Allow the on-demand UnityPy install.
        keep_engine: Leave the extracted engine assets on disk afterwards.

    Returns:
        True when ``dest`` holds the DB21 geometry.
    """
    if dest.is_file() and dest.stat().st_size >= MIN_REAL_BYTES and not force:
        print(f"already present: {dest} ({dest.stat().st_size:,} bytes)")
        return verify_obj(dest)

    archive = None if force else find_cached_zip()
    if archive is None:
        archive = DUCKIEMATRIX_CACHE / DUCKIEMATRIX_URL.rsplit("/", 1)[-1]
        if not download_stream(DUCKIEMATRIX_URL, archive, force):
            return False
    else:
        print(f"using the engine archive already on disk: {archive}")

    unity_py = ensure_unitypy(install=install)
    if unity_py is None:
        return False

    engine_dir = DUCKIEMATRIX_CACHE / "engine"
    data_dir = extract_engine(archive, engine_dir, force)
    if data_dir is None:
        return False
    try:
        return export_db21(unity_py, data_dir, dest)
    finally:
        # Only ever remove what this script extracted, and only from its own cache directory.
        if not keep_engine and engine_dir.is_dir():
            shutil.rmtree(engine_dir, ignore_errors=True)
            print(f"removed the extracted engine assets ({engine_dir}); the archive is kept")


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument list, or None for ``sys.argv``.

    Returns:
        Process exit code: 0 when at least one DB21-generation model is in place.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="redownload and re-extract everything")
    parser.add_argument(
        "--skip-duckiematrix",
        action="store_true",
        help="do not touch the 123 MB engine build; fetch the DB18-era glTF only",
    )
    parser.add_argument(
        "--no-install",
        dest="install",
        action="store_false",
        help="never pip-install UnityPy; fail the Duckiematrix path instead",
    )
    parser.add_argument(
        "--keep-engine",
        action="store_true",
        help="keep the extracted engine assets under _refs/duckiematrix/engine",
    )
    args = parser.parse_args(argv)

    real_db21 = False
    if not args.skip_duckiematrix:
        print("== DB21 (the real one), from Duckietown's Duckiematrix engine ==")
        real_db21 = fetch_duckiematrix(
            DEST / "db21j" / "main.obj",
            force=args.force,
            install=args.install,
            keep_engine=args.keep_engine,
        )

    print("== DB18-era glTF, the fallback ==")
    gltf = fetch(DB18_GLTF_URL, DEST / "db21" / "main.gltf", args.force)

    if real_db21:
        print("done. The viewer and scripts/train.py draw the real DB21 on the next launch.")
    elif gltf:
        print(
            "the real DB21 is NOT in place; the DB18-era glTF is. The viewer will draw that and "
            "say so. Re-run without --skip-duckiematrix, or with --force, to try again."
        )
    else:
        print("nothing fetched; the viewer falls back to per-part or primitive visuals.")
    return 0 if (real_db21 or gltf) else 1


if __name__ == "__main__":
    sys.exit(main())
