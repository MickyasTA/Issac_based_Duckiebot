"""Author the city as **text** ``.usda`` using standalone ``usd-core`` -- no Kit boot required.

Why text USD only: the clean-room gate of SPEC v2 S3.4 bans every binary or opaque geometry
format (``.usd``, ``.usdc``, ``.usdz``, ``.obj``, ``.glb``, ...) from the repository, so the only
USD this package ever writes is a readable, diffable ``.usda``. Generated stages are gitignored
and rebuilt from the YAML maps by ``scripts/build_city.py``.

Vertex budget: rule 3 of the clean-room gate allows a city ``.usda`` at most
``4 * tile_count + 64`` mesh points. This builder emits exactly one quad (4 points) per grid
tile in a single merged ``UsdGeom.Mesh`` with one ``UsdGeom.Subset`` per tile kind. Walls, sign
posts and off-road distractors are ``Cube``, ``Cylinder`` and ``Sphere`` prims, which contribute
no mesh points at all; the only other meshes are the sign cards, which need UV coordinates and
so cost 4 points each against the 64-point slack (hence the cap of 16 cards). See
:func:`city_vertex_count`.

The gate cannot know a stage's tile count without being told, so :func:`build_city_usda` writes
it into the layer metadata as ``customLayerData = { int tiles = N }``. That single line is the
contract between this module and ``scripts/check_clean_room.py``; it is exercised end to end by
``tests/unit/test_usd_builder.py::test_generated_city_tree_passes_the_clean_room_gate``, which
builds a real tree and runs the real gate over it. No prim customData key may end in ``tile``:
the gate parses the header for an integer tile count, and a float-valued key with such a name
used to be read as "zero tiles", which collapsed the budget and failed every city file.

Materials: each subset binds a material that carries **both** a ``UsdPreviewSurface`` network
(so the stage renders correctly in ``usdview``, in CI and in any non-Omniverse consumer) and an
``OmniPBR`` MDL shader on the ``mdl`` output (so Isaac Sim's RTX renderer picks OmniPBR, whose
``diffuse_tint`` / ``albedo_brightness`` / ``reflection_roughness_constant`` inputs are the
scalar handles the per-episode visual DR of SPEC v2 S7.1 layer 2b writes at runtime). Texture
*assignments* are authored once, here; nothing at runtime ever swaps a texture.

Physics: the city stage contains **no colliders at all** (SPEC v2 S3.3 / risk item K). Walls are
visual only. The single physics surface is the separately authored ground plane produced by
:func:`build_ground_usda`.

USD runtime discovery
---------------------
:func:`ensure_usd` looks for ``pxr`` in this order:

1. an ordinary import, which succeeds when ``usd-core`` is pip-installed (the CI and tools-venv
   path, and the one SPEC v2 M0 provisions);
2. the USD libraries shipped inside an Isaac Sim install, found via ``isaacsim/extscache/
   omni.usd.libs-*``. Isaac's own venv has the libraries but not the pip package, and this
   fallback lets the city build run there without a Kit boot.

If neither works it raises :class:`UsdUnavailableError` with the exact command to run.
"""

from __future__ import annotations

import math
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Final

import numpy as np

from .maps import CityMap
from .spec import IDEAL_PALETTE, NOMINAL_TILE_SPEC, ColorPalette, TileSpec
from .tiles import EDGE_BASIS, SIGN_KINDS, TILE_KINDS, rotated_uv_corners

__all__ = [
    "DEFAULT_ROAD_Z",
    "UsdModules",
    "UsdUnavailableError",
    "build_city_usda",
    "build_ground_usda",
    "city_vertex_count",
    "ensure_usd",
]

#: Height of the road quads above the physics ground plane, in metres. Two millimetres is far
#: below the 5 mm lane-graph overlay tolerance of milestone M2 and removes all z-fighting.
DEFAULT_ROAD_Z: Final[float] = 0.002

#: Wall geometry (SPEC v2 S3.3): 0.30 m tall, 0.02 m thick, visual only.
WALL_HEIGHT_M: Final[float] = 0.30
WALL_THICKNESS_M: Final[float] = 0.02

#: Traffic-sign geometry (SPEC v2 S3.3): an 85 x 155 mm card on an 11 mm post, centre 130 mm up.
SIGN_CARD_W_M: Final[float] = 0.085
SIGN_CARD_H_M: Final[float] = 0.155
SIGN_POST_D_M: Final[float] = 0.011
SIGN_CENTER_Z_M: Final[float] = 0.130

_ISAAC_USD_GLOB: Final[str] = "isaacsim/extscache/omni.usd.libs-*"


class UsdUnavailableError(ImportError):
    """Raised when no USD runtime can be found, with actionable instructions."""


@dataclass(frozen=True)
class UsdModules:
    """The handful of ``pxr`` modules this builder uses.

    Attributes:
        Usd: ``pxr.Usd``.
        UsdGeom: ``pxr.UsdGeom``.
        UsdShade: ``pxr.UsdShade``.
        UsdPhysics: ``pxr.UsdPhysics``.
        Sdf: ``pxr.Sdf``.
        Gf: ``pxr.Gf``.
        Vt: ``pxr.Vt``.
        version: The USD version tuple, for diagnostics.
        source: ``"usd-core"`` or ``"isaac-sim"``.
    """

    Usd: ModuleType
    UsdGeom: ModuleType
    UsdShade: ModuleType
    UsdPhysics: ModuleType
    Sdf: ModuleType
    Gf: ModuleType
    Vt: ModuleType
    version: tuple[int, ...]
    source: str


_USD_CACHE: UsdModules | None = None


def _import_pxr(source: str) -> UsdModules:
    """Import the pxr modules and package them.

    Args:
        source: Label recorded on the result, for diagnostics.

    Returns:
        The imported modules.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

    return UsdModules(
        Usd=Usd,
        UsdGeom=UsdGeom,
        UsdShade=UsdShade,
        UsdPhysics=UsdPhysics,
        Sdf=Sdf,
        Gf=Gf,
        Vt=Vt,
        version=tuple(Usd.GetVersion()),
        source=source,
    )


def _isaac_usd_dir() -> Path | None:
    """Locate the USD libraries bundled with an Isaac Sim install, if one is importable.

    Returns:
        The ``omni.usd.libs-*`` directory, or ``None``.
    """
    roots: list[Path] = []
    env_root = os.environ.get("DUCKIEBOT_ISAAC_USD_DIR")
    if env_root:
        roots.append(Path(env_root))
    for entry in sys.path:
        if not entry:
            continue
        roots.extend(sorted(Path(entry).glob(_ISAAC_USD_GLOB)))
    for root in roots:
        if (root / "bin").is_dir():
            return root
    return None


def ensure_usd() -> UsdModules:
    """Return the ``pxr`` modules, importing a USD runtime if necessary.

    The result is cached, so repeated calls are free.

    Returns:
        The :class:`UsdModules` bundle.

    Raises:
        UsdUnavailableError: If neither ``usd-core`` nor an Isaac Sim USD runtime is available.
            The message names the exact ``pip install`` command and the environment variable that
            can point at an Isaac install.
    """
    global _USD_CACHE
    if _USD_CACHE is not None:
        return _USD_CACHE
    try:
        _USD_CACHE = _import_pxr("usd-core")
        return _USD_CACHE
    except ImportError:
        pass
    isaac_dir = _isaac_usd_dir()
    if isaac_dir is not None:
        try:
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(str(isaac_dir / "bin"))
            sys.path.insert(0, str(isaac_dir))
            _USD_CACHE = _import_pxr("isaac-sim")
            return _USD_CACHE
        except (ImportError, OSError):
            pass
    raise UsdUnavailableError(
        "No USD runtime found, so the city .usda cannot be authored.\n"
        "Fix it in one of two ways:\n"
        "  1. pip install usd-core          "
        "(the supported path; SPEC v2 M0 installs it into the tools venv)\n"
        "  2. set DUCKIEBOT_ISAAC_USD_DIR to an Isaac Sim "
        "'isaacsim/extscache/omni.usd.libs-*' directory that contains a 'bin' folder.\n"
        "Texture generation (duckiebot_rl.city.tiles) and map generation "
        "(duckiebot_rl.city.maps) do not need USD and keep working without it."
    )


def city_vertex_count(city: CityMap, n_signs: int = 0) -> int:
    """Total mesh points the generated ``.usda`` of ``city`` will contain.

    The clean-room gate of SPEC v2 S3.4 allows ``4 * tile_count + 64``; this builder emits
    ``4 * tile_count`` for the road plus 4 per sign card and nothing else.

    Args:
        city: The map.
        n_signs: Number of sign cards that will be emitted.

    Returns:
        ``4 * n_rows * n_cols + 4 * n_signs``.
    """
    return 4 * city.n_rows * city.n_cols + 4 * int(n_signs)


# ------------------------------------------------------------------------------- primitives
def _define_xform(
    usd: UsdModules,
    stage: Any,
    path: str,
    translate: Sequence[float] | None = None,
    rotate_z_deg: float | None = None,
    scale: Sequence[float] | None = None,
) -> Any:
    """Define an Xform with an optional TRS, in that order."""
    xform = usd.UsdGeom.Xform.Define(stage, path)
    if translate is not None:
        xform.AddTranslateOp().Set(usd.Gf.Vec3d(*[float(v) for v in translate]))
    if rotate_z_deg is not None:
        xform.AddRotateZOp().Set(float(rotate_z_deg))
    if scale is not None:
        xform.AddScaleOp().Set(usd.Gf.Vec3f(*[float(v) for v in scale]))
    return xform


def _define_box(
    usd: UsdModules,
    stage: Any,
    path: str,
    center: Sequence[float],
    size: Sequence[float],
    rotate_z_deg: float = 0.0,
) -> Any:
    """Define a unit ``Cube`` scaled to ``size`` and placed at ``center``.

    Using ``Cube`` rather than a ``Mesh`` keeps the file's mesh-point count at zero, which is
    what the clean-room vertex budget cares about.
    """
    _define_xform(usd, stage, path, translate=center, rotate_z_deg=rotate_z_deg, scale=size)
    cube = usd.UsdGeom.Cube.Define(stage, f"{path}/Geom")
    cube.CreateSizeAttr(1.0)
    cube.CreateExtentAttr([usd.Gf.Vec3f(-0.5, -0.5, -0.5), usd.Gf.Vec3f(0.5, 0.5, 0.5)])
    return cube


def _define_cylinder(
    usd: UsdModules, stage: Any, path: str, center: Sequence[float], radius: float, height: float
) -> Any:
    """Define a z-aligned ``Cylinder``."""
    _define_xform(usd, stage, path, translate=center)
    cyl = usd.UsdGeom.Cylinder.Define(stage, f"{path}/Geom")
    cyl.CreateAxisAttr("Z")
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(height))
    cyl.CreateExtentAttr(
        [usd.Gf.Vec3f(-radius, -radius, -height / 2), usd.Gf.Vec3f(radius, radius, height / 2)]
    )
    return cyl


def _define_sphere(usd: UsdModules, stage: Any, path: str, center: Sequence[float], radius: float) -> Any:
    """Define a ``Sphere``."""
    _define_xform(usd, stage, path, translate=center)
    sph = usd.UsdGeom.Sphere.Define(stage, f"{path}/Geom")
    sph.CreateRadiusAttr(float(radius))
    sph.CreateExtentAttr([usd.Gf.Vec3f(-radius, -radius, -radius), usd.Gf.Vec3f(radius, radius, radius)])
    return sph


def _define_quad(usd: UsdModules, stage: Any, path: str, width: float, height: float) -> Any:
    """Define a textured quad in the local YZ plane, facing local ``+x``.

    Sign cards need UVs, and ``UsdGeom.Cube`` carries none, so cards are the one place this
    builder emits mesh points besides the road. Four points per card against the 64-point slack
    in the clean-room budget caps the card count at 16; :func:`build_city_usda` enforces that.

    Args:
        usd: The pxr module bundle.
        stage: Stage being authored.
        path: Prim path of the mesh.
        width: Card width along local ``y``, in metres.
        height: Card height along local ``z``, in metres.

    Returns:
        The ``UsdGeom.Mesh``.
    """
    hw, hh = width / 2.0, height / 2.0
    mesh = usd.UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(
        [
            usd.Gf.Vec3f(0.0, -hw, -hh),
            usd.Gf.Vec3f(0.0, hw, -hh),
            usd.Gf.Vec3f(0.0, hw, hh),
            usd.Gf.Vec3f(0.0, -hw, hh),
        ]
    )
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateNormalsAttr([usd.Gf.Vec3f(1.0, 0.0, 0.0)] * 4)
    mesh.SetNormalsInterpolation(usd.UsdGeom.Tokens.faceVarying)
    mesh.CreateSubdivisionSchemeAttr(usd.UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateExtentAttr([usd.Gf.Vec3f(0.0, -hw, -hh), usd.Gf.Vec3f(0.0, hw, hh)])
    usd.UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", usd.Sdf.ValueTypeNames.TexCoord2fArray, usd.UsdGeom.Tokens.faceVarying
    ).Set([usd.Gf.Vec2f(1, 0), usd.Gf.Vec2f(0, 0), usd.Gf.Vec2f(0, 1), usd.Gf.Vec2f(1, 1)])
    return mesh


def _define_material(
    usd: UsdModules,
    stage: Any,
    path: str,
    diffuse_color: tuple[float, float, float],
    texture: str | None = None,
    roughness: float = 0.8,
    tint: tuple[float, float, float] = (1.0, 1.0, 1.0),
    albedo_brightness: float = 1.0,
) -> Any:
    """Define a material with a UsdPreviewSurface network and an OmniPBR mdl output.

    Args:
        usd: The pxr module bundle.
        stage: The stage being authored.
        path: Prim path of the material.
        diffuse_color: Base colour used when there is no texture, and as the preview tint.
        texture: Relative asset path of a diffuse texture, or ``None``.
        roughness: Roughness constant, exposed to DR as ``reflection_roughness_constant``.
        tint: OmniPBR ``diffuse_tint``; the per-variant colour handle.
        albedo_brightness: OmniPBR ``albedo_brightness``; the per-variant luminance handle.

    Returns:
        The ``UsdShade.Material``.
    """
    material = usd.UsdShade.Material.Define(stage, path)

    preview = usd.UsdShade.Shader.Define(stage, f"{path}/Preview")
    preview.CreateIdAttr("UsdPreviewSurface")
    preview.CreateInput("roughness", usd.Sdf.ValueTypeNames.Float).Set(float(roughness))
    preview.CreateInput("metallic", usd.Sdf.ValueTypeNames.Float).Set(0.0)
    preview.CreateInput("specular", usd.Sdf.ValueTypeNames.Float).Set(0.2)
    if texture is None:
        preview.CreateInput("diffuseColor", usd.Sdf.ValueTypeNames.Color3f).Set(usd.Gf.Vec3f(*diffuse_color))
    else:
        reader = usd.UsdShade.Shader.Define(stage, f"{path}/StReader")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", usd.Sdf.ValueTypeNames.Token).Set("st")
        reader.CreateOutput("result", usd.Sdf.ValueTypeNames.Float2)

        sampler = usd.UsdShade.Shader.Define(stage, f"{path}/DiffuseTex")
        sampler.CreateIdAttr("UsdUVTexture")
        sampler.CreateInput("file", usd.Sdf.ValueTypeNames.Asset).Set(usd.Sdf.AssetPath(texture))
        sampler.CreateInput("wrapS", usd.Sdf.ValueTypeNames.Token).Set("clamp")
        sampler.CreateInput("wrapT", usd.Sdf.ValueTypeNames.Token).Set("clamp")
        sampler.CreateInput("sourceColorSpace", usd.Sdf.ValueTypeNames.Token).Set("sRGB")
        sampler.CreateInput("scale", usd.Sdf.ValueTypeNames.Float4).Set(
            usd.Gf.Vec4f(tint[0], tint[1], tint[2], 1.0)
        )
        sampler.CreateInput("st", usd.Sdf.ValueTypeNames.Float2).ConnectToSource(
            reader.ConnectableAPI(), "result"
        )
        sampler.CreateOutput("rgb", usd.Sdf.ValueTypeNames.Float3)
        preview.CreateInput("diffuseColor", usd.Sdf.ValueTypeNames.Color3f).ConnectToSource(
            sampler.ConnectableAPI(), "rgb"
        )
    preview.CreateOutput("surface", usd.Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(preview.ConnectableAPI(), "surface")

    omni = usd.UsdShade.Shader.Define(stage, f"{path}/OmniPBR")
    omni.SetSourceAsset(usd.Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    omni.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    omni.CreateIdAttr("OmniPBR")
    if texture is not None:
        omni.CreateInput("diffuse_texture", usd.Sdf.ValueTypeNames.Asset).Set(usd.Sdf.AssetPath(texture))
        omni.CreateInput("project_uvw", usd.Sdf.ValueTypeNames.Bool).Set(False)
        omni.CreateInput("texture_rotate", usd.Sdf.ValueTypeNames.Float).Set(0.0)
    omni.CreateInput("diffuse_color_constant", usd.Sdf.ValueTypeNames.Color3f).Set(
        usd.Gf.Vec3f(*diffuse_color)
    )
    # The three scalars below are the ONLY things the runtime DR event terms write.
    omni.CreateInput("diffuse_tint", usd.Sdf.ValueTypeNames.Color3f).Set(usd.Gf.Vec3f(*tint))
    omni.CreateInput("albedo_brightness", usd.Sdf.ValueTypeNames.Float).Set(float(albedo_brightness))
    omni.CreateInput("reflection_roughness_constant", usd.Sdf.ValueTypeNames.Float).Set(float(roughness))
    omni.CreateInput("metallic_constant", usd.Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput("mdl").ConnectToSource(omni.ConnectableAPI(), "out")
    material.CreateDisplacementOutput("mdl").ConnectToSource(omni.ConnectableAPI(), "out")
    material.CreateVolumeOutput("mdl").ConnectToSource(omni.ConnectableAPI(), "out")
    return material


# ------------------------------------------------------------------------------ city stage
def _road_mesh(
    usd: UsdModules, stage: Any, city: CityMap, root: str, road_z: float
) -> tuple[Any, dict[str, list[int]]]:
    """Author the merged road mesh: one quad per tile, four points per quad.

    Args:
        usd: The pxr module bundle.
        stage: Stage being authored.
        city: The map.
        root: Prim path of the city root.
        road_z: Height of the quads above ``z = 0``.

    Returns:
        ``(mesh, faces_by_kind)`` where ``faces_by_kind`` maps a tile kind to its face indices.
    """
    n_tiles = city.n_rows * city.n_cols
    points = np.empty((n_tiles * 4, 3), dtype=np.float32)
    uvs = np.empty((n_tiles * 4, 2), dtype=np.float32)
    faces_by_kind: dict[str, list[int]] = {}
    pitch = city.tile_size
    ox, oy = city.origin_xy

    for face, (row, col, tile) in enumerate(city.iter_tiles()):
        x0 = ox + col * pitch
        y0 = oy + (city.n_rows - 1 - row) * pitch
        corners = ((x0, y0), (x0 + pitch, y0), (x0 + pitch, y0 + pitch), (x0, y0 + pitch))
        for k, (x, y) in enumerate(corners):
            points[face * 4 + k] = (x, y, road_z)
        for k, uv in enumerate(rotated_uv_corners(tile.rot)):
            uvs[face * 4 + k] = uv
        faces_by_kind.setdefault(tile.kind, []).append(face)

    mesh = usd.UsdGeom.Mesh.Define(stage, f"{root}/RoadMesh")
    mesh.CreatePointsAttr(usd.Vt.Vec3fArray.FromNumpy(points))
    mesh.CreateFaceVertexIndicesAttr(usd.Vt.IntArray.FromNumpy(np.arange(n_tiles * 4, dtype=np.int32)))
    mesh.CreateFaceVertexCountsAttr(usd.Vt.IntArray.FromNumpy(np.full(n_tiles, 4, dtype=np.int32)))
    mesh.CreateSubdivisionSchemeAttr(usd.UsdGeom.Tokens.none)
    normals = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (n_tiles * 4, 1))
    mesh.CreateNormalsAttr(usd.Vt.Vec3fArray.FromNumpy(normals))
    mesh.SetNormalsInterpolation(usd.UsdGeom.Tokens.faceVarying)
    mesh.CreateExtentAttr(
        [
            usd.Gf.Vec3f(float(points[:, 0].min()), float(points[:, 1].min()), float(road_z)),
            usd.Gf.Vec3f(float(points[:, 0].max()), float(points[:, 1].max()), float(road_z)),
        ]
    )
    usd.UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", usd.Sdf.ValueTypeNames.TexCoord2fArray, usd.UsdGeom.Tokens.faceVarying
    ).Set(usd.Vt.Vec2fArray.FromNumpy(uvs))
    return mesh, faces_by_kind


def _wall_boxes(city: CityMap) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
    """Four perimeter walls around the whole grid, as ``(center, size)`` pairs."""
    ox, oy = city.origin_xy
    width = city.n_cols * city.tile_size
    depth = city.n_rows * city.tile_size
    t, h = WALL_THICKNESS_M, WALL_HEIGHT_M
    cx, cy = ox + width / 2.0, oy + depth / 2.0
    return [
        ((cx, oy + depth + t / 2.0, h / 2.0), (width + 2 * t, t, h)),
        ((cx, oy - t / 2.0, h / 2.0), (width + 2 * t, t, h)),
        ((ox - t / 2.0, cy, h / 2.0), (t, depth, h)),
        ((ox + width + t / 2.0, cy, h / 2.0), (t, depth, h)),
    ]


def _sign_slots(city: CityMap, max_signs: int) -> list[tuple[float, float, float]]:
    """Roadside sign positions: outside the outer corner of every curve and intersection tile.

    Args:
        city: The map.
        max_signs: Cap on the number of slots returned.

    Returns:
        ``(x, y, yaw_deg)`` slots, deterministic in row-major tile order.
    """
    out: list[tuple[float, float, float]] = []
    inset = 0.42 * city.tile_size
    for row, col, tile in city.iter_tiles():
        if tile.kind not in ("curve", "threeway", "fourway"):
            continue
        cx, cy = city.tile_center_xy(row, col)
        closed = [e for e in ("N", "E", "S", "W") if e not in tile.open_edges]
        for edge in closed:
            (nx, ny), _ = EDGE_BASIS[edge]
            yaw = math.degrees(math.atan2(-ny, -nx))
            out.append((cx + nx * inset, cy + ny * inset, yaw))
            if len(out) >= max_signs:
                return out
    return out


def build_city_usda(
    city: CityMap,
    out_path: str | Path,
    texture_dir: str = "../textures/bucket_00",
    spec: TileSpec = NOMINAL_TILE_SPEC,
    palette: ColorPalette = IDEAL_PALETTE,
    road_z: float = DEFAULT_ROAD_Z,
    walls: bool | None = None,
    max_signs: int = 8,
    sign_texture_dir: str | None = None,
    n_distractors: int = 4,
    seed: int | None = None,
    root_prim: str = "/City",
    roughness: float = 0.8,
    tint: tuple[float, float, float] = (1.0, 1.0, 1.0),
    albedo_brightness: float = 1.0,
) -> Path:
    """Author one city variant as a text ``.usda`` file.

    Args:
        city: A validated map. Its ``meta`` supplies ``walls``, ``geometry_bucket`` and
            ``palette_index`` when those arguments are left at their defaults.
        out_path: Destination ``.usda`` path. Parent directories are created.
        texture_dir: Asset path of the tile textures, **relative to** ``out_path`` (forward
            slashes). One PNG per tile kind is expected inside it.
        spec: Marking geometry the textures were painted with. It is rescaled to the map's own
            tile pitch before being recorded as stage metadata, so a consumer reading
            ``duckiebot:lane_width_m`` gets the lane width the stage actually shows (``w_ep``)
            rather than the bucket's authoring width. See :meth:`~.spec.TileSpec.rescaled`.
        palette: Colours the textures were painted with; used for the untextured fallbacks
            (walls, distractors) and for the preview-surface base colours.
        road_z: Height of the road quads above the ground plane.
        walls: Emit perimeter walls. ``None`` reads ``city.meta["walls"]``, defaulting to ``True``.
        max_signs: Cap on roadside sign distractors; ``0`` disables them. Capped at 16 because
            each textured card costs 4 mesh points against the 64-point clean-room slack.
        sign_texture_dir: Asset path of the procedural sign faces relative to ``out_path``.
            ``None`` uses flat-coloured cards and emits no card textures.
        n_distractors: Number of off-road primitive distractors (SPEC v2 S7.2 axis V13).
        seed: Seed for the distractor placement; ``None`` uses ``city.meta["variant_index"]``
            or ``0``, so the same map always yields the same stage.
        root_prim: Path of the stage's default prim.
        roughness: Base roughness of the road materials, and the authored value of the OmniPBR
            ``reflection_roughness_constant`` handle.
        tint: Authored value of the OmniPBR ``diffuse_tint`` handle on the road materials. This
            plus ``albedo_brightness`` and ``roughness`` are the three scalars the runtime
            visual DR writes; see :func:`~.spec.variant_material_scalars`.
        albedo_brightness: Authored value of the OmniPBR ``albedo_brightness`` handle.

    Returns:
        The written path.

    Raises:
        UsdUnavailableError: If no USD runtime is available.
        ValueError: If ``out_path`` does not end in ``.usda``.
    """
    usd = ensure_usd()
    out = Path(out_path)
    if out.suffix != ".usda":
        raise ValueError(f"the clean-room gate allows text USD only; expected a .usda path, got {out.name!r}")
    if max_signs > 16:
        raise ValueError(
            f"max_signs must be <= 16 so the sign cards fit the clean-room 64-point slack, got {max_signs}"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    city.validate()

    if walls is None:
        walls = bool(city.meta.get("walls", True))
    if seed is None:
        seed = int(city.meta.get("variant_index", 0))
    rng = np.random.default_rng(seed)
    tex = texture_dir.rstrip("/")

    n_tiles = int(city.n_rows * city.n_cols)
    # The marking geometry was painted at the bucket's own pitch and is UV-mapped onto quads of
    # the map's pitch, so every marking scales by map_pitch / bucket_pitch. Record the spec the
    # stage actually shows, not the one the texture was painted with: w_ep is the SPEC v2 S5.4
    # reward denominator and an 8% error there silently mis-scales the lateral penalty.
    shown = spec.rescaled(float(city.tile_size))

    stage = usd.Usd.Stage.CreateNew(str(out))
    usd.UsdGeom.SetStageUpAxis(stage, usd.UsdGeom.Tokens.z)
    usd.UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    # Layer metadata, not prim customData: rule 3 of scripts/check_clean_room.py reads the tile
    # count out of the layer header to size the mesh-point budget at 4 * tiles + 64. USD writes
    # a python int as "int tiles = N", which is exactly the typed form the gate anchors on.
    stage.GetRootLayer().customLayerData = {
        "tiles": n_tiles,
        "rows": int(city.n_rows),
        "cols": int(city.n_cols),
        "generator": "duckiebot_rl.city.usd_builder",
    }
    root = usd.UsdGeom.Xform.Define(stage, root_prim)
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().SetCustomDataByKey("duckiebot:map_name", city.name)
    root.GetPrim().SetCustomDataByKey("duckiebot:tile_size_m", float(city.tile_size))
    root.GetPrim().SetCustomDataByKey("duckiebot:lane_width_m", float(shown.lane_width_m))
    # Named "_frac", never "_tile": a key ending in "tile" followed by a float used to be parsed
    # by the clean-room gate as a tile count of zero, which flagged every generated city as a
    # licensing violation. The gate is anchored now, and so is the name.
    root.GetPrim().SetCustomDataByKey(
        "duckiebot:lane_center_offset_frac", float(shown.lane_center_offset_tile)
    )
    root.GetPrim().SetCustomDataByKey("duckiebot:generator", "duckiebot_rl.city.usd_builder")

    looks = f"{root_prim}/Looks"
    usd.UsdGeom.Scope.Define(stage, looks)

    mesh, faces_by_kind = _road_mesh(usd, stage, city, root_prim, road_z)
    usd.UsdGeom.Subset.SetFamilyType(mesh, "materialBind", usd.UsdGeom.Tokens.partition)
    base_colors = {
        "straight": palette.road,
        "curve": palette.road,
        "threeway": palette.road,
        "fourway": palette.road,
        "asphalt": palette.asphalt,
        "grass": palette.grass,
        "empty": palette.asphalt,
    }
    for kind in TILE_KINDS:
        faces = faces_by_kind.get(kind)
        if not faces:
            continue
        subset = usd.UsdGeom.Subset.CreateGeomSubset(
            mesh,
            f"subset_{kind}",
            usd.UsdGeom.Tokens.face,
            usd.Vt.IntArray.FromNumpy(np.asarray(faces, dtype=np.int32)),
            "materialBind",
            usd.UsdGeom.Tokens.partition,
        )
        material = _define_material(
            usd,
            stage,
            f"{looks}/mat_{kind}",
            base_colors[kind],
            texture=f"{tex}/{kind}.png",
            roughness=roughness,
            tint=tint,
            albedo_brightness=albedo_brightness,
        )
        usd.UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(material)

    if walls:
        wall_mat = _define_material(
            usd, stage, f"{looks}/mat_wall", (0.64, 0.71, 0.28), texture=None, roughness=0.9
        )
        usd.UsdGeom.Scope.Define(stage, f"{root_prim}/Walls")
        for i, (center, size) in enumerate(_wall_boxes(city)):
            path = f"{root_prim}/Walls/wall_{i:02d}"
            _define_box(usd, stage, path, center, size)
            usd.UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(path)).Bind(wall_mat)

    slots = _sign_slots(city, max_signs) if max_signs > 0 else []
    if slots:
        usd.UsdGeom.Scope.Define(stage, f"{root_prim}/Signs")
        post_mat = _define_material(usd, stage, f"{looks}/mat_post", (0.15, 0.15, 0.15), roughness=0.7)
        for i, (x, y, yaw) in enumerate(slots):
            sign_kind = SIGN_KINDS[i % len(SIGN_KINDS)]
            card_color = ((0.85, 0.1, 0.1), (0.1, 0.2, 0.8), (0.95, 0.9, 0.1))[i % 3]
            card_tex = f"{sign_texture_dir.rstrip('/')}/sign_{sign_kind}.png" if sign_texture_dir else None
            card_mat = _define_material(
                usd, stage, f"{looks}/mat_sign_{i:02d}", card_color, texture=card_tex, roughness=0.6
            )
            group = f"{root_prim}/Signs/sign_{i:02d}"
            _define_xform(usd, stage, group, translate=(x, y, 0.0), rotate_z_deg=yaw)
            post_h = SIGN_CENTER_Z_M - SIGN_CARD_H_M / 2.0
            _define_cylinder(
                usd, stage, f"{group}/Post", (0.0, 0.0, post_h / 2.0), SIGN_POST_D_M / 2.0, post_h
            )
            usd.UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(f"{group}/Post")).Bind(post_mat)
            _define_xform(usd, stage, f"{group}/Card", translate=(0.0, 0.0, SIGN_CENTER_Z_M))
            _define_quad(usd, stage, f"{group}/Card/Geom", SIGN_CARD_W_M, SIGN_CARD_H_M)
            card_prim = stage.GetPrimAtPath(f"{group}/Card/Geom")
            usd.UsdShade.MaterialBindingAPI.Apply(card_prim).Bind(card_mat)

    off_road = [(r, c) for r, c, t in city.iter_tiles() if not t.drivable]
    if n_distractors > 0 and off_road:
        usd.UsdGeom.Scope.Define(stage, f"{root_prim}/Distractors")
        for i in range(n_distractors):
            row, col = off_road[int(rng.integers(len(off_road)))]
            cx, cy = city.tile_center_xy(row, col)
            jitter = rng.uniform(-0.35, 0.35, size=2) * city.tile_size
            color = tuple(float(v) for v in rng.uniform(0.05, 0.95, size=3))
            path = f"{root_prim}/Distractors/prop_{i:02d}"
            mat = _define_material(usd, stage, f"{looks}/mat_prop_{i:02d}", color, roughness=0.75)
            shape = int(rng.integers(3))
            size = float(rng.uniform(0.05, 0.20))
            if shape == 0:
                _define_box(
                    usd,
                    stage,
                    path,
                    (cx + jitter[0], cy + jitter[1], size / 2.0),
                    (size, size, size),
                    rotate_z_deg=float(rng.uniform(0.0, 360.0)),
                )
            elif shape == 1:
                _define_sphere(usd, stage, path, (cx + jitter[0], cy + jitter[1], size / 2.0), size / 2.0)
            else:
                _define_cylinder(
                    usd, stage, path, (cx + jitter[0], cy + jitter[1], size / 2.0), size / 3.0, size
                )
            usd.UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(path)).Bind(mat)

    stage.GetRootLayer().Save()
    return out


def build_ground_usda(
    out_path: str | Path,
    half_extent_m: float = 50.0,
    albedo: tuple[float, float, float] = (0.35, 0.35, 0.36),
    static_friction: float = 1.0,
    dynamic_friction: float = 0.9,
    restitution: float = 0.0,
    root_prim: str = "/Ground",
) -> Path:
    """Author the single physics ground plane (SPEC v2 S5.1, replaces ``GroundPlaneCfg``).

    This is the only prim in the whole scene that carries a collider: the city stages are
    visual-only. The plane is a four-point quad, so it also sits inside the clean-room vertex
    budget with room to spare.

    Args:
        out_path: Destination ``.usda`` path.
        half_extent_m: Half side length of the square plane, in metres.
        albedo: Base colour; the runtime DR randomises it through the OmniPBR tint.
        static_friction: Physics material static friction.
        dynamic_friction: Physics material dynamic friction; clamped to ``<= static_friction``.
        restitution: Physics material restitution.
        root_prim: Path of the stage's default prim.

    Returns:
        The written path.

    Raises:
        UsdUnavailableError: If no USD runtime is available.
        ValueError: If ``out_path`` does not end in ``.usda``.
    """
    usd = ensure_usd()
    out = Path(out_path)
    if out.suffix != ".usda":
        raise ValueError(f"expected a .usda path, got {out.name!r}")
    out.parent.mkdir(parents=True, exist_ok=True)
    dynamic_friction = min(dynamic_friction, static_friction)

    stage = usd.Usd.Stage.CreateNew(str(out))
    usd.UsdGeom.SetStageUpAxis(stage, usd.UsdGeom.Tokens.z)
    usd.UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = usd.UsdGeom.Xform.Define(stage, root_prim)
    stage.SetDefaultPrim(root.GetPrim())

    h = float(half_extent_m)
    mesh = usd.UsdGeom.Mesh.Define(stage, f"{root_prim}/Plane")
    mesh.CreatePointsAttr(
        [
            usd.Gf.Vec3f(-h, -h, 0.0),
            usd.Gf.Vec3f(h, -h, 0.0),
            usd.Gf.Vec3f(h, h, 0.0),
            usd.Gf.Vec3f(-h, h, 0.0),
        ]
    )
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateNormalsAttr([usd.Gf.Vec3f(0.0, 0.0, 1.0)] * 4)
    mesh.SetNormalsInterpolation(usd.UsdGeom.Tokens.faceVarying)
    mesh.CreateSubdivisionSchemeAttr(usd.UsdGeom.Tokens.none)
    mesh.CreateExtentAttr([usd.Gf.Vec3f(-h, -h, 0.0), usd.Gf.Vec3f(h, h, 0.0)])
    usd.UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", usd.Sdf.ValueTypeNames.TexCoord2fArray, usd.UsdGeom.Tokens.faceVarying
    ).Set([usd.Gf.Vec2f(0, 0), usd.Gf.Vec2f(1, 0), usd.Gf.Vec2f(1, 1), usd.Gf.Vec2f(0, 1)])

    usd.UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    mesh_collision = usd.UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
    mesh_collision.CreateApproximationAttr("none")

    usd.UsdGeom.Scope.Define(stage, f"{root_prim}/Looks")
    visual = _define_material(usd, stage, f"{root_prim}/Looks/mat_ground", albedo, roughness=0.9)
    usd.UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(visual)

    physics_mat = usd.UsdShade.Material.Define(stage, f"{root_prim}/Looks/physics_ground")
    api = usd.UsdPhysics.MaterialAPI.Apply(physics_mat.GetPrim())
    api.CreateStaticFrictionAttr(float(static_friction))
    api.CreateDynamicFrictionAttr(float(dynamic_friction))
    api.CreateRestitutionAttr(float(restitution))
    usd.UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
        physics_mat, usd.UsdShade.Tokens.weakerThanDescendants, "physics"
    )

    stage.GetRootLayer().Save()
    return out
