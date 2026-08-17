"""Millimetre-accurate dimensional specification for the clean-room Duckietown city.

Everything in this module is a *dimensional fact* taken from published Duckietown appearance
specifications and from SPEC v2 section S3.3. Dimensions are facts, not copyrightable
expression: no Duckietown image, mesh or texture asset is used anywhere in this package.

Provenance of every number is given in the field docstrings below. The three consumers of this
module -- ``tiles`` (texture painting), ``lane_graph`` (reward ground truth) and ``usd_builder``
(stage authoring) -- all read the *same* :class:`TileSpec` instance so that the painted markings
and the analytic lane centrelines cannot drift apart (SPEC v2 milestone M2 requires agreement
within 5 mm).

Coordinate and sign conventions (SPEC v2 S2 / S5, restated here because they are load bearing)
-----------------------------------------------------------------------------------------------
* World frame is REP-103: ``x`` forward/east, ``y`` left/north, ``z`` up. Tiles are laid out in
  the ``z = 0`` plane.
* A tile with grid index ``(col, row)`` has its centre at ``((col + 0.5) * pitch,
  (row + 0.5) * pitch)``. Tile-local coordinates run over ``[-pitch/2, +pitch/2]`` in both axes.
* Tile rotation is an integer number of quarter turns **counter-clockwise about +z**.
* Traffic is right-hand: driving along a lane, the yellow centre tape is on your **left** and the
  white edge tape is on your **right**.
* Lateral offset ``d > 0`` means the robot is displaced to the **left** of its lane centreline,
  i.e. toward the yellow tape (SPEC v2 S2). ``psi > 0`` means the robot heading is rotated
  counter-clockwise (left) of the lane tangent.

The 0.22 vs 0.20 lane-centre fork
---------------------------------
``duckietown-world/tile_template.py`` places the lane centreline at ``0.22`` tile units from the
tile centreline (128.7 mm at pitch 0.585) while ``gym-duckietown``'s reward Bezier places it at
``0.20`` (117.0 mm). SPEC v2 S3.3 quotes 0.22 together with curve radii 0.28/0.72 tile.
Those two conventions are *not* simultaneously satisfiable with the tape dimensions that S3.3
also fixes (yellow 24 mm on the tile centreline, clear lane 210 mm, white 48 mm), which place the
geometric centre of the clear lane at ``12 + 105 = 117 mm = 0.20`` tile and therefore give curve
lane radii ``0.30``/``0.70`` tile.

Because M2 grades the lane graph *against the rendered markings* (5 mm tolerance, and the fork is
11.7 mm), this module **derives** the lane centre from the tape layout by default, so the reward
ground truth is by construction the geometric centre of the painted lane. Set
:attr:`TileSpec.lane_center_offset_mm` explicitly to force the 0.22 convention if a downstream
comparison with duckietown-world is ever needed; :data:`DUCKIETOWN_WORLD_LANE_OFFSET_TILE` is
provided for that purpose.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass, replace
from typing import Final

import numpy as np

__all__ = [
    "DUCKIETOWN_WORLD_LANE_OFFSET_TILE",
    "GEOMETRY_BUCKET_COUNT",
    "IDEAL_PALETTE",
    "NOMINAL_TILE_SPEC",
    "PHOTO_PALETTE",
    "ROBOT_WIDTH_M",
    "ColorPalette",
    "TileSpec",
    "TileSpecRanges",
    "geometry_buckets",
    "hue_shift_rgb",
    "palette_buckets",
    "variant_material_scalars",
]

RGB = tuple[float, float, float]

#: duckietown-world's lane-centre offset in tile units. See the module docstring.
DUCKIETOWN_WORLD_LANE_OFFSET_TILE: Final[float] = 0.22

#: Robot width used by the S5.4 progress gate (SPEC v2 S5.4, ``W_R``).
ROBOT_WIDTH_M: Final[float] = 0.131

#: Number of quantised marking-geometry buckets (SPEC v2 S3.3 / S7.2 axis V9).
GEOMETRY_BUCKET_COUNT: Final[int] = 16


def hue_shift_rgb(rgb: RGB, hue_deg: float = 0.0, sat_scale: float = 1.0, val_scale: float = 1.0) -> RGB:
    """Shift an sRGB triple in HSV space.

    Args:
        rgb: Colour as three floats in ``[0, 1]``.
        hue_deg: Hue rotation in degrees (wraps).
        sat_scale: Multiplicative saturation factor.
        val_scale: Multiplicative value factor.

    Returns:
        The shifted colour, clipped to ``[0, 1]``.
    """
    h, s, v = colorsys.rgb_to_hsv(*rgb)
    h = (h + hue_deg / 360.0) % 1.0
    s = min(1.0, max(0.0, s * sat_scale))
    v = min(1.0, max(0.0, v * val_scale))
    out = colorsys.hsv_to_rgb(h, s, v)
    return (float(out[0]), float(out[1]), float(out[2]))


@dataclass(frozen=True)
class ColorPalette:
    """sRGB colours of every painted element, as floats in ``[0, 1]``.

    Two endpoints are provided as module constants: :data:`IDEAL_PALETTE` (the idealised
    segmentation-friendly colours) and :data:`PHOTO_PALETTE` (medians measured from photographs
    of real tape). SPEC v2 S3.3 requires domain randomisation *between* these endpoints.
    """

    road: RGB = (0.0, 0.0, 0.0)
    white: RGB = (1.0, 1.0, 1.0)
    yellow: RGB = (1.0, 1.0, 0.0)
    red: RGB = (1.0, 0.0, 0.0)
    grass: RGB = (0.369, 0.514, 0.161)
    asphalt: RGB = (0.239, 0.223, 0.192)

    def blend(self, other: ColorPalette, t: float) -> ColorPalette:
        """Linearly interpolate every channel toward ``other``.

        Args:
            other: The far endpoint.
            t: Blend factor; ``0`` returns ``self``, ``1`` returns ``other``.

        Returns:
            The blended palette.
        """
        t = float(np.clip(t, 0.0, 1.0))

        def mix(a: RGB, b: RGB) -> RGB:
            return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)

        return ColorPalette(
            road=mix(self.road, other.road),
            white=mix(self.white, other.white),
            yellow=mix(self.yellow, other.yellow),
            red=mix(self.red, other.red),
            grass=mix(self.grass, other.grass),
            asphalt=mix(self.asphalt, other.asphalt),
        )

    def jittered(
        self,
        yellow_hue_deg: float = 0.0,
        yellow_sat_scale: float = 1.0,
        white_hue_deg: float = 0.0,
        road_value_scale: float = 1.0,
    ) -> ColorPalette:
        """Apply the S7.2 V7 tape-tint randomisation axes.

        Args:
            yellow_hue_deg: Yellow hue rotation in degrees (SPEC range +/- 20).
            yellow_sat_scale: Yellow saturation scale (SPEC range 0.75 .. 1.10).
            white_hue_deg: White hue rotation in degrees (SPEC range +/- 8).
            road_value_scale: Road luminance scale.

        Returns:
            A new palette with the jitters applied.
        """
        return replace(
            self,
            yellow=hue_shift_rgb(self.yellow, yellow_hue_deg, yellow_sat_scale, 1.0),
            white=hue_shift_rgb(self.white, white_hue_deg, 1.0, 1.0),
            road=hue_shift_rgb(self.road, 0.0, 1.0, road_value_scale),
        )


#: Idealised colours, from the synthetic tile style (road #000000, white #FFFFFF,
#: yellow #FFFF00, red #FF0000). Used as one DR endpoint and for segmentation ground truth.
IDEAL_PALETTE: Final[ColorPalette] = ColorPalette(
    road=(0.0, 0.0, 0.0),
    white=(1.0, 1.0, 1.0),
    yellow=(1.0, 1.0, 0.0),
    red=(1.0, 0.0, 0.0),
    grass=(0.369, 0.514, 0.161),
    asphalt=(0.239, 0.223, 0.192),
)

#: Photographic endpoint: road #403E38, white #EFEFEE, yellow #EDE04A, red #EC3743,
#: grass #5E8329, asphalt #3D3931 (SPEC v2 S3.3).
PHOTO_PALETTE: Final[ColorPalette] = ColorPalette(
    road=(0x40 / 255, 0x3E / 255, 0x38 / 255),
    white=(0xEF / 255, 0xEF / 255, 0xEE / 255),
    yellow=(0xED / 255, 0xE0 / 255, 0x4A / 255),
    red=(0xEC / 255, 0x37 / 255, 0x43 / 255),
    grass=(0x5E / 255, 0x83 / 255, 0x29 / 255),
    asphalt=(0x3D / 255, 0x39 / 255, 0x31 / 255),
)


@dataclass(frozen=True)
class TileSpec:
    """Every tile and marking dimension, in millimetres.

    All fields are millimetres unless the name says otherwise. Derived properties return metres,
    because metres are what the simulator, the lane graph and the USD stage speak.

    Attributes:
        tile_pitch_mm: Tile repeat distance. 585 mm is the pitch used by 38 of the 44 shipped
            Duckietown maps; the physical tile is 610 mm across the interlocking dents, and the
            dents overlap. SPEC v2 S3.3 DR range 570 .. 615.
        white_tape_mm: Width of the solid white lane-edge tape (48 mm, official spec; 48.3 mm
            measured). SPEC DR range 40 .. 56.
        yellow_tape_mm: Width of the dashed yellow centre tape, a half-width tape roll (24 mm
            official; 23.4 mm measured). SPEC DR range 19 .. 31.
        red_tape_mm: Width of the red stop bar (48 mm).
        red_tape_len_mm: Length of the red stop bar across the lane (210 mm official; the stop
            line spans only the incoming lane, not the full tile).
        clear_lane_mm: Clear drivable width from the inner edge of the yellow tape to the inner
            edge of the white tape (210 mm nominal). SPEC DR range 170 .. 280.
        dash_mm: Yellow dash mark length (50 mm official; 41.6 mm measured in the sim texture).
            SPEC DR range 45 .. 55.
        gap_mm: Yellow dash gap length (25 mm official; 16.1 mm measured). SPEC DR range 20 .. 30.
        dash_phase: Dash pattern phase as a fraction of the dash period, in ``[0, 1)``.
        stop_line_inset_mm: Distance from the tile edge to the outer edge of the red stop bar.
            7.5 mm matches the measured 4-way texture (bands at 7.3 .. 58.8 mm from the edge).
        yellow_stub_mm: Length of the yellow centre stubs at intersection entries, measured
            inward from the tile edge (60 mm, official 4-way assembly instructions).
        corner_white_mm: Requested length of the white corner pieces at intersection entries,
            measured inward from the tile edge. The painter clamps it to
            ``pitch/2 - white_center_offset`` so a corner arm never intrudes into the crossing
            box; at nominal values that clamp is 46.5 mm.
        lane_center_offset_mm: Explicit override for the lane centreline offset from the tile
            centreline. ``None`` (the default) derives it from the tape layout; see the module
            docstring for the 0.22 vs 0.20 fork.
        edge_soft_mm: Soft edge width used when rasterising markings, in millimetres. Purely
            cosmetic anti-aliasing help on top of the 2x supersample.
    """

    tile_pitch_mm: float = 585.0
    white_tape_mm: float = 48.0
    yellow_tape_mm: float = 24.0
    red_tape_mm: float = 48.0
    red_tape_len_mm: float = 210.0
    clear_lane_mm: float = 210.0
    dash_mm: float = 50.0
    gap_mm: float = 25.0
    dash_phase: float = 0.0
    stop_line_inset_mm: float = 7.5
    yellow_stub_mm: float = 60.0
    corner_white_mm: float = 105.0
    lane_center_offset_mm: float | None = None
    edge_soft_mm: float = 0.0

    # ---------------------------------------------------------------- derived, all in metres
    @property
    def pitch_m(self) -> float:
        """Tile repeat distance in metres."""
        return self.tile_pitch_mm / 1000.0

    @property
    def half_m(self) -> float:
        """Half tile pitch in metres (tile-local coordinates span ``[-half, +half]``)."""
        return self.tile_pitch_mm / 2000.0

    @property
    def yellow_half_m(self) -> float:
        """Half width of the yellow centre tape, in metres."""
        return self.yellow_tape_mm / 2000.0

    @property
    def white_half_m(self) -> float:
        """Half width of the white edge tape, in metres."""
        return self.white_tape_mm / 2000.0

    @property
    def white_center_offset_m(self) -> float:
        """Distance from the tile centreline to the *centre* of the white edge tape, in metres."""
        return (self.yellow_tape_mm / 2.0 + self.clear_lane_mm + self.white_tape_mm / 2.0) / 1000.0

    @property
    def white_inner_offset_m(self) -> float:
        """Distance from the tile centreline to the inner edge of the white tape, in metres."""
        return (self.yellow_tape_mm / 2.0 + self.clear_lane_mm) / 1000.0

    @property
    def lane_center_offset_m(self) -> float:
        """Distance from the tile centreline to a lane centreline, in metres.

        Derived as ``yellow_tape/2 + clear_lane/2`` unless :attr:`lane_center_offset_mm`
        overrides it. At nominal values this is 0.117 m (= 0.20 tile), the geometric centre of
        the painted clear lane.
        """
        if self.lane_center_offset_mm is not None:
            return self.lane_center_offset_mm / 1000.0
        return (self.yellow_tape_mm / 2.0 + self.clear_lane_mm / 2.0) / 1000.0

    @property
    def lane_center_offset_tile(self) -> float:
        """The lane centreline offset expressed in tile units (0.20 at nominal values)."""
        return self.lane_center_offset_m / self.pitch_m

    @property
    def lane_width_m(self) -> float:
        """Clear drivable lane width in metres. This is ``w_ep`` in the SPEC v2 S5.4 reward."""
        return self.clear_lane_mm / 1000.0

    @property
    def curve_radius_inner_m(self) -> float:
        """Lane-centre radius of the inner (right-turn) arc of a curve tile, in metres."""
        return self.half_m - self.lane_center_offset_m

    @property
    def curve_radius_outer_m(self) -> float:
        """Lane-centre radius of the outer (left-turn) arc of a curve tile, in metres."""
        return self.half_m + self.lane_center_offset_m

    @property
    def curve_radius_yellow_m(self) -> float:
        """Radius of the yellow centre arc of a curve tile (always half a tile), in metres."""
        return self.half_m

    @property
    def dash_period_m(self) -> float:
        """Yellow dash repeat distance (mark + gap) in metres."""
        return (self.dash_mm + self.gap_mm) / 1000.0

    def rescaled(self, pitch_m: float) -> TileSpec:
        """Return this spec with every dimension scaled so the tile pitch becomes ``pitch_m``.

        A layout variant carries its own tile pitch while its markings come from a shared
        geometry bucket. Scaling the whole spec keeps every marking at the same fraction of the
        tile, which is exactly how a texture painted at one pitch behaves when it is mapped onto
        a quad of another pitch.

        Args:
            pitch_m: Target tile pitch in metres.

        Returns:
            A rescaled, validated :class:`TileSpec`.

        Raises:
            ValueError: If ``pitch_m`` is not positive.
        """
        if pitch_m <= 0.0:
            raise ValueError(f"pitch_m must be > 0, got {pitch_m}")
        k = pitch_m * 1000.0 / self.tile_pitch_mm
        out = replace(
            self,
            tile_pitch_mm=self.tile_pitch_mm * k,
            white_tape_mm=self.white_tape_mm * k,
            yellow_tape_mm=self.yellow_tape_mm * k,
            red_tape_mm=self.red_tape_mm * k,
            red_tape_len_mm=self.red_tape_len_mm * k,
            clear_lane_mm=self.clear_lane_mm * k,
            dash_mm=self.dash_mm * k,
            gap_mm=self.gap_mm * k,
            stop_line_inset_mm=self.stop_line_inset_mm * k,
            yellow_stub_mm=self.yellow_stub_mm * k,
            corner_white_mm=self.corner_white_mm * k,
            lane_center_offset_mm=(
                None if self.lane_center_offset_mm is None else self.lane_center_offset_mm * k
            ),
            edge_soft_mm=self.edge_soft_mm * k,
        )
        out.validate()
        return out

    def validate(self) -> None:
        """Raise :class:`ValueError` if the dimensions cannot produce a physical tile.

        Raises:
            ValueError: If any dimension is non-positive, or if the markings do not fit inside
                the tile pitch, or the dash pattern is degenerate.
        """
        positives = {
            "tile_pitch_mm": self.tile_pitch_mm,
            "white_tape_mm": self.white_tape_mm,
            "yellow_tape_mm": self.yellow_tape_mm,
            "red_tape_mm": self.red_tape_mm,
            "red_tape_len_mm": self.red_tape_len_mm,
            "clear_lane_mm": self.clear_lane_mm,
            "dash_mm": self.dash_mm,
            "gap_mm": self.gap_mm,
        }
        for name, value in positives.items():
            if value <= 0.0:
                raise ValueError(f"TileSpec.{name} must be > 0, got {value}")
        if not 0.0 <= self.dash_phase < 1.0:
            raise ValueError(f"TileSpec.dash_phase must be in [0, 1), got {self.dash_phase}")
        outer = self.white_center_offset_m + self.white_half_m
        if outer >= self.half_m:
            raise ValueError(
                f"markings do not fit: white tape outer edge at {outer * 1000:.1f} mm exceeds the "
                f"tile half-pitch {self.half_m * 1000:.1f} mm"
            )
        inner_white_arc_r = self.half_m - self.white_center_offset_m
        if inner_white_arc_r <= self.white_half_m:
            raise ValueError(
                f"inner white arc radius {inner_white_arc_r * 1000:.1f} mm is not larger than the "
                f"white tape half-width {self.white_half_m * 1000:.1f} mm; the arc would self-overlap"
            )
        if self.stop_line_inset_mm < 0.0 or self.yellow_stub_mm <= 0.0 or self.corner_white_mm <= 0.0:
            raise ValueError("intersection marking lengths must be positive (inset may be zero)")


#: The nominal, un-randomised tile geometry of SPEC v2 S3.3.
NOMINAL_TILE_SPEC: Final[TileSpec] = TileSpec()


@dataclass(frozen=True)
class TileSpecRanges:
    """Uniform sampling ranges for the S7.2 axis V9 (marking geometry) domain randomisation.

    Each field is a ``(low, high)`` pair in the same unit as the corresponding
    :class:`TileSpec` field. The defaults are exactly the SPEC v2 S3.3 / S7.2 ranges.

    .. warning::
       The SPEC v2 V9 ranges are **jointly infeasible** at their upper ends. The markings must
       fit inside the tile: ``yellow/2 + clear_lane + white < pitch/2``. At the nominal pitch of
       585 mm with 24 mm yellow and 48 mm white that caps ``clear_lane`` at 232.5 mm, well below
       the quoted upper bound of 280 mm; at the minimum pitch of 570 mm with the widest tapes the
       cap falls to 210.5 mm. :meth:`sample` therefore *clamps* the clear-lane draw to the
       feasible interval implied by the other draws, leaving ``min_shoulder_mm`` of asphalt
       outside the white tape. The effective upper bound is reported by
       :meth:`feasible_clear_lane_mm`.
    """

    tile_pitch_mm: tuple[float, float] = (570.0, 615.0)
    white_tape_mm: tuple[float, float] = (40.0, 56.0)
    yellow_tape_mm: tuple[float, float] = (19.0, 31.0)
    clear_lane_mm: tuple[float, float] = (170.0, 280.0)
    dash_mm: tuple[float, float] = (45.0, 55.0)
    gap_mm: tuple[float, float] = (20.0, 30.0)
    dash_phase: tuple[float, float] = (0.0, 1.0)
    min_shoulder_mm: float = 5.0

    def feasible_clear_lane_mm(
        self, tile_pitch_mm: float, yellow_tape_mm: float, white_tape_mm: float
    ) -> float:
        """Largest clear lane width that still fits inside the tile.

        Args:
            tile_pitch_mm: Tile pitch of the variant being sampled.
            yellow_tape_mm: Yellow tape width of the variant.
            white_tape_mm: White tape width of the variant.

        Returns:
            The upper bound on ``clear_lane_mm`` in millimetres, leaving
            :attr:`min_shoulder_mm` of asphalt outside the white tape.
        """
        return tile_pitch_mm / 2.0 - yellow_tape_mm / 2.0 - white_tape_mm - self.min_shoulder_mm

    def sample(self, rng: np.random.Generator, alpha: float = 1.0) -> TileSpec:
        """Draw one :class:`TileSpec` from these ranges.

        Args:
            rng: Seeded numpy generator; the only source of randomness.
            alpha: Curriculum scalar in ``[0, 1]`` (SPEC v2 S7.4). ``0`` returns the nominal
                spec, ``1`` samples the full range; intermediate values shrink each range
                symmetrically around the nominal value.

        Returns:
            A validated :class:`TileSpec`. The clear lane width is clamped to the feasible
            interval; see the class-level warning.

        Raises:
            ValueError: If the sampled tape widths leave no feasible clear lane at all.
        """
        alpha = float(np.clip(alpha, 0.0, 1.0))
        nominal = NOMINAL_TILE_SPEC

        def draw(name: str) -> float:
            low, high = getattr(self, name)
            nom = getattr(nominal, name)
            low = nom + (low - nom) * alpha
            high = nom + (high - nom) * alpha
            return float(rng.uniform(low, high))

        pitch = draw("tile_pitch_mm")
        white = draw("white_tape_mm")
        yellow = draw("yellow_tape_mm")
        clear = draw("clear_lane_mm")
        clear_max = self.feasible_clear_lane_mm(pitch, yellow, white)
        clear_min = min(self.clear_lane_mm[0], clear_max)
        if clear_max <= 0.0:
            raise ValueError(
                f"no feasible clear lane: pitch {pitch:.1f} mm with yellow {yellow:.1f} mm and "
                f"white {white:.1f} mm leaves {clear_max:.1f} mm"
            )
        spec = TileSpec(
            tile_pitch_mm=pitch,
            white_tape_mm=white,
            yellow_tape_mm=yellow,
            clear_lane_mm=float(np.clip(clear, clear_min, clear_max)),
            dash_mm=draw("dash_mm"),
            gap_mm=draw("gap_mm"),
            dash_phase=draw("dash_phase") % 1.0,
        )
        spec.validate()
        return spec


def geometry_buckets(
    count: int = GEOMETRY_BUCKET_COUNT,
    seed: int = 0,
    ranges: TileSpecRanges | None = None,
    alpha: float = 1.0,
) -> tuple[TileSpec, ...]:
    """Quantise the V9 marking-geometry DR axis into a fixed set of buckets.

    SPEC v2 S3.3 and S5.6 require the marking textures to collapse to 16 geometry buckets shared
    across the 64 city variants, with the per-variant *colour* carried by material scalar tints
    rather than by unique textures. This function produces those buckets deterministically.

    Bucket 0 is always the nominal spec so that a DR-off run is exactly reproducible.

    Args:
        count: Number of buckets to produce.
        seed: Seed for the internal generator.
        ranges: Sampling ranges; defaults to :class:`TileSpecRanges`.
        alpha: Curriculum scalar forwarded to :meth:`TileSpecRanges.sample`.

    Returns:
        A tuple of ``count`` validated :class:`TileSpec` instances.

    Raises:
        ValueError: If ``count`` is not positive.
    """
    if count <= 0:
        raise ValueError(f"count must be > 0, got {count}")
    rng = np.random.default_rng(seed)
    ranges = ranges or TileSpecRanges()
    out = [NOMINAL_TILE_SPEC]
    while len(out) < count:
        out.append(ranges.sample(rng, alpha=alpha))
    return tuple(out[:count])


def variant_material_scalars(index: int, seed: int = 0, alpha: float = 1.0) -> tuple[RGB, float, float]:
    """Per-variant OmniPBR scalar handles authored into the city stage.

    SPEC v2 S3.3 and S7.1 layer 2b require per-variant *colour* to be carried by material scalars
    rather than by unique textures, and require the runtime domain randomisation to write only
    those same three scalars (never a texture assignment). This function produces their
    authored-at-build-time values; the reset event terms later overwrite them per episode.

    Args:
        index: Variant index; the values are a deterministic function of ``(index, seed)``.
        seed: Base seed.
        alpha: Curriculum scalar in ``[0, 1]`` shrinking every range toward the neutral value.

    Returns:
        ``(diffuse_tint, albedo_brightness, reflection_roughness_constant)``.
    """
    alpha = float(np.clip(alpha, 0.0, 1.0))
    rng = np.random.default_rng((int(seed), int(index)))
    tint = tuple(1.0 + (float(rng.uniform(0.80, 1.05)) - 1.0) * alpha for _ in range(3))
    brightness = 1.0 + (float(rng.uniform(0.70, 1.25)) - 1.0) * alpha
    roughness = 0.8 + (float(rng.uniform(0.50, 0.95)) - 0.8) * alpha
    return (float(tint[0]), float(tint[1]), float(tint[2])), brightness, roughness


def palette_buckets(count: int = 64, seed: int = 0, alpha: float = 1.0) -> tuple[ColorPalette, ...]:
    """Deterministic per-variant colour palettes blended between the two endpoints.

    Palette 0 is always :data:`IDEAL_PALETTE` so a DR-off run is reproducible.

    Args:
        count: Number of palettes to produce (one per city variant).
        seed: Seed for the internal generator.
        alpha: Curriculum scalar in ``[0, 1]`` scaling the blend and the tint jitters.

    Returns:
        A tuple of ``count`` :class:`ColorPalette` instances.

    Raises:
        ValueError: If ``count`` is not positive.
    """
    if count <= 0:
        raise ValueError(f"count must be > 0, got {count}")
    alpha = float(np.clip(alpha, 0.0, 1.0))
    rng = np.random.default_rng(seed)
    out = [IDEAL_PALETTE]
    while len(out) < count:
        base = IDEAL_PALETTE.blend(PHOTO_PALETTE, float(rng.uniform(0.0, 1.0)) * alpha)
        out.append(
            base.jittered(
                yellow_hue_deg=float(rng.uniform(-20.0, 20.0)) * alpha,
                yellow_sat_scale=1.0 + (float(rng.uniform(0.75, 1.10)) - 1.0) * alpha,
                white_hue_deg=float(rng.uniform(-8.0, 8.0)) * alpha,
                road_value_scale=1.0 + (float(rng.uniform(0.7, 1.3)) - 1.0) * alpha,
            )
        )
    return tuple(out[:count])
