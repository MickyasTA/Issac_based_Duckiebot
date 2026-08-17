"""Pre-flight sanity check on what the policy actually sees (SPEC v2 S4.3, S5.2, S6.7).

Run it with the Isaac Sim interpreter, before any multi-day campaign::

    & d:/Personal/personal/wheeled_quadruped_robot/.venv/Scripts/python.exe `
        scripts/check_obs.py --num_envs 6 --headless --enable_cameras

    # with the photometric randomisation at the alpha a resumed run would use
    & $ISAAC scripts/check_obs.py --num_envs 6 --headless --enable_cameras `
        --visual-dr --alpha-vis 0.6

Why this script exists
----------------------
A vision policy that will not learn is, far more often than not, looking at something other than
what its author believes, and a reward curve cannot tell you that. One contact sheet can. The
failures this catches in under three minutes, each of which otherwise costs a day of compute:

* **Black camera.** ``TiledCamera`` returns an all-zero buffer for the first renders after the
  RTX pipeline is created, and it returns an all-zero buffer forever if the annotator was never
  attached. ``--settle-steps`` walks the camera forward and the per-step brightness trace printed
  below shows exactly which of the two you have.
* **Upside-down or wrongly cropped image.** S4.3 step 7 drops the TOP 16 rows of the 96x64
  downsampled frame. If the crop were applied to the bottom, or the render were flipped, the
  observation would be sky and the horizon statistic below would inverted.
* **Dead DR pipeline.** With ``--visual-dr`` the per-env frames must differ from each other; if
  the exposure/gain/tint axes are silently disabled every env looks identical, which is reported
  as a zero cross-env spread.
* **Wrong channel order.** Yellow lane tape read as blue is instantly visible in the contact
  sheet and shows up as a zero yellow-detection rate.
* **Stale frame ring.** The three panels of a stack are ``t``, ``t-2`` and ``t-4``; three
  identical panels mean the ring is not advancing.

Lane-frame ground truth (``--lane-check``, on by default)
---------------------------------------------------------
The observation can be perfect and the *reward* still wrong, if the lane query disagrees with the
geometry that was rendered. The second half of this script closes that loop: it places each robot
at a commanded signed lateral offset ``d_cmd`` on a straight lane segment using
:func:`duckiebot_rl.envs.obstacles.lane_frame_to_world`, steps once, and compares the offset the
environment reports (``BatchedLaneGraph.query(...).d``, the number the S5.4 reward integrates)
against ``d_cmd``. Those are two independent directions of the same map -- a forward placement and
an inverse projection -- so agreement to a few millimetres is a real round trip and not a
tautology.

The rendered image is tied to the same number by the yellow-centroid column printed for every
offset. ``d > 0`` is toward the yellow centre tape (S2), so as ``d_cmd`` sweeps from the white
edge toward the yellow line the centroid of the detected yellow pixels must march to the RIGHT
across the frame, and the reported correlation must be positive. A lane graph mirrored against
the city USD passes the numeric round trip and fails this one.

Outputs
-------
``docs/img/obs_check.png``
    The labelled contact sheet: one row per environment, columns ``RAW RENDER``, ``OBS T``,
    ``OBS T-2``, ``OBS T-4`` and ``YELLOW MASK``.
``docs/img/obs_check_lane.png``
    The lane-offset sweep, written only when ``--lane-check`` runs.
stdout
    Per-channel min/max/mean/std, saturated-pixel fractions, the yellow-detection rate, the
    horizon statistic, the per-step brightness trace, and the lane round-trip table.

Exit code
---------
0 if every hard check passed, 1 otherwise. The hard checks are: no all-black observation, no
channel fully saturated, the frame ring advancing, and (when ``--lane-check`` runs) a lane
round-trip error under ``--lane-tol``. That makes this script usable as a pre-launch gate in a
campaign script, not only as something to look at.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from isaaclab.app import AppLauncher

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


DEFAULT_OUT = "docs/img/obs_check.png"
"""Contact sheet destination, relative to the repository root."""

YELLOW_HUE_RANGE_DEG: tuple[float, float] = (35.0, 72.0)
"""Hue window accepted as "yellow tape", in degrees.

These numbers are measured, not guessed, and the first version of this script got them wrong in
the direction that matters. A window of 30..80 deg with a saturation floor of 0.30 reported a
100% detection rate on six environments while the mask overlay was in fact painting the GRASS:
the rendered lawn sits at hue 67..79 with saturation 0.46..0.57, which walks straight into a
window chosen from the authored palette alone. Sampled off the rendered 96x48 observations:

* tape, brightest saturated pixels: ``(241, 243, 141)``, ``(245, 226, 101)``, ``(239, 228, 110)``
  -- hue 61.2, 52.1, 54.9 deg, saturation 0.42..0.59, value 0.94..0.96;
* grass, the bulk of the mid-tone saturated pixels: hue 67..79 deg, value 0.45..0.75.

Hue alone does not separate them cleanly (they meet around 62..67 deg); hue plus
:data:`YELLOW_MIN_VALUE` does, because the tape is retro-bright and the lawn is not.
"""

YELLOW_MIN_SATURATION = 0.35
"""Minimum HSV saturation, which is what rejects grey road, white edge tape and the sky."""

YELLOW_MIN_VALUE = 0.78
"""Minimum HSV value, which is what rejects grass.

The load-bearing threshold. Rendered grass tops out around 0.75 under the S5.1 dome plus sun,
while the tape sits at 0.94..0.96, so this cut removes essentially all of the lawn and none of
the tape. It is why the per-environment counts fell from 541..1347 "yellow" pixels (grass) to
0..165 (dashes) when it was introduced.
"""

YELLOW_MIN_PIXELS = 12
"""Pixels of a 96x48 observation that must pass the test before the line counts as detected.

Twelve is about one 24 mm dash at mid range. It sits above the residual speckle the detector
still picks off sunlit grass right at the horizon (single digits per frame) and below the
100..170 pixels a real dash contributes.

Known limitation, stated rather than hidden: the S7.2 V7 axis rotates the tape hue by up to
+/- 20 deg, and at the +20 end the tape genuinely renders chartreuse and overlaps the lawn in
hue. Under heavy visual DR this count is a FLOOR on how much tape is visible, not an exact
measurement. It is a black-camera and channel-order detector, not a segmentation model.
"""

YELLOW_CENTROID_ROW_FRACTION = 0.5
"""Bottom fraction of the frame used for the yellow centroid column.

The centroid is a lateral-position cue, so it must come from the near field. Computing it over
the whole frame lets a few horizon-grass pixels 40 rows away drag it sideways, which is exactly
what broke the monotonic-in-``d`` check on the first run of this script.
"""

_LABEL_H = 22
"""Pixel height reserved above each contact-sheet row for its text label."""

_BG = (18, 18, 22)
"""Contact-sheet background, matching :mod:`duckiebot_rl.viz.render`."""


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser, with the AppLauncher arguments appended last.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        description="Dump and validate the preprocessed observation before a long training run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num_envs", type=int, default=6, help="parallel environments to boot")
    parser.add_argument("--rows", type=int, default=6, help="environments drawn on the contact sheet")
    parser.add_argument("--seed", type=int, default=0, help="master seed")
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=12,
        help="control steps taken before the observation is captured; the brightness trace covers them",
    )
    parser.add_argument("--visual-dr", action="store_true", help="enable the S4.3/S7.2 photometric DR")
    parser.add_argument("--alpha-vis", type=float, default=1.0, help="visual DR curriculum alpha")
    parser.add_argument("--dynamics-dr", action="store_true", help="enable the S7.3 dynamics DR")
    parser.add_argument("--alpha-dyn", type=float, default=1.0, help="dynamics DR curriculum alpha")
    parser.add_argument("--obstacles", action="store_true", help="spawn the S5.1 obstacle field")
    parser.add_argument("--city-root", default=None, help="directory holding the generated city USD")
    parser.add_argument("--num-variants", type=int, default=8, help="city layouts to load")
    parser.add_argument("--out", default=DEFAULT_OUT, help="contact sheet destination PNG")
    parser.add_argument("--scale", type=int, default=4, help="integer magnification of each tile")
    parser.add_argument(
        "--no-lane-check",
        dest="lane_check",
        action="store_false",
        help="skip the lane-frame ground-truth round trip",
    )
    parser.add_argument(
        "--lane-tol",
        type=float,
        default=0.005,
        help="maximum |d_reported - d_commanded| accepted by the lane round trip, in metres",
    )
    parser.add_argument("--json", default="", help="also write the full report as JSON here")

    # MUST be last: add_app_launcher_args inspects the parser it is given.
    AppLauncher.add_app_launcher_args(parser)
    return parser


# =============================================================================================
# Pure image analysis. No Isaac, no torch: everything below operates on uint8 numpy arrays.
# =============================================================================================


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB image to HSV without pulling in OpenCV or matplotlib.

    Args:
        rgb: ``(H, W, 3)`` array, uint8 or float in ``[0, 255]``.

    Returns:
        ``(H, W, 3)`` float32 array; hue in degrees ``[0, 360)``, saturation and value in
        ``[0, 1]``.

    Raises:
        ValueError: If the input is not ``(H, W, 3)``.
    """
    array = np.asarray(rgb, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) RGB image, got shape {array.shape}")
    scaled = array / 255.0
    r, g, b = scaled[..., 0], scaled[..., 1], scaled[..., 2]
    value = scaled.max(axis=2)
    minimum = scaled.min(axis=2)
    chroma = value - minimum

    safe_chroma = np.where(chroma == 0.0, 1.0, chroma)
    hue = np.zeros_like(value)
    hue = np.where(value == r, ((g - b) / safe_chroma) % 6.0, hue)
    hue = np.where(value == g, (b - r) / safe_chroma + 2.0, hue)
    hue = np.where(value == b, (r - g) / safe_chroma + 4.0, hue)
    hue = np.where(chroma == 0.0, 0.0, hue * 60.0)

    saturation = np.where(value == 0.0, 0.0, chroma / np.where(value == 0.0, 1.0, value))
    return np.stack([hue, saturation, value], axis=-1).astype(np.float32)


def yellow_mask(frame: np.ndarray) -> np.ndarray:
    """Return the boolean mask of pixels that look like the yellow centre tape.

    The test is in HSV rather than in raw RGB because the S7.2 photometric axes move exposure,
    gain and white balance, all of which change RGB magnitudes while leaving hue roughly where it
    was. A raw ``R > k and G > k and B < k`` test reports "no lane line" on a dimmed frame.

    Args:
        frame: ``(H, W, 3)`` uint8 RGB observation frame.

    Returns:
        ``(H, W)`` bool array.
    """
    hsv = rgb_to_hsv(frame)
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    low, high = YELLOW_HUE_RANGE_DEG
    return (hue >= low) & (hue <= high) & (saturation >= YELLOW_MIN_SATURATION) & (value >= YELLOW_MIN_VALUE)


def yellow_centroid_column(mask: np.ndarray, near_field_only: bool = True) -> float:
    """Return the mean column index of the masked pixels, or NaN when nothing is masked.

    Args:
        mask: ``(H, W)`` bool array from :func:`yellow_mask`.
        near_field_only: Restrict the average to the bottom
            :data:`YELLOW_CENTROID_ROW_FRACTION` of the frame, which is where the ego lane's tape
            is and where the horizon is not.

    Returns:
        Column index in ``[0, W - 1]``, or ``float("nan")``.
    """
    window = mask
    if near_field_only:
        start = int(mask.shape[0] * (1.0 - YELLOW_CENTROID_ROW_FRACTION))
        window = mask[start:]
    if not window.any():
        return float("nan")
    return float(np.nonzero(window)[1].mean())


def frame_statistics(frame: np.ndarray) -> dict[str, Any]:
    """Compute the per-channel and whole-frame statistics of one observation frame.

    Args:
        frame: ``(H, W, 3)`` uint8 RGB frame.

    Returns:
        A dict with per-channel min/max/mean/std, the saturated-pixel fractions, the yellow
        detection result and the horizon statistic.
    """
    array = np.asarray(frame)
    channels = [array[..., c].astype(np.float32) for c in range(array.shape[2])]
    mask = yellow_mask(array)
    rows = array.shape[0]
    third = max(rows // 3, 1)
    top_mean = float(array[:third].astype(np.float32).mean())
    bottom_mean = float(array[-third:].astype(np.float32).mean())
    return {
        "min": [float(c.min()) for c in channels],
        "max": [float(c.max()) for c in channels],
        "mean": [round(float(c.mean()), 3) for c in channels],
        "std": [round(float(c.std()), 3) for c in channels],
        "frac_zero": round(float((array == 0).mean()), 5),
        "frac_255": round(float((array == 255).mean()), 5),
        "yellow_pixels": int(mask.sum()),
        "yellow_detected": bool(int(mask.sum()) >= YELLOW_MIN_PIXELS),
        "yellow_centroid_col": round(yellow_centroid_column(mask), 2),
        "top_third_mean": round(top_mean, 3),
        "bottom_third_mean": round(bottom_mean, 3),
    }


def mask_overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return a dimmed copy of ``frame`` with the masked pixels painted magenta.

    Magenta because it is the one hue that cannot be confused with anything the city renders:
    road, tape, grass and sky are all outside it.

    Args:
        frame: ``(H, W, 3)`` uint8 RGB frame.
        mask: ``(H, W)`` bool array.

    Returns:
        ``(H, W, 3)`` uint8 RGB image.
    """
    out = (np.asarray(frame).astype(np.float32) * 0.35).astype(np.uint8)
    out[mask] = (255, 0, 220)
    return out


def upscale(tile: np.ndarray, scale: int) -> np.ndarray:
    """Nearest-neighbour magnify a tile by an integer factor.

    Nearest neighbour, never a smooth resize: a 96x48 observation shown with interpolation hides
    exactly the single-pixel artefacts this sheet exists to reveal.

    Args:
        tile: ``(H, W, 3)`` uint8 image.
        scale: Integer magnification, at least 1.

    Returns:
        ``(H * scale, W * scale, 3)`` uint8 image.

    Raises:
        ValueError: If ``scale`` is below 1.
    """
    if scale < 1:
        raise ValueError(f"scale must be >= 1, got {scale}")
    return np.repeat(np.repeat(np.asarray(tile), scale, axis=0), scale, axis=1)


def contact_sheet(rows: list[dict[str, Any]], title: str, gap: int = 10) -> np.ndarray:
    """Compose the labelled contact sheet.

    Args:
        rows: One dict per row, with keys ``label`` (str) and ``tiles`` (list of
            ``(caption, image)`` pairs, all images already magnified).
        title: Text drawn across the top of the sheet.
        gap: Pixels of background between tiles.

    Returns:
        ``(H, W, 3)`` uint8 RGB image.

    Raises:
        ValueError: If ``rows`` is empty.
    """
    from duckiebot_rl.viz.render import draw_text

    if not rows:
        raise ValueError("contact_sheet needs at least one row")

    widths = [sum(tile.shape[1] + gap for _, tile in row["tiles"]) + gap for row in rows]
    heights = [max(tile.shape[0] for _, tile in row["tiles"]) for row in rows]
    title_h = 30
    caption_h = 16
    width = max(widths)
    height = title_h + sum(h + _LABEL_H + caption_h + gap for h in heights) + gap

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:, :] = _BG
    draw_text(canvas, gap, 6, title, color=(235, 235, 240), scale=2)

    y = title_h
    for row, row_h in zip(rows, heights, strict=True):
        draw_text(canvas, gap, y + 3, row["label"], color=(250, 210, 90), scale=2)
        top = y + _LABEL_H + caption_h
        x = gap
        for caption, tile in row["tiles"]:
            draw_text(canvas, x, y + _LABEL_H, caption, color=(150, 200, 250), scale=1)
            tile_h, tile_w = tile.shape[:2]
            canvas[top : top + tile_h, x : x + tile_w] = tile
            canvas[top - 1, x - 1 : x + tile_w + 1] = (90, 90, 100)
            canvas[top + tile_h, x - 1 : x + tile_w + 1] = (90, 90, 100)
            canvas[top - 1 : top + tile_h + 1, x - 1] = (90, 90, 100)
            canvas[top - 1 : top + tile_h + 1, x + tile_w] = (90, 90, 100)
            x += tile_w + gap
        y = top + row_h + gap
    return canvas


# =============================================================================================
# Isaac-side collection
# =============================================================================================


def build_settings(args: argparse.Namespace) -> Any:
    """Translate the command line into :class:`LaneFollowSettings`.

    Args:
        args: Parsed arguments.

    Returns:
        The populated settings object.
    """
    from duckiebot_rl.envs.env_cfg import CitySettings, LaneFollowSettings, ObstacleSettings

    return LaneFollowSettings(
        num_envs=args.num_envs,
        device=args.device,
        seed=args.seed,
        city=CitySettings(root=args.city_root, num_variants=args.num_variants),
        obstacles=ObstacleSettings(enabled=args.obstacles),
        use_image=True,
        visual_dr=args.visual_dr,
        dynamics_dr=args.dynamics_dr,
        dr_alpha_vis=args.alpha_vis if args.visual_dr else 0.0,
        dr_alpha_dyn=args.alpha_dyn if args.dynamics_dr else 0.0,
    )


def settle(env: Any, steps: int) -> list[float]:
    """Step the environment with a zero command and record the mean raw camera brightness.

    The trace is the diagnostic, not the side effect. A camera that is black because the RTX
    pipeline has not produced a frame yet shows zeros followed by a jump; a camera that is black
    because its annotator was never attached shows zeros all the way down.

    Args:
        env: The constructed environment.
        steps: Control steps to take.

    Returns:
        One mean-brightness value per step, over the whole batch and all colour channels.
    """
    import torch

    zero = torch.zeros(env.num_envs, 2, device=env.device)
    trace: list[float] = []
    for _ in range(max(steps, 0)):
        env.step(zero)
        rgb = env.onboard_camera.data.output["rgb"]
        trace.append(round(float(rgb[..., :3].float().mean()), 4))
    return trace


def collect_rows(env: Any, count: int, scale: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build the contact-sheet rows and the per-environment statistics.

    Args:
        env: The constructed environment, already settled.
        count: Environments to include.
        scale: Integer magnification for the observation tiles.

    Returns:
        ``(rows, stats)``; ``rows`` feeds :func:`contact_sheet`, ``stats`` is JSON-serialisable.
    """
    from duckiebot_rl.dr.preprocess import FRAME_STACK_OFFSETS

    raw = env.onboard_camera.data.output["rgb"][..., :3].detach().cpu().numpy().astype(np.uint8)
    stacked = env.stacked_obs.detach().cpu().numpy().astype(np.uint8)
    lane_d = env.lane_offset.detach().cpu().numpy()
    lane_psi = env.lane_heading_error.detach().cpu().numpy()
    variants = env.variant_index.detach().cpu().numpy()

    rows: list[dict[str, Any]] = []
    stats: list[dict[str, Any]] = []
    for i in range(min(count, env.num_envs)):
        frames = [stacked[i, :, :, c * 3 : c * 3 + 3] for c in range(stacked.shape[2] // 3)]
        newest = frames[0]
        row_stats = frame_statistics(newest)
        row_stats["env"] = i
        row_stats["variant"] = int(variants[i])
        row_stats["lane_d_m"] = round(float(lane_d[i]), 4)
        row_stats["lane_psi_deg"] = round(float(np.degrees(lane_psi[i])), 2)
        row_stats["raw_mean"] = round(float(raw[i].mean()), 3)
        # Two frames of the same stack being byte-identical means the ring did not advance.
        row_stats["ring_advancing"] = bool(any(not np.array_equal(frames[0], other) for other in frames[1:]))
        stats.append(row_stats)

        tiles: list[tuple[str, np.ndarray]] = [("RAW RENDER 192X128", upscale(raw[i], max(scale // 2, 1)))]
        for offset, frame in zip(FRAME_STACK_OFFSETS, frames, strict=False):
            caption = "OBS T" if offset == 0 else f"OBS T-{offset}"
            tiles.append((f"{caption} 96X48", upscale(frame, scale)))
        tiles.append(
            (
                f"YELLOW MASK {row_stats['yellow_pixels']} PX",
                upscale(mask_overlay(newest, yellow_mask(newest)), scale),
            )
        )
        label = (
            f"ENV {i}  MAP {row_stats['variant']:03d}  D {row_stats['lane_d_m']:+.3f} M  "
            f"PSI {row_stats['lane_psi_deg']:+.1f} DEG  MEAN {row_stats['mean'][0]:.0f}/"
            f"{row_stats['mean'][1]:.0f}/{row_stats['mean'][2]:.0f}  "
            f"YELLOW {'YES' if row_stats['yellow_detected'] else 'NO'}"
        )
        rows.append({"label": label, "tiles": tiles})
    return rows, stats


def lane_round_trip(env: Any, tolerance: float, scale: int) -> dict[str, Any]:
    """Place each robot at a commanded lane offset and check the reported offset agrees.

    Every environment is moved onto the longest straight segment of its own map variant, at the
    same arc length, with zero heading error, and given its own commanded lateral offset spread
    across the clear lane width. One control step later the environment's own lane query is read
    back. The physical drift over that step is reported alongside the error, because a large
    error with a large drift means the robot moved, not that the geometry disagrees.

    Args:
        env: The constructed environment.
        tolerance: Maximum accepted ``|d_reported - d_commanded|`` in metres.
        scale: Integer magnification for the sweep image tiles.

    Returns:
        A report dict with one entry per environment plus the pass/fail verdict.
    """
    import torch

    from duckiebot_rl.envs.obstacles import lane_frame_to_world

    lane = env.lane_graph
    n = env.num_envs
    device = env.device
    variant = env.variant_index

    # The longest straight segment of each env's own variant: straight so that "lateral offset"
    # is unambiguous, longest so that the arc length below is comfortably inside it.
    straight = lane.seg_valid & ~lane.seg_is_arc
    lengths = torch.where(straight, lane.seg_length, torch.zeros_like(lane.seg_length))
    segment = lengths.argmax(dim=1)[variant]
    seg_len = lane.seg_length[variant, segment]
    arc = 0.5 * seg_len

    half = 0.5 * lane.lane_width.to(device)[variant]
    fractions = torch.linspace(-0.8, 0.8, n, device=device) if n > 1 else torch.zeros(1, device=device)
    commanded = fractions * half

    x, y, yaw = lane_frame_to_world(lane, variant, segment, arc, commanded)
    root = env.robot.data.default_root_state.clone()
    origins = env.scene.env_origins
    root[:, 0] = x + origins[:, 0]
    root[:, 1] = y + origins[:, 1]
    root[:, 2] = env.settings.params.base_link_height_m + origins[:, 2]
    half_yaw = 0.5 * yaw
    root[:, 3] = torch.cos(half_yaw)
    root[:, 4] = 0.0
    root[:, 5] = 0.0
    root[:, 6] = torch.sin(half_yaw)
    root[:, 7:] = 0.0
    env.robot.write_root_state_to_sim(root)

    # Keep the placement alive: a truncation or a termination inside the step would reset the
    # robot to a random spawn and the read-back would measure that instead.
    env.episode_length_buf[:] = 0
    before = torch.stack([x, y], dim=-1).clone()
    env.step(torch.zeros(n, 2, device=env.device))

    reported = env.lane_offset.detach()
    after = (env.robot.data.root_pos_w[:, :2] - origins[:, :2]).detach()
    drift = torch.linalg.vector_norm(after - before, dim=-1)
    error = (reported - commanded).abs()

    stacked = env.stacked_obs.detach().cpu().numpy().astype(np.uint8)
    entries: list[dict[str, Any]] = []
    tiles: list[tuple[str, np.ndarray]] = []
    for i in range(n):
        newest = stacked[i, :, :, :3]
        mask = yellow_mask(newest)
        entries.append(
            {
                "env": i,
                "variant": int(variant[i]),
                "segment": int(segment[i]),
                "d_commanded_m": round(float(commanded[i]), 5),
                "d_reported_m": round(float(reported[i]), 5),
                "error_mm": round(float(error[i]) * 1000.0, 3),
                "drift_mm": round(float(drift[i]) * 1000.0, 3),
                "psi_reported_deg": round(float(np.degrees(float(env.lane_heading_error[i]))), 3),
                "yellow_pixels": int(mask.sum()),
                "yellow_centroid_col": round(yellow_centroid_column(mask), 2),
            }
        )
        tiles.append((f"D {float(commanded[i]) * 1000:+.0f} MM", upscale(newest, scale)))

    worst = float(error.max()) * 1000.0
    centroids = [e["yellow_centroid_col"] for e in entries]
    # Only offsets that actually show tape in the near field can vote. An offset whose dash
    # happens to fall in a gap contributes no centroid and must not be read as a violation.
    voting = [
        (e["d_commanded_m"], e["yellow_centroid_col"])
        for e in entries
        if e["yellow_pixels"] >= YELLOW_MIN_PIXELS and not np.isnan(e["yellow_centroid_col"])
    ]
    # A correlation rather than strict monotonicity, because each column of the sweep is a
    # DIFFERENT map variant (one stage per env, fixed at scene build), with its own lane width
    # and its own dash phase. Run with `--num-variants 1` for a same-map sweep if you want the
    # strict version. The sign is what carries the meaning: d > 0 is toward the yellow tape, so
    # the tape must move to the RIGHT across the frame as d rises, i.e. a POSITIVE correlation.
    # A lane graph mirrored against the city USD would put a large negative number here.
    correlation = float("nan")
    if len(voting) >= 3:
        d_values = np.array([v[0] for v in voting], dtype=np.float64)
        columns = np.array([v[1] for v in voting], dtype=np.float64)
        if d_values.std() > 0 and columns.std() > 0:
            correlation = float(np.corrcoef(d_values, columns)[0, 1])
    return {
        "entries": entries,
        "worst_error_mm": round(worst, 3),
        "tolerance_mm": round(tolerance * 1000.0, 3),
        "passed": bool(worst <= tolerance * 1000.0),
        "yellow_centroid_columns": centroids,
        "centroid_samples": len(voting),
        "centroid_correlation_with_d": round(correlation, 4),
        "centroid_sign_correct": bool(correlation > 0.5),
        "tiles": tiles,
    }


def print_report(report: dict[str, Any]) -> None:
    """Print the human-readable half of the report.

    Args:
        report: The assembled report dict.
    """
    print("\n[check_obs] settle trace (mean raw camera brightness per control step):")
    print("  " + "  ".join(f"{v:.2f}" for v in report["settle_trace"]))

    print("\n[check_obs] per-environment observation statistics (newest frame of the stack):")
    header = (
        f"  {'env':>3} {'map':>4} {'d[m]':>8} {'psi[deg]':>9} "
        f"{'meanR':>7} {'meanG':>7} {'meanB':>7} {'stdR':>7} "
        f"{'zero%':>7} {'sat%':>7} {'ypx':>5} {'ycol':>7} {'ring':>5}"
    )
    print(header)
    for s in report["envs"]:
        print(
            f"  {s['env']:>3} {s['variant']:>4} {s['lane_d_m']:>8.4f} {s['lane_psi_deg']:>9.2f} "
            f"{s['mean'][0]:>7.2f} {s['mean'][1]:>7.2f} {s['mean'][2]:>7.2f} {s['std'][0]:>7.2f} "
            f"{100 * s['frac_zero']:>7.2f} {100 * s['frac_255']:>7.2f} {s['yellow_pixels']:>5} "
            f"{s['yellow_centroid_col']:>7.2f} {s['ring_advancing']!s:>5}"
        )

    agg = report["aggregate"]
    print("\n[check_obs] aggregate over the sampled environments:")
    for key in (
        "channel_min",
        "channel_max",
        "channel_mean",
        "channel_std",
        "frac_zero",
        "frac_255",
        "yellow_detection_rate",
        "cross_env_spread",
        "horizon_top_over_bottom",
    ):
        print(f"  {key:<28s} {agg[key]}")

    lane = report.get("lane_check")
    if lane:
        print("\n[check_obs] lane-frame ground-truth round trip:")
        print(
            f"  {'env':>3} {'map':>4} {'seg':>4} {'d_cmd[mm]':>10} {'d_rep[mm]':>10} "
            f"{'err[mm]':>9} {'drift[mm]':>10} {'psi[deg]':>9} {'ypx':>5} {'ycol':>7}"
        )
        for e in lane["entries"]:
            print(
                f"  {e['env']:>3} {e['variant']:>4} {e['segment']:>4} "
                f"{1000 * e['d_commanded_m']:>10.2f} {1000 * e['d_reported_m']:>10.2f} "
                f"{e['error_mm']:>9.3f} {e['drift_mm']:>10.3f} {e['psi_reported_deg']:>9.3f} "
                f"{e['yellow_pixels']:>5} {e['yellow_centroid_col']:>7.2f}"
            )
        print(
            f"  worst error {lane['worst_error_mm']:.3f} mm against a tolerance of "
            f"{lane['tolerance_mm']:.1f} mm: {'PASS' if lane['passed'] else 'FAIL'}"
        )
        print(
            f"  yellow centroid column vs commanded d: correlation "
            f"{lane['centroid_correlation_with_d']:+.3f} over {lane['centroid_samples']} offsets "
            f"that showed tape, positive is the correct sign "
            f"({'PASS' if lane['centroid_sign_correct'] else 'FAIL'})"
        )

    print("\n[check_obs] verdict:")
    for name, ok in report["checks"].items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")


def main() -> int:
    """Boot Isaac, capture the observation, write the artefacts and report.

    Returns:
        0 if every hard check passed, 1 otherwise.
    """
    args, _unknown = _PARSER.parse_known_args()
    # Kit tears the process down from C++ on `simulation_app.close()`, which never flushes a
    # block-buffered Python stdout. Every print below would be lost the moment this script is
    # run with its output redirected to a file, which is exactly how a pre-launch gate is run.
    sys.stdout.reconfigure(line_buffering=True)

    from duckiebot_rl.envs.env_cfg import lane_follow_env_cfg
    from duckiebot_rl.envs.lane_follow_env import DuckiebotLaneFollowEnv
    from duckiebot_rl.sim2sim.track import write_png

    settings = build_settings(args)
    print(f"[check_obs] settings: {json.dumps(settings.summary(), sort_keys=True)}")

    cfg = lane_follow_env_cfg(settings)
    env = DuckiebotLaneFollowEnv(cfg)
    env.reset()

    trace = settle(env, args.settle_steps)
    rows, stats = collect_rows(env, args.rows, args.scale)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = _REPO_ROOT / out_path
    title = (
        f"OBS CHECK  N={args.num_envs}  VISUAL DR {'ON' if args.visual_dr else 'OFF'} "
        f"ALPHA {settings.dr_alpha_vis:.2f}  SETTLE {args.settle_steps} STEPS"
    )
    write_png(out_path, contact_sheet(rows, title))
    print(f"[check_obs] contact sheet written to {out_path.as_posix()}")

    channel_stack = np.stack([s["mean"] for s in stats], axis=0)
    aggregate = {
        "channel_min": [min(s["min"][c] for s in stats) for c in range(3)],
        "channel_max": [max(s["max"][c] for s in stats) for c in range(3)],
        "channel_mean": [round(float(channel_stack[:, c].mean()), 3) for c in range(3)],
        "channel_std": [round(float(np.mean([s["std"][c] for s in stats])), 3) for c in range(3)],
        "frac_zero": round(float(np.mean([s["frac_zero"] for s in stats])), 5),
        "frac_255": round(float(np.mean([s["frac_255"] for s in stats])), 5),
        "yellow_detection_rate": round(float(np.mean([s["yellow_detected"] for s in stats])), 3),
        # Zero spread with DR on means the photometric axes are not reaching the frame.
        "cross_env_spread": round(float(channel_stack.std(axis=0).mean()), 4),
        "horizon_top_over_bottom": round(
            float(np.mean([s["top_third_mean"] for s in stats]))
            / max(float(np.mean([s["bottom_third_mean"] for s in stats])), 1e-6),
            4,
        ),
    }

    report: dict[str, Any] = {
        "settings": settings.summary(),
        "settle_trace": trace,
        "envs": stats,
        "aggregate": aggregate,
        "contact_sheet": out_path.as_posix(),
    }

    if args.lane_check:
        lane = lane_round_trip(env, args.lane_tol, args.scale)
        lane_tiles = lane.pop("tiles")
        lane_path = out_path.with_name(f"{out_path.stem}_lane{out_path.suffix}")
        write_png(
            lane_path,
            contact_sheet(
                [{"label": "LANE OFFSET SWEEP, LEFT (WHITE EDGE) TO RIGHT (YELLOW)", "tiles": lane_tiles}],
                "LANE-FRAME GROUND TRUTH: COMMANDED OFFSET VS RENDERED VIEW",
            ),
        )
        lane["sweep_sheet"] = lane_path.as_posix()
        report["lane_check"] = lane
        print(f"[check_obs] lane sweep sheet written to {lane_path.as_posix()}")

    checks = {
        "no all-black observation": all(max(s["max"]) > 0 for s in stats),
        "no channel fully saturated": aggregate["frac_255"] < 0.5,
        "frame ring advancing": all(s["ring_advancing"] for s in stats),
        "yellow centre line visible somewhere": aggregate["yellow_detection_rate"] > 0.0,
    }
    if args.lane_check:
        checks["lane round trip within tolerance"] = bool(report["lane_check"]["passed"])
    report["checks"] = checks

    print_report(report)

    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[check_obs] report written to {json_path.as_posix()}")

    env.close()
    return 0 if all(checks.values()) else 1


_PARSER = build_parser()

if __name__ == "__main__":
    _args, _hydra = _PARSER.parse_known_args()
    # A vision check without cameras is not a check; force the flag rather than fail obscurely
    # three minutes into a Kit boot.
    _args.enable_cameras = True
    _app_launcher = AppLauncher(_args)
    _simulation_app = _app_launcher.app

    _code = main()
    _simulation_app.close()
    raise SystemExit(_code)
