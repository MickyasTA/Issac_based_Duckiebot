"""Frame capture, video encoding, and the "what the policy sees" observation snapshot.

Three artefacts, all written into the run directory defined by the run-directory contract:

``video/latest_rollout.mp4``
    The rollout, encoded with ``imageio`` plus ``imageio-ffmpeg``.
``video/latest_rollout.gif``
    The same rollout, loopable and embeddable straight into a README.
``obs/latest_obs.png``
    The stacked observation the policy actually consumed, de-normalised, split back into its
    three frames, tiled left to right and labelled with each frame's time offset.

Every one of them is encoded to a sibling temporary (``latest_rollout.tmp.mp4``) and then
moved into place with
:func:`~duckiebot_rl.viz.watcher.atomic_replace`, because a dashboard or a browser reading
``latest_rollout.gif`` while it is being written would otherwise see a truncated file. Encoders
are handed the temporary path, never the final one, so a crashed encode leaves no half-written
artefact under a name a reader looks for, and the temporary is removed on failure.

Why the observation snapshot matters
------------------------------------
A vision policy that will not learn is, far more often than not, looking at something other than
what its author believes. The observation is a ``(48, 96, 9)`` uint8 stack: three RGB frames at
``t``, ``t-2`` and ``t-4``, produced by the shared S4.3 chain in
:mod:`duckiebot_rl.dr.preprocess`. Bugs that are invisible in a reward curve and obvious in this
one PNG include a stale frame ring (three identical panels), a channel-order slip (blue lane
markings), an inverted crop (sky instead of road), and photometric randomisation cranked so far
that the yellow dashes have vanished.

Dependencies, and what happens without them
-------------------------------------------
The PNG path needs nothing beyond numpy: it goes through
:func:`duckiebot_rl.sim2sim.track.write_png`, a standard-library PNG writer, and text labels are
drawn with a 5x7 bitmap font defined in this module. So the observation snapshot works in a bare
tools venv. Video is different: MP4 needs ``imageio`` and ``imageio-ffmpeg``, GIF needs
``imageio`` or ``Pillow``. :func:`write_rollout` degrades in that order and reports what it did in
:class:`RolloutArtifacts.message` rather than failing the viewer.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from duckiebot_rl.dr.preprocess import FRAME_STACK_OFFSETS
from duckiebot_rl.sim2sim.track import write_png
from duckiebot_rl.viz.watcher import atomic_replace

__all__ = [
    "DEFAULT_FPS",
    "GIF_NAME",
    "MP4_NAME",
    "OBS_NAME",
    "RolloutArtifacts",
    "available_encoders",
    "denormalize_stack",
    "draw_text",
    "observation_panel",
    "split_stack",
    "write_gif",
    "write_mp4",
    "write_obs_snapshot",
    "write_rollout",
]

DEFAULT_FPS = 20
"""Frames per second for both the MP4 and the GIF. Close to the 15 Hz control rate."""

MP4_NAME = "latest_rollout.mp4"
"""File name of the rollout video inside ``<run_dir>/video``."""

GIF_NAME = "latest_rollout.gif"
"""File name of the loopable rollout inside ``<run_dir>/video``."""

OBS_NAME = "latest_obs.png"
"""File name of the observation snapshot inside ``<run_dir>/obs``."""

_MP4_HINT = (
    "MP4 encoding needs imageio and imageio-ffmpeg. Install them into the interpreter running "
    "the viewer, e.g.\n  "
    "d:/Personal/personal/mujoco_venv/Scripts/python.exe -m pip install imageio imageio-ffmpeg"
)
_GIF_HINT = (
    "GIF encoding needs imageio or Pillow. Install one into the interpreter running the viewer, "
    "e.g.\n  d:/Personal/personal/mujoco_venv/Scripts/python.exe -m pip install imageio"
)

# --------------------------------------------------------------------------------------------
# 5x7 bitmap font
#
# Five column bytes per glyph, bit 0 is the TOP row, bit 6 the bottom. This is the classic 5x7
# ASCII cell. It lives here rather than in Pillow so that the observation snapshot, which is the
# single most useful debugging artefact this module produces, has no optional dependency at all.
# --------------------------------------------------------------------------------------------
_FONT: dict[str, tuple[int, int, int, int, int]] = {
    " ": (0x00, 0x00, 0x00, 0x00, 0x00),
    "-": (0x08, 0x08, 0x08, 0x08, 0x08),
    "+": (0x08, 0x08, 0x3E, 0x08, 0x08),
    ".": (0x00, 0x00, 0x40, 0x00, 0x00),
    ",": (0x00, 0x50, 0x30, 0x00, 0x00),
    ":": (0x00, 0x36, 0x36, 0x00, 0x00),
    "=": (0x14, 0x14, 0x14, 0x14, 0x14),
    "/": (0x20, 0x10, 0x08, 0x04, 0x02),
    "_": (0x40, 0x40, 0x40, 0x40, 0x40),
    "(": (0x00, 0x1C, 0x22, 0x41, 0x00),
    ")": (0x00, 0x41, 0x22, 0x1C, 0x00),
    "%": (0x23, 0x13, 0x08, 0x64, 0x62),
    "#": (0x14, 0x7F, 0x14, 0x7F, 0x14),
    "0": (0x3E, 0x51, 0x49, 0x45, 0x3E),
    "1": (0x00, 0x42, 0x7F, 0x40, 0x00),
    "2": (0x42, 0x61, 0x51, 0x49, 0x46),
    "3": (0x21, 0x41, 0x45, 0x4B, 0x31),
    "4": (0x18, 0x14, 0x12, 0x7F, 0x10),
    "5": (0x27, 0x45, 0x45, 0x45, 0x39),
    "6": (0x3C, 0x4A, 0x49, 0x49, 0x30),
    "7": (0x01, 0x71, 0x09, 0x05, 0x03),
    "8": (0x36, 0x49, 0x49, 0x49, 0x36),
    "9": (0x06, 0x49, 0x49, 0x29, 0x1E),
    "A": (0x7E, 0x11, 0x11, 0x11, 0x7E),
    "B": (0x7F, 0x49, 0x49, 0x49, 0x36),
    "C": (0x3E, 0x41, 0x41, 0x41, 0x22),
    "D": (0x7F, 0x41, 0x41, 0x22, 0x1C),
    "E": (0x7F, 0x49, 0x49, 0x49, 0x41),
    "F": (0x7F, 0x09, 0x09, 0x09, 0x01),
    "G": (0x3E, 0x41, 0x49, 0x49, 0x7A),
    "H": (0x7F, 0x08, 0x08, 0x08, 0x7F),
    "I": (0x00, 0x41, 0x7F, 0x41, 0x00),
    "J": (0x20, 0x40, 0x41, 0x3F, 0x01),
    "K": (0x7F, 0x08, 0x14, 0x22, 0x41),
    "L": (0x7F, 0x40, 0x40, 0x40, 0x40),
    "M": (0x7F, 0x02, 0x0C, 0x02, 0x7F),
    "N": (0x7F, 0x04, 0x08, 0x10, 0x7F),
    "O": (0x3E, 0x41, 0x41, 0x41, 0x3E),
    "P": (0x7F, 0x09, 0x09, 0x09, 0x06),
    "Q": (0x3E, 0x41, 0x51, 0x21, 0x5E),
    "R": (0x7F, 0x09, 0x19, 0x29, 0x46),
    "S": (0x46, 0x49, 0x49, 0x49, 0x31),
    "T": (0x01, 0x01, 0x7F, 0x01, 0x01),
    "U": (0x3F, 0x40, 0x40, 0x40, 0x3F),
    "V": (0x1F, 0x20, 0x40, 0x20, 0x1F),
    "W": (0x3F, 0x40, 0x38, 0x40, 0x3F),
    "X": (0x63, 0x14, 0x08, 0x14, 0x63),
    "Y": (0x07, 0x08, 0x70, 0x08, 0x07),
    "Z": (0x61, 0x51, 0x49, 0x45, 0x43),
}
_GLYPH_W = 5
_GLYPH_H = 7
_GLYPH_ADVANCE = 6


def draw_text(
    canvas: np.ndarray,
    x: int,
    y: int,
    text: str,
    color: tuple[int, int, int] = (255, 255, 255),
    scale: int = 1,
) -> np.ndarray:
    """Draw ASCII text onto an RGB canvas in place with the built-in 5x7 font.

    Characters with no glyph are drawn as a blank cell rather than raising, so a label can carry
    arbitrary text without the caller sanitising it.

    Args:
        canvas: ``(H, W, 3)`` uint8 array, modified in place.
        x: Left edge of the first glyph, in pixels.
        y: Top edge of the text, in pixels.
        text: The string to draw. Lowercase letters are drawn as uppercase.
        color: RGB colour of the lit pixels.
        scale: Integer pixel magnification.

    Returns:
        The same canvas, for chaining.

    Raises:
        ValueError: If the canvas is not ``(H, W, 3)`` or ``scale`` is not positive.
    """
    if canvas.ndim != 3 or canvas.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) canvas, got shape {canvas.shape}")
    if scale < 1:
        raise ValueError(f"scale must be >= 1, got {scale}")
    height, width = canvas.shape[:2]
    rgb = np.asarray(color, dtype=np.uint8)
    cursor = int(x)
    for char in text.upper():
        glyph = _FONT.get(char)
        if glyph is not None:
            for col, bits in enumerate(glyph):
                for row in range(_GLYPH_H):
                    if not bits >> row & 1:
                        continue
                    px = cursor + col * scale
                    py = int(y) + row * scale
                    if px + scale <= 0 or py + scale <= 0 or px >= width or py >= height:
                        continue
                    canvas[max(py, 0) : py + scale, max(px, 0) : px + scale] = rgb
        cursor += _GLYPH_ADVANCE * scale
    return canvas


def text_width(text: str, scale: int = 1) -> int:
    """Return the pixel width :func:`draw_text` will occupy.

    Args:
        text: The string that would be drawn.
        scale: The same magnification that will be passed to :func:`draw_text`.

    Returns:
        Width in pixels, including the trailing inter-glyph gap of the last character.
    """
    return len(text) * _GLYPH_ADVANCE * scale


def available_encoders() -> dict[str, bool]:
    """Report which optional encoders this interpreter can use.

    Returns:
        Mapping with keys ``imageio``, ``imageio_ffmpeg`` and ``pillow``.
    """
    import importlib.util

    return {
        "imageio": importlib.util.find_spec("imageio") is not None,
        "imageio_ffmpeg": importlib.util.find_spec("imageio_ffmpeg") is not None,
        "pillow": importlib.util.find_spec("PIL") is not None,
    }


def _as_frame_list(frames: Iterable[np.ndarray]) -> list[np.ndarray]:
    """Normalise an iterable of frames to a list of ``(H, W, 3)`` uint8 arrays.

    Args:
        frames: Frames as uint8 in ``[0, 255]`` or float in ``[0, 1]``.

    Returns:
        The frames as uint8 RGB arrays.

    Raises:
        ValueError: If the sequence is empty, a frame is not ``(H, W, 3)``, or the frames differ
            in shape (which every encoder rejects, with a much worse message).
    """
    out: list[np.ndarray] = []
    for frame in frames:
        array = np.asarray(frame)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"expected (H, W, 3) RGB frames, got shape {array.shape}")
        if array.dtype != np.uint8:
            scale = 255.0 if float(np.nanmax(array, initial=0.0)) <= 1.0 else 1.0
            array = np.clip(array.astype(np.float32) * scale, 0, 255).astype(np.uint8)
        out.append(array)
    if not out:
        raise ValueError("no frames to encode")
    shapes = {frame.shape for frame in out}
    if len(shapes) != 1:
        raise ValueError(f"all frames must share one shape, got {sorted(shapes)}")
    return out


def _tmp_path(destination: Path) -> Path:
    """Return the temporary path an artefact is encoded into before being moved into place.

    The suffix is preserved (``latest_rollout.tmp.mp4``, not ``latest_rollout.mp4.tmp``) because
    ``imageio`` selects its backend from the file extension and refuses a ``.tmp`` URI outright.
    Readers looking for finished artefacts match the exact contract names, so an extra dotted
    component is never mistaken for one.

    Args:
        destination: The final artefact path.

    Returns:
        A sibling temporary path with the same suffix.
    """
    return destination.with_name(f"{destination.stem}.tmp{destination.suffix}")


def write_mp4(
    path: str | os.PathLike[str],
    frames: Sequence[np.ndarray],
    fps: int = DEFAULT_FPS,
    quality: int = 8,
) -> Path:
    """Encode frames to an MP4, written atomically.

    Args:
        path: Destination ``.mp4``. Parent directories are created.
        frames: RGB frames.
        fps: Frames per second.
        quality: ``imageio`` quality, 0 to 10.

    Returns:
        The destination path.

    Raises:
        RuntimeError: If ``imageio`` or ``imageio-ffmpeg`` is unavailable.
        ValueError: If the frames are unusable.
    """
    encoders = available_encoders()
    if not (encoders["imageio"] and encoders["imageio_ffmpeg"]):
        raise RuntimeError(_MP4_HINT)
    import imageio.v2 as imageio

    array = _as_frame_list(frames)
    # H.264 requires even dimensions; padding by a row/column is invisible and avoids a
    # confusing "width not divisible by 2" failure deep inside ffmpeg.
    array = [_pad_even(frame) for frame in array]

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(destination)
    try:
        # format is named explicitly so the encoder choice does not silently depend on how
        # _tmp_path() happens to spell the temporary file name.
        writer = imageio.get_writer(str(tmp), format="FFMPEG", fps=fps, quality=quality, macro_block_size=1)
        try:
            for frame in array:
                writer.append_data(frame)
        finally:
            writer.close()
    except Exception:
        with suppress(OSError):
            tmp.unlink()
        raise
    return atomic_replace(tmp, destination)


def _pad_even(frame: np.ndarray) -> np.ndarray:
    """Pad a frame to even height and width by replicating the last row and column.

    Args:
        frame: ``(H, W, 3)`` uint8 array.

    Returns:
        A frame with even dimensions; the input itself when it is already even.
    """
    height, width = frame.shape[:2]
    pad_h, pad_w = height % 2, width % 2
    if not (pad_h or pad_w):
        return frame
    return np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")


def write_gif(
    path: str | os.PathLike[str],
    frames: Sequence[np.ndarray],
    fps: int = DEFAULT_FPS,
    loop: int = 0,
) -> Path:
    """Encode frames to a looping GIF, written atomically.

    Args:
        path: Destination ``.gif``. Parent directories are created.
        frames: RGB frames.
        fps: Frames per second.
        loop: GIF loop count; 0 means loop forever.

    Returns:
        The destination path.

    Raises:
        RuntimeError: If neither ``Pillow`` nor ``imageio`` is available.
        ValueError: If the frames are unusable.

    Note:
        Pillow is tried first even though ``imageio`` is the preferred encoder elsewhere. The
        meaning of ``imageio``'s GIF ``duration`` argument changed between releases when its GIF
        backend moved onto Pillow, so the same call produces a 20x playback-speed difference
        depending on the installed version. Pillow's own ``duration`` is unambiguously
        milliseconds per frame, so going straight to it makes the GIF play at the requested rate
        on every machine.
    """
    encoders = available_encoders()
    if not (encoders["imageio"] or encoders["pillow"]):
        raise RuntimeError(_GIF_HINT)

    array = _as_frame_list(frames)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(destination)
    duration_ms = max(round(1000.0 / max(fps, 1)), 20)
    try:
        if encoders["pillow"]:
            from PIL import Image

            images = [Image.fromarray(frame) for frame in array]
            images[0].save(
                str(tmp),
                format="GIF",
                save_all=True,
                append_images=images[1:],
                duration=duration_ms,
                loop=loop,
            )
        else:
            import imageio.v2 as imageio

            imageio.mimsave(str(tmp), array, format="GIF", duration=duration_ms / 1000.0, loop=loop)
    except Exception:
        with suppress(OSError):
            tmp.unlink()
        raise
    return atomic_replace(tmp, destination)


@dataclass
class RolloutArtifacts:
    """What :func:`write_rollout` managed to produce.

    Attributes:
        mp4: Path of the written MP4, or None.
        gif: Path of the written GIF, or None.
        obs: Path of the written observation snapshot, or None.
        frames: Number of frames encoded.
        message: Human-readable summary, including the reason for anything that was skipped.
        warnings: One entry per artefact that could not be produced.
    """

    mp4: Path | None = None
    gif: Path | None = None
    obs: Path | None = None
    frames: int = 0
    message: str = ""
    warnings: list[str] = field(default_factory=list)


def write_rollout(
    run_dir: str | os.PathLike[str],
    frames: Sequence[np.ndarray],
    fps: int = DEFAULT_FPS,
    stacked_obs: np.ndarray | None = None,
    obs_title: str = "",
) -> RolloutArtifacts:
    """Write ``video/latest_rollout.{mp4,gif}`` and optionally ``obs/latest_obs.png``.

    Neither encoder is mandatory. If MP4 is impossible the GIF is still written, and the reason is
    recorded in the result rather than raised, because losing a visualisation must never take down
    a viewer that is otherwise working.

    Args:
        run_dir: The run directory, ``runs/<run_id>``.
        frames: RGB frames of the rollout, as rendered by the backend.
        fps: Frames per second for both encoders.
        stacked_obs: The final stacked observation, ``(H, W, 9)``, or None to skip the snapshot.
        obs_title: Title drawn across the top of the observation snapshot.

    Returns:
        A :class:`RolloutArtifacts` describing what was written.
    """
    root = Path(run_dir)
    result = RolloutArtifacts(frames=len(frames))
    video_dir = root / "videos"

    if not frames:
        result.warnings.append("no frames captured; nothing to encode")
    else:
        try:
            result.mp4 = write_mp4(video_dir / MP4_NAME, frames, fps=fps)
        except (RuntimeError, ValueError, OSError) as exc:
            result.warnings.append(f"mp4 skipped: {exc}")
        try:
            result.gif = write_gif(video_dir / GIF_NAME, frames, fps=fps)
        except (RuntimeError, ValueError, OSError) as exc:
            result.warnings.append(f"gif skipped: {exc}")

    if stacked_obs is not None:
        try:
            result.obs = write_obs_snapshot(root / "obs" / OBS_NAME, stacked_obs, title=obs_title)
        except (ValueError, OSError) as exc:
            result.warnings.append(f"obs snapshot skipped: {exc}")

    written = [str(p) for p in (result.mp4, result.gif, result.obs) if p is not None]
    result.message = (
        f"wrote {len(written)} artefact(s) from {result.frames} frames: " + ", ".join(written)
        if written
        else "wrote nothing"
    )
    if result.warnings:
        result.message += " | " + " | ".join(result.warnings)
    return result


def denormalize_stack(stacked: np.ndarray) -> np.ndarray:
    """Return a stacked observation as viewable uint8, whatever scale it arrives in.

    The S4.3 chain ends at uint8, so the common case is the identity. Callers who captured the
    observation further down the pipeline may hand over float data in ``[0, 1]`` (post ``/255``
    inside the encoder) or in ``[0, 255]``; both are recovered here so the snapshot always shows
    the image as the encoder would see it.

    Args:
        stacked: ``(H, W, C)`` array, uint8 or float.

    Returns:
        A ``(H, W, C)`` uint8 array.

    Raises:
        ValueError: If the array is not 3-dimensional.
    """
    array = np.asarray(stacked)
    if array.ndim != 3:
        raise ValueError(f"expected a (H, W, C) stacked observation, got shape {array.shape}")
    if array.dtype == np.uint8:
        return array
    values = array.astype(np.float32)
    peak = float(np.nanmax(np.abs(values), initial=0.0))
    scale = 255.0 if peak <= 1.0 else 1.0
    return np.clip(values * scale, 0, 255).astype(np.uint8)


def split_stack(stacked: np.ndarray, channels: int = 3) -> list[np.ndarray]:
    """Split a channel-concatenated observation back into its individual frames.

    Args:
        stacked: ``(H, W, C)`` array with ``C`` a multiple of ``channels``.
        channels: Channels per frame, 3 for RGB.

    Returns:
        A list of ``(H, W, channels)`` arrays, newest first, matching the S4.3 step 10 layout.

    Raises:
        ValueError: If the channel count is not a multiple of ``channels``.
    """
    array = np.asarray(stacked)
    if array.ndim != 3 or array.shape[2] % channels:
        raise ValueError(f"cannot split shape {array.shape} into frames of {channels} channels")
    count = array.shape[2] // channels
    return [array[:, :, i * channels : (i + 1) * channels] for i in range(count)]


def observation_panel(
    stacked: np.ndarray,
    offsets: Sequence[int] = FRAME_STACK_OFFSETS,
    scale: int = 4,
    title: str = "",
    gap: int = 8,
) -> np.ndarray:
    """Build the tiled, labelled image of what the policy sees.

    Args:
        stacked: The stacked observation, ``(H, W, 3 * n)``, uint8 or float.
        offsets: Control-step offsets of the stacked frames, newest first. Used only for labels.
        scale: Integer magnification; a ``96x48`` frame is unreadable at native size.
        title: Optional title drawn across the top.
        gap: Pixels of background between tiles.

    Returns:
        An ``(H, W, 3)`` uint8 RGB image.

    Raises:
        ValueError: If ``scale`` or ``gap`` is negative, or the stack cannot be split.
    """
    if scale < 1:
        raise ValueError(f"scale must be >= 1, got {scale}")
    if gap < 0:
        raise ValueError(f"gap must be >= 0, got {gap}")

    frames = split_stack(denormalize_stack(stacked))
    tiles = [np.repeat(np.repeat(frame, scale, axis=0), scale, axis=1) for frame in frames]
    tile_h, tile_w = tiles[0].shape[:2]

    label_h = _GLYPH_H * 2 + 6
    title_h = (_GLYPH_H * 2 + 8) if title else 0
    width = len(tiles) * tile_w + (len(tiles) + 1) * gap
    height = title_h + label_h + tile_h + 2 * gap

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:, :] = (18, 18, 22)

    if title:
        draw_text(canvas, gap, gap // 2 + 2, title, color=(235, 235, 240), scale=2)

    for index, tile in enumerate(tiles):
        left = gap + index * (tile_w + gap)
        top = title_h + label_h
        offset = offsets[index] if index < len(offsets) else index
        label = "T" if offset == 0 else f"T-{offset}"
        draw_text(
            canvas,
            left,
            title_h + 3,
            f"{label}  {tile_w // scale}X{tile_h // scale}",
            color=(250, 210, 90),
            scale=2,
        )
        canvas[top : top + tile_h, left : left + tile_w] = tile
        # A one-pixel frame around each tile so a black observation is still visibly a tile.
        canvas[top - 1, left - 1 : left + tile_w + 1] = (90, 90, 100)
        canvas[top + tile_h, left - 1 : left + tile_w + 1] = (90, 90, 100)
        canvas[top - 1 : top + tile_h + 1, left - 1] = (90, 90, 100)
        canvas[top - 1 : top + tile_h + 1, left + tile_w] = (90, 90, 100)
    return canvas


def write_obs_snapshot(
    path: str | os.PathLike[str],
    stacked: np.ndarray,
    offsets: Sequence[int] = FRAME_STACK_OFFSETS,
    scale: int = 4,
    title: str = "",
) -> Path:
    """Write the observation panel to a PNG, atomically.

    Args:
        path: Destination ``.png``. Parent directories are created.
        stacked: The stacked observation, ``(H, W, 3 * n)``.
        offsets: Control-step offsets of the stacked frames, newest first.
        scale: Integer magnification.
        title: Optional title drawn across the top.

    Returns:
        The destination path.

    Raises:
        ValueError: If the observation cannot be tiled.
    """
    panel = observation_panel(stacked, offsets=offsets, scale=scale, title=title)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(destination)
    try:
        write_png(tmp, panel)
    except Exception:
        with suppress(OSError):
            tmp.unlink()
        raise
    return atomic_replace(tmp, destination)


def annotate_frame(frame: np.ndarray, lines: Sequence[str], scale: int = 2) -> np.ndarray:
    """Return a copy of ``frame`` with status lines burned into its top-left corner.

    Used by the viewer so the recorded video carries the checkpoint iteration it was produced
    from, which is what makes a saved rollout self-describing.

    Args:
        frame: ``(H, W, 3)`` uint8 RGB frame.
        lines: Text lines to draw, top to bottom.
        scale: Font magnification.

    Returns:
        A new annotated frame; the input is not modified.

    Raises:
        ValueError: If the frame is not ``(H, W, 3)``.
    """
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) frame, got shape {array.shape}")
    canvas = np.ascontiguousarray(array.astype(np.uint8).copy())
    step = _GLYPH_H * scale + 3
    band = 4 + step * len(lines)
    if band < canvas.shape[0]:
        strip = canvas[:band].astype(np.float32) * 0.35
        canvas[:band] = strip.astype(np.uint8)
    for row, line in enumerate(lines):
        draw_text(canvas, 4, 3 + row * step, line, color=(255, 255, 255), scale=scale)
    return canvas


def capture_frames(source: Any, count: int) -> list[np.ndarray]:  # pragma: no cover - convenience
    """Pull ``count`` frames from any object exposing ``render_frame()``.

    Args:
        source: An object with a ``render_frame()`` method returning ``(H, W, 3)`` uint8.
        count: Number of frames to pull.

    Returns:
        The captured frames.
    """
    return [np.asarray(source.render_frame()) for _ in range(int(count))]
