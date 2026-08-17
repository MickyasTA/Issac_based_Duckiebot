"""render: valid GIF and PNG on disk, no temporary files left behind, no hard dependencies.

The PNG path is pure numpy and must always work. The video paths are optional, so they are
skipped, never failed, when their encoder is absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from duckiebot_rl.dr.preprocess import FRAME_STACK_OFFSETS, OBS_CHANNELS, OBS_H, OBS_W
from duckiebot_rl.viz.render import (
    GIF_NAME,
    MP4_NAME,
    OBS_NAME,
    annotate_frame,
    available_encoders,
    denormalize_stack,
    draw_text,
    observation_panel,
    split_stack,
    text_width,
    write_gif,
    write_obs_snapshot,
    write_rollout,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
GIF_MAGIC = (b"GIF87a", b"GIF89a")


def make_frames(count: int = 6, height: int = 32, width: int = 48) -> list[np.ndarray]:
    """Return deterministic RGB frames that differ from one another."""
    rng = np.random.default_rng(0)
    base = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    return [np.clip(base.astype(np.int16) + 10 * i, 0, 255).astype(np.uint8) for i in range(count)]


def make_stack(height: int = OBS_H, width: int = OBS_W) -> np.ndarray:
    """Return a stacked observation shaped like the real one."""
    rng = np.random.default_rng(1)
    return rng.integers(0, 256, (height, width, OBS_CHANNELS), dtype=np.uint8)


def assert_no_temporaries(directory: Path) -> None:
    """Assert no ``.tmp`` file survived anywhere under ``directory``."""
    leftovers = sorted(p.as_posix() for p in directory.rglob("*") if ".tmp" in p.name)
    assert leftovers == [], f"temporary files left behind: {leftovers}"


# ------------------------------------------------------------------------------------------ gif


def test_write_gif_produces_a_valid_gif_and_no_temporary(tmp_path: Path):
    if not (available_encoders()["pillow"] or available_encoders()["imageio"]):
        pytest.skip("neither Pillow nor imageio is installed")

    path = write_gif(tmp_path / "videos" / GIF_NAME, make_frames(), fps=10)

    assert path.is_file()
    assert path.read_bytes()[:6] in GIF_MAGIC
    assert path.stat().st_size > 100
    assert_no_temporaries(tmp_path)


def test_write_gif_accepts_float_frames(tmp_path: Path):
    if not (available_encoders()["pillow"] or available_encoders()["imageio"]):
        pytest.skip("neither Pillow nor imageio is installed")
    frames = [frame.astype(np.float32) / 255.0 for frame in make_frames(3)]
    path = write_gif(tmp_path / "f.gif", frames)
    assert path.read_bytes()[:6] in GIF_MAGIC


def test_write_gif_rejects_ragged_frames(tmp_path: Path):
    frames = [np.zeros((8, 8, 3), np.uint8), np.zeros((9, 8, 3), np.uint8)]
    with pytest.raises(ValueError, match="share one shape"):
        write_gif(tmp_path / "bad.gif", frames)
    assert_no_temporaries(tmp_path)


def test_write_gif_rejects_an_empty_sequence(tmp_path: Path):
    with pytest.raises(ValueError, match="no frames"):
        write_gif(tmp_path / "empty.gif", [])


# ------------------------------------------------------------------------------------------ mp4


def test_write_mp4_when_ffmpeg_is_available(tmp_path: Path):
    encoders = available_encoders()
    if not (encoders["imageio"] and encoders["imageio_ffmpeg"]):
        pytest.skip("imageio-ffmpeg is not installed")
    from duckiebot_rl.viz.render import write_mp4

    # An odd width exercises the even-dimension padding H.264 requires.
    path = write_mp4(tmp_path / "videos" / MP4_NAME, make_frames(width=49), fps=10)

    assert path.is_file()
    assert path.stat().st_size > 0
    assert b"ftyp" in path.read_bytes()[:64]
    assert_no_temporaries(tmp_path)


# -------------------------------------------------------------------------- observation snapshot


def test_write_obs_snapshot_produces_a_valid_png_and_no_temporary(tmp_path: Path):
    path = write_obs_snapshot(tmp_path / "obs" / OBS_NAME, make_stack(), title="ITERATION 12")

    assert path.is_file()
    assert path.read_bytes()[: len(PNG_MAGIC)] == PNG_MAGIC
    assert_no_temporaries(tmp_path)


def test_observation_panel_tiles_every_stacked_frame():
    stack = make_stack()
    panel = observation_panel(stack, scale=2, gap=4)

    assert panel.ndim == 3
    assert panel.shape[2] == 3
    assert panel.dtype == np.uint8
    # Three tiles of 2x the observation width, plus four gaps.
    assert panel.shape[1] == 3 * (OBS_W * 2) + 4 * 4
    assert panel.shape[0] > OBS_H * 2


def test_observation_panel_shows_the_actual_pixels():
    stack = np.zeros((OBS_H, OBS_W, OBS_CHANNELS), dtype=np.uint8)
    stack[..., 0:3] = (200, 30, 40)
    panel = observation_panel(stack, scale=1, gap=2, title="")
    assert np.any(np.all(panel == (200, 30, 40), axis=-1)), "the newest frame must appear in the panel"


def test_split_stack_returns_one_frame_per_offset():
    frames = split_stack(make_stack())
    assert len(frames) == len(FRAME_STACK_OFFSETS)
    assert all(frame.shape == (OBS_H, OBS_W, 3) for frame in frames)


def test_split_stack_rejects_a_bad_channel_count():
    with pytest.raises(ValueError, match="cannot split"):
        split_stack(np.zeros((4, 4, 8), np.uint8))


@pytest.mark.parametrize(
    ("array", "expected_peak"),
    [
        (np.full((4, 4, 3), 255, np.uint8), 255),
        (np.ones((4, 4, 3), np.float32), 255),
        (np.full((4, 4, 3), 255.0, np.float32), 255),
    ],
)
def test_denormalize_stack_recovers_uint8_from_any_scale(array, expected_peak):
    out = denormalize_stack(array)
    assert out.dtype == np.uint8
    assert out.max() == expected_peak


def test_denormalize_stack_rejects_a_non_image():
    with pytest.raises(ValueError, match="H, W, C"):
        denormalize_stack(np.zeros((4, 4), np.uint8))


# ------------------------------------------------------------------------------------ text drawing


def test_draw_text_lights_pixels_and_stays_in_bounds():
    canvas = np.zeros((16, 80, 3), np.uint8)
    draw_text(canvas, 2, 2, "T-2", color=(255, 255, 255))
    assert canvas.sum() > 0
    assert canvas.shape == (16, 80, 3)


def test_draw_text_clips_instead_of_raising():
    canvas = np.zeros((10, 10, 3), np.uint8)
    draw_text(canvas, -20, -20, "OFFSCREEN")
    draw_text(canvas, 200, 200, "OFFSCREEN")
    draw_text(canvas, 2, 2, "unknown glyph: é")  # no glyph, drawn as a blank cell


def test_draw_text_rejects_a_bad_canvas():
    with pytest.raises(ValueError, match="H, W, 3"):
        draw_text(np.zeros((4, 4), np.uint8), 0, 0, "X")


def test_text_width_matches_the_drawn_extent():
    assert text_width("ABC", scale=2) == 3 * 6 * 2


def test_annotate_frame_does_not_modify_its_input():
    frame = np.full((40, 90, 3), 120, np.uint8)
    original = frame.copy()
    out = annotate_frame(frame, ["EP 1 STEP 7", "ITER 42"])
    assert out.shape == frame.shape
    np.testing.assert_array_equal(frame, original)
    assert not np.array_equal(out, original)


# ------------------------------------------------------------------------------- write_rollout


def test_write_rollout_writes_into_the_contract_paths(tmp_path: Path):
    result = write_rollout(tmp_path, make_frames(4), fps=10, stacked_obs=make_stack(), obs_title="T")

    assert result.frames == 4
    assert result.obs == tmp_path / "obs" / OBS_NAME
    assert result.obs.is_file()
    if result.gif is not None:
        assert result.gif == tmp_path / "videos" / GIF_NAME
        assert result.gif.read_bytes()[:6] in GIF_MAGIC
    if result.mp4 is not None:
        assert result.mp4 == tmp_path / "videos" / MP4_NAME
    assert "wrote" in result.message
    assert_no_temporaries(tmp_path)


def test_write_rollout_degrades_instead_of_raising_when_there_are_no_frames(tmp_path: Path):
    result = write_rollout(tmp_path, [], stacked_obs=make_stack())

    assert result.mp4 is None
    assert result.gif is None
    assert result.obs is not None and result.obs.is_file()
    assert any("no frames" in warning for warning in result.warnings)
    assert_no_temporaries(tmp_path)


def test_write_rollout_reports_an_unusable_observation_without_raising(tmp_path: Path):
    result = write_rollout(tmp_path, make_frames(2), stacked_obs=np.zeros((4, 4, 8), np.uint8))

    assert result.obs is None
    assert any("obs snapshot skipped" in warning for warning in result.warnings)
    assert_no_temporaries(tmp_path)
