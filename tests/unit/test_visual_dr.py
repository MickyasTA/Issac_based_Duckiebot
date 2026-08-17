"""Unit tests for the per-step photometric domain randomization (SPEC v2 S7.2).

Covers: shape/dtype/range preservation for every augmentation, per-env independence, exact
identity at zero curriculum strength, determinism under a seed, and the scene-side / camera-mount
range books.
"""

from __future__ import annotations

import math

import pytest
import torch

from duckiebot_rl.dr import visual as V
from duckiebot_rl.dr.curriculum import Range

N = 6
H, W = 128, 192


def _img(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.rand((N, 3, H, W), generator=g)
    x[:, :, :, 90:92] = 1.0
    x[:, :, 60:90, 20:60] = 0.0
    return x


def _dr(seed: int = 0, **kw: object) -> V.VisualDR:
    return V.VisualDR(N, V.VisualDRCfg(**kw), generator=torch.Generator().manual_seed(seed))


def _ones(v: float = 1.0) -> torch.Tensor:
    return torch.full((N,), v)


def _zeros() -> torch.Tensor:
    return torch.zeros(N)


# ---------------------------------------------------------------------------- whole chain


def test_chain_preserves_shape_dtype_and_range():
    x = _img()
    dr = _dr(1)
    out, params = dr.randomize(x, alpha=1.0)
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    assert torch.isfinite(out).all()
    assert params.exposure_log2.shape == (N,)


def test_chain_is_identity_at_alpha_zero():
    x = _img(2)
    out, params = _dr(3).randomize(x, alpha=0.0)
    assert torch.allclose(out, x, atol=1e-6)
    # Every axis collapsed to its nominal value.
    assert torch.allclose(params.exposure_log2, _zeros())
    assert torch.allclose(params.gamma, _ones())
    assert torch.allclose(params.noise_sigma, _zeros())
    assert torch.allclose(params.blur_len_px, _zeros())
    assert torch.allclose(params.jpeg_quality, _ones(100.0))


def test_chain_is_not_identity_at_alpha_one():
    x = _img(4)
    out, _ = _dr(5).randomize(x, alpha=1.0)
    assert (out - x).abs().max() > 0.01


def test_per_env_independence_of_params_and_pixels():
    x = _img(6).expand(N, 3, H, W).clone()
    x[:] = x[0:1]  # every env sees the SAME image
    out, params = _dr(7).randomize(x, alpha=1.0)
    for name, t in params.as_dict().items():
        if name in ("frame_repeat_p",):
            continue
        assert t.float().std() > 0.0, f"{name} is identical across envs"
    for i in range(1, N):
        assert not torch.equal(out[0], out[i])


def test_determinism_under_seed():
    x = _img(8)
    a, pa = _dr(42).randomize(x, alpha=1.0)
    b, pb = _dr(42).randomize(x, alpha=1.0)
    assert torch.equal(a, b)
    assert torch.equal(pa.jpeg_quality, pb.jpeg_quality)
    c, _ = _dr(43).randomize(x, alpha=1.0)
    assert not torch.equal(a, c)


def test_curriculum_alpha_monotonically_widens_the_effect():
    x = _img(9)
    deltas = []
    for alpha in (0.0, 0.25, 1.0):
        out, _ = _dr(11).randomize(x, alpha=alpha)
        deltas.append(float((out - x).abs().mean()))
    assert deltas[0] < deltas[1] < deltas[2]


def test_operator_is_usable_as_the_preprocess_hook():
    from duckiebot_rl.dr import preprocess as pp

    frames = torch.randint(0, 256, (N, pp.RENDER_H, pp.RENDER_W, 3), dtype=torch.uint8)
    dr = _dr(12)
    out = pp.preprocess_frame(frames, photometric=dr.operator(dr.sample(alpha=1.0)))
    assert out.shape == (N, pp.OBS_H, pp.OBS_W, 3)
    assert out.dtype == torch.uint8


# ---------------------------------------------------------------------------- single operators


def test_color_identity_and_effects():
    x = _img(13)
    same = V.apply_color(x, _zeros(), _ones(), _ones(), _ones(), _ones(), _ones(), _zeros())
    assert torch.allclose(same, x, atol=1e-6)
    bright = V.apply_color(x, _ones(0.5), _ones(), _ones(), _ones(), _ones(), _ones(), _zeros())
    assert float(bright.mean()) > float(x.mean())
    gray = V.apply_color(x, _zeros(), _ones(), _ones(), _zeros(), _ones(), _ones(), _zeros())
    assert float((gray[:, 0] - gray[:, 1]).abs().max()) < 1e-5
    assert float(gray.min()) >= 0.0 and float(gray.max()) <= 1.0


def test_noise_identity_and_effects():
    x = _img(14)
    g = torch.Generator().manual_seed(0)
    assert torch.equal(V.apply_noise(x, _zeros(), _zeros(), generator=g), x)
    noisy = V.apply_noise(x, _ones(0.05), _zeros(), generator=g)
    assert noisy.shape == x.shape
    assert float(noisy.min()) >= 0.0 and float(noisy.max()) <= 1.0
    assert float((noisy - x).abs().mean()) > 0.0


def test_motion_blur_identity_and_effects():
    x = _img(15)
    assert torch.equal(V.apply_motion_blur(x, _zeros(), _zeros()), x)
    blurred = V.apply_motion_blur(x, _ones(8.0), _zeros())
    assert blurred.shape == x.shape

    # A horizontal blur must reduce horizontal gradient energy.
    def hgrad(t: torch.Tensor) -> float:
        return float((t[..., 1:] - t[..., :-1]).abs().mean())

    assert hgrad(blurred) < hgrad(x)


def test_motion_blur_rejects_even_taps():
    with pytest.raises(ValueError, match="odd integer"):
        V.apply_motion_blur(_img(), _ones(), _zeros(), taps=6)


def test_defocus_identity_and_effects():
    x = _img(16)
    assert torch.allclose(V.apply_defocus(x, _zeros()), x, atol=0.0)
    out = V.apply_defocus(x, _ones(1.2))
    assert out.shape == x.shape
    assert float((out - x).abs().mean()) > 0.0
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_defocus_is_per_env():
    x = _img(17)
    sigma = torch.tensor([0.0, 0.0, 0.5, 0.5, 1.2, 1.2])
    out = V.apply_defocus(x, sigma)
    assert torch.equal(out[0], x[0])
    assert not torch.equal(out[4], x[4])


def test_vignette_identity_and_effects():
    x = torch.ones((N, 3, H, W))
    assert torch.equal(V.apply_vignette(x, _zeros()), x)
    out = V.apply_vignette(x, _ones(0.35))
    assert float(out[0, 0, H // 2, W // 2]) == pytest.approx(1.0, abs=1e-3)
    assert float(out[0, 0, 0, 0]) == pytest.approx(1.0 - 0.35, abs=1e-3)
    assert float(out.min()) >= 0.0


def test_chromatic_aberration_identity_and_effects():
    x = _img(18)
    assert torch.equal(V.apply_chromatic_aberration(x, _zeros()), x)
    out = V.apply_chromatic_aberration(x, _ones(1.5))
    assert torch.equal(out[:, 1], x[:, 1])  # green is the reference channel
    assert not torch.equal(out[:, 0], x[:, 0])


def test_lens_distortion_identity_and_effects():
    x = _img(19)
    assert torch.equal(V.apply_lens_distortion(x, _zeros()), x)
    out = V.apply_lens_distortion(x, _ones(0.06))
    assert out.shape == x.shape
    assert not torch.equal(out, x)


def test_jpeg_identity_at_quality_100_and_artifacts_below():
    x = _img(20)
    assert torch.equal(V.apply_jpeg(x, _ones(100.0)), x)
    out = V.apply_jpeg(x, _ones(30.0))
    assert out.shape == x.shape
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    assert float((out - x).abs().mean()) > 1e-3
    # Higher quality must be closer to the original than lower quality.
    hi = V.apply_jpeg(x, _ones(95.0))
    assert float((hi - x).abs().mean()) < float((out - x).abs().mean())


def test_jpeg_rejects_non_multiple_of_eight():
    with pytest.raises(ValueError, match="divisible by 8"):
        V.apply_jpeg(torch.rand((1, 3, 10, 16)), torch.tensor([50.0]))


def test_operators_reject_bad_images():
    with pytest.raises(ValueError, match="C=3"):
        V.apply_vignette(torch.rand((1, 1, 8, 8)), torch.zeros(1))
    with pytest.raises(ValueError, match="float"):
        V.apply_vignette(torch.zeros((1, 3, 8, 8), dtype=torch.uint8), torch.zeros(1))


def test_frame_repeat_probabilities():
    g = torch.Generator().manual_seed(0)
    assert not V.sample_frame_repeat(torch.zeros(1000), generator=g).any()
    assert V.sample_frame_repeat(torch.ones(1000), generator=g).all()
    m = V.sample_frame_repeat(torch.full((20000,), 0.10), generator=g)
    assert 0.08 < float(m.float().mean()) < 0.12


def test_motion_blur_length_scales_with_speed():
    dr = _dr(21)
    slow = dr.sample(alpha=1.0, speed_frac=torch.zeros(N))
    assert torch.allclose(slow.blur_len_px, _zeros())
    fast = _dr(21).sample(alpha=1.0, speed_frac=torch.ones(N))
    assert float(fast.blur_len_px.max()) > 0.0


def test_disabling_ops_short_circuits_them():
    x = _img(22)
    out, _ = _dr(23, enable_jpeg=False, enable_motion_blur=False).randomize(x, alpha=1.0)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------- range books


def test_visual_range_book_covers_the_s7_2_step_axes():
    book = V.default_visual_ranges()
    for key in (
        "exposure_log2",
        "gamma",
        "contrast",
        "saturation",
        "wb_r",
        "wb_b",
        "noise_sigma",
        "shot_scale",
        "blur_len_px",
        "jpeg_quality",
        "vignette",
        "ca_px",
        "defocus_sigma_px",
        "distort_k1",
        "frame_repeat_p",
    ):
        assert isinstance(book[key], Range)


def test_scene_params_respect_ranges_and_collapse_to_nominal():
    g = torch.Generator().manual_seed(0)
    book = V.default_scene_ranges()
    full = V.sample_scene_params(512, 1.0, generator=g)
    for name, rng in book.items():
        t = full[name]
        assert float(t.min()) >= rng.lo - 1e-5, name
        assert float(t.max()) <= rng.hi + 1e-5, name
    nominal = V.sample_scene_params(8, 0.0, generator=g)
    for name, rng in book.items():
        assert torch.allclose(nominal[name].float(), torch.full_like(nominal[name].float(), rng.nominal))
    assert full["lamp_offset_m"].shape == (512, 2)


def test_camera_mount_matches_spec_s2_nominal():
    g = torch.Generator().manual_seed(0)
    m = V.sample_camera_mount(4, 0.0, generator=g)
    assert torch.allclose(m["height_m"], torch.full((4,), 0.101))
    assert torch.allclose(m["pitch_down_rad"], torch.full((4,), 25.3 * math.pi / 180.0))
    assert torch.allclose(m["forward_m"], torch.full((4,), 0.078))
    assert torch.allclose(m["base_z_m"], torch.full((4,), 0.101 - 0.0318), atol=1e-7)
    wide = V.sample_camera_mount(256, 1.0, generator=g)
    assert float(wide["height_m"].min()) >= 0.090 - 1e-6
    assert float(wide["height_m"].max()) <= 0.120 + 1e-6
    assert float(wide["pitch_down_rad"].min()) >= 15.0 * math.pi / 180.0 - 1e-6
    assert float(wide["pitch_down_rad"].max()) <= 28.0 * math.pi / 180.0 + 1e-6
