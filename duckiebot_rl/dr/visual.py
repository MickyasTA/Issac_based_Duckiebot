"""Per-step, per-env photometric domain randomization (SPEC v2 S7.2, layer 1).

Everything here is pure torch and runs *inside* the preprocessing chain (S4.3 step 3), i.e. on
the observation tensor at RENDER resolution, before the fixed anti-alias blur and the box
downsample. It never runs inside the PPO loss (augmenting there corrupts the importance ratio,
S7.1) and it never touches USD.

Axes implemented here (S7.2):

===== ============================================= ==================================
Axis   Parameter                                     Function
===== ============================================= ==================================
V10    Camera mount pose                             :func:`sample_camera_mount`
V11    Principal point                               ``dr.preprocess.shift_principal_point``
V12    Residual distortion (barrel/pincushion k1)    :func:`apply_lens_distortion`
V14    Exposure / gamma / contrast / saturation / WB :func:`apply_color`
V15    Sensor noise (gaussian + shot)                :func:`apply_noise`
V16    Motion blur (directional, speed scaled)       :func:`apply_motion_blur`
V17    JPEG artifacts (8x8 DCT quantization)         :func:`apply_jpeg`
V18    Vignette / chromatic aberration / defocus     :func:`apply_vignette` + friends
V19    Frame repeat                                  :func:`sample_frame_repeat`
===== ============================================= ==================================

The scene-side axes V1-V9 and V13 (lights, materials, walls, distractors) cannot be applied to a
tensor: they are written into USD by the Isaac event terms in ``events_visual.py``. Their
*ranges* still live here, in :func:`default_scene_ranges` / :func:`sample_scene_params`, so that
one file owns every number of the S7.2 table and the curriculum scalar gates them all through
the same :class:`~duckiebot_rl.dr.curriculum.Range` rule.

Design rules honoured by every operator in this module:

* **Batched, no python loop over envs.** Every parameter is a ``(N,)`` tensor and every op is a
  single batched kernel (grouped convolutions and ``grid_sample`` do the per-env work).
* **Identity at zero curriculum strength.** At ``alpha = 0`` every axis collapses to its nominal
  (identity) value and the chain is a no-op to float tolerance; ops built on ``grid_sample`` are
  additionally hard-gated with ``torch.where`` so they are bit-exact identities.
* **Explicit generator.** No global RNG is ever touched.
* **Shape/dtype/range preserving.** In and out: NCHW float32 in [0, 1].

Measured cost (RTX 3080 Laptop, N = 256, render 192x128, fp32, torch 2.7 + cu128)
--------------------------------------------------------------------------------
The chain is memory-bound: it touches an N x 3 x 128 x 192 fp32 tensor (75 MB) once or twice
per operator. Per control step::

    sample params 2.3 ms   distortion 2.1   colour 4.5   motion blur (7 taps) 10.8
    vignette 0.8           chromatic 2.0    defocus 3.0  noise 3.6   JPEG 13.3
    apply() total 41 ms;  full step incl. preprocess tail + frame ring 51 ms

i.e. the photometric DR alone caps throughput at about 5.0k env-steps/s at N = 256, against the
4-10k env-steps/s SPEC v2 S5.6 expectation for the whole loop. Two knobs recover most of it:
``VisualDRCfg(enable_jpeg=False, motion_blur_taps=5)`` measures 34.8 ms (7.4k env-steps/s).
Report this to [env] when the M6 N-sweep is run; do NOT silently drop axes to hit a number.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from duckiebot_rl.dr.curriculum import Range, RangeBook

__all__ = [
    "VisualDR",
    "VisualDRCfg",
    "VisualParams",
    "apply_chromatic_aberration",
    "apply_color",
    "apply_defocus",
    "apply_jpeg",
    "apply_lens_distortion",
    "apply_motion_blur",
    "apply_noise",
    "apply_vignette",
    "default_scene_ranges",
    "default_visual_ranges",
    "sample_camera_mount",
    "sample_frame_repeat",
    "sample_scene_params",
]

_LUMA = (0.299, 0.587, 0.114)

# Small device-side constants are cached per (device, dtype). Rebuilding them from python
# tuples on every call costs a host-to-device copy per operator per control step, which at
# N = 256 and 15 Hz was a measurable fraction of the whole DR budget.
_CACHE: dict[tuple, torch.Tensor] = {}


def _cached(key: tuple, build: Callable[[], torch.Tensor]) -> torch.Tensor:
    """Return a cached read-only constant tensor, building it on first use.

    Args:
        key: Cache key; must include device and dtype.
        build: Zero-argument builder invoked on a miss.

    Returns:
        The cached tensor. Callers must treat it as immutable.
    """
    t = _CACHE.get(key)
    if t is None:
        t = build()
        _CACHE[key] = t
    return t


# Standard JPEG Annex-K quantization tables (quality 50 baseline).
_Q_LUMA: tuple[tuple[int, ...], ...] = (
    (16, 11, 10, 16, 24, 40, 51, 61),
    (12, 12, 14, 19, 26, 58, 60, 55),
    (14, 13, 16, 24, 40, 57, 69, 56),
    (14, 17, 22, 29, 51, 87, 80, 62),
    (18, 22, 37, 56, 68, 109, 103, 77),
    (24, 35, 55, 64, 81, 104, 113, 92),
    (49, 64, 78, 87, 103, 121, 120, 101),
    (72, 92, 95, 98, 112, 100, 103, 99),
)
_Q_CHROMA: tuple[tuple[int, ...], ...] = (
    (17, 18, 24, 47, 99, 99, 99, 99),
    (18, 21, 26, 66, 99, 99, 99, 99),
    (24, 26, 56, 99, 99, 99, 99, 99),
    (47, 66, 99, 99, 99, 99, 99, 99),
    (99, 99, 99, 99, 99, 99, 99, 99),
    (99, 99, 99, 99, 99, 99, 99, 99),
    (99, 99, 99, 99, 99, 99, 99, 99),
    (99, 99, 99, 99, 99, 99, 99, 99),
)


# --------------------------------------------------------------------------------------------
# Range books (every number is a SPEC v2 S7.2 table cell)
# --------------------------------------------------------------------------------------------


def default_visual_ranges() -> dict[str, Range]:
    """Return the per-step photometric DR axes of SPEC v2 S7.2.

    Returns:
        A range book keyed by parameter name. Each entry carries its identity (nominal) value so
        that the curriculum scalar ``alpha_vis`` interpolates the axis on and off.
    """
    return {
        # V14 exposure / gamma / contrast / saturation / white balance
        "exposure_log2": Range(-1.0, 1.0, 0.0, "linear", "log2 stops"),
        "gamma": Range(0.7, 1.5, 1.0, "linear", ""),
        "contrast": Range(0.7, 1.3, 1.0, "linear", ""),
        "saturation": Range(0.5, 1.5, 1.0, "linear", ""),
        "wb_r": Range(0.85, 1.15, 1.0, "linear", ""),
        "wb_b": Range(0.85, 1.15, 1.0, "linear", ""),
        # Hue is not a separate S7.2 row (V14 covers chroma through white balance and V7 covers
        # tape hue in the material terms). It is kept here as a small extra photometric axis
        # because the deployed camera's auto-white-balance drifts hue as well as channel gain.
        "hue_deg": Range(-10.0, 10.0, 0.0, "linear", "deg"),
        # V15 sensor noise. sigma is log-uniform in [0.5, 10] / 255 with identity 0, hence
        # log_from_zero: at alpha = 0 the axis is exactly off.
        "noise_sigma": Range(0.5 / 255.0, 10.0 / 255.0, 0.0, "log_from_zero", "intensity"),
        "shot_scale": Range(0.002, 0.05, 0.0, "log_from_zero", "intensity/sqrt(I)"),
        # V16 motion blur (length is scaled by the per-env speed fraction at sample time)
        "blur_len_px": Range(0.0, 12.0, 0.0, "linear", "render px"),
        "blur_angle_rad": Range(0.0, math.pi, 0.0, "linear", "rad"),
        # V17 JPEG. Identity is quality 100 (no quantization), which sits outside the S7.2
        # table range U(30, 95): at alpha = 1 the live range is exactly the table.
        "jpeg_quality": Range(30.0, 95.0, 100.0, "linear", "quality", nominal_outside=True),
        # V18 vignette / chromatic aberration / defocus
        "vignette": Range(0.0, 0.35, 0.0, "linear", "corner falloff"),
        "ca_px": Range(0.0, 1.5, 0.0, "linear", "render px at edge"),
        "defocus_sigma_px": Range(0.0, 1.2, 0.0, "linear", "render px"),
        # V12 residual distortion after rectification
        "distort_k1": Range(-0.06, 0.06, 0.0, "linear", ""),
        # V19 frame repeat probability
        "frame_repeat_p": Range(0.0, 0.10, 0.0, "linear", "probability"),
    }


def default_scene_ranges() -> dict[str, Range]:
    """Return the scene-side visual axes V1-V9 / V13 of SPEC v2 S7.2.

    These are consumed by the Isaac event terms (``events_visual.py``), which write OmniPBR
    scalars and light attributes. They live here so that the S7.2 table has exactly one home.

    Returns:
        A range book keyed by parameter name.
    """
    return {
        # V1 / V4 intensities
        "sun_intensity_scale": Range(0.3, 3.0, 1.0, "log", "x nominal"),
        "lamp_intensity": Range(50.0, 2000.0, 300.0, "log", "cd"),
        "lamp_radius_m": Range(0.02, 0.6, 0.1, "linear", "m"),
        "lamp_offset_m": Range(-2.0, 2.0, 0.0, "linear", "m"),
        # V2r sun direction
        "sun_elevation_deg": Range(25.0, 70.0, 50.0, "linear", "deg"),
        "sun_azimuth_deg": Range(0.0, 360.0, 180.0, "linear", "deg"),
        # V2b dome
        "dome_yaw_deg": Range(0.0, 360.0, 180.0, "linear", "deg"),
        "dome_index": Range(0.0, 5.0, 0.0, "int", "hdri index"),
        # V3 colour temperature
        "color_temp_k": Range(2700.0, 6500.0, 5000.0, "linear", "K"),
        # V6 road albedo / roughness
        "road_luminance": Range(0.02, 0.16, 0.05, "linear", ""),
        "road_roughness": Range(0.5, 0.95, 0.8, "linear", ""),
        # V7 tape tint
        "yellow_hue_deg": Range(-20.0, 20.0, 0.0, "linear", "deg"),
        "yellow_sat_scale": Range(0.75, 1.10, 1.0, "linear", ""),
        "white_hue_deg": Range(-8.0, 8.0, 0.0, "linear", "deg"),
        # V8 tape roughness / specular
        "tape_roughness": Range(0.25, 0.90, 0.6, "linear", ""),
        "tape_specular": Range(0.0, 0.5, 0.2, "linear", ""),
        # V9 marking geometry (baked into the 16 texture buckets; the live lane width per
        # episode still comes from the variant metadata and is needed by the reward gate)
        "lane_width_m": Range(0.17, 0.28, 0.21, "linear", "m"),
        "centerline_offset_m": Range(-0.015, 0.015, 0.0, "linear", "m"),
        "tile_pitch_m": Range(0.570, 0.615, 0.585, "linear", "m"),
        # V13 walls / distractors
        "wall_albedo_scale": Range(0.6, 1.4, 1.0, "linear", ""),
        "num_distractors": Range(0.0, 8.0, 4.0, "int", "count"),
    }


def default_camera_mount_ranges() -> dict[str, Range]:
    """Return the V10 camera-mount axes of SPEC v2 S7.2 (nominal values from S2).

    Returns:
        A range book keyed by parameter name. Heights and forward offsets are in metres in the
        ground frame; ``pitch_down`` is the positive scalar the ROS quaternion helper consumes.
    """
    return {
        "height_m": Range(0.090, 0.120, 0.101, "linear", "m above ground"),
        "pitch_down_deg": Range(15.0, 28.0, 25.3, "linear", "deg"),
        "forward_m": Range(0.058, 0.085, 0.078, "linear", "m"),
        "yaw_deg": Range(-3.0, 3.0, 0.0, "linear", "deg"),
        "roll_deg": Range(-3.0, 3.0, 0.0, "linear", "deg"),
    }


# --------------------------------------------------------------------------------------------
# Parameter container
# --------------------------------------------------------------------------------------------


@dataclass
class VisualParams:
    """Per-env photometric parameters for one control step.

    Every field is a ``(N,)`` float32 tensor on the DR device. The container is a plain
    dataclass so it can be logged, asserted on and replayed in tests.

    Attributes:
        exposure_log2: V14 exposure in stops.
        gamma: V14 gamma.
        contrast: V14 contrast gain around the per-image luma mean.
        saturation: V14 saturation gain.
        hue_deg: Extra hue rotation in degrees (see :func:`default_visual_ranges`).
        wb_r: V14 red white-balance gain.
        wb_b: V14 blue white-balance gain.
        noise_sigma: V15 gaussian sigma in intensity units.
        shot_scale: V15 shot-noise scale (std = ``shot_scale * sqrt(I)``).
        blur_len_px: V16 motion-blur length in render px (already speed scaled).
        blur_angle_rad: V16 motion-blur direction.
        jpeg_quality: V17 JPEG quality in [30, 100]; 100 disables the op.
        vignette: V18 corner falloff.
        ca_px: V18 lateral chromatic aberration at the image edge, in render px.
        defocus_sigma_px: V18 defocus sigma in render px.
        distort_k1: V12 residual radial distortion coefficient.
        frame_repeat_p: V19 per-env probability that the next frame repeats.
    """

    exposure_log2: torch.Tensor
    gamma: torch.Tensor
    contrast: torch.Tensor
    saturation: torch.Tensor
    hue_deg: torch.Tensor
    wb_r: torch.Tensor
    wb_b: torch.Tensor
    noise_sigma: torch.Tensor
    shot_scale: torch.Tensor
    blur_len_px: torch.Tensor
    blur_angle_rad: torch.Tensor
    jpeg_quality: torch.Tensor
    vignette: torch.Tensor
    ca_px: torch.Tensor
    defocus_sigma_px: torch.Tensor
    distort_k1: torch.Tensor
    frame_repeat_p: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        """Return the parameters as a plain dict.

        Returns:
            ``{field_name: tensor}`` for every field.
        """
        return dict(self.__dict__)


# --------------------------------------------------------------------------------------------
# Operators
# --------------------------------------------------------------------------------------------


def _check_image(x: torch.Tensor) -> tuple[int, int, int, int]:
    """Validate an NCHW image batch.

    Args:
        x: The tensor to check.

    Returns:
        The ``(N, C, H, W)`` shape.

    Raises:
        ValueError: If the tensor is not a 4-D 3-channel float tensor.
    """
    if x.ndim != 4 or x.shape[1] != 3:
        raise ValueError(f"expected NCHW float image with C=3, got {tuple(x.shape)}")
    if not torch.is_floating_point(x):
        raise ValueError(f"expected a float tensor, got {x.dtype}")
    n, c, h, w = x.shape
    return n, c, h, w


def _luma(x: torch.Tensor) -> torch.Tensor:
    """Compute Rec.601 luma.

    Args:
        x: NCHW RGB tensor.

    Returns:
        An ``(N, 1, H, W)`` luma tensor.
    """
    w = _cached(
        ("luma", x.device, x.dtype),
        lambda: torch.tensor(_LUMA, dtype=x.dtype, device=x.device).view(1, 3, 1, 1),
    )
    return (x * w).sum(dim=1, keepdim=True)


def _hue_matrix(deg: torch.Tensor) -> torch.Tensor:
    """Build the luma-preserving hue-rotation matrices.

    Args:
        deg: ``(N,)`` rotation angles in degrees.

    Returns:
        An ``(N, 3, 3)`` matrix batch; the identity at 0 degrees.
    """
    th = deg * (math.pi / 180.0)
    c, s = torch.cos(th), torch.sin(th)
    r0 = torch.stack(
        [0.213 + c * 0.787 - s * 0.213, 0.715 - c * 0.715 - s * 0.715, 0.072 - c * 0.072 + s * 0.928], -1
    )
    r1 = torch.stack(
        [0.213 - c * 0.213 + s * 0.143, 0.715 + c * 0.285 + s * 0.140, 0.072 - c * 0.072 - s * 0.283], -1
    )
    r2 = torch.stack(
        [0.213 - c * 0.213 - s * 0.787, 0.715 - c * 0.715 + s * 0.715, 0.072 + c * 0.928 + s * 0.072], -1
    )
    return torch.stack([r0, r1, r2], dim=-2)


def apply_color(
    x: torch.Tensor,
    exposure_log2: torch.Tensor,
    gamma: torch.Tensor,
    contrast: torch.Tensor,
    saturation: torch.Tensor,
    wb_r: torch.Tensor,
    wb_b: torch.Tensor,
    hue_deg: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply V14 exposure, white balance, contrast, saturation, hue and gamma.

    Order (fixed, matches a real ISP): exposure -> white balance -> contrast -> saturation ->
    hue -> gamma, with a clamp to [0, 1] before the gamma so the power is always well defined.

    Args:
        x: NCHW float image in [0, 1].
        exposure_log2: ``(N,)`` exposure in stops.
        gamma: ``(N,)`` gamma exponent.
        contrast: ``(N,)`` contrast gain around the per-image luma mean.
        saturation: ``(N,)`` saturation gain.
        wb_r: ``(N,)`` red gain.
        wb_b: ``(N,)`` blue gain.
        hue_deg: Optional ``(N,)`` hue rotation in degrees.

    Returns:
        The randomized NCHW image, clamped to [0, 1].
    """
    n = _check_image(x)[0]
    dev, dt = x.device, x.dtype
    lw = torch.tensor(_LUMA, device=dev, dtype=dt)

    # Exposure, white balance, contrast, saturation and hue are ALL affine in RGB, and the only
    # image-dependent quantity among them is the scalar luma mean used by contrast - which can
    # be obtained from the per-channel means of the input. So the whole block collapses to one
    # 3x3 matrix plus a bias per env: a single pass over the N x 3 x 128 x 192 tensor instead of
    # roughly a dozen. (Measured: 6.6 ms -> 2.4 ms per step at N = 256 on an RTX 3080 Laptop.)
    gains = torch.stack([wb_r, torch.ones_like(wb_r), wb_b], dim=1).to(dev, dt)
    gains = gains * torch.exp2(exposure_log2.to(dev, dt)).view(n, 1)
    mean_lum = ((x.mean(dim=(2, 3)) * gains) * lw).sum(dim=1)

    c = contrast.to(dev, dt).view(n, 1, 1)
    sat = saturation.to(dev, dt).view(n, 1, 1)
    eye = torch.eye(3, device=dev, dtype=dt).unsqueeze(0)
    m_sat = sat * eye + (1.0 - sat) * lw.view(1, 1, 3).expand(1, 3, 3)
    m = m_sat * (c * gains.view(n, 1, 3))
    bias = (m_sat @ ((1.0 - c.view(n, 1)) * mean_lum.view(n, 1)).expand(n, 3).unsqueeze(-1)).squeeze(-1)
    if hue_deg is not None:
        m_hue = _hue_matrix(hue_deg.to(dev, dt))
        m = m_hue @ m
        bias = (m_hue @ bias.unsqueeze(-1)).squeeze(-1)

    x = torch.einsum("nij,njhw->nihw", m, x) + bias.view(n, 3, 1, 1)
    x = x.clamp(0.0, 1.0)
    return torch.pow(x, gamma.to(dev, dt).view(n, 1, 1, 1))


def apply_noise(
    x: torch.Tensor,
    noise_sigma: torch.Tensor,
    shot_scale: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Apply V15 gaussian read noise plus signal-dependent shot noise.

    Shot noise is modelled with the usual gaussian approximation to a Poisson photon count:
    ``std = shot_scale * sqrt(I)``. Both terms are exactly zero at ``alpha = 0``.

    Args:
        x: NCHW float image in [0, 1].
        noise_sigma: ``(N,)`` gaussian sigma in intensity units.
        shot_scale: ``(N,)`` shot-noise scale.
        generator: Torch generator (determinism).

    Returns:
        The noisy image, clamped to [0, 1].
    """
    n = _check_image(x)[0]
    eps = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
    sigma = noise_sigma.to(x.device, x.dtype).view(n, 1, 1, 1)
    shot = shot_scale.to(x.device, x.dtype).view(n, 1, 1, 1) * torch.sqrt(x.clamp_min(0.0))
    return (x + eps * (sigma + shot)).clamp(0.0, 1.0)


def _base_grid(n: int, h: int, w: int, device: Any, dtype: Any) -> torch.Tensor:
    """Build the identity sampling grid for ``grid_sample`` (align_corners=True).

    Args:
        n: Batch size.
        h: Height.
        w: Width.
        device: Torch device.
        dtype: Torch dtype.

    Returns:
        An ``(N, H, W, 2)`` grid in normalized [-1, 1] coordinates.
    """

    def _build() -> torch.Tensor:
        ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([gx, gy], dim=-1).unsqueeze(0)

    return _cached(("grid", h, w, device, dtype), _build).expand(n, h, w, 2)


def _sample(x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Bilinear ``grid_sample`` with border padding and ``align_corners=True``.

    Args:
        x: NCHW image.
        grid: ``(N, H, W, 2)`` sampling grid.

    Returns:
        The resampled image.
    """
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)


def apply_motion_blur(
    x: torch.Tensor, length_px: torch.Tensor, angle_rad: torch.Tensor, *, taps: int = 7
) -> torch.Tensor:
    """Apply V16 directional motion blur.

    The blur is a line integral of ``taps`` bilinear samples spread over ``length_px`` in the
    direction ``angle_rad``. The AI-DO evaluator synthesizes blur by overlaying many sub-frames,
    which is 10-20x slower per observation; a directional convolution is the documented
    substitute (10_report_06 V51).

    Args:
        x: NCHW float image.
        length_px: ``(N,)`` blur length in render px (0 disables the op for that env).
        angle_rad: ``(N,)`` blur direction.
        taps: Number of samples along the line (odd).

    Returns:
        The blurred image; envs with ``length_px == 0`` are returned bit-exact.

    Raises:
        ValueError: If ``taps`` is even or smaller than 3.
    """
    if taps < 3 or taps % 2 == 0:
        raise ValueError(f"taps must be an odd integer >= 3, got {taps}")
    n, _, h, w = _check_image(x)
    length = length_px.to(x.device, x.dtype).view(n, 1, 1, 1)
    ang = angle_rad.to(x.device, x.dtype).view(n, 1, 1, 1)
    base = _base_grid(n, h, w, x.device, x.dtype)
    px_x = 2.0 / max(w - 1, 1)
    px_y = 2.0 / max(h - 1, 1)
    acc = torch.zeros_like(x)
    for i in range(taps):
        t = (i / (taps - 1)) - 0.5
        dx = (length * t * torch.cos(ang)) * px_x
        dy = (length * t * torch.sin(ang)) * px_y
        offset = torch.cat([dx, dy], dim=-1).view(n, 1, 1, 2)
        acc = acc + _sample(x, base + offset)
    blurred = acc / float(taps)
    return torch.where((length_px.to(x.device) > 0).view(n, 1, 1, 1), blurred, x)


def apply_defocus(x: torch.Tensor, sigma_px: torch.Tensor, *, radius: int = 2) -> torch.Tensor:
    """Apply V18 defocus as a per-env separable gaussian blur.

    Implemented as a grouped convolution over ``N * C`` groups so every env gets its own kernel
    in a single kernel launch (no python loop over envs).

    Args:
        x: NCHW float image.
        sigma_px: ``(N,)`` gaussian sigma in render px (0 is an exact identity).
        radius: Kernel half-width.

    Returns:
        The defocused image.
    """
    n, c, h, w = _check_image(x)
    s = sigma_px.to(x.device, x.dtype).clamp_min(1e-6).view(n, 1)
    t = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype).view(1, -1)
    k = torch.exp(-0.5 * (t / s) ** 2)
    k = k / k.sum(dim=1, keepdim=True)
    kk = k.repeat_interleave(c, dim=0)
    xr = x.reshape(1, n * c, h, w)
    xr = F.conv2d(F.pad(xr, (radius, radius, 0, 0), mode="replicate"), kk.view(n * c, 1, 1, -1), groups=n * c)
    xr = F.conv2d(F.pad(xr, (0, 0, radius, radius), mode="replicate"), kk.view(n * c, 1, -1, 1), groups=n * c)
    return xr.reshape(n, c, h, w)


def apply_vignette(x: torch.Tensor, strength: torch.Tensor) -> torch.Tensor:
    """Apply V18 vignetting.

    Args:
        x: NCHW float image.
        strength: ``(N,)`` falloff at the image corner (0 is an exact identity).

    Returns:
        The vignetted image.
    """
    n, _, h, w = _check_image(x)

    def _build() -> torch.Tensor:
        grid = _base_grid(1, h, w, x.device, x.dtype)[0]
        return ((grid[..., 0] ** 2 + grid[..., 1] ** 2) / 2.0).view(1, 1, h, w)

    r2 = _cached(("vign", h, w, x.device, x.dtype), _build)
    return x * (1.0 - strength.to(x.device, x.dtype).view(n, 1, 1, 1) * r2)


def apply_chromatic_aberration(x: torch.Tensor, shift_px: torch.Tensor) -> torch.Tensor:
    """Apply V18 lateral chromatic aberration.

    Red is magnified and blue demagnified (radially) so that the offset reaches ``shift_px`` at
    the image corner, which is how a cheap wide-angle lens misregisters channels.

    Args:
        x: NCHW float image.
        shift_px: ``(N,)`` lateral shift in render px at the image corner.

    Returns:
        The aberrated image; envs with ``shift_px == 0`` are returned bit-exact.
    """
    n, _, h, w = _check_image(x)
    half_diag = 0.5 * math.hypot(w - 1, h - 1)
    e = (shift_px.to(x.device, x.dtype) / half_diag).view(n, 1, 1, 1)
    base = _base_grid(n, h, w, x.device, x.dtype)
    r = _sample(x[:, 0:1], base * (1.0 - e))
    b = _sample(x[:, 2:3], base * (1.0 + e))
    out = torch.cat([r, x[:, 1:2], b], dim=1)
    return torch.where((shift_px.to(x.device) > 0).view(n, 1, 1, 1), out, x)


def apply_lens_distortion(x: torch.Tensor, k1: torch.Tensor) -> torch.Tensor:
    """Apply V12 residual radial distortion left over after rectification.

    Args:
        x: NCHW float image.
        k1: ``(N,)`` radial coefficient; negative is barrel, positive pincushion.

    Returns:
        The warped image; envs with ``k1 == 0`` are returned bit-exact.
    """
    n, _, h, w = _check_image(x)
    base = _base_grid(n, h, w, x.device, x.dtype)
    r2 = (base[..., 0] ** 2 + base[..., 1] ** 2).unsqueeze(-1)
    grid = base * (1.0 + k1.to(x.device, x.dtype).view(n, 1, 1, 1) * r2)
    return torch.where((k1.to(x.device) != 0).view(n, 1, 1, 1), _sample(x, grid), x)


def _dct_matrix(device: Any, dtype: Any) -> torch.Tensor:
    """Build the orthonormal 8x8 DCT-II matrix.

    Args:
        device: Torch device.
        dtype: Torch dtype.

    Returns:
        An ``(8, 8)`` matrix ``M`` such that ``M @ B @ M.T`` is the 2-D DCT of block ``B``.
    """

    def _build() -> torch.Tensor:
        k = torch.arange(8, device=device, dtype=dtype).view(8, 1)
        nn = torch.arange(8, device=device, dtype=dtype).view(1, 8)
        m = torch.cos(math.pi * (2.0 * nn + 1.0) * k / 16.0)
        scale = torch.full((8, 1), math.sqrt(2.0 / 8.0), device=device, dtype=dtype)
        scale[0, 0] = math.sqrt(1.0 / 8.0)
        return m * scale

    return _cached(("dct", device, dtype), _build)


def _quant_tables(quality: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale the JPEG Annex-K tables to a per-env quality.

    Args:
        quality: ``(N,)`` quality in [1, 100].

    Returns:
        ``(luma, chroma)`` tables of shape ``(N, 8, 8)``.
    """
    q = quality.clamp(1.0, 100.0).view(-1, 1, 1)
    scale = torch.where(q < 50.0, 5000.0 / q, 200.0 - 2.0 * q)
    dev, dt = quality.device, quality.dtype
    base_l = _cached(("qluma", dev, dt), lambda: torch.tensor(_Q_LUMA, device=dev, dtype=dt).unsqueeze(0))
    base_c = _cached(("qchroma", dev, dt), lambda: torch.tensor(_Q_CHROMA, device=dev, dtype=dt).unsqueeze(0))
    tl = torch.floor((base_l * scale + 50.0) / 100.0).clamp(1.0, 255.0)
    tc = torch.floor((base_c * scale + 50.0) / 100.0).clamp(1.0, 255.0)
    return tl, tc


def apply_jpeg(x: torch.Tensor, quality: torch.Tensor) -> torch.Tensor:
    """Apply V17 JPEG-like 8x8 block DCT quantization.

    The Duckiebot publishes ``CompressedImage`` (JPEG) over ROS, so ringing on the high-contrast
    white-on-black lane edges is a real, systematic artefact of the deployed observation. This is
    a faithful-but-batched approximation: full-range YCbCr, Annex-K quantization tables scaled by
    quality, no chroma subsampling and no entropy coding (neither changes the reconstructed
    pixels beyond the quantization already applied).

    Args:
        x: NCHW float image in [0, 1]; H and W must be multiples of 8.
        quality: ``(N,)`` JPEG quality; ``>= 100`` disables the op for that env.

    Returns:
        The compressed-then-decompressed image, clamped to [0, 1].

    Raises:
        ValueError: If H or W is not a multiple of 8.
    """
    n, c, h, w = _check_image(x)
    if h % 8 or w % 8:
        raise ValueError(f"apply_jpeg needs H and W divisible by 8, got {(h, w)}")
    q = quality.to(x.device, x.dtype)

    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 0.5 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 0.5 + 0.5 * r - 0.418688 * g - 0.081312 * b
    ycc = torch.stack([y, cb, cr], dim=1) * 255.0 - 128.0

    # 2-D DCT without a single permute: the natural (n, c, h/8, 8, w/8, 8) view already has the
    # row-of-block axis at dim 3 and the column-in-block axis at dim 5, so both transforms are
    # contiguous matmuls. Permuting to (..., 8, 8) blocks instead costs a 75 MB copy per call at
    # N = 256 and turns the GEMM into ~300k 8x8x8 batched matmuls.
    m = _dct_matrix(x.device, x.dtype)
    nb = n * c * (h // 8)
    rows = torch.matmul(m, ycc.reshape(nb, 8, w))
    coeff = torch.matmul(rows.reshape(nb, 8, w // 8, 8), m.t())

    tl, tc = _quant_tables(q)
    tab = torch.stack([tl, tc, tc], dim=1).view(n, c, 1, 8, 1, 8)
    coeff = coeff.reshape(n, c, h // 8, 8, w // 8, 8)
    coeff = torch.round(coeff / tab) * tab

    cols = torch.matmul(coeff.reshape(nb, 8, w // 8, 8), m)
    ycc = torch.matmul(m.t(), cols.reshape(nb, 8, w)).reshape(n, c, h, w)
    ycc = (ycc + 128.0) / 255.0

    y, cb, cr = ycc[:, 0], ycc[:, 1] - 0.5, ycc[:, 2] - 0.5
    out = torch.stack([y + 1.402 * cr, y - 0.344136 * cb - 0.714136 * cr, y + 1.772 * cb], dim=1)
    out = out.clamp(0.0, 1.0)
    return torch.where((q >= 100.0).view(n, 1, 1, 1), x, out)


def sample_frame_repeat(prob: torch.Tensor, *, generator: torch.Generator | None = None) -> torch.Tensor:
    """Sample the V19 frame-repeat mask.

    Args:
        prob: ``(N,)`` per-env repeat probability.
        generator: Torch generator (determinism).

    Returns:
        A ``(N,)`` bool tensor; True means the frame ring should re-push the previous frame
        (see :meth:`duckiebot_rl.dr.preprocess.FrameStack.push`).
    """
    u = torch.rand(prob.shape, generator=generator, device=prob.device, dtype=torch.float32)
    return u < prob.to(torch.float32)


def sample_camera_mount(
    num_envs: int,
    alpha: float | torch.Tensor = 1.0,
    *,
    book: RangeBook | None = None,
    generator: torch.Generator | None = None,
    device: Any = None,
    boundary: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Sample the V10 camera-mount pose per env.

    The returned ``pitch_down_rad`` is the positive scalar consumed by the shared
    ``quat_cam_ros`` helper (SPEC v2 S2): no code outside that helper touches camera signs.

    Args:
        num_envs: Number of envs.
        alpha: Curriculum scalar.
        book: Optional override of :func:`default_camera_mount_ranges`.
        generator: Torch generator (determinism).
        device: Torch device.
        boundary: Optional ADR boundary-probe mask.

    Returns:
        A dict with ``height_m``, ``forward_m``, ``pitch_down_rad``, ``yaw_rad``, ``roll_rad``
        and ``base_z_m`` (the mount height expressed in the base_link frame, i.e. minus the
        0.0318 m axle height of S2).
    """
    b = dict(book or default_camera_mount_ranges())
    out = {
        k: b[k].sample(num_envs, alpha, generator=generator, device=device, boundary=boundary)
        for k in ("height_m", "pitch_down_deg", "forward_m", "yaw_deg", "roll_deg")
    }
    deg2rad = math.pi / 180.0
    return {
        "height_m": out["height_m"],
        "forward_m": out["forward_m"],
        "base_z_m": out["height_m"] - 0.0318,
        "pitch_down_rad": out["pitch_down_deg"] * deg2rad,
        "yaw_rad": out["yaw_deg"] * deg2rad,
        "roll_rad": out["roll_deg"] * deg2rad,
    }


def sample_scene_params(
    num_envs: int,
    alpha: float | torch.Tensor = 1.0,
    *,
    book: RangeBook | None = None,
    generator: torch.Generator | None = None,
    device: Any = None,
    boundary: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Sample the scene-side visual axes V1-V9 / V13 for the Isaac event terms.

    Args:
        num_envs: Number of envs.
        alpha: Curriculum scalar ``alpha_vis``.
        book: Optional override of :func:`default_scene_ranges`.
        generator: Torch generator (determinism).
        device: Torch device.
        boundary: Optional ADR boundary-probe mask.

    Returns:
        A dict of ``(N,)`` tensors, one per axis of the book. ``lamp_offset_m`` is returned as
        ``(N, 2)`` because the lamp moves in x and y.
    """
    b = dict(book or default_scene_ranges())
    out: dict[str, torch.Tensor] = {}
    for name, rng in b.items():
        shape: int | tuple[int, ...] = (num_envs, 2) if name == "lamp_offset_m" else num_envs
        bnd = boundary
        if bnd is not None and name == "lamp_offset_m":
            bnd = bnd.view(-1, 1).expand(num_envs, 2)
        out[name] = rng.sample(shape, alpha, generator=generator, device=device, boundary=bnd)
    return out


# --------------------------------------------------------------------------------------------
# The step operator
# --------------------------------------------------------------------------------------------


@dataclass
class VisualDRCfg:
    """Configuration of the per-step photometric DR chain.

    Attributes:
        ranges: Range book; defaults to :func:`default_visual_ranges`.
        motion_blur_taps: Number of line samples in :func:`apply_motion_blur`.
        defocus_radius: Kernel half-width of :func:`apply_defocus`.
        enable_jpeg: Set False to skip V17 (it is the most expensive op; the ablation configs
            in ``configs/train/ablations`` use this).
        enable_motion_blur: Set False to skip V16.
    """

    ranges: dict[str, Range] = field(default_factory=default_visual_ranges)
    motion_blur_taps: int = 7
    defocus_radius: int = 2
    enable_jpeg: bool = True
    enable_motion_blur: bool = True


class VisualDR:
    """Samples and applies the per-step photometric DR chain (SPEC v2 S4.3 step 3).

    The chain order is fixed by the spec: V14 colour -> V16 motion blur -> V18 vignette / CA /
    defocus -> V15 noise -> V17 JPEG. V12 residual distortion is applied first, because it is a
    property of the optics rather than of the sensor. V11 principal-point jitter is deliberately
    *not* here: it is step 4 of the chain and lives in ``preprocess.shift_principal_point``.

    Args:
        num_envs: Number of parallel envs.
        cfg: Configuration; defaults to :class:`VisualDRCfg`.
        device: Torch device on which parameters are sampled.
        generator: Torch generator used for every draw (determinism).
    """

    def __init__(
        self,
        num_envs: int,
        cfg: VisualDRCfg | None = None,
        *,
        device: Any = None,
        generator: torch.Generator | None = None,
    ) -> None:
        self.num_envs = int(num_envs)
        self.cfg = cfg or VisualDRCfg()
        self.device = device
        self.generator = generator

    def sample(
        self,
        alpha: float | torch.Tensor = 1.0,
        *,
        speed_frac: torch.Tensor | None = None,
        boundary: torch.Tensor | None = None,
    ) -> VisualParams:
        """Sample one step of per-env photometric parameters.

        Args:
            alpha: Curriculum scalar ``alpha_vis`` (float or per-env tensor).
            speed_frac: Optional ``(N,)`` speed fraction in [0, 1] used to scale the motion-blur
                length (S7.2 V16: "length prop. to speed"). ``None`` means full length.
            boundary: Optional ADR boundary-probe mask.

        Returns:
            A :class:`VisualParams` with one entry per env.
        """
        b = self.cfg.ranges
        kw = {"generator": self.generator, "device": self.device, "boundary": boundary}
        p = {k: b[k].sample(self.num_envs, alpha, **kw) for k in b}
        blur = p["blur_len_px"]
        if speed_frac is not None:
            blur = blur * speed_frac.to(blur.device, blur.dtype).clamp(0.0, 1.0)
        return VisualParams(
            exposure_log2=p["exposure_log2"],
            gamma=p["gamma"],
            contrast=p["contrast"],
            saturation=p["saturation"],
            hue_deg=p["hue_deg"],
            wb_r=p["wb_r"],
            wb_b=p["wb_b"],
            noise_sigma=p["noise_sigma"],
            shot_scale=p["shot_scale"],
            blur_len_px=blur,
            blur_angle_rad=p["blur_angle_rad"],
            jpeg_quality=p["jpeg_quality"],
            vignette=p["vignette"],
            ca_px=p["ca_px"],
            defocus_sigma_px=p["defocus_sigma_px"],
            distort_k1=p["distort_k1"],
            frame_repeat_p=p["frame_repeat_p"],
        )

    def apply(self, x: torch.Tensor, params: VisualParams) -> torch.Tensor:
        """Apply a sampled parameter set to an image batch.

        Args:
            x: NCHW float32 image in [0, 1] at render resolution.
            params: Parameters from :meth:`sample`.

        Returns:
            The randomized image, same shape/dtype, clamped to [0, 1].
        """
        x = apply_lens_distortion(x, params.distort_k1)
        x = apply_color(
            x,
            params.exposure_log2,
            params.gamma,
            params.contrast,
            params.saturation,
            params.wb_r,
            params.wb_b,
            params.hue_deg,
        )
        if self.cfg.enable_motion_blur:
            x = apply_motion_blur(
                x, params.blur_len_px, params.blur_angle_rad, taps=self.cfg.motion_blur_taps
            )
        x = apply_vignette(x, params.vignette)
        x = apply_chromatic_aberration(x, params.ca_px)
        x = apply_defocus(x, params.defocus_sigma_px, radius=self.cfg.defocus_radius)
        x = apply_noise(x, params.noise_sigma, params.shot_scale, generator=self.generator)
        if self.cfg.enable_jpeg:
            x = apply_jpeg(x, params.jpeg_quality)
        return x.clamp(0.0, 1.0)

    def operator(self, params: VisualParams) -> Callable[[torch.Tensor], torch.Tensor]:
        """Bind parameters into a one-argument operator.

        This is what gets handed to ``preprocess_frame(photometric=...)`` so the DR runs at
        exactly S4.3 step 3.

        Args:
            params: Parameters from :meth:`sample`.

        Returns:
            A callable mapping an NCHW image to the randomized NCHW image.
        """

        def _op(x: torch.Tensor) -> torch.Tensor:
            return self.apply(x, params)

        return _op

    def randomize(
        self,
        x: torch.Tensor,
        alpha: float | torch.Tensor = 1.0,
        *,
        speed_frac: torch.Tensor | None = None,
        boundary: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, VisualParams]:
        """Sample and apply in one call.

        Args:
            x: NCHW float32 image in [0, 1].
            alpha: Curriculum scalar ``alpha_vis``.
            speed_frac: Optional ``(N,)`` speed fraction for the motion-blur length.
            boundary: Optional ADR boundary-probe mask.

        Returns:
            ``(randomized_image, params)``.
        """
        params = self.sample(alpha, speed_frac=speed_frac, boundary=boundary)
        return self.apply(x, params), params
