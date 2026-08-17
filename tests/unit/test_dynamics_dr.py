"""Unit tests for the dynamics DR sampler (SPEC v2 S7.3).

Covers: every sampled range is respected, alpha = 0 reproduces the nominal robot exactly, envs
are independent, draws are reproducible under a seed, partial resets touch only the resetting
envs, and the derived quantities (per-wheel radius, kinetic friction, encoder model) are correct.
"""

from __future__ import annotations

import math

import pytest
import torch

from duckiebot_rl.dr.dynamics import (
    ENCODER_TICKS_PER_REV,
    NOMINAL_WHEEL_RADIUS_M,
    DynamicsDRCfg,
    DynamicsRandomizer,
    default_dynamics_ranges,
    quantize_encoder,
)

# Parameter fields that map 1:1 onto a range-book entry.
DIRECT_FIELDS = (
    "baseline_m",
    "mass_kg",
    "izz_scale",
    "motor_gain",
    "motor_trim",
    "battery_sag",
    "battery_decay_per_s",
    "deadband_duty",
    "delay_steps",
    "delay_frac",
    "motor_lag_alpha",
    "obs_delay_steps",
    "control_jitter",
    "drag_u1",
    "drag_w1",
    "slip_std",
    "encoder_dropout_p",
    "push_interval_s",
    "push_dv",
    "push_dyaw",
    "tile_friction_scale",
    "spawn_lateral_m",
    "spawn_heading_deg",
    "effort_limit_nm",
    "brake_beta",
    "armature_scale",
    "joint_friction_nm",
    "friction_static",
)


def _rand(n: int = 1024, seed: int = 0) -> DynamicsRandomizer:
    return DynamicsRandomizer(n, generator=torch.Generator().manual_seed(seed))


def test_all_direct_axes_respect_their_ranges_at_alpha_one():
    r = _rand()
    p = r.resample(alpha=1.0).as_dict()
    book = default_dynamics_ranges()
    for name in DIRECT_FIELDS:
        rng = book[name]
        t = p[name].float()
        assert float(t.min()) >= rng.lo - 1e-5, f"{name} below {rng.lo}"
        assert float(t.max()) <= rng.hi + 1e-5, f"{name} above {rng.hi}"


def test_alpha_zero_reproduces_the_nominal_robot():
    r = _rand(64, seed=1)
    p = r.resample(alpha=0.0).as_dict()
    book = default_dynamics_ranges()
    for name in DIRECT_FIELDS:
        t = p[name].float()
        assert torch.allclose(t, torch.full_like(t, book[name].nominal), atol=1e-6), name
    assert torch.allclose(p["wheel_radius_m"], torch.full((64, 2), NOMINAL_WHEEL_RADIUS_M))
    assert torch.allclose(p["com_offset_m"][:, 0], torch.full((64,), -0.015))
    assert torch.allclose(p["com_offset_m"][:, 2], torch.full((64,), 0.015))


def test_shapes_and_dtypes():
    p = _rand(32).resample(alpha=1.0)
    assert p.wheel_radius_m.shape == (32, 2)
    assert p.deadband_duty.shape == (32, 2)
    assert p.friction_static.shape == (32, 2)
    assert p.com_offset_m.shape == (32, 3)
    assert p.baseline_m.shape == (32,)
    assert p.delay_steps.dtype == torch.long
    assert p.obs_delay_steps.dtype == torch.long


def test_integer_axes_take_only_legal_values():
    p = _rand(2048).resample(alpha=1.0)
    assert set(p.delay_steps.unique().tolist()) <= {1, 2, 3}
    assert set(p.obs_delay_steps.unique().tolist()) <= {0, 1, 2}


def test_wheel_radius_asymmetry_is_bounded_and_two_sided():
    p = _rand(4096, seed=2).resample(alpha=1.0)
    lo = NOMINAL_WHEEL_RADIUS_M * 0.95 * 0.94
    hi = NOMINAL_WHEEL_RADIUS_M * 1.05 * 1.06
    assert float(p.wheel_radius_m.min()) >= lo - 1e-9
    assert float(p.wheel_radius_m.max()) <= hi + 1e-9
    diff = p.wheel_radius_m[:, 0] - p.wheel_radius_m[:, 1]
    assert float(diff.max()) > 0.0 and float(diff.min()) < 0.0


def test_kinetic_friction_never_exceeds_static():
    p = _rand(4096, seed=3).resample(alpha=1.0)
    assert bool((p.friction_kinetic <= p.friction_static + 1e-6).all())
    assert float((p.friction_static - p.friction_kinetic).max()) > 0.0


def test_per_wheel_friction_can_be_disabled():
    r = DynamicsRandomizer(
        128,
        DynamicsDRCfg(per_wheel_friction=False),
        generator=torch.Generator().manual_seed(4),
    )
    p = r.resample(alpha=1.0)
    assert torch.equal(p.friction_static[:, 0], p.friction_static[:, 1])


def test_per_env_independence():
    p = _rand(512, seed=5).resample(alpha=1.0).as_dict()
    for name, t in p.items():
        assert t.float().std() > 0.0, f"{name} is identical across envs"


def test_reproducible_under_seed():
    a = _rand(64, seed=7).resample(alpha=1.0).as_dict()
    b = _rand(64, seed=7).resample(alpha=1.0).as_dict()
    c = _rand(64, seed=8).resample(alpha=1.0).as_dict()
    for name, t in a.items():
        assert torch.equal(t, b[name]), name
    assert not torch.equal(a["motor_gain"], c["motor_gain"])


def test_partial_resample_touches_only_the_given_envs():
    r = _rand(16, seed=9)
    before = {k: v.clone() for k, v in r.resample(alpha=1.0).as_dict().items()}
    ids = torch.tensor([2, 5, 11])
    after = r.resample(ids, alpha=1.0).as_dict()
    mask = torch.ones(16, dtype=torch.bool)
    mask[ids] = False
    for name, t in after.items():
        assert torch.equal(t[mask], before[name][mask]), f"{name} changed on non-reset envs"
    assert not torch.equal(after["motor_gain"][ids], before["motor_gain"][ids])


def test_boundary_probe_samples_the_live_clamps():
    r = _rand(256, seed=10)
    boundary = torch.ones(256, dtype=torch.bool)
    p = r.resample(alpha=1.0, boundary=boundary)
    book = default_dynamics_ranges()
    gain = p.motor_gain
    lo, hi = book["motor_gain"].lo, book["motor_gain"].hi
    assert torch.all(((gain - lo).abs() < 1e-6) | ((gain - hi).abs() < 1e-6))
    assert bool(((gain - lo).abs() < 1e-6).any())
    assert bool(((gain - hi).abs() < 1e-6).any())


def test_push_and_spawn_samplers():
    r = _rand(1024, seed=12)
    dv, dyaw = r.sample_push(alpha=1.0)
    assert dv.shape == (1024,) and dyaw.shape == (1024,)
    assert float(dv.abs().max()) <= 0.15 + 1e-6
    assert float(dyaw.abs().max()) <= 0.6 + 1e-6
    lateral, heading = r.spawn_pose(alpha=1.0)
    assert float(lateral.abs().max()) <= 0.06 + 1e-6
    assert float(heading.abs().max()) <= 25.0 * math.pi / 180.0 + 1e-6
    zero_dv, _ = r.sample_push(alpha=0.0)
    assert torch.allclose(zero_dv, torch.zeros(1024))


def test_state_dict_round_trip():
    a = _rand(8, seed=13)
    a.resample(alpha=1.0)
    state = {k: v.clone() for k, v in a.state_dict().items()}
    b = _rand(8, seed=99)
    b.resample(alpha=1.0)
    b.load_state_dict(state)
    for name, t in b.state_dict().items():
        assert torch.equal(t, state[name]), name


def test_state_dict_load_rejects_missing_field():
    r = _rand(4)
    state = dict(r.state_dict())
    del state["motor_gain"]
    with pytest.raises(KeyError, match="motor_gain"):
        r.load_state_dict(state)


# ---------------------------------------------------------------------------- encoder model


def test_encoder_quantization_is_on_the_tick_grid():
    dt = 1.0 / 15.0
    quantum = 2.0 * math.pi / (ENCODER_TICKS_PER_REV * dt)
    w = torch.tensor([[0.0, 3.7], [12.4, -8.1]])
    out = quantize_encoder(w, dt, torch.zeros(2), tick_noise=0.0)
    ticks = out / quantum
    assert torch.allclose(ticks, torch.round(ticks), atol=1e-5)
    assert float((out - w).abs().max()) <= quantum


def test_encoder_dropout_zeroes_readings():
    g = torch.Generator().manual_seed(0)
    w = torch.full((256, 2), 10.0)
    kept = quantize_encoder(w, 1 / 15.0, torch.zeros(256), tick_noise=0.0, generator=g)
    assert bool((kept != 0).all())
    dropped = quantize_encoder(w, 1 / 15.0, torch.ones(256), tick_noise=0.0, generator=g)
    assert bool((dropped == 0).all())


def test_encoder_noise_is_reproducible():
    w = torch.full((32, 2), 5.0)
    a = quantize_encoder(w, 1 / 15.0, torch.zeros(32), generator=torch.Generator().manual_seed(3))
    b = quantize_encoder(w, 1 / 15.0, torch.zeros(32), generator=torch.Generator().manual_seed(3))
    assert torch.equal(a, b)
