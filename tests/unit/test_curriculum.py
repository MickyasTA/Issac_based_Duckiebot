"""Unit tests for the two-scalar auto-DR curriculum (SPEC v2 S7.4) and the Range rule.

Covers: the ADR expand/hold/contract rule, clamping, exact serialization and restore (S6.9 makes
the curriculum a mandatory checkpoint field), boundary-probe assignment, the interpolation rule
that every DR axis is sampled through, and the hard-example miner.
"""

from __future__ import annotations

import math

import pytest
import torch

from duckiebot_rl.dr.curriculum import (
    BlockSampler,
    CurriculumCfg,
    HardExampleMiner,
    HardExampleMinerCfg,
    Range,
    TwoScalarADR,
    sample_book,
)


def _adr(**kw: float) -> TwoScalarADR:
    return TwoScalarADR(CurriculumCfg(**kw))


# ---------------------------------------------------------------------------- the ADR rule


def test_expands_on_success():
    adr = _adr()
    assert adr.alpha_vis == 0.0
    adr.record("vis", [10.0] * 30)
    assert adr.update()["vis"] == "expand"
    assert adr.alpha_vis == pytest.approx(0.02)
    assert adr.alpha_dyn == 0.0


def test_holds_between_the_thresholds():
    adr = _adr(init_alpha_vis=0.5)
    adr.record("vis", [6.0] * 30)
    assert adr.update()["vis"] == "hold"
    assert adr.alpha_vis == pytest.approx(0.5)


def test_contracts_on_failure():
    adr = _adr(init_alpha_dyn=0.5)
    adr.record("dyn", [1.0] * 30)
    assert adr.update()["dyn"] == "contract"
    assert adr.alpha_dyn == pytest.approx(0.48)


def test_waits_until_the_buffer_is_full():
    adr = _adr()
    adr.record("vis", [10.0] * 29)
    assert adr.update()["vis"] == "wait"
    assert adr.alpha_vis == 0.0
    adr.record("vis", [10.0])
    assert adr.update()["vis"] == "expand"


def test_buffer_is_cleared_after_an_update():
    adr = _adr()
    adr.record("vis", [10.0] * 30)
    adr.update()
    assert adr.metrics()["curriculum/buffer_vis"] == 0.0
    assert adr.update()["vis"] == "wait"


def test_alphas_are_clamped():
    adr = _adr(init_alpha_vis=0.99, init_alpha_dyn=0.01)
    for _ in range(5):
        adr.record("vis", [10.0] * 30)
        adr.record("dyn", [0.0] * 30)
        adr.update()
    assert adr.alpha_vis == pytest.approx(1.0)
    assert adr.alpha_dyn == pytest.approx(0.0)


def test_thresholds_are_the_spec_values():
    cfg = CurriculumCfg()
    assert (cfg.expand_threshold, cfg.contract_threshold) == (8.0, 4.0)
    assert (cfg.buffer_size, cfg.step, cfg.boundary_prob) == (30, 0.02, 0.1)


def test_unknown_scalar_raises():
    adr = _adr()
    with pytest.raises(KeyError):
        adr.record("visual", [1.0])
    with pytest.raises(KeyError):
        adr.alpha("visual")


# ---------------------------------------------------------------------------- probes


def test_assign_probes_is_mutually_exclusive_and_hits_the_target_rate():
    adr = _adr()
    g = torch.Generator().manual_seed(0)
    masks = adr.assign_probes(20000, generator=g)
    assert not bool((masks["vis"] & masks["dyn"]).any())
    rate = float((masks["vis"] | masks["dyn"]).float().mean())
    assert 0.08 < rate < 0.12
    assert 0.4 < float(masks["vis"].sum()) / float((masks["vis"] | masks["dyn"]).sum()) < 0.6


def test_assign_probes_is_deterministic():
    adr = _adr()
    a = adr.assign_probes(128, generator=torch.Generator().manual_seed(3))
    b = adr.assign_probes(128, generator=torch.Generator().manual_seed(3))
    assert torch.equal(a["vis"], b["vis"]) and torch.equal(a["dyn"], b["dyn"])


# ---------------------------------------------------------------------------- serialization


def test_state_dict_restores_exactly():
    adr = _adr()
    g = torch.Generator().manual_seed(1)
    adr.record("vis", [10.0] * 30)
    adr.update()
    adr.record("vis", [7.5, 9.25])
    adr.record("dyn", [3.0] * 12)
    state = adr.state_dict()

    restored = _adr()
    restored.load_state_dict(state)
    assert restored.state_dict() == state
    assert restored.alpha_vis == adr.alpha_vis
    assert restored.metrics() == adr.metrics()

    # And it keeps behaving identically afterwards.
    for obj in (adr, restored):
        obj.record("vis", [10.0] * 28)
    assert adr.update() == restored.update()
    assert adr.alpha_vis == restored.alpha_vis
    assert torch.equal(
        adr.assign_probes(16, generator=torch.Generator().manual_seed(5))["vis"],
        restored.assign_probes(16, generator=g.manual_seed(5))["vis"],
    )


def test_state_dict_is_json_compatible():
    import json

    adr = _adr()
    adr.record("dyn", [1.0, 2.0])
    assert json.loads(json.dumps(adr.state_dict())) == adr.state_dict()


def test_load_state_dict_rejects_missing_mandatory_fields():
    adr = _adr()
    with pytest.raises(KeyError, match="alpha"):
        adr.load_state_dict({"buffers": {"vis": [], "dyn": []}})
    with pytest.raises(KeyError, match="buffers"):
        adr.load_state_dict({"alpha": {"vis": 0.0, "dyn": 0.0}})


# ---------------------------------------------------------------------------- the Range rule


def test_range_collapses_to_nominal_at_alpha_zero():
    r = Range(0.5, 1.5, 1.0)
    t = r.sample(256, 0.0, generator=torch.Generator().manual_seed(0))
    assert torch.allclose(t, torch.ones(256))


def test_range_covers_the_clamps_at_alpha_one():
    r = Range(0.5, 1.5, 1.0)
    t = r.sample(20000, 1.0, generator=torch.Generator().manual_seed(0))
    assert float(t.min()) >= 0.5 and float(t.max()) <= 1.5
    assert float(t.min()) < 0.52 and float(t.max()) > 1.48


def test_range_interpolates_linearly():
    r = Range(0.0, 10.0, 2.0)
    assert r.live(0.0) == (2.0, 2.0)
    assert r.live(0.5) == pytest.approx((1.0, 6.0))
    assert r.live(1.0) == pytest.approx((0.0, 10.0))


def test_log_from_zero_axis_is_off_at_alpha_zero():
    r = Range(0.5 / 255, 10.0 / 255, 0.0, "log_from_zero")
    g = torch.Generator().manual_seed(0)
    assert torch.equal(r.sample(64, 0.0, generator=g), torch.zeros(64))
    t = r.sample(4096, 1.0, generator=g)
    assert float(t.min()) >= 0.5 / 255 - 1e-9
    assert float(t.max()) <= 10.0 / 255 + 1e-9


def test_log_axis_interpolates_multiplicatively():
    r = Range(0.5, 2.0, 1.0, "log")
    g = torch.Generator().manual_seed(0)
    assert torch.allclose(r.sample(32, 0.0, generator=g), torch.ones(32))
    t = r.sample(8192, 1.0, generator=g)
    assert float(t.min()) >= 0.5 - 1e-6 and float(t.max()) <= 2.0 + 1e-6
    half = r.sample(8192, 0.5, generator=g)
    assert float(half.min()) >= math.sqrt(0.5) - 1e-6
    assert float(half.max()) <= math.sqrt(2.0) + 1e-6


def test_int_axis_returns_long_and_covers_the_range():
    r = Range(1.0, 3.0, 2.0, "int")
    g = torch.Generator().manual_seed(0)
    t = r.sample(4096, 1.0, generator=g)
    assert t.dtype == torch.long
    assert set(t.unique().tolist()) == {1, 2, 3}
    assert set(r.sample(64, 0.0, generator=g).unique().tolist()) == {2}


def test_boundary_probe_returns_only_clamps():
    r = Range(0.5, 1.5, 1.0)
    g = torch.Generator().manual_seed(0)
    b = torch.ones(4096, dtype=torch.bool)
    t = r.sample(4096, 1.0, generator=g, boundary=b)
    assert torch.all(((t - 0.5).abs() < 1e-6) | ((t - 1.5).abs() < 1e-6))
    half = r.sample(4096, 0.5, generator=g, boundary=b)
    assert torch.all(((half - 0.75).abs() < 1e-6) | ((half - 1.25).abs() < 1e-6))


def test_range_validates_its_definition():
    with pytest.raises(ValueError, match="hi"):
        Range(2.0, 1.0, 1.5)
    with pytest.raises(ValueError, match="outside"):
        Range(0.0, 1.0, 5.0)
    with pytest.raises(ValueError, match="log mode needs lo"):
        Range(0.0, 1.0, 0.5, "log")
    Range(30.0, 95.0, 100.0, "linear", nominal_outside=True)


def test_sample_book_and_unknown_axis():
    book = {"a": Range(0.0, 1.0, 0.5), "b": Range(-1.0, 1.0, 0.0)}
    out = sample_book(book, ["a", "b"], 8, 1.0, generator=torch.Generator().manual_seed(0))
    assert set(out) == {"a", "b"} and out["a"].shape == (8,)
    with pytest.raises(KeyError, match="unknown DR axis"):
        sample_book(book, ["c"], 4, 1.0)


# ---------------------------------------------------------------------------- hard examples


def test_hard_example_miner_tracks_and_biases():
    m = HardExampleMiner(HardExampleMinerCfg(num_tiles=20, ema=0.5, hard_fraction=0.25))
    for _ in range(5):
        m.update(list(range(20)), [float(i) for i in range(20)])
    table = m.error_table
    assert table[19] > table[0]
    g = torch.Generator().manual_seed(0)
    tiles = m.sample_tiles(20000, generator=g)
    # The worst decile (2 of 20 tiles) should get ~25% + 10% * 75% of the draws.
    frac = float(((tiles == 19) | (tiles == 18)).float().mean())
    assert 0.28 < frac < 0.38


def test_hard_example_miner_serializes():
    m = HardExampleMiner(HardExampleMinerCfg(num_tiles=5))
    m.update([0, 1], [1.0, 2.0])
    state = m.state_dict()
    other = HardExampleMiner(HardExampleMinerCfg(num_tiles=5))
    other.load_state_dict(state)
    assert other.state_dict() == state
    with pytest.raises(KeyError, match="error"):
        other.load_state_dict({"count": [0] * 5})


def test_hard_example_miner_validates_input():
    m = HardExampleMiner(HardExampleMinerCfg(num_tiles=3))
    with pytest.raises(ValueError, match="equal length"):
        m.update([0, 1], [1.0])
    with pytest.raises(ValueError, match="outside"):
        m.update([7], [1.0])


# =============================================================================================
# BlockSampler (profile rank 4)
# =============================================================================================


def _mixed_book() -> dict[str, Range]:
    """Return a book exercising all four sampling modes at once.

    Returns:
        ``{axis_name: Range}``.
    """
    return {
        "lin": Range(-1.0, 1.0, 0.0, "linear"),
        "lin2": Range(0.7, 1.5, 1.0, "linear"),
        "logz": Range(0.5 / 255.0, 10.0 / 255.0, 0.0, "log_from_zero"),
        "logm": Range(0.3, 3.0, 1.0, "log"),
        "idx": Range(0.0, 5.0, 0.0, "int"),
        "outside": Range(30.0, 95.0, 100.0, "linear", nominal_outside=True),
    }


@pytest.mark.parametrize("alpha", [0.0, 0.33, 1.0])
def test_block_sampler_matches_range_sample_given_the_same_uniforms(alpha):
    """Every mode, bit for bit. The grouping must not change one ulp of the arithmetic."""
    book = _mixed_book()
    block = BlockSampler(book, list(book), alpha)
    u = block.draw(64, generator=torch.Generator().manual_seed(31))
    out = block.transform(u)
    for row, name in enumerate(block.order):
        assert torch.equal(out[name], book[name]._sample_scalar_alpha(u[row], alpha)), name


def test_block_sampler_order_is_a_permutation_of_the_requested_keys():
    """Rows are grouped by mode so each group is a contiguous slice; nothing may be lost."""
    book = _mixed_book()
    block = BlockSampler(book, list(book), 1.0)
    assert sorted(block.order) == sorted(book)
    assert block.num_axes == len(book)


def test_block_sampler_preserves_int_dtype():
    """An ``int`` axis indexes an HDRI table; a float would be a silent wrong texture."""
    book = _mixed_book()
    block = BlockSampler(book, list(book), 1.0)
    out = block.sample(32, generator=torch.Generator().manual_seed(2))
    assert out["idx"].dtype == torch.long
    assert out["lin"].dtype == torch.float32
    assert int(out["idx"].min()) >= 0 and int(out["idx"].max()) <= 5


def test_block_sampler_boundary_forces_the_live_clamps():
    """Same ADR rule as ``Range.sample``, drawn independently per axis."""
    book = _mixed_book()
    alpha = 0.6
    block = BlockSampler(book, list(book), alpha)
    out = block.sample(
        1024, generator=torch.Generator().manual_seed(8), boundary=torch.ones(1024, dtype=torch.bool)
    )
    lo, hi = book["lin"].live(alpha)
    on_edge = torch.isclose(out["lin"], torch.tensor(lo)) | torch.isclose(out["lin"], torch.tensor(hi))
    assert bool(on_edge.all())
    assert 0.35 < float((out["lin"] > 0.5 * (lo + hi)).float().mean()) < 0.65


def test_block_sampler_draw_is_one_call_per_uniform_block():
    """The whole point: 17 axes cost one ``torch.rand``, or two with a boundary mask."""
    book = _mixed_book()
    block = BlockSampler(book, list(book), 1.0)
    assert block.draw(8, generator=torch.Generator().manual_seed(0)).shape == (len(book), 8)


def test_block_sampler_matches_reports_a_stale_cache():
    """``VisualDR`` caches one of these; the ADR loop moves alpha and must invalidate it."""
    book = _mixed_book()
    block = BlockSampler(book, list(book), 0.5)
    assert block.matches(list(book), 0.5, None) is True
    assert block.matches(list(book), 0.6, None) is False
    assert block.matches(list(book)[:-1], 0.5, None) is False


def test_block_sampler_rejects_an_unknown_axis():
    """A typo in a range book must fail at construction, not produce a silently missing axis."""
    with pytest.raises(KeyError, match="nope"):
        BlockSampler(_mixed_book(), ["lin", "nope"], 1.0)


def test_block_sampler_transform_rejects_a_mis_shaped_block():
    """A row-count mismatch would silently pair axes with the wrong uniforms."""
    book = _mixed_book()
    block = BlockSampler(book, list(book), 1.0)
    with pytest.raises(ValueError, match="rows"):
        block.transform(torch.rand(len(book) - 1, 8))
