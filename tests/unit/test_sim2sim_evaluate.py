"""Aggregation of the S8.4 evaluation records must be reproducible (owner ``[sim2sim]``).

Interpreter: numpy only. :mod:`duckiebot_rl.sim2sim.evaluate` imports ``mujoco`` lazily, inside the
environment constructor, so the summarizing half of the module is importable everywhere.

The defect these tests guard: ``run_condition`` collects worker results with ``imap_unordered``, so
the order of ``records`` varies from run to run, and ``_bootstrap_median_ci`` resampled with a fixed
index matrix. The multiset was order-invariant but the realized resample was not, so ``ci_low`` and
``ci_high`` moved between identical invocations of the same seeds over the same episodes. S8.4 makes
the confidence interval part of the headline C1-vs-C5 framing, and a number that changes when you
re-run the same command cannot go in a report.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from duckiebot_rl.sim2sim import evaluate as _evaluate  # noqa: E402


def _records(count: int = 60, seed: int = 0) -> list[dict[str, Any]]:
    """Build a synthetic set of episode records spanning three seeds.

    Args:
        count: how many records to build.
        seed: RNG seed for the synthetic metric values.

    Returns:
        A list of records shaped like the ones ``_run_block`` emits.
    """
    rng = np.random.default_rng(seed)
    out: list[dict[str, Any]] = []
    for index in range(count):
        record: dict[str, Any] = {
            "condition": "C5",
            "seed": index % 3,
            "episode": index,
            "reason": "timeout" if index % 4 else "off_drivable",
            "collisions": 0,
        }
        for key in _evaluate._METRIC_KEYS:
            record[key] = float(rng.normal(10.0, 3.0))
        record["success"] = float(index % 4 != 0)
        out.append(record)
    return out


def test_bootstrap_ci_is_invariant_to_sample_order() -> None:
    """The same sample in a different order gives the same confidence interval."""
    rng_values = np.random.default_rng(5).normal(3.0, 1.0, size=80).tolist()
    shuffled = list(rng_values)
    random.Random(1).shuffle(shuffled)

    first = _evaluate._bootstrap_median_ci(rng_values, np.random.default_rng(0))
    second = _evaluate._bootstrap_median_ci(shuffled, np.random.default_rng(0))
    assert first == second


def test_summarize_is_byte_identical_under_record_reordering() -> None:
    """Two shuffled copies of one set of episodes summarize identically, byte for byte.

    ``imap_unordered`` is why this matters: the same run really does produce the records in a
    different order every time.
    """
    records = _records()
    shuffled_a = list(records)
    shuffled_b = list(records)
    random.Random(2).shuffle(shuffled_a)
    random.Random(99).shuffle(shuffled_b)

    first = json.dumps(_evaluate.summarize(shuffled_a), sort_keys=True)
    second = json.dumps(_evaluate.summarize(shuffled_b), sort_keys=True)
    assert first == second

    # And re-running the same call twice is stable too, which is the property a report needs.
    assert first == json.dumps(_evaluate.summarize(records), sort_keys=True)


def test_summarize_reports_every_s8_4_metric() -> None:
    """All eleven reported metrics survive aggregation with a median, an IQR and a CI."""
    summary = _evaluate.summarize(_records())
    assert summary["episodes"] == 60
    assert summary["seeds"] == [0, 1, 2]
    for key in _evaluate._METRIC_KEYS:
        entry = summary["per_metric"][key]
        assert entry["q25"] <= entry["median"] <= entry["q75"]
        assert entry["ci_low"] <= entry["ci_high"]
    assert sum(summary["failure_reasons"].values()) == 60


def test_conditions_do_not_pin_the_episode_length() -> None:
    """No condition overrides ``episode_length_s``; the S8.4 cap has one owner, ``max_seconds``.

    ``run_condition`` writes ``max_seconds`` into the env config unconditionally and raises if a
    condition tries to disagree with it. This asserts the shipped conditions stay out of that fight.
    """
    for condition in _evaluate.CONDITIONS.values():
        assert "episode_length_s" not in condition.overrides


def test_c6_description_states_its_coverage_gaps() -> None:
    """C6 advertises exactly which S7.3 and S7.2 axes it cannot realize.

    A reader comparing C5 with C6 has to be able to see that the dynamics jitter is a subset, and
    which subset, without reading ``_apply_body_dr``.
    """
    description = _evaluate.CONDITIONS["C6"].description
    for token in ("V1-V8", "V13", "D10", "D11", "D14", "D15"):
        assert token in description, f"C6 does not declare its {token} coverage gap"


@pytest.mark.parametrize("values", [[], [float("nan")]])
def test_bootstrap_ci_of_an_empty_sample_is_nan(values: list[float]) -> None:
    """An all-NaN or empty metric yields NaN bounds rather than an exception."""
    low, high = _evaluate._bootstrap_median_ci(values, np.random.default_rng(0))
    assert low != low and high != high


def test_mujoco_spin_guard_uses_a_moving_anchor_like_the_isaac_one() -> None:
    """S8.3 parity: the twin's spin guard must not fire on a finished lap either.

    The Isaac guard and this one are separate implementations of the same S5.5 condition, so a
    fix to one is only half a fix. Anchoring to the spawn made both fire on success, and it was
    THIS copy that produced the measured artifact: 117 of 120 C5 episodes reported as "spin" at
    a median 0.964 laps and 3.3 cm lane RMS, which collapsed the headline transfer number from
    18.3 m (2.07 laps, 100% success) to 7.96 m. Asserting on the source keeps the two in step
    without booting a simulator.
    """
    from pathlib import Path

    source = Path("duckiebot_rl/sim2sim/env.py").read_text(encoding="utf-8")
    assert "_spin_ref_xy" in source, "the MuJoCo spin guard must use the moving anchor"
    assert "self._spin_ref_xy)" in source or "self._spin_ref_xy," in source, (
        "displacement must be measured from the moving anchor, not the spawn"
    )
    spin_block = source[source.index("_yaw_integral += abs") : source.index('return True, "spin"')]
    assert "_start_xy" not in spin_block, (
        "the spin guard still measures displacement from the spawn; on a closed loop that fires "
        "on lap completion, which is the regression this pins"
    )
