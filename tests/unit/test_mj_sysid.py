r"""The sysid self-test must reproduce the numbers its own documentation reports (SPEC v2 S8.2).

Interpreter: needs ``mujoco`` and ``numpy``. Run with::

    d:/Personal/personal/mujoco_venv/Scripts/python.exe -m pytest tests/unit/test_mj_sysid.py \\
        --run-mujoco -q

The defect: the module docstring's results table quoted measurements the shipped code does not
produce, all of them in the optimistic direction (``kv`` within 4.6% where the code gives -7.142%,
armature 6.1e-4 where it gives 4.338e-4, a 1.3 mm straight-line endpoint where it gives 20.31 mm),
and it described the perturbation as "friction x2.5" where the CLI applies x0.4. For a public repo
whose credibility rests on its numbers being reproducible from the checked-in code, that is the
worst class of defect, so the table is now generated and this file is what keeps it generated:

* :data:`ARTIFACT` is written by ``scripts/eval_sim2sim.py sysid --out`` and checked in;
* the first test re-runs the shipped self-test and asserts it still produces those numbers;
* the second test asserts the module docstring still quotes them.

Change the code and one of the three has to be regenerated, deliberately.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

pytestmark = pytest.mark.mujoco
pytest.importorskip("mujoco", reason="run these with the tools venv (mujoco_venv)")

from duckiebot_rl.sim2sim import sysid as _sysid  # noqa: E402

ARTIFACT = _REPO_ROOT / _sysid.SELFTEST_ARTIFACT

#: Numbers the module docstring states, and the tolerance each is asserted at. The tolerances are
#: loose enough to survive a MuJoCo patch release and far tighter than the discrepancies that made
#: this file necessary (the old table was wrong by factors of 1.4 to 15).
DOCUMENTED = {
    "r_eff_pct": (-0.030, 0.01),
    "b_eff_pct": (0.045, 0.01),
    "kv_pct": (-7.142, 0.05),
    "armature": (4.338e-4, 2e-6),
    "frictionloss": (9.645e-3, 5e-5),
    "endpoint_straight_mm": (20.31, 0.1),
    "endpoint_arc_mm": (301.39, 2.0),
}


@pytest.fixture(scope="module")
def result() -> _sysid.SysIdResult:
    """Run the shipped synthetic self-test once for this module."""
    return _sysid.run_selftest()[0]


def test_the_artifact_is_checked_in_and_readable() -> None:
    """``docs/sim2sim_sysid_selftest.json`` exists, so the documented table has a source."""
    assert ARTIFACT.is_file(), (
        f"{ARTIFACT} is missing. Regenerate it with: scripts/eval_sim2sim.py sysid --out "
        f"{_sysid.SELFTEST_ARTIFACT}"
    )
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert payload["accepted"] is False
    assert set(payload["stage2"]) == {"armature", "frictionloss", "damping"}


def test_the_self_test_still_produces_the_documented_numbers(result: _sysid.SysIdResult) -> None:
    """Re-running the shipped self-test reproduces every number in the module docstring."""
    measured = {
        "r_eff_pct": result.stage1_error_pct["r_eff"],
        "b_eff_pct": result.stage1_error_pct["b_eff"],
        "kv_pct": result.stage1_error_pct["kv"],
        "armature": result.stage2["armature"],
        "frictionloss": result.stage2["frictionloss"],
        "endpoint_straight_mm": result.residuals["endpoint_straight"]["final_pos_mm"],
        "endpoint_arc_mm": result.residuals["endpoint_arc"]["final_pos_mm"],
    }
    for key, (expected, tolerance) in DOCUMENTED.items():
        assert measured[key] == pytest.approx(expected, abs=tolerance), (
            f"the self-test now measures {key} = {measured[key]!r}, but the sysid module docstring "
            f"and {_sysid.SELFTEST_ARTIFACT} say {expected!r}. Regenerate both, and state what "
            f"changed; do not adjust the tolerance."
        )


def test_the_checked_in_artifact_matches_a_fresh_run(result: _sysid.SysIdResult) -> None:
    """The artifact is regenerable: a fresh run reproduces the file that is checked in."""
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert payload["stage2"]["armature"] == pytest.approx(result.stage2["armature"], rel=1e-6)
    assert payload["stage2"]["frictionloss"] == pytest.approx(result.stage2["frictionloss"], rel=1e-6)
    assert payload["endpoint_error_mm"] == pytest.approx(result.endpoint_error_mm, rel=1e-6)
    assert payload["accepted"] == result.accepted


def test_the_docstring_states_the_numbers_it_measures() -> None:
    """Every documented figure appears verbatim in the module docstring.

    This is the anti-drift assertion. If the code changes and the table is not regenerated, the
    previous test fails; if the table is edited by hand into something the code does not produce,
    this one fails.
    """
    doc = _sysid.__doc__ or ""
    for token in ("-0.030%", "+0.045%", "-7.142%", "4.338e-4", "9.645e-3", "20.31 mm", "301.39 mm"):
        assert token in doc, f"the sysid module docstring no longer states {token}"
    assert "not yet met" in doc, "the docstring must say plainly that the S8.2 gate is not met"


def test_the_docstring_states_the_perturbation_the_code_applies() -> None:
    """The described experiment is the executed one, down to the joint-friction direction.

    The old text said "joint friction x2.5" while the code applied x0.4, so a reader could not
    reproduce the run even in principle.
    """
    doc = _sysid.__doc__ or ""
    perturbation = _sysid.SELFTEST_PERTURBATION
    assert perturbation["frictionloss"] == pytest.approx(0.004)
    assert perturbation["armature"] == pytest.approx(6.0e-4)
    assert perturbation["kv"] == pytest.approx(0.042)
    assert "x1.03" in doc and "x0.965" in doc
    assert "4.0e-3" in doc and "x0.4" in doc, "the docstring must state the friction scale applied"
    assert "x2.5" not in doc, "the x2.5 joint-friction claim is not what the code applies"


def test_acceptance_is_reported_honestly(result: _sysid.SysIdResult) -> None:
    """The self-test does not pass S8.2, and says so rather than quoting the parts that do."""
    assert result.accepted is False
    assert result.endpoint_error_mm > 25.0
    assert result.dr_coverage["armature"]["covered"] is False


def test_the_self_test_is_deterministic() -> None:
    """Two runs of the same self-test give identical parameters, so the table is reproducible."""
    first = _sysid.run_selftest(outer_passes=1, max_iter=10)[0]
    second = _sysid.run_selftest(outer_passes=1, max_iter=10)[0]
    assert first.stage2 == second.stage2
    assert first.stage1_error_pct == second.stage1_error_pct
