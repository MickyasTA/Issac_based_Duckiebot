"""Pytest configuration shared by every test in the repository.

Three things happen here:

1. The repository root is put on ``sys.path`` so that ``duckiebot_rl`` imports from a fresh
   clone without an editable install. CI installs the package properly; developers should not
   have to.
2. The marker vocabulary is registered and enforced. The default test run is the CPU-only,
   Isaac-free, GPU-free, MuJoCo-free subset, which is exactly what runs in CI. Anything that
   needs a simulator, a GPU or minutes of wall-clock is marked and skipped unless explicitly
   opted into.
3. ``DUCKIEBOT_RL_STRICT_FP32`` is exported for every test session (SPEC v2 S6.5): TF32 is
   disabled and the PPO ratio assert tightens from 5e-3 to 1e-5. The M4 gate runs in this mode,
   so the tests do too.
4. The ``count_host_syncs`` fixture is published, so that any module can assert that a hot path
   performs no device-to-host synchronisation. The M-phase profile found 23.4 synchronising
   calls per control step in a rollout whose GPU sat at 12% utilisation, and several of the
   optimisations that removed them are one careless ``.item()`` away from coming back.

Opt-in flags::

    pytest --run-isaac     also run tests marked `isaac`  (needs Isaac Sim 5.1 + Isaac Lab)
    pytest --run-gpu       also run tests marked `gpu`    (needs a CUDA device)
    pytest --run-mujoco    also run tests marked `mujoco` (needs the mujoco package)
    pytest --runslow       also run tests marked `slow`
    pytest --run-all       all of the above
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Marker name -> (command-line flag, skip reason when the flag is absent).
_OPT_IN_MARKERS: dict[str, tuple[str, str]] = {
    "isaac": ("--run-isaac", "needs Isaac Sim 5.1 / Isaac Lab; run with --run-isaac"),
    "gpu": ("--run-gpu", "needs a CUDA GPU; run with --run-gpu"),
    "mujoco": ("--run-mujoco", "needs the MuJoCo venv; run with --run-mujoco"),
    "slow": ("--runslow", "slow test; run with --runslow"),
}

_MARKER_HELP = {
    "isaac": "needs Isaac Sim 5.1 / Isaac Lab (never runs in CI)",
    "gpu": "needs a CUDA GPU (never runs in CI)",
    "mujoco": "needs the MuJoCo venv and the mujoco package",
    "slow": "takes more than ~30 s",
}


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the opt-in flags."""
    group = parser.getgroup("duckiebot-rl")
    for marker, (flag, _) in _OPT_IN_MARKERS.items():
        group.addoption(
            flag,
            action="store_true",
            default=False,
            help=f"run tests marked '{marker}' ({_MARKER_HELP[marker]})",
        )
    group.addoption(
        "--run-all",
        action="store_true",
        default=False,
        help="run every opt-in marker: isaac, gpu, mujoco, slow",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register markers and pin the numeric mode for the whole session."""
    for marker, description in _MARKER_HELP.items():
        config.addinivalue_line("markers", f"{marker}: {description}")
    os.environ.setdefault("DUCKIEBOT_RL_STRICT_FP32", "1")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip opt-in markers unless their flag (or --run-all) was given."""
    run_all = config.getoption("--run-all")
    for marker, (flag, reason) in _OPT_IN_MARKERS.items():
        if run_all or config.getoption(flag):
            continue
        skip = pytest.mark.skip(reason=reason)
        for item in items:
            if marker in item.keywords:
                item.add_marker(skip)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path of the repository root.

    Returns:
        The repository root directory.
    """
    return _REPO_ROOT


@pytest.fixture(scope="session")
def strict_fp32() -> bool:
    """Whether the session runs in strict fp32 mode (TF32 disabled).

    Returns:
        True when ``DUCKIEBOT_RL_STRICT_FP32`` is set to a truthy value.
    """
    return os.environ.get("DUCKIEBOT_RL_STRICT_FP32", "0") not in ("", "0", "false", "False")


@pytest.fixture
def count_host_syncs() -> Callable[[], object]:
    """Return a context manager that counts device-to-host synchronising tensor calls.

    Usage::

        with count_host_syncs() as syncs:
            hot_path()
            assert syncs() == 0

    On a CPU-only runner none of the counted calls actually stalls anything, which is precisely
    why counting the CALL is the right instrument: the call site is identical on CUDA, where it
    costs a pipeline stall, and CI has no GPU. ``torch.Tensor`` attributes are restored on exit
    even if the body raises.

    Returns:
        A zero-argument context-manager factory yielding a counter callable.
    """
    names = ("item", "tolist", "cpu", "numpy", "__bool__", "__float__", "__int__", "nonzero")

    @contextmanager
    def counter() -> Iterator[Callable[[], int]]:
        """Patch the synchronising methods for the duration of the block.

        Yields:
            A callable returning the number of synchronising calls made so far.
        """
        originals = {name: getattr(torch.Tensor, name) for name in names}
        total = [0]

        def wrap(fn: Callable) -> Callable:
            """Return ``fn`` wrapped in a call counter.

            Args:
                fn: The original ``torch.Tensor`` method.

            Returns:
                The counting wrapper.
            """

            def wrapper(*args: object, **kwargs: object) -> object:
                total[0] += 1
                return fn(*args, **kwargs)

            return wrapper

        for name, fn in originals.items():
            setattr(torch.Tensor, name, wrap(fn))
        try:
            yield lambda: total[0]
        finally:
            for name, fn in originals.items():
                setattr(torch.Tensor, name, fn)

    return counter
