# Contributing

Thank you for looking. This is a portfolio and research repository, so the bar is less about
process and more about not breaking the two things that make it worth reading: the clean-room
asset position and the from-scratch reinforcement-learning implementation.

## Ground rules

1. **No reinforcement-learning libraries.** Stable-Baselines3, rl_games, skrl, tianshou and
   friends are forbidden anywhere in the learning path. The whole point is that PPO here is
   readable, testable and written from scratch. Pure PyTorch only.
2. **No Duckietown assets, ever.** No mesh, texture or scene file from `gym-duckietown`,
   `duckietown-world`, `dt-core` or any other upstream repository, converted or not. Dimensional
   facts from public documentation are fine and must be cited. `scripts/check_clean_room.py`
   enforces this and runs in CI and in pre-commit; if it fails, that is a licensing problem, not
   a style problem.
3. **Windows first.** Every script must run on Windows 11. No `os.uname`, no `/tmp`, no
   bash-only tricks, forward slashes in paths. CI additionally runs the CPU half on Linux.
4. **No stubs.** No `TODO`, no `pass  # implement later`. If it is committed, it runs and it is
   tested.
5. **No hardware claims.** There is no physical robot in this project. Anything that would need
   one is labelled a design target, never a measurement.

## Environment

See [docs/setup_windows.md](docs/setup_windows.md) for the full setup, including the GPU TDR
registry change that every multi-day run depends on.

```powershell
$ISAAC = "d:/Personal/personal/wheeled_quadruped_robot/.venv/Scripts/python.exe"
& $ISAAC -m pip install -e ".[dev,export,cv]"
& $ISAAC -m pre_commit install
```

## Style

* Python 3.11. Type hints on every function, including return types.
* Google-style docstrings on every module, class and public function.
* Ruff clean at line length 110: `ruff check .` and `ruff format --check .` both pass.
* Deterministic and seedable: seed `torch`, `numpy` **and** the standard-library `random`.
* Comments explain why, not what. A comment that restates the code is noise; a comment that
  records the failure mode a line prevents is the reason the line survives review.

## Tests

The default test run is CPU-only, simulator-free, and is exactly what CI runs.

```powershell
$TOOLS = "d:/Personal/personal/mujoco_venv/Scripts/python.exe"

& $ISAAC -m pytest tests/unit -q                          # default
& $ISAAC -m pytest --run-all tests/                       # everything: needs Isaac, a GPU, MuJoCo
& $TOOLS -m pytest --run-mujoco tests/unit/test_mj*.py    # the MuJoCo half, in the MuJoCo venv
```

The MuJoCo tests are the one group that is not run from the Isaac venv: `mujoco` is installed in
a separate environment that deliberately has no torch, so those modules are selected by path
rather than by asking the Isaac venv to collect a suite it cannot import.

Markers, all deselected by default:

| Marker | Needs | Flag |
|---|---|---|
| `isaac` | Isaac Sim 5.1 and Isaac Lab | `--run-isaac` |
| `gpu` | a CUDA device | `--run-gpu` |
| `mujoco` | the MuJoCo venv | `--run-mujoco` |
| `slow` | more than about 30 s | `--runslow` |

Prefer tests that need neither a GPU nor a simulator. A test that only runs on the author's
machine protects nothing.

## Ownership of shared files

Three modules are consumed by training, sim-to-sim and deployment at once, and a one-line change
in any of them moves the sim-to-real gap without moving a single test:

* `duckiebot_rl/dr/preprocess.py`
* `duckiebot_rl/camera_math.py`
* `duckiebot_rl/wrappers/delay.py`

Changes there require agreement from the PPO, sim-to-sim and deployment owners, and must come
with the parity tests updated in the same commit.

## Commits and pull requests

* Present tense, imperative subject line, under 72 characters.
* One logical change per commit.
* Commit messages must **not** contain `Co-Authored-By` trailers or any generated-with footer.
* A pull request should say what it changes, which specification section it implements or
  contradicts, and what evidence exists that it works (ideally the test that would have caught
  the bug).

## Reporting a problem

Open an issue with the exact command, the full traceback, the output of

```powershell
& $ISAAC -c "import sys, torch; print(sys.version); print(torch.__version__, torch.cuda.is_available())"
```

and, for anything that ran on the GPU, the `nvidia-smi` output at the time. For a training run
that died overnight, check the Windows event log for display-driver reset events first: that is
the TDR watchdog, and it is documented in the setup guide.
