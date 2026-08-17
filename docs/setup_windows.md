# Windows setup

This project runs on **Windows 11 only**. There is no Linux path and no WSL fallback: that is a
project decision, not an omission. Every script, path and command in this repository is written
for Windows, and CI additionally proves that the CPU-only half works on ubuntu-latest.

Target machine for all numbers in this repository:

| Component | Value |
|---|---|
| OS | Windows 11 Pro 10.0.26200 |
| GPU | NVIDIA RTX 3080 Laptop, 8 GB VRAM (hard constraint) |
| Python | 3.11.4 |
| Isaac Sim | 5.1.0 |
| Isaac Lab | 2.3.2.post1 |
| PyTorch | 2.7.0+cu128 |
| MuJoCo | 3.11.0, separate virtual environment |

---

## 1. The two virtual environments

This repository does **not** create or manage the Isaac environment, and `isaacsim` and
`isaaclab` are deliberately **not** dependencies in `pyproject.toml`. They are multi-gigabyte
CUDA wheels pinned to a single Kit build; listing them would break every CPU-only job (CI, unit
tests, MuJoCo, ONNX export) for no benefit. All Isaac imports in this codebase are guarded and
raise an actionable error when the environment is wrong.

| Environment | Path on the development machine | Holds | Used for |
|---|---|---|---|
| Isaac venv | `d:/Personal/personal/wheeled_quadruped_robot/.venv` | Isaac Sim 5.1.0, Isaac Lab 2.3.2.post1, torch 2.7.0+cu128, gymnasium 1.2.0 | training, evaluation in Isaac, ONNX export |
| MuJoCo / tools venv | `d:/Personal/personal/mujoco_venv` | mujoco 3.11, numpy, glfw, PyOpenGL; plus what setup adds: CPU torch, opencv-python-headless, Pillow, onnxruntime, usd-core | sim-to-sim, offline USD authoring, texture generation, preprocessing parity |

Convenient shell variables:

```powershell
$ISAAC  = "d:/Personal/personal/wheeled_quadruped_robot/.venv/Scripts/python.exe"
$TOOLS  = "d:/Personal/personal/mujoco_venv/Scripts/python.exe"
```

Two facts worth knowing before you lose an afternoon to them:

* **`pxr` is not importable from the Isaac venv**, even after `import isaacsim`. The USD python
  bindings that ship inside Isaac Sim need a full Kit boot. Offline USD authoring therefore uses
  a plain `usd-core` pip install in the tools venv, which is why the city generator runs there
  and not in the Isaac venv.
* **Isaac cloud asset paths moved in 5.1.** Robots now live under a vendor directory:
  `Isaac/Robots/NVIDIA/Jetbot/jetbot.usd` resolves, while the 4.5-era
  `Isaac/Robots/Jetbot/jetbot.usd` returns 404. The asset root is
  `https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1`.

---

## 2. The GPU TDR watchdog (do this before any long run)

This is the single most important Windows-specific setting in the project.

Windows runs a **Timeout Detection and Recovery** watchdog on the display driver. If one GPU
operation does not return within `TdrDelay` seconds (default **2**), Windows resets the driver.
A large Isaac Lab stage build, a heavy render step at 256 environments, or a texture upload
spike can all exceed two seconds. When the reset fires, the training process dies with a CUDA
error that looks exactly like a bug in the training code, usually several hours into a run.

Raise both timeouts to 60 seconds:

```powershell
# ELEVATED PowerShell
$key = "HKLM:\SYSTEM\CurrentControlSet\Control\GraphicsDrivers"
New-ItemProperty -Path $key -Name "TdrDelay"    -PropertyType DWord -Value 60 -Force
New-ItemProperty -Path $key -Name "TdrDdiDelay" -PropertyType DWord -Value 60 -Force
```

Verify, then **reboot**. The change does not take effect until you do:

```powershell
reg query "HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers" /v TdrDelay
# TdrDelay    REG_DWORD    0x3c        (60)
```

If a run dies anyway, check the Windows event log for display-driver reset events
(`Event Viewer -> Windows Logs -> System`, source `Display` or `nvlddmkm`) before assuming the
fault is in the code.

Setting these to 0 disables the watchdog entirely. Do not do that: a genuinely hung kernel then
requires a hard power cycle.

---

## 3. Sleep, hibernate and display timeouts

A machine that suspends during a multi-day run loses the CUDA context. Disable standby and
hibernate for both AC and battery:

```powershell
powercfg /change standby-timeout-ac 0
powercfg /change standby-timeout-dc 0
powercfg /change hibernate-timeout-ac 0
powercfg /change hibernate-timeout-dc 0
powercfg /change monitor-timeout-ac 0
```

Also worth doing before a multi-day run:

* Set Windows Update to a fixed active-hours window, or pause updates. An automatic restart
  ends the run.
* Use the high-performance power plan on AC and keep the laptop on mains.
* Keep the machine physically ventilated. The 8 GB VRAM budget is tight enough that a thermal
  throttle changes throughput numbers and makes benchmark rows non-comparable.

---

## 4. One-shot setup script

Everything above, plus the package installs, is scripted:

```powershell
# From an ELEVATED PowerShell, in the repository root
.\scripts\setup_windows.ps1

# Without administrator rights: packages only, no registry or power changes
.\scripts\setup_windows.ps1 -SkipRegistry -SkipPower
```

The script installs into the Isaac venv: `onnx`, `onnxruntime`, `ruff`, `pytest`. The critic
found that `onnxruntime` was missing from the Isaac venv, which silently disabled the export
parity gate, so it is now installed explicitly and is a declared extra of this package.

It installs into the tools venv: CPU `torch` (from the PyTorch CPU index, not the CUDA one),
`opencv-python-headless`, `pillow`, `onnxruntime`, `usd-core`, `pyyaml`.

---

## 5. Install this package

```powershell
cd d:/Personal/personal/Issac_based_Duckiebot
& $ISAAC -m pip install -e ".[dev,export,cv]"
```

Available extras:

| Extra | Contents | Where |
|---|---|---|
| `dev` | pytest, pytest-cov, ruff, pre-commit, gymnasium | both venvs |
| `export` | onnx, onnxruntime | Isaac venv (and the robot image) |
| `mujoco` | mujoco, glfw, PyOpenGL | tools venv |
| `usd` | usd-core | tools venv |
| `cv` | opencv-python-headless | tools venv, deploy tests |

---

## 6. Acceptance check (milestone M0)

All of these must pass before any training starts:

```powershell
# 1. The licensing gate
& $ISAAC scripts/check_clean_room.py --verbose        # exit code 0

# 2. Tools venv has everything the offline pipeline needs
& $TOOLS -c "import torch, cv2, PIL, onnxruntime, pxr; print('tools venv OK')"

# 3. Isaac venv can export and verify ONNX
& $ISAAC -c "import onnxruntime; print(onnxruntime.__version__)"

# 4. The TDR change is live (after a reboot)
reg query "HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers" /v TdrDelay

# 5. The CPU test suite, which is exactly what CI runs
& $ISAAC -m pytest tests/unit -q
```

---

## 7. Running training

```powershell
& $ISAAC scripts/train.py --task Duckiebot-LaneFollow-v0 --num_envs 256 --headless --enable_cameras
```

Notes that save time:

* Launch through the Isaac venv python directly. Do not use `isaaclab.bat`.
* `--enable_cameras` is mandatory for any run that uses the tiled camera.
* There is deliberately **no** `--rendering_mode` flag. The environment configuration is the
  single source of truth for rendering, because a CLI override that silently changes the
  renderer between a benchmark and a training run makes every VRAM number meaningless.
* Expect 4,000 to 10,000 environment steps per second at 256 environments, including the
  re-render overhead from environment resets. That is 6 to 10 hours for a 150M-step
  lane-following run and 15 to 28 hours for a 400M-step run with obstacles.
* Watch VRAM with `nvidia-smi`, never with `torch.cuda.memory_allocated`. The Kit renderer's
  allocations are invisible to torch, and the budget gate is 7.2 GiB by `nvidia-smi` at 256
  environments.

Resume after any interruption:

```powershell
& $ISAAC scripts/train.py --task Duckiebot-LaneFollow-v0 --num_envs 256 --headless --enable_cameras --resume training_results/<run_id>
# or, for the newest run under training_results/:
& $ISAAC scripts/train.py --task Duckiebot-LaneFollow-v0 --num_envs 256 --headless --enable_cameras --resume latest
```

The checkpoint restores the learner exactly (bit-identical on CPU, asserted by a test) and the
environment stream statistically. Curriculum scalars and randomization state are mandatory
fields in the checkpoint: without them a resume would restart domain randomization from zero
with no visible symptom.

---

## 8. Deployment toolchain (offline)

There is no robot in this project. The export path is exercised entirely offline:

```powershell
& $ISAAC scripts/export_policy.py --checkpoint checkpoints/best.pt --out-dir exports/
```

That writes `policy_opset13.onnx` (TensorRT 8.2, Jetson Nano, static batch 1),
`policy_opset18.onnx` (TensorRT 10, Orin Nano), a JSON sidecar for each with sha256,
preprocessing constants, action units and training provenance, plus `policy_traced.pt`, the
TorchScript artifact that drives the Isaac and MuJoCo evaluations so that no evaluation number
can come from a different forward pass than the one that was exported.

The TensorRT engine builds are **documented, not run**, and are recorded in the sidecars:

```bash
# On a Jetson Nano (JetPack 4.6.x, TensorRT 8.2.1.8), not executed in this project:
sudo nvpmodel -m 0 && sudo jetson_clocks
trtexec --onnx=policy_opset13.onnx --fp16 --workspace=512 --saveEngine=policy_nano.plan

# On an Orin Nano (JetPack 6.x, TensorRT 10.x):
trtexec --onnx=policy_opset18.onnx --fp16 --saveEngine=policy_orin.plan
```

FP16 only, no INT8. The first-hardware-contact gate, whenever hardware exists, is 1000 real
frames compared torch-fp32 against TensorRT-fp16 with a maximum absolute difference below 1e-2.

---

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Run dies with a CUDA error after hours, driver reset in the event log | TDR watchdog | Section 2, then reboot |
| Run dies overnight, machine was idle | Standby or hibernate | Section 3 |
| `ModuleNotFoundError: pxr` in the Isaac venv | USD bindings need a Kit boot | Run USD authoring in the tools venv with `usd-core` |
| 404 on an Isaac cloud asset path | Paths moved to vendor directories in 5.1 | Use `Isaac/Robots/NVIDIA/...` |
| Throughput collapses more than 2x between two benchmark points | VRAM spilling to system memory through WDDM | Treat as out-of-memory; drop the environment count or the minibatch size |
| `onnxruntime` missing | Not part of the Isaac install | `& $ISAAC -m pip install onnxruntime` |
| ONNX export prints a Unicode error on Windows | The exporter writes emoji to a cp1252 console | Already handled: the exporter switches the console to replacement characters for the duration |
