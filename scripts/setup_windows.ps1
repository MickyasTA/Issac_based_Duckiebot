<#
.SYNOPSIS
    One-shot Windows setup for the duckiebot-rl project (SPEC v2 milestone M0).

.DESCRIPTION
    This project runs on Windows 11 only. There is no Linux or WSL fallback. Two things make a
    multi-day GPU training run fail on Windows that never come up on Linux, and both are fixed
    here:

      1. The TDR watchdog. Windows resets the display driver when a single GPU operation takes
         longer than TdrDelay seconds (default 2). A large Isaac Lab stage build or a heavy
         render step can exceed that, and the reset kills the run with a CUDA error that looks
         like a bug in the training code. Raising TdrDelay and TdrDdiDelay to 60 s is the
         standard mitigation. This requires administrator rights and a reboot.

      2. Sleep and hibernate. A machine that suspends mid-run loses the CUDA context. Standby
         and hibernate are disabled for both AC and DC power.

    It also installs the CPU-side packages that the specification requires in each virtual
    environment. Isaac Sim and Isaac Lab are never installed by this script: they are already
    present in the Isaac venv and are not pip dependencies of this project.

.PARAMETER SkipRegistry
    Do not touch the registry. Use this when running without administrator rights; the package
    installation still happens.

.PARAMETER SkipPower
    Do not change the power plan.

.PARAMETER IsaacPython
    Full path to the Isaac Sim virtual environment python.

.PARAMETER MujocoPython
    Full path to the MuJoCo (tools) virtual environment python.

.EXAMPLE
    # Run from an elevated PowerShell in the repository root:
    .\scripts\setup_windows.ps1

.EXAMPLE
    # Non-elevated: install packages only.
    .\scripts\setup_windows.ps1 -SkipRegistry -SkipPower

.NOTES
    Verify afterwards, then REBOOT for the TDR change to take effect:
        reg query "HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers" /v TdrDelay
#>

[CmdletBinding()]
param(
    [switch]$SkipRegistry,
    [switch]$SkipPower,
    [string]$IsaacPython = "d:/Personal/personal/wheeled_quadruped_robot/.venv/Scripts/python.exe",
    [string]$MujocoPython = "d:/Personal/personal/mujoco_venv/Scripts/python.exe"
)

$ErrorActionPreference = "Stop"

function Write-Step($message) {
    Write-Host ""
    Write-Host "=== $message ===" -ForegroundColor Cyan
}

function Test-Administrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# ------------------------------------------------------------------------------------------
# 1. TDR watchdog
# ------------------------------------------------------------------------------------------
if (-not $SkipRegistry) {
    Write-Step "GPU TDR watchdog (HKLM GraphicsDrivers)"
    if (-not (Test-Administrator)) {
        Write-Warning "Not running as administrator. Skipping the registry change."
        Write-Warning "Re-run from an elevated PowerShell, or pass -SkipRegistry to silence this."
    }
    else {
        $key = "HKLM:\SYSTEM\CurrentControlSet\Control\GraphicsDrivers"
        New-ItemProperty -Path $key -Name "TdrDelay"    -PropertyType DWord -Value 60 -Force | Out-Null
        New-ItemProperty -Path $key -Name "TdrDdiDelay" -PropertyType DWord -Value 60 -Force | Out-Null
        $tdrDelay = (Get-ItemProperty -Path $key -Name "TdrDelay").TdrDelay
        $tdrDdi = (Get-ItemProperty -Path $key -Name "TdrDdiDelay").TdrDdiDelay
        Write-Host "TdrDelay    = $tdrDelay"
        Write-Host "TdrDdiDelay = $tdrDdi"
        Write-Warning "A REBOOT is required before these take effect."
    }
}

# ------------------------------------------------------------------------------------------
# 2. Sleep and hibernate
# ------------------------------------------------------------------------------------------
if (-not $SkipPower) {
    Write-Step "Power settings (no standby, no hibernate, no display timeout)"
    powercfg /change standby-timeout-ac 0
    powercfg /change standby-timeout-dc 0
    powercfg /change hibernate-timeout-ac 0
    powercfg /change hibernate-timeout-dc 0
    powercfg /change monitor-timeout-ac 0
    Write-Host "standby and hibernate timeouts set to 0 (never)"
}

# ------------------------------------------------------------------------------------------
# 3. Python packages
# ------------------------------------------------------------------------------------------
function Install-Packages($python, $label, $packages) {
    Write-Step "$label : $python"
    if (-not (Test-Path $python)) {
        Write-Warning "$label python not found at $python. Skipping."
        return
    }
    & $python -m pip install --upgrade pip
    & $python -m pip install @packages
    if ($LASTEXITCODE -ne 0) {
        throw "$label package installation failed with exit code $LASTEXITCODE"
    }
}

# The Isaac venv already has Isaac Sim 5.1.0, Isaac Lab 2.3.2.post1, torch 2.7.0+cu128 and
# gymnasium. It only lacks the export toolchain. NEVER pip install isaacsim or isaaclab here.
Install-Packages $IsaacPython "Isaac venv" @("onnx>=1.16", "onnxruntime>=1.17", "ruff>=0.6", "pytest>=8.0")

# The MuJoCo venv is also the tools venv: it authors USD offline (usd-core, because `pxr` is
# not importable from a plain venv without a full Kit boot), generates textures (Pillow),
# implements the cv2 half of the preprocessing parity test, and runs the CPU torch twin of the
# training-time preprocessing chain.
Install-Packages $MujocoPython "MuJoCo/tools venv" @(
    "--index-url", "https://download.pytorch.org/whl/cpu", "torch"
)
Install-Packages $MujocoPython "MuJoCo/tools venv (rest)" @(
    "opencv-python-headless>=4.9", "pillow>=10.0", "onnxruntime>=1.17", "usd-core>=24.05", "pyyaml>=6.0"
)

# ------------------------------------------------------------------------------------------
# 4. Verification (the M0 acceptance test)
# ------------------------------------------------------------------------------------------
Write-Step "Verification"

if (Test-Path $IsaacPython) {
    & $IsaacPython -c "import onnxruntime; print('Isaac venv onnxruntime', onnxruntime.__version__)"
}
if (Test-Path $MujocoPython) {
    & $MujocoPython -c "import torch, cv2, PIL, onnxruntime, pxr; print('tools venv imports OK')"
}

if (-not $SkipRegistry -and (Test-Administrator)) {
    reg query "HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers" /v TdrDelay
}

Write-Step "Done"
Write-Host "Remaining manual steps:"
Write-Host "  1. Reboot so the TDR change takes effect."
Write-Host "  2. python scripts/check_clean_room.py      (must exit 0)"
Write-Host "  3. python -m pytest tests/unit -q          (must be green)"
