<#
.SYNOPSIS
    Orphan-safe launcher for any Isaac Sim / Kit process in this project.

.DESCRIPTION
    Every Kit exit on this machine can leave an orphaned python.exe holding 2-8 GB of commit and
    up to 1.3 GB of VRAM. Booting a second Kit on top of one of those is how five of ten launch
    attempts were lost in one evening: the new process either fails to allocate, or it runs with
    a degraded VRAM budget and produces numbers that look like a regression but are not.

    This wrapper makes the documented preamble mechanical rather than remembered:

      1. Kill leftover python.exe processes that belong to THIS project - matched on the Isaac
         virtualenv path and on this repository's path in the command line, never on the bare
         image name, so an unrelated Python session is left alone.
      2. Wait for the driver and the memory manager to actually release what those processes
         held. VRAM and commit are both freed lazily; the wait is not a superstition.
      3. Refuse to launch unless the GPU is back to idle VRAM and the host has enough free
         commit, because launching under either condition is what produces the unexplainable run.
      4. Assert that no Kit process is already alive. Only one may exist at a time here.
      5. Exec the command, tee-ing everything into a log file inside the run directory.

    It changes no system setting. It adds no Windows Defender exclusion, and it does not touch
    the pagefile: that is system-managed and altering it is the user's decision, not this
    script's.

.PARAMETER Exec
    The command to run and all of its arguments, given last: the executable first, then
    everything it should receive. Isaac Lab's own flags are double-dashed ('--num_envs',
    '--headless'), which PowerShell passes straight through, so no separator token is needed and
    none should be used - a bare '--' is parsed as an empty parameter name and fails to bind.

.PARAMETER LogDir
    Directory to write the launch log into. Created when missing. Defaults to '.tmp/launches'.

.PARAMETER MinFreeCommitGb
    Minimum free commit required before launching.

.PARAMETER MaxIdleVramMib
    Highest 'nvidia-smi' used-memory reading that still counts as an idle GPU.

.PARAMETER SettleSeconds
    Seconds to wait after killing orphans, before re-checking the resources.

.PARAMETER Force
    Launch even if the resource preconditions are not met. The reason is still logged.

.EXAMPLE
    ./scripts/launch_run.ps1 d:/Personal/personal/wheeled_quadruped_robot/.venv/Scripts/python.exe `
        scripts/train.py --task Duckiebot-LaneFollow-v0 --num_envs 64 --headless --enable_cameras

.EXAMPLE
    ./scripts/launch_run.ps1 -LogDir training_results/latest python.exe scripts/train.py --num_envs 256
#>

[CmdletBinding()]
param(
    [string] $LogDir,
    [double] $MinFreeCommitGb = 9.0,
    [double] $MaxIdleVramMib = 200.0,
    [int]    $SettleSeconds = 15,
    [switch] $Force,

    [Parameter(Mandatory = $true, Position = 0, ValueFromRemainingArguments = $true)]
    [string[]] $Exec
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$RepoRoot = Split-Path -Parent $PSScriptRoot
$IsaacVenv = 'wheeled_quadruped_robot\.venv'

# Tolerate a leading '--' if one survives from a caller that wrote the separator out of habit.
if ($Exec.Count -gt 0 -and $Exec[0] -eq '--') { $Exec = @($Exec[1..($Exec.Count - 1)]) }
if ($Exec.Count -eq 0) { throw 'no command given: pass the executable and its arguments last' }
$Command = $Exec[0]
$CommandArgs = @()
if ($Exec.Count -gt 1) { $CommandArgs = @($Exec[1..($Exec.Count - 1)]) }

# ------------------------------------------------------------------ logging

if (-not $LogDir) { $LogDir = Join-Path $RepoRoot '.tmp\launches' }
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Force -Path $LogDir | Out-Null }
$LogFile = Join-Path $LogDir ("launch_{0}.log" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))

function Write-Line {
    param([string] $Message, [string] $Level = 'info')
    $stamp = (Get-Date -Format 'HH:mm:ss')
    $line = "[$stamp] [$Level] $Message"
    Add-Content -Path $LogFile -Value $line -Encoding utf8
    if ($Level -eq 'error') { Write-Host $line -ForegroundColor Red }
    elseif ($Level -eq 'warn') { Write-Host $line -ForegroundColor Yellow }
    else { Write-Host $line }
}

# ------------------------------------------------------------------ probes

function Get-FreeCommitGb {
    <#  Free commit in GiB, or NaN when the probe fails. #>
    try {
        $os = Get-CimInstance Win32_OperatingSystem -ErrorAction Stop
        return [double] $os.FreeVirtualMemory / 1MB
    } catch {
        return [double]::NaN
    }
}

function Get-UsedVramMib {
    <#  Used VRAM in MiB across all GPUs, or NaN when nvidia-smi is unavailable. #>
    try {
        $raw = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
        if ($LASTEXITCODE -ne 0 -or -not $raw) { return [double]::NaN }
        $total = 0.0
        foreach ($line in @($raw)) {
            $text = "$line".Trim()
            if ($text) { $total += [double] $text }
        }
        return $total
    } catch {
        return [double]::NaN
    }
}

function Get-ProjectPythonProcesses {
    <#  python.exe processes belonging to this project: the Isaac venv, or this repo on the
        command line. Never every python.exe on the machine. #>
    $matched = @()
    try {
        $all = Get-CimInstance Win32_Process -Filter "Name = 'python.exe' OR Name = 'pythonw.exe'" -ErrorAction Stop
    } catch {
        Write-Line "could not enumerate processes: $($_.Exception.Message)" 'warn'
        return @()
    }
    foreach ($process in $all) {
        if ($process.ProcessId -eq $PID) { continue }
        $haystack = "$($process.ExecutablePath) $($process.CommandLine)"
        if ($haystack -match [regex]::Escape($IsaacVenv) -or $haystack -match [regex]::Escape($RepoRoot)) {
            $matched += $process
        }
    }
    return $matched
}

function Get-KitProcesses {
    <#  Live Kit / Isaac Sim processes. Only one may exist at a time on this machine. #>
    try {
        return @(Get-Process -Name 'kit', 'isaac-sim', 'omni*' -ErrorAction SilentlyContinue)
    } catch {
        return @()
    }
}

# ------------------------------------------------------------------ preamble

Write-Line "launch wrapper starting in $RepoRoot"
Write-Line "command: $Command $($CommandArgs -join ' ')"
Write-Line "log: $LogFile"

$orphans = @(Get-ProjectPythonProcesses)
if ($orphans.Count -gt 0) {
    foreach ($orphan in $orphans) {
        $ws = [math]::Round($orphan.WorkingSetSize / 1GB, 2)
        Write-Line "killing orphan pid $($orphan.ProcessId) (working set $ws GB): $($orphan.CommandLine)" 'warn'
        try {
            Stop-Process -Id $orphan.ProcessId -Force -ErrorAction Stop
        } catch {
            Write-Line "could not kill pid $($orphan.ProcessId): $($_.Exception.Message)" 'warn'
        }
    }
    Write-Line "waiting $SettleSeconds s for the driver and the memory manager to release"
    Start-Sleep -Seconds $SettleSeconds
} else {
    Write-Line 'no project python orphans found'
}

$kit = @(Get-KitProcesses)
if ($kit.Count -gt 0) {
    $names = ($kit | ForEach-Object { "$($_.ProcessName)($($_.Id))" }) -join ', '
    Write-Line "a Kit process is already alive: $names. Only one may exist at a time." 'error'
    if (-not $Force) { exit 2 }
    Write-Line 'continuing anyway because -Force was given' 'warn'
}

$vram = Get-UsedVramMib
$commit = Get-FreeCommitGb
Write-Line ("resources: used VRAM {0} MiB (idle threshold {1}), free commit {2:N2} GB (minimum {3:N2})" -f `
        $vram, $MaxIdleVramMib, $commit, $MinFreeCommitGb)

$blockers = @()
if ([double]::IsNaN($vram)) {
    Write-Line 'nvidia-smi did not answer; VRAM precondition skipped' 'warn'
} elseif ($vram -gt $MaxIdleVramMib) {
    $blockers += ("VRAM still held: {0} MiB > {1} MiB" -f $vram, $MaxIdleVramMib)
}
if ([double]::IsNaN($commit)) {
    Write-Line 'commit probe did not answer; commit precondition skipped' 'warn'
} elseif ($commit -lt $MinFreeCommitGb) {
    $blockers += ("free commit too low: {0:N2} GB < {1:N2} GB" -f $commit, $MinFreeCommitGb)
}

if ($blockers.Count -gt 0) {
    foreach ($blocker in $blockers) { Write-Line $blocker 'error' }
    if (-not $Force) {
        Write-Line 'refusing to launch. Re-run with -Force to override, or free the resources first.' 'error'
        exit 3
    }
    Write-Line 'launching anyway because -Force was given' 'warn'
}

# ------------------------------------------------------------------ launch

Write-Line 'preamble clear, launching'
$started = Get-Date
& $Command @CommandArgs 2>&1 | Tee-Object -FilePath $LogFile -Append
$code = $LASTEXITCODE
$elapsed = (Get-Date) - $started
Write-Line ("command exited with code {0} after {1:N0} s" -f $code, $elapsed.TotalSeconds)

$leftover = @(Get-ProjectPythonProcesses)
if ($leftover.Count -gt 0) {
    Write-Line "$($leftover.Count) project python process(es) survived the exit; the next launch will clear them" 'warn'
}
$vramAfter = Get-UsedVramMib
Write-Line ("used VRAM after exit: {0} MiB" -f $vramAfter)

exit $code
