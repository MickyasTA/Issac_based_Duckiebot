# Live training: the run directory, the logger and the dashboard

Training a lane-following policy takes hours to days. This page is about the two things that make
that survivable: a **run directory** that is written atomically enough to be read while it is being
written, and a **dashboard PNG** you can leave open in an image viewer and glance at.

Three components, in `duckiebot_rl/viz/`:

| Module | What it is | Needs |
| --- | --- | --- |
| `run_dir.py` | The run-directory layout, implemented once. Ids, atomic writers, tolerant readers, the heartbeat, the metrics appender, the checkpoint index. | standard library only |
| `logger.py` | `TrainLogger`, the small API a training script calls. | torch (lazily, only to save checkpoints) |
| `plots.py`, `dashboard.py` | The figures and the composite `figures/latest.png`. | matplotlib (`[viz]` extra) |

The dashboard runs **out of process** and costs **zero VRAM**. It reads two files and draws with
the Agg backend. That is deliberate: headless Isaac training is budgeted at 6.4 to 7.6 GiB of the
8 GiB on this machine, so nothing in the monitoring path is allowed to allocate on the GPU.

---

## 1. Quick start

Terminal 1, training (whatever your training entry point is; it calls `TrainLogger`):

```powershell
d:/Personal/personal/wheeled_quadruped_robot/.venv/Scripts/python.exe scripts/train.py --seed 0
```

Terminal 2, the dashboard:

```powershell
d:/Personal/personal/wheeled_quadruped_robot/.venv/Scripts/python.exe scripts/dashboard.py --watch
```

With no `--run`, it picks the newest directory under `./runs`. Then open

```
runs/<run_id>/figures/latest.png
```

in any image viewer that reloads on change, and leave it open. The file is replaced atomically, so
the viewer never catches a half-written PNG.

`--watch` is safe to start **before** training does; it polls for the directory. It is safe to leave
running **after** training ends; it renders once more and exits.

---

## 2. The run directory contract

Every run writes exactly this tree, under `runs/<run_id>/`, where the id is
`<UTCtimestamp>_<name>_seed<N>`, for example `20260817T104500Z_lanefollow_seed0`:

```
runs/<run_id>/
  config.yaml            the full resolved config, written once at start
  status.json            the heartbeat, rewritten atomically every iteration
  metrics.jsonl          append-only, one flat JSON object per iteration, fsync'd
  checkpoints/
    latest.pt            always the most recent          (atomic)
    best.pt              best by the model-selection metric (atomic)
    iter_%08d.pt         periodic archive
    index.json           {latest: {...}, best: {...}} with iteration, metric, sha256, mtime
  figures/
    latest.png           THE composite dashboard
    <panel>.png          each panel standalone (written with --panels)
  video/
    latest_rollout.mp4   most recent policy rollout
    latest_rollout.gif   same, loopable, README-embeddable
  obs/
    latest_obs.png       what the policy actually sees, de-normalised and tiled
  tb/                    TensorBoard mirror, when tensorboard is importable
```

`video/` and `obs/` are written by the rollout tooling, not by the logger. `run_dir.py` owns their
paths so nothing else has to hardcode them, and the dashboard footer reports whether they exist.

**Nothing outside `run_dir.py` may hardcode one of these paths.** Ask a `RunDir` instead:

```python
from duckiebot_rl.viz import RunDir

run = RunDir.open("runs/20260817T104500Z_lanefollow_seed0")
run.latest_checkpoint  # .../checkpoints/latest.pt
run.archive_checkpoint(500)  # .../checkpoints/iter_00000500.pt
run.figure_path("ep_return")  # .../figures/ep_return.png
```

### `status.json`

```json
{
  "schema_version": 1,
  "run_id": "20260817T104500Z_lanefollow_seed0",
  "pid": 21948,
  "state": "running",
  "iteration": 412,
  "total_timesteps": 54001664,
  "wall_clock_s": 7381.4,
  "steps_per_s": 7315.6,
  "best_metric_name": "ep_return_mean",
  "best_metric_value": 102.87,
  "best_iteration": 388,
  "last_update_utc": "2026-08-17T12:43:01Z",
  "vram_used_mb": 6842.0,
  "num_envs": 4096,
  "device": "cuda:0",
  "git_commit": "a5e3f41f74bbf557806aaaac26b76bd936fc67ca"
}
```

`state` is one of `running`, `finished`, `crashed`. `last_update_utc` is stamped by the writer, not
the caller, so the freshness the dashboard reports is the freshness of the write.

### `metrics.jsonl`

One flat JSON object per line, per iteration, appended and fsync'd. Keys are free-form and the
dashboard **discovers** them, but these must be present:

```
iteration, total_timesteps, wall_clock_s,
ep_return_mean, ep_return_std, ep_len_mean,
policy_loss, value_loss, entropy, approx_kl, clipfrac,
explained_variance, grad_norm, learning_rate,
lane_dev_rms_m, lane_dev_max_m, success_rate,
alpha_vis, alpha_dyn
```

`TrainLogger` warns once (a `RuntimeWarning`, never an exception) if the first row you log is
missing any of them. Non-finite values are written as `null`, because JSON has no `NaN` and a
reader that has to special-case a bare `NaN` literal is a reader that dies at 3 a.m.

Optional but understood: `target_kl`, which draws the target line on the PPO health panel at its
real value instead of the 0.015 default.

### The atomicity rule

Every whole-file writer writes `<name>.tmp` in the same directory, flushes, fsyncs and then
`Path.replace()`s it into position. That call is atomic on Windows and POSIX alike, so a reader
sees either the old file or the new one, never a half-written one.

`metrics.jsonl` is the exception: it is append-only, so a reader can catch a torn **final line**
instead. `RunDir.read_metrics()` drops it.

There is one Windows-specific hazard, and both sides handle it. `os.replace` needs delete access
to the destination, and CPython opens files without `FILE_SHARE_DELETE`. So a reader that happens
to hold `status.json` open for the microsecond the writer swaps it makes the **writer** raise
`PermissionError` on Windows, where identical code never fails on Linux. Writers retry the replace;
readers retry the open. `tests/unit/test_run_dir.py` runs a reader in a loop against a live writer
and asserts that every single read decodes to a complete payload.

---

## 3. The integration contract for `scripts/train.py`

This is the whole API. It is intentionally four calls, and it is meant to stay stable.

```python
from duckiebot_rl.viz import TrainLogger
from duckiebot_rl.ppo.checkpoint import save_checkpoint

log = TrainLogger.create(
    "runs",  # the runs root; the run id is built for you
    name="lanefollow",
    seed=cfg.seed,
    config=cfg,  # written verbatim to config.yaml, once
    num_envs=cfg.num_envs,  # reported in the heartbeat
    device=str(device),
    best_metric="ep_return_mean",  # what best.pt means
    best_mode="max",  # or "min", for e.g. lane_dev_rms_m
    archive_every=50,  # every 50th checkpoint also lands in iter_%08d.pt
)

for iteration in range(1, cfg.num_iterations + 1):
    ...  # rollout, GAE, update

    log.log_iteration(
        {
            "iteration": iteration,
            "total_timesteps": global_step,
            "ep_return_mean": ...,
            "ep_return_std": ...,
            "ep_len_mean": ...,
            "policy_loss": ...,
            "value_loss": ...,
            "entropy": ...,
            "approx_kl": ...,
            "clipfrac": ...,
            "explained_variance": ...,
            "grad_norm": ...,
            "learning_rate": ...,
            "lane_dev_rms_m": ...,
            "lane_dev_max_m": ...,
            "success_rate": ...,
            "alpha_vis": adr.alpha_vis,
            "alpha_dyn": adr.alpha_dyn,
            # anything else you like: it gets its own panel automatically
            "collisions_per_min": ...,
        }
    )

    if iteration % cfg.save_every == 0:
        log.save_checkpoint(
            lambda path: save_checkpoint(path, learner, iteration, global_step, adr.state_dict()),
            metric=metrics["ep_return_mean"],
        )

log.finish()
```

### `TrainLogger.create(runs_root, name, seed, config, run_id=None, **kwargs)`

Resolves the run id, creates the tree, writes `config.yaml` and the first heartbeat. Pass `run_id`
to re-attach to an existing directory, which is how a resume keeps writing to the same run.

### `log.log_iteration(dict) -> dict`

Appends the row, rewrites `status.json`, mirrors numeric entries to TensorBoard. Fills in
`iteration`, `total_timesteps` and `wall_clock_s` if you omit them. Returns the row as written.

### `log.save_checkpoint(state, metric=..., is_best=None, iteration=None, archive=None)`

`state` is either

* **a callable** that writes the checkpoint to the path it is handed. **Prefer this**: it lets you
  delegate to `duckiebot_rl.ppo.checkpoint.save_checkpoint`, which already saves the model,
  optimiser, running normalisers, the mandatory curriculum state and every RNG stream; or
* **a mapping**, which the logger serialises with `torch.save` under the same atomicity rule.

The logger owns model selection. Give it `metric` and it decides whether this becomes `best.pt`,
records both files in `checkpoints/index.json` with their SHA-256, and reports the best value and
its iteration in the heartbeat. Pass `is_best=True`/`False` to override.

### `log.finish(state="finished")`

Writes the terminal heartbeat, closes TensorBoard, renders a last figure. Idempotent.

Use the logger as a context manager and a raising loop records `state: crashed` for you:

```python
with TrainLogger.create("runs", name="lanefollow", seed=0, config=cfg) as log:
    ...
```

### Two more methods you probably do not need

* `log.heartbeat()` rewrites `status.json` without logging a row. Call it inside a long iteration
  (a slow evaluation, a video rollout) so the dashboard does not paint the run as stale.
* `log.render_dashboard()` renders the figure from inside the training process. Training does not
  need it, since `scripts/dashboard.py --watch` renders out of process for free; it exists so an
  unattended run leaves a final figure behind. It never raises: a plotting bug must not kill a
  training run that has been going for two days.

### What the logger deliberately does not do

It does not own your config schema, it does not decide when to checkpoint, and it does not touch
the GPU except to read `torch.cuda.memory_reserved()` for the heartbeat.

---

## 4. The dashboard

```powershell
python scripts/dashboard.py --run runs/<run_id> [--watch] [--interval 20] [--once] [--panels]
```

| Flag | Meaning |
| --- | --- |
| `--run <dir>` | The run directory. Omit it and the newest under `--runs-root` (default `./runs`) is used. |
| `--once` | Render once and exit. This is the default. |
| `--watch` | Re-render whenever `metrics.jsonl` grows, until the run reports finished or crashed. |
| `--interval <s>` | Poll period in watch mode; default 20 s. |
| `--panels` | Also write `figures/<panel>.png` for every panel, not just the composite. |
| `--dpi <n>` | Output resolution; 100 gives the 1600 px reading width. |
| `--keep-going` | In watch mode, do not exit when the run ends. |
| `--no-summary` | Suppress the text summary. |

Example session:

```
[dashboard] watching runs\20260817T120000Z_livedemo_seed0 every 3s; Ctrl+C to stop
[dashboard] run directory does not exist yet, polling for it
[dashboard] render 1: runs\...\figures\latest.png (64,905 bytes)
[dashboard] render 2: runs\...\figures\latest.png (220,736 bytes)
...
run        : 20260817T120000Z_livedemo_seed0
state      : FINISHED
iterations : 70 logged, last iteration 70
best       : ep_return_mean = 112.5 at iteration 70
best       : best.pt iter 70 ep_return_mean=112.5 sha256 9dbc36e15de3
```

### The header strip

Run id, state, iteration, environment steps, wall clock, throughput, the best metric and the
iteration that produced it, and how old the heartbeat is. The strip is colour-coded **and**
labelled, so the colour is reinforcement and never the only carrier of meaning:

| State | Reads |
| --- | --- |
| running, heartbeat under 2 minutes old | `RUNNING`, green |
| running, heartbeat over 2 minutes old | `STALE, no heartbeat for 0:47:38`, amber |
| finished | `FINISHED`, green |
| crashed | `CRASHED`, red |
| no `status.json` yet | `NO STATUS YET`, grey |

Two minutes is longer than any single PPO iteration on the target hardware, so amber always means
something is genuinely wrong: the trainer is wedged, throttled, or gone.

### The panels

1. **episode return**: mean over a +/-1 standard-deviation band.
2. **episode length**: steps per episode. On this task it is a survival proxy; it should rise.
3. **lane deviation**: RMS and worst-case distance from the lane centre, in metres.
4. **success rate**: fraction of episodes reaching the goal, on a pinned 0 to 100 % scale so
   progress cannot be exaggerated by an auto-scaled axis.
5. **PPO health**: approximate KL and clip fraction, with the target-KL threshold drawn. KL walking
   past its target while the clip fraction climbs is the signature of too large a step.
6. **critic explained variance**, with the zero reference drawn. Negative means the critic is worse
   than predicting the mean, which is why the line is there.
7. **losses**: policy and value on one axis, switched to symlog when their magnitudes differ by
   more than 50x.
8. **policy entropy** and **learning rate**, as two stacked charts in one cell.
9. **DR curriculum**: `alpha_vis` and `alpha_dyn` on a pinned 0 to 1 scale.

Then **one panel per discovered key**. Anything numeric you logged that no fixed panel claims gets
its own panel appended, in first-seen order, capped at 12 so the figure stays openable. That is why
`grad_norm` appears even though it has no hand-written panel.

### Reading the curves

Each series is drawn twice: the raw values faintly, and an exponential moving average on top. Never
smoothing alone, because the smoothed curve hides exactly the variance spike you opened the figure
to find. The last value of each series is marked with a dot and printed beside it.

Colours come from one validated colourblind-safe categorical order, defined once in
`plots.PALETTE` and `plots.SERIES`. Single-series panels all use slot 1, so a colour never means two
different things in two panels, and the status colours (green, amber, red) are reserved for run
state and are never reused for a series.

---

## 5. Degenerate cases

These all render, none of them raise. They are covered by `tests/unit/test_dashboard.py`:

* zero rows logged, or a run directory containing nothing at all;
* exactly one row (single markers, not invisible zero-length lines);
* a column that is entirely NaN, or a metric that only starts appearing halfway through a run
  (drawn as a gap, not as a drop to zero);
* a corrupt `status.json`, or a torn final line in `metrics.jsonl`;
* a crashed run;
* a run whose only extra keys are strings.

A panel that somehow fails still prints its exception into its own cell rather than costing you the
whole figure.

---

## 6. Installation and degradation

The plotting layer is an optional extra:

```powershell
d:/Personal/personal/wheeled_quadruped_robot/.venv/Scripts/python.exe -m pip install "duckiebot-rl[viz]"
```

If matplotlib is missing, `scripts/dashboard.py` exits 1 with the exact pip command, and
`TrainLogger` keeps working: the run directory, the heartbeat, the metrics and the checkpoints are
all pure standard library plus torch. You lose the picture, never the data.

Because `run_dir.py` imports nothing outside the standard library, you can run the dashboard from
the MuJoCo venv, or from any interpreter with matplotlib, against a run being written by the Isaac
venv. That is the recommended arrangement: it keeps the training process untouched.

TensorBoard mirroring is on by default and degrades silently when TensorBoard is not importable.

---

## 7. Reading a run programmatically

```python
from duckiebot_rl.viz import RunDir

run = RunDir.open("runs/20260817T104500Z_lanefollow_seed0")

status = run.read_status()  # RunStatus | None
rows = run.read_metrics()  # list[dict], torn final line dropped
index = run.read_index()  # {"latest": {...}, "best": {...}}
config = run.read_config()  # the resolved config

print(status.state, status.iteration, index["best"]["sha256"])
```

Every reader tolerates the file vanishing, being replaced mid-read, or being unreadable, and
returns `None` or an empty container rather than raising. Poll them as fast as you like.

For a text summary of a run, without matplotlib doing any drawing:

```python
from duckiebot_rl.viz import summarise

print(summarise("runs/20260817T104500Z_lanefollow_seed0"))
```
