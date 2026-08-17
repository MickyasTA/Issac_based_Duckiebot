# The live viewer

Training runs headless. `scripts/live_view.py` attaches to its run directory, loads the newest
checkpoint into a network it builds exactly once, drives an episode, and hot-reloads the moment
the trainer publishes a newer checkpoint. Nothing restarts: not the simulator, not the network,
not the window.

```powershell
$TOOLS = "d:/Personal/personal/mujoco_venv/Scripts/python.exe"

# follow the newest run under runs/, forever
& $TOOLS scripts/live_view.py --loop

# follow one run, record video and observation snapshots into it
& $TOOLS scripts/live_view.py --run runs/20260817T104500Z_lanefollow_seed0 --loop --record --map loop_big
```

---

## 1. Which interpreter runs it (this was ambiguous; it is not any more)

The viewer needs **`mujoco` and `torch` in one process**: `mujoco` to simulate, `torch` to
evaluate the policy. As measured on this machine before any change was made:

| Package | Isaac venv (`wheeled_quadruped_robot/.venv`) | Tools venv (`mujoco_venv`) |
|---|---|---|
| `torch` | 2.7.0+cu128 | **was missing** |
| `mujoco` | **not installed, and not going to be** | 3.11.0 |
| `imageio`, `imageio-ffmpeg` | present | **was missing** |
| `Pillow` | present | **was missing** |
| `numpy`, `glfw` | numpy only | both |

Neither venv could run the viewer. The resolution follows what `docs/setup_windows.md` already
specified for the tools venv rather than inventing a third arrangement: **CPU torch goes into the
tools venv**, and `mujoco` stays out of the Isaac venv. Installing Isaac Sim's dependency tree
next to MuJoCo is the risk this project has consistently declined to take, and the CPU torch wheel
is ~200 MB against the multi-gigabyte CUDA one.

What was installed, and the resulting state:

```powershell
& $TOOLS -m pip install --index-url https://download.pytorch.org/whl/cpu torch
& $TOOLS -m pip install imageio imageio-ffmpeg pillow pyyaml
```

```
tools venv torch 2.13.0+cpu mujoco 3.11.0 numpy 2.4.6
```

**The answer, stated plainly:**

| Task | Interpreter |
|---|---|
| `scripts/live_view.py --backend mujoco` (the default) | **tools venv** |
| `scripts/live_view.py --backend isaac --allow-isaac-vram` | **Isaac venv** |
| `pytest tests/unit` (including the three viewer test files) | **either**; both pass |

The unit tests need neither `mujoco` nor a GPU, which is why they run in both.

---

## 2. Why MuJoCo is the default backend

Headless Isaac training on the target RTX 3080 Laptop is budgeted at **6.4 to 7.6 GiB of 8 GiB**.
A second Isaac Kit process costs another **2.5 to 3.5 GiB of baseline** before it renders a single
frame. 6.4 + 2.5 = 8.9 GiB against an 8 GiB card: an Isaac viewer next to an Isaac trainer does not
run slowly, it out-of-memories, and it takes the training run with it.

So `--backend isaac` refuses to start unless you pass `--allow-isaac-vram`, and the refusal prints
that arithmetic instead of saying "not supported". The MuJoCo backend runs on the CPU, costs zero
VRAM, and drives the *same* task: it wraps the existing `duckiebot_rl.sim2sim.env.MjDuckiebotEnv`
in `obs_mode="rgb_vec"`, so the action path and the S4.3 observation chain are the ones the
sim-to-sim evaluation uses, not a second implementation.

`duckiebot_rl/envs/` is owned by another module. The Isaac backend therefore sits behind exactly
one lazy import in `duckiebot_rl/viz/backends.py:_import_isaac_env_factory`, which names the four
locations it looks in and the keyword contract a factory must satisfy
(`map`, `num_envs`, `device`, `render` -> an object with `reset`, `step`, `render_frame`, `close`,
`control_dt`). Until that module lands the viewer says so; nothing else depends on it.

---

## 3. What the trainer has to do: nothing new

The viewer reads the run directory that `duckiebot_rl.viz.logger.TrainLogger` already writes. Its
`save_checkpoint()` calls `RunDir.record_checkpoint()`, which hashes the checkpoint and records it
in `checkpoints/index.json`. That index *is* the contract, and there is no second API to remember:

```python
log.save_checkpoint(
    lambda path: save_checkpoint(path, learner, iteration, steps, curriculum),
    iteration=it,
    metric=ep_return_mean,
)
```

| The trainer guarantees | The viewer guarantees |
|---|---|
| `latest.pt` / `best.pt` written atomically (`.tmp` then replace) | never writes into `checkpoints/` |
| the index entry is recorded **after** the `.pt` is in its final place | never loads a file whose content disagrees with the recorded hash |
| nothing else: `status.json`, `metrics.jsonl` and `figures/` are not required | `--record` writes only `video/` and `obs/` |

**Windows sharing violation.** `os.replace` onto a file another process holds open fails with
`PermissionError`. The viewer holds a checkpoint open only for one read, and
`duckiebot_rl.viz.watcher.atomic_replace` retries the collision. Use it rather than a bare
`os.replace` in any writer: losing a training run to a visualisation would be an absurd way to
lose a training run.

---

## 4. Why the watcher verifies hashes

Atomic writes protect a reader from a *torn* file. They do not protect it from a stale one. Two
failures survive `.tmp`-then-replace on its own:

1. the index lands before the `.pt` does, or vice versa, so the viewer loads yesterday's policy
   while printing today's iteration, and every conclusion drawn from the window is wrong;
2. the file is genuinely truncated or corrupt, and `torch.load` dies inside the render loop.

`CheckpointWatcher` therefore hashes the file it is about to hand out and compares against the
index. A mismatch is retried a few times (the normal mid-replace case), then recorded as a
rejection and skipped until the file changes again. It never raises for a missing run directory, a
missing index, a vanished file or a partial one; it returns `None` and explains itself in
`watcher.last_error`. `--allow-unindexed` downgrades to a weaker size-stability check for run
directories produced by something that does not write an index.

---

## 5. Refusing an architecture mismatch

`PolicyHost` builds the `ActorCritic` once; every later load is a `load_state_dict` into the same
module. Before touching a weight it validates in two passes: the checkpoint's recorded
`config.network` against the live one on the shape-relevant fields, then every parameter name and
shape against the live module (which catches checkpoints with no config block). Either failure
raises `ArchitectureMismatch` naming the offending fields, and the host keeps serving the policy
it already had, rather than emitting a wall of torch size-mismatch text halfway through a rollout.

`act()` returns the Gaussian **mean** by default. Visualising a sampled action mixes policy
improvement with exploration noise and makes two rollouts of the same checkpoint incomparable;
`--stochastic` samples when the exploration distribution is what you want to see. The host also
applies the checkpoint's own running `vec` statistics, because an unnormalised `vec` puts the
fusion layer off the manifold the actor trained on and the policy drives off the road while
looking perfectly healthy.

---

## 6. What `--record` writes

| Path | Needs | Without it |
|---|---|---|
| `video/latest_rollout.mp4` | `imageio` + `imageio-ffmpeg` | skipped, reason reported |
| `video/latest_rollout.gif` | `Pillow` or `imageio` | skipped, reason reported |
| `obs/latest_obs.png` | nothing beyond numpy | always written |

Install the optional encoders with `pip install "duckiebot-rl[record]"`.

The observation snapshot is the most useful of the three. It de-normalises the stacked
observation, splits it back into its three frames, tiles them and labels each with its time
offset, so bugs that are invisible in a reward curve are obvious at a glance: a stale frame ring
(three identical panels), a channel-order slip (blue lane markings), an inverted crop (sky instead
of road), photometric randomisation cranked until the yellow dashes vanish. It is drawn with a
built-in 5x7 bitmap font and a standard-library PNG writer specifically so it has no optional
dependency and cannot be the thing that fails.

Every artefact is encoded to a sibling temporary (`latest_rollout.tmp.mp4`) and then moved into
place atomically. The suffix is preserved because `imageio` picks its backend from the file
extension and refuses a `.tmp` URI outright.

---

## 7. Verified end to end

A run directory was built with two different policies, the viewer was started in `--loop --record`
mode against it, and `latest.pt` was swapped underneath the running process:

```
[live_view] START   latest=latest.pt iteration=100 ep_return_mean=12.5 sha256=a0e3645e09cb
[live_view] backend=mujoco map=loop_small control_dt=0.0667s action=deterministic mean
[live_view] episode 17: steps=24 return=-6.07 lane_dev_rms=0.1293 m reason=off_drivable reloads=0 iteration=100
[live_view] RELOAD  latest=latest.pt iteration=200 ep_return_mean=41.75 sha256=8f6a3a1e0f7b
[live_view]         reload #2 ... iteration=200 ep_return_mean=41.75 vec-normalised alpha_vis=0.35 alpha_dyn=0.2
[live_view]         action on the same observation: [0.59308 0.39034] -> [ 0.82574 -0.72858] (max|delta|=1.118920, CHANGED)
[live_view] episode 18: steps=20 return=-15.78 lane_dev_rms=0.0574 m reason=off_drivable reloads=1 iteration=200
```

The reload line reports the checkpoint, the iteration and the metric, and the line after it feeds
the **same observation** through the policy before and after the swap. That is the point: it shows
the weights actually took effect, not merely that a file was read.
