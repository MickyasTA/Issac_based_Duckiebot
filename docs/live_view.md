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

One episode encodes to one shape. On the Isaac backend the chase camera needs a render or two
before it produces a render product, so the first frames of an episode are the robot's own
128x192 onboard view and the rest are the 360x640 chase view; the encoders take a single shape
per file and used to refuse the whole episode with `all frames must share one shape`. The viewer
now reconciles the frames itself (`live_view.harmonize_frames`): the majority shape wins, a short
leading run of odd frames is dropped, and a longer one is resampled by nearest neighbour instead.

Every artefact is encoded to a sibling temporary (`latest_rollout.tmp.mp4`) and then moved into
place atomically. The suffix is preserved because `imageio` picks its backend from the file
extension and refuses a `.tmp` URI outright.

---

## 7. Which robot you are looking at: `--robot-mesh`

`--robot-mesh {db21j,db17,primitive}` picks the robot visual the Isaac backend draws. It is
viewer-side only: the meshes are attached as pure visual references and the primitive visual
scopes are hidden, so physics, collisions and the camera are byte-for-byte identical under all
three and a recorded episode drives the same way whichever you choose.

| Value | What is drawn | Needs |
|---|---|---|
| `db21j` (default, also `--db21j`) | the real latest-generation DB21, extracted from Duckietown's own Duckiematrix engine: camera mast with fisheye, Jetson devkit, top/bottom/interaction plates, LED bumpers, caster | `_refs/visual_mesh/db21j/main.obj` |
| `db21j`, second choice | the DB18-era glTF duckietown-world publishes as `duckiebot3`, used only when the OBJ is absent and labelled as what it is on stdout | `_refs/visual_mesh/db21/main.gltf` |
| `db17` | the older per-part DB17 OBJs, assembled and colored here because the OBJs carry no material | `chassis.obj` and friends under `_refs/visual_mesh` |
| `primitive` | the boxes and cylinders the environment itself builds | nothing |

`duckiebot3` is a misleading upstream directory name: that glTF is export_DB18's asset, not a
DB21. The only published copy of the real DB21 is inside the Duckiematrix simulator build, which
is why `scripts/fetch_visual_mesh.py` downloads that engine and rips the model out of its Unity
assets with UnityPy (`pip install -e .[mesh]`, or let the script install it).

The meshes derive from Duckietown CAD, are not redistributable and are therefore never committed.
`scripts/fetch_visual_mesh.py` places them under `_refs/`. A missing mesh is not an error: the
viewer prints what it looked for, points at the fetch script, and downgrades `db21j` to the glTF,
then to `db17` and then to `primitive` rather than failing to start. A misspelled selector *is* an
error, and is rejected before Kit boots so the typo costs no minutes.

The same flag, the same vocabulary and the same default now exist on `scripts/train.py` and
`scripts/check_obs.py`, where the attachment covers every one of the N environments rather than
just env 0. `tests/unit/test_robot_mesh.py` carries the proof that the robot's own mesh cannot
reach its own camera: every vertex falls inside the 0.05 m near plane, at the nominal camera pose
and at every corner of the declared V10 camera randomisation box.

---

## 8. Watching every layout at once: `--num-envs`

`--num-envs N` above 1 switches the viewer into **parallel view mode**, which is Isaac-only:

```powershell
$ISAAC = "d:/Personal/personal/wheeled_quadruped_robot/.venv/Scripts/python.exe"

# 64 different cities in one scene, 64 robots, one policy, free-fly the Kit viewport over them
& $ISAAC scripts/live_view.py --backend isaac --allow-isaac-vram --window --num-envs 64
```

One Isaac scene holds N cities side by side on the usual `env_spacing` grid, each with its own
robot, and **one** policy drives all of them:
`PolicyHost.act_batch(obs)` runs the actor once over an `(N, obs)` batch and returns `(N, act_dim)`.
Nothing loops over environments; `ActorCritic` is natively batched and always was. The batch is a
batch of the same policy the single-environment view shows: a batch of N copies of one observation
returns N copies of the action `act()` gives for it, to float32 noise (measured disagreement about
5e-10, which is the GEMM kernel a different batch size selects, not a different policy). That
equality is a unit test, not a claim.

What changes, and why:

| | `--num-envs 1` | `--num-envs N` |
|---|---|---|
| Loop | one episode at a time, reset between them | reset once, then step forever |
| Episode bookkeeping | per-episode return, lane-deviation RMS, termination reason | none: the environments reset themselves inside `step`, which is what makes the grid a continuous picture |
| Picture | chase camera, `obs/live_frame.png` stream | the Kit viewport itself, so pass `--window` |
| `--record` | writes `video/latest_rollout.{mp4,gif}` | **refused**, with a message: a chase camera follows one robot and there are N of them; a video of the grid is the training evaluation recorder's job |
| `--episodes`, `--loop` | control how many episodes run | do not apply; the grid runs until Ctrl-C or the window closes |
| Progress | one line per episode | one line every ~5 s: steps/s, env-steps/s, mean reward, resets, reloads, iteration |

Hot-reload works exactly as it does for one robot, and reaches all N at once: the watcher is polled
on a wall-clock timer (so its cadence does not change with the grid's speed), the weights are
swapped into the same live network, and the before/after action probe prints for environment 0
rather than 64 rows of numbers.

**Varied layouts.** A single-environment scene pins the stage list to the one layout `--map` names,
because `MultiUsdFileCfg` assigns assets by `index % len` and one environment is always index 0.
For a grid that is exactly wrong: N environments over a one-entry list is N copies of one city.
`backends._parallel_factory_kwargs` widens it, preferring the environment factory's own
`allow_multi` keyword when it advertises one, and otherwise passing a `city` override that keeps
the build root `--map` resolved to and grows the variant list to `city_000 .. city_{N-1}`. So in
parallel mode `--map` selects the **build root** (`build/city`, `build/city_hard`, ...) rather than a
single layout.

Refusals, all before anything slow happens:

```
[live_view] --num-envs 64 is an Isaac-only feature, and --backend mujoco was requested.
  the parallel grid is one Isaac scene holding num_envs cities side by side, each with its own
  robot, all driven by one policy in one batched forward pass
  ...
```

and, when Kit would come up headless, `no --window, so Kit runs headless and NOTHING appears on
screen; add --window to actually watch the grid`.

Ctrl-C and closing the window both leave through the same bounded teardown as a single-environment
run: the loop notices `is_running()` going false, returns its summary, and `finally` closes the
backend and joins Kit's `close()` with a 30 s bound.

---

## 9. Verified end to end

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
