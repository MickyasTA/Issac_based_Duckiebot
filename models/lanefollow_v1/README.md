# Pretrained lane-following policy (`lanefollow_v1`)

A vision-based lane-following policy for a Duckiebot DB21, trained from scratch with PPO in
Isaac Lab. It sees a camera image and its own proprioception, and outputs a velocity command.
No ground-truth pose, no lane state and no map is available to it at inference time.

## What is measured

| where | distance | laps | lane RMS | success |
|---|---|---|---|---|
| Isaac Lab, held-out eval | 21.8 tiles | - | - | - |
| **MuJoCo** (different engine AND renderer) | **18.29 m** | **2.18** | **3.28 cm** | **100%** |

The MuJoCo row is 120 episodes over 3 seeds, 45 s cap, on an 8.12 m closed loop; 113 of those 120
episodes reached the time cap without failing at all. Full matrix and reproduction commands live
in [`docs/results/`](../../docs/results/).

Trained under domain randomisation at full strength (`alpha_vis = alpha_dyn = 1.0`).

**Honest limitation.** With the S7.3 dynamics randomisation applied on top of the engine change
(condition C6), the same policy manages only 2.26 m. It has also never touched physical hardware.
Treat a real-robot run as an experiment rather than a deployment, and start on a closed course.

## Files

| file | what it is | use it for |
|---|---|---|
| `policy_traced.pt` | TorchScript, batch 1 | Python inference anywhere torch runs; this is the artifact the MuJoCo numbers above were produced with |
| `policy_opset13.onnx` | ONNX opset 13 | Jetson Nano, TensorRT 8.2 |
| `policy_opset18.onnx` | ONNX opset 18 | Orin Nano, TensorRT 10 |
| `policy_opset*.json` | sidecars | full I/O contract, preprocessing constants, provenance, sha256 |
| `checkpoint_model_best.pth` | training checkpoint | resume training, or re-export |
| `SHA256SUMS` | checksums | `sha256sum -c SHA256SUMS` |

Both ONNX graphs pass an offline parity gate against torch on 1000 random frames at **9.9e-06**
maximum absolute action difference, against a 1e-05 threshold.

## Inputs and outputs

Two inputs:

- **`image`**, `uint8`, shape `(1, 48, 96, 9)`, **NHWC**, **RGB**. The 9 channels are 3 stacked
  frames of 3 channels each, oldest first, sampled every 2 control steps.
- **`vec`**, `float32`, shape `(1, 8)`:
  `[prev_v, prev_omega, prev2_v, prev2_omega, wheel_left, wheel_right, yaw_rate, body_speed]`.
  Wheel speeds in rad/s, yaw rate in rad/s, body speed in m/s. The two previous actions are the
  clipped values that were actually sent, in `[-1, 1]`.

Two outputs, each `float32` `(1, 2)`: **`action`** (sampled from the policy distribution) and
**`mu`** (its mean). **Use `mu` for deployment** — it is the deterministic action.

Convert `mu` into a velocity command:

```
v     = 0.5 * 0.6 * (clip(mu[0], -1, 1) + 1)     # m/s,   v_max     = 0.6
omega = 4.0 * clip(mu[1], -1, 1)                 # rad/s, omega_max = 4.0
```

On a Duckiebot that is a `duckietown_msgs/Twist2DStamped` published to
`car_cmd_switch_node/cmd` at 15 Hz.

## Preprocessing (this has to match exactly)

The policy was trained on one specific image pipeline; feeding it raw camera frames will not
work. Starting from the canonical camera model (192x128, fx = fy = 65.98, cx = 96.0, cy = 64.0,
HFOV 111 degrees):

1. rectify to that canonical model,
2. blur with the separable kernel `[0.00256, 0.16555, 0.66378, 0.16555, 0.00256]`, `replicate`
   padding,
3. box-downsample by 2, giving 96x64,
4. crop the top 16 rows, giving 96x48,
5. stack 3 such frames sampled every 2nd control step, oldest channel first.

Steps 1 to 4 are the robot's job: nothing is baked into the graph. The exact constants are in the
sidecar JSON under `preprocess`, which is the authority if this summary and the code ever differ.

## Run it

TorchScript:

```python
import torch

policy = torch.jit.load("policy_traced.pt").eval()
image = torch.zeros(1, 48, 96, 9, dtype=torch.uint8)  # your preprocessed stack
vec = torch.zeros(1, 8, dtype=torch.float32)
with torch.no_grad():
    action, mu = policy(image, vec)

v = 0.5 * 0.6 * (mu[0, 0].clamp(-1, 1) + 1)
omega = 4.0 * mu[0, 1].clamp(-1, 1)
```

ONNX Runtime, which is what would run on the robot:

```python
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("policy_opset13.onnx", providers=["CPUExecutionProvider"])
image = np.zeros((1, 48, 96, 9), dtype=np.uint8)
vec = np.zeros((1, 8), dtype=np.float32)
action, mu = session.run(None, {"image": image, "vec": vec})
```

TensorRT, documented but never run here, because no hardware was involved in this project:

```
nvpmodel -m 0 && jetson_clocks
trtexec --onnx=policy_opset13.onnx --fp16 --workspace=512 --saveEngine=policy_nano.plan
```

Or watch it drive in the MuJoCo twin instead, which needs no robot:

```
<mujoco_venv>/Scripts/python.exe scripts/eval_sim2sim.py build --out build/sim2sim
<mujoco_venv>/Scripts/python.exe scripts/eval_sim2sim.py eval \
    --policy models/lanefollow_v1/policy_traced.pt --obs-mode rgb_vec \
    --conditions C5 --seeds 0 --episodes 8 --episode-seconds 45 --workers 1
```

## Re-export, or continue training

```
python scripts/export_policy.py \
    --checkpoint models/lanefollow_v1/checkpoint_model_best.pth --out-dir models/my_export

python scripts/train.py --headless --enable_cameras --num_envs 64 --robot-mesh primitive \
    --resume models/lanefollow_v1/checkpoint_model_best.pth
```

## Provenance

Run `20260821T173917Z_lanefollow_seed0_leaky_seed0`, checkpoint selected on held-out evaluation
distance (21.819 tiles at iteration 5900). Apache-2.0, the same licence as the repository.

If you do run this on a real Duckiebot, please open an issue with what happened, good or bad.
Closing that loop is the single most useful thing anyone can contribute to this project.
