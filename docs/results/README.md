# Results

## M10 sim-to-sim transfer (SPEC v2 S8.4)

One policy, exported to TorchScript, driven in MuJoCo: a different physics engine and a
different renderer from the Isaac Lab environment it was trained in. 40 episodes x 3 seeds per
condition, 45 s cap, 8.12 m closed loop.

| condition | episodes | distance (median) | laps | lane RMS | survival | success |
|---|---|---|---|---|---|---|
| **C5** nominal | 120 | **18.29 m** | **2.18** | **3.28 cm** | 45.0 s | **100%** |
| **C6** + dynamics DR | 120 | 2.26 m | 0.28 | 5.60 cm | 7.4 s | 0% |

C5 across seeds: 18.273 +/- 0.081 m, 95% CI [18.196, 18.389]. Failures: 113 of 120 episodes
reached the 45 s cap without failing at all; 5 off-drivable, 2 rollover.

**Reading C5.** The policy drives more than two full laps at 3.3 cm lane RMS in a simulator it
never saw, and in 94% of episodes it simply runs out of clock rather than failing. Vision, the
lane-following behaviour and the action path all survive a complete change of engine and
renderer.

**Reading C6.** Layering the S7.3 dynamics randomisation on top of the engine change collapses
it to a quarter lap, mostly off-drivable. The policy was trained under Isaac's dynamics
randomisation at alpha 0.92, so this is not simply "unseen randomisation": it is the compound of
two different physics engines AND randomised parameters, and it is the honest open weakness of
the current checkpoint. It is the first thing to attack before any real-robot attempt.

Reproduce:

    d:/Personal/personal/wheeled_quadruped_robot/.venv/Scripts/python.exe scripts/export_policy.py ^
        --checkpoint <run>/model_best.pth --out-dir exports/<name>
    d:/Personal/personal/mujoco_venv/Scripts/python.exe scripts/eval_sim2sim.py build --out build/sim2sim
    d:/Personal/personal/mujoco_venv/Scripts/python.exe scripts/eval_sim2sim.py eval ^
        --policy exports/<name>/policy_traced.pt --obs-mode rgb_vec --conditions C5 C6 ^
        --seeds 0 1 2 --episodes 40 --episode-seconds 45 --workers 1 ^
        --out docs/results/transfer_C5_C6.json

Raw records: `transfer_C5_C6.json`.

## M11 export parity (SPEC v2 S9.1)

Both deployment targets export from one checkpoint and pass the onnxruntime parity gate against
torch on 1000 random frames:

| target | opset | runtime | action max abs diff |
|---|---|---|---|
| Jetson Nano | 13 | TensorRT 8.2 | 5.245e-06 |
| Orin Nano | 18 | TensorRT 10 | 5.245e-06 |

Gate threshold is 1e-05. TorchScript (`policy_traced.pt`) is the artifact the C5/C6 evaluation
above actually drives, which is the S9.1 requirement that the exported graph, not a
reimplementation, is what gets measured.
