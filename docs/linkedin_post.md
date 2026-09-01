# LinkedIn post draft
`
I brought Duckietown into NVIDIA Isaac Sim, and taught a robot to drive with pixels alone.

Over the past weeks I built an open-source integration of the Duckietown lane-following
challenge into Isaac Sim 5.1 / Isaac Lab: a procedural city generator, a vision-based PPO
agent written from scratch (no RL libraries), and 64 different city layouts training in
parallel on a single laptop GPU. The policy sees only a 96x48 camera stream, and it now holds
its lane at about 1.5 cm RMS deviation through hairpin turns it once cut straight across.

The inspiration comes from my internship at IRT Saint-Exupery, where I worked up close with
the hardest problem in robot learning: the sim-to-real gap. Isaac Sim is one of the most
promising answers I have touched. Its flexibility is the point: tiled cameras render every
robot's viewpoint in one pass, domain randomization is a first-class citizen, and the same
scene can host anything from procedural tiles to a real-world environment reconstructed with
3D Gaussian Splatting, so the pixels the policy trains on can get arbitrarily close to the
pixels a real robot will see. That direction, 3DGS-reconstructed environments inside Isaac for
real-to-sim, is next on my roadmap.

What I did not expect to learn is how hard the REWARD is compared to the network. The
optimizer found every crack I left open, and each fix is now a test in the repo:

- The policy learned to cut corners because lane deviation was measured to the NEAREST lane
  centreline, so crossing the line teleported the error back to zero. Fix: continuity
  constrained lane matching, so the measurement follows the robot and cheating becomes visible.
- It then learned that dying early was cheaper than driving badly. Fix: a survival income
  sized so that no recoverable state is worth abandoning.
- It then learned to park and collect that income while its vision encoder quietly died a
  dead-ReLU death, pushed by a KL-starved learning-rate spiral. Fix: motion-gated income,
  leaky activations so the encoder cannot die, a hard learning-rate ceiling, and per-iteration
  encoder liveness telemetry so blindness is an alarm, not a mystery.

Honest evaluation went from 1.8 tiles of progress per episode to over 20 after those fixes,
with every metric designed so it cannot be gamed. A MuJoCo twin of the whole task verifies
sim-to-sim transfer along the way.

Everything else I believe in is in there too: a clean-room asset policy (not one Duckietown
file is redistributed; the city is generated, the real DB21 robot model is fetched at use time
from Duckietown's own public simulator), 1,200+ unit tests, and reproducible runs.

The repo is open source, and I would love company: collaborate on the environments, challenge
the reward, bring your own maps, or help me close the loop by testing the trained policy on a
real Duckiebot.

A special thank you to my internship advisors, Valentin Guillet and David Bertoin, for
teaching me reinforcement learning and sim-to-real the way it is actually practiced, and for
advice that keeps paying off long after the internship ended. This project carries their
fingerprints.

Repo: https://github.com/MickyasTA/Issac_based_Duckiebot

#IsaacSim #ReinforcementLearning #Robotics #SimToReal #Duckietown #NVIDIA #GaussianSplatting
#OpenSource #RobotLearning #PPO`


---

# Short version (fits the 3,000 character limit)

I brought Duckietown into NVIDIA Isaac Sim, and taught a robot to drive from raw pixels.

An open-source integration of the Duckietown lane-following challenge into Isaac Sim 5.1 / Isaac Lab: a procedural city generator, a from-scratch vision PPO agent (no RL libraries), and 64 city layouts training in parallel on one laptop GPU. The policy's only view of the world is a 96x48 camera stream plus its own proprioception (wheel speeds, past actions); no ground-truth state ever reaches the actor. It now holds its lane at ~1.5 cm RMS through hairpins it once cut straight across.

The inspiration comes from my internship at IRT Saint-Exupery, where I worked on the hardest problem in robot learning: the sim-to-real gap. Isaac Sim is the most promising answer I have touched, and its flexibility is the point: the same scene can host procedural tiles today and a real-world environment reconstructed with 3D Gaussian Splatting tomorrow, closing the visual gap for real-to-sim. That is next on my roadmap.

The surprise: the reward is harder than the network. The optimizer found every crack I left open, and each fix is now a test in the repo:
- It cut corners because lane error was measured to the nearest lane, so crossing the line reset it to zero. Fix: continuity-constrained matching.
- It learned dying early was cheaper than driving badly. Fix: a survival income.
- It learned to park and collect that income while its vision encoder died a silent dead-ReLU death. Fix: motion-gated income, leaky activations, an lr ceiling, and encoder liveness alarms.

Honest evaluation went from 1.8 to 20+ tiles per episode. A MuJoCo twin verifies sim-to-sim transfer. Clean-room assets, 1,200+ tests, reproducible runs.

The repo is open source and I would love company: bring your maps, challenge the reward, or help test the policy on a real Duckiebot.

A special thank you to my internship advisors, Valentin Guillet and David Bertoin, for teaching me RL and sim-to-real as they are actually practiced. This project carries their fingerprints.

Repo: https://github.com/MickyasTA/Issac_based_Duckiebot

#IsaacSim #ReinforcementLearning #Robotics #SimToReal #Duckietown #OpenSource
