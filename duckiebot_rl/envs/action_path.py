"""The SPEC v2 S5.3 action path in torch, batched over ``N`` envs.

S5.3 says the chain is "one class, with a torch-free numpy twin verified equal by unit test".
:class:`duckiebot_rl.sim2sim.env.ActionPath` is the numpy half and already ships; this is the
torch half that the Isaac environment runs. The two are kept step-for-step identical on purpose,
including step 4's asymmetric ``max`` form: a "cleverer" symmetric brake here would be a silent
divergence from the dynamics the policy was trained against, and it would land in every
sim-to-sim transfer number without appearing in any diff of the reward or the network.

It lives in its own module rather than inside ``lane_follow_env.py`` because that file imports
Isaac Lab at module scope. Six steps of motor modelling that only run inside a Kit process are
six steps nobody can unit-test, and this is the part of the environment where a sign error is
both easy to make and invisible until a sim-to-real gap shows up months later.

The six steps, and what each one is protecting against
------------------------------------------------------

1. **Inverse kinematics** with the per-episode randomised gain, trim, baseline and per-wheel
   radius (D1, D2, D5). Radius asymmetry is what makes a straight-line command curve.
2. **Actuation delay** (D8): a whole-step ring plus sub-step linear interpolation, then a
   first-order motor lag. S2 calls the 0.150 s delay the dominant dynamics gap. The ring is
   PRE-FILLED with zeros on reset, never emptied: with an empty ring the delayed read clamps to
   the newest entry and the first ``delay_steps`` control steps of every episode would bypass
   the delay entirely and answer the first command at full commanded wheel speed.
3. **Dead-band** (D7): below the release duty the hardware COASTS. Commanding the CURRENT wheel
   speed makes the implicit velocity servo produce ~zero torque, which is a coast. Commanding
   zero would be an active brake, which the DB21 cannot do (critic item F).
4. **Brake authority** (D18): back-EMF braking is partial, so a falling target is floored at
   ``w_current - beta * DW_MAX``.
5. **Wheel slip** (D12) and **battery sag** (D6) scale the realised target.
6. **Clamp** to the wheel velocity limit, then the caller writes it with
   ``Articulation.set_joint_velocity_target``.

No physical drive parameter is ever written per reset. ``randomize_actuator_gains`` forces a CPU
sync on an implicit actuator and would fire on ~43% of steps at ``N = 256`` (critic item E); D17,
the per-env effort limit, is the one exception and is written once at startup.
"""

from __future__ import annotations

import torch

from duckiebot_rl.assets.params import DUCKIEBOT, DuckiebotParams
from duckiebot_rl.dr.delay import DelayBuffer

__all__ = ["TorchActionPath"]


class TorchActionPath:
    """Batched torch implementation of the S5.3 action path.

    Args:
        num_envs: Number of parallel envs.
        control_dt: Control period in seconds.
        params: Shared robot parameters.
        device: Torch device.
        generator: Torch generator, for the per-step slip noise.
        max_delay_steps: Ring depth in control steps. Defaults to the larger of the D8 upper
            bound and the nominal ``actuation_delay_s / control_dt``, so neither the DR range
            nor the nominal delay can silently be clipped.
    """

    def __init__(
        self,
        num_envs: int,
        control_dt: float = DUCKIEBOT.control_dt_s,
        params: DuckiebotParams = DUCKIEBOT,
        device: torch.device | str = "cpu",
        generator: torch.Generator | None = None,
        max_delay_steps: int | None = None,
    ) -> None:
        self.num_envs = int(num_envs)
        self.control_dt = float(control_dt)
        self.params = params
        self.device = torch.device(device)
        self.generator = generator

        nominal_delay = round(params.actuation_delay_s / control_dt)
        self.max_delay_steps = (
            max(int(params.dr_delay_control_steps[1]), nominal_delay)
            if max_delay_steps is None
            else int(max_delay_steps)
        )
        self.delay = DelayBuffer(
            self.num_envs,
            (2,),
            max_delay=self.max_delay_steps,
            dtype=torch.float32,
            device=self.device,
            backend="torch",
            interpolate=True,
        )
        self.lagged = torch.zeros(self.num_envs, 2, dtype=torch.float32, device=self.device)
        self.episode_time_s = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.delay.set_delay(nominal_delay, 0.0)

    @property
    def nominal_delay_steps(self) -> int:
        """Return the whole-step actuation delay implied by S2 at this control period."""
        return round(self.params.actuation_delay_s / self.control_dt)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Clear the lag state and pre-fill the delay ring with zero targets.

        Args:
            env_ids: Long tensor of env indices, or None for all envs.
        """
        if env_ids is None:
            self.delay.reset(None, None)
            self.lagged.zero_()
            self.episode_time_s.zero_()
            return
        ids = env_ids.to(device=self.device, dtype=torch.long)
        self.delay.reset(ids, None)
        self.lagged[ids] = 0.0
        self.episode_time_s[ids] = 0.0

    def set_delay(self, delay_steps: torch.Tensor, delay_frac: torch.Tensor) -> None:
        """Apply the D8 per-env delay drawn on episode reset.

        Args:
            delay_steps: ``(N,)`` integer whole-step delay.
            delay_frac: ``(N,)`` sub-step fraction in ``[0, 1)``.
        """
        steps = torch.clamp(delay_steps.to(torch.long), 0, self.max_delay_steps)
        self.delay.set_delay(steps, delay_frac.to(torch.float32))

    def __call__(
        self,
        action: torch.Tensor,
        wheel_velocity: torch.Tensor,
        dr: object,
        apply_dr: bool = True,
    ) -> torch.Tensor:
        """Map a batch of policy actions to a batch of wheel-velocity targets.

        Args:
            action: ``(N, 2)`` raw policy action. Clipped to ``[-1, 1]`` here, which is the only
                place clipping happens; the buffer stores the unclipped Gaussian sample.
            wheel_velocity: ``(N, 2)`` measured ``(left, right)`` wheel speeds in rad/s.
            dr: A :class:`duckiebot_rl.dr.dynamics.DynamicsParams`. Typed as ``object`` so this
                module needs no import from the DR package at type-check time.
            apply_dr: When False, the drive is nominal: unit gain, zero trim, nominal radius and
                baseline, no dead-band, no slip, full brake authority. The delay ring still runs
                at the S2 nominal, because the 0.150 s actuation delay is a property of the
                hardware and not a randomization axis.

        Returns:
            ``(N, 2)`` wheel-velocity targets in rad/s, clamped to the S2 velocity limit.
        """
        params = self.params
        a = torch.clamp(action, -1.0, 1.0)
        v_cmd = 0.5 * params.v_cmd_max_m_s * (a[:, 0] + 1.0)
        om_cmd = params.omega_cmd_max_rad_s * a[:, 1]

        if apply_dr:
            radius = dr.wheel_radius_m  # type: ignore[attr-defined]
            baseline = dr.baseline_m  # type: ignore[attr-defined]
            gain = dr.motor_gain  # type: ignore[attr-defined]
            trim = dr.motor_trim  # type: ignore[attr-defined]
            deadband = dr.deadband_duty  # type: ignore[attr-defined]
            brake_beta = dr.brake_beta  # type: ignore[attr-defined]
            slip_std = dr.slip_std  # type: ignore[attr-defined]
            sag = dr.battery_sag - dr.battery_decay_per_s * self.episode_time_s  # type: ignore[attr-defined]
        else:
            ones = torch.ones(self.num_envs, device=self.device)
            radius = torch.full((self.num_envs, 2), params.wheel_radius_m, device=self.device)
            baseline = ones * params.wheel_baseline_m
            gain = ones * params.motor_gain_nominal
            trim = ones * params.motor_trim_nominal
            deadband = torch.zeros(self.num_envs, 2, device=self.device)
            brake_beta = ones
            slip_std = torch.zeros(self.num_envs, device=self.device)
            sag = ones

        # 1. inverse kinematics with randomized gain, trim, baseline and per-wheel radius.
        half_turn = 0.5 * om_cmd * baseline
        target = torch.stack(
            [
                (v_cmd - half_turn) / radius[:, 0] * gain * (1.0 - trim),
                (v_cmd + half_turn) / radius[:, 1] * gain * (1.0 + trim),
            ],
            dim=1,
        )

        # 2. actuation delay D8, then the first-order motor lag.
        delayed = self.delay.step(target)
        alpha = dr.motor_lag_alpha.unsqueeze(-1) if apply_dr else 1.0  # type: ignore[attr-defined]
        self.lagged = self.lagged + alpha * (delayed - self.lagged)
        target = self.lagged.clone()

        # 3. dead-band D7: below the release duty the hardware COASTS, it does not brake.
        duty = target / params.motor_constant_k_rad_s_per_duty
        threshold = torch.clamp(deadband, min=params.pwm_release_duty)
        coasting = duty.abs() < threshold
        target = torch.where(coasting, wheel_velocity, target)

        # 4. brake authority D18: bound how fast a target may fall. The spec's asymmetric max()
        #    form is reproduced verbatim; see the module docstring.
        floor = wheel_velocity - brake_beta.unsqueeze(-1) * params.brake_dw_max_rad_s_per_step
        target = torch.maximum(target, floor)

        # 5. wheel slip D12 and battery sag D6 scale the realized target.
        noise = torch.rand(target.shape, generator=self.generator, device=self.device, dtype=target.dtype)
        slip = 1.0 + (2.0 * noise - 1.0) * slip_std.unsqueeze(-1)
        target = target * sag.unsqueeze(-1) * slip

        # 6. clamp to the S2 wheel velocity limit.
        limit = params.wheel_velocity_limit_rad_s
        self.episode_time_s += self.control_dt
        return torch.clamp(target, -limit, limit)

    def state_dict(self) -> dict[str, object]:
        """Serialise the delay ring and the lag state for a checkpoint.

        Returns:
            A mapping with ``delay``, ``lagged`` and ``episode_time_s``.
        """
        return {
            "delay": self.delay.state_dict(),
            "lagged": self.lagged.clone(),
            "episode_time_s": self.episode_time_s.clone(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore state produced by :meth:`state_dict`.

        Args:
            state: The mapping returned by :meth:`state_dict`.

        Raises:
            KeyError: If a field is missing.
        """
        for key in ("delay", "lagged", "episode_time_s"):
            if key not in state:
                raise KeyError(f"action-path state is missing field {key!r}")
        self.delay.load_state_dict(state["delay"])  # type: ignore[arg-type]
        self.lagged.copy_(state["lagged"])  # type: ignore[arg-type]
        self.episode_time_s.copy_(state["episode_time_s"])  # type: ignore[arg-type]
