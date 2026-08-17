"""Configuration for the Duckiebot lane-following environment (SPEC v2 S5).

This module is the single source of truth for every number in S5: the rates, the episode length,
the observation and action spaces, the scene graph, the renderer settings and the VRAM budget.

Two layers, and the split is deliberate
---------------------------------------

* :class:`LaneFollowSettings` and everything it aggregates are plain dataclasses with no Isaac
  import at all. They validate themselves, they carry the whole S5 numeric content, and
  ``tests/unit/test_env_cfg.py`` checks them on a CPU-only runner. This is the same pattern
  :mod:`duckiebot_rl.assets.robot_cfg` uses, for the same reason: Isaac Sim is not a pip
  dependency of this repository and CI has no GPU.
* :func:`lane_follow_env_cfg` turns those settings into a real ``DirectRLEnvCfg``, importing
  ``isaaclab`` lazily inside the function body.

The renderer is configured HERE and only here
---------------------------------------------

Critic item G: ``--rendering_mode`` on the command line outranks ``RenderCfg.rendering_mode``
(``simulation_context.py:741-745``). If ``train.py`` exposed a rendering flag, an ablation could
silently run a different renderer than the config records, and the resulting number would be
unfalsifiable. ``scripts/train.py`` therefore exposes no rendering flag at all and
:class:`RenderingSettings` is the only place the renderer is chosen.

``antialiasing_mode`` is set through ``RenderCfg``, never through a raw ``/rtx/post/aa/op``
write, and the Isaac Lab setter wraps its own call in ``except Exception: pass`` - a failure is
SILENT and falls back to the preset's DLSS. :func:`expected_carb_settings` returns what M6 must
read back to prove the setting took, which is S4.4 acceptance item 3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from duckiebot_rl.assets.params import DUCKIEBOT, DuckiebotParams
from duckiebot_rl.dr.preprocess import OBS_CHANNELS, OBS_H, OBS_W, RENDER_H, RENDER_W

__all__ = [
    "ACT_DIM",
    "ANTIALIASING_CODES",
    "ANTIALIASING_SETTING",
    "CITY_USD_SEARCH_ROOTS",
    "GPU_MEMORY_BUDGET_SETTING",
    "PRIV_DIM",
    "VEC_DIM",
    "CitySettings",
    "LaneFollowSettings",
    "ObstacleSettings",
    "RateSettings",
    "RenderingSettings",
    "SpaceSettings",
    "VramBudget",
    "action_space_spec",
    "expected_carb_settings",
    "impoola_activation_floats",
    "lane_follow_env_cfg",
    "observation_space_spec",
    "resolve_city_assets",
    "state_space_spec",
    "vram_budget",
]

VEC_DIM: int = 8
"""Width of the actor's proprioceptive observation (SPEC v2 S5.2).

``[a_prev(2), a_prev2(2), wheel_speed_L, wheel_speed_R, imu_yaw_rate, odom_v]``, with the wheel
speeds encoder-quantized to 135 ticks/rev and subject to the D13 dropout.
"""

PRIV_DIM: int = 14
"""Width of the critic's privileged observation (SPEC v2 S5.2).

The 8 actor entries followed by ``[d, psi, curvature at +0.3 m lookahead,
dist_nearest_obstacle, rel_speed_nearest_obstacle, lane_progress_speed]``.
"""

ACT_DIM: int = 2
"""Action dimensionality: ``(a_v, a_omega)`` in ``[-1, 1]^2`` (SPEC v2 S5.3)."""

CITY_USD_SEARCH_ROOTS: tuple[str, ...] = ("build/city", "assets/city", "assets/usd")
"""Repository-relative directories searched for the generated city stages, in order.

``scripts/build_city.py`` defaults to ``--out build/city`` and writes ``<out>/usd/<name>.usda``
plus ``<out>/maps/<name>.yaml``. The README's quickstart passes ``--out assets/usd``. Both
layouts, and a flat directory with no ``usd/`` subdirectory, resolve here rather than failing
with an opaque USD stage-open error.
"""


# =============================================================================================
# Settings
# =============================================================================================


@dataclass(frozen=True)
class RateSettings:
    """Physics, control and rendering rates (SPEC v2 S5.2).

    Attributes:
        sim_dt_s: Physics step. 1/240 s, matching the MuJoCo harness exactly so the C0-vs-C5
            transfer delta carries no integration-rate confound (critic item J).
        decimation: Physics steps per control step. 16, giving 15 Hz control, which is the
            deployed ``car_cmd_switch_node`` rate.
        render_interval: Physics steps per render. Equal to ``decimation``: one render per
            control step, no more.
        episode_length_s: Truncation horizon in seconds.
        num_rerenders_on_reset: Extra full renders performed whenever ANY env resets. Must be 1
            for a camera env: with 0, ``TiledCamera`` hands the new episode the last frame of the
            previous one, because the annotator only advances on ``sim.render()``. The cost is
            budgeted, not free (S5.2, critic item I).
        stagger_initial_episode_length: Randomise ``episode_length_buf`` at the very first reset
            so that resets do not all land on the same step. Without it the rerender factor is a
            periodic spike instead of the ~43% steady-state average the budget assumes.
    """

    sim_dt_s: float = DUCKIEBOT.sim_dt_s
    decimation: int = DUCKIEBOT.decimation
    render_interval: int = DUCKIEBOT.decimation
    episode_length_s: float = 30.0
    num_rerenders_on_reset: int = 1
    stagger_initial_episode_length: bool = True

    def __post_init__(self) -> None:
        """Validate the rates.

        Raises:
            ValueError: If a rate is non-positive, if the render interval does not match the
                decimation, or if ``num_rerenders_on_reset`` is not 1.
        """
        if self.sim_dt_s <= 0.0 or self.decimation <= 0:
            raise ValueError(
                f"sim_dt_s and decimation must be positive, got ({self.sim_dt_s}, {self.decimation})"
            )
        if self.render_interval != self.decimation:
            raise ValueError(
                f"render_interval ({self.render_interval}) must equal decimation ({self.decimation}): "
                "S5.2 budgets exactly one render per control step"
            )
        if self.episode_length_s <= 0.0:
            raise ValueError(f"episode_length_s must be positive, got {self.episode_length_s}")
        if self.num_rerenders_on_reset != 1:
            raise ValueError(
                "num_rerenders_on_reset must be 1 (SPEC v2 S6.7 guard 4): with 0 the first "
                "observation of every new episode is the last frame of the previous one"
            )

    @property
    def control_dt_s(self) -> float:
        """Return the control period in seconds."""
        return self.sim_dt_s * self.decimation

    @property
    def control_hz(self) -> float:
        """Return the control rate in hertz."""
        return 1.0 / self.control_dt_s

    @property
    def max_episode_length(self) -> int:
        """Return the truncation horizon in control steps.

        Matches ``DirectRLEnv.max_episode_length``, which is
        ``ceil(episode_length_s / (sim.dt * decimation))``.
        """
        return math.ceil(self.episode_length_s / self.control_dt_s)


@dataclass(frozen=True)
class RenderingSettings:
    """Renderer configuration (SPEC v2 S5.2). The ONLY source of truth; see the module docstring.

    Attributes:
        rendering_mode: Kit rendering preset.
        antialiasing_mode: Passed through ``RenderCfg``, read back at M6.
        enable_shadows: Shadows on, per the S7.2 V5 row.
        gpu_memory_budget_mb: Caps the resource manager's GPU budget so a texture spill to host
            memory fails loudly instead of silently halving throughput (S5.6).

            SPEC v2 names the carb key ``rtx-transient.resourcemanager.
            maxTextureStreamingBudgetMB``. **That key does not exist in Isaac Sim 5.1.** Probed
            on this build it reads ``None``, and ``RenderCfg.carb_settings`` raises
            ``ValueError: '...' does not map to a carb setting``
            (``simulation_context.py:795``) rather than being ignored, so passing it makes the
            environment unconstructible. The key that does exist and expresses the same budget
            is ``rtx-transient.resourcemanager.UJITSO.GPUMemoryBudgetMB`` (default 2048 MB),
            alongside ``enableTextureStreaming``, which the M6 readback also checks.
        dome_light_upper_lower_strategy: Left at None, i.e. the performance preset's value of 3.
            Strategy 0 contradicts the preset and strategy 4 needs a denoiser the preset
            disables (critic item E); the consequence, that the dome barely lights diffuse
            surfaces, is accepted and is why V2r randomises a DistantLight instead.
        viewer_resolution: Resolution of the debug viewport when not headless.
    """

    rendering_mode: str = "performance"
    antialiasing_mode: str = "FXAA"
    enable_shadows: bool = True
    gpu_memory_budget_mb: int = 1024
    dome_light_upper_lower_strategy: int | None = None
    viewer_resolution: tuple[int, int] = (1280, 720)

    def __post_init__(self) -> None:
        """Validate the renderer choices.

        Raises:
            ValueError: If the rendering or antialiasing mode is not one Isaac Lab accepts.
        """
        if self.rendering_mode not in ("performance", "balanced", "quality"):
            raise ValueError(f"unknown rendering_mode {self.rendering_mode!r}")
        if self.antialiasing_mode not in ("Off", "FXAA", "DLSS", "TAA", "DLAA"):
            raise ValueError(f"unknown antialiasing_mode {self.antialiasing_mode!r}")


@dataclass(frozen=True)
class SpaceSettings:
    """Observation, state and action space shapes (SPEC v2 S5.2).

    Attributes:
        render_width: Camera render width in pixels.
        render_height: Camera render height in pixels.
        obs_width: Observation width after the S4.3 box downsample.
        obs_height: Observation height after the crop.
        obs_channels: Stacked channels, ``3 * len(FRAME_STACK_OFFSETS)``.
        vec_dim: Actor vector width.
        priv_dim: Critic privileged vector width.
        act_dim: Action width.
    """

    render_width: int = RENDER_W
    render_height: int = RENDER_H
    obs_width: int = OBS_W
    obs_height: int = OBS_H
    obs_channels: int = OBS_CHANNELS
    vec_dim: int = VEC_DIM
    priv_dim: int = PRIV_DIM
    act_dim: int = ACT_DIM

    def __post_init__(self) -> None:
        """Validate the shapes.

        Raises:
            ValueError: If the observation dimensions are not divisible by 8 (three stride-2
                pools in the Impoola encoder), or if the privileged vector is not a superset of
                the actor vector.
        """
        if self.obs_height % 8 or self.obs_width % 8:
            raise ValueError(
                f"obs dimensions must be divisible by 8 for the three stride-2 pools, got "
                f"({self.obs_height}, {self.obs_width})"
            )
        if self.priv_dim < self.vec_dim:
            raise ValueError(f"priv_dim ({self.priv_dim}) must be >= vec_dim ({self.vec_dim})")

    @property
    def rgb_shape(self) -> tuple[int, int, int]:
        """Return the NHWC-per-sample image observation shape ``(H, W, C)``."""
        return (self.obs_height, self.obs_width, self.obs_channels)

    @property
    def render_shape(self) -> tuple[int, int, int]:
        """Return the shape of one raw rendered frame, ``(H, W, 3)``."""
        return (self.render_height, self.render_width, 3)


@dataclass(frozen=True)
class CitySettings:
    """Which city stages the scene uses and how they are assigned to envs (SPEC v2 S5.1, S7.1).

    Attributes:
        root: Directory holding the generated city assets, or None to search
            :data:`CITY_USD_SEARCH_ROOTS` under the repository root.
        num_variants: Number of training layouts.
        variant_seed: Seed passed to ``duckiebot_rl.city.maps.variant_maps``; must match the
            ``--seed`` used by ``scripts/build_city.py`` or the lane graph will describe a
            different city than the one on screen.
        geometry_buckets: Number of marking-geometry texture buckets.
        eval_maps: Number of held-out layouts. Never used for training; ``scripts/play.py`` and
            the S8.4 protocol are the only consumers.
        random_choice: ``MultiUsdFileCfg.random_choice``. MUST stay False: with True the 64
            assets are sampled with replacement across 256 envs, roughly 1.2 layouts get zero
            envs and the counts are uneven; with False the spawner uses ``index % len`` and every
            layout gets exactly 4 envs, deterministically (critic item D).
        env_spacing: Grid spacing between envs, in metres.
        placement_half_extent: Half-extent of the per-env placement box. Everything the city
            generator emits, including off-road distractors, stays inside it.
    """

    root: str | None = None
    num_variants: int = 64
    variant_seed: int = 0
    geometry_buckets: int = 16
    eval_maps: int = 4
    random_choice: bool = False
    env_spacing: float = 8.0
    placement_half_extent: float = 3.6

    def __post_init__(self) -> None:
        """Validate the city settings.

        Raises:
            ValueError: If ``random_choice`` is True, if a count is non-positive, or if the
                placement box does not fit inside the env spacing.
        """
        if self.random_choice:
            raise ValueError(
                "MultiUsdFileCfg.random_choice must be False (critic item D): sampling 64 "
                "layouts with replacement over 256 envs leaves some layouts with zero envs"
            )
        if self.num_variants <= 0 or self.geometry_buckets <= 0:
            raise ValueError("num_variants and geometry_buckets must be positive")
        if 2.0 * self.placement_half_extent > self.env_spacing:
            raise ValueError(
                f"placement box 2 x {self.placement_half_extent} m does not fit inside "
                f"env_spacing {self.env_spacing} m"
            )


@dataclass(frozen=True)
class ObstacleSettings:
    """Obstacle field configuration (SPEC v2 S5.1, S5.5, S7.4).

    Attributes:
        enabled: Spawn the obstacle collection at all. Stages 0 and 1 of the task curriculum
            train without it, which also removes its VRAM and its PhysX cost.
        stage: Curriculum stage, 2 for "one leading NPC" and 3 for the full scenario sampler.
        density: Activation probability of each non-leading eligible slot.
        margin_m: The S5.5 geometric safety margin added to each obstacle radius.
    """

    enabled: bool = False
    stage: int = 3
    density: float = 0.5
    margin_m: float = 0.12

    def __post_init__(self) -> None:
        """Validate the obstacle settings.

        Raises:
            ValueError: If the density is outside ``[0, 1]`` or the margin is negative.
        """
        if not 0.0 <= self.density <= 1.0:
            raise ValueError(f"density must be in [0, 1], got {self.density}")
        if self.margin_m < 0.0:
            raise ValueError(f"margin_m must be >= 0, got {self.margin_m}")


@dataclass
class LaneFollowSettings:
    """The complete S5 environment specification, with no Isaac import anywhere.

    Attributes:
        num_envs: Parallel environment count ``N``.
        device: Torch / sim device string.
        seed: Environment seed. Seeds python ``random`` too, because ``MultiUsdFileCfg`` picks
            layouts with the stdlib generator (critic item D).
        rates: See :class:`RateSettings`.
        rendering: See :class:`RenderingSettings`.
        spaces: See :class:`SpaceSettings`.
        city: See :class:`CitySettings`.
        obstacles: See :class:`ObstacleSettings`.
        params: The shared robot parameter set.
        robot_usd_path: Path to the imported and patched robot USD, or None for the default.
        use_image: False drops the camera entirely and runs the vec-only mode of S6.2, which is
            what milestones M3 and M5 use.
        visual_dr: Enable the S4.3 step-3 photometric randomization.
        dynamics_dr: Enable the S7.3 dynamics randomization in the action path.
        dr_alpha_vis: Initial visual curriculum scalar.
        dr_alpha_dyn: Initial dynamics curriculum scalar.
    """

    num_envs: int = 256
    device: str = "cuda:0"
    seed: int = 0
    rates: RateSettings = field(default_factory=RateSettings)
    rendering: RenderingSettings = field(default_factory=RenderingSettings)
    spaces: SpaceSettings = field(default_factory=SpaceSettings)
    city: CitySettings = field(default_factory=CitySettings)
    obstacles: ObstacleSettings = field(default_factory=ObstacleSettings)
    params: DuckiebotParams = DUCKIEBOT
    robot_usd_path: str | None = None
    use_image: bool = True
    visual_dr: bool = True
    dynamics_dr: bool = True
    dr_alpha_vis: float = 0.0
    dr_alpha_dyn: float = 0.0

    def __post_init__(self) -> None:
        """Validate the aggregate.

        Raises:
            ValueError: If ``num_envs`` is non-positive or a curriculum scalar is out of range.
        """
        if self.num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {self.num_envs}")
        for name in ("dr_alpha_vis", "dr_alpha_dyn"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")

    @property
    def control_dt_s(self) -> float:
        """Return the control period in seconds."""
        return self.rates.control_dt_s

    @property
    def max_episode_length(self) -> int:
        """Return the truncation horizon in control steps."""
        return self.rates.max_episode_length

    def summary(self) -> dict[str, Any]:
        """Return a flat, JSON-serialisable summary for checkpoints and TensorBoard.

        Returns:
            A dict of the scalars a reader needs to reproduce the run.
        """
        return {
            "num_envs": self.num_envs,
            "device": self.device,
            "seed": self.seed,
            "sim_dt_s": self.rates.sim_dt_s,
            "decimation": self.rates.decimation,
            "control_hz": self.rates.control_hz,
            "episode_length_s": self.rates.episode_length_s,
            "max_episode_length": self.rates.max_episode_length,
            "num_rerenders_on_reset": self.rates.num_rerenders_on_reset,
            "rendering_mode": self.rendering.rendering_mode,
            "antialiasing_mode": self.rendering.antialiasing_mode,
            "rgb_shape": list(self.spaces.rgb_shape),
            "vec_dim": self.spaces.vec_dim,
            "priv_dim": self.spaces.priv_dim,
            "act_dim": self.spaces.act_dim,
            "num_variants": self.city.num_variants,
            "variant_seed": self.city.variant_seed,
            "env_spacing": self.city.env_spacing,
            "obstacles_enabled": self.obstacles.enabled,
            "obstacle_stage": self.obstacles.stage,
            "use_image": self.use_image,
            "visual_dr": self.visual_dr,
            "dynamics_dr": self.dynamics_dr,
        }


# =============================================================================================
# Gym spaces
# =============================================================================================


def observation_space_spec(settings: LaneFollowSettings) -> Any:
    """Return the actor observation space exactly as SPEC v2 S5.2 states it.

    ``Dict(rgb=Box(0, 255, (48, 96, 9), uint8), vec=Box(-inf, inf, (8,), float32))``. Real
    ``gymnasium`` spaces are built rather than the shorthand ``list[int]`` form, because the
    shorthand always produces ``Box(-inf, inf, ..., float32)`` and would misdeclare the image as
    an unbounded float tensor. ``spec_to_gym_space`` passes a ``gym.spaces.Space`` straight
    through, so this is a supported input.

    Args:
        settings: The environment settings.

    Returns:
        A ``gymnasium.spaces.Dict``; in vec-only mode, just the ``vec`` Box.
    """
    import gymnasium as gym
    import numpy as np

    vec = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(settings.spaces.vec_dim,), dtype=np.float32)
    if not settings.use_image:
        return gym.spaces.Dict({"vec": vec})
    rgb = gym.spaces.Box(low=0, high=255, shape=settings.spaces.rgb_shape, dtype=np.uint8)
    return gym.spaces.Dict({"rgb": rgb, "vec": vec})


def state_space_spec(settings: LaneFollowSettings) -> Any:
    """Return the asymmetric critic's state space (SPEC v2 S5.2).

    The image is the same tensor the actor sees; only the vector differs, carrying the six
    privileged fields the actor has no access to.

    Args:
        settings: The environment settings.

    Returns:
        A ``gymnasium.spaces.Dict``.
    """
    import gymnasium as gym
    import numpy as np

    priv = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(settings.spaces.priv_dim,), dtype=np.float32)
    if not settings.use_image:
        return gym.spaces.Dict({"vec_priv": priv})
    rgb = gym.spaces.Box(low=0, high=255, shape=settings.spaces.rgb_shape, dtype=np.uint8)
    return gym.spaces.Dict({"rgb": rgb, "vec_priv": priv})


def action_space_spec(settings: LaneFollowSettings) -> Any:
    """Return the action space ``Box(-1, 1, (2,), float32)`` (SPEC v2 S5.2).

    Args:
        settings: The environment settings.

    Returns:
        A ``gymnasium.spaces.Box``.
    """
    import gymnasium as gym
    import numpy as np

    return gym.spaces.Box(low=-1.0, high=1.0, shape=(settings.spaces.act_dim,), dtype=np.float32)


# =============================================================================================
# VRAM budget (SPEC v2 S5.6)
# =============================================================================================


def impoola_activation_floats(
    obs_height: int = OBS_H,
    obs_width: int = OBS_W,
    channels: tuple[int, ...] = (16, 32, 32),
    tensors_per_residual_block: int = 4,
    residual_blocks: int = 2,
) -> int:
    """Return the stored activation floats of one Impoola forward pass, per sample.

    Each ConvSequence is ``Conv3x3 -> MaxPool3x3/stride2 -> 2 residual blocks``, and each
    residual block stores 4 tensors at the post-pool resolution (conservative: it counts the
    ReLU outputs). At the S5.6 numbers this returns 389,376 floats = 1.56 MB fp32 per sample,
    which at minibatch 512 over two towers is the 1.49 GiB line of the budget table and is what
    forces ``num_minibatches = 16`` (critic item B: v1's minibatch 4096 needed 11.9 GiB).

    Args:
        obs_height: Observation height in pixels.
        obs_width: Observation width in pixels.
        channels: Per-ConvSequence output channels.
        tensors_per_residual_block: Stored tensors per residual block.
        residual_blocks: Residual blocks per ConvSequence.

    Returns:
        Activation floats per sample.
    """
    height, width = obs_height, obs_width
    total = 0
    for out_channels in channels:
        conv = out_channels * height * width
        height, width = height // 2, width // 2
        pooled = out_channels * height * width
        residual = tensors_per_residual_block * residual_blocks * pooled
        total += conv + pooled + residual
    return total


@dataclass(frozen=True)
class VramBudget:
    """The SPEC v2 S5.6 budget table at a given ``N`` and minibatch size.

    Every entry is ``(low, high)`` GiB. The consumers that this repository computes exactly
    (rollout buffer, camera output, PPO activations) are computed; the ones owned by Kit,
    PhysX and the driver are the spec's measured or estimated ranges.

    Attributes:
        num_envs: The ``N`` the table was computed for.
        minibatch_size: The PPO minibatch size.
        rollout_steps: The rollout length ``T``.
        items: Ordered mapping of budget line to ``(low, high)`` GiB.
    """

    num_envs: int
    minibatch_size: int
    rollout_steps: int
    items: dict[str, tuple[float, float]]

    @property
    def total(self) -> tuple[float, float]:
        """Return the ``(low, high)`` total in GiB."""
        low = sum(value[0] for value in self.items.values())
        high = sum(value[1] for value in self.items.values())
        return low, high

    @property
    def midpoint(self) -> float:
        """Return the midpoint of the total, in GiB."""
        low, high = self.total
        return 0.5 * (low + high)

    def as_table(self) -> str:
        """Render the budget as a fixed-width text table for the M6 report.

        Returns:
            A multi-line string.
        """
        width = max(len(name) for name in self.items)
        lines = [f"{'item'.ljust(width)}   low    high  (GiB)"]
        for name, (low, high) in self.items.items():
            lines.append(f"{name.ljust(width)}  {low:5.2f}  {high:5.2f}")
        low, high = self.total
        lines.append(f"{'TOTAL'.ljust(width)}  {low:5.2f}  {high:5.2f}")
        return "\n".join(lines)


def vram_budget(
    num_envs: int = 256,
    minibatch_size: int = 512,
    rollout_steps: int = 32,
    spaces: SpaceSettings | None = None,
) -> VramBudget:
    """Return the S5.6 VRAM budget with the computable lines actually computed.

    The M6 gate accepts ``N = 256`` only under 7.2 GiB measured by ``nvidia-smi``, never by
    ``torch.cuda`` accounting: Kit's allocations are invisible to torch, so a torch-side number
    would pass a run that is in fact spilling to host memory.

    Args:
        num_envs: Parallel environment count.
        minibatch_size: PPO minibatch size.
        rollout_steps: Rollout length ``T``.
        spaces: Observation shapes; defaults to the S5.2 shapes.

    Returns:
        The populated :class:`VramBudget`.
    """
    sp = spaces or SpaceSettings()
    gib = float(1 << 30)
    stacked_bytes = math.prod(sp.rgb_shape)
    rollout_gib = rollout_steps * num_envs * stacked_bytes / gib
    # vec + vec_priv + action + mu + log_std + logp + value + reward + 3 flags, all f32/bool.
    scalar_bytes = (sp.vec_dim + sp.priv_dim + 3 * sp.act_dim + 6) * 4 + 3
    rollout_gib += rollout_steps * num_envs * scalar_bytes / gib
    camera_gib = num_envs * sp.render_height * sp.render_width * 4 / gib
    activation_gib = 2.0 * minibatch_size * impoola_activation_floats(sp.obs_height, sp.obs_width) * 4 / gib

    items: dict[str, tuple[float, float]] = {
        "Windows WDDM / desktop reserve": (0.6, 1.0),
        "Kit + RTX renderer + BLAS + shaders": (2.0, 2.8),
        "City textures": (0.13, 0.13),
        "Dome HDRIs": (0.13, 0.13),
        "Tiled render target + G-buffers": (0.19, 0.25),
        "TiledCamera RGBA output": (camera_gib, camera_gib),
        "PhysX GPU buffers": (0.2, 0.3),
        "torch context + preprocess transients": (0.5, 0.5),
        "Rollout buffer (stacked uint8 obs)": (rollout_gib, rollout_gib),
        "PPO activations (two towers, fp32)": (activation_gib, activation_gib),
        "Params + grads + Adam moments": (0.02, 0.02),
        "Fragmentation slack ~12%": (0.8, 0.8),
    }
    return VramBudget(
        num_envs=num_envs, minibatch_size=minibatch_size, rollout_steps=rollout_steps, items=items
    )


GPU_MEMORY_BUDGET_SETTING: str = "/rtx-transient/resourcemanager/UJITSO/GPUMemoryBudgetMB"
"""Carb path of the resource manager's GPU budget, in megabytes.

Probed on Isaac Sim 5.1: default 2048. See :class:`RenderingSettings.gpu_memory_budget_mb` for
why the key SPEC v2 names is not usable on this build.
"""

ANTIALIASING_SETTING: str = "/rtx/post/aa/op"
"""Carb path holding the antialiasing operator, as an integer code."""

ANTIALIASING_CODES: dict[str, int] = {"Off": 0, "TAA": 1, "FXAA": 2, "DLSS": 3, "DLAA": 4}
"""Antialiasing name to carb integer code.

``RenderCfg.antialiasing_mode`` takes the NAME, but ``/rtx/post/aa/op`` stores an integer, so a
readback that compares against the string can never match no matter what the renderer did. The
performance preset leaves the setting at 3 (DLSS), which is exactly the value a silently failed
FXAA write would leave behind: ``rep.settings.set_render_rtx_realtime`` is wrapped in
``except Exception: pass`` at ``simulation_context.py:799-805``.
"""


def expected_carb_settings(rendering: RenderingSettings | None = None) -> dict[str, Any]:
    """Return the carb settings M6 must read back to prove the render config took effect.

    SPEC v2 S4.4 acceptance item 3: the Isaac Lab antialiasing setter is wrapped in
    ``except Exception: pass``, so a failure is silent and the run falls back to the preset's
    DLSS. Reading the setting back is the only way to know.

    Args:
        rendering: The renderer settings; defaults to :class:`RenderingSettings`.

    Returns:
        Mapping of carb setting path to expected value. Every path in it exists on Isaac Sim
        5.1; a readback of a nonexistent path returns ``None`` and would pass a comparison
        against ``None`` while proving nothing.
    """
    cfg = rendering or RenderingSettings()
    settings: dict[str, Any] = {
        ANTIALIASING_SETTING: ANTIALIASING_CODES[cfg.antialiasing_mode],
        GPU_MEMORY_BUDGET_SETTING: cfg.gpu_memory_budget_mb,
        "/rtx-transient/resourcemanager/enableTextureStreaming": True,
        "/rtx/shadows/enabled": cfg.enable_shadows,
    }
    if cfg.dome_light_upper_lower_strategy is not None:
        settings["/rtx/domeLight/upperLowerStrategy"] = cfg.dome_light_upper_lower_strategy
    return settings


# =============================================================================================
# Asset resolution
# =============================================================================================


def _repo_root() -> Path:
    """Return the repository root, two levels above this file."""
    return Path(__file__).resolve().parents[2]


def resolve_city_assets(city: CitySettings, include_eval: bool = False) -> tuple[list[str], list[str], str]:
    """Locate the generated city stages and their map YAMLs.

    Args:
        city: City settings carrying the root and the variant count.
        include_eval: Append the held-out eval layouts after the training ones.

    Returns:
        ``(usd_paths, map_paths, root)``. Both lists are in variant order and are the same
        length; ``root`` is the directory that was used, as a forward-slash string.

    Raises:
        FileNotFoundError: If no search root contains the stages, with the exact list of
            directories that were tried and the command that generates them.
    """
    # An explicit root is authoritative. Falling through to the defaults when it comes up empty
    # would quietly load a DIFFERENT city than the caller asked for, and the lane graph would
    # then describe a layout nobody selected.
    roots: list[Path] = (
        [Path(city.root)]
        if city.root is not None
        else [_repo_root() / name for name in CITY_USD_SEARCH_ROOTS]
    )

    names = [f"city_{index:03d}" for index in range(city.num_variants)]
    if include_eval:
        names.extend(f"eval_{index:02d}" for index in range(city.eval_maps))

    tried: list[str] = []
    for root in roots:
        for usd_dir, map_dir in ((root / "usd", root / "maps"), (root, root)):
            tried.append(usd_dir.as_posix())
            usd_paths = [usd_dir / f"{name}.usda" for name in names]
            if not all(path.is_file() for path in usd_paths):
                continue
            map_paths = [map_dir / f"{name}.yaml" for name in names]
            if not all(path.is_file() for path in map_paths):
                continue
            return (
                [path.as_posix() for path in usd_paths],
                [path.as_posix() for path in map_paths],
                root.as_posix(),
            )

    raise FileNotFoundError(
        f"could not find {len(names)} city stage(s) plus their map YAMLs. Looked in: "
        + ", ".join(tried)
        + ". Generate them with: python scripts/build_city.py --all --out build/city "
        f"--variants {city.num_variants} --seed {city.variant_seed} "
        f"--buckets {city.geometry_buckets}"
    )


def resolve_ground_usd(root: str) -> str:
    """Return the path of the generated ground plane inside a city build root.

    Args:
        root: A directory returned by :func:`resolve_city_assets`.

    Returns:
        The ground stage path with forward slashes.

    Raises:
        FileNotFoundError: If neither candidate location holds ``ground.usda``.
    """
    for candidate in (Path(root) / "usd" / "ground.usda", Path(root) / "ground.usda"):
        if candidate.is_file():
            return candidate.as_posix()
    raise FileNotFoundError(f"ground.usda not found under {root}; run scripts/build_city.py to generate it")


# =============================================================================================
# The Isaac config
# =============================================================================================


def lane_follow_env_cfg(settings: LaneFollowSettings | None = None, **overrides: Any) -> Any:
    """Build the ``DirectRLEnvCfg`` for the lane-following task.

    Isaac Lab is imported inside this function, so importing this module on a CPU-only runner
    stays free. The scene is declared as a full ``InteractiveSceneCfg`` subclass rather than
    assembled imperatively in ``_setup_scene``, which means ``InteractiveScene`` performs the
    heterogeneous clone and the automatic ``filter_collisions`` pass itself (critic item E: with
    ``replicate_physics=False`` and ``filter_collisions=True`` that call is already made for you
    at ``interactive_scene.py:214-215``, and a manual call afterwards hits an early-out and does
    nothing).

    Args:
        settings: The plain settings. Defaults to :class:`LaneFollowSettings`.
        **overrides: Field values applied to a copy of ``settings`` before the config is built.

    Returns:
        A fully populated ``DuckiebotLaneFollowEnvCfg``.

    Raises:
        ImportError: If Isaac Lab is not importable in this interpreter.
        FileNotFoundError: If the robot USD or the city stages have not been built.
    """
    base = settings or LaneFollowSettings()
    if overrides:
        current = {f.name: getattr(base, f.name) for f in fields(base)}
        unknown = set(overrides) - set(current)
        if unknown:
            raise ValueError(f"unknown LaneFollowSettings field(s): {sorted(unknown)}")
        current.update(overrides)
        base = LaneFollowSettings(**current)

    try:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg
        from isaaclab.envs import DirectRLEnvCfg
        from isaaclab.scene import InteractiveSceneCfg
        from isaaclab.sensors import TiledCameraCfg
        from isaaclab.sim import PhysxCfg, RenderCfg, SimulationCfg
        from isaaclab.sim.spawners.wrappers import MultiUsdFileCfg
        from isaaclab.utils import configclass
    except ImportError as exc:  # pragma: no cover - exercised only inside Isaac
        raise ImportError(
            "Isaac Lab is not importable from this interpreter. lane_follow_env_cfg() only runs "
            "inside the Isaac Sim python environment, and the AppLauncher must have been "
            "constructed first. CPU-only tooling should use LaneFollowSettings, vram_budget() "
            "and the *_space_spec() helpers, none of which need Isaac."
        ) from exc

    from duckiebot_rl.assets.robot_cfg import DEFAULT_USD_PATH, duckiebot_articulation_cfg
    from duckiebot_rl.envs.camera_math import pinhole_camera_kwargs, quat_cam_ros
    from duckiebot_rl.envs.obstacles import obstacle_collection_cfg

    params = base.params
    usd_paths, map_paths, city_root = resolve_city_assets(base.city)
    ground_path = resolve_ground_usd(city_root)
    robot_usd = base.robot_usd_path or DEFAULT_USD_PATH
    if not Path(robot_usd).is_file():
        raise FileNotFoundError(
            f"robot USD not found at {robot_usd}; build it with python tools/import_urdf_headless.py"
        )

    camera_cfg = TiledCameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{params.base_link_name}/front_cam",
        offset=TiledCameraCfg.OffsetCfg(
            pos=params.camera_pos_base_frame_m,
            rot=quat_cam_ros(math.radians(params.camera_pitch_down_deg)),
            convention="ros",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(**pinhole_camera_kwargs(params)),
        width=base.spaces.render_width,
        height=base.spaces.render_height,
        # Left at the default False. The consequence is documented rather than hidden: camera
        # pose data (`camera.data.pos_w`) stays permanently stale, so any debug overlay must
        # recompute the pose from the robot root state and the known mount offset (S5.1).
        update_latest_camera_pose=False,
    )

    @configclass
    class DuckiebotSceneCfg(InteractiveSceneCfg):
        """Scene graph of SPEC v2 S5.1. Declaration order is spawn order."""

        ground: AssetBaseCfg = AssetBaseCfg(
            prim_path="/World/ground",
            spawn=sim_utils.UsdFileCfg(usd_path=ground_path),
            # -1 puts the plane in the global collision group, which is also what makes
            # InteractiveScene add it to the global prim paths of the automatic collision filter.
            collision_group=-1,
        )
        dome_light: AssetBaseCfg = AssetBaseCfg(
            prim_path="/World/DomeLight",
            spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(0.9, 0.92, 1.0)),
        )
        sun_light: AssetBaseCfg = AssetBaseCfg(
            prim_path="/World/SunLight",
            spawn=sim_utils.DistantLightCfg(
                intensity=1500.0,
                angle=1.0,
                enable_color_temperature=True,
                color_temperature=5200.0,
            ),
        )
        city: AssetBaseCfg = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/City",
            spawn=MultiUsdFileCfg(usd_path=list(usd_paths), random_choice=base.city.random_choice),
        )
        robot = duckiebot_articulation_cfg(usd_path=robot_usd, params=params)
        # Declared AFTER the robot: InteractiveScene spawns cfg fields in declaration order, and
        # the camera prim lives under a robot link that must already exist. Declared as a field
        # rather than attached afterwards so that `None` is a first-class value (vec-only mode)
        # and the entity walk skips it cleanly.
        camera = camera_cfg if base.use_image else None
        lamp: AssetBaseCfg = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Lamp",
            spawn=sim_utils.SphereLightCfg(intensity=300.0, radius=0.1),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 1.6)),
        )
        obstacles = obstacle_collection_cfg() if base.obstacles.enabled else None

    scene_cfg = DuckiebotSceneCfg(
        num_envs=base.num_envs,
        env_spacing=base.city.env_spacing,
        # False is mandatory: USD-level per-env visual randomization needs heterogeneous envs,
        # and MultiUsdFileCfg gives each env a different city stage.
        replicate_physics=False,
        # clone_in_fabric requires replicate_physics=True, so it must be False here.
        clone_in_fabric=False,
        filter_collisions=True,
    )

    render_cfg = RenderCfg(
        rendering_mode=base.rendering.rendering_mode,
        antialiasing_mode=base.rendering.antialiasing_mode,
        enable_shadows=base.rendering.enable_shadows,
        dome_light_upper_lower_strategy=base.rendering.dome_light_upper_lower_strategy,
        carb_settings={GPU_MEMORY_BUDGET_SETTING: base.rendering.gpu_memory_budget_mb},
    )

    @configclass
    class DuckiebotLaneFollowEnvCfg(DirectRLEnvCfg):
        """The SPEC v2 S5 environment config."""

        decimation: int = base.rates.decimation
        episode_length_s: float = base.rates.episode_length_s
        num_rerenders_on_reset: int = base.rates.num_rerenders_on_reset
        seed: int = base.seed
        sim: SimulationCfg = SimulationCfg(
            dt=base.rates.sim_dt_s,
            render_interval=base.rates.render_interval,
            device=base.device,
            render=render_cfg,
            physx=PhysxCfg(
                gpu_max_rigid_contact_count=2**20,
                gpu_heap_capacity=2**23,
                gpu_collision_stack_size=2**24,
            ),
        )
        # The local is deliberately named `scene_cfg`: inside a class body the target of
        # `scene: ... = scene` is a class-local name, so the right-hand side would resolve
        # to the not-yet-defined class attribute rather than to the enclosing function's
        # variable, and the class body raises NameError.
        scene: InteractiveSceneCfg = scene_cfg
        observation_space = observation_space_spec(base)
        state_space = state_space_spec(base)
        action_space = action_space_spec(base)
        # No UI window: it is dead weight headless and it holds a reference back to the env.
        ui_window_class_type = None
        # Carried on the cfg so the environment, the checkpoint writer and the TensorBoard run
        # metadata all read the same object instead of three partial copies.
        settings: LaneFollowSettings = base

    cfg = DuckiebotLaneFollowEnvCfg()
    cfg.viewer.resolution = base.rendering.viewer_resolution
    # Attached after construction rather than declared as class attributes: a list default on a
    # configclass body is a mutable class attribute, and the config is built exactly once per
    # process, so there is nothing for a per-instance default factory to protect.
    cfg.map_paths = list(map_paths)
    cfg.city_root = city_root
    return cfg
