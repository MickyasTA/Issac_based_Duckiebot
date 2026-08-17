"""Consistency of the SPEC v2 S5 environment configuration (owner ``[env]``).

None of this needs Isaac. ``duckiebot_rl.envs.env_cfg`` keeps the whole S5 numeric content in
plain dataclasses precisely so that the rates, the space shapes and the VRAM arithmetic are
checked on the CPU-only CI runner, and only the ``DirectRLEnvCfg`` assembly is behind a lazy
``isaaclab`` import.

Three families of assertion:

* **Rates.** ``sim.dt`` and ``decimation`` must give 15 Hz, which is the deployed
  ``car_cmd_switch_node`` rate, and 30 s must give 450 control steps.
* **Spaces.** The declared observation space has to match, shape for shape and dtype for dtype,
  what ``_get_observations`` will actually produce. A mismatch here is invisible until a network
  is built against the wrong width.
* **VRAM.** The S5.6 budget is the reason ``num_minibatches`` is 16 rather than v1's 2. The
  activation arithmetic is recomputed from the observation shape, so shrinking the image without
  re-deriving the budget cannot pass unnoticed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from duckiebot_rl.assets.params import DUCKIEBOT  # noqa: E402
from duckiebot_rl.dr.preprocess import OBS_CHANNELS, OBS_H, OBS_W, RENDER_H, RENDER_W  # noqa: E402
from duckiebot_rl.envs import env_cfg as ec  # noqa: E402
from duckiebot_rl.ppo.config import NetworkConfig, PPOConfig  # noqa: E402

# =============================================================================================
# Rates
# =============================================================================================


def test_control_rate_is_fifteen_hertz():
    """1/240 s physics with decimation 16 is the deploy rate, and nothing else is acceptable."""
    rates = ec.RateSettings()
    assert rates.sim_dt_s == pytest.approx(1.0 / 240.0)
    assert rates.decimation == 16
    assert rates.control_hz == pytest.approx(15.0, abs=1e-9)
    assert rates.control_dt_s == pytest.approx(1.0 / 15.0, abs=1e-12)


def test_control_rate_matches_the_shared_robot_parameters():
    """The rates come from assets/params.py, not from a second copy declared here."""
    rates = ec.RateSettings()
    assert rates.sim_dt_s == DUCKIEBOT.sim_dt_s
    assert rates.decimation == DUCKIEBOT.decimation
    assert rates.control_dt_s == pytest.approx(DUCKIEBOT.control_dt_s, abs=1e-12)


def test_physics_rate_matches_the_mujoco_harness():
    """Critic item J: a 2x integration-rate difference would confound the C0-vs-C5 delta."""
    from duckiebot_rl.sim2sim.env import MjEnvCfg

    mj = MjEnvCfg()
    physics_dt = mj.physics_dt if mj.physics_dt is not None else DUCKIEBOT.sim_dt_s
    decimation = mj.decimation if mj.decimation is not None else DUCKIEBOT.decimation
    rates = ec.RateSettings()
    assert physics_dt == pytest.approx(rates.sim_dt_s)
    assert decimation == rates.decimation


def test_episode_horizon_is_450_control_steps():
    """30 s at 15 Hz. Isaac computes ceil(episode_length_s / (sim.dt * decimation))."""
    rates = ec.RateSettings()
    assert rates.episode_length_s == 30.0
    assert rates.max_episode_length == 450


def test_one_render_per_control_step():
    """render_interval must equal decimation; anything smaller renders more than once per step."""
    with pytest.raises(ValueError, match="render_interval"):
        ec.RateSettings(render_interval=8)


def test_rerender_on_reset_is_mandatory():
    """S6.7 guard 4: with 0, the first frame of a new episode is the last frame of the old one."""
    with pytest.raises(ValueError, match="num_rerenders_on_reset"):
        ec.RateSettings(num_rerenders_on_reset=0)


def test_nonsensical_rates_are_rejected():
    with pytest.raises(ValueError, match="must be positive"):
        ec.RateSettings(sim_dt_s=0.0)
    with pytest.raises(ValueError, match="episode_length_s"):
        ec.RateSettings(episode_length_s=-1.0)


# =============================================================================================
# Spaces
# =============================================================================================


def test_observation_shapes_come_from_the_preprocess_constants():
    """The env cfg cannot drift from the S4.3 chain that produces the tensor."""
    spaces = ec.SpaceSettings()
    assert spaces.rgb_shape == (OBS_H, OBS_W, OBS_CHANNELS) == (48, 96, 9)
    assert spaces.render_shape == (RENDER_H, RENDER_W, 3) == (128, 192, 3)


def test_vector_widths_are_the_spec_widths():
    """8 proprioceptive entries for the actor, 14 for the asymmetric critic."""
    assert ec.VEC_DIM == 8
    assert ec.PRIV_DIM == 14
    assert ec.ACT_DIM == 2


def test_privileged_vector_extends_the_actor_vector_by_exactly_six_fields():
    """d, psi, curvature, obstacle distance, closing speed, along-lane speed."""
    assert ec.PRIV_DIM - ec.VEC_DIM == 6


def test_env_spaces_match_the_ppo_network_config():
    """The learner is built from NetworkConfig; the two declarations must agree."""
    spaces = ec.SpaceSettings()
    net = NetworkConfig()
    assert (net.obs_height, net.obs_width, net.obs_channels) == spaces.rgb_shape
    assert net.vec_dim == spaces.vec_dim
    assert net.priv_dim == spaces.priv_dim
    assert net.act_dim == spaces.act_dim


def test_observation_space_is_the_exact_spec_box():
    """uint8 in [0, 255] for the image, float32 unbounded for the vector."""
    import numpy as np

    settings = ec.LaneFollowSettings()
    space = ec.observation_space_spec(settings)
    assert set(space.spaces) == {"rgb", "vec"}
    assert space["rgb"].shape == (48, 96, 9)
    assert space["rgb"].dtype == np.uint8
    assert space["rgb"].low.min() == 0 and space["rgb"].high.max() == 255
    assert space["vec"].shape == (8,)
    assert space["vec"].dtype == np.float32


def test_state_space_carries_the_privileged_vector():
    settings = ec.LaneFollowSettings()
    space = ec.state_space_spec(settings)
    assert set(space.spaces) == {"rgb", "vec_priv"}
    assert space["vec_priv"].shape == (14,)


def test_action_space_is_the_unit_box():
    import numpy as np

    space = ec.action_space_spec(ec.LaneFollowSettings())
    assert space.shape == (2,)
    assert space.dtype == np.float32
    assert space.low.tolist() == [-1.0, -1.0]
    assert space.high.tolist() == [1.0, 1.0]


def test_vec_only_mode_drops_the_image_from_both_spaces():
    """M3 and M5 train state-based; the image key must be absent, not present and zeroed."""
    settings = ec.LaneFollowSettings(use_image=False)
    assert set(ec.observation_space_spec(settings).spaces) == {"vec"}
    assert set(ec.state_space_spec(settings).spaces) == {"vec_priv"}


def test_observation_dimensions_survive_three_stride_two_pools():
    """The Impoola encoder halves the map three times; 48x96 -> 6x12."""
    spaces = ec.SpaceSettings()
    assert spaces.obs_height % 8 == 0 and spaces.obs_width % 8 == 0
    with pytest.raises(ValueError, match="divisible by 8"):
        ec.SpaceSettings(obs_height=50)


# =============================================================================================
# VRAM budget (SPEC v2 S5.6)
# =============================================================================================


def test_impoola_activation_arithmetic_matches_the_spec():
    """S5.6 shows the arithmetic: 239,616 + 119,808 + 29,952 = 389,376 floats per sample."""
    assert ec.impoola_activation_floats() == 389_376


def test_activation_bytes_per_sample():
    """1.56 MB fp32 per sample is the number the minibatch size is derived from."""
    megabytes = ec.impoola_activation_floats() * 4 / 1e6
    assert megabytes == pytest.approx(1.56, abs=0.01)


def test_two_towers_at_minibatch_512_cost_the_budgeted_activation_memory():
    """Critic item B: v1's minibatch 4096 needed 11.9 GiB by this same arithmetic."""
    budget = ec.vram_budget(num_envs=256, minibatch_size=512)
    low, _high = budget.items["PPO activations (two towers, fp32)"]
    assert low == pytest.approx(1.49, abs=0.02)


def test_the_v1_minibatch_would_have_blown_the_whole_card():
    """Recompute the number that forced num_minibatches from 2 to 16."""
    budget = ec.vram_budget(num_envs=256, minibatch_size=4096)
    low, _high = budget.items["PPO activations (two towers, fp32)"]
    assert low == pytest.approx(11.9, abs=0.2)


def test_rollout_buffer_line_is_the_stacked_uint8_store():
    """32 * 256 * 48 * 96 * 9 bytes = 324 MiB, the cost of storing stacks instead of frames."""
    budget = ec.vram_budget()
    low, _high = budget.items["Rollout buffer (stacked uint8 obs)"]
    image_mib = 32 * 256 * 48 * 96 * 9 / (1 << 20)
    assert image_mib == pytest.approx(324.0, abs=0.5)
    assert low == pytest.approx(image_mib / 1024.0, abs=0.02)


def test_total_budget_lands_on_the_spec_table():
    """S5.6 quotes 6.4 - 7.6 GiB, midpoint ~7.1, against a 7.2 GiB M6 gate at N = 256."""
    budget = ec.vram_budget()
    low, high = budget.total
    assert low == pytest.approx(6.4, abs=0.1)
    assert high == pytest.approx(7.7, abs=0.2)
    assert 6.9 < budget.midpoint < 7.3


def test_budget_table_renders_every_line():
    text = ec.vram_budget().as_table()
    assert "TOTAL" in text
    assert text.count("\n") == len(ec.vram_budget().items) + 1


def test_the_ppo_minibatch_size_is_the_one_the_budget_assumes():
    """The budget is computed at minibatch 512; PPOConfig must actually produce 512."""
    cfg = PPOConfig()
    assert cfg.num_envs == 256
    assert cfg.num_steps == 32
    assert cfg.num_minibatches == 16
    assert cfg.minibatch_size == 512
    assert cfg.batch_size % cfg.num_minibatches == 0


def test_default_settings_and_ppo_config_agree_on_the_env_count():
    assert ec.LaneFollowSettings().num_envs == PPOConfig().num_envs


# =============================================================================================
# Renderer: the single source of truth
# =============================================================================================


def test_renderer_defaults_are_the_spec_choices():
    """Performance preset, FXAA, shadows on, 1024 MB pinned texture budget."""
    rendering = ec.RenderingSettings()
    assert rendering.rendering_mode == "performance"
    assert rendering.antialiasing_mode == "FXAA"
    assert rendering.enable_shadows is True
    assert rendering.gpu_memory_budget_mb == 1024


def test_dome_strategy_is_left_at_the_preset_value():
    """Critic item E: strategy 0 contradicts the preset and 4 needs a disabled denoiser."""
    assert ec.RenderingSettings().dome_light_upper_lower_strategy is None


def test_unknown_renderer_modes_are_rejected():
    with pytest.raises(ValueError, match="rendering_mode"):
        ec.RenderingSettings(rendering_mode="ultra")
    with pytest.raises(ValueError, match="antialiasing_mode"):
        ec.RenderingSettings(antialiasing_mode="MSAA")


def test_carb_readback_compares_against_the_integer_code_not_the_name():
    """S4.4 item 3: the Isaac setter swallows exceptions, so the value must be read back.

    ``RenderCfg.antialiasing_mode`` takes the NAME but ``/rtx/post/aa/op`` stores an integer, so
    a readback that compared against ``"FXAA"`` could never match and would report a failure on
    a correctly configured renderer, or be quietly dropped.
    """
    expected = ec.expected_carb_settings()
    assert expected[ec.ANTIALIASING_SETTING] == ec.ANTIALIASING_CODES["FXAA"] == 2
    assert expected[ec.GPU_MEMORY_BUDGET_SETTING] == 1024
    assert expected["/rtx/shadows/enabled"] is True


def test_carb_readback_uses_only_settings_that_exist_in_isaac_sim_5_1():
    """The key SPEC v2 names for the texture budget does not exist on this build.

    Probed on Isaac Sim 5.1, ``/rtx-transient/resourcemanager/maxTextureStreamingBudgetMB``
    reads ``None``, and ``RenderCfg.carb_settings`` raises rather than ignoring an unknown key
    (``simulation_context.py:795``), so passing it makes the environment unconstructible.
    """
    expected = ec.expected_carb_settings()
    assert "maxTextureStreamingBudgetMB" not in " ".join(expected)
    assert ec.GPU_MEMORY_BUDGET_SETTING.endswith("/UJITSO/GPUMemoryBudgetMB")


def test_train_py_does_not_add_a_rendering_flag_of_its_own():
    """Critic item G: --rendering_mode on the CLI outranks RenderCfg and would break ablations.

    ``AppLauncher.add_app_launcher_args`` registers the flag unconditionally, so the assertion
    that can be made is that ``train.py`` never adds it itself and never sets it.
    """
    source = (_REPO_ROOT / "scripts" / "train.py").read_text(encoding="utf-8")
    added = [line for line in source.splitlines() if "add_argument(" in line and "rendering_mode" in line]
    assert added == []
    assert "AppLauncher.add_app_launcher_args(parser)" in source


def test_train_py_refuses_an_explicit_rendering_override():
    """Since the flag cannot be removed, using it has to be a hard error, not a silent override."""
    import argparse
    import importlib.util

    path = _REPO_ROOT / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("_train_for_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError:
        pytest.skip("scripts/train.py needs Isaac Lab, which this interpreter does not have")

    module.reject_rendering_override(argparse.Namespace(rendering_mode=None))
    with pytest.raises(SystemExit, match="outranks RenderCfg"):
        module.reject_rendering_override(argparse.Namespace(rendering_mode="quality"))


# =============================================================================================
# City and obstacles
# =============================================================================================


def test_layout_selection_is_deterministic():
    """Critic item D: random_choice=True leaves ~1.2 of 64 layouts with zero envs at N=256."""
    assert ec.CitySettings().random_choice is False
    with pytest.raises(ValueError, match="random_choice"):
        ec.CitySettings(random_choice=True)


def test_layouts_divide_evenly_across_the_default_env_count():
    """64 layouts over 256 envs is exactly 4 envs each under index % len."""
    city = ec.CitySettings()
    assert ec.LaneFollowSettings().num_envs % city.num_variants == 0


def test_placement_box_fits_inside_the_env_spacing():
    """A 3.6 m half-extent inside 8.0 m spacing leaves 0.8 m of gap between cities."""
    city = ec.CitySettings()
    assert 2.0 * city.placement_half_extent <= city.env_spacing
    with pytest.raises(ValueError, match="placement box"):
        ec.CitySettings(placement_half_extent=4.5)


def test_camera_far_clip_cannot_see_the_neighbouring_city():
    """6 m far clip against 8 m spacing: a neighbour is never in range."""
    _near, far = DUCKIEBOT.camera_clipping_range_m
    assert far < ec.CitySettings().env_spacing


def test_obstacle_settings_validate_their_ranges():
    with pytest.raises(ValueError, match="density"):
        ec.ObstacleSettings(density=1.5)
    with pytest.raises(ValueError, match="margin_m"):
        ec.ObstacleSettings(margin_m=-0.01)


def test_obstacles_are_off_by_default():
    """Task-curriculum stages 0 and 1 train without obstacles, and pay none of their cost."""
    assert ec.LaneFollowSettings().obstacles.enabled is False


# =============================================================================================
# The aggregate
# =============================================================================================


def test_curriculum_scalars_are_validated():
    with pytest.raises(ValueError, match="dr_alpha_vis"):
        ec.LaneFollowSettings(dr_alpha_vis=1.5)


def test_summary_is_json_serialisable_and_complete():
    """The summary is what lands in the checkpoint and in the TensorBoard run metadata."""
    import json

    summary = ec.LaneFollowSettings().summary()
    json.dumps(summary)
    for key in ("control_hz", "max_episode_length", "rgb_shape", "num_rerenders_on_reset", "seed"):
        assert key in summary
    assert summary["control_hz"] == pytest.approx(15.0)
    assert summary["max_episode_length"] == 450


def test_missing_city_assets_produce_an_actionable_error():
    """The failure has to name the directories tried and the command that fixes it."""
    city = ec.CitySettings(root="does/not/exist", num_variants=4)
    with pytest.raises(FileNotFoundError) as excinfo:
        ec.resolve_city_assets(city)
    message = str(excinfo.value)
    assert "does/not/exist" in message
    assert "scripts/build_city.py" in message


def test_isaac_free_import_surface():
    """Importing the env package must not drag in Isaac, or CI cannot run any of it.

    Checked in a fresh subprocess rather than against ``sys.modules``: another test in the same
    session may legitimately have imported Isaac Lab, and an order-dependent assertion here
    would pass or fail depending on the collection order rather than on the import graph.
    """
    import subprocess

    script = (
        "import sys\n"
        "import duckiebot_rl.envs as envs\n"
        "import duckiebot_rl.envs.env_cfg\n"
        "import duckiebot_rl.envs.rewards\n"
        "import duckiebot_rl.envs.terminations\n"
        "import duckiebot_rl.envs.camera_math\n"
        "import duckiebot_rl.envs.action_path\n"
        "import duckiebot_rl.envs.obstacles\n"
        "import duckiebot_rl.camera_math\n"
        "leaked = sorted(m for m in sys.modules if m.split('.')[0] in "
        "('isaaclab', 'isaacsim', 'omni', 'carb', 'pxr'))\n"
        "assert not leaked, leaked\n"
        "assert envs.LaneFollowSettings().num_envs == 256\n"
        "print('clean')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "clean" in result.stdout


# =============================================================================================
# Regression guard on an [assets]-owned setting that only the [env] can observe
# =============================================================================================


def test_wheel_joints_use_a_force_drive():
    """The wheel drive must be a FORCE drive, or the robot cannot move at all.

    The imported USD carries ``drive:angular:physics:type = "acceleration"`` on both wheel
    joints, because that is what the URDF importer writes. Isaac Lab overwrites stiffness,
    damping, effort limit, velocity limit and armature from the actuator config at every
    articulation init, but never the drive TYPE. Under an acceleration drive PhysX scales the
    gains by the joint's effective inertia, so the S2 damping of 0.05 N.m.s/rad against an
    18.87 rad/s target yields ``2.25e-4 x 0.94 = 2.1e-4`` N.m rather than the intended 0.15 N.m
    - 48x below the 0.010 N.m joint friction. The wheels then never break static friction, and
    ``data.applied_torque`` still reports the effort limit because that field is Isaac Lab's own
    analytic estimate rather than a PhysX readback, so nothing looks wrong anywhere.

    The setting lives in the ``[assets]``-owned spawn config, but only the environment can
    observe the consequence, which is why the guard lives here.
    """
    from duckiebot_rl.assets.robot_cfg import spawn_property_spec

    assert spawn_property_spec()["joint_drive_props"]["drive_type"] == "force"
