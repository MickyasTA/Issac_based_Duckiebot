"""The live viewer's command line, its backend plumbing and its frame-shape guard.

Everything here runs on numpy alone. Isaac Sim is never imported: the environment factory is
replaced with a recorder, and ``duckiebot_rl.envs.viz_env`` is only inspected, never called into
the parts that need Kit.
"""

from __future__ import annotations

import argparse
import dataclasses
import types

import numpy as np
import pytest
import torch

from duckiebot_rl.envs import viz_env
from duckiebot_rl.viz import backends
from scripts import live_view


class _Recorder:
    """Stands in for the Isaac environment factory and remembers how it was called."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> str:
        self.calls.append(dict(kwargs))
        return "isaac-env"


def _parse(*argv: str) -> argparse.Namespace:
    return live_view.build_parser().parse_args(list(argv))


# --------------------------------------------------------------------------- the flag


def test_robot_mesh_defaults_to_db21j():
    assert _parse().robot_mesh == "db21j"
    assert live_view.DEFAULT_ROBOT_MESH == "db21j"


@pytest.mark.parametrize("choice", ["db21j", "db17", "primitive"])
def test_robot_mesh_accepts_every_choice(choice):
    assert _parse("--robot-mesh", choice).robot_mesh == choice


def test_robot_mesh_rejects_an_unknown_name():
    with pytest.raises(SystemExit):
        _parse("--robot-mesh", "db99")


def test_db21j_alias_sets_the_same_dest():
    assert _parse("--db21j").robot_mesh == "db21j"
    # the alias is a store_const on the same dest, so a later --robot-mesh still wins
    assert _parse("--db21j", "--robot-mesh", "db17").robot_mesh == "db17"
    assert _parse("--robot-mesh", "db17", "--db21j").robot_mesh == "db21j"


def test_db21j_alias_does_not_clobber_the_default():
    # a store_const alias whose default leaked would leave robot_mesh None when unused
    assert _parse("--map", "city_7").robot_mesh == "db21j"


def test_backend_choices_match_the_environment_module():
    # backends.py spells the vocabulary out so the viewer imports without duckiebot_rl.envs;
    # this is the test that keeps the two copies honest.
    assert backends.ROBOT_MESH_NAMES == viz_env.ROBOT_MESH_CHOICES
    assert backends.DEFAULT_ROBOT_MESH == viz_env.DEFAULT_ROBOT_MESH


# ------------------------------------------------------------------- the plumbing


def test_backend_kwargs_carries_the_mesh_and_the_rest_of_the_command_line():
    kwargs = live_view.backend_kwargs(_parse("--robot-mesh", "db17", "--map", "city_9", "--seed", "3"))
    assert kwargs["robot_mesh"] == "db17"
    assert kwargs["map_name"] == "city_9"
    assert kwargs["seed"] == 3
    assert kwargs["backend"] == "mujoco"


def test_make_backend_forwards_robot_mesh_to_the_isaac_factory(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(backends, "_import_isaac_env_factory", lambda: recorder)

    args = _parse("--backend", "isaac", "--allow-isaac-vram", "--robot-mesh", "db17")
    assert backends.make_backend(**live_view.backend_kwargs(args)) == "isaac-env"

    (call,) = recorder.calls
    assert call["robot_mesh"] == "db17"
    assert call["num_envs"] == 1
    assert call["render"] is True


def test_make_backend_defaults_the_isaac_mesh_to_db21j(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(backends, "_import_isaac_env_factory", lambda: recorder)

    backends.make_backend(**live_view.backend_kwargs(_parse("--backend", "isaac", "--allow-isaac-vram")))
    assert recorder.calls[0]["robot_mesh"] == "db21j"


def test_make_backend_does_not_hand_robot_mesh_to_mujoco(monkeypatch):
    seen: dict[str, object] = {}

    def fake_mujoco(**kwargs: object) -> str:
        seen.update(kwargs)
        return "mujoco-env"

    monkeypatch.setattr(backends, "MujocoBackend", fake_mujoco)
    assert backends.make_backend(**live_view.backend_kwargs(_parse("--robot-mesh", "db17"))) == "mujoco-env"
    assert "robot_mesh" not in seen


def test_make_viz_env_takes_robot_mesh_and_checks_it_before_kit():
    import inspect

    signature = inspect.signature(viz_env.make_viz_env)
    assert signature.parameters["robot_mesh"].default == "db21j"
    # the validation has to happen before _require_kit(), or a typo costs a minute of Kit boot
    with pytest.raises(ValueError, match="unknown robot_mesh"):
        viz_env.make_viz_env(robot_mesh="db99")


def test_resolve_robot_mesh_passes_primitive_through_untouched():
    assert viz_env.resolve_robot_mesh("primitive") == "primitive"
    assert viz_env.resolve_robot_mesh(" PRIMITIVE ") == "primitive"


def test_resolve_robot_mesh_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="db99"):
        viz_env.resolve_robot_mesh("db99")


def test_resolve_robot_mesh_falls_back_when_the_gltf_is_missing(monkeypatch, capsys):
    # db21j has two sources now, the Duckiematrix OBJ ahead of the DB18-era glTF; both have to
    # be absent before the fallback is reached. tests/unit/test_robot_mesh.py owns the ordering.
    monkeypatch.setattr(viz_env, "_find_db21j_obj", lambda: None)
    monkeypatch.setattr(viz_env, "_find_db21j_gltf", lambda: None)
    monkeypatch.setattr(viz_env, "_find_visual_mesh_dir", lambda: "somewhere/meshes")
    assert viz_env.resolve_robot_mesh("db21j") == "db17"
    printed = capsys.readouterr().out
    assert "fetch_visual_mesh.py" in printed
    assert "db17" in printed


def test_resolve_robot_mesh_falls_back_to_primitive_when_nothing_is_on_disk(monkeypatch, capsys):
    monkeypatch.setattr(viz_env, "_find_db21j_obj", lambda: None)
    monkeypatch.setattr(viz_env, "_find_db21j_gltf", lambda: None)
    monkeypatch.setattr(viz_env, "_find_visual_mesh_dir", lambda: None)
    assert viz_env.resolve_robot_mesh("db21j") == "primitive"
    assert "primitive" in capsys.readouterr().out


def test_attach_real_visuals_does_nothing_for_primitive(monkeypatch):
    # if it reached the Kit imports this test would not be running outside Isaac at all
    def explode() -> object:
        raise AssertionError("the primitive path must not look for meshes")

    monkeypatch.setattr(viz_env, "_find_db21j_obj", explode)
    monkeypatch.setattr(viz_env, "_find_db21j_gltf", explode)
    monkeypatch.setattr(viz_env, "_find_visual_mesh_dir", explode)
    assert viz_env._attach_real_visuals(object(), robot_mesh="primitive") is None


def test_attach_real_visuals_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="unknown robot_mesh"):
        viz_env._attach_real_visuals(object(), robot_mesh="db99")


# ------------------------------------------------------- the mixed-shape recording guard


def _frame(height: int, width: int, value: int) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


def test_harmonize_frames_leaves_a_uniform_episode_alone():
    frames = [_frame(360, 640, i) for i in range(5)]
    out = live_view.harmonize_frames(frames)
    assert [f.shape for f in out] == [(360, 640, 3)] * 5
    assert np.array_equal(out[2], frames[2])


def test_harmonize_frames_handles_an_empty_episode():
    assert live_view.harmonize_frames([]) == []


def test_harmonize_frames_drops_the_onboard_warmup_prefix():
    # the reported bug: 128x192 onboard frames until the 360x640 chase camera comes up
    frames = [_frame(128, 192, 1), _frame(128, 192, 2)] + [_frame(360, 640, i) for i in range(20)]
    out = live_view.harmonize_frames(frames)
    assert len(out) == 20
    assert {f.shape for f in out} == {(360, 640, 3)}


def test_harmonize_frames_resizes_when_the_prefix_is_too_long_to_drop():
    frames = [_frame(128, 192, 1)] * 8 + [_frame(360, 640, 2)] * 10
    out = live_view.harmonize_frames(frames)
    assert len(out) == 18
    assert {f.shape for f in out} == {(360, 640, 3)}
    assert out[0].dtype == np.uint8
    assert int(out[0][0, 0, 0]) == 1


def test_harmonize_frames_resizes_a_mismatch_that_is_not_a_prefix():
    frames = [_frame(360, 640, 0), _frame(128, 192, 1), _frame(360, 640, 2)]
    out = live_view.harmonize_frames(frames)
    assert [f.shape for f in out] == [(360, 640, 3)] * 3


def test_harmonize_frames_output_satisfies_the_encoder_guard():
    from duckiebot_rl.viz.render import _as_frame_list

    frames = [_frame(128, 192, 1), _frame(360, 640, 2), _frame(360, 640, 3), _frame(360, 640, 4)]
    with pytest.raises(ValueError, match="share one shape"):
        _as_frame_list(frames)
    assert len(_as_frame_list(live_view.harmonize_frames(frames))) == 3


def test_harmonize_frames_drops_frames_it_cannot_resample():
    frames = [_frame(360, 640, 0), np.zeros((128, 192, 4), dtype=np.uint8), _frame(360, 640, 2)]
    out = live_view.harmonize_frames(frames)
    assert [f.shape for f in out] == [(360, 640, 3)] * 2


def test_resize_nearest_upscales_and_downscales_without_touching_the_dtype():
    source = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    up = live_view._resize_nearest(source, 4, 6)
    assert up.shape == (4, 6, 3)
    assert up.dtype == np.uint8
    assert np.array_equal(up[0, 0], source[0, 0])
    down = live_view._resize_nearest(source, 1, 1)
    assert down.shape == (1, 1, 3)


# ============================================================ the parallel grid view
#
# Everything below runs on numpy and CPU torch tensors. The "Isaac environment" is a handful of
# dataclasses: what is under test is the plumbing, the refusals and the loop, none of which need
# a simulator to be wrong.


class _FakeVecEnv:
    """The vectorized environment the grid drives: batched tensors in, batched tensors out."""

    def __init__(self, num_envs: int = 4, vec_dim: int = 3) -> None:
        self.num_envs = num_envs
        self.vec_dim = vec_dim
        self.device = "cpu"
        self.actions: list[torch.Tensor] = []
        self.reset_seeds: list[object] = []
        self.closed = False
        self.step_count = 0

    def _obs(self) -> dict[str, dict[str, torch.Tensor]]:
        value = float(self.step_count)
        return {"policy": {"vec": torch.full((self.num_envs, self.vec_dim), value)}}

    def reset(self, seed: object = None) -> tuple[dict[str, object], dict[str, object]]:
        self.reset_seeds.append(seed)
        return self._obs(), {}

    def step(self, action: torch.Tensor) -> tuple[object, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        self.actions.append(action.clone())
        self.step_count += 1
        terminated = torch.zeros(self.num_envs, dtype=torch.bool)
        terminated[0] = self.step_count % 3 == 0
        return (
            self._obs(),
            torch.full((self.num_envs,), 0.25),
            terminated,
            torch.zeros(self.num_envs, dtype=torch.bool),
            {"log": {}},
        )

    def close(self) -> None:
        self.closed = True


class _FakeAdapter:
    """What the environment factory returns: the viewer adapter, with its vectorized env inside."""

    def __init__(self, num_envs: int = 4) -> None:
        self.env = _FakeVecEnv(num_envs=num_envs)
        self.control_dt = 0.05
        self.closed = False

    def close(self) -> None:
        self.closed = True
        self.env.close()


class _GridFactory:
    """An environment factory that hands back a fake adapter and remembers its keywords."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.adapter: _FakeAdapter | None = None

    def __call__(self, **kwargs: object) -> _FakeAdapter:
        self.calls.append(dict(kwargs))
        self.adapter = _FakeAdapter(num_envs=int(kwargs.get("num_envs", 1)))
        return self.adapter


class _FakeHost:
    """Stands in for PolicyHost: batched actions, a reload counter, no torch checkpoint."""

    def __init__(self, act_dim: int = 2) -> None:
        self.act_dim = act_dim
        self.loaded = True
        self.reload_count = 0
        self.state = types.SimpleNamespace(iteration=11)
        self.batch_sizes: list[int] = []

    def act_batch(self, obs: dict[str, object]) -> np.ndarray:
        rows = int(np.asarray(obs["vec"]).shape[0])
        self.batch_sizes.append(rows)
        return np.full((rows, self.act_dim), 0.5, dtype=np.float32)

    def act(self, obs: dict[str, object], stochastic: bool = False) -> np.ndarray:
        return np.zeros(self.act_dim, dtype=np.float32)

    def load_from_info(self, info: object) -> object:
        self.reload_count += 1
        self.state = types.SimpleNamespace(iteration=getattr(info, "iteration", -1))
        return types.SimpleNamespace(describe=lambda: "fake host state")


class _FakeInfo:
    """A CheckpointInfo stand-in: only what apply_reload actually reads."""

    iteration = 42

    def describe(self) -> str:
        return "fake checkpoint iteration=42"


class _FakeWatcher:
    """A CheckpointWatcher stand-in that publishes one new checkpoint, then nothing."""

    def __init__(self, publish: int = 0) -> None:
        self.publish = publish
        self.polls = 0

    def poll(self) -> object:
        self.polls += 1
        return _FakeInfo() if self.polls == self.publish else None


@dataclasses.dataclass
class _FakeCity:
    """A CitySettings stand-in, so dataclasses.replace does what it does on the real one."""

    root: str | None = "build/city"
    num_variants: int = 1
    variant_names: tuple[str, ...] | None = ("city_007",)


# ------------------------------------------------------------- the cross-module contract


def _stub_viz_env_construction(monkeypatch) -> dict[str, object]:
    """Let ``make_viz_env`` run to the point of building its settings, with no Isaac anywhere.

    ``duckiebot_rl.envs.lane_follow_env`` imports Isaac Lab at module scope and raises without
    Kit, so the import itself is satisfied with a stand-in module rather than skipped.

    Args:
        monkeypatch: The pytest fixture.

    Returns:
        A dict that receives the ``settings`` and ``overrides`` the config builder was called with.
    """
    import sys

    from duckiebot_rl.envs import env_cfg

    seen: dict[str, object] = {}

    def fake_cfg(settings: object, **overrides: object) -> str:
        seen["settings"] = settings
        seen["overrides"] = dict(overrides)
        return "cfg"

    def fake_env(cfg: object, render_mode: object = None) -> object:
        return types.SimpleNamespace(cfg=cfg, render_mode=render_mode)

    stand_in = types.ModuleType("duckiebot_rl.envs.lane_follow_env")
    stand_in.DuckiebotLaneFollowEnv = fake_env
    monkeypatch.setitem(sys.modules, "duckiebot_rl.envs.lane_follow_env", stand_in)
    monkeypatch.setattr(env_cfg, "lane_follow_env_cfg", fake_cfg)
    monkeypatch.setattr(viz_env, "_require_kit", lambda: None)
    monkeypatch.setattr(viz_env, "resolve_city_selection", lambda name: _FakeCity())
    monkeypatch.setattr(viz_env, "_attach_real_visuals", lambda env, robot_mesh="db21j": None)
    monkeypatch.setattr(viz_env, "IsaacVizEnv", lambda env: types.SimpleNamespace(env=env))
    return seen


def test_make_viz_env_passes_num_envs_through_for_the_grid(monkeypatch):
    """The grid's whole premise: --num-envs N must reach the settings unclamped.

    This is the one contract the parallel view cannot check at runtime. Clamping N back to 1 does
    not raise and does not log: the Kit viewport simply shows a single city where the user asked
    for 64, and the status line still reports healthy steps/s. So it is pinned here.
    """
    seen = _stub_viz_env_construction(monkeypatch)
    viz_env.make_viz_env(map="city_007", num_envs=64)
    assert seen["settings"].num_envs == 64, "make_viz_env clamped the grid back to one environment"


def test_make_viz_env_still_defaults_to_a_single_environment(monkeypatch):
    seen = _stub_viz_env_construction(monkeypatch)
    viz_env.make_viz_env(map="city_007")
    assert seen["settings"].num_envs == 1


def test_make_viz_env_unstaggers_a_single_environment(monkeypatch):
    """A one-robot viewer episode must run its full horizon, not a staggered fraction of it.

    ``stagger_initial_episode_length`` randomises ``episode_length_buf`` at the very first reset
    so 64 training environments do not all reset on the same step. Inherited by the viewer it
    randomises where episode 1 truncates instead (observed: a requested 15 s showcase episode
    ended after 5 control steps). The grid keeps the stagger for the same reason training does.
    """
    seen = _stub_viz_env_construction(monkeypatch)
    viz_env.make_viz_env(map="city_007", episode_length_s=15.0)
    assert seen["settings"].rates.stagger_initial_episode_length is False
    assert seen["settings"].rates.episode_length_s == 15.0


def test_make_viz_env_keeps_the_stagger_for_the_grid(monkeypatch):
    seen = _stub_viz_env_construction(monkeypatch)
    viz_env.make_viz_env(map="city_007", num_envs=64)
    assert seen["settings"].rates.stagger_initial_episode_length is True


def test_make_viz_env_forwards_a_widened_city_selection(monkeypatch):
    """The layout list the grid widens has to survive as a LaneFollowSettings override.

    ``_parallel_factory_kwargs`` hands ``city=`` to the factory, which forwards it through
    ``**overrides``; if that stopped landing on the settings, 64 environments would share the one
    layout ``--map`` named.
    """
    seen = _stub_viz_env_construction(monkeypatch)
    widened = dataclasses.replace(_FakeCity(), num_variants=64, variant_names=None)
    viz_env.make_viz_env(map="city_007", num_envs=64, city=widened)
    assert seen["overrides"]["city"] is widened


# ------------------------------------------------------------------------------ the flag


def test_num_envs_defaults_to_one():
    assert _parse().num_envs == 1


def test_num_envs_parses():
    assert _parse("--num-envs", "64").num_envs == 64


def test_backend_kwargs_carries_num_envs():
    assert live_view.backend_kwargs(_parse("--num-envs", "64"))["num_envs"] == 64
    assert live_view.backend_kwargs(_parse())["num_envs"] == 1


# ------------------------------------------------------------------------ the guardrails


def test_parallel_mode_plan_says_nothing_for_a_single_environment():
    assert live_view.parallel_mode_plan(_parse()) == (None, [])
    assert live_view.parallel_mode_plan(_parse("--num-envs", "1", "--record")) == (None, [])


def test_parallel_mode_plan_refuses_a_mujoco_grid():
    refusal, notes = live_view.parallel_mode_plan(_parse("--num-envs", "8"))
    assert refusal is not None
    assert "--backend isaac" in refusal
    assert "8" in refusal
    assert notes == []


def test_parallel_mode_plan_refuses_a_count_below_one():
    refusal, _notes = live_view.parallel_mode_plan(_parse("--num-envs", "0"))
    assert refusal is not None and "pass 1 or more" in refusal


def test_parallel_mode_plan_names_what_it_switches_off():
    _refusal, notes = live_view.parallel_mode_plan(
        _parse("--backend", "isaac", "--allow-isaac-vram", "--window", "--num-envs", "64")
    )
    joined = " ".join(notes)
    assert "chase camera" in joined
    assert "live_frame" in joined
    assert "--episodes and --loop do not apply" in joined
    assert "--record" not in joined


def test_parallel_mode_plan_refuses_to_record_a_grid():
    _refusal, notes = live_view.parallel_mode_plan(
        _parse("--backend", "isaac", "--allow-isaac-vram", "--window", "--num-envs", "64", "--record")
    )
    record_note = [note for note in notes if "--record" in note]
    assert len(record_note) == 1
    assert "IGNORED" in record_note[0]
    assert "--num-envs 1" in record_note[0]


def test_parallel_mode_plan_warns_when_nothing_would_be_on_screen():
    _refusal, notes = live_view.parallel_mode_plan(
        _parse("--backend", "isaac", "--allow-isaac-vram", "--num-envs", "64")
    )
    assert any("add --window" in note for note in notes)


def test_parallel_mode_plan_points_at_the_gpu_when_the_policy_is_on_the_cpu():
    _refusal, notes = live_view.parallel_mode_plan(
        _parse("--backend", "isaac", "--allow-isaac-vram", "--window", "--num-envs", "64")
    )
    assert any("--device cuda:0" in note for note in notes)

    _refusal, notes = live_view.parallel_mode_plan(
        _parse(
            "--backend",
            "isaac",
            "--allow-isaac-vram",
            "--window",
            "--num-envs",
            "64",
            "--device",
            "cuda:0",
        )
    )
    assert not any("--device" in note for note in notes)


def test_main_refuses_a_mujoco_grid_before_touching_a_run_directory(capsys):
    # the refusal has to land before the run directory is resolved and the checkpoint waited for
    assert live_view.main(["--num-envs", "8", "--runs-root", "definitely/not/here"]) == 5
    assert "--backend isaac" in capsys.readouterr().err


# --------------------------------------------------------------------- the backend plumbing


def test_make_backend_builds_the_grid_and_wraps_the_adapter(monkeypatch):
    factory = _GridFactory()
    monkeypatch.setattr(backends, "_import_isaac_env_factory", lambda: factory)
    monkeypatch.setattr(backends, "_parallel_factory_kwargs", lambda *a, **k: {})

    args = _parse("--backend", "isaac", "--allow-isaac-vram", "--num-envs", "64")
    backend = backends.make_backend(**live_view.backend_kwargs(args))

    assert isinstance(backend, backends.ParallelIsaacBackend)
    assert backend.num_envs == 64
    assert factory.calls[0]["num_envs"] == 64
    assert factory.calls[0]["render"] is True


def test_make_backend_leaves_a_single_environment_adapter_unwrapped(monkeypatch):
    factory = _GridFactory()
    monkeypatch.setattr(backends, "_import_isaac_env_factory", lambda: factory)

    backend = backends.make_backend(
        **live_view.backend_kwargs(_parse("--backend", "isaac", "--allow-isaac-vram"))
    )
    assert not isinstance(backend, backends.ParallelIsaacBackend)
    assert factory.calls[0]["num_envs"] == 1
    assert "city" not in factory.calls[0] and "allow_multi" not in factory.calls[0]


def test_make_backend_refuses_a_mujoco_grid():
    with pytest.raises(ValueError, match="Isaac-only"):
        backends.make_backend(backend="mujoco", num_envs=16)


def test_parallel_factory_kwargs_defers_to_a_factory_that_advertises_allow_multi():
    def factory(
        map: str = "loop_small", num_envs: int = 1, allow_multi: bool = False, **kwargs: object
    ) -> None:
        return None

    assert backends._parallel_factory_kwargs(factory, "city_7", 64) == {"allow_multi": True}


def test_parallel_factory_kwargs_widens_the_layout_list_itself(monkeypatch):
    # without allow_multi the grid would get 64 copies of the one layout --map named, because
    # MultiUsdFileCfg assigns assets by index % len over a one-entry list
    monkeypatch.setattr(viz_env, "resolve_city_selection", lambda name: _FakeCity())

    def factory(**kwargs: object) -> None:
        return None

    extra = backends._parallel_factory_kwargs(factory, "city_007", 64)
    assert extra["city"].num_variants == 64
    assert extra["city"].variant_names is None
    assert extra["city"].root == "build/city", "the build root --map picked has to survive"


def test_parallel_factory_kwargs_gives_up_quietly_when_the_layouts_cannot_be_resolved(monkeypatch, capsys):
    def explode(name: str) -> None:
        raise FileNotFoundError("no stages built")

    monkeypatch.setattr(viz_env, "resolve_city_selection", explode)
    assert backends._parallel_factory_kwargs(lambda **kwargs: None, "city_007", 64) == {}
    assert "keeping the factory's own layout selection" in capsys.readouterr().out


# ------------------------------------------------------------------- the parallel backend


def test_parallel_backend_needs_the_vectorized_environment():
    with pytest.raises(RuntimeError, match="'env' attribute"):
        backends.ParallelIsaacBackend(object(), num_envs=8)


def test_parallel_backend_resets_once_and_keeps_the_batch_axis():
    adapter = _FakeAdapter(num_envs=6)
    backend = backends.ParallelIsaacBackend(adapter, num_envs=6)

    obs = backend.reset(seed=3)
    assert adapter.env.reset_seeds == [3]
    assert obs["vec"].shape == (6, adapter.env.vec_dim), "the policy half, batch axis intact"
    assert backend.control_dt == 0.05


def test_parallel_backend_steps_every_environment_with_one_batched_action():
    adapter = _FakeAdapter(num_envs=6)
    backend = backends.ParallelIsaacBackend(adapter, num_envs=6)
    backend.reset()

    actions = np.tile(np.array([0.2, -0.4], dtype=np.float32), (6, 1))
    obs, reward, terminated, truncated, info = backend.step(actions)

    assert len(adapter.env.actions) == 1, "one env.step per control step, not one per environment"
    assert tuple(adapter.env.actions[0].shape) == (6, 2)
    assert obs["vec"].shape == (6, adapter.env.vec_dim)
    assert tuple(reward.shape) == (6,)
    assert tuple(terminated.shape) == (6,) and tuple(truncated.shape) == (6,)
    assert isinstance(info, dict)


def test_parallel_backend_clips_actions_into_the_action_box():
    adapter = _FakeAdapter(num_envs=4)
    backend = backends.ParallelIsaacBackend(adapter, num_envs=4)
    backend.step(np.full((4, 2), 7.5, dtype=np.float32))
    backend.step(torch.full((4, 2), -7.5))

    assert float(adapter.env.actions[0].max()) == 1.0
    assert float(adapter.env.actions[1].min()) == -1.0


def test_parallel_backend_has_no_frame_to_render_and_says_so():
    backend = backends.ParallelIsaacBackend(_FakeAdapter(), num_envs=4)
    assert backend.render_frame() is None


def test_parallel_backend_close_goes_through_the_adapter():
    adapter = _FakeAdapter()
    backends.ParallelIsaacBackend(adapter, num_envs=4).close()
    assert adapter.closed and adapter.env.closed


# ------------------------------------------------------------------------- the parallel loop


def _grid(num_envs: int = 4) -> backends.ParallelIsaacBackend:
    return backends.ParallelIsaacBackend(_FakeAdapter(num_envs=num_envs), num_envs=num_envs)


def test_run_parallel_steps_every_environment_until_the_limit():
    backend, host = _grid(4), _FakeHost()
    summary = live_view.run_parallel(
        backend=backend,
        host=host,
        watcher=_FakeWatcher(),
        seed=0,
        poll_seconds=1e9,
        status_seconds=1e9,
        max_steps=10,
    )

    assert summary["steps"] == 10
    assert summary["env_steps"] == 40
    assert summary["reason"] == "step limit"
    assert host.batch_sizes == [4] * 10, "one batched forward pass per control step"
    assert len(backend.adapter.env.reset_seeds) == 1, "the grid resets exactly once"
    assert summary["resets"] == 3, "the environments reset themselves as they finish"
    assert summary["mean_reward"] == pytest.approx(0.25)


def test_run_parallel_stops_when_the_window_closes():
    backend, host = _grid(4), _FakeHost()
    calls = {"n": 0}

    def is_running() -> bool:
        calls["n"] += 1
        return calls["n"] < 3

    summary = live_view.run_parallel(
        backend=backend,
        host=host,
        watcher=_FakeWatcher(),
        seed=0,
        poll_seconds=1e9,
        status_seconds=1e9,
        is_running=is_running,
        max_steps=1000,
    )
    assert summary["reason"] == "window closed"
    assert summary["steps"] == 3


def test_run_parallel_hot_reloads_without_restarting_anything(capsys):
    backend, host = _grid(4), _FakeHost()
    summary = live_view.run_parallel(
        backend=backend,
        host=host,
        watcher=_FakeWatcher(publish=2),
        seed=0,
        poll_seconds=0.0,
        status_seconds=1e9,
        max_steps=5,
    )

    assert summary["reloads"] == 1
    assert host.reload_count == 1
    assert host.state.iteration == 42, "the newest weights are what the grid keeps driving with"
    assert summary["steps"] == 5, "the loop never restarted"
    assert len(backend.adapter.env.reset_seeds) == 1
    assert "RELOAD" in capsys.readouterr().out


def test_run_parallel_prints_a_status_line(capsys):
    live_view.run_parallel(
        backend=_grid(8),
        host=_FakeHost(),
        watcher=_FakeWatcher(),
        seed=0,
        poll_seconds=1e9,
        status_seconds=0.0,
        max_steps=2,
    )
    printed = capsys.readouterr().out
    assert "steps/s" in printed
    assert "env-steps/s" in printed
    assert "mean_reward" in printed
    assert "reloads=" in printed


def test_first_env_obs_slices_environment_zero():
    obs = {"vec": torch.arange(12.0).reshape(4, 3), "rgb": np.zeros((4, 2, 2, 3), dtype=np.uint8)}
    single = live_view._first_env_obs(obs)
    assert single is not None
    assert tuple(single["vec"].shape) == (3,)
    assert single["rgb"].shape == (2, 2, 3)


def test_first_env_obs_gives_up_rather_than_raising():
    assert live_view._first_env_obs({"vec": 3.0}) is None


def test_host_sum_reduces_tensors_and_arrays_and_survives_nonsense():
    assert live_view._host_sum(torch.full((4,), 0.5)) == pytest.approx(2.0)
    assert live_view._host_sum(np.ones(3, dtype=np.float32)) == pytest.approx(3.0)
    assert live_view._host_sum(torch.tensor([True, False, True])) == pytest.approx(2.0)
    assert live_view._host_sum(object()) == 0.0


def test_batch_sum_agrees_with_host_sum_without_leaving_the_device():
    """The step path accumulates with _batch_sum and only the status line pays for a sync.

    The reduction has to stay a torch tensor: converting here instead would reintroduce the
    three per-step GPU stalls the parallel backend hands back raw tensors to avoid.
    """
    for value in (torch.full((4,), 0.5), np.ones(3, dtype=np.float32), torch.tensor([True, False, True])):
        assert live_view._host_sum(live_view._batch_sum(value)) == pytest.approx(live_view._host_sum(value))
    assert isinstance(live_view._batch_sum(torch.ones(4)), torch.Tensor), "a sync was forced"
    assert live_view._batch_sum(2.5) == pytest.approx(2.5)
    assert live_view._batch_sum(object()) == 0
