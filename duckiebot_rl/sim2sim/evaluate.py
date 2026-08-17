"""The sim-to-sim evaluation harness (SPEC v2 S8.4).

Runs a trained policy for N episodes per condition per seed and reports the S8.4 metrics with
confidence intervals. Two things the critic asked for explicitly are built in rather than bolted on:

**A wall-clock budget.** The v1 matrix was 8 conditions x 500 episodes x 5 seeds x 45 s = 20,000
episodes, uncosted; at MuJoCo's single-process speed that alone is most of a day. Every run here
carries a :class:`Budget`: the expected wall clock is computed *before* the run from a measured
per-episode cost, printed, and compared against the S8.4 table afterwards. A run that overshoots its
budget by more than 50% is flagged in the report rather than quietly accepted.

**Process-level parallelism.** MuJoCo stepping and offscreen rendering are CPU bound and release
nothing useful to threads, so episodes are distributed over processes. Each worker builds its own
environment once and then runs many episodes, because scene construction (and texture generation)
dominates a single short episode. The default worker count leaves one core for the OS. Workers are
spawned, which is the only mode Windows has, so everything crossing the boundary is picklable: a
worker receives a plain :class:`WorkerSpec` and loads the policy itself.

Metrics, per S8.4: lane RMS and max ``|d|``, survival time, fractional laps, success rate, a
failure-reason histogram, collisions per minute, the four AI-DO-style metrics, and the corner-cut
statistics from S5.4. Reported per condition as median with interquartile range across episodes, and
mean plus standard deviation across seeds, with a bootstrap confidence interval on the median.

The headline is the C1-versus-C5 pair. It is a robustness probe across two independent physics
engines and two independent renderers. It is not a real-world predictor and the report says so.
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import statistics
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from ._resolve import SharedModuleUnavailable, environment_report
from .env import EpisodeMetrics, MjDuckiebotEnv, MjEnvCfg

__all__ = [
    "CONDITIONS",
    "Budget",
    "Condition",
    "EvaluationReport",
    "PolicySpec",
    "WorkerSpec",
    "load_policy",
    "run_condition",
    "run_episode",
    "run_matrix",
    "summarize",
]

_METRIC_KEYS = (
    "distance_m",
    "laps",
    "duration_s",
    "lane_rms_m",
    "lane_max_m",
    "lane_abs_integral_ms",
    "out_of_lane_integral_ms",
    "p_far",
    "collisions_per_min",
    "success",
    "return",
)


@dataclass(frozen=True)
class Condition:
    """One row of the S8.4 evaluation matrix.

    Attributes:
        name: the S8.4 condition label, for example ``"C5"``.
        description: what the condition probes.
        overrides: :class:`MjEnvCfg` fields to override for this condition.
        budget_min: the S8.4 wall-clock allowance for 3 seeds, in minutes.
    """

    name: str
    description: str
    overrides: dict[str, Any] = field(default_factory=dict)
    budget_min: float = 9.0


CONDITIONS: dict[str, Condition] = {
    "C5": Condition(
        name="C5",
        description="MuJoCo nominal: identified physics, no randomization.",
        overrides={"dynamics_dr": False, "photometric_dr": False},
        budget_min=9.0,
    ),
    "C6": Condition(
        name="C6",
        description=(
            "MuJoCo plus its own randomization: layer-1 photometric DR inside the shared "
            "preprocess chain, and the identified-dynamics jitter of S7.3. Dynamics coverage is "
            "D3-D8, D12, D13, D16, D17, D18 and the drive-train pair (armature, joint friction) "
            "the S8.2 identification fits. Documented coverage gaps, all of them structural rather "
            "than forgotten: scene-graph visual axes V1-V8 and V13 have no MuJoCo counterpart; "
            "D10 control-period jitter is excluded because S8.3 item 3 locks the decimation; and "
            "D11 drag, D14 external pushes and D15 per-tile friction patches are not modelled by "
            "this harness."
        ),
        overrides={"dynamics_dr": True, "photometric_dr": True},
        budget_min=9.0,
    ),
}
"""The MuJoCo half of the S8.4 matrix. C0-C4 and C8 run in Isaac; C7 is the optional referee."""


@dataclass
class Budget:
    """Wall-clock accounting for one condition.

    Attributes:
        episodes: total episodes to run.
        seconds_per_episode: measured cost of one episode, in seconds.
        workers: number of worker processes.
        allowance_s: the S8.4 allowance for this condition, in seconds.
        elapsed_s: measured wall clock, filled in after the run.
    """

    episodes: int
    seconds_per_episode: float
    workers: int
    allowance_s: float
    elapsed_s: float = float("nan")

    @property
    def projected_s(self) -> float:
        """Projected wall clock in seconds, assuming perfect worker utilization."""
        return self.episodes * self.seconds_per_episode / max(1, self.workers)

    @property
    def overrun_ratio(self) -> float:
        """Measured wall clock divided by the S8.4 allowance."""
        return self.elapsed_s / self.allowance_s if self.allowance_s > 0 else float("nan")

    @property
    def within_budget(self) -> bool:
        """True when the run stayed inside the S8.4 allowance plus the 50% tolerance."""
        ratio = self.overrun_ratio
        return bool(ratio == ratio and ratio <= 1.5)


@dataclass(frozen=True)
class PolicySpec:
    """How a worker should obtain the policy.

    Attributes:
        kind: ``"torchscript"`` loads a traced module, ``"constant"`` and ``"zero"`` are built-in
            scripted policies used for smoke tests and for the wall-clock calibration run.
        path: the artifact path, for ``"torchscript"``.
        device: torch device string.
        action: the constant action, for ``"constant"``.
    """

    kind: str = "zero"
    path: str = ""
    device: str = "cpu"
    action: tuple[float, float] = (0.0, 0.0)


def load_policy(spec: PolicySpec) -> Callable[[dict[str, np.ndarray]], np.ndarray]:
    """Return a callable mapping an observation dict to an action.

    The TorchScript path is deliberate: SPEC v2 S8.3 item 5 requires that the *same* traced artifact
    drives the Isaac evaluation, the MuJoCo evaluation and the ONNX parity test, so that no
    evaluation can silently run a differently-defined network from the one that was trained.

    Args:
        spec: how to obtain the policy.

    Returns:
        The policy callable.

    Raises:
        SharedModuleUnavailable: if a TorchScript policy is requested without torch installed.
        ValueError: if the policy kind is unknown.
    """
    if spec.kind == "zero":
        return lambda _obs: np.zeros(2, dtype=np.float32)
    if spec.kind == "constant":
        action = np.asarray(spec.action, dtype=np.float32)
        return lambda _obs: action
    if spec.kind != "torchscript":
        raise ValueError(f"unknown policy kind {spec.kind!r}")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the interpreter
        raise SharedModuleUnavailable(
            "a TorchScript policy needs torch, which this interpreter does not have. Install CPU "
            "torch into the tools venv (SPEC v2 M0):\n  "
            "d:/Personal/personal/mujoco_venv/Scripts/python.exe -m pip install torch "
            "--index-url https://download.pytorch.org/whl/cpu"
        ) from exc
    module = torch.jit.load(spec.path, map_location=spec.device).eval()

    @torch.no_grad()
    def policy(obs: dict[str, np.ndarray]) -> np.ndarray:
        args = []
        if "rgb" in obs:
            args.append(torch.from_numpy(obs["rgb"]).unsqueeze(0).to(spec.device))
        if "vec" in obs:
            args.append(torch.from_numpy(obs["vec"]).unsqueeze(0).to(spec.device))
        out = module(*args)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out.squeeze(0).cpu().numpy().astype(np.float32)

    return policy


def run_episode(
    env: MjDuckiebotEnv,
    policy: Callable[[dict[str, np.ndarray]], np.ndarray],
    seed: int,
    max_seconds: float = 45.0,
) -> EpisodeMetrics:
    """Run one episode and return its metrics.

    Args:
        env: the environment; it is reset here.
        policy: the policy callable.
        seed: episode seed; it drives the spawn pose and the randomization sample.
        max_seconds: the S8.4 episode cap, in seconds of simulated time.

    Returns:
        The completed :class:`EpisodeMetrics`.
    """
    obs, _info = env.reset(seed=seed)
    steps = round(max_seconds / env.control_dt)
    for _ in range(steps):
        obs, _reward, terminated, truncated, _info = env.step(policy(obs))
        if terminated or truncated:
            break
    return env.metrics


@dataclass(frozen=True)
class WorkerSpec:
    """A picklable unit of work: one seed's worth of episodes for one condition.

    Attributes:
        condition: the condition name.
        cfg_kwargs: :class:`MjEnvCfg` fields, already merged with the condition overrides.
        policy: how to obtain the policy.
        seed: the training seed this block belongs to.
        episodes: how many episodes to run.
        episode_offset: index of the first episode, so seeds never overlap.
        max_seconds: the S8.4 episode cap.
    """

    condition: str
    cfg_kwargs: dict[str, Any]
    policy: PolicySpec
    seed: int
    episodes: int
    episode_offset: int
    max_seconds: float


def _run_block(spec: WorkerSpec) -> list[dict[str, Any]]:
    """Worker entry point: build one environment and run a block of episodes through it.

    Args:
        spec: the unit of work.

    Returns:
        One metrics dict per episode, each tagged with its condition and seed.
    """
    env = MjDuckiebotEnv(MjEnvCfg(**spec.cfg_kwargs))
    policy = load_policy(spec.policy)
    out: list[dict[str, Any]] = []
    try:
        for index in range(spec.episodes):
            episode_seed = spec.seed * 1_000_003 + spec.episode_offset + index
            metrics = run_episode(env, policy, episode_seed, spec.max_seconds)
            record = metrics.as_dict()
            record["condition"] = spec.condition
            record["seed"] = spec.seed
            record["episode"] = spec.episode_offset + index
            duration = float(record["duration_s"]) or 1e-9
            record["collisions_per_min"] = 60.0 * float(record["collisions"]) / duration
            out.append(record)
    finally:
        env.close()
    return out


def _bootstrap_median_ci(
    values: Sequence[float], rng: np.random.Generator, resamples: int = 2000, alpha: float = 0.05
) -> tuple[float, float]:
    """Return a percentile bootstrap confidence interval for the median.

    Args:
        values: the sample.
        rng: the generator used for resampling.
        resamples: number of bootstrap resamples.
        alpha: two-sided significance level.

    The sample is sorted before resampling. The bootstrap draws a fixed index matrix from ``rng``,
    so ``array[draws]`` depends on the ORDER of the sample, not only on its multiset: the median of
    each resample changes if the same episodes arrive in a different order. Episode records come
    back from ``imap_unordered``, whose order varies run to run, so without this sort the reported
    ``ci_low``/``ci_high`` moved between identical invocations of the same seed over the same
    episodes. A confidence interval that is not reproducible cannot go in a report.

    Returns:
        ``(low, high)``, or ``(nan, nan)`` for an empty sample.
    """
    array = np.sort(np.asarray([v for v in values if v == v], dtype=np.float64))
    if array.size == 0:
        return float("nan"), float("nan")
    draws = rng.integers(0, array.size, size=(resamples, array.size))
    medians = np.median(array[draws], axis=1)
    return float(np.percentile(medians, 100 * alpha / 2)), float(
        np.percentile(medians, 100 * (1 - alpha / 2))
    )


def summarize(records: Sequence[dict[str, Any]], seed: int = 0) -> dict[str, Any]:
    """Aggregate episode records into the S8.4 reported quantities.

    Args:
        records: episode metric dicts, all from one condition.
        seed: seed for the bootstrap resampling, so the confidence intervals are reproducible.

    Returns:
        A dict with ``episodes``, ``per_metric`` (median, IQR, bootstrap CI, and the across-seed
        mean and standard deviation) and ``failure_reasons``.
    """
    rng = np.random.default_rng(seed)
    seeds = sorted({int(r["seed"]) for r in records})
    per_metric: dict[str, dict[str, float]] = {}
    for key in _METRIC_KEYS:
        values = [float(r[key]) for r in records if key in r and r[key] == r[key]]
        if not values:
            continue
        low, high = _bootstrap_median_ci(values, rng)
        per_seed = [
            statistics.median([float(r[key]) for r in records if int(r["seed"]) == s and key in r])
            for s in seeds
        ]
        per_metric[key] = {
            "median": float(np.median(values)),
            "q25": float(np.percentile(values, 25)),
            "q75": float(np.percentile(values, 75)),
            "ci_low": low,
            "ci_high": high,
            "seed_mean": float(np.mean(per_seed)) if per_seed else float("nan"),
            "seed_sd": float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0,
        }
    reasons: dict[str, int] = {}
    for record in records:
        reasons[str(record["reason"])] = reasons.get(str(record["reason"]), 0) + 1
    return {
        "episodes": len(records),
        "seeds": seeds,
        "per_metric": per_metric,
        "failure_reasons": reasons,
    }


@dataclass
class EvaluationReport:
    """The full result of an evaluation run.

    Attributes:
        conditions: per-condition summaries.
        budgets: per-condition wall-clock accounting.
        records: every episode record, kept so the report can be re-aggregated.
        provenance: environment and parameter provenance, so a number can always be traced.
        caveat: the framing sentence that must accompany any reported delta.
    """

    conditions: dict[str, dict[str, Any]]
    budgets: dict[str, Budget]
    records: list[dict[str, Any]]
    provenance: dict[str, Any]
    caveat: str = (
        "The C1-versus-C5 delta is a robustness probe across two independent physics engines and "
        "two independent renderers. It is not a prediction of real-world performance; no physical "
        "robot was involved at any point in this project."
    )

    def save(self, path: str | Path) -> Path:
        """Write the report as JSON.

        Args:
            path: destination file.

        Returns:
            The destination path.
        """
        payload = {
            "conditions": self.conditions,
            "budgets": {k: asdict(v) for k, v in self.budgets.items()},
            "records": self.records,
            "provenance": self.provenance,
            "caveat": self.caveat,
        }
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return destination

    def report(self) -> str:
        """Render a human-readable table."""
        lines = ["sim-to-sim evaluation (SPEC v2 S8.4)", f"  caveat: {self.caveat}"]
        for name, provenance in self.provenance.items():
            lines.append(f"  {name}: {provenance}")
        header = (
            f"  {'condition':<10}{'eps':>5}{'distance m':>16}{'laps':>10}"
            f"{'lane RMS m':>12}{'survival s':>12}{'success':>10}{'coll/min':>10}"
        )
        lines.append(header)

        def show(metrics: dict[str, Any], key: str, digits: int = 3) -> str:
            """Format one metric's median, or a dash when the metric is absent."""
            entry = metrics.get(key)
            return "-" if entry is None else f"{entry['median']:.{digits}f}"

        for name, summary in self.conditions.items():
            metrics = summary["per_metric"]
            lines.append(
                f"  {name:<10}{summary['episodes']:>5}"
                f"{show(metrics, 'distance_m'):>16}{show(metrics, 'laps'):>10}"
                f"{show(metrics, 'lane_rms_m', 4):>12}{show(metrics, 'duration_s', 1):>12}"
                f"{show(metrics, 'success', 2):>10}{show(metrics, 'collisions_per_min', 2):>10}"
            )
        for name, summary in self.conditions.items():
            entry = summary["per_metric"].get("distance_m")
            if entry is not None:
                lines.append(
                    f"  {name} distance: median {entry['median']:.3f} m "
                    f"IQR [{entry['q25']:.3f}, {entry['q75']:.3f}] "
                    f"95% CI [{entry['ci_low']:.3f}, {entry['ci_high']:.3f}] "
                    f"across-seed {entry['seed_mean']:.3f} +/- {entry['seed_sd']:.3f}"
                )
            lines.append(f"  {name} failures: {summary['failure_reasons']}")
        lines.append("  wall clock")
        for name, budget in self.budgets.items():
            state = "within budget" if budget.within_budget else "OVER BUDGET"
            lines.append(
                f"    {name:<6} {budget.elapsed_s:8.1f} s over {budget.workers} workers "
                f"(projected {budget.projected_s:.1f} s, S8.4 allowance {budget.allowance_s:.0f} s) "
                f"-> {state}"
            )
        return "\n".join(lines)


def _default_workers() -> int:
    """Return a worker count that leaves one core for the operating system."""
    return max(1, (os.cpu_count() or 2) - 1)


def run_condition(
    condition: Condition,
    policy: PolicySpec,
    base_cfg: MjEnvCfg,
    seeds: Sequence[int] = (0, 1, 2),
    episodes_per_seed: int = 200,
    max_seconds: float = 45.0,
    workers: int | None = None,
    calibrate_episodes: int = 2,
) -> tuple[list[dict[str, Any]], Budget]:
    """Run every episode of one condition, in parallel, with a wall-clock budget.

    Args:
        condition: the condition to run.
        policy: how workers should obtain the policy.
        base_cfg: the base environment configuration; the condition's overrides are applied on top.
        seeds: the training seeds being evaluated.
        episodes_per_seed: episodes per seed.
        max_seconds: the S8.4 episode cap in simulated seconds.
        workers: process count; None leaves one core for the operating system.
        calibrate_episodes: episodes run in-process first to measure the per-episode cost, so the
            projected wall clock is printed before the long run rather than discovered after it.

    Returns:
        ``(records, budget)``.
    """
    workers = workers if workers is not None else _default_workers()
    cfg_kwargs = {**asdict(replace(base_cfg)), **condition.overrides}
    # The S8.4 episode cap has exactly one source of truth. The env truncates at
    # cfg.episode_length_s, so a caller passing a base_cfg that still carries the 30 s training
    # horizon would silently run 33% shorter episodes than the frozen protocol states, which moves
    # survival time, fractional laps and distance -- three reported metrics.
    declared = float(cfg_kwargs.get("episode_length_s", max_seconds))
    if abs(declared - float(max_seconds)) > 1e-9 and "episode_length_s" in condition.overrides:
        raise ValueError(
            f"condition {condition.name!r} overrides episode_length_s to {declared} s while "
            f"run_condition was called with max_seconds={max_seconds} s. The S8.4 cap must have "
            f"one value; set them to the same number or drop the override."
        )
    cfg_kwargs["episode_length_s"] = float(max_seconds)
    total = len(seeds) * episodes_per_seed

    started = time.time()
    calibration = _run_block(
        WorkerSpec(
            condition=condition.name,
            cfg_kwargs=cfg_kwargs,
            policy=policy,
            seed=int(seeds[0]),
            episodes=max(1, calibrate_episodes),
            episode_offset=10_000_000,
            max_seconds=max_seconds,
        )
    )
    per_episode = (time.time() - started) / max(1, len(calibration))
    budget = Budget(
        episodes=total,
        seconds_per_episode=per_episode,
        workers=workers,
        allowance_s=condition.budget_min * 60.0,
    )

    specs = [
        WorkerSpec(
            condition=condition.name,
            cfg_kwargs=cfg_kwargs,
            policy=policy,
            seed=int(seed),
            episodes=episodes_per_seed,
            episode_offset=0,
            max_seconds=max_seconds,
        )
        for seed in seeds
    ]
    # Split each seed's block across workers so a 3-seed run can still use every core.
    blocks: list[WorkerSpec] = []
    per_worker = max(1, math.ceil(episodes_per_seed / max(1, workers)))
    for spec in specs:
        offset = 0
        while offset < spec.episodes:
            count = min(per_worker, spec.episodes - offset)
            blocks.append(replace(spec, episodes=count, episode_offset=offset))
            offset += count

    started = time.time()
    records: list[dict[str, Any]] = []
    if workers <= 1 or len(blocks) == 1:
        for block in blocks:
            records.extend(_run_block(block))
    else:
        context = mp.get_context("spawn")
        with context.Pool(processes=min(workers, len(blocks))) as pool:
            for chunk in pool.imap_unordered(_run_block, blocks):
                records.extend(chunk)
    budget.elapsed_s = time.time() - started
    # imap_unordered returns blocks in completion order, which is wall-clock dependent. Every
    # reported quantity downstream must not be: sort into (condition, seed, episode) order so the
    # same seeds over the same episodes give byte-identical output on every run.
    records.sort(key=lambda r: (str(r["condition"]), int(r["seed"]), int(r["episode"])))
    return records, budget


def run_matrix(
    policy: PolicySpec,
    base_cfg: MjEnvCfg | None = None,
    conditions: Iterable[str] = ("C5", "C6"),
    seeds: Sequence[int] = (0, 1, 2),
    episodes_per_seed: int = 200,
    max_seconds: float = 45.0,
    workers: int | None = None,
) -> EvaluationReport:
    """Run the MuJoCo half of the S8.4 matrix.

    Args:
        policy: how workers should obtain the policy.
        base_cfg: base environment configuration; defaults to a 45 s evaluation episode.
        conditions: condition names to run.
        seeds: training seeds being evaluated.
        episodes_per_seed: episodes per seed per condition.
        max_seconds: the S8.4 episode cap.
        workers: process count.

    Returns:
        The assembled :class:`EvaluationReport`.

    Raises:
        KeyError: if a condition name is not in :data:`CONDITIONS`.
    """
    base_cfg = base_cfg if base_cfg is not None else MjEnvCfg(episode_length_s=max_seconds)
    probe = MjDuckiebotEnv(MjEnvCfg(**asdict(base_cfg)))
    provenance: dict[str, Any] = dict(probe.provenance())
    provenance["interpreter"] = environment_report()["python"]
    if base_cfg.obs_mode == "rgb_vec":
        probe.scene.assert_vision_ready()
    probe.close()

    summaries: dict[str, dict[str, Any]] = {}
    budgets: dict[str, Budget] = {}
    all_records: list[dict[str, Any]] = []
    for name in conditions:
        if name not in CONDITIONS:
            raise KeyError(f"unknown condition {name!r}; known: {sorted(CONDITIONS)}")
        condition = CONDITIONS[name]
        records, budget = run_condition(
            condition,
            policy,
            base_cfg,
            seeds=seeds,
            episodes_per_seed=episodes_per_seed,
            max_seconds=max_seconds,
            workers=workers,
        )
        summaries[name] = summarize(records)
        summaries[name]["description"] = condition.description
        budgets[name] = budget
        all_records.extend(records)
    return EvaluationReport(conditions=summaries, budgets=budgets, records=all_records, provenance=provenance)
