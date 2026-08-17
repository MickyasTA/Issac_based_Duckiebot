"""Two-scalar automatic domain-randomization curriculum (SPEC v2 S7.4).

This module owns two things:

1. :class:`Range` - the *one* place where the curriculum interpolation rule of SPEC v2 S7.4
   ("every parameter's live range = nominal +/- alpha * (clamp - nominal)") is implemented.
   Every visual and dynamics DR axis is declared as a :class:`Range` and sampled through it, so
   the curriculum scalars automatically gate every axis without any per-axis special casing.
2. :class:`TwoScalarADR` - the ADR loop over the two scalars ``alpha_vis`` and ``alpha_dyn``,
   plus :class:`HardExampleMiner`. Both carry fully serializable state, because SPEC v2 S6.9
   makes the curriculum state a *mandatory* checkpoint field: without it a resume silently
   restarts domain randomization at alpha = 0.

ADR rule (S7.4, verbatim): boundary-sample probability 0.1, buffer 30 episodes, step 0.02,
expand when the mean lane-frame consecutive distance is >= 8 tiles, contract below 4.

Determinism: every stochastic method takes an explicit :class:`torch.Generator`; nothing here
touches global RNG state.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch

__all__ = [
    "CurriculumCfg",
    "HardExampleMiner",
    "HardExampleMinerCfg",
    "Range",
    "RangeBook",
    "TwoScalarADR",
    "sample_book",
]

SampleMode = Literal["linear", "log", "log_from_zero", "int"]


@dataclass(frozen=True)
class Range:
    """A single domain-randomization axis with a curriculum-aware sampler.

    The axis is defined by a nominal (identity / real-world best-estimate) value and the two
    clamps of its widest allowed range. A curriculum scalar ``alpha`` in [0, 1] interpolates
    between "always nominal" (alpha = 0) and "the full table range" (alpha = 1), which is the
    S7.4 rule.

    Attributes:
        lo: Lower clamp of the widest range (alpha = 1).
        hi: Upper clamp of the widest range (alpha = 1).
        nominal: The value the axis collapses to at alpha = 0. For ``log_from_zero`` axes the
            identity element is 0.0 and this field is ignored.
        mode: Interpolation/sampling mode.

            * ``linear``: ``U(nominal + a*(lo-nominal), nominal + a*(hi-nominal))``.
            * ``log``: log-uniform between the clamps, interpolated multiplicatively toward
              ``nominal``; requires ``lo``, ``hi`` and ``nominal`` all > 0.
            * ``log_from_zero``: log-uniform between the clamps, scaled by ``alpha``. Used for
              strictly positive axes whose identity element is 0 (noise sigma, blur length),
              where a multiplicative interpolation toward 0 is undefined.
            * ``int``: like ``linear`` but rounded to an inclusive discrete uniform.
        unit: Free-text unit; documentation only.
        nominal_outside: Set True for the rare axis whose identity element lies *outside* the
            table clamps (JPEG quality is the example: the table is U(30, 95) but "no
            compression" is quality 100). The interpolation rule is unchanged; this flag only
            switches off the typo guard that otherwise requires ``lo <= nominal <= hi``.
    """

    lo: float
    hi: float
    nominal: float = 0.0
    mode: SampleMode = "linear"
    unit: str = ""
    nominal_outside: bool = False

    def __post_init__(self) -> None:
        """Validate the range definition.

        Raises:
            ValueError: If the clamps are inverted, the nominal value is outside the clamps, or
                a log mode is used with a non-positive bound.
        """
        if self.hi < self.lo:
            raise ValueError(f"Range hi ({self.hi}) < lo ({self.lo})")
        check_nominal = self.mode in ("linear", "int", "log") and not self.nominal_outside
        if check_nominal and not self.lo <= self.nominal <= self.hi:
            raise ValueError(f"nominal {self.nominal} outside [{self.lo}, {self.hi}]")
        if self.mode in ("log", "log_from_zero") and self.lo <= 0.0:
            raise ValueError(f"log mode needs lo > 0, got {self.lo}")
        if self.mode == "log" and self.nominal <= 0.0:
            raise ValueError("log mode needs nominal > 0 (use log_from_zero when identity is 0)")

    def live(self, alpha: float) -> tuple[float, float]:
        """Return the live (curriculum-scaled) clamps for a scalar alpha.

        Args:
            alpha: Curriculum scalar in [0, 1].

        Returns:
            The ``(lo, hi)`` interval actually sampled at this alpha.
        """
        a = float(alpha)
        if self.mode == "log_from_zero":
            return (a * self.lo, a * self.hi)
        if self.mode == "log":
            n = self.nominal
            return (n * (self.lo / n) ** a, n * (self.hi / n) ** a)
        lo = self.nominal + a * (self.lo - self.nominal)
        hi = self.nominal + a * (self.hi - self.nominal)
        if self.mode == "int":
            return (float(round(lo)), float(round(hi)))
        return (lo, hi)

    def sample(
        self,
        shape: int | tuple[int, ...],
        alpha: float | torch.Tensor = 1.0,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
        boundary: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Draw independent samples for this axis.

        Args:
            shape: Output shape. An int ``n`` means ``(n,)``.
            alpha: Curriculum scalar in [0, 1]. May be a float or a tensor broadcastable to
                ``shape`` (per-env alphas).
            generator: Torch generator used for every draw (determinism).
            device: Device of the returned tensor.
            boundary: Optional bool tensor broadcastable to ``shape``. Where it is True the
                sample is forced to one of the two live clamps (each with probability 0.5),
                which is the ADR boundary-sampling probe.

        Returns:
            A tensor of shape ``shape``; dtype is ``torch.long`` for ``int`` mode and
            ``torch.float32`` otherwise.
        """
        size = (shape,) if isinstance(shape, int) else tuple(shape)
        u = torch.rand(size, generator=generator, device=device, dtype=torch.float32)
        if boundary is not None:
            edge = torch.rand(size, generator=generator, device=device, dtype=torch.float32)
            u = torch.where(boundary.to(torch.bool), (edge < 0.5).to(u.dtype), u)
        if not isinstance(alpha, torch.Tensor):
            return self._sample_scalar_alpha(u, float(alpha))
        a = alpha.to(device=u.device, dtype=torch.float32)

        if self.mode == "log_from_zero":
            log_lo, log_hi = math.log(self.lo), math.log(self.hi)
            return a * torch.exp(log_lo + u * (log_hi - log_lo))
        if self.mode == "log":
            n = self.nominal
            log_lo, log_hi = math.log(self.lo / n), math.log(self.hi / n)
            return n * torch.exp(a * (log_lo + u * (log_hi - log_lo)))

        lo = self.nominal + a * (self.lo - self.nominal)
        hi = self.nominal + a * (self.hi - self.nominal)
        if self.mode == "int":
            lo_i = torch.round(lo)
            hi_i = torch.round(hi)
            span = (hi_i - lo_i + 1.0).clamp(min=1.0)
            off = torch.clamp(torch.floor(u * span), max=span - 1.0)
            return (lo_i + off).to(torch.long)
        return lo + u * (hi - lo)

    def _sample_scalar_alpha(self, u: torch.Tensor, a: float) -> torch.Tensor:
        """Sample with a python-float alpha (the hot path).

        Keeping alpha a python scalar lets the interpolated bounds be folded into the single
        elementwise kernel that transforms ``u``. The tensor path would instead allocate a
        device tensor per axis per step, and at 17 visual axes x 256 envs x 15 Hz that host-to-
        device traffic was measurably more expensive than the sampling itself.

        Args:
            u: Uniform [0, 1) draws with the desired output shape.
            a: Curriculum scalar.

        Returns:
            The sampled tensor.
        """
        if self.mode == "log_from_zero":
            log_lo, log_hi = math.log(self.lo), math.log(self.hi)
            return torch.exp(u * (log_hi - log_lo) + log_lo) * a
        if self.mode == "log":
            n = self.nominal
            e_lo, e_hi = a * math.log(self.lo / n), a * math.log(self.hi / n)
            return torch.exp(u * (e_hi - e_lo) + e_lo) * n
        lo = self.nominal + a * (self.lo - self.nominal)
        hi = self.nominal + a * (self.hi - self.nominal)
        if self.mode == "int":
            lo_i, hi_i = round(lo), round(hi)
            span = max(hi_i - lo_i + 1, 1)
            return (torch.clamp(torch.floor(u * span), max=span - 1.0) + lo_i).to(torch.long)
        return u * (hi - lo) + lo


RangeBook = Mapping[str, Range]
"""A named collection of DR axes (one entry per SPEC v2 S7 table row)."""


def sample_book(
    book: RangeBook,
    keys: Sequence[str],
    shape: int | tuple[int, ...],
    alpha: float | torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
    boundary: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Sample several axes of a range book at once.

    Args:
        book: The range book to read from.
        keys: Axis names to sample.
        shape: Output shape passed to every :meth:`Range.sample`.
        alpha: Curriculum scalar (float or per-env tensor).
        generator: Torch generator (determinism).
        device: Device of the returned tensors.
        boundary: Optional ADR boundary-probe mask.

    Returns:
        A dict mapping each requested key to its sampled tensor.

    Raises:
        KeyError: If a requested key is not in the book.
    """
    out: dict[str, torch.Tensor] = {}
    for k in keys:
        if k not in book:
            raise KeyError(f"unknown DR axis {k!r}; book has {sorted(book)}")
        out[k] = book[k].sample(shape, alpha, generator=generator, device=device, boundary=boundary)
    return out


@dataclass
class CurriculumCfg:
    """Configuration of the two-scalar ADR loop (SPEC v2 S7.4).

    Attributes:
        boundary_prob: Probability that a resetting env becomes a boundary probe for one of the
            two scalars.
        buffer_size: Number of probe episodes pooled before a scalar is updated.
        step: Additive change applied to a scalar on expand/contract.
        expand_threshold: Mean lane-frame consecutive distance (tiles) at or above which the
            scalar expands.
        contract_threshold: Mean distance below which the scalar contracts.
        alpha_min: Lower clamp of both scalars.
        alpha_max: Upper clamp of both scalars.
        init_alpha_vis: Initial visual scalar.
        init_alpha_dyn: Initial dynamics scalar.
    """

    boundary_prob: float = 0.1
    buffer_size: int = 30
    step: float = 0.02
    expand_threshold: float = 8.0
    contract_threshold: float = 4.0
    alpha_min: float = 0.0
    alpha_max: float = 1.0
    init_alpha_vis: float = 0.0
    init_alpha_dyn: float = 0.0


class TwoScalarADR:
    """Automatic domain randomization over the scalars ``alpha_vis`` and ``alpha_dyn``.

    Usage per training iteration:

    1. On env reset, call :meth:`assign_probes` to decide which envs probe which scalar and to
       obtain the per-env boundary masks handed to :meth:`Range.sample`.
    2. When a probe env finishes an episode, call :meth:`record` with its success metric (mean
       lane-frame consecutive distance, in tiles).
    3. Call :meth:`update` once per iteration; it expands/contracts a scalar as soon as that
       scalar's buffer is full, then clears the buffer.

    The class is free of any Isaac/gym import so it can be unit tested on CPU and restored from
    a checkpoint without a simulator.
    """

    SCALARS: tuple[str, str] = ("vis", "dyn")

    def __init__(self, cfg: CurriculumCfg | None = None) -> None:
        self.cfg = cfg or CurriculumCfg()
        self._alpha: dict[str, float] = {
            "vis": float(self.cfg.init_alpha_vis),
            "dyn": float(self.cfg.init_alpha_dyn),
        }
        self._buffers: dict[str, list[float]] = {"vis": [], "dyn": []}
        self._n_expand: dict[str, int] = {"vis": 0, "dyn": 0}
        self._n_contract: dict[str, int] = {"vis": 0, "dyn": 0}
        self._n_recorded: dict[str, int] = {"vis": 0, "dyn": 0}

    @property
    def alpha_vis(self) -> float:
        """Current visual curriculum scalar in [0, 1]."""
        return self._alpha["vis"]

    @property
    def alpha_dyn(self) -> float:
        """Current dynamics curriculum scalar in [0, 1]."""
        return self._alpha["dyn"]

    def alpha(self, name: str) -> float:
        """Return one scalar by name.

        Args:
            name: ``"vis"`` or ``"dyn"``.

        Returns:
            The scalar value.

        Raises:
            KeyError: If ``name`` is not a known scalar.
        """
        if name not in self._alpha:
            raise KeyError(f"unknown curriculum scalar {name!r}")
        return self._alpha[name]

    def assign_probes(
        self,
        num_envs: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
    ) -> dict[str, torch.Tensor]:
        """Assign ADR boundary probes to envs.

        Each env independently becomes a probe with probability ``cfg.boundary_prob``; a probe
        is assigned to exactly one scalar (50/50), so the two masks are mutually exclusive.

        Args:
            num_envs: Number of envs being (re)assigned.
            generator: Torch generator (determinism).
            device: Device of the returned masks.

        Returns:
            ``{"vis": bool tensor (num_envs,), "dyn": bool tensor (num_envs,)}``.
        """
        r = torch.rand(num_envs, generator=generator, device=device)
        which = torch.rand(num_envs, generator=generator, device=device)
        is_probe = r < self.cfg.boundary_prob
        return {"vis": is_probe & (which < 0.5), "dyn": is_probe & (which >= 0.5)}

    def record(self, name: str, values: Iterable[float]) -> None:
        """Record finished probe episodes for one scalar.

        Args:
            name: ``"vis"`` or ``"dyn"``.
            values: Success metric per finished probe episode (mean lane-frame consecutive
                distance, in tiles).

        Raises:
            KeyError: If ``name`` is not a known scalar.
        """
        if name not in self._buffers:
            raise KeyError(f"unknown curriculum scalar {name!r}")
        buf = self._buffers[name]
        for v in values:
            buf.append(float(v))
            self._n_recorded[name] += 1

    def update(self) -> dict[str, str]:
        """Apply the ADR rule to every scalar whose buffer is full.

        Returns:
            ``{scalar: action}`` where action is ``"expand"``, ``"contract"``, ``"hold"``
            (buffer full but the metric sat between the thresholds) or ``"wait"`` (buffer not
            full yet).
        """
        out: dict[str, str] = {}
        for name in self.SCALARS:
            buf = self._buffers[name]
            if len(buf) < self.cfg.buffer_size:
                out[name] = "wait"
                continue
            mean = sum(buf) / len(buf)
            buf.clear()
            if mean >= self.cfg.expand_threshold:
                self._alpha[name] = min(self.cfg.alpha_max, self._alpha[name] + self.cfg.step)
                self._n_expand[name] += 1
                out[name] = "expand"
            elif mean < self.cfg.contract_threshold:
                self._alpha[name] = max(self.cfg.alpha_min, self._alpha[name] - self.cfg.step)
                self._n_contract[name] += 1
                out[name] = "contract"
            else:
                out[name] = "hold"
        return out

    def metrics(self) -> dict[str, float]:
        """Return curriculum diagnostics for TensorBoard.

        Returns:
            A flat dict of scalar metrics.
        """
        return {
            "curriculum/alpha_vis": self._alpha["vis"],
            "curriculum/alpha_dyn": self._alpha["dyn"],
            "curriculum/buffer_vis": float(len(self._buffers["vis"])),
            "curriculum/buffer_dyn": float(len(self._buffers["dyn"])),
            "curriculum/expands_vis": float(self._n_expand["vis"]),
            "curriculum/expands_dyn": float(self._n_expand["dyn"]),
            "curriculum/contracts_vis": float(self._n_contract["vis"]),
            "curriculum/contracts_dyn": float(self._n_contract["dyn"]),
        }

    def state_dict(self) -> dict[str, Any]:
        """Serialize the full curriculum state (SPEC v2 S6.9 mandatory checkpoint field).

        Returns:
            A JSON-compatible dict holding both scalars, both success buffers and the counters.
        """
        return {
            "version": 1,
            "alpha": dict(self._alpha),
            "buffers": {k: list(v) for k, v in self._buffers.items()},
            "n_expand": dict(self._n_expand),
            "n_contract": dict(self._n_contract),
            "n_recorded": dict(self._n_recorded),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore state produced by :meth:`state_dict`.

        Args:
            state: The mapping returned by :meth:`state_dict`.

        Raises:
            KeyError: If a mandatory field is missing. A resume must never silently restart the
                curriculum at alpha = 0 (SPEC v2 S6.9).
        """
        for key in ("alpha", "buffers"):
            if key not in state:
                raise KeyError(f"curriculum state is missing mandatory field {key!r}")
        self._alpha = {k: float(v) for k, v in state["alpha"].items()}
        self._buffers = {k: [float(x) for x in v] for k, v in state["buffers"].items()}
        zero = {"vis": 0, "dyn": 0}
        self._n_expand = {k: int(v) for k, v in state.get("n_expand", zero).items()}
        self._n_contract = {k: int(v) for k, v in state.get("n_contract", zero).items()}
        self._n_recorded = {k: int(v) for k, v in state.get("n_recorded", zero).items()}


@dataclass
class HardExampleMinerCfg:
    """Configuration of the hard-example spawn miner (SPEC v2 S7.4).

    Attributes:
        num_tiles: Number of drivable spawn tiles tracked.
        ema: Exponential-moving-average coefficient of the per-tile error estimate.
        hard_fraction: Fraction of spawns biased to the worst decile.
        decile: Fraction of visited tiles considered "worst" (0.1 = worst decile).
    """

    num_tiles: int = 1
    ema: float = 0.05
    hard_fraction: float = 0.25
    decile: float = 0.1


class HardExampleMiner:
    """Tracks per-tile tracking error and biases spawns toward the worst tiles.

    SPEC v2 S7.4: "every 200k steps bias 25% of spawns to the worst-decile tiles by tracking
    error". The table is part of the checkpoint contract (S6.9).
    """

    def __init__(self, cfg: HardExampleMinerCfg | None = None) -> None:
        self.cfg = cfg or HardExampleMinerCfg()
        self._error = [0.0] * int(self.cfg.num_tiles)
        self._count = [0] * int(self.cfg.num_tiles)

    @property
    def error_table(self) -> list[float]:
        """Per-tile EMA of the tracking error (read-only copy)."""
        return list(self._error)

    def update(self, tile_ids: Sequence[int], errors: Sequence[float]) -> None:
        """Fold finished-episode errors into the per-tile EMA table.

        Args:
            tile_ids: Spawn tile index per finished episode.
            errors: Tracking error (e.g. time-integrated |d|) per finished episode.

        Raises:
            ValueError: If the sequences differ in length or a tile id is out of range.
        """
        if len(tile_ids) != len(errors):
            raise ValueError("tile_ids and errors must have equal length")
        a = float(self.cfg.ema)
        for t, e in zip(tile_ids, errors, strict=True):
            i = int(t)
            if not 0 <= i < len(self._error):
                raise ValueError(f"tile id {i} outside [0, {len(self._error)})")
            if self._count[i]:
                self._error[i] = (1.0 - a) * self._error[i] + a * float(e)
            else:
                self._error[i] = float(e)
            self._count[i] += 1

    def sample_tiles(
        self,
        num: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Sample spawn tiles, biasing ``hard_fraction`` of them to the worst decile.

        Args:
            num: Number of spawn tiles to draw.
            generator: Torch generator (determinism).
            device: Device of the returned tensor.

        Returns:
            A ``(num,)`` long tensor of tile indices.
        """
        n_tiles = max(len(self._error), 1)
        uniform = torch.randint(0, n_tiles, (num,), generator=generator, device=device, dtype=torch.long)
        visited = [i for i, c in enumerate(self._count) if c > 0]
        if not visited:
            return uniform
        k = max(1, round(self.cfg.decile * len(visited)))
        worst = sorted(visited, key=lambda i: self._error[i], reverse=True)[:k]
        worst_t = torch.tensor(worst, dtype=torch.long, device=device)
        pick = torch.randint(0, k, (num,), generator=generator, device=device, dtype=torch.long)
        take_hard = torch.rand(num, generator=generator, device=device) < self.cfg.hard_fraction
        return torch.where(take_hard, worst_t[pick], uniform)

    def state_dict(self) -> dict[str, Any]:
        """Serialize the mining table.

        Returns:
            A JSON-compatible dict with the EMA table and the visit counts.
        """
        return {"version": 1, "error": list(self._error), "count": list(self._count)}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a table produced by :meth:`state_dict`.

        Args:
            state: The mapping returned by :meth:`state_dict`.

        Raises:
            KeyError: If the ``error`` field is missing.
        """
        if "error" not in state:
            raise KeyError("hard-example miner state is missing mandatory field 'error'")
        self._error = [float(x) for x in state["error"]]
        self._count = [int(x) for x in state.get("count", [0] * len(self._error))]
