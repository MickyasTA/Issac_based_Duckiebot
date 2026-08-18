"""Hold one live :class:`~duckiebot_rl.ppo.networks.ActorCritic` and hot-swap its weights.

The live viewer's whole point is that training stays headless while a second window shows the
newest policy driving. That only works if loading a new checkpoint costs nothing structural: the
network object, the simulator, the frame ring and the window all have to survive the swap. So
:class:`PolicyHost` builds the network **once** and every subsequent load is a
``load_state_dict`` into the same module, in place.

The dangerous failure mode this module exists to prevent is loading a checkpoint whose
architecture differs from the live network. ``nn.Module.load_state_dict`` would raise a wall of
size-mismatch text from deep inside torch, halfway through a rollout, and the operator would have
to work out which of a dozen shape fields moved. Instead :meth:`PolicyHost.load_checkpoint`
validates first, in two passes:

1. **Config pass.** If the checkpoint records ``config.network`` it is rebuilt into a
   :class:`~duckiebot_rl.ppo.config.NetworkConfig` and compared field by field against the live
   one, on the fields that change tensor shapes.
2. **State-dict pass.** Every parameter and buffer name and shape is compared against the live
   module. This catches checkpoints that record no config at all.

Either failure raises :class:`ArchitectureMismatch`, which names the offending fields, and the
live network is left untouched, still serving the previous policy.

Observation normalisation
-------------------------
``vec`` reaches the policy already normalised during training, so the viewer has to reproduce
that. The statistics come from the checkpoint itself, preferring the learner's own
``learner.vec_norm`` buffers and falling back to the flat ``running_norm.vec`` mirror that
``duckiebot_rl.deploy.export_onnx`` reads. Skipping this step is not a cosmetic error: an
unnormalised ``vec`` moves the fusion-layer input off the manifold the actor was trained on and
the policy drives off the road while looking perfectly healthy.

Determinism
-----------
:meth:`PolicyHost.act` returns the Gaussian **mean** by default. Visualising a sampled action
would mix policy improvement with exploration noise and make two consecutive rollouts of the same
checkpoint incomparable. ``stochastic=True`` samples instead, for when the exploration
distribution is what you actually want to see.

One observation or many
-----------------------
:meth:`PolicyHost.act` is the single-environment call the chase-camera viewer uses.
:meth:`PolicyHost.act_batch` is the same computation with the batch axis kept, for the parallel
grid viewer that drives every environment of a 64-city grid from one policy. Both funnel through
:meth:`PolicyHost._actions`, so the two can never drift: the actor-critic is natively batched and
neither path loops over environments. Both accept numpy arrays and torch tensors, including the
CUDA tensors the Isaac environment hands back, because a numpy detour through a CUDA tensor
raises rather than copies.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from duckiebot_rl.ppo.config import NetworkConfig
from duckiebot_rl.ppo.networks import ActorCritic

__all__ = ["ArchitectureMismatch", "HostState", "PolicyHost", "network_config_from_checkpoint"]

SHAPE_FIELDS: tuple[str, ...] = (
    "use_image",
    "obs_height",
    "obs_width",
    "obs_channels",
    "vec_dim",
    "priv_dim",
    "act_dim",
    "encoder_channels",
    "encoder_out",
    "hidden_dim",
    "squash",
)
"""``NetworkConfig`` fields that change tensor shapes or the forward path.

Everything else in ``NetworkConfig`` (init gains, ``sigma_init``, the ``log_std`` clamps) only
affects how a network was *initialised* or how its output is post-processed, so a checkpoint that
differs there is still loadable and is not a mismatch.
"""

_STATE_DICT_PATHS: tuple[tuple[str, ...], ...] = (
    ("learner", "model"),
    ("model",),
    ("agent",),
)


class ArchitectureMismatch(RuntimeError):
    """Raised when a checkpoint's architecture does not match the live network.

    Raised *before* any weight is touched, so the host keeps serving the policy it already had.
    """


@dataclass(frozen=True)
class HostState:
    """What the host is currently serving.

    Attributes:
        source: Path the weights were loaded from, or ``"<payload>"`` for an in-memory load.
        iteration: PPO iteration recorded in the checkpoint or supplied by the caller.
        metric_name: Model-selection metric name, or ``""``.
        metric_value: Model-selection metric value, or None.
        sha256: Content hash the caller verified, or ``""``.
        reload_count: How many successful loads this host has performed, starting at 1.
        alpha_vis: Visual curriculum scalar recorded in the checkpoint, or None.
        alpha_dyn: Dynamics curriculum scalar recorded in the checkpoint, or None.
        normalized_vec: True when vector statistics were found and are being applied.
    """

    source: str
    iteration: int
    metric_name: str
    metric_value: float | None
    sha256: str
    reload_count: int
    alpha_vis: float | None
    alpha_dyn: float | None
    normalized_vec: bool

    def describe(self) -> str:
        """Return the one-line banner the viewer prints on reload.

        Returns:
            A human-readable summary of the loaded policy.
        """
        metric = "n/a"
        if self.metric_value is not None:
            metric = f"{self.metric_name or 'metric'}={self.metric_value:.4g}"
        alpha = ""
        if self.alpha_vis is not None or self.alpha_dyn is not None:
            alpha = f" alpha_vis={self.alpha_vis} alpha_dyn={self.alpha_dyn}"
        norm = "vec-normalised" if self.normalized_vec else "vec-RAW(no stats in checkpoint)"
        return (
            f"reload #{self.reload_count} from {self.source} iteration={self.iteration} "
            f"{metric} {norm}{alpha}"
        )


def _extract_state_dict(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Find the actor-critic state dict inside a checkpoint payload.

    Args:
        payload: A loaded checkpoint.

    Returns:
        The model state dict.

    Raises:
        ArchitectureMismatch: If no recognisable model state dict is present.
    """
    for path in _STATE_DICT_PATHS:
        node: Any = payload
        for key in path:
            node = node.get(key) if isinstance(node, Mapping) else None
            if node is None:
                break
        if isinstance(node, Mapping) and node:
            return dict(node)
    raise ArchitectureMismatch(
        "checkpoint carries no actor-critic state dict; looked for "
        + ", ".join("'" + ".".join(p) + "'" for p in _STATE_DICT_PATHS)
    )


def network_config_from_checkpoint(payload: Mapping[str, Any]) -> NetworkConfig | None:
    """Rebuild the :class:`NetworkConfig` a checkpoint was trained with.

    Args:
        payload: A loaded checkpoint.

    Returns:
        The recorded network configuration, or None when the checkpoint records no config. Unknown
        keys are dropped so a config written by a newer trainer still loads.
    """
    raw = payload.get("config")
    if not isinstance(raw, Mapping):
        return None
    network = raw.get("network")
    if not isinstance(network, Mapping):
        return None
    valid = {field.name for field in dataclasses.fields(NetworkConfig)}
    kwargs = {key: value for key, value in network.items() if key in valid}
    if "encoder_channels" in kwargs:
        kwargs["encoder_channels"] = tuple(kwargs["encoder_channels"])
    try:
        return NetworkConfig(**kwargs)
    except (TypeError, ValueError) as exc:  # pragma: no cover - a corrupt config block
        raise ArchitectureMismatch(f"checkpoint records an invalid network config: {exc}") from exc


def _compare_configs(live: NetworkConfig, incoming: NetworkConfig) -> list[str]:
    """Return human-readable differences on the shape-relevant fields.

    Args:
        live: The configuration the host was built with.
        incoming: The configuration recorded in the checkpoint.

    Returns:
        One string per differing field; empty when the two agree.
    """
    diffs: list[str] = []
    for field in SHAPE_FIELDS:
        here = getattr(live, field)
        there = getattr(incoming, field)
        if isinstance(here, (list, tuple)) or isinstance(there, (list, tuple)):
            here, there = tuple(here), tuple(there)
        if here != there:
            diffs.append(f"{field}: live={here!r} checkpoint={there!r}")
    return diffs


def _compare_state_dicts(live: Mapping[str, Any], incoming: Mapping[str, Any]) -> list[str]:
    """Return human-readable differences between two state dicts.

    Args:
        live: ``state_dict()`` of the live module.
        incoming: The checkpoint's state dict.

    Returns:
        One string per missing key, unexpected key or shape mismatch; empty when they agree.
    """
    diffs: list[str] = []
    missing = sorted(set(live) - set(incoming))
    unexpected = sorted(set(incoming) - set(live))
    if missing:
        diffs.append(f"missing from checkpoint: {', '.join(missing[:6])}")
    if unexpected:
        diffs.append(f"unexpected in checkpoint: {', '.join(unexpected[:6])}")
    for key in sorted(set(live) & set(incoming)):
        here = tuple(live[key].shape)
        there = tuple(np.shape(incoming[key]))
        if here != there:
            diffs.append(f"{key}: live={here} checkpoint={there}")
    return diffs


def _vec_stats(payload: Mapping[str, Any], vec_dim: int) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Recover the actor's running vector-observation statistics from a checkpoint.

    The learner's own ``vec_norm`` buffers are preferred because they are what the rollout used.
    The flat ``running_norm`` mirror written for the ONNX exporter is the fallback.

    Args:
        payload: A loaded checkpoint.
        vec_dim: Expected vector width.

    Returns:
        ``(mean, std)`` as 1-D float32 tensors, or None when the checkpoint carries no usable
        statistics of the right width.
    """
    candidates: list[Mapping[str, Any]] = []
    learner = payload.get("learner")
    if isinstance(learner, Mapping) and isinstance(learner.get("vec_norm"), Mapping):
        candidates.append(learner["vec_norm"])
    running = payload.get("running_norm")
    if isinstance(running, Mapping) and isinstance(running.get("vec"), Mapping):
        candidates.append(running["vec"])

    for entry in candidates:
        if entry.get("mean") is None:
            continue
        mean = torch.as_tensor(entry["mean"], dtype=torch.float32).reshape(-1)
        std_raw = entry.get("std")
        if std_raw is not None:
            std = torch.as_tensor(std_raw, dtype=torch.float32).reshape(-1)
        elif entry.get("var") is not None:
            std = torch.sqrt(torch.as_tensor(entry["var"], dtype=torch.float32).reshape(-1) + 1e-8)
        else:
            continue
        if mean.numel() == vec_dim and std.numel() == vec_dim:
            return mean, std.clamp_min(1e-6)
    return None


class PolicyHost:
    """Own one actor-critic and hot-reload its weights without rebuilding anything.

    Args:
        network_cfg: Architecture to build. When None the host stays unbuilt until the first
            :meth:`load_checkpoint`, which then adopts the checkpoint's own architecture.
        device: Torch device string.
        stochastic: Sample from the policy instead of returning the mean.
        vec_clip: Symmetric clip applied after vector normalisation, matching
            ``PPOConfig.vec_clip``.
        normalize_vec: Apply the checkpoint's running vector statistics. Turning this off is an
            ablation, not a convenience: the policy was trained on normalised input.
        seed: Seed for the sampling generator, used only when ``stochastic`` is True.

    Attributes:
        device: The torch device in use.
        stochastic: Whether :meth:`act` samples.
        vec_clip: The post-normalisation clip.
        normalize_vec: Whether normalisation is applied.
        state: What the host is currently serving, or None before the first load.
    """

    def __init__(
        self,
        network_cfg: NetworkConfig | None = None,
        device: str | torch.device = "cpu",
        stochastic: bool = False,
        vec_clip: float = 5.0,
        normalize_vec: bool = True,
        seed: int | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.stochastic = bool(stochastic)
        self.vec_clip = float(vec_clip)
        self.normalize_vec = bool(normalize_vec)
        self.state: HostState | None = None
        self._cfg: NetworkConfig | None = None
        self._agent: ActorCritic | None = None
        self._vec_mean: torch.Tensor | None = None
        self._vec_std: torch.Tensor | None = None
        self._reload_count = 0
        self._generator = torch.Generator(device="cpu")
        if seed is not None:
            self._generator.manual_seed(int(seed))
        if network_cfg is not None:
            self._build(network_cfg)

    @classmethod
    def from_checkpoint(
        cls,
        path: str | os.PathLike[str],
        device: str | torch.device = "cpu",
        stochastic: bool = False,
        vec_clip: float = 5.0,
        normalize_vec: bool = True,
        seed: int | None = None,
    ) -> PolicyHost:
        """Build a host whose architecture is taken from a checkpoint, then load it.

        Args:
            path: Checkpoint file.
            device: Torch device string.
            stochastic: Sample instead of returning the mean.
            vec_clip: Symmetric clip applied after vector normalisation.
            normalize_vec: Apply the checkpoint's running vector statistics.
            seed: Seed for the sampling generator.

        Returns:
            A host already serving the checkpoint's policy.
        """
        host = cls(
            network_cfg=None,
            device=device,
            stochastic=stochastic,
            vec_clip=vec_clip,
            normalize_vec=normalize_vec,
            seed=seed,
        )
        host.load_checkpoint(path)
        return host

    def _build(self, cfg: NetworkConfig) -> None:
        """Construct the live network. Called at most once per host.

        Args:
            cfg: The architecture to build.
        """
        self._cfg = cfg
        self._agent = ActorCritic(cfg).to(self.device).eval()
        for parameter in self._agent.parameters():
            parameter.requires_grad_(False)

    @property
    def agent(self) -> ActorCritic:
        """The live actor-critic.

        Returns:
            The network instance, whose identity is stable across reloads.

        Raises:
            RuntimeError: If no checkpoint has been loaded and no config was supplied.
        """
        if self._agent is None:
            raise RuntimeError("PolicyHost has no network yet; call load_checkpoint() first")
        return self._agent

    @property
    def network_cfg(self) -> NetworkConfig:
        """The architecture the live network was built with.

        Returns:
            The network configuration.

        Raises:
            RuntimeError: If no checkpoint has been loaded and no config was supplied.
        """
        if self._cfg is None:
            raise RuntimeError("PolicyHost has no network yet; call load_checkpoint() first")
        return self._cfg

    @property
    def loaded(self) -> bool:
        """True once at least one checkpoint has been loaded successfully."""
        return self.state is not None

    @property
    def reload_count(self) -> int:
        """Number of successful checkpoint loads performed by this host."""
        return self._reload_count

    def load_checkpoint(
        self,
        source: str | os.PathLike[str] | Mapping[str, Any],
        iteration: int | None = None,
        metric_name: str = "",
        metric_value: float | None = None,
        sha256: str = "",
    ) -> HostState:
        """Validate a checkpoint and hot-swap the live weights.

        Nothing is mutated until every check passes, so a rejected checkpoint leaves the host
        serving exactly the policy it served before.

        Args:
            source: Checkpoint path, or an already-loaded payload mapping.
            iteration: Override for the iteration reported in :class:`HostState`.
            metric_name: Model-selection metric name to report.
            metric_value: Model-selection metric value to report.
            sha256: Content hash the caller already verified, recorded for the banner.

        Returns:
            The new :class:`HostState`.

        Raises:
            ArchitectureMismatch: If the checkpoint's architecture differs from the live network,
                or if it carries no recognisable state dict.
        """
        if isinstance(source, Mapping):
            payload: Mapping[str, Any] = source
            origin = "<payload>"
        else:
            payload = torch.load(Path(source), map_location=self.device, weights_only=False)
            origin = str(Path(source))

        incoming_state = _extract_state_dict(payload)
        incoming_cfg = network_config_from_checkpoint(payload)

        if self._agent is None:
            if incoming_cfg is None:
                raise ArchitectureMismatch(
                    f"{origin} records no 'config.network' block, so the architecture cannot be "
                    "inferred. Construct PolicyHost(network_cfg=NetworkConfig(...)) explicitly."
                )
            self._build(incoming_cfg)

        live_cfg = self.network_cfg
        if incoming_cfg is not None:
            diffs = _compare_configs(live_cfg, incoming_cfg)
            if diffs:
                raise ArchitectureMismatch(
                    f"{origin} was trained with a different network architecture; refusing to "
                    "load it into the live network.\n  " + "\n  ".join(diffs)
                )

        live_state = self.agent.state_dict()
        diffs = _compare_state_dicts(live_state, incoming_state)
        if diffs:
            raise ArchitectureMismatch(
                f"{origin} state dict does not fit the live network; refusing to load it.\n  "
                + "\n  ".join(diffs)
            )

        self.agent.load_state_dict(incoming_state)
        self.agent.eval()

        stats = _vec_stats(payload, live_cfg.vec_dim) if self.normalize_vec else None
        if stats is None:
            self._vec_mean = None
            self._vec_std = None
        else:
            self._vec_mean = stats[0].to(self.device).view(1, -1)
            self._vec_std = stats[1].to(self.device).view(1, -1)

        curriculum = payload.get("curriculum")
        alpha_vis = alpha_dyn = None
        if isinstance(curriculum, Mapping):
            alpha_vis = _maybe_float(curriculum.get("alpha_vis"))
            alpha_dyn = _maybe_float(curriculum.get("alpha_dyn"))

        self._reload_count += 1
        self.state = HostState(
            source=origin,
            iteration=int(payload.get("iteration", -1)) if iteration is None else int(iteration),
            metric_name=metric_name,
            metric_value=metric_value,
            sha256=sha256,
            reload_count=self._reload_count,
            alpha_vis=alpha_vis,
            alpha_dyn=alpha_dyn,
            normalized_vec=self._vec_mean is not None,
        )
        return self.state

    def load_from_info(self, info: Any) -> HostState:
        """Load the checkpoint a :class:`~duckiebot_rl.viz.watcher.CheckpointInfo` points at.

        Args:
            info: A ``CheckpointInfo`` from :class:`~duckiebot_rl.viz.watcher.CheckpointWatcher`.

        Returns:
            The new :class:`HostState`.
        """
        return self.load_checkpoint(
            info.path,
            iteration=info.iteration,
            metric_name=info.metric_name,
            metric_value=info.metric_value,
            sha256=info.sha256,
        )

    def _prepare(self, obs: Mapping[str, Any]) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Turn an environment observation into the tensors the actor expects.

        Args:
            obs: Observation mapping with a ``vec`` entry and, in image mode, an ``rgb`` entry.
                Either may already carry a leading batch dimension.

        Returns:
            ``(image, vec)``; ``image`` is None in vec-only mode.

        Raises:
            KeyError: If a required observation entry is missing.
            ValueError: If ``vec`` has the wrong width.
        """
        cfg = self.network_cfg
        if "vec" not in obs:
            raise KeyError("observation has no 'vec' entry")
        vec = _as_tensor(obs["vec"], self.device, dtype=torch.float32)
        if vec.ndim == 1:
            vec = vec.unsqueeze(0)
        if vec.shape[-1] != cfg.vec_dim:
            raise ValueError(f"vec has width {vec.shape[-1]}, network expects {cfg.vec_dim}")
        if self._vec_mean is not None and self._vec_std is not None:
            vec = ((vec - self._vec_mean) / self._vec_std).clamp(-self.vec_clip, self.vec_clip)

        image: torch.Tensor | None = None
        if cfg.use_image:
            if "rgb" not in obs:
                raise KeyError("observation has no 'rgb' entry but the network uses an image encoder")
            image = _as_tensor(obs["rgb"], self.device)
            if image.ndim == 3:
                image = image.unsqueeze(0)
        return image, vec

    def _actions(self, image: torch.Tensor | None, vec: torch.Tensor, sample: bool) -> torch.Tensor:
        """Evaluate the actor over a prepared batch and return the batched action.

        The one place the policy is run. :meth:`act` and :meth:`act_batch` differ only in what
        they do with the batch axis afterwards, which is what keeps the single-environment view
        and the N-environment grid provably showing the same policy.

        Args:
            image: NHWC image batch, or None in vec-only mode.
            vec: ``(B, vec_dim)`` already-normalised vector batch.
            sample: Sample from the Gaussian instead of taking its mean.

        Returns:
            A ``(B, act_dim)`` tensor, tanh-squashed when the network config asks for it.
        """
        features = self.agent.actor_features(image, vec)
        action = self.agent.policy_head.mean(features)
        if sample:
            log_std = self.agent.policy_head.clamped_log_std().expand_as(action)
            noise = torch.randn(action.shape, generator=self._generator).to(action.device)
            action = action + log_std.exp() * noise
        if self.network_cfg.squash:
            action = torch.tanh(action)
        return action

    @torch.no_grad()
    def act(self, obs: Mapping[str, Any], stochastic: bool | None = None) -> np.ndarray:
        """Return the action for one observation.

        Args:
            obs: Observation mapping as produced by the MuJoCo or Isaac environment.
            stochastic: Override the host's sampling mode for this call.

        Returns:
            A ``(act_dim,)`` float32 array. It is the raw Gaussian mean by default, unclipped, in
            the ``[-1, 1]`` action box the environments expect; callers clip it.

        Raises:
            RuntimeError: If no checkpoint has been loaded.
        """
        if not self.loaded:
            raise RuntimeError("PolicyHost.act() called before any checkpoint was loaded")
        image, vec = self._prepare(obs)
        sample = self.stochastic if stochastic is None else bool(stochastic)
        action = self._actions(image, vec, sample)
        return action.squeeze(0).detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def act_batch(self, obs: Mapping[str, Any], stochastic: bool | None = None) -> np.ndarray:
        """Return one action per environment for a batched observation.

        What the parallel grid viewer calls: N environments, one policy, one forward pass.
        Nothing here loops over environments, because :class:`ActorCritic` is natively batched,
        and the running vector statistics loaded from the checkpoint are the same ones
        :meth:`act` applies, broadcast over the batch.

        A batch of N copies of one observation therefore yields N copies of the action
        :meth:`act` returns for it in deterministic mode, to float32 noise: a batch of 64 picks a
        different GEMM kernel than a batch of 1 and the measured disagreement is around 5e-10,
        which is the arithmetic, not a different policy. Sampled actions differ per row on
        purpose: one draw of shape ``(N, act_dim)`` is not N draws of shape ``(1, act_dim)``.

        Args:
            obs: Observation mapping whose entries carry a leading environment axis. Numpy
                arrays and torch tensors are both accepted, including CUDA tensors handed
                straight out of the Isaac environment.
            stochastic: Override the host's sampling mode for this call.

        Returns:
            An ``(N, act_dim)`` float32 array, unclipped, in the ``[-1, 1]`` action box the
            environments expect; callers clip it.

        Raises:
            RuntimeError: If no checkpoint has been loaded.
            ValueError: If ``vec`` carries no environment axis. Squeezing that case through
                silently would drive env 0 and leave the other N-1 robots on stale actions.
        """
        if not self.loaded:
            raise RuntimeError("PolicyHost.act_batch() called before any checkpoint was loaded")
        if "vec" in obs and getattr(obs["vec"], "ndim", 2) != 2:
            raise ValueError(
                "act_batch() expects a batched 'vec' of shape (num_envs, vec_dim); got ndim="
                f"{obs['vec'].ndim}. Use act() for a single observation."
            )
        image, vec = self._prepare(obs)
        sample = self.stochastic if stochastic is None else bool(stochastic)
        action = self._actions(image, vec, sample)
        return action.detach().cpu().numpy().astype(np.float32)

    def describe(self) -> str:
        """Return a one-line description of what is loaded.

        Returns:
            The current state's banner, or a note that nothing is loaded yet.
        """
        return "no checkpoint loaded" if self.state is None else self.state.describe()


def _as_tensor(value: Any, device: torch.device, dtype: Any = None) -> torch.Tensor:
    """Move one observation entry onto the policy device without a numpy detour.

    The single-environment viewer hands numpy in. The parallel grid viewer hands the Isaac
    environment its own CUDA tensors, and ``np.asarray`` on a CUDA tensor raises
    ``TypeError: can't convert cuda:0 device type tensor to numpy`` rather than copying, so the
    detour is not merely wasteful there, it is wrong.

    Args:
        value: A torch tensor, a numpy array, or anything ``np.asarray`` accepts.
        device: Target torch device.
        dtype: Target dtype, or None to keep the source dtype. Images keep theirs on purpose:
            the encoder's own ``prepare`` step accepts uint8 or float32 in ``[0, 255]`` and
            casting here would hide a wrong one.

    Returns:
        A tensor on ``device``, detached from any autograd graph.
    """
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
        if dtype is not None:
            return tensor.to(device=device, dtype=dtype)
        return tensor.to(device=device)
    return torch.as_tensor(np.asarray(value), dtype=dtype, device=device)


def _maybe_float(value: Any) -> float | None:
    """Coerce a curriculum field to float, tolerating tensors and missing values.

    Args:
        value: Raw value from the checkpoint.

    Returns:
        The float, or None.
    """
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
