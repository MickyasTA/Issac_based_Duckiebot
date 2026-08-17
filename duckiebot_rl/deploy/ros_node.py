"""ROS 1 node skeleton for running the exported policy on a Duckiebot (SPEC v2 S9.2).

THIS FILE HAS NEVER BEEN RUN ON A ROBOT, AND CANNOT BE RUN IN THIS REPOSITORY.
There is no physical Duckiebot in this project, no ROS installation on the development machine
(Windows 11), and no Duckietown ML base image. What this module provides is:

* the exact, testable pure functions that define the robot-side half of the S4.3 preprocessing
  chain, in numpy only, so that they can be proven byte-equal to the training path offline
  (``duckiebot_rl.deploy.parity.numpy_tail_parity``);
* a frame ring with the same stacking semantics as the trainer (t, t-2, t-4);
* an onnxruntime inference wrapper with the same input contract as the exported graph;
* the node class itself, whose ROS imports happen inside ``__init__`` so that this module
  stays importable, lintable and unit-testable without ROS.

Deliberate constraints from the specification:

* The node NEVER imports torch. Inference is onnxruntime, or TensorRT once an engine exists.
* Publishing rate is 15 Hz, matching the training control rate exactly.
* The colour order is RGB. ``cv2.imdecode`` returns BGR; the conversion is explicit and is the
  single most common silent sim-to-real bug, so it has its own assertion in the fixture tests.
* A watchdog publishes zero commands when the newest image is older than 0.2 s, and shutdown
  publishes zeros five times.
* ``v`` is capped at a conservative value for first runs; the cap is a node parameter, not
  something baked into the exported graph.

Topics (Duckietown ROS 1 Noetic conventions):
    subscribe  ``~/camera_node/image/compressed``   sensor_msgs/CompressedImage
    subscribe  ``~/camera_node/camera_info``        sensor_msgs/CameraInfo   (latched, once)
    publish    ``~/car_cmd_switch_node/cmd``        duckietown_msgs/Twist2DStamped
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "BOX",
    "CONTROL_RATE_HZ",
    "CROP_TOP",
    "KERNEL5",
    "K_CANON",
    "OBS_H",
    "OBS_W",
    "PRE_BLUR_SIGMA_PX",
    "RENDER_H",
    "RENDER_W",
    "STACK_LEN",
    "STACK_STRIDE",
    "WATCHDOG_TIMEOUT_S",
    "FrameRing",
    "LaneFollowingNode",
    "NodeConfig",
    "OnnxPolicy",
    "bgr_to_rgb",
    "build_rectify_map",
    "decode_compressed_image",
    "main",
    "preprocess_tail_numpy",
    "rectify_to_canonical",
]

# --------------------------------------------------------------------------------------------
# Constants. Duplicated from duckiebot_rl.deploy.export_onnx ON PURPOSE: importing that module
# would pull in torch, and the robot image must not have torch. Equality of the two copies is
# asserted by tests/unit/test_export.py, so the duplication cannot drift silently.
# --------------------------------------------------------------------------------------------

RENDER_W = 192
RENDER_H = 128
OBS_W = 96
OBS_H = 48
CROP_TOP = 16
BOX = 2
KERNEL5: tuple[float, float, float, float, float] = (0.00256, 0.16555, 0.66378, 0.16555, 0.00256)
STACK_LEN = 3
STACK_STRIDE = 2
OBS_CHANNELS = STACK_LEN * 3
VEC_DIM = 8

CANONICAL_FX = 65.98
CANONICAL_CX = 96.0
CANONICAL_CY = 64.0
K_CANON: np.ndarray = np.array(
    [[CANONICAL_FX, 0.0, CANONICAL_CX], [0.0, CANONICAL_FX, CANONICAL_CY], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
"""Canonical pinhole camera matrix at 192x128 (S4.1).

The robot rectifies DIRECTLY to this matrix. ``cv2.getOptimalNewCameraMatrix`` is not used:
the new camera matrix is chosen, not derived, so that the on-robot geometry equals the trained
geometry by construction rather than by luck.
"""

PRE_BLUR_SIGMA_PX = 1.0
"""Gaussian pre-blur applied at 640x480 before the remap (S4.3 robot path).

It compensates for the roughly 3.3x decimation performed by the rectification remap, which
otherwise aliases; the simulator gets the same low-pass for free from its 2x supersampled
render plus box downsample.
"""

CONTROL_RATE_HZ = 15.0
WATCHDOG_TIMEOUT_S = 0.2
SHUTDOWN_ZERO_PUBLISHES = 5
DEFAULT_V_CAP_MPS = 0.25
"""Conservative forward-speed cap for first hardware runs (S9.2)."""


# --------------------------------------------------------------------------------------------
# Pure functions: the robot-side preprocessing chain
# --------------------------------------------------------------------------------------------


def _require_cv2() -> Any:
    """Import OpenCV with an actionable error message.

    Returns:
        The ``cv2`` module.

    Raises:
        ImportError: When OpenCV is not installed.
    """
    try:
        import cv2  # optional dependency, resolved at call time
    except ImportError as exc:  # pragma: no cover - exercised only on machines without cv2
        raise ImportError(
            "OpenCV is required for JPEG decode and rectification on the robot. "
            "Install it with: pip install 'duckiebot-rl[cv]'"
        ) from exc
    return cv2


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Swap the channel order of an HWC image.

    ``cv2.imdecode`` produces BGR while the policy was trained on RGB. Getting this wrong
    costs roughly nothing in loss curves and everything in lane keeping, because yellow and
    blue swap places. It is therefore an explicit, separately tested step.

    Args:
        image: ``(H, W, 3)`` array.

    Returns:
        A new array with the channel order reversed.

    Raises:
        ValueError: If the input is not a 3-channel HWC image.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) image, got shape {image.shape}")
    return np.ascontiguousarray(image[:, :, ::-1])


def decode_compressed_image(payload: bytes | np.ndarray) -> np.ndarray:
    """Decode a ``sensor_msgs/CompressedImage`` payload into an RGB array.

    Args:
        payload: Raw JPEG bytes (the ``data`` field of the message).

    Returns:
        ``(H, W, 3)`` uint8 RGB array.

    Raises:
        ValueError: If the payload cannot be decoded.
    """
    cv2 = _require_cv2()
    buffer = np.frombuffer(payload, dtype=np.uint8) if isinstance(payload, bytes) else payload
    decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("cv2.imdecode returned None; the compressed payload is not a valid image")
    return bgr_to_rgb(decoded)


def build_rectify_map(
    k_raw: Sequence[Sequence[float]] | np.ndarray,
    d_raw: Sequence[float] | np.ndarray,
    size: tuple[int, int] = (RENDER_W, RENDER_H),
) -> tuple[np.ndarray, np.ndarray]:
    """Build the undistort/rectify maps from the raw camera to the canonical pinhole.

    Args:
        k_raw: ``3x3`` intrinsic matrix from ``camera_node/camera_info``.
        d_raw: plumb_bob distortion coefficients.
        size: Output ``(width, height)``; the canonical render size.

    Returns:
        Tuple ``(map1, map2)`` suitable for ``cv2.remap``.
    """
    cv2 = _require_cv2()
    k = np.asarray(k_raw, dtype=np.float64).reshape(3, 3)
    d = np.asarray(d_raw, dtype=np.float64).reshape(-1)
    return cv2.initUndistortRectifyMap(k, d, None, K_CANON, size, cv2.CV_32FC1)


def rectify_to_canonical(
    image_rgb: np.ndarray,
    map1: np.ndarray,
    map2: np.ndarray,
    *,
    pre_blur_sigma: float = PRE_BLUR_SIGMA_PX,
) -> np.ndarray:
    """Pre-blur and rectify a full-resolution camera frame to the canonical render.

    Args:
        image_rgb: ``(480, 640, 3)`` uint8 RGB frame.
        map1: First remap table from :func:`build_rectify_map`.
        map2: Second remap table.
        pre_blur_sigma: Gaussian sigma in pixels applied before the remap.

    Returns:
        ``(128, 192, 3)`` uint8 RGB canonical render.
    """
    cv2 = _require_cv2()
    source = image_rgb
    if pre_blur_sigma > 0.0:
        source = cv2.GaussianBlur(source, ksize=(0, 0), sigmaX=pre_blur_sigma, sigmaY=pre_blur_sigma)
    return cv2.remap(source, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _blur_axis_numpy(x: np.ndarray, axis: int, kernel: Sequence[float]) -> np.ndarray:
    """Apply a 1-D kernel along one spatial axis with replicate padding.

    Args:
        x: ``(H, W, C)`` float32 array.
        axis: 0 for height, 1 for width.
        kernel: Odd-length tap sequence.

    Returns:
        The filtered array, same shape as the input.
    """
    pad = (len(kernel) - 1) // 2
    pad_width = [(0, 0), (0, 0), (0, 0)]
    pad_width[axis] = (pad, pad)
    padded = np.pad(x, pad_width, mode="edge")
    out = np.zeros_like(x)
    length = x.shape[axis]
    for index, tap in enumerate(kernel):
        window = padded[index : index + length] if axis == 0 else padded[:, index : index + length]
        out += np.float32(tap) * window
    return out


def preprocess_tail_numpy(
    render_hwc: np.ndarray,
    *,
    kernel: Sequence[float] = KERNEL5,
    box: int = BOX,
    crop_top: int = CROP_TOP,
    obs_h: int = OBS_H,
) -> np.ndarray:
    """Run S4.3 steps 5 to 8 in numpy: blur, exact box downsample, crop, quantise.

    This is the robot-side twin of ``duckiebot_rl.deploy.export_onnx.BakedPreprocess``. The two
    are compared frame by frame in the parity gate; the S4.3 rule is at most 1 least-significant
    bit of difference, and in practice they agree exactly on the large majority of pixels.

    Args:
        render_hwc: ``(128, 192, C)`` uint8 canonical render. ``C`` is 3 for a single frame or
            9 for an already stacked triple; the operator sequence is per-channel either way.
        kernel: Separable blur taps applied along width then height.
        box: Box-downsample factor.
        crop_top: Rows removed from the top after downsampling.
        obs_h: Output height.

    Returns:
        ``(48, 96, C)`` uint8 observation tile.

    Raises:
        ValueError: If the input geometry does not match the canonical render.
    """
    if render_hwc.ndim != 3 or render_hwc.shape[0] != RENDER_H or render_hwc.shape[1] != RENDER_W:
        raise ValueError(f"expected a ({RENDER_H}, {RENDER_W}, C) render, got shape {render_hwc.shape}")
    x = render_hwc.astype(np.float32) / np.float32(255.0)
    x = _blur_axis_numpy(x, axis=1, kernel=kernel)
    x = _blur_axis_numpy(x, axis=0, kernel=kernel)
    height, width, channels = x.shape
    pooled = x.reshape(height // box, box, width // box, box, channels).mean(axis=(1, 3))
    cropped = pooled[crop_top : crop_top + obs_h]
    return np.clip(np.round(cropped * np.float32(255.0)), 0, 255).astype(np.uint8)


class FrameRing:
    """Fixed-depth ring of preprocessed frames with the trainer's stacking semantics.

    The observation is the channel-wise concatenation of frames ``(t, t-2, t-4)``. On reset
    the ring is filled with copies of the first frame, so the first observation of an episode
    is three copies of one frame, exactly as in the simulator.

    Attributes:
        depth: Ring depth; must be at least ``(stack_len - 1) * stride + 1``.
        stack_len: Number of stacked frames.
        stride: Control steps between stacked frames.
    """

    def __init__(self, depth: int = 5, stack_len: int = STACK_LEN, stride: int = STACK_STRIDE) -> None:
        """Create an empty ring.

        Args:
            depth: Ring depth in frames.
            stack_len: Number of frames in the stack.
            stride: Control steps between stacked frames.

        Raises:
            ValueError: If the depth cannot serve the requested stack.
        """
        needed = (stack_len - 1) * stride + 1
        if depth < needed:
            raise ValueError(f"depth {depth} is too small for stack_len {stack_len} stride {stride}")
        self.depth = int(depth)
        self.stack_len = int(stack_len)
        self.stride = int(stride)
        self._buffer: deque[np.ndarray] = deque(maxlen=self.depth)

    def reset(self, frame: np.ndarray) -> None:
        """Fill the ring with copies of one frame.

        Args:
            frame: ``(48, 96, 3)`` uint8 preprocessed frame.
        """
        self._buffer.clear()
        for _ in range(self.depth):
            self._buffer.append(np.array(frame, copy=True))

    def push(self, frame: np.ndarray) -> None:
        """Append a frame, initialising the ring on the first call.

        Args:
            frame: ``(48, 96, 3)`` uint8 preprocessed frame.
        """
        if not self._buffer:
            self.reset(frame)
            return
        self._buffer.append(np.array(frame, copy=True))

    @property
    def ready(self) -> bool:
        """Whether the ring holds enough frames to build an observation."""
        return len(self._buffer) == self.depth

    def observation(self) -> np.ndarray:
        """Build the stacked observation.

        Returns:
            ``(1, 48, 96, 9)`` uint8 array ready for the exported graph.

        Raises:
            RuntimeError: If called before any frame was pushed.
        """
        if not self._buffer:
            raise RuntimeError("FrameRing is empty; call push() or reset() first")
        frames = list(self._buffer)
        picks = [frames[-1 - i * self.stride] for i in range(self.stack_len)]
        return np.concatenate(picks, axis=2)[None, ...]


class OnnxPolicy:
    """onnxruntime wrapper with the exported graph's input contract.

    Attributes:
        path: Path of the loaded ``.onnx`` file.
        input_names: Graph input names, image first.
        output_names: Graph output names, ``action`` first.
    """

    def __init__(self, onnx_path: str | Path, providers: Sequence[str] | None = None) -> None:
        """Load an ONNX artifact.

        Args:
            onnx_path: Path to the exported graph.
            providers: onnxruntime execution providers; CPU only by default, which is what the
                Nano actually uses when TensorRT is not available.

        Raises:
            ImportError: When onnxruntime is not installed.
            FileNotFoundError: When the artifact is missing.
        """
        try:
            import onnxruntime as ort  # optional dependency
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required to run the policy. Install with: pip install 'duckiebot-rl[export]'"
            ) from exc
        self.path = Path(onnx_path)
        if not self.path.is_file():
            raise FileNotFoundError(f"ONNX artifact not found: {self.path}")
        self._session = ort.InferenceSession(
            str(self.path), providers=list(providers) if providers else ["CPUExecutionProvider"]
        )
        self.input_names = [inp.name for inp in self._session.get_inputs()]
        self.output_names = [out.name for out in self._session.get_outputs()]
        self._image_dtype = np.uint8 if "uint8" in self._session.get_inputs()[0].type else np.float32

    def infer(self, observation: np.ndarray, vec: np.ndarray) -> tuple[float, float]:
        """Run one inference.

        Args:
            observation: ``(1, 48, 96, 9)`` uint8 stacked observation.
            vec: ``(1, 8)`` float32 raw observation vector.

        Returns:
            Tuple ``(v, omega)`` in m/s and rad/s.
        """
        feeds = {
            self.input_names[0]: observation.astype(self._image_dtype, copy=False),
            self.input_names[1]: np.asarray(vec, dtype=np.float32).reshape(1, -1),
        }
        outputs = self._session.run(self.output_names, feeds)
        action = np.asarray(outputs[0]).reshape(-1)
        return float(action[0]), float(action[1])


@dataclass
class NodeConfig:
    """Runtime configuration of the lane-following node.

    Attributes:
        onnx_path: Exported policy to load.
        vehicle_name: Duckiebot hostname, used to build the topic namespace.
        v_cap: Hard cap applied to the commanded forward speed.
        control_rate_hz: Publish rate.
        watchdog_timeout_s: Age above which the newest image is considered stale.
        providers: onnxruntime execution providers.
    """

    onnx_path: str
    vehicle_name: str = "duckiebot"
    v_cap: float = DEFAULT_V_CAP_MPS
    control_rate_hz: float = CONTROL_RATE_HZ
    watchdog_timeout_s: float = WATCHDOG_TIMEOUT_S
    providers: tuple[str, ...] = ("CPUExecutionProvider",)


class LaneFollowingNode:
    """ROS 1 node that drives a Duckiebot with the exported policy.

    NOT RUNNABLE HERE. Constructing this class imports ``rospy``, ``sensor_msgs`` and
    ``duckietown_msgs``, none of which exist on the development machine. The class is written
    out in full anyway so that the deployment path is reviewable, and so that the fixture
    tests can drive :meth:`process_frame`, which is pure and ROS-free.

    Attributes:
        config: The node configuration.
        ring: The frame ring feeding the stacked observation.
        policy: The onnxruntime inference wrapper.
    """

    def __init__(self, config: NodeConfig) -> None:
        """Initialise the node, its ROS handles and the policy.

        Args:
            config: Node configuration.

        Raises:
            ImportError: When the ROS 1 stack is unavailable, which is always the case in this
                repository. The message says so explicitly instead of failing obscurely.
        """
        self.config = config
        self.ring = FrameRing()
        self.policy = OnnxPolicy(config.onnx_path, providers=config.providers)
        self._maps: tuple[np.ndarray, np.ndarray] | None = None
        self._last_action = (0.0, 0.0)
        self._last_image_time = 0.0
        self._latest_frame: np.ndarray | None = None

        try:
            import rospy  # ROS is a runtime-only dependency of this class
            from duckietown_msgs.msg import Twist2DStamped
            from sensor_msgs.msg import CameraInfo, CompressedImage
        except ImportError as exc:  # pragma: no cover - never importable in this repository
            raise ImportError(
                "ROS 1 (rospy, sensor_msgs, duckietown_msgs) is not available. This node runs "
                "only inside a Duckietown ML base image on the robot; on a workstation use "
                "duckiebot_rl.deploy.parity and the fixture tests instead."
            ) from exc

        self._rospy = rospy
        self._twist_cls = Twist2DStamped
        rospy.init_node("rl_lane_following", anonymous=False)
        namespace = f"/{config.vehicle_name}"
        self._pub = rospy.Publisher(f"{namespace}/car_cmd_switch_node/cmd", Twist2DStamped, queue_size=1)
        rospy.Subscriber(
            f"{namespace}/camera_node/camera_info", CameraInfo, self.on_camera_info, queue_size=1
        )
        rospy.Subscriber(
            f"{namespace}/camera_node/image/compressed",
            CompressedImage,
            self.on_image,
            queue_size=1,
            buff_size=2**24,
        )
        rospy.on_shutdown(self.on_shutdown)

    # -- callbacks -----------------------------------------------------------------------

    def on_camera_info(self, msg: Any) -> None:
        """Build the rectification maps once, from the first CameraInfo message.

        Args:
            msg: ``sensor_msgs/CameraInfo``.
        """
        if self._maps is not None:
            return
        self._maps = build_rectify_map(np.asarray(msg.K).reshape(3, 3), np.asarray(msg.D))
        self._rospy.loginfo("rectification maps built for the canonical 192x128 pinhole")

    def on_image(self, msg: Any) -> None:
        """Store the newest compressed frame; inference happens in the fixed-rate loop.

        Keeping decode and inference out of the callback is what makes the keep-newest
        policy real: an overrun callback would otherwise queue stale frames.

        Args:
            msg: ``sensor_msgs/CompressedImage``.
        """
        self._latest_frame = np.frombuffer(msg.data, dtype=np.uint8)
        self._last_image_time = time.monotonic()

    def on_shutdown(self) -> None:
        """Publish zero commands repeatedly so the robot stops on exit."""
        for _ in range(SHUTDOWN_ZERO_PUBLISHES):
            self.publish(0.0, 0.0)
            time.sleep(0.02)

    # -- pure logic ----------------------------------------------------------------------

    def process_frame(self, jpeg_payload: bytes, vec: np.ndarray) -> tuple[float, float]:
        """Decode, preprocess and infer for one frame. Pure and ROS-free.

        Args:
            jpeg_payload: Raw JPEG bytes from the compressed image topic.
            vec: ``(8,)`` raw observation vector.

        Returns:
            Tuple ``(v, omega)``, with ``v`` already capped.

        Raises:
            RuntimeError: If called before the rectification maps exist.
        """
        if self._maps is None:
            raise RuntimeError("camera_info has not arrived yet; no rectification maps")
        rgb = decode_compressed_image(jpeg_payload)
        canonical = rectify_to_canonical(rgb, self._maps[0], self._maps[1])
        tile = preprocess_tail_numpy(canonical)
        self.ring.push(tile)
        v, omega = self.policy.infer(self.ring.observation(), np.asarray(vec, dtype=np.float32))
        return min(v, self.config.v_cap), omega

    def set_rectify_maps(self, map1: np.ndarray, map2: np.ndarray) -> None:
        """Inject rectification maps directly. Used by the fixture tests.

        Args:
            map1: First remap table.
            map2: Second remap table.
        """
        self._maps = (map1, map2)

    # -- ROS plumbing --------------------------------------------------------------------

    def publish(self, v: float, omega: float) -> None:
        """Publish one Twist2DStamped command.

        Args:
            v: Forward speed in m/s.
            omega: Yaw rate in rad/s.
        """
        msg = self._twist_cls()
        msg.header.stamp = self._rospy.Time.now()
        msg.v = float(v)
        msg.omega = float(omega)
        self._pub.publish(msg)

    def spin(self) -> None:
        """Run the fixed-rate control loop until ROS shuts the node down."""
        rate = self._rospy.Rate(self.config.control_rate_hz)
        vec = np.zeros(VEC_DIM, dtype=np.float32)
        while not self._rospy.is_shutdown():
            stale = (time.monotonic() - self._last_image_time) > self.config.watchdog_timeout_s
            if self._latest_frame is None or stale or self._maps is None:
                self.publish(0.0, 0.0)
            else:
                try:
                    v, omega = self.process_frame(self._latest_frame.tobytes(), vec)
                except (ValueError, RuntimeError) as exc:
                    self._rospy.logwarn(f"inference failed, publishing zeros: {exc}")
                    v, omega = 0.0, 0.0
                self._last_action = (v, omega)
                self.publish(v, omega)
            rate.sleep()


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point used by the DTProject launcher on the robot.

    Args:
        argv: Argument list; ``sys.argv[1:]`` when ``None``.

    Returns:
        Process exit code.
    """
    import argparse  # keeps the module import cheap on the robot

    parser = argparse.ArgumentParser(description="Duckiebot RL lane-following node (robot only).")
    parser.add_argument("--onnx", required=True, help="exported policy artifact")
    parser.add_argument("--vehicle", default="duckiebot", help="vehicle hostname")
    parser.add_argument("--v-cap", type=float, default=DEFAULT_V_CAP_MPS, help="forward-speed cap")
    args = parser.parse_args(argv)
    node = LaneFollowingNode(NodeConfig(onnx_path=args.onnx, vehicle_name=args.vehicle, v_cap=args.v_cap))
    node.spin()
    return 0


if __name__ == "__main__":  # pragma: no cover - robot-only entry point
    raise SystemExit(main())
