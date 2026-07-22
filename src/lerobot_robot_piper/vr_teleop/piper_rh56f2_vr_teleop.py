#!/usr/bin/env python3
"""VR teleop bridge for Piper arm + RH56F2 hand.

This is the hardware-side bridge:

  VR frame -> ee.* target pose + hand.* target angles -> PiperRH56F2Follower

The first version accepts JSON lines so it can be connected to different VR
frontends without tying the robot driver to one specific headset SDK. The
official vr_teleop input code can later call ``VRFrameTeleop.step(...)``
directly instead of going through stdin.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import threading
import time
from http import server
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lerobot_robot_piper.config_piper_rh56f2_follower import PiperRH56F2FollowerConfig
from lerobot_robot_piper.piper_rh56f2_follower import EE_POSE_NAMES, PiperRH56F2Follower
from lerobot_robot_piper.rh56f2_hand import (
    DEFAULT_CLOSED,
    DEFAULT_OPEN,
    HAND_NAMES,
    RH56F2Hand,
    RH56F2HandConfig,
)


class CameraWebPreview:
    """Serve the latest annotated frame as a browser MJPEG stream."""

    def __init__(self, host: str, port: int):
        import cv2

        self.cv2 = cv2
        self._lock = threading.Lock()
        self._jpeg = b""
        preview = self

        class Handler(server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/":
                    body = (
                        "<html><head><title>D455 hand teleop</title></head>"
                        "<body style='margin:0;background:#111'>"
                        "<img src='/stream.mjpg' style='max-width:100vw;max-height:100vh'>"
                        "</body></html>"
                    ).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if self.path != "/stream.mjpg":
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        with preview._lock:
                            jpeg = preview._jpeg
                        if jpeg:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                            self.wfile.write(jpeg)
                            self.wfile.write(b"\r\n")
                            self.wfile.flush()
                        time.sleep(0.04)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *_args: object) -> None:
                return

        self.httpd = server.ThreadingHTTPServer((host, port), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/"

    def submit(self, frame: np.ndarray) -> None:
        ok, encoded = self.cv2.imencode(".jpg", frame, [int(self.cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            with self._lock:
                self._jpeg = encoded.tobytes()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def _clip_value(goal: float, current: float, max_delta: float | None) -> float:
    if max_delta is None:
        return goal
    return current + min(max(goal - current, -max_delta), max_delta)


@dataclass
class VRFrame:
    """One normalized VR tracking frame.

    ``wrist_xyz_m`` is the operator hand/wrist position in meters.
    ``wrist_rpy_deg`` is optional roll/pitch/yaw in degrees.
    ``finger_curls`` maps RH56F2 finger names to 0.0=open and 1.0=closed.
    ``deadman`` must be true before motion commands are sent.
    """

    wrist_xyz_m: tuple[float, float, float]
    wrist_rpy_deg: tuple[float, float, float] | None
    finger_curls: dict[str, float]
    deadman: bool
    landmarks: list[float] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "VRFrame":
        wrist = data.get("wrist_xyz_m")
        if not isinstance(wrist, list | tuple) or len(wrist) != 3:
            raise ValueError("frame needs wrist_xyz_m: [x, y, z]")

        raw_rpy = data.get("wrist_rpy_deg")
        rpy = None
        if raw_rpy is not None:
            if not isinstance(raw_rpy, list | tuple) or len(raw_rpy) != 3:
                raise ValueError("wrist_rpy_deg must be [rx, ry, rz]")
            rpy = tuple(float(v) for v in raw_rpy)

        curls_raw = data.get("finger_curls", {})
        if not isinstance(curls_raw, dict):
            raise ValueError("finger_curls must be an object")

        return cls(
            wrist_xyz_m=tuple(float(v) for v in wrist),
            wrist_rpy_deg=rpy,
            finger_curls={str(k): float(v) for k, v in curls_raw.items()},
            deadman=bool(data.get("deadman", False)),
            landmarks=(
                [float(v) for v in data["landmarks"]]
                if isinstance(data.get("landmarks"), list)
                else None
            ),
        )


def _parse_csv_line(message: str, prefix: str, count: int) -> list[float] | None:
    prefix = prefix.lower()
    for line in message.splitlines():
        if not line.strip().lower().startswith(prefix):
            continue
        _, _, rest = line.partition(":")
        values: list[float] = []
        for item in rest.split(","):
            try:
                values.append(float(item.strip()))
            except ValueError:
                break
        if len(values) == count:
            return values
    return None


def _parse_right_wrist(message: str) -> list[float] | None:
    return _parse_csv_line(message, "right wrist", 7)


def _parse_right_landmarks(message: str) -> list[float] | None:
    return _parse_csv_line(message, "right landmarks", 63)


def _distance(points: list[float], first: int, second: int) -> float:
    first_offset = first * 3
    second_offset = second * 3
    return math.sqrt(
        sum(
            (points[first_offset + axis] - points[second_offset + axis]) ** 2
            for axis in range(3)
        )
    )


def _point(points: list[float], index: int) -> np.ndarray:
    offset = index * 3
    return np.asarray(points[offset : offset + 3], dtype=float)


def _joint_angle(points: list[float], first: int, vertex: int, second: int) -> float:
    """Return the 3-D angle at a hand landmark in radians."""
    a = _point(points, first) - _point(points, vertex)
    b = _point(points, second) - _point(points, vertex)
    denominator = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-8)
    cosine = float(np.dot(a, b) / denominator)
    return math.acos(float(np.clip(cosine, -1.0, 1.0)))


def landmarks_to_simple_curls(points: list[float]) -> dict[str, float]:
    """Convert 21 VR landmarks into safe 0=open, 1=closed curl values."""
    if len(points) != 63:
        raise ValueError("expected 63 hand landmark values")

    finger_indices = {
        "index": (5, 6, 7, 8),
        "middle": (9, 10, 11, 12),
        "ring": (13, 14, 15, 16),
        "little": (17, 18, 19, 20),
    }
    curls: dict[str, float] = {}
    for name, (mcp, pip, dip, tip) in finger_indices.items():
        pip_bend = (math.pi - _joint_angle(points, mcp, pip, dip)) / (math.pi - 0.55)
        dip_bend = (math.pi - _joint_angle(points, pip, dip, tip)) / (math.pi - 0.55)
        angle_curl = 0.5 * (pip_bend + dip_bend)

        # Use the finger shape relative to the wrist and palm size. This is
        # invariant to image translation and hand scale, and gives a larger
        # signal when the fingertip folds toward the palm in a fist.
        mcp_from_wrist = _distance(points, mcp, 0) / max(_distance(points, 0, 9), 1e-6)
        tip_from_wrist = _distance(points, tip, 0) / max(_distance(points, 0, 9), 1e-6)
        fold_curl = (mcp_from_wrist - tip_from_wrist + 0.10) / 0.45

        # Blend relative folding and joint angles, then add gain so a real
        # fist reaches the RH56F2 closed range instead of stopping halfway.
        curls[name] = float(np.clip(1.35 * (0.65 * fold_curl + 0.35 * angle_curl), 0.0, 1.0))

    # Normalize thumb measurements by palm size.  These are distances between
    # landmarks, so moving or rotating the whole hand does not change them.
    palm_scale = max(_distance(points, 0, 9), 1e-6)
    thumb_to_wrist = _distance(points, 4, 0) / palm_scale
    thumb_to_index_mcp = _distance(points, 4, 5) / palm_scale
    thumb_to_middle_mcp = _distance(points, 4, 9) / palm_scale
    thumb_bend_angle = (math.pi - _joint_angle(points, 2, 3, 4)) / (math.pi - 0.65)

    curls["thumb_bend"] = float(np.clip(
        0.65 * ((1.05 - thumb_to_wrist) / 0.55) + 0.35 * thumb_bend_angle,
        0.0,
        1.0,
    ))
    # Use both index- and middle-MCP distances.  A thumb across the palm is
    # close to both; an abducted thumb is far from both.  This is much less
    # sensitive to wrist rotation than a screen-space left/right test.
    thumb_open_fraction = np.clip(
        0.55 * ((thumb_to_index_mcp - 0.28) / 0.62)
        + 0.45 * ((thumb_to_middle_mcp - 0.36) / 0.62),
        0.0,
        1.0,
    )
    curls["thumb_swing"] = float(1.0 - thumb_open_fraction)
    return curls


class Quest3UDPInput:
    """Read the upstream Quest 3 Hand Tracking Streamer UDP format."""

    def __init__(self, port: int):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(("0.0.0.0", port))
        self.socket.setblocking(False)
        self.port = port
        self.received_packets = 0
        self.valid_frames = 0
        self.invalid_packets = 0

    def close(self) -> None:
        self.socket.close()

    def poll(self) -> VRFrame | None:
        latest: bytes | None = None
        while True:
            try:
                latest, _ = self.socket.recvfrom(65536)
            except BlockingIOError:
                break
        if latest is None:
            return None

        self.received_packets += 1

        message = latest.decode("utf-8", errors="ignore")
        wrist = _parse_right_wrist(message)
        landmarks = _parse_right_landmarks(message)
        if wrist is None or landmarks is None:
            self.invalid_packets += 1
            return None

        # Quest/Unity frame -> Piper frame. Keep orientation disabled until
        # the real wrist-axis calibration is confirmed on the hardware.
        wrist_xyz_m = (-wrist[0], -wrist[2], wrist[1])
        self.valid_frames += 1
        return VRFrame(
            wrist_xyz_m=wrist_xyz_m,
            wrist_rpy_deg=None,
            finger_curls=landmarks_to_simple_curls(landmarks),
            deadman=True,
            landmarks=landmarks,
        )


class RealSenseHandInput:
    """Read D455 color frames and convert MediaPipe landmarks to VRFrame.

    The first D455 phase deliberately uses only hand landmarks.  Depth and
    wrist pose will be added later for arm teleoperation; RH56F2 hand control
    only needs the normalized finger curl values produced here.
    """

    def __init__(
        self,
        model_path: Path,
        serial: str,
        width: int,
        height: int,
        fps: int,
        show_camera: bool,
        preview_port: int,
        min_detection: float,
        min_tracking: float,
        min_presence: float,
        min_score: float,
    ):
        try:
            import cv2
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError(
                "D455 input needs pyrealsense2, mediapipe, and opencv-python. "
                "Install pyrealsense2 in the same Python environment used to run teleop."
            ) from exc

        if not model_path.exists():
            raise RuntimeError(
                f"MediaPipe model not found: {model_path}. "
                "Use --realsense-model to pass gesture_recognizer.task."
            )

        self.cv2 = cv2
        self.mp = mp
        self.show_camera = show_camera
        self.window_name = "D455 hand teleop"
        self.preview = CameraWebPreview("127.0.0.1", preview_port) if show_camera else None
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self._pipeline_started = False
        if serial:
            self.config.enable_device(serial)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        options = vision.GestureRecognizerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=min_detection,
            min_hand_presence_confidence=min_presence,
            min_tracking_confidence=min_tracking,
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(options)
        self.last_timestamp_ms = 0
        self.min_score = min_score
        self.received_frames = 0
        self.valid_frames = 0
        self.no_hand_frames = 0
        self.port = None
        self._smoothed_curls: dict[str, float] | None = None
        self.curl_smoothing = 0.35
        self.last_raw_curls: dict[str, float] | None = None
        self.last_smoothed_curls: dict[str, float] | None = None

    def close(self) -> None:
        self.recognizer.close()
        if self._pipeline_started:
            self.pipeline.stop()
        if self.preview is not None:
            self.preview.close()

    def start(self) -> None:
        self.pipeline.start(self.config)
        self._pipeline_started = True

    def poll(self) -> VRFrame | None:
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None

        self.received_frames += 1
        image_bgr = np.asanyarray(color_frame.get_data())
        if self.preview is not None:
            self.preview.submit(image_bgr)
        image_rgb = self.cv2.cvtColor(image_bgr, self.cv2.COLOR_BGR2RGB)
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=image_rgb)
        timestamp_ms = int(time.monotonic() * 1000)
        if timestamp_ms <= self.last_timestamp_ms:
            timestamp_ms = self.last_timestamp_ms + 1
        self.last_timestamp_ms = timestamp_ms
        result = self.recognizer.recognize_for_video(mp_image, timestamp_ms)

        if not result.hand_landmarks:
            self.no_hand_frames += 1
            self._smoothed_curls = None
            self.last_raw_curls = None
            self.last_smoothed_curls = None
            return None

        hand_index = 0
        if result.handedness:
            for index, handedness in enumerate(result.handedness):
                if handedness and handedness[0].category_name.lower() == "right":
                    hand_index = index
                    break
        hand = result.hand_landmarks[hand_index]
        landmarks = [float(value) for landmark in hand for value in (landmark.x, landmark.y, landmark.z)]
        self.valid_frames += 1
        if self.show_camera:
            h, w = image_bgr.shape[:2]
            for start, end in (
                (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
                (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
            ):
                p1 = (int(hand[start].x * w), int(hand[start].y * h))
                p2 = (int(hand[end].x * w), int(hand[end].y * h))
                self.cv2.line(image_bgr, p1, p2, (0, 220, 255), 2)
            for landmark in hand:
                self.cv2.circle(image_bgr, (int(landmark.x * w), int(landmark.y * h)), 4, (0, 80, 255), -1)
            label = "hand detected"
            if result.gestures and result.gestures[0]:
                category = result.gestures[0][0]
                if float(category.score) >= self.min_score:
                    label = f"{category.category_name} {float(category.score):.2f}"
            self.cv2.putText(image_bgr, label, (20, 35), self.cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if self.preview is not None:
                self.preview.submit(image_bgr)

        # Wrist coordinates are normalized image coordinates for now. They
        # are harmless in hand-only mode and reserved for later arm mapping.
        wrist = hand[0]
        raw_curls = landmarks_to_simple_curls(landmarks)
        self.last_raw_curls = raw_curls
        if self._smoothed_curls is None:
            smoothed_curls = raw_curls
        else:
            smoothed_curls = {
                name: self.curl_smoothing * raw_curls[name]
                + (1.0 - self.curl_smoothing) * self._smoothed_curls[name]
                for name in raw_curls
            }
        self._smoothed_curls = smoothed_curls
        self.last_smoothed_curls = smoothed_curls

        return VRFrame(
            wrist_xyz_m=(float(wrist.x), float(wrist.y), float(wrist.z)),
            wrist_rpy_deg=None,
            finger_curls=smoothed_curls,
            deadman=True,
            landmarks=landmarks,
        )


class ArmPoseMapper:
    """Map relative VR wrist motion to Piper end-effector pose commands."""

    def __init__(self, xyz_scale: float, rpy_scale: float):
        self.xyz_scale = xyz_scale
        self.rpy_scale = rpy_scale
        self._vr_anchor: tuple[float, float, float] | None = None
        self._rpy_anchor: tuple[float, float, float] | None = None
        self._ee_anchor: dict[str, float] | None = None

    def reset(self, frame: VRFrame, current_ee: dict[str, float]) -> None:
        self._vr_anchor = frame.wrist_xyz_m
        self._rpy_anchor = frame.wrist_rpy_deg
        self._ee_anchor = dict(current_ee)

    def map(self, frame: VRFrame, current_ee: dict[str, float]) -> dict[str, float]:
        if self._vr_anchor is None or self._ee_anchor is None:
            self.reset(frame, current_ee)

        assert self._vr_anchor is not None
        assert self._ee_anchor is not None

        dx = (frame.wrist_xyz_m[0] - self._vr_anchor[0]) * 1000.0 * self.xyz_scale
        dy = (frame.wrist_xyz_m[1] - self._vr_anchor[1]) * 1000.0 * self.xyz_scale
        dz = (frame.wrist_xyz_m[2] - self._vr_anchor[2]) * 1000.0 * self.xyz_scale

        target = {
            "ee.x": self._ee_anchor["ee.x"] + dx,
            "ee.y": self._ee_anchor["ee.y"] + dy,
            "ee.z": self._ee_anchor["ee.z"] + dz,
            "ee.rx": self._ee_anchor["ee.rx"],
            "ee.ry": self._ee_anchor["ee.ry"],
            "ee.rz": self._ee_anchor["ee.rz"],
        }

        if frame.wrist_rpy_deg is not None and self._rpy_anchor is not None:
            target["ee.rx"] += (frame.wrist_rpy_deg[0] - self._rpy_anchor[0]) * self.rpy_scale
            target["ee.ry"] += (frame.wrist_rpy_deg[1] - self._rpy_anchor[1]) * self.rpy_scale
            target["ee.rz"] += (frame.wrist_rpy_deg[2] - self._rpy_anchor[2]) * self.rpy_scale

        return target


class RH56F2SimpleRetargeter:
    """Simple curl-to-register mapping before full AnyDexRetarget integration."""

    def __init__(self, thumb_swing_closed: float = 500.0):
        # RH56F2 SDK right-hand calibration from the supplied controller:
        # fingers 1740->900, thumb bend 1450->1100, thumb swing 1750->500.
        self.open_pose = {
            "little": 1740.0,
            "ring": 1740.0,
            "middle": 1740.0,
            "index": 1740.0,
            "thumb_bend": 1450.0,
            "thumb_swing": 1750.0,
        }
        self.closed_pose = {
            "little": 900.0,
            "ring": 900.0,
            "middle": 900.0,
            "index": 900.0,
            "thumb_bend": 1100.0,
            "thumb_swing": 500.0,
        }
        self.thumb_swing_closed = float(thumb_swing_closed)

    def map(self, curls: dict[str, float]) -> dict[str, float]:
        action: dict[str, float] = {}
        for name in HAND_NAMES:
            curl = float(curls.get(name, curls.get("all", 0.0)))
            curl = min(max(curl, 0.0), 1.0)
            opened = self.open_pose[name]
            closed = self.thumb_swing_closed if name == "thumb_swing" else self.closed_pose[name]
            action[f"hand.{name}.pos"] = opened + curl * (closed - opened)
        return action


def parse_qpos_groups(value: str) -> list[list[int]]:
    """Parse five RH56F2 finger groups, e.g. ``0,1;2,3;4,5;6,7;8,9``."""
    groups: list[list[int]] = []
    for group in value.split(";"):
        indices = [int(item.strip()) for item in group.split(",") if item.strip()]
        if not indices:
            raise ValueError("each AnyDex qpos group must contain an index")
        groups.append(indices)
    if len(groups) != 5:
        raise ValueError("AnyDex qpos groups must be thumb,index,middle,ring,little")
    return groups


class AnyDexRH56F2Retargeter:
    """Use AnyDexRetarget as a hand-pose front end for RH56F2.

    AnyDexRetarget targets a modeled hand, not RH56F2. The configured qpos
    groups therefore become normalized finger curls before RH56F2 register
    angles are generated. The groups must be calibrated for the selected
    AnyDex robot model.
    """

    def __init__(self, config_path: Path, anydex_root: Path, qpos_groups: str):
        if str(anydex_root) not in sys.path:
            sys.path.insert(0, str(anydex_root))
        try:
            from anydexretarget import Retargeter
        except ImportError as exc:
            raise RuntimeError(
                "AnyDexRetarget is unavailable. Install its dependencies and "
                "pass --anydex-root to its checkout."
            ) from exc

        self.retargeter = Retargeter.from_yaml(str(config_path), "right")
        self.groups = parse_qpos_groups(qpos_groups)
        lower = np.asarray(self.retargeter.optimizer.opt_lower_bounds, dtype=float)
        upper = np.asarray(self.retargeter.optimizer.opt_upper_bounds, dtype=float)
        self.lower = lower
        self.span = np.maximum(upper - lower, 1e-6)

    def map_landmarks(self, landmarks: list[float]) -> dict[str, float]:
        if len(landmarks) != 63:
            raise ValueError("AnyDex input needs 63 hand landmark values")
        qpos = np.asarray(
            self.retargeter.retarget(np.asarray(landmarks, dtype=float).reshape(21, 3)),
            dtype=float,
        )
        curls: dict[str, float] = {}
        names = ["thumb_bend", "index", "middle", "ring", "little"]
        for name, group in zip(names, self.groups, strict=True):
            if any(index >= len(qpos) for index in group):
                raise ValueError(f"AnyDex qpos group {group} exceeds output size {len(qpos)}")
            normalized = [(qpos[index] - self.lower[index]) / self.span[index] for index in group]
            curls[name] = float(np.clip(np.mean(normalized), 0.0, 1.0))
        curls["thumb_swing"] = 0.0
        return RH56F2SimpleRetargeter().map(curls)


class DryRunRobot:
    """Small stand-in used to verify mapping before connecting real hardware."""

    def __init__(self, start_ee: dict[str, float]):
        self.obs = dict(start_ee)
        self.obs.update({f"hand.{name}.pos": DEFAULT_OPEN[name] for name in HAND_NAMES})

    def connect(self) -> None:
        pass

    @property
    def is_connected(self) -> bool:
        return True

    def get_observation(self) -> dict[str, float]:
        return dict(self.obs)

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        self.obs.update(action)
        print(json.dumps(action, sort_keys=True))
        return action

    def disconnect(self) -> None:
        pass


class HandOnlyRobot:
    """Real RH56F2 hand with a fake arm pose for low-risk teleop checks."""

    def __init__(
        self,
        start_ee: dict[str, float],
        hand_port: str,
        hand_id: int,
        hand_speed: int,
        hand_force: int,
        max_hand_delta: float | None,
    ):
        self.obs = dict(start_ee)
        self.max_hand_delta = max_hand_delta
        self.hand = RH56F2Hand(
            RH56F2HandConfig(
                port=hand_port,
                hand_id=hand_id,
                speed=hand_speed,
                force=hand_force,
            )
        )

    def connect(self) -> None:
        self.hand.connect()

    @property
    def is_connected(self) -> bool:
        return self.hand.is_connected

    def get_observation(self) -> dict[str, float]:
        obs = dict(self.obs)
        hand_pos = self.hand.read_positions("angleAct")
        obs.update({f"hand.{name}.pos": value for name, value in hand_pos.items()})
        hand_force = self.hand.read_positions("forceAct")
        obs.update({f"hand.{name}.force": value for name, value in hand_force.items()})
        return obs

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        self.obs.update({name: float(action[name]) for name in EE_POSE_NAMES if name in action})
        hand_action = {
            name: float(action[f"hand.{name}.pos"])
            for name in HAND_NAMES
            if f"hand.{name}.pos" in action
        }
        if not hand_action:
            return {}

        current = self.hand.read_positions("angleAct")
        clipped = {
            name: _clip_value(value, current[name], self.max_hand_delta)
            for name, value in hand_action.items()
        }
        self.hand.set_angles(clipped)
        return {f"hand.{name}.pos": value for name, value in clipped.items()}

    def disconnect(self) -> None:
        self.hand.disconnect()


class VRFrameTeleop:
    def __init__(self, robot: object, xyz_scale: float, rpy_scale: float, hand_mapper: object):
        self.robot = robot
        self.arm_mapper = ArmPoseMapper(xyz_scale=xyz_scale, rpy_scale=rpy_scale)
        self.hand_mapper = hand_mapper
        self._last_deadman = False

    def _hand_action(self, frame: VRFrame) -> dict[str, float]:
        if isinstance(self.hand_mapper, AnyDexRH56F2Retargeter):
            if frame.landmarks is None:
                return {}
            return self.hand_mapper.map_landmarks(frame.landmarks)
        return self.hand_mapper.map(frame.finger_curls)

    def step(self, frame: VRFrame) -> dict[str, float]:
        obs = self.robot.get_observation()
        current_ee = {name: float(obs[name]) for name in EE_POSE_NAMES}

        if not frame.deadman:
            self._last_deadman = False
            self.arm_mapper.reset(frame, current_ee)
            return {}

        if not self._last_deadman:
            self.arm_mapper.reset(frame, current_ee)
        self._last_deadman = True

        action = self.arm_mapper.map(frame, current_ee)
        action.update(self._hand_action(frame))
        return self.robot.send_action(action)


def iter_json_frames(stream: Iterable[str]) -> Iterable[VRFrame]:
    for line in stream:
        line = line.strip()
        if not line:
            continue
        yield VRFrame.from_dict(json.loads(line))


def build_robot(args: argparse.Namespace) -> object:
    start_ee = {
        "ee.x": args.start_x,
        "ee.y": args.start_y,
        "ee.z": args.start_z,
        "ee.rx": args.start_rx,
        "ee.ry": args.start_ry,
        "ee.rz": args.start_rz,
    }

    if not args.connect:
        return DryRunRobot(start_ee)

    if args.hand_only:
        return HandOnlyRobot(
            start_ee=start_ee,
            hand_port=args.hand_port,
            hand_id=args.hand_id,
            hand_speed=args.hand_speed,
            hand_force=args.hand_force,
            max_hand_delta=args.max_hand_delta,
        )

    return PiperRH56F2Follower(
        PiperRH56F2FollowerConfig(
            can_port=args.can,
            speed_rate=args.speed,
            max_ee_delta_mm=args.max_ee_delta_mm,
            max_ee_delta_deg=args.max_ee_delta_deg,
            hand_port=args.hand_port,
            hand_id=args.hand_id,
            hand_speed=args.hand_speed,
            hand_force=args.hand_force,
            max_hand_delta=args.max_hand_delta,
        )
    )


def build_input(args: argparse.Namespace) -> Quest3UDPInput | RealSenseHandInput | None:
    if args.input_source == "quest3":
        return Quest3UDPInput(args.port)
    if args.input_source == "realsense":
        return RealSenseHandInput(
            model_path=args.realsense_model,
            serial=args.realsense_serial,
            width=args.realsense_width,
            height=args.realsense_height,
            fps=args.realsense_fps,
            show_camera=args.show_camera,
            preview_port=args.preview_port,
            min_detection=args.min_detection,
            min_tracking=args.min_tracking,
            min_presence=args.min_presence,
            min_score=args.min_score,
        )
    return None


def build_hand_mapper(args: argparse.Namespace) -> object:
    if args.hand_mode == "simple":
        return RH56F2SimpleRetargeter(thumb_swing_closed=args.thumb_swing_closed)
    if args.hand_config is None:
        raise ValueError("--hand-config is required with --hand-mode anydex")
    return AnyDexRH56F2Retargeter(
        config_path=args.hand_config,
        anydex_root=args.anydex_root,
        qpos_groups=args.anydex_qpos_groups,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connect", action="store_true", help="Connect real Piper + RH56F2 hardware.")
    parser.add_argument("--input-source", choices=["stdin", "quest3", "realsense"], default="stdin")
    parser.add_argument("--port", type=int, default=9000, help="Quest 3 UDP input port.")
    parser.add_argument(
        "--realsense-model",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "rock_paper_scissors" / "gesture_recognizer.task",
        help="MediaPipe hand model used with --input-source realsense.",
    )
    parser.add_argument("--realsense-serial", default="", help="Optional D455 serial number.")
    parser.add_argument("--realsense-width", type=int, default=640)
    parser.add_argument("--realsense-height", type=int, default=480)
    parser.add_argument("--realsense-fps", type=int, default=30)
    parser.add_argument("--show-camera", action="store_true", help="Show D455 image and detected hand landmarks.")
    parser.add_argument(
        "--print-curls",
        action="store_true",
        help="Print normalized finger curls and RH56F2 target angles while running.",
    )
    parser.add_argument("--preview-port", type=int, default=8765, help="Local browser preview port.")
    parser.add_argument("--min-detection", type=float, default=0.7)
    parser.add_argument("--min-tracking", type=float, default=0.5)
    parser.add_argument("--min-presence", type=float, default=0.5)
    parser.add_argument("--min-score", type=float, default=0.5)
    parser.add_argument("--hand-mode", choices=["simple", "anydex"], default="simple")
    parser.add_argument("--hand-config", type=Path)
    parser.add_argument("--anydex-root", type=Path, default=Path("third_party/AnyDexRetarget"))
    parser.add_argument(
        "--anydex-qpos-groups",
        default="0,1;2,3;4,5;6,7;8,9",
        help="Five qpos groups: thumb,index,middle,ring,little.",
    )
    parser.add_argument("--hand-only", action="store_true", help="Only connect RH56F2; keep Piper disabled.")
    parser.add_argument("--can", default="can0")
    parser.add_argument("--hand-port", default="/dev/ttyUSB0")
    parser.add_argument("--hand-id", type=int, default=1)
    parser.add_argument("--hand-speed", type=int, default=800)
    parser.add_argument("--hand-force", type=int, default=1500)
    parser.add_argument(
        "--thumb-swing-closed",
        type=float,
        default=500.0,
        help="RH56F2 thumb side-swing target when fully curled; calibrate on hardware.",
    )
    parser.add_argument("--speed", type=int, default=20)
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument("--xyz-scale", type=float, default=0.6)
    parser.add_argument("--rpy-scale", type=float, default=0.5)
    parser.add_argument("--max-ee-delta-mm", type=float, default=10.0)
    parser.add_argument("--max-ee-delta-deg", type=float, default=5.0)
    parser.add_argument("--max-hand-delta", type=float, default=80.0)
    parser.add_argument("--start-x", type=float, default=300.0)
    parser.add_argument("--start-y", type=float, default=0.0)
    parser.add_argument("--start-z", type=float, default=250.0)
    parser.add_argument("--start-rx", type=float, default=0.0)
    parser.add_argument("--start-ry", type=float, default=0.0)
    parser.add_argument("--start-rz", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    robot = build_robot(args)
    hand_mapper = build_hand_mapper(args)
    teleop = VRFrameTeleop(
        robot,
        xyz_scale=args.xyz_scale,
        rpy_scale=args.rpy_scale,
        hand_mapper=hand_mapper,
    )
    interval_s = 1.0 / args.rate_hz
    print(f"Initializing input source: {args.input_source} (show_camera={args.show_camera})", flush=True)
    vr_input: Quest3UDPInput | RealSenseHandInput | None = None
    last_frame_time = time.monotonic()
    last_status_time = time.monotonic()
    last_curls_time = time.monotonic()

    try:
        vr_input = build_input(args)
        print("Input source initialized.", flush=True)
        robot.connect()
        if vr_input is None:
            print("VR teleop ready. Reading normalized JSON frames from stdin.")
            for frame in iter_json_frames(sys.stdin):
                started = time.time()
                teleop.step(frame)
                elapsed = time.time() - started
                if elapsed < interval_s:
                    time.sleep(interval_s - elapsed)
        else:
            if isinstance(vr_input, RealSenseHandInput):
                vr_input.start()
                print(
                    "D455 hand teleop ready. Show your right hand to the camera. "
                    "Waiting for hand landmarks...",
                    flush=True,
                )
                if vr_input.preview is not None:
                    print(f"Camera preview: {vr_input.preview.url}", flush=True)
            else:
                print(
                    f"VR teleop ready. Listening for Quest 3 UDP on 0.0.0.0:{vr_input.port}. "
                    "Waiting for valid hand frames...",
                    flush=True,
                )
            while True:
                started = time.time()
                frame = vr_input.poll()
                if frame is not None:
                    last_frame_time = time.monotonic()
                    action = teleop.step(frame)
                    if args.print_curls and time.monotonic() - last_curls_time >= 0.5:
                        print(
                            "curls=" + json.dumps(frame.finger_curls, sort_keys=True)
                            + " targets=" + json.dumps(action, sort_keys=True),
                            flush=True,
                        )
                        last_curls_time = time.monotonic()
                elif time.monotonic() - last_frame_time > 0.25:
                    teleop.step(
                        VRFrame(
                            wrist_xyz_m=(0.0, 0.0, 0.0),
                            wrist_rpy_deg=None,
                            finger_curls={},
                            deadman=False,
                        )
                    )
                if time.monotonic() - last_status_time >= 2.0:
                    if isinstance(vr_input, RealSenseHandInput):
                        print(
                            "D455 status: "
                            f"frames={vr_input.received_frames} "
                            f"valid_hand={vr_input.valid_frames} "
                            f"no_hand={vr_input.no_hand_frames}",
                            flush=True,
                        )
                    else:
                        print(
                            "Quest 3 status: "
                            f"packets={vr_input.received_packets} "
                            f"valid={vr_input.valid_frames} "
                            f"invalid={vr_input.invalid_packets}",
                            flush=True,
                        )
                    last_status_time = time.monotonic()
                elapsed = time.time() - started
                if elapsed < interval_s:
                    time.sleep(interval_s - elapsed)
    except KeyboardInterrupt:
        print("\nStopping VR teleoperation.")
    finally:
        if vr_input is not None:
            vr_input.close()
        if getattr(robot, "is_connected", False):
            robot.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
