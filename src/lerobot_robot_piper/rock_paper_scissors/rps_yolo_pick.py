#!/usr/bin/env python3
"""D435i RPS flow with D405 YOLO target pick."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


PACKAGE_PARENT = Path(__file__).resolve().parents[2]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from lerobot_robot_piper.claw_machine.lerobot_claw import (  # noqa: E402
    ClawMachineController,
    ClawMachineTaskConfig,
    DEFAULT_START_JOINTS,
    DEFAULT_START_POSE,
    fmt_joints,
    parse_joint_degrees,
    pose_from_values,
)
from lerobot_robot_piper.rock_paper_scissors.ball_tactile_classifier.claw_integration import (  # noqa: E402
    BallClassifierConfig,
)
from lerobot_robot_piper.config_piper_rh56f2_follower import (  # noqa: E402
    PiperRH56F2FollowerConfig,
)
from lerobot_robot_piper.piper_rh56f2_follower import PiperRH56F2Follower  # noqa: E402
from lerobot_robot_piper.rock_paper_scissors.rh56f2_rps_demo import (  # noqa: E402
    D455ColorCamera,
    HandGestureRecognizer,
)
from lerobot_robot_piper.rh56f2_hand import DEFAULT_CLOSED, DEFAULT_OPEN  # noqa: E402
from lerobot_robot_piper.rock_paper_scissors.rps_prize_controller import (  # noqa: E402
    rotate_d405_box_ccw_90,
    rotate_d405_ccw_90,
    rotate_d405_point_ccw_90,
)
from lerobot_robot_piper.rock_paper_scissors.solve_eye_hand_calibration import (  # noqa: E402
    make_transform,
    rpy_to_matrix,
)
from lerobot_robot_piper.rock_paper_scissors.test_yolo_d405_ball import depth_at_box  # noqa: E402
from lerobot_robot_piper.rock_paper_scissors.eye_to_hand_calibration.homography_tabletop_runtime import (  # noqa: E402
    PixelDetection,
    apply_homography,
    best_ball_pixel,
    draw_preview,
    load_homography,
    median_detection,
    update_stable,
)
from lerobot_robot_piper.rock_paper_scissors.eye_to_hand_calibration.planar_grasp_geometry import (  # noqa: E402
    radial_flange_xy,
    solve_planar_joint_target,
)


GESTURES = ("rock", "paper", "scissors")
BEATS = {"rock": "scissors", "paper": "rock", "scissors": "paper"}
DISPLAY = {"rock": "Rock", "paper": "Paper", "scissors": "Scissors"}
CAMERA_GESTURES = {"Rock": "rock", "Paper": "paper", "Scissors": "scissors"}
DEFAULT_D435I_SERIAL = "261722071542"
DEFAULT_FIXED_D405_SERIAL = "315122271151"
DEFAULT_D405_SERIAL = DEFAULT_FIXED_D405_SERIAL
DEFAULT_GESTURE_MODEL = Path(__file__).with_name("gesture_recognizer.task")
DEFAULT_CALIBRATION = Path(__file__).with_name("eye_hand_calibration_yolo11x_clean2.json")
DEFAULT_HOMOGRAPHY_CALIBRATION = Path(__file__).with_name("eye_to_hand_calibration") / "homography_position_calibration.json"
DEFAULT_BALL_MODEL = Path(__file__).with_name("yolo11x.pt")
DEFAULT_HOMOGRAPHY_BALL_MODEL = Path(__file__).with_name("yolo26n.pt")
BALL_CLASSIFIER_DIR = Path(__file__).with_name("ball_tactile_classifier")
DEFAULT_TACTILE_MODEL = BALL_CLASSIFIER_DIR / "model.json"
DEFAULT_TACTILE_OUTPUT = BALL_CLASSIFIER_DIR / "live_predictions.csv"
DEFAULT_TACTILE_REFERENCE_SAMPLES = BALL_CLASSIFIER_DIR / "samples.csv"
FIXED_GRAB_XYZ_M = (0.30455, 0.02575, 0.25000)
DEFAULT_DROP_POSE = [-185.449, 434.149, 135.053, 68.125, 72.173, 172.599]
DEFAULT_APPROACH_HEIGHT_M = 0.08
DEFAULT_RPS_JOINTS = [-1.391, 53.150, -56.325, 2.544, -17.309, 101.959]

SCISSORS = dict(DEFAULT_CLOSED)
SCISSORS.update(
    {
        "little": 900,
        "ring": 900,
        "middle": 1720,
        "index": 1720,
        "thumb_bend": 1130,
        "thumb_swing": 1700,
    }
)
RPS_HAND_POSES = {
    "rock": dict(DEFAULT_CLOSED),
    "paper": dict(DEFAULT_OPEN),
    "scissors": SCISSORS,
}


@dataclass(frozen=True)
class YoloBallObservation:
    confidence: float
    box: tuple[int, int, int, int]
    pixel: tuple[int, int]
    depth_m: float
    camera_xyz_m: tuple[float, float, float]


@dataclass(frozen=True)
class PickTarget:
    base_xyz_m: tuple[float, float, float]
    camera_xyz_m: tuple[float, float, float]
    confidence: float
    pixel: tuple[int, int]
    depth_m: float


def parse_roi(value: str) -> tuple[int, int, int, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("expected x0,y0,x1,y1")
    try:
        x0, y0, x1, y1 = [int(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ROI values must be integers") from exc
    if x1 <= x0 or y1 <= y0:
        raise argparse.ArgumentTypeError("ROI must satisfy x1>x0 and y1>y0")
    return x0, y0, x1, y1


def parse_xyz(value: str) -> list[float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected X,Y,Z in metres")
    try:
        return [float(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("X,Y,Z must be numbers") from exc


def parse_pose_values(value: str) -> list[float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 6:
        raise argparse.ArgumentTypeError("expected X,Y,Z,RX,RY,RZ in mm/deg")
    try:
        return [float(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("pose values must be numbers") from exc


def piper_pose_to_base_tool(current_pose: dict[str, float]) -> np.ndarray:
    xyz_m = np.array(
        [
            current_pose["ee.x"] / 1000.0,
            current_pose["ee.y"] / 1000.0,
            current_pose["ee.z"] / 1000.0,
        ],
        dtype=float,
    )
    rotation = rpy_to_matrix(current_pose["ee.rx"], current_pose["ee.ry"], current_pose["ee.rz"])
    return make_transform(rotation, xyz_m)


def median_ball_observation(observations: list[YoloBallObservation]) -> YoloBallObservation | None:
    if not observations:
        return None
    pixels = np.median([item.pixel for item in observations], axis=0)
    camera_xyz = np.median([item.camera_xyz_m for item in observations], axis=0)
    depths = np.median([item.depth_m for item in observations])
    confidences = np.median([item.confidence for item in observations])
    widths = [item.box[2] - item.box[0] for item in observations]
    heights = [item.box[3] - item.box[1] for item in observations]
    center_x = int(round(pixels[0]))
    center_y = int(round(pixels[1]))
    width = int(round(float(np.median(widths))))
    height = int(round(float(np.median(heights))))
    return YoloBallObservation(
        confidence=float(confidences),
        box=(
            center_x - width // 2,
            center_y - height // 2,
            center_x + width // 2,
            center_y + height // 2,
        ),
        pixel=(center_x, center_y),
        depth_m=float(depths),
        camera_xyz_m=tuple(float(value) for value in camera_xyz),
    )


class D405YoloTargeter:
    def __init__(self, args: argparse.Namespace):
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError("pyrealsense2 is required for D405 targeting") from exc

        self.rs = rs
        self.args = args
        self.tool_camera = np.asarray(json.loads(args.calibration.read_text())["T_tool_camera"], dtype=float)
        self.model = YOLO(str(args.ball_model))
        self.ball_class_ids = {
            class_id
            for class_id, name in self.model.names.items()
            if str(name).lower() == "sports ball"
        }
        if not self.ball_class_ids:
            raise RuntimeError(f"YOLO model has no sports ball class: {self.model.names}")

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        if args.ball_serial:
            self.config.enable_device(args.ball_serial)
        self.config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
        self.config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
        self.align = rs.align(rs.stream.color)
        self.depth_scale = 0.001
        self.started = False

    def start(self) -> None:
        if self.started:
            return
        profile = self.pipeline.start(self.config)
        self.depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
        self.started = True

    def stop(self) -> None:
        if self.started:
            self.pipeline.stop()
            self.started = False

    def warmup(self, duration_s: float) -> None:
        if duration_s <= 0:
            return
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            frames = self.pipeline.wait_for_frames()
            self.align.process(frames)

    def prime(self) -> None:
        color, _ = self.read_observation()
        display = rotate_d405_ccw_90(color)
        if not self.args.no_window:
            cv2.imshow("D405 YOLO target - q to abort", display)
            cv2.waitKey(1)

    def _prediction_kwargs(self) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "conf": self.args.ball_conf,
            "imgsz": self.args.ball_imgsz,
            "verbose": False,
        }
        if self.args.ball_device:
            kwargs["device"] = self.args.ball_device
        return kwargs

    def _candidate_score(self, observation: YoloBallObservation) -> float:
        if self.args.ball_select == "confidence":
            return observation.confidence
        image_center = (self.args.width / 2.0, self.args.height / 2.0)
        distance = float(np.hypot(observation.pixel[0] - image_center[0], observation.pixel[1] - image_center[1]))
        if self.args.ball_select == "center":
            return -distance
        if self.args.ball_select == "near":
            return -observation.depth_m
        raise RuntimeError(f"unknown ball selection mode: {self.args.ball_select}")

    def read_observation(self) -> tuple[np.ndarray, YoloBallObservation | None]:
        frames = self.align.process(self.pipeline.wait_for_frames())
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return np.zeros((self.args.height, self.args.width, 3), dtype=np.uint8), None

        color = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        result = self.model.predict(color, **self._prediction_kwargs())[0]
        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        candidates: list[YoloBallObservation] = []

        if result.boxes is not None:
            for box_tensor, confidence_tensor, class_tensor in zip(
                result.boxes.xyxy,
                result.boxes.conf,
                result.boxes.cls,
            ):
                if int(class_tensor.item()) not in self.ball_class_ids:
                    continue
                x0, y0, x1, y1 = [int(value) for value in box_tensor.tolist()]
                x0 = max(0, min(self.args.width - 1, x0))
                y0 = max(0, min(self.args.height - 1, y0))
                x1 = max(x0 + 1, min(self.args.width, x1))
                y1 = max(y0 + 1, min(self.args.height, y1))
                center = ((x0 + x1) // 2, (y0 + y1) // 2)
                if not (
                    self.args.ball_roi[0] <= center[0] <= self.args.ball_roi[2]
                    and self.args.ball_roi[1] <= center[1] <= self.args.ball_roi[3]
                ):
                    continue
                depth_result = depth_at_box(depth_frame, depth_image, (x0, y0, x1, y1), self.depth_scale)
                if depth_result is None:
                    continue
                depth_m, pixel = depth_result
                point = self.rs.rs2_deproject_pixel_to_point(intrinsics, list(pixel), depth_m)
                candidates.append(
                    YoloBallObservation(
                        confidence=float(confidence_tensor.item()),
                        box=(x0, y0, x1, y1),
                        pixel=pixel,
                        depth_m=depth_m,
                        camera_xyz_m=tuple(float(value) for value in point),
                    )
                )

        if not candidates:
            return color, None
        return color, max(candidates, key=self._candidate_score)

    def acquire_target(self, controller: ClawMachineController) -> PickTarget | None:
        observations: list[YoloBallObservation] = []
        deadline = time.monotonic() + self.args.ball_timeout
        last_status_t = 0.0
        candidate_seen = False
        window = "D405 YOLO target - q to abort"
        if not self.args.no_window:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)

        print(f"Refreshing D405 frames for {self.args.ball_warmup:.1f}s after reaching start pose.")
        self.warmup(self.args.ball_warmup)

        while time.monotonic() < deadline:
            color, candidate = self.read_observation()
            stable: YoloBallObservation | None = None
            if candidate is None:
                observations.clear()
            else:
                if observations:
                    previous = observations[-1]
                    pixel_jump = float(np.hypot(candidate.pixel[0] - previous.pixel[0], candidate.pixel[1] - previous.pixel[1]))
                    depth_jump = abs(candidate.depth_m - previous.depth_m)
                    if pixel_jump > self.args.ball_max_pixel_jump or depth_jump > self.args.ball_max_depth_jump:
                        observations.clear()
                observations = (observations + [candidate])[-self.args.ball_stable_frames:]
                if len(observations) >= self.args.ball_stable_frames:
                    stable = median_ball_observation(observations)
                candidate_seen = True

            now = time.monotonic()
            if now - last_status_t >= 1.0:
                if candidate is None:
                    print(
                        "D405 YOLO: no ball candidate "
                        f"(conf>={self.args.ball_conf}, roi={self.args.ball_roi})"
                    )
                else:
                    print(
                        "D405 YOLO: "
                        f"conf={candidate.confidence:.3f} "
                        f"depth={candidate.depth_m:.4f}m "
                        f"pixel={candidate.pixel} "
                        f"stable={len(observations)}/{self.args.ball_stable_frames}"
                    )
                last_status_t = now

            display = rotate_d405_ccw_90(color)
            if candidate is None:
                cv2.putText(display, "YOLO BALL NOT FOUND", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                rotated_box = rotate_d405_box_ccw_90(candidate.box, color.shape[1])
                rotated_center = rotate_d405_point_ccw_90(candidate.pixel, color.shape[1])
                cv2.rectangle(display, (rotated_box[0], rotated_box[1]), (rotated_box[2], rotated_box[3]), (0, 255, 0), 2)
                cv2.drawMarker(display, rotated_center, (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
                cv2.putText(
                    display,
                    f"BALL conf={candidate.confidence:.2f} depth={candidate.depth_m:.3f}m stable={len(observations)}/{self.args.ball_stable_frames}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (0, 255, 0),
                    2,
                )
            cv2.putText(display, "q: abort target acquisition", (20, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
            if not self.args.no_window:
                cv2.imshow(window, display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    return None

            if stable is None:
                continue

            base_tool = piper_pose_to_base_tool(controller.current_pose())
            camera_point = np.array([*stable.camera_xyz_m, 1.0])
            base_point = base_tool @ self.tool_camera @ camera_point
            base_xyz = [float(value) for value in base_point[:3]]
            base_xyz[0] += self.args.grab_x_offset
            base_xyz[1] += self.args.grab_y_offset
            if self.args.target_z_mode == "fixed":
                base_xyz[2] = self.args.fixed_grab_z
            else:
                base_xyz[2] = base_xyz[2] + self.args.grab_z_offset
            base_xyz[2] = max(self.args.min_grab_z, min(self.args.max_grab_z, base_xyz[2]))
            return PickTarget(
                base_xyz_m=tuple(base_xyz),
                camera_xyz_m=stable.camera_xyz_m,
                confidence=stable.confidence,
                pixel=stable.pixel,
                depth_m=stable.depth_m,
            )

        if candidate_seen:
            print(
                f"[warn] D405 saw ball candidates but none stayed stable for "
                f"{self.args.ball_stable_frames} frames within {self.args.ball_timeout:.1f}s"
            )
        else:
            print(
                f"[warn] no D405 YOLO ball candidate above conf={self.args.ball_conf} "
                f"within {self.args.ball_timeout:.1f}s"
            )
        return None


@dataclass(frozen=True)
class HomographyPlanarTarget:
    confidence: float
    pixel: tuple[int, int]
    ball_xy_m: tuple[float, float]
    flange_xy_m: tuple[float, float]
    theta_deg: float
    joints: list[float]
    fk_error_mm: float


class D405HomographyPlanarTargeter:
    def __init__(self, args: argparse.Namespace):
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError("pyrealsense2 is required for D405 homography targeting") from exc

        self.rs = rs
        self.args = args
        self.homography = load_homography(args.homography_calibration)
        self.model = YOLO(str(args.homography_ball_model))
        self.ball_class_ids = {
            class_id
            for class_id, name in self.model.names.items()
            if str(name).lower() == "sports ball"
        }
        if not self.ball_class_ids:
            raise RuntimeError(f"YOLO model has no sports ball class: {self.model.names}")

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        if args.ball_serial:
            self.config.enable_device(args.ball_serial)
        self.config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
        self.started = False

    def start(self) -> None:
        if self.started:
            return
        self.pipeline.start(self.config)
        self.started = True

    def stop(self) -> None:
        if self.started:
            self.pipeline.stop()
            self.started = False

    def _prediction_kwargs(self) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "conf": self.args.homography_conf,
            "imgsz": self.args.homography_imgsz,
            "verbose": False,
        }
        if self.args.ball_device:
            kwargs["device"] = self.args.ball_device
        return kwargs

    def read_detection(self) -> tuple[np.ndarray, PixelDetection | None]:
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return np.zeros((self.args.height, self.args.width, 3), dtype=np.uint8), None
        color = np.asanyarray(color_frame.get_data())
        result = self.model.predict(color, **self._prediction_kwargs())[0]
        detection = best_ball_pixel(
            result,
            self.ball_class_ids,
            self.args.ball_roi,
            self.args.width,
            self.args.height,
        )
        return color, detection

    def warmup(self, duration_s: float) -> None:
        deadline = time.monotonic() + max(0.0, duration_s)
        while time.monotonic() < deadline:
            self.pipeline.wait_for_frames()

    def prime(self) -> None:
        color, detection = self.read_detection()
        target_xy = apply_homography(self.homography, detection.pixel) if detection is not None else None
        if not self.args.no_window:
            cv2.imshow(
                "D405 homography tabletop target",
                draw_preview(color, detection, target_xy, 1 if detection else 0, self.args.ball_stable_frames),
            )
            cv2.waitKey(1)

    def acquire_target(self) -> HomographyPlanarTarget | None:
        detections: list[PixelDetection] = []
        deadline = time.monotonic() + self.args.ball_timeout
        last_status_t = 0.0
        window = "D405 homography tabletop target"
        if not self.args.no_window:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)

        print(f"Refreshing D405 frames for {self.args.ball_warmup:.1f}s before homography target.")
        self.warmup(self.args.ball_warmup)

        while time.monotonic() < deadline:
            color, detection = self.read_detection()
            detections = update_stable(
                detections,
                detection,
                self.args.ball_stable_frames,
                self.args.ball_max_pixel_jump,
            )
            stable = median_detection(detections) if len(detections) >= self.args.ball_stable_frames else None
            ball_xy_m = apply_homography(self.homography, stable.pixel) if stable is not None else None

            now = time.monotonic()
            if now - last_status_t >= 1.0:
                if detection is None:
                    print(
                        "D405 homography YOLO: no ball candidate "
                        f"(conf>={self.args.homography_conf}, roi={self.args.ball_roi})"
                    )
                else:
                    print(
                        "D405 homography YOLO: "
                        f"conf={detection.confidence:.3f} pixel={detection.pixel} "
                        f"stable={len(detections)}/{self.args.ball_stable_frames}"
                    )
                last_status_t = now

            if not self.args.no_window:
                preview_xy = ball_xy_m
                cv2.imshow(
                    window,
                    draw_preview(color, stable or detection, preview_xy, len(detections), self.args.ball_stable_frames),
                )
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    return None

            if stable is None or ball_xy_m is None:
                continue

            flange_xy_m, theta_deg = radial_flange_xy(ball_xy_m, self.args.radial_offset_mm)
            joints, fk, error_mm = solve_planar_joint_target(
                flange_xy_m,
                self.args.lift_z * 1000.0,
                self.args.planar_j4_deg,
                self.args.planar_j6_deg,
                self.args.planar_j5_seed_deg,
            )
            fk_xyz = fk[:3, 3]
            print(
                "homography target: "
                f"pixel={stable.pixel} conf={stable.confidence:.3f} "
                f"ball_xy=({ball_xy_m[0]:.4f},{ball_xy_m[1]:.4f})m "
                f"theta={theta_deg:.2f}deg radial_offset={self.args.radial_offset_mm:.1f}mm"
            )
            print(
                "planar_joint: "
                + ",".join(f"{value:.3f}" for value in joints)
                + f" fk_xyz=({fk_xyz[0]:.4f},{fk_xyz[1]:.4f},{fk_xyz[2]:.4f})m "
                + f"error={error_mm:.2f}mm"
            )
            return HomographyPlanarTarget(
                confidence=stable.confidence,
                pixel=stable.pixel,
                ball_xy_m=ball_xy_m,
                flange_xy_m=flange_xy_m,
                theta_deg=theta_deg,
                joints=joints,
                fk_error_mm=error_mm,
            )

        print(f"[warn] no stable D405 homography ball target within {self.args.ball_timeout:.1f}s")
        return None


def choose_system_gesture(
    player: str,
    player_win_probability: float,
    tie_probability: float,
    rng: random.Random,
    force_player_win: bool,
) -> str:
    if force_player_win:
        return BEATS[player]
    roll = rng.random()
    if roll < tie_probability:
        return player
    if roll < tie_probability + player_win_probability:
        return BEATS[player]
    return next(gesture for gesture, beaten in BEATS.items() if beaten == player)


def outcome(player: str, system: str) -> str:
    if player == system:
        return "TIE"
    return "PLAYER_WIN" if BEATS[player] == system else "SYSTEM_WIN"


def pose_from_xyz(current_pose: dict[str, float], xyz_m: list[float]) -> dict[str, float]:
    pose = dict(current_pose)
    pose["ee.x"] = xyz_m[0] * 1000.0
    pose["ee.y"] = xyz_m[1] * 1000.0
    pose["ee.z"] = xyz_m[2] * 1000.0
    return pose


def approach_xyz_from_args(args: argparse.Namespace) -> list[float]:
    if args.hover_xyz is not None:
        return list(args.hover_xyz)
    return [
        args.grab_xyz[0],
        args.grab_xyz[1],
        args.grab_xyz[2] + args.approach_height,
    ]


def set_rps_hand_pose(controller: ClawMachineController, gesture: str, delay_s: float) -> None:
    pose = RPS_HAND_POSES[gesture]
    controller.set_hand_speed(controller.config.hand_speed, f"{gesture} hand speed")
    if gesture == "paper":
        controller.set_hand_pose(
            {
                "thumb_swing": pose["thumb_swing"],
                "thumb_bend": pose["thumb_bend"],
            },
            "paper thumb stage",
        )
        time.sleep(delay_s)
    elif gesture == "rock":
        controller.set_hand_pose(
            {
                "little": pose["little"],
                "ring": pose["ring"],
                "middle": pose["middle"],
                "index": pose["index"],
            },
            "rock finger stage",
        )
        time.sleep(delay_s)
    elif gesture == "scissors":
        controller.set_hand_pose(
            {
                "little": pose["little"],
                "ring": pose["ring"],
                "middle": pose["middle"],
                "index": pose["index"],
            },
            "scissors finger stage",
        )
        time.sleep(delay_s)
    controller.set_hand_pose(pose, f"show {gesture}")


def disconnect_without_disable_prompt(robot: PiperRH56F2Follower) -> None:
    if robot.piper is not None:
        robot.piper.DisconnectPort()
    if robot.hand is not None:
        robot.hand.disconnect()
    for camera in robot.cameras.values():
        camera.disconnect()
    robot._is_connected = False


def status_hex(value: object) -> str:
    try:
        return f"0x{int(value):x}"
    except (TypeError, ValueError):
        return str(value)


def print_raw_status(robot: PiperRH56F2Follower, label: str) -> None:
    if robot.piper is None:
        return
    status = robot.piper.GetArmStatus().arm_status
    enable = list(robot.piper.GetArmEnableStatus())
    print(
        f"{label}: ctrl={status_hex(status.ctrl_mode)} "
        f"mode={status_hex(status.mode_feed)} "
        f"arm={status_hex(status.arm_status)} "
        f"enable={enable}"
    )


def make_controller(args: argparse.Namespace) -> tuple[PiperRH56F2Follower, ClawMachineController]:
    ball_classifier_config = None
    if args.classify_ball:
        ball_classifier_config = BallClassifierConfig(
            model=args.ball_tactile_model,
            output=args.ball_tactile_output,
            visual_reference_samples=args.ball_tactile_visual_reference_samples,
            contact_threshold=args.ball_contact_threshold,
            hover_duration=args.ball_hover_duration,
            hover_rate_hz=args.ball_hover_rate_hz,
            squeeze_delta=args.ball_squeeze_delta,
            squeeze_duration=args.ball_squeeze_duration,
            ab_squeeze_test=args.ball_ab_squeeze_test,
            ab_squeeze_threshold=args.ball_ab_squeeze_threshold,
            ab_squeeze_a_standard=args.ball_ab_squeeze_a_standard,
            ab_squeeze_b_standard=args.ball_ab_squeeze_b_standard,
            ab_squeeze_mode=args.ball_ab_squeeze_mode,
            low_confidence_c_squeeze_threshold=args.ball_low_confidence_c_squeeze_threshold,
            ab_friction_threshold=args.ball_ab_friction_threshold,
            ab_friction_finger=args.ball_ab_friction_finger,
            ab_friction_feature=args.ball_ab_friction_feature,
            ab_proximity_assist=args.ball_ab_proximity_assist,
            ab_proximity_index_force_threshold=args.ball_ab_proximity_index_force_threshold,
            ab_proximity_thumb_threshold=args.ball_ab_proximity_thumb_threshold,
            ab_proximity_a_direction=args.ball_ab_proximity_a_direction,
            ab_proximity_min_samples=args.ball_ab_proximity_min_samples,
            bc_proximity_assist=args.ball_bc_proximity_assist,
            bc_proximity_thumb_threshold=args.ball_bc_proximity_thumb_threshold,
            bc_proximity_middle_threshold=args.ball_bc_proximity_middle_threshold,
            notes="rps_homography_grasp",
        )
    robot = PiperRH56F2Follower(
        PiperRH56F2FollowerConfig(
            can_port=args.can,
            speed_rate=args.speed,
            hand_port=args.hand_port,
            hand_id=args.hand_id,
            hand_speed=args.hand_speed,
            hand_force=args.hand_force,
            max_ee_delta_mm=None,
            max_ee_delta_deg=None,
            max_hand_delta=None,
        )
    )
    controller = ClawMachineController(
        robot,
        ClawMachineTaskConfig(
            grab_z=args.grab_xyz[2] * 1000.0,
            start_pose=pose_from_values(list(DEFAULT_START_POSE)),
            start_joints=list(DEFAULT_START_JOINTS),
            lift_z=approach_xyz_from_args(args)[2] * 1000.0,
            speed_rate=args.speed,
            rate_hz=args.rate_hz,
            start_duration_s=args.start_duration,
            hover_duration_s=args.hover_duration,
            vertical_duration_s=args.vertical_duration,
            return_duration_s=args.return_duration,
            hand_speed=args.hand_speed,
            pre_grab_open_speed=args.pre_grab_open_speed,
            hand_settle_s=args.hand_settle,
            pre_grab_open_settle_s=args.pre_grab_open_settle,
            auto_position_tolerance_mm=args.position_tolerance_mm,
            auto_rpy_tolerance_deg=args.rpy_tolerance_deg,
            result_gesture=False,
            drop_pose=pose_from_values(list(args.drop_pose)),
            transfer_duration_s=args.transfer_duration,
            drop_open_settle_s=args.drop_open_settle,
            ball_classifier_config=ball_classifier_config,
        ),
    )
    return robot, controller


def run_fixed_pick(
    controller: ClawMachineController,
    args: argparse.Namespace,
    rps_joints: list[float],
) -> bool:
    print("Moving to claw-machine start joints before fixed pick.")
    if not controller.move_joints_for(
        list(DEFAULT_START_JOINTS),
        args.speed,
        args.start_duration,
        "start MOVE_J",
    ):
        return False

    approach_xyz = approach_xyz_from_args(args)
    start_pose = controller.current_pose()
    approach_pose = pose_from_xyz(start_pose, approach_xyz)
    grab_pose = pose_from_xyz(approach_pose, args.grab_xyz)
    lift_pose = pose_from_xyz(approach_pose, approach_xyz)

    print(f"approach pose: {controller.format_pose(approach_pose)}")
    print(f"grab pose:     {controller.format_pose(grab_pose)}")
    print(f"lift pose:     {controller.format_pose(lift_pose)}")

    if not controller.move_ee_for(
        approach_pose,
        args.hover_duration,
        "approach MOVE_P",
        require_reached=True,
    ):
        return False

    controller.open_while_descending()
    if args.pre_grab_open_settle > 0:
        if not controller.wait_with_stop(args.pre_grab_open_settle):
            controller.hold_current_position()
            return False

    if not controller.move_ee_for(
        grab_pose,
        args.vertical_duration,
        "descend MOVE_P",
        require_reached=True,
    ):
        return False

    controller.set_hand_force(args.hand_force, "max grasp force")
    if not controller.close_at_grab_adaptive():
        return False
    if args.hand_settle > 0:
        if not controller.wait_with_stop(args.hand_settle):
            controller.hold_current_position()
            return False

    if not controller.move_ee_for(
        lift_pose,
        args.vertical_duration,
        "lift MOVE_P",
        require_reached=True,
    ):
        return False

    if args.check_held:
        held = controller.held_by_force()
        print(f"held_by_force={held}")

    print("Returning to captured RPS arm joints.")
    if not controller.move_joints_for(
        rps_joints,
        args.speed,
        args.return_duration,
        "return RPS MOVE_J",
    ):
        return False

    if args.ready_hand_after_return:
        set_rps_hand_pose(controller, args.ready_hand_after_return, args.hand_stage_delay)
    return True


def refine_target_at_hover(
    controller: ClawMachineController,
    targeter: D405YoloTargeter,
    args: argparse.Namespace,
    previous_grab_xyz: list[float],
) -> list[float] | None:
    print("Refining target from current hover pose with D405 YOLO.")
    target = targeter.acquire_target(controller)
    if target is None:
        print("[warn] hover refinement failed: no stable D405 target")
        return None

    refined_grab_xyz = list(target.base_xyz_m)
    dx = refined_grab_xyz[0] - previous_grab_xyz[0]
    dy = refined_grab_xyz[1] - previous_grab_xyz[1]
    dz = refined_grab_xyz[2] - previous_grab_xyz[2]
    xy_delta = float(np.hypot(dx, dy))
    print(
        "hover refined target: "
        f"conf={target.confidence:.3f} depth={target.depth_m:.4f}m "
        f"base={target.base_xyz_m} "
        f"delta=(dx={dx:.4f}, dy={dy:.4f}, dz={dz:.4f}, xy={xy_delta:.4f})m"
    )

    if xy_delta > args.refine_max_xy:
        print(
            f"[warn] hover refinement rejected: xy correction {xy_delta:.4f}m "
            f"> --refine-max-xy {args.refine_max_xy:.4f}m"
        )
        return None
    return refined_grab_xyz


def run_yolo_pick(
    controller: ClawMachineController,
    targeter: D405YoloTargeter,
    args: argparse.Namespace,
    rps_joints: list[float],
) -> bool:
    print("Moving to claw-machine start joints before D405 target acquisition.")
    if not controller.move_joints_for(
        list(DEFAULT_START_JOINTS),
        args.speed,
        args.start_duration,
        "start MOVE_J",
    ):
        return False

    print("Acquiring stable D405 YOLO ball target.")
    target = targeter.acquire_target(controller)
    if target is None:
        return False

    grab_xyz = list(target.base_xyz_m)
    approach_xyz = [grab_xyz[0], grab_xyz[1], grab_xyz[2] + args.approach_height]
    start_pose = controller.current_pose()
    approach_pose = pose_from_xyz(start_pose, approach_xyz)
    grab_pose = pose_from_xyz(approach_pose, grab_xyz)
    lift_pose = pose_from_xyz(approach_pose, approach_xyz)

    print(
        "D405 target: "
        f"conf={target.confidence:.3f} depth={target.depth_m:.4f}m "
        f"camera={target.camera_xyz_m} base={target.base_xyz_m}"
    )
    print(f"target z mode: {args.target_z_mode}, final grab XYZ(m): {tuple(grab_xyz)}")
    print(f"approach pose: {controller.format_pose(approach_pose)}")
    print(f"grab pose:     {controller.format_pose(grab_pose)}")
    print(f"lift pose:     {controller.format_pose(lift_pose)}")

    if args.hover_only:
        print("hover-only mode: moving to approach pose, no descent, no gripper close.")
        ok = controller.move_ee_for(
            approach_pose,
            args.hover_duration,
            "approach MOVE_P",
            require_reached=True,
        )
        if ok and args.refine_at_hover:
            refined_target = refine_target_at_hover(controller, targeter, args, grab_xyz)
            if refined_target is None:
                return False
            grab_xyz = refined_target
            approach_xyz = [grab_xyz[0], grab_xyz[1], grab_xyz[2] + args.approach_height]
            current_pose = controller.current_pose()
            approach_pose = pose_from_xyz(current_pose, approach_xyz)
            print(f"refined hover pose: {controller.format_pose(approach_pose)}")
            ok = controller.move_ee_for(
                approach_pose,
                args.refine_duration,
                "refined approach MOVE_P",
                require_reached=True,
            )
        if ok:
            print("Returning to captured RPS arm joints.")
            ok = controller.move_joints_for(
                rps_joints,
                args.speed,
                args.return_duration,
                "return RPS MOVE_J",
            )
        return ok

    if not controller.move_ee_for(
        approach_pose,
        args.hover_duration,
        "approach MOVE_P",
        require_reached=True,
    ):
        return False

    if args.refine_at_hover:
        refined_target = refine_target_at_hover(controller, targeter, args, grab_xyz)
        if refined_target is None:
            return False
        grab_xyz = refined_target
        approach_xyz = [grab_xyz[0], grab_xyz[1], grab_xyz[2] + args.approach_height]
        current_pose = controller.current_pose()
        approach_pose = pose_from_xyz(current_pose, approach_xyz)
        grab_pose = pose_from_xyz(approach_pose, grab_xyz)
        lift_pose = pose_from_xyz(approach_pose, approach_xyz)
        print(f"refined approach pose: {controller.format_pose(approach_pose)}")
        print(f"refined grab pose:     {controller.format_pose(grab_pose)}")
        if not controller.move_ee_for(
            approach_pose,
            args.refine_duration,
            "refined approach MOVE_P",
            require_reached=True,
        ):
            return False

    controller.open_while_descending()
    if args.pre_grab_open_settle > 0:
        if not controller.wait_with_stop(args.pre_grab_open_settle):
            controller.hold_current_position()
            return False

    if not controller.move_ee_for(
        grab_pose,
        args.vertical_duration,
        "descend MOVE_P",
        require_reached=True,
    ):
        return False

    controller.set_hand_force(args.hand_force, "max grasp force")
    if not controller.close_at_grab_adaptive():
        return False
    if args.hand_settle > 0:
        if not controller.wait_with_stop(args.hand_settle):
            controller.hold_current_position()
            return False

    if not controller.move_ee_for(
        lift_pose,
        args.vertical_duration,
        "lift MOVE_P",
        require_reached=True,
    ):
        return False

    if args.check_held:
        held = controller.held_by_force()
        print(f"held_by_force={held}")

    print("Returning to captured RPS arm joints.")
    if not controller.move_joints_for(
        rps_joints,
        args.speed,
        args.return_duration,
        "return RPS MOVE_J",
    ):
        return False

    if args.ready_hand_after_return:
        set_rps_hand_pose(controller, args.ready_hand_after_return, args.hand_stage_delay)
    return True


def run_homography_planar_pick(
    controller: ClawMachineController,
    targeter: D405HomographyPlanarTargeter,
    args: argparse.Namespace,
) -> bool:
    print("Moving to claw-machine start joints before homography target acquisition.")
    if not controller.move_joints_for(
        list(DEFAULT_START_JOINTS),
        args.speed,
        args.start_duration,
        "start MOVE_J",
    ):
        return False

    print("Acquiring stable fixed-D405 homography ball target.")
    target = targeter.acquire_target()
    if target is None:
        return False

    if target.fk_error_mm > args.max_planar_fk_error_mm:
        print(
            f"[warn] planar FK error {target.fk_error_mm:.2f}mm "
            f"> --max-planar-fk-error-mm {args.max_planar_fk_error_mm:.2f}mm"
        )
        return False

    print("Moving planar joints to radial flange hover above ball.")
    if not controller.move_joints_until_reached(
        target.joints,
        args.speed,
        args.planar_duration,
        args.planar_hold_after_reached,
        "planar target MOVE_J",
        tolerance_deg=args.planar_joint_tolerance_deg,
    ):
        return False

    hover_pose = controller.current_pose()
    grab_pose = dict(hover_pose)
    grab_pose["ee.z"] = args.fixed_grab_z * 1000.0
    lift_pose = dict(hover_pose)
    lift_pose["ee.z"] = max(hover_pose["ee.z"], args.lift_z * 1000.0)
    drop_pose = pose_from_values(list(args.drop_pose))

    print("Running homography planar pick cycle")
    print(f"  hover: {controller.format_pose(hover_pose)}")
    print(f"  grab:  {controller.format_pose(grab_pose)}")
    print(f"  lift:  {controller.format_pose(lift_pose)}")
    print(f"  drop:  {controller.format_pose(drop_pose)}")
    print(f"  start: {controller.format_pose(pose_from_values(list(DEFAULT_START_POSE)))}")

    previous_lift_z = controller.config.lift_z
    previous_grab_z = controller.config.grab_z
    try:
        controller.config.grab_z = grab_pose["ee.z"]
        controller.config.lift_z = lift_pose["ee.z"]
        return controller.run_pick_cycle(
            pose_from_values(list(DEFAULT_START_POSE)),
            hover_pose,
            drop_pose,
        )
    finally:
        controller.config.lift_z = previous_lift_z
        controller.config.grab_z = previous_grab_z


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--can", default="can0")
    parser.add_argument("--hand-port", default="/dev/ttyUSB0")
    parser.add_argument("--hand-id", type=int, default=1)
    parser.add_argument("--hand-speed", type=int, default=2500)
    parser.add_argument("--hand-force", type=int, default=1500)
    parser.add_argument("--speed", type=int, default=8)
    parser.add_argument("--rate-hz", type=float, default=40.0)
    parser.add_argument("--start-duration", type=float, default=20.0)
    parser.add_argument("--hover-duration", type=float, default=8.0)
    parser.add_argument("--vertical-duration", type=float, default=4.0)
    parser.add_argument("--transfer-duration", type=float, default=8.0)
    parser.add_argument("--return-duration", type=float, default=8.0)
    parser.add_argument("--gesture-serial", default=DEFAULT_D435I_SERIAL)
    parser.add_argument("--gesture-model", type=Path, default=DEFAULT_GESTURE_MODEL)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--stable-frames", type=int, default=5)
    parser.add_argument("--min-detection", type=float, default=0.7)
    parser.add_argument("--min-tracking", type=float, default=0.5)
    parser.add_argument("--min-presence", type=float, default=0.5)
    parser.add_argument("--min-score", type=float, default=0.5)
    parser.add_argument("--keyboard", action="store_true", help="use keyboard r/p/s instead of D435i")
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--print-vision", action="store_true")
    parser.add_argument(
        "--pick-source",
        choices=("yolo", "fixed", "homography"),
        default="yolo",
        help="yolo uses D405+YOLO+hand-eye; homography uses fixed D405 tabletop XY; fixed keeps old --grab-xyz",
    )
    parser.add_argument("--grab-xyz", type=parse_xyz, default=list(FIXED_GRAB_XYZ_M))
    parser.add_argument(
        "--approach-height",
        type=float,
        default=DEFAULT_APPROACH_HEIGHT_M,
        help="metres above --grab-xyz used for the internal safe approach/lift point",
    )
    parser.add_argument(
        "--hover-xyz",
        type=parse_xyz,
        default=None,
        help="advanced override for approach/lift X,Y,Z in metres; default is grab XYZ plus approach height",
    )
    parser.add_argument("--ball-serial", default=DEFAULT_D405_SERIAL)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--ball-model", type=Path, default=DEFAULT_BALL_MODEL)
    parser.add_argument("--ball-roi", type=parse_roi, default=(0, 0, 640, 480))
    parser.add_argument("--ball-conf", type=float, default=0.70)
    parser.add_argument("--ball-imgsz", type=int, default=1280)
    parser.add_argument("--ball-device", default="0", help="ultralytics device for D405 YOLO, e.g. 0 or cpu")
    parser.add_argument("--ball-timeout", type=float, default=12.0)
    parser.add_argument(
        "--ball-warmup",
        type=float,
        default=0.2,
        help="seconds to refresh D405 frames after moving to claw-machine start pose",
    )
    parser.add_argument("--ball-stable-frames", type=int, default=8)
    parser.add_argument("--ball-max-pixel-jump", type=float, default=25.0)
    parser.add_argument("--ball-max-depth-jump", type=float, default=0.02)
    parser.add_argument(
        "--ball-select",
        choices=("center", "confidence", "near"),
        default="center",
        help="which detected sports ball to grab when several are visible",
    )
    parser.add_argument(
        "--target-z-mode",
        choices=("fixed", "vision"),
        default="fixed",
        help="fixed uses --fixed-grab-z; vision uses calibrated Z plus --grab-z-offset",
    )
    parser.add_argument("--fixed-grab-z", type=float, default=FIXED_GRAB_XYZ_M[2])
    parser.add_argument("--homography-calibration", type=Path, default=DEFAULT_HOMOGRAPHY_CALIBRATION)
    parser.add_argument("--homography-ball-model", type=Path, default=DEFAULT_HOMOGRAPHY_BALL_MODEL)
    parser.add_argument("--homography-conf", type=float, default=0.05)
    parser.add_argument("--homography-imgsz", type=int, default=960)
    parser.add_argument("--radial-offset-mm", type=float, default=45.0)
    parser.add_argument("--planar-j4-deg", type=float, default=0.0)
    parser.add_argument("--planar-j6-deg", type=float, default=0.0)
    parser.add_argument("--planar-j5-seed-deg", type=float, default=13.0)
    parser.add_argument("--planar-duration", type=float, default=5.0)
    parser.add_argument("--planar-joint-tolerance-deg", type=float, default=1.5)
    parser.add_argument("--planar-hold-after-reached", type=float, default=0.15)
    parser.add_argument("--max-planar-fk-error-mm", type=float, default=3.0)
    parser.add_argument("--lift-z", type=float, default=0.285, help="metres used as hover/lift Z after homography grasp")
    parser.add_argument("--drop-pose", type=parse_pose_values, default=list(DEFAULT_DROP_POSE))
    parser.add_argument("--grab-x-offset", type=float, default=0.0, help="metres added to vision target X before approach/grab")
    parser.add_argument("--grab-y-offset", type=float, default=0.0, help="metres added to vision target Y before approach/grab")
    parser.add_argument("--grab-z-offset", type=float, default=0.0)
    parser.add_argument("--min-grab-z", type=float, default=0.15)
    parser.add_argument("--max-grab-z", type=float, default=0.35)
    parser.add_argument(
        "--hover-only",
        action="store_true",
        help="after player win, move only to visual approach pose; no descent or gripper close",
    )
    parser.add_argument(
        "--refine-at-hover",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="after the first approach move, detect the ball again and correct X/Y before descending",
    )
    parser.add_argument(
        "--refine-max-xy",
        type=float,
        default=0.08,
        help="maximum accepted hover refinement correction in metres",
    )
    parser.add_argument(
        "--refine-duration",
        type=float,
        default=3.0,
        help="seconds used for the small MOVE_P correction after hover refinement",
    )
    parser.add_argument("--rps-joints", type=parse_joint_degrees, default=list(DEFAULT_RPS_JOINTS))
    parser.add_argument("--player-win-probability", type=float, default=0.50)
    parser.add_argument("--tie-probability", type=float, default=0.0)
    parser.add_argument("--force-player-win", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--hand-stage-delay", type=float, default=0.15)
    parser.add_argument(
        "--rps-result-hold",
        type=float,
        default=1.0,
        help="seconds to hold the robot's RPS result gesture before pick/return",
    )
    parser.add_argument("--pre-grab-open-speed", type=int, default=2500)
    parser.add_argument("--pre-grab-open-settle", type=float, default=1.0)
    parser.add_argument("--hand-settle", type=float, default=1.0)
    parser.add_argument("--drop-open-settle", type=float, default=4.0)
    parser.add_argument("--position-tolerance-mm", type=float, default=2.0)
    parser.add_argument("--rpy-tolerance-deg", type=float, default=2.0)
    parser.add_argument("--check-held", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--classify-ball",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="after lift hover, update the same LIVEBOARD flow as predict_live.py",
    )
    parser.add_argument("--ball-tactile-model", type=Path, default=DEFAULT_TACTILE_MODEL)
    parser.add_argument("--ball-tactile-output", type=Path, default=DEFAULT_TACTILE_OUTPUT)
    parser.add_argument("--ball-tactile-visual-reference-samples", type=Path, default=DEFAULT_TACTILE_REFERENCE_SAMPLES)
    parser.add_argument("--ball-contact-threshold", type=float, default=70.0)
    parser.add_argument("--ball-hover-duration", type=float, default=5.0)
    parser.add_argument("--ball-hover-rate-hz", type=float, default=10.0)
    parser.add_argument("--ball-squeeze-delta", type=float, default=40.0)
    parser.add_argument("--ball-squeeze-duration", type=float, default=3.0)
    parser.add_argument("--ball-ab-squeeze-test", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ball-ab-squeeze-threshold", type=float, default=190.0)
    parser.add_argument("--ball-ab-squeeze-a-standard", type=float, default=238.0)
    parser.add_argument("--ball-ab-squeeze-b-standard", type=float, default=142.5)
    parser.add_argument(
        "--ball-ab-squeeze-mode",
        choices=("friction", "shape", "curve", "threshold"),
        default="friction",
    )
    parser.add_argument("--ball-low-confidence-c-squeeze-threshold", type=float, default=0.0)
    parser.add_argument("--ball-ab-friction-threshold", type=float, default=0.1464)
    parser.add_argument(
        "--ball-ab-friction-finger",
        choices=("index", "middle", "thumb"),
        default="middle",
    )
    parser.add_argument(
        "--ball-ab-friction-feature",
        choices=("last", "mean", "max", "late_slope"),
        default="last",
    )
    parser.add_argument("--ball-ab-proximity-assist", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ball-ab-proximity-index-force-threshold", type=float, default=70.0)
    parser.add_argument("--ball-ab-proximity-thumb-threshold", type=float, default=169619.0)
    parser.add_argument(
        "--ball-ab-proximity-a-direction",
        choices=(">=", "<="),
        default="<=",
    )
    parser.add_argument("--ball-ab-proximity-min-samples", type=float, default=5.0)
    parser.add_argument("--ball-bc-proximity-assist", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ball-bc-proximity-thumb-threshold", type=float, default=180000.0)
    parser.add_argument("--ball-bc-proximity-middle-threshold", type=float, default=100000.0)
    parser.add_argument(
        "--ready-hand-after-return",
        choices=("rock", "paper", "scissors"),
        default=None,
        help="optional hand gesture after returning; default keeps the grasp closed",
    )
    parser.add_argument("--one-shot", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--yes", action="store_true")
    parser.add_argument(
        "--disconnect-prompt",
        action="store_true",
        help="use follower disconnect prompt; default disconnects without DisableArm",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not 0 <= args.speed <= 100:
        raise SystemExit("--speed must be between 0 and 100")
    for name in ("rate_hz", "start_duration", "hover_duration", "vertical_duration", "transfer_duration", "return_duration"):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        raise SystemExit("--width, --height, and --fps must be positive")
    if args.stable_frames <= 0:
        raise SystemExit("--stable-frames must be positive")
    if not 0.0 <= args.player_win_probability <= 1.0:
        raise SystemExit("--player-win-probability must be in [0, 1]")
    if not 0.0 <= args.tie_probability < 1.0:
        raise SystemExit("--tie-probability must be in [0, 1)")
    if args.player_win_probability + args.tie_probability > 1.0:
        raise SystemExit("--player-win-probability + --tie-probability must be <= 1")
    if args.approach_height < 0:
        raise SystemExit("--approach-height must be non-negative")
    if args.rps_result_hold < 0:
        raise SystemExit("--rps-result-hold must be non-negative")
    if args.pick_source == "yolo":
        if not args.calibration.exists():
            raise SystemExit(f"--calibration does not exist: {args.calibration}")
        if not args.ball_model.exists():
            raise SystemExit(f"--ball-model does not exist: {args.ball_model}")
    if args.pick_source == "homography":
        if not args.homography_calibration.exists():
            raise SystemExit(f"--homography-calibration does not exist: {args.homography_calibration}")
        if not args.homography_ball_model.exists():
            raise SystemExit(f"--homography-ball-model does not exist: {args.homography_ball_model}")
        if args.radial_offset_mm < 0:
            raise SystemExit("--radial-offset-mm must be non-negative")
        if args.planar_duration <= 0 or args.planar_joint_tolerance_deg <= 0:
            raise SystemExit("--planar-duration and --planar-joint-tolerance-deg must be positive")
        if args.max_planar_fk_error_mm < 0:
            raise SystemExit("--max-planar-fk-error-mm must be non-negative")
        if args.lift_z <= 0:
            raise SystemExit("--lift-z must be positive")
    if not 0.0 <= args.ball_conf <= 1.0:
        raise SystemExit("--ball-conf must be in [0, 1]")
    if args.ball_imgsz <= 0:
        raise SystemExit("--ball-imgsz must be positive")
    if args.ball_timeout <= 0:
        raise SystemExit("--ball-timeout must be positive")
    if args.ball_stable_frames <= 0:
        raise SystemExit("--ball-stable-frames must be positive")
    if args.ball_max_pixel_jump <= 0 or args.ball_max_depth_jump <= 0:
        raise SystemExit("--ball-max-pixel-jump and --ball-max-depth-jump must be positive")
    if args.fixed_grab_z <= 0:
        raise SystemExit("--fixed-grab-z must be positive")
    if args.min_grab_z > args.max_grab_z:
        raise SystemExit("--min-grab-z must be <= --max-grab-z")
    if args.refine_max_xy <= 0:
        raise SystemExit("--refine-max-xy must be positive")
    if args.refine_duration <= 0:
        raise SystemExit("--refine-duration must be positive")
    if args.drop_open_settle < 0 or args.hand_settle < 0 or args.pre_grab_open_settle < 0:
        raise SystemExit("hand settle times must be non-negative")
    if args.classify_ball:
        if not args.ball_tactile_model.exists():
            raise SystemExit(f"--ball-tactile-model does not exist: {args.ball_tactile_model}")
        if not args.ball_tactile_visual_reference_samples.exists():
            raise SystemExit(
                f"--ball-tactile-visual-reference-samples does not exist: "
                f"{args.ball_tactile_visual_reference_samples}"
            )
        if args.ball_contact_threshold < 0:
            raise SystemExit("--ball-contact-threshold must be non-negative")
        if args.ball_hover_duration <= 0 or args.ball_hover_rate_hz <= 0:
            raise SystemExit("--ball-hover-duration/rate must be positive")
        if args.ball_squeeze_delta <= 0 or args.ball_squeeze_duration <= 0:
            raise SystemExit("--ball-squeeze-delta/duration must be positive")


def read_player_gesture() -> str | None:
    value = input("Player gesture [r/p/s/q]: ").strip().lower()
    aliases = {
        "r": "rock",
        "rock": "rock",
        "p": "paper",
        "paper": "paper",
        "s": "scissors",
        "scissors": "scissors",
        "q": None,
        "quit": None,
        "exit": None,
    }
    if value not in aliases:
        print("Ignored. Use r, p, s, or q.")
        return ""
    return aliases[value]


def wait_for_camera_gesture(
    camera: D455ColorCamera,
    recognizer: HandGestureRecognizer,
    args: argparse.Namespace,
) -> str | None:
    stable_gesture = ""
    stable_count = 0
    last_print_t = 0.0
    window = "D435i RPS - q to quit"
    if not args.no_window:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    while True:
        frame = camera.read()
        if frame is None:
            continue
        if not args.no_mirror:
            frame = cv2.flip(frame, 1)
        annotated, gesture_name = recognizer.get_gesture(frame)
        player = CAMERA_GESTURES.get(gesture_name, "")
        if player and player == stable_gesture:
            stable_count += 1
        elif player:
            stable_gesture = player
            stable_count = 1
        else:
            stable_gesture = ""
            stable_count = 0

        if args.print_vision and time.monotonic() - last_print_t >= 1.0:
            print(
                "vision: "
                f"raw={recognizer.last_category} "
                f"score={recognizer.last_score:.2f} "
                f"hands={recognizer.last_hand_count} "
                f"fallback={recognizer.last_fallback} "
                f"mapped={gesture_name} "
                f"stable={stable_gesture or 'none'}:{stable_count}/{args.stable_frames}"
            )
            last_print_t = time.monotonic()

        if not args.no_window:
            cv2.putText(
                annotated,
                f"gesture={gesture_name} stable={stable_count}/{args.stable_frames}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 255, 0),
                2,
            )
            cv2.imshow(window, annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return None

        if stable_gesture and stable_count >= args.stable_frames:
            print(f"Detected player gesture: {DISPLAY[stable_gesture]}")
            return stable_gesture


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)

    print("SAFETY: D435i is used only for RPS; D405 is used only after the player wins.")
    print("Player win triggers pick and return to configured RPS pose.")
    print(f"gesture serial: {args.gesture_serial or 'any RealSense color camera'}")
    print(f"pick source: {args.pick_source}")
    if args.pick_source == "fixed":
        print(f"grab XYZ(m):     {tuple(args.grab_xyz)}")
        print(f"approach XYZ(m): {tuple(approach_xyz_from_args(args))}")
    elif args.pick_source == "homography":
        print(f"D405 serial: {args.ball_serial or 'any RealSense color camera'}")
        print(f"YOLO model: {args.homography_ball_model}")
        print(f"homography calibration: {args.homography_calibration}")
        print(f"fixed grab Z(m): {args.fixed_grab_z}")
        print(f"lift Z(m): {args.lift_z}")
        print(f"radial offset(mm): {args.radial_offset_mm}")
        print(f"drop pose(mm/deg): {','.join(f'{value:.3f}' for value in args.drop_pose)}")
    else:
        print(f"D405 serial: {args.ball_serial or 'any RealSense depth camera'}")
        print(f"YOLO model: {args.ball_model}")
        print(f"calibration: {args.calibration}")
        print(f"target z mode: {args.target_z_mode}")
        if args.target_z_mode == "fixed":
            print(f"fixed grab Z(m): {args.fixed_grab_z}")
        else:
            print(f"vision grab Z offset(m): {args.grab_z_offset}")
        print(f"grab XY offset(m): ({args.grab_x_offset}, {args.grab_y_offset})")
        print(f"grab Z clamp(m): [{args.min_grab_z}, {args.max_grab_z}]")
        if args.hover_only:
            print("hover-only: no descent and no gripper close after D405 target acquisition.")
    if args.classify_ball:
        dashboard = args.ball_tactile_output.with_name("live_dashboard.html")
        print(f"tactile model: {args.ball_tactile_model}")
        print(f"tactile LIVEBOARD csv: {args.ball_tactile_output}")
        print(f"tactile LIVEBOARD html: {dashboard}")
    print(f"RPS joints: {fmt_joints(args.rps_joints)}")
    print(f"claw start joints: {fmt_joints(list(DEFAULT_START_JOINTS))}")
    if not args.yes:
        if input("Type RPS_PICK to connect and enable Piper/RH56F2: ").strip() != "RPS_PICK":
            print("Aborted before connecting.")
            return 0

    rng = random.Random(args.seed)
    robot, controller = make_controller(args)
    camera = None
    recognizer = None
    targeter = None
    try:
        if not args.keyboard:
            camera = D455ColorCamera(args.gesture_serial, args.width, args.height, args.fps)
            camera.start()
            recognizer = HandGestureRecognizer(
                args.gesture_model,
                args.min_detection,
                args.min_tracking,
                args.min_presence,
                args.min_score,
            )
        if args.pick_source == "yolo":
            targeter = D405YoloTargeter(args)
            targeter.start()
            print("Priming D405 YOLO once before the RPS game.")
            targeter.prime()
        elif args.pick_source == "homography":
            targeter = D405HomographyPlanarTargeter(args)
            targeter.start()
            print("Priming fixed D405 homography YOLO once before the RPS game.")
            targeter.prime()
        robot.connect()
        print_raw_status(robot, "after connect")
        rps_joints = list(args.rps_joints)
        print("Moving to configured RPS arm joints.")
        if not controller.move_joints_for(
            rps_joints,
            args.speed,
            args.return_duration,
            "RPS ready MOVE_J",
        ):
            raise RuntimeError("RPS ready MOVE_J failed")
        set_rps_hand_pose(controller, "paper", args.hand_stage_delay)

        while True:
            if args.keyboard:
                player = read_player_gesture()
            else:
                print("Show rock/paper/scissors to D435i.")
                player = wait_for_camera_gesture(camera, recognizer, args)
            if player is None:
                break
            if player == "":
                continue

            system = choose_system_gesture(
                player,
                args.player_win_probability,
                args.tie_probability,
                rng,
                args.force_player_win,
            )
            set_rps_hand_pose(controller, system, args.hand_stage_delay)
            if args.rps_result_hold > 0:
                time.sleep(args.rps_result_hold)
            result = outcome(player, system)
            print(
                f"player={DISPLAY[player]} system={DISPLAY[system]} result={result}"
            )
            if result == "PLAYER_WIN":
                if not args.yes:
                    answer = input("Player won. Press Enter to pick, or type skip: ").strip().lower()
                    if answer == "skip":
                        continue
                if args.pick_source == "fixed":
                    ok = run_fixed_pick(controller, args, rps_joints)
                    print(f"fixed pick {'complete' if ok else 'failed'}")
                elif args.pick_source == "homography":
                    if targeter is None:
                        raise RuntimeError("D405 homography targeter is not initialized")
                    ok = run_homography_planar_pick(controller, targeter, args)
                    print(f"homography pick {'complete' if ok else 'failed'}")
                else:
                    if targeter is None:
                        raise RuntimeError("D405 YOLO targeter is not initialized")
                    ok = run_yolo_pick(controller, targeter, args, rps_joints)
                    print(f"YOLO pick {'complete' if ok else 'failed'}")
                if args.one_shot:
                    return 0 if ok else 1
            else:
                print("Player did not win; pick is not triggered.")
            if args.one_shot:
                return 0
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted. Motors were not disabled by this script.")
        return 130
    except Exception as exc:
        print(f"\n[warn] {exc}")
        try:
            print_raw_status(robot, "failure status")
        except Exception:
            pass
        return 1
    finally:
        if targeter is not None:
            targeter.stop()
        if recognizer is not None:
            recognizer.close()
        if camera is not None:
            try:
                camera.stop()
            except Exception as exc:
                print(f"[warn] D435i stop skipped: {exc}")
        cv2.destroyAllWindows()
        if robot.is_connected:
            if args.disconnect_prompt:
                robot.disconnect()
            else:
                disconnect_without_disable_prompt(robot)
                print("Disconnected without sending DisableArm.")


if __name__ == "__main__":
    raise SystemExit(main())
