#!/usr/bin/env python3
"""Complete LeRobot-style claw-machine controller for Piper + RH56F2.

This is the LeRobot equivalent of the direct-SDK claw_machine workflow:

  setup MOVE_J -> keyboard/gamepad MOVE_J teleop -> pick/drop MOVE_P cycle
  -> force-based held check -> result gesture -> back to teleop.

The controller talks to hardware through the LeRobot Robot surface:

  - robot.get_observation()
  - robot.send_action(action)
"""

from __future__ import annotations

import argparse
import os
import select
import struct
import sys
import termios
import threading
import time
import tty
from copy import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from lerobot.processor import RobotAction, RobotObservation

try:
    from ..config_piper_rh56f2_follower import PiperRH56F2FollowerConfig
    from ..piper_follower import JOINT_LIMITS_DEG, JOINT_NAMES
    from ..piper_rh56f2_follower import EE_POSE_NAMES, PiperRH56F2Follower
    from ..rh56f2_hand import DEFAULT_CLOSED, DEFAULT_OPEN, HAND_NAMES
    from ..rock_paper_scissors.ball_tactile_classifier.claw_integration import (
        DEFAULT_MODEL as DEFAULT_BALL_MODEL,
        DEFAULT_OUTPUT as DEFAULT_BALL_OUTPUT,
        DEFAULT_REFERENCE_SAMPLES as DEFAULT_BALL_REFERENCE_SAMPLES,
        BallClassifierConfig,
        HeldBallClassifier,
    )
    from ..rock_paper_scissors.ball_tactile_classifier.common import (
        BALL_SAFE_CLOSED as BALL_CLASSIFIER_CLOSED,
        CLOSE_PHASES as BALL_CLASSIFIER_CLOSE_PHASES,
        CLOSE_STEP_BY_NAME as BALL_CLASSIFIER_CLOSE_STEP_BY_NAME,
        FINGER_NAMES as BALL_CLASSIFIER_FINGER_NAMES,
        force_delta as ball_force_delta,
    )
except ImportError:  # Allow running from this directory with: python lerobot_claw.py
    package_parent = Path(__file__).resolve().parents[2]
    if str(package_parent) not in sys.path:
        sys.path.insert(0, str(package_parent))
    from lerobot_robot_piper.config_piper_rh56f2_follower import PiperRH56F2FollowerConfig
    from lerobot_robot_piper.piper_follower import JOINT_LIMITS_DEG, JOINT_NAMES
    from lerobot_robot_piper.piper_rh56f2_follower import EE_POSE_NAMES, PiperRH56F2Follower
    from lerobot_robot_piper.rh56f2_hand import DEFAULT_CLOSED, DEFAULT_OPEN, HAND_NAMES
    from lerobot_robot_piper.rock_paper_scissors.ball_tactile_classifier.claw_integration import (
        DEFAULT_MODEL as DEFAULT_BALL_MODEL,
        DEFAULT_OUTPUT as DEFAULT_BALL_OUTPUT,
        DEFAULT_REFERENCE_SAMPLES as DEFAULT_BALL_REFERENCE_SAMPLES,
        BallClassifierConfig,
        HeldBallClassifier,
    )
    from lerobot_robot_piper.rock_paper_scissors.ball_tactile_classifier.common import (
        BALL_SAFE_CLOSED as BALL_CLASSIFIER_CLOSED,
        CLOSE_PHASES as BALL_CLASSIFIER_CLOSE_PHASES,
        CLOSE_STEP_BY_NAME as BALL_CLASSIFIER_CLOSE_STEP_BY_NAME,
        FINGER_NAMES as BALL_CLASSIFIER_FINGER_NAMES,
        force_delta as ball_force_delta,
    )


JS_EVENT_FORMAT = "IhBB"
JS_EVENT_SIZE = struct.calcsize(JS_EVENT_FORMAT)
JS_EVENT_BUTTON = 0x01
JS_EVENT_AXIS = 0x02
JS_EVENT_INIT = 0x80

DEFAULT_START_POSE = [282.857, -2.963, 364.752, 172.794, 54.610, 171.120]
DEFAULT_START_JOINTS = [-0.807, 71.692, -65.543, 1.088, 33.928, 3.761]
JOINT_KEYS = [f"{name}.pos" for name in JOINT_NAMES]

BALL_READY_OPEN = dict(DEFAULT_OPEN)
BALL_READY_OPEN.update(
    {
        "little": 1800,
        "ring": 1800,
        "middle": 1800,
        "index": 1800,
        "thumb_bend": 1500,
        "thumb_swing": 1050,
    }
)

BALL_CLOSED = dict(DEFAULT_CLOSED)
BALL_CLOSED.update(
    {
        "little": 1200,
        "ring": 1220,
        "middle": 1350,
        "index": 1350,
        "thumb_bend": 1350,
        "thumb_swing": 1050,
    }
)


def ball_configure_hand(hand: object, speed: int, force: int) -> None:
    hand.write_positions(
        "speedSet",
        {name: float(speed) for name in BALL_CLASSIFIER_FINGER_NAMES},
    )
    hand.write_positions(
        "forceSet",
        {name: float(force) for name in BALL_CLASSIFIER_FINGER_NAMES},
    )
    print(f"hand limits: speed={speed} force={force}")

THUMB_GESTURE = dict(DEFAULT_CLOSED)
THUMB_GESTURE.update({"thumb_bend": 1500, "thumb_swing": 1800})

GRASP_HAND_SPEED = 800
GRASP_HAND_FORCE = 600
GRASP_MAX_FORCE_DELTA = 900.0
GRASP_MAX_CLOSE_DURATION_S = 3.0


class ActionRobot(Protocol):
    """Small LeRobot Robot surface used by this controller."""

    def get_observation(self) -> RobotObservation:
        ...

    def send_action(self, action: RobotAction) -> RobotAction:
        ...


class RawTerminal:
    def __enter__(self) -> "RawTerminal":
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)


class LinuxJoystick:
    """Small reader for /dev/input/js* devices."""

    def __init__(self, device: str):
        self.device = device
        self.fd: int | None = None
        self.axes: dict[int, float] = {}
        self.button_presses: set[int] = set()

    def __enter__(self) -> "LinuxJoystick":
        self.fd = os.open(self.device, os.O_RDONLY | os.O_NONBLOCK)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def read_events(self) -> None:
        if self.fd is None:
            raise RuntimeError("joystick is not open")
        while select.select([self.fd], [], [], 0.0)[0]:
            try:
                data = os.read(self.fd, JS_EVENT_SIZE)
            except BlockingIOError:
                return
            if len(data) != JS_EVENT_SIZE:
                return

            _, value, event_type, number = struct.unpack(JS_EVENT_FORMAT, data)
            if event_type & JS_EVENT_INIT:
                continue
            event_type &= ~JS_EVENT_INIT
            if event_type == JS_EVENT_AXIS:
                self.axes[number] = max(-1.0, min(1.0, value / 32767.0))
            elif event_type == JS_EVENT_BUTTON and value == 1:
                self.button_presses.add(number)

    def discard_events(self) -> None:
        if self.fd is None:
            raise RuntimeError("joystick is not open")
        while select.select([self.fd], [], [], 0.0)[0]:
            try:
                data = os.read(self.fd, JS_EVENT_SIZE)
            except BlockingIOError:
                break
            if len(data) != JS_EVENT_SIZE:
                break
        self.axes.clear()
        self.button_presses.clear()

    def axis(self, number: int, deadzone: float) -> float:
        value = self.axes.get(number, 0.0)
        if abs(value) < deadzone:
            return 0.0
        sign = 1.0 if value > 0.0 else -1.0
        return sign * ((abs(value) - deadzone) / (1.0 - deadzone))

    def pop_button(self, number: int) -> bool:
        if number not in self.button_presses:
            return False
        self.button_presses.remove(number)
        return True


class GamepadStopMonitor:
    """Monitor the stop button without consuming the teleop joystick stream."""

    def __init__(self, device: str, stop_button: int, on_stop: callable) -> None:
        self.device = device
        self.stop_button = stop_button
        self.on_stop = on_stop
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        try:
            with LinuxJoystick(self.device) as joystick:
                while not self.stop_event.is_set():
                    joystick.read_events()
                    if joystick.pop_button(self.stop_button):
                        self.on_stop()
                        return
                    joystick.discard_events()
                    self.stop_event.wait(0.01)
        except OSError as exc:
            print(f"\n[warn] B stop monitor unavailable: {exc}")

    def close(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)


@dataclass
class ClawMachineTaskConfig:
    """Task parameters, using LeRobot action/observation units."""

    grab_z: float
    drop_pose: dict[str, float] | None = None
    start_pose: dict[str, float] = field(default_factory=lambda: pose_from_values(DEFAULT_START_POSE))
    start_joints: list[float] = field(default_factory=lambda: list(DEFAULT_START_JOINTS))
    lift_z: float | None = None
    speed_rate: int = 8
    rate_hz: float = 40.0
    feedback_timeout_s: float = 8.0
    start_duration_s: float = 8.0
    hover_z: float | None = None
    hover_duration_s: float = 6.0
    vertical_duration_s: float = 4.0
    transfer_duration_s: float = 8.0
    return_duration_s: float = 8.0
    soft_arrival: bool = True
    soft_arrival_min_speed: int = 2
    soft_arrival_joint_slow_deg: float = 12.0
    soft_arrival_pose_slow_mm: float = 80.0
    soft_arrival_pose_slow_deg: float = 20.0
    auto_position_tolerance_mm: float = 2.0
    auto_rpy_tolerance_deg: float = 2.0
    j1_step_deg: float = 2.0
    reach_step_deg: float = 2.0
    reach_transition_j2_deg: float = 90.0
    reach_pre_j2_gain: float = 1.0
    reach_pre_j3_gain: float = -0.85
    reach_pre_j5_gain: float = -0.05
    reach_post_j2_gain: float = 1.0
    reach_post_j3_gain: float = -1.2
    reach_post_j5_gain: float = -0.15
    hand_speed: int = 800
    pre_grab_open_speed: int = 1800
    adaptive_close_force_threshold: float = 300.0
    adaptive_close_step_deg: float = 25.0
    adaptive_close_rear_step_deg: float = 35.0
    adaptive_close_settle_s: float = 0.06
    grasp_mode: int = 0
    hand_settle_s: float = 0.0
    pre_grab_open_settle_s: float = 0.0
    drop_open_settle_s: float = 4.0
    held_force_threshold: float = 100.0
    held_force_fingers: list[str] = field(
        default_factory=lambda: list(HAND_NAMES)
    )
    held_force_alt_fingers: list[str] = field(default_factory=list)
    held_force_count: int = 2
    held_required_samples: int = 15
    held_check_duration_s: float = 2.5
    held_check_rate_hz: float = 5.0
    result_gesture: bool = True
    result_gesture_speed: int = 20
    result_gesture_j2_back_deg: float = 30.0
    result_gesture_j6_deg: float = 90.0
    result_thumb_speed: int = 2500
    result_thumb_settle_s: float = 0.0
    result_gesture_duration_s: float = 6.0
    result_gesture_hold_after_s: float = 0.0
    result_gesture_return_duration_s: float = 2.5
    failed_return_hold_s: float = 0.5
    control: str = "keyboard"
    gamepad_device: str = "/dev/input/js0"
    gamepad_deadzone: float = 0.18
    gamepad_axis_x: int = 0
    gamepad_axis_y: int = 1
    gamepad_invert_y: bool = True
    gamepad_j1_speed_dps: float = 8.0
    gamepad_reach_speed_dps: float = 6.0
    gamepad_axis_curve: float = 1.8
    gamepad_print_interval: float = 0.2
    gamepad_lead_limit_deg: float = 1.5
    gamepad_stop_reset: bool = True
    gamepad_pick_button: int = 0
    gamepad_stop_button: int = 1
    ball_classifier_config: BallClassifierConfig | None = None


def pose_from_values(values: list[float]) -> dict[str, float]:
    if len(values) != 6:
        raise ValueError("expected 6 pose values: X,Y,Z,RX,RY,RZ")
    return dict(zip(EE_POSE_NAMES, values, strict=True))


def parse_pose_mm_deg(value: str) -> dict[str, float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 6:
        raise argparse.ArgumentTypeError("expected X,Y,Z,RX,RY,RZ")
    try:
        values = [float(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("pose values must be numbers") from exc
    return pose_from_values(values)


def parse_joint_degrees(value: str) -> list[float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 6:
        raise argparse.ArgumentTypeError("expected J1,J2,J3,J4,J5,J6")
    try:
        values = [float(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("joint values must be numbers") from exc
    if not joint_limits_ok(values):
        raise argparse.ArgumentTypeError("joint target is outside Piper limits")
    return values


def parse_name_list(value: str) -> list[str]:
    names = [part.strip() for part in value.split(",") if part.strip()]
    if not names:
        raise argparse.ArgumentTypeError("expected comma-separated names")
    return names


def read_key(timeout_s: float = 0.1) -> str | None:
    ready, _, _ = select.select([sys.stdin], [], [], timeout_s)
    if not ready:
        return None
    key = sys.stdin.read(1)
    while select.select([sys.stdin], [], [], 0.0)[0]:
        key = sys.stdin.read(1)
    return key


def shaped_axis(value: float, curve: float) -> float:
    if value == 0.0:
        return 0.0
    sign = 1.0 if value > 0.0 else -1.0
    return sign * (abs(value) ** curve)


def joint_limits_ok(joints: list[float]) -> bool:
    return all(
        lo <= value <= hi
        for value, (lo, hi) in zip(joints, JOINT_LIMITS_DEG.values(), strict=True)
    )


def fmt_joints(values: list[float]) -> str:
    return " ".join(f"{value:8.3f}" for value in values)


def key_to_joint_move(key: str) -> tuple[str, int] | None:
    key = key.lower()
    if key == "w":
        return "forward", 0
    if key == "s":
        return "back", 1
    if key == "d":
        return "right", 2
    if key == "a":
        return "left", 3
    return None


def apply_joint_step(
    target: list[float],
    direction: int,
    config: ClawMachineTaskConfig,
    locked_j4: float,
    locked_j5: float,
    locked_j6: float,
) -> tuple[list[float], str]:
    def apply_reach_gains(
        joints: list[float],
        signed_step: float,
        gains: tuple[float, float, float],
    ) -> None:
        joints[1] += signed_step * gains[0]
        joints[2] += signed_step * gains[1]
        joints[4] += signed_step * gains[2]

    def apply_reach(joints: list[float], sign: float) -> str:
        j2 = joints[1]
        reach_step = config.reach_step_deg
        transition = config.reach_transition_j2_deg
        pre_gains = (
            config.reach_pre_j2_gain,
            config.reach_pre_j3_gain,
            config.reach_pre_j5_gain,
        )
        post_gains = (
            config.reach_post_j2_gain,
            config.reach_post_j3_gain,
            config.reach_post_j5_gain,
        )

        label = "pre"
        if sign > 0:
            if j2 < transition:
                pre_j2_delta = min(reach_step * pre_gains[0], transition - j2)
                if pre_j2_delta > 0:
                    pre_step = pre_j2_delta / pre_gains[0]
                    apply_reach_gains(joints, pre_step, pre_gains)
                remaining = reach_step - max(0.0, pre_j2_delta / pre_gains[0])
                if remaining > 1e-6:
                    apply_reach_gains(joints, remaining, post_gains)
                    label = "pre->post"
            else:
                apply_reach_gains(joints, reach_step, post_gains)
                label = "post"
        else:
            if j2 > transition:
                post_j2_delta = min(reach_step * post_gains[0], j2 - transition)
                if post_j2_delta > 0:
                    post_step = post_j2_delta / post_gains[0]
                    apply_reach_gains(joints, -post_step, post_gains)
                remaining = reach_step - max(0.0, post_j2_delta / post_gains[0])
                if remaining > 1e-6:
                    apply_reach_gains(joints, -remaining, pre_gains)
                    label = "post->pre"
                else:
                    label = "post"
            else:
                apply_reach_gains(joints, -reach_step, pre_gains)
                label = "pre"
        return label

    next_target = list(target)
    if direction == 0:
        phase = apply_reach(next_target, 1.0)
    elif direction == 1:
        phase = apply_reach(next_target, -1.0)
        next_target[4] = locked_j5
    elif direction == 2:
        next_target[0] += config.j1_step_deg
        phase = "base"
    elif direction == 3:
        next_target[0] -= config.j1_step_deg
        phase = "base"
    else:
        raise ValueError(f"bad direction: {direction}")

    next_target[3] = locked_j4
    next_target[5] = locked_j6
    return next_target, phase


def velocity_config(
    config: ClawMachineTaskConfig,
    dt_s: float,
    j1_axis: float = 0.0,
    reach_axis: float = 0.0,
) -> ClawMachineTaskConfig:
    local_config = copy(config)
    local_config.j1_step_deg = config.gamepad_j1_speed_dps * dt_s * abs(j1_axis)
    local_config.reach_step_deg = config.gamepad_reach_speed_dps * dt_s * abs(reach_axis)
    return local_config


def clamp_target_lead(target: list[float], actual: list[float], max_lead_deg: float) -> list[float]:
    if max_lead_deg <= 0.0:
        return target
    limited = list(target)
    for idx, actual_value in enumerate(actual):
        limited[idx] = min(
            max(limited[idx], actual_value - max_lead_deg),
            actual_value + max_lead_deg,
        )
    return limited


def restore_locked_wrist(target: list[float], locked_j4: float, locked_j6: float) -> list[float]:
    locked = list(target)
    locked[3] = locked_j4
    locked[5] = locked_j6
    return locked


class ClawMachineController:
    def __init__(self, robot: ActionRobot, config: ClawMachineTaskConfig):
        self.robot = robot
        self.config = config
        self._emergency_stop = threading.Event()
        self._close_peak_active_count = 0
        self._close_peak_names: set[str] = set()
        self.ball_classifier = (
            HeldBallClassifier(config.ball_classifier_config)
            if config.ball_classifier_config is not None
            else None
        )

    def request_emergency_stop(self) -> None:
        if not self._emergency_stop.is_set():
            self._emergency_stop.set()
            print("\n[B] Emergency stop requested: holding current position.")

    def emergency_stop_requested(self) -> bool:
        return self._emergency_stop.is_set()

    def wait_with_stop(self, duration_s: float) -> bool:
        return not self._emergency_stop.wait(max(duration_s, 0.0))

    def hold_current_position(self) -> None:
        try:
            current = self.current_joints()
            if self.send_joint_once(current):
                print(f"Emergency stop: holding current joints {fmt_joints(current)}")
        except Exception as exc:
            print(f"[warn] emergency hold failed: {exc}")

    def observation(self) -> RobotObservation:
        return self.robot.get_observation()

    def raw_hand(self) -> object | None:
        return getattr(self.robot, "hand", None)

    def current_pose(self) -> dict[str, float]:
        obs = self.observation()
        return {name: float(obs[name]) for name in EE_POSE_NAMES}

    def current_joints(self) -> list[float]:
        obs = self.observation()
        return [float(obs[key]) for key in JOINT_KEYS]

    def joint_action(self, joints: list[float], speed: int | None = None) -> RobotAction:
        action: RobotAction = {key: value for key, value in zip(JOINT_KEYS, joints, strict=True)}
        action["arm.speed_rate"] = float(self.config.speed_rate if speed is None else speed)
        return action

    def ee_action(self, pose: dict[str, float], speed: int | None = None) -> RobotAction:
        action: RobotAction = dict(pose)
        action["arm.speed_rate"] = float(self.config.speed_rate if speed is None else speed)
        return action

    def hand_pose_action(self, pose: dict[str, float]) -> RobotAction:
        return {f"hand.{name}.pos": value for name, value in pose.items()}

    def set_hand_pose(self, pose: dict[str, float], label: str) -> bool:
        self.robot.send_action(self.hand_pose_action(pose))
        print(f"{label}: hand command sent")
        return True

    def set_hand_speed(self, speed: int | None, label: str) -> bool:
        if speed is None:
            return True
        action = {f"hand.{name}.speed": float(speed) for name in HAND_NAMES}
        self.robot.send_action(action)
        print(f"{label}: hand speed set to {speed}")
        return True

    def set_hand_force(self, force: int | None, label: str) -> bool:
        if force is None:
            return True
        action = {f"hand.{name}.force_limit": float(force) for name in HAND_NAMES}
        self.robot.send_action(action)
        print(f"{label}: hand force set to {force}")
        return True

    def set_hand_mode(self, mode: int) -> None:
        self.robot.send_action({"hand.mode": float(mode)})
        print(f"grasp mode: {mode}")

    def send_joint_once(self, target: list[float], speed: int | None = None) -> bool:
        if not joint_limits_ok(target):
            print(f"\n[warn] MOVE_J target outside limits: {fmt_joints(target)}")
            return False
        self.robot.send_action(self.joint_action(target, speed))
        return True

    def soft_speed_from_ratio(self, base_speed: int, ratio: float) -> int:
        if not self.config.soft_arrival:
            return base_speed
        min_speed = min(base_speed, max(1, int(self.config.soft_arrival_min_speed)))
        ratio = max(0.0, min(1.0, ratio))
        return int(round(min_speed + (base_speed - min_speed) * ratio))

    def joint_soft_speed(self, target: list[float], base_speed: int) -> int:
        current = self.current_joints()
        j1_j2_error = max(
            abs(current[0] - target[0]),
            abs(current[1] - target[1]),
        )
        ratio = j1_j2_error / max(self.config.soft_arrival_joint_slow_deg, 1e-6)
        return self.soft_speed_from_ratio(base_speed, ratio)

    def pose_soft_speed(self, pose: dict[str, float], base_speed: int) -> int:
        actual = self.current_pose()
        xyz_error, rpy_error = self.pose_error_mm_deg(actual, pose)
        ratio = max(
            xyz_error / max(self.config.soft_arrival_pose_slow_mm, 1e-6),
            rpy_error / max(self.config.soft_arrival_pose_slow_deg, 1e-6),
        )
        return self.soft_speed_from_ratio(base_speed, ratio)

    def move_joints_for(
        self,
        target: list[float],
        speed: int,
        duration_s: float,
        label: str,
        soft_arrival: bool = True,
    ) -> bool:
        deadline = time.time() + duration_s
        interval_s = 1.0 / self.config.rate_hz
        while time.time() < deadline:
            if self.emergency_stop_requested():
                self.hold_current_position()
                return False
            command_speed = self.joint_soft_speed(target, speed) if soft_arrival else speed
            if not self.send_joint_once(target, command_speed):
                return False
            if not self.wait_with_stop(interval_s):
                self.hold_current_position()
                return False
        print(f"{label}: joints {fmt_joints(target)}")
        return True

    def move_joints_until_reached(
        self,
        target: list[float],
        speed: int,
        max_duration_s: float,
        hold_after_reached_s: float,
        label: str,
        tolerance_deg: float = 1.0,
    ) -> bool:
        deadline = time.time() + max_duration_s
        hold_deadline: float | None = None
        interval_s = 1.0 / self.config.rate_hz
        while time.time() < deadline:
            if self.emergency_stop_requested():
                self.hold_current_position()
                return False
            command_speed = self.joint_soft_speed(target, speed)
            if not self.send_joint_once(target, command_speed):
                return False
            current = self.current_joints()
            reached = all(
                abs(current[index] - target[index]) <= tolerance_deg
                for index in range(min(len(current), len(target)))
            )
            if reached:
                if hold_deadline is None:
                    hold_deadline = time.time() + hold_after_reached_s
                elif time.time() >= hold_deadline:
                    print(f"{label}: joints {fmt_joints(target)}")
                    return True
            else:
                hold_deadline = None
            if not self.wait_with_stop(interval_s):
                self.hold_current_position()
                return False
        print(f"{label}: joints {fmt_joints(target)}")
        return True

    def pose_error_mm_deg(
        self,
        actual: dict[str, float],
        target: dict[str, float],
    ) -> tuple[float, float]:
        xyz_error = max(abs(actual[name] - target[name]) for name in EE_POSE_NAMES[:3])
        rpy_error = max(abs(actual[name] - target[name]) for name in EE_POSE_NAMES[3:])
        return xyz_error, rpy_error

    def move_ee_for(
        self,
        pose: dict[str, float],
        duration_s: float,
        label: str,
        require_reached: bool = False,
    ) -> bool:
        interval_s = 1.0 / self.config.rate_hz
        deadline = time.time() + duration_s
        count = 0
        while time.time() < deadline:
            if self.emergency_stop_requested():
                self.hold_current_position()
                return False
            command_speed = self.pose_soft_speed(pose, self.config.speed_rate)
            self.robot.send_action(self.ee_action(pose, command_speed))
            if not self.wait_with_stop(interval_s):
                self.hold_current_position()
                return False

            if count % max(1, int(self.config.rate_hz / 2)) == 0:
                actual = self.current_pose()
                xyz_error, rpy_error = self.pose_error_mm_deg(actual, pose)
                print(
                    f"\r{label}: xyz_err={xyz_error:.3f}mm "
                    f"rpy_err={rpy_error:.3f}deg pose=[{self.format_pose(actual)}]",
                    end="",
                    flush=True,
                )
                if (
                    xyz_error <= self.config.auto_position_tolerance_mm
                    and rpy_error <= self.config.auto_rpy_tolerance_deg
                ):
                    print()
                    return True
            count += 1

        actual = self.current_pose()
        xyz_error, rpy_error = self.pose_error_mm_deg(actual, pose)
        reached = (
            xyz_error <= self.config.auto_position_tolerance_mm
            and rpy_error <= self.config.auto_rpy_tolerance_deg
        )
        state = "done" if reached or not require_reached else "timeout"
        print(
            f"{label} {state}: xyz_err={xyz_error:.3f}mm rpy_err={rpy_error:.3f}deg "
            f"pose=[{self.format_pose(actual)}]"
        )
        return reached or not require_reached

    def held_by_force(self) -> bool:
        required_duration_s = self.config.held_check_duration_s
        interval_s = 1.0 / self.config.held_check_rate_hz
        check_started = time.monotonic()
        # The required duration is measured from the first qualifying sample.
        # Leave enough total time for pressure to become stable after lifting.
        deadline = check_started + max(required_duration_s * 2.0, required_duration_s + 1.0)
        active_since: float | None = None
        active_samples = 0
        best_duration_s = 0.0
        best_samples = 0
        tracked_names = sorted(
            set(self.config.held_force_fingers) | set(self.config.held_force_alt_fingers)
        )
        last_active = {name: 0.0 for name in tracked_names}
        best_group = ""

        def qualifies(active_names: list[str]) -> bool:
            thumb_active = any(
                name in {"thumb_bend", "thumb_swing"} for name in active_names
            )
            return (
                len(active_names) >= self.config.held_force_count
                and thumb_active
                and self._close_peak_active_count >= 3
            )

        while time.monotonic() < deadline:
            if self.emergency_stop_requested():
                self.hold_current_position()
                return False
            obs = self.observation()
            sample_time = time.monotonic()
            last_active = {
                name: abs(float(obs.get(f"hand.{name}.force", 0.0)))
                for name in tracked_names
            }
            active_names = [
                name
                for name in tracked_names
                if last_active.get(name, 0.0) >= self.config.held_force_threshold
            ]
            thumb_active = any(
                name in {"thumb_bend", "thumb_swing"} for name in active_names
            )
            if qualifies(active_names):
                if active_since is None:
                    active_since = sample_time
                    active_samples = 0
                active_samples += 1
                active_duration_s = sample_time - active_since
                if active_duration_s > best_duration_s:
                    best_duration_s = active_duration_s
                    best_samples = active_samples
                best_group = ",".join(active_names)
                if active_duration_s >= required_duration_s:
                    force_text = " ".join(
                        f"{name}={value:.1f}" for name, value in last_active.items()
                    )
                    print(
                        f"held check: {force_text}, threshold="
                        f"{self.config.held_force_threshold:.1f}, "
                        f"active={len(active_names)}/{self.config.held_force_count}, "
                        f"thumb_active={thumb_active}, "
                        f"close_peak={self._close_peak_active_count}/3, "
                        f"continuous={active_duration_s:.2f}s/"
                        f"{required_duration_s:.2f}s, samples={active_samples}, "
                        f"group={best_group}, held=True"
                    )
                    return True
            else:
                active_since = None
                active_samples = 0
            if not self.wait_with_stop(interval_s):
                self.hold_current_position()
                return False

        # The final sample may arrive one control interval before the timeout.
        # If the final reading is still valid, include that interval in the
        # continuous-pressure duration instead of dropping it at the deadline.
        try:
            final_obs = self.observation()
            final_time = time.monotonic()
            last_active = {
                name: abs(float(final_obs.get(f"hand.{name}.force", 0.0)))
                for name in tracked_names
            }
            final_active_names = [
                name
                for name in tracked_names
                if last_active.get(name, 0.0) >= self.config.held_force_threshold
            ]
            final_thumb_active = any(
                name in {"thumb_bend", "thumb_swing"}
                for name in final_active_names
            )
            if (
                active_since is not None
                and qualifies(final_active_names)
            ):
                best_duration_s = max(best_duration_s, final_time - active_since)
                best_group = ",".join(final_active_names)
        except Exception:
            pass

        held = best_duration_s >= required_duration_s
        force_text = " ".join(f"{name}={value:.1f}" for name, value in last_active.items())
        print(
            f"held check: {force_text}, threshold={self.config.held_force_threshold:.1f}, "
            f"active_count={sum(value >= self.config.held_force_threshold for value in last_active.values())}/"
            f"{self.config.held_force_count}, "
            f"close_peak={self._close_peak_active_count}/3, "
            f"best_continuous={best_duration_s:.2f}s/{required_duration_s:.2f}s, "
            f"samples={best_samples}, best_group={best_group or 'none'}, held={held}"
        )
        return held

    def close_for_teleop(self) -> None:
        self.set_hand_speed(self.config.hand_speed, "teleop close speed")
        self.set_hand_pose(DEFAULT_CLOSED, "close for teleop")

    def open_while_descending(self) -> None:
        self.set_hand_speed(self.config.pre_grab_open_speed, "fast open while descending")
        self.set_hand_pose(BALL_READY_OPEN, "wide open and swing thumb inward while descending")

    def close_at_grab(self) -> bool:
        self.set_hand_speed(self.config.hand_speed, "restore close speed")
        return self.set_hand_pose(BALL_CLOSED, "close ball grasp")

    def close_at_grab_adaptive(
        self,
        ball_trial: object | None = None,
        ball_hand: object | None = None,
    ) -> bool:
        if self.config.grasp_mode != 0:
            print(
                f"[warn] grasp mode {self.config.grasp_mode} is disabled for the stable path; "
                "using hand mode 0"
            )
        return self._close_at_grab_fixed(ball_trial, ball_hand)

    def _close_at_grab_fixed(
        self,
        ball_trial: object | None = None,
        ball_hand: object | None = None,
    ) -> bool:
        if ball_hand is not None:
            ball_configure_hand(ball_hand, GRASP_HAND_SPEED, GRASP_HAND_FORCE)
        else:
            self.set_hand_speed(GRASP_HAND_SPEED, "adaptive close speed")
            self.set_hand_force(GRASP_HAND_FORCE, "adaptive close force")
        baseline_obs = self.observation()
        self._close_peak_names = set()
        self._close_peak_active_count = 0
        current_target = {
            name: float(baseline_obs[f"hand.{name}.pos"])
            for name in BALL_CLASSIFIER_FINGER_NAMES
        }
        contacted: set[str] = set()
        contact_threshold = (
            self.ball_classifier.config.contact_threshold
            if self.ball_classifier is not None
            else self.config.held_force_threshold
        )
        print(
            "Ball grasp close: predict_live rhythm "
            "phases=little_thumb/ring_thumb/middle_index, "
            "offsets=0.00/0.15/0.30s, step_settle=0.06s, speed=800"
        )

        started_at = time.monotonic()
        while True:
            if self.emergency_stop_requested():
                self.hold_current_position()
                return False

            elapsed_s = time.monotonic() - started_at
            action_values: dict[str, float] = {}
            for _, names, offset_s in BALL_CLASSIFIER_CLOSE_PHASES:
                if elapsed_s < offset_s:
                    continue
                for name in names:
                    goal = BALL_CLASSIFIER_CLOSED[name]
                    if current_target[name] > goal:
                        current_target[name] = max(
                            goal,
                            current_target[name] - BALL_CLASSIFIER_CLOSE_STEP_BY_NAME[name],
                        )
                        action_values[name] = current_target[name]

            if action_values:
                if ball_hand is not None and hasattr(ball_hand, "set_angles"):
                    ball_hand.set_angles(action_values)
                else:
                    self.robot.send_action(
                        {f"hand.{name}.pos": value for name, value in action_values.items()}
                    )
            if not self.wait_with_stop(self.config.adaptive_close_settle_s):
                self.hold_current_position()
                return False

            frame = None
            if self.ball_classifier is not None and ball_trial is not None and ball_hand is not None:
                try:
                    frame = self.ball_classifier.record_grasp_frame(ball_hand, ball_trial, started_at)
                except Exception as exc:
                    print(f"\n[warn] grasp frame read failed: {exc}")
            if frame is not None:
                force_text = []
                for name in BALL_CLASSIFIER_FINGER_NAMES:
                    delta = ball_force_delta(frame.forces, ball_trial.baseline_forces, name)
                    force_text.append(f"{name}={delta:.0f}")
                    if delta >= contact_threshold:
                        contacted.add(name)
                max_delta = max(
                    ball_force_delta(frame.forces, ball_trial.baseline_forces, name)
                    for name in BALL_CLASSIFIER_FINGER_NAMES
                )
                print(
                    "\rclose contact "
                    f"{len(contacted)}/6 "
                    + " ".join(force_text),
                    end="",
                    flush=True,
                )
            else:
                obs = self.observation()
                force_text = []
                for name in BALL_CLASSIFIER_FINGER_NAMES:
                    force = abs(float(obs.get(f"hand.{name}.force", 0.0)))
                    force_text.append(f"{name}={force:.0f}")
                    if force >= contact_threshold:
                        contacted.add(name)
                max_delta = max(
                    abs(float(obs.get(f"hand.{name}.force", 0.0)))
                    for name in BALL_CLASSIFIER_FINGER_NAMES
                )
                print(
                    "\rclose contact "
                    f"{len(contacted)}/6 "
                    + " ".join(force_text),
                    end="",
                    flush=True,
                )

            self._close_peak_names.update(contacted)
            self._close_peak_active_count = len(self._close_peak_names)
            if max_delta >= GRASP_MAX_FORCE_DELTA:
                print("\nmax close force reached; stopping close.")
                break
            all_goals_reached = all(
                current_target[name] <= BALL_CLASSIFIER_CLOSED[name]
                for name in BALL_CLASSIFIER_FINGER_NAMES
            )
            if elapsed_s >= GRASP_MAX_CLOSE_DURATION_S:
                print(f"\nmax close duration reached: {GRASP_MAX_CLOSE_DURATION_S:.2f}s")
                break
            if all_goals_reached:
                break

        print()
        print(
            "Ball grasp close complete; fixed sequence finished. "
            f"close_peak={self._close_peak_active_count}/6 "
            f"({','.join(sorted(self._close_peak_names)) or 'none'})"
        )
        return True

    def _close_at_grab_feedback(self) -> bool:
        self.set_hand_speed(GRASP_HAND_SPEED, "adaptive close speed")
        self.set_hand_force(GRASP_HAND_FORCE, "adaptive close force")
        baseline_obs = self.observation()
        self._close_peak_names = set()
        self._close_peak_active_count = 0
        current_target = {
            name: float(baseline_obs[f"hand.{name}.pos"])
            for name in HAND_NAMES
        }
        phases = [
            ("little + thumb swing", {"little": 1200.0, "thumb_swing": 900.0}, 0.00),
            ("ring + thumb bend", {"ring": 1220.0, "thumb_bend": 1350.0}, 0.15),
            ("middle + index", {"middle": 1350.0, "index": 1350.0}, 0.30),
        ]
        steps = {
            "little": 120.0,
            "ring": 70.0,
            "middle": 60.0,
            "index": 60.0,
            "thumb_bend": 50.0,
            "thumb_swing": 150.0,
        }
        target_force = 300.0 if self.config.grasp_mode == 1 else 220.0
        correction = 12.0 if self.config.grasp_mode == 1 else 6.0
        force_band = 50.0 if self.config.grasp_mode == 1 else 70.0
        label = "force closed-loop" if self.config.grasp_mode == 1 else "impedance"
        print(
            f"Ball grasp close: mode={self.config.grasp_mode} ({label}), "
            f"target_force={target_force:.0f}, correction={correction:.0f}"
        )
        started_at = time.monotonic()
        started_phases: set[str] = set()
        completed_phases: set[str] = set()
        while len(completed_phases) < len(phases):
            if self.emergency_stop_requested():
                self.hold_current_position()
                return False
            elapsed = time.monotonic() - started_at
            action: RobotAction = {}
            active_names: list[str] = []
            for phase_name, goals, phase_offset in phases:
                if elapsed < phase_offset or phase_name in completed_phases:
                    continue
                if phase_name not in started_phases:
                    started_phases.add(phase_name)
                    print(f"Grasp phase started: {phase_name}")
                remaining = False
                for name, goal in goals.items():
                    force = abs(float(self.observation().get(f"hand.{name}.force", 0.0)))
                    if self.config.grasp_mode == 1 and force >= self.config.adaptive_close_force_threshold:
                        continue
                    if current_target[name] > goal:
                        remaining = True
                        current_target[name] = max(goal, current_target[name] - steps[name])
                    action[f"hand.{name}.pos"] = current_target[name]
                    active_names.append(name)
                if not remaining:
                    completed_phases.add(phase_name)
                    print(f"Grasp phase complete: {phase_name}")
            if action:
                self.robot.send_action(action)
                obs = self.observation()
                for name in HAND_NAMES:
                    force = abs(float(obs.get(f"hand.{name}.force", 0.0)))
                    if force >= self.config.adaptive_close_force_threshold:
                        self._close_peak_names.add(name)
                self._close_peak_active_count = len(self._close_peak_names)
                if self.config.grasp_mode == 2:
                    for name in active_names:
                        force = abs(float(obs.get(f"hand.{name}.force", 0.0)))
                        if force > target_force + force_band:
                            current_target[name] = min(current_target[name] + correction, 1800.0)
                        elif force < target_force - force_band:
                            current_target[name] = max(current_target[name] - correction, 850.0)
            if not self.wait_with_stop(self.config.adaptive_close_settle_s):
                self.hold_current_position()
                return False
        maintain_until = time.monotonic() + 1.0
        while time.monotonic() < maintain_until:
            if self.emergency_stop_requested():
                self.hold_current_position()
                return False
            obs = self.observation()
            action = {}
            for name in HAND_NAMES:
                force = abs(float(obs.get(f"hand.{name}.force", 0.0)))
                if force > target_force + force_band:
                    current_target[name] = min(current_target[name] + correction, 1800.0)
                elif force < target_force - force_band:
                    current_target[name] = max(current_target[name] - correction, 850.0)
                action[f"hand.{name}.pos"] = current_target[name]
            self.robot.send_action(action)
            if not self.wait_with_stop(self.config.adaptive_close_settle_s):
                self.hold_current_position()
                return False
        print(
            f"Ball grasp close complete; mode={self.config.grasp_mode}, "
            f"close_peak={self._close_peak_active_count}/6"
        )
        return True

    def open_at_drop(self) -> bool:
        return self.set_hand_pose(DEFAULT_OPEN, "open")

    def close_while_returning(self) -> None:
        self.set_hand_pose(DEFAULT_CLOSED, "close while returning")

    def show_thumb_gesture(self) -> bool:
        self.set_hand_speed(self.config.result_thumb_speed, "thumb gesture speed")
        return self.set_hand_pose(THUMB_GESTURE, "thumb gesture")

    def run_result_gesture(self, held: bool) -> bool:
        if not self.config.result_gesture:
            return True

        original = self.current_joints()
        gesture = list(original)
        gesture[1] -= self.config.result_gesture_j2_back_deg
        gesture[5] += (
            self.config.result_gesture_j6_deg
            if held
            else -self.config.result_gesture_j6_deg
        )
        label = "success thumbs-up" if held else "empty thumbs-down"

        print(f"Result gesture: {label}")
        print(f"  original joints: {fmt_joints(original)}")
        print(f"  gesture joints:  {fmt_joints(gesture)}")

        self.show_thumb_gesture()
        if self.config.result_thumb_settle_s > 0:
            if not self.wait_with_stop(self.config.result_thumb_settle_s):
                self.hold_current_position()
                return False

        if not self.move_joints_for(
            gesture,
            self.config.result_gesture_speed,
            self.config.result_gesture_duration_s,
            label,
            soft_arrival=False,
        ):
            return False
        if self.config.result_gesture_hold_after_s > 0:
            if not self.move_joints_for(
                gesture,
                self.config.result_gesture_speed,
                self.config.result_gesture_hold_after_s,
                "result gesture hold",
                soft_arrival=False,
            ):
                return False
        if not self.move_joints_for(
            original,
            self.config.result_gesture_speed,
            self.config.result_gesture_return_duration_s,
            "result gesture return",
            soft_arrival=False,
        ):
            return False
        self.close_while_returning()
        return True

    def move_to_start_and_hover(self) -> tuple[dict[str, float], dict[str, float]]:
        start_pose = dict(self.config.start_pose)
        print("Selecting LeRobot joint_*.pos MOVE_J action for initial move...")
        print(f"Moving to configured start joints: {fmt_joints(self.config.start_joints)}")
        if not self.move_joints_for(
            self.config.start_joints,
            self.config.speed_rate,
            self.config.start_duration_s,
            "start MOVE_J",
        ):
            raise RuntimeError("start move failed")

        hover_pose = self.current_pose()
        if self.config.hover_z is not None:
            hover_pose = dict(hover_pose)
            hover_pose["ee.z"] = self.config.hover_z
            print(f"Moving to keyboard/gamepad hover Z: {self.format_pose(hover_pose)}")
            if not self.move_ee_for(hover_pose, self.config.hover_duration_s, "hover"):
                raise RuntimeError("hover move failed")
        return start_pose, hover_pose

    def run_pick_cycle(
        self,
        start_pose: dict[str, float],
        hover_pose: dict[str, float],
        drop_pose: dict[str, float],
    ) -> bool:
        grab_pose = dict(hover_pose)
        grab_pose["ee.z"] = self.config.grab_z
        lift_pose = dict(hover_pose)
        if self.config.lift_z is not None:
            lift_pose["ee.z"] = self.config.lift_z

        print()
        print("Running LeRobot pick cycle")
        print(f"  hover: {self.format_pose(hover_pose)}")
        print(f"  grab:  {self.format_pose(grab_pose)}")
        print(f"  lift:  {self.format_pose(lift_pose)}")
        print(f"  drop:  {self.format_pose(drop_pose)}")
        print(f"  start: {self.format_pose(start_pose)}")

        self.open_while_descending()
        if self.emergency_stop_requested():
            self.hold_current_position()
            return False
        if not self.move_ee_for(
            grab_pose,
            self.config.vertical_duration_s,
            "descend",
            require_reached=True,
        ):
            print("[warn] descend failed")
            return False

        if self.config.pre_grab_open_settle_s > 0:
            if not self.wait_with_stop(self.config.pre_grab_open_settle_s):
                self.hold_current_position()
                return False
        if self.emergency_stop_requested():
            self.hold_current_position()
            return False
        ball_trial = None
        ball_hand = self.raw_hand()
        if self.ball_classifier is not None:
            if ball_hand is None:
                print("[warn] ball classifier unavailable: robot has no raw RH56F2 hand handle.")
            else:
                try:
                    ball_trial = self.ball_classifier.begin_trial(ball_hand)
                    print("ball classifier: baseline captured before closing.")
                except Exception as exc:
                    print(f"[warn] ball classifier baseline failed: {exc}")
        if not self.close_at_grab_adaptive(ball_trial, ball_hand):
            print("[warn] adaptive close failed")
            return False
        print("Grasp close complete; lifting immediately to check hover.")

        if not self.move_ee_for(
            lift_pose,
            self.config.vertical_duration_s,
            "lift",
            require_reached=True,
        ):
            print("[warn] lift failed")
            return False

        if self.ball_classifier is not None and ball_trial is not None and ball_hand is not None:
            try:
                classification_row = self.ball_classifier.classify_held(ball_hand, ball_trial)
                predicted_label = str(classification_row.get("predicted_label") or "")
                held_at_lift = predicted_label not in {"", "NONE"}
            except Exception as exc:
                print(f"[warn] ball classifier failed: {exc}")
                held_at_lift = False
        else:
            held_at_lift = self.held_by_force()
        if self.emergency_stop_requested():
            self.hold_current_position()
            return False

        if not held_at_lift:
            print("Grasp not confirmed at lift; skipping drop and returning to start MOVE_J.")
            if not self.move_joints_until_reached(
                self.config.start_joints,
                self.config.speed_rate,
                self.config.return_duration_s,
                self.config.failed_return_hold_s,
                "return MOVE_J",
            ):
                print("[warn] return MOVE_J failed")
                return False
            if not self.run_result_gesture(False):
                print("[warn] result gesture failed")
                return False
            return True

        if not self.move_ee_for(
            drop_pose,
            self.config.transfer_duration_s,
            "drop move",
            require_reached=True,
        ):
            print("[warn] drop move failed")
            return False

        held_at_drop = held_at_lift
        self.open_at_drop()
        if not self.wait_with_stop(self.config.drop_open_settle_s):
            self.hold_current_position()
            return False

        self.close_while_returning()
        if not self.move_ee_for(
            start_pose,
            self.config.return_duration_s,
            "return",
            require_reached=True,
        ):
            print("[warn] return failed")
            return False
        if not self.run_result_gesture(held_at_drop):
            print("[warn] result gesture failed")
            return False
        return True

    def capture_teleop_reference(self) -> tuple[list[float], float, float, float]:
        joint_target = self.current_joints()
        return joint_target, joint_target[3], joint_target[4], joint_target[5]

    def print_state(self) -> None:
        print()
        print(f"pose:   {self.format_pose(self.current_pose())}")
        print(f"joints: {fmt_joints(self.current_joints())}")

    def run_keyboard_loop(
        self,
        start_pose: dict[str, float],
        keyboard_pose: dict[str, float],
    ) -> None:
        joint_target, locked_j4, locked_j5, locked_j6 = self.capture_teleop_reference()
        drop_pose = dict(self.config.drop_pose or start_pose)

        print(f"Captured start pose: {self.format_pose(start_pose)}")
        print(f"Keyboard pose:       {self.format_pose(keyboard_pose)}")
        print(f"Keyboard joints:     {fmt_joints(joint_target)}")
        print(
            "Locked joints:       "
            f"J4={locked_j4:.3f} J5-back={locked_j5:.3f} J6={locked_j6:.3f}"
        )
        print(f"Drop pose:           {self.format_pose(drop_pose)}")
        self.close_for_teleop()
        print("Keyboard control is active.")
        print("Use WASD, space to grab, P to print, R to reset, Q to quit.")

        with RawTerminal():
            while True:
                key = read_key(0.1)
                if key is None:
                    continue

                key_lower = key.lower()
                if key_lower == "q":
                    print("\nquit")
                    break
                if key_lower == "p":
                    self.print_state()
                    continue
                if key_lower == "r":
                    joint_target, locked_j4, locked_j5, locked_j6 = (
                        self.capture_teleop_reference()
                    )
                    print(f"\nreference reset joints: {fmt_joints(joint_target)}")
                    continue
                if key == " ":
                    hover_pose = self.current_pose()
                    ok = self.run_pick_cycle(start_pose, hover_pose, drop_pose)
                    joint_target, locked_j4, locked_j5, locked_j6 = (
                        self.capture_teleop_reference()
                    )
                    print(
                        f"cycle {'complete' if ok else 'stopped'}; "
                        f"current joints={fmt_joints(joint_target)}"
                    )
                    continue

                move = key_to_joint_move(key)
                if move is None:
                    print(f"\nignored key: {key!r}")
                    continue

                label, direction = move
                joint_target, phase = apply_joint_step(
                    joint_target,
                    direction,
                    self.config,
                    locked_j4,
                    locked_j5,
                    locked_j6,
                )
                print(
                    f"\rkey {label} [{phase}]: target joints {fmt_joints(joint_target)}",
                    end="",
                    flush=True,
                )
                if not self.send_joint_once(joint_target):
                    print("[warn] nudge failed; stop sending commands and reset if needed.")
                    break

    def run_gamepad_loop(
        self,
        start_pose: dict[str, float],
        keyboard_pose: dict[str, float],
    ) -> None:
        joint_target, locked_j4, locked_j5, locked_j6 = self.capture_teleop_reference()
        drop_pose = dict(self.config.drop_pose or start_pose)

        print(f"Captured start pose: {self.format_pose(start_pose)}")
        print(f"Keyboard/gamepad pose: {self.format_pose(keyboard_pose)}")
        print(f"Gamepad start joints: {fmt_joints(joint_target)}")
        self.print_gamepad_help()
        self.close_for_teleop()

        with LinuxJoystick(self.config.gamepad_device) as joystick:
            last_loop = time.monotonic()
            last_print = 0.0
            was_moving = False
            while True:
                if self.emergency_stop_requested():
                    self.hold_current_position()
                    return
                now = time.monotonic()
                dt_s = min(max(now - last_loop, 0.0), 0.1)
                last_loop = now

                joystick.read_events()

                if joystick.pop_button(self.config.gamepad_stop_button):
                    print(f"\ngamepad button {self.config.gamepad_stop_button}: emergency stop")
                    self.request_emergency_stop()
                    self.hold_current_position()
                    return
                if joystick.pop_button(3):
                    print("\ngamepad button 3: print state")
                    self.print_state()
                if joystick.pop_button(2):
                    print("\ngamepad button 2: reset joint reference")
                    joint_target, locked_j4, locked_j5, locked_j6 = (
                        self.capture_teleop_reference()
                    )
                    print(f"\nreference reset joints: {fmt_joints(joint_target)}")

                if joystick.pop_button(self.config.gamepad_pick_button):
                    print(f"\ngamepad button {self.config.gamepad_pick_button}: pick cycle")
                    joystick.discard_events()
                    try:
                        hover_pose = self.current_pose()
                        ok = self.run_pick_cycle(start_pose, hover_pose, drop_pose)
                    finally:
                        joystick.discard_events()
                    if self.emergency_stop_requested():
                        self.hold_current_position()
                        return
                    joint_target, _, locked_j5, _ = self.capture_teleop_reference()
                    joint_target = restore_locked_wrist(joint_target, locked_j4, locked_j6)
                    if not self.send_joint_once(joint_target):
                        print("[warn] failed to restore locked J4/J6 after returning to start.")
                        break
                    print(
                        f"cycle {'complete' if ok else 'stopped'}; "
                        f"current joints={fmt_joints(joint_target)} "
                        f"locked J4={locked_j4:.3f} J6={locked_j6:.3f}"
                    )

                x_axis = shaped_axis(
                    joystick.axis(self.config.gamepad_axis_x, self.config.gamepad_deadzone),
                    self.config.gamepad_axis_curve,
                )
                y_axis = shaped_axis(
                    joystick.axis(self.config.gamepad_axis_y, self.config.gamepad_deadzone),
                    self.config.gamepad_axis_curve,
                )

                moved = False
                labels = []
                phases = []
                if x_axis > 0:
                    local_config = velocity_config(self.config, dt_s, j1_axis=x_axis)
                    joint_target, phase = apply_joint_step(
                        joint_target, 2, local_config, locked_j4, locked_j5, locked_j6
                    )
                    labels.append(f"right {abs(x_axis):.2f}")
                    phases.append(phase)
                    moved = True
                elif x_axis < 0:
                    local_config = velocity_config(self.config, dt_s, j1_axis=x_axis)
                    joint_target, phase = apply_joint_step(
                        joint_target, 3, local_config, locked_j4, locked_j5, locked_j6
                    )
                    labels.append(f"left {abs(x_axis):.2f}")
                    phases.append(phase)
                    moved = True

                reach_axis = -y_axis if self.config.gamepad_invert_y else y_axis
                if reach_axis > 0:
                    local_config = velocity_config(self.config, dt_s, reach_axis=reach_axis)
                    joint_target, phase = apply_joint_step(
                        joint_target, 0, local_config, locked_j4, locked_j5, locked_j6
                    )
                    labels.append(f"forward {abs(reach_axis):.2f}")
                    phases.append(phase)
                    moved = True
                elif reach_axis < 0:
                    local_config = velocity_config(self.config, dt_s, reach_axis=reach_axis)
                    joint_target, phase = apply_joint_step(
                        joint_target, 1, local_config, locked_j4, locked_j5, locked_j6
                    )
                    labels.append(f"back {abs(reach_axis):.2f}")
                    phases.append(phase)
                    moved = True

                if moved:
                    actual_joints = self.current_joints()
                    joint_target = clamp_target_lead(
                        joint_target,
                        actual_joints,
                        self.config.gamepad_lead_limit_deg,
                    )
                    if now - last_print >= self.config.gamepad_print_interval:
                        print(
                            f"\rgamepad {'+'.join(labels)} [{'/'.join(phases)}]: "
                            f"joints {fmt_joints(joint_target)}",
                            end="",
                            flush=True,
                        )
                        last_print = now
                    if not self.send_joint_once(joint_target):
                        print("[warn] gamepad nudge failed; stop sending commands and reset if needed.")
                        break
                    was_moving = True
                elif was_moving and self.config.gamepad_stop_reset:
                    joint_target = self.current_joints()
                    if not self.send_joint_once(joint_target):
                        print("[warn] gamepad stop failed; reset if needed.")
                        break
                    print(
                        f"\rgamepad stop: holding current joints {fmt_joints(joint_target)}",
                        end="",
                        flush=True,
                    )
                    was_moving = False

                select.select([], [], [], 1.0 / self.config.rate_hz)

    def run(self) -> None:
        self.print_state()
        if self.emergency_stop_requested():
            self.hold_current_position()
            return
        start_pose, keyboard_pose = self.move_to_start_and_hover()
        if self.emergency_stop_requested():
            self.hold_current_position()
            return
        if self.config.control == "once":
            hover_pose = self.current_pose()
            drop_pose = dict(self.config.drop_pose or start_pose)
            self.run_pick_cycle(start_pose, hover_pose, drop_pose)
        elif self.config.control == "gamepad":
            self.run_gamepad_loop(start_pose, keyboard_pose)
        else:
            self.run_keyboard_loop(start_pose, keyboard_pose)

    def print_gamepad_help(self) -> None:
        print("Gamepad control is active.")
        print(f"Device: {self.config.gamepad_device}")
        print("Left stick X: J1 left/right")
        print("Left stick Y: reach forward/back")
        print(
            "Gamepad speed: "
            f"J1 {self.config.gamepad_j1_speed_dps:.2f} deg/s, "
            f"reach {self.config.gamepad_reach_speed_dps:.2f} deg/s, "
            f"curve {self.config.gamepad_axis_curve:.2f}"
        )
        print(
            "Stop behavior: "
            f"reset target on stick release={self.config.gamepad_stop_reset}, "
            f"target lead limit={self.config.gamepad_lead_limit_deg:.2f} deg"
        )
        print(f"A / button {self.config.gamepad_pick_button}: pick cycle")
        print(
            f"B / button {self.config.gamepad_stop_button}: "
            "emergency stop, hold position, then D prompt"
        )
        print("X / button 2: reset joint reference")
        print("Y / button 3: print current pose")

    @staticmethod
    def format_pose(pose: dict[str, float]) -> str:
        return (
            f"X={pose['ee.x']:8.3f} Y={pose['ee.y']:8.3f} Z={pose['ee.z']:8.3f} "
            f"RX={pose['ee.rx']:8.3f} RY={pose['ee.ry']:8.3f} RZ={pose['ee.rz']:8.3f}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--can", default="can0")
    parser.add_argument("--hand-port", default="/dev/ttyUSB0")
    parser.add_argument("--hand-id", type=int, default=1)
    parser.add_argument("--hand-speed", type=int, default=800)
    parser.add_argument("--hand-force", type=int, default=1500)
    parser.add_argument("--speed", type=int, default=8)
    parser.add_argument("--rate-hz", type=float, default=40.0)
    parser.add_argument("--feedback-timeout", type=float, default=8.0)
    parser.add_argument("--j1-step-deg", type=float, default=2.0)
    parser.add_argument("--reach-step-deg", type=float, default=2.0)
    parser.add_argument("--reach-transition-j2-deg", type=float, default=90.0)
    parser.add_argument("--reach-pre-j2-gain", type=float, default=1.0)
    parser.add_argument("--reach-pre-j3-gain", type=float, default=-0.85)
    parser.add_argument("--reach-pre-j5-gain", type=float, default=-0.05)
    parser.add_argument("--reach-post-j2-gain", type=float, default=1.0)
    parser.add_argument("--reach-post-j3-gain", type=float, default=-1.2)
    parser.add_argument("--reach-post-j5-gain", type=float, default=-0.15)
    parser.add_argument("--start", type=parse_pose_mm_deg, default=pose_from_values(DEFAULT_START_POSE))
    parser.add_argument("--start-joints", type=parse_joint_degrees, default=list(DEFAULT_START_JOINTS))
    parser.add_argument("--start-duration", type=float, default=8.0)
    parser.add_argument("--hover-z", type=float)
    parser.add_argument("--hover-duration", type=float, default=6.0)
    parser.add_argument("--grab-z", type=float, required=True)
    parser.add_argument("--lift-z", type=float)
    parser.add_argument("--vertical-duration", type=float, default=4.0)
    parser.add_argument("--transfer-duration", type=float, default=8.0)
    parser.add_argument("--return-duration", type=float, default=8.0)
    parser.add_argument("--soft-arrival", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--soft-arrival-min-speed", type=int, default=2)
    parser.add_argument("--soft-arrival-joint-slow-deg", type=float, default=12.0)
    parser.add_argument("--soft-arrival-pose-slow-mm", type=float, default=80.0)
    parser.add_argument("--soft-arrival-pose-slow-deg", type=float, default=20.0)
    parser.add_argument("--auto-position-tolerance-mm", type=float, default=2.0)
    parser.add_argument("--auto-rpy-tolerance-deg", type=float, default=2.0)
    parser.add_argument("--drop", type=parse_pose_mm_deg)
    parser.add_argument("--pre-grab-open-speed", type=int, default=1800)
    parser.add_argument("--adaptive-close-force-threshold", type=float, default=300.0)
    parser.add_argument("--adaptive-close-step-deg", type=float, default=25.0)
    parser.add_argument("--adaptive-close-rear-step-deg", type=float, default=35.0)
    parser.add_argument("--adaptive-close-settle", type=float, default=0.06)
    parser.add_argument("--grasp-mode", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--hand-settle", type=float, default=0.0)
    parser.add_argument("--pre-grab-open-settle", type=float, default=0.0)
    parser.add_argument("--drop-open-settle", type=float, default=4.0)
    parser.add_argument("--held-force-threshold", type=float, default=100.0)
    parser.add_argument(
        "--held-force-fingers",
        type=parse_name_list,
        default=list(HAND_NAMES),
    )
    parser.add_argument(
        "--held-force-alt-fingers",
        type=parse_name_list,
        default=[],
    )
    parser.add_argument("--held-force-count", type=int, default=2)
    parser.add_argument("--held-check-duration", type=float, default=2.5)
    parser.add_argument("--held-check-rate-hz", type=float, default=5.0)
    parser.add_argument("--held-required-samples", type=int, default=15)
    parser.add_argument("--result-gesture", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--result-gesture-speed", type=int, default=20)
    parser.add_argument("--result-gesture-j2-back-deg", type=float, default=30.0)
    parser.add_argument("--result-gesture-j6-deg", type=float, default=90.0)
    parser.add_argument("--result-thumb-speed", type=int, default=2500)
    parser.add_argument("--result-thumb-settle", type=float, default=0.0)
    parser.add_argument("--result-gesture-duration", type=float, default=6.0)
    parser.add_argument("--result-gesture-hold-after", type=float, default=0.0)
    parser.add_argument("--result-gesture-return-duration", type=float, default=2.5)
    parser.add_argument("--failed-return-hold", type=float, default=0.5)
    parser.add_argument("--control", choices=("keyboard", "gamepad", "once"), default="keyboard")
    parser.add_argument("--gamepad-device", default="/dev/input/js0")
    parser.add_argument("--gamepad-deadzone", type=float, default=0.18)
    parser.add_argument("--gamepad-axis-x", type=int, default=0)
    parser.add_argument("--gamepad-axis-y", type=int, default=1)
    parser.add_argument("--gamepad-invert-y", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gamepad-j1-speed-dps", type=float, default=8.0)
    parser.add_argument("--gamepad-reach-speed-dps", type=float, default=6.0)
    parser.add_argument("--gamepad-axis-curve", type=float, default=1.8)
    parser.add_argument("--gamepad-print-interval", type=float, default=0.2)
    parser.add_argument("--gamepad-lead-limit-deg", type=float, default=1.5)
    parser.add_argument("--gamepad-stop-reset", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gamepad-pick-button", type=int, default=0)
    parser.add_argument("--gamepad-stop-button", type=int, default=1)
    parser.add_argument(
        "--classify-ball",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="At lift hover, decide NONE/A/B/C with the tactile classifier.",
    )
    parser.add_argument("--ball-model", type=Path, default=DEFAULT_BALL_MODEL)
    parser.add_argument("--ball-output", type=Path, default=DEFAULT_BALL_OUTPUT)
    parser.add_argument("--ball-visual-reference-samples", type=Path, default=DEFAULT_BALL_REFERENCE_SAMPLES)
    parser.add_argument("--ball-contact-threshold", type=float, default=70.0)
    parser.add_argument("--ball-hover-duration", type=float, default=1.5)
    parser.add_argument("--ball-hover-rate-hz", type=float, default=10.0)
    parser.add_argument("--ball-squeeze-delta", type=float, default=40.0)
    parser.add_argument("--ball-squeeze-duration", type=float, default=3.0)
    parser.add_argument("--ball-ab-squeeze-test", action=argparse.BooleanOptionalAction, default=False)
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
    parser.add_argument("--yes", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not 0 <= args.speed <= 100:
        raise ValueError("--speed must be in [0, 100]")
    if args.rate_hz <= 0:
        raise ValueError("--rate-hz must be positive")
    if args.soft_arrival_min_speed <= 0:
        raise ValueError("--soft-arrival-min-speed must be positive")
    if (
        args.soft_arrival_joint_slow_deg <= 0
        or args.soft_arrival_pose_slow_mm <= 0
        or args.soft_arrival_pose_slow_deg <= 0
    ):
        raise ValueError("soft arrival thresholds must be positive")
    if args.reach_step_deg <= 0:
        raise ValueError("--reach-step-deg must be positive")
    if args.reach_pre_j2_gain <= 0 or args.reach_post_j2_gain <= 0:
        raise ValueError("reach J2 gains must be positive")
    if args.hand_settle < 0 or args.pre_grab_open_settle < 0 or args.drop_open_settle < 0:
        raise ValueError("hand settle times must be non-negative")
    if args.hand_speed <= 0 or args.pre_grab_open_speed <= 0:
        raise ValueError("hand speeds must be positive")
    if args.adaptive_close_force_threshold < 0:
        raise ValueError("--adaptive-close-force-threshold must be non-negative")
    if args.adaptive_close_step_deg <= 0 or args.adaptive_close_rear_step_deg <= 0:
        raise ValueError("adaptive close steps must be positive")
    if args.adaptive_close_settle < 0:
        raise ValueError("--adaptive-close-settle must be non-negative")
    if args.grasp_mode not in (0, 1, 2):
        raise ValueError("--grasp-mode must be 0, 1, or 2")
    if args.held_force_threshold < 0:
        raise ValueError("--held-force-threshold must be non-negative")
    if not 1 <= args.held_force_count <= len(HAND_NAMES):
        raise ValueError(f"--held-force-count must be in [1, {len(HAND_NAMES)}]")
    if args.held_check_duration <= 0 or args.held_check_rate_hz <= 0:
        raise ValueError("held check duration/rate must be positive")
    if args.held_required_samples <= 0:
        raise ValueError("--held-required-samples must be positive")
    invalid_force_names = set(args.held_force_fingers) - set(HAND_NAMES)
    if invalid_force_names:
        raise ValueError(f"bad --held-force-fingers names: {sorted(invalid_force_names)}")
    invalid_alt_force_names = set(args.held_force_alt_fingers) - set(HAND_NAMES)
    if invalid_alt_force_names:
        raise ValueError(f"bad --held-force-alt-fingers names: {sorted(invalid_alt_force_names)}")
    if not 0 <= args.result_gesture_speed <= 100:
        raise ValueError("--result-gesture-speed must be in [0, 100]")
    if args.result_thumb_speed <= 0:
        raise ValueError("--result-thumb-speed must be positive")
    if args.result_gesture_j2_back_deg < 0 or args.result_gesture_j6_deg < 0:
        raise ValueError("result gesture angles must be non-negative")
    if (
        args.result_thumb_settle < 0
        or args.result_gesture_duration < 0
        or args.result_gesture_hold_after < 0
        or args.result_gesture_return_duration < 0
        or args.failed_return_hold < 0
    ):
        raise ValueError("result gesture/failed return durations must be non-negative")
    if not 0.0 <= args.gamepad_deadzone < 1.0:
        raise ValueError("--gamepad-deadzone must be in [0, 1)")
    if args.gamepad_j1_speed_dps <= 0 or args.gamepad_reach_speed_dps <= 0:
        raise ValueError("gamepad speed values must be positive")
    if args.gamepad_axis_curve < 1.0:
        raise ValueError("--gamepad-axis-curve must be >= 1")
    if args.gamepad_print_interval < 0:
        raise ValueError("--gamepad-print-interval must be non-negative")
    if args.gamepad_lead_limit_deg < 0:
        raise ValueError("--gamepad-lead-limit-deg must be non-negative")
    if args.gamepad_pick_button < 0 or args.gamepad_stop_button < 0:
        raise ValueError("gamepad button numbers must be non-negative")
    if args.gamepad_pick_button == args.gamepad_stop_button:
        raise ValueError("--gamepad-pick-button and --gamepad-stop-button must differ")
    if args.classify_ball:
        if not args.ball_model.exists():
            raise ValueError(f"--ball-model does not exist: {args.ball_model}")
        if not args.ball_visual_reference_samples.exists():
            raise ValueError(f"--ball-visual-reference-samples does not exist: {args.ball_visual_reference_samples}")
        if args.ball_contact_threshold < 0:
            raise ValueError("--ball-contact-threshold must be non-negative")
        if args.ball_hover_duration <= 0 or args.ball_hover_rate_hz <= 0:
            raise ValueError("--ball-hover-duration/rate must be positive")
        if args.ball_squeeze_delta <= 0 or args.ball_squeeze_duration <= 0:
            raise ValueError("--ball-squeeze-delta/duration must be positive")
        if args.ball_ab_squeeze_a_standard == args.ball_ab_squeeze_b_standard:
            raise ValueError("--ball-ab-squeeze-a-standard and --ball-ab-squeeze-b-standard must differ")
        if not 0 <= args.ball_low_confidence_c_squeeze_threshold <= 1:
            raise ValueError("--ball-low-confidence-c-squeeze-threshold must be between 0 and 1")
        if args.ball_ab_friction_threshold < 0:
            raise ValueError("--ball-ab-friction-threshold must be non-negative")
        if args.ball_ab_proximity_index_force_threshold < 0 or args.ball_ab_proximity_thumb_threshold <= 0:
            raise ValueError("--ball-ab-proximity thresholds must be valid")
        if args.ball_ab_proximity_min_samples < 0:
            raise ValueError("--ball-ab-proximity-min-samples must be non-negative")
        if args.ball_bc_proximity_thumb_threshold <= 0 or args.ball_bc_proximity_middle_threshold <= 0:
            raise ValueError("--ball-bc-proximity thresholds must be positive")


def config_from_args(args: argparse.Namespace) -> ClawMachineTaskConfig:
    ball_classifier_config = None
    if args.classify_ball:
        ball_classifier_config = BallClassifierConfig(
            model=args.ball_model,
            output=args.ball_output,
            visual_reference_samples=args.ball_visual_reference_samples,
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
        )
    return ClawMachineTaskConfig(
        grab_z=args.grab_z,
        drop_pose=args.drop,
        start_pose=args.start,
        start_joints=args.start_joints,
        lift_z=args.lift_z,
        speed_rate=args.speed,
        rate_hz=args.rate_hz,
        feedback_timeout_s=args.feedback_timeout,
        start_duration_s=args.start_duration,
        hover_z=args.hover_z,
        hover_duration_s=args.hover_duration,
        vertical_duration_s=args.vertical_duration,
        transfer_duration_s=args.transfer_duration,
        return_duration_s=args.return_duration,
        soft_arrival=args.soft_arrival,
        soft_arrival_min_speed=args.soft_arrival_min_speed,
        soft_arrival_joint_slow_deg=args.soft_arrival_joint_slow_deg,
        soft_arrival_pose_slow_mm=args.soft_arrival_pose_slow_mm,
        soft_arrival_pose_slow_deg=args.soft_arrival_pose_slow_deg,
        auto_position_tolerance_mm=args.auto_position_tolerance_mm,
        auto_rpy_tolerance_deg=args.auto_rpy_tolerance_deg,
        j1_step_deg=args.j1_step_deg,
        reach_step_deg=args.reach_step_deg,
        reach_transition_j2_deg=args.reach_transition_j2_deg,
        reach_pre_j2_gain=args.reach_pre_j2_gain,
        reach_pre_j3_gain=args.reach_pre_j3_gain,
        reach_pre_j5_gain=args.reach_pre_j5_gain,
        reach_post_j2_gain=args.reach_post_j2_gain,
        reach_post_j3_gain=args.reach_post_j3_gain,
        reach_post_j5_gain=args.reach_post_j5_gain,
        hand_speed=args.hand_speed,
        pre_grab_open_speed=args.pre_grab_open_speed,
        adaptive_close_force_threshold=args.adaptive_close_force_threshold,
        adaptive_close_step_deg=args.adaptive_close_step_deg,
        adaptive_close_rear_step_deg=args.adaptive_close_rear_step_deg,
        adaptive_close_settle_s=args.adaptive_close_settle,
        grasp_mode=args.grasp_mode,
        hand_settle_s=args.hand_settle,
        pre_grab_open_settle_s=args.pre_grab_open_settle,
        drop_open_settle_s=args.drop_open_settle,
        held_force_threshold=args.held_force_threshold,
        held_force_fingers=args.held_force_fingers,
        held_force_alt_fingers=args.held_force_alt_fingers,
        held_force_count=args.held_force_count,
        held_required_samples=args.held_required_samples,
        held_check_duration_s=args.held_check_duration,
        held_check_rate_hz=args.held_check_rate_hz,
        result_gesture=args.result_gesture,
        result_gesture_speed=args.result_gesture_speed,
        result_gesture_j2_back_deg=args.result_gesture_j2_back_deg,
        result_gesture_j6_deg=args.result_gesture_j6_deg,
        result_thumb_speed=args.result_thumb_speed,
        result_thumb_settle_s=args.result_thumb_settle,
        result_gesture_duration_s=args.result_gesture_duration,
        result_gesture_hold_after_s=args.result_gesture_hold_after,
        result_gesture_return_duration_s=args.result_gesture_return_duration,
        failed_return_hold_s=args.failed_return_hold,
        control=args.control,
        gamepad_device=args.gamepad_device,
        gamepad_deadzone=args.gamepad_deadzone,
        gamepad_axis_x=args.gamepad_axis_x,
        gamepad_axis_y=args.gamepad_axis_y,
        gamepad_invert_y=args.gamepad_invert_y,
        gamepad_j1_speed_dps=args.gamepad_j1_speed_dps,
        gamepad_reach_speed_dps=args.gamepad_reach_speed_dps,
        gamepad_axis_curve=args.gamepad_axis_curve,
        gamepad_print_interval=args.gamepad_print_interval,
        gamepad_lead_limit_deg=args.gamepad_lead_limit_deg,
        gamepad_stop_reset=args.gamepad_stop_reset,
        gamepad_pick_button=args.gamepad_pick_button,
        gamepad_stop_button=args.gamepad_stop_button,
        ball_classifier_config=ball_classifier_config,
    )


def main() -> int:
    args = build_arg_parser().parse_args()
    validate_args(args)

    print("LeRobot claw-machine teleop")
    print(f"control={args.control}, CAN={args.can}, hand={args.hand_port}")
    print(f"grab Z: {args.grab_z:.3f} mm")
    print(
        "MOVE_J teleop: "
        f"j1_step={args.j1_step_deg:.3f} reach_step={args.reach_step_deg:.3f} "
        f"transition_j2={args.reach_transition_j2_deg:.3f}"
    )
    print(
        "Reach gains: "
        f"pre=({args.reach_pre_j2_gain:.2f},{args.reach_pre_j3_gain:.2f},"
        f"{args.reach_pre_j5_gain:.2f}) "
        f"post=({args.reach_post_j2_gain:.2f},{args.reach_post_j3_gain:.2f},"
        f"{args.reach_post_j5_gain:.2f})"
    )
    print("Initial position uses joint_*.pos MOVE_J; hover/pick/drop use ee.* MOVE_P; teleop uses MOVE_J.")
    if args.classify_ball:
        print(f"ball classifier enabled: model={args.ball_model} output={args.ball_output}")
    if not args.yes:
        answer = input("Type YES to connect the robot and run LeRobot claw control: ").strip()
        if answer != "YES":
            print("Aborted.")
            return 1

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
    controller = ClawMachineController(robot, config_from_args(args))
    stop_monitor = None
    if args.control == "gamepad":
        stop_monitor = GamepadStopMonitor(
            args.gamepad_device,
            args.gamepad_stop_button,
            controller.request_emergency_stop,
        )
        stop_monitor.start()

    try:
        robot.connect()
        controller.run()
    except KeyboardInterrupt:
        print("\nInterrupted. Motors were not disabled by this script.")
    except Exception as exc:
        print(f"\n[warn] {exc}")
        try:
            if robot.is_connected:
                controller.print_state()
        except Exception:
            pass
        return 1
    finally:
        if stop_monitor is not None:
            stop_monitor.close()
        if robot.is_connected:
            robot.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
