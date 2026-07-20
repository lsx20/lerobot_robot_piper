#!/usr/bin/env python3
"""LeRobot-style claw-machine controller for Piper + RH56F2.

This module is the bridge from the older direct-SDK claw scripts to LeRobot's
Robot interface. It only talks to the robot through:

  - robot.get_observation()
  - robot.send_action(...)

That makes the claw workflow look like a hand-written policy on top of a
LeRobot-compatible robot.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from typing import Protocol

from lerobot.processor import RobotAction, RobotObservation

from ..config_piper_rh56f2_follower import PiperRH56F2FollowerConfig
from ..piper_rh56f2_follower import EE_POSE_NAMES, PiperRH56F2Follower
from .claw_hand_grasp import BALL_CLOSED, BALL_READY_OPEN


class ActionRobot(Protocol):
    """Small part of LeRobot's Robot API needed by this controller."""

    def get_observation(self) -> RobotObservation:
        ...

    def send_action(self, action: RobotAction) -> RobotAction:
        ...


@dataclass
class ClawMachineTaskConfig:
    """Task-level parameters for one claw-machine pick/drop cycle.

    Units match ``PiperRH56F2Follower``:
      - end-effector x/y/z are millimeters
      - end-effector rx/ry/rz are degrees
      - RH56F2 hand values are vendor register angle units
    """

    grab_z: float
    drop_pose: dict[str, float]
    start_pose: dict[str, float] | None = None
    lift_z: float | None = None
    speed_rate: int = 30
    rate_hz: float = 10.0
    start_duration_s: float = 6.0
    vertical_duration_s: float = 4.0
    transfer_duration_s: float = 8.0
    return_duration_s: float = 8.0
    hand_settle_s: float = 1.0
    pre_grab_open_settle_s: float = 0.5
    drop_open_settle_s: float = 1.0
    held_force_threshold: float = 130.0
    held_force_fingers: list[str] = field(
        default_factory=lambda: ["thumb_bend", "thumb_swing", "index", "middle"]
    )
    held_required_samples: int = 3
    held_check_duration_s: float = 1.0
    held_check_rate_hz: float = 5.0


def pose_from_raw(raw_pose: list[int]) -> dict[str, float]:
    """Convert Piper SDK raw 0.001 mm/deg pose into LeRobot ee action keys."""
    if len(raw_pose) != 6:
        raise ValueError("expected 6 raw pose values: X,Y,Z,RX,RY,RZ")
    return {
        name: value / 1000.0
        for name, value in zip(EE_POSE_NAMES, raw_pose, strict=True)
    }


def parse_pose_mm_deg(value: str) -> dict[str, float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 6:
        raise argparse.ArgumentTypeError("expected X,Y,Z,RX,RY,RZ")
    try:
        values = [float(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("pose values must be numbers") from exc
    return dict(zip(EE_POSE_NAMES, values, strict=True))


def ee_pose_from_observation(obs: RobotObservation) -> dict[str, float]:
    return {name: float(obs[name]) for name in EE_POSE_NAMES}


class ClawMachineController:
    def __init__(self, robot: ActionRobot, config: ClawMachineTaskConfig):
        self.robot = robot
        self.config = config

    def set_hand_pose(self, pose: dict[str, float]) -> None:
        action = {f"hand.{name}.pos": value for name, value in pose.items()}
        self.robot.send_action(action)

    def move_ee_for(self, pose: dict[str, float], duration_s: float, label: str) -> None:
        interval_s = 1.0 / self.config.rate_hz
        deadline = time.time() + duration_s
        while time.time() < deadline:
            self.robot.send_action(pose)
            time.sleep(interval_s)
        print(f"{label}: {self.format_pose(pose)}")

    def held_by_force(self) -> bool:
        deadline = time.time() + self.config.held_check_duration_s
        interval_s = 1.0 / self.config.held_check_rate_hz
        consecutive = 0
        best = 0

        while time.time() < deadline:
            obs = self.robot.get_observation()
            active = [
                abs(float(obs.get(f"hand.{name}.force", 0.0)))
                >= self.config.held_force_threshold
                for name in self.config.held_force_fingers
            ]
            if all(active):
                consecutive += 1
                best = max(best, consecutive)
            else:
                consecutive = 0
            time.sleep(interval_s)

        held = best >= self.config.held_required_samples
        print(f"held check: best={best}/{self.config.held_required_samples}, held={held}")
        return held

    def run_pick_cycle(self, hover_pose: dict[str, float] | None = None) -> bool:
        obs = self.robot.get_observation()
        hover = dict(hover_pose or ee_pose_from_observation(obs))
        start = dict(self.config.start_pose or hover)
        drop = dict(self.config.drop_pose)

        grab = dict(hover)
        grab["ee.z"] = self.config.grab_z

        lift = dict(hover)
        if self.config.lift_z is not None:
            lift["ee.z"] = self.config.lift_z

        print("Running LeRobot claw pick cycle")
        print(f"  hover: {self.format_pose(hover)}")
        print(f"  grab:  {self.format_pose(grab)}")
        print(f"  lift:  {self.format_pose(lift)}")
        print(f"  drop:  {self.format_pose(drop)}")
        print(f"  start: {self.format_pose(start)}")

        self.set_hand_pose(BALL_READY_OPEN)
        self.move_ee_for(grab, self.config.vertical_duration_s, "descend")

        if self.config.pre_grab_open_settle_s > 0:
            time.sleep(self.config.pre_grab_open_settle_s)
        self.set_hand_pose(BALL_CLOSED)
        time.sleep(self.config.hand_settle_s)

        self.move_ee_for(lift, self.config.vertical_duration_s, "lift")
        self.move_ee_for(drop, self.config.transfer_duration_s, "drop move")

        held = self.held_by_force()
        self.set_hand_pose(BALL_READY_OPEN)
        time.sleep(self.config.drop_open_settle_s)

        self.set_hand_pose(BALL_CLOSED)
        self.move_ee_for(start, self.config.return_duration_s, "return")
        return held

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
    parser.add_argument("--speed", type=int, default=30)
    parser.add_argument("--rate-hz", type=float, default=10.0)
    parser.add_argument("--grab-z", type=float, required=True)
    parser.add_argument("--lift-z", type=float)
    parser.add_argument("--drop", type=parse_pose_mm_deg, required=True)
    parser.add_argument("--start", type=parse_pose_mm_deg)
    parser.add_argument("--vertical-duration", type=float, default=4.0)
    parser.add_argument("--transfer-duration", type=float, default=8.0)
    parser.add_argument("--return-duration", type=float, default=8.0)
    parser.add_argument("--yes", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if not args.yes:
        answer = input("Type YES to connect the robot and run one LeRobot claw cycle: ").strip()
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
        )
    )
    task = ClawMachineTaskConfig(
        grab_z=args.grab_z,
        lift_z=args.lift_z,
        drop_pose=args.drop,
        start_pose=args.start,
        speed_rate=args.speed,
        rate_hz=args.rate_hz,
        vertical_duration_s=args.vertical_duration,
        transfer_duration_s=args.transfer_duration,
        return_duration_s=args.return_duration,
    )
    controller = ClawMachineController(robot, task)

    try:
        robot.connect()
        held = controller.run_pick_cycle()
        print(f"LeRobot claw cycle complete; held={held}")
    finally:
        if robot.is_connected:
            robot.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
