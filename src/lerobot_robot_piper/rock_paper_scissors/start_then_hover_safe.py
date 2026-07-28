#!/usr/bin/env python3
"""Recover Piper, move to the latest start joints, then hover at a fixed target."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


CLAW_MACHINE_DIR = Path(__file__).resolve().parents[1] / "claw_machine"
if str(CLAW_MACHINE_DIR) not in sys.path:
    sys.path.insert(0, str(CLAW_MACHINE_DIR))

from claw_init import (  # noqa: E402
    RawTerminal,
    connect_piper,
    end_pose_raw,
    enable_all,
    fmt_joints,
    joints_deg,
    pose_mm_deg,
    read_key,
    wait_for_real_feedback,
)
from lerobot_claw import DEFAULT_START_JOINTS, DEFAULT_START_POSE  # noqa: E402


FIXED_HOVER_XYZ_M = (0.30455, 0.02575, 0.25000)


def parse_xyz(value: str) -> list[float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected X,Y,Z in metres")
    try:
        return [float(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("X,Y,Z must be numbers") from exc


def status(piper: object) -> tuple[int, int, int, list[bool]]:
    arm = piper.GetArmStatus().arm_status
    return (
        int(arm.ctrl_mode),
        int(arm.mode_feed),
        int(arm.arm_status),
        list(piper.GetArmEnableStatus()),
    )


def print_status(piper: object, label: str) -> None:
    ctrl, mode, arm, enabled = status(piper)
    print(
        f"{label}: ctrl=0x{ctrl:x} mode=0x{mode:x} arm=0x{arm:x} "
        f"enable={enabled} pose=[{pose_mm_deg(end_pose_raw(piper))}]"
    )


def wait_ready(piper: object, speed: int, mode: int, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        piper.MotionCtrl_1(0x02, 0x00, 0x02)
        piper.MotionCtrl_2(0x01, mode, speed, 0x00)
        piper.EnableArm(7, 0x02)
        time.sleep(0.05)
        ctrl, mode_feed, arm, enabled = status(piper)
        print(
            f"ready check: ctrl=0x{ctrl:x} mode=0x{mode_feed:x} "
            f"arm=0x{arm:x} enable={enabled}"
        )
        if ctrl == 0x01 and mode_feed == mode and arm == 0x00 and enabled and all(enabled):
            return True
    return False


def clipped_joint_target(current: list[float], goal: list[float], max_delta_deg: float = 5.0) -> list[float]:
    return [
        current_value + max(-max_delta_deg, min(max_delta_deg, goal_value - current_value))
        for current_value, goal_value in zip(current, goal, strict=True)
    ]


def send_joint_step(piper: object, goal: list[float], speed: int) -> bool:
    current = joints_deg(piper)
    command = clipped_joint_target(current, goal)
    piper.MotionCtrl_2(0x01, 0x01, speed, 0x00)
    piper.JointCtrl(*[int(round(value * 1000.0)) for value in command])
    time.sleep(0.01)
    _, _, arm, enabled = status(piper)
    if arm != 0x00 or not enabled or not all(enabled):
        for _ in range(20):
            piper.EnableArm(7, 0x02)
            time.sleep(0.05)
            _, _, arm, enabled = status(piper)
            if arm == 0x00 and enabled and all(enabled):
                return True
        print(f"\n[warn] MOVE_J feedback failed: arm=0x{arm:x} enable={enabled}")
        return False
    return True


def send_pose_phase(
    piper: object,
    target: list[int],
    speed: int,
    duration_s: float,
    rate_hz: float,
    label: str,
) -> str:
    paused = False
    deadline = time.monotonic() + duration_s
    interval = 1.0 / rate_hz
    with RawTerminal():
        while time.monotonic() < deadline:
            key = read_key(0.0)
            if key == " ":
                paused = not paused
                print("\nPAUSED: holding current pose." if paused else f"\nRESUMED: moving during {label}.")
            elif key in {"q", "Q"}:
                return "q"
            _, _, arm_before, enabled_before = status(piper)
            if arm_before != 0x00 or not enabled_before or not all(enabled_before):
                print(f"\n[STOP] feedback invalid before EndPoseCtrl: arm=0x{arm_before:x} enable={enabled_before}")
                return "fault"
            if paused:
                hold = end_pose_raw(piper)
            else:
                current = end_pose_raw(piper)
                hold = []
                for index, (current_raw, goal_raw) in enumerate(zip(current, target, strict=True)):
                    max_delta_raw = 20_000 if index < 3 else 10_000
                    delta_raw = max(-max_delta_raw, min(max_delta_raw, goal_raw - current_raw))
                    hold.append(current_raw + delta_raw)
            piper.MotionCtrl_2(0x01, 0x00, speed, 0x00)
            piper.EndPoseCtrl(*hold)
            time.sleep(min(interval, 0.03))
            _, _, arm_after, enabled_after = status(piper)
            if arm_after != 0x00 or not enabled_after or not all(enabled_after):
                print(f"\n[STOP] Piper rejected/left protection after EndPoseCtrl: arm=0x{arm_after:x} enable={enabled_after}")
                return "fault"
            remaining = interval - min(interval, 0.03)
            if remaining > 0:
                time.sleep(remaining)
    return "done"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--can", default="can0")
    parser.add_argument(
        "--xyz",
        type=parse_xyz,
        default=list(FIXED_HOVER_XYZ_M),
        help="fixed hover X,Y,Z in metres; no camera is used",
    )
    parser.add_argument("--speed", type=int, default=2)
    parser.add_argument("--rate-hz", type=float, default=40.0)
    parser.add_argument("--start-duration", type=float, default=15.0)
    parser.add_argument("--hover-duration", type=float, default=10.0)
    parser.add_argument("--feedback-timeout", type=float, default=10.0)
    parser.add_argument("--fraction", type=float, default=0.2, help="fraction of the requested XYZ displacement to test")
    parser.add_argument("--yes", action="store_true", help="start without the confirmation prompt")
    args = parser.parse_args()

    if not 0 <= args.speed <= 100:
        raise SystemExit("--speed must be between 0 and 100")
    if args.rate_hz <= 0 or args.start_duration <= 0 or args.hover_duration <= 0:
        raise SystemExit("durations and --rate-hz must be positive")
    if not 0.0 < args.fraction <= 1.0:
        raise SystemExit("--fraction must be in (0, 1]")

    hover_target = [
        int(round(args.xyz[0] * 1_000_000.0)),
        int(round(args.xyz[1] * 1_000_000.0)),
        int(round(args.xyz[2] * 1_000_000.0)),
    ]
    print("SAFETY: one continuous test; start pose then hover only.")
    print("No camera, descent, gripper, or grasp command is issued.")
    print("SPACE = pause/hold during hover, q = stop, Ctrl+C = abort.")
    print("Recovery may temporarily change motor enable feedback; support the arm.")
    print(f"Latest start pose: {tuple(DEFAULT_START_POSE)}")
    print(f"Latest start joints: {fmt_joints(list(DEFAULT_START_JOINTS))}")
    print(f"Hover target: {pose_mm_deg(hover_target + [0, 0, 0])}")
    if not args.yes:
        if input("Type START_HOVER to continue: ").strip() != "START_HOVER":
            print("Aborted before connecting.")
            return 0

    piper = connect_piper(args)
    try:
        wait_for_real_feedback(piper, args.feedback_timeout)
        print_status(piper, "initial")

        print("Selecting CAN_CTRL before enabling, matching lerobot_claw.py...")
        piper.MotionCtrl_1(0x02, 0x00, 0x02)
        time.sleep(0.05)
        if not enable_all(piper, args.feedback_timeout):
            raise RuntimeError("Could not enable all joints after recovery.")
        if not wait_ready(piper, args.speed, 0x01, args.feedback_timeout):
            print_status(piper, "MOVE_J not ready")
            raise RuntimeError("MOVE_J still not ready; refusing to move to start.")

        start_joints = list(DEFAULT_START_JOINTS)
        print("Priming MOVE_J with the current joints, matching PiperRH56F2Follower.connect()...")
        current_joints = joints_deg(piper)
        piper.MotionCtrl_2(0x01, 0x01, args.speed, 0x00)
        piper.JointCtrl(*[int(round(value * 1000.0)) for value in current_joints])
        time.sleep(0.3)
        if not enable_all(piper, args.feedback_timeout):
            print_status(piper, "MOVE_J prime failed")
            raise RuntimeError("Piper did not remain enabled after the MOVE_J prime command.")
        print(f"Moving to latest start joints: {fmt_joints(start_joints)}")
        deadline = time.monotonic() + args.start_duration
        while time.monotonic() < deadline:
            if not send_joint_step(piper, start_joints, args.speed):
                raise RuntimeError("MOVE_J start command failed.")
            time.sleep(1.0 / args.rate_hz)
        print_status(piper, "at start")

        print("Switching to MOVE_P using the same mode transition as lerobot_claw.py...")
        if not wait_ready(piper, args.speed, 0x00, args.feedback_timeout):
            print_status(piper, "MOVE_P not ready")
            raise RuntimeError("MOVE_P still not ready; refusing to send hover target.")

        current = end_pose_raw(piper)
        partial_target = [
            current[index] + int(round((hover_target[index] - current[index]) * args.fraction))
            for index in range(3)
        ] + current[3:]
        xy_target = list(current)
        xy_target[0] = partial_target[0]
        xy_target[1] = partial_target[1]
        z_target = list(xy_target)
        z_target[2] = partial_target[2]
        print(f"Testing fraction: {args.fraction:.3f} of requested XYZ displacement")
        print(f"XY target (hold current Z): {pose_mm_deg(xy_target)}")
        result = send_pose_phase(piper, xy_target, args.speed, args.hover_duration, args.rate_hz, "XY phase")
        if result != "done":
            print_status(piper, f"XY {result}")
            return 0
        print(f"Z target (hold X/Y): {pose_mm_deg(z_target)}")
        result = send_pose_phase(piper, z_target, args.speed, args.hover_duration, args.rate_hz, "Z phase")
        print_status(piper, f"hover {result}")
        return 0
    except KeyboardInterrupt:
        print("\nCtrl+C received; stopped sending motion commands.")
        return 130
    finally:
        try:
            piper.DisconnectPort()
        except Exception as exc:
            print(f"[warn] disconnect failed: {exc}")
        print("Motors were not explicitly disabled by this script.")


if __name__ == "__main__":
    raise SystemExit(main())
