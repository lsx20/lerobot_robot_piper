#!/usr/bin/env python3
"""Verify Piper joint motion and Cartesian end-pose control.

This script is intentionally conservative:
  1. Prints pose, FK-from-joints, joint feedback, and controller status.
  2. Optionally performs a small joint move first.
  3. Sends a small Cartesian Z move with repeated EndPoseCtrl commands.

Use Ctrl-C to stop. Keep one hand near the emergency stop.
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import time
from dataclasses import dataclass

from piper_sdk import C_PiperInterface_V2


JOINT_LIMITS_DEG = {
    1: (-150.0, 150.0),
    2: (0.0, 180.0),
    3: (-170.0, 0.0),
    4: (-100.0, 100.0),
    5: (-70.0, 70.0),
    6: (-120.0, 120.0),
}


@dataclass
class State:
    pose: list[float]
    fk: list[float] | None
    joints: list[float]


def mmdeg_from_end_pose_msg(piper: C_PiperInterface_V2) -> list[float]:
    ep = piper.GetArmEndPoseMsgs().end_pose
    return [
        ep.X_axis / 1000.0,
        ep.Y_axis / 1000.0,
        ep.Z_axis / 1000.0,
        ep.RX_axis / 1000.0,
        ep.RY_axis / 1000.0,
        ep.RZ_axis / 1000.0,
    ]


def joints_deg_from_msg(piper: C_PiperInterface_V2) -> list[float]:
    js = piper.GetArmJointMsgs().joint_state
    return [
        js.joint_1 / 1000.0,
        js.joint_2 / 1000.0,
        js.joint_3 / 1000.0,
        js.joint_4 / 1000.0,
        js.joint_5 / 1000.0,
        js.joint_6 / 1000.0,
    ]


def fk_feedback(piper: C_PiperInterface_V2) -> list[float] | None:
    try:
        fk = piper.GetFK("feedback")
        if fk:
            return [float(v) for v in fk[-1]]
    except Exception as exc:
        print(f"[warn] GetFK('feedback') failed: {exc}")
    return None


def read_state(piper: C_PiperInterface_V2) -> State:
    return State(
        pose=mmdeg_from_end_pose_msg(piper),
        fk=fk_feedback(piper),
        joints=joints_deg_from_msg(piper),
    )


def fmt(values: list[float] | None) -> str:
    if values is None:
        return "None"
    return " ".join(f"{v:9.3f}" for v in values)


def max_abs_delta(a: list[float] | None, b: list[float] | None, n: int | None = None) -> float:
    if a is None or b is None:
        return float("nan")
    if n is None:
        n = min(len(a), len(b))
    return max(abs(a[i] - b[i]) for i in range(n))


def xyz_error(target: list[float] | None, values: list[float] | None) -> float:
    if target is None or values is None:
        return float("nan")
    return math.sqrt(sum((target[i] - values[i]) ** 2 for i in range(3)))


def print_state(
    label: str,
    state: State,
    initial: State | None = None,
    target: list[float] | None = None,
) -> None:
    if initial is None:
        print(f"{label:>10} pose xyz/rpy: {fmt(state.pose)}")
        print(f"{label:>10} fk   xyz/rpy: {fmt(state.fk)}")
        print(f"{label:>10} joints deg : {fmt(state.joints)}")
        return

    pose_d = max_abs_delta(initial.pose, state.pose, 3)
    fk_d = max_abs_delta(initial.fk, state.fk, 3)
    joint_d = max_abs_delta(initial.joints, state.joints)
    pose_err = xyz_error(target, state.pose)
    fk_err = xyz_error(target, state.fk)
    print(
        f"{label:>10} pose xyz/rpy: {fmt(state.pose)} | d_xyz={pose_d:7.3f} mm"
        f" | err={pose_err:7.3f} mm"
    )
    print(
        f"{label:>10} fk   xyz/rpy: {fmt(state.fk)} | d_xyz={fk_d:7.3f} mm"
        f" | err={fk_err:7.3f} mm"
    )
    print(
        f"{label:>10} joints deg : {fmt(state.joints)} | d_jnt={joint_d:7.3f} deg"
    )


def print_status(piper: C_PiperInterface_V2) -> None:
    print("\n--- controller status ---")
    for name in ("GetArmStatus", "GetArmCtrlCode151", "GetArmEnableStatus"):
        try:
            print(f"{name}: {getattr(piper, name)()}")
        except Exception as exc:
            print(f"{name}: failed: {exc}")
    print("-------------------------\n")


def prompt_before_disable(piper: C_PiperInterface_V2) -> None:
    print()
    print("WARNING: disabling Piper motors may make the arm drop.")
    print("Hold/support the arm before disabling.")
    answer = input("Type D then Enter to disable arm motors, or press Enter to keep motors enabled: ").strip()
    if answer == "D":
        piper.DisableArm(7)
        print("Piper arm motors disabled.")
    else:
        print("Piper arm motors left enabled.")


def ensure_enabled(piper: C_PiperInterface_V2) -> None:
    print("Enabling Piper...")
    for _ in range(500):
        if piper.EnablePiper():
            print("Piper enabled.")
            return
        time.sleep(0.01)
    raise RuntimeError("EnablePiper() did not succeed within 5 seconds")


def enable_fk(piper: C_PiperInterface_V2) -> None:
    try:
        piper.EnableFkCal()
        time.sleep(0.2)
        print(f"SDK FK calculation enabled: {piper.isCalFk()}")
    except Exception as exc:
        print(f"[warn] failed to enable SDK FK calculation: {exc}")


def send_hold_joint(piper: C_PiperInterface_V2, joints_deg: list[float], speed: int) -> None:
    values = [int(round(v * 1000.0)) for v in joints_deg]
    piper.MotionCtrl_2(0x01, 0x01, speed, 0x00, 0, 0x01)
    piper.JointCtrl(*values)


def joint_smoke_test(
    piper: C_PiperInterface_V2,
    speed: int,
    joint_index: int,
    delta_deg: float,
    duration_s: float,
    rate_hz: float,
) -> None:
    start = read_state(piper)
    target = list(start.joints)
    lo, hi = JOINT_LIMITS_DEG[joint_index]
    idx = joint_index - 1
    target[idx] = min(max(target[idx] + delta_deg, lo), hi)

    if math.isclose(target[idx], start.joints[idx], abs_tol=0.001):
        print(f"[warn] Joint {joint_index} target is clamped to current value; skipping joint test.")
        return

    print("\n=== joint smoke test ===")
    print(f"Moving joint_{joint_index} by about {target[idx] - start.joints[idx]:.3f} deg")
    dt = 1.0 / rate_hz
    end_t = time.time() + duration_s
    count = 0
    while time.time() < end_t:
        send_hold_joint(piper, target, speed)
        if count % max(1, int(rate_hz / 5)) == 0:
            print_state("joint", read_state(piper), start)
        count += 1
        time.sleep(dt)

    final = read_state(piper)
    print_state("joint end", final, start)
    print(
        "Joint test result:",
        "JOINTS_CHANGED" if max_abs_delta(start.joints, final.joints) > 0.5 else "NO_JOINT_CHANGE",
    )

    print("Returning to original joint position...")
    end_t = time.time() + duration_s
    while time.time() < end_t:
        send_hold_joint(piper, start.joints, speed)
        time.sleep(dt)


def cartesian_test(
    piper: C_PiperInterface_V2,
    speed: int,
    move_mode: int,
    installation_pos: int,
    dx_mm: float,
    dy_mm: float,
    dz_mm: float,
    fixed_target: list[float] | None,
    duration_s: float,
    rate_hz: float,
) -> None:
    start = read_state(piper)
    if fixed_target is None:
        x, y, z, rx, ry, rz = start.pose
        target = [
            x + dx_mm,
            y + dy_mm,
            z + dz_mm,
            rx,
            ry,
            rz,
        ]
    else:
        target = fixed_target
    target_i = [int(round(v * 1000.0)) for v in target]

    print("\n=== Cartesian EndPoseCtrl test ===")
    print(f"move_mode: 0x{move_mode:02X} ({'MOVE L' if move_mode == 0x02 else 'MOVE P'})")
    print(f"installation_pos: 0x{installation_pos:02X}")
    print(f"target xyz/rpy: {fmt(target)}")

    try:
        piper.MotionCtrl_1(0x02, 0x00, 0x02)
        time.sleep(0.05)
    except Exception as exc:
        print(f"[warn] MotionCtrl_1 skipped: {exc}")

    dt = 1.0 / rate_hz
    end_t = time.time() + duration_s
    count = 0
    while time.time() < end_t:
        piper.MotionCtrl_2(0x01, move_mode, speed, 0x00, 0, installation_pos)
        piper.EndPoseCtrl(*target_i)

        if count % max(1, int(rate_hz / 5)) == 0:
            print_state("cart", read_state(piper), start, target)
        count += 1
        time.sleep(dt)

    final = read_state(piper)
    print_state("cart end", final, start, target)

    pose_delta = max_abs_delta(start.pose, final.pose, 3)
    fk_delta = max_abs_delta(start.fk, final.fk, 3)
    joint_delta = max_abs_delta(start.joints, final.joints)
    pose_err = xyz_error(target, final.pose)
    fk_err = xyz_error(target, final.fk)

    print("\n=== interpretation ===")
    if pose_err <= 5.0 and (math.isnan(fk_err) or fk_err <= 10.0):
        print("Cartesian target was reached within tolerance.")
    elif joint_delta > 0.5 and (math.isnan(fk_delta) or fk_delta > 2.0):
        print("Physical motion likely happened, but it did not converge to the commanded Cartesian target.")
    elif pose_delta > 2.0 and joint_delta <= 0.5:
        print("Only end-pose message changed: joints did not move, so this is not physical motion.")
    elif pose_delta <= 2.0 and joint_delta <= 0.5:
        print("No meaningful movement feedback detected.")
    else:
        print("Mixed result: inspect pose/FK/joint deltas above.")
    print(f"Final pose target error: {pose_err:.3f} mm")
    print(f"Final FK target error: {fk_err:.3f} mm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--can", default="can0", help="CAN device, e.g. can0 or piper_left")
    parser.add_argument("--speed", type=int, default=15, help="Motion speed percent, 0-100")
    parser.add_argument("--rate-hz", type=float, default=50.0, help="Command send rate")
    parser.add_argument("--duration", type=float, default=4.0, help="Seconds for each test")
    parser.add_argument("--mode", choices=("p", "l"), default="p", help="p=MOVE P, l=MOVE L")
    parser.add_argument(
        "--installation-pos",
        type=lambda value: int(value, 0),
        default=0x01,
        choices=(0x01, 0x02, 0x03),
        help="Piper installation position: 1=horizontal upright, 2=left-side mount, 3=right-side mount",
    )
    parser.add_argument("--dx-mm", type=float, default=0.0)
    parser.add_argument("--dy-mm", type=float, default=0.0)
    parser.add_argument("--dz-mm", type=float, default=10.0)
    parser.add_argument(
        "--demo-target",
        choices=("low", "high"),
        help="Use SDK demo target instead of current pose plus delta: low=(57,0,215,0,85,0), high=(57,0,260,0,85,0)",
    )
    parser.add_argument("--skip-joint-test", action="store_true")
    parser.add_argument("--joint-index", type=int, default=1, choices=range(1, 7))
    parser.add_argument("--joint-delta-deg", type=float, default=3.0)
    parser.add_argument(
        "--no-disable-prompt",
        action="store_true",
        help="Exit without asking whether to disable arm motors",
    )
    parser.add_argument("--yes", action="store_true", help="Skip interactive confirmation")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 <= args.speed <= 100:
        raise ValueError("--speed must be in [0, 100]")
    if args.rate_hz <= 0:
        raise ValueError("--rate-hz must be positive")

    stop = False

    def handle_sigint(_signum, _frame):
        nonlocal stop
        stop = True
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_sigint)

    move_mode = 0x02 if args.mode == "l" else 0x00
    demo_targets = {
        "low": [57.0, 0.0, 215.0, 0.0, 85.0, 0.0],
        "high": [57.0, 0.0, 260.0, 0.0, 85.0, 0.0],
    }
    fixed_target = demo_targets.get(args.demo_target)

    print("About to test a real Piper arm.")
    print(f"CAN={args.can}, speed={args.speed}, cartesian delta=({args.dx_mm}, {args.dy_mm}, {args.dz_mm}) mm")
    print(f"installation_pos=0x{args.installation_pos:02X}")
    if fixed_target is not None:
        print(f"Using SDK demo target: {args.demo_target} {fmt(fixed_target)}")
    print("Make sure the workspace is clear and emergency stop is reachable.")
    if not args.yes:
        answer = input("Type YES to continue: ").strip()
        if answer != "YES":
            print("Aborted.")
            return 1

    piper = C_PiperInterface_V2(args.can, dh_is_offset=1, start_sdk_fk_cal=True)
    piper.ConnectPort()
    ensure_enabled(piper)
    enable_fk(piper)
    time.sleep(0.5)

    initial = read_state(piper)
    print("\n=== initial feedback ===")
    print_state("initial", initial)
    print_status(piper)

    try:
        if not args.skip_joint_test:
            joint_smoke_test(
                piper=piper,
                speed=args.speed,
                joint_index=args.joint_index,
                delta_deg=args.joint_delta_deg,
                duration_s=args.duration,
                rate_hz=args.rate_hz,
            )
            time.sleep(0.5)
            print_status(piper)

        cartesian_test(
            piper=piper,
            speed=args.speed,
            move_mode=move_mode,
            installation_pos=args.installation_pos,
            dx_mm=args.dx_mm,
            dy_mm=args.dy_mm,
            dz_mm=args.dz_mm,
            fixed_target=fixed_target,
            duration_s=args.duration,
            rate_hz=args.rate_hz,
        )
        print_status(piper)
    except KeyboardInterrupt:
        if stop:
            print("\nInterrupted by user.")
    finally:
        try:
            current_joints = joints_deg_from_msg(piper)
            for _ in range(10):
                send_hold_joint(piper, current_joints, max(1, args.speed))
                time.sleep(0.02)
            if not args.no_disable_prompt:
                prompt_before_disable(piper)
        except Exception as exc:
            print(f"[warn] failed during final hold/disable prompt: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
