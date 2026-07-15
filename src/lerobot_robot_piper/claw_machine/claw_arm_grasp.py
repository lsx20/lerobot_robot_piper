#!/usr/bin/env python3
"""Arm-side MOVE_P pick/drop cycle for the claw-machine workflow."""

from __future__ import annotations

import time
from argparse import Namespace

from piper_sdk import C_PiperInterface_V2

from claw_hand_grasp import (
    close_at_grab,
    close_while_returning,
    open_at_drop,
    open_while_descending,
)
from claw_init import pose_mm_deg, send_movep_for


def run_pick_cycle(
    piper: C_PiperInterface_V2,
    hand: object | None,
    args: Namespace,
    start_pose: list[int],
    hover_pose: list[int],
    drop_pose: list[int],
) -> bool:
    grab_pose = list(hover_pose)
    grab_pose[2] = int(round(args.grab_z * 1000.0))
    lift_pose = list(hover_pose)
    if args.lift_z is not None:
        lift_pose[2] = int(round(args.lift_z * 1000.0))

    print()
    print("Running pick cycle")
    print(f"  hover: {pose_mm_deg(hover_pose)}")
    print(f"  grab:  {pose_mm_deg(grab_pose)}")
    print(f"  lift:  {pose_mm_deg(lift_pose)}")
    print(f"  drop:  {pose_mm_deg(drop_pose)}")
    print(f"  start: {pose_mm_deg(start_pose)}")

    open_while_descending(hand)
    if not send_movep_for(
        piper,
        grab_pose,
        args.speed,
        args.vertical_duration,
        args.rate_hz,
        "descend",
        args.auto_position_tolerance_mm,
        args.auto_rpy_tolerance_deg,
        True,
    ):
        print("[warn] descend failed")
        return False

    if args.pre_grab_open_settle > 0:
        time.sleep(args.pre_grab_open_settle)
    close_at_grab(hand)
    time.sleep(args.hand_settle)

    if not send_movep_for(
        piper,
        lift_pose,
        args.speed,
        args.vertical_duration,
        args.rate_hz,
        "lift",
        args.auto_position_tolerance_mm,
        args.auto_rpy_tolerance_deg,
        True,
    ):
        print("[warn] lift failed")
        return False

    if not send_movep_for(
        piper,
        drop_pose,
        args.speed,
        args.transfer_duration,
        args.rate_hz,
        "drop move",
        args.auto_position_tolerance_mm,
        args.auto_rpy_tolerance_deg,
        True,
    ):
        print("[warn] drop move failed")
        return False

    open_at_drop(hand)
    time.sleep(args.drop_open_settle)

    close_while_returning(hand)
    if not send_movep_for(
        piper,
        start_pose,
        args.speed,
        args.return_duration,
        args.rate_hz,
        "return",
        args.auto_position_tolerance_mm,
        args.auto_rpy_tolerance_deg,
        True,
    ):
        print("[warn] return failed")
        return False
    return True
