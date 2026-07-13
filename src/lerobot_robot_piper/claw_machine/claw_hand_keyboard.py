#!/usr/bin/env python3
"""Hand setup during the manual keyboard phase."""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from rh56f2_hand import DEFAULT_CLOSED, RH56F2Hand, RH56F2HandConfig
except Exception:  # pragma: no cover - arm-only mode.
    DEFAULT_CLOSED = {}
    RH56F2Hand = None
    RH56F2HandConfig = None

from claw_hand_grasp import set_hand


def connect_hand(args: Namespace) -> object | None:
    if args.no_hand:
        return None
    if RH56F2Hand is None or RH56F2HandConfig is None:
        raise RuntimeError("RH56F2 hand module could not be imported")

    hand = RH56F2Hand(
        RH56F2HandConfig(
            port=args.hand_port,
            hand_id=args.hand_id,
            speed=args.hand_speed,
            force=args.hand_force,
        )
    )
    hand.connect()
    print("RH56F2 hand connected.")
    set_hand(hand, DEFAULT_CLOSED, "initial close")
    return hand


def close_for_keyboard(hand: object | None) -> None:
    set_hand(hand, DEFAULT_CLOSED, "keyboard close")


def disconnect_hand(hand: object | None) -> None:
    if hand is None:
        return
    try:
        hand.disconnect()
    except Exception as exc:
        print(f"[warn] hand disconnect failed: {exc}")
