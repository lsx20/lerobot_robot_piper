#!/usr/bin/env python3
"""Hand-side grasp actions for the claw-machine workflow."""

from __future__ import annotations

import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from rh56f2_hand import DEFAULT_CLOSED, DEFAULT_OPEN
except Exception:  # pragma: no cover - arm-only mode.
    DEFAULT_CLOSED = {}
    DEFAULT_OPEN = {}


BALL_READY_OPEN = dict(DEFAULT_OPEN)
BALL_READY_OPEN.update(
    {
        # Keep the fingers and thumb bend open, but swing the thumb inward
        # while the arm is descending toward the ball.
        "thumb_swing": 900,
    }
)

BALL_CLOSED = dict(DEFAULT_CLOSED)
BALL_CLOSED.update(
    {
        # At grab height, curl the thumb and four fingers while keeping the
        # thumb swung inward around the ball.
        "thumb_bend": 1060,
        "thumb_swing": 900,
    }
)


def set_hand(hand: object | None, pose: dict[str, float], label: str) -> bool:
    if hand is None:
        print(f"{label}: hand disabled; skipped")
        return True
    try:
        hand.set_angles(pose)
    except Exception as exc:
        print(f"[warn] {label}: hand command failed: {exc}")
        return False
    print(f"{label}: hand command sent")
    return True


def set_hand_async(hand: object | None, pose: dict[str, float], label: str) -> None:
    if hand is None:
        print(f"{label}: hand disabled; skipped")
        return

    def worker() -> None:
        set_hand(hand, pose, label)

    threading.Thread(target=worker, daemon=True).start()


def open_while_descending(hand: object | None) -> None:
    set_hand_async(
        hand,
        BALL_READY_OPEN,
        "open fingers and swing thumb inward while descending",
    )


def close_at_grab(hand: object | None) -> bool:
    return set_hand(hand, BALL_CLOSED, "close ball grasp")


def open_at_drop(hand: object | None) -> bool:
    return set_hand(hand, DEFAULT_OPEN, "open")


def close_while_returning(hand: object | None) -> None:
    set_hand_async(hand, DEFAULT_CLOSED, "close while returning")
