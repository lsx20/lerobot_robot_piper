import logging
import time
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots.robot import Robot
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from .config_piper_rh56f2_follower import PiperRH56F2FollowerConfig
from .piper_follower import JOINT_LIMITS_DEG, JOINT_NAMES
from .rh56f2_hand import HAND_NAMES, RH56F2Hand, RH56F2HandConfig

logger = logging.getLogger(__name__)


def _clip_step(goal: float, current: float, max_delta: float | None) -> float:
    if max_delta is None:
        return goal
    return current + float(np.clip(goal - current, -max_delta, max_delta))


class PiperRH56F2Follower(Robot):
    """LeRobot-compatible Piper arm + RH56F2 dexterous hand.

    API units:
      - arm joint positions: degrees
      - hand positions: RH56F2 register angle units
    """

    config_class = PiperRH56F2FollowerConfig
    name = "piper_rh56f2_follower"

    def __init__(self, config: PiperRH56F2FollowerConfig):
        super().__init__(config)
        self.config = config
        self.piper: Any = None
        self.hand: RH56F2Hand | None = None
        self._is_connected = False
        self.cameras = make_cameras_from_configs(config.cameras)

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        features: dict[str, type | tuple] = {f"{name}.pos": float for name in JOINT_NAMES}
        features.update({f"hand.{name}.pos": float for name in HAND_NAMES})
        for cam_name in self.cameras:
            cam_cfg = self.config.cameras[cam_name]
            features[cam_name] = (cam_cfg.height, cam_cfg.width, 3)
        return features

    @cached_property
    def action_features(self) -> dict[str, type]:
        features: dict[str, type] = {f"{name}.pos": float for name in JOINT_NAMES}
        features.update({f"hand.{name}.pos": float for name in HAND_NAMES})
        return features

    @property
    def is_connected(self) -> bool:
        return (
            self._is_connected
            and self.hand is not None
            and self.hand.is_connected
            and all(cam.is_connected for cam in self.cameras.values())
        )

    @property
    def is_calibrated(self) -> bool:
        return True

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        from piper_sdk import C_PiperInterface_V2

        self.piper = C_PiperInterface_V2(self.config.can_port)
        self.piper.ConnectPort()
        time.sleep(0.2)

        self.piper.MotionCtrl_1(0x02, 0x00, 0x02)
        time.sleep(0.05)
        self.piper.MotionCtrl_2(0x00, 0x01, 0, 0x00)
        time.sleep(0.05)
        self.piper.MotionCtrl_2(0x01, 0x01, self.config.speed_rate, 0x00)

        start = self._arm_current_deg()
        while not self.piper.EnablePiper():
            time.sleep(0.01)
        self._send_arm_deg(start, clip_limits=False)

        self.hand = RH56F2Hand(
            RH56F2HandConfig(
                port=self.config.hand_port,
                baudrate=self.config.hand_baudrate,
                hand_id=self.config.hand_id,
                speed=self.config.hand_speed,
                force=self.config.hand_force,
            )
        )
        self.hand.connect()

        for cam in self.cameras.values():
            cam.connect()

        self._is_connected = True
        logger.info("PiperRH56F2Follower connected: can=%s hand=%s", self.config.can_port, self.config.hand_port)

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        if self.hand is not None:
            self.hand.configure()

    def _arm_current_deg(self) -> dict[str, float]:
        joint_msgs = self.piper.GetArmJointMsgs()
        js = joint_msgs.joint_state
        values = [js.joint_1, js.joint_2, js.joint_3, js.joint_4, js.joint_5, js.joint_6]
        return {f"{name}.pos": value / 1000.0 for name, value in zip(JOINT_NAMES, values, strict=True)}

    def _send_arm_deg(self, arm_action: dict[str, float], clip_limits: bool = True) -> None:
        current = self._arm_current_deg()
        values: list[int] = []
        for name in JOINT_NAMES:
            key = f"{name}.pos"
            goal = float(arm_action.get(key, current[key]))
            should_clip = (
                clip_limits
                and self.config.clip_arm_to_sdk_limits
                and (name != "joint_6" or self.config.clip_joint6_to_sdk_limits)
            )
            if should_clip:
                lo, hi = JOINT_LIMITS_DEG[name]
                goal = float(np.clip(goal, lo, hi))
            goal = _clip_step(goal, current[key], self.config.max_arm_delta_deg)
            values.append(int(round(goal * 1000)))
        self.piper.MotionCtrl_2(0x01, 0x01, self.config.speed_rate, 0x00)
        self.piper.JointCtrl(*values)

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        obs: RobotObservation = {}
        obs.update(self._arm_current_deg())

        hand_pos = self.hand.read_positions("angleAct")
        obs.update({f"hand.{name}.pos": value for name, value in hand_pos.items()})

        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.read_latest()

        return obs

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        sent: RobotAction = {}

        arm_action = {key: float(value) for key, value in action.items() if key in {f"{n}.pos" for n in JOINT_NAMES}}
        if arm_action:
            self._send_arm_deg(arm_action)
            sent.update(arm_action)

        hand_action = {}
        for name in HAND_NAMES:
            key = f"hand.{name}.pos"
            if key in action:
                hand_action[name] = float(action[key])
        if hand_action:
            current = self.hand.read_positions("angleAct")
            clipped = {
                name: _clip_step(value, current[name], self.config.max_hand_delta)
                for name, value in hand_action.items()
            }
            self.hand.set_angles(clipped)
            sent.update({f"hand.{name}.pos": value for name, value in clipped.items()})

        return sent

    @check_if_not_connected
    def disconnect(self) -> None:
        if self.piper is not None:
            print()
            print("WARNING: disabling Piper motors may make the arm drop.")
            print("Hold/support the arm and hand before disabling.")
            confirm = input(
                "Type D then Enter to disable arm motors, or press Enter to keep motors enabled: "
            ).strip()
            if confirm == "D":
                self.piper.DisableArm(7)
                logger.info("Piper arm motors disabled by user confirmation.")
            else:
                logger.info("Piper arm motors left enabled; no DisableArm command sent.")
        if self.hand is not None:
            self.hand.disconnect()
        for cam in self.cameras.values():
            cam.disconnect()
        self._is_connected = False
        logger.info("PiperRH56F2Follower disconnected.")
