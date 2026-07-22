# Piper + RH56F2 VR Teleop

This folder is the integration point for VR teleoperation.

## Goal

Use VR hand tracking to control both:

- Piper arm end-effector pose: `ee.x`, `ee.y`, `ee.z`, `ee.rx`, `ee.ry`, `ee.rz`
- RH56F2 dexterous hand angles: `hand.<finger>.pos`

The hardware side reuses `PiperRH56F2Follower`, which already contains the
lessons from the claw-machine debugging work:

- Piper MOVE_P end-effector control through `ee.*`
- RH56F2 RS485 hand control through `hand.*.pos`
- hand force feedback through `hand.*.force`
- per-step motion limits for arm and hand
- no automatic motor disable on exit unless the user confirms

## Current Bridge

`piper_rh56f2_vr_teleop.py` accepts normalized JSON frames from stdin.

Example dry run:

```bash
printf '%s\n' \
  '{"deadman": false, "wrist_xyz_m": [0.0, 0.0, 0.0], "finger_curls": {"all": 0.0}}' \
  '{"deadman": true, "wrist_xyz_m": [0.02, 0.00, 0.00], "finger_curls": {"all": 0.5}}' \
  | python3 -m lerobot_robot_piper.vr_teleop.piper_rh56f2_vr_teleop
```

Real hardware mode must be explicit:

```bash
python3 -m lerobot_robot_piper.vr_teleop.piper_rh56f2_vr_teleop \
  --connect \
  --can can0 \
  --hand-port /dev/ttyUSB0
```

## Expected VR Frame

```json
{
  "deadman": true,
  "wrist_xyz_m": [0.02, 0.00, 0.01],
  "wrist_rpy_deg": [0.0, 0.0, 5.0],
  "finger_curls": {
    "thumb_bend": 0.2,
    "thumb_swing": 0.1,
    "index": 0.7,
    "middle": 0.7,
    "ring": 0.4,
    "little": 0.4
  }
}
```

## D455 Hand-Only Teleoperation

The D455 can now be used instead of Quest 3 for the first phase. The input
module reads a color frame, runs the existing `gesture_recognizer.task` hand
model, flattens the 21 landmarks into 63 numbers, computes normalized finger
curls, and creates the same `VRFrame` used by the Quest 3 path.

The data path is:

```text
D455 color image -> MediaPipe landmarks -> finger_curls -> VRFrame
    -> RH56F2SimpleRetargeter -> hand.<finger>.pos -> /dev/ttyUSB0
```

Install the RealSense Python binding in the same environment used to run the
teleop command:

```bash
python3 -m pip install pyrealsense2
```

First run a dry-run. It opens the D455 but does not connect to hardware and
prints the generated RH56F2 actions. With `--show-camera`, open the printed
browser URL, normally `http://127.0.0.1:8765/`, to view the live annotated
stream:

```bash
PYTHONPATH=/home/zhiyu/robot_ws/lerobot_robot_piper/src \
python3 -u -m lerobot_robot_piper.vr_teleop.piper_rh56f2_vr_teleop \
  --input-source realsense \
  --show-camera
```

The browser preview is used instead of an OpenCV window because MediaPipe's
EGL context can block Qt/OpenCV GUI calls on some desktop environments.

For the first real test, connect only the hand. Keep the Piper arm disabled:

```bash
PYTHONPATH=/home/zhiyu/robot_ws/lerobot_robot_piper/src \
python3 -u -m lerobot_robot_piper.vr_teleop.piper_rh56f2_vr_teleop \
  --input-source realsense \
  --show-camera \
  --connect --hand-only \
  --hand-port /dev/ttyUSB0 \
  --hand-speed 300 \
  --max-hand-delta 30 \
  --thumb-swing-closed 500
```

Removing the hand from the camera causes the deadman state to become false
and stops new motion commands. Press `Ctrl-C` to stop. This phase uses the
image-normalized wrist coordinates only as a placeholder; they are ignored by
`HandOnlyRobot` and are not yet used to move Piper.

## Next Step

The simple curl mapping is only the first hardware validation layer. The
upstream Quest 3 UDP format is now accepted directly with `--input-source
quest3`. After the arm and hand follow safely, replace
`RH56F2SimpleRetargeter` with an AnyDexRetarget-backed retargeter that converts
full hand landmarks into RH56F2 joint targets.

Quest 3 input:

```bash
python3 -m lerobot_robot_piper.vr_teleop.piper_rh56f2_vr_teleop \
  --input-source quest3 --port 9000
```

The command above is dry-run by default. Add `--connect` only after the input
and mapping have been checked. The first real-hardware test should use low
`--speed`, `--max-ee-delta-mm`, and `--max-hand-delta` values.

## AnyDex Mode

AnyDexRetarget has models for several hands, but not RH56F2. Therefore its
output must be reduced to five finger-curl values before RH56F2 register
angles are sent. The qpos groups are ordered as:

```text
thumb,index,middle,ring,little
```

Example command using an Inspire model as the retargeting front end:

```bash
python3 -m lerobot_robot_piper.vr_teleop.piper_rh56f2_vr_teleop \
  --input-source quest3 \
  --hand-mode anydex \
  --anydex-root /path/to/AnyDexRetarget \
  --hand-config /path/to/AnyDexRetarget/example/config/adaptive/quest3/quest3_inspire_hand.yaml \
  --anydex-qpos-groups '0,1;2,3;4,5;6,7;8,9'
```

The example qpos groups are placeholders until RH56F2 is calibrated against
the selected AnyDex model. Do not connect real hardware until each finger
group and its open/closed direction has been checked in dry-run output.
