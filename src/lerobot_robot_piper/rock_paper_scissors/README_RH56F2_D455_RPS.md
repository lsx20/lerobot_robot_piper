# RH56F2 + D455 Rock Paper Scissors

This directory contains a RH56F2 version of the OmniHand/MediaPipe demo from
the reference article.

The reference article uses OmniHand 2025 with a CAN-FD SDK:

- `OmniHand2025`
- `create_hand_by_hcan`
- ten active joint angles

This project uses your existing RH56F2 RS485 driver instead:

- `RH56F2Hand`
- `/dev/ttyUSB0`
- six register angles: little, ring, middle, index, thumb_bend, thumb_swing

## 1. Install Camera And Vision Dependencies

First make sure the D455 works in Intel RealSense Viewer.

Then install Python dependencies.

Check the Python version inside the active environment first:

```bash
python3 --version
```

For Python 3.13, use the newer MediaPipe wheel. Do not pin numpy to 1.26.4:

```bash
python3 -m pip install pyrealsense2 mediapipe==0.10.35
```

Download the MediaPipe Gesture Recognizer model once:

```bash
wget -O gesture_recognizer.task https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task
```

For Python 3.10 to 3.12, the article's old version can work, but the simpler
command below is usually enough:

```bash
python3 -m pip install pyrealsense2 mediapipe
```

The demo script uses the newer MediaPipe Tasks API, so it also needs
`gesture_recognizer.task` in this directory, or a path passed with
`--gesture-model`.

If your OpenCV build has issues after changing numpy, reinstall OpenCV in the
same Python environment.

## 2. Find The D455

Use one of these:

```bash
realsense-viewer
lerobot-find-cameras realsense
```

Write down the D455 serial number. You can still run these scripts without
`--serial` if only one RealSense camera is connected.

## 3. Test D455 Only

```bash
python3 test_d455_camera.py
```

With an explicit serial number:

```bash
python3 test_d455_camera.py --serial 1234567890
```

Press `q` to quit.

## 4. Test RH56F2 Only

Keep the arm still and make sure the hand has empty space around it.

```bash
python3 test_rh56f2_rps.py --port /dev/ttyUSB0 --cycle
```

If the hand has not reached each pose before the next command, increase the
cycle delay:

```bash
python3 test_rh56f2_rps.py --port /dev/ttyUSB0 --cycle --delay 4
```

To test faster gesture motion, increase hand speed and shorten the staged
delay:

```bash
python3 test_rh56f2_rps.py --port /dev/ttyUSB0 --cycle --speed 1200 --stage-delay 0.05 --delay 1
```

If the USB-RS485 adapter appears as another port:

```bash
ls /dev/ttyUSB* /dev/ttyACM*
```

Then pass the correct port with `--port`.

## 5. Run The Full Demo

First run dry mode. This uses the camera and recognition, but does not connect
the RH56F2:

```bash
python3 rh56f2_rps_demo.py --dry-run
```

For vision debugging, print the raw MediaPipe category and confidence:

```bash
python3 rh56f2_rps_demo.py --dry-run --print-vision
```

If the raw category stays `None` but `hands=1`, the script falls back to a
landmark geometry classifier. In that case, check the `fallback=` field. The
demo can trigger as long as `fallback` becomes `Rock`, `Paper`, or `Scissors`.

If a hand is visible but the score is low, lower the thresholds:

```bash
python3 rh56f2_rps_demo.py --dry-run --print-vision --min-score 0.2 --min-detection 0.4 --min-presence 0.4
```

If the model is somewhere else:

```bash
python3 rh56f2_rps_demo.py --dry-run --gesture-model /path/to/gesture_recognizer.task
```

Then connect the RH56F2:

```bash
python3 rh56f2_rps_demo.py --hand-port /dev/ttyUSB0
```

For a faster game, increase hand speed and shorten stage, settle, and cooldown
times:

```bash
python3 rh56f2_rps_demo.py --hand-port /dev/ttyUSB0 --hand-speed 1200 --stage-delay 0.05 --motion-settle 0.5 --cooldown 0.5
```

If the hand changes too quickly or does not finish moving, increase them:

```bash
python3 rh56f2_rps_demo.py --hand-port /dev/ttyUSB0 --motion-settle 3 --cooldown 3
```

With an explicit D455 serial number:

```bash
python3 rh56f2_rps_demo.py --serial 1234567890 --hand-port /dev/ttyUSB0
```

Press `q` to quit.

## Gesture Mapping

Human gesture from MediaPipe:

- Rock: zero or one extended finger
- Scissors: index and middle extended
- Paper: three or four extended fingers

Robot reply:

- Human Rock -> RH56F2 Paper
- Human Paper -> RH56F2 Scissors
- Human Scissors -> RH56F2 Rock

## RH56F2 Pose Tuning

The current pose values are conservative starting points:

- Rock: `DEFAULT_CLOSED`
- Paper: `DEFAULT_OPEN`
- Scissors: index and middle open, ring and little closed

If your RH56F2 finger directions differ, tune the values in
`test_rh56f2_rps.py` first. After the hand looks correct, copy the same values
into `rh56f2_rps_demo.py`.

Move in small increments, about 50 to 100 register units at a time.
