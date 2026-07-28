#!/usr/bin/env python3
"""Run pretrained YOLO sports-ball detection on the hand-mounted D405."""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", default="260322279862")
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def depth_at_box(depth_frame: object, depth_image: np.ndarray, box: tuple[int, int, int, int], depth_scale: float) -> tuple[float, tuple[int, int]] | None:
    x0, y0, x1, y1 = box
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    inner_x0 = x0 + int(width * 0.30)
    inner_x1 = x1 - int(width * 0.30)
    inner_y0 = y0 + int(height * 0.30)
    inner_y1 = y1 - int(height * 0.30)
    region = depth_image[inner_y0:inner_y1, inner_x0:inner_x1].astype(np.float32) * depth_scale
    valid = region[region > 0.05]
    if valid.size == 0:
        return None
    depth_m = float(np.median(valid))
    center = ((x0 + x1) // 2, (y0 + y1) // 2)
    return depth_m, center


def main() -> int:
    args = parse_args()
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise SystemExit("pyrealsense2 is not installed in this Python environment") from exc

    model = YOLO(args.model)
    ball_class_ids = {class_id for class_id, name in model.names.items() if name == "sports ball"}
    if not ball_class_ids:
        raise SystemExit(f"Model does not contain a 'sports ball' class: {model.names}")

    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    align = rs.align(rs.stream.color)
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())

    print("SAFETY: YOLO and D405 read-only test.")
    print("No Piper connection, enable, disable, or motion commands are used.")
    print("Press q to quit; press p to print the best detected ball.")
    last_print: tuple[float, tuple[float, float, float], float] | None = None
    try:
        while True:
            frames = align.process(pipeline.wait_for_frames())
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            color = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            result = model.predict(color, conf=args.conf, imgsz=args.imgsz, device=0, verbose=False)[0]
            display = color.copy()
            detections: list[tuple[float, tuple[int, int, int, int], float, tuple[float, float, float]]] = []
            intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
            if result.boxes is not None:
                for box_tensor, confidence_tensor, class_tensor in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                    class_id = int(class_tensor.item())
                    confidence = float(confidence_tensor.item())
                    if class_id not in ball_class_ids:
                        continue
                    x0, y0, x1, y1 = [int(value) for value in box_tensor.tolist()]
                    x0 = max(0, min(args.width - 1, x0))
                    y0 = max(0, min(args.height - 1, y0))
                    x1 = max(x0 + 1, min(args.width, x1))
                    y1 = max(y0 + 1, min(args.height, y1))
                    depth_result = depth_at_box(depth_frame, depth_image, (x0, y0, x1, y1), depth_scale)
                    if depth_result is None:
                        continue
                    depth_m, center = depth_result
                    point = rs.rs2_deproject_pixel_to_point(intrinsics, list(center), depth_m)
                    point_xyz = tuple(float(value) for value in point)
                    detections.append((confidence, (x0, y0, x1, y1), depth_m, point_xyz))
                    cv2.rectangle(display, (x0, y0), (x1, y1), (0, 255, 0), 2)
                    cv2.drawMarker(display, center, (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
                    cv2.putText(display, f"BALL {confidence:.2f} depth={depth_m:.3f}m", (x0, max(24, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 0), 2)
            detections.sort(key=lambda item: item[0], reverse=True)
            if detections:
                best = detections[0]
                last_print = (best[2], best[3], best[0])
                cv2.putText(display, f"balls={len(detections)}  q:quit p:print", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            else:
                cv2.putText(display, "BALL NOT FOUND  q:quit", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 255), 2)
            cv2.imshow("YOLO D405 ball test", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("p") and last_print is not None:
                print(f"confidence={last_print[2]:.3f} depth_m={last_print[0]:.4f} camera_xyz_m={last_print[1]}")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
