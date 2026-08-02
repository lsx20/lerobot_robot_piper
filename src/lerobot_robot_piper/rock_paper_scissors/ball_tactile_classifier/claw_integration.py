#!/usr/bin/env python3
"""Ball classification adapter for the claw-machine pick cycle.

This module assumes the arm has already grasped and lifted the object. It only
reads RH56F2 tactile feedback and optionally performs the A/B squeeze check.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

from . import collect_lift_samples as lift
from .common import (
    FINGER_NAMES,
    TouchFrame,
    TouchTrial,
    ab_shape_decision,
    append_feature_row,
    detect_missed_grasp,
    extract_features,
    load_model,
    now_trial_id,
    predict_row,
)
from .predict_live import (
    apply_proximity_assist,
    friction_ab_label,
    init_ab_proximity_fields,
    load_ab_squeeze_references,
    quality_number,
    squeeze_ab_curve_label,
    squeeze_ab_label,
    squeeze_series_from_row,
)
from .visualize_live import (
    render_dashboard_from_csv,
    render_dashboard_preview_from_csv,
    render_pending_dashboard_from_csv,
)


DEFAULT_MODEL = Path(__file__).with_name("model_with_newB.json")
DEFAULT_OUTPUT = Path(__file__).with_name("claw_predictions.csv")
DEFAULT_DASHBOARD = Path(__file__).with_name("live_dashboard.html")
DEFAULT_REFERENCE_SAMPLES = Path(__file__).with_name("samples_with_newB.csv")


@dataclass
class BallClassifierConfig:
    model: Path = DEFAULT_MODEL
    output: Path = DEFAULT_OUTPUT
    visual_reference_samples: Path = DEFAULT_REFERENCE_SAMPLES
    contact_threshold: float = 70.0
    min_active_contacts: float | None = None
    min_hover_samples: float | None = None
    hover_duration: float = 1.5
    hover_rate_hz: float = 10.0
    hand_read_retries: int = 5
    hand_read_retry_delay: float = 0.03
    tactile_data: bool = True
    tactile_required: bool = False
    squeeze_delta: float = 40.0
    squeeze_pre_duration: float = 0.4
    squeeze_duration: float = 3.0
    squeeze_rate_hz: float = 20.0
    squeeze_max_force_delta: float = 900.0
    squeeze_baseline_hover_samples: int = 10
    squeeze_middle_touch_threshold: float = 80.0
    squeeze_middle_seek_step: float = 15.0
    squeeze_middle_seek_max_delta: float = 160.0
    squeeze_middle_seek_settle: float = 0.08
    squeeze_test: bool = False
    ab_squeeze_test: bool = False
    ab_squeeze_threshold: float = 190.0
    ab_squeeze_a_standard: float = 238.0
    ab_squeeze_b_standard: float = 142.5
    ab_squeeze_mode: str = "friction"
    ab_squeeze_reference_samples: Path = DEFAULT_REFERENCE_SAMPLES
    low_confidence_c_squeeze_threshold: float = 0.0
    ab_friction_finger: str = "middle"
    ab_friction_feature: str = "last"
    ab_friction_threshold: float = 0.1464
    ab_friction_a_direction: str = ">="
    ab_shape_late_slope_threshold: float = -5.0
    ab_shape_rebound_threshold: float = 15.0
    ab_shape_peak_pos_threshold: float = 0.5
    ab_shape_min_a_score: int = 2
    ab_proximity_assist: bool = True
    ab_proximity_index_force_threshold: float = 70.0
    ab_proximity_thumb_threshold: float = 169619.0
    ab_proximity_a_direction: str = "<="
    ab_proximity_min_samples: float = 5.0
    bc_proximity_assist: bool = True
    bc_proximity_thumb_threshold: float = 180000.0
    bc_proximity_middle_threshold: float = 100000.0
    notes: str = "claw_machine"


def _number(value: object, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _lift_args(config: BallClassifierConfig) -> argparse.Namespace:
    return argparse.Namespace(
        hover_duration=config.hover_duration,
        hover_rate_hz=config.hover_rate_hz,
        hand_read_retries=config.hand_read_retries,
        hand_read_retry_delay=config.hand_read_retry_delay,
        tactile_data=config.tactile_data,
        tactile_required=config.tactile_required,
        squeeze_delta=config.squeeze_delta,
        squeeze_pre_duration=config.squeeze_pre_duration,
        squeeze_duration=config.squeeze_duration,
        squeeze_rate_hz=config.squeeze_rate_hz,
        squeeze_max_force_delta=config.squeeze_max_force_delta,
        squeeze_baseline_hover_samples=config.squeeze_baseline_hover_samples,
        squeeze_middle_touch_threshold=config.squeeze_middle_touch_threshold,
        squeeze_middle_seek_step=config.squeeze_middle_seek_step,
        squeeze_middle_seek_max_delta=config.squeeze_middle_seek_max_delta,
        squeeze_middle_seek_settle=config.squeeze_middle_seek_settle,
        squeeze_test=config.squeeze_test,
        ab_squeeze_test=config.ab_squeeze_test,
        ab_squeeze_threshold=config.ab_squeeze_threshold,
        ab_squeeze_a_standard=config.ab_squeeze_a_standard,
        ab_squeeze_b_standard=config.ab_squeeze_b_standard,
        ab_squeeze_mode=config.ab_squeeze_mode,
        low_confidence_c_squeeze_threshold=config.low_confidence_c_squeeze_threshold,
        ab_friction_finger=config.ab_friction_finger,
        ab_friction_feature=config.ab_friction_feature,
        ab_friction_threshold=config.ab_friction_threshold,
        ab_friction_a_direction=config.ab_friction_a_direction,
        ab_shape_late_slope_threshold=config.ab_shape_late_slope_threshold,
        ab_shape_rebound_threshold=config.ab_shape_rebound_threshold,
        ab_shape_peak_pos_threshold=config.ab_shape_peak_pos_threshold,
        ab_shape_min_a_score=config.ab_shape_min_a_score,
        ab_proximity_assist=config.ab_proximity_assist,
        ab_proximity_index_force_threshold=config.ab_proximity_index_force_threshold,
        ab_proximity_thumb_threshold=config.ab_proximity_thumb_threshold,
        ab_proximity_a_direction=config.ab_proximity_a_direction,
        ab_proximity_min_samples=config.ab_proximity_min_samples,
        bc_proximity_assist=config.bc_proximity_assist,
        bc_proximity_thumb_threshold=config.bc_proximity_thumb_threshold,
        bc_proximity_middle_threshold=config.bc_proximity_middle_threshold,
    )


class HeldBallClassifier:
    """Two-stage classifier for an object already held by the claw machine."""

    def __init__(self, config: BallClassifierConfig):
        self.config = config
        self.model = load_model(config.model)
        self.min_active_contacts = (
            float(self.model.get("min_active_contacts", 0.0))
            if config.min_active_contacts is None
            else float(config.min_active_contacts)
        )
        self.min_hover_samples = (
            float(self.model.get("min_hover_samples", 0.0))
            if config.min_hover_samples is None
            else float(config.min_hover_samples)
        )
        self.ab_squeeze_references = load_ab_squeeze_references(config.ab_squeeze_reference_samples)

    def begin_trial(self, hand: object, repeat_index: int = 0) -> TouchTrial:
        args = _lift_args(self.config)
        baseline = lift.read_touch_frame(hand, 0.0, args)
        try:
            render_pending_dashboard_from_csv(
                self.config.output,
                DEFAULT_DASHBOARD,
                last=20,
                reference_samples=self.config.visual_reference_samples,
                notes="claw grasp in progress",
            )
        except Exception as exc:
            print(f"[warn] pending live dashboard refresh failed: {exc}")
        return TouchTrial(
            label="unknown",
            trial_id=now_trial_id(),
            repeat_index=repeat_index,
            baseline_angles=baseline.angles,
            baseline_forces=baseline.forces,
            contact_threshold=self.config.contact_threshold,
            notes=self.config.notes,
        )

    def record_grasp_frame(self, hand: object, trial: TouchTrial) -> None:
        args = _lift_args(self.config)
        trial.frames.append(lift.read_touch_frame(hand, 0.0, args))

    def record_observation_frame(
        self,
        trial: TouchTrial,
        observation: dict[str, object],
        started_at: float,
    ) -> None:
        angles = {
            name: float(observation.get(f"hand.{name}.pos", trial.baseline_angles.get(name, 0.0)))
            for name in FINGER_NAMES
        }
        forces = {
            name: float(observation.get(f"hand.{name}.force", trial.baseline_forces.get(name, 0.0)))
            for name in FINGER_NAMES
        }
        trial.frames.append(TouchFrame(time.monotonic() - started_at, angles, forces))

    def refresh_decision_dashboard(self, row: dict[str, object]) -> None:
        try:
            render_dashboard_preview_from_csv(
                self.config.output,
                DEFAULT_DASHBOARD,
                row,
                last=20,
                reference_samples=self.config.visual_reference_samples,
            )
        except Exception as exc:
            print(f"[warn] decision dashboard refresh failed: {exc}")

    def classify_held(self, hand: object, trial: TouchTrial) -> dict[str, object]:
        args = _lift_args(self.config)
        if not trial.frames:
            self.record_grasp_frame(hand, trial)

        lift.collect_hover(hand, trial, args)
        row = extract_features(trial)

        active_contacts = _number(row.get("active_contact_count"))
        hover_samples = _number(row.get("hover_sample_count"))
        if active_contacts < self.min_active_contacts or hover_samples < self.min_hover_samples:
            row["prediction_status"] = "skipped_quality"
            row["predicted_label"] = ""
            row["prediction_confidence"] = ""
            row["prediction_distance"] = ""
            self.refresh_decision_dashboard(row)
            append_feature_row(self.config.output, row)
            print(
                "ball_type=unknown "
                f"status=skipped_quality active={active_contacts:.0f}/{self.min_active_contacts:g} "
                f"hover_samples={hover_samples:.0f}/{self.min_hover_samples:g}"
            )
            return row

        miss_hit, miss_stats = detect_missed_grasp(row)
        row["miss_grasp_active_contact_count"] = miss_stats["active_contact_count"]
        row["miss_grasp_size_closure_mean"] = miss_stats["size_closure_mean"]
        row["miss_grasp_size_contact_angle_mean"] = miss_stats["size_contact_angle_mean"]
        row["miss_grasp_hover_sample_count"] = miss_stats["hover_sample_count"]
        row["miss_grasp_hover_thumb_force_delta_mean"] = miss_stats["hover_thumb_force_delta_mean"]
        row["miss_grasp_hover_force_delta_sum_mean"] = miss_stats["hover_force_delta_sum_mean"]
        row["miss_grasp_final_force_delta_sum"] = miss_stats["final_force_delta_sum"]
        row["miss_grasp_low_hover_after_lift"] = miss_stats["low_hover_after_lift"]
        row["miss_grasp_deep_empty_close"] = miss_stats["deep_empty_close"]
        row["ab_squeeze_triggered"] = 0.0
        row["ab_squeeze_threshold"] = float(args.ab_squeeze_threshold)
        row["ab_squeeze_a_standard"] = float(args.ab_squeeze_a_standard)
        row["ab_squeeze_b_standard"] = float(args.ab_squeeze_b_standard)
        row["ab_squeeze_middle_force_delta_max"] = ""
        row["ab_squeeze_curve_distance_a"] = ""
        row["ab_squeeze_curve_distance_b"] = ""
        row["ab_shape_late_slope"] = ""
        row["ab_shape_rebound"] = ""
        row["ab_shape_peak_pos"] = ""
        row["ab_shape_score_a"] = ""
        row["ab_friction_finger"] = args.ab_friction_finger
        row["ab_friction_feature"] = args.ab_friction_feature
        row["ab_friction_value"] = ""
        row["ab_friction_threshold"] = float(args.ab_friction_threshold)
        row["ab_friction_a_direction"] = args.ab_friction_a_direction
        init_ab_proximity_fields(row, args)
        if miss_hit:
            row["prediction_status"] = "miss_grasp"
            row["predicted_label"] = "NONE"
            row["prediction_confidence"] = 1.0
            row["prediction_distance"] = ""
            row["stage1_predicted_label"] = ""
            row["stage1_prediction_confidence"] = ""
            row["stage1_prediction_distance"] = ""
            self.refresh_decision_dashboard(row)
            append_feature_row(self.config.output, row)
            print(
                "ball_type=NONE "
                "stage1=- "
                "confidence=1.00 "
                "status=miss_grasp "
                f"active={miss_stats['active_contact_count']:.0f} "
                f"closure={miss_stats['size_closure_mean']:.1f} "
                f"angle={miss_stats['size_contact_angle_mean']:.1f} "
                f"hover={miss_stats['hover_sample_count']:.0f} "
                f"hover_sum={miss_stats['hover_force_delta_sum_mean']:.1f} "
                f"thumb={miss_stats['hover_thumb_force_delta_mean']:.1f} "
                f"force_sum={miss_stats['final_force_delta_sum']:.1f} "
                f"low_hover={miss_stats['low_hover_after_lift']:.0f}"
            )
            return row

        stage1 = predict_row(row, self.model)
        stage1_label = str(stage1["label"])
        row["stage1_predicted_label"] = stage1_label
        row["stage1_prediction_confidence"] = float(stage1["confidence"])
        row["stage1_prediction_distance"] = float(stage1["distance"])
        print(
            f"stage1_prediction={stage1_label} "
            f"confidence={float(stage1['confidence']):.2f} "
            f"distance={float(stage1['distance']):.2f}"
        )

        low_confidence_c = (
            stage1_label == "C"
            and float(stage1["confidence"]) < args.low_confidence_c_squeeze_threshold
        )
        squeeze_triggered = (
            args.squeeze_test
            or (args.ab_squeeze_test and stage1_label in {"A", "B"})
            or low_confidence_c
        )
        if squeeze_triggered:
            if stage1_label in {"A", "B"} and args.ab_squeeze_test:
                reason = "A/B hardness check"
            elif low_confidence_c:
                reason = (
                    "low-confidence C fallback "
                    f"({float(stage1['confidence']):.2f} < {args.low_confidence_c_squeeze_threshold:.2f})"
                )
            else:
                reason = "forced squeeze test"
            print(f"squeeze test triggered: {reason}")
            lift.collect_squeeze(hand, trial, args)
            row = extract_features(trial)
            row["stage1_predicted_label"] = stage1_label
            row["stage1_prediction_confidence"] = float(stage1["confidence"])
            row["stage1_prediction_distance"] = float(stage1["distance"])
            row["ab_squeeze_triggered"] = 1.0
            row["ab_squeeze_threshold"] = float(args.ab_squeeze_threshold)
            row["ab_squeeze_a_standard"] = float(args.ab_squeeze_a_standard)
            row["ab_squeeze_b_standard"] = float(args.ab_squeeze_b_standard)
            row["ab_squeeze_middle_force_delta_max"] = quality_number(row, "squeeze_middle_force_delta_max")
            row["ab_friction_finger"] = args.ab_friction_finger
            row["ab_friction_feature"] = args.ab_friction_feature
            row["ab_friction_value"] = ""
            row["ab_friction_threshold"] = float(args.ab_friction_threshold)
            row["ab_friction_a_direction"] = args.ab_friction_a_direction
            init_ab_proximity_fields(row, args)
            if (stage1_label in {"A", "B"} and args.ab_squeeze_test) or low_confidence_c:
                middle_force = quality_number(row, "squeeze_middle_force_delta_max")
                middle_ready = quality_number(row, "squeeze_middle_ready") >= 0.5
                row["ab_squeeze_curve_distance_a"] = ""
                row["ab_squeeze_curve_distance_b"] = ""
                row["ab_shape_late_slope"] = ""
                row["ab_shape_rebound"] = ""
                row["ab_shape_peak_pos"] = ""
                row["ab_shape_score_a"] = ""
                if not middle_ready:
                    final_label = stage1_label
                    ab_confidence = float(stage1["confidence"])
                    row["prediction_status"] = (
                        "low_confidence_c_squeeze_no_middle_contact"
                        if low_confidence_c
                        else "ab_squeeze_no_middle_contact"
                    )
                    print("ab_squeeze skipped: middle finger did not confirm contact.")
                elif args.ab_squeeze_mode == "friction":
                    final_label, ab_confidence, friction_value, friction_series = friction_ab_label(row, args)
                    row["ab_friction_value"] = friction_value if friction_series else ""
                    if not final_label:
                        final_label = stage1_label
                        ab_confidence = float(stage1["confidence"])
                        row["prediction_status"] = "ab_friction_no_touchdata"
                        print("ab_friction skipped: no touchData friction series.")
                elif args.ab_squeeze_mode == "shape":
                    series_name, series = squeeze_series_from_row(row)
                    if series_name == "middle" and series:
                        final_label, ab_confidence, shape_metrics = ab_shape_decision(
                            series,
                            late_slope_threshold=args.ab_shape_late_slope_threshold,
                            rebound_threshold=args.ab_shape_rebound_threshold,
                            peak_pos_threshold=args.ab_shape_peak_pos_threshold,
                            min_a_score=args.ab_shape_min_a_score,
                        )
                        row["ab_shape_late_slope"] = shape_metrics["curve_late_slope"]
                        row["ab_shape_rebound"] = shape_metrics["curve_rebound"]
                        row["ab_shape_peak_pos"] = shape_metrics["curve_peak_pos"]
                        row["ab_shape_score_a"] = shape_metrics["ab_shape_score_a"]
                    else:
                        final_label, ab_confidence = squeeze_ab_label(middle_force, args)
                        row["prediction_status"] = "ab_squeeze_no_middle_series"
                        print("ab_squeeze shape fallback: no middle squeeze curve, using threshold.")
                elif args.ab_squeeze_mode == "curve":
                    final_label, ab_confidence, ab_distances = squeeze_ab_curve_label(
                        row,
                        self.ab_squeeze_references,
                        args,
                    )
                    row["ab_squeeze_curve_distance_a"] = ab_distances["A"]
                    row["ab_squeeze_curve_distance_b"] = ab_distances["B"]
                    if not final_label:
                        final_label = stage1_label
                        ab_confidence = float(stage1["confidence"])
                        row["prediction_status"] = "ab_squeeze_no_reference"
                        print("ab_squeeze skipped: no labelled squeeze reference curves yet.")
                else:
                    final_label, ab_confidence = squeeze_ab_label(middle_force, args)
                if row.get("prediction_status") not in {"ab_squeeze_no_reference", "ab_squeeze_no_middle_series", "ab_friction_no_touchdata"}:
                    if row.get("prediction_status") not in {
                        "ab_squeeze_no_middle_contact",
                        "low_confidence_c_squeeze_no_middle_contact",
                    }:
                        if low_confidence_c:
                            row["prediction_status"] = (
                                "low_confidence_c_friction"
                                if args.ab_squeeze_mode == "friction"
                                else "low_confidence_c_squeeze"
                            )
                        else:
                            row["prediction_status"] = "ab_friction" if args.ab_squeeze_mode == "friction" else "ab_squeeze"
                row["predicted_label"] = final_label
                row["prediction_confidence"] = ab_confidence
                row["prediction_distance"] = ""
                print(
                    f"ab_squeeze middle_force={middle_force:.1f} "
                    f"mode={args.ab_squeeze_mode} "
                    f"final_prediction={final_label} "
                    f"confidence={ab_confidence:.2f}"
                )
                if args.ab_squeeze_mode == "friction":
                    print(
                        "ab_friction "
                        f"{row.get('ab_friction_finger')}.{row.get('ab_friction_feature')}="
                        f"{row.get('ab_friction_value') or '-'} "
                        f"rule: A if value {row.get('ab_friction_a_direction')} "
                        f"{row.get('ab_friction_threshold')} else B"
                    )
                if args.ab_squeeze_mode == "shape":
                    print(
                        "ab_shape "
                        f"late_slope={row.get('ab_shape_late_slope') or '-'} "
                        f"rebound={row.get('ab_shape_rebound') or '-'} "
                        f"peak_pos={row.get('ab_shape_peak_pos') or '-'} "
                        f"score_A={row.get('ab_shape_score_a') or '-'}"
                    )
            else:
                apply_proximity_assist(
                    row,
                    stage1_label,
                    float(stage1["confidence"]),
                    float(stage1["distance"]),
                    args,
                )
        else:
            apply_proximity_assist(
                row,
                stage1_label,
                float(stage1["confidence"]),
                float(stage1["distance"]),
                args,
            )

        self.refresh_decision_dashboard(row)
        append_feature_row(self.config.output, row)

        friction_text = ""
        if row.get("ab_squeeze_triggered") == 1.0:
            value = row.get("ab_friction_value")
            friction_text = (
                f" friction_{self.config.ab_friction_finger}_{self.config.ab_friction_feature}="
                f"{value if value != '' else '-'} threshold={self.config.ab_friction_threshold:g}"
            )
        print(
            f"ball_type={row.get('predicted_label')} "
            f"stage1={stage1_label} "
            f"confidence={float(row.get('prediction_confidence') or 0.0):.2f} "
            f"status={row.get('prediction_status')} "
            f"distance={row.get('prediction_distance') or '-'}"
            f"{friction_text}"
        )
        return row
