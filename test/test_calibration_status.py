"""Unit tests for calibration readiness status JSON."""

from __future__ import annotations

import json
import unittest

from hand_eye_calibration.calibration_status import (
    build_calibration_status,
    status_to_json,
)


class CalibrationStatusTest(unittest.TestCase):
    def test_not_ready_with_few_samples(self) -> None:
        status = build_calibration_status(
            sample_count=2,
            diversity={
                "sample_count": 2,
                "translation_span_m": [0.01, 0.01, 0.01],
                "max_rotation_from_first_deg": 10.0,
                "guidance": ["Need more samples from different wrist poses."],
            },
            residuals=None,
            last_sample_metrics=None,
        )
        self.assertEqual(status["readiness"], "not_ready")
        self.assertFalse(status["ready_to_save"])

    def test_collecting_when_duplicate_pose(self) -> None:
        status = build_calibration_status(
            sample_count=8,
            diversity={
                "sample_count": 8,
                "translation_span_m": [0.08, 0.06, 0.04],
                "max_rotation_from_first_deg": 30.0,
                "guidance": ["Pose diversity looks reasonable."],
            },
            residuals={
                "mean_translation_m": 0.002,
                "max_translation_m": 0.004,
                "mean_rotation_deg": 0.5,
                "max_rotation_deg": 1.0,
            },
            last_sample_metrics={
                "robot_delta_translation_m": 0.005,
                "robot_delta_rotation_deg": 2.0,
                "marker_distance_m": 0.4,
                "marker_view_angle_deg": 20.0,
            },
        )
        self.assertEqual(status["readiness"], "collecting")
        self.assertIn("close to the previous", status["last_sample_warning"])

    def test_excellent_when_all_metrics_good(self) -> None:
        status = build_calibration_status(
            sample_count=15,
            diversity={
                "sample_count": 15,
                "translation_span_m": [0.12, 0.08, 0.06],
                "max_rotation_from_first_deg": 45.0,
                "guidance": ["Pose diversity looks reasonable."],
            },
            residuals={
                "mean_translation_m": 0.0015,
                "max_translation_m": 0.003,
                "mean_rotation_deg": 0.4,
                "max_rotation_deg": 0.8,
            },
            last_sample_metrics={
                "robot_delta_translation_m": 0.08,
                "robot_delta_rotation_deg": 12.0,
                "marker_distance_m": 0.45,
                "marker_view_angle_deg": 25.0,
            },
        )
        self.assertEqual(status["readiness"], "excellent")
        self.assertTrue(status["ready_to_save"])
        parsed = json.loads(status_to_json(status))
        self.assertEqual(parsed["sample_count"], 15)


if __name__ == "__main__":
    unittest.main()
