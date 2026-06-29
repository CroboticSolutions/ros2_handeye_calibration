"""
ChArUco board detection tests.

Run offline (no ROS):
  cd arms_ws/src/ros2_handeye_calibration
  python3 -m pytest test/test_charuco_detection.py -v

Or:
  python3 test/test_charuco_detection.py
"""

from __future__ import annotations

import unittest
from pathlib import Path

import cv2
import numpy as np

from charuco_detection_helpers import (
    DEFAULT_DETECTOR_SPEC,
    LIVE_BOARD_SPEC,
    BoardSpec,
    detect_charuco,
    load_gray_image,
    render_synthetic_board,
    count_raw_markers,
    best_matching_spec,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
LIVE_SNAPSHOT = FIXTURES / "live_board_snapshot.jpg"


class CharucoDetectionTest(unittest.TestCase):
    def test_synthetic_default_spec_detects_board(self) -> None:
        spec = BoardSpec.from_dict(DEFAULT_DETECTOR_SPEC)
        gray = render_synthetic_board(spec)
        result = detect_charuco(gray, spec)
        self.assertGreaterEqual(result.n_markers, 20, "synthetic board should expose many markers")
        self.assertGreaterEqual(result.n_corners, 40, "synthetic board should expose many charuco corners")

    def test_live_snapshot_has_visible_aruco_markers(self) -> None:
        self.assertTrue(LIVE_SNAPSHOT.is_file(), f"missing fixture {LIVE_SNAPSHOT}")
        gray = load_gray_image(str(LIVE_SNAPSHOT))
        n = count_raw_markers(gray, "DICT_4X4_100")
        self.assertGreaterEqual(n, 10, "camera sees ArUco markers but raw count is low")

    def test_live_snapshot_fails_with_default_9x13_spec(self) -> None:
        """Documents why GUI/detector defaults miss the printed board."""
        gray = load_gray_image(str(LIVE_SNAPSHOT))
        spec = BoardSpec.from_dict(DEFAULT_DETECTOR_SPEC)
        result = detect_charuco(gray, spec)
        self.assertEqual(
            result.n_corners,
            0,
            "default 9x13 spec must NOT match the physical 13x15 board",
        )

    def test_live_snapshot_detects_with_observed_13x15_spec(self) -> None:
        gray = load_gray_image(str(LIVE_SNAPSHOT))
        spec = BoardSpec.from_dict(LIVE_BOARD_SPEC)
        result = detect_charuco(gray, spec)
        self.assertGreaterEqual(
            result.n_corners,
            8,
            f"expected >=8 ChArUco corners with 13x15 spec, got {result.n_corners}",
        )
        self.assertGreaterEqual(result.n_markers, 20)

    def test_live_snapshot_pose_with_approximate_intrinsics(self) -> None:
        gray = load_gray_image(str(LIVE_SNAPSHOT))
        h, w = gray.shape[:2]
        spec = BoardSpec.from_dict(LIVE_BOARD_SPEC)
        fx = fy = float(w)
        cx, cy = w / 2.0, h / 2.0
        k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        result = detect_charuco(gray, spec, camera_matrix=k, dist_coeffs=np.zeros((5, 1)))
        self.assertTrue(result.pose_ok, "pose should succeed with enough corners + intrinsics")
        self.assertIsNotNone(result.reproj_error_px)

    def test_sweep_finds_13x15_on_live_snapshot(self) -> None:
        gray = load_gray_image(str(LIVE_SNAPSHOT))
        best = best_matching_spec(gray, min_corners=8)
        self.assertIsNotNone(best, "sweep should find at least one matching board spec")
        assert best is not None
        self.assertEqual(best.spec.squares_x, 13)
        self.assertEqual(best.spec.squares_y, 15)
        self.assertIn(best.spec.aruco_dictionary, ("DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250"))


if __name__ == "__main__":
    unittest.main()
