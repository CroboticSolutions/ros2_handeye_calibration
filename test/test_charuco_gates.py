"""
Tests for the ChArUco detector's pose-quality gates.

These gates decide whether a board pose is published to TF at all, so they
directly control which samples reach hand-eye calibration. The thresholds are
pinned against this repo's real fixture so a future tuning change that would
silently stop accepting good captures fails here instead of in the lab.

Run offline (needs a ROS 2 environment sourced):
  cd arms_ws/src/ros2_handeye_calibration
  python3 -m pytest test/test_charuco_gates.py -v
"""

from __future__ import annotations

import unittest
from pathlib import Path

import cv2
import numpy as np

try:
    from hand_eye_calibration.charuco_detector import CharucoBoardDetector
    ROS_AVAILABLE = True
except ImportError:  # pragma: no cover - environment without ROS
    ROS_AVAILABLE = False

from charuco_detection_helpers import (
    DEFAULT_DETECTOR_SPEC,
    BoardSpec,
    build_charuco_board,
    load_gray_image,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
LIVE_SNAPSHOT = FIXTURES / "live_board_snapshot.jpg"

# Defaults declared by charuco_detector.py; kept in sync deliberately.
MIN_CORNERS_DEFAULT = 24
MIN_SPREAD_DEFAULT = 0.04


def _detect(gray, try_refine: bool):
    spec = BoardSpec.from_dict(DEFAULT_DETECTOR_SPEC)
    board = build_charuco_board(spec)
    detector_params = cv2.aruco.DetectorParameters()
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
    charuco_params = cv2.aruco.CharucoParameters()
    charuco_params.tryRefineMarkers = try_refine
    detector = cv2.aruco.CharucoDetector(board, charuco_params, detector_params)
    corners, ids, _, _ = detector.detectBoard(gray)
    return corners, ids


@unittest.skipUnless(ROS_AVAILABLE, "ROS 2 (rclpy) not available")
class CornerSpreadTest(unittest.TestCase):
    def test_live_capture_passes_both_default_gates(self) -> None:
        gray = load_gray_image(str(LIVE_SNAPSHOT))
        corners, ids = _detect(gray, try_refine=True)
        self.assertIsNotNone(ids)
        spread = CharucoBoardDetector.corner_spread(corners, gray.shape)

        self.assertGreaterEqual(
            len(ids), MIN_CORNERS_DEFAULT,
            f"a good live capture must clear min_charuco_corners={MIN_CORNERS_DEFAULT}")
        self.assertGreaterEqual(
            spread, MIN_SPREAD_DEFAULT,
            f"a good live capture must clear min_charuco_corner_spread={MIN_SPREAD_DEFAULT}")

    def test_try_refine_markers_recovers_more_corners(self) -> None:
        """Why try_refine_markers defaults to True: it roughly doubles usable corners."""
        gray = load_gray_image(str(LIVE_SNAPSHOT))
        _, ids_off = _detect(gray, try_refine=False)
        _, ids_on = _detect(gray, try_refine=True)
        n_off = 0 if ids_off is None else len(ids_off)
        n_on = 0 if ids_on is None else len(ids_on)
        self.assertGreater(n_on, n_off)

    def test_clustered_corners_score_lower_than_spread_out_corners(self) -> None:
        shape = (800, 1280)
        clustered = np.array([[[100.0 + i % 5, 100.0 + i // 5]] for i in range(25)])
        spread_out = np.array([
            [[float(x), float(y)]]
            for x in (100, 400, 700, 1000, 1150)
            for y in (80, 250, 400, 600, 750)
        ])
        s_clustered = CharucoBoardDetector.corner_spread(clustered, shape)
        s_spread = CharucoBoardDetector.corner_spread(spread_out, shape)

        self.assertLess(s_clustered, MIN_SPREAD_DEFAULT)
        self.assertGreater(s_spread, MIN_SPREAD_DEFAULT)
        self.assertLess(s_clustered, s_spread)

    def test_degenerate_inputs_do_not_raise(self) -> None:
        self.assertEqual(CharucoBoardDetector.corner_spread(np.empty((0, 1, 2)), (800, 1280)), 0.0)
        self.assertEqual(CharucoBoardDetector.corner_spread(np.array([[[5.0, 5.0]]]), (800, 1280)), 0.0)
        self.assertEqual(CharucoBoardDetector.corner_spread(np.array([[[5.0, 5.0]]]), (0, 0)), 0.0)


if __name__ == "__main__":
    unittest.main()
