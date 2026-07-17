#!/usr/bin/env python3
"""
Diagnose ChArUco board detection on a live ROS image or saved snapshot.

Examples:
  # One frame from the running camera:
  source /opt/ros/jazzy/setup.bash
  python3 scripts/diagnose_charuco_board.py --ros-topic /oak/rgb/image_raw

  # Saved image (e.g. from test fixture):
  python3 scripts/diagnose_charuco_board.py --image test/fixtures/live_board_snapshot.jpg

  # Sweep grid/dictionary to suggest board spec:
  python3 scripts/diagnose_charuco_board.py --image test/fixtures/live_board_snapshot.jpg --sweep
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

# Allow running from package root without install.
_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT / "test") not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT / "test"))

from charuco_detection_helpers import (  # noqa: E402
    COMMON_DICTIONARIES,
    DEFAULT_DETECTOR_SPEC,
    LIVE_BOARD_SPEC,
    BoardSpec,
    best_matching_spec,
    count_raw_markers,
    detect_charuco,
    load_gray_image,
    sweep_board_specs,
)


def _grab_ros_frame(topic: str, timeout_sec: float = 5.0):
    import rclpy
    from cv_bridge import CvBridge
    from rclpy.node import Node
    from sensor_msgs.msg import Image

    class Grab(Node):
        def __init__(self):
            super().__init__("charuco_diagnose_grab")
            self.bridge = CvBridge()
            self.gray = None

        def cb(self, msg: Image):
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    rclpy.init()
    node = Grab()
    node.create_subscription(Image, topic, node.cb, 10)
    import time

    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline and node.gray is None:
        rclpy.spin_once(node, timeout_sec=0.1)
    gray = node.gray
    node.destroy_node()
    rclpy.shutdown()
    if gray is None:
        raise RuntimeError(f"No frame received on {topic} within {timeout_sec}s")
    return gray


def _print_result(label: str, result) -> None:
    s = result.spec
    print(
        f"  {label}: dict={s.aruco_dictionary} grid={s.squares_x}x{s.squares_y} "
        f"square={s.square_length_m*1000:.1f}mm marker={s.marker_length_m*1000:.1f}mm "
        f"-> markers={result.n_markers} corners={result.n_corners}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose ChArUco board detection")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="Path to BGR/gray snapshot")
    src.add_argument("--ros-topic", help="Grab one frame from this sensor_msgs/Image topic")
    parser.add_argument("--spec-json", help="Board spec JSON (squares_x, squares_y, ...)")
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Grid/dictionary sweep (15 mm / 11 mm default) and print best matches",
    )
    parser.add_argument(
        "--full-sweep",
        action="store_true",
        help="Also try extra square/marker sizes and dictionaries (slower)",
    )
    parser.add_argument(
        "--save-annotated",
        help="Write annotated overlay PNG for the given spec (or best sweep hit)",
    )
    args = parser.parse_args()

    if args.image:
        gray = load_gray_image(args.image)
        print(f"Loaded {args.image} shape={gray.shape[1]}x{gray.shape[0]}")
    else:
        gray = _grab_ros_frame(args.ros_topic)
        print(f"Grabbed ROS frame shape={gray.shape[1]}x{gray.shape[0]} from {args.ros_topic}")

    print("\nRaw ArUco marker counts (no board geometry):")
    for dn in COMMON_DICTIONARIES[:4]:
        print(f"  {dn}: {count_raw_markers(gray, dn)}")

    if args.spec_json:
        spec = BoardSpec.from_dict(json.loads(args.spec_json))
    else:
        spec = BoardSpec.from_dict(DEFAULT_DETECTOR_SPEC)

    print("\nDetection with configured spec:")
    _print_result("configured", detect_charuco(gray, spec))

    live_hint = BoardSpec.from_dict(LIVE_BOARD_SPEC)
    print(
        f"\nDetection with observed live board hint "
        f"({live_hint.squares_x}x{live_hint.squares_y} {live_hint.aruco_dictionary}):"
    )
    hint_result = detect_charuco(gray, live_hint)
    _print_result("live hint", hint_result)

    best = None
    if args.sweep or hint_result.n_corners == 0:
        print("\nSweeping board specs...")
        sweep_kwargs = {"min_corners": 8}
        if args.full_sweep:
            sweep_kwargs.update(
                {
                    "squares_x_range": range(5, 16),
                    "squares_y_range": range(5, 16),
                    "square_lengths_mm": (15.0, 20.0, 25.0, 30.0),
                    "marker_lengths_mm": (11.0, 15.0, 18.0, 22.0),
                    "dictionaries": COMMON_DICTIONARIES,
                }
            )
        hits = sweep_board_specs(gray, **sweep_kwargs)
        if not hits:
            print("  No spec yielded >=8 ChArUco corners.")
        else:
            print(f"  Top {min(5, len(hits))} matches:")
            for hit in hits[:5]:
                _print_result("match", hit)
            best = hits[0]
            print(
                "\nSuggested GUI board spec JSON:\n"
                + json.dumps(best.spec.as_dict(), indent=2)
            )

    out_spec = best.spec if best is not None else (live_hint if hint_result.n_corners >= 8 else spec)
    if args.save_annotated:
        board = __import__("charuco_detection_helpers", fromlist=["build_charuco_board"]).build_charuco_board(out_spec)
        det = cv2.aruco.CharucoDetector(board)
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cc, ci, _, _ = det.detectBoard(gray)
        if ci is not None and len(ci) > 0:
            cv2.aruco.drawDetectedCornersCharuco(bgr, cc, ci, (0, 255, 0))
        cv2.imwrite(args.save_annotated, bgr)
        print(f"\nWrote annotated image to {args.save_annotated}")

    if hint_result.n_corners >= 8:
        if spec.squares_x == live_hint.squares_x and spec.squares_y == live_hint.squares_y:
            return 0
        print(
            f"\nLikely cause: configured spec ({spec.squares_x}x{spec.squares_y}) does not "
            f"match the printed board ({live_hint.squares_x}x{live_hint.squares_y}). "
            "Update the board spec in the calibration GUI (X/Y are orientation-sensitive)."
        )
        return 2

    return 1 if hint_result.n_corners < 8 else 0


if __name__ == "__main__":
    raise SystemExit(main())
