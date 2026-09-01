#!/usr/bin/env python3
"""
ChArUco board detector + pose publisher for hand-eye calibration.

Replaces the single-ArUco-marker detector. Subscribes to the camera image +
camera_info, detects a ChArUco board, estimates its pose with solvePnP, and:

  * broadcasts TF  camera_optical_frame -> board_frame  (the transform the
    hand_eye_calibration collector looks up as tracking_base -> tracking_marker),
  * publishes std_msgs/Bool on chessboard_visible_topic so the GUI can show a
    live "board detected" indicator,
  * (optional) publishes an annotated image for debugging,
  * accepts a live board spec on board_spec_topic (std_msgs/String JSON) so the
    GUI can change dictionary / square count / sizes without relaunching.

Board defaults match the printed Calib.io-style board:
  13 x 9 squares (landscape), 15 mm square, 11 mm marker, 4x4 ArUco dictionary.

NOTE on orientation: OpenCV ChArUco layouts are NOT symmetric — (13, 9) and
(9, 13) place markers differently, so a swapped X/Y never detects. The printed
board is 13 wide x 9 high (58 markers, ids 0-57, verified from live captures).

NOTE on dictionary size: a 13x9 ChArUco board needs 58 markers, so DICT_4X4_50
(only 50 ids) is too small. Default is DICT_4X4_100. If detection never locks on,
the printed board was generated with a different dictionary -- change it from the
GUI (Board spec -> Dictionary) or via the `aruco_dictionary` parameter.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from scipy.spatial.transform import Rotation as Rot
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, String
from tf2_ros import TransformBroadcaster


def _lookup_aruco_dictionary(name: str):
    """Resolve a dictionary by DICT_* name or numeric id."""
    normalized = str(name or "").strip()
    if normalized.isdigit():
        return cv2.aruco.getPredefinedDictionary(int(normalized))
    if not normalized.startswith("DICT_"):
        normalized = f"DICT_{normalized}"
    if not hasattr(cv2.aruco, normalized):
        raise ValueError(f"Unknown ArUco dictionary: {name!r}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, normalized))


class CharucoBoardDetector(Node):
    def __init__(self):
        super().__init__("charuco_detector")

        # Topics / frames
        self.declare_parameter("image_topic", "/oak/rgb/image_raw")
        self.declare_parameter("camera_info_topic", "/oak/rgb/camera_info")
        self.declare_parameter("board_frame", "charuco_board")
        # Empty -> use the optical frame from the image/camera_info header.
        self.declare_parameter("camera_optical_frame", "")
        self.declare_parameter("annotated_topic", "/charuco_detector/image_annotated")
        self.declare_parameter("chessboard_visible_topic", "/hand_eye_calibration/chessboard_visible")
        self.declare_parameter("board_spec_topic", "/hand_eye_calibration/board_spec")
        self.declare_parameter("publish_annotated", True)

        # Board geometry (the printed board)
        self.declare_parameter("squares_x", 13)
        self.declare_parameter("squares_y", 9)
        self.declare_parameter("square_length_m", 0.015)
        self.declare_parameter("marker_length_m", 0.011)
        self.declare_parameter("aruco_dictionary", "DICT_4X4_100")
        # Quality gate: minimum interior ChArUco corners required for a pose.
        # A 13x9 board exposes up to 96 interior corners. Measured on this
        # repo's fixtures: a good live capture yields ~79, a half-visible board
        # ~30, so 24 rejects genuinely weak detections while staying permissive.
        # Raising this trades capture rate for pose conditioning.
        self.declare_parameter("min_charuco_corners", 24)
        # Corners clustered in one small image region give a poorly conditioned
        # PnP even when there are many of them. This is the RMS radius of the
        # corners about their centroid, normalized by the image diagonal.
        # Measured: full-frame synthetic 0.26, good live capture 0.09,
        # half-visible board 0.06. The default only catches pathological
        # clustering; raise it if you want to insist on fuller board coverage.
        self.declare_parameter("min_charuco_corner_spread", 0.04)
        # Reject (do not publish) poses whose reprojection error exceeds this.
        self.declare_parameter("max_reproj_error_px", 2.0)
        # Set false to publish high-error poses anyway (diagnostics only —
        # feeding them into hand-eye calibration degrades its precision).
        self.declare_parameter("reject_on_reproj_error", True)
        # Re-detect markers from the interpolated ChArUco corners. Measured on
        # this repo's fixtures: recovers 79 corners vs 39 on a real capture, and
        # is slightly more accurate against a known synthetic ground-truth pose
        # (0.777 mm vs 0.833 mm median position error over 40 views).
        self.declare_parameter("try_refine_markers", True)

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        self.board_frame = str(self.get_parameter("board_frame").value)
        self.camera_optical_frame = str(self.get_parameter("camera_optical_frame").value)
        self.annotated_topic = str(self.get_parameter("annotated_topic").value)
        self.publish_annotated = bool(self.get_parameter("publish_annotated").value)
        self.min_charuco_corners = int(self.get_parameter("min_charuco_corners").value)
        self.min_charuco_corner_spread = float(self.get_parameter("min_charuco_corner_spread").value)
        self.max_reproj_error_px = float(self.get_parameter("max_reproj_error_px").value)
        self.reject_on_reproj_error = bool(self.get_parameter("reject_on_reproj_error").value)
        self.try_refine_markers = bool(self.get_parameter("try_refine_markers").value)

        self.spec = {
            "squares_x": int(self.get_parameter("squares_x").value),
            "squares_y": int(self.get_parameter("squares_y").value),
            "square_length_m": float(self.get_parameter("square_length_m").value),
            "marker_length_m": float(self.get_parameter("marker_length_m").value),
            "aruco_dictionary": str(self.get_parameter("aruco_dictionary").value),
        }

        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None
        self._info_frame = None
        self._warned_no_info = False
        # Rejection bookkeeping so a persistently-rejecting gate explains itself
        # instead of silently publishing nothing (see _note_rejection).
        self._reject_streak = 0
        self._reject_reason = None
        self._last_reproj_px = None
        self._build_board()

        self.tf_broadcaster = TransformBroadcaster(self)
        self.visible_pub = self.create_publisher(
            Bool, str(self.get_parameter("chessboard_visible_topic").value), 10
        )
        # WebRTC (server_ros.py IMAGE_QOS) is RELIABLE. qos_profile_sensor_data
        # is BEST_EFFORT — DDS drops every annotated frame while
        # chessboard_visible (RELIABLE Bool) still lights the GUI.
        annotated_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.annotated_pub = (
            self.create_publisher(Image, self.annotated_topic, annotated_qos)
            if self.publish_annotated
            else None
        )

        self.create_subscription(
            CameraInfo, self.camera_info_topic, self._camera_info_cb, 10
        )
        self.create_subscription(
            Image, self.image_topic, self._image_cb, qos_profile_sensor_data
        )
        # GUI -> node live board reconfiguration.
        self.create_subscription(
            String, str(self.get_parameter("board_spec_topic").value), self._board_spec_cb, 1
        )

        self.get_logger().info(
            f"ChArUco detector up: image={self.image_topic}, info={self.camera_info_topic}, "
            f"board={self.spec['squares_x']}x{self.spec['squares_y']} "
            f"square={self.spec['square_length_m']*1000:.0f}mm marker={self.spec['marker_length_m']*1000:.0f}mm "
            f"dict={self.spec['aruco_dictionary']}, board_frame={self.board_frame}"
        )
        self.get_logger().info(
            f"Quality gates: min_charuco_corners={self.min_charuco_corners}, "
            f"min_charuco_corner_spread={self.min_charuco_corner_spread:.3f}, "
            f"max_reproj_error_px={self.max_reproj_error_px:.2f} "
            f"(reject={self.reject_on_reproj_error}), try_refine_markers={self.try_refine_markers}"
        )

    # -- board construction ------------------------------------------------
    def _build_board(self):
        dictionary = _lookup_aruco_dictionary(self.spec["aruco_dictionary"])
        size = (self.spec["squares_x"], self.spec["squares_y"])
        self.board = cv2.aruco.CharucoBoard(
            size,
            self.spec["square_length_m"],
            self.spec["marker_length_m"],
            dictionary,
        )

        detector_params = cv2.aruco.DetectorParameters()
        # Explicitly OFF (this also happens to be the OpenCV default): refining
        # ArUco marker corners near the chessboard squares biases the corners
        # that ChArUco's interior-corner interpolation depends on, unless the
        # marker/square margin is large. See the OpenCV ChArUco tutorial.
        detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE

        charuco_params = cv2.aruco.CharucoParameters()
        # Interior ChArUco corners are subpixel-refined by the detector itself,
        # independently of the marker setting above. Re-detecting markers from
        # the board layout (tryRefineMarkers) recovers markers that the first
        # pass missed, which mostly helps exactly when detection is degraded.
        charuco_params.tryRefineMarkers = self.try_refine_markers

        # Modern (OpenCV >= 4.7) detector. Falls back handled by exception in callback.
        self.charuco_detector = cv2.aruco.CharucoDetector(self.board, charuco_params, detector_params)
        self.dictionary = dictionary

    def _board_spec_cb(self, msg: String):
        try:
            data = json.loads(msg.data)
        except (ValueError, TypeError) as exc:
            self.get_logger().warning(f"Ignoring invalid board_spec JSON: {exc}")
            return
        new_spec = dict(self.spec)
        for key in ("squares_x", "squares_y"):
            if key in data:
                new_spec[key] = int(data[key])
        for key in ("square_length_m", "marker_length_m"):
            if key in data:
                new_spec[key] = float(data[key])
        if "aruco_dictionary" in data:
            new_spec["aruco_dictionary"] = str(data["aruco_dictionary"])
        if new_spec == self.spec:
            return
        prev = self.spec
        self.spec = new_spec
        try:
            self._build_board()
            self.get_logger().info(f"Board spec updated from GUI: {self.spec}")
        except Exception as exc:  # noqa: BLE001 - revert on bad spec
            self.spec = prev
            self._build_board()
            self.get_logger().error(f"Bad board spec {new_spec}: {exc}; kept {prev}")

    # -- camera info -------------------------------------------------------
    def _camera_info_cb(self, msg: CameraInfo):
        self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d, dtype=np.float64).reshape(-1, 1)
        self._info_frame = msg.header.frame_id

    # -- detection ---------------------------------------------------------
    def _optical_frame(self, header_frame: str) -> str:
        if self.camera_optical_frame:
            return self.camera_optical_frame
        return self._info_frame or header_frame or "camera_optical_frame"

    def _publish_visible(self, visible: bool):
        self.visible_pub.publish(Bool(data=bool(visible)))

    @staticmethod
    def corner_spread(charuco_corners, image_shape) -> float:
        """RMS radius of the corners about their centroid, over the image diagonal."""
        pts = np.asarray(charuco_corners, dtype=np.float64).reshape(-1, 2)
        if len(pts) < 2:
            return 0.0
        diag = float(np.hypot(image_shape[0], image_shape[1]))
        if diag <= 0.0:
            return 0.0
        centred = pts - pts.mean(axis=0)
        return float(np.sqrt((centred ** 2).sum(axis=1).mean()) / diag)

    def _note_rejection(self, reason: str):
        """Log why a pose was rejected, escalating if it keeps happening.

        A hard quality gate that silently publishes nothing looks identical to
        'the board is not visible', so say which gate is firing and what to do.
        """
        self._reject_streak += 1
        self._reject_reason = reason
        if self._reject_streak in (1, 10) or self._reject_streak % 100 == 0:
            self.get_logger().warning(
                f"ChArUco pose rejected ({self._reject_streak} in a row): {reason}"
            )
            if self._reject_streak >= 10:
                self.get_logger().warning(
                    "Nothing is being published to TF, so calibration capture will fail. "
                    "Check camera focus/lighting and board visibility; verify camera_info "
                    "intrinsics are for this resolution; then relax the gates via "
                    "min_charuco_corners / min_charuco_corner_spread / max_reproj_error_px "
                    "(or set reject_on_reproj_error:=false to publish anyway)."
                )

    def _note_accepted(self):
        if self._reject_streak:
            self.get_logger().info(
                f"ChArUco pose accepted again after {self._reject_streak} rejected frame(s)."
            )
        self._reject_streak = 0
        self._reject_reason = None

    def _image_cb(self, msg: Image):
        if self.camera_matrix is None:
            if not self._warned_no_info:
                self.get_logger().warning(
                    f"No CameraInfo on {self.camera_info_topic} yet; cannot estimate board pose."
                )
                self._warned_no_info = True
            self._publish_visible(False)
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warning(f"cv_bridge failed: {exc}")
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        charuco_corners, charuco_ids = self._detect(gray)
        n_corners = 0 if charuco_ids is None else len(charuco_ids)

        annotated = frame if self.annotated_pub is not None else None
        pose_ok = False
        spread = self.corner_spread(charuco_corners, gray.shape) if n_corners else 0.0
        if n_corners < self.min_charuco_corners:
            if n_corners:
                self._note_rejection(
                    f"only {n_corners} ChArUco corners (need min_charuco_corners="
                    f"{self.min_charuco_corners}) — move closer or expose more of the board"
                )
        elif spread < self.min_charuco_corner_spread:
            self._note_rejection(
                f"corners clustered in one region (spread={spread:.3f} < "
                f"min_charuco_corner_spread={self.min_charuco_corner_spread:.3f}) — "
                "a clustered set gives a poorly conditioned pose"
            )
        else:
            pose_ok = self._estimate_and_publish(
                charuco_corners, charuco_ids, msg.header.frame_id, annotated, msg.header.stamp
            )

        self._publish_visible(pose_ok)

        if self.annotated_pub is not None and annotated is not None:
            if charuco_ids is not None and n_corners > 0:
                cv2.aruco.drawDetectedCornersCharuco(
                    annotated, charuco_corners, charuco_ids, (0, 255, 0)
                )
            label = (
                f"corners {n_corners}/{self.min_charuco_corners}  "
                f"spread {spread:.3f}  "
                f"{'POSE OK' if pose_ok else 'no pose'}"
            )
            if pose_ok and self._last_reproj_px is not None:
                label += f"  reproj {self._last_reproj_px:.2f}px"
            cv2.putText(
                annotated, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 0) if pose_ok else (0, 0, 255), 2, cv2.LINE_AA,
            )
            if not pose_ok and self._reject_reason:
                cv2.putText(
                    annotated, self._reject_reason[:70], (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA,
                )
            out = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            out.header = msg.header
            self.annotated_pub.publish(out)

    def _detect(self, gray):
        """Return (charuco_corners, charuco_ids) using whichever OpenCV API exists."""
        try:
            charuco_corners, charuco_ids, _, _ = self.charuco_detector.detectBoard(gray)
            return charuco_corners, charuco_ids
        except AttributeError:
            # Legacy OpenCV (< 4.7) fallback.
            marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary)
            if marker_ids is None or len(marker_ids) == 0:
                return None, None
            _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                marker_corners, marker_ids, gray, self.board
            )
            return charuco_corners, charuco_ids

    def _estimate_and_publish(self, charuco_corners, charuco_ids, header_frame, annotated, stamp):
        try:
            obj_points, img_points = self.board.matchImagePoints(charuco_corners, charuco_ids)
        except AttributeError:
            obj_points, img_points = None, None
        if obj_points is None or len(obj_points) < 4:
            self._note_rejection("could not match enough board points to image points")
            return False

        ok, rvec, tvec = cv2.solvePnP(
            obj_points, img_points, self.camera_matrix, self.dist_coeffs
        )
        if not ok:
            self._note_rejection(f"solvePnP failed on {len(obj_points)} corners")
            return False

        # Reprojection error gates whether this pose is trustworthy enough to
        # publish at all — a high-error pose fed into hand-eye calibration
        # silently degrades its precision, so reject rather than just warn.
        proj, _ = cv2.projectPoints(
            obj_points, rvec, tvec, self.camera_matrix, self.dist_coeffs
        )
        reproj = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - img_points.reshape(-1, 2), axis=1)))
        self._last_reproj_px = reproj
        if reproj > self.max_reproj_error_px:
            reason = (
                f"reprojection error {reproj:.2f}px > max_reproj_error_px="
                f"{self.max_reproj_error_px:.2f} over {len(obj_points)} corners — "
                "check square/marker sizes, dictionary, focus, and that camera_info "
                "intrinsics match this image resolution"
            )
            if self.reject_on_reproj_error:
                self._note_rejection(reason)
                return False
            self.get_logger().warning(f"Publishing low-quality ChArUco pose anyway: {reason}")

        self._note_accepted()
        rot = Rot.from_rotvec(rvec.reshape(3))
        qx, qy, qz, qw = (float(v) for v in rot.as_quat())

        if annotated is not None:
            cv2.drawFrameAxes(
                annotated, self.camera_matrix, self.dist_coeffs, rvec, tvec,
                self.spec["square_length_m"] * 2.0,
            )

        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = self._optical_frame(header_frame)
        t.child_frame_id = self.board_frame
        t.transform.translation.x = float(tvec[0])
        t.transform.translation.y = float(tvec[1])
        t.transform.translation.z = float(tvec[2])
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(t)
        return True


def main():
    rclpy.init()
    node = CharucoBoardDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
