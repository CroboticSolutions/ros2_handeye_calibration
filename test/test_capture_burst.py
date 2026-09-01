"""
Integration test for the ROS layer of hand-eye capture.

This covers the part the pure-math backend tests cannot: that
``capture_point`` actually runs to completion inside a live executor and
collects several time-synchronized frames.

It is a regression test for a real bug — the burst loop originally pumped the
executor itself with ``rclpy.spin_once`` from inside the service callback,
which raises ``RuntimeError: Executor is already spinning`` and aborted every
capture. Nothing in the backend tests could see that.

Run offline (needs a ROS 2 environment sourced):
  cd arms_ws/src/ros2_handeye_calibration
  python3 -m pytest test/test_capture_burst.py -v
"""

from __future__ import annotations

import os
import threading
import time
import unittest

os.environ.setdefault("ROS_DOMAIN_ID", "94")
os.environ.setdefault("ROS_LOCALHOST_ONLY", "1")

try:
    import rclpy
    import rclpy.time
    from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
    from rclpy.node import Node
    from geometry_msgs.msg import TransformStamped
    from std_srvs.srv import Trigger
    from tf2_ros import TransformBroadcaster

    from hand_eye_calibration.node import DataCollector

    ROS_AVAILABLE = True
except ImportError:  # pragma: no cover - environment without ROS
    ROS_AVAILABLE = False
    Node = object  # so the helper class below can still be defined


CAM_FRAME = "test_cam_optical"
BOARD_FRAME = "test_board"
BASE_FRAME = "test_base"
EFFECTOR_FRAME = "test_ee"


class _FakeStack(Node):
    """Stands in for the robot + charuco_detector: publishes both TFs at 30 Hz."""

    def __init__(self) -> None:
        super().__init__("fake_stack")
        self.br = TransformBroadcaster(self)
        self.ticks = 0
        self.create_timer(1.0 / 30.0, self.tick)

    def tick(self) -> None:
        self.ticks += 1
        now = self.get_clock().now().to_msg()
        # Robot pose: static, so burst averaging should see no motion.
        robot = TransformStamped()
        robot.header.stamp = now
        robot.header.frame_id = BASE_FRAME
        robot.child_frame_id = EFFECTOR_FRAME
        robot.transform.translation.x = 0.4
        robot.transform.translation.z = 0.3
        robot.transform.rotation.w = 1.0
        # Board pose: tiny per-frame jitter, like real ChArUco pose noise.
        board = TransformStamped()
        board.header.stamp = now
        board.header.frame_id = CAM_FRAME
        board.child_frame_id = BOARD_FRAME
        board.transform.translation.x = 0.001 * ((self.ticks % 3) - 1)
        board.transform.translation.z = 0.5
        board.transform.rotation.w = 1.0
        self.br.sendTransform([robot, board])


@unittest.skipUnless(ROS_AVAILABLE, "ROS 2 (rclpy) not available")
class CaptureBurstTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        rclpy.init()

        cls.stack = _FakeStack()
        cls.stack_exec = SingleThreadedExecutor()
        cls.stack_exec.add_node(cls.stack)
        cls.stack_thread = threading.Thread(target=cls.stack_exec.spin, daemon=True)
        cls.stack_thread.start()

        cls.node = DataCollector()
        # The node reads its parameters into attributes in __init__, so drive
        # the test by setting those directly — simpler than ROS CLI overrides,
        # which cannot express the empty topic names we want here.
        cls.node.tracking_base_frame = CAM_FRAME
        cls.node.tracking_marker_frame = BOARD_FRAME
        cls.node.robot_base_frame = BASE_FRAME
        cls.node.robot_effector_frame = EFFECTOR_FRAME
        cls.node.calibration_type = "eye-in-hand"
        # Empty topics keep preflight from blocking on wait_for_message.
        cls.node.pointcloud_topic = ""
        cls.node.camera_info_topic = ""
        cls.node.capture_burst_duration_s = 1.5
        cls.node.capture_burst_samples = 5

        # Must match main(): a single-threaded executor cannot service TF while
        # the capture callback is blocked waiting for it.
        cls.executor = MultiThreadedExecutor()
        cls.executor.add_node(cls.node)
        cls.exec_thread = threading.Thread(target=cls.executor.spin, daemon=True)
        cls.exec_thread.start()

        cls.client = cls.node.create_client(Trigger, "hand_eye_calibration/capture_point")
        assert cls.client.wait_for_service(timeout_sec=10.0), "capture_point service never came up"
        time.sleep(1.5)  # let the TF buffer fill

    @classmethod
    def tearDownClass(cls) -> None:
        cls.executor.shutdown()
        cls.stack_exec.shutdown()
        cls.node.destroy_node()
        cls.stack.destroy_node()
        rclpy.shutdown()

    def _capture(self, timeout=25.0):
        future = self.client.call_async(Trigger.Request())
        deadline = time.time() + timeout
        while not future.done() and time.time() < deadline:
            time.sleep(0.05)
        self.assertTrue(future.done(), "capture_point never returned (deadlock or crash)")
        return future.result()

    def test_capture_point_succeeds_and_collects_a_full_burst(self) -> None:
        before = len(self.node.robot_samples)
        result = self._capture()

        self.assertTrue(result.success, f"capture_point failed: {result.message}")
        self.assertEqual(len(self.node.robot_samples), before + 1)

        metrics = self.node.sample_metrics[-1]
        self.assertGreaterEqual(
            metrics["burst_frame_count"], 2,
            "burst collected almost nothing — the executor is not servicing TF "
            "while the capture callback runs",
        )

    def test_static_robot_reports_no_motion_during_burst(self) -> None:
        self._capture()
        metrics = self.node.sample_metrics[-1]
        self.assertLess(metrics["burst_robot_translation_dev_m"], 1e-6)
        self.assertLess(metrics["burst_robot_rotation_dev_deg"], 1e-3)

    def test_estimate_uncertainty_service_responds(self) -> None:
        """Wiring check for the on-demand uncertainty service.

        The fake stack holds the robot still, so every sample is the same pose
        and the calibration is degenerate by construction — the point here is
        that the service answers instead of hanging or letting an exception
        escape into the executor. The numerical behaviour is covered by the
        backend's bootstrap tests against a known ground truth.
        """
        client = self.node.create_client(Trigger, "hand_eye_calibration/estimate_uncertainty")
        self.assertTrue(
            client.wait_for_service(timeout_sec=10.0),
            "estimate_uncertainty service was never advertised",
        )
        future = client.call_async(Trigger.Request())
        deadline = time.time() + 60.0
        while not future.done() and time.time() < deadline:
            time.sleep(0.05)
        self.assertTrue(future.done(), "estimate_uncertainty never returned")
        self.assertTrue(future.result().message, "service gave no explanation")

    def test_repeated_captures_keep_working(self) -> None:
        """The executor must stay healthy across captures, not die on the first."""
        before = len(self.node.robot_samples)
        for _ in range(3):
            result = self._capture()
            self.assertTrue(result.success, f"capture_point failed: {result.message}")
        self.assertEqual(len(self.node.robot_samples), before + 3)


if __name__ == "__main__":
    unittest.main()
