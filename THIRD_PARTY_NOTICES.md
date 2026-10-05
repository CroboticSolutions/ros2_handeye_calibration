# Intrinsic calibration adaptation

Upstream: https://github.com/intrinsic-ai/intrinsic-core
Pinned revision: c61bf075f2335371c6367b61117e8a62bb960c3b
Copyright 2026 Intrinsic Innovation LLC. Apache License 2.0.
Original source checkout: /root/intrinsic_calibration_upstream (reference only).

Adapted files: hand_eye_calibration/intrinsic_solver.py and intrinsic_acquisition.py.
Sources: intrinsic_perception/intrinsic/perception/calibration/camera_to_robot_calibration.cc,
pose_sampling.h; skills/calibration/sample_calibration_poses.cc,
collect_calibration_data.cc; intrinsic_sdk/intrinsic/eigenmath/pose3_utils.h.

Modifications: Python/SciPy implementation of the Shah + joint pose objective;
ROS2/MoveIt integration; preserved physical feedback, collision, timing and stop
checks; always planned initial moves; finite/degeneracy guards; additional pose
bootstrap and independent validation. No Intrinsic gRPC/ICON runtime is installed
or claimed. No upstream algorithm selection or Ceres binary is being executed.

The original Apache-2.0 license is included as LICENSE.intrinsic.
