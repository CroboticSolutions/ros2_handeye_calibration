# Automatic Piper camera calibration

In the Piper + Gazebo profile, position the stationary arm so the camera sees the
ChArUco board, then press **Calibrate**. Keep the board stationary throughout.
The routine collects distinct views, validates the camera transform against
held-out views, returns to the initial joint position and saves the calibration.
**Stop** cancels the active controller goal; stopping does not return the robot.
The existing **Apply** button applies the saved mount transform separately.

Before any motion, the worker checks candidate views and reduces orbit size or
uses smaller wrist rotations when needed. It rejects duplicate views and requires
rotation diversity. If the initial pose cannot provide enough reachable views,
it asks for a different initial pose without moving.

The worker uses `/compute_ik` (`avoid_collisions=false`) and
`/arm_controller/follow_joint_trajectory`. It does not create a planning scene or
check collisions. It checks URDF joint bounds, IK branch jumps, fresh joint
feedback, controller completion and settling before acquiring fresh synchronized
board/robot frames. Failed poses are skipped. The GUI locks manual capture,
board changes and Apply while the automatic sequence is active.

At least 12 training samples and 2 independent validation views are required.
Validation compares the board pose in the robot base frame and currently allows
at most 10 mm translation and 3 degrees rotation deviation. These are acceptance
thresholds, not a guaranteed accuracy specification. Failed or stopped runs do
not replace the saved calibration. Successful saves preserve the previous file
as `hand_eye_calibration.yaml.previous` and replace the destination atomically.

Services (std_srvs/Trigger):
- `/hand_eye_calibration/auto_start`
- `/hand_eye_calibration/auto_stop`

Progress is the `automatic` object in the existing JSON String status topic
`/hand_eye_calibration/status`. It contains state, active, pose, total, accepted,
message and independent validation metrics. Calibration results continue to use
the existing collector schema and include `automatic_validation` in saved YAML.

The launch profile enables this for the Piper simulated camera; other profiles
retain manual calibration. `auto_enabled` controls availability. Node parameters
`auto_group`, `auto_ik_link`, `auto_controller` and `auto_joint_names` define the
Piper integration. The camera mount in TF is only used for approximate viewpoint
generation; the solver estimates the mount from synchronized observations.

Verification includes exact-transform recovery for the generated sequence,
complete sequence/holdout isolation, failed validation preserving the saved
result, stale detection, duplicate Start, unreachable views, Stop and late goal
acceptance after cancellation. Live checks can verify IK and service wiring
without sending a trajectory to the robot.

## Keeping the board in view

The entire board (including a conservative 10 mm white border) must fit with a
10% image margin before starting. Dimensions come from the same launch values
and live GUI board spec as the detector. Projection uses CameraInfo intrinsics,
distortion and its optical frame. Unsupported camera models fail before motion.
Candidate views retain their requested tilt; if needed, their translation is
adjusted to center the board and increase distance. Smaller orbits are tried
before wrist-only alternatives. URDF forward kinematics checks joint-space
paths at intervals of at most 0.015 rad, including intermediate views.

Each accepted joint path is executed as one smooth controller trajectory, with
zero velocity at its endpoints. The previous 0.06 rad stop-and-observe subdivision
has been removed. Existing duration limits remain (minimum 2.5 simulation seconds,
distance / 0.12); these describe duration, not a peak velocity limit for cubic
interpolation. The robot settles and waits for a post-stop image only at sampling
poses or when a correction is required.

During motion a 20 Hz watchdog checks board framing and fresh joint/camera data.
Detections older than 0.35 simulation seconds, or no new image timestamp for 0.75
wall seconds, abort motion. A margin below 8% triggers cancellation; the worker
waits for the controller's terminal result and for settling, then computes a
translation correction preserving the target tilt. Recovery permits a 3% margin
along its checked path and restores 10% at its destination. A margin below 3%
or lost detection aborts the run. At most four trajectory attempts are allowed.
This is monitored trajectory execution with occasional corrections, not a
continuous velocity visual servo. Sampling still requires a fresh stationary view.

Loss of fresh detection, a measured margin violation, unreachable corrections,
or exhausted correction attempts stops the sequence without saving. The robot
stays at its current position on such failures; it does not move blindly to
recover detection. Return motion uses the same checks and may need a framing
correction rather than reaching the exact original joint position. Stop remains
available during all steps. Board spec or camera intrinsic changes abort a run.
The sampled geometric checks do not model occlusion or guarantee visibility
between samples; fresh feedback limits further travel when prediction is wrong.
