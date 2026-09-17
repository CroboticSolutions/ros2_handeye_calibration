# Automatic hand-eye calibration without an initial mount transform

Start with a stationary ChArUco board fully inside the RGB image, with at least
10% image margin. Press **Calibrate**. The same acquisition algorithm is used for
real and simulated robots with a configured serial revolute chain, MoveIt group
and FollowJointTrajectory controller. Opening the page does not send motion.

## Inputs and frame contract

The worker requires joint feedback, robot URDF joint limits, a robot-base to
robot-effector chain, RGB CameraInfo, and detected board poses in that same RGB
optical frame. The board detector uses camera intrinsics and the printed board
dimensions. It does not need a robot-to-camera transform.

No existing camera mount TF, saved calibration, camera IK, or orbit planned from
nominal camera extrinsics is used. Robot FK stops at the robot effector, before
any camera mount joint. CameraInfo and detector optical-frame names must agree;
a mismatch fails before motion rather than silently applying a mount transform.

## Acquisition

1. Capture an initial observation and probe an unknown joint direction with a
   0.01 rad increment. Continue from the current pose; there is no per-joint
   excursion back to the start.
2. Learn a local mapping from measured joint increments to normalized board-edge
   pixels (an image Jacobian). Only observed joint directions may be predicted.
   Keep the last 32 measurements near the current configuration; large prediction
   errors discard confidence and stale models.
3. Bootstrap samples require at least 2.5 degrees of rotation from every retained
   view; translation alone does not count as a new sample. Favor rotation about
   undersampled axes so a large number of translations cannot fill the budget.
   Rank candidate increments by new robot-pose information, rotation diversity
   and predicted image margin. Include small centering combinations derived from
   the measured image Jacobian. This requires no initial camera mount transform.
4. Start with 0.01 rad probes, use 0.02 rad steps for learned directions, and permit
   up to 0.05 rad only after successful prediction checks and repeated evidence
   in that direction. During bootstrap, joint positions remain within 0.35 rad
   of the initial configuration. Collision checks and live image monitoring apply to every step.
5. Stop and observe after each selected increment. If a move loses its margin,
   reacquire the stationary board and retrace ONLY the last observed segment.
   There is no scheduled return to the initial pose, including on success.
6. Solve the initial mount estimate from six training samples. The sixth sample
   must establish sufficient rotation about two axes. Use those same samples to
   reject non-finite or inconsistent estimates (10 mm / 3 degrees); this is an
   internal fit check, not independent validation. No extra views are collected.
   Using that approximate estimate, generate camera views around the measured board centre with
   6–30 degree rotations about multiple axes and distances at 92%, 100% and 108%
   of the initial viewing depth. Solve for coordinated joint targets within the
   robot's actual joint limits. Rank targets using predicted image sensitivity to
   both the camera mount and the unknown board pose, marginalizing the board
   nuisance parameters. This equal-pixel-noise approximation is used for ranking
   only; final uncertainty still uses measured bootstrap fits.
   Request complete joint paths from MoveIt `/plan_kinematic_path`; an invalid
   straight connection alone does not reject a target. Check every returned
   segment for predicted board visibility, preserve detours and execute stopped
   increments along that path. Recheck collisions before each command. Bound
   requests to six per selection and paths to 2.4 rad accumulated maximum-joint
   distance. Remove only duplicate and collinear path points, retaining corners.
   Execute in increments of at most 0.05 rad with live tracking, refreshing
   the board prediction from actual observations before every increment.
   Refinement samples must differ from every retained view by at least 5 degrees,
   or by both 2.5 degrees and 30 mm. Parallel translations alone are rejected.
   Capture the nine additional samples only at selected target endpoints, never
   at intermediate transit stops. Re-estimate the mount and regenerate targets
   after every accepted sample. If no planned view remains, stop explicitly. Penalize
   repeated local destinations, while allowing transit through known positions.
   Reject similar measurements, not necessary transit poses. Stop after eight
   local attempts or 16 non-progressing attempts without a new view. A checked
   route that measurably advances towards its selected target may use up to 48
   increments without a new view; route advancement never resets sample counts.
   Failures preserve the saved calibration. Status separates accepted views,
   movement attempts, stagnation limits and target rejection reasons.
7. Keep all six initial and nine targeted observations in the calibration dataset.
   Assess uncertainty from all 15 samples while stationary. No bootstrap holdouts
   or final independent validation observations are acquired.
8. Save only if internal consistency and uncertainty (default 2 mm) pass. Failure
   preserves the previous result and never extends the sample budget. The legacy
   `auto_max_training_samples` option does not change the fixed 6+9 acquisition.
   Travel increments and rejected observations do not consume samples.

Internal fit checks and uncertainty do not measure independent absolute accuracy.
The GUI and saved metadata explicitly record the absence of independent validation.
The loop remains bounded by movement attempts and tracking/visibility watchdogs.

## Robot integration

At start, derive controlled joint names from the configured base-to-effector URDF
chain. Match its DOFs to one SRDF group from `/move_group/get_parameters` and one
active JointTrajectoryController from `/controller_manager/list_controllers`.
`auto_group` and `auto_controller` launch arguments resolve ambiguous deployments;
GUI launch configuration also supports `CALIB_MOVEIT_GROUP` and
`CALIB_TRAJECTORY_ACTION`. Robot base and effector frames must match the selected
robot (`CALIB_ROBOT_BASE_FRAME`, `CALIB_ROBOT_EFFECTOR_FRAME` or launch overrides).
No saved joint-pose sequence or nominal camera mount is required. Current support
is for bounded independent revolute joints; unsupported chains fail before motion.
Missing/ambiguous configuration, controller or planner also fails before motion.
Collision checking is enabled for both real and simulated GUI profiles.

## Motion and interruption behavior

Real profiles require `/check_state_validity`. Every joint segment is checked
against the current MoveIt scene at <=0.015 rad intervals and rechecked before
submission to `/arm_controller/follow_joint_trajectory`. Attached bodies are
retained. Camera visibility and collision failures have separate reasons.
These discrete checks cover modeled geometry, not unmodeled obstacles.

Each motion uses zero endpoint velocities. The real-profile duration bounds the
commanded cubic interpolation to 0.06 rad/s and 0.12 rad/s² at full acquisition speed (twice the previous speed scale). Recovery returns
remain at half speed, slower on retries. Segments have a 0.5 second duration floor before scaling; the
velocity/acceleration bounds determine longer durations when needed. Acquisition is intentionally stop-and-observe
and can take several minutes; it is not continuous visual servoing.

A 20 Hz motion monitor uses recent accepted detections, permits single rejected
images within a 350 ms pose-age bound, and watches advancing timestamps on a
wall-clock deadline. It cancels at an 8% image margin; the hard margin is 3%.
After cancellation it waits for the controller terminal result and settling.
Reacquisition waits at most five seconds, requiring multiple accepted timestamps
spanning 200 ms. No motion resumes while the board is absent. A successfully
reacquired view leads to a checked return of the last segment and a different direction,
not a large camera-pose correction. A recorded return can start in the 3–8% warning
band, but hard margins, timestamps, joint bounds and collisions remain checked.

**Stop** cancels motion and interrupts waiting; it does not automatically return
the robot. Hard visibility loss, stale joints, controller failure or unavailable
collision checking stops the run. Failed/stopped runs do not replace the saved
calibration. Successful saves preserve the previous YAML atomically. **Apply**
remains a separate user action.

## Computation and visualization

The nonlinear solver precomputes relative sample pairs and evaluates their
residuals in batches. The same all-pairs objective, robust loss and 40 bootstrap
resamples are retained. An exact-input cache avoids repeating uncertainty at save
when the transform, measurements, algorithm and bootstrap settings are unchanged.

RViz can display a Marker on `/hand_eye_calibration/board_estimate`, in the robot
base frame. The translucent orange rectangle is an estimated board pose, not a
measured ground truth or a collision object. It is published only after a mount
estimate passes the six-sample internal fit check, and is removed when the run ends. Motion
uses current observed image geometry and collision checks, not the marker.

## Services and verification

- `/hand_eye_calibration/auto_start` and `/auto_stop` (Trigger).
- `/hand_eye_calibration/status` contains `automatic`, including `phase`
  (`bootstrap` / `refinement`), initial/targeted sample counts and internal fit consistency. `validation` is null.

Tests use an independent virtual camera with a hidden, nontrivial mount transform.
Any initial camera TF access raises an exception. Tests verify transform recovery,
six initial plus nine targeted samples, rejection of poisoned estimates, bounded increments,
collision rejection, tracking loss, cancellation and Stop. Hardware read-only
checks do not constitute a completed moving calibration or absolute metrology.
