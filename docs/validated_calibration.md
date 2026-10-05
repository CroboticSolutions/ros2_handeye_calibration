> This page documents the legacy `reprojection` pipeline. The default is now
> `intrinsic_pose`; see [Intrinsic-style architecture](intrinsic_architecture.md).

# Hand-eye: reprojection solve, held-out validation, datasets

What changed in September 2026 and how to use it. The closed-form path is
still there (`solver:=axxb`), but it is no longer the default.

## Pipeline

1. `charuco_detector` publishes the board TF **and** the raw corners of every
   accepted frame on `/charuco_detector/observation` (`std_msgs/String` JSON:
   stamp, ids, pixels, board points, CameraInfo, board spec).
2. `hand_eye_calibration` buffers observations by stamp and measures the
   camera latency (arrival time − header stamp). The status topic reports it as
   `camera_latency` (median / p95).
3. A capture uses only frames that
   - are younger than `max(1.0 s, p95 latency + 0.25 s)`,
   - were taken while the robot was still (the robot pose `stationary_window_s`
     = 0.3 s before the frame equals the pose at the frame), and
   - in automatic mode, were exposed after the robot settled (frame stamp ≥
     settle time + 50 ms; this compares stamps with stamps, so it holds for any
     latency).
4. Solve on the training samples:
   closed form (5 OpenCV methods, best AX=XB residual) → MAD outlier rejection
   → **joint least squares on corner reprojection error** over the camera mount
   `X` and the board pose in the robot base `B` (optionally `fx fy cx cy` with
   `estimate_intrinsics:=true`). Whole views that are reprojection outliers are
   dropped (Huber loss, ≤ 20 %).
5. Evidence, computed at save time:
   - **board spread**: every view's own PnP board pose mapped into the base with
     `X`; the board is static, so the spread is a direct consistency number in mm;
   - **leave-one-out**: refit without each view, predict it (board position in
     mm, corner pixels);
   - **validation views** (automatic mode, 5 by default): poses collected after
     the training set, never used in the solve;
   - **bootstrap σ** over poses (the independent unit).
6. Acceptance gate (`acceptance.py`, parameters `accept_*`):

   | check | default |
   |---|---|
   | training poses | ≥ 12 |
   | bootstrap σ, worst direction | ≤ 2 mm |
   | board spread RMS | ≤ 3 mm |
   | leave-one-out board position RMS | ≤ 3 mm |
   | leave-one-out reprojection RMS | ≤ 3 px |
   | validation poses, max board position error | ≤ 4 mm |

   Pass → `calibration_file` is replaced (old copy in `.previous`).
   Fail → the active file is untouched; the candidate goes to
   `<calibration_file>.rejected.yaml` and the service reports which check
   failed. `acceptance_mode:=warn` saves anyway with a warning (diagnostics only).
7. Every save, and every failed automatic run with ≥ 4 samples, writes a
   dataset to `dataset_dir` (default `~/.ros/hand_eye_calibration_runs/<UTC>/`):
   `dataset.json` (poses, raw corners per frame, joints, roles, camera, URDF),
   `images/NN.png`, `calibration.yaml`.

These numbers are still internal to the robot + camera. The only external
check is the touch-off below.

## Offline re-solve

```bash
ros2 run hand_eye_calibration handeye_offline_solve ~/.ros/hand_eye_calibration_runs/<run>
ros2 run hand_eye_calibration handeye_offline_solve <run> --intrinsics      # also refine fx fy cx cy
ros2 run hand_eye_calibration handeye_offline_solve <run> --joint-offsets   # also joint zero offsets
ros2 run hand_eye_calibration handeye_offline_solve <run> --yaml out.yaml   # write a calibration YAML
```

It prints closed form, AX=XB refined and reprojection results side by side with
the evidence above. Exit code 0 = accepted, 2 = rejected. Joint offsets (joints
2…n−1; the first is absorbed by the board pose, the last by the mount) show
whether the robot's kinematic model, not the camera, limits accuracy.

## Automatic mode

- Freshness limits (1.0 / 0.35 / 0.2 s) scale with the measured latency.
- Targeted views: 8–38° camera rotations, ±60° roll, depth ×0.85/1.2, board
  normal ≤ 50° from the optical axis, camera within `auto_max_camera_excursion_m`
  (0.30 m) of the start.
- Initialization is assessed from six poses onward, after at least 20° pairwise
  rotation and 10° span on a second principal rotation axis. Local candidate
  scoring rewards progress toward missing coverage. The initial fit must also
  pass consistency, bootstrap uncertainty and successive-estimate stability checks.
- Training uses `auto_min_training_samples=20` through
  `auto_max_training_samples=30`. Full quality is assessed every two additional
  training views; final acceptance thresholds are unchanged. An invalid maximum
  below the minimum fails at startup rather than silently changing it.
- The common board pose comes from the joint pixel fit (or all-view consensus),
  never from the most recent single observation.
- After training passes, freeze the estimate and collect `auto_validation_views=5`
  independent held-out poses (minimum three). They never update the training fit.
  Frozen-prediction validation requires maximum board-centre error ≤4 mm and
  rotation error ≤3°, followed by the existing save acceptance gate.
- Build/install the package and restart the collector after changing launch
  defaults. Updating Python source alone can leave the old installed launch active.
- Watchdog: with `auto_heartbeat_timeout_s > 0` (the GUI launch passes 3 s) the
  run needs a heartbeat on `hand_eye_calibration/auto_heartbeat`; the GUI page
  publishes it every second. Silence cancels the trajectory and stops.
- The GUI asks for confirmation before the first motion.

## Touch-off (physical check)

Measures hand-eye + TCP + robot error together, which is what a seam sees.

1. Board fixed and detected, robot still:
   `ros2 service call /hand_eye_calibration/touchoff_reference std_srvs/srv/Trigger {}`
   (stores the board pose in base through the live camera TF and through the
   in-memory candidate, if any).
2. Jog the tool tip (`touchoff_tcp_frame`, default `arm_tcp`) onto ChArUco corner
   `touchoff_corner_id` (default 0; the annotated image shows corner ids).
   Do not move the board.
3. `ros2 service call /hand_eye_calibration/touchoff_capture std_srvs/srv/Trigger {}`
   → error vector in mm per reference; logged to
   `<dataset_dir>/touchoff/touchoff_log.yaml` and shown in the GUI.

Repeat on 3–5 corners across the board.

## Apply on the welding-gun stack

`piper_orbbec_femto_bolt_welding_gun` now has an Apply path
(`handeye_welding_gun_apply`):

- camera: rewrites the **real** origin passed to the femto_bolt macro in
  `piper_femto_bolt_handeye_macros.xacro` as `T_cal · inv(camera_base_link →
  optical)`, with the inner chain from the running driver's TF (`--nominal` also
  updates the simulation value from the femto_bolt xacro);
- TCP: rewrites the real-robot branch of `arm_tcp_joint` in
  `piper_welding_gun.urdf.xacro`, tool +Z → arm_tcp +X, keeping the roll
  closest to the current origin (reproduces the 23 September hand bake exactly).

Both refuse a YAML whose acceptance failed and keep a `.before_apply` copy.
robot_state_publisher is updated live; **MoveIt needs a hardware stack restart**.

# Tool TCP (pivot): gate, reorientation check, datasets

Why: the first welding-gun TCP had 4 touches whose rotations were mostly about
one axis (principal spans 85° / 15° / 3°), RMS 3.5 mm and a 29 mm
leave-one-out shift. The old `ready_to_save` only required 4 touches; the
"≥ 15° per flange axis" check was advisory and basis dependent.

Industrial practice (ABB 4/5/6-point, KUKA XYZ 4-point + ABC, FANUC 6-point
for bent torches, Yaskawa/UR reorientation test): touches from clearly
different orientations about several axes, mean error ≲ 0.5-1 mm, orientation
defined explicitly, then a reorientation check where the tip must stay on the
spike. Cut the wire to a fixed stick-out (or use a rigid pin) before touching.

## Acceptance (`tcp_quality.py`, parameters `accept_*`)

| check | default |
|---|---|
| tip touches | ≥ 8 |
| second principal rotation span | ≥ 20° (rotation about two non-parallel axes) |
| pivot RMS | ≤ 1 mm |
| max leave-one-out TCP shift | ≤ 1.5 mm |
| reorientation / validation touches | ≥ 3, max ≤ 1 mm (not in the fit) |
| axis mode: alignments | ≥ 3, spread ≤ 1° |
| axis mode: neck angle vs CAD (`cad_axis_angle_deg`, 35°) | ≤ 5° |

Save always writes a dataset (`~/.ros/tool_tcp_calibration_runs/<UTC>/dataset.json`:
flange poses, joints, rounds, URDF). The active `tool_tcp_calibration.yaml` is
replaced only on pass (old copy in `.previous`); otherwise
`tool_tcp_calibration.yaml.rejected.yaml`. `acceptance_mode:=warn` saves anyway.

## Tool frame (6-point style)

+Z = wire axis from the alignment round (average of ≥ 3 alignments to the spike
direction `spike_axis_base`). Roll is explicit (`tcp_roll_convention:=bend_plane`):
+X lies in the torch-neck bend plane (wire axis + flange Z), pointing back
along the flange axis. The welding-gun Apply then maps tool +Z to arm_tcp +X.

## Reorientation check (Validate tab)

`tool_tcp_calibration/reorient_next` moves the robot, slowly and
collision-checked (MoveIt Cartesian path, same executor safeguards as automatic
hand-eye), in one path: lift the tip to 30 mm above the fitted spike point,
rotate the tool about its fitted tip (±25° tilt about two horizontal axes, ±45°
about the spike axis; 6 poses), descend to 5 mm above the spike. The operator
closes the 5 mm by hand and presses Capture point in the Validate round; the
touch is scored against the frozen fit. `motion_stop` cancels; with
`motion_heartbeat_timeout_s` (GUI: 3 s) the motion stops when the page closes.
If the path is not 100 % feasible nothing is sent.

## Offline

```bash
ros2 run hand_eye_calibration tcp_offline_solve ~/.ros/tool_tcp_calibration_runs/<run>
ros2 run hand_eye_calibration tcp_offline_solve <run> --joint-offsets
```

`--joint-offsets` fits the tip, the spike point and zero offsets of joints
2…n−1 together (joint 1 is absorbed by the spike point, the last joint by the
tip). A large residual drop means the robot's kinematic model, not the TCP,
limits accuracy.

## Solver selection and refinement checks (2026-09-30)

The five OpenCV methods are now ranked by leave-one-pose-out AX=XB residuals:
fit N−1 poses and score only pairs involving the omitted pose. A method with
any failed fold is ineligible. `algorithm_selection` records every method's
score and fold count. Dedicated validation poses never enter this selection.

AX=XB exclusions remain excluded from the final reprojection fit. Reports
separate `axxb_rejected_views`, `reprojection_rejected_views`, `kept_views`, and
the union `rejected_views`; node indices refer to the original capture order.
Final residual reporting uses the final survivor set.

On Save and in the offline CLI, a nested leave-one-pose-out comparison rebuilds
algorithm selection, AX=XB rejection/refinement, and pixel refinement using
only the training portion of each fold. It predicts **every** original training
pose, including poses rejected by the full-data fit. This replaces the previous
warm-started LOO metric in those paths; reported values are not directly
comparable with older retained-only LOO reports. Dedicated validation remains
separate and is not used to pick a solver.

`refinement_non_regression` blocks acceptance if pixel refinement materially
worsens held-out RMS pixels, board position, or orientation relative to AX=XB.
Non-regression limits are 1.05 × baseline plus respectively 0.05 px, 0.5 mm,
and 0.1 degrees. These account for small fit/noise differences; they do **not**
relax the existing absolute quality thresholds. Failure blocks saving a new
active candidate; it does not silently apply an AX=XB fallback. Full comparison
runs only for full evaluation, not each acquisition preview, and adds compute
time. Fold transforms and errors are retained in `refinement_comparison`.

After installing this change, restart only the calibration application to load
new Python code. Export/save any needed in-memory samples first. No robot or
controller restart is required for the solver change. Already applied hand-eye
and TCP calibrations are not replaced by a build or an offline evaluation.
