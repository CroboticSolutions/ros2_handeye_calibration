# Intrinsic-style ROS 2 hand-eye calibration

Default solver: `intrinsic_pose`. Source reference:
https://github.com/intrinsic-ai/intrinsic-core/tree/c61bf075f2335371c6367b61117e8a62bb960c3b/intrinsic_perception/intrinsic/perception/calibration

A sparse, pinned source checkout is available at `/root/intrinsic_calibration_upstream`.
This is a ROS/MoveIt adaptation of the public algorithm and acquisition architecture,
not an installed Intrinsic service stack. Their gRPC/ICON services and Ceres runtime
are not used. See `THIRD_PARTY_NOTICES.md` and `LICENSE.intrinsic`.

## Pipeline

1. Verify current joint feedback, controller, collision scene and board framing.
2. Sample eight systematic flange poses: ±XYZ and two diagonal axes, with 20 mm
   translation and 12° rotation relative to the initial reference. Every move
   requires a timed, collision-checked MoveIt trajectory, including pre-calibration.
3. After feedback settles, wait an additional interruptible second, then capture
   a stationary time-synchronized burst. Default: return by a checked path to the
   reference pose between captures (`intrinsic_return_to_base:=true`).
4. With at least six successful initial captures and non-degenerate rotation
   information, compute the initial joint pose estimate. Reject it for planning
   if maximum pose error exceeds 10 mm or 6°.
5. Freeze that planning estimate. Collect 20 main poses (`auto_min_training_samples`)
   in a box of half-size 60/60/40 mm around the reference camera, looking at the
   board centre plus uniform local XYZ tilts ±15°/±15°/±35°. Check image margin,
   configured camera excursion, numerical IK, and the actual planned path.
   Do not refit or discard views while collecting this batch.
6. Solve all accepted initial and main pose pairs together. Shah initializes X/Y;
   SciPy least-squares optimizes their pose residual with squared loss and the
   initial residual variance weighting used by Intrinsic. No sample rejection,
   no AX=XB algorithm sweep, no pixel-based refinement in this mode.
7. Local acceptance extensions: minimum 20 training views, rotation observability,
   pose RMS ≤3 mm / 1°, bootstrap position sigma ≤2 mm (configurable existing
   `auto_max_position_sigma_m` for automatic acquisition). These are our defaults,
   not claimed to be upstream universal thresholds.
8. Freeze training and collect five additional independent views (minimum three).
   Their maximum prediction error must be ≤4 mm / 3°. They cannot affect training,
   initial values, weighting, bootstrap or model selection. Save reuses the cached
   training fit and evaluates the held-out views.

Typically 28 training + 5 validation samples. Failed path candidates are resampled
within a finite candidate budget; three unsuccessful captures in succession stop
collection. A failed run keeps raw evidence and does not replace the active file.
Old calibration/TCP files are not applied or overwritten by installing this change.

The original `reprojection` and `axxb` solvers remain explicitly selectable for
comparison/rollback; their legacy acquisition path is not the new default.
`auto_max_training_samples` belongs to that legacy path; the Intrinsic-style path
uses a fixed main batch plus the initial and validation phases.

## Use

Launch the existing calibration page/collector; its installed launch defaults to
`solver:=intrinsic_pose`. Preserve the actual camera topics and printed board
configuration. Robot motion starts only through the existing Calibrate action.

Offline replay (does not apply or write an active calibration):

```bash
source /opt/ros/jazzy/setup.bash
source /root/arms_ws/install/setup.bash
ros2 run hand_eye_calibration handeye_intrinsic_solve \
  /root/.ros/hand_eye_calibration_runs/20260930_151536 \
  --bootstrap 40 --output /tmp/intrinsic_151536.json
```

Exit 0 means local acceptance passed; exit 2 means rejected. Reports identify
`solver: intrinsic_pose`, `algorithm: SHAH+NONLINEAR`, `optimizer: scipy_least_squares`
and the pinned upstream revision. This is the same objective adapted to SciPy,
not a claim of binary/numerical identity with Ceres.

## Verification (2026-10-01)

- Synthetic known transforms recover camera and board; deliberately corrupted
  views stay in the solve and cause rejection.
- Actual ROS collector tested in isolated domain153, with temporary calibration
  files; save, rejection, unchanged training under altered held-out data.
- Virtual acquisition verifies exactly two fits (after 8 and 28 training views),
  followed by 5 held-out views. No refitting during main/validation collection.
- Recorded run151536: all16 views retained; pose RMS2.802 mm/0.531°, max6.188 mm/1.022°,
  sigma5.560 mm. Rejected: insufficient sample count and position uncertainty.
- Recorded PiPER kinematics:8/8 initial IK;50/120 main candidates meet both predicted
  framing and IK. This test does not check collision paths or prove physical accuracy.
- No physical robot movement performed during migration.

Evidence: `/root/arms_ws/diagnostics/intrinsic_migration_20261001`.

### Initial acquisition recovery (2026-10-01)

The ROS adaptation now targets eight diverse initial captures using at most
24 deterministic candidates: eight axes at 6 degrees / 20 mm, then replacement
candidates at 4 degrees / 15 mm and 3 degrees / 10 mm. IK, planning, capture and
near-duplicate failures are counted separately in `planner_rejections`; initial
candidate progress is published and logged. All paths still use checked MoveIt
trajectories, and capture retains existing detection/stationarity gates. At least
six accepted views with multi-axis observability remain mandatory; final fitting
and independent validation thresholds are unchanged. This recovery extends the
upstream fixed-eight pre-calibration strategy. It reacts to tool occlusion via
failed detection and smaller alternative views; it does not predict occlusion
from the tool mesh. Main sampling is unchanged.
