"""
Build JSON pivot (tool-TCP) calibration readiness status for the GUI coach panel.
Mirrors calibration_status.py's readiness/checklist shape so the frontend can
reuse the same panel component.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

MIN_SAMPLES = 4
TARGET_SAMPLES = 10
MIN_ROTATION_SPAN_DEG = 40.0
MIN_AXIS_ROTATION_SPAN_DEG = 15.0

# "Good" thresholds gate "excellent"; "warn" thresholds gate "ready_to_save".
GOOD_RMS_RESIDUAL_M = 0.001
WARN_RMS_RESIDUAL_M = 0.001
GOOD_MAX_RESIDUAL_M = 0.002
WARN_MAX_RESIDUAL_M = 0.002
MAX_OUTLIER_FRACTION = 0.20
MAX_CI95_HALF_WIDTH_M = 0.001
MAX_LOO_SHIFT_M = 0.001

MIN_VALIDATION_SAMPLES = 5
TARGET_VALIDATION_SAMPLES = 10
VALIDATION_RMS_LIMIT_M = 0.001
VALIDATION_MAX_LIMIT_M = 0.002

CONDITION_NUMBER_WARN = 50.0

# Axis (align-to-spike) thresholds.
# One alignment pose already yields an axis; more are averaged to reduce the
# operator's manual-alignment noise.
MIN_ALIGN_SAMPLES = 1
TARGET_ALIGN_SAMPLES = 3
# Spread of the per-pose axis directions about their mean == axis uncertainty.
GOOD_AXIS_ANGLE_DEG = 1.0
WARN_AXIS_ANGLE_DEG = 3.0


def _orientation_span_deg(flange_samples: list[list[float]]) -> float:
    """Max rotation (deg) of any sample's flange orientation from the first sample."""
    if len(flange_samples) < 2:
        return 0.0
    # Deferred import: only rotation math needed here, keep this module import-light.
    from scipy.spatial.transform import Rotation as Rot

    first = Rot.from_quat(flange_samples[0][3:])
    deltas = [
        (first.inv() * Rot.from_quat(sample[3:])).magnitude()
        for sample in flange_samples[1:]
    ]
    return float(math.degrees(max(deltas)))


def _orientation_axis_spans_deg(flange_samples: list[list[float]]) -> list[float]:
    """Rotation-vector coverage about the first pose's local X/Y/Z axes."""
    if len(flange_samples) < 2:
        return [0.0, 0.0, 0.0]
    from scipy.spatial.transform import Rotation as Rot

    first = Rot.from_quat(flange_samples[0][3:])
    vectors = np.asarray(
        [(first.inv() * Rot.from_quat(sample[3:])).as_rotvec() for sample in flange_samples]
    )
    spans = np.degrees(np.max(vectors, axis=0) - np.min(vectors, axis=0))
    return [float(abs(v)) for v in spans]


def _checklist(
    sample_count: int,
    orientation_span_deg: float,
    axis_spans_deg: list[float],
    pivot: dict[str, Any] | None,
    last_residual_m: float | None,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = [
        {
            "id": "min_samples",
            "label": f"Minimum {MIN_SAMPLES} samples",
            "ok": sample_count >= MIN_SAMPLES,
            "detail": f"{sample_count} / {MIN_SAMPLES}",
        },
        {
            "id": "target_samples",
            "label": f"Target {TARGET_SAMPLES} samples",
            "ok": sample_count >= TARGET_SAMPLES,
            "detail": f"{sample_count} / {TARGET_SAMPLES}",
        },
        {
            "id": "orientation_spread",
            "label": "Total wrist orientation coverage",
            "ok": sample_count < 2 or orientation_span_deg >= MIN_ROTATION_SPAN_DEG,
            "detail": f"{orientation_span_deg:.1f}° / {MIN_ROTATION_SPAN_DEG:.0f}°",
        },
        {
            "id": "axis_coverage",
            "label": "Rotation coverage about X / Y / Z",
            "ok": sample_count < 2 or all(v >= MIN_AXIS_ROTATION_SPAN_DEG for v in axis_spans_deg),
            "detail": (
                f"{axis_spans_deg[0]:.0f}° / {axis_spans_deg[1]:.0f}° / "
                f"{axis_spans_deg[2]:.0f}° (each ≥ {MIN_AXIS_ROTATION_SPAN_DEG:.0f}°)"
            ),
        },
    ]

    if pivot is not None:
        rms = pivot["rms_residual_m"]
        inlier_max = pivot.get("inlier_max_residual_m", pivot["max_residual_m"])
        items.append(
            {
                "id": "rms_residual",
                "label": "RMS pivot error",
                "ok": rms <= WARN_RMS_RESIDUAL_M,
                "detail": f"{rms * 1000:.2f} mm (good < {GOOD_RMS_RESIDUAL_M * 1000:.0f} mm)",
            }
        )
        items.append(
            {
                "id": "max_inlier_residual",
                "label": "Worst accepted touch",
                "ok": inlier_max <= WARN_MAX_RESIDUAL_M,
                "detail": f"{inlier_max * 1000:.2f} mm / {WARN_MAX_RESIDUAL_M * 1000:.0f} mm",
            }
        )
        outlier_count = pivot.get("outlier_count", 0)
        outlier_fraction = outlier_count / sample_count if sample_count else 0.0
        items.append(
            {
                "id": "robust_inliers",
                "label": "RANSAC accepted touches",
                "ok": outlier_fraction <= MAX_OUTLIER_FRACTION,
                "detail": f"{pivot.get('inlier_count', sample_count)} accepted, {outlier_count} rejected",
            }
        )
        items.append(
            {
                "id": "tcp_uncertainty",
                "label": "TCP 95% confidence half-width",
                "ok": pivot.get("max_tcp_ci95_half_width_m", float("inf")) <= MAX_CI95_HALF_WIDTH_M,
                "detail": f"{pivot.get('max_tcp_ci95_half_width_m', float('inf')) * 1000:.2f} mm",
            }
        )
        items.append(
            {
                "id": "leave_one_out",
                "label": "Leave-one-out stability",
                "ok": pivot.get("max_loo_tcp_shift_m", float("inf")) <= MAX_LOO_SHIFT_M,
                "detail": f"max shift {pivot.get('max_loo_tcp_shift_m', float('inf')) * 1000:.2f} mm",
            }
        )
        items.append(
            {
                "id": "condition_number",
                "label": "Orientation diversity (conditioning)",
                "ok": pivot["condition_number"] <= CONDITION_NUMBER_WARN,
                "detail": f"condition number {pivot['condition_number']:.1f}",
            }
        )

    if last_residual_m is not None:
        items.append(
            {
                "id": "last_sample",
                "label": "Last sample quality",
                "ok": last_residual_m <= WARN_MAX_RESIDUAL_M,
                "detail": f"{last_residual_m * 1000:.2f} mm from fitted tip",
            }
        )

    return items


def _readiness(
    sample_count: int,
    orientation_span_deg: float,
    axis_spans_deg: list[float],
    pivot: dict[str, Any] | None,
) -> str:
    if sample_count < MIN_SAMPLES:
        return "not_ready"

    orientation_ok = (
        orientation_span_deg >= MIN_ROTATION_SPAN_DEG
        and all(v >= MIN_AXIS_ROTATION_SPAN_DEG for v in axis_spans_deg)
    )
    if pivot is None:
        return "collecting"

    rms = pivot["rms_residual_m"]
    max_res = pivot.get("inlier_max_residual_m", pivot["max_residual_m"])
    condition_ok = pivot["condition_number"] <= CONDITION_NUMBER_WARN
    outlier_fraction = pivot.get("outlier_count", 0) / sample_count
    robust_ok = outlier_fraction <= MAX_OUTLIER_FRACTION
    uncertainty_ok = (
        pivot.get("max_tcp_ci95_half_width_m", float("inf")) <= MAX_CI95_HALF_WIDTH_M
        and pivot.get("max_loo_tcp_shift_m", float("inf")) <= MAX_LOO_SHIFT_M
    )
    residuals_ok = rms <= WARN_RMS_RESIDUAL_M and max_res <= WARN_MAX_RESIDUAL_M

    excellent = (
        sample_count >= TARGET_SAMPLES
        and orientation_ok
        and condition_ok
        and robust_ok
        and uncertainty_ok
        and rms <= GOOD_RMS_RESIDUAL_M
        and max_res <= GOOD_MAX_RESIDUAL_M
    )
    if excellent:
        return "excellent"

    ready = orientation_ok and condition_ok and robust_ok and uncertainty_ok and residuals_ok
    if ready:
        return "ready_to_save"

    return "collecting"


def _xyz(v: list[float] | tuple[float, ...] | Any) -> dict[str, float]:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def _sample_viz(
    flange_samples: list[list[float]],
    residuals: list[float] | None,
    outlier_indices: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Flange poses in the robot base frame for the GUI 3D overlay."""
    out: list[dict[str, Any]] = []
    outliers = set(outlier_indices or [])
    for i, sample in enumerate(flange_samples):
        entry: dict[str, Any] = {
            "x": float(sample[0]),
            "y": float(sample[1]),
            "z": float(sample[2]),
            "qx": float(sample[3]),
            "qy": float(sample[4]),
            "qz": float(sample[5]),
            "qw": float(sample[6]),
            "residual_m": None,
            "is_outlier": i in outliers,
        }
        if residuals is not None and i < len(residuals):
            entry["residual_m"] = float(residuals[i])
        out.append(entry)
    return out


def build_pivot_status(
    *,
    flange_samples: list[list[float]],
    pivot: dict[str, Any] | None,
) -> dict[str, Any]:
    sample_count = len(flange_samples)
    orientation_span_deg = _orientation_span_deg(flange_samples)
    axis_spans_deg = _orientation_axis_spans_deg(flange_samples)
    residuals = (
        pivot["per_sample_residuals_m"]
        if pivot is not None and pivot["per_sample_residuals_m"]
        else None
    )
    last_residual_m = residuals[-1] if residuals else None

    readiness = _readiness(sample_count, orientation_span_deg, axis_spans_deg, pivot)
    # Four diverse poses are the standard minimum pivot solve.  Quality and
    # validation diagnostics remain visible, but an experienced operator may
    # save a provisional calibration as soon as a full-rank four-pose fit exists.
    ready_to_save = sample_count >= MIN_SAMPLES and pivot is not None

    guidance: list[str] = []
    if sample_count < MIN_SAMPLES:
        guidance.append(f"Need at least {MIN_SAMPLES} touches before a fit is possible.")
    elif orientation_span_deg < MIN_ROTATION_SPAN_DEG:
        guidance.append(
            "Need more wrist orientation variation between touches (rotate the wrist more between samples)."
        )
    elif any(v < MIN_AXIS_ROTATION_SPAN_DEG for v in axis_spans_deg):
        guidance.append(
            "Rotate about all three wrist axes; current X/Y/Z coverage is "
            + "/".join(f"{v:.0f}°" for v in axis_spans_deg)
            + "."
        )
    if pivot is not None and pivot["condition_number"] > CONDITION_NUMBER_WARN:
        guidance.append("Orientations are too similar to separate translation from the pivot point reliably.")
    if pivot is not None and pivot.get("outlier_indices"):
        numbers = ", ".join(str(i + 1) for i in pivot["outlier_indices"])
        guidance.append(f"RANSAC rejected touch(es) {numbers}; remove and re-capture them.")
    if pivot is not None and pivot.get("influential_sample_indices"):
        numbers = ", ".join(str(i + 1) for i in pivot["influential_sample_indices"])
        guidance.append(f"Touch(es) {numbers} have high leave-one-out influence; inspect them.")
    if last_residual_m is not None and last_residual_m > WARN_MAX_RESIDUAL_M:
        guidance.append(
            f"Last touch is {last_residual_m * 1000:.1f} mm off the fitted tip point — "
            "consider removing it and re-touching."
        )
    if not guidance:
        if readiness == "excellent":
            guidance.append("Excellent — no need to collect more samples. Save when ready.")
        elif readiness == "ready_to_save":
            guidance.append("Ready to save — quality is sufficient.")
        else:
            guidance.append("Keep collecting touches from varied wrist orientations.")

    summary = guidance[0]

    payload: dict[str, Any] = {
        "sample_count": sample_count,
        "min_samples": MIN_SAMPLES,
        "target_samples": TARGET_SAMPLES,
        "readiness": readiness,
        "ready_to_save": ready_to_save,
        "summary": summary,
        "guidance": guidance,
        "checklist": _checklist(
            sample_count, orientation_span_deg, axis_spans_deg, pivot, last_residual_m
        ),
        "orientation_span_deg": orientation_span_deg,
        "orientation_axis_spans_deg": axis_spans_deg,
        "last_sample_residual_m": last_residual_m,
        # Always present so the GUI 3D scene can draw flange poses as they arrive.
        "samples": _sample_viz(
            flange_samples,
            residuals,
            pivot.get("outlier_indices") if pivot is not None else None,
        ),
    }
    if pivot is not None:
        payload["estimate"] = {
            "tx": pivot["tcp_translation"][0],
            "ty": pivot["tcp_translation"][1],
            "tz": pivot["tcp_translation"][2],
        }
        payload["fixed_point"] = _xyz(pivot["fixed_point"])
        payload["condition_number"] = pivot["condition_number"]
        payload["rms_residual_m"] = pivot["rms_residual_m"]
        payload["max_residual_m"] = pivot["max_residual_m"]
        payload["per_sample_residuals_m"] = pivot["per_sample_residuals_m"]
        payload["method"] = pivot.get("method")
        payload["inlier_count"] = pivot.get("inlier_count")
        payload["outlier_count"] = pivot.get("outlier_count")
        payload["outlier_indices"] = pivot.get("outlier_indices", [])
        payload["influential_sample_indices"] = pivot.get("influential_sample_indices", [])
        payload["max_tcp_ci95_half_width_m"] = pivot.get("max_tcp_ci95_half_width_m")
        payload["max_loo_tcp_shift_m"] = pivot.get("max_loo_tcp_shift_m")
    return payload


def build_validation_status(
    flange_samples: list[list[float]],
    validation: dict[str, Any] | None,
) -> dict[str, Any]:
    count = len(flange_samples)
    rms = validation.get("rms_residual_m") if validation else None
    max_res = validation.get("max_residual_m") if validation else None
    quality_ok = bool(
        validation
        and rms is not None
        and max_res is not None
        and rms <= VALIDATION_RMS_LIMIT_M
        and max_res <= VALIDATION_MAX_LIMIT_M
    )
    ready = count >= MIN_VALIDATION_SAMPLES and quality_ok
    if count < MIN_VALIDATION_SAMPLES:
        readiness = "not_ready"
    elif not quality_ok:
        readiness = "collecting"
    elif count >= TARGET_VALIDATION_SAMPLES:
        readiness = "excellent"
    else:
        readiness = "ready_to_save"

    checklist = [
        {
            "id": "validation_samples",
            "label": f"Held-out validation touches",
            "ok": count >= MIN_VALIDATION_SAMPLES,
            "detail": f"{count} / {MIN_VALIDATION_SAMPLES} minimum ({TARGET_VALIDATION_SAMPLES} target)",
        },
        {
            "id": "validation_rms",
            "label": "Validation RMS",
            "ok": rms is not None and rms <= VALIDATION_RMS_LIMIT_M,
            "detail": "n/a" if rms is None else f"{rms * 1000:.2f} mm / 1.00 mm",
        },
        {
            "id": "validation_max",
            "label": "Worst validation touch",
            "ok": max_res is not None and max_res <= VALIDATION_MAX_LIMIT_M,
            "detail": "n/a" if max_res is None else f"{max_res * 1000:.2f} mm / 2.00 mm",
        },
    ]
    if count < MIN_VALIDATION_SAMPLES:
        guidance = [
            "Keep the tip on the same spike and capture new wrist orientations; these poses are not used in the fit."
        ]
    elif quality_ok:
        guidance = ["Held-out validation passed."]
    else:
        guidance = [
            "Validation failed: re-check spike contact and robot kinematics before saving."
        ]
    residuals = validation.get("per_sample_residuals_m") if validation else None
    return {
        "sample_count": count,
        "min_samples": MIN_VALIDATION_SAMPLES,
        "target_samples": TARGET_VALIDATION_SAMPLES,
        "readiness": readiness,
        "ready_to_save": ready,
        "summary": guidance[0],
        "guidance": guidance,
        "checklist": checklist,
        "orientation_span_deg": _orientation_span_deg(flange_samples),
        "orientation_axis_spans_deg": _orientation_axis_spans_deg(flange_samples),
        "last_sample_residual_m": residuals[-1] if residuals else None,
        "samples": _sample_viz(flange_samples, residuals),
        "rms_residual_m": rms,
        "max_residual_m": max_res,
        "p95_residual_m": validation.get("p95_residual_m") if validation else None,
        "per_sample_residuals_m": residuals,
    }


def build_align_status(flange_samples: list[list[float]]) -> dict[str, Any]:
    """
    Readiness/checklist for the axis-alignment round. Unlike the pivot rounds
    this is not a least-squares fit: each captured pose is the flange while the
    tool's straight segment is held collinear with the (known-direction) spike.
    One pose already defines the axis; more are averaged. Shape is kept
    compatible with build_pivot_status so the GUI can reuse the round panel.
    """
    count = len(flange_samples)
    if count < MIN_ALIGN_SAMPLES:
        readiness = "not_ready"
    elif count >= TARGET_ALIGN_SAMPLES:
        readiness = "excellent"
    else:
        readiness = "ready_to_save"
    ready_to_save = count >= MIN_ALIGN_SAMPLES

    checklist = [
        {
            "id": "align_min_samples",
            "label": f"Minimum {MIN_ALIGN_SAMPLES} alignment pose",
            "ok": count >= MIN_ALIGN_SAMPLES,
            "detail": f"{count} / {MIN_ALIGN_SAMPLES}",
        },
        {
            "id": "align_target_samples",
            "label": f"Target {TARGET_ALIGN_SAMPLES} poses (averaged)",
            "ok": count >= TARGET_ALIGN_SAMPLES,
            "detail": f"{count} / {TARGET_ALIGN_SAMPLES}",
        },
    ]

    guidance: list[str] = []
    if count < MIN_ALIGN_SAMPLES:
        guidance.append(
            "Make the tool's straight tip segment collinear with the spike, then capture a pose."
        )
    elif count < TARGET_ALIGN_SAMPLES:
        guidance.append(
            "One pose is enough, but capture a few (re-aligning each time) so they can be averaged."
        )
    else:
        guidance.append("Enough alignment poses captured.")

    return {
        "sample_count": count,
        "min_samples": MIN_ALIGN_SAMPLES,
        "target_samples": TARGET_ALIGN_SAMPLES,
        "readiness": readiness,
        "ready_to_save": ready_to_save,
        "summary": guidance[0],
        "guidance": guidance,
        "checklist": checklist,
        "orientation_span_deg": 0.0,
        "last_sample_residual_m": None,
        "samples": _sample_viz(flange_samples, None),
    }


def build_axis_status(
    *,
    tip_ready: bool,
    align_ready: bool,
    axis: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Readiness/checklist for the axis combine step (align-to-spike method).
    ``axis`` is the dict from PivotCalibrationBackend.compute_axis_from_alignment,
    or None when it has not been computed yet.
    """
    spread_deg = axis.get("alignment_spread_deg") if axis else None
    sample_count = axis.get("sample_count") if axis else None

    checklist: list[dict[str, Any]] = [
        {
            "id": "tip_round_ready",
            "label": "Tip round ready",
            "ok": tip_ready,
            "detail": "position of the tool tip is fitted" if tip_ready else "collect more tip touches",
        },
        {
            "id": "align_round_ready",
            "label": "Alignment pose captured",
            "ok": align_ready,
            "detail": "tool aligned to the spike" if align_ready else "capture at least one alignment pose",
        },
    ]
    if spread_deg is not None:
        detail = (
            f"±{spread_deg:.2f}° across {sample_count} poses (good ≤ {GOOD_AXIS_ANGLE_DEG:.0f}°)"
            if sample_count and sample_count > 1
            else "single pose — capture more to estimate spread"
        )
        checklist.append(
            {
                "id": "axis_angle",
                "label": "Tool-axis spread",
                "ok": (sample_count or 0) <= 1 or spread_deg <= WARN_AXIS_ANGLE_DEG,
                "detail": detail,
            }
        )

    guidance: list[str] = []
    if not tip_ready or not align_ready:
        guidance.append(
            "Do the tip round, then hold the tool collinear with the spike and capture "
            "an alignment pose before computing the axis."
        )
    elif axis is None:
        guidance.append("Ready — press \"Compute axis\" to fit the tool orientation from the alignment.")
    else:
        if spread_deg is not None and sample_count and sample_count > 1 and spread_deg > WARN_AXIS_ANGLE_DEG:
            guidance.append(
                f"Alignment poses disagree by ±{spread_deg:.1f}° — re-align more carefully "
                "(or add poses) so the averaged axis is tighter."
            )
        if sample_count == 1:
            guidance.append(
                "Only one alignment pose — its accuracy equals your manual alignment. "
                "Capture a few more to average out the eyeballing."
            )

    ok = bool(
        axis is not None
        and tip_ready
        and align_ready
        and (
            spread_deg is None
            or not sample_count
            or sample_count <= 1
            or spread_deg <= WARN_AXIS_ANGLE_DEG
        )
    )
    if ok and not guidance:
        guidance.append("Axis looks good — save to write the calibrated orientation.")

    payload: dict[str, Any] = {
        "computed": axis is not None,
        "ok": ok,
        "checklist": checklist,
        "guidance": guidance,
    }
    if axis is not None:
        payload["alignment_spread_deg"] = spread_deg
        payload["sample_count"] = sample_count
        payload["spike_axis_base"] = axis.get("spike_axis_base")
        payload["axis_dir"] = axis["axis_dir"]
        payload["quaternion"] = {
            "qx": axis["quaternion"][0],
            "qy": axis["quaternion"][1],
            "qz": axis["quaternion"][2],
            "qw": axis["quaternion"][3],
        }
    return payload


def build_tool_tcp_status(
    *,
    mode: str,
    active_round: str,
    tip_samples: list[list[float]],
    tip_pivot: dict[str, Any] | None,
    validation_samples: list[list[float]],
    validation_result: dict[str, Any] | None,
    align_samples: list[list[float]],
    axis: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Top-level status for the GUI. Wraps the tip pivot round and (in axis mode)
    the axis-alignment round, plus the axis combine readiness. ``mode`` is
    "position" (single-point, orientation defaults to the flange) or "axis"
    (tip position + tool axis from the align-to-spike round). The alignment
    round is published under the "axis_ref" key for GUI backward-compat.
    """
    tip = build_pivot_status(flange_samples=tip_samples, pivot=tip_pivot)
    validation = build_validation_status(validation_samples, validation_result)
    align = build_align_status(align_samples)
    axis_status = build_axis_status(
        tip_ready=tip["ready_to_save"],
        align_ready=align["ready_to_save"],
        axis=axis,
    )
    # Base-frame tip location (for the 3D overlay), when the tip fit exists.
    if tip.get("fixed_point") is not None:
        axis_status["tip_fixed_point"] = tip["fixed_point"]

    if mode == "axis":
        ready_to_save = axis_status["ok"]
    else:
        ready_to_save = tip["ready_to_save"]

    return {
        "mode": mode,
        "active_round": active_round,
        "ready_to_save": ready_to_save,
        "tip": tip,
        "validation": validation,
        "axis_ref": align,
        "axis": axis_status,
    }


def status_to_json(status: dict[str, Any]) -> str:
    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: finite(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [finite(item) for item in value]
        return value

    return json.dumps(finite(status), separators=(",", ":"), allow_nan=False)
