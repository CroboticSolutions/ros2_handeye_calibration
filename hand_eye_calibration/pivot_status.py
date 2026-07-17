"""
Build JSON pivot (tool-TCP) calibration readiness status for the GUI coach panel.
Mirrors calibration_status.py's readiness/checklist shape so the frontend can
reuse the same panel component.
"""

from __future__ import annotations

import json
import math
from typing import Any

MIN_SAMPLES = 4
TARGET_SAMPLES = 10
MIN_ROTATION_SPAN_DEG = 25.0

# "Good" thresholds gate "excellent"; "warn" thresholds gate "ready_to_save".
GOOD_RMS_RESIDUAL_M = 0.001
WARN_RMS_RESIDUAL_M = 0.003
GOOD_MAX_RESIDUAL_M = 0.002
WARN_MAX_RESIDUAL_M = 0.005

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


def _checklist(
    sample_count: int,
    orientation_span_deg: float,
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
            "label": "Wrist orientation coverage",
            "ok": sample_count < 2 or orientation_span_deg >= MIN_ROTATION_SPAN_DEG,
            "detail": f"{orientation_span_deg:.1f}° / {MIN_ROTATION_SPAN_DEG:.0f}°",
        },
    ]

    if pivot is not None:
        rms = pivot["rms_residual_m"]
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
    pivot: dict[str, Any] | None,
) -> str:
    if sample_count < MIN_SAMPLES:
        return "not_ready"

    orientation_ok = sample_count < 2 or orientation_span_deg >= MIN_ROTATION_SPAN_DEG
    if pivot is None:
        return "collecting"

    rms = pivot["rms_residual_m"]
    max_res = pivot["max_residual_m"]
    condition_ok = pivot["condition_number"] <= CONDITION_NUMBER_WARN
    residuals_ok = rms <= WARN_RMS_RESIDUAL_M and max_res <= WARN_MAX_RESIDUAL_M

    excellent = (
        sample_count >= TARGET_SAMPLES
        and orientation_ok
        and condition_ok
        and rms <= GOOD_RMS_RESIDUAL_M
        and max_res <= GOOD_MAX_RESIDUAL_M
    )
    if excellent:
        return "excellent"

    ready = orientation_ok and condition_ok and residuals_ok
    if ready:
        return "ready_to_save"

    return "collecting"


def _xyz(v: list[float] | tuple[float, ...] | Any) -> dict[str, float]:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def _sample_viz(
    flange_samples: list[list[float]],
    residuals: list[float] | None,
) -> list[dict[str, Any]]:
    """Flange poses in the robot base frame for the GUI 3D overlay."""
    out: list[dict[str, Any]] = []
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
    residuals = (
        pivot["per_sample_residuals_m"]
        if pivot is not None and pivot["per_sample_residuals_m"]
        else None
    )
    last_residual_m = residuals[-1] if residuals else None

    readiness = _readiness(sample_count, orientation_span_deg, pivot)
    ready_to_save = readiness in ("ready_to_save", "excellent")

    guidance: list[str] = []
    if sample_count < MIN_SAMPLES:
        guidance.append(f"Need at least {MIN_SAMPLES} touches before a fit is possible.")
    elif orientation_span_deg < MIN_ROTATION_SPAN_DEG:
        guidance.append(
            "Need more wrist orientation variation between touches (rotate the wrist more between samples)."
        )
    if pivot is not None and pivot["condition_number"] > CONDITION_NUMBER_WARN:
        guidance.append("Orientations are too similar to separate translation from the pivot point reliably.")
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
        "checklist": _checklist(sample_count, orientation_span_deg, pivot, last_residual_m),
        "orientation_span_deg": orientation_span_deg,
        "last_sample_residual_m": last_residual_m,
        # Always present so the GUI 3D scene can draw flange poses as they arrive.
        "samples": _sample_viz(flange_samples, residuals),
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
    return payload


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
        "axis_ref": align,
        "axis": axis_status,
    }


def status_to_json(status: dict[str, Any]) -> str:
    return json.dumps(status, separators=(",", ":"))
