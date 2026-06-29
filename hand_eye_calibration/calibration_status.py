"""
Build JSON calibration readiness status for the GUI coach panel.
"""

from __future__ import annotations

import json
from typing import Any

MIN_SAMPLES = 4
TARGET_SAMPLES = 15
MIN_TRANSLATION_SPAN_M = 0.05
MIN_ROTATION_SPAN_DEG = 25.0
TARGET_SAMPLES_IDEAL = 10

MAX_MEAN_RESIDUAL_TRANS_M = 0.003
MAX_MEAN_RESIDUAL_ROT_DEG = 1.0
WARN_MEAN_RESIDUAL_TRANS_M = 0.005
WARN_MEAN_RESIDUAL_ROT_DEG = 2.0

DUPLICATE_TRANS_M = 0.015
DUPLICATE_ROT_DEG = 5.0


def _sample_warnings(metrics: dict[str, Any] | None) -> list[str]:
    if not metrics:
        return []
    warnings: list[str] = []
    dt = metrics.get("robot_delta_translation_m")
    dr = metrics.get("robot_delta_rotation_deg")
    if dt is not None and dr is not None:
        if dt < DUPLICATE_TRANS_M and dr < DUPLICATE_ROT_DEG:
            warnings.append(
                "Last sample is very close to the previous pose — rotate or move the arm more."
            )
    angle = metrics.get("marker_view_angle_deg")
    if angle is not None and angle > 70.0:
        warnings.append("Board viewed at a steep angle — tilt the arm or board for a clearer view.")
    dist = metrics.get("marker_distance_m")
    if dist is not None and (dist < 0.15 or dist > 1.5):
        warnings.append("Board distance is outside the usual 0.15–1.5 m range.")
    return warnings


def _checklist(
    sample_count: int,
    diversity: dict[str, Any],
    residuals: dict[str, Any] | None,
    last_warnings: list[str],
) -> list[dict[str, Any]]:
    span = diversity.get("translation_span_m") or [0.0, 0.0, 0.0]
    max_span = max(float(v) for v in span) if span else 0.0
    max_rot = float(diversity.get("max_rotation_from_first_deg") or 0.0)

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
            "id": "rotation_spread",
            "label": "Wrist rotation coverage",
            "ok": sample_count < 2 or max_rot >= MIN_ROTATION_SPAN_DEG,
            "detail": f"{max_rot:.1f}° / {MIN_ROTATION_SPAN_DEG:.0f}°",
        },
        {
            "id": "translation_spread",
            "label": "Arm translation spread",
            "ok": sample_count < 2 or max_span >= MIN_TRANSLATION_SPAN_M,
            "detail": f"{max_span * 100:.1f} cm / {MIN_TRANSLATION_SPAN_M * 100:.0f} cm",
        },
    ]

    if residuals:
        mt = float(residuals.get("mean_translation_m", 999.0))
        mr = float(residuals.get("mean_rotation_deg", 999.0))
        items.append(
            {
                "id": "residual_translation",
                "label": "Mean calibration error (translation)",
                "ok": mt <= WARN_MEAN_RESIDUAL_TRANS_M,
                "detail": f"{mt * 1000:.1f} mm (good < {MAX_MEAN_RESIDUAL_TRANS_M * 1000:.0f} mm)",
            }
        )
        items.append(
            {
                "id": "residual_rotation",
                "label": "Mean calibration error (rotation)",
                "ok": mr <= WARN_MEAN_RESIDUAL_ROT_DEG,
                "detail": f"{mr:.2f}° (good < {MAX_MEAN_RESIDUAL_ROT_DEG:.1f}°)",
            }
        )

    if last_warnings:
        items.append(
            {
                "id": "last_sample",
                "label": "Last sample quality",
                "ok": False,
                "detail": last_warnings[0],
            }
        )
    elif sample_count > 0:
        items.append(
            {
                "id": "last_sample",
                "label": "Last sample quality",
                "ok": True,
                "detail": "Good sample",
            }
        )

    return items


def _readiness(
    sample_count: int,
    diversity: dict[str, Any],
    residuals: dict[str, Any] | None,
    last_warnings: list[str],
) -> str:
    if sample_count < MIN_SAMPLES:
        return "not_ready"

    span = diversity.get("translation_span_m") or [0.0, 0.0, 0.0]
    max_span = max(float(v) for v in span) if span else 0.0
    max_rot = float(diversity.get("max_rotation_from_first_deg") or 0.0)
    guidance = diversity.get("guidance") or []

    diversity_ok = (
        sample_count < 2
        or (max_span >= MIN_TRANSLATION_SPAN_M and max_rot >= MIN_ROTATION_SPAN_DEG)
    )
    residuals_ok = True
    if residuals:
        mt = float(residuals.get("mean_translation_m", 999.0))
        mr = float(residuals.get("mean_rotation_deg", 999.0))
        residuals_ok = mt <= WARN_MEAN_RESIDUAL_TRANS_M and mr <= WARN_MEAN_RESIDUAL_ROT_DEG

    has_blocking = bool(last_warnings) or any(
        g
        for g in guidance
        if "Need more" in g or "More samples recommended" in g
    )

    excellent = (
        sample_count >= TARGET_SAMPLES
        and diversity_ok
        and residuals_ok
        and not last_warnings
        and (not residuals or (
            float(residuals.get("mean_translation_m", 1.0)) <= MAX_MEAN_RESIDUAL_TRANS_M
            and float(residuals.get("mean_rotation_deg", 10.0)) <= MAX_MEAN_RESIDUAL_ROT_DEG
        ))
    )
    if excellent:
        return "excellent"

    ready = (
        sample_count >= MIN_SAMPLES
        and diversity_ok
        and residuals_ok
        and not last_warnings
    )
    if ready and sample_count >= TARGET_SAMPLES_IDEAL:
        return "ready_to_save"

    if sample_count >= MIN_SAMPLES and not has_blocking and residuals_ok:
        return "ready_to_save"

    return "collecting"


def build_calibration_status(
    *,
    sample_count: int,
    diversity: dict[str, Any],
    residuals: dict[str, Any] | None,
    last_sample_metrics: dict[str, Any] | None,
    estimate: list[float] | None = None,
) -> dict[str, Any]:
    last_warnings = _sample_warnings(last_sample_metrics)
    readiness = _readiness(sample_count, diversity, residuals, last_warnings)
    guidance = list(diversity.get("guidance") or [])
    guidance.extend(last_warnings)

    if readiness == "excellent":
        summary = "Excellent — no need to collect more samples. Save when ready."
    elif readiness == "ready_to_save":
        summary = "Ready to save — quality is sufficient."
    elif sample_count < MIN_SAMPLES:
        summary = f"Need at least {MIN_SAMPLES} samples before calibration is valid."
    else:
        summary = "Keep collecting — improve pose diversity before saving."

    checklist = _checklist(sample_count, diversity, residuals, last_warnings)
    ready_to_save = readiness in ("ready_to_save", "excellent")

    payload: dict[str, Any] = {
        "sample_count": sample_count,
        "min_samples": MIN_SAMPLES,
        "target_samples": TARGET_SAMPLES,
        "readiness": readiness,
        "ready_to_save": ready_to_save,
        "summary": summary,
        "guidance": guidance,
        "checklist": checklist,
        "diversity": diversity,
        "residuals": residuals,
        "last_sample_warning": last_warnings[0] if last_warnings else None,
    }
    if estimate is not None and len(estimate) >= 7:
        payload["estimate"] = {
            "tx": estimate[0],
            "ty": estimate[1],
            "tz": estimate[2],
            "qx": estimate[3],
            "qy": estimate[4],
            "qz": estimate[5],
            "qw": estimate[6],
        }
    return payload


def status_to_json(status: dict[str, Any]) -> str:
    return json.dumps(status, separators=(",", ":"))
