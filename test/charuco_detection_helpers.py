"""
Pure OpenCV ChArUco detection helpers for tests and diagnostics.

Does not import ROS — safe to run offline with pytest or the diagnose script.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

# Printed board observed in live OAK-D captures (Calib.io-style, DICT_4X4).
LIVE_BOARD_SPEC = {
    "squares_x": 13,
    "squares_y": 15,
    "square_length_m": 0.015,
    "marker_length_m": 0.011,
    "aruco_dictionary": "DICT_4X4_100",
}

# Defaults wired into charuco_detector / calibration.launch.py (mismatch live board).
DEFAULT_DETECTOR_SPEC = {
    "squares_x": 9,
    "squares_y": 13,
    "square_length_m": 0.015,
    "marker_length_m": 0.011,
    "aruco_dictionary": "DICT_4X4_100",
}

COMMON_DICTIONARIES = (
    "DICT_4X4_50",
    "DICT_4X4_100",
    "DICT_4X4_250",
    "DICT_4X4_1000",
    "DICT_5X5_100",
    "DICT_5X5_250",
    "DICT_6X6_250",
    "DICT_7X7_100",
)


@dataclass(frozen=True)
class BoardSpec:
    squares_x: int
    squares_y: int
    square_length_m: float
    marker_length_m: float
    aruco_dictionary: str

    @classmethod
    def from_dict(cls, data: dict) -> "BoardSpec":
        return cls(
            squares_x=int(data["squares_x"]),
            squares_y=int(data["squares_y"]),
            square_length_m=float(data["square_length_m"]),
            marker_length_m=float(data["marker_length_m"]),
            aruco_dictionary=str(data["aruco_dictionary"]),
        )

    def as_dict(self) -> dict:
        return {
            "squares_x": self.squares_x,
            "squares_y": self.squares_y,
            "square_length_m": self.square_length_m,
            "marker_length_m": self.marker_length_m,
            "aruco_dictionary": self.aruco_dictionary,
        }


@dataclass(frozen=True)
class DetectionResult:
    spec: BoardSpec
    n_markers: int
    n_corners: int
    marker_ids: tuple[int, ...]
    pose_ok: bool
    reproj_error_px: float | None

    @property
    def passes_min_corners(self) -> bool:
        return self.n_corners >= 8


def lookup_aruco_dictionary(name: str):
    normalized = str(name or "").strip()
    if normalized.isdigit():
        return cv2.aruco.getPredefinedDictionary(int(normalized))
    if not normalized.startswith("DICT_"):
        normalized = f"DICT_{normalized}"
    if not hasattr(cv2.aruco, normalized):
        raise ValueError(f"Unknown ArUco dictionary: {name!r}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, normalized))


def build_charuco_board(spec: BoardSpec) -> cv2.aruco.CharucoBoard:
    dictionary = lookup_aruco_dictionary(spec.aruco_dictionary)
    return cv2.aruco.CharucoBoard(
        (spec.squares_x, spec.squares_y),
        spec.square_length_m,
        spec.marker_length_m,
        dictionary,
    )


def load_gray_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def count_raw_markers(gray: np.ndarray, dictionary_name: str) -> int:
    dictionary = lookup_aruco_dictionary(dictionary_name)
    _corners, ids, _ = cv2.aruco.ArucoDetector(dictionary).detectMarkers(gray)
    return 0 if ids is None else len(ids)


def detect_charuco(
    gray: np.ndarray,
    spec: BoardSpec,
    *,
    camera_matrix: np.ndarray | None = None,
    dist_coeffs: np.ndarray | None = None,
    min_corners_for_pose: int = 8,
) -> DetectionResult:
    board = build_charuco_board(spec)
    detector = cv2.aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)

    n_markers = 0 if marker_ids is None else len(marker_ids)
    n_corners = 0 if charuco_ids is None else len(charuco_ids)
    ids_tuple = tuple(
        sorted(int(i) for i in marker_ids.reshape(-1))
    ) if marker_ids is not None else ()

    pose_ok = False
    reproj = None
    if n_corners >= min_corners_for_pose and camera_matrix is not None:
        try:
            obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
        except AttributeError:
            obj_points, img_points = None, None
        if obj_points is not None and len(obj_points) >= 4:
            dist = dist_coeffs if dist_coeffs is not None else np.zeros((5, 1))
            ok, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist)
            if ok:
                proj, _ = cv2.projectPoints(obj_points, rvec, tvec, camera_matrix, dist)
                reproj = float(
                    np.mean(
                        np.linalg.norm(
                            proj.reshape(-1, 2) - img_points.reshape(-1, 2),
                            axis=1,
                        )
                    )
                )
                pose_ok = True

    return DetectionResult(
        spec=spec,
        n_markers=n_markers,
        n_corners=n_corners,
        marker_ids=ids_tuple,
        pose_ok=pose_ok,
        reproj_error_px=reproj,
    )


def render_synthetic_board(
    spec: BoardSpec,
    *,
    pixels_per_square: int = 100,
    border_px: int = 20,
) -> np.ndarray:
    board = build_charuco_board(spec)
    w = spec.squares_x * pixels_per_square + 2 * border_px
    h = spec.squares_y * pixels_per_square + 2 * border_px
    img = board.generateImage((w, h))
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def sweep_board_specs(
    gray: np.ndarray,
    *,
    squares_x_range: Iterable[int] = range(8, 16),
    squares_y_range: Iterable[int] = range(8, 16),
    square_lengths_mm: Iterable[float] = (15.0,),
    marker_lengths_mm: Iterable[float] = (11.0,),
    dictionaries: Iterable[str] = ("DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250"),
    min_corners: int = 8,
) -> list[DetectionResult]:
    hits: list[DetectionResult] = []
    for sx in squares_x_range:
        for sy in squares_y_range:
            for sq_mm in square_lengths_mm:
                for mk_mm in marker_lengths_mm:
                    if mk_mm >= sq_mm:
                        continue
                    for dn in dictionaries:
                        spec = BoardSpec(sx, sy, sq_mm / 1000.0, mk_mm / 1000.0, dn)
                        result = detect_charuco(gray, spec)
                        if result.n_corners >= min_corners:
                            hits.append(result)
    hits.sort(
        key=lambda r: (
            r.n_corners,
            r.n_markers,
            r.spec.squares_x * r.spec.squares_y,
        ),
        reverse=True,
    )
    return hits


def best_matching_spec(
    gray: np.ndarray,
    *,
    min_corners: int = 8,
) -> DetectionResult | None:
    hits = sweep_board_specs(gray, min_corners=min_corners)
    return hits[0] if hits else None
