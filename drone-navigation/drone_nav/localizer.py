"""Localising a single drone frame on the satellite map."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import Config
from . import features as ft


@dataclass
class FramePose:
    """Result of localising one frame. All coordinates are in original-map pixels."""

    index: int
    success: bool
    center: np.ndarray = field(default_factory=lambda: np.full(2, np.nan))
    corners: np.ndarray = field(default_factory=lambda: np.full((4, 2), np.nan))
    angle_deg: float = float("nan")
    scale: float = float("nan")
    n_matches: int = 0
    n_inliers: int = 0
    matrix: np.ndarray | None = None  # 2x3, original frame px -> original map px
    reason: str = ""  # why localisation failed; empty on success


def similarity_params(matrix: np.ndarray) -> tuple[float, float]:
    """Extract (rotation angle in degrees, uniform scale) from a 2x3 similarity matrix.

    The angle is the rotation of the frame's axes in map image coordinates (y down),
    so a frame whose "up" points to the map's right has angle ~ +90.
    """
    a, b = float(matrix[0, 0]), float(matrix[1, 0])
    return math.degrees(math.atan2(b, a)), math.hypot(a, b)


def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 2x3 affine matrix to an (N, 2) array of points."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return points @ matrix[:, :2].T + matrix[:, 2]


def frame_geometry(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Centre and corners (clockwise from top-left) of a frame in its own pixel coordinates."""
    center = np.array([width / 2.0, height / 2.0])
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float64)
    return center, corners


class MapLocalizer:
    """Computes map features once and localises any number of frames against them."""

    def __init__(self, map_image: np.ndarray, config: Config) -> None:
        self.config = config
        self.map_shape = map_image.shape[:2]
        # Full resolution, unlimited keypoints: capping them leaves map regions featureless.
        map_sift = ft.create_sift(config.map_nfeatures)
        self.map_points, map_desc = ft.detect_and_compute(map_sift, ft.to_gray(map_image))
        if map_desc is None or len(self.map_points) < config.knn_k:
            raise ValueError("Not enough SIFT features found on the map.")
        self.matcher = ft.build_flann_matcher(
            map_desc, config.flann_algorithm_kdtree, config.flann_trees, config.flann_checks
        )
        self.frame_sift = ft.create_sift(config.frame_nfeatures)

    @property
    def n_map_features(self) -> int:
        """Number of SIFT keypoints detected on the map."""
        return len(self.map_points)

    def localize(self, frame: np.ndarray, index: int = 0) -> FramePose:
        """Find where ``frame`` lies on the map. Never raises on bad frames: returns success=False."""
        cfg = self.config
        height, width = frame.shape[:2]

        small = ft.resize(ft.to_gray(frame), cfg.frame_scale)
        frame_points, frame_desc = ft.detect_and_compute(self.frame_sift, small)
        if frame_desc is None or len(frame_points) < cfg.knn_k:
            return FramePose(index, False, reason="no features in frame")

        knn = self.matcher.knnMatch(frame_desc, k=cfg.knn_k)
        good = ft.ratio_test(knn, cfg.ratio_test)
        n_matches = len(good)
        if n_matches < cfg.min_inliers:
            return FramePose(index, False, n_matches=n_matches, reason="too few matches")

        # Back to original-frame pixels so the matrix maps the original frame to the map.
        src = frame_points[[m.queryIdx for m in good]] / cfg.frame_scale
        dst = self.map_points[[m.trainIdx for m in good]]

        # Similarity (shift + rotation + uniform scale): the data has no perspective, and
        # a homography only adds unstable degrees of freedom (see README, "Approach").
        matrix, inlier_mask = cv2.estimateAffinePartial2D(
            src,
            dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=cfg.ransac_reproj_threshold,
            maxIters=cfg.ransac_max_iters,
            confidence=cfg.ransac_confidence,
            refineIters=cfg.ransac_refine_iters,
        )
        if matrix is None:
            return FramePose(index, False, n_matches=n_matches, reason="RANSAC failed")

        n_inliers = int(inlier_mask.sum())
        angle, scale = similarity_params(matrix)
        center, corners = frame_geometry(width, height)
        pose = FramePose(
            index=index,
            success=True,
            center=transform_points(matrix, center)[0],
            corners=transform_points(matrix, corners),
            angle_deg=angle,
            scale=scale,
            n_matches=n_matches,
            n_inliers=n_inliers,
            matrix=matrix,
        )

        if n_inliers < cfg.min_inliers:
            pose.success, pose.reason = False, f"only {n_inliers} inliers"
        elif cfg.expected_scale is not None and (
            abs(scale / cfg.expected_scale - 1.0) > cfg.scale_tolerance
        ):
            pose.success, pose.reason = False, f"implausible scale {scale:.3f}"
        return pose
