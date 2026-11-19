"""SIFT feature extraction and FLANN matching helpers."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np


def to_gray(image: np.ndarray) -> np.ndarray:
    """Convert a BGR (or already grayscale) image to single-channel uint8."""
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def resize(image: np.ndarray, scale: float) -> np.ndarray:
    """Resize by a uniform factor; INTER_AREA avoids aliasing when shrinking."""
    if scale == 1.0:
        return image
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=interpolation)


def create_sift(nfeatures: int) -> cv2.SIFT:
    """Create a SIFT detector; ``nfeatures=0`` keeps all keypoints."""
    return cv2.SIFT_create(nfeatures=nfeatures)


def detect_and_compute(
    sift: cv2.SIFT, gray: np.ndarray
) -> tuple[np.ndarray, np.ndarray | None]:
    """Detect keypoints and compute descriptors.

    Returns keypoint coordinates as an (N, 2) float32 array (cheaper to keep than
    cv2.KeyPoint objects for 150k map points) and float32 descriptors or None.
    """
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    points = np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 2)
    if descriptors is not None:
        descriptors = descriptors.astype(np.float32, copy=False)
    return points, descriptors


def build_flann_matcher(
    train_descriptors: np.ndarray, algorithm: int, trees: int, checks: int
) -> cv2.FlannBasedMatcher:
    """Build a FLANN KD-tree index over the (map) descriptors once, for reuse across frames."""
    matcher = cv2.FlannBasedMatcher(
        dict(algorithm=algorithm, trees=trees), dict(checks=checks)
    )
    matcher.add([train_descriptors])
    matcher.train()
    return matcher


def ratio_test(
    knn_matches: Sequence[Sequence[cv2.DMatch]], ratio: float
) -> list[cv2.DMatch]:
    """Lowe's ratio test: keep a match only if it is clearly better than the runner-up."""
    good = []
    for candidates in knn_matches:
        if len(candidates) < 2:
            continue
        best, second = candidates[0], candidates[1]
        if best.distance < ratio * second.distance:
            good.append(best)
    return good
