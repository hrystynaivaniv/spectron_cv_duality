"""Turning per-frame poses into a continuous trajectory."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.signal import savgol_filter

from .localizer import FramePose


def centers_array(poses: Sequence[FramePose]) -> np.ndarray:
    """(N, 2) array of frame centres; rows of failed frames are NaN."""
    out = np.full((len(poses), 2), np.nan)
    for i, pose in enumerate(poses):
        if pose.success:
            out[i] = pose.center
    return out


def interpolate_missing(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fill NaN rows by linear interpolation over the frame index.

    Gaps at the ends are filled with the nearest known position (no extrapolation,
    which could fly off the map). Returns (filled points, boolean mask of filled rows).
    """
    points = np.asarray(points, dtype=np.float64)
    missing = np.isnan(points).any(axis=1)
    if missing.all():
        raise ValueError("No frame was localised; cannot build a trajectory.")
    filled = points.copy()
    if missing.any():
        idx = np.arange(len(points))
        known = ~missing
        for axis in range(points.shape[1]):
            filled[missing, axis] = np.interp(idx[missing], idx[known], points[known, axis])
    return filled, missing


def smooth_trajectory(points: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    """Savitzky-Golay smoothing along the track.

    Chosen over a moving average because it preserves the curvature of turns.
    The window is shrunk for short tracks; too-short tracks are returned unchanged.
    """
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    window = min(window, n if n % 2 == 1 else n - 1)
    if window <= polyorder:
        return points.copy()
    return savgol_filter(points, window_length=window, polyorder=polyorder, axis=0, mode="interp")
