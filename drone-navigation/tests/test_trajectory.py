"""Trajectory: NaN interpolation and smoothing."""

from __future__ import annotations

import numpy as np
import pytest

from drone_nav.localizer import FramePose
from drone_nav.trajectory import centers_array, interpolate_missing, smooth_trajectory

NAN = np.nan


def test_interpolates_interior_gap_linearly() -> None:
    pts = np.array([[0, 0], [NAN, NAN], [NAN, NAN], [30, 60], [40, 80]], float)
    filled, mask = interpolate_missing(pts)
    np.testing.assert_allclose(filled, [[0, 0], [10, 20], [20, 40], [30, 60], [40, 80]])
    assert mask.tolist() == [False, True, True, False, False]


def test_end_gaps_use_nearest_known_point() -> None:
    pts = np.array([[NAN, NAN], [5, 5], [7, 9], [NAN, NAN]], float)
    filled, mask = interpolate_missing(pts)
    np.testing.assert_allclose(filled, [[5, 5], [5, 5], [7, 9], [7, 9]])
    assert mask.tolist() == [True, False, False, True]


def test_no_gaps_is_identity() -> None:
    pts = np.arange(10, dtype=float).reshape(5, 2)
    filled, mask = interpolate_missing(pts)
    np.testing.assert_array_equal(filled, pts)
    assert not mask.any()


def test_all_missing_raises() -> None:
    with pytest.raises(ValueError, match="No frame was localised"):
        interpolate_missing(np.full((3, 2), NAN))


def test_centers_array_marks_failed_frames() -> None:
    poses = [
        FramePose(0, True, center=np.array([1.0, 2.0])),
        FramePose(1, False),
        FramePose(2, True, center=np.array([3.0, 4.0])),
    ]
    c = centers_array(poses)
    np.testing.assert_array_equal(c[[0, 2]], [[1, 2], [3, 4]])
    assert np.isnan(c[1]).all()


def test_smoothing_keeps_quadratic_and_reduces_noise() -> None:
    t = np.arange(40, dtype=float)
    clean = np.stack([63.4 * t, 0.5 * (t - 20) ** 2], axis=1)
    # A quadratic is reproduced exactly by a polyorder-2 Savitzky-Golay filter.
    np.testing.assert_allclose(smooth_trajectory(clean, 5, 2), clean, atol=1e-6)

    noisy = clean + np.random.default_rng(0).normal(0, 2.0, clean.shape)
    smoothed = smooth_trajectory(noisy, 5, 2)
    assert np.abs(smoothed - clean).mean() < np.abs(noisy - clean).mean()


def test_smoothing_short_tracks() -> None:
    three = np.array([[0, 0], [1, 5], [2, 0]], float)
    out = smooth_trajectory(three, 5, 2)  # window shrinks to 3 -> exact fit of 3 points
    np.testing.assert_allclose(out, three, atol=1e-9)
    two = np.array([[0, 0], [1, 1]], float)
    np.testing.assert_array_equal(smooth_trajectory(two, 5, 2), two)
