"""Localizer recovers known shift / rotation / scale from synthetic frames."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from drone_nav.config import Config
from drone_nav.localizer import MapLocalizer, similarity_params
from synthetic import frame_to_map_matrix, make_textured_map, render_frame

FRAME_SIZE = 640
# Same geometry as the real data: a 640 px frame covers ~162 map pixels.
SCALE = 0.253


@pytest.fixture(scope="module")
def synth_map() -> np.ndarray:
    return make_textured_map(800, 1200, seed=3)


@pytest.fixture(scope="module")
def localizer(synth_map: np.ndarray) -> MapLocalizer:
    return MapLocalizer(synth_map, Config())


@pytest.mark.parametrize(
    "center, angle",
    [((300.0, 300.0), 45.2), ((600.0, 420.0), 134.9), ((900.0, 500.0), -30.0), ((450.0, 550.0), 180.0)],
)
def test_recovers_known_pose(localizer: MapLocalizer, synth_map: np.ndarray, center, angle) -> None:
    frame = render_frame(synth_map, center, angle, SCALE, FRAME_SIZE)
    pose = localizer.localize(frame, index=7)

    assert pose.success, pose.reason
    assert pose.index == 7
    np.testing.assert_allclose(pose.center, center, atol=1.5)
    assert abs((pose.angle_deg - angle + 180) % 360 - 180) < 1.0
    assert pose.scale == pytest.approx(SCALE, rel=0.02)
    assert pose.n_inliers >= Config().min_inliers
    assert pose.n_inliers <= pose.n_matches

    expected = frame_to_map_matrix(center, angle, SCALE, FRAME_SIZE)
    corners = np.array([[0, 0], [FRAME_SIZE, 0], [FRAME_SIZE, FRAME_SIZE], [0, FRAME_SIZE]], float)
    expected_corners = corners @ expected[:, :2].T + expected[:, 2]
    np.testing.assert_allclose(pose.corners, expected_corners, atol=2.5)


def test_unrelated_frame_fails(localizer: MapLocalizer) -> None:
    """A frame from a different 'world' must be reported as failed, not mislocalised."""
    other = make_textured_map(800, 1200, seed=99)
    frame = render_frame(other, (500.0, 400.0), 10.0, SCALE, FRAME_SIZE)
    pose = localizer.localize(frame)
    assert not pose.success
    assert pose.reason


def test_blank_frame_fails(localizer: MapLocalizer) -> None:
    pose = localizer.localize(np.full((FRAME_SIZE, FRAME_SIZE, 3), 128, np.uint8))
    assert not pose.success
    assert pose.reason == "no features in frame"


def test_scale_check_rejects_unexpected_scale(synth_map: np.ndarray) -> None:
    """With an expected scale far from the true one, an otherwise good match is rejected."""
    loc = MapLocalizer(synth_map, replace(Config(), expected_scale=0.5))
    pose = loc.localize(render_frame(synth_map, (600.0, 400.0), 20.0, SCALE, FRAME_SIZE))
    assert not pose.success
    assert "scale" in pose.reason


def test_similarity_params() -> None:
    m = frame_to_map_matrix((10.0, 20.0), 30.0, 0.5, 100)
    angle, scale = similarity_params(m)
    assert angle == pytest.approx(30.0)
    assert scale == pytest.approx(0.5)
