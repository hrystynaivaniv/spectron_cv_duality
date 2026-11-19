"""Synthetic map and frame generators for tests (no real data needed)."""

from __future__ import annotations

import math

import cv2
import numpy as np


def make_textured_map(height: int, width: int, seed: int = 0) -> np.ndarray:
    """Random BGR 'map': smooth colour blobs + many random shapes, rich in SIFT corners."""
    rng = np.random.default_rng(seed)
    coarse = rng.integers(0, 256, size=(max(height // 40, 2), max(width // 40, 2), 3), dtype=np.uint8)
    image = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)

    n_shapes = height * width // 900
    for _ in range(n_shapes):
        color = tuple(int(c) for c in rng.integers(0, 256, size=3))
        x, y = int(rng.integers(0, width)), int(rng.integers(0, height))
        size = int(rng.integers(3, 25))
        kind = rng.integers(0, 3)
        if kind == 0:
            cv2.circle(image, (x, y), size, color, -1)
        elif kind == 1:
            cv2.rectangle(image, (x, y), (x + size, y + int(rng.integers(3, 25))), color, -1)
        else:
            x2, y2 = x + int(rng.integers(-40, 40)), y + int(rng.integers(-40, 40))
            cv2.line(image, (x, y), (x2, y2), color, int(rng.integers(1, 4)))

    noise = rng.normal(0, 6, size=image.shape)
    image = np.clip(image.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(image, (3, 3), 0)


def frame_to_map_matrix(
    center: tuple[float, float], angle_deg: float, scale: float, size: int
) -> np.ndarray:
    """2x3 similarity mapping frame pixels to map pixels, with the frame centre on ``center``."""
    t = math.radians(angle_deg)
    a = scale * np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    offset = np.asarray(center, dtype=np.float64) - a @ np.array([size / 2.0, size / 2.0])
    return np.hstack([a, offset[:, None]])


def render_frame(
    map_image: np.ndarray, center: tuple[float, float], angle_deg: float, scale: float, size: int
) -> np.ndarray:
    """Cut a rotated, zoomed square 'drone frame' out of the map with known parameters."""
    matrix = frame_to_map_matrix(center, angle_deg, scale, size)
    # WARP_INVERSE_MAP: matrix maps destination (frame) pixels to source (map) pixels.
    return cv2.warpAffine(
        map_image,
        matrix,
        (size, size),
        flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT,
    )
