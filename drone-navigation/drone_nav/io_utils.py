"""Reading the map and listing drone frames in the correct order."""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

_NUMBER_RE = re.compile(r"\d+")


def read_image(path: str | Path) -> np.ndarray:
    """Read a colour image (BGR). Raises FileNotFoundError / ValueError with a clear message."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    # imdecode + fromfile instead of imread: works with non-ASCII paths on Windows.
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def frame_sort_key(path: Path) -> tuple[int, int, str]:
    """Sort key using the last number in the file name (crop_2 before crop_10).

    Files without a number go after numbered ones, in alphabetical order.
    """
    numbers = _NUMBER_RE.findall(path.stem)
    if numbers:
        return (0, int(numbers[-1]), path.name)
    return (1, 0, path.name)


def list_frames(frames_dir: str | Path, extensions: tuple[str, ...]) -> list[Path]:
    """Return image files in ``frames_dir`` sorted numerically by the number in their name."""
    frames_dir = Path(frames_dir)
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
    exts = {e.lower() for e in extensions}
    frames = [p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    if not frames:
        raise FileNotFoundError(
            f"No frame images ({', '.join(sorted(exts))}) found in: {frames_dir}"
        )
    return sorted(frames, key=frame_sort_key)
