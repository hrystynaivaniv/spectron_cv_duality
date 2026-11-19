"""Frame listing: numeric order and clear errors."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from drone_nav.config import Config
from drone_nav.io_utils import list_frames, read_image

EXTS = Config().frame_extensions


def _touch(folder: Path, names: list[str]) -> None:
    for name in names:
        (folder / name).write_bytes(b"")


def test_numeric_not_alphabetical_order(tmp_path: Path) -> None:
    _touch(tmp_path, [f"crop_{i}.png" for i in (10, 2, 0, 1, 49, 11, 20, 3)])
    names = [p.name for p in list_frames(tmp_path, EXTS)]
    assert names == [f"crop_{i}.png" for i in (0, 1, 2, 3, 10, 11, 20, 49)]


def test_ignores_non_images_and_unnumbered_go_last(tmp_path: Path) -> None:
    _touch(tmp_path, ["crop_2.png", "notes.txt", "crop_1.JPG", "extra.png", ".gitkeep"])
    names = [p.name for p in list_frames(tmp_path, EXTS)]
    assert names == ["crop_1.JPG", "crop_2.png", "extra.png"]


def test_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Frames directory not found"):
        list_frames(tmp_path / "nope", EXTS)


def test_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No frame images"):
        list_frames(tmp_path, EXTS)


def test_read_image_roundtrip_and_errors(tmp_path: Path) -> None:
    img = np.random.default_rng(0).integers(0, 256, (20, 30, 3), dtype=np.uint8)
    path = tmp_path / "карта.png"  # non-ASCII path must work on Windows too
    ok, buf = cv2.imencode(".png", img)
    assert ok
    buf.tofile(str(path))
    np.testing.assert_array_equal(read_image(path), img)

    with pytest.raises(FileNotFoundError):
        read_image(tmp_path / "missing.png")
    (tmp_path / "broken.png").write_bytes(b"not an image")
    with pytest.raises(ValueError, match="Cannot decode"):
        read_image(tmp_path / "broken.png")
