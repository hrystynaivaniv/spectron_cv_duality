"""End-to-end run of main.py on a small synthetic dataset."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

import main
from synthetic import make_textured_map, render_frame


def test_cli_end_to_end(tmp_path: Path) -> None:
    world = make_textured_map(700, 1100, seed=5)
    cv2.imwrite(str(tmp_path / "map.png"), world)
    frames = tmp_path / "crops"
    frames.mkdir()
    truth = [(250.0 + 120 * i, 350.0 + 40 * np.sin(i), 30.0 + 10 * i) for i in range(6)]
    for i, (x, y, a) in enumerate(truth):
        img = render_frame(world, (x, y), a, 0.253, 640)
        if i == 3:  # an unusable frame in the middle must be interpolated
            img = np.full_like(img, 128)
        cv2.imwrite(str(frames / f"crop_{i}.png"), img)

    out = tmp_path / "out"
    assert main.main(["--map", str(tmp_path / "map.png"), "--frames", str(frames),
                      "--out", str(out), "--no-smooth"]) == 0

    rows = list(csv.DictReader((out / "trajectory.csv").open()))
    assert [r["status"] for r in rows] == ["ok", "ok", "ok", "interpolated", "ok", "ok"]
    for r, (x, y, _) in zip(rows, truth):
        if r["status"] == "ok":
            assert abs(float(r["x"]) - x) < 1.5 and abs(float(r["y"]) - y) < 1.5
    route = cv2.imread(str(out / "route.png"))
    assert route is not None and route.shape == world.shape


def test_cli_reports_missing_inputs(tmp_path: Path, capsys) -> None:
    assert main.main(["--map", str(tmp_path / "none.png"), "--frames", str(tmp_path)]) == 1
    assert "Image not found" in capsys.readouterr().err
