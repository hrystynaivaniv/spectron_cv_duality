"""Run the full pipeline (main.py) on every example and save one route picture per example.

Examples:
  * task      – the original task data (data/global_map.png + data/crops);
  * generated – 5 random flights x 3 degradation levels (clean / realistic / harsh) cut out of the
                same map with known ground truth (see synthetic_flights.py).

For each example the output folder holds route.png, trajectory.csv (and flyover.mp4 with --video).
For generated flights the error against the ground truth is printed and written to summary.csv.

Example:
    python experiments/run_all.py --out outputs/examples
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthetic_flights import LEVELS, N_FRAMES, ROOT, degrade, make_flight, render  # noqa: E402

from drone_nav.io_utils import read_image  # noqa: E402


def run_main(map_path: Path, frames: Path, out: Path, video: bool) -> None:
    """Run the real CLI exactly as a user would (without the preview window)."""
    cmd = [sys.executable, str(ROOT / "main.py"), "--map", str(map_path), "--frames", str(frames),
           "--out", str(out), "--no-show"] + (["--video"] if video else [])
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def read_track(out: Path) -> list[dict]:
    with (out / "trajectory.csv").open() as fh:
        return list(csv.DictReader(fh))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--map", type=Path, default=ROOT / "data/global_map.png")
    p.add_argument("--frames", type=Path, default=ROOT / "data/crops", help="original task frames")
    p.add_argument("--out", type=Path, default=ROOT / "outputs/examples")
    p.add_argument("--flights", type=int, default=5, help="generated flights per degradation level")
    p.add_argument("--video", action="store_true", help="also render flyover.mp4 for every example")
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    map_img = read_image(args.map)

    summary = []

    def record(name: str, rows: list[dict], gt: np.ndarray | None) -> None:
        n_ok = sum(r["status"] == "ok" for r in rows)
        s = {"example": name, "frames": len(rows), "localised": n_ok}
        if gt is not None:
            pos = np.array([[float(r["x_raw"]), float(r["y_raw"])] for r in rows])
            ok = np.array([r["status"] == "ok" for r in rows])
            err = np.linalg.norm(pos[ok] - gt[ok], axis=1)
            s["centre_err_median_px"] = round(float(np.median(err)), 2)
            s["centre_err_max_px"] = round(float(err.max()), 2)
        summary.append(s)
        print(s)

    # 1. The original task data.
    run_main(args.map, args.frames, args.out / "task", args.video)
    record("task", read_track(args.out / "task"), None)

    # 2. Generated flights with ground truth.
    for li, lv in enumerate(LEVELS):
        for f in range(args.flights):
            name = f"{lv.name}_flight{f}"
            rng = np.random.default_rng(1000 + f)  # same routes for every level
            centres, heading, scale = make_flight(map_img.shape[:2], rng)
            noise_rng = np.random.default_rng(100 * li + f)
            with tempfile.TemporaryDirectory() as tmp:
                for i in range(N_FRAMES):
                    frame = degrade(render(map_img, centres[i], heading[i], scale[i]), lv, noise_rng)
                    cv2.imwrite(str(Path(tmp) / f"crop_{i}.png"), frame)
                run_main(args.map, Path(tmp), args.out / name, args.video)
            record(name, read_track(args.out / name), centres)

    with (args.out / "summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["example", "frames", "localised",
                                           "centre_err_median_px", "centre_err_max_px"], restval="")
        w.writeheader()
        w.writerows(summary)
    print(f"\n{len(summary)} examples, pictures: {args.out}/<example>/route.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
