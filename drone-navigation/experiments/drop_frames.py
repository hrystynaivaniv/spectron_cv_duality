"""Does the route survive missing photos? Remove frames from the task data and measure.

Reference = positions of all 50 frames from the full run (every frame passed the
checks with >= 104 inliers). Then frames are removed in three ways:

1. random   – drop 10 … 60 % of frames at random (first and last kept), 50 trials each;
2. gap      – drop a contiguous run of 1 … 15 frames at every possible position;
3. pipeline – end-to-end through main.py: removed frames replaced by unusable images
              (black / pure noise / an unrelated picture) so the localiser must reject them
              and interpolation must fill them in.

Every frame is localised independently, so removing frames never changes the
positions of the others (no drift). The only error is at the missing frames, where the
position is interpolated over the frame number. We compare linear interpolation (what
the pipeline uses) with a cubic spline.

Example:
    python experiments/drop_frames.py --out outputs/experiments
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import CubicSpline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from drone_nav.config import Config  # noqa: E402
from drone_nav.io_utils import list_frames, read_image  # noqa: E402
from drone_nav.localizer import MapLocalizer  # noqa: E402
from drone_nav.trajectory import centers_array  # noqa: E402

FOOTPRINT_PX = 640 * 0.2535  # side of one frame on the map


def fill(ref: np.ndarray, keep: np.ndarray, method: str) -> np.ndarray:
    """Positions for all frames from the kept ones, interpolated over the frame number."""
    idx = np.arange(len(ref))
    if method == "linear":
        return np.stack([np.interp(idx, idx[keep], ref[keep, a]) for a in range(2)], axis=1)
    return CubicSpline(idx[keep], ref[keep], axis=0)(idx)


def errors(ref: np.ndarray, keep: np.ndarray, method: str) -> np.ndarray:
    """Distance from the reference for the removed frames only."""
    return np.linalg.norm(fill(ref, keep, method)[~keep] - ref[~keep], axis=1)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--map", type=Path, default=ROOT / "data/global_map.png")
    p.add_argument("--frames", type=Path, default=ROOT / "data/crops")
    p.add_argument("--out", type=Path, default=ROOT / "outputs/experiments")
    p.add_argument("--trials", type=int, default=50)
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = Config()
    map_img = read_image(args.map)
    frames = list_frames(args.frames, cfg.frame_extensions)
    loc = MapLocalizer(map_img, cfg)
    poses = [loc.localize(read_image(f), i) for i, f in enumerate(frames)]
    assert all(p.success for p in poses), "reference run must localise every frame"
    ref = centers_array(poses)
    n = len(ref)
    rng = np.random.default_rng(0)

    # 1. Random removal --------------------------------------------------
    random_rows = []
    for pct in (10, 20, 30, 40, 50, 60):
        k = round(n * pct / 100)
        res = {m: [] for m in ("linear", "cubic")}
        for _ in range(args.trials):
            keep = np.ones(n, bool)
            keep[rng.choice(np.arange(1, n - 1), size=k, replace=False)] = False
            for m in res:
                res[m].append(errors(ref, keep, m))
        row = {"removed_pct": pct, "removed_frames": k}
        for m, e in res.items():
            e = np.concatenate(e)
            row[f"{m}_median_px"] = round(float(np.median(e)), 1)
            row[f"{m}_p95_px"] = round(float(np.percentile(e, 95)), 1)
            row[f"{m}_max_px"] = round(float(e.max()), 1)
        row["max_as_footprint_pct"] = round(100 * row["linear_max_px"] / FOOTPRINT_PX, 1)
        random_rows.append(row)
        print("random", row)

    # 2. Contiguous gaps -------------------------------------------------
    gap_rows = []
    for g in (1, 2, 3, 5, 8, 10, 15):
        res = {m: [] for m in ("linear", "cubic")}
        for start in range(1, n - 1 - g + 1):
            keep = np.ones(n, bool)
            keep[start:start + g] = False
            for m in res:
                res[m].append(errors(ref, keep, m).max())
        row = {"gap_frames": g,
               "gap_length_px": round(g * float(np.median(np.linalg.norm(np.diff(ref, axis=0), axis=1))))}
        for m, e in res.items():
            row[f"{m}_mean_px"] = round(float(np.mean(e)), 1)
            row[f"{m}_worst_px"] = round(float(np.max(e)), 1)
        row["worst_as_footprint_pct"] = round(100 * row["linear_worst_px"] / FOOTPRINT_PX, 1)
        gap_rows.append(row)
        print("gap", row)

    # 3. End-to-end through main.py -------------------------------------
    # Replace 30 % of frames (incl. a 5-frame gap) with images the localiser must reject.
    bad = sorted(set(rng.choice(np.arange(1, n - 1), size=10, replace=False).tolist()) | {22, 23, 24, 25, 26})
    with tempfile.TemporaryDirectory() as tmp:
        tmp_frames = Path(tmp) / "crops"
        tmp_frames.mkdir()
        unrelated = cv2.resize(cv2.flip(map_img[:400, :400], -1), (640, 640))
        unrelated = cv2.GaussianBlur(unrelated, (0, 0), 25)  # featureless, like fog or lens cap
        for i, f in enumerate(frames):
            if i in bad:
                kind = bad.index(i) % 3
                img = (np.zeros((640, 640, 3), np.uint8) if kind == 0 else
                       rng.integers(0, 256, (640, 640, 3), dtype=np.uint8) if kind == 1 else unrelated)
                cv2.imwrite(str(tmp_frames / f.name), img)
            else:
                shutil.copy(f, tmp_frames / f.name)
        e2e_out = args.out / "drop_e2e"
        subprocess.run([sys.executable, str(ROOT / "main.py"), "--map", str(args.map), "--frames", str(tmp_frames),
                        "--out", str(e2e_out), "--no-show", "--no-smooth"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with (e2e_out / cfg.trajectory_filename).open() as fh:
        traj = list(csv.DictReader(fh))
    flagged = [int(r["index"]) for r in traj if r["status"] == "interpolated"]
    track = np.array([[float(r["x"]), float(r["y"])] for r in traj])
    e2e_err = np.linalg.norm(track[bad] - ref[bad], axis=1)
    ok_err = np.linalg.norm(np.delete(track, bad, 0) - np.delete(ref, bad, 0), axis=1)
    e2e = {"replaced_frames": len(bad), "flagged_interpolated": len(flagged), "all_flagged": flagged == bad,
           "kept_frames_max_shift_px": round(float(ok_err.max()), 3),
           "interp_median_px": round(float(np.median(e2e_err)), 1), "interp_max_px": round(float(e2e_err.max()), 1)}
    print("e2e", e2e, "replaced:", bad)

    for name, rows in (("drop_random.csv", random_rows), ("drop_gap.csv", gap_rows), ("drop_e2e.csv", [e2e])):
        with (args.out / name).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
