"""Generate new drone flights with known ground truth and measure localisation accuracy.

The task data has no ground-truth coordinates, so here we create our own datasets:
random smooth flights over the satellite map, every frame cut out with a known centre,
heading and altitude (scale), then degraded like a real camera would degrade it
(exposure, colour cast, sensor noise, defocus, JPEG, clouds). The unchanged pipeline
(MapLocalizer + interpolation) is run on them and compared with the truth.

Example:
    python experiments/synthetic_flights.py --map data/global_map.png --out outputs/experiments
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from drone_nav.config import Config  # noqa: E402
from drone_nav.io_utils import read_image  # noqa: E402
from drone_nav.localizer import MapLocalizer  # noqa: E402
from drone_nav.trajectory import centers_array, interpolate_missing  # noqa: E402
from drone_nav.visualize import draw_route  # noqa: E402

FRAME_SIZE = 640
BASE_SCALE = 0.253  # frame px -> map px, same as the task data
STEP_PX = 78  # distance between consecutive frames on the map, as in the task data
N_FRAMES = 50


@dataclass(frozen=True)
class Level:
    """How strongly frames are degraded."""

    name: str
    gain: float  # +- relative brightness change
    color: float  # +- per-channel gain (colour cast / other season)
    noise: float  # sensor noise sigma, grey levels
    blur: float  # Gaussian blur sigma, px (defocus / motion)
    jpeg: int  # JPEG quality, 100 = off
    clouds: float  # max fraction of the frame hidden by a cloud


LEVELS = [
    Level("clean", 0.0, 0.0, 0.0, 0.0, 100, 0.0),
    Level("realistic", 0.2, 0.10, 6.0, 1.0, 75, 0.0),
    Level("harsh", 0.4, 0.20, 15.0, 2.0, 40, 0.25),
]


def make_flight(map_hw: tuple[int, int], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random smooth flight: centres (N,2), camera heading (N,) in degrees, scale (N,).

    The drone moves with a fixed step and a slowly drifting course; near the map edge
    it steers back to the centre so every footprint stays on the map.
    """
    h, w = map_hw
    margin = FRAME_SIZE * BASE_SCALE * 1.15 * math.sqrt(2) / 2 + 20  # rotated footprint fits
    pos = np.array([rng.uniform(margin, w - margin), rng.uniform(margin, h - margin)])
    course = rng.uniform(0, 2 * math.pi)
    turn = 0.0
    centres, courses = [], []
    for _ in range(N_FRAMES):
        centres.append(pos.copy())
        courses.append(course)
        turn = 0.8 * turn + rng.normal(0, math.radians(5))  # smooth turning rate
        nxt = pos + STEP_PX * np.array([math.cos(course + turn), math.sin(course + turn)])
        if not (margin < nxt[0] < w - margin and margin < nxt[1] < h - margin):
            to_centre = math.atan2(h / 2 - pos[1], w / 2 - pos[0])
            diff = (to_centre - course + math.pi) % (2 * math.pi) - math.pi
            turn = float(np.clip(diff, -math.radians(35), math.radians(35)))
        course += turn
        pos = pos + STEP_PX * np.array([math.cos(course), math.sin(course)])
    centres = np.array(centres)
    # Camera "up" follows the course plus a fixed random mount offset: any angle 0..360.
    heading = (np.degrees(np.array(courses)) + rng.uniform(0, 360)) % 360
    # Altitude changes slowly by up to +-8 %.
    phase = rng.uniform(0, 2 * math.pi)
    scale = BASE_SCALE * (1 + 0.08 * np.sin(np.linspace(0, 2 * math.pi, N_FRAMES) + phase))
    return centres, heading, scale


def frame_to_map(center: np.ndarray, angle_deg: float, scale: float) -> np.ndarray:
    """2x3 similarity mapping frame px to map px with the frame centre on ``center``."""
    t = math.radians(angle_deg)
    a = scale * np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    offset = center - a @ np.array([FRAME_SIZE / 2, FRAME_SIZE / 2])
    return np.hstack([a, offset[:, None]])


def render(map_img: np.ndarray, center: np.ndarray, angle: float, scale: float) -> np.ndarray:
    """Cut the frame out of the map (map -> frame resampling, like a camera looking down)."""
    m = frame_to_map(center, angle, scale)
    return cv2.warpAffine(map_img, m, (FRAME_SIZE, FRAME_SIZE),
                          flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)


def degrade(img: np.ndarray, lv: Level, rng: np.random.Generator) -> np.ndarray:
    """Apply camera / weather effects of the given level."""
    out = img.astype(np.float32)
    if lv.gain or lv.color:
        gains = (1 + rng.uniform(-lv.gain, lv.gain)) * (1 + rng.uniform(-lv.color, lv.color, size=3))
        out *= gains.astype(np.float32)
    if lv.clouds:
        mask = np.zeros(img.shape[:2], np.float32)
        r = int(FRAME_SIZE * math.sqrt(rng.uniform(0, lv.clouds) / math.pi))
        if r > 0:
            c = (int(rng.integers(0, FRAME_SIZE)), int(rng.integers(0, FRAME_SIZE)))
            cv2.circle(mask, c, r, 1.0, -1)
            mask = cv2.GaussianBlur(mask, (0, 0), r / 3 + 1)[..., None]
            out = out * (1 - mask) + 235 * mask
    if lv.blur:
        out = cv2.GaussianBlur(out, (0, 0), lv.blur)
    if lv.noise:
        out += rng.normal(0, lv.noise, size=out.shape).astype(np.float32)
    out = np.clip(out, 0, 255).astype(np.uint8)
    if lv.jpeg < 100:
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, lv.jpeg])
        out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return out


def angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest absolute difference between angles in degrees."""
    return np.abs((a - b + 180) % 360 - 180)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--map", type=Path, default=ROOT / "data/global_map.png")
    p.add_argument("--out", type=Path, default=ROOT / "outputs/experiments")
    p.add_argument("--flights", type=int, default=5, help="flights per degradation level")
    p.add_argument("--save-frames", action="store_true", help="also write the generated frames")
    args = p.parse_args(argv)

    cfg = Config()
    map_img = read_image(args.map)
    loc = MapLocalizer(map_img, cfg)
    args.out.mkdir(parents=True, exist_ok=True)

    rows, summary = [], []
    for li, lv in enumerate(LEVELS):
        errs, aerrs, serrs, track_errs, n_ok, n_all = [], [], [], [], 0, 0
        for f in range(args.flights):
            rng = np.random.default_rng(1000 + f)  # same routes for every level
            centres, heading, scale = make_flight(map_img.shape[:2], rng)
            noise_rng = np.random.default_rng(100 * li + f)  # deterministic per level+flight
            poses = []
            for i in range(N_FRAMES):
                frame = degrade(render(map_img, centres[i], heading[i], scale[i]), lv, noise_rng)
                if args.save_frames:
                    d = args.out / "synthetic_frames" / f"{lv.name}_flight{f}"
                    d.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(d / f"crop_{i}.png"), frame)
                pose = loc.localize(frame, i)
                poses.append(pose)
                row = {"level": lv.name, "flight": f, "frame": i, "ok": pose.success,
                       "gt_x": centres[i, 0], "gt_y": centres[i, 1], "gt_angle": heading[i], "gt_scale": scale[i],
                       "x": pose.center[0], "y": pose.center[1], "angle": pose.angle_deg % 360,
                       "scale": pose.scale, "inliers": pose.n_inliers}
                rows.append(row)
                n_all += 1
                if pose.success:
                    n_ok += 1
                    errs.append(float(np.linalg.norm(pose.center - centres[i])))
                    aerrs.append(float(angle_diff(np.array(pose.angle_deg), np.array(heading[i]))))
                    serrs.append(abs(pose.scale / scale[i] - 1))
            track, interp = interpolate_missing(centers_array(poses))
            track_errs.extend(np.linalg.norm(track - centres, axis=1).tolist())
            if f == 0:
                img = draw_route(map_img, poses, track, interp, cfg)
                for c in centres:  # ground truth on top: small red crosses
                    cv2.drawMarker(img, (int(c[0]), int(c[1])), (40, 40, 255), cv2.MARKER_CROSS, 18, 3)
                cv2.imwrite(str(args.out / f"synthetic_{lv.name}_flight0.png"), img)
        e, te = np.array(errs), np.array(track_errs)
        summary.append({
            "level": lv.name, "frames": n_all, "localised": n_ok, "rate_pct": round(100 * n_ok / n_all, 1),
            "centre_err_median_px": round(float(np.median(e)), 2), "centre_err_p95_px": round(float(np.percentile(e, 95)), 2),
            "centre_err_max_px": round(float(e.max()), 2), "angle_err_median_deg": round(float(np.median(aerrs)), 3),
            "angle_err_max_deg": round(float(np.max(aerrs)), 3), "scale_err_median_pct": round(100 * float(np.median(serrs)), 3),
            "track_err_max_px": round(float(te.max()), 2),
        })

    with (args.out / "synthetic_frames.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with (args.out / "synthetic_summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    for s in summary:
        print(s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
