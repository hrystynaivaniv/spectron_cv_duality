"""Find the breaking point: push one degradation at a time until localisation fails.

Uses the generated flights from synthetic_flights.py (known ground truth). For each
setting it reports the share of frames that pass the pipeline's own checks and how many
of those are actually correct (centre within 10 map px of the truth).

Example:
    python experiments/stress_test.py --out outputs/experiments
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthetic_flights import FRAME_SIZE, N_FRAMES, ROOT, make_flight, render  # noqa: E402

from drone_nav.config import Config  # noqa: E402
from drone_nav.io_utils import read_image  # noqa: E402
from drone_nav.localizer import MapLocalizer  # noqa: E402

CORRECT_PX = 10.0  # a localised frame counts as correct if its centre is this close to the truth


def noise(img, sigma, rng):
    return np.clip(img + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


def blur(img, sigma, rng):
    return img if sigma == 0 else cv2.GaussianBlur(img, (0, 0), sigma)


def contrast(img, k, rng):
    """Flatten contrast around the mean: k=1 unchanged, k=0.1 almost grey (haze, dusk)."""
    m = img.mean()
    return np.clip((img.astype(np.float32) - m) * k + m, 0, 255).astype(np.uint8)


def contrast_stretched(img, k, rng):
    """Same haze, then a 1-99 percentile contrast stretch before matching (proposed fix)."""
    img = contrast(img, k, rng)
    lo, hi = np.percentile(img, (1, 99))
    return np.clip((img.astype(np.float32) - lo) * 255 / (hi - lo + 1e-6), 0, 255).astype(np.uint8)


def clouds(img, frac, rng):
    """Hide ``frac`` of the frame behind soft white cloud blobs."""
    if frac == 0:
        return img
    mask = np.zeros(img.shape[:2], np.float32)
    while mask.mean() < frac:
        r = int(rng.integers(60, 200))
        cv2.circle(mask, (int(rng.integers(0, FRAME_SIZE)), int(rng.integers(0, FRAME_SIZE))), r, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), 15)[..., None]
    return (img * (1 - mask) + 235 * mask).astype(np.uint8)


SWEEPS = {
    "noise_sigma": (noise, [0, 10, 20, 30, 45, 60, 80]),
    "blur_sigma_px": (blur, [0, 2, 4, 6, 8, 10]),
    "clouds_pct": (clouds, [0, 0.2, 0.4, 0.6, 0.75, 0.85]),
    "contrast_k": (contrast, [1.0, 0.5, 0.3, 0.2, 0.1, 0.05]),
    "contrast_k_stretched": (contrast_stretched, [1.0, 0.5, 0.3, 0.2, 0.1, 0.05]),
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--map", type=Path, default=ROOT / "data/global_map.png")
    p.add_argument("--out", type=Path, default=ROOT / "outputs/experiments")
    p.add_argument("--flights", type=int, default=2)
    args = p.parse_args(argv)

    map_img = read_image(args.map)
    loc = MapLocalizer(map_img, Config())
    flights = [make_flight(map_img.shape[:2], np.random.default_rng(1000 + f)) for f in range(args.flights)]
    clean = [[render(map_img, c[i], h[i], s[i]) for i in range(N_FRAMES)] for c, h, s in flights]

    rows = []
    for name, (fn, values) in SWEEPS.items():
        for v in values:
            rng = np.random.default_rng(7)
            n = ok = correct = 0
            errs = []
            for (centres, _, _), frames in zip(flights, clean):
                for i, fr in enumerate(frames):
                    pose = loc.localize(fn(fr, v, rng), i)
                    n += 1
                    if pose.success:
                        ok += 1
                        e = float(np.linalg.norm(pose.center - centres[i]))
                        errs.append(e)
                        correct += e <= CORRECT_PX
            row = {"sweep": name, "value": v, "frames": n, "localised_pct": round(100 * ok / n, 1),
                   "correct_pct": round(100 * correct / n, 1),
                   "wrong_accepted": ok - correct,
                   "median_err_px": round(float(np.median(errs)), 2) if errs else math.nan}
            rows.append(row)
            print(row)

    with (args.out / "stress_summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
