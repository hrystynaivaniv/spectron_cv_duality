"""Compare outputs/trajectory.csv with the reference trajectory table in research.md.

Usage:
    python tools/check_reference.py [--csv outputs/trajectory.csv] [--research research.md]
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

# Reference rows look like: | 12 | 1081 | 495 | 91.8 | 0.254 | 294/294 | 321/353 |
_ROW = re.compile(r"^\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*(-?[\d.]+)\s*\|\s*([\d.]+)\s*\|")


def load_reference(path: Path) -> dict[int, tuple[float, float, float, float]]:
    """Parse (x, y, angle, scale) per frame from the markdown table."""
    ref = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = _ROW.match(line.strip())
        if m:
            i, x, y, a, s = m.groups()
            ref[int(i)] = (float(x), float(y), float(a), float(s))
    if not ref:
        raise ValueError(f"No reference table found in {path}")
    return ref


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, default=Path("outputs/trajectory.csv"))
    p.add_argument("--research", type=Path, default=Path("research.md"))
    p.add_argument("--tolerance", type=float, default=2.0, help="Max allowed centre error, px.")
    args = p.parse_args()

    ref = load_reference(args.research)
    rows = list(csv.DictReader(args.csv.open()))
    raw_err, final_err, ang_err, scales = [], [], [], []
    worst = (-1, 0.0)
    for r in rows:
        i = int(r["index"])
        if i not in ref:
            continue
        x, y, a, _ = ref[i]
        final_err.append(np.hypot(float(r["x"]) - x, float(r["y"]) - y))
        if r["status"] != "ok":
            continue
        e = np.hypot(float(r["x_raw"]) - x, float(r["y_raw"]) - y)
        raw_err.append(e)
        worst = max(worst, (i, e), key=lambda t: t[1])
        ang_err.append(abs((float(r["angle_deg"]) - a + 180) % 360 - 180))
        scales.append(float(r["scale"]))

    n_ok = sum(r["status"] == "ok" for r in rows)
    print(f"Localised {n_ok}/{len(rows)} frames; compared with {len(ref)} reference rows")
    if not raw_err:
        print("No localised frames to compare.")
        return 1
    print(f"Measured centre error: max {max(raw_err):.2f} px (frame {worst[0]}), mean {np.mean(raw_err):.2f} px")
    print(f"Final track error (after interpolation/smoothing): max {max(final_err):.2f} px")
    print(f"Angle error: max {max(ang_err):.2f} deg (reference rounded to 0.1)")
    print(f"Scale: {min(scales):.4f} .. {max(scales):.4f}")
    ok = max(raw_err) <= args.tolerance and n_ok == len(ref)
    print("PASS" if ok else f"FAIL (tolerance {args.tolerance} px, all frames must be localised)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
