"""Reconstruct a drone route from aerial frames and a satellite map.

Example:
    python main.py --map data/global_map.png --frames data/crops --out outputs
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from drone_nav.config import Config
from drone_nav.io_utils import list_frames, read_image
from drone_nav.localizer import FramePose, MapLocalizer
from drone_nav.trajectory import centers_array, interpolate_missing, smooth_trajectory
from drone_nav.visualize import draw_route, write_flyover_video


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line interface."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--map", required=True, type=Path, help="Path to the satellite map image.")
    p.add_argument("--frames", required=True, type=Path, help="Directory with drone frames (crop_N.png).")
    p.add_argument("--out", default=Path("outputs"), type=Path, help="Output directory (default: outputs).")
    p.add_argument("--video", action="store_true", help="Also render a fly-over video.")
    p.add_argument("--no-show", action="store_true", help="Do not open a window with the route at the end.")
    p.add_argument("--no-smooth", action="store_true", help="Disable Savitzky-Golay smoothing.")
    return p.parse_args(argv)


def show_route(route: np.ndarray, max_w: int = 1500, max_h: int = 850) -> None:
    """Show the route image in a window (scaled to fit the screen); any key closes it."""
    scale = min(max_w / route.shape[1], max_h / route.shape[0], 1.0)
    view = cv2.resize(route, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    try:
        cv2.imshow("Drone route (press any key to close)", view)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:  # no display (server / CI): the file is still saved
        print("No display available, route window skipped.")


def write_csv(
    path: Path,
    frames: list[Path],
    poses: list[FramePose],
    track: np.ndarray,
    raw: np.ndarray,
    interpolated: np.ndarray,
) -> None:
    """One row per frame. x, y = final (interpolated/smoothed) track; x_raw, y_raw = measured centre."""
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "x", "y", "angle_deg", "scale", "n_matches", "n_inliers", "status",
                    "frame", "x_raw", "y_raw"])
        for i, pose in enumerate(poses):
            w.writerow([
                i, f"{track[i, 0]:.2f}", f"{track[i, 1]:.2f}",
                f"{pose.angle_deg:.2f}", f"{pose.scale:.4f}", pose.n_matches, pose.n_inliers,
                "interpolated" if interpolated[i] else "ok",
                frames[i].name, f"{raw[i, 0]:.2f}", f"{raw[i, 1]:.2f}",
            ])


def main(argv: list[str] | None = None) -> int:
    """Run the full pipeline; returns a process exit code."""
    args = parse_args(argv)
    cfg = Config()
    if args.no_smooth:
        cfg = replace(cfg, smooth=False)

    t_start = time.perf_counter()
    try:
        map_image = read_image(args.map)
        frames = list_frames(args.frames, cfg.frame_extensions)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Map {map_image.shape[1]}x{map_image.shape[0]}, {len(frames)} frames")

    t0 = time.perf_counter()
    localizer = MapLocalizer(map_image, cfg)
    t_map = time.perf_counter() - t0
    print(f"Map features: {localizer.n_map_features} ({t_map:.1f} s)")

    t0 = time.perf_counter()
    poses: list[FramePose] = []
    for i, path in enumerate(tqdm(frames, desc="Localising", unit="frame")):
        try:
            pose = localizer.localize(read_image(path), i)
        except ValueError as e:  # undecodable file: mark as failed, keep going
            pose = FramePose(i, False, reason=str(e))
        if not pose.success:
            tqdm.write(f"  frame {i} ({path.name}) not localised: {pose.reason}")
        poses.append(pose)
    t_loc = time.perf_counter() - t0

    raw = centers_array(poses)
    try:
        track, interpolated = interpolate_missing(raw)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    if cfg.smooth:
        track = smooth_trajectory(track, cfg.savgol_window, cfg.savgol_polyorder)

    write_csv(args.out / cfg.trajectory_filename, frames, poses, track, raw, interpolated)
    route = draw_route(map_image, poses, track, interpolated, cfg)
    cv2.imwrite(str(args.out / cfg.route_filename), route)

    if args.video:
        write_flyover_video(map_image, poses, track, interpolated,
                            lambda i: read_image(frames[i]), args.out / cfg.video_filename, cfg)

    n_ok = sum(p.success for p in poses)
    inliers = [p.n_inliers for p in poses if p.success]
    print(f"\nLocalised {n_ok}/{len(poses)} frames"
          + (f" (inliers min {min(inliers)}, median {int(np.median(inliers))})" if inliers else ""))
    if n_ok < len(poses):
        print(f"Interpolated: {[p.index for p in poses if not p.success]}")
    print(f"Smoothing: {'on' if cfg.smooth else 'off'}")
    print(f"Time: map features {t_map:.1f} s, localisation {t_loc:.1f} s "
          f"({t_loc / len(poses) * 1000:.0f} ms/frame), total {time.perf_counter() - t_start:.1f} s")
    print(f"Results in: {args.out}")
    if not args.no_show:
        show_route(route)
    return 0


if __name__ == "__main__":
    sys.exit(main())
