"""Route image and optional fly-over video."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np
from tqdm import tqdm

from .config import Config
from .localizer import FramePose

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_AA = cv2.LINE_AA


def _pt(p: np.ndarray) -> tuple[int, int]:
    return int(round(float(p[0]))), int(round(float(p[1])))


def _put_text(
    img: np.ndarray,
    text: str,
    center: np.ndarray,
    font_scale: float,
    color: tuple[int, int, int],
    thickness: int,
    outline_color: tuple[int, int, int],
    outline: int,
) -> None:
    """Draw outlined text centred on ``center``, clamped to stay inside the image."""
    (tw, th), baseline = cv2.getTextSize(text, _FONT, font_scale, thickness)
    h, w = img.shape[:2]
    margin = outline + 2
    x = int(np.clip(center[0] - tw / 2, margin, w - tw - margin))
    y = int(np.clip(center[1] + th / 2, th + margin, h - baseline - margin))
    cv2.putText(img, text, (x, y), _FONT, font_scale, outline_color, thickness + outline, _AA)
    cv2.putText(img, text, (x, y), _FONT, font_scale, color, thickness, _AA)


def _put_tag(
    img: np.ndarray,
    text: str,
    center: np.ndarray,
    font_scale: float,
    bg_color: tuple[int, int, int],
    cfg: Config,
) -> None:
    """Draw text on a solid coloured box (readable on any map texture), clamped inside the image."""
    thickness = cfg.label_thickness + 1
    (tw, th), baseline = cv2.getTextSize(text, _FONT, font_scale, thickness)
    pad = cfg.tag_padding
    bw, bh = tw + 2 * pad, th + baseline + 2 * pad
    h, w = img.shape[:2]
    x0 = int(np.clip(center[0] - bw / 2, 0, w - bw - 1))
    y0 = int(np.clip(center[1] - bh / 2, 0, h - bh - 1))
    cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + bh), bg_color, -1)
    cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + bh), cfg.color_text, cfg.marker_border)
    cv2.putText(img, text, (x0 + pad, y0 + pad + th), _FONT, font_scale, cfg.color_text, thickness, _AA)


def _dashed_line(
    img: np.ndarray, p0: np.ndarray, p1: np.ndarray, color, thickness: int, dash: float
) -> None:
    """OpenCV has no dashed lines; draw alternating short segments."""
    length = float(np.linalg.norm(p1 - p0))
    if length < 1e-6:
        return
    n = max(int(length // dash), 1)
    for k in range(0, n, 2):
        a = p0 + (p1 - p0) * (k / n)
        b = p0 + (p1 - p0) * (min(k + 1, n) / n)
        cv2.line(img, _pt(a), _pt(b), color, thickness, _AA)


def _path_normals(track: np.ndarray) -> np.ndarray:
    """Unit normal to the path at each point (from central differences)."""
    tangents = np.gradient(track, axis=0) if len(track) > 1 else np.array([[1.0, 0.0]])
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = np.divide(tangents, norms, out=np.tile([[1.0, 0.0]], (len(tangents), 1)), where=norms > 1e-9)
    return np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)


def _path_tangent(track: np.ndarray, i: int) -> np.ndarray:
    """Unit direction of travel at an end point of the track."""
    if len(track) < 2:
        return np.array([1.0, 0.0])
    d = track[1] - track[0] if i == 0 else track[-1] - track[-2]
    n = np.linalg.norm(d)
    return d / n if n > 1e-9 else np.array([1.0, 0.0])


def _draw_route_line(img: np.ndarray, track: np.ndarray, interpolated: np.ndarray, cfg: Config) -> None:
    """Solid line between measured points, dashed where either end was interpolated."""
    for i in range(len(track) - 1):
        if interpolated[i] or interpolated[i + 1]:
            _dashed_line(img, track[i], track[i + 1], cfg.color_point_interp, cfg.line_thickness,
                         dash=cfg.point_radius)
        else:
            cv2.line(img, _pt(track[i]), _pt(track[i + 1]), cfg.color_route, cfg.line_thickness, _AA)


def _draw_point(img: np.ndarray, p: np.ndarray, interpolated: bool, radius: int, cfg: Config) -> None:
    if interpolated:
        cv2.circle(img, _pt(p), radius, cfg.color_point_interp, cfg.label_thickness + 1, _AA)
    else:
        cv2.circle(img, _pt(p), radius, cfg.color_point_ok, -1, _AA)
        cv2.circle(img, _pt(p), radius, cfg.color_text_outline, cfg.marker_border, _AA)


def _draw_legend(img: np.ndarray, title: str, cfg: Config) -> None:
    """Semi-transparent legend box in the top-left corner."""
    entries = ["route", "ok", "interp", "box", "start", "finish"]
    labels = {
        "route": "Route (localised)",
        "ok": "Localised frame centre (N = frame index)",
        "interp": "Interpolated position (frame not localised)",
        "box": "Frame footprint (size + rotation)",
        "start": "START",
        "finish": "FINISH",
    }
    pad, lh, fs = cfg.legend_padding, cfg.legend_line_height, cfg.legend_font_scale
    icon_w = 3 * cfg.point_radius
    text_w = max(cv2.getTextSize(s, _FONT, fs, cfg.label_thickness)[0][0] for s in [title, *labels.values()])
    box_w = pad * 3 + icon_w + text_w
    box_h = pad * 2 + lh * (len(entries) + 1)

    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (box_w, box_h), cfg.color_legend_bg, -1)
    cv2.addWeighted(overlay, cfg.legend_alpha, img, 1 - cfg.legend_alpha, 0, dst=img)

    y = pad + lh // 2
    cv2.putText(img, title, (pad, y + lh // 4), _FONT, fs, cfg.color_text, cfg.label_thickness, _AA)
    for key in entries:
        y += lh
        c = np.array([pad + icon_w / 2, y])
        if key == "route":
            cv2.line(img, _pt(c - [icon_w / 2, 0]), _pt(c + [icon_w / 2, 0]), cfg.color_route, cfg.line_thickness, _AA)
        elif key == "ok":
            _draw_point(img, c, False, cfg.point_radius, cfg)
        elif key == "interp":
            _draw_point(img, c, True, cfg.point_radius, cfg)
        elif key == "box":
            half = cfg.point_radius
            cv2.rectangle(img, _pt(c - half), _pt(c + half), cfg.color_frame_box, cfg.frame_box_thickness, _AA)
        else:
            color = cfg.color_start if key == "start" else cfg.color_finish
            cv2.circle(img, _pt(c), cfg.point_radius, color, -1, _AA)
            cv2.circle(img, _pt(c), cfg.point_radius, cfg.color_text, cfg.marker_border, _AA)
        cv2.putText(img, labels[key], (pad * 2 + icon_w, y + lh // 4), _FONT, fs, cfg.color_text,
                    cfg.label_thickness, _AA)


def draw_route(
    map_image: np.ndarray,
    poses: Sequence[FramePose],
    track: np.ndarray,
    interpolated: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    """Render the route over the map: footprints, line, numbered points, START/FINISH, legend."""
    img = (map_image.astype(np.float32) * cfg.map_dim_factor).astype(np.uint8)

    # Footprints first (bottom layer), blended so the map underneath stays visible.
    boxes = img.copy()
    for pose in poses:
        if pose.success:
            cv2.polylines(boxes, [np.round(pose.corners).astype(np.int32)], True,
                          cfg.color_frame_box, cfg.frame_box_thickness, _AA)
    cv2.addWeighted(boxes, cfg.frame_box_alpha, img, 1 - cfg.frame_box_alpha, 0, dst=img)

    _draw_route_line(img, track, interpolated, cfg)

    normals = _path_normals(track)
    last = len(track) - 1
    for i, p in enumerate(track):
        if i in (0, last):
            continue
        _draw_point(img, p, bool(interpolated[i]), cfg.point_radius, cfg)
        # Alternate label sides so consecutive numbers never sit next to each other.
        side = 1.0 if i % 2 == 0 else -1.0
        _put_text(img, str(i), p + side * normals[i] * (cfg.point_radius + cfg.label_offset_px),
                  cfg.label_font_scale, cfg.color_text, cfg.label_thickness,
                  cfg.color_text_outline, cfg.label_outline)

    # START is labelled behind the first point and FINISH beyond the last one,
    # i.e. off the path where no other labels are.
    for i, name, color, sign in ((0, "START", cfg.color_start, -1.0), (last, "FINISH", cfg.color_finish, 1.0)):
        p = track[i]
        cv2.circle(img, _pt(p), cfg.endpoint_radius, color, -1, _AA)
        cv2.circle(img, _pt(p), cfg.endpoint_radius, cfg.color_text, cfg.marker_border + 1, _AA)
        _put_text(img, str(i), p, cfg.label_font_scale, cfg.color_text, cfg.label_thickness,
                  cfg.color_text_outline, cfg.label_outline)
        anchor = p + sign * _path_tangent(track, i) * (cfg.endpoint_radius + cfg.endpoint_label_offset_px)
        _put_tag(img, name, anchor, cfg.endpoint_font_scale, color, cfg)
        if len(track) == 1:
            break

    n_ok = int((~interpolated).sum())
    _draw_legend(img, f"Drone route: {n_ok}/{len(track)} frames localised", cfg)
    return img


def write_flyover_video(
    map_image: np.ndarray,
    poses: Sequence[FramePose],
    track: np.ndarray,
    interpolated: np.ndarray,
    load_frame: Callable[[int], np.ndarray],
    out_path: str | Path,
    cfg: Config,
) -> None:
    """Zoomed 'chase camera' over the map: travelled route, current footprint, drone-frame inset."""
    out_w, out_h = cfg.video_size
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*cfg.video_fourcc), cfg.video_fps,
                             (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer for {out_path} (codec {cfg.video_fourcc}).")

    z = cfg.video_zoom
    n = len(track)
    steps = [(i, k / cfg.video_substeps) for i in range(n - 1) for k in range(cfg.video_substeps)]
    steps.append((n - 1, 0.0))
    inset_cache: dict[int, np.ndarray] = {}

    try:
        for i, t in tqdm(steps, desc="Video", unit="frame"):
            j = min(i + 1, n - 1)
            cam = (1 - t) * track[i] + t * track[j]
            # Map -> output transform keeping the camera position in the centre.
            m = np.array([[z, 0, out_w / 2 - z * cam[0]], [0, z, out_h / 2 - z * cam[1]]])
            view = cv2.warpAffine(map_image, m, (out_w, out_h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT)

            def to_view(pts: np.ndarray) -> np.ndarray:
                return np.asarray(pts, dtype=np.float64).reshape(-1, 2) * z + m[:, 2]

            # Current footprint, interpolated between the two neighbouring measured frames.
            a, b = poses[i], poses[j]
            if a.success and b.success:
                corners = (1 - t) * a.corners + t * b.corners
            else:
                corners = a.corners if a.success else (b.corners if b.success else None)
            if corners is not None:
                cv2.polylines(view, [np.round(to_view(corners)).astype(np.int32)], True,
                              cfg.color_frame_box, cfg.frame_box_thickness + 1, _AA)

            travelled = np.vstack([track[: i + 1], cam[None]])
            vt = to_view(travelled)
            _draw_route_line(view, vt, np.append(interpolated[: i + 1], interpolated[j]), cfg)
            for k, p in enumerate(vt[:-1]):
                color = cfg.color_start if k == 0 else (cfg.color_point_interp if interpolated[k] else cfg.color_point_ok)
                cv2.circle(view, _pt(p), cfg.video_point_radius, color, -1, _AA)
            cv2.circle(view, _pt(vt[-1]), cfg.point_radius, cfg.color_finish, -1, _AA)

            # Drone-camera inset in the top-right corner.
            if i not in inset_cache:
                inset_cache.clear()
                inset_cache[i] = cv2.resize(load_frame(i), (cfg.video_inset_size, cfg.video_inset_size),
                                            interpolation=cv2.INTER_AREA)
            s, mg = cfg.video_inset_size, cfg.video_inset_margin
            x0 = out_w - s - mg
            view[mg:mg + s, x0:x0 + s] = inset_cache[i]
            cv2.rectangle(view, (x0, mg), (x0 + s, mg + s), cfg.color_text, cfg.marker_border)

            status = "interpolated" if interpolated[i] else f"heading {poses[i].angle_deg:.0f} deg"
            caption = f"Frame {i}/{n - 1}  x={cam[0]:.0f}  y={cam[1]:.0f}  {status}"
            _put_text(view, caption, np.array([out_w / 2, out_h - cfg.video_caption_margin]), cfg.video_font_scale,
                      cfg.color_text, cfg.label_thickness, cfg.color_text_outline, cfg.label_outline)
            writer.write(view)
    finally:
        writer.release()
