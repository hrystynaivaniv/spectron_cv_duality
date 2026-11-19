"""All tunable parameters of the pipeline in one place.

Default values come from a preliminary study of the task data (summarised in README.md).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """Parameters for feature extraction, matching, pose checks, trajectory and drawing."""

    # --- Input -----------------------------------------------------------
    # Extensions recognised as frame images inside the frames directory.
    frame_extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    # --- Features --------------------------------------------------------
    # Frames are ~4x more detailed than the map; shrinking them to map scale makes
    # ~99% of good matches RANSAC inliers (vs ~80-85% at full size) and is faster.
    frame_scale: float = 0.25
    # 0 = no limit. Capping map keypoints (e.g. 40k) leaves whole map regions
    # without features and drops some frames to ~20 inliers.
    map_nfeatures: int = 0
    frame_nfeatures: int = 0

    # --- Matching --------------------------------------------------------
    flann_algorithm_kdtree: int = 1
    flann_trees: int = 5
    flann_checks: int = 64
    knn_k: int = 2
    ratio_test: float = 0.75

    # --- Transform estimation -------------------------------------------
    # Threshold is in original-map pixels.
    ransac_reproj_threshold: float = 5.0
    ransac_max_iters: int = 2000
    ransac_confidence: float = 0.99
    ransac_refine_iters: int = 10

    # --- Pose sanity checks ---------------------------------------------
    min_inliers: int = 20
    # Frame-to-map scale is constant (0.253-0.254) for this dataset; a pose far
    # from it means a wrong match. ``None`` disables the check.
    expected_scale: float | None = 0.253
    scale_tolerance: float = 0.15  # relative, i.e. +-15%

    # --- Trajectory ------------------------------------------------------
    smooth: bool = True
    savgol_window: int = 5
    savgol_polyorder: int = 2

    # --- Route image -----------------------------------------------------
    route_filename: str = "route.png"
    trajectory_filename: str = "trajectory.csv"
    # Map is dimmed so the route overlay stands out.
    map_dim_factor: float = 0.65
    line_thickness: int = 4
    point_radius: int = 11
    endpoint_radius: int = 22
    frame_box_thickness: int = 2
    marker_border: int = 2
    frame_box_alpha: float = 0.55
    label_font_scale: float = 0.95
    label_thickness: int = 2
    label_outline: int = 5
    # Labels alternate sides of the path so neighbours (~77 px apart) do not collide.
    label_offset_px: int = 26
    endpoint_font_scale: float = 1.3
    # START/FINISH tags sit this far beyond the end points, along the path direction.
    endpoint_label_offset_px: int = 75
    tag_padding: int = 10
    legend_font_scale: float = 0.9
    legend_padding: int = 20
    legend_line_height: int = 42
    legend_alpha: float = 0.8
    # Colours are BGR.
    color_route: tuple[int, int, int] = (0, 215, 255)
    color_point_ok: tuple[int, int, int] = (255, 200, 0)
    color_point_interp: tuple[int, int, int] = (200, 0, 255)
    color_frame_box: tuple[int, int, int] = (255, 255, 255)
    color_start: tuple[int, int, int] = (0, 200, 0)
    color_finish: tuple[int, int, int] = (0, 0, 230)
    color_text: tuple[int, int, int] = (255, 255, 255)
    color_text_outline: tuple[int, int, int] = (0, 0, 0)
    color_legend_bg: tuple[int, int, int] = (30, 30, 30)

    # --- Fly-over video --------------------------------------------------
    video_filename: str = "flyover.mp4"
    video_fourcc: str = "mp4v"
    video_size: tuple[int, int] = (1280, 720)  # (width, height)
    video_fps: int = 25
    # Intermediate camera positions between consecutive frames, for smooth motion.
    video_substeps: int = 6
    # Output pixels per map pixel; >1 zooms into the map.
    video_zoom: float = 1.6
    video_inset_size: int = 260
    video_inset_margin: int = 16
    video_font_scale: float = 0.9
    video_caption_margin: int = 30
    video_point_radius: int = 5


DEFAULT_CONFIG = Config()
