"""
This file implements segmentation of a provided csv of tag positions into
'movement' and 'rest' sections based on the speed of tag movement.

One function will:
* Identify each tag's speed at each frame, as change in x,y coordinates of the tag's position.
* For consecutive frame pairs where a tag is missing, set that tag's speed to 0.
* Optionally print the speed of each tag over time in a graph.

Another function will:
* Identify consecutive groups of frames where, for any of the tags, their speed exceeds a
  provided threshold.
* Return the beginning and end of such consecutive frames, in frame index.

A third function will:
* Take a video and the corresponding frame indices for start and end of consecutive groups.
* Render each group separately into their video files and store.
* Optionally, this video will include 60 frames before and after the video, plus a count
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from find_apriltags import draw_yaw_arrow
except ImportError:
    from opencv.find_apriltags import draw_yaw_arrow

try:
    from apriltag_tracking import gray_to_gradient_for_lk
except ImportError:
    from opencv.apriltag_tracking import gray_to_gradient_for_lk


def compute_tag_speeds(
    csv_path: str | Path,
    plot: bool = False,
    show_plot: bool = True,
    save_plot_path: str | Path | None = None,
    threshold: float | None = None,
) -> pd.DataFrame:
    """
    Compute each tag's speed at each frame as the change in (center_x, center_y) image
    position from the previous frame. For consecutive frame pairs where a tag is missing,
    speed is set to 0.

    Args:
        csv_path: Path to the tag CSV (must include: frame, tag_id, center_x, center_y).
        plot: If True, plot speed per tag over time.
        show_plot: If True and plot is True, display the plot with plt.show().
        save_plot_path: If set and plot is True, save the figure to this path.

        threshold: If set and plot is True, show threshold line and shade where speed exceeds it.

    Returns:
        DataFrame with columns: frame, tag_id, speed. One row per (frame, tag_id).
        Speed at frame 0 is 0 (no previous frame). Speed is in image units (pixels) per frame.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    for col in ["frame", "tag_id", "center_x", "center_y"]:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' in CSV. Columns: {list(df.columns)}")

    all_frames = sorted(df["frame"].unique())
    all_tag_ids = sorted(df["tag_id"].unique())
    df = df.sort_values(["tag_id", "frame"]).reset_index(drop=True)

    rows: list[dict] = []
    for tag_id in all_tag_ids:
        group = df[df["tag_id"] == tag_id]
        if group.empty:
            for f in all_frames:
                rows.append({"frame": f, "tag_id": tag_id, "speed": 0.0})
            continue
        frames = group["frame"].values
        x = group["center_x"].values
        y = group["center_y"].values
        frame_to_idx = {f: i for i, f in enumerate(frames)}

        for k, f in enumerate(all_frames):
            if f == all_frames[0]:
                speed = 0.0  # No previous frame
            else:
                prev_f = f - 1
                if prev_f in frame_to_idx and f in frame_to_idx:
                    i_prev = frame_to_idx[prev_f]
                    i_curr = frame_to_idx[f]
                    dx = x[i_curr] - x[i_prev]
                    dy = y[i_curr] - y[i_prev]
                    speed = float(np.sqrt(dx * dx + dy * dy))
                else:
                    speed = 0.0  # Tag missing in consecutive frame pair
            rows.append({"frame": f, "tag_id": tag_id, "speed": speed})

    speeds_df = pd.DataFrame(rows)

    if plot:
        _plot_speeds(
            speeds_df,
            show_plot=show_plot,
            save_path=save_plot_path,
            threshold=threshold,
        )

    return speeds_df


def _plot_speeds(
    speeds_df: pd.DataFrame,
    show_plot: bool = True,
    save_path: str | Path | None = None,
    threshold: float | None = None,
    segments: list[tuple[int, int]] | None = None,
) -> None:
    """
    Plot speed of each tag over time (frame).

    If threshold is set: draw a horizontal line at that value and shade the background
    for frame ranges where any tag's speed exceeds the threshold.

    If segments is set: shade the background for those (start_frame, end_frame) ranges
    (e.g. the segments returned by find_movement_segments), with a distinct color.
    """
    tag_ids = sorted(speeds_df["tag_id"].unique())
    fig, ax = plt.subplots(figsize=(10, 4))

    # Shade segment ranges first (behind everything) if provided
    if segments:
        for start_f, end_f in segments:
            ax.axvspan(
                start_f,
                end_f + 1,
                color="green",
                alpha=0.15,
                label="Segment" if start_f == segments[0][0] else None,
            )
    # Shade frame ranges where max speed > threshold (if threshold provided)
    if threshold is not None:
        max_speed_per_frame = speeds_df.groupby("frame", as_index=False)["speed"].max()
        frames = max_speed_per_frame["frame"].values
        max_speeds = max_speed_per_frame["speed"].values
        above = max_speeds > threshold
        i = 0
        first_span = True
        while i < len(frames):
            if not above[i]:
                i += 1
                continue
            start_f = int(frames[i])
            while i < len(frames) and above[i]:
                i += 1
            end_f = int(frames[i - 1])
            ax.axvspan(
                start_f,
                end_f + 1,
                color="orange",
                alpha=0.2,
                label="Above threshold" if first_span else None,
            )
            first_span = False

    for tag_id in tag_ids:
        sub = speeds_df[speeds_df["tag_id"] == tag_id]
        ax.plot(sub["frame"], sub["speed"], label=f"Tag {tag_id}", alpha=0.8)

    if threshold is not None:
        ax.axhline(threshold, color="red", linestyle="--", linewidth=1.5, label="Threshold")

    ax.set_xlabel("Frame")
    ax.set_ylabel("Speed")
    ax.set_title("Tag speed over time (center_x, center_y change per frame)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close()


def _get_tag_base_colors(tag_ids: list[int]) -> dict[int, tuple[int, int, int]]:
    """
    Assign a distinct base BGR color for each tag_id using a matplotlib colormap.
    """
    cmap = plt.get_cmap("tab10")
    colors: dict[int, tuple[int, int, int]] = {}
    n = max(1, len(tag_ids))
    for i, tid in enumerate(tag_ids):
        r, g, b, _ = cmap(i % n)
        # Convert from 0-1 RGB to 0-255 BGR
        colors[tid] = (int(255 * b), int(255 * g), int(255 * r))
    return colors


def _lighten_towards_white(
    bgr: tuple[int, int, int],
    t: float,
) -> tuple[int, int, int]:
    """
    Interpolate color so that t=0 is very light (near white),
    and t=1 is the original base color.
    """
    t = float(np.clip(t, 0.0, 1.0))
    b, g, r = bgr
    wb = 255
    wg = 255
    wr = 255
    lb = int((1.0 - t) * wb + t * b)
    lg = int((1.0 - t) * wg + t * g)
    lr = int((1.0 - t) * wr + t * r)
    return (lb, lg, lr)


def _draw_dotted_line(
    img: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 1,
    segment_length: int = 10,
    gap_length: int = 6,
) -> None:
    """
    Draw a dotted line between two points by alternating short line segments and gaps.
    """
    x1, y1 = pt1
    x2, y2 = pt2
    dx = x2 - x1
    dy = y2 - y1
    dist = float(np.hypot(dx, dy))
    if dist == 0:
        return
    direction = (dx / dist, dy / dist)
    step = segment_length + gap_length
    num_steps = int(dist // step) + 1
    for i in range(num_steps):
        start_d = i * step
        end_d = min(start_d + segment_length, dist)
        if end_d <= start_d:
            continue
        sx = int(x1 + direction[0] * start_d)
        sy = int(y1 + direction[1] * start_d)
        ex = int(x1 + direction[0] * end_d)
        ey = int(y1 + direction[1] * end_d)
        cv2.line(img, (sx, sy), (ex, ey), color, thickness, lineType=cv2.LINE_AA)


def render_segment_tag_positions(
    csv_or_df: str | Path | pd.DataFrame,
    segment: tuple[int, int],
    output_path: str | Path,
    connect_lines: bool = True,
    dotted_for_gaps: bool = True,
    image_size: tuple[int, int] | None = None,
    background_color: tuple[int, int, int] = (0, 0, 0),
) -> Path:
    """
    Render the positions of each tag within a single segment onto a single image.

    For each tag, points are drawn at (center_x, center_y) for frames within the
    [segment_start, segment_end] range. Each tag has an associated base color; for that
    tag, earlier frames are rendered with lighter shades and later frames with darker
    shades approaching the base color (time gradation).

    Optionally, positions for consecutive frames are linked with lines:
    - If two frames are consecutive (frame_{i+1} == frame_i + 1), draw a solid line.
    - If there is a gap (frame_{i+1} > frame_i + 1) and dotted_for_gaps is True,
      draw a dotted line between the two points.

    Args:
        csv_or_df: Path to a CSV or a pre-loaded DataFrame with columns:
            frame, tag_id, center_x, center_y.
        segment: (start_frame, end_frame) inclusive.
        output_path: Path where the rendered image will be saved (PNG recommended).
        connect_lines: If True, connect positions for each tag over time.
        dotted_for_gaps: If True, connect non-consecutive frames with dotted lines.
        image_size: Optional (width, height). If None, the size is inferred from
            max(center_x), max(center_y) within the segment, with a small margin.
        background_color: BGR background color for the canvas.

    Returns:
        Path to the saved image.
    """
    if isinstance(csv_or_df, (str, Path)):
        csv_path = Path(csv_or_df)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        df = csv_or_df.copy()

    for col in ["frame", "tag_id", "center_x", "center_y"]:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' in data. Columns: {list(df.columns)}")

    start_frame, end_frame = segment
    if end_frame < start_frame:
        raise ValueError(f"Invalid segment: end_frame ({end_frame}) < start_frame ({start_frame})")

    sub = df[(df["frame"] >= start_frame) & (df["frame"] <= end_frame)].copy()
    if sub.empty:
        raise ValueError(f"No tag positions found in segment frames [{start_frame}, {end_frame}]")

    # Determine canvas size
    if image_size is not None:
        width, height = image_size
    else:
        margin = 10
        max_x = float(sub["center_x"].max())
        max_y = float(sub["center_y"].max())
        width = int(max_x) + margin
        height = int(max_y) + margin
        width = max(width, 1)
        height = max(height, 1)

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:, :] = np.array(background_color, dtype=np.uint8)

    tag_ids = sorted(sub["tag_id"].unique())
    base_colors = _get_tag_base_colors(tag_ids)

    for tid in tag_ids:
        tdf = sub[sub["tag_id"] == tid].sort_values("frame")
        if tdf.empty:
            continue
        frames = tdf["frame"].to_numpy()
        xs = tdf["center_x"].to_numpy()
        ys = tdf["center_y"].to_numpy()
        n = len(frames)
        base_color = base_colors[tid]

        # Draw points with time-based light-to-dark gradation
        for idx in range(n):
            # Earlier frames: t close to 0 -> very light; later: t close to 1 -> base color
            t = idx / (n - 1) if n > 1 else 1.0
            color = _lighten_towards_white(base_color, t)
            x = int(round(xs[idx]))
            y = int(round(ys[idx]))
            if 0 <= x < width and 0 <= y < height:
                cv2.circle(canvas, (x, y), 4, color, thickness=-1, lineType=cv2.LINE_AA)

        # Draw connecting lines for motion track
        if connect_lines and n >= 2:
            for i in range(n - 1):
                x1 = int(round(xs[i]))
                y1 = int(round(ys[i]))
                x2 = int(round(xs[i + 1]))
                y2 = int(round(ys[i + 1]))
                if not (0 <= x1 < width and 0 <= y1 < height and 0 <= x2 < width and 0 <= y2 < height):
                    continue
                # Use darker shade for the line (later frame color)
                t_line = (i + 1) / (n - 1) if n > 1 else 1.0
                line_color = _lighten_towards_white(base_color, t_line)
                frame_curr = int(frames[i])
                frame_next = int(frames[i + 1])
                if frame_next == frame_curr + 1:
                    cv2.line(
                        canvas,
                        (x1, y1),
                        (x2, y2),
                        line_color,
                        thickness=2,
                        lineType=cv2.LINE_AA,
                    )
                elif dotted_for_gaps:
                    _draw_dotted_line(
                        canvas,
                        (x1, y1),
                        (x2, y2),
                        line_color,
                        thickness=2,
                    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)
    return output_path


def find_movement_segments(
    speeds_df: pd.DataFrame,
    threshold: float,
    min_segment_size: int = 25,
) -> list[tuple[int, int]]:
    """
    Find consecutive groups of frames where any tag's speed exceeds the given threshold.
    Returns (start_frame, end_frame) for each group, inclusive on both ends.
    Segments with fewer than min_segment_size frames are discarded.

    Args:
        speeds_df: DataFrame from compute_tag_speeds (columns: frame, tag_id, speed).
        threshold: Speed threshold; frames with max tag speed above this are "movement".
        min_segment_size: Minimum number of frames in a segment; smaller segments are dropped (default 25).

    Returns:
        List of (start_frame, end_frame) in ascending order by start_frame.
    """
    if speeds_df.empty:
        return []

    # Per-frame maximum speed over all tags
    max_speed_per_frame = speeds_df.groupby("frame", as_index=False)["speed"].max()
    frames = max_speed_per_frame["frame"].values
    max_speeds = max_speed_per_frame["speed"].values

    above = max_speeds > threshold
    segments: list[tuple[int, int]] = []
    i = 0
    while i < len(frames):
        if not above[i]:
            i += 1
            continue
        start_frame = int(frames[i])
        while i < len(frames) and above[i]:
            i += 1
        end_frame = int(frames[i - 1])
        size = end_frame - start_frame + 1
        if size >= min_segment_size:
            segments.append((start_frame, end_frame))

    return segments


def render_segment_videos(
    video_path: str | Path,
    segments: list[tuple[int, int]],
    output_dir: str | Path,
    padding_frames: int = 60,
    show_count: bool = True,
    pose_csv: str | Path | None = None,
    camera_intrinsics: tuple[float, float, float, float] | None = None,
    yaw_arrow_length_m: float = 0.05,
    yaw_arrow_length_px: float = 40.0,
    gradient_videos: bool = False,
    draw_tag_overlay: bool = True,
) -> list[Path]:
    """
    Render each segment as a separate video file. Optionally include padding_frames
    before and after each segment and draw a frame count on each frame.

    When pose_csv is provided, each tag's yaw is drawn as an arrow on the frame.
    If camera_intrinsics is also provided (and the CSV has x, y, z, yaw_deg, pitch_deg,
    roll_deg), the arrow is the projected 3D X axis (same as run_on_source in
    find_apriltags). Otherwise the arrow is drawn in the image plane from yaw_deg.

    When gradient_videos is True, frames are converted to Sobel gradients (using
    gray_to_gradient_for_lk) before any overlays. Two videos are rendered per segment:
    one for Sobel X and one for Sobel Y.

    When draw_tag_overlay is False, the tag borders and corner counts aren't drawn on 
    the frames.

    Args:
        video_path: Path to the source video.
        segments: List of (start_frame, end_frame) from find_movement_segments.
        output_dir: Directory where output videos will be written.
        padding_frames: Number of frames to include before and after each segment (default 60).
        show_count: If True, draw frame index (and optionally segment index) on each frame.
        pose_csv: Optional path to a CSV with pose per (frame, tag_id). Must have center_x,
            center_y, yaw_deg; for 3D arrows also x, y, z, pitch_deg, roll_deg.
        camera_intrinsics: Optional (fx, fy, cx, cy) for 3D yaw arrow projection.
        yaw_arrow_length_m: Length of the 3D yaw axis in meters (when using camera_intrinsics).
        yaw_arrow_length_px: Length of the 2D yaw arrow in pixels (when not using 3D).
        gradient_videos: If True, convert each frame to gradients and write two videos per
            segment (Sobel X and Sobel Y); overlays are drawn on top of the gradient frames.
        draw_tag_overlay: If False, do not draw tag borders and corner counts.

    Returns:
        List of paths to the written video files.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pose CSV for yaw arrows and tag corner visualization if requested
    pose_by_frame: dict[int, list[dict]] = {}
    has_corner_columns = False
    if pose_csv is not None:
        pose_path = Path(pose_csv)
        if not pose_path.exists():
            raise FileNotFoundError(f"Pose CSV not found: {pose_path}")
        pose_df = pd.read_csv(pose_path)
        base_required_cols = ["frame", "tag_id", "center_x", "center_y", "yaw_deg"]
        for col in base_required_cols:
            if col not in pose_df.columns:
                raise ValueError(
                    f"Pose CSV must have columns frame, tag_id, center_x, center_y, yaw_deg. "
                    f"Found: {list(pose_df.columns)}"
                )
        corner_cols = [
            "c0_x",
            "c0_y",
            "c1_x",
            "c1_y",
            "c2_x",
            "c2_y",
            "c3_x",
            "c3_y",
        ]
        has_corner_columns = all(col in pose_df.columns for col in corner_cols)

        for frame_idx, group in pose_df.groupby("frame", sort=True):
            pose_by_frame[int(frame_idx)] = group.to_dict("records")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames_video <= 0:
        total_frames_video = 2**31 - 1  # unknown length

    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    if fourcc == 0 or fourcc == -1:
        fourcc = cv2.VideoWriter_fourcc(*"MP4V")

    written: list[Path] = []
    for seg_idx, (start, end) in enumerate(segments):
        read_start = max(0, start - padding_frames)
        read_end = min(total_frames_video - 1, end + padding_frames)
        num_frames = read_end - read_start + 1

        if gradient_videos:
            out_name_x = f"{video_path.stem}_segment_{seg_idx:03d}_sobel_x_frames_{read_start}-{read_end}.mp4"
            out_name_y = f"{video_path.stem}_segment_{seg_idx:03d}_sobel_y_frames_{read_start}-{read_end}.mp4"
            out_path_x = output_dir / out_name_x
            out_path_y = output_dir / out_name_y
            writer_x = cv2.VideoWriter(str(out_path_x), fourcc, fps, (width, height))
            writer_y = cv2.VideoWriter(str(out_path_y), fourcc, fps, (width, height))
            if not writer_x.isOpened():
                raise IOError(f"Could not create video: {out_path_x}")
            if not writer_y.isOpened():
                raise IOError(f"Could not create video: {out_path_y}")
            writers = [(writer_x, out_path_x), (writer_y, out_path_y)]
        else:
            out_name = f"{video_path.stem}_segment_{seg_idx:03d}_frames_{read_start}-{read_end}.mp4"
            out_path = output_dir / out_name
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
            if not writer.isOpened():
                raise IOError(f"Could not create video: {out_path}")
            writers = [(writer, out_path)]

        cap.set(cv2.CAP_PROP_POS_FRAMES, read_start)
        for local_idx in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            global_frame = read_start + local_idx

            if gradient_videos:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                grad_x = gray_to_gradient_for_lk(gray, mode="x")
                grad_y = gray_to_gradient_for_lk(gray, mode="y")
                frame_x = cv2.cvtColor(grad_x, cv2.COLOR_GRAY2BGR)
                frame_y = cv2.cvtColor(grad_y, cv2.COLOR_GRAY2BGR)
                output_frames = [(frame_x, writer_x), (frame_y, writer_y)]
            else:
                output_frames = [(frame, writer)]

            for out_frame, out_writer in output_frames:
                if show_count:
                    text = f"Frame {global_frame}"
                    if padding_frames > 0:
                        if global_frame < start:
                            text += f" (pre -{start - global_frame})"
                        elif global_frame > end:
                            text += f" (post +{global_frame - end})"
                    cv2.putText(
                        out_frame,
                        text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
                
                # Draw yaw arrow and tag geometry for each tag in this frame when pose_csv is provided
                for row in pose_by_frame.get(global_frame, []):
                    center_x = float(row["center_x"])
                    center_y = float(row["center_y"])
                    yaw_deg = float(row["yaw_deg"])
                    pitch_deg = row["pitch_deg"] if row.get("pitch_deg", "") != "" else None
                    roll_deg = row["roll_deg"] if row.get("roll_deg", "") != "" else None
                    x_cam = row["x"] if row.get("x", "") != "" else None
                    y_cam = row["y"] if row.get("y", "") != "" else None
                    z_cam = row["z"] if row.get("z", "") != "" else None
                    tracked_flag = row.get("tracked", 0)
                    tag_id = row.get("tag_id", None)
                    try:
                        tracked_flag = int(tracked_flag)
                    except Exception:
                        tracked_flag = 0
                    # Pass 3D pose only when all values are present and non-NaN
                    def _f(v):
                        if v is None or (isinstance(v, float) and np.isnan(v)):
                            return None
                        return float(v)
                    # Color arrows differently when the tag location was tracked vs detected.
                    # BGR: detected=red, tracked=yellow.
                    arrow_color = (0, 255, 255) if tracked_flag == 1 else (0, 0, 255)

                    # Optionally draw the tag's quadrilateral boundary and per-corner indices
                    if has_corner_columns and draw_tag_overlay:
                        try:
                            c0 = (int(round(float(row["c0_x"]))), int(round(float(row["c0_y"]))))
                            c1 = (int(round(float(row["c1_x"]))), int(round(float(row["c1_y"]))))
                            c2 = (int(round(float(row["c2_x"]))), int(round(float(row["c2_y"]))))
                            c3 = (int(round(float(row["c3_x"]))), int(round(float(row["c3_y"]))))
                            corners = [c0, c1, c2, c3]
                            # Draw boundary lines in the same color as the yaw arrow
                            for p_start, p_end in zip(corners, corners[1:] + corners[:1]):
                                cv2.line(
                                    out_frame,
                                    p_start,
                                    p_end,
                                    arrow_color,
                                    thickness=2,
                                    lineType=cv2.LINE_AA,
                                )
                            # Label each corner with its index (0–3) near the corner location
                            for idx, (cx_i, cy_i) in enumerate(corners):
                                cv2.putText(
                                    out_frame,
                                    str(idx),
                                    (cx_i + 3, cy_i - 3),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.4,
                                    arrow_color,
                                    1,
                                    lineType=cv2.LINE_AA,
                                )
                        except Exception:
                            # If any corner values are missing or invalid, skip drawing corners for this tag.
                            pass

                    draw_yaw_arrow(
                        out_frame,
                        center_x,
                        center_y,
                        yaw_deg,
                        pitch_deg=_f(pitch_deg),
                        roll_deg=_f(roll_deg),
                        x_cam=_f(x_cam),
                        y_cam=_f(y_cam),
                        z_cam=_f(z_cam),
                        camera_intrinsics=camera_intrinsics,
                        axis_length_m=yaw_arrow_length_m,
                        arrow_length_px=yaw_arrow_length_px,
                        color=arrow_color,
                    )
                    # Draw the tag ID label near the tag center, similar to run_on_source.
                    if tag_id is not None:
                        try:
                            tag_label = str(int(tag_id))
                        except Exception:
                            tag_label = str(tag_id)
                        cv2.putText(
                            out_frame,
                            tag_label,
                            (int(round(center_x)) + 5, int(round(center_y)) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            arrow_color,
                            1,
                            lineType=cv2.LINE_AA,
                        )
                out_writer.write(out_frame)

        for w, p in writers:
            w.release()
            written.append(p)

    cap.release()
    return written


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Segment pose CSV into movement/rest by tag speed; optionally render segment videos."
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to the tag CSV (frame, tag_id, center_x, center_y).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.001,
        help="Speed threshold for movement (default: 0.001).",
    )
    parser.add_argument(
        "--min-segment-size",
        type=int,
        default=25,
        help="Minimum segment length in frames; smaller segments are discarded (default: 25).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot tag speeds over time.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the plot (use with --plot and -o).",
    )
    parser.add_argument(
        "-o", "--output-plot",
        type=Path,
        default=None,
        help="Save speed plot to this path.",
    )
    parser.add_argument(
        "--output-plot-segments",
        type=Path,
        default=None,
        help="Save speed plot with segments highlighted (after find_movement_segments).",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Video file to split into segment clips.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for segment videos (requires --video).",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=60,
        help="Frames to include before/after each segment (default: 60).",
    )
    parser.add_argument(
        "--pose-csv",
        type=Path,
        default=None,
        help="Pose CSV (frame, tag_id, center_x, center_y, yaw_deg, ...) to draw yaw arrows on video. Set to csv_path if not given but --video is.",
    )
    parser.add_argument(
        "--intrinsics",
        type=str,
        default=None,
        help="Camera intrinsics 'fx,fy,cx,cy' for 3D yaw arrow projection (optional).",
    )
    parser.add_argument(
        "--gradient-videos",
        action="store_true",
        help="Render gradient videos: two videos per segment (Sobel X and Sobel Y) with frames converted via gray_to_gradient_for_lk before overlays.",
    )
    parser.add_argument(
        "--no-tag-overlays",
        action="store_true",
        help="Do not draw tag borders and corner counts on frames.",
    )
    args = parser.parse_args()

    # If we are running video segmentation and no explicit pose CSV is provided,
    # default the pose CSV to the main input CSV path.
    if args.video is not None and args.pose_csv is None:
        args.pose_csv = args.csv_path

    speeds_df = compute_tag_speeds(
        args.csv_path,
        plot=args.plot,
        show_plot=not args.no_show,
        save_plot_path=args.output_plot,
        threshold=args.threshold,
    )
    print(f"Computed speeds for {len(speeds_df)} (frame, tag_id) pairs.")

    segments = find_movement_segments(
        speeds_df,
        args.threshold,
        min_segment_size=args.min_segment_size,
    )
    print(f"Found {len(segments)} movement segment(s) above threshold {args.threshold}:")
    for start, end in segments:
        print(f"  frames {start}–{end}")

    if args.plot:
        _plot_speeds(
            speeds_df,
            show_plot=not args.no_show,
            save_path=args.output_plot_segments,
            threshold=args.threshold,
            segments=segments,
        )

    if args.video is not None:
        out_dir = args.output_dir or (args.video.parent / "segments")
        camera_intrinsics = None
        if args.intrinsics is not None:
            parts = [p.strip() for p in args.intrinsics.split(",")]
            if len(parts) != 4:
                raise ValueError("--intrinsics must be 'fx,fy,cx,cy' (four numbers)")
            camera_intrinsics = tuple(float(p) for p in parts)
        paths = render_segment_videos(
            args.video,
            segments,
            out_dir,
            padding_frames=args.padding,
            show_count=True,
            pose_csv=args.pose_csv,
            camera_intrinsics=camera_intrinsics,
            gradient_videos=args.gradient_videos,
            draw_tag_overlay=not args.no_tag_overlays,
        )
        print(f"Wrote {len(paths)} video(s) to {out_dir}")


if __name__ == "__main__":
    main()
