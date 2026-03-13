"""
AprilTag detection from a video file or live camera.

REQUIRED CONFIGURATION:
-----------------------
- Tag family: This script uses "tag36h11" by default (most common).
  Other families supported by pupil-apriltags: tag25h9, tag16h5, tagCircle21h7,
  tagCircle49h12, tagStandard41h12, tagStandard52h13.
  Your physical tags must match the family you configure below.
  Generate tags at: https://chev.me/arucogen/ or April Robotics official repo.

- Input: Pass a video path for file, or use --camera [ID] for live camera.

REAL-TIME VIDEO LIBRARIES (for camera capture):
- OpenCV (cv2.VideoCapture): Already used here; works with USB webcams and many
  built-in cameras. Cross-platform, simple API. Best default choice.
- PyAV (av): FFmpeg bindings; more control over backends and codecs if you need
  a specific pipeline (e.g. hardware decode).
- picamera2: For Raspberry Pi Camera Module (official, Linux/RPi only).
- GStreamer (OpenCV backend CAP_GSTREAMER): Good on Linux for complex pipelines
  or low-latency capture; set cv2.VideoCapture(gst_pipeline_string).
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from pupil_apriltags import Detector
from tqdm import tqdm

from apriltag_tracking import (
    FrameDetections,
    detections_to_frame_detections,
    track_pose_detections_with_fallback,
)

"""
How to collect data from the video:
- The video is a series of movement, consisting of ballstic one, and fixed ones
- I could identify when the ballistic movements start by setting a threshold on the speed of movement of AprilTags (any)
- I could divide the set of movements into: each move, start and end; and the wait in between.
- Then I could collect data from the apriltags, between each point in time


"""

PRINT = False

# --- Configuration: match these to your setup ---
TAG_FAMILY = "tag36h11"  # Must match the April tag family printed on your tags
# For real-time: increase quad_decimate (e.g. 2.0) and nthreads; set refine_edges=1
DETECTOR_KWARGS = dict(
    families=TAG_FAMILY,
    nthreads=1,
    quad_decimate=1.0,   # 2 = run quad detection on 1/2 resolution (faster, slightly less accurate)
    quad_sigma=0,
    refine_edges=0, # outside of 0, improves detection (?) but reduces accuracy
    decode_sharpening=0.25, #extend of sharpening of image, set to 0 might help with computation?
    debug=0,
)


def get_detector():
    return Detector(**DETECTOR_KWARGS)


def frame_to_grayscale(frame: np.ndarray) -> np.ndarray:
    """Convert BGR frame to grayscale uint8. Required by pupil_apriltags."""
    if len(frame.shape) == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def detect_apriltags(detector: Detector, frame: np.ndarray) -> list:
    """
    Return list of detections; each has .center, .corners, .tag_id, .tag_family, etc.
    """
    gray = frame_to_grayscale(frame)
    return detector.detect(gray)


def detect_apriltags_with_pose(
    detector: Detector,
    frame: np.ndarray,
    camera_params: tuple[float, float, float, float],
    tag_size: float,
) -> list:
    """
    Detect AprilTags and estimate pose (position & orientation) using camera intrinsics.
    camera_params: (fx, fy, cx, cy). tag_size: physical tag size in meters.
    Each detection will have .pose_R, .pose_t, .pose_err when successful.
    """
    gray = frame_to_grayscale(frame)
    return detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=camera_params,
        tag_size=tag_size,
    )


def rotation_matrix_to_euler_deg(R: np.ndarray) -> tuple[float, float, float]:
    """
    Convert 3x3 rotation matrix to Euler angles (yaw, pitch, roll) in degrees.
    Convention: rotation order ZYX.
    """
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        yaw = np.arctan2(R[1, 0], R[0, 0])
        pitch = np.arctan2(-R[2, 0], sy)
        roll = np.arctan2(R[2, 1], R[2, 2])
    else:
        yaw = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        roll = 0.0
    return (
        np.degrees(yaw),
        np.degrees(pitch),
        np.degrees(roll),
    )


def euler_deg_to_rotation_matrix(
    yaw_deg: float, pitch_deg: float, roll_deg: float
) -> np.ndarray:
    """
    Convert Euler angles (yaw, pitch, roll) in degrees to a 3x3 rotation matrix.
    Convention: rotation order ZYX (same as rotation_matrix_to_euler_deg).
    So R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    """
    y = np.radians(yaw_deg)
    p = np.radians(pitch_deg)
    r = np.radians(roll_deg)
    cx, sx = np.cos(y), np.sin(y)
    cy, sy = np.cos(p), np.sin(p)
    cz, sz = np.cos(r), np.sin(r)
    # Rx(roll)
    Rx = np.array([
        [1, 0, 0],
        [0, cz, -sz],
        [0, sz, cz],
    ], dtype=np.float64)
    # Ry(pitch)
    Ry = np.array([
        [cy, 0, sy],
        [0, 1, 0],
        [-sy, 0, cy],
    ], dtype=np.float64)
    # Rz(yaw)
    Rz = np.array([
        [cx, -sx, 0],
        [sx, cx, 0],
        [0, 0, 1],
    ], dtype=np.float64)
    return Rz @ Ry @ Rx


def project_camera_to_image(
    x_cam: float, y_cam: float, z_cam: float,
    fx: float, fy: float, cx: float, cy: float,
) -> tuple[float, float] | None:
    """Project a 3D point in camera frame to pixel (u, v). Returns None if behind camera."""
    if z_cam <= 0:
        return None
    u = fx * x_cam / z_cam + cx
    v = fy * y_cam / z_cam + cy
    return (u, v)


def homography_to_rotation(
    H_cam_to_tag: np.ndarray,
    camera_intrinsics: tuple[float, float, float, float],
) -> np.ndarray:
    """
    Approximate rotation matrix from image->tag-plane homography and intrinsics.

    H_cam_to_tag maps image pixels -> tag-plane coords. We invert it to get the usual
    tag-plane -> image homography and then decompose:
        s * p_img = K [r1 r2 t] p_tag
    returning R = [r1 r2 r3] with r3 = r1 x r2, re-orthonormalized.
    """
    fx, fy, cx, cy = camera_intrinsics
    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    K_inv = np.linalg.inv(K)

    # Convert image->tag homography to tag->image for decomposition.
    H_tag_to_img = np.linalg.inv(H_cam_to_tag)
    h1 = H_tag_to_img[:, 0]
    h2 = H_tag_to_img[:, 1]

    Kinv_h1 = K_inv @ h1
    Kinv_h2 = K_inv @ h2
    lam = 1.0 / np.linalg.norm(Kinv_h1)
    r1 = lam * Kinv_h1
    r2 = lam * Kinv_h2
    r3 = np.cross(r1, r2)

    R_approx = np.column_stack((r1, r2, r3))
    # Orthonormalize to the closest proper rotation matrix.
    U, _, Vt = np.linalg.svd(R_approx)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0
        R = U @ Vt
    return R


def draw_yaw_arrow(
    frame: np.ndarray,
    center_x: float,
    center_y: float,
    yaw_deg: float,
    *,
    pitch_deg: Optional[float] = None,
    roll_deg: Optional[float] = None,
    x_cam: Optional[float] = None,
    y_cam: Optional[float] = None,
    z_cam: Optional[float] = None,
    camera_intrinsics: Optional[tuple[float, float, float, float]] = None,
    axis_length_m: float = 0.05,
    arrow_length_px: float = 40.0,
    color: tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
) -> None:
    """
    Draw the tag's yaw direction as an arrow on the image.

    If camera_intrinsics and full 3D pose (x_cam, y_cam, z_cam, yaw_deg, pitch_deg, roll_deg)
    are provided, the arrow is the projection of the tag's X axis (forward) in 3D, matching
    run_on_source. Otherwise, draw a 2D arrow from (center_x, center_y) in the direction
    given by yaw_deg in the image plane, with length arrow_length_px.

    Args:
        frame: BGR image to draw on (modified in place).
        center_x, center_y: Tag center in pixel coordinates.
        yaw_deg: Yaw angle in degrees.
        pitch_deg, roll_deg: Optional, for 3D projection.
        x_cam, y_cam, z_cam: Optional, tag position in camera frame (meters).
        camera_intrinsics: Optional (fx, fy, cx, cy).
        axis_length_m: Length of the 3D axis in meters (used for 3D projection).
        arrow_length_px: Length of the arrow in pixels (used for 2D fallback).
        color: BGR color for the arrow.
        thickness: Line thickness.
    """
    cx_pt = (int(round(center_x)), int(round(center_y)))
    use_3d = (
        camera_intrinsics is not None
        and len(camera_intrinsics) == 4
        and x_cam is not None
        and y_cam is not None
        and z_cam is not None
        and pitch_deg is not None
        and roll_deg is not None
    )
    if use_3d:
        fx, fy, cx_f, cy_f = camera_intrinsics
        R = euler_deg_to_rotation_matrix(yaw_deg, pitch_deg, roll_deg)
        t = np.array([x_cam, y_cam, z_cam], dtype=np.float64)
        tip_cam = t + axis_length_m * R[:, 0]
        pt = project_camera_to_image(
            float(tip_cam[0]), float(tip_cam[1]), float(tip_cam[2]),
            fx, fy, cx_f, cy_f,
        )
        if pt is not None:
            tip_pt = (int(round(pt[0])), int(round(pt[1])))
            cv2.arrowedLine(
                frame, cx_pt, tip_pt, color, thickness, tipLength=0.2, line_type=cv2.LINE_AA
            )
        return
    # 2D fallback: yaw in image plane (0 = right, 90 = down)
    rad = np.radians(yaw_deg)
    dx = arrow_length_px * np.cos(rad)
    dy = arrow_length_px * np.sin(rad)
    tip_pt = (int(round(center_x + dx)), int(round(center_y + dy)))
    cv2.arrowedLine(
        frame, cx_pt, tip_pt, color, thickness, tipLength=0.2, line_type=cv2.LINE_AA
    )


# Tests which ports have a camera available; taken from: https://stackoverflow.com/questions/57577445/list-available-cameras-opencv-python
def list_ports():
    """
    Test the ports and returns a tuple with the available ports and the ones that are working.
    """
    non_working_ports = []
    dev_port = 0
    working_ports = []
    available_ports = []
    while len(non_working_ports) < 6: # if there are more than 5 non working ports stop the testing. 
        camera = cv2.VideoCapture(dev_port)
        if not camera.isOpened():
            non_working_ports.append(dev_port)
            print("Port %s is not working." %dev_port)
        else:
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                print("Port %s is working and reads images (%s x %s)" %(dev_port,h,w))
                working_ports.append(dev_port)
            else:
                print("Port %s for camera ( %s x %s) is present but does not reads." %(dev_port,h,w))
                available_ports.append(dev_port)
        dev_port +=1
    return available_ports,working_ports,non_working_ports


def _write_pose_csv(out_path: Path, csv_rows: list[dict]) -> None:
    """Write pose rows to CSV using the standard field layout."""
    if not csv_rows:
        return
    fieldnames = [
        "frame",
        "tag_id",
        "tracked",
        "x",
        "y",
        "z",
        "yaw_deg",
        "pitch_deg",
        "roll_deg",
        "center_x",
        "center_y",
        "c0_x",
        "c0_y",
        "c1_x",
        "c1_y",
        "c2_x",
        "c2_y",
        "c3_x",
        "c3_y",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)


def _process_frame_and_collect(
    frame_idx: int,
    frame: np.ndarray,
    detector: Detector,
    *,
    use_pose: bool,
    camera_intrinsics: Optional[tuple[float, float, float, float]] = None,
    tag_size_m: Optional[float] = None,
    use_fallback_tracking: bool = False,
    prev_gray: Optional[np.ndarray] = None,
    prev_frame_dets: Optional[FrameDetections] = None,
    csv_rows: Optional[list[dict]] = None,
) -> tuple[list, Optional[np.ndarray], Optional[FrameDetections]]:
    """
    Detect tags in a frame, optionally perform fallback tracking, and append pose rows.

    Returns:
        detections, new_prev_gray, new_prev_frame_dets
    """
    gray = frame_to_grayscale(frame)

    if use_pose:
        detections = detect_apriltags_with_pose(
            detector, frame, camera_intrinsics, tag_size_m  # type: ignore[arg-type]
        )
        raw_next_fd = detections_to_frame_detections(
            detections,
            frame_idx=frame_idx,
            compute_homography=False,
            tag_size=float(tag_size_m) if tag_size_m is not None else 1.0,
        )
        if (
            use_fallback_tracking
            and prev_gray is not None
            and prev_frame_dets is not None
        ):
            frame_dets = track_pose_detections_with_fallback(
                prev_gray,
                prev_frame_dets,
                gray,
                raw_next_fd,
                frame_idx=frame_idx,
                tag_size=float(tag_size_m) if tag_size_m is not None else 1.0,
            )
            existing_ids = {int(d.tag_id) for d in detections}
            for tag_id, ts in frame_dets.tags.items():
                if tag_id in existing_ids:
                    continue
                # Synthetic detection without pose for tracked-only tags.
                detections.append(
                    type(
                        "Det",
                        (),
                        {
                            "tag_id": ts.tag_id,
                            "center": ts.center,
                            "corners": ts.corners,
                        },
                    )()
                )
            prev_frame_dets = frame_dets
        else:
            prev_frame_dets = raw_next_fd
    else:
        detections = detector.detect(gray)
        prev_frame_dets = None

    prev_gray = gray

    if use_pose and csv_rows is not None and prev_frame_dets is not None:
        for tag_id, ts in prev_frame_dets.tags.items():
            x, y = ts.center
            c = ts.corners  # 4x2 in pixel coords
            tracked_flag = 1 if ts.source == "tracked" else 0

            pose_R, pose_t = ts.pose_R, ts.pose_t
            x_cam = y_cam = z_cam = None
            yaw_deg = pitch_deg = roll_deg = None

            if pose_R is not None and pose_t is not None:
                # Direct pose from detector.
                t = np.asarray(pose_t, dtype=np.float64).reshape(-1)
                x_cam, y_cam, z_cam = float(t[0]), float(t[1]), float(t[2])
                yaw_deg, pitch_deg, roll_deg = rotation_matrix_to_euler_deg(
                    np.asarray(pose_R, dtype=np.float64).reshape(3, 3)
                )
            elif tracked_flag == 1 and ts.H_cam_to_tag is not None and camera_intrinsics is not None:
                # Tracked-only: no x,y,z; recover orientation from homography.
                R_from_H = homography_to_rotation(ts.H_cam_to_tag, camera_intrinsics)
                yaw_deg, pitch_deg, roll_deg = rotation_matrix_to_euler_deg(R_from_H)

            csv_rows.append({
                    "frame": frame_idx,
                    "tag_id": int(tag_id),
                    "tracked": tracked_flag,  # 0=detected, 1=tracked
                    "x": x_cam,
                    "y": y_cam,
                    "z": z_cam,
                    "yaw_deg": yaw_deg,
                    "pitch_deg": pitch_deg,
                    "roll_deg": roll_deg,
                    "center_x": float(x),
                    "center_y": float(y),
                    "c0_x": float(c[0][0]),
                    "c0_y": float(c[0][1]),
                    "c1_x": float(c[1][0]),
                    "c1_y": float(c[1][1]),
                    "c2_x": float(c[2][0]),
                    "c2_y": float(c[2][1]),
                    "c3_x": float(c[3][0]),
                    "c3_y": float(c[3][1]),
                })

            if PRINT and yaw_deg is not None:
                print(
                    f"Frame {frame_idx}: tag_id={tag_id} tracked={tracked_flag} center=({x:.1f}, {y:.1f}) "
                    f"pose x={x_cam:.4f} y={y_cam:.4f} z={z_cam:.4f} "
                    f"yaw={yaw_deg:.2f}° pitch={pitch_deg:.2f}° roll={roll_deg:.2f}°"
                )

    if not use_pose and PRINT:
        for d in detections:
            x, y = d.center
            print(
                f"Frame {frame_idx}: tag_id={d.tag_id} center=({x:.1f}, {y:.1f}) corners={d.corners.tolist()}"
            )

    return detections, prev_gray, prev_frame_dets


def run_on_source(
    source: str | Path | int,
    *,
    show: bool = True,
    max_frames: int | None = None,
    segments: Optional[list[tuple[int, int]]] = None,
    width: int | None = None,
    height: int | None = None,
    camera_intrinsics: Optional[tuple[float, float, float, float]] = None,
    tag_size_m: Optional[float] = None,
    output_csv: Optional[str | Path] = None,
    use_fallback_tracking: bool = False,
) -> None:
    """
    Run AprilTag detection on a video file or live camera.

    source: Path to a video file  (str or Path), or camera device index (int).
    max_frames: Number of frames before ending the process, defaults to None for videos and 100 for camera.
    segments: Optional list of (start, end) frame indices to analyze for video files only.
        Each tuple is interpreted as [start, end) in 0-based frame indices. When provided
        for a video source, segments take precedence over max_frames and one CSV is
        written per segment if output_csv is set.
    camera_intrinsics: Optional (fx, fy, cx, cy) for pose estimation. When set with tag_size_m, enables pose.
    tag_size_m: Physical AprilTag size in meters. When set with camera_intrinsics, enables pose estimation.
    output_csv: If set and pose estimation is enabled, writes position & angle per tag per frame to this CSV file.
        Adds column `tracked` where 0=detected, 1=tracked (fallback tracking).
    """
    use_pose = (
        camera_intrinsics is not None
        and len(camera_intrinsics) == 4
        and tag_size_m is not None
        and tag_size_m > 0
    )
    if output_csv and not use_pose:
        print("Warning: output_csv is ignored when pose estimation is not enabled (need camera_intrinsics and tag_size_m).", file=sys.stderr)

    is_camera = isinstance(source, int)
    if is_camera:
        if segments is not None:
            print("Error: segments are only supported for video files, not cameras.", file=sys.stderr)
            sys.exit(1)
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Error: could not open camera {source}", file=sys.stderr)
            sys.exit(1)
        if width is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        else:
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        if height is not None:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        else:
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        # Set max frame to stop processing at
        if max_frames is None: 
            max_frames = 100
        print(f"**Using camera with index {source}, {width}x{height} at {fps}fps.")
    else:
        path = Path(source)
        if not path.is_file():
            print(f"Error: file not found: {path}", file=sys.stderr)
            sys.exit(1)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            print(f"Error: could not open video: {path}", file=sys.stderr)
            sys.exit(1)

        total_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_in_video <= 0:
            # Some codecs don't report frame count; use max_frames or a large number for tqdm
            total_in_video = max_frames if max_frames is not None else 0
        if segments is not None:
            # Validate and normalize segments for video sources.
            cleaned_segments: list[tuple[int, int]] = []
            for start, end in segments:
                if start < 0 or end <= start:
                    print(f"Error: invalid segment ({start}, {end}); start must be >= 0 and end > start.",
                           file=sys.stderr,)
                    sys.exit(1)
                if start >= total_in_video:
                    print(f"Warning: segment ({start}, {end}) starts beyond video length ({total_in_video}); skipping.",
                         file=sys.stderr,)
                    continue
                if end > total_in_video:
                    print(f"Warning: segment ({start}, {end}) extends beyond video length ({total_in_video}); clamping end to {total_in_video}.",
                          file=sys.stderr,)
                    end = total_in_video
                cleaned_segments.append((start, end))
            if not cleaned_segments:
                print("Error: no valid segments to process after validation.", file=sys.stderr)
                sys.exit(1)
            cleaned_segments.sort(key=lambda s: s[0])
            segments = cleaned_segments
            if max_frames is not None:
                print("Warning: segments specified for video; ignoring max_frames.",
                    file=sys.stderr,)
                max_frames = None
            total_for_tqdm = sum(end - start for start, end in segments)
        else:
            if max_frames is not None:
                total_for_tqdm = min(total_in_video, max_frames)
            else:
                total_for_tqdm = total_in_video

    detector = get_detector()
    window_name = "AprilTags (camera)" if is_camera else "AprilTags"
    # Shared state for processing
    prev_gray: Optional[np.ndarray] = None
    prev_frame_dets: Optional[FrameDetections] = None
    stop_requested = False

    # Non-segmented path (camera or whole video)
    def _process_frames(
        cap,
        detector,
        use_pose,
        camera_intrinsics,
        tag_size_m,
        use_fallback_tracking,
        show,
        window_name,
        output_csv,
        csv_rows,
        segment_range=None,
        max_frames=None,
        pbar=None,
        stop_requested=False,
    ):
        """
        Unified frame processing loop for both simple and segmented video.
        - segment_range: (start, end) tuple for range of frames to process; or None for unsegmented.
        - max_frames: max number of frames to process (applied only for unsegmented).
        - csv_rows: list to which results are appended.
        """
        prev_gray = None
        prev_frame_dets = None

        if segment_range is not None:
            start, end = segment_range
            frame_idx = start
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            def should_continue(i): return i < end
        else:
            frame_idx = 0
            def should_continue(i): return True if (max_frames is None) else i < max_frames

        while should_continue(frame_idx):
            ret, frame = cap.read()
            if not ret:
                break

            dets, prev_gray, prev_frame_dets = _process_frame_and_collect(
                frame_idx,
                frame,
                detector,
                use_pose=use_pose,
                camera_intrinsics=camera_intrinsics,
                tag_size_m=tag_size_m,
                use_fallback_tracking=use_fallback_tracking,
                prev_gray=prev_gray,
                prev_frame_dets=prev_frame_dets,
                csv_rows=csv_rows if (use_pose and output_csv) else None,
            )

            if show:
                for d in dets:
                    pts = d.corners.astype(np.int32)
                    cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
                    cx, cy = int(d.center[0]), int(d.center[1])
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    if use_pose and hasattr(d, "pose_t") and d.pose_t is not None:
                        fx, fy, cx_f, cy_f = camera_intrinsics
                        t = d.pose_t.flatten()
                        R = d.pose_R
                        axis_length = 0.05
                        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
                        for i in range(3):
                            tip_cam = t + axis_length * R[:, i]
                            pt = project_camera_to_image(
                                float(tip_cam[0]), float(tip_cam[1]), float(tip_cam[2]),
                                fx, fy, cx_f, cy_f,
                            )
                            if pt is not None:
                                tip_pt = (int(round(pt[0])), int(round(pt[1])))
                                cv2.arrowedLine(
                                    frame,
                                    (cx, cy),
                                    tip_pt,
                                    colors[i],
                                    2,
                                    tipLength=0.2,
                                )
                        label = str(d.tag_id)
                    else:
                        label = str(d.tag_id)
                    cv2.putText(
                        frame,
                        label,
                        (cx + 5, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop_requested = True
                    break

            frame_idx += 1
            if pbar is not None:
                pbar.update(1)

        return stop_requested, frame_idx

    if segments is None:
        csv_rows: list[dict] = []  # for output_csv when use_pose
        pbar = None
        if not is_camera:
            pbar = tqdm(total=total_for_tqdm, unit="frame", desc="Detecting AprilTags")
        stop_requested, _ = _process_frames(
            cap,
            detector,
            use_pose,
            camera_intrinsics,
            tag_size_m,
            use_fallback_tracking,
            show,
            window_name,
            output_csv,
            csv_rows,
            segment_range=None,
            max_frames=max_frames,
            pbar=pbar,
        )
        if pbar is not None:
            pbar.close()

        cap.release()
        if show:
            cv2.destroyAllWindows()

        if output_csv and csv_rows:
            out_path = Path(output_csv)
            _write_pose_csv(out_path, csv_rows)
            print(f"Pose data written to {out_path} ({len(csv_rows)} rows).")
        return

    # Segmented video path
    if output_csv is None:
        print("Error: segments requires output_csv to be set to write per-segment CSVs.", file=sys.stderr)
        sys.exit(1)

    pbar = tqdm(total=total_for_tqdm, unit="frame", desc="Detecting AprilTags") if total_for_tqdm else None

    for start, end in segments:
        csv_rows_segment: list[dict] = []
        stop_requested, last_frame_idx = _process_frames(
            cap,
            detector,
            use_pose,
            camera_intrinsics,
            tag_size_m,
            use_fallback_tracking,
            show,
            window_name,
            output_csv,
            csv_rows_segment,
            segment_range=(start, end),
            max_frames=None,
            pbar=pbar,
        )
        # Write CSV for this segment
        if csv_rows_segment:
            base_path = Path(output_csv)
            segment_out = base_path.with_name(f"{base_path.stem}_{start}-{end}{base_path.suffix}")
            _write_pose_csv(segment_out, csv_rows_segment)
            print(f"Pose data written to {segment_out} ({len(csv_rows_segment)} rows).")

        if stop_requested:
            break

    if pbar is not None:
        pbar.close()

    cap.release()
    if show:
        cv2.destroyAllWindows()


def analyze_video_apriltags(
    video_path: str | Path,
    tag_ids: Optional[list[int]] = None,
    output_txt: Optional[str | Path] = None,
    max_frames: Optional[int] = None,
) -> dict:
    """
    Run AprilTag detection on all video frames without rendering.
    Uses tqdm for progress. Tracks which frames contain which tags and writes results to a txt file.

    Args:
        video_path: Path to the video file.
        tag_ids: Optional list of tag IDs to focus on. If None, all detected tags are tracked.
        output_txt: Path for the output txt file. If None, uses video_path with .txt extension.
        max_frames: If set, only analyze the first N frames.

    Returns:
        Dict with:
          - total_frames: int (number of frames actually analyzed)
          - frame_to_tags: dict[frame_index, list[tag_id]]
          - tag_frame_count: dict[tag_id, int]  (how many frames contain that tag)
          - frames_missing_tag: dict[tag_id, list[frame_index]]  (frames where tag is not present)
    """
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {path}")

    total_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_in_video <= 0:
        # Some codecs don't report frame count; use max_frames or a large number for tqdm
        total_in_video = max_frames if max_frames is not None else 0
    if max_frames is not None:
        total_for_tqdm = min(total_in_video, max_frames)
    else:
        total_for_tqdm = total_in_video

    detector = get_detector()
    frame_to_tags: dict[int, list[int]] = {}

    with tqdm(total=total_for_tqdm, unit="frame", desc="Detecting AprilTags") as pbar:
        frame_idx = 0
        while True:
            if max_frames is not None and frame_idx >= max_frames:
                break
            ret, frame = cap.read()
            if not ret:
                break
            gray = frame_to_grayscale(frame)
            detections = detector.detect(gray)
            tag_ids_in_frame = [d.tag_id for d in detections]
            frame_to_tags[frame_idx] = tag_ids_in_frame
            frame_idx += 1
            pbar.update(1)

    cap.release()

    total_frames = len(frame_to_tags)  # number of frames actually analyzed

    all_tag_ids: set[int] = set()
    for tags in frame_to_tags.values():
        all_tag_ids.update(tags)

    if tag_ids is not None:
        focus_tags = set(tag_ids)
        # Include focus tags even if never detected (so we report "missing" for all frames)
        tags_to_report = focus_tags
    else:
        tags_to_report = all_tag_ids

    tag_frame_count: dict[int, int] = {tid: 0 for tid in tags_to_report}
    frames_missing_tag: dict[int, list[int]] = {tid: [] for tid in tags_to_report}

    for fid, tags_in_frame in frame_to_tags.items():
        for tid in tags_to_report:
            if tid in tags_in_frame:
                tag_frame_count[tid] += 1
            else:
                frames_missing_tag[tid].append(fid)

    result = {
        "total_frames": total_frames,
        "frame_to_tags": frame_to_tags,
        "tag_frame_count": tag_frame_count,
        "frames_missing_tag": frames_missing_tag,
    }

    out_path = Path(output_txt) if output_txt else path.parent / (path.stem + "_apriltags.txt")
    _write_analysis_txt(
        out_path,
        total_frames=total_frames,
        tag_frame_count=result["tag_frame_count"],
        frames_missing_tag=result["frames_missing_tag"],
        frame_to_tags=result["frame_to_tags"],
        focus_tag_ids=tag_ids,
    )
    print(f"Results written to {out_path}")

    return result


def _write_analysis_txt(
    path: Path,
    total_frames: int,
    tag_frame_count: dict[int, int],
    frames_missing_tag: dict[int, list[int]],
    frame_to_tags: dict[int, list[int]],
    focus_tag_ids: Optional[list[int]] = None,
) -> None:
    """Write analysis summary and per-tag stats to a text file."""
    lines = [
        "AprilTag video analysis",
        "=" * 60,
        f"Total frames: {total_frames}",
        "",
    ]
    if focus_tag_ids is not None:
        lines.append(f"Focus tags (requested): {sorted(focus_tag_ids)}")
        lines.append("")
    lines.append("Stats: frames containing each tag (out of all frames)")
    lines.append("-" * 60)
    for tid in sorted(tag_frame_count.keys()):
        count = tag_frame_count[tid]
        pct = (100.0 * count / total_frames) if total_frames else 0
        lines.append(f"  Tag {tid}: {count}/{total_frames} frames ({pct:.1f}%)")
    lines.append("")
    lines.append("Frames missing each tag (frame indices 0-based)")
    lines.append("-" * 60)
    for tid in sorted(frames_missing_tag.keys()):
        missing = frames_missing_tag[tid]
        if len(missing) <= 30:
            lines.append(f"  Tag {tid}: {missing}")
        else:
            lines.append(f"  Tag {tid}: {len(missing)} frames — first 20: {missing[:20]} ... last 5: {missing[-5:]}")
    lines.append("")
    lines.append("Per-frame tag presence (frame_index -> tag IDs)")
    lines.append("-" * 60)
    for fid in sorted(frame_to_tags.keys()):
        tags = sorted(frame_to_tags[fid])
        lines.append(f"  Frame {fid}: {tags}")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Detect AprilTags in a video file or live camera.",
    )
    parser.add_argument(
        "video",
        nargs="?",
        help="Path to video file (e.g. input.mov). Omit if using --camera.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        nargs="?",
        const=0,
        metavar="ID",
        help="Use live camera instead of file. Optional ID (default 0).",
    )
    parser.add_argument("--no-show", action="store_true", help="Disable display window")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames (default: 100 for camera; for --analyze, only analyze first N frames).")
    parser.add_argument("--width", type=int, default=None, help="Camera frame width (e.g. 640)")
    parser.add_argument("--height", type=int, default=None, help="Camera frame height")
    parser.add_argument(
        "--nthreads",
        type=int,
        default=None,
        help="Override AprilTag detector nthreads (default from DETECTOR_KWARGS).",
    )
    parser.add_argument(
        "--quad-sigma",
        type=float,
        default=None,
        help="Override AprilTag detector quad_sigma (default from DETECTOR_KWARGS).",
    )
    parser.add_argument(
        "--decode-sharpening",
        type=float,
        default=None,
        help="Override AprilTag detector decode_sharpening (default from DETECTOR_KWARGS).",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run headless analysis: detect tags on all frames (no rendering), show tqdm progress, write stats to a txt file.",
    )
    parser.add_argument(
        "--tags",
        type=int,
        nargs="*",
        default=None,
        metavar="ID",
        help="Optional list of tag IDs to focus on (for --analyze). Stats and missing-frame lists are reported only for these tags.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="PATH",
        help="Output txt path for --analyze (default: <video>_apriltags.txt).",
    )
    parser.add_argument(
        "--intrinsics",
        type=str,
        default=None,
        metavar="fx,fy,cx,cy",
        help="Camera intrinsics for pose estimation (e.g. 1371,1371,960,540). Requires --tag-size.",
    )
    parser.add_argument(
        "--tag-size",
        type=float,
        default=None,
        metavar="M",
        help="AprilTag size in meters for pose estimation. Requires --intrinsics.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        metavar="PATH",
        help="Write pose (frame, tag_id, tracked, x, y, z, yaw_deg, pitch_deg, roll_deg) to CSV. Requires pose estimation (--intrinsics and --tag-size).",
    )
    parser.add_argument(
        "--fallback-tracking",
        action="store_true",
        help="When pose estimation is OFF, recover missing tags using per-tag optical-flow tracking between frames.",
    )
    parser.add_argument(
        "--segments",
        type=str,
        default=None,
        metavar="LIST",
        help=(
            "Optional list of frame segments for video files only, in the form "
            "'start1:end1,start2:end2'. Each segment is [start, end) in 0-based frame indices. "
            "When provided, segments take precedence over --max-frames and one CSV is written per segment."
        ),
    )
    args = parser.parse_args()

    # Apply optional detector overrides before creating any Detector instances.
    if args.nthreads is not None:
        DETECTOR_KWARGS["nthreads"] = int(args.nthreads)
    if args.quad_sigma is not None:
        DETECTOR_KWARGS["quad_sigma"] = float(args.quad_sigma)
    if args.decode_sharpening is not None:
        DETECTOR_KWARGS["decode_sharpening"] = float(args.decode_sharpening)

    camera_intrinsics: Optional[tuple[float, float, float, float]] = None
    if args.intrinsics:
        parts = [p.strip() for p in args.intrinsics.split(",")]
        if len(parts) != 4:
            print("Error: --intrinsics must be fx,fy,cx,cy (four numbers).", file=sys.stderr)
            sys.exit(1)
        try:
            camera_intrinsics = (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))
        except ValueError:
            print("Error: --intrinsics must be four numbers.", file=sys.stderr)
            sys.exit(1)

    if args.analyze:
        video = args.video
        if not video:
            print("Error: --analyze requires a video path.", file=sys.stderr)
            sys.exit(1)
        if args.camera is not None:
            print("Error: --analyze cannot be used with --camera.", file=sys.stderr)
            sys.exit(1)
        try:
            analyze_video_apriltags(
                video_path=video,
                tag_ids=args.tags,
                output_txt=args.output,
                max_frames=args.max_frames,
            )
        except (FileNotFoundError, IOError) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        return

    segments: Optional[list[tuple[int, int]]] = None
    if args.segments:
        try:
            segments_list: list[tuple[int, int]] = []
            for part in args.segments.split(","):
                part = part.strip()
                if not part:
                    continue
                start_str, end_str = part.split(":")
                start = int(start_str)
                end = int(end_str)
                if start < 0 or end <= start:
                    raise ValueError
                segments_list.append((start, end))
            if not segments_list:
                raise ValueError
            segments = segments_list
        except ValueError:
            print(
                "Error: --segments must be of the form 'start1:end1,start2:end2' with start >= 0 and end > start.",
                file=sys.stderr,
            )
            sys.exit(1)

    if args.camera is not None:
        run_on_source(
            args.camera,
            show=not args.no_show,
            max_frames=args.max_frames,
            width=args.width,
            height=args.height,
            camera_intrinsics=camera_intrinsics,
            tag_size_m=args.tag_size,
            output_csv=args.output_csv,
            use_fallback_tracking=args.fallback_tracking,
            segments=segments,
        )
    else:
        run_on_source(
            args.video or "input.mov",
            show=not args.no_show,
            max_frames=args.max_frames,
            camera_intrinsics=camera_intrinsics,
            tag_size_m=args.tag_size,
            output_csv=args.output_csv,
            use_fallback_tracking=args.fallback_tracking,
            segments=segments,
        )


if __name__ == "__main__":
    # (1371.02, 1371.02, 960.00, 540.00)
    main()