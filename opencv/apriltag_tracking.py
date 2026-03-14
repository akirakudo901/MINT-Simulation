from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, Optional, Tuple

import cv2
import numpy as np
from pupil_apriltags import Detector


Corners = np.ndarray  # shape (4, 2), float32/float64 in pixel coords
Homography = np.ndarray  # shape (3, 3), float64
Center = Tuple[float, float]


@dataclass(frozen=True)
class TagState:
    tag_id: int
    corners: Corners  # (4, 2) in pixels, order as returned by pupil_apriltags
    center: Center
    # Optional image->tag-plane homography; may be unused when pose is available.
    H_cam_to_tag: Optional[Homography] = None
    # Optional full pose from pupil_apriltags (camera frame).
    pose_R: Optional[np.ndarray] = None  # 3x3
    pose_t: Optional[np.ndarray] = None  # 3x1 or (3,)
    pose_err: Optional[float] = None
    # "detected" | "tracked"
    source: str = "detected"
    confidence: float = 1.0


@dataclass(frozen=True)
class FrameDetections:
    frame_idx: int
    tags: Dict[int, TagState]


def _yaw_rad_from_rotation_matrix(R: np.ndarray) -> float:
    """Extract yaw (rotation around Z) from 3x3 rotation matrix. Returns radians."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-9:
        return float(np.arctan2(R[1, 0], R[0, 0]))
    return float(np.arctan2(-R[1, 2], R[1, 1]))


def _yaw_rad_from_tag_state(tag: TagState) -> float:
    """Yaw in radians: from pose_R if available, else image-plane angle of first edge."""
    if tag.pose_R is not None:
        return _yaw_rad_from_rotation_matrix(tag.pose_R)
    corners = np.asarray(tag.corners, dtype=np.float64).reshape(4, 2)
    return float(np.arctan2(corners[1, 1] - corners[0, 1], corners[1, 0] - corners[0, 0]))


class MotionHistory:
    """
    Per-tag motion history used to extrapolate an initial flow guess for LK tracking.
    Stores the last maxlen (frame_idx, center_xy, yaw_rad) per tag. Extrapolation
    uses velocity (center displacement / frame) and angular velocity (delta_yaw / frame).
    """

    __slots__ = ("_per_tag", "_maxlen")

    def __init__(self, maxlen: int = 5):
        self._maxlen = max(2, maxlen)
        self._per_tag: Dict[int, Deque[Tuple[int, np.ndarray, float]]] = {}

    def update(self, frame_detections: FrameDetections, frame_idx: int) -> None:
        """Append (frame_idx, center, yaw_rad) for each tag in frame_detections."""
        for tag_id, tag in frame_detections.tags.items():
            center = np.asarray([tag.center[0], tag.center[1]], dtype=np.float64)
            yaw = _yaw_rad_from_tag_state(tag)
            if tag_id not in self._per_tag:
                self._per_tag[tag_id] = deque(maxlen=self._maxlen)
            self._per_tag[tag_id].append((frame_idx, center, yaw))

    def get_predicted_corners(
        self,
        tag_id: int,
        prev_corners: np.ndarray,
        prev_center: np.ndarray,
        prev_frame_idx: int,
        next_frame_idx: int,
    ) -> Optional[np.ndarray]:
        """
        Predict corners at next_frame_idx assuming motion along an arc of unknown
        radius. Rotation is taken from yaw change; radius is solved from the
        isosceles triangle (apex angle = yaw change, base = displacement between
        first and last history). Center of rotation is placed on the side of the
        chord given by the sign of yaw change. Returns (4, 2) float64 or None.
        """
        history = self._per_tag.get(tag_id)
        if not history or len(history) < 2:
            return None
        # Use last entry as "current" (prev frame); ensure it matches prev_frame_idx
        f_last, c_last, y_last = history[-1]
        if f_last != prev_frame_idx:
            return None
        prev_corners = np.asarray(prev_corners, dtype=np.float64).reshape(4, 2)
        prev_center = np.asarray(prev_center, dtype=np.float64).reshape(2)

        # Velocity: (last_center - first_center) / delta_frames (robust over K frames)
        frames = list(history)
        f0, c0, y0 = frames[0]
        
        c0, c_last = np.asarray(c0, dtype=np.float64), np.asarray(c_last, dtype=np.float64)
        delta_f = float(f_last - f0)
        if delta_f <= 0:
            return None

        yaw_diff = float(np.asarray(y_last, dtype=np.float64) - np.asarray(y0, dtype=np.float64))
        yaw_diff = np.arctan2(np.sin(yaw_diff), np.cos(yaw_diff))
        omega = yaw_diff / delta_f
        dt = float(next_frame_idx - prev_frame_idx)
        angle = omega * dt
        c, s = np.cos(angle), np.sin(angle)
        R_2d = np.array([[c, -s], [s, c]], dtype=np.float64)

        # Arc-based center prediction
        d = float(np.linalg.norm(c_last - c0))
        theta_abs = abs(yaw_diff)
        if theta_abs < 1e-9 or d < 1e-9:
            # Near-linear or no translation: use constant-velocity center
            velocity = (c_last - c0) / delta_f
            predicted_center = prev_center + velocity * dt
        else:
            # Isosceles: apex angle = theta, base = d => chord = 2*R*sin(theta/2) => R = d / (2*sin(theta/2))
            sin_half = np.sin(theta_abs / 2.0)
            if sin_half < 1e-9:
                velocity = (c_last - c0) / delta_f
                predicted_center = prev_center + velocity * dt
            else:
                R = d / (2.0 * sin_half)
                M = (c0 + c_last) * 0.5
                V = c_last - c0
                half_chord_sq = (d / 2.0) ** 2
                h_sq = R * R - half_chord_sq
                h = np.sqrt(max(0.0, h_sq))
                perp = np.array([-V[1], V[0]], dtype=np.float64)
                perp_norm = np.linalg.norm(perp)
                if perp_norm < 1e-12:
                    velocity = (c_last - c0) / delta_f
                    predicted_center = prev_center + velocity * dt
                else:
                    n = (np.sign(yaw_diff) if yaw_diff != 0 else 1.0) * perp / perp_norm
                    O = M + h * n
                    predicted_center = O + (R_2d @ (prev_center - O))

        offsets = (prev_corners - prev_center) @ R_2d.T
        predicted_corners = predicted_center + offsets
        return predicted_corners.astype(np.float64)


def detections_to_frame_detections(
    detections: Iterable,
    *,
    frame_idx: int,
    compute_homography: bool = False,
    tag_size: float = 1.0,
) -> FrameDetections:
    """
    Convert a sequence of pupil_apriltags detections into FrameDetections, without
    computing any homographies or poses.
    """
    tags: Dict[int, TagState] = {}
    for d in detections:
        corners = np.asarray(d.corners, dtype=np.float64).reshape(4, 2)
        center = (float(d.center[0]), float(d.center[1]))

        # If pose is available (from pupil_apriltags pose mode), prefer that over
        # computing a homography. This "replaces" the homography with the pose.
        pose_R: Optional[np.ndarray] = None
        pose_t: Optional[np.ndarray] = None
        pose_err: Optional[float] = None
        if hasattr(d, "pose_R") and getattr(d, "pose_R") is not None:
            pose_R = np.asarray(d.pose_R, dtype=np.float64).reshape(3, 3)
            if hasattr(d, "pose_t") and getattr(d, "pose_t") is not None:
                pose_t = np.asarray(d.pose_t, dtype=np.float64).reshape(-1)
            if hasattr(d, "pose_err"):
                try:
                    pose_err = float(d.pose_err)
                except Exception:
                    pose_err = None

        H: Optional[Homography]
        conf: float
        if pose_R is not None:
            # Confidence based on pose error if available; fall back to 1.0.
            if pose_err is not None:
                conf = float(np.exp(-abs(pose_err)))
            else:
                conf = 1.0
            H = None
        elif compute_homography:
            H, rmse = _fit_and_score_homography_cam_to_tag(
                corners, tag_size=tag_size
            )
            conf = float(np.exp(-rmse))
        else:
            H = None
            conf = 1.0
        tags[int(d.tag_id)] = TagState(
            tag_id=int(d.tag_id),
            corners=corners,
            center=center,
            H_cam_to_tag=H,
            pose_R=pose_R,
            pose_t=pose_t,
            pose_err=pose_err,
            source="detected",
            confidence=conf,
        )
    return FrameDetections(frame_idx=frame_idx, tags=tags)


def _tag_corners_tagcoords(tag_size: float = 1.0) -> np.ndarray:
    # Canonical tag-plane coordinates for the 4 corners.
    # We use a unit square by default; tag_size scales it.
    s = float(tag_size)
    return np.array([[0.0, 0.0], [s, 0.0], [s, s], [0.0, s]], dtype=np.float64)


def homography_cam_to_tag_from_corners(
    corners_px: np.ndarray, *, tag_size: float = 1.0
) -> Homography:
    """
    Build H_cam_to_tag from the detected pixel-space corners.
    H maps image pixels (camera coords) -> tag-plane coords.
    """
    img = np.asarray(corners_px, dtype=np.float64).reshape(4, 2)
    tag = _tag_corners_tagcoords(tag_size=tag_size)
    # OpenCV expects float32; getPerspectiveTransform gives tag<-img transform.
    H = cv2.getPerspectiveTransform(img.astype(np.float32), tag.astype(np.float32))
    return H


def _apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    ph = np.concatenate([pts, ones], axis=1)  # (N,3)
    q = (H @ ph.T).T
    q = q[:, :2] / q[:, 2:3]
    return q


def _is_convex_quad(pts: np.ndarray) -> bool:
    pts = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    # Convex if cross products of successive edges have same sign.
    def cross_z(a: np.ndarray, b: np.ndarray) -> float:
        return float(a[0] * b[1] - a[1] * b[0])

    signs = []
    for i in range(4):
        p0 = pts[i]
        p1 = pts[(i + 1) % 4]
        p2 = pts[(i + 2) % 4]
        v1 = p1 - p0
        v2 = p2 - p1
        signs.append(cross_z(v1, v2))
    # Allow consistent sign with some tolerance
    pos = sum(s > 1e-9 for s in signs)
    neg = sum(s < -1e-9 for s in signs)
    return (pos == 4) or (neg == 4)


def _quad_area(pts: np.ndarray) -> float:
    pts = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    return float(abs(cv2.contourArea(pts.astype(np.float32))))


def _fit_and_score_homography_cam_to_tag(
    corners_px: np.ndarray, *, tag_size: float = 1.0
) -> Tuple[Optional[np.ndarray], float]:
    """
    Returns (H_cam_to_tag, reprojection_rmse_in_tagcoords).
    Lower RMSE is better.
    """
    corners_px = np.asarray(corners_px, dtype=np.float64).reshape(4, 2)
    tag = _tag_corners_tagcoords(tag_size=tag_size)
    H = homography_cam_to_tag_from_corners(corners_px, tag_size=tag_size)
    pred_tag = _apply_homography(H, corners_px)
    err = pred_tag - tag
    rmse = float(np.sqrt(np.mean(np.sum(err * err, axis=1))))
    return H, rmse


def detect_to_frame_detections(
    detector: Detector,
    frame_gray: np.ndarray,
    *,
    frame_idx: int,
    tag_size: float = 1.0,
) -> FrameDetections:
    dets = detector.detect(frame_gray)
    tags: Dict[int, TagState] = {}
    for d in dets:
        corners = np.asarray(d.corners, dtype=np.float64).reshape(4, 2)
        center = (float(d.center[0]), float(d.center[1]))
        H, rmse = _fit_and_score_homography_cam_to_tag(corners, tag_size=tag_size)
        # Convert rmse to a coarse confidence in [0,1]
        conf = float(np.exp(-rmse))
        tags[int(d.tag_id)] = TagState(
            tag_id=int(d.tag_id),
            corners=corners,
            center=center,
            H_cam_to_tag=H,
            source="detected",
            confidence=conf,
        )
    return FrameDetections(frame_idx=frame_idx, tags=tags)


def gray_to_gradient_for_lk(
    gray: np.ndarray,
    *,
    mode: str = "xy",
    ksize: int = 3,
    blur_sigma: float = 0.0,
) -> np.ndarray:
    """
    Preprocess grayscale image for LK tracking so that edges (white on one side,
    black on the other) are matched to points with the same gradient property.

    LK minimizes intensity difference in a window. By passing gradient images
    instead of raw intensity, "brightness constancy" becomes gradient constancy:
    a point is matched where the gradient (direction and sense) is the same.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale image (uint8 or float).
    mode : str
        - "xy": 2-channel image (I_x, I_y). Matches full gradient vector;
          any edge direction is matched by same orientation and sense.
        - "x": 1-channel I_x. Strong for horizontal edges (white-left/black-right
          vs black-left/white-right).
        - "y": 1-channel I_y. Strong for vertical edges.
    ksize : int
        Sobel kernel size (1, 3, 5, or 7).
    blur_sigma : float
        If > 0, apply Gaussian blur before Sobel to reduce noise (e.g. 0.5–1.0).

    Returns
    -------
    np.ndarray
        uint8 image, 1 or 2 channels. Gradient values are mapped to [0, 255]
        with zero gradient at 128 so same gradient → same value.
    """
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if blur_sigma > 0:
        gray = cv2.GaussianBlur(gray, (0, 0), blur_sigma)
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)
    # Map gradient to [0, 255] with zero at 128 (Sobel on uint8 is in ~[-1020, 1020])
    scale = 127.0 / max(1e-6, (255.0 * 4))  # 4 from typical 3x3 Sobel scale
    def to_uint8(g: np.ndarray) -> np.ndarray:
        return np.clip(128 + g * scale, 0, 255).astype(np.uint8)
    if mode == "x":
        return to_uint8(sobel_x)
    if mode == "y":
        return to_uint8(sobel_y)
    if mode == "xy":
        return np.stack([to_uint8(sobel_x), to_uint8(sobel_y)], axis=-1)
    raise ValueError("mode must be 'x', 'y', or 'xy'")


def _lk_track_points(
    prev_gray: np.ndarray,
    next_gray: np.ndarray,
    pts_prev: np.ndarray,
    *,
    next_pts_initial: Optional[np.ndarray] = None,
    win_size: Tuple[int, int] = (31, 31),
    max_level: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    p0 = np.asarray(pts_prev, dtype=np.float32).reshape(-1, 1, 2)
    if next_pts_initial is not None:
        p1_init = np.asarray(next_pts_initial, dtype=np.float32).reshape(-1, 1, 2)
        flags = cv2.OPTFLOW_USE_INITIAL_FLOW
    else:
        p1_init = None
        flags = 0
    p1, st, err = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        next_gray,
        p0,
        p1_init,
        winSize=win_size,
        maxLevel=max_level,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        flags=flags,
    )
    return p1.reshape(-1, 2).astype(np.float64), st.reshape(-1), err.reshape(-1)


def _refine_corners_subpix(
    gray: np.ndarray, corners: np.ndarray, *, win: Tuple[int, int] = (7, 7)
) -> np.ndarray:
    pts = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)
    # cornerSubPix requires float32 image
    img = gray
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # Slight smoothing helps in light blur/noise
    img = cv2.GaussianBlur(img, (3, 3), 0)
    refined = cv2.cornerSubPix(
        img,
        pts,
        winSize=win,
        zeroZone=(-1, -1),
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.001),
    )
    return refined.reshape(-1, 2).astype(np.float64)


def _recover_missing_tags_with_tracking(
    prev_frame_gray: np.ndarray,
    prev_detections: FrameDetections,
    next_frame_gray: np.ndarray,
    raw_next: FrameDetections,
    *,
    frame_idx: int,
    tag_size: float = 1.0,
    max_corner_err_px: float = 25.0,
    min_area_px2: float = 25.0,
    max_tagcoord_rmse: float = 0.25,
    motion_history: Optional[MotionHistory] = None,
    prev_frame_idx: Optional[int] = None,
    lk_win_size: Tuple[int, int] = (31, 31),
    lk_gradient_preprocess: bool = False,
) -> FrameDetections:
    """
    Generic helper for both fallback tracking modes.

    Given previous-frame detections and raw detections for the next frame,
    fill in tags that went missing by tracking their corners with LK optical flow.
    When motion_history and prev_frame_idx are provided, uses velocity and angular
    velocity from recent frames to supply an initial flow guess (OPTFLOW_USE_INITIAL_FLOW).
    """
    prev_ids = set(prev_detections.tags.keys())
    next_ids = set(raw_next.tags.keys())
    missing = sorted(prev_ids - next_ids)
    if not missing:
        return raw_next

    if lk_gradient_preprocess:
        prev_lk = gray_to_gradient_for_lk(prev_frame_gray, mode="xy")
        next_lk = gray_to_gradient_for_lk(next_frame_gray, mode="xy")
    else:
        prev_lk = prev_frame_gray
        next_lk = next_frame_gray

    final_tags = dict(raw_next.tags)
    next_frame_idx = frame_idx
    use_initial_flow = (
        motion_history is not None
        and prev_frame_idx is not None
        and prev_frame_idx < next_frame_idx
    )

    for tag_id in missing:
        prev_tag = prev_detections.tags.get(tag_id)
        if prev_tag is None:
            continue
        prev_corners = np.asarray(prev_tag.corners, dtype=np.float64).reshape(4, 2)
        prev_center = np.array([prev_tag.center[0], prev_tag.center[1]], dtype=np.float64)

        next_pts_initial: Optional[np.ndarray] = None
        if use_initial_flow:
            next_pts_initial = motion_history.get_predicted_corners(
                tag_id, prev_corners, prev_center, prev_frame_idx, next_frame_idx
            )

        tracked, st, err = _lk_track_points(
            prev_lk,
            next_lk,
            prev_corners,
            next_pts_initial=next_pts_initial,
            win_size=lk_win_size,
        )
        if int(np.sum(st)) < 3:
            continue
        if float(np.max(err[st.astype(bool)])) > max_corner_err_px:
            continue

        refined = _refine_corners_subpix(next_frame_gray, tracked)

        if not _is_convex_quad(refined):
            continue
        if _quad_area(refined) < min_area_px2:
            continue

        H, rmse = _fit_and_score_homography_cam_to_tag(refined, tag_size=tag_size)
        if rmse > max_tagcoord_rmse:
            continue

        center = refined.mean(axis=0)
        flow_quality = float(np.exp(-float(np.mean(err[st.astype(bool)])) / 10.0))
        reproj_quality = float(np.exp(-rmse))
        conf = float(np.clip(0.5 * flow_quality + 0.5 * reproj_quality, 0.0, 1.0))

        final_tags[tag_id] = TagState(
            tag_id=tag_id,
            corners=refined,
            center=(float(center[0]), float(center[1])),
            H_cam_to_tag=H,
            source="tracked",
            confidence=conf,
        )

    return FrameDetections(frame_idx=frame_idx, tags=final_tags)


def track_apriltags_with_fallback(
    detector: Detector,
    prev_frame_gray: np.ndarray,
    prev_detections: FrameDetections,
    next_frame_gray: np.ndarray,
    *,
    frame_idx: int,
    tag_size: float = 1.0,
    max_corner_err_px: float = 25.0,
    min_area_px2: float = 25.0,
    max_tagcoord_rmse: float = 0.25,
    motion_history: Optional[MotionHistory] = None,
    prev_frame_idx: Optional[int] = None,
) -> FrameDetections:
    """
    Detect tags in next_frame_gray; for tags missing relative to prev_detections,
    attempt recovery using per-tag LK tracking + local corner refinement.
    Optionally use motion_history and prev_frame_idx to seed LK with an extrapolated
    initial flow (velocity + angular velocity from recent frames).

    Notes:
    - This is intentionally *per-tag* (no global motion), suitable for fixed camera + locally moving tags.
    - H_cam_to_tag is computed from corners as image->tag-plane homography.
    """
    raw = detect_to_frame_detections(
        detector, next_frame_gray, frame_idx=frame_idx, tag_size=tag_size
    )
    return _recover_missing_tags_with_tracking(
        prev_frame_gray,
        prev_detections,
        next_frame_gray,
        raw,
        frame_idx=frame_idx,
        tag_size=tag_size,
        max_corner_err_px=max_corner_err_px,
        min_area_px2=min_area_px2,
        max_tagcoord_rmse=max_tagcoord_rmse,
        motion_history=motion_history,
        prev_frame_idx=prev_frame_idx,
    )


def track_pose_detections_with_fallback(
    prev_frame_gray: np.ndarray,
    prev_detections: FrameDetections,
    next_frame_gray: np.ndarray,
    raw_next: FrameDetections,
    *,
    frame_idx: int,
    tag_size: float = 1.0,
    max_corner_err_px: float = 25.0,
    min_area_px2: float = 25.0,
    max_tagcoord_rmse: float = 0.25,
    motion_history: Optional[MotionHistory] = None,
    prev_frame_idx: Optional[int] = None,
    lk_win_size: Tuple[int, int] = (31, 31),
    lk_gradient_preprocess: bool = False,
) -> FrameDetections:
    """
    Given previous-frame detections and raw detections for the next frame (both
    based on the pose-estimation detector output), fill in tags that went
    missing by tracking their corners with LK optical flow.
    Optionally use motion_history and prev_frame_idx to seed LK with an
    extrapolated initial flow (velocity + angular velocity from recent frames).

    This variant uses only image-space tracking but still computes a
    camera->tag-plane homography from the recovered corners for convenience.
    """
    return _recover_missing_tags_with_tracking(
        prev_frame_gray,
        prev_detections,
        next_frame_gray,
        raw_next,
        frame_idx=frame_idx,
        tag_size=tag_size,
        max_corner_err_px=max_corner_err_px,
        min_area_px2=min_area_px2,
        max_tagcoord_rmse=max_tagcoord_rmse,
        motion_history=motion_history,
        prev_frame_idx=prev_frame_idx,
        lk_win_size=lk_win_size,
        lk_gradient_preprocess=lk_gradient_preprocess,
    )