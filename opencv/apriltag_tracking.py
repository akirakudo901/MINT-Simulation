from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

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
    # OpenCV expects float32/64; getPerspectiveTransform gives tag<-img transform.
    H = cv2.getPerspectiveTransform(img.astype(np.float64), tag.astype(np.float64))
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


def _lk_track_points(
    prev_gray: np.ndarray,
    next_gray: np.ndarray,
    pts_prev: np.ndarray,
    *,
    win_size: Tuple[int, int] = (31, 31),
    max_level: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    p0 = np.asarray(pts_prev, dtype=np.float32).reshape(-1, 1, 2)
    p1, st, err = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        next_gray,
        p0,
        None,
        winSize=win_size,
        maxLevel=max_level,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        flags=0,
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
) -> FrameDetections:
    """
    Detect tags in next_frame_gray; for tags missing relative to prev_detections,
    attempt recovery using per-tag LK tracking + local corner refinement.

    Notes:
    - This is intentionally *per-tag* (no global motion), suitable for fixed camera + locally moving tags.
    - H_cam_to_tag is computed from corners as image->tag-plane homography.
    """
    raw = detect_to_frame_detections(
        detector, next_frame_gray, frame_idx=frame_idx, tag_size=tag_size
    )
    prev_ids = set(prev_detections.tags.keys())
    next_ids = set(raw.tags.keys())
    missing = sorted(prev_ids - next_ids)
    if not missing:
        return raw

    final_tags = dict(raw.tags)

    for tag_id in missing:
        prev_tag = prev_detections.tags.get(tag_id)
        if prev_tag is None:
            continue
        prev_corners = np.asarray(prev_tag.corners, dtype=np.float64).reshape(4, 2)

        tracked, st, err = _lk_track_points(prev_frame_gray, next_frame_gray, prev_corners)
        if int(np.sum(st)) < 3:
            continue
        if float(np.max(err[st.astype(bool)])) > max_corner_err_px:
            continue

        # Local subpixel refinement in the new frame.
        refined = _refine_corners_subpix(next_frame_gray, tracked)

        if not _is_convex_quad(refined):
            continue
        if _quad_area(refined) < min_area_px2:
            continue

        H, rmse = _fit_and_score_homography_cam_to_tag(refined, tag_size=tag_size)
        if rmse > max_tagcoord_rmse:
            continue

        center = refined.mean(axis=0)
        # Confidence: combine flow quality + homography reprojection quality.
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
) -> FrameDetections:
    """
    Given previous-frame detections and raw detections for the next frame (both
    based on the pose-estimation detector output), fill in tags that went
    missing by tracking their corners with LK optical flow.

    This variant uses only image-space tracking but still computes a
    camera->tag-plane homography from the recovered corners for convenience.
    """
    prev_ids = set(prev_detections.tags.keys())
    next_ids = set(raw_next.tags.keys())
    missing = sorted(prev_ids - next_ids)
    if not missing:
        return raw_next

    final_tags = dict(raw_next.tags)

    for tag_id in missing:
        prev_tag = prev_detections.tags.get(tag_id)
        if prev_tag is None:
            continue
        prev_corners = np.asarray(prev_tag.corners, dtype=np.float64).reshape(4, 2)

        tracked, st, err = _lk_track_points(prev_frame_gray, next_frame_gray, prev_corners)
        if int(np.sum(st)) < 3:
            continue
        if float(np.max(err[st.astype(bool)])) > max_corner_err_px:
            continue

        refined = _refine_corners_subpix(next_frame_gray, tracked)

        if not _is_convex_quad(refined):
            continue
        if _quad_area(refined) < min_area_px2:
            continue

        # Fit homography and reject if reprojection in tag coords is too large.
        H, rmse = _fit_and_score_homography_cam_to_tag(
            refined, tag_size=tag_size
        )
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

