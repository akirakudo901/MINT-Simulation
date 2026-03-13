"""
Camera calibration to obtain intrinsics (fx, fy, cx, cy) for AprilTag pose estimation.

Use either:
  1. Checkerboard calibration (accurate): print a checkerboard, capture ~15–20 images
     at different angles/distances, then run this script on those images.
  2. Rough estimate (no calibration): assume principal point at image center and
     approximate fx from field-of-view (see get_rough_camera_params() below).

Checkerboard:
- Use a grid of *internal* corners, e.g. 9x6 means 9 columns and 6 rows of *inner* corners
  (squares touching each other). Print from: https://markhedleyjones.com/projects/calibration-checkerboard-collection
  or generate with OpenCV / online generators.
- Measure one square size in meters and set CHECKERBOARD_SQUARE_SIZE below.

Usage:
  # Calibrate from a folder of checkerboard images:
  python calibrate_camera.py --images path/to/calib_images/*.jpg --cols 9 --rows 6 --square-size 0.025

  # Or capture from live camera (press SPACE to capture, Q when done):
  python calibrate_camera.py --camera 0 --cols 9 --rows 6 --square-size 0.025 --capture-dir ./calib_frames

  # Output: prints (fx, fy, cx, cy) and saves to JSON for use in find_apriltags.py
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def find_checkerboard_corners(
    gray: np.ndarray,
    pattern_size: tuple[int, int],
    subpix_criteria: tuple,
) -> np.ndarray | None:
    """Find checkerboard corners; returns refined corners or None."""
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    if not ret:
        return None
    corners = cv2.cornerSubPix(
        gray, corners, (11, 11), (-1, -1), subpix_criteria
    )
    return corners


def calibrate_from_images(
    image_paths: list[Path],
    pattern_size: tuple[int, int],
    square_size_m: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Run OpenCV calibration on checkerboard images.

    Returns:
        camera_matrix: 3x3 (fx, fy, cx, cy in standard positions)
        dist_coeffs: distortion coefficients
        mean_error: re-projection error in pixels
    """
    # Object points: 3D coordinates of checkerboard corners (Z=0 plane)
    # Units in meters if square_size_m is in meters
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(
        -1, 2
    )
    objp *= square_size_m

    objpoints: list[np.ndarray] = []
    imgpoints: list[np.ndarray] = []
    image_size = None

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            print(f"  Skip (cannot read): {path}", file=sys.stderr)
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = gray.shape[::-1]

        corners = find_checkerboard_corners(gray, pattern_size, criteria)
        if corners is None:
            print(f"  Skip (no pattern): {path}", file=sys.stderr)
            continue

        objpoints.append(objp)
        imgpoints.append(corners)

    if len(objpoints) < 5:
        raise RuntimeError(
            f"Need at least 5 valid checkerboard images; got {len(objpoints)}. "
            "Capture more at different angles and distances."
        )

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    # Re-projection error
    mean_error = 0.0
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
        )
        mean_error += cv2.norm(imgpoints[i], proj, cv2.NORM_L2) / len(proj)
    mean_error /= len(objpoints)

    return camera_matrix, dist_coeffs, mean_error


def camera_matrix_to_intrinsics(camera_matrix: np.ndarray) -> tuple[float, float, float, float]:
    """Extract (fx, fy, cx, cy) for pupil_apriltags camera_params."""
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])
    return (fx, fy, cx, cy)


def get_rough_camera_params(
    width: int,
    height: int,
    fov_deg: float = 60.0,
) -> tuple[float, float, float, float]:
    """
    Approximate camera_params without calibration.

    Assumes principal point at image center and estimates fx, fy from
    horizontal field-of-view (in degrees). Use only for quick tests;
    pose scale/angle will be approximate.

    fov_deg: horizontal field of view (e.g. 60–90 for typical webcams).
    """
    cx = width / 2.0
    cy = height / 2.0
    fov_rad = np.radians(fov_deg)
    fx = (width / 2.0) / np.tan(fov_rad / 2.0)
    fy = fx  # assume square pixels
    return (fx, fy, cx, cy)


def capture_calibration_frames(
    camera_id: int,
    pattern_size: tuple[int, int],
    output_dir: Path,
    count: int = 20,
) -> list[Path]:
    """
    Live capture: show camera feed, press SPACE to capture a frame, Q to finish.
    Returns list of saved image paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_id}")

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    saved: list[Path] = []
    print(f"Capture {count} checkerboard images. SPACE = capture, Q = done.")
    while len(saved) < count:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = find_checkerboard_corners(gray, pattern_size, criteria)
        if corners is not None:
            cv2.drawChessboardCorners(
                frame, pattern_size, corners, True
            )
            status = "FOUND - press SPACE to capture"
        else:
            status = "no pattern"
        cv2.putText(
            frame, f"{status}  [{len(saved)}/{count}]",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )
        cv2.imshow("Calibration capture", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" ") and corners is not None:
            path = output_dir / f"calib_{len(saved):03d}.jpg"
            cv2.imwrite(str(path), frame)
            saved.append(path)
            print(f"  Saved {path.name}")

    cap.release()
    cv2.destroyAllWindows()
    return saved


def main():
    parser = argparse.ArgumentParser(
        description="Camera calibration for (fx, fy, cx, cy) used by AprilTag pose estimation.",
    )
    parser.add_argument(
        "--images",
        nargs="+",
        help="Paths to checkerboard images (e.g. calib/*.jpg).",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        metavar="ID",
        help="Use live camera to capture frames; SPACE to capture, Q to quit.",
    )
    parser.add_argument(
        "--capture-dir",
        type=Path,
        default=Path("calib_frames"),
        help="Directory to save captured frames (with --camera).",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=9,
        help="Number of internal checkerboard columns (default 9).",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=6,
        help="Number of internal checkerboard rows (default 6).",
    )
    parser.add_argument(
        "--square-size",
        type=float,
        default=0.025,
        metavar="M",
        help="Checkerboard square size in meters (default 0.025 = 25mm).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save intrinsics to JSON (camera_params and image size).",
    )
    parser.add_argument(
        "--rough",
        action="store_true",
        help="Skip calibration; print rough (fx,fy,cx,cy) from image size and FOV.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Image width for --rough (default 640).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Image height for --rough (default 480).",
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=70,
        help="Horizontal FOV in degrees for --rough (default 70).",
    )
    args = parser.parse_args()

    if args.rough:
        fx, fy, cx, cy = get_rough_camera_params(
            args.width, args.height, args.fov
        )
        print("Rough camera_params (no calibration):")
        print(f"  camera_params = ({fx:.2f}, {fy:.2f}, {cx:.2f}, {cy:.2f})")
        if args.output:
            data = {
                "camera_params": [fx, fy, cx, cy],
                "image_size": [args.width, args.height],
                "note": "rough estimate from FOV",
            }
            args.output.write_text(json.dumps(data, indent=2))
            print(f"  Saved to {args.output}")
        return

    pattern_size = (args.cols, args.rows)

    if args.camera is not None:
        paths = capture_calibration_frames(
            args.camera, pattern_size, args.capture_dir
        )
        if not paths:
            print("No images captured.", file=sys.stderr)
            sys.exit(1)
        image_paths = paths
    elif args.images:
        image_paths = []
        for p in args.images:
            image_paths.extend(Path(x).resolve() for x in glob.glob(p))
        image_paths = sorted(set(image_paths))
        if not image_paths:
            print("No images found.", file=sys.stderr)
            sys.exit(1)
    else:
        print("Provide --images <paths> or --camera <id>.", file=sys.stderr)
        sys.exit(1)

    print(f"Calibrating from {len(image_paths)} images, pattern {pattern_size}, square {args.square_size}m")
    camera_matrix, dist_coeffs, err = calibrate_from_images(
        image_paths, pattern_size, args.square_size
    )
    fx, fy, cx, cy = camera_matrix_to_intrinsics(camera_matrix)

    print(f"Re-projection error: {err:.4f} pixels")
    print("camera_params for pupil_apriltags:")
    print(f"  camera_params = ({fx:.2f}, {fy:.2f}, {cx:.2f}, {cy:.2f})")
    print("Distortion coefficients (for cv2.undistort if needed):")
    print(f"  dist_coeffs = {dist_coeffs.flatten().tolist()}")

    if args.output:
        # Get image size from first image
        sample = cv2.imread(str(image_paths[0]))
        h, w = sample.shape[:2] if sample is not None else (0, 0)
        data = {
            "camera_params": [fx, fy, cx, cy],
            "dist_coeffs": dist_coeffs.flatten().tolist(),
            "image_size": [w, h],
            "reprojection_error_px": err,
        }
        args.output.write_text(json.dumps(data, indent=2))
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
