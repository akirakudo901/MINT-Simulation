"""
Measure latency of real-time capture and AprilTag detection.

METHODOLOGY
----------
- Capture latency: time from "request next frame" to "frame available in memory"
  (cap.read()). This includes driver/camera exposure, transfer, and decode.
- Detection latency: time to convert frame to grayscale and run detector.detect().
- End-to-end: capture + detection per frame.

Use time.perf_counter() for high-resolution timing. Run for N frames and report
mean, std, and percentiles so you can see variance (e.g. GC, thermal throttling).

Run with display off (--no-show) for more accurate timings. Optional: reduce
resolution in OpenCV (cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)) to test impact.

Usage (run from opencv/ so find_apriltags imports):
  python measure_latency.py [--camera 0] [--frames 300] [--no-show]
  python measure_latency.py --no-show --width 640 --frames 500
"""

import argparse
import statistics
import sys
import time

import cv2
import numpy as np

# Reuse detector config and helpers from find_apriltags
from find_apriltags import detect_apriltags, frame_to_grayscale, get_detector


def measure(
    camera_id: int = 0,
    num_frames: int = 300,
    show: bool = False,
    width: int | None = None,
    height: int | None = None,
) -> None:
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: could not open camera {camera_id}", file=sys.stderr)
        sys.exit(1)

    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    detector = get_detector()

    capture_ms: list[float] = []
    detection_ms: list[float] = []
    end_to_end_ms: list[float] = []

    print(f"Measuring over {num_frames} frames (camera {camera_id}, show={show})...")
    warmup = 5
    for i in range(num_frames + warmup):
        # --- Capture latency: time to get one frame ---
        t0 = time.perf_counter()
        ret, frame = cap.read()
        t1 = time.perf_counter()
        if not ret or frame is None:
            print("Warning: failed to read frame", file=sys.stderr)
            continue

        # --- Detection latency: grayscale + detector.detect ---
        t2 = time.perf_counter()
        gray = frame_to_grayscale(frame)
        detections = detector.detect(gray)
        t3 = time.perf_counter()

        if i >= warmup:
            cap_ms = (t1 - t0) * 1000
            det_ms = (t3 - t2) * 1000
            e2e_ms = (t3 - t0) * 1000
            capture_ms.append(cap_ms)
            detection_ms.append(det_ms)
            end_to_end_ms.append(e2e_ms)

        if show:
            for d in detections:
                pts = d.corners.astype(np.int32)
                cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
                cx, cy = int(d.center[0]), int(d.center[1])
                cv2.putText(frame, str(d.tag_id), (cx + 5, cy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Latency measure", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if show:
        cv2.destroyAllWindows()

    if not capture_ms:
        print("No frames collected.", file=sys.stderr)
        sys.exit(1)

    n = len(capture_ms)

    def stats(name: str, ms: list[float]) -> None:
        mean = statistics.mean(ms)
        stdev = statistics.stdev(ms) if n > 1 else 0.0
        p50 = statistics.median(ms)
        p99 = sorted(ms)[int(0.99 * n) - 1] if n >= 100 else p50
        print(f"  {name}: mean={mean:.2f} ms  std={stdev:.2f}  p50={p50:.2f} ms  p99={p99:.2f} ms")

    print("\nLatency (ms) over", n, "frames:")
    stats("Capture (cap.read)", capture_ms)
    stats("Detection (gray + detect)", detection_ms)
    stats("End-to-end (capture + detection)", end_to_end_ms)
    print(f"\nImplied max FPS (1 / e2e): ~{1000 / statistics.mean(end_to_end_ms):.1f}")


def main():
    parser = argparse.ArgumentParser(description="Measure capture and AprilTag detection latency.")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default 0)")
    parser.add_argument("--frames", type=int, default=300, help="Number of frames to measure")
    parser.add_argument("--no-show", action="store_true", help="Disable display (recommended for timing)")
    parser.add_argument("--width", type=int, default=None, help="Force frame width (e.g. 640)")
    parser.add_argument("--height", type=int, default=None, help="Force frame height")
    args = parser.parse_args()

    measure(
        camera_id=args.camera,
        num_frames=args.frames,
        show=not args.no_show,
        width=args.width,
        height=args.height,
    )


if __name__ == "__main__":
    main()
