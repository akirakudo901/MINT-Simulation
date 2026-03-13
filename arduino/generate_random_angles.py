#!/usr/bin/env python3
"""
Generate N random angles for hip, knee, and ankle joints.
Saves to a file and optionally sends to Arduino via Serial.
"""

import argparse
import random
import time
from pathlib import Path

try:
    import serial
except ImportError:
    serial = None

# Joint ranges (degrees)
HIP_RANGE = (0, 180)
KNEE_RANGE = (0, 150)
ANKLE_RANGE = (0, 120)


def generate_angles(n: int) -> list[tuple[int, int, int]]:
    """Generate n random (hip, knee, ankle) tuples with uniform sampling."""
    return [
        (
            random.randint(*HIP_RANGE),
            random.randint(*KNEE_RANGE),
            random.randint(*ANKLE_RANGE),
        )
        for _ in range(n)
    ]


def save_to_file(angles: list[tuple[int, int, int]], filepath: Path) -> None:
    """Save angles to a text file, one triplet per line: hip,knee,ankle."""
    with open(filepath, "w") as f:
        for hip, knee, ankle in angles:
            f.write(f"{hip},{knee},{ankle}\n")


def load_from_file(filepath: Path) -> list[tuple[int, int, int]]:
    """Load angles from a text file."""
    angles = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) == 3:
                angles.append((int(parts[0]), int(parts[1]), int(parts[2])))
    return angles


def send_to_arduino(
    angles: list[tuple[int, int, int]],
    port: str = None,
    baud: int = 9600,
) -> None:
    """Send angle triplets to Arduino over Serial."""
    if serial is None:
        raise ImportError("pyserial is required for sending to Arduino. Install with: pip install pyserial")

    import glob

    # Auto-detect Arduino port if not specified
    if not port:
        for pattern in ["/dev/cu.usb*", "/dev/tty.usb*", "/dev/cu.usbserial*", "COM*"]:
            matches = glob.glob(pattern)
            if matches:
                port = matches[0]
                break
        if not port:
            raise OSError("No serial port found. Specify with -p (e.g. -p /dev/cu.usbserial-1420)")

    with serial.Serial(port, baud, timeout=1) as ser:
        time.sleep(2)  # Allow Arduino to reset after serial connect
        for i, (hip, knee, ankle) in enumerate(angles):
            msg = f"{hip},{knee},{ankle}\n"
            ser.write(msg.encode())
            print(f"[{i + 1}/{len(angles)}] Sent hip={hip}, knee={knee}, ankle={ankle}")
            # Wait for Arduino to process (adjust as needed)
            time.sleep(1)
            while ser.in_waiting:
                line = ser.readline().decode().strip()
                if line:
                    print(f"  Arduino: {line}")


def main():
    parser = argparse.ArgumentParser(description="Generate and apply random joint angles")
    parser.add_argument("n", type=int, default=10, nargs="?", help="Number of angle sets (default: 10)")
    parser.add_argument(
        "-o", "--output",
        default="random_angles.txt",
        help="Output file path (default: random_angles.txt)",
    )
    parser.add_argument(
        "-s", "--send",
        action="store_true",
        help="After generating/saving, read from file and send to Arduino",
    )
    parser.add_argument(
        "-p", "--port",
        default=None,
        help="Serial port for Arduino (e.g. /dev/cu.usbserial-1420 or COM3). Auto-detects if omitted.",
    )
    parser.add_argument(
        "--load-only",
        action="store_true",
        help="Skip generation; load from existing file and send to Arduino",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent.resolve()
    filepath = script_dir / args.output

    if args.load_only:
        angles = load_from_file(filepath)
        print(f"Loaded {len(angles)} angle sets from {filepath}")
    else:
        angles = generate_angles(args.n)
        save_to_file(angles, filepath)
        print(f"Generated {args.n} angle sets, saved to {filepath}")

    if args.send:
        send_to_arduino(angles, port=args.port)
    else:
        print("Use -s to send angles to Arduino after generating.")
        print("Example: python generate_random_angles.py 10 -s")


if __name__ == "__main__":
    main()
