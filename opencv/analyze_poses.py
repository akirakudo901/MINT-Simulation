"""
This file analyzes csvs obtained by extracting pose estimates of AprilTags running
`run_on_source` in the `find_apriltags.py` file.

One function computes the mean and standard deviation for the yaw, pitch and roll estimates,
separately for each tag id, in degrees. It then plots this into a bar graph.
"""

import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd


def analyze_pose_angles(
    csv_path: str | Path,
    show_plot: bool = True,
    save_path: str | Path | None = None,
    sections: Optional[Sequence[Tuple[int, int]]] = None,
) -> Union[pd.DataFrame, dict]:
    """
    Load a pose CSV, compute mean and standard deviation of yaw, pitch, and roll (in degrees)
    per tag_id, and plot them as bar graphs.

    Args:
        csv_path: Path to the CSV file (e.g. from find_apriltags.run_on_source).
        show_plot: If True, display the plot with plt.show().
        save_path: If set, save the figure to this path (e.g. 'poses_analysis.png').
        sections: Optional sequence of (start_frame, end_frame) tuples. When provided, the
            function still computes and plots overall per-tag statistics, but additionally
            computes the same statistics restricted to each frame interval.

    Returns:
        If ``sections`` is None:
            A DataFrame with columns:
              tag_id, yaw_mean, yaw_std, pitch_mean, pitch_std, roll_mean, roll_std.

        If ``sections`` is provided:
            A dict with keys:
              - "overall": overall per-tag statistics DataFrame as above.
              - "sections": DataFrame with the same statistics computed separately for each
                (start_frame, end_frame) interval, and extra columns:
                section_index, frame_start, frame_end.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    angle_cols = ["yaw_deg", "pitch_deg", "roll_deg"]
    for col in angle_cols:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' in CSV. Columns: {list(df.columns)}")

    # Aggregate mean and std per tag_id (overall)
    stats = df.groupby("tag_id")[angle_cols].agg(["mean", "std"]).reset_index()
    stats.columns = [
        "tag_id",
        "yaw_mean", "yaw_std",
        "pitch_mean", "pitch_std",
        "roll_mean", "roll_std",
    ]

    # Bar plot: one subplot per angle (yaw, pitch, roll) for overall stats
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    tag_ids = stats["tag_id"].astype(str)
    x = range(len(tag_ids))
    width = 0.35

    for ax, angle_name, mean_col, std_col in [
        (axes[0], "Yaw (deg)", "yaw_mean", "yaw_std"),
        (axes[1], "Pitch (deg)", "pitch_mean", "pitch_std"),
        (axes[2], "Roll (deg)", "roll_mean", "roll_std"),
    ]:
        means = stats[mean_col]
        stds = stats[std_col].fillna(0)
        ax.bar(x, means, width, yerr=stds, capsize=4, color="steelblue", edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(tag_ids)
        ax.set_xlabel("Tag ID")
        ax.set_ylabel(angle_name)
        ax.set_title(f"{angle_name} — mean ± std")
        ax.axhline(0, color="gray", linewidth=0.5)

    plt.suptitle("Pose angle statistics per AprilTag ID", fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close()

    if sections is None:
        # Backwards-compatible: only return overall stats
        return stats

    # Additionally compute per-section statistics restricted to the specified frame ranges.
    section_rows: List[pd.DataFrame] = []
    for idx, (start_frame, end_frame) in enumerate(sections):
        section_df = df[(df["frame"] >= start_frame) & (df["frame"] <= end_frame)]
        if section_df.empty:
            # Skip empty ranges to avoid confusing all-NaN rows
            continue

        section_stats = (
            section_df.groupby("tag_id")[angle_cols]
            .agg(["mean", "std"])
            .reset_index()
        )
        section_stats.columns = [
            "tag_id",
            "yaw_mean",
            "yaw_std",
            "pitch_mean",
            "pitch_std",
            "roll_mean",
            "roll_std",
        ]
        section_stats["section_index"] = idx
        section_stats["frame_start"] = start_frame
        section_stats["frame_end"] = end_frame
        section_rows.append(section_stats)

    if section_rows:
        sections_stats_df = pd.concat(section_rows, ignore_index=True)
    else:
        # No valid sections produced any data; return an empty sections DataFrame.
        sections_stats_df = pd.DataFrame(
            columns=[
                "tag_id",
                "yaw_mean",
                "yaw_std",
                "pitch_mean",
                "pitch_std",
                "roll_mean",
                "roll_std",
                "section_index",
                "frame_start",
                "frame_end",
            ]
        )

    return {
        "overall": stats,
        "sections": sections_stats_df,
    }


def _parse_sections_arg(sections_arg: Optional[str]) -> Optional[List[Tuple[int, int]]]:
    """
    Parse a command-line --sections argument of the form:
        "start1:end1,start2:end2,..."
    into a list of (start, end) integer tuples.
    """
    if not sections_arg:
        return None

    result: List[Tuple[int, int]] = []
    for part in sections_arg.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            start_str, end_str = part.split(":")
            start = int(start_str)
            end = int(end_str)
        except ValueError as exc:
            raise ValueError(
                f"Invalid section specifier '{part}'. Expected 'start:end'."
            ) from exc
        if end < start:
            raise ValueError(
                f"Invalid section '{part}': end frame must be >= start frame."
            )
        result.append((start, end))

    return result or None


def main():
    parser = argparse.ArgumentParser(
        description="Analyze pose CSV: mean/std of yaw, pitch, roll per tag_id and plot bar graphs."
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parent.parent / "poses_full.csv",
        help="Path to the pose CSV (default: poses_full.csv in project root)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the plot (useful when only saving).",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Save the figure to this path (e.g. poses_analysis.png).",
    )
    parser.add_argument(
        "--sections",
        type=str,
        default=None,
        help=(
            "Optional comma-separated frame ranges to analyze separately, "
            'e.g. "0:100,200:300".'
        ),
    )
    args = parser.parse_args()

    sections = _parse_sections_arg(args.sections)

    result = analyze_pose_angles(
        args.csv_path,
        show_plot=not args.no_show,
        save_path=args.output,
        sections=sections,
    )

    # Print summaries
    if isinstance(result, dict):
        overall = result["overall"]
        sections_df = result["sections"]

        print("Overall per-tag statistics (degrees):")
        print(overall.to_string(index=False))
        print()

        if not sections_df.empty:
            print("Per-section per-tag statistics (degrees):")
            print(sections_df.to_string(index=False))
        else:
            print("No valid sections produced any statistics (ranges may be empty).")
    else:
        print("Per-tag statistics (degrees):")
        print(result.to_string(index=False))


if __name__ == "__main__":
    main()
