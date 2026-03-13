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
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection


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


def _plot_time_series_with_tracked(
    ax: plt.Axes,
    frames: pd.Series,
    values: pd.Series,
    tracked: Optional[pd.Series],
    ylabel: str,
    *,
    color_detected: str = "tab:blue",
    color_tracked: str = "tab:orange",
) -> None:
    """
    Plot a 1D time series as a continuous line, recoloring segments based on `tracked` flag.

    `tracked` should be 0 for detected and 1 for tracked; if None, a single color is used.
    Frames where `values` is NaN create gaps (no line is drawn across them).
    """
    x = np.asarray(frames, dtype=float)
    y = np.asarray(values, dtype=float)
    tracked_arr = np.asarray(tracked, dtype=float) if tracked is not None else None

    if x.size < 2:
        return

    segments = []
    colors = []
    x_min, x_max = float("inf"), float("-inf")

    prev_segment = None
    prev_color = None
    for i in range(len(x) - 1):
        y0, y1 = y[i], y[i + 1]
        if np.isnan(y0) or np.isnan(y1):
            # Do not create a segment across NaNs; this produces a visual gap.
            prev_segment = None
            prev_color = None
            continue

        x0, x1 = x[i], x[i + 1]
        x_min = min(x_min, x0, x1)
        x_max = max(x_max, x0, x1)

        if tracked_arr is not None:
            flag = 1 if tracked_arr[i] == 1 else 0
            curr_color = color_tracked if flag == 1 else color_detected
        else:
            curr_color = color_detected

        # If we have a previous segment, try to elongate it
        if (
            prev_segment is not None
            and prev_color == curr_color
            and np.isclose(prev_segment[-1][0], x0)
            and np.isclose(prev_segment[-1][1], y0)
        ):
            # Elongate the latest segment by extending to the new point
            prev_segment.append([x1, y1])
        else:
            # Add new segment
            segments.append([[x0, y0], [x1, y1]])
            colors.append(curr_color)
            prev_segment = segments[-1]
            prev_color = curr_color

    if not segments:
        return

    lc = LineCollection(segments, colors=colors, linewidths=1.5)
    ax.add_collection(lc)

    if x_min != float("inf") and x_max != float("-inf"):
        ax.set_xlim(float(x_min), float(x_max))

    # Fit y-axis to the plotted data range
    # Flatten all y-values from the segments and filter out NaNs
    all_y = np.concatenate([[pt[1] for pt in seg] for seg in segments])
    all_y = all_y[~np.isnan(all_y)]
    if all_y.size > 0:
        y_min, y_max = np.min(all_y), np.max(all_y)
        if not np.isclose(y_min, y_max):
            # Add a small margin
            margin = 0.05 * (y_max - y_min)
            ax.set_ylim(y_min - margin, y_max + margin)
        else:
            # Single value: still add a margin
            ax.set_ylim(y_min - 1, y_max + 1)

    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.3)


def analyze_pose_trajectories(
    csv_path: str | Path,
    show_plot: bool = True,
    save_path: str | Path | None = None,
    sections: Optional[Sequence[Tuple[int, int]]] = None,
    tag_ids: Optional[Sequence[int]] = None,
) -> None:
    """
    Plot center coordinates (X, Y) and yaw over time from a pose CSV.

    The CSV is expected to be produced by `find_apriltags.run_on_source` with pose
    estimation enabled, so it should contain at least the columns:
        - frame
        - tag_id
        - center_x, center_y
        - yaw_deg
    Optionally, when fallback tracking is enabled and pose is written, the CSV
    also contains:
        - tracked (0 = detected, 1 = tracked)

    For each provided frame interval in ``sections`` (or for the full video if
    ``sections`` is None), this function produces a figure with three stacked
    subplots:
        1) center_x vs. frame
        2) center_y vs. frame
        3) yaw_deg vs. frame

    Within each subplot, the line is continuous over time, but segments
    corresponding to detected frames (tracked == 0) and tracked frames
    (tracked == 1) are colored differently.

    Args:
        csv_path: Path to the pose CSV.
        show_plot: If True, display figures with plt.show().
        save_path: If set, save figures to this base path. When multiple
            sections or tag IDs are present, suffixes like
            ``_sec{idx}_tag{tid}`` are added before the file extension.
        sections: Optional list of (start_frame, end_frame) tuples indicating
            which parts of the video to analyze. If None, the whole CSV range
            is used as a single section.
        tag_ids: Optional list of tag IDs to include. If None, all tag IDs
            present in the CSV are plotted.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = ["frame", "tag_id", "center_x", "center_y", "yaw_deg"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns {missing} in CSV. "
            f"Available columns: {list(df.columns)}"
        )

    has_tracked = "tracked" in df.columns

    if tag_ids is not None:
        tag_ids_set = set(int(t) for t in tag_ids)
        df = df[df["tag_id"].isin(tag_ids_set)]
        if df.empty:
            raise ValueError(
                f"No rows left after filtering for tag_ids={sorted(tag_ids_set)}."
            )

    if sections is None:
        section_specs: List[Tuple[Optional[int], Optional[int]]] = [(None, None)]
    else:
        section_specs = [(int(s), int(e)) for (s, e) in sections]

    # Prepare base path for saving (if requested).
    save_base: Optional[Path]
    save_suffix: str
    if save_path is not None:
        save_base = Path(save_path)
        if save_base.suffix:
            save_suffix = save_base.suffix
        else:
            save_suffix = ".png"
    else:
        save_base = None
        save_suffix = ""

    unique_tag_ids = sorted(df["tag_id"].unique())

    for sec_idx, (start_f, end_f) in enumerate(section_specs):
        if start_f is not None and end_f is not None:
            sec_df = df[(df["frame"] >= start_f) & (df["frame"] <= end_f)]
        else:
            sec_df = df

        if sec_df.empty:
            continue

        for tag_id in unique_tag_ids:
            tag_df = sec_df[sec_df["tag_id"] == tag_id].sort_values("frame")
            if tag_df.shape[0] < 2:
                continue

            frames = tag_df["frame"]
            cx = tag_df["center_x"]
            cy = tag_df["center_y"]
            yaw = tag_df["yaw_deg"]
            tracked_series = tag_df["tracked"] if has_tracked else None

            fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

            _plot_time_series_with_tracked(
                axes[0],
                frames,
                cx,
                tracked_series,
                ylabel="center_x (px)",
            )
            _plot_time_series_with_tracked(
                axes[1],
                frames,
                cy,
                tracked_series,
                ylabel="center_y (px)",
            )
            _plot_time_series_with_tracked(
                axes[2],
                frames,
                yaw,
                tracked_series,
                ylabel="yaw (deg)",
            )

            axes[2].set_xlabel("Frame index")

            title_parts = [f"Tag {tag_id} center & yaw"]
            if start_f is not None and end_f is not None:
                title_parts.append(f"frames {start_f}–{end_f}")
            fig.suptitle(" — ".join(title_parts))

            # Legend proxies for detected vs tracked coloring when available.
            if has_tracked:
                proxy_detected = plt.Line2D(
                    [], [], color="tab:blue", label="detected (tracked=0)"
                )
                proxy_tracked = plt.Line2D(
                    [], [], color="tab:orange", label="tracked (tracked=1)"
                )
                axes[0].legend(handles=[proxy_detected, proxy_tracked], loc="best")

            fig.tight_layout(rect=(0, 0, 1, 0.96))

            if save_base is not None:
                if start_f is not None and end_f is not None:
                    suffix = f"_sec{sec_idx}_tag{tag_id}{save_suffix}"
                else:
                    suffix = f"_overall_tag{tag_id}{save_suffix}"
                out_path = save_base.with_name(save_base.stem + suffix)
                fig.savefig(out_path, dpi=150, bbox_inches="tight")

            if show_plot:
                plt.show()
            else:
                plt.close(fig)


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
    # main()

    # CSV = r"/Users/akirakudo/Desktop/code/python/MINT/MINT-Simulation/poses/poses_allslowmo_1500_quadSigma0.8_sharpening0.25.csv"
    CSV = r"/Users/akirakudo/Desktop/code/python/MINT/MINT-Simulation/poses/poses_allslowmo_2500_with_detect.csv"

    analyze_pose_trajectories(
        csv_path=CSV,
        show_plot=True,
        save_path=None,
        sections=[(1236, 1338)],
        tag_ids=None,
    )