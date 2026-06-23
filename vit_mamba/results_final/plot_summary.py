"""
plot_summary.py

Reads summary.csv (Model, Accuracy (%), ms/image, noise F1, pulse F1,
ramp F1, sine F1, square F1) and produces two PNG charts:

  1. inference_speed.png  - bar chart of ms/image per model
  2. f1_comparison.png    - grouped bar chart of per-class F1 scores

Colors and layout mirror the reference charts (dark navy + light blue,
value labels above each bar, light horizontal gridlines).

Usage:
    python plot_summary.py [path_to_summary.csv] [output_dir]

Both arguments are optional; defaults are "summary.csv" and the
current directory.
"""

import sys
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch

# Colors matched to the in-chat visualizations
DARK_BLUE = "#0c447c"
LIGHT_BLUE = "#85b7eb"
GRID_COLOR = "#d9d9d9"
TEXT_COLOR = "#333333"


def rounded_bar(ax, x_center, bottom, width, height, color, radius_pts=8.0):
    """Draw a single bar with rounded TOP corners and a flat bottom.

    matplotlib's ax.bar() has no border-radius option, so the bar is built
    by hand as a Path: straight up the left side, a quadratic-bezier
    corner into the top edge, straight across, a matching corner down the
    right side, then straight back down to a flat bottom. The corner
    radius is specified in points and converted separately for x and y so
    it looks like a consistent, isotropic radius on screen regardless of
    the data's x/y scale.
    """
    ax_bbox = ax.get_window_extent()
    px_per_xdata = ax_bbox.width / (ax.get_xlim()[1] - ax.get_xlim()[0])
    px_per_ydata = ax_bbox.height / (ax.get_ylim()[1] - ax.get_ylim()[0])

    rx = min(radius_pts / px_per_xdata, width / 2)
    ry = min(radius_pts / px_per_ydata, height) if height > 0 else 0

    x0, x1 = x_center - width / 2, x_center + width / 2
    y0, y1 = bottom, bottom + height

    verts = [
        (x0, y0),
        (x0, y1 - ry),
        (x0, y1),
        (x0 + rx, y1),
        (x1 - rx, y1),
        (x1, y1),
        (x1, y1 - ry),
        (x1, y0),
        (x0, y0),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.LINETO,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.LINETO,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.LINETO,
        MplPath.CLOSEPOLY,
    ]
    patch = PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", zorder=3)
    ax.add_patch(patch)
    return patch


def load_summary(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def plot_inference_speed(rows: list[dict], out_path: Path) -> None:
    models = [r["Model"] for r in rows]
    speeds = [float(r["ms/image"]) for r in rows]
    colors = [DARK_BLUE, LIGHT_BLUE]

    fig, ax = plt.subplots(figsize=(6, 5))

    # Fix axis limits up front -- rounded_bar needs stable xlim/ylim to
    # convert the point-based corner radius into data units.
    ax.set_xlim(-0.6, len(models) - 0.4)
    ax.set_ylim(0, max(speeds) * 1.25)
    fig.canvas.draw()  # forces a layout pass so get_window_extent() is valid

    bar_width = 0.5
    for i, (model, speed, color) in enumerate(zip(models, speeds, colors)):
        rounded_bar(ax, i, 0, bar_width, speed, color, radius_pts=10)
        ax.text(
            i,
            speed + max(speeds) * 0.03,
            f"{speed:.2f} ms",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="medium",
            color=DARK_BLUE,
        )

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=14, color=TEXT_COLOR)
    ax.set_ylabel("ms / image", fontsize=13, color=TEXT_COLOR)
    ax.set_title("Inference speed (ms / image)", fontsize=16, pad=20)
    ax.tick_params(axis="x", labelsize=14, colors=TEXT_COLOR)
    ax.tick_params(axis="y", labelsize=12, colors=TEXT_COLOR)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID_COLOR)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_f1_comparison(rows: list[dict], out_path: Path) -> None:
    classes = ["noise", "pulse", "ramp", "sine", "square"]
    col_names = [f"{c} F1" for c in classes]

    # rows[0] = MambaVision, rows[1] = ViT (order as they appear in csv)
    series = {r["Model"]: [float(r[c]) * 100 for c in col_names] for r in rows}
    model_names = list(series.keys())

    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.set_xlim(-0.6, len(classes) - 0.4)
    ax.set_ylim(95, 100)  # hard cap at 100% -- F1 cannot exceed 100%
    fig.canvas.draw()  # forces a layout pass so get_window_extent() is valid

    bottom = 95
    for i in range(len(classes)):
        v1 = series[model_names[0]][i]
        v2 = series[model_names[1]][i]
        rounded_bar(ax, i - width / 2, bottom, width, v1 - bottom, DARK_BLUE, radius_pts=6)
        rounded_bar(ax, i + width / 2, bottom, width, v2 - bottom, LIGHT_BLUE, radius_pts=6)
        ax.text(i - width / 2, v1 + 0.15, f"{v1:.1f}%", ha="center", va="bottom",
                fontsize=11, fontweight="medium", color=DARK_BLUE)
        ax.text(i + width / 2, v2 + 0.15, f"{v2:.1f}%", ha="center", va="bottom",
                fontsize=11, fontweight="medium", color=DARK_BLUE)

    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, fontsize=14, color=TEXT_COLOR)
    ax.tick_params(axis="y", labelsize=12, colors=TEXT_COLOR)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_title("Per-class F1 score (%)", fontsize=16, pad=55)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID_COLOR)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=DARK_BLUE, label=model_names[0]),
        plt.Rectangle((0, 0), 1, 1, fc=LIGHT_BLUE, label=model_names[1]),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=2,
        frameon=False,
        fontsize=13,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("summary.csv")
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_summary(csv_path)

    plot_inference_speed(rows, out_dir / "inference_speed.png")
    plot_f1_comparison(rows, out_dir / "f1_comparison.png")

    print(f"Saved {out_dir / 'inference_speed.png'}")
    print(f"Saved {out_dir / 'f1_comparison.png'}")


if __name__ == "__main__":
    main()
