"""
visualize_training_quality.py
=============================
Side-by-side view of waveform image + generated training text.
Helps you judge whether the description matches the actual signal.

Usage:
    python visualize_training_quality.py                    # all rows in training_data.csv
    python visualize_training_quality.py --limit 5          # first 5 rows
    python visualize_training_quality.py --waveform sine    # filter by waveform type
    python visualize_training_quality.py --index 3 7 12     # specific row indices
"""

import argparse
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import pandas as pd

BG      = "#0f1117"
BG2     = "#1a1d27"
BORDER  = "#2e3347"
WHITE   = "#e8eaf0"
GRAY    = "#8892a4"
GREEN   = "#00d97e"
BLUE    = "#4da6ff"
YELLOW  = "#ffd166"
RED     = "#ff6b6b"
PURPLE  = "#c77dff"


def section_color(header: str) -> str:
    h = header.upper()
    if "DIAGNOSIS" in h:
        return YELLOW
    if "CORRECTION VECTOR" in h:
        return GREEN
    if "CORRECTION" in h:
        return BLUE
    return WHITE


def render_text_panel(ax, description: str, waveform: str, split: str):
    ax.set_facecolor(BG2)
    ax.axis("off")

    # Parse sections
    sections = []
    current_header = None
    current_lines  = []
    for raw_line in description.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if any(line.startswith(h) for h in
               ("DIAGNOSIS:", "CORRECTION VECTOR:", "CORRECTION:")):
            if current_header is not None:
                sections.append((current_header, "\n".join(current_lines)))
            # Split header from inline text
            colon = line.index(":") + 1
            current_header = line[:colon]
            rest = line[colon:].strip()
            current_lines = [rest] if rest else []
        else:
            current_lines.append(line)
    if current_header is not None:
        sections.append((current_header, "\n".join(current_lines)))

    y = 0.97
    wrap_width = 52   # chars before wrapping

    for header, body in sections:
        color = section_color(header)

        # Header
        ax.text(0.03, y, header, transform=ax.transAxes,
                va="top", color=color, fontsize=8.5, fontweight="bold",
                fontfamily="monospace")
        y -= 0.055

        # Body — word-wrap each line
        for raw in body.splitlines():
            wrapped = textwrap.fill(raw.strip(), width=wrap_width)
            for wline in wrapped.splitlines():
                ax.text(0.05, y, wline, transform=ax.transAxes,
                        va="top", color=WHITE, fontsize=7.8,
                        fontfamily="monospace")
                y -= 0.045
            y -= 0.005   # small gap between lines

        y -= 0.02   # gap between sections

    # Waveform tag badge
    wf_colors = {"sine": BLUE, "square": GREEN, "ramp": YELLOW,
                 "noise": PURPLE, "pulse": RED}
    badge_color = wf_colors.get(waveform, GRAY)
    ax.text(0.97, 0.97, f"{split} / {waveform}",
            transform=ax.transAxes, va="top", ha="right",
            color=badge_color, fontsize=7.5, fontweight="bold",
            bbox=dict(facecolor=BG, edgecolor=badge_color,
                      boxstyle="round,pad=0.3", linewidth=1.2))


def make_figure(rows: list[dict], out_path: str):
    n = len(rows)
    fig_h = max(4, n * 3.8)
    fig = plt.figure(figsize=(16, fig_h), facecolor=BG)
    fig.suptitle(
        "Training Data Quality Check — Waveform Image vs Generated Description",
        color=WHITE, fontsize=11, fontweight="bold", y=1.002
    )

    outer = gridspec.GridSpec(n, 1, figure=fig,
                              hspace=0.08,
                              left=0.01, right=0.99,
                              top=0.995, bottom=0.005)

    for row_idx, row in enumerate(rows):
        inner = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer[row_idx],
            width_ratios=[1, 1.25], wspace=0.03
        )

        # ── Left: waveform image ───────────────────────────────────────────────
        ax_img = fig.add_subplot(inner[0])
        ax_img.set_facecolor(BG2)
        img_path = row["image_path"]
        try:
            img = mpimg.imread(img_path)
            ax_img.imshow(img)
        except Exception:
            ax_img.text(0.5, 0.5, f"Image not found:\n{img_path}",
                        transform=ax_img.transAxes, ha="center", va="center",
                        color=RED, fontsize=7)
        ax_img.axis("off")

        # Filename label below image
        fname = Path(img_path).name
        ax_img.set_title(fname, color=GRAY, fontsize=7, pad=3)

        # ── Right: text panel ──────────────────────────────────────────────────
        ax_txt = fig.add_subplot(inner[1])
        render_text_panel(ax_txt, row["description"],
                          row["waveform_label"], row["split"])

        # Row border
        for ax in (ax_img, ax_txt):
            for sp in ax.spines.values():
                sp.set_edgecolor(BORDER)
                sp.set_linewidth(0.8)

    plt.savefig(out_path, dpi=130, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close()
    print(f"Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",    default="./training_data.csv")
    parser.add_argument("--output",   default="./training_quality_check.png")
    parser.add_argument("--limit",    type=int, default=None,
                        help="Max rows to show")
    parser.add_argument("--waveform", default=None,
                        choices=["sine", "square", "noise", "ramp", "pulse"],
                        help="Filter by waveform type")
    parser.add_argument("--index",    nargs="+", type=int, default=None,
                        help="Specific row indices to show (0-based)")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")

    if args.waveform:
        df = df[df["waveform_label"] == args.waveform].reset_index(drop=True)
        print(f"Filtered to {len(df)} rows for waveform='{args.waveform}'")

    if args.index is not None:
        df = df.iloc[args.index].reset_index(drop=True)
    elif args.limit:
        df = df.head(args.limit)

    if df.empty:
        print("No rows to display.")
        return

    rows = df.to_dict("records")
    print(f"Rendering {len(rows)} sample(s)...")
    make_figure(rows, args.output)


if __name__ == "__main__":
    main()
