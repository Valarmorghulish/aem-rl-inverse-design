# -*- coding: utf-8 -*-


"""Plot RL iteration distributions for the PAEK and PAES settings."""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore")
matplotlib.rcParams["font.family"] = "Arial"

# --- Paths -------------------------------------------------------------------
BASE_PATH = os.path.abspath(
    os.environ.get(
        "AEM_RL_ROOT",
        os.path.join(os.path.dirname(__file__), "..", ".."),
    )
)
OUTPUT_PATH = os.path.join(
    os.environ.get(
        "AEM_RL_FIGURE_OUTPUT",
        os.path.join(BASE_PATH, "outputs", "figures"),
    ),
    "rl_iteration_distributions",
)
os.makedirs(OUTPUT_PATH, exist_ok=True)

# --- Color palette -----------------------------------------------------------
ITER_COLORS = {
    10: "#707070",  # medium gray
    20: "#8C1A4A",  # deep burgundy
    30: "#1B4F8A",  # deep navy blue
    40: "#D94E1F",  # burnt orange
    50: "#2AAD9B",  # teal
    60: "#2E9EC6",  # sky blue
    70: "#E8A020",  # amber / gold
}

ITER_NAMES = {
    10: "Iter 10",
    20: "Iter 20",
    30: "Iter 30",
    40: "Iter 40",
    50: "Iter 50",
    60: "Iter 60",
    70: "Iter 70",
}

# --- Dataset configurations ---------------------------------------------------
DATASETS = {
    "PAEK": {
        "folder": "rl_replay_plots_50PAEK",
        "file_pattern": "checkpoint_iter_{iter}_valid_samples.csv",
        "iters": [10, 20, 30, 40, 50],
    },
    "PAES": {
        "folder": "rl_replay_plots_50PAES",
        "file_pattern": "checkpoint_iter_{iter}_valid_samples_cumulative.csv",
        "iters": [10, 20, 30, 40, 50, 60, 70],
    },
}

# --- Property configurations --------------------------------------------------
PROPERTIES = {
    "Conductivity": {
        "x_min": 20,
        "x_max": 130,
        "xticks": [20, 40, 60, 80, 100, 120],
        "xlabel": "Predicted Conductivity (mS/cm)",
    },
    "SR": {
        "x_min": 20,
        "x_max": 80,
        "xticks": [20, 30, 40, 50, 60, 70, 80],
        "xlabel": "Predicted SR (%)",
    },
}

# --- Tuning -------------------------------------------------------------------
BW_MULTIPLIER = 1.8
FILL_ALPHA = 0.80  # fill transparency per ridge
FILL_ALPHA_MAX = 0.65  # used for swatch matching

# Ridge layout
RIDGE_SPACING = 0.35  # fraction of global_peak for row spacing (smaller = closer)
# 0.25 = very tight, 0.5 = more open

# Fixed figure size - IDENTICAL for ALL plots regardless of iteration count
FIG_W = 3.8
FIG_H = 3.6

# --- Helpers -----------------------------------------------------------------


def find_column(df: pd.DataFrame, keyword: str):
    for col in df.columns:
        if keyword.lower() in col.lower():
            return col
    return None


def plot_kde_ridge(polymer: str, prop: str, prop_cfg: dict, ds_cfg: dict) -> None:
    x_min, x_max = prop_cfg["x_min"], prop_cfg["x_max"]
    x_plot = np.linspace(x_min, x_max, 1000)

    # -- 1. Load all curves, sorted Iter 10 to Iter N (bottom to top) --------
    curves = []
    for it in sorted(ds_cfg["iters"]):  # ascending: Iter 10 first
        fpath = os.path.join(
            BASE_PATH, ds_cfg["folder"], ds_cfg["file_pattern"].format(iter=it)
        )
        if not os.path.isfile(fpath):
            print(f"  [warning]  File not found: {fpath}")
            continue
        df = pd.read_csv(fpath)
        col = find_column(df, prop)
        if col is None:
            print(f"  [warning]  Column '{prop}' missing in {os.path.basename(fpath)}.")
            continue
        data = df[col].dropna().values
        data = data[(data >= x_min) & (data <= x_max)]
        if len(data) < 10:
            print(
                f"  [warning]  Too few points ({len(data)}) - {os.path.basename(fpath)}"
            )
            continue
        kde = gaussian_kde(data, bw_method="scott")
        kde.set_bandwidth(bw_method=kde.factor * BW_MULTIPLIER)
        y = kde(x_plot)
        curves.append((it, y))

    if not curves:
        print(f"  [skip]  Nothing to plot for {polymer} - {prop}")
        return

    n = len(curves)

    # -- 2. Compute vertical layout ----------------------------------------
    global_peak = max(c[1].max() for c in curves)

    # Fixed row spacing as fraction of global peak - same regardless of n
    row_height = global_peak * RIDGE_SPACING
    # Position: Iter 10 = TOP (offset largest), Iter 70 = BOTTOM (offset 0)
    # Z-order:  Iter 10 drawn FIRST (back), Iter 70 drawn LAST (front)
    # curves is sorted ascending [Iter10, Iter20, ..., Iter70]
    offsets = [(n - 1 - i) * row_height for i in range(n)]
    # offsets[0] = Iter10 = (n-1)*row_height  to top
    # offsets[-1] = IterN = 0                 to bottom
    total_height_data = offsets[0] + global_peak * 1.08

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    # -- 3. Draw: Iter 10 first (back/top), Iter 70 last (front/bottom) ----
    for z, ((it, y), offset) in enumerate(zip(curves, offsets)):
        color = ITER_COLORS.get(it, "#555555")
        ax.fill_between(
            x_plot,
            offset,
            y + offset,
            color=color,
            alpha=FILL_ALPHA,
            linewidth=0,
            zorder=z + 2,
        )
        ax.hlines(offset, x_min, x_max, colors="#cccccc", linewidth=0.7, zorder=z + 1)

    # -- 4. Style ----------------------------------------------------------
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.8)
    ax.spines["bottom"].set_color("#1A1A1A")

    ax.set_xlim(x_min, x_max)
    ax.set_xticks(prop_cfg["xticks"])
    ax.tick_params(
        axis="x",
        which="both",
        direction="out",
        length=5,
        width=1.5,
        labelsize=15,
        colors="#1A1A1A",
        pad=4,
    )
    for lbl in ax.get_xticklabels():
        lbl.set_fontweight("bold")

    # Y axis: hide ticks/labels
    ax.tick_params(axis="y", left=False, labelleft=False)
    # Scale y-limits to always fill the fixed figure height
    ax.set_ylim(bottom=-row_height * 0.08, top=total_height_data)

    ax.set_xlabel(
        prop_cfg["xlabel"], fontsize=15, fontweight="bold", color="#1A1A1A", labelpad=6
    )
    ax.set_ylabel("")
    ax.grid(False)

    plt.tight_layout(pad=0.6)

    # -- 5. Save -----------------------------------------------------------
    stem = f"{polymer}_{prop}_kde"
    fig.savefig(
        os.path.join(OUTPUT_PATH, stem + ".png"),
        dpi=1200,
        format="png",
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        os.path.join(OUTPUT_PATH, stem + ".tif"),
        dpi=1200,
        format="tiff",
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)
    print(f"  [saved]  {stem}.png / .tif")


# --- Color swatch -------------------------------------------------------------


def save_color_swatch() -> None:
    all_iters = [10, 20, 30, 40, 50, 60, 70]
    n = len(all_iters)

    sw_w = 1.00
    sw_h = 0.46
    gap = 0.18
    lbl_gap = 0.20
    pad_x = 0.25
    pad_y = 0.28

    fig_w = pad_x * 2 + sw_w + lbl_gap + 1.6
    fig_h = pad_y * 2 + n * sw_h + (n - 1) * gap

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    for i, it in enumerate(all_iters):
        y_top = fig_h - pad_y - i * (sw_h + gap)
        y_bot = y_top - sw_h
        yc = (y_top + y_bot) / 2

        rect = mpatches.FancyBboxPatch(
            (pad_x, y_bot),
            sw_w,
            sw_h,
            boxstyle="round,pad=0.02",
            facecolor=ITER_COLORS[it],
            alpha=FILL_ALPHA_MAX,
            edgecolor="none",
        )
        ax.add_patch(rect)

        ax.text(
            pad_x + sw_w + lbl_gap,
            yc,
            ITER_NAMES[it],
            ha="left",
            va="center",
            fontsize=16,
            fontweight="bold",
            fontfamily="Arial",
            color="#1A1A1A",
        )

    stem = "iter_color_swatch"
    fig.savefig(
        os.path.join(OUTPUT_PATH, stem + ".png"),
        dpi=1200,
        format="png",
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        os.path.join(OUTPUT_PATH, stem + ".tif"),
        dpi=1200,
        format="tiff",
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)
    print(f"  [saved]  {stem}.png / .tif")


# --- Main ---------------------------------------------------------------------


def main():
    print("=" * 62)
    print("  KDE Ridge Plot Generator v2")
    print(f"  Output to {OUTPUT_PATH}")
    print("=" * 62)

    total = 0
    for polymer, ds_cfg in DATASETS.items():
        for prop, prop_cfg in PROPERTIES.items():
            print(f"\n[{polymer}]  {prop}")
            plot_kde_ridge(polymer, prop, prop_cfg, ds_cfg)
            total += 1

    print("\n[Color Swatch]")
    save_color_swatch()

    print("\n" + "=" * 62)
    print(f"  Done - {total} KDE plots + 1 color swatch")
    print(f"  Saved to: {OUTPUT_PATH}")
    print("=" * 62)


if __name__ == "__main__":
    main()
