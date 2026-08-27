# -*- coding: utf-8 -*-


"""Plot conductivity and swelling-ratio distributions for eight generators.

The script produces two filled kernel-density plots and a colour swatch. The
bandwidth, axis limits, colours, and export resolution match the manuscript
analysis.

Datasets
--------
  Pretrain  - Unbiased_stable_100.csv
  Fine-tune - Finetuned_stable_100.csv
  PAP       - PAP_stable_500.csv
  PBF       - PBF_stable_500.csv
  PPO       - PPO_stable_100.csv
  PAEK      - PAEK_stable_100.csv
  PAEKS     - PAEKS_stable_100.csv
  PAES      - PAES_stable_100.csv

Run: python plotting/figures/generate_model_distributions.py
"""

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

# --- Paths --------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(
    os.environ.get(
        "AEM_RL_ROOT",
        os.path.join(os.path.dirname(__file__), "..", ".."),
    )
)
ROOT = os.path.join(PROJECT_ROOT, "generated_8models_500pairs_full_pipeline")
OUTPUT = os.path.join(
    os.environ.get(
        "AEM_RL_FIGURE_OUTPUT",
        os.path.join(PROJECT_ROOT, "outputs", "figures"),
    ),
    "model_distributions",
)
os.makedirs(OUTPUT, exist_ok=True)

# --- Source files -------------------------------------------------------------
SOURCES = {
    "Pretrain": os.path.join(ROOT, "Unbiased", "Unbiased_stable_100.csv"),
    "Fine-tune": os.path.join(ROOT, "Finetuned", "Finetuned_stable_100.csv"),
    "PAP": os.path.join(ROOT, "PAP", "PAP_stable_500.csv"),
    "PBF": os.path.join(ROOT, "PBF", "PBF_stable_500.csv"),
    "PPO": os.path.join(ROOT, "PPO", "PPO_stable_100.csv"),
    "PAEK": os.path.join(ROOT, "PAEK", "PAEK_stable_100.csv"),
    "PAEKS": os.path.join(ROOT, "PAEKS", "PAEKS_stable_100.csv"),
    "PAES": os.path.join(ROOT, "PAES", "PAES_stable_100.csv"),
}

# draw order within each plot (controls legend narrative; occlusion handled by peak sort)
DRAW_ORDER = [
    "Pretrain",
    "Fine-tune",
    "PAP",
    "PBF",
    "PPO",
    "PAEK",
    "PAEKS",
    "PAES",
]

# --- Color palette ------------------------------------------------------------
#   Original 5  (unchanged)
#   New 3       - chosen to be clearly distinct from all existing colors
GROUP_COLORS = {
    # -- original 5 ------------------------------------------
    "Pretrain": "#8DA9C4",  # dusty steel blue
    "Fine-tune": "#4A7C59",  # forest green
    "PAP": "#C1666B",  # muted coral red
    "PBF": "#E09F3E",  # warm amber
    "PPO": "#7B5EA7",  # soft violet
    # -- new 3 -----------------------------------------------
    "PAEK": "#2A9D8F",  # deep teal          (cool, distinct from blue & green)
    "PAEKS": "#A05A2C",  # burnt sienna brown (warm-dark, distinct from amber & coral)
    "PAES": "#6B8F71",  # sage / olive green (muted mid-green, distinct from forest)
}

FILL_ALPHA = 0.42  # alpha for filled area - lower = more see-through at overlaps
BW_MULT = 1.8  # smoothing factor on Scott bandwidth

# --- Property configs ---------------------------------------------------------
PROPERTIES = {
    "Conductivity": {
        "col": "predicted_conductivity_mScm",
        "x_min": 20,
        "x_max": 130,
        "xticks": [20, 40, 60, 80, 100, 120],
        "xlabel": "Conductivity (mS/cm)",
    },
    "SR": {
        "col": "predicted_SR_pct",
        "x_min": 20,
        "x_max": 80,
        "xticks": [20, 30, 40, 50, 60, 70, 80],
        "xlabel": "SR (%)",
    },
}

FIG_W, FIG_H = 3.8, 3.0

# --- KDE helper ---------------------------------------------------------------


def smooth_kde(data: np.ndarray, x_plot: np.ndarray) -> np.ndarray:
    kde = gaussian_kde(data, bw_method="scott")
    kde.set_bandwidth(bw_method=kde.factor * BW_MULT)
    return kde(x_plot)


def find_col(df: pd.DataFrame, name: str):
    """Case-insensitive column lookup - exact match first, then substring."""
    for c in df.columns:
        if c.strip().lower() == name.strip().lower():
            return c
    for c in df.columns:
        if name.lower() in c.lower():
            return c
    return None


# --- Plot function -------------------------------------------------------------


def plot_all(prop_name: str, prop: dict) -> None:
    x_min, x_max = prop["x_min"], prop["x_max"]
    x_plot = np.linspace(x_min, x_max, 1000)

    # -- 1. load all curves ------------------------------------------------
    curves = []
    for label in DRAW_ORDER:
        fpath = SOURCES[label]
        if not os.path.isfile(fpath):
            print(f"  [warning]  Not found : {fpath}")
            continue
        df = pd.read_csv(fpath)
        col = find_col(df, prop["col"])
        if col is None:
            print(
                f"  [warning]  Column '{prop['col']}' missing in "
                f"{os.path.basename(fpath)}. Cols: {list(df.columns)}"
            )
            continue
        data = df[col].dropna().values
        data = data[(data >= x_min) & (data <= x_max)]
        if len(data) < 10:
            print(f"  [warning]  Too few points ({len(data)}) - {label} [{prop_name}]")
            continue
        y = smooth_kde(data, x_plot)
        curves.append((float(y.max()), label, y))
        print(f"      {label:12s}  n={len(data):5d}  peak={y.max():.4f}")

    if not curves:
        print(f"  [skip]  Nothing to plot [{prop_name}]")
        return

    # -- 2. sort: tallest first to drawn at lowest z; shorter always visible -
    curves.sort(key=lambda c: c[0], reverse=True)

    # -- 3. draw ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    for z, (_, label, y) in enumerate(curves):
        color = GROUP_COLORS[label]
        ax.fill_between(
            x_plot,
            0,
            y,
            color=color,
            alpha=FILL_ALPHA,
            linewidth=0,
            zorder=z + 1,
        )
        # thin definition edge - same hue, slightly more opaque

    # -- 4. style - bottom spine only --------------------------------------
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.8)
    ax.spines["bottom"].set_color("#1A1A1A")

    ax.set_xlim(x_min, x_max)
    ax.set_xticks(prop["xticks"])
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

    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(
        prop["xlabel"], fontsize=15, fontweight="bold", color="#1A1A1A", labelpad=6
    )
    ax.set_ylabel("")
    ax.grid(False)

    plt.tight_layout(pad=0.6)

    # -- 5. save -----------------------------------------------------------
    stem = f"all8_{prop_name}_kde"
    fig.savefig(
        os.path.join(OUTPUT, stem + ".png"),
        dpi=1200,
        format="png",
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        os.path.join(OUTPUT, stem + ".tif"),
        dpi=1200,
        format="tiff",
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)
    print(f"  [saved]  {stem}.png / .tif")


# --- Color swatch -------------------------------------------------------------


def save_swatch() -> None:
    """Vertical swatch - one row per label, manuscript style, Arial bold."""
    labels = list(GROUP_COLORS.keys())  # preserves insertion order (Python 3.7+)
    n = len(labels)

    sw_w = 0.62  # color block width  (inches)
    row_h = 0.46  # row height         (inches)
    lbl_w = 1.20  # text area width    (inches)
    pad_x = 0.18
    pad_y = 0.22

    fig_w = pad_x * 2 + sw_w + lbl_w
    fig_h = pad_y * 2 + n * row_h

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    for i, label in enumerate(labels):
        yc = fig_h - pad_y - (i + 0.5) * row_h

        rect = mpatches.FancyBboxPatch(
            (pad_x, yc - row_h * 0.36),
            sw_w,
            row_h * 0.72,
            boxstyle="round,pad=0.01",
            facecolor=GROUP_COLORS[label],
            edgecolor="none",
        )
        ax.add_patch(rect)

        ax.text(
            pad_x + sw_w + 0.10,
            yc,
            label,
            ha="left",
            va="center",
            fontsize=11,
            fontweight="bold",
            fontfamily="Arial",
            color="#1A1A1A",
        )

    stem = "group_color_swatch_v2"
    fig.savefig(
        os.path.join(OUTPUT, stem + ".png"),
        dpi=1200,
        format="png",
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        os.path.join(OUTPUT, stem + ".tif"),
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
    print("=" * 64)
    print("  KDE Filled Plot Generator v2 - 8 datasets, 2 plots")
    print(f"  Output to {OUTPUT}")
    print("=" * 64)

    for prop_name, prop in PROPERTIES.items():
        print(f"\n-- {prop_name} --------------------------------------")
        plot_all(prop_name, prop)

    print("\n-- Color Swatch ------------------------------------------")
    save_swatch()

    print("\n" + "=" * 64)
    print("  Done - 2 KDE plots + 1 color swatch")
    print(f"  Saved to: {OUTPUT}")
    print("=" * 64)


if __name__ == "__main__":
    main()
