# -*- coding: utf-8 -*-


"""Compare Pretrain, Fine-tune, PAP, PBF, and PPO property distributions."""

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
    "rl_model_comparison",
)
os.makedirs(OUTPUT, exist_ok=True)

# --- Source files -------------------------------------------------------------
SOURCES = {
    "Pretrain": os.path.join(ROOT, "Unbiased", "Unbiased_stable_100.csv"),
    "Fine-tune": os.path.join(ROOT, "Finetuned", "Finetuned_stable_100.csv"),
    "PAP": os.path.join(ROOT, "PAP", "PAP_stable_500.csv"),
    "PBF": os.path.join(ROOT, "PBF", "PBF_stable_500.csv"),
    "PPO": os.path.join(ROOT, "PPO", "PPO_stable_100.csv"),
}

# --- Plot groups - each tuple defines one figure -----------------------------
PLOT_GROUPS = [
    ("PAP", ["Pretrain", "Fine-tune", "PAP"]),
    ("PBF", ["Pretrain", "Fine-tune", "PBF"]),
    ("PPO", ["Pretrain", "Fine-tune", "PPO"]),
]

# --- Color palette - top-journal, filled-density style -----------------------
#   Soft but distinct; chosen for overlap readability with alpha fill
GROUP_COLORS = {
    "Pretrain": "#8DA9C4",  # dusty steel blue   (neutral baseline)
    "Fine-tune": "#4A7C59",  # forest green       (trained reference)
    "PAP": "#C1666B",  # muted coral red
    "PBF": "#E09F3E",  # warm amber
    "PPO": "#7B5EA7",  # soft violet
}

FILL_ALPHA = 0.45  # transparency for filled area
BW_MULT = 1.8  # smoothing multiplier on Scott bandwidth

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
    for c in df.columns:
        if c.strip().lower() == name.strip().lower():
            return c
    for c in df.columns:
        if name.lower() in c.lower():
            return c
    return None


# --- Plot function -------------------------------------------------------------


def plot_group(group_name: str, labels: list, prop_name: str, prop: dict):
    x_min, x_max = prop["x_min"], prop["x_max"]
    x_plot = np.linspace(x_min, x_max, 1000)

    # collect curves
    curves = []
    for label in labels:
        fpath = SOURCES[label]
        if not os.path.isfile(fpath):
            print(f"  [warning]  Not found: {fpath}")
            continue
        df = pd.read_csv(fpath)
        col = find_col(df, prop["col"])
        if col is None:
            print(
                f"  [warning]  Column '{prop['col']}' missing in {os.path.basename(fpath)}. "
                f"Cols: {list(df.columns)}"
            )
            continue
        data = df[col].dropna().values
        data = data[(data >= x_min) & (data <= x_max)]
        if len(data) < 10:
            print(f"  [warning]  Too few points ({len(data)}) - {label} [{prop_name}]")
            continue
        y = smooth_kde(data, x_plot)
        curves.append((float(y.max()), label, y))

    if not curves:
        print(f"  [skip]  Nothing to plot - {group_name} [{prop_name}]")
        return

    # draw tallest peak first (lowest z) to shorter curves visible on top
    curves.sort(key=lambda c: c[0], reverse=True)

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
        # thin top edge in same color, slightly more opaque, for definition

    # -- Style -------------------------------------------------------------
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

    stem = f"{group_name}_{prop_name}_kde"
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


def save_swatch():
    labels = list(GROUP_COLORS.keys())
    n = len(labels)

    sw_w, row_h = 0.60, 0.46
    lbl_w = 1.10
    pad_x, pad_y = 0.18, 0.22

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

    stem = "group_color_swatch"
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
    print("=" * 62)
    print("  KDE Filled Plot Generator - manuscript style")
    print(f"  Output to {OUTPUT}")
    print("=" * 62)

    for group_name, labels in PLOT_GROUPS:
        for prop_name, prop in PROPERTIES.items():
            print(f"\n[{group_name}]  {prop_name}")
            plot_group(group_name, labels, prop_name, prop)

    print("\n[Color Swatch]")
    save_swatch()

    print("\n" + "=" * 62)
    print("  Done - 6 KDE plots + 1 color swatch")
    print(f"  Saved to: {OUTPUT}")
    print("=" * 62)


if __name__ == "__main__":
    main()
