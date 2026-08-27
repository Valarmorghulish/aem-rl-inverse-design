# -*- coding: utf-8 -*-


"""Plot conductivity-swelling Pareto fronts for PAP, PBF, and PPO."""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")
matplotlib.rcParams["font.family"] = "Arial"

# PATHS
BASE = os.path.abspath(
    os.environ.get(
        "AEM_RL_ROOT",
        os.path.join(os.path.dirname(__file__), "..", ".."),
    )
)
GEN_BASE = os.path.join(BASE, "generated_aem_pairs_strict_all_models_with_predictor")
OUTPUT = os.path.join(
    os.environ.get(
        "AEM_RL_FIGURE_OUTPUT",
        os.path.join(BASE, "outputs", "figures"),
    ),
    "pareto_fronts",
)
os.makedirs(OUTPUT, exist_ok=True)

GEN_PATHS = {
    "PAP": os.path.join(GEN_BASE, "PAP", "PAP_unique_500_pairs.csv"),
    "PBF": os.path.join(GEN_BASE, "PBF", "PBF_unique_500_pairs.csv"),
    "PPO": os.path.join(GEN_BASE, "PPO", "PPO_unique_500_pairs.csv"),
}

# VISUAL STYLE
CANDIDATE_COLORS = {
    "PAP": "#E05A63",  # deep coral
    "PBF": "#E8A228",  # warm amber
    "PPO": "#7B5EA7",  # violet
}
PARETO_COLOR = "#1B4F8A"  # deep navy
REFLINE_COLOR = "#888888"  # mid-gray reference lines
REFLINE_LW = 1.2

CANDIDATE_ALPHA = 0.35
PARETO_ALPHA = 1.00

S_CAND = 28
S_PARETO = 70

FIG_W, FIG_H = 4.0, 3.6
DPI = 1200

FONT_LABEL = 18
FONT_TICK = 16
FONT_LEGEND = 11
LW_SPINE = 1.6

# Reference line positions
REF_COND = 100  # mS/cm - vertical dashed line
REF_SR = 30  # %     - horizontal dashed line

# PARETO FRONT (maximize conductivity, minimize SR)


def pareto_front_fast(cond: np.ndarray, sr: np.ndarray) -> np.ndarray:
    """O(n log n): sort by cond desc, track running min SR."""
    order = np.lexsort((sr, -cond))
    mask = np.zeros(len(cond), dtype=bool)
    min_sr = np.inf
    for idx in order:
        if sr[idx] < min_sr:
            mask[idx] = True
            min_sr = sr[idx]
    return mask


# COLUMN DETECTION


def detect_col(df, *keywords):
    for kw in keywords:
        for c in df.columns:
            if kw.lower() == c.lower().strip():
                return c
    for kw in keywords:
        for c in df.columns:
            if kw.lower() in c.lower():
                return c
    return None


# PASS 1 - unified axis limits
print("=" * 60)
print("  Pareto Front Plot Generator")
print(f"  Output -> {OUTPUT}")
print("=" * 60)
print("\n[Pass 1] Scanning data ...")

all_cond, all_sr = [], []
data_cache = {}

for name, path in GEN_PATHS.items():
    if not os.path.isfile(path):
        print(f"  WARNING  Not found: {path}")
        continue
    df = pd.read_csv(path)
    ccol = detect_col(df, "predicted_conductivity")
    scol = detect_col(df, "predicted_sr")
    if ccol is None or scol is None:
        print(f"  WARNING  Missing columns in {name}: " f"cond='{ccol}' sr='{scol}'")
        print(f"           Columns: {list(df.columns)}")
        continue
    cond = pd.to_numeric(df[ccol], errors="coerce").values
    sr = pd.to_numeric(df[scol], errors="coerce").values
    valid = np.isfinite(cond) & np.isfinite(sr)
    cond, sr = cond[valid], sr[valid]
    pareto = pareto_front_fast(cond, sr)
    data_cache[name] = {"cond": cond, "sr": sr, "pareto": pareto}
    all_cond.extend(cond.tolist())
    all_sr.extend(sr.tolist())
    print(
        f"  {name}: {valid.sum()} pts  "
        f"cond=[{cond.min():.1f}, {cond.max():.1f}]  "
        f"sr=[{sr.min():.1f}, {sr.max():.1f}]  "
        f"Pareto={pareto.sum()}"
    )

if not data_cache:
    raise RuntimeError("No data loaded.")

all_cond = np.array(all_cond)
all_sr = np.array(all_sr)
c_pad = (all_cond.max() - all_cond.min()) * 0.06
s_pad = (all_sr.max() - all_sr.min()) * 0.06
X_MIN, X_MAX = all_cond.min() - c_pad, all_cond.max() + c_pad
Y_MIN = min(all_sr.min() - s_pad, 18.0)  # ensure y=20 tick is visible
Y_MAX = all_sr.max() + s_pad

xl = mticker.MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10])
X_TICKS = [t for t in xl.tick_values(X_MIN, X_MAX) if X_MIN <= t <= X_MAX]
# Y ticks: every 10 units, from 20 upward to cover all data
Y_TICKS = [t for t in np.arange(20, Y_MAX + 1, 10) if Y_MIN <= t <= Y_MAX]

print(f"\n  Unified x: [{X_MIN:.1f}, {X_MAX:.1f}]")
print(f"  Unified y: [{Y_MIN:.1f}, {Y_MAX:.1f}]  (plotted largetosmall)")

# PASS 2 - Plot
print("\n[Pass 2] Generating plots ...")

for name, d in data_cache.items():
    cond, sr, pareto = d["cond"], d["sr"], d["pareto"]
    col = CANDIDATE_COLORS[name]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    # -- Reference lines ---------------------------------------------------
    ax.axvline(
        REF_COND,
        color=REFLINE_COLOR,
        linewidth=REFLINE_LW,
        linestyle="--",
        zorder=1,
        alpha=0.80,
    )
    ax.axhline(
        REF_SR,
        color=REFLINE_COLOR,
        linewidth=REFLINE_LW,
        linestyle="--",
        zorder=1,
        alpha=0.80,
    )

    # -- Generated candidates ----------------------------------------------
    ax.scatter(
        cond[~pareto],
        sr[~pareto],
        c=col,
        s=S_CAND,
        alpha=CANDIDATE_ALPHA,
        linewidths=0,
        zorder=2,
        label=name,  # Polymer name used as the legend label.
    )

    # -- Pareto front ------------------------------------------------------
    pidx = np.where(pareto)[0]
    order = np.argsort(cond[pidx])
    ax.plot(
        cond[pidx][order],
        sr[pidx][order],
        color=PARETO_COLOR,
        linewidth=1.4,
        linestyle="--",
        alpha=0.75,
        zorder=3,
    )
    ax.scatter(
        cond[pareto],
        sr[pareto],
        c=PARETO_COLOR,
        s=S_PARETO,
        alpha=PARETO_ALPHA,
        linewidths=0.5,
        edgecolors="white",
        zorder=4,
        label="Pareto Front",
    )

    # -- Axes styling ------------------------------------------------------
    for sp in ("top", "right", "left", "bottom"):
        ax.spines[sp].set_visible(True)
        ax.spines[sp].set_linewidth(LW_SPINE)
        ax.spines[sp].set_color("#1A1A1A")

    ax.set_xlim(X_MIN, X_MAX)
    # y-axis inverted: large at bottom of data range becomes top of axis
    ax.set_ylim(Y_MAX, Y_MIN)  # Invert by swapping the limits.
    ax.set_xticks(X_TICKS)
    ax.set_yticks(Y_TICKS)
    ax.tick_params(
        axis="both",
        direction="out",
        length=5,
        width=LW_SPINE,
        labelsize=FONT_TICK,
        colors="#1A1A1A",
        pad=3,
    )
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")

    ax.set_xlabel(
        "Conductivity (mS/cm)",
        fontsize=FONT_LABEL,
        fontweight="bold",
        color="#1A1A1A",
        labelpad=5,
    )
    ax.set_ylabel(
        "SR (%)", fontsize=FONT_LABEL, fontweight="bold", color="#1A1A1A", labelpad=5
    )
    ax.grid(False)

    # -- Legend - top-right, avoid data -----------------------------------
    leg = ax.legend(
        loc="upper left",
        fontsize=FONT_LEGEND,
        frameon=True,
        framealpha=0.88,
        edgecolor="#cccccc",
        handletextpad=0.3,
        handlelength=0.9,
        borderpad=0.35,
        labelspacing=0.25,
    )
    leg.get_frame().set_linewidth(0.8)
    for txt in leg.get_texts():
        txt.set_fontweight("bold")
    for h in leg.legend_handles:
        try:
            h.set_sizes([22])
        except AttributeError:
            pass

    plt.tight_layout(pad=0.8)

    stem = f"pareto_{name}"
    fig.savefig(
        os.path.join(OUTPUT, stem + ".png"),
        dpi=DPI,
        format="png",
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        os.path.join(OUTPUT, stem + ".tif"),
        dpi=DPI,
        format="tiff",
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)
    print(f"  [saved]  {stem}.png / .tif")

print(f"\nDone - {len(data_cache)} plots saved to: {OUTPUT}")
