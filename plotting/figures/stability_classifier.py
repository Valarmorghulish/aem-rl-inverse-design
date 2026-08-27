# -*- coding: utf-8 -*-


import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

matplotlib.rcParams["font.family"] = "Arial"


PROJECT_ROOT = os.path.abspath(
    os.environ.get(
        "AEM_RL_ROOT",
        os.path.join(os.path.dirname(__file__), "..", ".."),
    )
)
WORK_DIR = os.path.abspath(
    os.environ.get(
        "AEM_RL_STABILITY_RESULTS",
        os.path.join(
            PROJECT_ROOT,
            "aem_catboost_WITHTIME_degMIN_desc20_thr0.5_bayes_tune",
        ),
    )
)
TUNED_DIR = os.path.join(WORK_DIR, "tuned_bayes")
OUTPUT_DIR = os.path.join(
    os.environ.get(
        "AEM_RL_FIGURE_OUTPUT",
        os.path.join(PROJECT_ROOT, "outputs", "figures"),
    ),
    "stability_classifier",
)
THRESHOLD = 0.5

os.makedirs(OUTPUT_DIR, exist_ok=True)


FIG_W, FIG_H = 4.8, 4.8
DPI = 1200


# [left, bottom, width, height], with width equal to height.
AX_RECT = [0.28, 0.23, 0.66, 0.66]

FONT_LABEL = 20
FONT_TICK = 18
FONT_TICK_BAR = 17
FONT_ANNOT = 15
LW_SPINE = 2.2
LW_CURVE = 2.8

COL_PAP = "#C05A63"
COL_FINETUNE = "#3D8A52"
COL_PBF = "#D4962A"
COL_PPO = "#7B5EA7"
DASH_COLOR = "#888888"


ALPHA_CM = 0.82
ALPHA_BAR = 0.82


def style_ax(ax, xlabel="", ylabel="", tick_labelsize=FONT_TICK):
    for sp in ax.spines.values():
        sp.set_linewidth(LW_SPINE)
        sp.set_color("#1A1A1A")

    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        length=6,
        width=2.0,
        labelsize=tick_labelsize,
        colors="#1A1A1A",
        pad=5,
    )

    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
        lbl.set_fontfamily("Arial")

    if xlabel:
        ax.set_xlabel(
            xlabel,
            fontsize=FONT_LABEL,
            fontweight="bold",
            fontfamily="Arial",
            color="#1A1A1A",
            labelpad=7,
        )

    if ylabel:
        ax.set_ylabel(
            ylabel,
            fontsize=FONT_LABEL,
            fontweight="bold",
            fontfamily="Arial",
            color="#1A1A1A",
            labelpad=8,
        )

    ax.grid(False)
    ax.set_box_aspect(1)


def save_fig(fig, name):
    """Save a 5760 x 5760 px panel without tight-bounding-box cropping."""
    png_path = os.path.join(OUTPUT_DIR, f"{name}.png")
    tif_path = os.path.join(OUTPUT_DIR, f"{name}.tif")

    fig.savefig(png_path, dpi=DPI, facecolor="white")

    fig.savefig(
        tif_path, dpi=DPI, facecolor="white", pil_kwargs={"compression": "tiff_lzw"}
    )

    plt.show()
    print(f"  [saved]  {name}.png / {name}.tif")


df_pred = pd.read_csv(os.path.join(TUNED_DIR, "unseen_test_predictions.csv"))

y_true = df_pred["y_pass"].astype(int).values
p_tuned = df_pred["prob_pass"].values
pred_tuned = (p_tuned >= THRESHOLD).astype(int)

auc = float(roc_auc_score(y_true, p_tuned))
acc = float(accuracy_score(y_true, pred_tuned))
f1 = float(f1_score(y_true, pred_tuned))

print(f"AUC={auc:.3f}  ACC={acc:.3f}  F1={f1:.3f}")


# Panel a: ROC curve
fpr, tpr, _ = roc_curve(y_true, p_tuned)


if fpr[0] != 0 or tpr[0] != 0:
    fpr = np.concatenate([[0], fpr])
    tpr = np.concatenate([[0], tpr])

fig_a = plt.figure(figsize=(FIG_W, FIG_H))
ax_a = fig_a.add_axes(AX_RECT)


ax_a.step(fpr, tpr, where="post", color=COL_PAP, lw=LW_CURVE)

ax_a.plot([0, 1], [0, 1], "--", color=DASH_COLOR, lw=1.8)

ann = f"AUC = {auc:.3f}\nACC = {acc:.3f}\nF1   = {f1:.3f}"
ax_a.text(
    0.97,
    0.05,
    ann,
    transform=ax_a.transAxes,
    ha="right",
    va="bottom",
    fontsize=FONT_ANNOT,
    fontweight="bold",
    fontfamily="Arial",
    color="#1A1A1A",
    bbox=dict(
        boxstyle="round,pad=0.40",
        facecolor="white",
        edgecolor="#bbbbbb",
        linewidth=1.2,
        alpha=0.96,
    ),
)


ax_a.set_xlim(-0.06, 1.02)
ax_a.set_ylim(-0.01, 1.04)

ax_a.set_xticks([0.00, 0.25, 0.50, 0.75, 1.00])
ax_a.set_yticks([0.00, 0.20, 0.40, 0.60, 0.80, 1.00])

style_ax(
    ax_a,
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    tick_labelsize=FONT_TICK,
)

save_fig(fig_a, "roc_curve")


# Panel b: Confusion matrix
cm = confusion_matrix(y_true, pred_tuned, labels=[0, 1])
tn, fp_v, fn, tp = cm.ravel()

fig_b = plt.figure(figsize=(FIG_W, FIG_H))
ax_b = fig_b.add_axes(AX_RECT)


cell_info = [
    (0, 0, COL_FINETUNE, tn, "white", ALPHA_CM),
    (0, 1, "white", fp_v, "#1A1A1A", 1.00),
    (1, 0, "white", fn, "#1A1A1A", 1.00),
    (1, 1, COL_PAP, tp, "white", ALPHA_CM),
]

for r, c, bg, val, txtcol, alpha in cell_info:
    ax_b.add_patch(
        matplotlib.patches.Rectangle(
            (c - 0.5, r - 0.5), 1, 1, linewidth=0, facecolor=bg, alpha=alpha, zorder=1
        )
    )

    ax_b.text(
        c,
        r,
        str(val),
        ha="center",
        va="center",
        fontsize=28,
        fontweight="bold",
        fontfamily="Arial",
        color=txtcol,
        zorder=2,
    )

ax_b.set_xlim(-0.5, 1.5)
ax_b.set_ylim(1.5, -0.5)

ax_b.set_xticks([0, 1])
ax_b.set_yticks([0, 1])

ax_b.set_xticklabels(
    ["Fail (0)", "Pass (1)"], fontsize=FONT_TICK, fontweight="bold", fontfamily="Arial"
)


ax_b.set_yticklabels(
    ["Fail (0)", "Pass (1)"],
    fontsize=FONT_TICK,
    fontweight="bold",
    fontfamily="Arial",
    rotation=90,
    va="center",
)

style_ax(ax_b, xlabel="Predicted Label", ylabel="True Label", tick_labelsize=FONT_TICK)


ax_b.tick_params(axis="y", pad=1)

ax_b.text(
    0.03,
    0.97,
    f"TN={tn}  FP={fp_v}\nFN={fn}  TP={tp}",
    transform=ax_b.transAxes,
    ha="left",
    va="top",
    fontsize=12,
    fontweight="bold",
    fontfamily="Arial",
    color="#1A1A1A",
    bbox=dict(
        boxstyle="round,pad=0.30",
        facecolor="white",
        edgecolor="#bbbbbb",
        linewidth=1.2,
        alpha=0.94,
    ),
)

save_fig(fig_b, "confusion_matrix")


# Panel c: Pass-rate bar chart
pass_rates = {"PAP": 93.6, "PBF": 65.4, "Fine-tune": 39.8, "PPO": 1.2}

bar_colors = {"PAP": COL_PAP, "PBF": COL_PBF, "Fine-tune": COL_FINETUNE, "PPO": COL_PPO}

labels = list(pass_rates.keys())
values = [pass_rates[k] for k in labels]
colors = [bar_colors[k] for k in labels]


x = np.array([0.0, 1.15, 2.55, 3.85])

fig_c = plt.figure(figsize=(FIG_W, FIG_H))
ax_c = fig_c.add_axes(AX_RECT)

bars = ax_c.bar(
    x, values, color=colors, alpha=ALPHA_BAR, edgecolor="none", width=0.55, zorder=3
)

for bar, val in zip(bars, values):
    ax_c.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1.5,
        f"{val:.1f}%",
        ha="center",
        va="bottom",
        fontsize=FONT_ANNOT,
        fontweight="bold",
        fontfamily="Arial",
        color="#1A1A1A",
    )

ax_c.set_xlim(-0.55, 4.35)
ax_c.set_ylim(0, 112)

ax_c.set_xticks(x)
ax_c.set_xticklabels(
    labels, fontsize=FONT_TICK_BAR, fontweight="bold", fontfamily="Arial"
)

ax_c.set_yticks([0, 20, 40, 60, 80, 100])

style_ax(ax_c, xlabel="", ylabel="Pass Rate on Valid (%)", tick_labelsize=FONT_TICK_BAR)

for lbl in ax_c.get_xticklabels():
    lbl.set_fontweight("bold")
    lbl.set_fontfamily("Arial")

save_fig(fig_c, "pass_rate_bar")


print(f"\nDone. Figures saved to {OUTPUT_DIR}")
