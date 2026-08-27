# -*- coding: utf-8 -*-


"""Generate the data-distribution, Pareto, and stability overview panels."""
import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import (
    AutoMinorLocator,
    MultipleLocator,
    NullLocator,
    MaxNLocator,
)
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import PchipInterpolator


PROJECT_ROOT = os.path.abspath(
    os.environ.get(
        "AEM_RL_ROOT",
        os.path.join(os.path.dirname(__file__), "..", ".."),
    )
)
OUTPUT_DIR = os.path.join(
    os.environ.get(
        "AEM_RL_FIGURE_OUTPUT",
        os.path.join(PROJECT_ROOT, "outputs", "figures"),
    ),
    "data_overview",
)
os.makedirs(OUTPUT_DIR, exist_ok=True)
file_path = os.path.join(PROJECT_ROOT, "Conductivity_data_merged_processed.csv")
data = pd.read_csv(file_path)

data_cleaned = (
    data[["Conductivity", "SR"]].apply(pd.to_numeric, errors="coerce").dropna()
)


plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.linewidth": 2.2,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 1.8,
        "ytick.major.width": 1.8,
        "xtick.minor.width": 1.2,
        "ytick.minor.width": 1.2,
        "xtick.major.size": 7,
        "ytick.major.size": 7,
        "xtick.minor.size": 4,
        "ytick.minor.size": 4,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "figure.dpi": 240,
        "savefig.dpi": 1200,
    }
)


def get_nice_ylim_and_ticks(y_max):
    """Return a rounded upper limit and evenly spaced y-axis ticks."""
    if y_max <= 10:
        step = 2
    elif y_max <= 20:
        step = 5
    elif y_max <= 50:
        step = 10
    elif y_max <= 100:
        step = 20
    else:
        step = 25

    y_top = math.ceil((y_max * 1.12) / step) * step
    ticks = np.arange(0, y_top + 0.1, step)
    return (0, y_top), ticks


def build_percent_curve(
    values, x_max, bins=30, sigma=1.0, n_dense=3000, preserve_peak=True
):
    values = pd.to_numeric(values, errors="coerce").dropna().to_numpy()
    values = values[(values >= 0) & (values <= x_max)]

    counts, edges = np.histogram(values, bins=bins, range=(0, x_max))
    percent = counts / counts.sum() * 100.0
    centers = (edges[:-1] + edges[1:]) / 2

    raw_peak = percent.max()

    percent_smooth = gaussian_filter1d(percent, sigma=sigma, mode="reflect")
    percent_smooth = np.clip(percent_smooth, 0, None)

    if preserve_peak and percent_smooth.max() > 0:
        percent_smooth = percent_smooth * (raw_peak / percent_smooth.max())

    x_nodes = np.r_[0.0, centers, x_max]
    y_nodes = np.r_[0.0, percent_smooth, 0.0]

    interp = PchipInterpolator(x_nodes, y_nodes)
    x_dense = np.linspace(0, x_max, n_dense)
    y_dense = np.clip(interp(x_dense), 0, None)

    return x_dense, y_dense, raw_peak


def style_axis(ax, xlabel, xlim, ylim, xticks, yticks):
    ax.set_xlabel(xlabel, fontsize=34, fontweight="bold")
    ax.set_ylabel("Intensity (%)", fontsize=34, fontweight="bold")

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=20,
        width=1.8,
        length=7,
        direction="in",
        pad=8,
    )
    ax.tick_params(axis="both", which="minor", width=1.2, length=4, direction="in")

    for spine in ax.spines.values():
        spine.set_linewidth(2.2)
        spine.set_color("black")


def plot_single(
    column, x_max, xlabel, xticks, out_prefix, color_main, bins=30, sigma=1.0
):
    fig, ax = plt.subplots(figsize=(7.4, 5.8), dpi=280)

    x, y, raw_peak = build_percent_curve(
        data_cleaned[column],
        x_max=x_max,
        bins=bins,
        sigma=sigma,
        n_dense=3000,
        preserve_peak=True,
    )

    ylim, yticks = get_nice_ylim_and_ticks(y.max())

    ax.fill_between(
        x, y, 0, color=color_main, alpha=0.28, linewidth=0, antialiased=True
    )

    ax.plot(
        x,
        y,
        color=color_main,
        linewidth=2.4,
        antialiased=True,
        solid_capstyle="round",
        solid_joinstyle="round",
    )

    style_axis(
        ax, xlabel=xlabel, xlim=(0, x_max), ylim=ylim, xticks=xticks, yticks=yticks
    )

    base = os.path.join(OUTPUT_DIR, out_prefix)
    fig.subplots_adjust(left=0.18, right=0.95, bottom=0.18, top=0.95)

    fig.savefig(base + ".png", dpi=1200, facecolor="white")
    fig.savefig(
        base + ".tif",
        dpi=1200,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(base + ".pdf", facecolor="white")
    fig.savefig(base + ".svg", facecolor="white")

    plt.show()
    plt.close(fig)


# 7. Conductivity

plot_single(
    column="Conductivity",
    x_max=265,
    xlabel="Conductivity (mS/cm)",
    xticks=np.arange(0, 251, 50),
    out_prefix="Conductivity_single_adaptive",
    color_main="#39BDB3",
    bins=30,
    sigma=0.9,
)

# 8. SR

plot_single(
    column="SR",
    x_max=170,
    xlabel="SR (%)",
    xticks=np.arange(0, 161, 20),
    out_prefix="SR_single_adaptive",
    color_main="#FF2F66",
    bins=30,
    sigma=0.85,
)


# Pareto front plot


plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.linewidth": 2.2,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 1.8,
        "ytick.major.width": 1.8,
        "xtick.minor.width": 1.2,
        "ytick.minor.width": 1.2,
        "xtick.major.size": 7,
        "ytick.major.size": 7,
        "xtick.minor.size": 4,
        "ytick.minor.size": 4,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "figure.dpi": 240,
        "savefig.dpi": 1200,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


FIG_W, FIG_H = 7.4, 5.8
LABEL_SIZE = 34
TICK_SIZE = 20
LEGEND_SIZE = 16
LINE_W = 2.4
SPINE_W = 2.2


LEFT_M, RIGHT_M, BOTTOM_M, TOP_M = 0.18, 0.95, 0.18, 0.95

GREEN_MAIN = "#39BDB3"
RED_MAIN = "#FF2F66"


file_path = os.path.join(PROJECT_ROOT, "Conductivity_data_merged_processed.csv")
data = pd.read_csv(file_path)

required_cols = ["Conductivity", "SR"]
for col in required_cols:
    if col not in data.columns:
        raise ValueError(f"Required column is missing: {col}")

data["Conductivity"] = pd.to_numeric(data["Conductivity"], errors="coerce")
data["SR"] = pd.to_numeric(data["SR"], errors="coerce")

plot_data = data.dropna(subset=["Conductivity", "SR"]).copy()
plot_data = plot_data[
    (plot_data["Conductivity"] > 0) & (plot_data["SR"] >= 0)
].reset_index(drop=True)

if plot_data.empty:
    raise ValueError("No valid observations remain after data cleaning.")


def compute_pareto_front_exact(df, x_col="Conductivity", y_col="SR"):
    x = df[x_col].to_numpy()
    y = df[y_col].to_numpy()
    n = len(df)
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        dominates_i = (x >= x[i]) & (y <= y[i]) & ((x > x[i]) | (y < y[i]))
        dominates_i[i] = False
        if np.any(dominates_i):
            is_pareto[i] = False

    pareto = df[is_pareto].copy()
    others = df[~is_pareto].copy()

    pareto = pareto.drop_duplicates(subset=[x_col, y_col])
    pareto = pareto.sort_values([x_col, y_col], ascending=[True, True]).reset_index(
        drop=True
    )
    others = others.reset_index(drop=True)
    return pareto, others


pareto_front, other_points = compute_pareto_front_exact(plot_data)

front_line = (
    pareto_front.groupby("Conductivity", as_index=False)["SR"]
    .min()
    .sort_values("Conductivity")
    .reset_index(drop=True)
)


def get_xmax(values):
    return int(math.ceil(np.nanmax(values) / 10.0) * 10)


def get_ymax(values):
    return int(math.ceil(np.nanmax(values) / 10.0) * 10)


xmax = get_xmax(plot_data["Conductivity"].to_numpy())
ymax = get_ymax(plot_data["SR"].to_numpy())

if xmax <= 120:
    x_major = 20
elif xmax <= 300:
    x_major = 50
elif xmax <= 600:
    x_major = 100
else:
    x_major = 200

x_minor = x_major / 2
y_major = 25


x_pad_left = 8
x_pad_right = 8
y_pad_top = 24
y_pad_bottom = 8


fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=240)
ax.set_facecolor("white")

ax.scatter(
    other_points["Conductivity"],
    other_points["SR"],
    s=62,
    color=GREEN_MAIN,
    alpha=0.80,
    label="Other Datapoints",
    zorder=1,
)

ax.plot(
    front_line["Conductivity"],
    front_line["SR"],
    linestyle="--",
    linewidth=LINE_W,
    color=RED_MAIN,
    zorder=2,
)

ax.scatter(
    pareto_front["Conductivity"],
    pareto_front["SR"],
    s=115,
    color=RED_MAIN,
    edgecolor="black",
    linewidth=1.6,
    label="Pareto Front",
    zorder=3,
)

ax.set_xlabel(
    "Conductivity (mS/cm)", fontsize=LABEL_SIZE, fontweight="bold", labelpad=8
)
ax.set_ylabel("SR (%)", fontsize=LABEL_SIZE, fontweight="bold", labelpad=8)

ax.set_xlim(-x_pad_left, xmax + x_pad_right)
ax.set_ylim(ymax + y_pad_bottom, -y_pad_top)

ax.xaxis.set_major_locator(MultipleLocator(x_major))
ax.xaxis.set_minor_locator(MultipleLocator(x_minor))
ax.yaxis.set_major_locator(MultipleLocator(y_major))
ax.yaxis.set_minor_locator(NullLocator())


ax.yaxis.tick_left()
ax.yaxis.set_label_position("left")

ax.tick_params(
    axis="x",
    which="major",
    labelsize=TICK_SIZE,
    width=1.8,
    length=7,
    direction="in",
    pad=8,
    bottom=True,
    top=False,
)
ax.tick_params(
    axis="x", which="minor", width=1.2, length=4, direction="in", bottom=True, top=False
)
ax.tick_params(
    axis="y",
    which="major",
    labelsize=TICK_SIZE,
    width=1.8,
    length=7,
    direction="in",
    pad=8,
    right=False,
    left=True,
    labelright=False,
    labelleft=True,
)

for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight("bold")

for spine in ax.spines.values():
    spine.set_linewidth(SPINE_W)
    spine.set_color("black")

ax.grid(False)

legend = ax.legend(
    loc="upper right",
    frameon=False,
    fontsize=LEGEND_SIZE,
    handlelength=2.0,
    borderpad=0.15,
    scatterpoints=1,
    markerscale=0.95,
)
for text in legend.get_texts():
    text.set_fontweight("bold")


fig.subplots_adjust(left=LEFT_M, right=RIGHT_M, bottom=BOTTOM_M, top=TOP_M)

fig.canvas.draw()


out_base = os.path.join(OUTPUT_DIR, "Pareto_front_left_yaxis")

fig.savefig(out_base + ".png", dpi=1200, facecolor="white")
fig.savefig(
    out_base + ".tif",
    dpi=1200,
    facecolor="white",
    pil_kwargs={"compression": "tiff_lzw"},
)
fig.savefig(out_base + ".pdf", facecolor="white")
fig.savefig(out_base + ".svg", facecolor="white")

plt.show()
plt.close(fig)


# Stability bubble plot


plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.linewidth": 2.2,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 1.8,
        "ytick.major.width": 1.8,
        "xtick.minor.width": 1.2,
        "ytick.minor.width": 1.2,
        "xtick.major.size": 7,
        "ytick.major.size": 7,
        "xtick.minor.size": 4,
        "ytick.minor.size": 4,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "figure.dpi": 240,
        "savefig.dpi": 1200,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    }
)


FIG_W, FIG_H = 7.4, 5.8
LABEL_SIZE = 34
TICK_SIZE = 20
LEGEND_SIZE = 16
CBAR_TICK_SIZE = 14
LEFT_M, RIGHT_M, BOTTOM_M, TOP_M = 0.18, 0.95, 0.18, 0.95

GREEN_MAIN = "#39BDB3"
RED_MAIN = "#FF2F66"
MID_MAIN = "#8EA0C7"
TEMP_CMAP = LinearSegmentedColormap.from_list(
    "temp_map_aligned", [RED_MAIN, MID_MAIN, GREEN_MAIN]
)

SIZE_MAP = {
    "<=1": 28,
    "(1,4]": 50,
    "(4,8]": 78,
    ">8": 112,
}
BUBBLE_SCALE = 1.0


DATA_PATH = os.path.abspath(
    os.environ.get(
        "AEM_RL_STABILITY_CSV",
        os.path.join(PROJECT_ROOT, "stability4_final.csv"),
    )
)
OUT_DIR = OUTPUT_DIR
os.makedirs(OUT_DIR, exist_ok=True)
PAIR_WITH_MODE_FILL = False


REMOVE_TIME_GE = 5000
TIME_TOL = 1e-8


SMI_A_COL = "Hydrophilic"
SMI_B_COL = "Hydrophobic"
COND_COL = "Cond"
TIME_COL = "time(h)"
TEMP_PLOT_COL = "stability_test_temp (C)"
NAOH_COL = "solvent_NaOH (M)"
KOH_COL = "solvent_KOH (M)"

EXPERIMENT_FEATURES_ALL = [
    "Hydrophilic_Fraction",
    "solvent_NaOH (M)",
    "solvent_KOH (M)",
    "RH (%)",
    "theor_IEC (meq/g)",
    "stability_test_temp (C)",
    "prop_test_temp (C)",
    "time(h)",
]

ROUND_MAP = {
    "Hydrophilic_Fraction": 4,
    "solvent_NaOH (M)": 4,
    "solvent_KOH (M)": 4,
    "RH (%)": 2,
    "theor_IEC (meq/g)": 4,
    "stability_test_temp (C)": 2,
    "prop_test_temp (C)": 2,
    "time(h)": 2,
}


def mode_fill_values(df: pd.DataFrame, cols: list) -> dict:
    fill = {}
    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce")
        m = x.mode(dropna=True)
        if len(m) > 0:
            fill[c] = float(m.iloc[0])
        else:
            med = x.median()
            fill[c] = float(med) if pd.notna(med) else 0.0
    return fill


def apply_mode_fill(df: pd.DataFrame, fill: dict) -> pd.DataFrame:
    out = df.copy()
    for c, v in fill.items():
        out[c] = pd.to_numeric(out[c], errors="coerce")
        out[c] = out[c].replace([np.inf, -np.inf], np.nan).fillna(v)
    return out


def conc_bin_label(c):
    if pd.isna(c):
        return np.nan
    if c <= 1:
        return "<=1"
    elif c <= 4:
        return "(1,4]"
    elif c <= 8:
        return "(4,8]"
    else:
        return ">8"


def exclude_long_time_points(
    df: pd.DataFrame, time_col: str, cutoff_ge: float | None, tol: float = 1e-8
) -> pd.DataFrame:
    """Remove samples with time >= cutoff_ge."""
    if cutoff_ge is None:
        return df.copy()

    out = df.copy()
    t = pd.to_numeric(out[time_col], errors="coerce")
    mask_remove = t >= (float(cutoff_ge) - tol)
    return out.loc[~mask_remove].copy()


df = pd.read_csv(DATA_PATH).replace([np.inf, -np.inf], np.nan)
need_cols = [SMI_A_COL, SMI_B_COL, TIME_COL, COND_COL] + EXPERIMENT_FEATURES_ALL
missing = [c for c in need_cols if c not in df.columns]
if missing:
    raise ValueError(f"Required columns are missing: {missing}")

df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce")
df[COND_COL] = pd.to_numeric(df[COND_COL], errors="coerce")
df = df.dropna(subset=[TIME_COL, COND_COL]).copy()
df = df[df[COND_COL] > 0].copy()
df["polymer_unique_id"] = df[SMI_A_COL].astype(str) + "|" + df[SMI_B_COL].astype(str)

for c, nd in ROUND_MAP.items():
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").round(nd)

pair_keys = ["polymer_unique_id"] + [
    c for c in EXPERIMENT_FEATURES_ALL if c != TIME_COL
]
pair_condition_cols = [c for c in EXPERIMENT_FEATURES_ALL if c != TIME_COL]

if PAIR_WITH_MODE_FILL:
    pair_mode_fill = mode_fill_values(df, pair_condition_cols)
    df = apply_mode_fill(df, pair_mode_fill)
    with open(os.path.join(OUT_DIR, "pair_mode_fill.json"), "w", encoding="utf-8") as f:
        json.dump(pair_mode_fill, f, ensure_ascii=False, indent=2)
else:
    df = df.dropna(subset=pair_condition_cols).copy()

fresh = df[df[TIME_COL] == 0].copy()
deg = df[df[TIME_COL] > 0].copy()

fresh_agg = fresh.groupby(pair_keys, as_index=False).agg(Cond_init=(COND_COL, "median"))
deg_time_nunique = (
    deg.groupby(pair_keys)[TIME_COL].nunique().reset_index(name="n_time_deg")
)
bad = deg_time_nunique[deg_time_nunique["n_time_deg"] > 1]
if len(bad) > 0:
    bad_path = os.path.join(OUT_DIR, "bad_pairs_multiple_deg_time.csv")
    bad.to_csv(bad_path, index=False, encoding="utf-8-sig")
    raise ValueError(
        "Multiple degraded-time values were detected for the same pair key. "
        f"Details were written to {bad_path}."
    )

deg_agg = deg.groupby(pair_keys, as_index=False).agg(
    Cond_deg=(COND_COL, "min"), time_deg=(TIME_COL, "median")
)
paired = pd.merge(fresh_agg, deg_agg, on=pair_keys, how="inner")
if len(paired) == 0:
    raise ValueError(
        "No matching fresh (time = 0) and degraded (time > 0) records were found."
    )

rep = df.drop_duplicates(subset=pair_keys)[pair_keys + [SMI_A_COL, SMI_B_COL]].copy()
paired = pd.merge(paired, rep, on=pair_keys, how="left")
paired["retention"] = paired["Cond_deg"] / paired["Cond_init"]
paired = paired.replace([np.inf, -np.inf], np.nan).dropna(subset=["retention"]).copy()
paired = paired[paired["retention"] > 0].copy()


paired = exclude_long_time_points(paired, "time_deg", REMOVE_TIME_GE, tol=TIME_TOL)
if len(paired) == 0:
    raise ValueError(
        f"No observations remain after excluding time >= {REMOVE_TIME_GE} h."
    )

vis = paired.copy()
vis["Retention (%)"] = pd.to_numeric(vis["retention"], errors="coerce") * 100.0
vis["Retention_plot (%)"] = vis["Retention (%)"].clip(lower=0, upper=100)
vis["Time (h)"] = pd.to_numeric(vis["time_deg"], errors="coerce")
vis["T_plot (C)"] = pd.to_numeric(vis[TEMP_PLOT_COL], errors="coerce")

naoh = pd.to_numeric(vis.get(NAOH_COL, 0), errors="coerce").fillna(0)
koh = pd.to_numeric(vis.get(KOH_COL, 0), errors="coerce").fillna(0)
vis["c_total (M)"] = naoh + koh
vis["c_bin"] = vis["c_total (M)"].apply(conc_bin_label)
vis["marker_size"] = vis["c_bin"].map(SIZE_MAP)
vis = vis.dropna(
    subset=[
        "Time (h)",
        "Retention_plot (%)",
        "T_plot (C)",
        "c_total (M)",
        "marker_size",
    ]
).copy()
plot_df = vis.copy()


fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=240)
ax.set_facecolor("white")

for spine in ax.spines.values():
    spine.set_linewidth(2.2)
    spine.set_color("black")

ax.tick_params(
    axis="both",
    which="major",
    direction="in",
    length=7,
    width=1.8,
    labelsize=TICK_SIZE,
    top=False,
    right=False,
    pad=8,
)
ax.tick_params(
    axis="both",
    which="minor",
    direction="in",
    length=4,
    width=1.2,
    top=False,
    right=False,
)
ax.yaxis.set_major_locator(MultipleLocator(20))
ax.yaxis.set_minor_locator(MultipleLocator(10))

norm = Normalize(
    vmin=float(plot_df["T_plot (C)"].min()), vmax=float(plot_df["T_plot (C)"].max())
)
main_sizes = plot_df["marker_size"].values * BUBBLE_SCALE

sc = ax.scatter(
    plot_df["Time (h)"],
    plot_df["Retention_plot (%)"],
    c=plot_df["T_plot (C)"],
    s=main_sizes,
    cmap=TEMP_CMAP,
    norm=norm,
    alpha=0.92,
    edgecolors="white",
    linewidths=0.65,
)

ax.set_xlabel("Time (h)", fontsize=LABEL_SIZE, fontweight="bold", labelpad=8)
ax.set_ylabel("Retention (%)", fontsize=LABEL_SIZE, fontweight="bold", labelpad=10)

x_min_data = float(plot_df["Time (h)"].min())
x_max_data = float(plot_df["Time (h)"].max())
x_range = max(x_max_data - x_min_data, 1.0)


x_left = max(0.0, x_min_data - 0.05 * x_range)
x_right = x_max_data + 0.05 * x_range
ax.set_xlim(x_left, x_right)
ax.set_ylim(-2, 102)


ax.xaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
ax.xaxis.set_minor_locator(AutoMinorLocator(2))

for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight("bold")


legend_x0, legend_y0, legend_w, legend_h = 0.68, 0.11, 0.15, 0.26
cbar_x0, cbar_y0, cbar_w, cbar_h = 0.86, 0.11, 0.032, 0.26

lax = ax.inset_axes([legend_x0, legend_y0, legend_w, legend_h])
lax.set_axis_off()
lax.set_xlim(0, 1)
lax.set_ylim(-0.08, 1.0)

legend_y = [0.76, 0.54, 0.31, 0.09]
legend_labels = ["<=1", "(1,4]", "(4,8]", ">8"]
legend_sizes = [SIZE_MAP[k] * BUBBLE_SCALE for k in legend_labels]

for y, s, lab in zip(legend_y, legend_sizes, legend_labels):
    lax.scatter([0.26], [y], s=s, color="black", edgecolors="black", linewidths=0.4)
    lax.text(
        0.47, y, lab, va="center", ha="left", fontsize=LEGEND_SIZE, fontweight="bold"
    )

cax = ax.inset_axes([cbar_x0, cbar_y0, cbar_w, cbar_h])
cbar = fig.colorbar(sc, cax=cax)
cbar.outline.set_linewidth(0.9)
cbar.ax.tick_params(
    labelsize=CBAR_TICK_SIZE, direction="in", length=2.5, width=0.9, pad=3
)
for label in cbar.ax.get_yticklabels():
    label.set_fontweight("bold")

cbar_ticks = [float(plot_df["T_plot (C)"].min()), float(plot_df["T_plot (C)"].max())]
cbar.set_ticks(cbar_ticks)
cbar.set_ticklabels([f"{int(round(cbar_ticks[0]))}", f"{int(round(cbar_ticks[1]))}"])

title_y = legend_y0 + legend_h + 0.016
ax.text(
    legend_x0 + legend_w * 0.52,
    title_y,
    r"$c$ (M)",
    transform=ax.transAxes,
    ha="center",
    va="bottom",
    fontsize=LEGEND_SIZE,
    fontweight="bold",
)
ax.text(
    cbar_x0 + cbar_w * 0.50,
    title_y,
    "T (°C)",
    transform=ax.transAxes,
    ha="center",
    va="bottom",
    fontsize=LEGEND_SIZE,
    fontweight="bold",
)


fig.subplots_adjust(left=LEFT_M, right=RIGHT_M, bottom=BOTTOM_M, top=TOP_M)

out_base = os.path.join(OUT_DIR, "stability_time_retention_bubble_unified_final")
fig.savefig(out_base + ".png", dpi=1200, facecolor="white")
fig.savefig(
    out_base + ".tif",
    dpi=1200,
    facecolor="white",
    pil_kwargs={"compression": "tiff_lzw"},
)
fig.savefig(out_base + ".pdf", facecolor="white")
fig.savefig(out_base + ".svg", facecolor="white")

plt.show()
plt.close(fig)
