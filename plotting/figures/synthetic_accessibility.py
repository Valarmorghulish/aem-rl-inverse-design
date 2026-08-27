# -*- coding: utf-8 -*-


import os
import os.path as op
import math
import gzip
import pickle
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.patches import FancyBboxPatch
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


PROJECT_ROOT = os.path.abspath(
    os.environ.get(
        "AEM_RL_ROOT",
        os.path.join(os.path.dirname(__file__), "..", ".."),
    )
)
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "Conductivity_data_merged_processed.csv")

FILES = {
    "AEM_Experimental": RAW_DATA_PATH,
    "PAES": os.path.join(
        PROJECT_ROOT,
        "generated_aem_pairs_strict_500_only_no_predictor",
        "PAES",
        "PAES_unique_500_pairs.csv",
    ),
    "PAEKS": os.path.join(
        PROJECT_ROOT,
        "generated_aem_pairs_strict_500_only_no_predictor",
        "PAEKS",
        "PAEKS_unique_500_pairs.csv",
    ),
    "PAEK": os.path.join(
        PROJECT_ROOT,
        "generated_aem_pairs_strict_500_only_no_predictor",
        "PAEK",
        "PAEK_unique_500_pairs.csv",
    ),
    "Finetuned": os.path.join(
        PROJECT_ROOT,
        "generated_aem_pairs_strict_all_models_with_predictor",
        "Finetuned",
        "Finetuned_unique_500_pairs.csv",
    ),
    "PAP": os.path.join(
        PROJECT_ROOT,
        "generated_aem_pairs_strict_all_models_with_predictor",
        "PAP",
        "PAP_unique_500_pairs.csv",
    ),
    "PBF": os.path.join(
        PROJECT_ROOT,
        "generated_aem_pairs_strict_all_models_with_predictor",
        "PBF",
        "PBF_unique_500_pairs.csv",
    ),
    "PPO": os.path.join(
        PROJECT_ROOT,
        "generated_aem_pairs_strict_all_models_with_predictor",
        "PPO",
        "PPO_unique_500_pairs.csv",
    ),
    "Unbiased": os.path.join(
        PROJECT_ROOT,
        "generated_aem_pairs_strict_all_models_with_predictor",
        "Unbiased",
        "Unbiased_unique_500_pairs.csv",
    ),
}

OUT_DIR = os.path.join(
    os.environ.get(
        "AEM_RL_FIGURE_OUTPUT",
        os.path.join(PROJECT_ROOT, "outputs", "figures"),
    ),
    "synthetic_accessibility",
)

FPSCORE_BASENAME = os.environ.get(
    "AEM_RL_FPSCORES",
    os.path.join(PROJECT_ROOT, "fpscores"),
)

_fscores = None


def readFragmentScores(name: str = FPSCORE_BASENAME):
    global _fscores

    if _fscores is not None:
        return

    if name == "fpscores":
        name = op.join(os.getcwd(), name)

    fname = f"{name}.pkl.gz"

    if not op.exists(fname):
        raise FileNotFoundError(
            f"Cannot find fragment score file: {fname}\n"
            "Please place fpscores.pkl.gz in the working directory, "
            "or set FPSCORE_BASENAME to its full path without the .pkl.gz suffix."
        )

    data = pickle.load(gzip.open(fname, "rb"))

    out_dict = {}

    for row in data:
        for j in range(1, len(row)):
            out_dict[row[j]] = float(row[0])

    _fscores = out_dict


def numBridgeheadsAndSpiro(mol, ri=None):
    n_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    n_bridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)

    return n_bridgehead, n_spiro


def calculateScore(mol):
    if mol is None:
        return np.nan

    readFragmentScores(FPSCORE_BASENAME)

    fp = rdMolDescriptors.GetMorganFingerprint(mol, 2)
    fps = fp.GetNonzeroElements()

    score1 = 0.0
    nf = 0

    for bit_id, v in fps.items():
        nf += v
        score1 += _fscores.get(bit_id, -4) * v

    if nf == 0:
        return np.nan

    score1 /= nf

    n_atoms = mol.GetNumAtoms()
    n_chiral_centers = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))

    ri = mol.GetRingInfo()

    n_bridgeheads, n_spiro = numBridgeheadsAndSpiro(mol, ri)
    n_macrocycles = sum(1 for ring in ri.AtomRings() if len(ring) > 8)

    size_penalty = n_atoms**1.005 - n_atoms
    stereo_penalty = math.log10(n_chiral_centers + 1)
    spiro_penalty = math.log10(n_spiro + 1)
    bridge_penalty = math.log10(n_bridgeheads + 1)
    macrocycle_penalty = math.log10(2) if n_macrocycles > 0 else 0.0

    score2 = (
        0.0
        - size_penalty
        - stereo_penalty
        - spiro_penalty
        - bridge_penalty
        - macrocycle_penalty
    )

    score3 = 0.0

    if n_atoms > len(fps):
        score3 = math.log(float(n_atoms) / len(fps)) * 0.5

    sascore = score1 + score2 + score3

    min_raw = -4.0
    max_raw = 2.5

    sascore = 11.0 - (sascore - min_raw + 1.0) / (max_raw - min_raw) * 9.0

    if sascore > 8.0:
        sascore = 8.0 + math.log(sascore + 1.0 - 9.0)

    if sascore > 10.0:
        sascore = 10.0
    elif sascore < 1.0:
        sascore = 1.0

    return float(sascore)


def strip_role_tokens(s: str) -> str:
    if s is None:
        return ""

    s = str(s).strip()

    return s.replace("<", "").replace(">", "").replace("^", "").replace("~", "")


def sa_score(smiles: str):
    smi = strip_role_tokens(smiles)
    mol = Chem.MolFromSmiles(smi)

    if mol is None:
        return np.nan

    readFragmentScores("fpscores")

    return calculateScore(mol)


HYDRO_CANDIDATES = [
    "Hydrophilic",
    "hydrophilic",
    "hydro_raw",
    "hydro",
    "hydro_smi",
    "hydrophilic_smiles",
    "hydrophilic_raw",
    "hydro_clean",
]

PHOBIC_CANDIDATES = [
    "Hydrophobic",
    "hydrophobic",
    "phobic_raw",
    "phobic",
    "phobic_smi",
    "hydrophobic_smiles",
    "hydrophobic_raw",
    "phobic_clean",
]


def find_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}

    for c in candidates:
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]

    return None


def resolve_segment_columns(df: pd.DataFrame) -> Tuple[str, str]:
    hydro_col = find_first_existing(df, HYDRO_CANDIDATES)
    phobic_col = find_first_existing(df, PHOBIC_CANDIDATES)

    if hydro_col is None or phobic_col is None:
        raise ValueError(
            f"Could not find hydrophilic/hydrophobic columns. "
            f"Available columns: {list(df.columns)}"
        )

    return hydro_col, phobic_col


plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.linewidth": 6.0,
        "xtick.major.width": 6.0,
        "ytick.major.width": 6.0,
        "xtick.major.size": 20,
        "ytick.major.size": 20,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.unicode_minus": False,
    }
)


XTICK_FONTSIZE = 58
CARD_FONTSIZE = 90

AXIS_LINEWIDTH = 7.0
TICK_WIDTH = 6.0
TICK_LENGTH = 20

FILL_ALPHA = 0.95

COLOR_PHI = "#F2AA24"
COLOR_PHO = "#8064A9"


def kde_curve(values: np.ndarray, xmin=1.0, xmax=10.0, n=900, bw_adjust=1.0):
    vals = values[np.isfinite(values)]

    if len(vals) < 2:
        return None, None

    if np.nanstd(vals) == 0:
        return None, None

    xs = np.linspace(xmin, xmax, n)

    kde = gaussian_kde(vals)
    kde.set_bandwidth(bw_method=kde.factor * bw_adjust)

    ys = kde(xs)

    return xs, ys


def format_axis(ax):
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_linewidth(AXIS_LINEWIDTH)

    ax.set_yticks([])

    ax.tick_params(
        axis="y",
        left=False,
        right=False,
        labelleft=False,
        labelright=False,
    )

    ax.set_xticks([2, 4, 6, 8, 10])

    ax.tick_params(
        axis="x",
        labelsize=XTICK_FONTSIZE,
        width=TICK_WIDTH,
        length=TICK_LENGTH,
        pad=14,
        top=False,
        bottom=True,
        labelbottom=True,
    )

    for tick in ax.get_xticklabels():
        tick.set_fontweight("bold")

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")


def plot_phi_pho_color_card(out_dir: str):
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    cards = [
        {
            "x": 0.08,
            "y": 0.58,
            "w": 0.42,
            "h": 0.20,
            "color": COLOR_PHI,
            "label": "Phi",
        },
        {
            "x": 0.08,
            "y": 0.22,
            "w": 0.42,
            "h": 0.20,
            "color": COLOR_PHO,
            "label": "Pho",
        },
    ]

    for item in cards:
        patch = FancyBboxPatch(
            (item["x"], item["y"]),
            item["w"],
            item["h"],
            boxstyle="round,pad=0.0,rounding_size=0.018",
            facecolor=item["color"],
            edgecolor="none",
            linewidth=0,
        )

        ax.add_patch(patch)

        ax.text(
            item["x"] + item["w"] + 0.07,
            item["y"] + item["h"] / 2,
            item["label"],
            ha="left",
            va="center",
            fontsize=CARD_FONTSIZE,
            fontweight="bold",
            color="#111111",
        )

    png_path = op.join(out_dir, "Phi_Pho_color_card.png")
    tif_path = op.join(out_dir, "Phi_Pho_color_card.tif")

    fig.savefig(png_path, dpi=1200, bbox_inches="tight", facecolor="white")
    fig.savefig(tif_path, dpi=1200, bbox_inches="tight", facecolor="white")

    plt.close(fig)

    print(f"[OK] Color card saved -> {png_path}")
    print(f"[OK] Color card saved -> {tif_path}")


def plot_sa_distribution_for_dataset(
    dataset_name: str,
    df: pd.DataFrame,
    hydro_col: str,
    phobic_col: str,
    out_dir: str,
):
    hydro_sa = pd.to_numeric(
        df[hydro_col].astype(str).map(sa_score),
        errors="coerce",
    ).to_numpy()

    phobic_sa = pd.to_numeric(
        df[phobic_col].astype(str).map(sa_score),
        errors="coerce",
    ).to_numpy()

    summary = {
        "dataset": dataset_name,
        "n_rows": len(df),
        "hydro_valid": int(np.isfinite(hydro_sa).sum()),
        "phobic_valid": int(np.isfinite(phobic_sa).sum()),
        "hydro_sa_mean": (
            float(np.nanmean(hydro_sa)) if np.isfinite(hydro_sa).any() else np.nan
        ),
        "phobic_sa_mean": (
            float(np.nanmean(phobic_sa)) if np.isfinite(phobic_sa).any() else np.nan
        ),
        "hydro_sa_median": (
            float(np.nanmedian(hydro_sa)) if np.isfinite(hydro_sa).any() else np.nan
        ),
        "phobic_sa_median": (
            float(np.nanmedian(phobic_sa)) if np.isfinite(phobic_sa).any() else np.nan
        ),
    }

    fig, axes = plt.subplots(1, 2, figsize=(17.5, 6.2))
    fig.patch.set_facecolor("white")

    panels = [
        (axes[0], hydro_sa, COLOR_PHI),
        (axes[1], phobic_sa, COLOR_PHO),
    ]

    ymax_all = 0.0
    curves = []

    for ax, vals, color in panels:
        xs, ys = kde_curve(
            vals,
            xmin=1.0,
            xmax=10.0,
            n=900,
            bw_adjust=1.0,
        )

        curves.append((ax, xs, ys, vals, color))

        if ys is not None:
            ymax_all = max(ymax_all, float(np.max(ys)))

    ymax = max(0.15, ymax_all * 1.08)

    for ax, xs, ys, vals, color in curves:
        vals = vals[np.isfinite(vals)]

        if len(vals) >= 2 and xs is not None and ys is not None:
            ax.fill_between(
                xs,
                0,
                ys,
                color=color,
                alpha=FILL_ALPHA,
                linewidth=0,
                edgecolor="none",
            )

        ax.set_xlim(1.0, 10.0)
        ax.set_ylim(0.0, ymax)
        ax.set_facecolor("white")

        format_axis(ax)

    fig.subplots_adjust(
        left=0.035,
        right=0.985,
        bottom=0.22,
        top=0.98,
        wspace=0.24,
    )

    png_path = op.join(out_dir, f"{dataset_name}_SA_distribution.png")
    tif_path = op.join(out_dir, f"{dataset_name}_SA_distribution.tif")

    fig.savefig(png_path, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(tif_path, dpi=600, bbox_inches="tight", facecolor="white")

    plt.close(fig)

    print(f"[OK] {dataset_name}")
    print(f"     hydro_col = {hydro_col}")
    print(f"     phobic_col = {phobic_col}")
    print(f"     saved -> {png_path}")
    print(f"     saved -> {tif_path}")

    return summary


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    plot_phi_pho_color_card(OUT_DIR)

    summaries = []

    for dataset_name, path in FILES.items():
        if not op.exists(path):
            print(f"[SKIP] {dataset_name}: file not found -> {path}")
            continue

        df = pd.read_csv(path)

        hydro_col, phobic_col = resolve_segment_columns(df)

        summaries.append(
            plot_sa_distribution_for_dataset(
                dataset_name=dataset_name,
                df=df,
                hydro_col=hydro_col,
                phobic_col=phobic_col,
                out_dir=OUT_DIR,
            )
        )

    if summaries:
        summary_df = pd.DataFrame(summaries)

        csv_path = op.join(OUT_DIR, "sa_score_summary_by_dataset.csv")

        summary_df.to_csv(csv_path, index=False)

        print(f"\n[OK] Summary saved -> {csv_path}")
        print(summary_df)
    else:
        print("No datasets were processed.")


if __name__ == "__main__":
    main()
