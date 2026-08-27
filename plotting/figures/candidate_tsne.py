# -*- coding: utf-8 -*-


"""Generate joint t-SNE projections for training and generated candidates."""

import os
import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sklearn
from sklearn.manifold import TSNE
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModel
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
matplotlib.rcParams["font.family"] = "Arial"

_sk_ver = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
_iter_key = "max_iter" if _sk_ver >= (1, 2) else "n_iter"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# PATH AUTO-DETECTION
def _candidate_base_dirs():
    candidates = []
    env_base = os.environ.get("AEM_RL_ROOT")
    if env_base:
        candidates.append(Path(env_base))
    try:
        candidates.append(Path(__file__).resolve().parent)
    except NameError:
        pass
    cwd = Path.cwd().resolve()
    candidates.extend([cwd, cwd.parent, Path(__file__).resolve().parents[2]])
    seen, uniq = set(), []
    for p in candidates:
        if str(p) not in seen:
            uniq.append(p)
            seen.add(str(p))
    return uniq


def _looks_like_rlpoly_root(base):
    return (
        (base / "Conductivity_data_merged_processed.csv").exists()
        or (base / "local_models" / "polyBERT").exists()
        or (base / "checkpoints").exists()
    )


def find_base_dir():
    for base in _candidate_base_dirs():
        if base.exists() and _looks_like_rlpoly_root(base):
            return base
    raise FileNotFoundError(
        "AEM-RL project root not found. Set the AEM_RL_ROOT environment variable."
    )


def resolve_path(name, *candidates, must_exist=True, expect_dir=None):
    for c in candidates:
        if not c:
            continue
        p = Path(c)
        if not must_exist:
            return p
        if p.exists():
            if expect_dir is True and not p.is_dir():
                continue
            if expect_dir is False and not p.is_file():
                continue
            return p
    raise FileNotFoundError(
        f"{name} not found. Tried: {[str(Path(c)) for c in candidates if c]}"
    )


BASE = find_base_dir()
LOCAL_BERT = resolve_path(
    "LOCAL_BERT",
    os.environ.get("AEM_RL_POLYBERT"),
    BASE / "local_models" / "polyBERT",
    expect_dir=True,
)
CKPT_PATTERN = str(BASE / "checkpoints" / "predictor_finetune_fold_{fold}")
ORIG_CSV = resolve_path(
    "ORIG_CSV", BASE / "Conductivity_data_merged_processed.csv", expect_dir=False
)
GEN_BASE = resolve_path(
    "GEN_BASE",
    BASE / "generated_aem_pairs_strict_all_models_with_predictor",
    expect_dir=True,
)
OUTPUT = (
    Path(
        os.environ.get(
            "AEM_RL_FIGURE_OUTPUT",
            BASE / "outputs" / "figures",
        )
    )
    / "candidate_tsne"
)
OUTPUT.mkdir(parents=True, exist_ok=True)

GEN_PATHS = {
    "PAP": GEN_BASE / "PAP" / "PAP_unique_500_pairs.csv",
    "PBF": GEN_BASE / "PBF" / "PBF_unique_500_pairs.csv",
    "PPO": GEN_BASE / "PPO" / "PPO_unique_500_pairs.csv",
}
N_FOLDS = 5
MAX_LEN = 256
BATCH_SIZE = 32

# VISUAL STYLE - manuscript version
ORIG_COLOR = "#4DBFB3"
GEN_COLORS = {"PAP": "#E05A63", "PBF": "#E8A228", "PPO": "#7B5EA7"}
CMAP_COND = "plasma"
CMAP_SR = "viridis"
FIG_W, FIG_H = 4.2, 4.2
DPI = 1200
S_ORIG = 32
S_GEN = 42
A_ORIG = 0.70
A_GEN = 0.92
CBAR_LABEL_SIZE = 20
CBAR_TICK_SIZE = 17
# colorbar geometry - appended OUTSIDE scatter, so scatter stays FIG_WxFIG_H
CBAR_W_IN = 0.18
CBAR_PAD_IN = 0.14
CBAR_LBL_IN = 0.60


# HELPERS
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


def clean_ax(ax):
    ax.set_axis_off()


def save_fig(fig, stem):
    fig.savefig(
        OUTPUT / f"{stem}.png",
        dpi=DPI,
        format="png",
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.15,
    )
    fig.savefig(
        OUTPUT / f"{stem}.tif",
        dpi=DPI,
        format="tiff",
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.15,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)
    print(f"    [saved]  {stem}.png / .tif")


# LOAD BERT ENCODERS FROM CHECKPOINTS
def load_bert_from_ckpt(ckpt_dir):
    ckpt_dir = Path(ckpt_dir)
    sf, pt = ckpt_dir / "model.safetensors", ckpt_dir / "pytorch_model.bin"
    if sf.exists():
        from safetensors.torch import load_file

        sd = load_file(str(sf))
    elif pt.exists():
        sd = torch.load(pt, map_location="cpu")
    else:
        raise FileNotFoundError(f"No weights in {ckpt_dir}")
    bert_sd = {
        k[len("bert_model.") :]: v for k, v in sd.items() if k.startswith("bert_model.")
    }
    if not bert_sd:
        bert_sd = dict(sd)
    vocab_size = bert_sd["embeddings.word_embeddings.weight"].shape[0]
    cfg = (
        AutoConfig.from_pretrained(str(ckpt_dir), local_files_only=True)
        if (ckpt_dir / "config.json").exists()
        else AutoConfig.from_pretrained(str(LOCAL_BERT), local_files_only=True)
    )
    cfg.vocab_size = vocab_size
    model = AutoModel.from_config(cfg)
    if model.get_input_embeddings().weight.shape[0] != vocab_size:
        model.resize_token_embeddings(vocab_size)
    missing = {
        k
        for k in model.load_state_dict(bert_sd, strict=False).missing_keys
        if not k.endswith("position_ids")
    }
    if missing:
        raise RuntimeError(f"Missing keys: {missing}")
    return model.eval().to(device)


print("=" * 60)
print(f"BASE: {BASE}\nLOCAL_BERT: {LOCAL_BERT}\nOUTPUT: {OUTPUT}")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(str(LOCAL_BERT), local_files_only=True)
bert_encoders = []
for fold in range(1, N_FOLDS + 1):
    fold_dir = Path(CKPT_PATTERN.format(fold=fold))
    if not fold_dir.exists():
        print(f"  WARNING  Fold dir not found: {fold_dir}")
        continue
    has_direct = (fold_dir / "model.safetensors").exists() or (
        fold_dir / "pytorch_model.bin"
    ).exists()
    if has_direct:
        ckpt_dir = fold_dir
    else:
        ckpts = sorted(
            glob.glob(str(fold_dir / "checkpoint-*")),
            key=lambda x: (
                int(os.path.basename(x).split("-")[-1])
                if os.path.basename(x).split("-")[-1].isdigit()
                else -1
            ),
        )
        if not ckpts:
            print(f"  WARNING  No checkpoint for fold {fold}")
            continue
        ckpt_dir = Path(ckpts[-1])
    print(f"  Fold {fold}: {ckpt_dir.name}")
    bert_encoders.append(load_bert_from_ckpt(ckpt_dir))

if not bert_encoders:
    raise RuntimeError("No BERT encoders loaded.")
print(f"  Loaded {len(bert_encoders)} encoders  |  device: {device}")


# EMBEDDING EXTRACTION
class SMILESPairDataset(Dataset):
    def __init__(self, hydro_list, phobic_list):
        self.hydro, self.phobic = hydro_list, phobic_list

    def __len__(self):
        return len(self.hydro)

    def __getitem__(self, idx):
        def enc(smi):
            s = str(smi) if pd.notna(smi) else ""
            t = tokenizer(
                s,
                truncation=True,
                padding="max_length",
                max_length=MAX_LEN,
                return_tensors="pt",
            )
            return t["input_ids"].squeeze(0), t["attention_mask"].squeeze(0)

        h_ids, h_mask = enc(self.hydro[idx])
        p_ids, p_mask = enc(self.phobic[idx])
        return h_ids, h_mask, p_ids, p_mask


def collate_fn(batch):
    h_ids, h_mask, p_ids, p_mask = zip(*batch)
    return (
        torch.stack(h_ids),
        torch.stack(h_mask),
        torch.stack(p_ids),
        torch.stack(p_mask),
    )


def extract_embeddings(hydro_list, phobic_list, desc="Embedding"):
    ds = SMILESPairDataset(hydro_list, phobic_list)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    all_fold_embs = []
    for i, bert in enumerate(bert_encoders, 1):
        fold_embs = []
        with torch.no_grad():
            for h_ids, h_mask, p_ids, p_mask in tqdm(
                loader, desc=f"  {desc}|fold{i}", leave=False
            ):
                h_cls = bert(
                    input_ids=h_ids.to(device), attention_mask=h_mask.to(device)
                ).last_hidden_state[:, 0, :]
                p_cls = bert(
                    input_ids=p_ids.to(device), attention_mask=p_mask.to(device)
                ).last_hidden_state[:, 0, :]
                fold_embs.append(torch.cat([h_cls, p_cls], dim=1).cpu().numpy())
        all_fold_embs.append(np.concatenate(fold_embs, axis=0))
    return np.mean(np.stack(all_fold_embs, axis=0), axis=0).astype(np.float32)


# STEP 1 - Original AEM
print("\n[1/4] Loading original AEM dataset ...")
df_raw = pd.read_csv(ORIG_CSV)
hydro_col = detect_col(df_raw, "Hydrophilic", "hydro_smi", "hydrophilic")
phobic_col = detect_col(df_raw, "Hydrophobic", "phobic_smi", "hydrophobic")
cond_col = detect_col(df_raw, "Conductivity", "conductivity", "cond")
sr_col = detect_col(df_raw, "SR", "swelling_ratio", "swelling", "water_uptake", "wu")
print(f"  hydro='{hydro_col}'  phobic='{phobic_col}'  cond='{cond_col}'  SR='{sr_col}'")

df_all = df_raw.dropna(subset=[hydro_col, phobic_col]).reset_index(drop=True)
df_struct = (
    df_all.drop_duplicates(subset=[hydro_col, phobic_col]).copy().reset_index(drop=True)
)
print(f"  Rows: {len(df_all)}  |  Unique pairs: {len(df_struct)}")

perf_cols = [c for c in [cond_col, sr_col] if c is not None]
if perf_cols:
    df_work = df_all.copy()
    for c in perf_cols:
        df_work[c] = pd.to_numeric(df_work[c], errors="coerce")
    agg = df_work.groupby([hydro_col, phobic_col])[perf_cols].mean().reset_index()
    df_struct = df_struct.drop(columns=perf_cols, errors="ignore").merge(
        agg, on=[hydro_col, phobic_col], how="left"
    )
    for c in perf_cols:
        print(f"  {c}: {df_struct[c].notna().sum()}/{len(df_struct)} pairs (all temps)")

df_orig = df_struct
fp_orig = extract_embeddings(
    df_orig[hydro_col].tolist(), df_orig[phobic_col].tolist(), desc="Original AEM"
)
print(f"  Embedding shape: {fp_orig.shape}")

# STEP 2 - Generated datasets
print("\n[2/4] Loading generated datasets ...")
gen_fps, gen_dfs = {}, {}
for name, path in GEN_PATHS.items():
    path = Path(path)
    if not path.is_file():
        print(f"  WARNING  Not found: {path}")
        continue
    df = pd.read_csv(path)
    g_hydro = detect_col(df, "hydro_smi", "Hydrophilic", "hydrophilic")
    g_phobic = detect_col(df, "phobic_smi", "Hydrophobic", "hydrophobic")
    if not g_hydro or not g_phobic:
        print(f"  WARNING  {name} missing hydro/phobic columns")
        continue
    df = df.dropna(subset=[g_hydro, g_phobic]).reset_index(drop=True)
    fps = extract_embeddings(df[g_hydro].tolist(), df[g_phobic].tolist(), desc=name)
    gen_fps[name] = fps
    gen_dfs[name] = df
    pred_c = detect_col(df, "predicted_conductivity")
    pred_s = detect_col(df, "predicted_sr")
    print(f"  {name}: {len(fps)} pairs | pred_cond='{pred_c}'  pred_sr='{pred_s}'")

available = list(gen_fps.keys())
if not available:
    raise RuntimeError("No generated datasets loaded.")

# STEP 3 - Joint t-SNE
print("\n[3/4] Running joint t-SNE ...")
all_fps = np.vstack([fp_orig] + [gen_fps[n] for n in available])
print(f"  Matrix: {all_fps.shape}")
emb_all = TSNE(
    n_components=2,
    perplexity=40,
    learning_rate="auto",
    init="pca",
    random_state=42,
    **{_iter_key: 1200},
).fit_transform(all_fps)
n_orig = len(fp_orig)
emb_orig = emb_all[:n_orig]
emb_gen, cursor = {}, n_orig
for name in available:
    k = len(gen_fps[name])
    emb_gen[name] = emb_all[cursor : cursor + k]
    cursor += k
print("  Done.")

# STEP 4 - Plots
print("\n[4/4] Generating plots ...")

# -- Part A: Diversity ---------------------------------------------------------
print("\n  [Part A] Diversity")
for name in available:
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.scatter(
        emb_orig[:, 0],
        emb_orig[:, 1],
        c=ORIG_COLOR,
        s=S_ORIG,
        alpha=A_ORIG,
        linewidths=0,
        zorder=2,
    )
    ax.scatter(
        emb_gen[name][:, 0],
        emb_gen[name][:, 1],
        c=GEN_COLORS[name],
        s=S_GEN,
        alpha=A_GEN,
        linewidths=0,
        zorder=3,
    )
    clean_ax(ax)
    plt.tight_layout(pad=0.4)
    save_fig(fig, f"tsne_diversity_{name}")


# -- Performance values & unified scale ---------------------------------------
def load_perf(col):
    if col is None:
        return None
    return pd.to_numeric(df_orig[col], errors="coerce").values.astype(np.float32)


def global_range(arr):
    if arr is None:
        return None, None
    v = np.isfinite(arr)
    return (None, None) if not v.any() else (float(arr[v].min()), float(arr[v].max()))


cond_vals = load_perf(cond_col)
sr_vals = load_perf(sr_col)
COND_VMIN, COND_VMAX = global_range(cond_vals)
SR_VMIN, SR_VMAX = global_range(sr_vals)

if cond_vals is not None:
    finite = cond_vals[np.isfinite(cond_vals)]
    print(
        f"  Conductivity: [{finite.min():.2f}, {finite.max():.2f}]  mean={finite.mean():.2f}"
    )
print(f"  Unified Conductivity scale: [{COND_VMIN:.2f}, {COND_VMAX:.2f}]")
print(f"  Unified SR scale:           [{SR_VMIN:.2f}, {SR_VMAX:.2f}]")

# -- Part B: Performance -------------------------------------------------------
print("\n  [Part B] Performance")


def perf_plot(name, orig_vals, pred_col_kw, cbar_label, file_suffix, cmap, vmin, vmax):
    """
    Scatter area = FIG_W x FIG_H (same as diversity plots).
    Colorbar appended outside to the right.
    alpha=0.45, fontweight=bold, unit label - all from manuscript version.
    """
    if orig_vals is None or vmin is None:
        return
    valid_orig = np.isfinite(orig_vals)
    if valid_orig.sum() < 10:
        return

    df_gen = gen_dfs[name]
    pred_col = detect_col(df_gen, pred_col_kw)
    if pred_col is not None:
        gen_vals = pd.to_numeric(df_gen[pred_col], errors="coerce").values.astype(
            np.float32
        )
        valid_gen = np.isfinite(gen_vals)
        print(f"    {name}: {valid_gen.sum()} generated pts with {file_suffix}")
    else:
        gen_vals = None
        valid_gen = None
        print(f"    INFO  '{pred_col_kw}' not found in {name}")

    # Figure: scatter = FIG_W x FIG_H, colorbar appended outside
    total_w = FIG_W + CBAR_PAD_IN + CBAR_W_IN + CBAR_LBL_IN
    fig = plt.figure(figsize=(total_w, FIG_H))
    ax = fig.add_axes([0.0, 0.0, FIG_W / total_w, 1.0])
    cax = fig.add_axes([(FIG_W + CBAR_PAD_IN) / total_w, 0.0, CBAR_W_IN / total_w, 1.0])

    sc = ax.scatter(
        emb_orig[valid_orig, 0],
        emb_orig[valid_orig, 1],
        c=orig_vals[valid_orig],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        s=S_ORIG,
        alpha=0.45,
        linewidths=0,
        zorder=2,
    )
    eg = emb_gen[name]
    if gen_vals is not None and valid_gen.sum() > 0:
        ax.scatter(
            eg[valid_gen, 0],
            eg[valid_gen, 1],
            c=gen_vals[valid_gen],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            s=S_ORIG,
            alpha=0.45,
            linewidths=0,
            zorder=3,
        )

    ax.set_aspect("equal", adjustable="datalim")

    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label(
        cbar_label,
        fontsize=CBAR_LABEL_SIZE,
        fontweight="bold",
        color="#1A1A1A",
        labelpad=10,
    )
    cbar.ax.tick_params(labelsize=CBAR_TICK_SIZE, width=1.0, length=4)
    for lbl in cbar.ax.get_yticklabels():
        lbl.set_fontweight("bold")
    cbar.outline.set_linewidth(0.8)

    clean_ax(ax)
    save_fig(fig, f"tsne_{file_suffix}_{name}")


for name in available:
    perf_plot(
        name,
        cond_vals,
        "predicted_conductivity",
        "Conductivity (mS/cm)",
        "conductivity",
        CMAP_COND,
        COND_VMIN,
        COND_VMAX,
    )
    perf_plot(name, sr_vals, "predicted_sr", "SR (%)", "SR", CMAP_SR, SR_VMIN, SR_VMAX)

# -- Color swatch --------------------------------------------------------------
print("\n  [Swatch]")
SWATCH = [
    ("Original AEM", ORIG_COLOR),
    ("PAP", GEN_COLORS["PAP"]),
    ("PBF", GEN_COLORS["PBF"]),
    ("PPO", GEN_COLORS["PPO"]),
]
n_sw = len(SWATCH)
sw_w, sw_h, gap, pad_x, pad_y = 1.10, 0.52, 0.22, 0.20, 0.25
fig_w = pad_x * 2 + sw_w + 0.18 + 2.0
fig_h = pad_y * 2 + n_sw * sw_h + (n_sw - 1) * gap
fig_sw = plt.figure(figsize=(fig_w, fig_h))
ax_sw = fig_sw.add_axes([0, 0, 1, 1])
ax_sw.set_xlim(0, fig_w)
ax_sw.set_ylim(0, fig_h)
ax_sw.axis("off")
for i, (label, color) in enumerate(SWATCH):
    y_top = fig_h - pad_y - i * (sw_h + gap)
    ax_sw.add_patch(
        mpatches.FancyBboxPatch(
            (pad_x, y_top - sw_h),
            sw_w,
            sw_h,
            boxstyle="round,pad=0.02",
            facecolor=color,
            edgecolor="none",
        )
    )
    ax_sw.text(
        pad_x + sw_w + 0.18,
        y_top - sw_h / 2,
        label,
        ha="left",
        va="center",
        fontsize=18,
        fontweight="bold",
        fontfamily="Arial",
        color="#1A1A1A",
    )
save_fig(fig_sw, "tsne_color_swatch")

print(f"\nDone - {len(available)*3} plots + 1 swatch to {OUTPUT}")
