# -*- coding: utf-8 -*-


"""Plot the final-layer averaged attention map of the predictor."""

import os
import json
import warnings
import joblib
import torch
import torch.nn as nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from transformers import AutoConfig, AutoModel, AutoTokenizer

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Paths and settings
PROJECT_ROOT = os.path.abspath(
    os.environ.get(
        "AEM_RL_ROOT",
        os.path.join(os.path.dirname(__file__), "..", ".."),
    )
)
LOCAL_BERT_PATH = os.path.abspath(
    os.environ.get(
        "AEM_RL_POLYBERT",
        os.path.join(PROJECT_ROOT, "local_models", "polyBERT"),
    )
)
ENSEMBLE_DIR = os.path.abspath(
    os.environ.get(
        "AEM_RL_PREDICTOR_ENSEMBLE",
        os.path.join(PROJECT_ROOT, "checkpoints", "transformer_predictor_ensemble"),
    )
)
OUTPUT_DIR = os.path.abspath(
    os.environ.get(
        "AEM_RL_FIGURE_OUTPUT",
        os.path.join(PROJECT_ROOT, "outputs", "figures"),
    )
)
os.makedirs(OUTPUT_DIR, exist_ok=True)
FOLD_TO_LOAD = 1

# representative example
SMILES = "C[N+]1(C)CCC(C2=CC=C(C3=CC=C(C4=CC=C([*])C=C4)C=C3)C=C2)[*]CC1"

# output file
OUT_FIG = os.path.join(OUTPUT_DIR, "fig_predictor_attention_average_final")

# display options
MAX_LENGTH = 128
REMOVE_SPECIAL_TOKENS_FROM_DISPLAY = True

# robust visualization
USE_QUANTILE_CLIP = True
Q_LOW = 0.02
Q_HIGH = 0.98

# colormap: low = yellow, high = dark purple
CMAP = matplotlib.colormaps["plasma_r"].copy()
CMAP.set_bad("white")

# style
DPI = 1200
FIG_W, FIG_H = 12.5, 12.5
TOKEN_FONT = 10
CB_FONT = 16
SPINE_W = 2.0
GRID_W = 0.28
TICK_LEN = 3.0
TICK_W = 1.4


# 2. Load predictor metadata
fold_dir = os.path.join(ENSEMBLE_DIR, f"fold_{FOLD_TO_LOAD}")

with open(os.path.join(ENSEMBLE_DIR, "metadata.json"), "r", encoding="utf-8") as f:
    meta = json.load(f)

n_targets = len(meta["multi_task_targets"])
unfreeze_layers = meta["unfreeze_layers_reg"]

print(f"targets = {meta['multi_task_targets']}")
print(f"unfreeze_layers = {unfreeze_layers}")

scaler_cont = joblib.load(os.path.join(fold_dir, "scaler_cont.pkl"))
ohe_cat = joblib.load(os.path.join(fold_dir, "ohe_cat.pkl"))
n_cont = len(scaler_cont.mean_)
n_cat = ohe_cat.get_feature_names_out().shape[0]

print(f"n_cont = {n_cont}, n_cat = {n_cat}")


# 3. Predictor definition
class AEMTransformerPredictor(nn.Module):
    def __init__(
        self,
        finetuned_bert_model,
        n_cont_cond,
        n_cat_cond,
        n_targets,
        dropout=0.1,
        unfreeze_layers_reg=4,
    ):
        super().__init__()
        self.bert_model = finetuned_bert_model

        total_layers = self.bert_model.config.num_hidden_layers

        for p in self.bert_model.parameters():
            p.requires_grad_(False)

        for i, layer in enumerate(self.bert_model.encoder.layer):
            if i >= total_layers - unfreeze_layers_reg:
                for p in layer.parameters():
                    p.requires_grad_(True)

        for p in self.bert_model.embeddings.parameters():
            p.requires_grad_(True)

        bert_dim = self.bert_model.config.hidden_size
        shared_in = bert_dim * 2 + n_cont_cond + n_cat_cond

        self.shared_head = nn.Sequential(
            nn.Linear(shared_in, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.task_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(512, 256),
                    nn.GELU(),
                    nn.Linear(256, 128),
                    nn.GELU(),
                    nn.Linear(128, 1),
                )
                for _ in range(n_targets)
            ]
        )

    def _get_smiles_embedding(self, input_ids, attention_mask):
        return self.bert_model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]

    def forward(
        self,
        hydro_input_ids,
        hydro_attention_mask,
        phobic_input_ids,
        phobic_attention_mask,
        conditions,
        labels=None,
        mask=None,
        **kwargs,
    ):
        h = self._get_smiles_embedding(hydro_input_ids, hydro_attention_mask)
        p = self._get_smiles_embedding(phobic_input_ids, phobic_attention_mask)
        x = self.shared_head(torch.cat([h, p, conditions], dim=1))
        out = torch.cat([head(x) for head in self.task_heads], dim=1)
        return out


# 4. Load predictor checkpoint
bert_cfg = AutoConfig.from_pretrained(fold_dir, local_files_only=True)
bert_base = AutoModel.from_config(bert_cfg)

model = AEMTransformerPredictor(
    finetuned_bert_model=bert_base,
    n_cont_cond=n_cont,
    n_cat_cond=n_cat,
    n_targets=n_targets,
    unfreeze_layers_reg=unfreeze_layers,
)

state = torch.load(os.path.join(fold_dir, "pytorch_model.bin"), map_location="cpu")

load_info = model.load_state_dict(state, strict=False)
print("Predictor weights loaded.")
if len(load_info.missing_keys) > 0:
    print("missing_keys:", load_info.missing_keys)
if len(load_info.unexpected_keys) > 0:
    print("unexpected_keys:", load_info.unexpected_keys)

encoder = model.bert_model.to(device)
encoder.eval()


# 5. Tokenize example SMILES
tok = AutoTokenizer.from_pretrained(LOCAL_BERT_PATH, local_files_only=True)

enc = tok(
    SMILES,
    return_tensors="pt",
    add_special_tokens=True,
    truncation=True,
    max_length=MAX_LENGTH,
)

ids = enc["input_ids"]
mask = enc["attention_mask"]
tokens = tok.convert_ids_to_tokens(ids[0])
S = len(tokens)

print(f"Sequence length = {S}")
print(f"Tokens = {tokens}")


# 6. Extract final-layer attentions
encoder.config.output_attentions = True

with torch.no_grad():
    outputs = encoder(
        input_ids=ids.to(device),
        attention_mask=mask.to(device),
        output_attentions=True,
        return_dict=True,
    )

if outputs.attentions is None:
    raise ValueError("No attention tensors were returned.")

# shape: (num_heads, seq_len, seq_len)
attn = outputs.attentions[-1].squeeze(0).detach().cpu().float().numpy()
H = attn.shape[0]
print(f"Final-layer attention extracted successfully: heads = {H}, seq_len = {S}")


# 7. Build averaged attention map
def clip_and_rescale(a, q_low=0.02, q_high=0.98):
    if q_low is None or q_high is None:
        lo = np.min(a)
        hi = np.max(a)
    else:
        lo = np.quantile(a, q_low)
        hi = np.quantile(a, q_high)

    a_clip = np.clip(a, lo, hi)
    return (a_clip - lo) / (hi - lo + 1e-12)


# averaged attention map across all heads in the final layer
a_avg_raw = attn.mean(axis=0)

if USE_QUANTILE_CLIP:
    a_avg = clip_and_rescale(a_avg_raw, q_low=Q_LOW, q_high=Q_HIGH)
else:
    a_avg = clip_and_rescale(a_avg_raw, q_low=None, q_high=None)

# optional display trim
if REMOVE_SPECIAL_TOKENS_FROM_DISPLAY and len(tokens) >= 2:
    display_tokens = tokens[1:-1]
    a_avg_disp = a_avg[1:-1, 1:-1]
else:
    display_tokens = tokens
    a_avg_disp = a_avg


# 8. Draw figure
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
fig.patch.set_facecolor("white")

im = ax.imshow(
    a_avg_disp, cmap=CMAP, aspect="auto", vmin=0.0, vmax=1.0, interpolation="nearest"
)

# ticks
ax.set_xticks(np.arange(len(display_tokens)))
ax.set_xticklabels(
    display_tokens,
    rotation=90,
    fontsize=TOKEN_FONT,
    fontfamily="monospace",
    fontweight="bold",
    ha="center",
)

ax.set_yticks(np.arange(len(display_tokens)))
ax.set_yticklabels(
    display_tokens, fontsize=TOKEN_FONT, fontfamily="monospace", fontweight="bold"
)

ax.tick_params(axis="both", which="major", length=TICK_LEN, width=TICK_W, pad=1.5)

# make sure tick labels are bold
for lbl in ax.get_xticklabels():
    lbl.set_fontweight("bold")
for lbl in ax.get_yticklabels():
    lbl.set_fontweight("bold")

# grid
ax.set_xticks(np.arange(-0.5, len(display_tokens), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(display_tokens), 1), minor=True)
ax.grid(which="minor", color="white", linewidth=GRID_W)
ax.tick_params(which="minor", length=0)

# thicker spines
for sp in ax.spines.values():
    sp.set_linewidth(SPINE_W)

# colorbar
cb = plt.colorbar(im, ax=ax, shrink=0.66, pad=0.03, aspect=24, fraction=0.048)
cb.ax.tick_params(labelsize=CB_FONT, width=1.6, length=4.0)
cb.outline.set_linewidth(SPINE_W)
cb.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

for lbl in cb.ax.get_yticklabels():
    lbl.set_fontweight("bold")

# save
try:
    plt.savefig(
        f"{OUT_FIG}.png",
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compress_level": 5},
    )
except TypeError:
    plt.savefig(f"{OUT_FIG}.png", dpi=DPI, bbox_inches="tight", facecolor="white")

plt.savefig(f"{OUT_FIG}.pdf", bbox_inches="tight", facecolor="white")

print(f"Saved: {OUT_FIG}.png")
print(f"Saved: {OUT_FIG}.pdf")
print(f"Actual number of heads averaged in the final layer = {H}")

plt.show()
