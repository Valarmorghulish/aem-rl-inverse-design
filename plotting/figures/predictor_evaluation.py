# -*- coding: utf-8 -*-


"""Reload five predictor folds and generate evaluation tables and parity plots."""

# This script evaluates saved checkpoints; it does not retrain the predictor.
# 1) Rebuild the original train/validation/test split and scalers
# 2) Load 5 fold checkpoints
# 3) Output:
#    - Table A: 5 fold-specific models on the SAME independent test set
#    - Table B: Ensemble performance on the SAME independent test set
# 4) Plot single-target TIFF figures (1200 dpi, manuscript style)

import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModel
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "Arial"

PROJECT_ROOT = os.path.abspath(
    os.environ.get(
        "AEM_RL_ROOT",
        os.path.join(os.path.dirname(__file__), "..", ".."),
    )
)
RAW_DATA_PATH = os.path.abspath(
    os.environ.get(
        "AEM_RL_DATA_CSV",
        os.path.join(PROJECT_ROOT, "Conductivity_data_merged_processed.csv"),
    )
)
LOCAL_BERT_PATH = os.path.abspath(
    os.environ.get(
        "AEM_RL_POLYBERT",
        os.path.join(PROJECT_ROOT, "local_models", "polyBERT"),
    )
)
CHECKPOINT_ROOT = os.path.abspath(
    os.environ.get(
        "AEM_RL_PREDICTOR_CHECKPOINTS",
        os.path.join(PROJECT_ROOT, "checkpoints"),
    )
)
OUTPUT_DIR = os.path.join(
    os.environ.get(
        "AEM_RL_FIGURE_OUTPUT",
        os.path.join(PROJECT_ROOT, "outputs", "figures"),
    ),
    "predictor_evaluation",
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALL_POTENTIAL_CONT_COLS = [
    "SolventBoilingPoint",
    "SolventMeltingPoint",
    "SolventDensity",
    "SolventLogP",
    "SolventViscosity",
    "SolventHeatofVaporization",
    "SolventSurfaceTension",
    "SolventWeight",
    "DryTemp",
    "HydrophilicFrac",
    "IEC",
    "Mn",
    "Mw",
    "Temperature",
    "RH",
    "Thickness",
    "MechTemp",
    "TenM",
]
ALL_POTENTIAL_CAT_COLS = [
    "PolymerArchitecture",
    "Solvent",
    "FabMeth",
    "memAlkalineStability",
    "Crosslinking",
]
ALL_POTENTIAL_TARGETS = ["Conductivity", "SR", "WU", "TenS"]

# ---------- basic ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if "GLOBAL_MAX_SMILES_LEN" not in globals():
    GLOBAL_MAX_SMILES_LEN = 256

# ---------- strict preprocessing ----------
df_eval = pd.read_csv(RAW_DATA_PATH)

df_eval["Hydrophilic"] = df_eval["Hydrophilic"].astype(str).replace("nan", np.nan)
df_eval["Hydrophobic"] = df_eval["Hydrophobic"].astype(str).replace("nan", np.nan)

cols_to_check_in_cont = [
    col for col in ALL_POTENTIAL_CONT_COLS if col not in ALL_POTENTIAL_TARGETS
]
for col in cols_to_check_in_cont + ALL_POTENTIAL_TARGETS:
    if col in df_eval.columns:
        df_eval[col] = pd.to_numeric(df_eval[col], errors="coerce")

input_missing_ratios = {
    col: df_eval[col].isnull().sum() / len(df_eval)
    for col in cols_to_check_in_cont
    if col in df_eval.columns
}
missing_df = pd.DataFrame(
    list(input_missing_ratios.items()), columns=["Feature", "Missing_Ratio"]
).sort_values(by="Missing_Ratio", ascending=False)

MISSING_THRESHOLD = 0.5
cols_to_drop = missing_df[missing_df["Missing_Ratio"] > MISSING_THRESHOLD][
    "Feature"
].tolist()
df_eval.drop(columns=cols_to_drop, inplace=True, errors="ignore")

condition_cont_cols = [
    col
    for col in ALL_POTENTIAL_CONT_COLS
    if col in df_eval.columns and col not in ALL_POTENTIAL_TARGETS
]
condition_cat_cols = [col for col in ALL_POTENTIAL_CAT_COLS if col in df_eval.columns]
multi_task_targets = [col for col in ALL_POTENTIAL_TARGETS if col in df_eval.columns]

df_eval.dropna(subset=multi_task_targets, how="all", inplace=True)
df_eval.reset_index(drop=True, inplace=True)

if "Conductivity" in df_eval.columns:
    df_eval["Conductivity"] = np.log1p(df_eval["Conductivity"])

print("multi_task_targets =", multi_task_targets)

# ---------- exact split ----------
train_val_df, test_df = train_test_split(df_eval, test_size=0.15, random_state=42)
train_val_df = train_val_df.copy().reset_index(drop=True)
test_df = test_df.copy().reset_index(drop=True)

N_SPLITS = 5
stratify_col = f"{multi_task_targets[0]}_present"
train_val_df[stratify_col] = train_val_df[multi_task_targets[0]].notna().astype(int)

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

tokenizer_reg = AutoTokenizer.from_pretrained(LOCAL_BERT_PATH)


# ---------- dataset/model ----------
class AEMRegressionDataset(Dataset):
    def __init__(
        self,
        dataframe,
        tokenizer,
        cont_cols,
        cat_cols,
        target_cols,
        scaler_cont,
        ohe_cat,
        scaler_targets,
        max_length=GLOBAL_MAX_SMILES_LEN,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cont_cols = cont_cols
        self.cat_cols = cat_cols
        self.target_cols = target_cols
        self.scaler_cont = scaler_cont
        self.ohe_cat = ohe_cat
        self.scaler_targets = scaler_targets

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[[idx]]
        hydro_smi = (
            str(row["Hydrophilic"].iloc[0])
            if pd.notna(row["Hydrophilic"].iloc[0])
            else ""
        )
        phobic_smi = (
            str(row["Hydrophobic"].iloc[0])
            if pd.notna(row["Hydrophobic"].iloc[0])
            else ""
        )

        hydro_tokens = self.tokenizer(
            hydro_smi,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        phobic_tokens = self.tokenizer(
            phobic_smi,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        conditions_cont = self.scaler_cont.transform(row[self.cont_cols]).flatten()
        conditions_cat = self.ohe_cat.transform(
            row[self.cat_cols].astype(str)
        ).flatten()
        conditions_vec = np.concatenate([conditions_cont, conditions_cat])

        targets_np = row[self.target_cols].values.astype(np.float32).flatten()
        mask = ~np.isnan(targets_np)
        targets_np_filled = np.nan_to_num(targets_np, nan=0.0)
        targets_scaled = self.scaler_targets.transform(
            targets_np_filled.reshape(1, -1)
        ).flatten()
        targets_scaled[~mask] = 0.0

        return {
            "hydro_input_ids": torch.tensor(
                hydro_tokens["input_ids"].squeeze(), dtype=torch.long
            ),
            "hydro_attention_mask": torch.tensor(
                hydro_tokens["attention_mask"].squeeze(), dtype=torch.long
            ),
            "phobic_input_ids": torch.tensor(
                phobic_tokens["input_ids"].squeeze(), dtype=torch.long
            ),
            "phobic_attention_mask": torch.tensor(
                phobic_tokens["attention_mask"].squeeze(), dtype=torch.long
            ),
            "conditions": torch.tensor(conditions_vec, dtype=torch.float),
            "labels": torch.tensor(targets_scaled, dtype=torch.float),
            "mask": torch.tensor(mask, dtype=torch.bool),
        }


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
        unfreeze_start_layer = total_layers - unfreeze_layers_reg

        for p in self.bert_model.parameters():
            p.requires_grad_(False)
        for i, layer in enumerate(self.bert_model.encoder.layer):
            if i >= unfreeze_start_layer:
                for p in layer.parameters():
                    p.requires_grad_(True)
        for p in self.bert_model.embeddings.parameters():
            p.requires_grad_(True)

        bert_output_dim = self.bert_model.config.hidden_size
        shared_input_size = (bert_output_dim * 2) + n_cont_cond + n_cat_cond

        self.shared_head = nn.Sequential(
            nn.Linear(shared_input_size, 1024),
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
        hydro_fp = self._get_smiles_embedding(hydro_input_ids, hydro_attention_mask)
        phobic_fp = self._get_smiles_embedding(phobic_input_ids, phobic_attention_mask)
        full_embedding = torch.cat([hydro_fp, phobic_fp, conditions], dim=1)
        shared_output = self.shared_head(full_embedding)
        logits = torch.cat([head(shared_output) for head in self.task_heads], dim=1)
        if labels is not None:
            loss_fct = nn.MSELoss(reduction="none")
            se = loss_fct(logits, labels)
            masked_se = torch.where(mask, se, torch.zeros_like(se))
            valid_count = torch.sum(mask)
            loss = (
                torch.sum(masked_se) / valid_count
                if valid_count > 0
                else torch.tensor(0.0, device=logits.device)
            )
            return loss, logits
        return logits


class AEMDataCollator:
    def __call__(self, features):
        return {
            key: torch.stack([f[key] for f in features]) for key in features[0].keys()
        }


# ---------- checkpoint helpers ----------
def get_fold_checkpoint_dir(fold_idx):
    fold_dir = os.path.join(CHECKPOINT_ROOT, f"predictor_finetune_fold_{fold_idx}")
    ckpts = sorted(
        glob.glob(os.path.join(fold_dir, "checkpoint-*")),
        key=lambda x: int(os.path.basename(x).split("-")[-1]),
    )
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found in {fold_dir}")
    return ckpts[-1]


def load_checkpoint_state_dict(ckpt_dir):
    sf_path = os.path.join(ckpt_dir, "model.safetensors")
    pt_path = os.path.join(ckpt_dir, "pytorch_model.bin")
    if os.path.exists(sf_path):
        from safetensors.torch import load_file

        return load_file(sf_path)
    elif os.path.exists(pt_path):
        return torch.load(pt_path, map_location="cpu")
    raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin in {ckpt_dir}")


def get_head_count_from_state_dict(state_dict):
    head_ids = sorted(
        {int(k.split(".")[1]) for k in state_dict if k.startswith("task_heads.")}
    )
    return len(head_ids), head_ids


def build_model_from_checkpoint(ckpt_dir, n_cont_cond, n_cat_cond, expected_n_targets):
    state_dict = load_checkpoint_state_dict(ckpt_dir)
    head_count, head_ids = get_head_count_from_state_dict(state_dict)
    print(f"Checkpoint heads = {head_count}, ids = {head_ids}")

    if head_count != expected_n_targets:
        raise RuntimeError(
            f"{expected_n_targets} targets vs {head_count} checkpoint heads"
        )

    emb_key = "bert_model.embeddings.word_embeddings.weight"
    ckpt_vocab_size = state_dict[emb_key].shape[0]

    ckpt_config_path = os.path.join(ckpt_dir, "config.json")
    reg_config = (
        AutoConfig.from_pretrained(ckpt_dir)
        if os.path.exists(ckpt_config_path)
        else AutoConfig.from_pretrained(LOCAL_BERT_PATH)
    )
    reg_config.vocab_size = ckpt_vocab_size

    base_bert = AutoModel.from_config(reg_config)
    if base_bert.get_input_embeddings().weight.shape[0] != ckpt_vocab_size:
        base_bert.resize_token_embeddings(ckpt_vocab_size)

    model = AEMTransformerPredictor(
        finetuned_bert_model=base_bert,
        n_cont_cond=n_cont_cond,
        n_cat_cond=n_cat_cond,
        n_targets=expected_n_targets,
        unfreeze_layers_reg=4,
    ).to(device)

    load_info = model.load_state_dict(state_dict, strict=False)
    allowed_miss = {k for k in load_info.missing_keys if k.endswith("position_ids")}
    real_missing = set(load_info.missing_keys) - allowed_miss
    if real_missing:
        raise RuntimeError(f"Missing keys: {sorted(real_missing)}")
    if load_info.unexpected_keys:
        raise RuntimeError(f"Unexpected keys: {sorted(load_info.unexpected_keys)}")
    model.eval()
    return model, head_count


# ---------- prediction ----------
def predict_dataframe(model, df_part, scaler_cont, ohe_cat, scaler_targets):
    ds = AEMRegressionDataset(
        df_part,
        tokenizer_reg,
        condition_cont_cols,
        condition_cat_cols,
        multi_task_targets,
        scaler_cont,
        ohe_cat,
        scaler_targets,
        max_length=GLOBAL_MAX_SMILES_LEN,
    )
    dl = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=AEMDataCollator())

    preds_scaled_all, labels_scaled_all, masks_all = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in dl:
            logits = model(
                hydro_input_ids=batch["hydro_input_ids"].to(device),
                hydro_attention_mask=batch["hydro_attention_mask"].to(device),
                phobic_input_ids=batch["phobic_input_ids"].to(device),
                phobic_attention_mask=batch["phobic_attention_mask"].to(device),
                conditions=batch["conditions"].to(device),
            )
            preds_scaled_all.append(logits.cpu().numpy())
            labels_scaled_all.append(batch["labels"].numpy())
            masks_all.append(batch["mask"].numpy())

    preds_scaled = np.concatenate(preds_scaled_all, axis=0)
    labels_scaled = np.concatenate(labels_scaled_all, axis=0)
    masks = np.concatenate(masks_all, axis=0)

    preds_orig = scaler_targets.inverse_transform(preds_scaled)
    labels_orig = scaler_targets.inverse_transform(labels_scaled)
    labels_orig[~masks] = np.nan
    return preds_orig, labels_orig, masks


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def to_plot_scale(target_name, arr):
    return np.expm1(arr) if target_name == "Conductivity" else arr


def compute_plot_metrics(y_true, y_pred):
    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
    }


# ---------- style / color ----------
TARGETS_TO_SHOW = ["Conductivity", "SR"]
UNITS = {"Conductivity": "mS/cm", "SR": "%"}
AXIS_LABELS = {"Conductivity": "Conductivity (mS/cm)", "SR": "SR (%)"}

# Top-journal palette - same teal/coral as t-SNE plots
TRAIN_COLOR = "#4DBFB3"  # teal
TEST_COLOR = "#E05A63"  # deep coral

for t in TARGETS_TO_SHOW:
    if t not in multi_task_targets:
        raise RuntimeError(f"{t} not in multi_task_targets: {multi_task_targets}")

fold_head_counts = []
fold_val_metrics = {t: {"r2": [], "mae": [], "rmse": []} for t in TARGETS_TO_SHOW}
fold_test_metrics_same_test = {
    t: (
        {"r2_log": [], "r2_raw": [], "mae": [], "rmse": []}
        if t == "Conductivity"
        else {"r2": [], "mae": [], "rmse": []}
    )
    for t in TARGETS_TO_SHOW
}
fold_test_rows = []
ensemble_train_preds = []
ensemble_test_preds = []
train_labels_ref = None
test_labels_ref = None

# main fold loop
for fold, (train_idx, val_idx) in enumerate(
    skf.split(train_val_df, train_val_df[stratify_col]), start=1
):
    print(f"\n===== RELOAD FOLD {fold}/{N_SPLITS} =====")

    train_df = train_val_df.iloc[train_idx].copy()
    val_df = train_val_df.iloc[val_idx].copy()

    cont_means = train_df[condition_cont_cols].mean()
    cat_modes = train_df[condition_cat_cols].mode().iloc[0]

    for col in condition_cont_cols:
        train_df[col].fillna(cont_means[col], inplace=True)
        val_df[col].fillna(cont_means[col], inplace=True)
    for col in condition_cat_cols:
        train_df[col] = train_df[col].astype(str).fillna(cat_modes[col])
        val_df[col] = val_df[col].astype(str).fillna(cat_modes[col])

    scaler_cont = StandardScaler().fit(train_df[condition_cont_cols])
    ohe_cat = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(
        train_df[condition_cat_cols]
    )

    train_targets_means = train_df[multi_task_targets].mean()
    train_targets_stds = train_df[multi_task_targets].std()
    train_targets_stds[train_targets_stds == 0] = 1.0

    scaler_targets = StandardScaler()
    scaler_targets.mean_ = train_targets_means.values
    scaler_targets.scale_ = train_targets_stds.values

    ckpt_dir = get_fold_checkpoint_dir(fold)
    print(f"Loading: {ckpt_dir}")
    model, head_count = build_model_from_checkpoint(
        ckpt_dir=ckpt_dir,
        n_cont_cond=len(scaler_cont.mean_),
        n_cat_cond=ohe_cat.get_feature_names_out().shape[0],
        expected_n_targets=len(multi_task_targets),
    )
    fold_head_counts.append(head_count)

    # fold-internal validation
    val_pred, val_true, val_mask = predict_dataframe(
        model, val_df.copy(), scaler_cont, ohe_cat, scaler_targets
    )
    for target_name in TARGETS_TO_SHOW:
        idx = multi_task_targets.index(target_name)
        y_t = val_true[:, idx]
        y_p = val_pred[:, idx]
        valid = ~np.isnan(y_t)
        if target_name == "Conductivity":
            r2_v = r2_score(y_t[valid], y_p[valid])
            y_t_r = np.expm1(y_t[valid])
            y_p_r = np.expm1(y_p[valid])
            mae_v = mean_absolute_error(y_t_r, y_p_r)
            rm_v = rmse(y_t_r, y_p_r)
        else:
            r2_v = r2_score(y_t[valid], y_p[valid])
            mae_v = mean_absolute_error(y_t[valid], y_p[valid])
            rm_v = rmse(y_t[valid], y_p[valid])
        fold_val_metrics[target_name]["r2"].append(r2_v)
        fold_val_metrics[target_name]["mae"].append(mae_v)
        fold_val_metrics[target_name]["rmse"].append(rm_v)

    # ensemble prediction
    train_eval_df = train_val_df.copy()
    test_eval_df = test_df.copy()
    for col in condition_cont_cols:
        train_eval_df[col].fillna(cont_means[col], inplace=True)
        test_eval_df[col].fillna(cont_means[col], inplace=True)
    for col in condition_cat_cols:
        train_eval_df[col] = train_eval_df[col].astype(str).fillna(cat_modes[col])
        test_eval_df[col] = test_eval_df[col].astype(str).fillna(cat_modes[col])

    train_pred, train_true, _ = predict_dataframe(
        model, train_eval_df, scaler_cont, ohe_cat, scaler_targets
    )
    test_pred, test_true, _ = predict_dataframe(
        model, test_eval_df, scaler_cont, ohe_cat, scaler_targets
    )

    ensemble_train_preds.append(train_pred)
    ensemble_test_preds.append(test_pred)
    if train_labels_ref is None:
        train_labels_ref = train_true.copy()
        test_labels_ref = test_true.copy()

    # Table A: fold on same test set
    row = {"Fold": f"Fold {fold}"}
    if "Conductivity" in TARGETS_TO_SHOW:
        idx = multi_task_targets.index("Conductivity")
        y_t_l = test_true[:, idx]
        y_p_l = test_pred[:, idx]
        valid = ~np.isnan(y_t_l)
        r2_l = r2_score(y_t_l[valid], y_p_l[valid])
        y_t_r = np.expm1(y_t_l[valid])
        y_p_r = np.expm1(y_p_l[valid])
        row.update(
            {
                "Cond_R2_log": r2_l,
                "Cond_R2_raw": r2_score(y_t_r, y_p_r),
                "Cond_MAE_mScm": mean_absolute_error(y_t_r, y_p_r),
                "Cond_RMSE_mScm": rmse(y_t_r, y_p_r),
            }
        )
        for k, v in [
            ("r2_log", r2_l),
            ("r2_raw", r2_score(y_t_r, y_p_r)),
            ("mae", mean_absolute_error(y_t_r, y_p_r)),
            ("rmse", rmse(y_t_r, y_p_r)),
        ]:
            fold_test_metrics_same_test["Conductivity"][k].append(v)
    if "SR" in TARGETS_TO_SHOW:
        idx = multi_task_targets.index("SR")
        y_t_s = test_true[:, idx]
        y_p_s = test_pred[:, idx]
        v_s = ~np.isnan(y_t_s)
        r2_s = r2_score(y_t_s[v_s], y_p_s[v_s])
        mae_s = mean_absolute_error(y_t_s[v_s], y_p_s[v_s])
        rm_s = rmse(y_t_s[v_s], y_p_s[v_s])
        row.update({"SR_R2": r2_s, "SR_MAE_pct": mae_s, "SR_RMSE_pct": rm_s})
        for k, v in [("r2", r2_s), ("mae", mae_s), ("rmse", rm_s)]:
            fold_test_metrics_same_test["SR"][k].append(v)
    fold_test_rows.append(row)

# ---------- summary ----------
print("\n" + "=" * 90)
print("Saved checkpoint output heads:", fold_head_counts)

print("\n" + "=" * 90)
print("5-fold cross-validation summary (internal validation folds)")
for target_name in TARGETS_TO_SHOW:
    r2_m = np.mean(fold_val_metrics[target_name]["r2"])
    r2_s = np.std(fold_val_metrics[target_name]["r2"], ddof=1)
    mae_m = np.mean(fold_val_metrics[target_name]["mae"])
    mae_s = np.std(fold_val_metrics[target_name]["mae"], ddof=1)
    rmse_m = np.mean(fold_val_metrics[target_name]["rmse"])
    rmse_s = np.std(fold_val_metrics[target_name]["rmse"], ddof=1)
    unit = UNITS[target_name]
    print(
        f"{target_name}: R²={r2_m:.3f}±{r2_s:.3f}  "
        f"MAE={mae_m:.2f}±{mae_s:.2f} {unit}  "
        f"RMSE={rmse_m:.2f}±{rmse_s:.2f} {unit}"
    )

fold_test_df = pd.DataFrame(fold_test_rows)
metric_cols_A = [c for c in fold_test_df.columns if c != "Fold"]
mean_row = {"Fold": "Mean±Std"}
for c in metric_cols_A:
    m, s = fold_test_df[c].mean(), fold_test_df[c].std(ddof=1)
    mean_row[c] = f"{m:.3f}±{s:.3f}"
disp = fold_test_df.copy()
for c in metric_cols_A:
    disp[c] = disp[c].map(lambda x: f"{x:.3f}")
disp = pd.concat([disp, pd.DataFrame([mean_row])], ignore_index=True)

print("\n" + "=" * 120)
print("Table A. Five fold-specific models on the SAME independent test set")
print(disp.to_string(index=False))
fold_test_df.to_csv(
    os.path.join(
        OUTPUT_DIR, "tableA_five_fold_models_on_same_independent_test_set.csv"
    ),
    index=False,
)

# ---------- ensemble ----------
ensemble_train_pred = np.mean(np.stack(ensemble_train_preds), axis=0)
ensemble_test_pred = np.mean(np.stack(ensemble_test_preds), axis=0)

ensemble_rows = []
if "Conductivity" in TARGETS_TO_SHOW:
    idx = multi_task_targets.index("Conductivity")
    y_t_l = test_labels_ref[:, idx]
    y_p_l = ensemble_test_pred[:, idx]
    valid = ~np.isnan(y_t_l)
    r2_l = r2_score(y_t_l[valid], y_p_l[valid])
    y_t_r = np.expm1(y_t_l[valid])
    y_p_r = np.expm1(y_p_l[valid])
    ensemble_rows.append(
        {
            "Target": "Conductivity",
            "R2_log": r2_l,
            "R2_raw": r2_score(y_t_r, y_p_r),
            "MAE_mScm": mean_absolute_error(y_t_r, y_p_r),
            "RMSE_mScm": rmse(y_t_r, y_p_r),
        }
    )
if "SR" in TARGETS_TO_SHOW:
    idx = multi_task_targets.index("SR")
    y_t_s = test_labels_ref[:, idx]
    y_p_s = ensemble_test_pred[:, idx]
    v_s = ~np.isnan(y_t_s)
    ensemble_rows.append(
        {
            "Target": "SR",
            "R2_log": np.nan,
            "R2_raw": r2_score(y_t_s[v_s], y_p_s[v_s]),
            "MAE_pct": mean_absolute_error(y_t_s[v_s], y_p_s[v_s]),
            "RMSE_pct": rmse(y_t_s[v_s], y_p_s[v_s]),
        }
    )

ensemble_df = pd.DataFrame(ensemble_rows)
ens_disp = ensemble_df.copy()
for c in ens_disp.columns:
    if c != "Target":
        ens_disp[c] = ens_disp[c].apply(lambda x: "" if pd.isna(x) else f"{x:.3f}")
print("\n" + "=" * 120)
print("Table B. Ensemble on independent test set")
print(ens_disp.to_string(index=False))
ensemble_df.to_csv(
    os.path.join(OUTPUT_DIR, "tableB_ensemble_on_same_independent_test_set.csv"),
    index=False,
)


# PLOTTING - manuscript style, Arial bold, thick spines


def plot_single_target(target_name, save_path_tiff):
    idx = multi_task_targets.index(target_name)

    y_train_true = train_labels_ref[:, idx]
    y_train_pred = ensemble_train_pred[:, idx]
    y_test_true = test_labels_ref[:, idx]
    y_test_pred = ensemble_test_pred[:, idx]

    train_valid = ~np.isnan(y_train_true)
    test_valid = ~np.isnan(y_test_true)

    y_tt = to_plot_scale(target_name, y_train_true[train_valid])
    y_tp = to_plot_scale(target_name, y_train_pred[train_valid])
    y_et = to_plot_scale(target_name, y_test_true[test_valid])
    y_ep = to_plot_scale(target_name, y_test_pred[test_valid])

    m_train = compute_plot_metrics(y_tt, y_tp)
    m_test = compute_plot_metrics(y_et, y_ep)

    # -- Figure ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.0, 6.5), facecolor="white")

    # -- Scatter -----------------------------------------------------------
    ax.scatter(
        y_tt,
        y_tp,
        s=28,
        c=TRAIN_COLOR,
        alpha=0.70,
        linewidths=0,
        zorder=2,
        label="Train",
    )
    ax.scatter(
        y_et, y_ep, s=38, c=TEST_COLOR, alpha=0.88, linewidths=0, zorder=3, label="Test"
    )

    # -- y = x diagonal ----------------------------------------------------
    all_v = np.concatenate([y_tt, y_tp, y_et, y_ep])
    lo = float(np.nanmin(all_v))
    hi = float(np.nanmax(all_v))
    pad = (hi - lo) * 0.05 if hi > lo else 1.0
    ax.plot(
        [lo - pad, hi + pad],
        [lo - pad, hi + pad],
        color="#1A1A1A",
        lw=2.0,
        ls="--",
        zorder=1,
    )
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)

    # -- Spines ------------------------------------------------------------
    for sp in ax.spines.values():
        sp.set_linewidth(2.2)
        sp.set_color("#1A1A1A")

    # -- Axis labels  (axis name, not "Value") -----------------------------
    ax_label = AXIS_LABELS[target_name]
    ax.set_xlabel(
        f"True {ax_label}", fontsize=22, fontweight="bold", color="#1A1A1A", labelpad=7
    )
    ax.set_ylabel(
        f"Predicted {ax_label}",
        fontsize=22,
        fontweight="bold",
        color="#1A1A1A",
        labelpad=7,
    )

    # -- Ticks -------------------------------------------------------------
    ax.tick_params(
        axis="both",
        which="both",
        direction="out",
        length=6,
        width=2.0,
        labelsize=18,
        colors="#1A1A1A",
        pad=4,
    )
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")

    # -- Metric table (top-left, larger, bold) -----------------------------
    cell_text = [
        [
            "Train",
            f'{m_train["r2"]:.3f}',
            f'{m_train["mae"]:.2f}',
            f'{m_train["rmse"]:.2f}',
        ],
        [
            "Test",
            f'{m_test["r2"]:.3f}',
            f'{m_test["mae"]:.2f}',
            f'{m_test["rmse"]:.2f}',
        ],
    ]
    table = ax.table(
        cellText=cell_text,
        colLabels=["", "R²", "MAE", "RMSE"],
        cellLoc="center",
        colLoc="center",
        bbox=[0.04, 0.78, 0.60, 0.21],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    for (row_i, col_i), cell in table.get_celld().items():
        cell.set_linewidth(1.1)
        cell.set_edgecolor("#888888")
        cell.set_facecolor("white")
        txt = cell.get_text()
        txt.set_fontweight("bold")
        txt.set_fontfamily("Arial")
        if row_i == 0:  # header row
            txt.set_fontsize(14)
        # Color-code the row label
        if col_i == 0 and row_i == 1:
            txt.set_color(TRAIN_COLOR)
        elif col_i == 0 and row_i == 2:
            txt.set_color(TEST_COLOR)

    # -- Legend ------------------------------------------------------------
    leg = ax.legend(
        loc="lower right",
        fontsize=16,
        frameon=True,
        framealpha=0.95,
        edgecolor="#cccccc",
        handletextpad=0.4,
        handlelength=1.2,
        borderpad=0.5,
    )
    leg.get_frame().set_linewidth(1.0)
    for txt in leg.get_texts():
        txt.set_fontweight("bold")
        txt.set_fontfamily("Arial")
    for h in leg.legend_handles:
        h.set_sizes([50])

    ax.grid(False)
    plt.tight_layout(pad=0.8)
    plt.savefig(
        save_path_tiff, dpi=1200, format="tiff", bbox_inches="tight", facecolor="white"
    )
    plt.show()
    plt.close(fig)
    print(f"Saved: {save_path_tiff}")


# ---------- draw ----------
if "Conductivity" in TARGETS_TO_SHOW:
    plot_single_target(
        "Conductivity",
        os.path.join(OUTPUT_DIR, "Conductivity_ensemble_scatter.tiff"),
    )

if "SR" in TARGETS_TO_SHOW:
    plot_single_target(
        "SR",
        os.path.join(OUTPUT_DIR, "SR_ensemble_scatter.tiff"),
    )
