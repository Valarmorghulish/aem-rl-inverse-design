"""
polyBERT-based multitask predictor for hydroxide conductivity and SR.

The architecture follows Section 2.3.1 of the accompanying paper:

  - polyBERT encoder shared between the hydrophilic and hydrophobic motifs:
        Kuenneth, C., & Ramprasad, R. (2023). polyBERT: a chemical
        language model to enable fully machine-driven ultrafast polymer
        informatics. *Nature Communications*, 14, 4099.
  - Two-step training: (i) self-supervised MLM on an AEM-related SMILES
    corpus to specialise the encoder, then (ii) supervised fine-tuning of
    the upper encoder layers together with a shared regression head and
    task-specific output layers for hydroxide conductivity (sigma) and
    swelling ratio (SR).
  - Masked multitask regression so that samples with only one of the two
    labels can still contribute to training.

This module deliberately keeps only the two targets used in the paper
(hydroxide conductivity and SR). Auxiliary targets such as water uptake or
tensile properties have been removed.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


# Targets fixed to (sigma, SR) per the paper.
TASK_NAMES: Tuple[str, str] = ("Conductivity", "SR")


# ---------------------------------------------------------------------------
# Configuration container
# ---------------------------------------------------------------------------
@dataclass
class PredictorConfig:
    """Top-level configuration for predictor training and inference."""

    aem_data_path: str
    polybert_path: str
    output_dir: str = "./checkpoints/predictor"

    # Column groups in the AEM CSV file
    smiles_hydrophilic_col: str = "Hydrophilic"
    smiles_hydrophobic_col: str = "Hydrophobic"
    cont_cols: Sequence[str] = field(
        default_factory=lambda: [
            "HydrophilicFrac",
            "IEC",
            "Temperature",
            "RH",
        ]
    )
    cat_cols: Sequence[str] = field(
        default_factory=lambda: ["PolymerArchitecture"]
    )
    target_cols: Sequence[str] = field(default_factory=lambda: list(TASK_NAMES))

    # Tokenisation
    max_smiles_len: int = 256

    # MLM stage
    mlm_epochs: int = 60
    mlm_batch_size: int = 8
    mlm_grad_accum: int = 4
    mlm_lr: float = 5e-5
    mlm_unfreeze_layers: int = 2
    mlm_augment_factor: int = 8

    # Regression stage
    reg_epochs: int = 80
    reg_batch_size: int = 8
    reg_eval_batch_size: int = 16
    reg_grad_accum: int = 2
    reg_lr: float = 2e-5
    reg_unfreeze_layers: int = 4
    reg_dropout: float = 0.1
    reg_warmup_ratio: float = 0.06
    reg_weight_decay: float = 0.01
    reg_early_stop_patience: int = 6

    # Cross-validation
    test_size: float = 0.15
    n_folds: int = 5
    random_state: int = 42


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
class AEMSmilesAugmentedDataset(Dataset):
    """SMILES-only dataset for the self-supervised MLM step.

    Each unique hydrophilic / hydrophobic SMILES is repeated ``augment_factor``
    times; one copy is the canonical SMILES, the rest are RDKit
    ``doRandom=True`` re-encodings of the same molecule.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        smiles_cols: Sequence[str],
        max_length: int = 128,
        augment_factor: int = 8,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        unique_smiles: List[str] = []
        for col in smiles_cols:
            unique_smiles.extend(dataframe[col].dropna().unique().tolist())
        unique_smiles = list(set(unique_smiles))

        self.smiles: List[str] = []
        for smi in tqdm(unique_smiles, desc="Augmenting SMILES"):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            self.smiles.append(smi)
            for _ in range(max(augment_factor - 1, 0)):
                self.smiles.append(Chem.MolToSmiles(mol, doRandom=True))

        np.random.shuffle(self.smiles)

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.smiles[idx],
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {k: v.squeeze() for k, v in enc.items()}


class AEMRegressionDataset(Dataset):
    """Joint hydrophilic/hydrophobic dataset for the supervised fine-tuning step."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        cont_cols: Sequence[str],
        cat_cols: Sequence[str],
        target_cols: Sequence[str],
        scaler_cont: StandardScaler,
        ohe_cat: OneHotEncoder,
        scaler_targets: StandardScaler,
        max_length: int = 256,
        smiles_hydrophilic_col: str = "Hydrophilic",
        smiles_hydrophobic_col: str = "Hydrophobic",
    ):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cont_cols = list(cont_cols)
        self.cat_cols = list(cat_cols)
        self.target_cols = list(target_cols)
        self.scaler_cont = scaler_cont
        self.ohe_cat = ohe_cat
        self.scaler_targets = scaler_targets
        self.smiles_hydrophilic_col = smiles_hydrophilic_col
        self.smiles_hydrophobic_col = smiles_hydrophobic_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[[idx]]
        hydro = self._safe_str(row[self.smiles_hydrophilic_col].iloc[0])
        phobic = self._safe_str(row[self.smiles_hydrophobic_col].iloc[0])

        h_tok = self.tokenizer(
            hydro,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        p_tok = self.tokenizer(
            phobic,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        cont_vec = self.scaler_cont.transform(row[self.cont_cols]).flatten()
        cat_vec = self.ohe_cat.transform(row[self.cat_cols].astype(str)).flatten()
        cond_vec = np.concatenate([cont_vec, cat_vec])

        targets = row[self.target_cols].values.astype(np.float32).flatten()
        mask = ~np.isnan(targets)
        targets_filled = np.nan_to_num(targets, nan=0.0)
        targets_scaled = self.scaler_targets.transform(
            targets_filled.reshape(1, -1)
        ).flatten()
        targets_scaled[~mask] = 0.0

        return {
            "hydro_input_ids": torch.tensor(
                h_tok["input_ids"].squeeze(), dtype=torch.long
            ),
            "hydro_attention_mask": torch.tensor(
                h_tok["attention_mask"].squeeze(), dtype=torch.long
            ),
            "phobic_input_ids": torch.tensor(
                p_tok["input_ids"].squeeze(), dtype=torch.long
            ),
            "phobic_attention_mask": torch.tensor(
                p_tok["attention_mask"].squeeze(), dtype=torch.long
            ),
            "conditions": torch.tensor(cond_vec, dtype=torch.float),
            "labels": torch.tensor(targets_scaled, dtype=torch.float),
            "mask": torch.tensor(mask, dtype=torch.bool),
        }

    @staticmethod
    def _safe_str(value) -> str:
        return str(value) if pd.notna(value) else ""


class AEMDataCollator:
    def __call__(self, features):
        return {
            key: torch.stack([f[key] for f in features])
            for key in features[0].keys()
        }


# ---------------------------------------------------------------------------
# MLM model wrapper and trainer
# ---------------------------------------------------------------------------
class PolyBertForMLM(nn.Module):
    """Lightweight MLM head on top of a polyBERT encoder."""

    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model
        self.mlm_head = nn.Linear(
            bert_model.config.hidden_size, bert_model.config.vocab_size
        )

    def forward(self, **kwargs):
        bert_inputs = {
            k: v
            for k, v in kwargs.items()
            if k in ("input_ids", "attention_mask", "token_type_ids")
        }
        return self.mlm_head(self.bert(**bert_inputs).last_hidden_state)


class _MlmTrainer(Trainer):
    """Trainer subclass implementing the cross-entropy MLM loss."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        logits = model(**inputs)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            logits.view(-1, self.model.bert.config.vocab_size),
            labels.view(-1),
        )
        return (loss, {"logits": logits}) if return_outputs else loss


# ---------------------------------------------------------------------------
# Predictor module (regression)
# ---------------------------------------------------------------------------
class AEMTransformerPredictor(nn.Module):
    """Two-tower polyBERT encoder + shared regression head + per-task heads."""

    def __init__(
        self,
        bert_model,
        n_cont_cond: int,
        n_cat_cond: int,
        n_targets: int = len(TASK_NAMES),
        dropout: float = 0.1,
        unfreeze_layers_reg: int = 4,
    ):
        super().__init__()
        self.bert_model = bert_model
        total_layers = self.bert_model.config.num_hidden_layers
        unfreeze_start = total_layers - unfreeze_layers_reg

        for p in self.bert_model.parameters():
            p.requires_grad_(False)
        for i, layer in enumerate(self.bert_model.encoder.layer):
            if i >= unfreeze_start:
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

    def _encode(self, input_ids, attention_mask):
        outputs = self.bert_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        return outputs.last_hidden_state[:, 0, :]

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
        h = self._encode(hydro_input_ids, hydro_attention_mask)
        p = self._encode(phobic_input_ids, phobic_attention_mask)
        emb = torch.cat([h, p, conditions], dim=1)
        shared = self.shared_head(emb)
        outs = [head(shared) for head in self.task_heads]
        logits = torch.cat(outs, dim=1)

        if labels is None:
            return logits

        loss_fct = nn.MSELoss(reduction="none")
        se = loss_fct(logits, labels)
        masked_se = torch.where(mask, se, torch.zeros_like(se))
        valid = mask.sum()
        loss = (
            masked_se.sum() / valid
            if valid > 0
            else torch.tensor(0.0, device=logits.device)
        )
        return loss, logits


# ---------------------------------------------------------------------------
# Top-level trainer
# ---------------------------------------------------------------------------
class MultitaskPredictorTrainer:
    """End-to-end trainer for the two-step polyBERT-based multitask predictor."""

    def __init__(self, config: PredictorConfig):
        self.cfg = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        os.makedirs(self.cfg.output_dir, exist_ok=True)

    # ---------- data loading & cleaning -------------------------------
    def _load_dataframe(self) -> pd.DataFrame:
        df = pd.read_csv(self.cfg.aem_data_path)
        df[self.cfg.smiles_hydrophilic_col] = (
            df[self.cfg.smiles_hydrophilic_col].astype(str).replace("nan", np.nan)
        )
        df[self.cfg.smiles_hydrophobic_col] = (
            df[self.cfg.smiles_hydrophobic_col].astype(str).replace("nan", np.nan)
        )
        for col in list(self.cfg.cont_cols) + list(self.cfg.target_cols):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=list(self.cfg.target_cols), how="all", inplace=True)
        df.reset_index(drop=True, inplace=True)
        # log1p transform on hydroxide conductivity (skewed, positive)
        if "Conductivity" in df.columns:
            df["Conductivity"] = np.log1p(df["Conductivity"])
        return df

    # ---------- step 1: self-supervised MLM ---------------------------
    def run_mlm_step(self, df: pd.DataFrame):
        cfg = self.cfg
        tokenizer = AutoTokenizer.from_pretrained(cfg.polybert_path)
        config = AutoConfig.from_pretrained(cfg.polybert_path)
        config.vocab_size = len(tokenizer)
        bert = AutoModel.from_pretrained(
            cfg.polybert_path, config=config, ignore_mismatched_sizes=True
        )

        for p in bert.parameters():
            p.requires_grad = False
        n_layers = bert.config.num_hidden_layers
        unfreeze_from = n_layers - cfg.mlm_unfreeze_layers
        for i, layer in enumerate(bert.encoder.layer):
            if i >= unfreeze_from:
                for p in layer.parameters():
                    p.requires_grad = True
        for p in bert.embeddings.parameters():
            p.requires_grad = True

        model = PolyBertForMLM(bert).to(self.device)
        dset = AEMSmilesAugmentedDataset(
            df,
            tokenizer,
            smiles_cols=(
                cfg.smiles_hydrophilic_col,
                cfg.smiles_hydrophobic_col,
            ),
            max_length=128,
            augment_factor=cfg.mlm_augment_factor,
        )
        collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=0.15
        )
        out_dir = os.path.join(cfg.output_dir, "mlm_finetune")
        args = TrainingArguments(
            output_dir=out_dir,
            overwrite_output_dir=True,
            num_train_epochs=cfg.mlm_epochs,
            per_device_train_batch_size=cfg.mlm_batch_size,
            gradient_accumulation_steps=cfg.mlm_grad_accum,
            learning_rate=cfg.mlm_lr,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            logging_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            remove_unused_columns=False,
            report_to="none",
            seed=cfg.random_state,
        )
        trainer = _MlmTrainer(
            model=model,
            args=args,
            data_collator=collator,
            train_dataset=dset,
        )
        trainer.train()
        return trainer.model.bert, tokenizer

    # ---------- step 2: supervised fine-tuning (5-fold CV) ------------
    def run_regression_step(self, df: pd.DataFrame, finetuned_bert, tokenizer):
        cfg = self.cfg
        train_val_df, test_df = train_test_split(
            df, test_size=cfg.test_size, random_state=cfg.random_state
        )
        test_df = test_df.reset_index(drop=True).copy()

        # Stratify on whether the first target is observed
        stratify_col = f"{cfg.target_cols[0]}_present"
        train_val_df = train_val_df.copy()
        train_val_df[stratify_col] = (
            train_val_df[cfg.target_cols[0]].notna().astype(int)
        )

        skf = StratifiedKFold(
            n_splits=cfg.n_folds, shuffle=True, random_state=cfg.random_state
        )

        fold_models: List[AEMTransformerPredictor] = []
        fold_scalers: List[Dict] = []
        fold_histories: List[pd.DataFrame] = []

        for fold, (tr_idx, va_idx) in enumerate(
            skf.split(train_val_df, train_val_df[stratify_col])
        ):
            print(f"===== FOLD {fold + 1}/{cfg.n_folds} =====")
            train_df = train_val_df.iloc[tr_idx].copy()
            val_df = train_val_df.iloc[va_idx].copy()
            scalers = self._fit_fold_scalers(train_df)
            self._fill_with_train_stats(train_df, scalers)
            self._fill_with_train_stats(val_df, scalers)

            train_dset = AEMRegressionDataset(
                train_df, tokenizer, cfg.cont_cols, cfg.cat_cols,
                cfg.target_cols, scalers["cont"], scalers["cat"],
                scalers["targets"], max_length=cfg.max_smiles_len,
                smiles_hydrophilic_col=cfg.smiles_hydrophilic_col,
                smiles_hydrophobic_col=cfg.smiles_hydrophobic_col,
            )
            val_dset = AEMRegressionDataset(
                val_df, tokenizer, cfg.cont_cols, cfg.cat_cols,
                cfg.target_cols, scalers["cont"], scalers["cat"],
                scalers["targets"], max_length=cfg.max_smiles_len,
                smiles_hydrophilic_col=cfg.smiles_hydrophilic_col,
                smiles_hydrophobic_col=cfg.smiles_hydrophobic_col,
            )

            n_cat = scalers["cat"].get_feature_names_out().shape[0]
            bert_clone = AutoModel.from_config(finetuned_bert.config)
            bert_clone.load_state_dict(finetuned_bert.state_dict())
            model = AEMTransformerPredictor(
                bert_clone,
                n_cont_cond=len(scalers["cont"].mean_),
                n_cat_cond=n_cat,
                n_targets=len(cfg.target_cols),
                dropout=cfg.reg_dropout,
                unfreeze_layers_reg=cfg.reg_unfreeze_layers,
            ).to(self.device)

            fold_dir = os.path.join(cfg.output_dir, f"reg_fold_{fold + 1}")
            args = TrainingArguments(
                output_dir=fold_dir,
                overwrite_output_dir=True,
                num_train_epochs=cfg.reg_epochs,
                per_device_train_batch_size=cfg.reg_batch_size,
                per_device_eval_batch_size=cfg.reg_eval_batch_size,
                gradient_accumulation_steps=cfg.reg_grad_accum,
                warmup_ratio=cfg.reg_warmup_ratio,
                weight_decay=cfg.reg_weight_decay,
                learning_rate=cfg.reg_lr,
                lr_scheduler_type="linear",
                logging_strategy="epoch",
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                fp16=torch.cuda.is_available(),
                report_to="none",
                seed=cfg.random_state,
                save_total_limit=1,
            )
            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=train_dset,
                eval_dataset=val_dset,
                data_collator=AEMDataCollator(),
                callbacks=[
                    EarlyStoppingCallback(
                        early_stopping_patience=cfg.reg_early_stop_patience
                    )
                ],
            )
            trainer.train()
            fold_histories.append(pd.DataFrame(trainer.state.log_history))
            fold_models.append(trainer.model)
            fold_scalers.append(scalers)

            # Per-fold validation R2
            preds_scaled = trainer.predict(val_dset).predictions
            labels_scaled = np.array(
                [item["labels"].numpy() for item in val_dset]
            )
            mask = np.array([item["mask"].numpy() for item in val_dset])
            preds_orig = scalers["targets"].inverse_transform(preds_scaled)
            labels_orig = scalers["targets"].inverse_transform(labels_scaled)
            labels_orig[~mask] = np.nan
            for i, name in enumerate(cfg.target_cols):
                valid = ~np.isnan(labels_orig[:, i])
                if valid.sum() > 1:
                    r2 = r2_score(labels_orig[valid, i], preds_orig[valid, i])
                    print(f"  Fold {fold + 1} {name} val R2 = {r2:.4f}")

        self._save_ensemble(fold_models, fold_scalers)
        return fold_models, fold_scalers, test_df, fold_histories

    # ---------- helpers ----------------------------------------------
    def _fit_fold_scalers(self, train_df: pd.DataFrame) -> Dict:
        cfg = self.cfg
        cont_means = train_df[list(cfg.cont_cols)].mean()
        cat_modes = train_df[list(cfg.cat_cols)].mode().iloc[0]
        train_df = train_df.copy()
        for c in cfg.cont_cols:
            train_df[c] = train_df[c].fillna(cont_means[c])
        for c in cfg.cat_cols:
            train_df[c] = train_df[c].astype(str).fillna(cat_modes[c])

        scaler_cont = StandardScaler().fit(train_df[list(cfg.cont_cols)])
        ohe_cat = OneHotEncoder(
            sparse_output=False, handle_unknown="ignore"
        ).fit(train_df[list(cfg.cat_cols)])
        means = train_df[list(cfg.target_cols)].mean()
        stds = train_df[list(cfg.target_cols)].std().replace(0, 1.0)
        scaler_targets = StandardScaler()
        scaler_targets.mean_ = means.values
        scaler_targets.scale_ = stds.values
        return {
            "cont": scaler_cont,
            "cat": ohe_cat,
            "targets": scaler_targets,
            "cont_means": cont_means,
            "cat_modes": cat_modes,
        }

    def _fill_with_train_stats(self, df: pd.DataFrame, scalers: Dict) -> None:
        for c in self.cfg.cont_cols:
            df[c] = df[c].fillna(scalers["cont_means"][c])
        for c in self.cfg.cat_cols:
            df[c] = df[c].astype(str).fillna(scalers["cat_modes"][c])

    def _save_ensemble(
        self,
        fold_models: List[AEMTransformerPredictor],
        fold_scalers: List[Dict],
    ) -> None:
        cfg = self.cfg
        ensemble_dir = os.path.join(cfg.output_dir, "ensemble")
        shutil.rmtree(ensemble_dir, ignore_errors=True)
        os.makedirs(ensemble_dir, exist_ok=True)
        for i, (model, scalers) in enumerate(zip(fold_models, fold_scalers)):
            d = os.path.join(ensemble_dir, f"fold_{i + 1}")
            os.makedirs(d, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(d, "pytorch_model.bin"))
            model.bert_model.config.save_pretrained(d)
            joblib.dump(scalers["cont"], os.path.join(d, "scaler_cont.pkl"))
            joblib.dump(scalers["cat"], os.path.join(d, "ohe_cat.pkl"))
            joblib.dump(scalers["targets"], os.path.join(d, "scaler_targets.pkl"))
        metadata = {
            "condition_cont_cols": list(cfg.cont_cols),
            "condition_cat_cols": list(cfg.cat_cols),
            "multi_task_targets": list(cfg.target_cols),
            "unfreeze_layers_reg": cfg.reg_unfreeze_layers,
            "n_folds": cfg.n_folds,
            "max_smiles_len": cfg.max_smiles_len,
            "polybert_path": cfg.polybert_path,
        }
        with open(os.path.join(ensemble_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    def fit(self):
        df = self._load_dataframe()
        finetuned_bert, tokenizer = self.run_mlm_step(df)
        return self.run_regression_step(df, finetuned_bert, tokenizer)


# ---------------------------------------------------------------------------
# Inference wrapper used by the RL reward function
# ---------------------------------------------------------------------------
class EnsembleTransformerPredictorForRL:
    """Load a saved K-fold ensemble and expose a single ``predict_one`` API."""

    def __init__(
        self,
        ensemble_base_dir: str,
        device: torch.device,
        polybert_path: Optional[str] = None,
    ):
        self.device = device
        with open(os.path.join(ensemble_base_dir, "metadata.json"), "r") as f:
            self.metadata = json.load(f)
        if polybert_path is None:
            polybert_path = self.metadata.get("polybert_path")
        if polybert_path is None or not os.path.isdir(polybert_path):
            raise FileNotFoundError(
                "polybert_path is required and must point to a local "
                "polyBERT checkpoint directory."
            )
        self.tokenizer = AutoTokenizer.from_pretrained(polybert_path)

        self.models: List[AEMTransformerPredictor] = []
        self.scalers: List[Dict] = []
        for i in range(1, self.metadata["n_folds"] + 1):
            d = os.path.join(ensemble_base_dir, f"fold_{i}")
            scaler_cont = joblib.load(os.path.join(d, "scaler_cont.pkl"))
            ohe_cat = joblib.load(os.path.join(d, "ohe_cat.pkl"))
            scaler_targets = joblib.load(os.path.join(d, "scaler_targets.pkl"))
            config = AutoConfig.from_pretrained(d)
            bert = AutoModel.from_config(config)
            n_cat = ohe_cat.get_feature_names_out().shape[0]
            model = AEMTransformerPredictor(
                bert,
                n_cont_cond=len(scaler_cont.mean_),
                n_cat_cond=n_cat,
                n_targets=len(self.metadata["multi_task_targets"]),
                unfreeze_layers_reg=self.metadata["unfreeze_layers_reg"],
            ).to(device)
            model.load_state_dict(
                torch.load(
                    os.path.join(d, "pytorch_model.bin"),
                    map_location=device,
                )
            )
            model.eval()
            self.models.append(model)
            self.scalers.append(
                {"cont": scaler_cont, "cat": ohe_cat, "targets": scaler_targets}
            )

    def predict_one(
        self,
        hydro_smi: str,
        phobic_smi: str,
        conditions: Dict[str, float],
    ) -> Dict[str, float]:
        """Run the K-fold ensemble on a single AEM and return averaged targets.

        ``conditions`` should map column names (continuous + categorical) to
        their values; missing keys are imputed with 0.0 (continuous) or
        ``"Unknown"`` (categorical).
        """
        max_len = int(self.metadata.get("max_smiles_len", 256))
        all_preds = []
        with torch.no_grad():
            for model, scalers in zip(self.models, self.scalers):
                cont_vals = [
                    conditions.get(c, 0.0)
                    for c in self.metadata["condition_cont_cols"]
                ]
                cont_scaled = scalers["cont"].transform(
                    np.asarray(cont_vals).reshape(1, -1)
                )
                cat_vals = [
                    str(conditions.get(c, "Unknown"))
                    for c in self.metadata["condition_cat_cols"]
                ]
                cat_ohe = scalers["cat"].transform(
                    np.asarray(cat_vals).reshape(1, -1)
                )
                cond_vec = torch.tensor(
                    np.concatenate([cont_scaled, cat_ohe], axis=1),
                    dtype=torch.float,
                ).to(self.device)
                h_tok = self.tokenizer(
                    hydro_smi,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_len,
                    padding="max_length",
                ).to(self.device)
                p_tok = self.tokenizer(
                    phobic_smi,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_len,
                    padding="max_length",
                ).to(self.device)
                pred_scaled = model(
                    hydro_input_ids=h_tok["input_ids"],
                    hydro_attention_mask=h_tok["attention_mask"],
                    phobic_input_ids=p_tok["input_ids"],
                    phobic_attention_mask=p_tok["attention_mask"],
                    conditions=cond_vec,
                )
                pred_orig = scalers["targets"].inverse_transform(
                    pred_scaled.cpu().numpy()
                )
                # Reverse the log1p transform applied during training
                if "Conductivity" in self.metadata["multi_task_targets"]:
                    idx = self.metadata["multi_task_targets"].index(
                        "Conductivity"
                    )
                    pred_orig[0, idx] = np.expm1(pred_orig[0, idx])
                all_preds.append(pred_orig[0])
        avg = np.mean(all_preds, axis=0)
        return {
            name: float(val)
            for name, val in zip(
                self.metadata["multi_task_targets"], avg.tolist()
            )
        }
