"""
Alkaline-stability binary classifier (Pass / Fail) on paired aging records.

Implementation outline (Section 2.3.2 of the paper):
    - Pair each fresh (time = 0) record with its degraded (time > 0)
      counterpart sharing identical polymer identity and operational
      conditions; compute conductivity retention from the median of the
      fresh measurements over the minimum of the degraded measurements.
    - Label pairs with retention >= 0.80 as Pass and the rest as Fail.
    - Concatenate operational variables and 2D Mordred descriptors of the
      hydrophilic and hydrophobic motifs as input features.
    - Train a CatBoost classifier under a polymer-grouped split to avoid
      train/test leakage of the same polymer identity.
    - Optional Bayesian hyper-parameter tuning around a fixed-parameter
      baseline using Optuna with the TPE sampler.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from rdkit import Chem
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class StabilityClassifierConfig:
    data_path: str
    work_dir: str = "./checkpoints/stability"

    # Column names in the paired-aging CSV
    smi_hydrophilic_col: str = "Hydrophilic"
    smi_hydrophobic_col: str = "Hydrophobic"
    cond_col: str = "Cond"
    time_col: str = "time(h)"

    experiment_features: Sequence[str] = field(
        default_factory=lambda: [
            "Hydrophilic_Fraction",
            "solvent_NaOH (M)",
            "solvent_KOH (M)",
            "RH (%)",
            "theor_IEC (meq/g)",
            "stability_test_temp (C)",
            "prop_test_temp (C)",
            "time(h)",
        ]
    )

    # Pass/Fail label
    retention_threshold: float = 0.80

    # Mordred descriptor filtering
    desc_missing_rate_max: float = 0.10
    desc_variance_threshold: float = 0.05
    desc_corr_threshold: float = 0.85
    desc_corr_max_features: int = 600
    desc_max_features: int = 20

    # Train/test split
    test_size: float = 0.15
    cv_folds: int = 5
    random_state: int = 42

    # CatBoost baseline
    cb_max_iters: int = 5000
    early_stopping_rounds: int = 80
    cb_fallback_iters: int = 1000
    threshold: float = 0.5
    cb_fixed_params: Dict = field(
        default_factory=lambda: dict(
            depth=4,
            learning_rate=0.05,
            l2_leaf_reg=5.0,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        )
    )

    # Optuna tuning toggle
    do_bayesian_tuning: bool = True
    n_trials: int = 40
    pruner_startup_trials: int = 8
    pruner_warmup_folds: int = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def standardize_polymer_smiles(smi: str):
    """Cap polymer connection sites with carbons so RDKit accepts the SMILES."""
    if not isinstance(smi, str) or not smi.strip() or smi.lower() == "nan":
        return None, None
    capped = (
        smi.strip()
        .replace("[*]", "C")
        .replace("*", "C")
        .replace(" C", "C")
        .replace("C ", "C")
    )
    try:
        mol = Chem.MolFromSmiles(capped, sanitize=False)
        if mol is None:
            return None, None
        Chem.SanitizeMol(mol)
        return mol, Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None, None


def calculate_block_descriptors(mol, prefix: str) -> pd.Series:
    """Compute Mordred 2D descriptors for one motif and prefix the column names."""
    if mol is None:
        return pd.Series(dtype="float64")
    from mordred import Calculator, descriptors as mordred_descriptors

    calc = Calculator(mordred_descriptors, ignore_3D=True)
    try:
        s = pd.Series(calc(mol).asdict())
        s = pd.to_numeric(s, errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        return s.add_prefix(f"{prefix}_")
    except Exception:
        return pd.Series(dtype="float64")


def mode_fill_values(df: pd.DataFrame, cols: Sequence[str]) -> Dict[str, float]:
    fill: Dict[str, float] = {}
    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce")
        m = x.mode(dropna=True)
        if len(m) > 0:
            fill[c] = float(m.iloc[0])
        else:
            med = x.median()
            fill[c] = float(med) if pd.notna(med) else 0.0
    return fill


def apply_mode_fill(df: pd.DataFrame, fill: Dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    for c, v in fill.items():
        out[c] = pd.to_numeric(out[c], errors="coerce")
        out[c] = out[c].replace([np.inf, -np.inf], np.nan).fillna(v)
    return out


def _corr_filter_columns(df_train: pd.DataFrame, threshold: float) -> List[str]:
    if df_train.shape[1] <= 1:
        return df_train.columns.tolist()
    corr = df_train.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop = [c for c in upper.columns if (upper[c] > threshold).any()]
    return [c for c in df_train.columns if c not in drop]


def fit_descriptor_filter(X_train: pd.DataFrame, cfg: StabilityClassifierConfig) -> Dict:
    """Run the missing-rate / variance / correlation cascade on descriptors."""
    X = X_train.replace([np.inf, -np.inf], np.nan).copy()
    cols_non_all_nan = X.columns[X.notna().any()].tolist()
    X = X[cols_non_all_nan]

    miss = X.isna().mean(axis=0)
    cols_miss_ok = miss[miss <= cfg.desc_missing_rate_max].index.tolist()
    X = X[cols_miss_ok]

    median = X.median(numeric_only=True)
    X = X.fillna(median)
    if X.shape[1] == 0:
        raise RuntimeError(
            "Descriptor matrix empty after missing-rate filter; "
            "try a larger desc_missing_rate_max."
        )

    vt = VarianceThreshold(threshold=cfg.desc_variance_threshold).fit(X)
    cols_vt = X.columns[vt.get_support()].tolist()
    X = X[cols_vt]
    if X.shape[1] == 0:
        raise RuntimeError(
            "Descriptor matrix empty after variance filter; "
            "try a smaller desc_variance_threshold."
        )

    if X.shape[1] > cfg.desc_corr_max_features:
        top = (
            X.var(axis=0)
            .sort_values(ascending=False)
            .index[: cfg.desc_corr_max_features]
            .tolist()
        )
        X = X[top]

    cols_corr_keep = _corr_filter_columns(X, cfg.desc_corr_threshold)
    X = X[cols_corr_keep]

    if X.shape[1] > cfg.desc_max_features:
        cols_final = (
            X.var(axis=0)
            .sort_values(ascending=False)
            .index[: cfg.desc_max_features]
            .tolist()
        )
    else:
        cols_final = X.columns.tolist()

    return {
        "cols_non_all_nan": cols_non_all_nan,
        "cols_miss_ok": cols_miss_ok,
        "median": median,
        "cols_vt": cols_vt,
        "cols_corr_keep": cols_corr_keep,
        "cols_final": cols_final,
    }


def transform_with_descriptor_filter(X: pd.DataFrame, filt: Dict) -> pd.DataFrame:
    X = X.replace([np.inf, -np.inf], np.nan).copy()

    def _align(cols):
        for c in cols:
            if c not in X.columns:
                X[c] = np.nan

    _align(filt["cols_non_all_nan"])
    X = X[filt["cols_non_all_nan"]]
    _align(filt["cols_miss_ok"])
    X = X[filt["cols_miss_ok"]].fillna(filt["median"])
    _align(filt["cols_vt"])
    X = X[filt["cols_vt"]].fillna(0.0)
    _align(filt["cols_corr_keep"])
    X = X[filt["cols_corr_keep"]].fillna(0.0)
    _align(filt["cols_final"])
    X = X[filt["cols_final"]].fillna(0.0)
    return X


# ---------------------------------------------------------------------------
# Pairing pipeline
# ---------------------------------------------------------------------------
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


def pair_aging_records(df: pd.DataFrame, cfg: StabilityClassifierConfig) -> pd.DataFrame:
    """Pair fresh (time=0) and degraded (time>0) measurements.

    Within a pair, sigma_init is the median of the fresh measurements and
    sigma_deg is the minimum of the degraded measurements, following Section
    2.3.2 of the paper.
    """
    df = df.copy()
    df[cfg.time_col] = pd.to_numeric(df[cfg.time_col], errors="coerce")
    df[cfg.cond_col] = pd.to_numeric(df[cfg.cond_col], errors="coerce")
    df = df.dropna(subset=[cfg.time_col, cfg.cond_col])
    df = df[df[cfg.cond_col] > 0].copy()
    df["polymer_unique_id"] = (
        df[cfg.smi_hydrophilic_col].astype(str)
        + "|"
        + df[cfg.smi_hydrophobic_col].astype(str)
    )

    def _smiles_ok(row):
        m1, _ = standardize_polymer_smiles(row[cfg.smi_hydrophilic_col])
        if m1 is None:
            return False
        m2, _ = standardize_polymer_smiles(row[cfg.smi_hydrophobic_col])
        return m2 is not None

    df = df[df.apply(_smiles_ok, axis=1)].reset_index(drop=True)

    # Round columns before grouping so floating-point noise does not break pairing
    for c, nd in ROUND_MAP.items():
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").round(nd)

    fresh = df[df[cfg.time_col] == 0]
    deg = df[df[cfg.time_col] > 0]
    pair_keys = ["polymer_unique_id"] + [
        c for c in cfg.experiment_features if c != cfg.time_col
    ]

    fresh_agg = fresh.groupby(pair_keys, as_index=False).agg(
        Cond_init=(cfg.cond_col, "median")
    )

    deg_time_n = deg.groupby(pair_keys)[cfg.time_col].nunique().reset_index(
        name="n_time_deg"
    )
    bad = deg_time_n[deg_time_n["n_time_deg"] > 1]
    if len(bad) > 0:
        raise ValueError(
            "Detected pair_keys with multiple distinct degraded times. "
            "These violate the strict pairing rule of Section 2.3.2."
        )

    deg_agg = deg.groupby(pair_keys, as_index=False).agg(
        Cond_deg=(cfg.cond_col, "min"),
        time_deg=(cfg.time_col, "median"),
    )

    paired = fresh_agg.merge(deg_agg, on=pair_keys, how="inner")
    rep = df.drop_duplicates(subset=pair_keys)[
        pair_keys + [cfg.smi_hydrophilic_col, cfg.smi_hydrophobic_col]
    ]
    paired = paired.merge(rep, on=pair_keys, how="left")
    paired["retention"] = paired["Cond_deg"] / paired["Cond_init"]
    paired = paired.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["retention"]
    )
    paired = paired[paired["retention"] > 0].copy()
    paired["y_pass"] = (
        paired["retention"] >= cfg.retention_threshold
    ).astype(int)
    paired["sample_id"] = [f"pair_{i:05d}" for i in range(len(paired))]
    return paired


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class StabilityClassifierTrainer:
    """End-to-end training driver for the alkaline-stability classifier."""

    def __init__(self, cfg: StabilityClassifierConfig):
        self.cfg = cfg
        _ensure_dir(cfg.work_dir)

    # ------------------------------------------------------------------
    def fit(self) -> Tuple[CatBoostClassifier, Dict]:
        cfg = self.cfg
        df = pd.read_csv(cfg.data_path).replace([np.inf, -np.inf], np.nan)
        paired = pair_aging_records(df, cfg)
        paired.to_csv(
            os.path.join(cfg.work_dir, "paired_dataset.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        print(f"Paired dataset size: {len(paired)}")
        print(
            f"Pass-class fraction: {paired['y_pass'].mean():.3f} "
            f"(threshold = {cfg.retention_threshold})"
        )

        # Build descriptor cache once per polymer
        unique_polys = (
            paired[
                [
                    "polymer_unique_id",
                    cfg.smi_hydrophilic_col,
                    cfg.smi_hydrophobic_col,
                ]
            ]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        print(
            f"Computing Mordred descriptors for {len(unique_polys)} unique polymers..."
        )
        cache: Dict[str, pd.Series] = {}
        for _, row in unique_polys.iterrows():
            mol_h, _ = standardize_polymer_smiles(row[cfg.smi_hydrophilic_col])
            mol_p, _ = standardize_polymer_smiles(row[cfg.smi_hydrophobic_col])
            cache[row["polymer_unique_id"]] = pd.concat(
                [
                    calculate_block_descriptors(mol_h, "BlockA"),
                    calculate_block_descriptors(mol_p, "BlockB"),
                ]
            )

        desc_df = pd.DataFrame(
            [cache[pid] for pid in paired["polymer_unique_id"]]
        ).replace([np.inf, -np.inf], np.nan)

        exp_cols_no_time = [
            c for c in cfg.experiment_features if c != cfg.time_col
        ]
        exp_df = paired[exp_cols_no_time].copy()
        exp_df[cfg.time_col] = paired["time_deg"].values
        exp_df = exp_df[list(cfg.experiment_features)].replace(
            [np.inf, -np.inf], np.nan
        )

        y_all = paired["y_pass"].astype(int).values
        groups_all = paired["polymer_unique_id"].values

        gss = GroupShuffleSplit(
            n_splits=1, test_size=cfg.test_size, random_state=cfg.random_state
        )
        train_idx, test_idx = next(gss.split(paired, y_all, groups=groups_all))
        train_idx = np.asarray(train_idx)
        test_idx = np.asarray(test_idx)
        if len(set(groups_all[train_idx]) & set(groups_all[test_idx])) != 0:
            raise RuntimeError(
                "Train/test split leaked polymer groups; check input data."
            )

        exp_train_raw = exp_df.iloc[train_idx].reset_index(drop=True)
        exp_test_raw = exp_df.iloc[test_idx].reset_index(drop=True)
        desc_train_raw = desc_df.iloc[train_idx].reset_index(drop=True)
        desc_test_raw = desc_df.iloc[test_idx].reset_index(drop=True)
        y_train = pd.Series(y_all[train_idx]).reset_index(drop=True)
        y_test = pd.Series(y_all[test_idx]).reset_index(drop=True)
        groups_train = pd.Series(groups_all[train_idx]).reset_index(drop=True)

        # Build feature matrices using train-only statistics (no leakage)
        exp_fill = mode_fill_values(exp_train_raw, list(cfg.experiment_features))
        exp_train = apply_mode_fill(exp_train_raw, exp_fill)
        exp_test = apply_mode_fill(exp_test_raw, exp_fill)
        desc_filter = fit_descriptor_filter(desc_train_raw, cfg)
        desc_train = transform_with_descriptor_filter(desc_train_raw, desc_filter)
        desc_test = transform_with_descriptor_filter(desc_test_raw, desc_filter)
        X_train = pd.concat(
            [exp_train.reset_index(drop=True), desc_train.reset_index(drop=True)],
            axis=1,
        )
        X_test = pd.concat(
            [exp_test.reset_index(drop=True), desc_test.reset_index(drop=True)],
            axis=1,
        )
        median = X_train.median(numeric_only=True)
        X_train = X_train.fillna(median)
        X_test = X_test.fillna(median)

        joblib.dump(exp_fill, os.path.join(cfg.work_dir, "exp_mode_fill.pkl"))
        joblib.dump(desc_filter, os.path.join(cfg.work_dir, "desc_filter.pkl"))
        joblib.dump(median, os.path.join(cfg.work_dir, "feature_median.pkl"))
        joblib.dump(
            X_train.columns.tolist(),
            os.path.join(cfg.work_dir, "feature_names.pkl"),
        )

        # Hold-out sub-split inside the training set for early stopping
        gss2 = GroupShuffleSplit(
            n_splits=1, test_size=0.12, random_state=cfg.random_state
        )
        sub_tr, sub_va = next(
            gss2.split(X_train, y_train, groups=groups_train)
        )
        X_sub_tr, y_sub_tr = X_train.iloc[sub_tr], y_train.iloc[sub_tr]
        X_sub_va, y_sub_va = X_train.iloc[sub_va], y_train.iloc[sub_va]

        # ---------- Baseline (fixed parameters) -----------------------
        params = dict(cfg.cb_fixed_params)
        params.setdefault("random_seed", cfg.random_state)
        cb_warmup = CatBoostClassifier(iterations=cfg.cb_max_iters, **params)
        cb_warmup.fit(
            X_sub_tr,
            y_sub_tr,
            eval_set=(X_sub_va, y_sub_va),
            use_best_model=True,
            verbose=False,
            early_stopping_rounds=cfg.early_stopping_rounds,
        )
        best_it = int(getattr(cb_warmup, "get_best_iteration", lambda: -1)() or -1)
        if best_it <= 0:
            best_it = cfg.cb_fallback_iters

        cb_baseline = CatBoostClassifier(iterations=best_it, **params)
        cb_baseline.fit(X_train, y_train, verbose=False)
        baseline_dir = _ensure_dir(os.path.join(cfg.work_dir, "baseline"))
        joblib.dump(
            cb_baseline, os.path.join(baseline_dir, "catboost_model.pkl")
        )
        with open(os.path.join(baseline_dir, "params.json"), "w") as f:
            json.dump(
                {"params": params, "best_iteration": best_it}, f, indent=2
            )
        baseline_metrics = self._evaluate(
            cb_baseline, X_test, y_test, cfg.threshold, prefix="[Baseline] "
        )
        with open(
            os.path.join(baseline_dir, "test_metrics.json"), "w"
        ) as f:
            json.dump(
                {"threshold": cfg.threshold, "metrics": baseline_metrics},
                f,
                indent=2,
            )

        if not cfg.do_bayesian_tuning:
            return cb_baseline, {
                "baseline_metrics": baseline_metrics,
                "best_iteration": best_it,
                "feature_columns": X_train.columns.tolist(),
            }

        # ---------- Optuna tuning around the baseline -----------------
        import optuna

        def objective(trial: "optuna.Trial") -> float:
            tuned = dict(cfg.cb_fixed_params)
            tuned["depth"] = trial.suggest_int("depth", 3, 6)
            tuned["learning_rate"] = trial.suggest_float(
                "learning_rate", 0.03, 0.08, log=True
            )
            tuned["l2_leaf_reg"] = trial.suggest_float(
                "l2_leaf_reg", 2.0, 12.0, log=True
            )
            tuned["random_strength"] = trial.suggest_float(
                "random_strength", 0.0, 8.0
            )
            tuned["bagging_temperature"] = trial.suggest_float(
                "bagging_temperature", 0.0, 1.0
            )
            tuned["rsm"] = trial.suggest_float("rsm", 0.75, 1.0)
            tuned["iterations"] = cfg.cb_max_iters
            tuned["early_stopping_rounds"] = cfg.early_stopping_rounds

            gkf = GroupKFold(n_splits=cfg.cv_folds)
            aucs = []
            for fold_i, (tr, va) in enumerate(
                gkf.split(exp_train_raw, y_train, groups=groups_train), start=1
            ):
                exp_tr = apply_mode_fill(
                    exp_train_raw.iloc[tr],
                    mode_fill_values(
                        exp_train_raw.iloc[tr], list(cfg.experiment_features)
                    ),
                )
                exp_va = apply_mode_fill(
                    exp_train_raw.iloc[va],
                    mode_fill_values(
                        exp_train_raw.iloc[tr], list(cfg.experiment_features)
                    ),
                )
                df_filter = fit_descriptor_filter(
                    desc_train_raw.iloc[tr], cfg
                )
                desc_tr = transform_with_descriptor_filter(
                    desc_train_raw.iloc[tr], df_filter
                )
                desc_va = transform_with_descriptor_filter(
                    desc_train_raw.iloc[va], df_filter
                )
                X_tr = pd.concat(
                    [
                        exp_tr.reset_index(drop=True),
                        desc_tr.reset_index(drop=True),
                    ],
                    axis=1,
                )
                X_va = pd.concat(
                    [
                        exp_va.reset_index(drop=True),
                        desc_va.reset_index(drop=True),
                    ],
                    axis=1,
                )
                med = X_tr.median(numeric_only=True)
                X_tr = X_tr.fillna(med)
                X_va = X_va.fillna(med)
                model = CatBoostClassifier(**tuned)
                model.fit(
                    X_tr,
                    y_train.iloc[tr],
                    eval_set=(X_va, y_train.iloc[va]),
                    use_best_model=True,
                    verbose=False,
                )
                p = model.predict_proba(X_va)[:, 1]
                aucs.append(float(roc_auc_score(y_train.iloc[va], p)))
                trial.report(float(np.mean(aucs)), step=fold_i)
                if (
                    fold_i >= cfg.pruner_warmup_folds
                    and trial.should_prune()
                ):
                    raise optuna.exceptions.TrialPruned()
            return float(np.mean(aucs))

        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=cfg.pruner_startup_trials,
            n_warmup_steps=cfg.pruner_warmup_folds,
        )
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=cfg.random_state),
            pruner=pruner,
        )
        # Seed the search with the baseline parameters
        study.enqueue_trial(
            {
                "depth": cfg.cb_fixed_params.get("depth", 4),
                "learning_rate": cfg.cb_fixed_params.get("learning_rate", 0.05),
                "l2_leaf_reg": cfg.cb_fixed_params.get("l2_leaf_reg", 5.0),
                "random_strength": 0.0,
                "bagging_temperature": 0.0,
                "rsm": 1.0,
            }
        )
        study.optimize(objective, n_trials=cfg.n_trials, show_progress_bar=True)

        tuned_params = dict(cfg.cb_fixed_params)
        tuned_params.update(study.best_params)
        tuned_params["iterations"] = cfg.cb_max_iters
        tuned_params["early_stopping_rounds"] = cfg.early_stopping_rounds

        # Re-pick best_iteration with the tuned params on the same hold-out
        cb_warmup_t = CatBoostClassifier(**tuned_params)
        cb_warmup_t.fit(
            X_sub_tr,
            y_sub_tr,
            eval_set=(X_sub_va, y_sub_va),
            use_best_model=True,
            verbose=False,
        )
        best_it_t = int(
            getattr(cb_warmup_t, "get_best_iteration", lambda: -1)() or -1
        )
        if best_it_t <= 0:
            best_it_t = cfg.cb_fallback_iters

        final_params = dict(tuned_params)
        final_params.pop("early_stopping_rounds", None)
        final_params["iterations"] = best_it_t
        cb_tuned = CatBoostClassifier(**final_params)
        cb_tuned.fit(X_train, y_train, verbose=False)

        tuned_dir = _ensure_dir(os.path.join(cfg.work_dir, "tuned"))
        joblib.dump(cb_tuned, os.path.join(tuned_dir, "catboost_model.pkl"))
        with open(os.path.join(tuned_dir, "params.json"), "w") as f:
            json.dump(
                {"params": final_params, "best_iteration": best_it_t},
                f,
                indent=2,
            )
        tuned_metrics = self._evaluate(
            cb_tuned, X_test, y_test, cfg.threshold, prefix="[Tuned] "
        )
        with open(os.path.join(tuned_dir, "test_metrics.json"), "w") as f:
            json.dump(
                {"threshold": cfg.threshold, "metrics": tuned_metrics},
                f,
                indent=2,
            )

        return cb_tuned, {
            "baseline_metrics": baseline_metrics,
            "tuned_metrics": tuned_metrics,
            "best_params": study.best_params,
            "feature_columns": X_train.columns.tolist(),
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _evaluate(
        model: CatBoostClassifier,
        X_te: pd.DataFrame,
        y_te,
        threshold: float = 0.5,
        prefix: str = "",
    ) -> Dict[str, float]:
        proba = model.predict_proba(X_te)[:, 1]
        pred = (proba >= threshold).astype(int)
        metrics = {
            "AUC": float(roc_auc_score(y_te, proba)),
            "AP": float(average_precision_score(y_te, proba)),
            "ACC": float(accuracy_score(y_te, pred)),
            "BACC": float(balanced_accuracy_score(y_te, pred)),
            "F1": float(f1_score(y_te, pred)),
            "MCC": float(matthews_corrcoef(y_te, pred)),
            "BRIER": float(brier_score_loss(y_te, proba)),
        }
        print(f"{prefix}metrics @ threshold={threshold}: {metrics}")
        return metrics


# ---------------------------------------------------------------------------
# Inference helper for use as a post-RL filter
# ---------------------------------------------------------------------------
class StabilityClassifier:
    """Load a trained classifier + its preprocessing artefacts and score
    candidate AEMs from raw SMILES + experimental conditions.
    """

    def __init__(self, work_dir: str, prefer_tuned: bool = True):
        self.work_dir = work_dir
        cand_dirs = ["tuned", "baseline"] if prefer_tuned else ["baseline", "tuned"]
        for sub in cand_dirs:
            model_path = os.path.join(work_dir, sub, "catboost_model.pkl")
            if os.path.exists(model_path):
                self.model: CatBoostClassifier = joblib.load(model_path)
                break
        else:
            raise FileNotFoundError(
                f"No CatBoost model found under {work_dir}/(tuned|baseline)."
            )
        self.exp_fill = joblib.load(
            os.path.join(work_dir, "exp_mode_fill.pkl")
        )
        self.desc_filter = joblib.load(
            os.path.join(work_dir, "desc_filter.pkl")
        )
        self.median = joblib.load(
            os.path.join(work_dir, "feature_median.pkl")
        )
        self.feature_names: List[str] = joblib.load(
            os.path.join(work_dir, "feature_names.pkl")
        )

    # ------------------------------------------------------------------
    def predict(
        self,
        hydrophilic_smi: str,
        hydrophobic_smi: str,
        experiment: Dict[str, float],
        threshold: float = 0.5,
    ) -> Dict:
        mol_h, _ = standardize_polymer_smiles(hydrophilic_smi)
        mol_p, _ = standardize_polymer_smiles(hydrophobic_smi)
        if mol_h is None or mol_p is None:
            raise ValueError("Could not standardise input SMILES.")
        d = pd.concat(
            [
                calculate_block_descriptors(mol_h, "BlockA"),
                calculate_block_descriptors(mol_p, "BlockB"),
            ]
        ).to_frame().T.replace([np.inf, -np.inf], np.nan)
        e = pd.DataFrame([experiment]).replace([np.inf, -np.inf], np.nan)
        e2 = apply_mode_fill(e, self.exp_fill)
        d2 = transform_with_descriptor_filter(d, self.desc_filter)

        X_new = pd.concat(
            [e2.reset_index(drop=True), d2.reset_index(drop=True)], axis=1
        )
        for c in self.feature_names:
            if c not in X_new.columns:
                X_new[c] = np.nan
        X_new = X_new[self.feature_names].fillna(self.median)
        prob = float(self.model.predict_proba(X_new)[:, 1][0])
        return {
            "prob_pass": prob,
            "pred_pass": int(prob >= threshold),
            "threshold": float(threshold),
        }
