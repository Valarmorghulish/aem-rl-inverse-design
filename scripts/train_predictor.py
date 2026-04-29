"""
Train the polyBERT-based multitask predictor for hydroxide conductivity and SR.

Usage
-----
    python scripts/train_predictor.py \
        --aem-csv data/Conductivity_data_merged_processed.csv \
        --polybert-path local_models/polyBERT \
        --output-dir checkpoints/predictor

This script trains the two-step predictor described in Section 2.3.1 of the
paper:
    1. Self-supervised MLM step on the AEM SMILES corpus, starting from the
       polyBERT initialisation, to specialise the encoder.
    2. Supervised fine-tuning with five-fold stratified cross-validation on
       the structure-condition-property records.

The supervised step uses a masked multitask MSE loss so that records with
only one of the two targets still contribute. Per the paper, only the two
predictor outputs Conductivity and SR are kept; auxiliary targets (water
uptake, tensile properties) are not used.

References
----------
- Kuenneth, C., & Ramprasad, R. (2023). polyBERT: a chemical language model
  to enable fully machine-driven ultrafast polymer informatics.
  *Nature Communications*, 14, 4099.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aem_rl.predictor import (  # noqa: E402
    MultitaskPredictorTrainer,
    PredictorConfig,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the polyBERT-based two-task AEM property predictor."
    )
    p.add_argument(
        "--aem-csv",
        required=True,
        help="Path to the curated AEM structure-condition-property CSV.",
    )
    p.add_argument(
        "--polybert-path",
        required=True,
        help="Local directory holding polyBERT (config.json, pytorch_model.bin, "
        "tokenizer.json, etc.).",
    )
    p.add_argument(
        "--output-dir",
        default="./checkpoints/predictor",
        help="Where to save the MLM and per-fold regression checkpoints.",
    )

    # Optional column overrides
    p.add_argument("--smiles-hydrophilic-col", default="Hydrophilic")
    p.add_argument("--smiles-hydrophobic-col", default="Hydrophobic")
    p.add_argument(
        "--cont-cols",
        nargs="+",
        default=["HydrophilicFrac", "IEC", "Temperature", "RH"],
        help="Continuous condition columns. Defaults match the paper's setting.",
    )
    p.add_argument(
        "--cat-cols",
        nargs="+",
        default=["PolymerArchitecture"],
        help="Categorical condition columns.",
    )

    # Stage hyper-parameters
    p.add_argument("--mlm-epochs", type=int, default=60)
    p.add_argument("--mlm-batch-size", type=int, default=8)
    p.add_argument("--mlm-grad-accum", type=int, default=4)
    p.add_argument("--mlm-lr", type=float, default=5e-5)
    p.add_argument("--mlm-unfreeze-layers", type=int, default=2)
    p.add_argument("--mlm-augment-factor", type=int, default=8)

    p.add_argument("--reg-epochs", type=int, default=80)
    p.add_argument("--reg-batch-size", type=int, default=8)
    p.add_argument("--reg-eval-batch-size", type=int, default=16)
    p.add_argument("--reg-grad-accum", type=int, default=2)
    p.add_argument("--reg-lr", type=float, default=2e-5)
    p.add_argument("--reg-unfreeze-layers", type=int, default=4)
    p.add_argument("--reg-dropout", type=float, default=0.1)
    p.add_argument("--reg-warmup-ratio", type=float, default=0.06)
    p.add_argument("--reg-weight-decay", type=float, default=0.01)
    p.add_argument("--reg-early-stop-patience", type=int, default=6)

    p.add_argument("--max-smiles-len", type=int, default=256)
    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def plot_training_curves(fold_histories, out_path: Path) -> None:
    """Plot the per-epoch validation loss for each fold (training-curve only)."""
    import pandas as pd  # local import keeps the script importable without pandas

    plt.figure(figsize=(10, 6))
    for i, log_df in enumerate(fold_histories):
        log_df = pd.DataFrame(log_df)
        if "eval_loss" in log_df.columns:
            eval_logs = log_df[log_df["eval_loss"].notna()]
            if not eval_logs.empty:
                plt.plot(
                    eval_logs["epoch"],
                    eval_logs["eval_loss"],
                    label=f"Fold {i + 1}",
                    alpha=0.7,
                )
    plt.title("K-Fold validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Masked-MSE loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    cfg = PredictorConfig(
        aem_data_path=args.aem_csv,
        polybert_path=args.polybert_path,
        output_dir=args.output_dir,
        smiles_hydrophilic_col=args.smiles_hydrophilic_col,
        smiles_hydrophobic_col=args.smiles_hydrophobic_col,
        cont_cols=tuple(args.cont_cols),
        cat_cols=tuple(args.cat_cols),
        max_smiles_len=args.max_smiles_len,
        mlm_epochs=args.mlm_epochs,
        mlm_batch_size=args.mlm_batch_size,
        mlm_grad_accum=args.mlm_grad_accum,
        mlm_lr=args.mlm_lr,
        mlm_unfreeze_layers=args.mlm_unfreeze_layers,
        mlm_augment_factor=args.mlm_augment_factor,
        reg_epochs=args.reg_epochs,
        reg_batch_size=args.reg_batch_size,
        reg_eval_batch_size=args.reg_eval_batch_size,
        reg_grad_accum=args.reg_grad_accum,
        reg_lr=args.reg_lr,
        reg_unfreeze_layers=args.reg_unfreeze_layers,
        reg_dropout=args.reg_dropout,
        reg_warmup_ratio=args.reg_warmup_ratio,
        reg_weight_decay=args.reg_weight_decay,
        reg_early_stop_patience=args.reg_early_stop_patience,
        test_size=args.test_size,
        n_folds=args.n_folds,
        random_state=args.random_state,
    )
    print("=" * 80)
    print("Predictor training configuration")
    print("=" * 80)
    print(f"  data_path   : {cfg.aem_data_path}")
    print(f"  polybert    : {cfg.polybert_path}")
    print(f"  output_dir  : {cfg.output_dir}")
    print(f"  cont cols   : {cfg.cont_cols}")
    print(f"  cat cols    : {cfg.cat_cols}")
    print(f"  targets     : {cfg.target_cols}")
    print(f"  cv folds    : {cfg.n_folds}")
    print("=" * 80)

    trainer = MultitaskPredictorTrainer(cfg)
    fold_models, fold_scalers, test_df, fold_histories = trainer.fit()

    out_path = Path(cfg.output_dir) / "kfold_validation_loss.png"
    plot_training_curves(fold_histories, out_path)
    print(f"Validation-loss curves saved to {out_path}.")

    print(
        f"Done. Saved K-fold ensemble to {Path(cfg.output_dir) / 'ensemble'}."
    )


if __name__ == "__main__":
    main()
