"""
Train the alkaline-stability binary classifier (Pass / Fail).

Usage
-----
    python scripts/train_stability_classifier.py \
        --data-csv data/stability4_final.csv \
        --work-dir checkpoints/stability

This script trains the CatBoost classifier described in Section 2.3.2 of the
paper. It performs:
    1. Pairing of fresh (time = 0) and degraded (time > 0) measurements
       sharing the same polymer identity and operational conditions.
    2. Feature construction from operational variables and 2D Mordred
       descriptors of the hydrophilic and hydrophobic motifs, with the
       train-only descriptor filtering cascade.
    3. A polymer-grouped train / test split to prevent leakage of the same
       polymer identity across splits.
    4. A CatBoost baseline (fixed parameters) followed by an Optuna-TPE
       Bayesian search around the baseline for fine refinement (toggle off
       with --no-tuning).

References
----------
- Schertzer, R. et al. (2026). [Curated AEM stability dataset cited in the
  paper.]
- Hu, et al. (2025). CRYSTAL: an attention-enhanced framework for AEM
  generative design with multi-property prediction.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aem_rl.stability_classifier import (  # noqa: E402
    StabilityClassifierConfig,
    StabilityClassifierTrainer,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the CatBoost alkaline-stability classifier."
    )
    p.add_argument(
        "--data-csv",
        required=True,
        help="Path to the paired AEM aging CSV (must contain Hydrophilic, "
        "Hydrophobic, Cond and time(h) columns, plus the operational "
        "variables listed in the paper).",
    )
    p.add_argument(
        "--work-dir",
        default="./checkpoints/stability",
        help="Output directory for the model and preprocessing artefacts.",
    )
    p.add_argument(
        "--retention-threshold",
        type=float,
        default=0.80,
        help="Conductivity-retention threshold for the Pass class.",
    )
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument(
        "--n-trials",
        type=int,
        default=40,
        help="Number of Optuna trials for the small-range Bayesian search.",
    )
    p.add_argument(
        "--no-tuning",
        action="store_true",
        help="Skip the Bayesian tuning step and only train the baseline.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = StabilityClassifierConfig(
        data_path=args.data_csv,
        work_dir=args.work_dir,
        retention_threshold=args.retention_threshold,
        threshold=args.threshold,
        cv_folds=args.cv_folds,
        test_size=args.test_size,
        random_state=args.random_state,
        n_trials=args.n_trials,
        do_bayesian_tuning=not args.no_tuning,
    )
    print("=" * 80)
    print("Stability-classifier training configuration")
    print("=" * 80)
    print(f"  data_path           : {cfg.data_path}")
    print(f"  work_dir            : {cfg.work_dir}")
    print(f"  retention threshold : {cfg.retention_threshold}")
    print(f"  cv folds            : {cfg.cv_folds}")
    print(f"  do bayesian tuning  : {cfg.do_bayesian_tuning}")
    print("=" * 80)

    trainer = StabilityClassifierTrainer(cfg)
    model, info = trainer.fit()
    print("Training complete.")
    print(f"  baseline metrics : {info.get('baseline_metrics')}")
    if "tuned_metrics" in info:
        print(f"  tuned metrics    : {info.get('tuned_metrics')}")
        print(f"  best params      : {info.get('best_params')}")


if __name__ == "__main__":
    main()
