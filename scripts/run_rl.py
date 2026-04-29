"""
RL stage: bias the fine-tuned generator toward target-property regions.

Usage
-----
    python scripts/run_rl.py \
        --finetuned checkpoints/generator/checkpoint_finetune.pth \
        --tokens checkpoints/generator/tokens.json \
        --aem-finetune-data checkpoints/generator/aem_finetune_unique_with_roles.csv \
        --predictor-ensemble checkpoints/predictor/ensemble \
        --polybert-path local_models/polyBERT \
        --family PAEK \
        --output-dir checkpoints/rl/PAEK

Loop (Section 2.4 of the paper)
-------------------------------
    for iteration in 1..N_ITERATIONS:
        for j in 1..N_POLICY_UPDATES:
            for k in 1..N_BATCH:
                hydro = generator.evaluate(prime='^')
                phobic = generator.evaluate(prime='~')
                reward, details = reward_fn(hydro, phobic, predictor)
                if reward <= 0: resample
            policy-gradient update on accepted (hydro, phobic) pairs
        every CHECKPOINT_EVERY iterations: sample N_CHECKPOINT_SAMPLES pairs
            and record their predicted-property distribution
        every SAVE_EVERY iterations: save the generator weights

The reward function follows Eq. 5 of the paper. Family-specific
structural-preference D(S_T) is plugged in via the ``--family`` argument
(currently PAEK and GENERIC are registered; new families can be added by
subclassing :class:`aem_rl.reward.FamilyConstraint`).

Reference
---------
Popova, M., Isayev, O., & Tropsha, A. (2018).
Deep reinforcement learning for de novo drug design.
Science Advances, 4(7), eaap7885.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rdkit import RDLogger  # noqa: E402
from torch.optim.lr_scheduler import StepLR  # noqa: E402
from tqdm.auto import tqdm, trange  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aem_rl.data import GeneratorData  # noqa: E402
from aem_rl.predictor import EnsembleTransformerPredictorForRL  # noqa: E402
from aem_rl.reinforcement import Reinforcement  # noqa: E402
from aem_rl.reward import RewardConfig, build_reward_fn  # noqa: E402
from aem_rl.stack_rnn import StackAugmentedRNN  # noqa: E402

RDLogger.DisableLog("rdApp.*")


# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the RL stage.")
    p.add_argument("--finetuned", required=True, help="Fine-tuned generator .pth.")
    p.add_argument("--tokens", required=True, help="tokens.json built during pretraining.")
    p.add_argument(
        "--aem-finetune-data",
        required=True,
        help="Role-prefixed AEM CSV produced by the fine-tune script.",
    )
    p.add_argument(
        "--predictor-ensemble",
        required=True,
        help="Directory containing the trained K-fold predictor ensemble.",
    )
    p.add_argument(
        "--polybert-path",
        required=True,
        help="Local polyBERT directory (config + tokenizer).",
    )
    p.add_argument(
        "--family",
        default="PAEK",
        help="AEM family for the structural-preference term D(S_T). "
        "Registered: PAEK, GENERIC.",
    )
    p.add_argument(
        "--output-dir",
        default="./checkpoints/rl",
        help="Where to save RL checkpoints, history and plots.",
    )

    # Architecture (must match the fine-tuned model)
    p.add_argument("--hidden-size", type=int, default=1500)
    p.add_argument("--stack-width", type=int, default=1500)
    p.add_argument("--stack-depth", type=int, default=200)
    p.add_argument("--max-len", type=int, default=256)

    # RL hyper-parameters
    p.add_argument("--n-iterations", type=int, default=50)
    p.add_argument("--n-policy-updates", type=int, default=15)
    p.add_argument("--n-batch", type=int, default=15)
    p.add_argument("--gamma", type=float, default=0.97)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--lr-step-size", type=int, default=100)
    p.add_argument("--lr-gamma", type=float, default=0.9)
    p.add_argument("--grad-clipping", type=float, default=1.0)
    p.add_argument("--max-attempts-per-sample", type=int, default=200)

    # Sampling / checkpoint cadence
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--n-eval-samples", type=int, default=200)
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--n-checkpoint-samples", type=int, default=200)
    p.add_argument("--save-every", type=int, default=10)
    return p.parse_args()


# ---------------------------------------------------------------------------
def evaluate_progress(
    generator,
    predictor,
    data: GeneratorData,
    n_to_generate: int,
    reward_fn,
    iter_num: int,
) -> Dict:
    """Generate samples without policy updates, then summarise property stats."""
    generator.eval()
    cond, sr = [], []
    for _ in tqdm(range(n_to_generate), desc=f"Eval @ iter {iter_num}", leave=False):
        with torch.no_grad():
            hydro = generator.evaluate(data, prime_str="^")
            phobic = generator.evaluate(data, prime_str="~")
        reward, details = reward_fn(hydro, phobic, predictor)
        if reward > 0 and details.get("is_valid", False):
            cond.append(details.get("predicted_conductivity", 0.0))
            sr.append(details.get("SR", float("nan")))
    generator.train()
    if not cond:
        tqdm.write(f"  Eval @ iter {iter_num}: no valid molecules generated.")
        return {"n_valid": 0}
    summary = {
        "n_valid": len(cond),
        "mean_cond": float(np.mean(cond)),
        "max_cond": float(np.max(cond)),
        "mean_sr": float(np.nanmean(sr)),
    }
    tqdm.write(
        f"  Eval @ iter {iter_num}: n_valid={summary['n_valid']}/{n_to_generate}, "
        f"mean Cond={summary['mean_cond']:.2f}, mean SR={summary['mean_sr']:.2f}, "
        f"max Cond={summary['max_cond']:.2f}"
    )
    return summary


def sample_property_distribution(
    generator,
    predictor,
    data: GeneratorData,
    reward_fn,
    n_samples: int,
    prop_name: str,
) -> List[float]:
    """Collect a distribution of one predicted property over ``n_samples`` valid pairs."""
    generator.eval()
    out: List[float] = []
    pbar = tqdm(range(n_samples), desc=f"Sampling {prop_name}", leave=False)
    for _ in pbar:
        with torch.no_grad():
            hydro = generator.evaluate(data, prime_str="^")
            phobic = generator.evaluate(data, prime_str="~")
        reward, details = reward_fn(hydro, phobic, predictor)
        if details.get("is_valid", False) and "Full Prediction" in details.get(
            "stage", ""
        ):
            v = details.get(prop_name, float("nan"))
            if v is not None and not math.isnan(v):
                out.append(float(v))
    generator.train()
    return out


def plot_training_curves(history: Dict, out_path: Path) -> None:
    """Plot the RL training-curve panels (loss, reward, conductivity, SR, lr)."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    fig.suptitle("RL training curves", fontsize=16)

    axes[0].plot(history["total_rewards"], label="Total")
    axes[0].plot(history["prop_rewards"], label="Property", linestyle="--")
    axes[0].set_title("Reward")
    axes[0].set_xlabel("Gradient updates")
    axes[0].grid(True, alpha=0.4)
    axes[0].legend()

    for ax, key, title in [
        (axes[1], "losses", "Loss"),
        (axes[2], "valid_ratios", "Acceptance ratio"),
        (axes[3], "avg_conductivity", "Avg predicted conductivity (mS/cm)"),
        (axes[4], "avg_sr", "Avg predicted SR (%)"),
        (axes[5], "lr", "Learning rate"),
    ]:
        ax.plot(history[key])
        ax.set_title(title)
        ax.set_xlabel("Gradient updates")
        ax.grid(True, alpha=0.4)
    if history["valid_ratios"]:
        axes[2].set_ylim(0, 1.05)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open(args.tokens, "r") as f:
        tokens = json.load(f)
    print(f"Loaded {len(tokens)}-character vocabulary.")

    gen_data = GeneratorData(
        training_data_path=args.aem_finetune_data,
        delimiter=",",
        cols_to_read=[0],
        keep_header=False,
        tokens=tokens,
        max_len=args.max_len,
        use_cuda=(device.type == "cuda"),
    )
    print(f"Generator dataset size: {gen_data.file_len}")

    print("Loading fine-tuned generator ...")
    agent = StackAugmentedRNN(
        input_size=len(tokens),
        hidden_size=args.hidden_size,
        output_size=len(tokens),
        layer_type="GRU",
        n_layers=1,
        is_bidirectional=False,
        has_stack=True,
        stack_width=args.stack_width,
        stack_depth=args.stack_depth,
        use_cuda=(device.type == "cuda"),
        optimizer_instance=torch.optim.Adadelta,
        lr=args.lr,
    )
    agent.load_model(args.finetuned)

    optimizer = torch.optim.Adadelta(agent.parameters(), lr=args.lr)
    agent.optimizer = optimizer
    scheduler = StepLR(
        optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
    )

    print("Loading predictor ensemble ...")
    predictor = EnsembleTransformerPredictorForRL(
        ensemble_base_dir=args.predictor_ensemble,
        device=device,
        polybert_path=args.polybert_path,
    )

    print(f"Building reward function for family '{args.family}' ...")
    reward_fn = build_reward_fn(args.family, RewardConfig())

    rl_engine = Reinforcement(
        generator=agent, predictor=predictor, get_reward_func=reward_fn
    )

    history: Dict[str, List[float]] = {
        "total_rewards": [],
        "prop_rewards": [],
        "losses": [],
        "valid_ratios": [],
        "avg_conductivity": [],
        "avg_sr": [],
        "lr": [],
    }
    checkpoint_samples: Dict[str, Dict[int, List[float]]] = {
        "predicted_conductivity": {},
        "SR": {},
    }
    nan_skips = 0

    print(
        f"Starting RL training: {args.n_iterations} iterations, "
        f"{args.n_policy_updates} policy updates per iter, batch={args.n_batch}."
    )
    for i in range(args.n_iterations):
        best_in_iter: Dict = {"total_reward": -1.0}
        pbar_inner = trange(
            args.n_policy_updates,
            desc=f"Iter {i + 1}/{args.n_iterations}",
            leave=True,
        )
        for _ in pbar_inner:
            avg_reward, loss, batch_debug, n_attempts = (
                rl_engine.policy_gradient_step(
                    data=gen_data,
                    n_batch=args.n_batch,
                    gamma=args.gamma,
                    grad_clipping=args.grad_clipping,
                    max_attempts_per_sample=args.max_attempts_per_sample,
                )
            )
            if loss is None or math.isnan(loss):
                nan_skips += 1
                tqdm.write(
                    f"  WARNING: NaN loss; skipped step (total skips: {nan_skips})."
                )
                continue
            history["total_rewards"].append(float(avg_reward))
            history["losses"].append(float(loss))
            history["lr"].append(scheduler.get_last_lr()[0])
            scheduler.step()

            valid = [d for d in batch_debug if d.get("is_valid")]
            prop_rewards = [d.get("total_reward", 0.0) for d in valid]
            avg_prop_reward = (
                float(np.mean(prop_rewards)) if prop_rewards else 0.0
            )
            cond_batch = [
                d.get("predicted_conductivity", 0.0) for d in valid
            ]
            sr_batch = [d.get("SR", float("nan")) for d in valid]
            mean_c = float(np.mean(cond_batch)) if cond_batch else 0.0
            mean_s = (
                float(np.nanmean(sr_batch))
                if any(not math.isnan(s) for s in sr_batch)
                else 0.0
            )
            accept = (args.n_batch / n_attempts) if n_attempts > 0 else 0.0

            history["prop_rewards"].append(avg_prop_reward)
            history["valid_ratios"].append(float(accept))
            history["avg_conductivity"].append(mean_c)
            history["avg_sr"].append(mean_s)

            pbar_inner.set_postfix(
                R=f"{avg_reward:.3f}",
                L=f"{loss:.3f}",
                C=f"{mean_c:.2f}",
                S=f"{mean_s:.2f}",
                Acc=f"{accept * 100:5.1f}%",
            )

            for d in valid:
                if d.get("total_reward", 0) > best_in_iter["total_reward"]:
                    best_in_iter = d.copy()

        last_mean = float(
            np.mean(history["total_rewards"][-args.n_policy_updates :])
        )
        tqdm.write(f"  Iter {i + 1}: avg reward = {last_mean:.4f}")
        if best_in_iter["total_reward"] > -1:
            elite = (
                " (elite)" if best_in_iter.get("elite_boost_applied") else ""
            )
            tqdm.write(
                f"  Best in iter (R={best_in_iter['total_reward']:.3f}, "
                f"C={best_in_iter.get('predicted_conductivity', 0):.2f}, "
                f"SR={best_in_iter.get('SR', 0):.2f}{elite}):"
            )
            tqdm.write(f"    H: {best_in_iter.get('hydro_smi', 'N/A')}")
            tqdm.write(f"    P: {best_in_iter.get('phobic_smi', 'N/A')}")

        if (i + 1) % args.eval_every == 0:
            evaluate_progress(
                agent,
                predictor,
                gen_data,
                args.n_eval_samples,
                reward_fn,
                iter_num=i + 1,
            )

        if (i + 1) % args.checkpoint_every == 0:
            iter_num = i + 1
            tqdm.write(f"  Checkpoint sampling at iter {iter_num} ...")
            checkpoint_samples["predicted_conductivity"][iter_num] = (
                sample_property_distribution(
                    agent,
                    predictor,
                    gen_data,
                    reward_fn,
                    args.n_checkpoint_samples,
                    "predicted_conductivity",
                )
            )
            checkpoint_samples["SR"][iter_num] = sample_property_distribution(
                agent,
                predictor,
                gen_data,
                reward_fn,
                args.n_checkpoint_samples,
                "SR",
            )

        if (i + 1) % args.save_every == 0 and i > 0:
            save_path = output_dir / f"checkpoint_RL_iter_{i + 1}.pth"
            agent.save_model(str(save_path))
            tqdm.write(f"  Saved checkpoint to {save_path}.")

    final_path = output_dir / "checkpoint_RL_final.pth"
    agent.save_model(str(final_path))
    print(f"Training complete. Final model saved to {final_path}.")

    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f)
    with open(output_dir / "checkpoint_samples.json", "w") as f:
        json.dump(checkpoint_samples, f)

    plot_training_curves(history, output_dir / "training_curves.png")
    print(f"Training curves saved to {output_dir / 'training_curves.png'}.")


if __name__ == "__main__":
    main()
