"""
Generator stage 2: Fine-tune the pretrained generator on AEM SMILES.

Usage
-----
    python scripts/train_generator_finetune.py \
        --aem-csv data/Conductivity_data_merged_processed.csv \
        --pretrained checkpoints/generator/checkpoint_pretrain.pth \
        --tokens checkpoints/generator/tokens.json \
        --output-dir checkpoints/generator

Steps
-----
1. Read the curated AEM CSV. Keep unique hydrophilic motifs that contain
   ``[N+]`` (cationic) and unique hydrophobic motifs that do not. Write a
   role-prefixed dataset (``^`` for Phi, ``~`` for Pho).
2. Re-instantiate the same stack-augmented GRU architecture used during
   pretraining (hidden = 1500, stack 1500 x 200) and load the pretrained
   weights.
3. Fine-tune for 500,000 iterations with a small Adadelta learning rate
   (5e-5) and SMILES augmentation disabled, so that the generator learns
   AEM-specific structural priors without losing the polymer syntax
   acquired in stage 1.

Reference
---------
Popova, M., Isayev, O., & Tropsha, A. (2018).
Deep reinforcement learning for de novo drug design.
Science Advances, 4(7), eaap7885.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aem_rl.data import GeneratorData  # noqa: E402
from aem_rl.stack_rnn import StackAugmentedRNN  # noqa: E402

RDLogger.DisableLog("rdApp.*")


# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune the pretrained generator on AEM SMILES."
    )
    p.add_argument("--aem-csv", required=True, help="Curated AEM CSV file.")
    p.add_argument(
        "--smiles-hydrophilic-col",
        default="Hydrophilic",
        help="Column name of the hydrophilic motif SMILES.",
    )
    p.add_argument(
        "--smiles-hydrophobic-col",
        default="Hydrophobic",
        help="Column name of the hydrophobic motif SMILES.",
    )
    p.add_argument(
        "--pretrained",
        required=True,
        help="Path to the pretrained generator checkpoint (.pth).",
    )
    p.add_argument(
        "--tokens",
        required=True,
        help="Path to the tokens.json produced during pretraining.",
    )
    p.add_argument(
        "--output-dir",
        default="./checkpoints/generator",
        help="Where to save the fine-tuned model and training artefacts.",
    )
    p.add_argument("--n-iterations", type=int, default=500_000)
    p.add_argument("--hidden-size", type=int, default=1500)
    p.add_argument("--stack-width", type=int, default=1500)
    p.add_argument("--stack-depth", type=int, default=200)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--print-every", type=int, default=500)
    p.add_argument("--plot-every", type=int, default=10)
    p.add_argument(
        "--augment",
        action="store_true",
        help="Enable SMILES augmentation during fine-tuning (off by default).",
    )
    p.add_argument(
        "--checkpoint-name",
        default="checkpoint_finetune.pth",
        help="Filename for the fine-tuned model.",
    )
    p.add_argument(
        "--finetune-data-name",
        default="aem_finetune_unique_with_roles.csv",
        help="Filename of the role-prefixed AEM dataset to be written.",
    )
    return p.parse_args()


def build_finetune_dataset(
    aem_csv: str,
    smi_h: str,
    smi_p: str,
    output_path: Path,
) -> int:
    """Write the role-prefixed AEM SMILES dataset; return the number of rows.

    Hydrophilic motifs must contain ``[N+]`` (cationic centre) and
    hydrophobic motifs must not. Both columns are deduplicated.
    """
    df = pd.read_csv(aem_csv)
    hydro = sorted(
        s
        for s in df[smi_h].dropna().astype(str).unique().tolist()
        if "[N+]" in s
    )
    phobic = sorted(
        s
        for s in df[smi_p].dropna().astype(str).unique().tolist()
        if "[N+]" not in s
    )
    rows = [f"^{s}" for s in hydro] + [f"~{s}" for s in phobic]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False, header=False)
    print(
        f"Wrote {len(rows)} role-prefixed AEM SMILES "
        f"({len(hydro)} hydrophilic, {len(phobic)} hydrophobic) "
        f"to {output_path}."
    )
    return len(rows)


def sanity_check_samples(model, gen_data, n: int = 5) -> None:
    print("\n--- Sanity check ---")
    for i in range(n):
        h_raw = model.evaluate(gen_data, prime_str="^")
        p_raw = model.evaluate(gen_data, prime_str="~")
        h_clean = h_raw.replace("^", "").replace("<", "").replace(">", "")
        p_clean = p_raw.replace("~", "").replace("<", "").replace(">", "")
        h_ok = Chem.MolFromSmiles(h_clean) is not None
        p_ok = Chem.MolFromSmiles(p_clean) is not None
        print(f" pair {i + 1}:")
        print(f"   ^ {h_clean!r} (parses={h_ok}, has [N+]={'[N+]' in h_clean})")
        print(f"   ~ {p_clean!r} (parses={p_ok}, has [N+]={'[N+]' in p_clean})")


# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    use_cuda = torch.cuda.is_available()
    print(f"CUDA available: {use_cuda}")

    with open(args.tokens, "r") as f:
        tokens = json.load(f)
    print(f"Loaded {len(tokens)}-character vocabulary from {args.tokens}.")

    finetune_csv = output_dir / args.finetune_data_name
    build_finetune_dataset(
        args.aem_csv,
        args.smiles_hydrophilic_col,
        args.smiles_hydrophobic_col,
        finetune_csv,
    )

    gen_data = GeneratorData(
        training_data_path=str(finetune_csv),
        delimiter=",",
        cols_to_read=[0],
        keep_header=False,
        tokens=tokens,
        max_len=args.max_len,
        use_cuda=use_cuda,
    )
    if gen_data.file_len == 0:
        raise RuntimeError(
            f"Fine-tune dataset at {finetune_csv} is empty; "
            "check that the AEM CSV has both hydrophilic and hydrophobic columns."
        )

    print(
        "Re-instantiating stack-augmented GRU "
        f"(hidden={args.hidden_size}, stack={args.stack_width}x{args.stack_depth}) ..."
    )
    model = StackAugmentedRNN(
        input_size=len(tokens),
        hidden_size=args.hidden_size,
        output_size=len(tokens),
        layer_type="GRU",
        n_layers=1,
        is_bidirectional=False,
        has_stack=True,
        stack_width=args.stack_width,
        stack_depth=args.stack_depth,
        use_cuda=use_cuda,
        optimizer_instance=torch.optim.Adadelta,
        lr=args.lr,
    )

    print(f"Loading pretrained weights from {args.pretrained} ...")
    model.load_model(args.pretrained)
    model.change_lr(args.lr)
    print(f"Fine-tune learning rate set to {args.lr}.")

    print(
        f"Fine-tuning for {args.n_iterations:,} iterations "
        f"(augment={'yes' if args.augment else 'no'}) ..."
    )
    losses = model.fit(
        gen_data,
        n_iterations=args.n_iterations,
        print_every=args.print_every,
        plot_every=args.plot_every,
        augment=args.augment,
    )

    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title("Generator fine-tuning loss")
    plt.xlabel(f"x{args.plot_every} iterations")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(output_dir / "finetune_loss.png", dpi=150)
    plt.close()
    print(f"Loss curve saved to {output_dir / 'finetune_loss.png'}.")

    ckpt_path = output_dir / args.checkpoint_name
    model.save_model(str(ckpt_path))
    print(f"Fine-tuned model saved to {ckpt_path}.")

    sanity_check_samples(model, gen_data)


if __name__ == "__main__":
    main()
