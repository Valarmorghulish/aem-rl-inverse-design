"""
Generator stage 1: Pretrain on the PI1M polymer corpus.

Usage
-----
    python scripts/train_generator_pretrain.py \
        --pi1m-csv data/PI1M.csv \
        --aem-csv data/Conductivity_data_merged_processed.csv \
        --output-dir checkpoints/generator

Steps
-----
1. Read the PI1M corpus, unify all polymerisation connection sites to
   ``[*]``, and write two role-prefixed copies of every polymer (``^``
   for hydrophilic, ``~`` for hydrophobic). The role prefixes let the
   single generator model produce both motif types from the same weights.
2. Build the character-level alphabet from the union of PI1M and the AEM
   SMILES so that the same vocabulary covers Pretrain, Fine-tune and RL
   stages.
3. Train a stack-augmented GRU (hidden = 1500, stack 1500 x 200) for
   2,000,000 iterations using SMILES augmentation with the Adadelta
   optimiser at lr = 0.01.

Reference
---------
Popova, M., Isayev, O., & Tropsha, A. (2018).
Deep reinforcement learning for de novo drug design.
Science Advances, 4(7), eaap7885.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from rdkit import RDLogger  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

# Make ``src`` importable when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aem_rl.data import GeneratorData  # noqa: E402
from aem_rl.stack_rnn import StackAugmentedRNN  # noqa: E402

RDLogger.DisableLog("rdApp.*")


# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pretrain the stack-augmented generator on PI1M."
    )
    p.add_argument("--pi1m-csv", required=True, help="Path to the PI1M corpus.")
    p.add_argument(
        "--pi1m-delimiter",
        default="\t",
        help="Delimiter for the PI1M file (default: tab).",
    )
    p.add_argument(
        "--aem-csv",
        required=True,
        help="Path to the curated AEM CSV (used only for vocabulary).",
    )
    p.add_argument(
        "--smiles-hydrophilic-col",
        default="Hydrophilic",
        help="Column name for the hydrophilic motif SMILES in the AEM CSV.",
    )
    p.add_argument(
        "--smiles-hydrophobic-col",
        default="Hydrophobic",
        help="Column name for the hydrophobic motif SMILES in the AEM CSV.",
    )
    p.add_argument(
        "--output-dir",
        default="./checkpoints/generator",
        help="Where to save tokens.json, the role-prefixed dataset and the model.",
    )
    p.add_argument("--n-iterations", type=int, default=2_000_000)
    p.add_argument("--hidden-size", type=int, default=1500)
    p.add_argument("--stack-width", type=int, default=1500)
    p.add_argument("--stack-depth", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--print-every", type=int, default=5000)
    p.add_argument("--plot-every", type=int, default=10)
    p.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable RDKit SMILES augmentation.",
    )
    p.add_argument(
        "--checkpoint-name",
        default="checkpoint_pretrain.pth",
        help="Filename for the final pretrained model.",
    )
    return p.parse_args()


def unify_star_wildcards(smi: str) -> str:
    """Rewrite every bare ``*`` as ``[*]``."""
    return re.sub(r"(?<!\[)\*(?!\])", "[*]", smi)


def build_role_prefixed_pi1m(
    pi1m_csv: str, output_path: Path, delimiter: str = "\t"
) -> Path:
    """Emit two role-prefixed copies of each PI1M SMILES (``^`` and ``~``)."""
    if output_path.exists():
        print(f"Role-prefixed PI1M file already at {output_path}; reusing.")
        return output_path
    print(f"Reading PI1M corpus from {pi1m_csv} ...")
    df = pd.read_csv(pi1m_csv, delimiter=delimiter, header=None)
    smiles = df[0].dropna().unique().tolist()
    rows: list[str] = []
    for smi in tqdm(smiles, desc="Unify [*] / add role prefixes"):
        u = unify_star_wildcards(smi)
        rows.append(f"^{u}")
        rows.append(f"~{u}")
    pd.DataFrame(rows).to_csv(output_path, index=False, header=False)
    print(f"Wrote {len(rows)} role-prefixed lines to {output_path}.")
    return output_path


def build_token_vocabulary(
    pi1m_csv: str,
    aem_csv: str,
    output_path: Path,
    smi_h: str,
    smi_p: str,
    pi1m_delimiter: str = "\t",
) -> list[str]:
    """Collect the alphabet from PI1M + AEM SMILES + role/end markers."""
    if output_path.exists():
        with open(output_path, "r") as f:
            tokens = json.load(f)
        print(f"Reusing existing token vocabulary ({len(tokens)} chars).")
        return tokens

    print("Building character-level vocabulary ...")
    pi1m = pd.read_csv(pi1m_csv, delimiter=pi1m_delimiter, header=None)
    pi1m_smiles = pi1m[0].dropna().unique().tolist()

    aem = pd.read_csv(aem_csv)
    aem_smiles = (
        pd.concat([aem[smi_h], aem[smi_p]]).dropna().unique().tolist()
    )

    chars = set("<>^~")
    for smi in pi1m_smiles + aem_smiles:
        chars.update(list(str(smi)))

    base = ["<", ">"]
    rest = sorted(c for c in chars if c not in base)
    tokens = base + rest
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(tokens, f)
    print(f"Vocabulary size = {len(tokens)} written to {output_path}.")
    return tokens


# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    use_cuda = torch.cuda.is_available()
    print(f"CUDA available: {use_cuda}")

    role_csv = output_dir / "PI1M_with_roles_unified_star.csv"
    build_role_prefixed_pi1m(
        args.pi1m_csv, role_csv, delimiter=args.pi1m_delimiter
    )

    tokens_path = output_dir / "tokens.json"
    tokens = build_token_vocabulary(
        args.pi1m_csv,
        args.aem_csv,
        tokens_path,
        smi_h=args.smiles_hydrophilic_col,
        smi_p=args.smiles_hydrophobic_col,
        pi1m_delimiter=args.pi1m_delimiter,
    )

    print("Loading role-aware generator dataset ...")
    gen_data = GeneratorData(
        training_data_path=str(role_csv),
        delimiter=",",
        cols_to_read=[0],
        keep_header=False,
        tokens=tokens,
        max_len=args.max_len,
        use_cuda=use_cuda,
    )
    print(f"Number of training SMILES: {gen_data.file_len}")
    print(f"Alphabet size: {gen_data.n_characters}")

    print(
        "Initialising stack-augmented GRU "
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

    print(
        f"Pretraining for {args.n_iterations:,} iterations "
        f"(augment={'no' if args.no_augment else 'yes'}) ..."
    )
    losses = model.fit(
        gen_data,
        n_iterations=args.n_iterations,
        print_every=args.print_every,
        plot_every=args.plot_every,
        augment=(not args.no_augment),
    )

    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title("Generator pretraining loss")
    plt.xlabel(f"x{args.plot_every} iterations")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(output_dir / "pretrain_loss.png", dpi=150)
    plt.close()
    print(f"Loss curve saved to {output_dir / 'pretrain_loss.png'}.")

    ckpt_path = output_dir / args.checkpoint_name
    model.save_model(str(ckpt_path))
    print(f"Pretrained model saved to {ckpt_path}.")

    # Quick sanity check
    print("\n--- Generation samples after pretraining ---")
    print(" ^ sample :", model.evaluate(gen_data, prime_str="^"))
    print(" ~ sample :", model.evaluate(gen_data, prime_str="~"))


if __name__ == "__main__":
    main()
