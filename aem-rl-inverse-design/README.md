# AEM-RL: Multi-objective Reinforcement-Learning-Enabled Inverse Design of Anion Exchange Membranes

This repository contains the code accompanying the paper
**"Multi-objective Reinforcement-Learning-Enabled Inverse Design of Anion
Exchange Membranes"** (Journal of Membrane Science).

The workflow couples a stack-augmented motif generator, a polyBERT-based
multitask property predictor for hydroxide conductivity and SR, and a
CatBoost alkaline-stability classifier through a reward-guided
reinforcement-learning loop. The output of the workflow is a prioritised
set of candidate AEM chemical structures.

The high-level structure follows Sections 2.2-2.4 of the paper:

```
PI1M corpus  ─►  Pretrain generator  (2,000,000 steps)
                     │
                     ▼
AEM SMILES   ─►  Fine-tune generator (500,000 steps)
                     │            ▲
                     ▼            │
            Reward-guided RL ─────┘  ◄── multitask predictor (sigma, SR)
                     │                ◄── alkaline-stability filter (post-RL)
                     ▼
            Candidate AEM motif pairs
```

## Repository layout

```
aem-rl-inverse-design/
├── LICENSE                       # MIT
├── README.md
├── requirements.txt
├── configs/                      # JSON configuration templates
├── data/
│   └── README.md                 # data availability statement
├── scripts/                      # training entry points
│   ├── train_generator_pretrain.py
│   ├── train_generator_finetune.py
│   ├── train_predictor.py
│   ├── train_stability_classifier.py
│   └── run_rl.py
└── src/
    └── aem_rl/                   # importable Python package
        ├── data.py
        ├── stack_rnn.py          # generator architecture
        ├── smiles_enumerator.py  # SMILES augmentation
        ├── reinforcement.py      # REINFORCE driver
        ├── reward.py             # Eq. 5 reward + family-specific D(S_T)
        ├── predictor.py          # polyBERT multitask predictor
        ├── stability_classifier.py
        ├── sa_score.py           # Ertl-Schuffenhauer SA score
        └── utils.py
```

## Installation

The code requires Python >= 3.9 and a CUDA-capable GPU is recommended for
generator and predictor training. CatBoost training runs on CPU.

```bash
# Clone the repository
git clone https://github.com/Valarmorghulish/aem-rl-inverse-design.git
cd aem-rl-inverse-design

# Create an isolated environment (recommended)
python -m venv .venv
source .venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt
```

Two external assets are required at runtime:

1. **polyBERT checkpoint** &mdash; download the polyBERT model from the
   official release and place its ``config.json``,
   ``pytorch_model.bin`` and tokenizer files into a local directory
   (referred to below as ``local_models/polyBERT``). The Hugging Face
   identifier is ``kuelumbus/polyBERT``.
2. **SA-score fragment table** &mdash; place ``fpscores.pkl.gz`` from the
   official RDKit Contrib release alongside ``src/aem_rl/sa_score.py``.

## Data availability

The curated AEM datasets used in the paper are available from the
corresponding authors on reasonable request. See ``data/README.md`` for
the expected file layout and column names. The PI1M corpus used for
pretraining is publicly released by Ma & Luo
(*Macromolecules* 2020, 53, 9621-9633).

## End-to-end training pipeline

The pipeline runs in five sequential stages.

### 1. Pretrain the generator on PI1M (2,000,000 steps)

```bash
python scripts/train_generator_pretrain.py \
    --pi1m-csv data/PI1M.csv \
    --aem-csv data/Conductivity_data_merged_processed.csv \
    --output-dir checkpoints/generator \
    --n-iterations 2000000
```

This unifies the polymerisation connection sites to ``[*]``, builds the
character-level vocabulary from the union of PI1M and the curated AEM
dataset, prepends the role markers ``^`` (hydrophilic) and ``~``
(hydrophobic) to every PI1M entry, and trains the stack-augmented GRU
(hidden = 1500, stack 1500 x 200) with SMILES augmentation enabled.

Outputs:
- ``checkpoints/generator/tokens.json``
- ``checkpoints/generator/PI1M_with_roles_unified_star.csv``
- ``checkpoints/generator/checkpoint_pretrain.pth``
- ``checkpoints/generator/pretrain_loss.png``

### 2. Fine-tune the generator on AEM SMILES (500,000 steps)

```bash
python scripts/train_generator_finetune.py \
    --aem-csv data/Conductivity_data_merged_processed.csv \
    --pretrained checkpoints/generator/checkpoint_pretrain.pth \
    --tokens checkpoints/generator/tokens.json \
    --output-dir checkpoints/generator \
    --n-iterations 500000 \
    --lr 5e-5
```

Outputs:
- ``checkpoints/generator/aem_finetune_unique_with_roles.csv``
- ``checkpoints/generator/checkpoint_finetune.pth``
- ``checkpoints/generator/finetune_loss.png``

### 3. Train the multitask conductivity / SR predictor

```bash
python scripts/train_predictor.py \
    --aem-csv data/Conductivity_data_merged_processed.csv \
    --polybert-path local_models/polyBERT \
    --output-dir checkpoints/predictor
```

The predictor uses the polyBERT encoder shared between the hydrophilic
and hydrophobic motifs, followed by a shared regression head and two
task-specific output layers (Conductivity, SR). Training proceeds in
two steps (Section 2.3.1 of the paper):

- Self-supervised MLM step on the AEM SMILES corpus to specialise the
  encoder (60 epochs, 8x4 effective batch, lr = 5e-5, last 2 encoder
  layers + embeddings unfrozen).
- Five-fold stratified cross-validation supervised fine-tuning with a
  masked multitask MSE loss (80 epochs, 8x2 effective batch, lr = 2e-5,
  last 4 encoder layers + embeddings + heads unfrozen).

Outputs:
- ``checkpoints/predictor/mlm_finetune/``
- ``checkpoints/predictor/reg_fold_{1..5}/``
- ``checkpoints/predictor/ensemble/``
- ``checkpoints/predictor/kfold_validation_loss.png``

### 4. Train the alkaline-stability classifier

```bash
python scripts/train_stability_classifier.py \
    --data-csv data/stability4_final.csv \
    --work-dir checkpoints/stability
```

The classifier pairs fresh (``time(h) = 0``) and degraded
(``time(h) > 0``) records sharing the same polymer identity and
operational conditions, defines a Pass / Fail label by the conductivity
retention threshold of 0.80, builds features from the operational
variables and 2D Mordred descriptors of the two motifs, and fits a
CatBoost classifier under a polymer-grouped train / test split. An
Optuna-TPE Bayesian search around the fixed-parameter baseline is run
by default.

Outputs:
- ``checkpoints/stability/baseline/`` (CatBoost baseline model and metrics)
- ``checkpoints/stability/tuned/`` (Optuna-tuned model and metrics)
- ``checkpoints/stability/exp_mode_fill.pkl``
- ``checkpoints/stability/desc_filter.pkl``
- ``checkpoints/stability/feature_median.pkl``
- ``checkpoints/stability/feature_names.pkl``
- ``checkpoints/stability/paired_dataset.csv``

### 5. RL stage with a chosen AEM-family setting

```bash
python scripts/run_rl.py \
    --finetuned checkpoints/generator/checkpoint_finetune.pth \
    --tokens checkpoints/generator/tokens.json \
    --aem-finetune-data checkpoints/generator/aem_finetune_unique_with_roles.csv \
    --predictor-ensemble checkpoints/predictor/ensemble \
    --polybert-path local_models/polyBERT \
    --family PAEK \
    --output-dir checkpoints/rl/PAEK
```

The RL loop runs 50 iterations, each with 15 policy-gradient updates of
batch 15 (Section 2.4 of the paper), saves the model every 10 iterations
and records the predicted-property distributions every 10 iterations.

Outputs:
- ``checkpoints/rl/<family>/checkpoint_RL_iter_{10,20,...}.pth``
- ``checkpoints/rl/<family>/checkpoint_RL_final.pth``
- ``checkpoints/rl/<family>/history.json``
- ``checkpoints/rl/<family>/checkpoint_samples.json``
- ``checkpoints/rl/<family>/training_curves.png``

## Reward function

The total reward implements Eq. 5 of the paper:

```
R_total(S_T) = I_valid * [ D(S_T) * G_cond(sigma_hat) * G_SR(SR_hat)
                           + alpha * D(S_T) ]
              * B_div(S_T) * B_elite(S_T)
```

with

- ``I_valid``: indicator returned by the validity / composition checks
  (RDKit parsability, two ``[*]`` markers per motif, cation count in
  [1, 3] on the hydrophilic motif, neutral hydrophobic motif, MW >= 80
  on both motifs, hydrophilic-fraction in [0.05, 0.95] for the target
  IEC, side-chain caps);
- ``G_cond``: piecewise-monotone gate with breakpoints 50 and 110
  mS/cm and slopes 0.45 / 0.90;
- ``G_SR``: threshold-shaped gate with breakpoints 28 % and 32 %;
- ``D(S_T)``: structural-preference term combining a cation-class score
  with a hydrophobic-complexity score, modulated by an AEM-family
  constraint (default for this release: PAEK, requiring at least one
  aryl ether and one aryl ketone in the motif pair);
- ``alpha = 0.20``;
- ``B_div``: 1.2 for unique motif pairs, 1.0 for repeats, with a
  rolling cache of 1000 pairs;
- ``B_elite``: 1.5 if the predicted hydroxide conductivity exceeds
  100 mS/cm and the predicted SR is below 30 %, otherwise 1.0.

The default values reproduce the runs in the paper. Override individual
values at the call site by passing a custom ``RewardConfig`` instance to
``RewardFunction``.

To add a new AEM family, subclass ``aem_rl.reward.FamilyConstraint``,
register it in ``aem_rl.reward.build_reward_fn`` and pass the new family
name through ``--family``.

## Reproducibility notes

- The fixed condition vector used during RL prediction is
  ``Temperature = 80 degC``, ``RH = 100%``, ``IEC = 2.0 mmol/g``,
  ``HydrophilicFrac`` solved from Eq. 4, and ``PolymerArchitecture``
  set to ``Block`` if the two motifs differ and ``Homo`` otherwise.
- Hydroxide conductivity is trained on the ``log1p`` scale and inverted
  with ``expm1`` at inference, following Section 2.3.1.
- All seeds are fixed to 42 by default (CLI ``--random-state``).

## Citation

Please cite the accompanying paper if you use this code:

```
@article{zhao2026aemrl,
  title={Multi-objective Reinforcement-Learning-Enabled Inverse Design
         of Anion Exchange Membranes},
  author={Zhao, Shengzhan and Feng, Wenzhuo and Liu, Lunyang and Li, Hongfei},
  journal={Journal of Membrane Science},
  year={2026}
}
```

The generator and the policy-gradient training driver follow ReLeaSE:

```
@article{popova2018release,
  title={Deep reinforcement learning for de novo drug design},
  author={Popova, Mariya and Isayev, Olexandr and Tropsha, Alexander},
  journal={Science Advances},
  volume={4}, number={7}, pages={eaap7885}, year={2018}
}
```

The differentiable stack augmentation follows Joulin & Mikolov, NeurIPS 2015.
The polyBERT encoder follows Kuenneth & Ramprasad, *Nat. Commun.* 14, 4099
(2023). The synthetic-accessibility score follows Ertl & Schuffenhauer,
*J. Cheminform.* 1, 8 (2009).

## License

MIT (see ``LICENSE``).
