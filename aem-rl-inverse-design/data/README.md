# Datasets

The training data used in the paper are not included in this repository.
The curated AEM datasets are available from the corresponding authors on
reasonable request. The PI1M corpus used for generator pretraining is
publicly distributed by Ma & Luo (Macromolecules 2020, 53, 9621-9633).

This page documents the file layout and column conventions that the
training scripts expect.

## 1. PI1M corpus (generator pretraining)

A plain-text or tab-separated file with one polymer SMILES per row and no
header. Connection sites may appear as either ``*`` or ``[*]``; the
pretrain script unifies them to ``[*]``.

Path passed via ``--pi1m-csv`` (default delimiter is ``\t``).

## 2. Curated AEM CSV (generator fine-tuning + multitask predictor)

Comma-separated file with at least the following columns:

| Column                | Type   | Description                                      |
|-----------------------|--------|--------------------------------------------------|
| ``Hydrophilic``       | str    | SMILES of the hydrophilic motif (must contain ``[N+]``) |
| ``Hydrophobic``       | str    | SMILES of the hydrophobic motif (no positive charge)    |
| ``HydrophilicFrac``   | float  | Hydrophilic-motif molar fraction                 |
| ``IEC``               | float  | Ion-exchange capacity (mmol/g)                   |
| ``Temperature``       | float  | Conductivity test temperature (degC)             |
| ``RH``                | float  | Relative humidity during the test (%)            |
| ``PolymerArchitecture``| str   | One of ``Block`` / ``Homo`` / ``Random`` / ...    |
| ``Conductivity``      | float  | Measured hydroxide conductivity (mS/cm)          |
| ``SR``                | float  | In-plane linear swelling ratio (%)               |

Records may have only one of ``Conductivity`` or ``SR`` populated; the
masked multitask MSE loss handles the missing entries automatically.

Optional auxiliary continuous columns may be present; they are simply
ignored by this release because the predictor only outputs Conductivity
and SR.

Path passed via ``--aem-csv``.

## 3. Paired alkaline-aging CSV (stability classifier)

Comma-separated file with one row per measured conductivity, including
both before-aging (``time(h) = 0``) and after-aging (``time(h) > 0``)
rows. The trainer pairs them automatically.

Required columns:

| Column                       | Type   | Description                              |
|------------------------------|--------|------------------------------------------|
| ``Hydrophilic``              | str    | Hydrophilic-motif SMILES                 |
| ``Hydrophobic``              | str    | Hydrophobic-motif SMILES                 |
| ``Cond``                     | float  | Hydroxide conductivity (mS/cm)           |
| ``time(h)``                  | float  | Aging time (0 for fresh)                 |
| ``Hydrophilic_Fraction``     | float  | Hydrophilic-motif molar fraction         |
| ``solvent_NaOH (M)``         | float  | NaOH concentration (M)                   |
| ``solvent_KOH (M)``          | float  | KOH concentration (M)                    |
| ``RH (%)``                   | float  | Relative humidity (%)                    |
| ``theor_IEC (meq/g)``        | float  | Theoretical IEC (mmol/g)                 |
| ``stability_test_temp (C)``  | float  | Aging temperature (degC)                 |
| ``prop_test_temp (C)``       | float  | Hydroxide-conductivity test temperature (degC) |

Pairing rule (Section 2.3.2 of the paper):

- For each combination of (polymer identity, all experimental conditions
  except aging time), the fresh conductivity ``sigma_init`` is the
  median of the rows with ``time(h) = 0`` and the degraded conductivity
  ``sigma_deg`` is the minimum of the rows with ``time(h) > 0``.
- A pair must have a single distinct value of ``time(h) > 0``; pairs
  violating this rule cause the script to abort.
- ``retention = sigma_deg / sigma_init`` is computed and the pair is
  labelled Pass if ``retention >= 0.80``, Fail otherwise.

Path passed via ``--data-csv``.

## 4. polyBERT checkpoint

A local directory containing the polyBERT model files
(``config.json``, ``pytorch_model.bin``, ``tokenizer.json``, etc.).
Download the official polyBERT release from Hugging Face
(``kuelumbus/polyBERT``) and pass the local directory via
``--polybert-path``.

## 5. SA-score fragment table

Place ``fpscores.pkl.gz`` from the official RDKit Contrib release in the
same directory as ``src/aem_rl/sa_score.py`` (or in the working
directory). Without this file the SA-score module will raise
``FileNotFoundError`` on first use.
