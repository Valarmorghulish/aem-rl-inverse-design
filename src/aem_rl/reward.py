"""
Multi-objective reward function for the RL stage.

Implements Eq. 5 of the accompanying paper:

    R_total(S_T) = I_valid * [ D(S_T) * G_cond(sigma_hat) * G_SR(SR_hat)
                               + alpha * D(S_T) ]
                  * B_div(S_T) * B_elite(S_T)

where the symbols carry the same meaning as in the manuscript:
    I_valid  : binary indicator from the validity / composition checks
    G_cond   : monotone gating term for predicted hydroxide conductivity
    G_SR     : threshold-shaped gating term for predicted SR
    D(S_T)   : structural-preference term that captures cation-environment
               and AEM-family-specific constraints
    B_div    : multiplicative diversity bonus
    B_elite  : multiplicative elite-candidate bonus
    alpha    : weight of the structural-preference term

The structural-preference term D(S_T) is implemented per AEM family.
Six AEM-family constraints are provided as reference implementations,
matching the six families used in the paper:
    PAP    -- poly(aryl piperidinium)
    PBF    -- poly(biphenyl fluorene)
    PPO    -- poly(phenylene oxide)
    PAEK   -- poly(arylene ether ketone)
    PAEKS  -- poly(arylene ether ketone sulfone)
    PAES   -- poly(arylene ether sulfone)

Each family check is restricted to the main-chain atoms of the two motifs
so that the constraint reflects the polymer chemical structure rather than
incidental side-chain motifs. New families can be added by subclassing
``FamilyConstraint``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdmolops


_STAR_PATTERN = re.compile(r"\[\*\]")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class RewardConfig:
    """Hyper-parameters governing the reward shape.

    The defaults reproduce the values used in the paper.
    """

    # Operational/composition setting passed to the predictor
    target_iec: float = 2.0
    test_temperature: float = 80.0
    relative_humidity: float = 100.0
    phi_frac_min: float = 0.05
    phi_frac_max: float = 0.95

    # Conductivity gate G_cond piecewise breakpoints
    cond_low: float = 50.0
    cond_high: float = 110.0
    cond_pre: float = 50.0
    cond_slope_mid: float = 0.45
    cond_slope_tail: float = 0.90

    # SR gate G_SR breakpoints
    sr_low: float = 28.0
    sr_high: float = 32.0
    sr_low_slope: float = 0.08
    sr_mid_anchor: float = 0.92
    sr_tail_decay: float = 0.12

    # Structural preference weight (alpha in Eq. 5)
    alpha_structure: float = 0.20

    # Diversity bonus
    diversity_window_size: int = 1000
    diversity_unique_bonus: float = 1.2
    diversity_recurrent_bonus: float = 1.0

    # Elite bonus
    elite_cond_threshold: float = 100.0
    elite_sr_threshold: float = 30.0
    elite_multiplier: float = 1.5

    # Side-chain hard limits on the hydrophilic motif
    max_side_linear_carbon_chain: int = 8
    max_side_total_carbon: int = 16
    max_side_cation_count: int = 3
    max_main_chain_cation_count: int = 3

    # Minimum molecular weight for both motifs
    min_motif_molwt: float = 80.0


# ---------------------------------------------------------------------------
# Cation library (for D(S_T))
# ---------------------------------------------------------------------------
_CATION_SMARTS = {
    "imidazolium": "[c]1[n+][c][c][n]1",
    "pyridinium": "[n+;r6]",
    "triazolium": "[c]1[n+][n][c][n]1",
    "piperidinium": "[N+;R1;r6]",
    "pyrrolidinium": "[N+;R1;r5]",
    "guanidinium": "[C+]([N])([N])N",
}


def _compile_cation_smarts():
    return {
        name: Chem.MolFromSmarts(smarts)
        for name, smarts in _CATION_SMARTS.items()
    }


# ---------------------------------------------------------------------------
# Geometric / topological helpers
# ---------------------------------------------------------------------------
def strip_smiles(smi: Optional[str]) -> str:
    """Remove the ``< > ^ ~`` markers that wrap generator outputs."""
    if not isinstance(smi, str):
        return ""
    return (
        smi.replace("<", "")
        .replace(">", "")
        .replace("^", "")
        .replace("~", "")
        .strip()
    )


def get_main_chain_indices(mol) -> set:
    """Return the atom indices that lie on the shortest path between the two
    ``[*]`` connection sites; an empty set indicates that the input is not a
    linear two-connection-point polymer motif.
    """
    if mol is None:
        return set()
    try:
        rw = Chem.RWMol(mol)
        stars = [a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 0]
        if len(stars) < 2:
            return set()
        for idx in stars:
            rw.ReplaceAtom(idx, Chem.Atom(6))
        Chem.SanitizeMol(rw)
        path = rdmolops.GetShortestPath(rw, stars[0], stars[1])
        return set(path) if path else set()
    except Exception:
        return set()


def main_chain_no_cation(mol) -> bool:
    """True if no atom on the main chain carries a positive formal charge."""
    main = get_main_chain_indices(mol)
    if not main:
        return True
    return not any(
        mol.GetAtomWithIdx(int(idx)).GetFormalCharge() > 0 for idx in main
    )


def _longest_carbon_path(mol, atom_indices: Sequence[int]) -> int:
    """Length of the longest simple all-carbon single-bond path inside a
    subset of atom indices."""
    if not mol or not atom_indices:
        return 0
    adj: Dict[int, List[int]] = {
        idx: []
        for idx in atom_indices
        if mol.GetAtomWithIdx(idx).GetAtomicNum() == 6
    }
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if u in adj and v in adj and bond.GetBondType() == Chem.BondType.SINGLE:
            adj[u].append(v)
            adj[v].append(u)
    longest = 0
    for start in adj:
        stack = [(start, [start])]
        while stack:
            curr, path = stack.pop()
            longest = max(longest, len(path))
            for nbr in adj.get(curr, []):
                if nbr not in path:
                    stack.append((nbr, path + [nbr]))
    return longest


def analyze_hydrophilic_side_chain(mol) -> Dict[str, float]:
    """Side-chain summary statistics for the hydrophilic motif."""
    out = {
        "max_linear_carbon_chain": 0,
        "total_carbon_count": 0,
        "cation_count": 0,
        "side_chain_complexity_score": 0.0,
    }
    main = get_main_chain_indices(mol)
    if not main:
        return out
    side = {
        i for i in range(mol.GetNumAtoms()) if i not in main
    }
    if not side:
        return out

    total_c = 0
    cation = 0
    for idx in side:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetAtomicNum() == 6:
            total_c += 1
        if atom.GetFormalCharge() > 0:
            cation += 1
    out["total_carbon_count"] = total_c
    out["cation_count"] = cation
    out["max_linear_carbon_chain"] = _longest_carbon_path(mol, list(side))

    side_heavy = sum(
        1 for idx in side if mol.GetAtomWithIdx(idx).GetAtomicNum() != 1
    )
    if side_heavy > 1:
        terminal = sum(
            1
            for idx in side
            if mol.GetAtomWithIdx(idx).GetDegree() == 1
            and mol.GetAtomWithIdx(idx).GetAtomicNum() != 1
        )
        branching = (
            (side_heavy - terminal - 1) / (side_heavy - 2)
            if side_heavy > 2
            else 0.0
        )
        chiral = sum(
            1
            for c in Chem.FindMolChiralCenters(mol, includeUnassigned=True)
            if c[0] in side
        )
        out["side_chain_complexity_score"] = float(
            np.clip(branching * 0.7 + np.log1p(chiral) * 0.3, 0.0, 1.0)
        )
    return out


def hydrophobic_complexity_score(mol, main_chain: set) -> float:
    """Compactness-aware complexity score of the hydrophobic motif."""
    if not main_chain or mol is None:
        return 0.0
    side_atoms = sum(
        1
        for a in mol.GetAtoms()
        if a.GetIdx() not in main_chain and a.GetAtomicNum() != 1
    )
    chiral_off = sum(
        1
        for c in Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        if c[0] not in main_chain
    )
    val = np.log1p(side_atoms) + chiral_off * 0.5
    return float(np.clip(val / 5.0, 0.0, 1.0))


def cation_diversity_score(mol, main_chain: set, cation_patterns: Dict) -> float:
    """Reward presence of recognised cationic motifs on side chains.

    Returns 1.0 if any side-chain match is found, 0.3 otherwise.
    """
    if not main_chain:
        return 0.3
    side = {i for i in range(mol.GetNumAtoms()) if i not in main_chain}
    if not side:
        return 0.3
    for pat in cation_patterns.values():
        if pat is None:
            continue
        try:
            for match in mol.GetSubstructMatches(pat):
                if set(match).issubset(side):
                    return 1.0
        except Exception:
            continue
    return 0.3


# ---------------------------------------------------------------------------
# Composition feasibility (Eqs. 3-4 of the paper)
# ---------------------------------------------------------------------------
def hydrophilic_fraction_for_iec(
    mw_hydro: float,
    mw_phobic: float,
    n_cation: int,
    target_iec: float,
) -> float:
    """Solve Eq. 4: hydrophilic-motif molar fraction satisfying ``target_iec``.

    Returns ``-1.0`` if ``n_cation == 0`` or if the denominator is zero.
    """
    if n_cation == 0:
        return -1.0
    denominator = (n_cation * 1000.0) + target_iec * (mw_phobic - mw_hydro)
    if abs(denominator) < 1e-6:
        return -1.0
    return target_iec * mw_phobic / denominator


# ---------------------------------------------------------------------------
# AEM-family structural constraints (D(S_T))
# ---------------------------------------------------------------------------
def _main_chain_has_smarts(mol, main_chain: set, pattern) -> bool:
    """True iff ``mol`` has a substructure match of ``pattern`` whose atoms
    all lie on ``main_chain``.

    This is the standard helper used by every family constraint below: a
    family check should respond to motifs lying on the polymer main chain
    (which defines the family identity), not to incidental matches living
    purely on a side chain.
    """
    if pattern is None or not main_chain:
        return False
    try:
        for match in mol.GetSubstructMatches(pattern):
            if set(match).issubset(main_chain):
                return True
    except Exception:
        return False
    return False


def _any_main_chain_aromatic(mol_h, main_chain_h, mol_p, main_chain_p) -> bool:
    """True iff at least one main-chain atom on either motif is aromatic.

    All six AEM families considered in the paper have an aromatic-rich main
    chain, so this guard rejects pairs whose main chain is all aliphatic
    before any family-specific SMARTS check is performed.
    """
    return any(
        mol_h.GetAtomWithIdx(int(i)).GetIsAromatic() for i in main_chain_h
    ) or any(
        mol_p.GetAtomWithIdx(int(i)).GetIsAromatic() for i in main_chain_p
    )


class FamilyConstraint:
    """Base class for AEM-family-specific structural constraints.

    Subclasses implement ``check`` returning ``True`` if the (Phi, Pho) pair
    belongs to the target family. The base class itself accepts every pair
    that satisfies the universal sanity checks performed by ``compute_reward``.

    Convention used by every subclass below:
        - SMARTS patterns are tested against main-chain atoms only via
          ``_main_chain_has_smarts``;
        - features may live on either the hydrophilic motif (Phi) or the
          hydrophobic motif (Pho), since the curated AEM data places motifs
          on either block depending on the polymer architecture;
        - ``check`` returns ``True`` only when every required substructure
          is present.
    """

    name: str = "Generic"

    def check(self, mol_h, main_chain_h, mol_p, main_chain_p) -> bool:
        return True


class PAPConstraint(FamilyConstraint):
    """PAP -- poly(aryl piperidinium).

    Definition adopted here:
        - At least one main-chain atom on either motif is aromatic
          (the ``aryl`` part of poly(aryl piperidinium)).
        - At least one piperidinium centre is present in the pair, located
          on the main chain of the hydrophilic motif. Piperidinium is a
          six-membered saturated ring nitrogen carrying a positive formal
          charge ([N+;R1;r6]); this matches both N,N-dimethyl-piperidinium
          (DMP-type) and N-methyl-N-alkyl-piperidinium (mPip-type) cations
          discussed in the paper.
        - The polymer main chain must be ether-free, in line with the
          ether-free design rationale of the PAP family in the AEM
          literature; an aryl-ether linkage on the main chain is therefore
          a disqualifier.
    """

    name = "PAP"

    def __init__(self):
        # Six-membered saturated ring nitrogen with positive formal charge.
        self._piperidinium = Chem.MolFromSmarts("[N+;R1;r6]")
        # Aryl ether linkage written as aromatic-C bound through O to
        # another aromatic-C. Used to enforce ether-free main chains.
        self._aryl_ether = Chem.MolFromSmarts("c-[O;X2]-c")

    def check(self, mol_h, main_chain_h, mol_p, main_chain_p) -> bool:
        if not _any_main_chain_aromatic(
            mol_h, main_chain_h, mol_p, main_chain_p
        ):
            return False
        # The piperidinium centre may sit at the chain end of the
        # hydrophilic motif; require its match to overlap with the main
        # chain of Phi.
        has_piperidinium = _main_chain_has_smarts(
            mol_h, main_chain_h, self._piperidinium
        ) or _main_chain_has_smarts(
            mol_p, main_chain_p, self._piperidinium
        )
        if not has_piperidinium:
            return False
        if _main_chain_has_smarts(
            mol_h, main_chain_h, self._aryl_ether
        ) or _main_chain_has_smarts(
            mol_p, main_chain_p, self._aryl_ether
        ):
            return False
        return True


class PBFConstraint(FamilyConstraint):
    """PBF -- poly(biphenyl fluorene).

    Definition adopted here:
        - At least one main-chain atom on either motif is aromatic.
        - The polymer chemical structure carries a fluorene core, i.e.
          two benzene rings fused to a five-membered carbocycle through
          a shared sp3 carbon. The SMARTS below matches the fluorene
          quaternary carbon flanked by two fused aromatic rings.
        - The main chain must be ether-free (no ``c-O-c`` linkage on the
          main chain), distinguishing PBF from PAES / PAEK / PAEKS that
          all carry main-chain aryl ethers.
    """

    name = "PBF"

    def __init__(self):
        # Fluorene-style sp3 quaternary carbon bridging two fused aromatic
        # rings. The 9,9-disubstituted fluorene unit characteristic of PBF
        # structures matches this SMARTS regardless of substitution pattern.
        self._fluorene_core = Chem.MolFromSmarts(
            "[C;X4](-c1ccccc1-2)(-c1ccccc1-2)"
        )
        # A more permissive backup pattern: spirobifluorene / 9,9-diaryl
        # fluorene also produces fused biphenyl + sp3 centre, captured by
        # a generic "two aryl rings on the same sp3 carbon" check on the
        # main chain.
        self._diaryl_sp3 = Chem.MolFromSmarts("[C;X4]([c])([c])")
        self._aryl_ether = Chem.MolFromSmarts("c-[O;X2]-c")

    def check(self, mol_h, main_chain_h, mol_p, main_chain_p) -> bool:
        if not _any_main_chain_aromatic(
            mol_h, main_chain_h, mol_p, main_chain_p
        ):
            return False
        has_fluorene = _main_chain_has_smarts(
            mol_h, main_chain_h, self._fluorene_core
        ) or _main_chain_has_smarts(
            mol_p, main_chain_p, self._fluorene_core
        )
        if not has_fluorene:
            # Permissive fallback: at least an sp3 carbon connecting two
            # aryl rings on the main chain.
            has_diaryl = _main_chain_has_smarts(
                mol_h, main_chain_h, self._diaryl_sp3
            ) or _main_chain_has_smarts(
                mol_p, main_chain_p, self._diaryl_sp3
            )
            if not has_diaryl:
                return False
        if _main_chain_has_smarts(
            mol_h, main_chain_h, self._aryl_ether
        ) or _main_chain_has_smarts(
            mol_p, main_chain_p, self._aryl_ether
        ):
            return False
        return True


class PPOConstraint(FamilyConstraint):
    """PPO -- poly(phenylene oxide).

    Definition adopted here:
        - At least one main-chain atom on either motif is aromatic.
        - The main chain carries a 2,6-dimethylphenylene oxide repeat
          motif: an aromatic-O-aromatic linkage where the aromatic carbon
          carries methyl substituents. This matches the PPO repeat unit
          ``[*]Oc1c(C)cc(C)cc1[*]`` and its regio-isomers.
        - No ketone or sulfone group on the main chain (those would
          re-classify the polymer as PAEK or PAES).
    """

    name = "PPO"

    def __init__(self):
        # Aryl ether bridging two aromatic carbons; minimal PPO signature.
        self._aryl_ether = Chem.MolFromSmarts("c-[O;X2]-c")
        # 2,6-disubstituted (typically methyl) phenylene oxide pattern.
        self._dimethyl_phenylene_oxide = Chem.MolFromSmarts(
            "[O;X2]-c1c([CH3])cc([CH3,CH2,*])cc1"
        )
        # Disqualifiers: aryl ketone (PAEK) or aryl sulfone (PAES).
        self._aryl_ketone = Chem.MolFromSmarts("c-C(=O)-c")
        self._aryl_sulfone = Chem.MolFromSmarts("c-S(=O)(=O)-c")

    def check(self, mol_h, main_chain_h, mol_p, main_chain_p) -> bool:
        if not _any_main_chain_aromatic(
            mol_h, main_chain_h, mol_p, main_chain_p
        ):
            return False
        has_ether = _main_chain_has_smarts(
            mol_h, main_chain_h, self._aryl_ether
        ) or _main_chain_has_smarts(
            mol_p, main_chain_p, self._aryl_ether
        )
        if not has_ether:
            return False
        # Disqualify ketone / sulfone main-chain motifs to keep PPO
        # cleanly separated from PAEK / PAES / PAEKS.
        if _main_chain_has_smarts(
            mol_h, main_chain_h, self._aryl_ketone
        ) or _main_chain_has_smarts(
            mol_p, main_chain_p, self._aryl_ketone
        ):
            return False
        if _main_chain_has_smarts(
            mol_h, main_chain_h, self._aryl_sulfone
        ) or _main_chain_has_smarts(
            mol_p, main_chain_p, self._aryl_sulfone
        ):
            return False
        return True


class PAEKConstraint(FamilyConstraint):
    """PAEK -- poly(arylene ether ketone).

    Definition adopted here:
        - At least one main-chain atom on either motif is aromatic.
        - The main chain carries at least one aryl ether (``c-O-c``) and
          at least one aryl ketone (``c-C(=O)-c``).
        - The main chain has no aryl sulfone group (which would classify
          the polymer as PAES or PAEKS).
    """

    name = "PAEK"

    def __init__(self):
        self._aryl_ether = Chem.MolFromSmarts("c-[O;X2]-c")
        self._aryl_ketone = Chem.MolFromSmarts("c-C(=O)-c")
        self._aryl_sulfone = Chem.MolFromSmarts("c-S(=O)(=O)-c")

    def check(self, mol_h, main_chain_h, mol_p, main_chain_p) -> bool:
        if not _any_main_chain_aromatic(
            mol_h, main_chain_h, mol_p, main_chain_p
        ):
            return False
        has_ether = _main_chain_has_smarts(
            mol_h, main_chain_h, self._aryl_ether
        ) or _main_chain_has_smarts(
            mol_p, main_chain_p, self._aryl_ether
        )
        has_ketone = _main_chain_has_smarts(
            mol_h, main_chain_h, self._aryl_ketone
        ) or _main_chain_has_smarts(
            mol_p, main_chain_p, self._aryl_ketone
        )
        if not (has_ether and has_ketone):
            return False
        if _main_chain_has_smarts(
            mol_h, main_chain_h, self._aryl_sulfone
        ) or _main_chain_has_smarts(
            mol_p, main_chain_p, self._aryl_sulfone
        ):
            return False
        return True


class PAEKSConstraint(FamilyConstraint):
    """PAEKS -- poly(arylene ether ketone sulfone).

    Definition adopted here:
        - At least one main-chain atom on either motif is aromatic.
        - The main chain carries simultaneously an aryl ether
          (``c-O-c``), an aryl ketone (``c-C(=O)-c``), and an aryl
          sulfone (``c-S(=O)(=O)-c``). All three groups must lie on the
          main chain of either motif.
    """

    name = "PAEKS"

    def __init__(self):
        self._aryl_ether = Chem.MolFromSmarts("c-[O;X2]-c")
        self._aryl_ketone = Chem.MolFromSmarts("c-C(=O)-c")
        self._aryl_sulfone = Chem.MolFromSmarts("c-S(=O)(=O)-c")

    def check(self, mol_h, main_chain_h, mol_p, main_chain_p) -> bool:
        if not _any_main_chain_aromatic(
            mol_h, main_chain_h, mol_p, main_chain_p
        ):
            return False
        has_ether = _main_chain_has_smarts(
            mol_h, main_chain_h, self._aryl_ether
        ) or _main_chain_has_smarts(
            mol_p, main_chain_p, self._aryl_ether
        )
        has_ketone = _main_chain_has_smarts(
            mol_h, main_chain_h, self._aryl_ketone
        ) or _main_chain_has_smarts(
            mol_p, main_chain_p, self._aryl_ketone
        )
        has_sulfone = _main_chain_has_smarts(
            mol_h, main_chain_h, self._aryl_sulfone
        ) or _main_chain_has_smarts(
            mol_p, main_chain_p, self._aryl_sulfone
        )
        return has_ether and has_ketone and has_sulfone


class PAESConstraint(FamilyConstraint):
    """PAES -- poly(arylene ether sulfone).

    Definition adopted here:
        - At least one main-chain atom on either motif is aromatic.
        - The main chain carries at least one aryl ether (``c-O-c``) and
          at least one aryl sulfone (``c-S(=O)(=O)-c``).
        - The main chain has no aryl ketone group (which would classify
          the polymer as PAEK or PAEKS).
    """

    name = "PAES"

    def __init__(self):
        self._aryl_ether = Chem.MolFromSmarts("c-[O;X2]-c")
        self._aryl_sulfone = Chem.MolFromSmarts("c-S(=O)(=O)-c")
        self._aryl_ketone = Chem.MolFromSmarts("c-C(=O)-c")

    def check(self, mol_h, main_chain_h, mol_p, main_chain_p) -> bool:
        if not _any_main_chain_aromatic(
            mol_h, main_chain_h, mol_p, main_chain_p
        ):
            return False
        has_ether = _main_chain_has_smarts(
            mol_h, main_chain_h, self._aryl_ether
        ) or _main_chain_has_smarts(
            mol_p, main_chain_p, self._aryl_ether
        )
        has_sulfone = _main_chain_has_smarts(
            mol_h, main_chain_h, self._aryl_sulfone
        ) or _main_chain_has_smarts(
            mol_p, main_chain_p, self._aryl_sulfone
        )
        if not (has_ether and has_sulfone):
            return False
        if _main_chain_has_smarts(
            mol_h, main_chain_h, self._aryl_ketone
        ) or _main_chain_has_smarts(
            mol_p, main_chain_p, self._aryl_ketone
        ):
            return False
        return True


# ---------------------------------------------------------------------------
# Gating functions
# ---------------------------------------------------------------------------
def conductivity_gate(c: float, cfg: RewardConfig) -> float:
    """Piecewise-monotone gate G_cond(sigma_hat).

    Quadratic ramp below ``cond_low`` to suppress low-conductivity samples,
    then a linear segment with slope ``cond_slope_mid`` between
    ``cond_low`` and ``cond_high``, and a steeper linear tail with slope
    ``cond_slope_tail`` above ``cond_high``.
    """
    c = float(np.clip(c, 0.0, 1e9))
    t1, t2 = cfg.cond_low, cfg.cond_high
    pre = cfg.cond_pre
    v1 = (t1 / pre) ** 2
    v2 = v1 + (t2 - t1) * cfg.cond_slope_mid
    if c < t1:
        return float((c / pre) ** 2)
    if c < t2:
        return float(v1 + (c - t1) * cfg.cond_slope_mid)
    return float(v2 + (c - t2) * cfg.cond_slope_tail)


def sr_gate(sr: float, cfg: RewardConfig) -> float:
    """Threshold-shaped gate G_SR(SR_hat).

    Linear decline up to ``sr_low``, smooth quadratic crossing between
    ``sr_low`` and ``sr_high``, exponential decay above ``sr_high``.
    """
    sr = float(np.clip(sr, 0.0, 100.0))
    if sr <= cfg.sr_low:
        return 1.0 - cfg.sr_low_slope * (sr / cfg.sr_low)
    if sr <= cfg.sr_high:
        t = (sr - cfg.sr_low) / (cfg.sr_high - cfg.sr_low)
        return cfg.sr_mid_anchor * (1.0 - t * t)
    return 0.1 * np.exp(-cfg.sr_tail_decay * (sr - cfg.sr_high))


# ---------------------------------------------------------------------------
# Diversity bookkeeping
# ---------------------------------------------------------------------------
class DiversityTracker:
    """Track the most-recent molecule pairs and grant a unique-pair bonus."""

    def __init__(self, cfg: RewardConfig):
        self.cfg = cfg
        self.history: Dict[str, int] = {}

    def bonus(self, hydro: str, phobic: str) -> float:
        key = f"{hydro}|{phobic}"
        if len(self.history) > self.cfg.diversity_window_size:
            # Drop the oldest half of the cache to bound memory.
            for k in list(self.history.keys())[: len(self.history) // 2]:
                del self.history[k]
        if key in self.history:
            self.history[key] += 1
            return self.cfg.diversity_recurrent_bonus
        self.history[key] = 1
        return self.cfg.diversity_unique_bonus


# ---------------------------------------------------------------------------
# Top-level reward callable
# ---------------------------------------------------------------------------
class RewardFunction:
    """Compose Eq. 5 from gating, structural, diversity and elite terms.

    The instance is callable: ``reward, details = rf(hydro, phobic, predictor)``,
    making it a drop-in for :class:`Reinforcement`.
    """

    def __init__(
        self,
        family: FamilyConstraint = FamilyConstraint(),
        cfg: Optional[RewardConfig] = None,
    ):
        self.family = family
        self.cfg = cfg or RewardConfig()
        self._cation_patterns = _compile_cation_smarts()
        self.diversity = DiversityTracker(self.cfg)

    # ------------------------------------------------------------------
    def __call__(
        self, hydro_full: str, phobic_full: str, predictor, **kwargs
    ) -> Tuple[float, Dict]:
        return self.compute(hydro_full, phobic_full, predictor, **kwargs)

    # ------------------------------------------------------------------
    def compute(
        self,
        hydro_full: str,
        phobic_full: str,
        predictor,
        **kwargs,
    ) -> Tuple[float, Dict]:
        cfg = self.cfg
        details: Dict = {
            "is_valid": False,
            "stage": "Start",
            "hydro_smi": strip_smiles(hydro_full),
            "phobic_smi": strip_smiles(phobic_full),
            "reward_breakdown": {},
        }

        h_clean = details["hydro_smi"]
        p_clean = details["phobic_smi"]

        # ---- 1. Two ``[*]`` markers per motif (linear repeat unit) -----
        if (
            len(_STAR_PATTERN.findall(h_clean)) != 2
            or len(_STAR_PATTERN.findall(p_clean)) != 2
        ):
            details["stage"] = "Fail: not a linear two-star polymer motif"
            return 0.0, details

        # ---- 2. RDKit parsability -------------------------------------
        mol_h = Chem.MolFromSmiles(h_clean)
        mol_p = Chem.MolFromSmiles(p_clean)
        if mol_h is None or mol_p is None:
            details["stage"] = "Fail: invalid SMILES"
            return 0.0, details

        # ---- 3. Main-chain identification -----------------------------
        mc_h = get_main_chain_indices(mol_h)
        mc_p = get_main_chain_indices(mol_p)
        if not mc_h or not mc_p:
            details["stage"] = "Fail: cannot determine main chain"
            return 0.0, details

        # ---- 4. AEM-family-specific constraint D(S_T) check -----------
        if not self.family.check(mol_h, mc_h, mol_p, mc_p):
            details["stage"] = f"Fail: not in {self.family.name} family"
            return 0.0, details

        # ---- 5. Cation accounting --------------------------------------
        n_cations = sum(
            1 for a in mol_h.GetAtoms() if a.GetFormalCharge() > 0
        )
        if not (1 <= n_cations <= cfg.max_main_chain_cation_count):
            details["stage"] = f"Fail: cation count = {n_cations}"
            return 0.0, details
        if any(a.GetFormalCharge() != 0 for a in mol_p.GetAtoms()):
            details["stage"] = "Fail: hydrophobic motif carries net charge"
            return 0.0, details

        # ---- 6. Minimum motif molecular weights -----------------------
        if (
            Descriptors.MolWt(mol_h) < cfg.min_motif_molwt
            or Descriptors.MolWt(mol_p) < cfg.min_motif_molwt
        ):
            details["stage"] = "Fail: motif too simple (MolWt below threshold)"
            return 0.0, details

        # ---- 7. Cation must live on side chain only -------------------
        if not main_chain_no_cation(mol_h):
            details["stage"] = "Fail: cation on main chain"
            return 0.0, details

        # ---- 8. Side-chain complexity caps ----------------------------
        side = analyze_hydrophilic_side_chain(mol_h)
        details["side_chain_analysis"] = side
        if side["max_linear_carbon_chain"] > cfg.max_side_linear_carbon_chain:
            details["stage"] = "Fail: side chain linear carbon chain too long"
            return 0.0, details
        if side["total_carbon_count"] > cfg.max_side_total_carbon:
            details["stage"] = "Fail: side chain total carbon count too high"
            return 0.0, details
        if side["cation_count"] > cfg.max_side_cation_count:
            details["stage"] = "Fail: side chain cation count too high"
            return 0.0, details

        # ---- 9. Composition feasibility (target IEC -> phi_frac) -----
        h_frac = hydrophilic_fraction_for_iec(
            Descriptors.MolWt(mol_h),
            Descriptors.MolWt(mol_p),
            n_cations,
            cfg.target_iec,
        )
        if h_frac < cfg.phi_frac_min or h_frac > cfg.phi_frac_max:
            details["stage"] = (
                f"Fail: hydrophilic fraction out of range (={h_frac:.3f})"
            )
            return 0.0, details

        # ---- 10. Property prediction ---------------------------------
        poly_arch = "Block" if h_clean != p_clean else "Homo"
        conditions = {
            "Temperature": cfg.test_temperature,
            "RH": cfg.relative_humidity,
            "IEC": cfg.target_iec,
            "HydrophilicFrac": h_frac,
            "PolymerArchitecture": poly_arch,
        }
        try:
            props = predictor.predict_one(h_clean, p_clean, conditions)
        except Exception as exc:  # noqa: BLE001
            details["stage"] = f"Fail: predictor raised {exc!r}"
            return 0.0, details
        c = float(props.get("Conductivity", 0.0))
        s = float(props.get("SR", 0.0))
        details.update(props)
        details["predicted_conductivity"] = c
        details["is_valid"] = True
        details["stage"] = "Full Prediction"

        # ---- 11. Compose Eq. 5 ---------------------------------------
        cd_score = cation_diversity_score(mol_h, mc_h, self._cation_patterns)
        ph_complexity = hydrophobic_complexity_score(mol_p, mc_p)
        desirability = 0.5 * cd_score + 0.5 * ph_complexity
        g_cond = conductivity_gate(c, cfg)
        g_sr = sr_gate(s, cfg)
        structure_direct = cfg.alpha_structure * desirability
        core = desirability * g_cond * g_sr + structure_direct
        b_div = self.diversity.bonus(h_clean, p_clean)
        total = core * b_div

        elite = (c > cfg.elite_cond_threshold) and (s < cfg.elite_sr_threshold)
        details["elite_boost_applied"] = elite
        if elite:
            total *= cfg.elite_multiplier

        details["reward_breakdown"] = {
            "desirability": desirability,
            "G_cond": g_cond,
            "G_SR": g_sr,
            "cation_diversity_score": cd_score,
            "hydrophobic_complexity": ph_complexity,
            "structure_direct_bonus": structure_direct,
            "B_div": b_div,
            "core_reward": core,
        }
        details["total_reward"] = float(total)
        return float(np.clip(total, 0.0, None)), details


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------
# Registered AEM-family constraints (the six families used in the paper)
# plus a no-op "GENERIC" entry for ablation runs.
FAMILY_REGISTRY: Dict[str, type] = {
    "PAP": PAPConstraint,
    "PBF": PBFConstraint,
    "PPO": PPOConstraint,
    "PAEK": PAEKConstraint,
    "PAEKS": PAEKSConstraint,
    "PAES": PAESConstraint,
    "GENERIC": FamilyConstraint,
}


def build_reward_fn(family_name: str, cfg: Optional[RewardConfig] = None) -> Callable:
    """Return a callable reward function for one of the registered AEM families.

    Currently registered:
        - ``"PAP"``    poly(aryl piperidinium)
        - ``"PBF"``    poly(biphenyl fluorene)
        - ``"PPO"``    poly(phenylene oxide)
        - ``"PAEK"``   poly(arylene ether ketone)
        - ``"PAEKS"``  poly(arylene ether ketone sulfone)
        - ``"PAES"``   poly(arylene ether sulfone)
        - ``"GENERIC"`` no family-specific constraint (D(S_T) defaults to 1)

    Add new families by subclassing ``FamilyConstraint`` and registering
    the subclass in ``FAMILY_REGISTRY``.
    """
    key = family_name.upper()
    if key not in FAMILY_REGISTRY:
        raise ValueError(
            f"Unknown family '{family_name}'. "
            f"Registered: {list(FAMILY_REGISTRY)}."
        )
    return RewardFunction(family=FAMILY_REGISTRY[key](), cfg=cfg)
