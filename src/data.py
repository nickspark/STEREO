import hashlib
import json
import os
import pickle
import re
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Sequence, Mapping

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset

from qc_mapping import map_atomic_qc_to_nodes
# Periodic table symbols indexed by atomic number
_PERIODIC_TABLE = [
    "", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]
_SYMBOL_TO_Z: Dict[str, int] = {sym: i for i, sym in enumerate(_PERIODIC_TABLE) if sym}

_XYZ_COLUMN_CANDIDATES = {
    "cat": ["xyz-cat-ref", "xyz-cat-ar"],
    "r1": ["xyz-r1-ref", "xyz-r1-ar"],
    "r2": ["xyz-r2-ref", "xyz-r2-ar"],
}

_BORYLATION_XYZ_COLUMN_CANDIDATES = {
    "cat": ["cat-xyz", "bvs_cat_xyz", "xyz-cat-ref", "xyz-cat-ar"],
    "r1": ["r1-xyz", "bvs_rc_xyz", "xyz-r1-ref", "xyz-r1-ar"],
    "r2": ["p1-xyz", "bvs_product_xyz", "xyz-p1-ref", "xyz-p1-ar"],
}

_BORYLATION_SMILES_COLUMN_CANDIDATES = {
    "cat": ["bvs_cat_smiles", "cat-smiles", "cat_smiles"],
    "r1": ["bvs_rc_smiles", "r1-smiles", "r1_smiles"],
    "r2": ["bvs_product_smiles", "p1-smiles", "p1_smiles", "r2-smiles", "r2_smiles"],
}

_BORYLATION_REF_COLUMN_CANDIDATES = ["ref-data", "ref_data", "reference", "ref"]

_BORYLATION_SIGNATURE_COLUMN_CANDIDATES = {
    "cat_smiles": _BORYLATION_SMILES_COLUMN_CANDIDATES["cat"],
    "r1_smiles": _BORYLATION_SMILES_COLUMN_CANDIDATES["r1"],
    "r2_smiles": _BORYLATION_SMILES_COLUMN_CANDIDATES["r2"],
    "cat_xyz": _BORYLATION_XYZ_COLUMN_CANDIDATES["cat"],
    "r1_xyz": _BORYLATION_XYZ_COLUMN_CANDIDATES["r1"],
    "r2_xyz": _BORYLATION_XYZ_COLUMN_CANDIDATES["r2"],
}

_TARGET_COLUMN_CANDIDATES = ["target_ee", "ee %", "ee%", "yield", "yield %", "yield%", "label（产率/选择性）"]

_REFERENCE_ID_COLUMN_CANDIDATES = [
    "Reference_ID",
    "reference_id",
    "reference-id",
    "reference",
    "reference_data",
    "ref_data",
    "ref-data",
    "ref",
]

_REFERENCE_ID_TEXT_COLUMN_CANDIDATES = [
    "xyz-cat-ref",
    "xyz-cat-ar",
]

_TEMPERATURE_COLUMN_CANDIDATES = ["temperature_c", "T/C", "t/c", "temperature", "temp", "T", "t"]

_REFERENCE_ID_PATTERN = re.compile(r"ref-(\d+)-", re.IGNORECASE)

_QC_COLUMN_GROUPS: Dict[str, List[str]] = {
    "charge": ["q(N):metal2", "q(N+1):metal2", "q(N-1):metal2"],
    "fukui": ["f-:metal2", "f+:metal2", "f0:metal2"],
    "softness": ["s-:metal2", "s+:metal2", "s0:metal2"],
    "reactivity": [
        "CDD:metal2",
        "Electrophilicity_index_metal2",
        "Nucleophilicity_index_metal2",
        "Electrophilicity_atom:metal2",
        "Nucleophilicity_atom:metal2",
    ],
    "energetics": [
        "Vertical_IP_metal2",
        "Vertical_EA_metal2",
        "Mulliken_electronegativity_metal2",
        "Chemical_potential_metal2",
        "Hardness_metal2",
        "Softness_metal2",
    ],
    "ratios": ["s+/s-:metal2", "s-/s+:metal2"],
    "steric": [
        "steric_cat_radius_gyration",
        "steric_r1_radius_gyration",
        "steric_r2_radius_gyration",
        "steric_cat_max_radius",
        "steric_r1_max_radius",
        "steric_r2_max_radius",
        "steric_total_vdw_volume",
        "steric_centroid_min_distance",
        "steric_centroid_mean_distance",
        "steric_global_crowding_index",
    ],
}

_CORE22_QC_COLUMNS: List[str] = [
    "q(N):metal2",
    "q(N+1):metal2",
    "q(N-1):metal2",
    "f-:metal2",
    "f+:metal2",
    "f0:metal2",
    "s-:metal2",
    "s+:metal2",
    "s0:metal2",
    "CDD:metal2",
    "Electrophilicity_index_metal2",
    "Nucleophilicity_index_metal2",
    "Electrophilicity_atom:metal2",
    "Nucleophilicity_atom:metal2",
    "Vertical_IP_metal2",
    "Vertical_EA_metal2",
    "Mulliken_electronegativity_metal2",
    "Chemical_potential_metal2",
    "Hardness_metal2",
    "Softness_metal2",
    "s+/s-:metal2",
    "s-/s+:metal2",
]

_STERIC_XYZ_COLUMN_CANDIDATES: Dict[str, List[str]] = {
    "cat": ["bvs_cat_xyz", "xyz-cat-ref", "xyz-cat-ar"],
    "r1": ["bvs_rc_xyz", "xyz-r1-ref", "xyz-r1-ar"],
    "r2": ["bvs_product_xyz", "xyz-r2-ref", "xyz-r2-ar", "xyz-p1-ref", "xyz-p1-ar"],
}

_VDW_RADIUS_BY_Z: Dict[int, float] = {
    1: 1.20,   # H
    5: 1.92,   # B
    6: 1.70,   # C
    7: 1.55,   # N
    8: 1.52,   # O
    9: 1.47,   # F
    14: 2.10,  # Si
    15: 1.80,  # P
    16: 1.80,  # S
    17: 1.75,  # Cl
    26: 2.00,  # Fe
    27: 2.00,  # Co
    28: 1.97,  # Ni
    29: 1.96,  # Cu
    35: 1.85,  # Br
    46: 2.10,  # Pd
    53: 1.98,  # I
    77: 2.00,  # Ir
}

_COVALENT_RADIUS_BY_Z: Dict[int, float] = {
    1: 0.31,
    5: 0.84,
    6: 0.76,
    7: 0.71,
    8: 0.66,
    9: 0.57,
    14: 1.11,
    15: 1.07,
    16: 1.05,
    17: 1.02,
    26: 1.24,
    27: 1.18,
    28: 1.17,
    29: 1.32,
    35: 1.20,
    44: 1.25,
    46: 1.39,
    77: 1.37,
}

_REACTION_CENTER_LATE_METALS = {"Ni", "Pd", "Pt", "Ir", "Rh", "Ru", "Cu", "Ag", "Au"}
_REACTION_CENTER_MID_METALS = {"Fe", "Co", "Mn", "Cr"}
_REACTION_CENTER_METALS = _REACTION_CENTER_LATE_METALS | _REACTION_CENTER_MID_METALS | {"V", "Ti", "Mo", "W"}
_REACTION_CENTER_DONOR_WEIGHTS = {"P": 1.10, "N": 0.95, "S": 0.90, "O": 0.65, "C": 0.18}
_REACTION_CENTER_HALIDES = {"F", "Cl", "Br", "I"}
_REACTION_CENTER_PAIR_FEATURE_DIM = 16
_INTERACTION_RELAY_CUTOFF = 3.5
_INTERACTION_STERIC_CUTOFF = 6.0
_INTERACTION_PAIR_MODE_CLASSIC = "classic"
_INTERACTION_PAIR_MODE_DENSE = "late_metal_dense"
_INTERACTION_PAIR_FEATURE_DIM = 6
_INTERACTION_PAIR_FEATURE_DIM_DENSE = 16
_INTERACTION_HETERO_ATOMS = {"N", "O", "P", "S"}
_NATIVE_PAIR_INTERACTION_TYPES = (
    "coordination",
    "h_bond",
    "covalent",
    "pi_stacking",
    "ionic",
    "van_der_waals",
)
_NATIVE_PAIR_TEMPERATURE_DIRECTIONS = (
    "favors_major_path_at_higher_temperature",
    "favors_major_path_at_lower_temperature",
    "compresses_selectivity_window",
    "broadly_neutral",
)
_NATIVE_PAIR_ATOM_SYMBOLS = (
    "Ir",
    "Rh",
    "Ru",
    "Pd",
    "Pt",
    "Ni",
    "Cu",
    "B",
    "N",
    "O",
    "P",
    "S",
    "C",
    "other",
)
_NATIVE_PAIR_MAX_TOKENS = 3
_NATIVE_PAIR_MAX_EXPERTS = 5
_NATIVE_PAIR_SUMMARY_FEATURE_DIM = (
    15 + len(_NATIVE_PAIR_TEMPERATURE_DIRECTIONS) + len(_NATIVE_PAIR_INTERACTION_TYPES)
)
_NATIVE_PAIR_TOKEN_FEATURE_DIM = (
    1 + len(_NATIVE_PAIR_INTERACTION_TYPES) + 2 * len(_NATIVE_PAIR_ATOM_SYMBOLS)
)
_NATIVE_PAIR_SUMMARY_FEATURE_NAMES = (
    "confidence",
    "pair_focus",
    "reaction_center_compatibility",
    "steric_accommodation",
    "electronic_complement",
    "temperature_alignment",
    "supports_major_path",
    "catalyst_atom_coverage",
    "reactant_atom_coverage",
    "joint_interaction_density",
    "mean_joint_confidence",
    "max_joint_confidence",
    "ranking_signal::compatibility_order_score",
    "ranking_signal::pairwise_preference_margin",
    "ranking_signal::ee_order_confidence",
) + tuple(f"temperature_direction::{name}" for name in _NATIVE_PAIR_TEMPERATURE_DIRECTIONS) + tuple(
    f"dominant_interaction::{name}" for name in _NATIVE_PAIR_INTERACTION_TYPES
)
_NATIVE_PAIR_TOKEN_FEATURE_NAMES = (
    "interaction_confidence",
) + tuple(f"interaction_type::{name}" for name in _NATIVE_PAIR_INTERACTION_TYPES) + tuple(
    f"catalyst_atom::{name}" for name in _NATIVE_PAIR_ATOM_SYMBOLS
) + tuple(f"reactant_atom::{name}" for name in _NATIVE_PAIR_ATOM_SYMBOLS)
_NATIVE_PAIR_SUMMARY_GROUPS = {
    "core_compatibility": (0, 1, 2, 3, 4),
    "temperature_signal": (5, 15, 16, 17, 18),
    "pathway_support": (6,),
    "evidence_coverage": (7, 8, 9, 10, 11),
    "ranking_signal": (12, 13, 14),
    "interaction_type_identity": (19, 20, 21, 22, 23, 24),
}
_NATIVE_PAIR_TOKEN_GROUPS = {
    "interaction_confidence": (0,),
    "interaction_type_identity": (1, 2, 3, 4, 5, 6),
    "interaction_atom_identity": tuple(range(7, _NATIVE_PAIR_TOKEN_FEATURE_DIM)),
}
_NATIVE_PAIR_FIELD_PROFILE_SPECS = {
    "full": {
        "description": "Preserve the native-pair tensors exactly as emitted.",
        "summary_group_scales": {},
        "token_group_scales": {},
    },
    "chemistry_core_compact": {
        "description": (
            "Keep the compact local chemistry core while removing template-prone temperature, "
            "interaction-identity, and atom-identity channels; downweight evidence-density counts."
        ),
        "summary_group_scales": {
            "temperature_signal": 0.0,
            "pathway_support": 0.6,
            "evidence_coverage": 0.35,
            "interaction_type_identity": 0.0,
        },
        "token_group_scales": {
            "interaction_type_identity": 0.0,
            "interaction_atom_identity": 0.0,
        },
    },
    "mask_core_compatibility": {
        "description": "Zero the core local compatibility scalars.",
        "summary_group_scales": {"core_compatibility": 0.0},
        "token_group_scales": {},
    },
    "mask_temperature_signal": {
        "description": "Zero the temperature-alignment scalar and temperature-direction one-hot channels.",
        "summary_group_scales": {"temperature_signal": 0.0},
        "token_group_scales": {},
    },
    "mask_pathway_support": {
        "description": "Zero the dominant-path support flag only.",
        "summary_group_scales": {"pathway_support": 0.0},
        "token_group_scales": {},
    },
    "mask_evidence_coverage": {
        "description": "Zero evidence-count and joint-density summary channels.",
        "summary_group_scales": {"evidence_coverage": 0.0},
        "token_group_scales": {},
    },
    "mask_interaction_type_identity": {
        "description": "Zero dominant-interaction and joint-interaction type identity channels.",
        "summary_group_scales": {"interaction_type_identity": 0.0},
        "token_group_scales": {"interaction_type_identity": 0.0},
    },
    "mask_interaction_confidence": {
        "description": "Zero per-interaction token confidence.",
        "summary_group_scales": {},
        "token_group_scales": {"interaction_confidence": 0.0},
    },
    "mask_interaction_atom_identity": {
        "description": "Zero catalyst/reactant atom identity one-hot channels in joint-interaction tokens.",
        "summary_group_scales": {},
        "token_group_scales": {"interaction_atom_identity": 0.0},
    },
}

_COORDINATION_FEATURE_COLS: List[str] = [
    "coordination_active",
    "coordination_is_n",
    "coordination_is_o",
    "coordination_is_s",
    "coordination_potency",
    "coordination_hetero_neighbor_ratio",
]


def _interaction_pair_feature_dim_for_mode(mode: str) -> int:
    resolved = str(mode or _INTERACTION_PAIR_MODE_CLASSIC).lower()
    if resolved == _INTERACTION_PAIR_MODE_CLASSIC:
        return _INTERACTION_PAIR_FEATURE_DIM
    if resolved == _INTERACTION_PAIR_MODE_DENSE:
        return _INTERACTION_PAIR_FEATURE_DIM_DENSE
    raise ValueError(f"Unknown interaction pair mode: {mode}")


def native_pair_field_profile_choices() -> Tuple[str, ...]:
    return tuple(_NATIVE_PAIR_FIELD_PROFILE_SPECS.keys())


def native_pair_field_groups() -> Dict[str, Dict[str, Tuple[int, ...]]]:
    return {
        "summary": {name: tuple(indices) for name, indices in _NATIVE_PAIR_SUMMARY_GROUPS.items()},
        "token": {name: tuple(indices) for name, indices in _NATIVE_PAIR_TOKEN_GROUPS.items()},
    }


def _native_pair_scale_vector(
    *,
    dim: int,
    groups: Mapping[str, Sequence[int]],
    overrides: Mapping[str, float],
) -> np.ndarray:
    vector = np.ones((dim,), dtype=np.float32)
    for group_name, scale in overrides.items():
        if group_name not in groups:
            raise ValueError(
                f"Unknown native-pair field group `{group_name}`. "
                f"Known groups: {sorted(groups)}"
            )
        vector[np.asarray(tuple(groups[group_name]), dtype=np.int64)] = float(scale)
    return vector


def native_pair_field_profile_spec(profile: str = "full") -> Dict[str, object]:
    resolved = str(profile or "full").lower()
    spec = _NATIVE_PAIR_FIELD_PROFILE_SPECS.get(resolved)
    if spec is None:
        raise ValueError(
            f"Unknown native_pair_field_profile `{profile}`. "
            f"Choices: {list(native_pair_field_profile_choices())}"
        )
    summary_group_scales = {
        str(name): float(scale) for name, scale in (spec.get("summary_group_scales") or {}).items()
    }
    token_group_scales = {
        str(name): float(scale) for name, scale in (spec.get("token_group_scales") or {}).items()
    }
    summary_scale_vector = _native_pair_scale_vector(
        dim=_NATIVE_PAIR_SUMMARY_FEATURE_DIM,
        groups=_NATIVE_PAIR_SUMMARY_GROUPS,
        overrides=summary_group_scales,
    )
    token_scale_vector = _native_pair_scale_vector(
        dim=_NATIVE_PAIR_TOKEN_FEATURE_DIM,
        groups=_NATIVE_PAIR_TOKEN_GROUPS,
        overrides=token_group_scales,
    )
    return {
        "profile": resolved,
        "description": str(spec.get("description") or ""),
        "summary_group_scales": summary_group_scales,
        "token_group_scales": token_group_scales,
        "summary_scale_vector": summary_scale_vector,
        "token_scale_vector": token_scale_vector,
        "summary_feature_names": tuple(_NATIVE_PAIR_SUMMARY_FEATURE_NAMES),
        "token_feature_names": tuple(_NATIVE_PAIR_TOKEN_FEATURE_NAMES),
    }

_RDKIT_DESCRIPTOR_NAMES = [
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "RingCount",
    "HeavyAtomCount",
    "FractionCSP3",
]

_MORGAN_FINGERPRINT_DEFAULT_BITS = 1024
_MORGAN_FINGERPRINT_DEFAULT_RADIUS = 2


def _rdkit_descriptor_vector(smiles: str) -> np.ndarray:
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
    except ImportError as exc:
        raise ImportError("RDKit is required for rdkit descriptor features.") from exc

    if not smiles or not isinstance(smiles, str):
        return np.zeros((len(_RDKIT_DESCRIPTOR_NAMES),), dtype=np.float32)
    try:
        from molecular_representation.core import safe_mol_from_smiles
    except Exception:  # noqa: BLE001
        safe_mol_from_smiles = None

    mol = safe_mol_from_smiles(smiles) if safe_mol_from_smiles else Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((len(_RDKIT_DESCRIPTOR_NAMES),), dtype=np.float32)

    values = [
        Descriptors.MolWt(mol),
        Crippen.MolLogP(mol),
        rdMolDescriptors.CalcTPSA(mol),
        rdMolDescriptors.CalcNumHBD(mol),
        rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),
        rdMolDescriptors.CalcNumRings(mol),
        Descriptors.HeavyAtomCount(mol),
        rdMolDescriptors.CalcFractionCSP3(mol),
    ]
    return np.asarray(values, dtype=np.float32)


def _morgan_fingerprint_vector(smiles: str, *, radius: int, n_bits: int) -> np.ndarray:
    try:
        from rdkit import DataStructs
        from rdkit.Chem import rdFingerprintGenerator
    except ImportError as exc:
        raise ImportError("RDKit is required for Morgan fingerprint features.") from exc

    if not smiles or not isinstance(smiles, str):
        return np.zeros((n_bits,), dtype=np.float32)
    try:
        from molecular_representation.core import safe_mol_from_smiles
    except Exception:  # noqa: BLE001
        safe_mol_from_smiles = None

    mol = safe_mol_from_smiles(smiles) if safe_mol_from_smiles else None
    if mol is None:
        return np.zeros((n_bits,), dtype=np.float32)

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=int(radius), fpSize=int(n_bits))
    fp = generator.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


@dataclass
class Sample:
    cat_z: torch.Tensor
    cat_pos: torch.Tensor
    cat_edge_index: torch.Tensor
    r1_z: torch.Tensor
    r1_pos: torch.Tensor
    r1_edge_index: torch.Tensor
    r2_z: torch.Tensor
    r2_pos: torch.Tensor
    r2_edge_index: torch.Tensor
    features: torch.Tensor
    target: torch.Tensor
    temperature: Optional[torch.Tensor] = None
    qc_features: Optional[torch.Tensor] = None
    cat_node_qc: Optional[torch.Tensor] = None
    r1_node_qc: Optional[torch.Tensor] = None
    r2_node_qc: Optional[torch.Tensor] = None
    llm_features: Optional[torch.Tensor] = None
    llm_confidence: Optional[torch.Tensor] = None
    exact_covered_indicator: Optional[torch.Tensor] = None
    literature: Optional[dict] = None
    native_pair_summary: Optional[torch.Tensor] = None
    native_pair_tokens: Optional[torch.Tensor] = None
    native_pair_token_mask: Optional[torch.Tensor] = None
    native_pair_expert_summaries: Optional[torch.Tensor] = None
    native_pair_expert_tokens: Optional[torch.Tensor] = None
    native_pair_expert_token_masks: Optional[torch.Tensor] = None
    native_pair_expert_priors: Optional[torch.Tensor] = None
    combined_z: Optional[torch.Tensor] = None
    combined_pos: Optional[torch.Tensor] = None
    combined_node_type: Optional[torch.Tensor] = None
    combined_edge_index: Optional[torch.Tensor] = None
    combined_edge_attr: Optional[torch.Tensor] = None
    interaction_cat_r1_index: Optional[torch.Tensor] = None
    interaction_cat_r1_features: Optional[torch.Tensor] = None
    interaction_cat_r2_index: Optional[torch.Tensor] = None
    interaction_cat_r2_features: Optional[torch.Tensor] = None
    reaction_center_cat_r1_index: Optional[torch.Tensor] = None
    reaction_center_cat_r1_features: Optional[torch.Tensor] = None
    reaction_center_cat_r2_index: Optional[torch.Tensor] = None
    reaction_center_cat_r2_features: Optional[torch.Tensor] = None
    reaction_center_targets: Optional[torch.Tensor] = None
    focus_kinetic_relay: Optional[torch.Tensor] = None
    electronic_frontier_score: Optional[torch.Tensor] = None
    group_id: Optional[int] = None
    rank_target: Optional[torch.Tensor] = None


def _parse_xyz_block(block: str) -> Tuple[List[int], List[List[float]]]:
    lines = [ln.strip() for ln in str(block).splitlines() if ln.strip()]
    if not lines:
        return [], []

    # If the first line is a single integer, treat as atom count header.
    first_tokens = lines[0].split()
    start_idx = 0
    if len(first_tokens) == 1 and re.fullmatch(r"\d+", first_tokens[0]):
        start_idx = 1

    zs: List[int] = []
    coords: List[List[float]] = []
    for ln in lines[start_idx:]:
        toks = ln.split()
        if len(toks) < 4:
            continue
        sym = toks[0]
        if re.fullmatch(r"\d+", sym):
            z = int(sym)
        else:
            sym = sym[0].upper() + sym[1:].lower()
            if sym not in _SYMBOL_TO_Z:
                raise ValueError(f"Unknown element symbol: {sym}")
            z = _SYMBOL_TO_Z[sym]
        try:
            x, y, zc = float(toks[1]), float(toks[2]), float(toks[3])
        except ValueError as exc:
            raise ValueError(f"Invalid coordinates line: {ln}") from exc
        zs.append(z)
        coords.append([x, y, zc])

    return zs, coords


_BORYLATION_3D_CACHE_VERSION = "borylation_3d_v2"


def _dataset_cache_root() -> Path:
    override = os.environ.get("CATA_DATASET_CACHE_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return Path("progress") / "dataset_cache"


def _borylation_3d_content_digest(
    df: pd.DataFrame,
    *,
    cat_xyz_col: str,
    r1_xyz_col: str,
    r2_xyz_col: str,
) -> str:
    hasher = hashlib.sha256()
    hasher.update(_BORYLATION_3D_CACHE_VERSION.encode("utf-8"))
    hasher.update(str(len(df)).encode("utf-8"))
    for column in (cat_xyz_col, r1_xyz_col, r2_xyz_col):
        hasher.update(column.encode("utf-8"))
        hasher.update(b"\0")
        for value in df[column].fillna("").astype(str).tolist():
            hasher.update(value.encode("utf-8"))
            hasher.update(b"\0")
    return hasher.hexdigest()[:16]


def _borylation_3d_cache_path(
    *,
    content_digest: str,
    use_coulomb: bool,
    use_geometry: bool,
) -> Path:
    mode = f"c{int(bool(use_coulomb))}g{int(bool(use_geometry))}"
    return _dataset_cache_root() / f"borylation3d_{content_digest}_{mode}.pkl"


def _load_cached_borylation_3d_features(
    *,
    cache_key: str,
    row_count: int,
    use_coulomb: bool,
    use_geometry: bool,
):
    cache_path = _borylation_3d_cache_path(
        content_digest=cache_key,
        use_coulomb=use_coulomb,
        use_geometry=use_geometry,
    )
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("rb") as f:
            payload = pickle.load(f)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("version") != _BORYLATION_3D_CACHE_VERSION:
        return None
    if payload.get("cache_key") != cache_key:
        return None
    if int(payload.get("row_count", -1)) != int(row_count):
        return None
    parsed_rows = payload.get("parsed_rows")
    if not isinstance(parsed_rows, list) or len(parsed_rows) != int(row_count):
        return None
    for key in ("max_cat", "max_r1", "max_r2"):
        value = payload.get(key)
        if not isinstance(value, (int, np.integer)):
            return None
    if use_coulomb:
        coulomb_features = payload.get("coulomb_features")
        if not isinstance(coulomb_features, np.ndarray) or coulomb_features.shape[0] != int(row_count):
            return None
    if use_geometry:
        geom_features = payload.get("geom_features")
        if not isinstance(geom_features, np.ndarray) or geom_features.shape[0] != int(row_count):
            return None
    return payload


def _save_cached_borylation_3d_features(
    *,
    cache_key: str,
    row_count: int,
    use_coulomb: bool,
    use_geometry: bool,
    payload: Dict[str, object],
) -> None:
    cache_path = _borylation_3d_cache_path(
        content_digest=cache_key,
        use_coulomb=use_coulomb,
        use_geometry=use_geometry,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def _compute_borylation_3d_features(
    df: pd.DataFrame,
    *,
    excel_path: str,
    cat_xyz_col: str,
    r1_xyz_col: str,
    r2_xyz_col: str,
    use_coulomb: bool,
    use_geometry: bool,
    use_cache: bool = True,
    write_cache: bool = True,
    max_atoms_override: Optional[Tuple[int, int, int]] = None,
):
    cache_key = _borylation_3d_content_digest(
        df,
        cat_xyz_col=cat_xyz_col,
        r1_xyz_col=r1_xyz_col,
        r2_xyz_col=r2_xyz_col,
    )
    if use_cache and max_atoms_override is None:
        cached = _load_cached_borylation_3d_features(
            cache_key=cache_key,
            row_count=len(df),
            use_coulomb=use_coulomb,
            use_geometry=use_geometry,
        )
        if cached is not None:
            return (
                cached["parsed_rows"],
                cached.get("coulomb_features"),
                cached.get("geom_features"),
                int(cached["max_cat"]),
                int(cached["max_r1"]),
                int(cached["max_r2"]),
            )

    parsed_rows: List[Dict[str, np.ndarray]] = []
    max_cat = max_r1 = max_r2 = 0
    for _, row in df.iterrows():
        cat_z, cat_pos = _parse_xyz_block(row[cat_xyz_col])
        r1_z, r1_pos = _parse_xyz_block(row[r1_xyz_col])
        r2_z, r2_pos = _parse_xyz_block(row[r2_xyz_col])
        cat_pos_arr = np.asarray(cat_pos, dtype=np.float32)
        r1_pos_arr = np.asarray(r1_pos, dtype=np.float32)
        r2_pos_arr = np.asarray(r2_pos, dtype=np.float32)
        parsed_rows.append(
            {
                "cat_z": np.asarray(cat_z, dtype=np.int64),
                "cat_pos": cat_pos_arr,
                "cat_edge_index": _build_knn_edges(cat_pos).cpu().numpy().astype(np.int64, copy=False),
                "r1_z": np.asarray(r1_z, dtype=np.int64),
                "r1_pos": r1_pos_arr,
                "r1_edge_index": _build_knn_edges(r1_pos).cpu().numpy().astype(np.int64, copy=False),
                "r2_z": np.asarray(r2_z, dtype=np.int64),
                "r2_pos": r2_pos_arr,
                "r2_edge_index": _build_knn_edges(r2_pos).cpu().numpy().astype(np.int64, copy=False),
            }
        )
        max_cat = max(max_cat, len(cat_z))
        max_r1 = max(max_r1, len(r1_z))
        max_r2 = max(max_r2, len(r2_z))

    if max_atoms_override is not None:
        max_cat, max_r1, max_r2 = (int(v) for v in max_atoms_override)
    coulomb_features = None
    geom_features = None
    if use_coulomb:
        coulomb_dim = max_cat + max_r1 + max_r2
        coulomb_features = np.zeros((len(df), coulomb_dim), dtype=np.float32)
    if use_geometry:
        geom_dim = (max_cat + max_r1 + max_r2) * 4
        geom_features = np.zeros((len(df), geom_dim), dtype=np.float32)

    for idx, parsed in enumerate(parsed_rows):
        cat_z = parsed["cat_z"].tolist()
        r1_z = parsed["r1_z"].tolist()
        r2_z = parsed["r2_z"].tolist()
        cat_pos = parsed["cat_pos"].tolist()
        r1_pos = parsed["r1_pos"].tolist()
        r2_pos = parsed["r2_pos"].tolist()
        if use_coulomb:
            cat_feat = _coulomb_eigenvalues(cat_z, cat_pos, max_cat)
            r1_feat = _coulomb_eigenvalues(r1_z, r1_pos, max_r1)
            r2_feat = _coulomb_eigenvalues(r2_z, r2_pos, max_r2)
            coulomb_features[idx] = np.concatenate([cat_feat, r1_feat, r2_feat], axis=0)
        if use_geometry:
            cat_geom = _flatten_geometry(cat_z, cat_pos, max_cat)
            r1_geom = _flatten_geometry(r1_z, r1_pos, max_r1)
            r2_geom = _flatten_geometry(r2_z, r2_pos, max_r2)
            geom_features[idx] = np.concatenate([cat_geom, r1_geom, r2_geom], axis=0)

    payload = {
        "version": _BORYLATION_3D_CACHE_VERSION,
        "cache_key": cache_key,
        "row_count": int(len(df)),
        "max_cat": int(max_cat),
        "max_r1": int(max_r1),
        "max_r2": int(max_r2),
        "source_path": str(Path(excel_path).resolve()),
        "xyz_columns": {
            "cat": cat_xyz_col,
            "r1": r1_xyz_col,
            "r2": r2_xyz_col,
        },
        "feature_flags": {
            "use_coulomb": bool(use_coulomb),
            "use_geometry": bool(use_geometry),
        },
        "created_at_utc": datetime.utcnow().isoformat() + "Z",
        "parsed_rows": parsed_rows,
        "coulomb_features": coulomb_features,
        "geom_features": geom_features,
    }
    if write_cache and max_atoms_override is None:
        _save_cached_borylation_3d_features(
            cache_key=cache_key,
            row_count=len(df),
            use_coulomb=use_coulomb,
            use_geometry=use_geometry,
            payload=payload,
        )
    return parsed_rows, coulomb_features, geom_features, max_cat, max_r1, max_r2


def describe_borylation_3d_cache(
    df: pd.DataFrame,
    *,
    excel_path: str,
    cat_xyz_col: str,
    r1_xyz_col: str,
    r2_xyz_col: str,
    use_coulomb: bool,
    use_geometry: bool,
) -> Dict[str, object]:
    cache_key = _borylation_3d_content_digest(
        df,
        cat_xyz_col=cat_xyz_col,
        r1_xyz_col=r1_xyz_col,
        r2_xyz_col=r2_xyz_col,
    )
    cache_path = _borylation_3d_cache_path(
        content_digest=cache_key,
        use_coulomb=use_coulomb,
        use_geometry=use_geometry,
    )
    payload = _load_cached_borylation_3d_features(
        cache_key=cache_key,
        row_count=len(df),
        use_coulomb=use_coulomb,
        use_geometry=use_geometry,
    )
    return {
        "cache_key": cache_key,
        "cache_path": str(cache_path),
        "exists": cache_path.exists(),
        "cache_root": str(_dataset_cache_root()),
        "source_path": str(Path(excel_path).resolve()),
        "row_count": int(len(df)),
        "xyz_columns": {
            "cat": cat_xyz_col,
            "r1": r1_xyz_col,
            "r2": r2_xyz_col,
        },
        "feature_flags": {
            "use_coulomb": bool(use_coulomb),
            "use_geometry": bool(use_geometry),
        },
        "metadata": {
            "version": None if payload is None else payload.get("version"),
            "created_at_utc": None if payload is None else payload.get("created_at_utc"),
            "max_cat": None if payload is None else int(payload.get("max_cat", 0)),
            "max_r1": None if payload is None else int(payload.get("max_r1", 0)),
            "max_r2": None if payload is None else int(payload.get("max_r2", 0)),
        },
    }


def _select_column(df: pd.DataFrame, candidates: List[str], label: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Missing required {label} column. Tried: {candidates}")


def _select_target_column(df: pd.DataFrame) -> str:
    for col in _TARGET_COLUMN_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(f"Missing target column. Tried: {_TARGET_COLUMN_CANDIDATES}")


def _find_temperature_column(df: pd.DataFrame) -> Optional[str]:
    for col in _TEMPERATURE_COLUMN_CANDIDATES:
        if col in df.columns:
            return col
    for col in df.columns:
        text = str(col).strip().lower()
        if text in {"t/c", "temperature", "temp"}:
            return str(col)
        if "temp" in text and "attempt" not in text:
            return str(col)
    return None


def _is_excluded_metadata_feature(column: object) -> bool:
    text = str(column).strip()
    if not text:
        return False
    lowered = text.lower()
    if lowered.startswith("bvs_"):
        return True
    if lowered == "observed_in_locked_table":
        return True
    return False


def _infer_qc_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for name, columns in _QC_COLUMN_GROUPS.items():
        present = [col for col in columns if col in df.columns]
        if present:
            groups[name] = present
    return groups


def _column_values(df: pd.DataFrame, column: str) -> np.ndarray:
    if column not in df.columns:
        return np.zeros((len(df),), dtype=np.float32)
    series = df[column]
    if series.isna().any():
        series = series.fillna(0.0)
    return series.to_numpy(dtype=np.float32, copy=True)


def _normalize_signature_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)) and not np.isfinite(float(value)):
        return ""
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        val = float(value)
        if val.is_integer():
            return str(int(val))
        return f"{val:.10g}"
    return " ".join(str(value).split())


def _signature_series(df: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    if not columns:
        raise ValueError("signature columns must not be empty")
    normalized = [df[col].map(_normalize_signature_value).astype(str) for col in columns]
    signature = normalized[0]
    for part in normalized[1:]:
        signature = signature + "\x1f" + part
    return signature


def _resolve_structural_signature_groups(
    df: pd.DataFrame,
    signature_columns: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    if signature_columns:
        columns = list(signature_columns)
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing structural signature columns: {missing}")
    else:
        columns = [
            _select_column(df, _BORYLATION_SIGNATURE_COLUMN_CANDIDATES["cat_smiles"], "cat smiles"),
            _select_column(df, _BORYLATION_SIGNATURE_COLUMN_CANDIDATES["r1_smiles"], "r1 smiles"),
            _select_column(df, _BORYLATION_SIGNATURE_COLUMN_CANDIDATES["r2_smiles"], "p1 smiles"),
            _select_column(df, _BORYLATION_SIGNATURE_COLUMN_CANDIDATES["cat_xyz"], "cat xyz"),
            _select_column(df, _BORYLATION_SIGNATURE_COLUMN_CANDIDATES["r1_xyz"], "r1 xyz"),
            _select_column(df, _BORYLATION_SIGNATURE_COLUMN_CANDIDATES["r2_xyz"], "p1 xyz"),
        ]
    signature_values = _signature_series(df, columns).to_numpy(dtype=object)
    groups = pd.factorize(signature_values, sort=False)[0].astype(np.int64)
    return groups, columns


def _extract_xyz_by_candidates(row: pd.Series, candidates: Sequence[str]) -> Tuple[List[int], List[List[float]]]:
    for col in candidates:
        if col not in row.index:
            continue
        value = row[col]
        if pd.isna(value):
            continue
        try:
            return _parse_xyz_block(value)
        except Exception:  # noqa: BLE001
            continue
    return [], []


def _component_steric_metrics(zs: Sequence[int], coords: Sequence[Sequence[float]]) -> Dict[str, float]:
    if not zs or not coords:
        return {
            "radius_gyration": 0.0,
            "max_radius": 0.0,
            "vdw_volume": 0.0,
        }
    pos = np.asarray(coords, dtype=np.float32)
    centroid = pos.mean(axis=0, keepdims=True)
    centered = pos - centroid
    radii = np.linalg.norm(centered, axis=1)
    rg = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    max_radius = float(radii.max()) if radii.size else 0.0
    vdw_radii = np.asarray([_VDW_RADIUS_BY_Z.get(int(z), 1.8) for z in zs], dtype=np.float32)
    vdw_volume = float(np.sum((4.0 / 3.0) * np.pi * np.power(vdw_radii, 3)))
    return {
        "radius_gyration": rg,
        "max_radius": max_radius,
        "vdw_volume": vdw_volume,
    }


def _build_steric_qc_frame(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame(index=df.index)

    records: List[Dict[str, float]] = []
    for _, row in df.iterrows():
        cat_z, cat_pos = _extract_xyz_by_candidates(row, _STERIC_XYZ_COLUMN_CANDIDATES["cat"])
        r1_z, r1_pos = _extract_xyz_by_candidates(row, _STERIC_XYZ_COLUMN_CANDIDATES["r1"])
        r2_z, r2_pos = _extract_xyz_by_candidates(row, _STERIC_XYZ_COLUMN_CANDIDATES["r2"])

        cat_metrics = _component_steric_metrics(cat_z, cat_pos)
        r1_metrics = _component_steric_metrics(r1_z, r1_pos)
        r2_metrics = _component_steric_metrics(r2_z, r2_pos)

        centroids: List[np.ndarray] = []
        for coords in (cat_pos, r1_pos, r2_pos):
            if not coords:
                continue
            arr = np.asarray(coords, dtype=np.float32)
            centroids.append(arr.mean(axis=0))
        centroid_distances: List[float] = []
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                centroid_distances.append(float(np.linalg.norm(centroids[i] - centroids[j])))

        all_points = [np.asarray(coords, dtype=np.float32) for coords in (cat_pos, r1_pos, r2_pos) if coords]
        if all_points:
            stacked = np.concatenate(all_points, axis=0)
            bbox_lengths = stacked.max(axis=0) - stacked.min(axis=0)
            bbox_volume = float(np.prod(np.maximum(bbox_lengths, 1e-6)))
        else:
            bbox_volume = 0.0

        total_vdw = cat_metrics["vdw_volume"] + r1_metrics["vdw_volume"] + r2_metrics["vdw_volume"]
        crowding = float(total_vdw / max(bbox_volume, 1e-6)) if total_vdw > 0.0 else 0.0

        records.append(
            {
                "steric_cat_radius_gyration": cat_metrics["radius_gyration"],
                "steric_r1_radius_gyration": r1_metrics["radius_gyration"],
                "steric_r2_radius_gyration": r2_metrics["radius_gyration"],
                "steric_cat_max_radius": cat_metrics["max_radius"],
                "steric_r1_max_radius": r1_metrics["max_radius"],
                "steric_r2_max_radius": r2_metrics["max_radius"],
                "steric_total_vdw_volume": total_vdw,
                "steric_centroid_min_distance": min(centroid_distances) if centroid_distances else 0.0,
                "steric_centroid_mean_distance": float(np.mean(centroid_distances)) if centroid_distances else 0.0,
                "steric_global_crowding_index": crowding,
            }
        )

    return pd.DataFrame.from_records(records, index=df.index).astype(np.float32)


def _split_indices(
    indices: np.ndarray,
    val_fraction: float,
    test_fraction: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(indices)
    if n == 0:
        return indices, indices[:0], indices[:0]
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1.")
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1.")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1.")
    n_val = max(1, int(round(val_fraction * n)))
    n_test = max(1, int(round(test_fraction * n))) if test_fraction > 0.0 else 0
    n_train = n - n_val - n_test
    if n_train <= 0:
        n_train = 1
        remaining = n - n_train
        if test_fraction > 0.0:
            total = val_fraction + test_fraction
            n_val = max(1, int(round(remaining * val_fraction / total)))
            n_test = max(0, remaining - n_val)
        else:
            n_val = remaining
            n_test = 0
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:n_train + n_val + n_test]
    return train_idx, val_idx, test_idx


def _parse_reference_id(value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return None
        if float(value).is_integer():
            return int(value)
    text = str(value).strip()
    if not text:
        return None
    match = _REFERENCE_ID_PATTERN.search(text)
    if match:
        return int(match.group(1))
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    return None


def _build_reference_groups(series: pd.Series) -> Optional[np.ndarray]:
    parsed: List[Optional[int]] = [_parse_reference_id(value) for value in series.tolist()]
    valid = sum(value is not None for value in parsed)
    if valid == 0:
        return None
    groups = np.empty((len(parsed),), dtype=np.int64)
    for idx, value in enumerate(parsed):
        if value is None:
            groups[idx] = -(idx + 1)
        else:
            groups[idx] = int(value)
    return groups


def _resolve_reference_groups(df: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[str]]:
    best_groups: Optional[np.ndarray] = None
    best_source: Optional[str] = None
    best_duplicates = -1
    candidate_cols = _REFERENCE_ID_COLUMN_CANDIDATES + _REFERENCE_ID_TEXT_COLUMN_CANDIDATES
    for col in candidate_cols:
        if col not in df.columns:
            continue
        groups = _build_reference_groups(df[col])
        if groups is None:
            continue
        unique_count = int(np.unique(groups).size)
        duplicates = len(groups) - unique_count
        if duplicates > best_duplicates and unique_count >= 2:
            best_groups = groups
            best_source = col
            best_duplicates = duplicates
    if best_groups is None or best_duplicates <= 0:
        if len(df) >= 2:
            return np.arange(len(df), dtype=np.int64), "row_index"
        return None, None
    return best_groups, best_source


def _group_split_indices(
    indices: np.ndarray,
    groups: np.ndarray,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(indices)
    if n == 0:
        return indices, indices[:0], indices[:0]
    if len(groups) != n:
        raise ValueError("groups must have the same length as indices.")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1.")
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1.")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1.")

    unique_groups = np.unique(groups)
    if unique_groups.size == n:
        shuffled = indices.copy()
        rng = np.random.default_rng(int(seed))
        rng.shuffle(shuffled)
        return _split_indices(
            shuffled,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
        )
    min_groups = 3 if test_fraction > 0.0 else 2
    if unique_groups.size < min_groups:
        raise ValueError(
            f"GroupShuffleSplit requires at least {min_groups} unique groups "
            f"when test_fraction={test_fraction}."
        )

    positions = np.arange(n)
    if test_fraction > 0.0:
        test_splitter = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=int(seed))
        train_val_pos, test_pos = next(test_splitter.split(positions, groups=groups))
    else:
        train_val_pos = positions
        test_pos = positions[:0]

    train_val_groups = groups[train_val_pos]
    if np.unique(train_val_groups).size < 2:
        raise ValueError("Not enough groups left after test split to build validation split.")

    val_ratio = val_fraction / (1.0 - test_fraction)
    val_splitter = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=int(seed) + 1)
    train_pos, val_pos = next(val_splitter.split(train_val_pos, groups=train_val_groups))

    train_idx = indices[train_val_pos[train_pos]]
    val_idx = indices[train_val_pos[val_pos]]
    test_idx = indices[test_pos]
    return train_idx, val_idx, test_idx


def _build_qc_features(
    df: pd.DataFrame,
    qc_fusion: str,
    qc_groups: Optional[Dict[str, List[str]]] = None,
) -> Tuple[np.ndarray, List[str], Dict[str, List[str]]]:
    qc_fusion = (qc_fusion or "none").lower()
    if qc_fusion == "none":
        return np.zeros((len(df), 0), dtype=np.float32), [], qc_groups or {}
    if qc_fusion not in {
        "pooled",
        "interaction",
        "pooled_interaction",
        "raw",
        "raw_interaction",
        "energetics",
        "core22",
    }:
        raise ValueError(f"Unknown qc_fusion strategy: {qc_fusion}")

    steric_df = _build_steric_qc_frame(df)
    if steric_df.empty:
        qc_df = df
    else:
        qc_df = df.copy()
        for col in steric_df.columns:
            qc_df[col] = steric_df[col]

    groups = qc_groups if qc_groups is not None else _infer_qc_groups(qc_df)
    features: List[np.ndarray] = []
    feature_cols: List[str] = []

    if qc_fusion == "core22":
        eps = 1e-6
        cat_rg = _column_values(qc_df, "steric_cat_radius_gyration")
        r1_rg = _column_values(qc_df, "steric_r1_radius_gyration")
        r2_rg = _column_values(qc_df, "steric_r2_radius_gyration")
        cat_max = _column_values(qc_df, "steric_cat_max_radius")
        r1_max = _column_values(qc_df, "steric_r1_max_radius")
        r2_max = _column_values(qc_df, "steric_r2_max_radius")
        total_vdw = _column_values(qc_df, "steric_total_vdw_volume")
        centroid_min = _column_values(qc_df, "steric_centroid_min_distance")
        centroid_mean = _column_values(qc_df, "steric_centroid_mean_distance")
        crowding = _column_values(qc_df, "steric_global_crowding_index")

        vertical_ip = total_vdw / (centroid_min + eps)
        vertical_ea = total_vdw / (centroid_mean + eps)
        hardness = vertical_ip - vertical_ea
        softness = 1.0 / (np.abs(hardness) + eps)
        electrophilicity = cat_rg / (centroid_mean + eps)
        nucleophilicity = (r1_rg + r2_rg) / (centroid_mean + eps)
        s_minus = centroid_min
        s_plus = centroid_mean
        proxies: Dict[str, np.ndarray] = {
            "q(N):metal2": crowding,
            "q(N+1):metal2": cat_rg - r1_rg,
            "q(N-1):metal2": cat_rg - r2_rg,
            "f-:metal2": r1_max - r2_max,
            "f+:metal2": r1_rg + r2_rg,
            "f0:metal2": cat_max,
            "s-:metal2": s_minus,
            "s+:metal2": s_plus,
            "s0:metal2": total_vdw / (cat_max + r1_max + r2_max + eps),
            "CDD:metal2": crowding * centroid_mean,
            "Electrophilicity_index_metal2": electrophilicity,
            "Nucleophilicity_index_metal2": nucleophilicity,
            "Electrophilicity_atom:metal2": cat_max / (centroid_min + eps),
            "Nucleophilicity_atom:metal2": (r1_max + r2_max) / (centroid_min + eps),
            "Vertical_IP_metal2": vertical_ip,
            "Vertical_EA_metal2": vertical_ea,
            "Mulliken_electronegativity_metal2": 0.5 * (vertical_ip + vertical_ea),
            "Chemical_potential_metal2": -0.5 * (vertical_ip + vertical_ea),
            "Hardness_metal2": hardness,
            "Softness_metal2": softness,
            "s+/s-:metal2": s_plus / (s_minus + eps),
            "s-/s+:metal2": s_minus / (s_plus + eps),
        }
        for col in _CORE22_QC_COLUMNS:
            if col in qc_df.columns:
                values = _column_values(qc_df, col)
            else:
                values = proxies[col].astype(np.float32, copy=False)
            features.append(values[:, None])
            feature_cols.append(col)

    if qc_fusion == "energetics" and groups:
        columns = groups.get("energetics", [])
        for col in columns:
            features.append(_column_values(qc_df, col)[:, None])
            feature_cols.append(col)

    if qc_fusion in {"raw", "raw_interaction"} and groups:
        for columns in groups.values():
            for col in columns:
                features.append(_column_values(qc_df, col)[:, None])
                feature_cols.append(col)

    if qc_fusion in {"pooled", "pooled_interaction"} and groups:
        for group_name, columns in groups.items():
            values = np.stack([_column_values(qc_df, col) for col in columns], axis=1)
            mean = values.mean(axis=1, keepdims=True)
            std = values.std(axis=1, keepdims=True)
            minv = values.min(axis=1, keepdims=True)
            maxv = values.max(axis=1, keepdims=True)
            features.extend([mean, std, minv, maxv])
            feature_cols.extend(
                [
                    f"qc_{group_name}_mean",
                    f"qc_{group_name}_std",
                    f"qc_{group_name}_min",
                    f"qc_{group_name}_max",
                ]
            )

    if qc_fusion in {"interaction", "pooled_interaction", "raw_interaction"}:
        qn = _column_values(qc_df, "q(N):metal2")
        qn_plus = _column_values(qc_df, "q(N+1):metal2")
        qn_minus = _column_values(qc_df, "q(N-1):metal2")
        f_plus = _column_values(qc_df, "f+:metal2")
        f_minus = _column_values(qc_df, "f-:metal2")
        f_zero = _column_values(qc_df, "f0:metal2")
        cdd = _column_values(qc_df, "CDD:metal2")
        s_plus = _column_values(qc_df, "s+:metal2")
        s_minus = _column_values(qc_df, "s-:metal2")
        vertical_ip = _column_values(qc_df, "Vertical_IP_metal2")
        vertical_ea = _column_values(qc_df, "Vertical_EA_metal2")
        electrophilicity = _column_values(qc_df, "Electrophilicity_index_metal2")
        nucleophilicity = _column_values(qc_df, "Nucleophilicity_index_metal2")

        charge_delta_plus = (qn_plus - qn)[:, None]
        charge_delta_minus = (qn - qn_minus)[:, None]
        fukui_balance = (f_plus - f_minus)[:, None]
        fukui_sum = (f_plus + f_minus)[:, None]
        softness_balance = (s_plus - s_minus)[:, None]
        ip_ea_gap = (vertical_ip - vertical_ea)[:, None]
        electro_nuc_balance = (electrophilicity - nucleophilicity)[:, None]
        q_charge_fplus = (qn * f_plus)[:, None]
        q_charge_fminus = (qn * f_minus)[:, None]
        cdd_fukui_balance = (cdd * (f_plus - f_minus))[:, None]
        fzero_charge = (f_zero * qn)[:, None]

        features.extend(
            [
                charge_delta_plus,
                charge_delta_minus,
                fukui_balance,
                fukui_sum,
                softness_balance,
                ip_ea_gap,
                electro_nuc_balance,
                q_charge_fplus,
                q_charge_fminus,
                cdd_fukui_balance,
                fzero_charge,
            ]
        )
        feature_cols.extend(
            [
                "qc_charge_delta_plus",
                "qc_charge_delta_minus",
                "qc_fukui_balance",
                "qc_fukui_sum",
                "qc_softness_balance",
                "qc_ip_ea_gap",
                "qc_electro_nuc_balance",
                "qc_charge_fplus",
                "qc_charge_fminus",
                "qc_cdd_fukui_balance",
                "qc_fzero_charge",
            ]
        )

    if not features:
        return np.zeros((len(df), 0), dtype=np.float32), [], groups

    feat_mat = np.concatenate(features, axis=1).astype(np.float32)
    return feat_mat, feature_cols, groups


def _load_qc_weight_map(path: Optional[str]) -> Dict[str, float]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    entries = payload.get("feature_importance", [])
    weight_map: Dict[str, float] = {}
    for entry in entries:
        name = entry.get("feature") or entry.get("name")
        if not name:
            continue
        weight_map[str(name)] = float(entry.get("delta_mae", 0.0))
    return weight_map


def _compute_qc_feature_weights(
    feature_cols: Sequence[str],
    weight_map: Dict[str, float],
    mode: str = "positive",
    scale: float = 0.5,
    min_weight: float = 0.25,
    max_weight: float = 2.0,
) -> np.ndarray:
    if not feature_cols or not weight_map:
        return np.ones((len(feature_cols),), dtype=np.float32)
    mode = (mode or "positive").lower()
    deltas = np.array([weight_map.get(name, 0.0) for name in feature_cols], dtype=np.float32)
    if mode == "positive":
        deltas = np.maximum(deltas, 0.0)
    elif mode == "signed":
        pass
    elif mode == "none":
        return np.ones((len(feature_cols),), dtype=np.float32)
    else:
        raise ValueError(f"Unknown qc_weight_mode: {mode}")
    max_abs = float(np.max(np.abs(deltas))) if deltas.size else 0.0
    if max_abs <= 1e-12:
        return np.ones((len(feature_cols),), dtype=np.float32)
    scaled = deltas / max_abs
    weights = 1.0 + scaled * float(scale)
    weights = np.clip(weights, min_weight, max_weight)
    return weights.astype(np.float32)


def _build_node_qc(
    zs: Sequence[int],
    coords: Sequence[Sequence[float]],
    qc_feat: Optional[np.ndarray],
    qc_node_mode: str,
    coordination_features: str = "none",
    coordination_node_scale: float = 1.0,
) -> np.ndarray:
    mode = (qc_node_mode or "none").lower()
    if qc_feat is None:
        qc_feat = np.zeros((0,), dtype=np.float32)
    qc_feat = np.asarray(qc_feat, dtype=np.float32).reshape(-1)
    if mode == "none" or qc_feat.size == 0:
        qc_node = np.zeros((len(zs), 0), dtype=np.float32)
    elif mode == "all":
        if len(zs) == 0:
            qc_node = np.zeros((0, qc_feat.size), dtype=np.float32)
        else:
            qc_node = np.repeat(qc_feat.reshape(1, -1), len(zs), axis=0)
    elif mode == "metal":
        qc_node = map_atomic_qc_to_nodes(zs, qc_feat)
    else:
        raise ValueError(f"Unknown qc_node_mode: {qc_node_mode}")
    coordination_node = _build_coordination_node_features(
        zs,
        coords,
        mode=coordination_features,
    )
    if coordination_node.size and coordination_node_scale != 1.0:
        coordination_node = coordination_node * float(coordination_node_scale)
    if qc_node.size and coordination_node.size:
        return np.concatenate([qc_node, coordination_node], axis=1)
    if qc_node.size:
        return qc_node
    if coordination_node.size:
        return coordination_node
    return np.zeros((len(zs), 0), dtype=np.float32)


def _build_bond_adjacency(
    zs: Sequence[int],
    coords: Sequence[Sequence[float]],
    scale: float = 1.22,
    min_cutoff: float = 0.9,
) -> List[List[int]]:
    adjacency: List[List[int]] = [[] for _ in range(len(zs))]
    if len(zs) <= 1 or not coords:
        return adjacency
    pos = np.asarray(coords, dtype=np.float32)
    dists = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    for i in range(len(zs)):
        radius_i = _COVALENT_RADIUS_BY_Z.get(int(zs[i]), 0.77)
        for j in range(i + 1, len(zs)):
            radius_j = _COVALENT_RADIUS_BY_Z.get(int(zs[j]), 0.77)
            cutoff = max(float(min_cutoff), float(scale) * (radius_i + radius_j))
            if float(dists[i, j]) <= cutoff:
                adjacency[i].append(j)
                adjacency[j].append(i)
    return adjacency


def _build_coordination_node_features(
    zs: Sequence[int],
    coords: Sequence[Sequence[float]],
    mode: str = "none",
) -> np.ndarray:
    mode = (mode or "none").lower()
    if mode == "none":
        return np.zeros((len(zs), 0), dtype=np.float32)
    if mode != "potency":
        raise ValueError(f"Unknown coordination feature mode: {mode}")
    if not zs:
        return np.zeros((0, len(_COORDINATION_FEATURE_COLS)), dtype=np.float32)

    adjacency = _build_bond_adjacency(zs, coords)
    features = np.zeros((len(zs), len(_COORDINATION_FEATURE_COLS)), dtype=np.float32)
    base_potency = {7: 1.00, 8: 0.82, 16: 1.12}
    hetero_set = {7, 8, 16}
    carbon_set = {6}
    phosphorus_set = {15}

    for idx, z in enumerate(zs):
        atomic_number = int(z)
        is_n = 1.0 if atomic_number == 7 else 0.0
        is_o = 1.0 if atomic_number == 8 else 0.0
        is_s = 1.0 if atomic_number == 16 else 0.0
        active = 1.0 if atomic_number in hetero_set else 0.0
        if not active:
            continue

        neighbors = adjacency[idx]
        neighbor_z = [int(zs[j]) for j in neighbors]
        heavy_degree = sum(1 for item in neighbor_z if item != 1)
        hetero_neighbors = sum(1 for item in neighbor_z if item in hetero_set)
        carbon_neighbors = sum(1 for item in neighbor_z if item in carbon_set)
        phosphorus_neighbors = sum(1 for item in neighbor_z if item in phosphorus_set)
        terminal_bonus = 0.08 if heavy_degree <= 1 else 0.0
        crowding_penalty = 0.07 * max(0, heavy_degree - 2)
        hetero_bonus = 0.10 * hetero_neighbors
        carbon_bonus = 0.05 * carbon_neighbors if atomic_number in {8, 16} else 0.03 * carbon_neighbors
        phosphorus_bonus = 0.06 * phosphorus_neighbors if atomic_number == 16 else 0.0
        potency = base_potency[atomic_number] + terminal_bonus + hetero_bonus + carbon_bonus + phosphorus_bonus - crowding_penalty
        potency = float(np.clip(potency, 0.0, 1.5))
        hetero_ratio = float(hetero_neighbors / max(1, heavy_degree))

        features[idx] = np.asarray(
            [active, is_n, is_o, is_s, potency, hetero_ratio],
            dtype=np.float32,
        )
    return features


def _coerce_numeric_columns(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == object:
            numeric = pd.to_numeric(df[col], errors="coerce")
            ratio = numeric.notna().mean()
            if ratio >= threshold:
                df[col] = numeric
    return df


def _read_table_file(path: str, *, header: int = 0) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, header=header)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, header=header, sep="\t")
    return pd.read_excel(path, header=header)


def _load_mlb_dataframe(excel_path: str, allow_missing_target: bool = False) -> Tuple[pd.DataFrame, Optional[str]]:
    for header in (0, 1):
        df = _read_table_file(excel_path, header=header)
        df = _coerce_numeric_columns(df)
        try:
            target_col = _select_target_column(df)
        except ValueError:
            if not allow_missing_target:
                continue
            target_col = None
        if all(
            any(col in df.columns for col in _BORYLATION_XYZ_COLUMN_CANDIDATES[key])
            for key in _BORYLATION_XYZ_COLUMN_CANDIDATES
        ):
            return df, target_col
    if allow_missing_target:
        raise ValueError(
            "Unsupported ml-borylation inference format. Expect xyz columns "
            "xyz-cat-ref/xyz-cat-ar, xyz-r1-ref/xyz-r1-ar, xyz-p1-ref/xyz-p1-ar."
        )
    raise ValueError(
        "Unsupported ml-borylation format. Expect target column like 'yield %' and xyz columns "
        "xyz-cat-ref/xyz-cat-ar, xyz-r1-ref/xyz-r1-ar, xyz-p1-ref/xyz-p1-ar."
    )


def _as_float32_vector(values: object) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError(f"Expected a 1D normalization vector, got shape {array.shape}.")
    return array.copy()


def _normalization_stats_from_mapping(stats: Mapping[str, object]) -> Tuple[np.ndarray, np.ndarray, np.float32, np.float32]:
    feat_mean = _as_float32_vector(stats["feat_mean"])
    feat_std = _as_float32_vector(stats["feat_std"])
    if feat_mean.shape != feat_std.shape:
        raise ValueError(
            f"Normalization feat_mean/feat_std shape mismatch: {feat_mean.shape} vs {feat_std.shape}."
        )
    y_mean = np.float32(stats.get("y_mean", 0.0))
    y_std = np.float32(stats.get("y_std", 1.0))
    if not np.isfinite(float(y_std)) or float(y_std) < 1e-8:
        y_std = np.float32(1.0)
    return feat_mean, feat_std, y_mean, y_std


def _load_mlcb_dataframe(excel_path: str) -> pd.DataFrame:
    for header in (1, 0):
        df = _read_table_file(excel_path, header=header)
        if "yield" not in df.columns:
            continue
        if all(
            any(col in df.columns for col in _XYZ_COLUMN_CANDIDATES[key])
            for key in _XYZ_COLUMN_CANDIDATES
        ):
            return df
    raise ValueError(
        "Unsupported ml-cb format. Expect 'yield' and xyz columns like "
        "xyz-cat-ref/xyz-cat-ar, xyz-r1-ref/xyz-r1-ar, xyz-r2-ref/xyz-r2-ar."
    )


def _build_knn_edges(coords: List[List[float]], k: int = 8) -> torch.Tensor:
    if not coords:
        return torch.zeros((2, 0), dtype=torch.long)

    pos = np.asarray(coords, dtype=np.float32)
    n = pos.shape[0]
    if n <= 1:
        return torch.zeros((2, 0), dtype=torch.long)

    k_eff = min(k, n - 1)
    dists = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    knn = np.argpartition(dists, kth=range(1, k_eff + 1), axis=1)[:, 1 : k_eff + 1]
    src = np.repeat(np.arange(n), k_eff)
    dst = knn.reshape(-1)
    edge_index = np.stack([src, dst], axis=0)
    return torch.from_numpy(edge_index.astype(np.int64))


def _edge_distances(coords: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    if edge_index.size == 0:
        return np.zeros((0,), dtype=np.float32)
    src = edge_index[0]
    dst = edge_index[1]
    diff = coords[src] - coords[dst]
    return np.linalg.norm(diff, axis=1).astype(np.float32)


def _build_inter_edges(
    pos_a: np.ndarray,
    pos_b: np.ndarray,
    offset_a: int,
    offset_b: int,
    cutoff: float,
) -> Tuple[List[Tuple[int, int]], List[float]]:
    if pos_a.size == 0 or pos_b.size == 0:
        return [], []
    diff = pos_a[:, None, :] - pos_b[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    idx = np.where(dist < cutoff)
    edges: List[Tuple[int, int]] = []
    dists: List[float] = []
    for i, j in zip(idx[0], idx[1]):
        d = float(dist[i, j])
        edges.append((offset_a + i, offset_b + j))
        edges.append((offset_b + j, offset_a + i))
        dists.append(d)
        dists.append(d)
    return edges, dists


def _build_combined_graph(
    cat_z: List[int],
    cat_pos: List[List[float]],
    r1_z: List[int],
    r1_pos: List[List[float]],
    r2_z: List[int],
    r2_pos: List[List[float]],
    intra_k: int = 8,
    inter_cutoff: float = 5.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    z_all = np.asarray(cat_z + r1_z + r2_z, dtype=np.int64)
    pos_all = np.asarray(cat_pos + r1_pos + r2_pos, dtype=np.float32)
    node_type = np.asarray(
        [0] * len(cat_z) + [1] * len(r1_z) + [2] * len(r2_z),
        dtype=np.int64,
    )

    offsets = [0, len(cat_z), len(cat_z) + len(r1_z)]

    edge_list: List[Tuple[int, int]] = []
    edge_type: List[int] = []
    pair_type: List[int] = []
    edge_dist: List[float] = []

    # Intra-molecule edges
    for mol_idx, (zs, pos, offset) in enumerate(
        ((cat_z, cat_pos, offsets[0]), (r1_z, r1_pos, offsets[1]), (r2_z, r2_pos, offsets[2]))
    ):
        if len(zs) == 0:
            continue
        edge_index = _build_knn_edges(pos, k=intra_k).numpy()
        if edge_index.size == 0:
            continue
        dist = _edge_distances(np.asarray(pos, dtype=np.float32), edge_index)
        for (src, dst), d in zip(edge_index.T, dist):
            edge_list.append((offset + int(src), offset + int(dst)))
            edge_type.append(0)
            pair_type.append(mol_idx)  # 0 cat-cat, 1 r1-r1, 2 r2-r2
            edge_dist.append(float(d))

    # Inter-molecule edges
    pairs = [
        (cat_pos, r1_pos, offsets[0], offsets[1], 3),
        (cat_pos, r2_pos, offsets[0], offsets[2], 4),
        (r1_pos, r2_pos, offsets[1], offsets[2], 5),
    ]
    for pos_a, pos_b, off_a, off_b, pair_code in pairs:
        edges, dists = _build_inter_edges(
            np.asarray(pos_a, dtype=np.float32),
            np.asarray(pos_b, dtype=np.float32),
            off_a,
            off_b,
            inter_cutoff,
        )
        if not edges:
            continue
        edge_list.extend(edges)
        edge_type.extend([1] * len(edges))
        pair_type.extend([pair_code] * len(edges))
        edge_dist.extend(dists)

    if not edge_list:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 3), dtype=torch.float32)
    else:
        edge_index = torch.tensor(np.array(edge_list).T, dtype=torch.long)
        edge_attr = torch.tensor(
            np.stack([edge_type, pair_type, edge_dist], axis=1),
            dtype=torch.float32,
        )

    return (
        torch.tensor(z_all, dtype=torch.long),
        torch.tensor(pos_all, dtype=torch.float32),
        torch.tensor(node_type, dtype=torch.long),
        edge_index,
        edge_attr,
    )


def _symbol_from_z(z: int) -> str:
    if 0 <= int(z) < len(_PERIODIC_TABLE):
        return _PERIODIC_TABLE[int(z)]
    return "X"


def _reaction_center_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)))


def _build_atomic_interaction_pairs(
    cat_z: Sequence[int],
    cat_pos: Sequence[Sequence[float]],
    partner_z: Sequence[int],
    partner_pos: Sequence[Sequence[float]],
    relay_cutoff: float = _INTERACTION_RELAY_CUTOFF,
    steric_cutoff: float = _INTERACTION_STERIC_CUTOFF,
    partner_component: str = "r1",
    mode: str = _INTERACTION_PAIR_MODE_CLASSIC,
) -> Tuple[torch.Tensor, torch.Tensor]:
    feature_dim = _interaction_pair_feature_dim_for_mode(mode)
    if not cat_z or not partner_z:
        return (
            torch.zeros((2, 0), dtype=torch.long),
            torch.zeros((0, feature_dim), dtype=torch.float32),
        )
    cat_pos_arr = np.asarray(cat_pos, dtype=np.float32)
    partner_pos_arr = np.asarray(partner_pos, dtype=np.float32)
    dists = np.linalg.norm(cat_pos_arr[:, None, :] - partner_pos_arr[None, :, :], axis=-1)
    cat_stats = _reaction_center_stats(cat_z, cat_pos, "cat")
    partner_stats = _reaction_center_stats(partner_z, partner_pos, partner_component)
    cat_centroid = np.mean(cat_pos_arr, axis=0)
    partner_centroid = np.mean(partner_pos_arr, axis=0)
    metal_idx = cat_stats.get("metal_idx")
    metal_coord = cat_pos_arr[int(metal_idx)] if metal_idx is not None and int(metal_idx) < len(cat_pos_arr) else cat_centroid
    indices: List[List[int]] = []
    features: List[np.ndarray] = []
    steric_cutoff = float(steric_cutoff)
    relay_cutoff = float(relay_cutoff)
    mode = str(mode or _INTERACTION_PAIR_MODE_CLASSIC).lower()
    for cat_idx, cat_atomic_num in enumerate(cat_z):
        cat_symbol = _symbol_from_z(cat_atomic_num)
        cat_is_metal = 1.0 if cat_symbol in _REACTION_CENTER_METALS else 0.0
        cat_donor = _REACTION_CENTER_DONOR_WEIGHTS.get(cat_symbol, 0.0)
        for partner_idx, partner_atomic_num in enumerate(partner_z):
            distance = float(dists[cat_idx, partner_idx])
            if distance > steric_cutoff:
                continue
            partner_symbol = _symbol_from_z(partner_atomic_num)
            partner_is_hetero = 1.0 if partner_symbol in _INTERACTION_HETERO_ATOMS else 0.0
            partner_donor = _REACTION_CENTER_DONOR_WEIGHTS.get(partner_symbol, 0.0)
            relay_active = 1.0 if (distance <= relay_cutoff and cat_is_metal > 0.0 and partner_is_hetero > 0.0) else 0.0
            steric_active = 1.0
            ligand_shell = 1.0 if (distance > relay_cutoff and distance <= steric_cutoff) else 0.0
            feature_row: List[float] = [
                distance,
                distance / steric_cutoff,
                relay_active,
                steric_active,
                cat_is_metal,
                partner_is_hetero + ligand_shell,
            ]
            if mode == _INTERACTION_PAIR_MODE_DENSE:
                cat_coord = cat_pos_arr[cat_idx]
                partner_coord = partner_pos_arr[partner_idx]
                pair_axis = _reaction_center_unit(partner_coord - cat_coord)
                cat_axis = _reaction_center_unit(cat_coord - metal_coord if cat_is_metal > 0.0 else cat_coord - cat_centroid)
                partner_axis = _reaction_center_unit(partner_coord - partner_centroid)
                approach_cos = 0.5 * (
                    abs(float(np.dot(cat_axis, pair_axis))) + abs(float(np.dot(partner_axis, -pair_axis)))
                )
                donor_alignment = _clip01(1.0 - min(abs(cat_donor - partner_donor), 1.0))
                cat_flux = float(cat_stats.get("electronic_flux", 0.0))
                partner_flux = float(partner_stats.get("electronic_flux", 0.0))
                late_metal = float(cat_stats.get("late_metal", 0.0))
                electronic_overlap = _clip01(
                    0.18
                    + 0.24 * relay_active
                    + 0.24 * cat_flux
                    + 0.18 * partner_flux
                    + 0.08 * donor_alignment
                    + 0.08 * approach_cos
                    + 0.12 * late_metal
                    - 0.10 * (distance / steric_cutoff)
                )
                steric_drag = _clip01(
                    0.10
                    + 0.22 * float(cat_stats.get("steric_pressure", 0.0))
                    + 0.26 * float(partner_stats.get("steric_pressure", 0.0))
                    + 0.18 * ligand_shell
                    + 0.14 * float(partner_stats.get("halide_fraction", 0.0))
                    + 0.10 * (distance / steric_cutoff)
                )
                coordination_proxy = _clip01(
                    0.12
                    + 0.30 * float(partner_stats.get("hetero_fraction", 0.0))
                    + 0.22 * float(partner_stats.get("donor_density", 0.0))
                    + 0.18 * float(cat_stats.get("donor_density", 0.0))
                    + 0.08 * partner_is_hetero
                    + 0.10 * late_metal
                )
                shell_match = _clip01(0.42 * partner_is_hetero + 0.30 * (1.0 - ligand_shell) + 0.28 * approach_cos)
                late_metal_relay = _clip01(
                    late_metal * (0.40 * electronic_overlap + 0.35 * coordination_proxy + 0.25 * donor_alignment)
                )
                feature_row.extend(
                    [
                        cat_flux,
                        partner_flux,
                        donor_alignment,
                        approach_cos,
                        late_metal,
                        electronic_overlap,
                        steric_drag,
                        coordination_proxy,
                        shell_match,
                        late_metal_relay,
                    ]
                )
            features.append(np.asarray(feature_row, dtype=np.float32))
            indices.append([int(cat_idx), int(partner_idx)])
    if not indices:
        return (
            torch.zeros((2, 0), dtype=torch.long),
            torch.zeros((0, feature_dim), dtype=torch.float32),
        )
    return (
        torch.tensor(np.asarray(indices, dtype=np.int64).T, dtype=torch.long),
        torch.tensor(np.stack(features, axis=0), dtype=torch.float32),
    )


def _reaction_center_unit(vector: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-8:
        return np.zeros_like(arr)
    return arr / norm


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _smooth_transition(value: float, center: float, width: float, *, descending: bool = False) -> float:
    scale = max(float(width), 1e-6)
    shifted = (float(center) - float(value)) if descending else (float(value) - float(center))
    return float(1.0 / (1.0 + np.exp(-(shifted / scale))))


def _reaction_center_stats(
    zs: Sequence[int],
    coords: Sequence[Sequence[float]],
    component: str,
) -> Dict[str, object]:
    atoms = [(_symbol_from_z(z), np.asarray(coord, dtype=np.float32)) for z, coord in zip(zs, coords)]
    if not atoms:
        return {
            "metal_idx": None,
            "metal_symbol": None,
            "shell_atoms": [],
            "electronic_flux": 0.0,
            "steric_pressure": 0.0,
            "hetero_fraction": 0.0,
            "halide_fraction": 0.0,
            "aromatic_proxy": 0.0,
            "donor_density": 0.0,
            "late_metal": 0.0,
        }

    heavy_atoms = max(1, sum(1 for symbol, _ in atoms if symbol != "H"))
    carbon = sum(1 for symbol, _ in atoms if symbol == "C")
    hetero = sum(1 for symbol, _ in atoms if symbol in {"N", "O", "P", "S"})
    halides = sum(1 for symbol, _ in atoms if symbol in _REACTION_CENTER_HALIDES)
    aromatic_proxy = carbon / float(heavy_atoms)
    hetero_fraction = hetero / float(heavy_atoms)
    halide_fraction = halides / float(heavy_atoms)
    donor_density = sum(_REACTION_CENTER_DONOR_WEIGHTS.get(symbol, 0.0) for symbol, _ in atoms) / float(heavy_atoms)
    heavy_scale = min(1.0, heavy_atoms / 40.0)

    metal_idx: Optional[int] = None
    metal_symbol: Optional[str] = None
    shell_atoms: List[int] = []
    donor_shell = 0.0
    halide_shell = 0.0
    carbon_shell = 0.0
    for idx, (symbol, _) in enumerate(atoms):
        if symbol not in _REACTION_CENTER_METALS:
            continue
        if metal_symbol is None:
            metal_idx = idx
            metal_symbol = symbol
            if symbol in _REACTION_CENTER_LATE_METALS:
                break
    if metal_idx is not None:
        metal_coord = atoms[metal_idx][1]
        ranked = sorted(
            (
                (_reaction_center_distance(metal_coord, coord), idx, symbol)
                for idx, (symbol, coord) in enumerate(atoms)
                if idx != metal_idx
            ),
            key=lambda item: item[0],
        )
        for dist, idx, symbol in ranked[:6]:
            if dist <= 0.0:
                continue
            shell_atoms.append(int(idx))
            donor_shell += _REACTION_CENTER_DONOR_WEIGHTS.get(symbol, 0.0)
            if symbol in _REACTION_CENTER_HALIDES:
                halide_shell += 1.0
            if symbol == "C":
                carbon_shell += 1.0
        if shell_atoms:
            donor_shell /= float(len(shell_atoms))
            halide_shell /= float(len(shell_atoms))
            carbon_shell /= float(len(shell_atoms))

    late_metal = 1.0 if metal_symbol in _REACTION_CENTER_LATE_METALS else 0.0
    lmct_score = _clip01(0.22 + 0.34 * late_metal + 0.18 * donor_shell + 0.15 * aromatic_proxy - 0.24 * halide_shell)
    reductive_score = _clip01(0.20 + 0.28 * late_metal + 0.18 * donor_density + 0.16 * carbon_shell - 0.18 * halide_shell)
    oxidative_friction = _clip01(0.18 + 0.35 * halide_shell + 0.12 * hetero_fraction - 0.14 * donor_shell)
    electronic_flux = _clip01(0.52 * lmct_score + 0.32 * reductive_score + 0.08 * heavy_scale - 0.18 * oxidative_friction)
    steric_pressure = _clip01(0.24 * heavy_scale + 0.24 * carbon_shell + 0.18 * halide_fraction + 0.12 * halide_shell)
    if component != "cat":
        electronic_flux = _clip01(
            0.38 + 0.27 * hetero_fraction + 0.22 * aromatic_proxy + 0.10 * halide_fraction - 0.08 * heavy_scale
        )
        steric_pressure = _clip01(0.16 + 0.18 * heavy_scale + 0.12 * halide_fraction)

    return {
        "metal_idx": metal_idx,
        "metal_symbol": metal_symbol,
        "shell_atoms": shell_atoms,
        "electronic_flux": float(electronic_flux),
        "steric_pressure": float(steric_pressure),
        "hetero_fraction": float(hetero_fraction),
        "halide_fraction": float(halide_fraction),
        "aromatic_proxy": float(aromatic_proxy),
        "donor_density": float(donor_density),
        "late_metal": float(late_metal),
    }


def _electronic_frontier_score(
    cat_stats: Mapping[str, object],
    r1_stats: Mapping[str, object],
    r2_stats: Mapping[str, object],
    temperature_c: float,
) -> float:
    metal = str(cat_stats.get("metal_symbol") or "")
    if metal not in {"Ru", "Ir"}:
        return 0.0
    ru_flux = _smooth_transition(float(r1_stats.get("electronic_flux", 0.0)), 0.58, 0.035) if metal == "Ru" else 0.0
    hetero_low_aromatic = (
        _smooth_transition(float(r1_stats.get("hetero_fraction", 0.0)), 0.28, 0.04)
        * _smooth_transition(float(r1_stats.get("aromatic_proxy", 0.0)), 0.72, 0.05, descending=True)
    )
    ir_low_r2_hetero = (
        _smooth_transition(float(r2_stats.get("hetero_fraction", 0.0)), 0.18, 0.035, descending=True)
        if metal == "Ir"
        else 0.0
    )
    coordination_hot = max(
        _smooth_transition(float(r1_stats.get("hetero_fraction", 0.0)), 0.20, 0.04),
        _smooth_transition(float(r2_stats.get("hetero_fraction", 0.0)), 0.20, 0.04),
    )
    coordination_temp = coordination_hot * _smooth_transition(float(temperature_c), 45.0, 4.0)
    complement = 1.0
    for component in (ru_flux, hetero_low_aromatic, ir_low_r2_hetero, coordination_temp):
        complement *= 1.0 - _clip01(float(component))
    return _clip01(1.0 - complement)


def _kinetic_relay_focus_flag(
    cat_stats: Mapping[str, object],
    r1_stats: Mapping[str, object],
    r2_stats: Mapping[str, object],
    temperature_c: float,
) -> float:
    metal = str(cat_stats.get("metal_symbol") or "")
    if metal not in {"Ru", "Ir"}:
        return 0.0
    ru_flux = metal == "Ru" and float(r1_stats.get("electronic_flux", 0.0)) >= 0.58
    hetero_low_aromatic = (
        float(r1_stats.get("hetero_fraction", 0.0)) >= 0.28
        and float(r1_stats.get("aromatic_proxy", 0.0)) <= 0.72
    )
    ir_low_r2_hetero = metal == "Ir" and float(r2_stats.get("hetero_fraction", 0.0)) <= 0.18
    coordination_hot = (
        float(r1_stats.get("hetero_fraction", 0.0)) >= 0.20
        or float(r2_stats.get("hetero_fraction", 0.0)) >= 0.20
    )
    temperature_hot = float(temperature_c) >= 45.0
    return 1.0 if (ru_flux or hetero_low_aromatic or ir_low_r2_hetero or (coordination_hot and temperature_hot)) else 0.0


def _reaction_center_candidates(
    zs: Sequence[int],
    coords: Sequence[Sequence[float]],
    component: str,
    stats: Mapping[str, object],
    limit: int = 4,
) -> List[int]:
    if not zs:
        return [0]
    atoms = [(_symbol_from_z(z), np.asarray(coord, dtype=np.float32)) for z, coord in zip(zs, coords)]
    centroid = np.mean(np.stack([coord for _, coord in atoms], axis=0), axis=0)
    scored: List[Tuple[float, int]] = []
    for idx, (symbol, coord) in enumerate(atoms):
        donor = _REACTION_CENTER_DONOR_WEIGHTS.get(symbol, 0.0)
        hetero = 1.0 if symbol in {"N", "O", "P", "S"} else 0.0
        carbon = 1.0 if symbol == "C" else 0.0
        shell_bonus = 1.0 if idx in (stats.get("shell_atoms") or []) else 0.0
        centroid_dist = float(np.linalg.norm(coord - centroid))
        if component == "cat" and idx == stats.get("metal_idx"):
            score = 10.0
        elif component == "cat":
            score = 4.0 * shell_bonus + 2.0 * donor + 0.6 * hetero + 0.2 * carbon - 0.05 * centroid_dist
        else:
            score = 3.0 * donor + 1.0 * hetero + 0.4 * carbon + 0.05 * centroid_dist
        scored.append((score, idx))
    scored.sort(reverse=True)
    chosen: List[int] = []
    if component == "cat" and stats.get("metal_idx") is not None:
        chosen.append(int(stats["metal_idx"]))
    for _, idx in scored:
        if idx not in chosen:
            chosen.append(int(idx))
        if len(chosen) >= limit:
            break
    return chosen[:limit] if chosen else [0]


def _reaction_center_pair_features(
    cat_idx: int,
    partner_idx: int,
    cat_coords: Sequence[Sequence[float]],
    partner_coords: Sequence[Sequence[float]],
    cat_zs: Sequence[int],
    partner_zs: Sequence[int],
    cat_stats: Mapping[str, object],
    partner_stats: Mapping[str, object],
    partner_component: str,
) -> Tuple[np.ndarray, float, float]:
    cat_coord = np.asarray(cat_coords[cat_idx], dtype=np.float32)
    partner_coord = np.asarray(partner_coords[partner_idx], dtype=np.float32)
    dist = _reaction_center_distance(cat_coord, partner_coord)
    dist_norm = min(dist / 6.0, 1.5)
    proximity = float(np.exp(-dist / 2.5))
    pair_is_r1 = 1.0 if partner_component == "r1" else 0.0
    pair_is_r2 = 1.0 if partner_component == "r2" else 0.0
    cat_symbol = _symbol_from_z(cat_zs[cat_idx]) if cat_zs else "X"
    partner_symbol = _symbol_from_z(partner_zs[partner_idx]) if partner_zs else "X"
    cat_donor = _REACTION_CENTER_DONOR_WEIGHTS.get(cat_symbol, 0.0)
    partner_donor = _REACTION_CENTER_DONOR_WEIGHTS.get(partner_symbol, 0.0)

    metal_idx = cat_stats.get("metal_idx")
    if metal_idx is not None:
        metal_coord = np.asarray(cat_coords[int(metal_idx)], dtype=np.float32)
        cat_axis = _reaction_center_unit(cat_coord - metal_coord if int(metal_idx) != cat_idx else cat_coord - np.mean(np.asarray(cat_coords, dtype=np.float32), axis=0))
    else:
        cat_axis = _reaction_center_unit(cat_coord - np.mean(np.asarray(cat_coords, dtype=np.float32), axis=0))
    partner_centroid = np.mean(np.asarray(partner_coords, dtype=np.float32), axis=0)
    partner_axis = _reaction_center_unit(partner_coord - partner_centroid)
    pair_axis = _reaction_center_unit(partner_coord - cat_coord)
    approach_cos = 0.5 * (abs(float(np.dot(cat_axis, pair_axis))) + abs(float(np.dot(partner_axis, -pair_axis))))

    cat_flux = float(cat_stats.get("electronic_flux", 0.0))
    partner_flux = float(partner_stats.get("electronic_flux", 0.0))
    cat_steric = float(cat_stats.get("steric_pressure", 0.0))
    partner_steric = float(partner_stats.get("steric_pressure", 0.0))
    cat_hetero = float(cat_stats.get("hetero_fraction", 0.0))
    partner_hetero = float(partner_stats.get("hetero_fraction", 0.0))
    cat_density = float(cat_stats.get("donor_density", 0.0))
    partner_density = float(partner_stats.get("donor_density", 0.0))
    late_metal = float(cat_stats.get("late_metal", 0.0))

    alignment = _clip01(
        0.18
        + 0.28 * cat_flux
        + 0.28 * partner_flux
        + 0.10 * (1.0 - min(abs(cat_donor - partner_donor), 1.0))
        + 0.10 * (1.0 - min(abs(cat_density - partner_density), 1.0))
        + 0.08 * proximity
        + 0.08 * approach_cos
        - 0.08 * min(dist_norm, 1.0)
    )
    blockade = _clip01(
        0.12
        + 0.24 * cat_steric
        + 0.34 * partner_steric
        + 0.10 * partner_hetero
        + 0.08 * (1.0 - approach_cos)
        + 0.12 * (1.0 - proximity)
        + 0.08 * min(dist_norm, 1.0)
    )
    features = np.asarray(
        [
            dist_norm,
            proximity,
            pair_is_r1,
            pair_is_r2,
            cat_donor,
            partner_donor,
            cat_flux,
            partner_flux,
            cat_steric,
            partner_steric,
            cat_hetero,
            partner_hetero,
            cat_density,
            partner_density,
            approach_cos,
            late_metal,
        ],
        dtype=np.float32,
    )
    return features, float(alignment), float(blockade)


def _build_reaction_center_coupling(
    cat_z: Sequence[int],
    cat_pos: Sequence[Sequence[float]],
    r1_z: Sequence[int],
    r1_pos: Sequence[Sequence[float]],
    r2_z: Sequence[int],
    r2_pos: Sequence[Sequence[float]],
) -> Dict[str, torch.Tensor]:
    cat_stats = _reaction_center_stats(cat_z, cat_pos, "cat")
    r1_stats = _reaction_center_stats(r1_z, r1_pos, "r1")
    r2_stats = _reaction_center_stats(r2_z, r2_pos, "r2")
    cat_candidates = _reaction_center_candidates(cat_z, cat_pos, "cat", cat_stats)
    r1_candidates = _reaction_center_candidates(r1_z, r1_pos, "r1", r1_stats)
    r2_candidates = _reaction_center_candidates(r2_z, r2_pos, "r2", r2_stats)

    def _pair_block(
        partner_component: str,
        partner_candidates: Sequence[int],
        partner_z: Sequence[int],
        partner_pos: Sequence[Sequence[float]],
        partner_stats: Mapping[str, object],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[float], List[float]]:
        indices: List[List[int]] = []
        features: List[np.ndarray] = []
        alignments: List[float] = []
        blockades: List[float] = []
        for cat_idx in cat_candidates:
            for partner_idx in partner_candidates:
                pair_feat, alignment, blockade = _reaction_center_pair_features(
                    cat_idx,
                    partner_idx,
                    cat_pos,
                    partner_pos,
                    cat_z,
                    partner_z,
                    cat_stats,
                    partner_stats,
                    partner_component,
                )
                indices.append([int(cat_idx), int(partner_idx)])
                features.append(pair_feat)
                alignments.append(alignment)
                blockades.append(blockade)
        if not indices:
            return (
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros((0, _REACTION_CENTER_PAIR_FEATURE_DIM), dtype=torch.float32),
                [],
                [],
            )
        return (
            torch.tensor(np.asarray(indices, dtype=np.int64).T, dtype=torch.long),
            torch.tensor(np.stack(features, axis=0), dtype=torch.float32),
            alignments,
            blockades,
        )

    cat_r1_index, cat_r1_features, cat_r1_align, cat_r1_block = _pair_block("r1", r1_candidates, r1_z, r1_pos, r1_stats)
    cat_r2_index, cat_r2_features, cat_r2_align, cat_r2_block = _pair_block("r2", r2_candidates, r2_z, r2_pos, r2_stats)
    alignments = cat_r1_align + cat_r2_align
    blockades = cat_r1_block + cat_r2_block
    targets = torch.tensor(
        [
            max(alignments) if alignments else 0.0,
            max(blockades) if blockades else 0.0,
        ],
        dtype=torch.float32,
    )
    return {
        "cat_r1_index": cat_r1_index,
        "cat_r1_features": cat_r1_features,
        "cat_r2_index": cat_r2_index,
        "cat_r2_features": cat_r2_features,
        "targets": targets,
    }


def _coulomb_eigenvalues(zs: List[int], coords: List[List[float]], max_atoms: int) -> np.ndarray:
    if not zs or max_atoms == 0:
        return np.zeros((max_atoms,), dtype=np.float32)

    n = len(zs)
    pos = np.asarray(coords, dtype=np.float32)
    z = np.asarray(zs, dtype=np.float32)
    mat = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        mat[i, i] = 0.5 * (z[i] ** 2.4)
    if n > 1:
        dists = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
        for i in range(n):
            for j in range(i + 1, n):
                val = z[i] * z[j] / (dists[i, j] + 1e-8)
                mat[i, j] = val
                mat[j, i] = val
    eigvals = np.linalg.eigvalsh(mat)
    eigvals = np.sort(eigvals)[::-1]
    if eigvals.size < max_atoms:
        eigvals = np.pad(eigvals, (0, max_atoms - eigvals.size))
    else:
        eigvals = eigvals[:max_atoms]
    return eigvals.astype(np.float32)


def _flatten_geometry(zs: List[int], coords: List[List[float]], max_atoms: int) -> np.ndarray:
    if not zs or max_atoms == 0:
        return np.zeros((max_atoms * 4,), dtype=np.float32)
    atoms = sorted(zip(zs, coords), key=lambda x: (x[0], x[1][0], x[1][1], x[1][2]))
    arr = np.zeros((max_atoms, 4), dtype=np.float32)
    for i, (z, coord) in enumerate(atoms[:max_atoms]):
        arr[i, 0] = float(z)
        arr[i, 1:] = np.asarray(coord, dtype=np.float32)
    return arr.reshape(-1)


def _build_mock_literature(idx: int) -> dict:
    from llm.mock_data import MOCK_EXTRACTIONS
    from llm.schema import Interaction

    raw = MOCK_EXTRACTIONS[idx % len(MOCK_EXTRACTIONS)]
    interactions = [Interaction(**item) for item in raw.get("interactions", [])]
    return {
        "cat": interactions,
        "r1": interactions,
        "r2": interactions,
        "meta": {
            "overall_confidence": raw.get("overall_confidence"),
            "precision_level": raw.get("literature_precision_level"),
        },
    }


def _load_literature_cache(path: Optional[str]) -> Optional[Dict[int, dict]]:
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        return None
    if isinstance(payload, dict) and "samples" in payload:
        payload = payload["samples"]
    if not isinstance(payload, dict):
        return None
    cache: Dict[int, dict] = {}
    for key, value in payload.items():
        if key == "meta":
            continue
        try:
            idx = int(key)
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict):
            cache[idx] = value
    return cache


def _extract_llm_confidence(literature: Optional[dict]) -> float:
    if not literature:
        return 0.0
    meta = literature.get("meta") or {}
    overall = meta.get("overall_confidence", literature.get("overall_confidence"))
    confidences: List[float] = []
    for component in ("cat", "r1", "r2"):
        for inter in literature.get(component) or []:
            if isinstance(inter, dict):
                conf = inter.get("confidence")
            else:
                conf = getattr(inter, "confidence", None)
            if conf is None:
                continue
            try:
                confidences.append(float(conf))
            except (TypeError, ValueError):
                continue
    if overall is None:
        overall_val = float(np.mean(confidences)) if confidences else 0.0
    else:
        try:
            overall_val = float(overall)
        except (TypeError, ValueError):
            overall_val = 0.0
    if not np.isfinite(overall_val):
        overall_val = 0.0
    return float(min(max(overall_val, 0.0), 1.0))


def _extract_exact_covered_indicator(literature: Optional[Mapping[str, object]]) -> float:
    if not isinstance(literature, Mapping):
        return 0.0
    meta = literature.get("meta")
    meta = meta if isinstance(meta, Mapping) else {}
    assignment_kind = str(meta.get("coverage_assignment_kind") or "").strip().lower()
    follow_up_assignment_kind = str(meta.get("follow_up_assignment_kind") or "").strip().lower()
    if "propagation" in assignment_kind or "propagation" in follow_up_assignment_kind:
        return 0.0
    return 1.0


def _extract_llm_row_provenance(literature: Optional[Mapping[str, object]]) -> str:
    if not isinstance(literature, Mapping):
        return "uncovered"
    return "exact" if _extract_exact_covered_indicator(literature) > 0.5 else "propagated"


def _validate_llm_row_scale(raw_scale: object, *, label: str) -> float:
    try:
        scale = float(raw_scale)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite non-negative float; got {raw_scale!r}.") from exc
    if not np.isfinite(scale) or scale < 0.0:
        raise ValueError(f"{label} must be a finite non-negative float; got {raw_scale!r}.")
    return float(scale)


def _apply_llm_feature_overrides(
    llm_feat: np.ndarray,
    *,
    literature: Optional[Mapping[str, object]],
    llm_feature_cols: Sequence[str],
) -> np.ndarray:
    if llm_feat.size == 0 or not isinstance(literature, Mapping):
        return llm_feat
    meta = literature.get("meta")
    if not isinstance(meta, Mapping):
        return llm_feat
    zero_columns = meta.get("llm_zero_feature_columns")
    if not isinstance(zero_columns, Sequence) or isinstance(zero_columns, (str, bytes)):
        return llm_feat

    name_to_index = {name: idx for idx, name in enumerate(llm_feature_cols)}
    overridden = np.asarray(llm_feat, dtype=np.float32).copy()
    for column in zero_columns:
        if not isinstance(column, str):
            continue
        index = name_to_index.get(column)
        if index is None:
            continue
        overridden[index] = 0.0
    return overridden


def _parse_llm_semantic_group_scales(raw_scales: object) -> Dict[str, float]:
    if raw_scales is None:
        return {}
    if isinstance(raw_scales, Mapping):
        items = raw_scales.items()
    elif isinstance(raw_scales, str):
        stripped = raw_scales.strip()
        items = [] if not stripped else [tuple(stripped.split("=", 1))]
    elif isinstance(raw_scales, Sequence):
        items = []
        for value in raw_scales:
            if isinstance(value, str):
                stripped = value.strip()
                if stripped:
                    items.append(tuple(stripped.split("=", 1)))
    else:
        raise ValueError(
            "llm_semantic_group_scales must be a mapping, 'group=scale' string, or sequence of such strings."
        )

    parsed: Dict[str, float] = {}
    for raw_name, raw_scale in items:
        group_name = str(raw_name or "").strip()
        if not group_name:
            raise ValueError("LLM semantic group scale keys must be non-empty.")
        if isinstance(raw_scale, str) and "=" in raw_scale:
            raise ValueError(
                f"Invalid llm semantic group scale entry `{raw_name}={raw_scale}`. Expected `group=scale`."
            )
        try:
            scale = float(raw_scale)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid llm semantic group scale for `{group_name}`: {raw_scale!r}."
            ) from exc
        if not np.isfinite(scale) or scale < 0.0:
            raise ValueError(
                f"LLM semantic group scale for `{group_name}` must be finite and non-negative; got {raw_scale!r}."
            )
        parsed[group_name] = scale
    return parsed


def _parse_llm_semantic_feature_scales(raw_scales: object) -> Dict[str, float]:
    if raw_scales is None:
        return {}
    if isinstance(raw_scales, Mapping):
        items = raw_scales.items()
    elif isinstance(raw_scales, str):
        stripped = raw_scales.strip()
        items = [] if not stripped else [tuple(stripped.split("=", 1))]
    elif isinstance(raw_scales, Sequence):
        items = []
        for value in raw_scales:
            if isinstance(value, str):
                stripped = value.strip()
                if stripped:
                    items.append(tuple(stripped.split("=", 1)))
    else:
        raise ValueError(
            "llm_semantic_feature_scales must be a mapping, 'feature=scale' string, or sequence of such strings."
        )

    parsed: Dict[str, float] = {}
    for raw_name, raw_scale in items:
        feature_name = str(raw_name or "").strip()
        if not feature_name:
            raise ValueError("LLM semantic feature scale keys must be non-empty.")
        if isinstance(raw_scale, str) and "=" in raw_scale:
            raise ValueError(
                f"Invalid llm semantic feature scale entry `{raw_name}={raw_scale}`. Expected `feature=scale`."
            )
        try:
            scale = float(raw_scale)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid llm semantic feature scale for `{feature_name}`: {raw_scale!r}."
            ) from exc
        if not np.isfinite(scale) or scale < 0.0:
            raise ValueError(
                f"LLM semantic feature scale for `{feature_name}` must be finite and non-negative; got {raw_scale!r}."
            )
        parsed[feature_name] = scale
    return parsed


def _build_llm_semantic_group_scale_vector(
    llm_feature_cols: Sequence[str],
    group_scales: Mapping[str, float],
) -> np.ndarray:
    vector = np.ones((len(llm_feature_cols),), dtype=np.float32)
    if not llm_feature_cols or not group_scales:
        return vector

    from llm.features import llm_semantic_group_columns_from_feature_columns

    group_columns = llm_semantic_group_columns_from_feature_columns(llm_feature_cols)
    name_to_index = {name: idx for idx, name in enumerate(llm_feature_cols)}
    valid_groups = {name for name, columns in group_columns.items() if columns}
    for group_name, scale in group_scales.items():
        if group_name not in valid_groups:
            raise ValueError(
                f"Unknown llm semantic group `{group_name}` for feature scaling. "
                f"Known non-empty groups: {sorted(valid_groups)}"
            )
        indexes = [name_to_index[column] for column in group_columns[group_name] if column in name_to_index]
        vector[np.asarray(indexes, dtype=np.int64)] = float(scale)
    return vector


def _build_llm_semantic_feature_scale_vector(
    llm_feature_cols: Sequence[str],
    feature_scales: Mapping[str, float],
) -> np.ndarray:
    vector = np.ones((len(llm_feature_cols),), dtype=np.float32)
    if not llm_feature_cols or not feature_scales:
        return vector

    name_to_index = {name: idx for idx, name in enumerate(llm_feature_cols)}
    missing = sorted(feature_name for feature_name in feature_scales if feature_name not in name_to_index)
    if missing:
        raise ValueError(
            f"Unknown llm semantic feature(s) for feature scaling: {missing}. "
            f"Known features: {list(llm_feature_cols)}"
        )
    for feature_name, scale in feature_scales.items():
        vector[name_to_index[feature_name]] = float(scale)
    return vector


def _native_pair_safe_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = float(default)
    if not np.isfinite(number):
        return float(default)
    return float(number)


def _native_pair_clip01(value: object, default: float = 0.0) -> float:
    number = _native_pair_safe_float(value, default)
    if number < 0.0:
        return 0.0
    if number > 1.0:
        return 1.0
    return float(number)


def _native_pair_one_hot(label: str, allowed: Sequence[str]) -> np.ndarray:
    vector = np.zeros((len(allowed),), dtype=np.float32)
    try:
        index = allowed.index(label)
    except ValueError:
        if allowed:
            vector[-1] = 1.0
    else:
        vector[index] = 1.0
    return vector


def _native_pair_atom_symbol(atom_label: object) -> str:
    if not isinstance(atom_label, str):
        return "other"
    token = atom_label.split(":", 1)[0].strip()
    if not token:
        return "other"
    normalized = token[0].upper() + token[1:].lower()
    return normalized if normalized in _NATIVE_PAIR_ATOM_SYMBOLS[:-1] else "other"


def _build_native_pair_tensors(
    literature: Optional[Mapping[str, object]],
    field_profile: str = "full",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    summary = np.zeros((_NATIVE_PAIR_SUMMARY_FEATURE_DIM,), dtype=np.float32)
    tokens = np.zeros(
        (_NATIVE_PAIR_MAX_TOKENS, _NATIVE_PAIR_TOKEN_FEATURE_DIM),
        dtype=np.float32,
    )
    token_mask = np.zeros((_NATIVE_PAIR_MAX_TOKENS,), dtype=np.float32)
    profile_spec = native_pair_field_profile_spec(field_profile)
    if not isinstance(literature, Mapping):
        return summary, tokens, token_mask

    pair = literature.get("cat_r1_pair")
    if not isinstance(pair, Mapping):
        return summary, tokens, token_mask

    temperature = pair.get("temperature_modulation")
    temperature = temperature if isinstance(temperature, Mapping) else {}
    dominant = pair.get("dominant_interaction")
    dominant = dominant if isinstance(dominant, Mapping) else {}
    evidence_atoms = pair.get("evidence_atoms")
    evidence_atoms = evidence_atoms if isinstance(evidence_atoms, Mapping) else {}
    joint_interactions = pair.get("joint_interactions")
    joint_interactions = joint_interactions if isinstance(joint_interactions, list) else []
    structured_joint_interactions = [
        item for item in joint_interactions if isinstance(item, Mapping)
    ]
    structured_joint_interactions = sorted(
        structured_joint_interactions,
        key=lambda item: _native_pair_clip01(item.get("confidence"), 0.0),
        reverse=True,
    )

    catalyst_atoms = evidence_atoms.get("catalyst_atoms")
    catalyst_atoms = catalyst_atoms if isinstance(catalyst_atoms, list) else []
    reactant_atoms = evidence_atoms.get("reactant_atoms")
    reactant_atoms = reactant_atoms if isinstance(reactant_atoms, list) else []

    joint_confidences = [
        _native_pair_clip01(item.get("confidence"), 0.0)
        for item in structured_joint_interactions
    ]
    mean_joint_confidence = (
        sum(joint_confidences) / len(joint_confidences) if joint_confidences else 0.0
    )
    max_joint_confidence = max(joint_confidences) if joint_confidences else 0.0

    dominant_type = str(dominant.get("interaction_type") or "").strip()
    if not dominant_type and structured_joint_interactions:
        dominant_type = str(
            structured_joint_interactions[0].get("interaction_type") or ""
        ).strip()
    temperature_direction = str(temperature.get("direction") or "").strip()
    ranking = pair.get("ranking_signals")
    ranking = ranking if isinstance(ranking, Mapping) else {}
    total_candidates = max(
        1,
        int(round(_native_pair_safe_float(ranking.get("total_candidates"), default=1.0))),
    )
    compatibility_rank = min(
        total_candidates,
        max(
            1,
            int(round(_native_pair_safe_float(ranking.get("compatibility_rank"), default=float(total_candidates)))),
        ),
    )
    if total_candidates <= 1:
        compatibility_order_score = 1.0
    else:
        compatibility_order_score = 1.0 - (
            float(compatibility_rank - 1) / float(total_candidates - 1)
        )

    scalar_values = [
        _native_pair_clip01(pair.get("confidence"), 0.0),
        _native_pair_clip01(pair.get("pair_focus"), 0.0),
        _native_pair_clip01(pair.get("reaction_center_compatibility"), 0.0),
        _native_pair_clip01(pair.get("steric_accommodation"), 0.0),
        _native_pair_clip01(pair.get("electronic_complement"), 0.0),
        _native_pair_clip01(temperature.get("temperature_alignment"), 0.0),
        1.0 if bool(dominant.get("supports_major_path", False)) else 0.0,
        min(len(catalyst_atoms) / 3.0, 1.0),
        min(len(reactant_atoms) / 3.0, 1.0),
        min(len(structured_joint_interactions) / float(_NATIVE_PAIR_MAX_TOKENS), 1.0),
        float(mean_joint_confidence),
        float(max_joint_confidence),
        float(compatibility_order_score),
        _native_pair_clip01(ranking.get("pairwise_preference_margin"), 0.0),
        _native_pair_clip01(ranking.get("ee_order_confidence"), 0.0),
    ]
    summary[: len(scalar_values)] = np.asarray(scalar_values, dtype=np.float32)

    offset = len(scalar_values)
    summary[offset : offset + len(_NATIVE_PAIR_TEMPERATURE_DIRECTIONS)] = _native_pair_one_hot(
        temperature_direction if temperature_direction in _NATIVE_PAIR_TEMPERATURE_DIRECTIONS else "broadly_neutral",
        _NATIVE_PAIR_TEMPERATURE_DIRECTIONS,
    )
    offset += len(_NATIVE_PAIR_TEMPERATURE_DIRECTIONS)
    summary[offset : offset + len(_NATIVE_PAIR_INTERACTION_TYPES)] = _native_pair_one_hot(
        dominant_type if dominant_type in _NATIVE_PAIR_INTERACTION_TYPES else "van_der_waals",
        _NATIVE_PAIR_INTERACTION_TYPES,
    )

    for idx, interaction in enumerate(structured_joint_interactions[:_NATIVE_PAIR_MAX_TOKENS]):
        interaction_type = str(interaction.get("interaction_type") or "").strip()
        catalyst_atom = _native_pair_atom_symbol(interaction.get("catalyst_atom"))
        reactant_atom = _native_pair_atom_symbol(interaction.get("reactant_atom"))
        token_parts = [
            np.asarray(
                [_native_pair_clip01(interaction.get("confidence"), 0.0)],
                dtype=np.float32,
            ),
            _native_pair_one_hot(
                interaction_type if interaction_type in _NATIVE_PAIR_INTERACTION_TYPES else "van_der_waals",
                _NATIVE_PAIR_INTERACTION_TYPES,
            ),
            _native_pair_one_hot(catalyst_atom, _NATIVE_PAIR_ATOM_SYMBOLS),
            _native_pair_one_hot(reactant_atom, _NATIVE_PAIR_ATOM_SYMBOLS),
        ]
        tokens[idx] = np.concatenate(token_parts, axis=0)
        token_mask[idx] = 1.0
    summary *= np.asarray(profile_spec["summary_scale_vector"], dtype=np.float32)
    tokens *= np.asarray(profile_spec["token_scale_vector"], dtype=np.float32).reshape(1, -1)
    return summary, tokens, token_mask


def _build_native_pair_expert_tensors(
    literature: Optional[Mapping[str, object]],
    field_profile: str = "full",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    summaries = np.zeros(
        (_NATIVE_PAIR_MAX_EXPERTS, _NATIVE_PAIR_SUMMARY_FEATURE_DIM),
        dtype=np.float32,
    )
    tokens = np.zeros(
        (_NATIVE_PAIR_MAX_EXPERTS, _NATIVE_PAIR_MAX_TOKENS, _NATIVE_PAIR_TOKEN_FEATURE_DIM),
        dtype=np.float32,
    )
    token_masks = np.zeros((_NATIVE_PAIR_MAX_EXPERTS, _NATIVE_PAIR_MAX_TOKENS), dtype=np.float32)
    priors = np.zeros((_NATIVE_PAIR_MAX_EXPERTS,), dtype=np.float32)
    if not isinstance(literature, Mapping):
        return summaries, tokens, token_masks, priors

    expert_entries = literature.get("semantic_experts")
    expert_entries = expert_entries if isinstance(expert_entries, list) else []
    structured_entries = [entry for entry in expert_entries if isinstance(entry, Mapping)]
    if not structured_entries:
        return summaries, tokens, token_masks, priors

    max_experts = min(len(structured_entries), _NATIVE_PAIR_MAX_EXPERTS)
    raw_prior_values: List[float] = []
    for idx in range(max_experts):
        entry = structured_entries[idx]
        pair_payload = entry.get("cat_r1_pair")
        if isinstance(pair_payload, Mapping):
            literature_payload: Mapping[str, object] = {
                "cat_r1_pair": pair_payload,
            }
        else:
            literature_payload = entry
        summary, token_tensor, token_mask = _build_native_pair_tensors(
            literature_payload,
            field_profile=field_profile,
        )
        summaries[idx] = summary
        tokens[idx] = token_tensor
        token_masks[idx] = token_mask
        raw_prior_values.append(
            max(
                _native_pair_safe_float(
                    entry.get("prior_weight", entry.get("routing_prior", entry.get("confidence_prior"))),
                    default=0.0,
                ),
                0.0,
            )
        )

    if raw_prior_values:
        prior_array = np.asarray(raw_prior_values, dtype=np.float32)
        if float(prior_array.sum()) <= 1e-8:
            prior_array = np.full_like(prior_array, 1.0 / float(len(raw_prior_values)))
        else:
            prior_array = prior_array / float(prior_array.sum())
        priors[: len(raw_prior_values)] = prior_array
    return summaries, tokens, token_masks, priors


def _map_literature_entry(
    entry: Optional[dict],
    cat_smiles: str,
    r1_smiles: str,
    r2_smiles: str,
    mol_cache: Dict[str, object],
) -> Optional[dict]:
    if entry is None:
        return None
    from llm.mapping import map_interactions_to_indices

    mapped = dict(entry)
    cat_key = "cat" if "cat" in mapped else "cat_interactions"
    r1_key = "r1" if "r1" in mapped else "r1_interactions"
    r2_key = "r2" if "r2" in mapped else "r2_interactions"
    mapped["cat"] = map_interactions_to_indices(mapped.get(cat_key), cat_smiles, mol_cache)
    mapped["r1"] = map_interactions_to_indices(mapped.get(r1_key), r1_smiles, mol_cache)
    mapped["r2"] = map_interactions_to_indices(mapped.get(r2_key), r2_smiles, mol_cache)
    return mapped


class MLCBDataset(Dataset):
    def __init__(
        self,
        excel_path: str,
        split: str,
        seed: int = 42,
        val_fraction: float = 0.1,
        test_fraction: float = 0.1,
        max_samples: Optional[int] = None,
        include_numeric: bool = True,
        extra_numeric_cols: Optional[Sequence[str]] = None,
        exclude_numeric_cols: Optional[Sequence[str]] = None,
        use_coulomb: bool = True,
        use_geometry: bool = True,
        qc_fusion: str = "none",
        qc_scale: float = 1.0,
        qc_node_scale: float = 1.0,
        qc_global_scale: float = 1.0,
        qc_node_mode: str = "none",
        qc_weight_path: Optional[str] = None,
        qc_weight_mode: str = "positive",
        qc_weight_scale: float = 0.5,
        qc_weight_min: float = 0.25,
        qc_weight_max: float = 2.0,
        coordination_features: str = "none",
        coordination_node_scale: float = 1.0,
        build_combined_graph: bool = False,
        build_atomic_interaction_pairs: bool = False,
        interaction_pair_mode: str = _INTERACTION_PAIR_MODE_CLASSIC,
        build_reaction_center_coupling: bool = False,
        combined_k: int = 8,
        combined_cutoff: float = 5.0,
        group_cols: Optional[Sequence[str]] = None,
        rank_target_col: Optional[str] = None,
        target_col: str = "yield",
    ):
        df = _load_mlcb_dataframe(excel_path)
        cat_xyz_col = _select_column(df, _XYZ_COLUMN_CANDIDATES["cat"], "cat xyz")
        r1_xyz_col = _select_column(df, _XYZ_COLUMN_CANDIDATES["r1"], "r1 xyz")
        r2_xyz_col = _select_column(df, _XYZ_COLUMN_CANDIDATES["r2"], "r2 xyz")

        # Numeric features (exclude target)
        if target_col not in df.columns:
            raise ValueError(f"Missing target column: {target_col}")
        numeric_cols = []
        if include_numeric:
            numeric_cols = [
                c
                for c in df.columns
                if c != target_col and df[c].dtype != object and not _is_excluded_metadata_feature(c)
            ]
        excluded_numeric = {c for c in (exclude_numeric_cols or []) if c}
        if excluded_numeric:
            missing_excluded = sorted(c for c in excluded_numeric if c not in df.columns)
            if missing_excluded:
                raise ValueError(f"Missing exclude numeric column(s): {missing_excluded}")
            numeric_cols = [c for c in numeric_cols if c not in excluded_numeric]
        extra_numeric = [c for c in (extra_numeric_cols or []) if c]
        for col in extra_numeric:
            if col not in df.columns:
                raise ValueError(f"Missing extra numeric column: {col}")
            if col == target_col:
                raise ValueError(f"Extra numeric column cannot be target column: {col}")
            if col in excluded_numeric:
                continue
            if col not in numeric_cols:
                numeric_cols.append(col)
        if rank_target_col is not None and rank_target_col not in df.columns:
            raise ValueError(f"Missing rank target column: {rank_target_col}")

        group_ids: Optional[List[int]] = None
        resolved_group_cols: List[str] = []
        if group_cols:
            resolved_group_cols = list(group_cols)
            missing = [col for col in resolved_group_cols if col not in df.columns]
            if missing:
                raise ValueError(f"Missing group columns for ml-cb dataset: {missing}")
            group_keys = list(zip(*[df[col].tolist() for col in resolved_group_cols]))
            unique_keys = {key: idx for idx, key in enumerate(sorted(set(group_keys)))}
            group_ids = [unique_keys[key] for key in group_keys]

        qc_features = None
        qc_feature_cols: List[str] = []
        qc_groups: Dict[str, List[str]] = {}
        qc_weights = None
        if qc_fusion and qc_fusion.lower() != "none":
            qc_mat, qc_feature_cols, qc_groups = _build_qc_features(df, qc_fusion)
            if qc_mat.shape[1] > 0:
                qc_features = qc_mat
            qc_weights = _compute_qc_feature_weights(
                qc_feature_cols,
                _load_qc_weight_map(qc_weight_path),
                mode=qc_weight_mode,
                scale=qc_weight_scale,
                min_weight=qc_weight_min,
                max_weight=qc_weight_max,
            )

        # Precompute Coulomb eigenvalue features if requested
        parsed_rows: Optional[List[Dict[str, np.ndarray]]] = None
        coulomb_features = None
        geom_features = None
        max_cat = max_r1 = max_r2 = 0
        if use_coulomb or use_geometry:
            parsed_rows, coulomb_features, geom_features, max_cat, max_r1, max_r2 = _compute_borylation_3d_features(
                df,
                excel_path=excel_path,
                cat_xyz_col=cat_xyz_col,
                r1_xyz_col=r1_xyz_col,
                r2_xyz_col=r2_xyz_col,
                use_coulomb=use_coulomb,
                use_geometry=use_geometry,
            )

        feature_cols = list(numeric_cols)
        if qc_feature_cols:
            feature_cols += qc_feature_cols
        if use_coulomb:
            feature_cols += [f"coulomb_eig_{i}" for i in range((max_cat + max_r1 + max_r2) if max_cat else 0)]
        if use_geometry:
            feature_cols += [f"geom_feat_{i}" for i in range((max_cat + max_r1 + max_r2) * 4 if max_cat else 0)]
        self.feature_cols = feature_cols
        self.qc_feature_cols = qc_feature_cols
        self.llm_feature_cols: List[str] = []
        self.qc_groups = qc_groups
        self.numeric_cols = numeric_cols
        self.qc_scale = float(qc_scale)
        self.qc_node_scale = float(qc_node_scale)
        self.qc_global_scale = float(qc_global_scale)
        self.qc_node_mode = (qc_node_mode or "none").lower()
        self.qc_feature_weights = qc_weights
        self.coordination_features = (coordination_features or "none").lower()
        self.coordination_node_scale = float(coordination_node_scale)
        self.coordination_feature_cols = (
            list(_COORDINATION_FEATURE_COLS) if self.coordination_features != "none" else []
        )
        self.build_atomic_interaction_pairs = bool(build_atomic_interaction_pairs)
        self.interaction_pair_mode = str(interaction_pair_mode or _INTERACTION_PAIR_MODE_CLASSIC).lower()
        self.interaction_pair_feature_dim = (
            _interaction_pair_feature_dim_for_mode(self.interaction_pair_mode) if self.build_atomic_interaction_pairs else 0
        )
        self.build_reaction_center_coupling = bool(build_reaction_center_coupling)
        self.reaction_center_feature_dim = (
            _REACTION_CENTER_PAIR_FEATURE_DIM if self.build_reaction_center_coupling else 0
        )
        qc_start = len(numeric_cols)
        qc_end = qc_start + len(qc_feature_cols)
        self._qc_feature_slice = slice(qc_start, qc_end)

        # Shuffle and split indices
        rng = np.random.default_rng(seed)
        indices = np.arange(len(df))
        rng.shuffle(indices)
        if max_samples is not None:
            max_samples = int(max_samples)
            if max_samples <= 0:
                raise ValueError("max_samples must be positive")
            if max_samples < len(indices):
                indices = indices[:max_samples]
        train_idx, val_idx, test_idx = _split_indices(
            indices,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
        )

        if split == "train":
            use_idx = train_idx
        elif split == "val":
            use_idx = val_idx
        elif split == "test":
            use_idx = test_idx
        else:
            raise ValueError(f"Unknown split: {split}")

        # Compute normalization stats on train split
        train_df = df.iloc[train_idx]
        if numeric_cols:
            numeric_mat = np.stack([_column_values(df, col) for col in numeric_cols], axis=1)
        else:
            numeric_mat = np.zeros((len(df), 0), dtype=np.float32)
        train_numeric = numeric_mat[train_idx]
        feat_parts = [train_numeric]
        if qc_features is not None:
            feat_parts.append(qc_features[train_idx])
        if coulomb_features is not None:
            feat_parts.append(coulomb_features[train_idx])
        if geom_features is not None:
            feat_parts.append(geom_features[train_idx])
        train_feats = np.concatenate(feat_parts, axis=1)

        self.feat_mean = train_feats.mean(axis=0, dtype=np.float64).astype(np.float32)
        self.feat_std = train_feats.std(axis=0, dtype=np.float64).astype(np.float32)
        self.feat_std[self.feat_std < 1e-4] = 1.0
        self.y_mean = np.float32(np.mean(train_df[target_col].to_numpy(dtype=np.float64)))
        self.y_std = np.float32(np.std(train_df[target_col].to_numpy(dtype=np.float64)))
        if self.y_std < 1e-8:
            self.y_std = np.float32(1.0)

        self.rank_target_col = rank_target_col
        self.rank_target_mean: Optional[np.float32] = None
        self.rank_target_std: Optional[np.float32] = None
        if rank_target_col is not None:
            rank_values = _column_values(train_df, rank_target_col)
            self.rank_target_mean = np.float32(np.mean(rank_values, dtype=np.float64))
            self.rank_target_std = np.float32(np.std(rank_values, dtype=np.float64))
            if self.rank_target_std < 1e-8:
                self.rank_target_std = np.float32(1.0)

        # Pre-parse samples
        self.samples: List[Sample] = []
        for _, row in df.iloc[use_idx].iterrows():
            cat_z, cat_pos = _parse_xyz_block(row[cat_xyz_col])
            r1_z, r1_pos = _parse_xyz_block(row[r1_xyz_col])
            r2_z, r2_pos = _parse_xyz_block(row[r2_xyz_col])

            cat_edge_index = _build_knn_edges(cat_pos)
            r1_edge_index = _build_knn_edges(r1_pos)
            r2_edge_index = _build_knn_edges(r2_pos)

            feat_parts = [numeric_mat[row.name]]
            if qc_features is not None:
                feat_parts.append(qc_features[row.name])
            if coulomb_features is not None:
                feat_parts.append(coulomb_features[row.name])
            if geom_features is not None:
                feat_parts.append(geom_features[row.name])
            feat = np.concatenate(feat_parts, axis=0)
            feat = (feat - self.feat_mean) / self.feat_std
            if self._qc_feature_slice.start != self._qc_feature_slice.stop:
                qc_feat = feat[self._qc_feature_slice].copy()
                if self.qc_scale != 1.0:
                    qc_feat *= self.qc_scale
                    feat[self._qc_feature_slice] *= self.qc_scale
                if self.qc_feature_weights is not None:
                    qc_feat *= self.qc_feature_weights
                    feat[self._qc_feature_slice] *= self.qc_feature_weights
                if self.qc_node_scale != 1.0:
                    qc_feat *= self.qc_node_scale
                if self.qc_global_scale != 1.0:
                    feat[self._qc_feature_slice] *= self.qc_global_scale
            else:
                qc_feat = np.zeros((0,), dtype=np.float32)
            cat_node_qc = _build_node_qc(
                cat_z,
                cat_pos,
                qc_feat,
                self.qc_node_mode,
                coordination_features=self.coordination_features,
                coordination_node_scale=self.coordination_node_scale,
            )
            r1_node_qc = _build_node_qc(
                r1_z,
                r1_pos,
                qc_feat,
                self.qc_node_mode,
                coordination_features=self.coordination_features,
                coordination_node_scale=self.coordination_node_scale,
            )
            r2_node_qc = _build_node_qc(
                r2_z,
                r2_pos,
                qc_feat,
                self.qc_node_mode,
                coordination_features=self.coordination_features,
                coordination_node_scale=self.coordination_node_scale,
            )
            y = (np.float32(row[target_col]) - self.y_mean) / self.y_std
            rank_target = None
            if rank_target_col is not None and self.rank_target_mean is not None and self.rank_target_std is not None:
                raw_rank = row[rank_target_col]
                if pd.isna(raw_rank):
                    raw_rank = float(self.rank_target_mean)
                raw_rank = np.float32(raw_rank)
                rank_target = torch.tensor(
                    [(raw_rank - self.rank_target_mean) / self.rank_target_std],
                    dtype=torch.float32,
                )

            combined_z = None
            combined_pos = None
            combined_node_type = None
            combined_edge_index = None
            combined_edge_attr = None
            interaction_cat_r1_index = None
            interaction_cat_r1_features = None
            interaction_cat_r2_index = None
            interaction_cat_r2_features = None
            if build_combined_graph:
                combined_z, combined_pos, combined_node_type, combined_edge_index, combined_edge_attr = (
                    _build_combined_graph(
                        cat_z,
                        cat_pos,
                        r1_z,
                        r1_pos,
                        r2_z,
                        r2_pos,
                        intra_k=combined_k,
                        inter_cutoff=combined_cutoff,
                    )
                )
            if self.build_atomic_interaction_pairs:
                interaction_cat_r1_index, interaction_cat_r1_features = _build_atomic_interaction_pairs(
                    cat_z,
                    cat_pos,
                    r1_z,
                    r1_pos,
                    partner_component="r1",
                    mode=self.interaction_pair_mode,
                )
                interaction_cat_r2_index, interaction_cat_r2_features = _build_atomic_interaction_pairs(
                    cat_z,
                    cat_pos,
                    r2_z,
                    r2_pos,
                    partner_component="r2",
                    mode=self.interaction_pair_mode,
                )
            reaction_center = None
            if self.build_reaction_center_coupling:
                reaction_center = _build_reaction_center_coupling(cat_z, cat_pos, r1_z, r1_pos, r2_z, r2_pos)
            cat_stats = _reaction_center_stats(cat_z, cat_pos, "cat")
            r1_stats = _reaction_center_stats(r1_z, r1_pos, "r1")
            r2_stats = _reaction_center_stats(r2_z, r2_pos, "r2")
            focus_kinetic_relay = _kinetic_relay_focus_flag(cat_stats, r1_stats, r2_stats, 25.0)
            electronic_frontier_score = _electronic_frontier_score(cat_stats, r1_stats, r2_stats, 25.0)

            group_id = group_ids[row.name] if group_ids is not None else None
            self.samples.append(
                Sample(
                    cat_z=torch.tensor(cat_z, dtype=torch.long),
                    cat_pos=torch.tensor(cat_pos, dtype=torch.float32),
                    cat_edge_index=cat_edge_index,
                    r1_z=torch.tensor(r1_z, dtype=torch.long),
                    r1_pos=torch.tensor(r1_pos, dtype=torch.float32),
                    r1_edge_index=r1_edge_index,
                    r2_z=torch.tensor(r2_z, dtype=torch.long),
                    r2_pos=torch.tensor(r2_pos, dtype=torch.float32),
                    r2_edge_index=r2_edge_index,
                    features=torch.tensor(feat, dtype=torch.float32),
                    target=torch.tensor([y], dtype=torch.float32),
                    qc_features=torch.tensor(qc_feat, dtype=torch.float32),
                    cat_node_qc=torch.tensor(cat_node_qc, dtype=torch.float32),
                    r1_node_qc=torch.tensor(r1_node_qc, dtype=torch.float32),
                    r2_node_qc=torch.tensor(r2_node_qc, dtype=torch.float32),
                    llm_features=None,
                    literature=None,
                    combined_z=combined_z,
                    combined_pos=combined_pos,
                    combined_node_type=combined_node_type,
                    combined_edge_index=combined_edge_index,
                    combined_edge_attr=combined_edge_attr,
                    interaction_cat_r1_index=interaction_cat_r1_index,
                    interaction_cat_r1_features=interaction_cat_r1_features,
                    interaction_cat_r2_index=interaction_cat_r2_index,
                    interaction_cat_r2_features=interaction_cat_r2_features,
                    reaction_center_cat_r1_index=None if reaction_center is None else reaction_center["cat_r1_index"],
                    reaction_center_cat_r1_features=None if reaction_center is None else reaction_center["cat_r1_features"],
                    reaction_center_cat_r2_index=None if reaction_center is None else reaction_center["cat_r2_index"],
                    reaction_center_cat_r2_features=None if reaction_center is None else reaction_center["cat_r2_features"],
                    reaction_center_targets=None if reaction_center is None else reaction_center["targets"],
                    focus_kinetic_relay=torch.tensor([focus_kinetic_relay], dtype=torch.float32),
                    electronic_frontier_score=torch.tensor([electronic_frontier_score], dtype=torch.float32),
                    group_id=group_id,
                    rank_target=rank_target,
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]

    @property
    def qc_feature_slice(self) -> slice:
        return self._qc_feature_slice


class MLBorylationDataset(Dataset):
    def __init__(
        self,
        excel_path: str,
        split: str,
        seed: int = 42,
        max_samples: Optional[int] = None,
        include_numeric: bool = True,
        exclude_numeric_cols: Optional[Sequence[str]] = None,
        use_coulomb: bool = True,
        use_geometry: bool = True,
        use_rdkit: bool = False,
        use_morgan_fingerprint: bool = False,
        morgan_fingerprint_bits: int = _MORGAN_FINGERPRINT_DEFAULT_BITS,
        morgan_fingerprint_radius: int = _MORGAN_FINGERPRINT_DEFAULT_RADIUS,
        use_ref_data: bool = False,
        qc_fusion: str = "none",
        qc_scale: float = 1.0,
        qc_node_scale: float = 1.0,
        qc_global_scale: float = 1.0,
        qc_node_mode: str = "none",
        qc_weight_path: Optional[str] = None,
        qc_weight_mode: str = "positive",
        qc_weight_scale: float = 0.5,
        qc_weight_min: float = 0.25,
        qc_weight_max: float = 2.0,
        include_literature: bool = False,
        literature_cache_path: Optional[str] = None,
        llm_feature_profile: str = "legacy",
        llm_missingness_contract: str = "profile_default",
        llm_semantic_group_scales: Optional[object] = None,
        llm_semantic_feature_scales: Optional[object] = None,
        llm_exact_row_scale: float = 1.0,
        llm_propagated_row_scale: float = 1.0,
        native_pair_field_profile: str = "full",
        coordination_features: str = "none",
        coordination_node_scale: float = 1.0,
        build_combined_graph: bool = False,
        build_atomic_interaction_pairs: bool = False,
        interaction_pair_mode: str = _INTERACTION_PAIR_MODE_CLASSIC,
        build_reaction_center_coupling: bool = False,
        combined_k: int = 8,
        combined_cutoff: float = 5.0,
        val_fraction: float = 0.1,
        test_fraction: float = 0.1,
        split_strategy: str = "auto",
        signature_columns: Optional[Sequence[str]] = None,
        allow_missing_target: bool = False,
        normalization_stats: Optional[Mapping[str, object]] = None,
        max_atoms_override: Optional[Tuple[int, int, int]] = None,
    ):
        df, target_col = _load_mlb_dataframe(excel_path, allow_missing_target=allow_missing_target)
        cat_xyz_col = _select_column(df, _BORYLATION_XYZ_COLUMN_CANDIDATES["cat"], "cat xyz")
        r1_xyz_col = _select_column(df, _BORYLATION_XYZ_COLUMN_CANDIDATES["r1"], "r1 xyz")
        r2_xyz_col = _select_column(df, _BORYLATION_XYZ_COLUMN_CANDIDATES["r2"], "p1 xyz")
        temperature_col = _find_temperature_column(df)
        has_targets = target_col is not None

        numeric_cols = []
        if include_numeric:
            numeric_cols = [
                c
                for c in df.columns
                if c != target_col and df[c].dtype != object and not _is_excluded_metadata_feature(c)
            ]
        excluded_numeric = {c for c in (exclude_numeric_cols or []) if c}
        if excluded_numeric:
            missing_excluded = sorted(c for c in excluded_numeric if c not in df.columns)
            if missing_excluded:
                raise ValueError(f"Missing exclude numeric column(s): {missing_excluded}")
            numeric_cols = [c for c in numeric_cols if c not in excluded_numeric]

        qc_features = None
        qc_feature_cols: List[str] = []
        qc_groups: Dict[str, List[str]] = {}
        qc_weights = None
        if qc_fusion and qc_fusion.lower() != "none":
            qc_mat, qc_feature_cols, qc_groups = _build_qc_features(df, qc_fusion)
            if qc_mat.shape[1] > 0:
                qc_features = qc_mat
            qc_weights = _compute_qc_feature_weights(
                qc_feature_cols,
                _load_qc_weight_map(qc_weight_path),
                mode=qc_weight_mode,
                scale=qc_weight_scale,
                min_weight=qc_weight_min,
                max_weight=qc_weight_max,
            )

        rdkit_features = None
        rdkit_feature_cols: List[str] = []
        morgan_features = None
        morgan_feature_cols: List[str] = []
        cat_smiles_col = None
        r1_smiles_col = None
        r2_smiles_col = None
        if use_rdkit:
            cat_smiles_col = _select_column(df, _BORYLATION_SMILES_COLUMN_CANDIDATES["cat"], "cat smiles")
            r1_smiles_col = _select_column(df, _BORYLATION_SMILES_COLUMN_CANDIDATES["r1"], "r1 smiles")
            r2_smiles_col = _select_column(df, _BORYLATION_SMILES_COLUMN_CANDIDATES["r2"], "p1 smiles")

            cache: Dict[str, np.ndarray] = {}
            num_desc = len(_RDKIT_DESCRIPTOR_NAMES)
            rdkit_features = np.zeros((len(df), num_desc * 3), dtype=np.float32)
            for idx, row in df.iterrows():
                def _get_vec(smiles: str) -> np.ndarray:
                    key = smiles if isinstance(smiles, str) else ""
                    if key not in cache:
                        cache[key] = _rdkit_descriptor_vector(key)
                    return cache[key]

                cat_vec = _get_vec(row.get(cat_smiles_col, ""))
                r1_vec = _get_vec(row.get(r1_smiles_col, ""))
                r2_vec = _get_vec(row.get(r2_smiles_col, ""))
                rdkit_features[idx, :num_desc] = cat_vec
                rdkit_features[idx, num_desc:2 * num_desc] = r1_vec
                rdkit_features[idx, 2 * num_desc:] = r2_vec

            rdkit_feature_cols = (
                [f"cat_{name}" for name in _RDKIT_DESCRIPTOR_NAMES]
                + [f"r1_{name}" for name in _RDKIT_DESCRIPTOR_NAMES]
                + [f"p1_{name}" for name in _RDKIT_DESCRIPTOR_NAMES]
            )
        if use_morgan_fingerprint:
            if cat_smiles_col is None:
                cat_smiles_col = _select_column(df, _BORYLATION_SMILES_COLUMN_CANDIDATES["cat"], "cat smiles")
            if r1_smiles_col is None:
                r1_smiles_col = _select_column(df, _BORYLATION_SMILES_COLUMN_CANDIDATES["r1"], "r1 smiles")
            if r2_smiles_col is None:
                r2_smiles_col = _select_column(df, _BORYLATION_SMILES_COLUMN_CANDIDATES["r2"], "p1 smiles")

            cache: Dict[str, np.ndarray] = {}
            fingerprint_bits = int(morgan_fingerprint_bits)
            fingerprint_radius = int(morgan_fingerprint_radius)
            if fingerprint_bits <= 0:
                raise ValueError("morgan_fingerprint_bits must be positive")
            if fingerprint_radius < 0:
                raise ValueError("morgan_fingerprint_radius must be non-negative")
            morgan_features = np.zeros((len(df), fingerprint_bits * 3), dtype=np.float32)
            for idx, row in df.iterrows():
                def _get_fp(smiles: str) -> np.ndarray:
                    key = smiles if isinstance(smiles, str) else ""
                    if key not in cache:
                        cache[key] = _morgan_fingerprint_vector(
                            key,
                            radius=fingerprint_radius,
                            n_bits=fingerprint_bits,
                        )
                    return cache[key]

                cat_fp = _get_fp(row.get(cat_smiles_col, ""))
                r1_fp = _get_fp(row.get(r1_smiles_col, ""))
                r2_fp = _get_fp(row.get(r2_smiles_col, ""))
                morgan_features[idx, :fingerprint_bits] = cat_fp
                morgan_features[idx, fingerprint_bits:2 * fingerprint_bits] = r1_fp
                morgan_features[idx, 2 * fingerprint_bits:] = r2_fp

            morgan_feature_cols = (
                [f"cat_morgan_bit_{i}" for i in range(fingerprint_bits)]
                + [f"r1_morgan_bit_{i}" for i in range(fingerprint_bits)]
                + [f"p1_morgan_bit_{i}" for i in range(fingerprint_bits)]
            )
        elif include_literature:
            cat_smiles_col = _select_column(df, _BORYLATION_SMILES_COLUMN_CANDIDATES["cat"], "cat smiles")
            r1_smiles_col = _select_column(df, _BORYLATION_SMILES_COLUMN_CANDIDATES["r1"], "r1 smiles")
            r2_smiles_col = _select_column(df, _BORYLATION_SMILES_COLUMN_CANDIDATES["r2"], "p1 smiles")

        ref_features = None
        ref_feature_cols: List[str] = []
        if use_ref_data:
            ref_col = _select_column(df, _BORYLATION_REF_COLUMN_CANDIDATES, "ref-data")
            ref_values = df[ref_col].fillna("unknown").astype(str).tolist()
            unique_refs = sorted(set(ref_values))
            ref_index = {val: idx for idx, val in enumerate(unique_refs)}
            ref_features = np.zeros((len(df), len(unique_refs)), dtype=np.float32)
            for idx, val in enumerate(ref_values):
                ref_features[idx, ref_index[val]] = 1.0
            ref_feature_cols = [f"ref_{val}" for val in unique_refs]

        llm_feature_cols: List[str] = []
        llm_feature_builder = None
        resolved_llm_feature_profile = str(llm_feature_profile or "legacy")
        resolved_llm_missingness_contract = str(llm_missingness_contract or "profile_default").strip().lower()
        resolved_llm_semantic_group_scales = _parse_llm_semantic_group_scales(llm_semantic_group_scales)
        resolved_llm_semantic_feature_scales = _parse_llm_semantic_feature_scales(llm_semantic_feature_scales)
        resolved_llm_exact_row_scale = _validate_llm_row_scale(
            llm_exact_row_scale,
            label="llm_exact_row_scale",
        )
        resolved_llm_propagated_row_scale = _validate_llm_row_scale(
            llm_propagated_row_scale,
            label="llm_propagated_row_scale",
        )
        llm_semantic_group_scale_vector = np.ones((0,), dtype=np.float32)
        llm_semantic_feature_scale_vector = np.ones((0,), dtype=np.float32)
        valid_llm_missingness_contracts = {"profile_default", "missingness_token"}
        if resolved_llm_missingness_contract not in valid_llm_missingness_contracts:
            raise ValueError(
                f"Unknown llm_missingness_contract: {llm_missingness_contract}. "
                f"Expected one of {sorted(valid_llm_missingness_contracts)}."
            )
        native_pair_profile_spec = native_pair_field_profile_spec(native_pair_field_profile)
        if include_literature:
            from llm.features import build_llm_feature_vector, llm_feature_columns

            llm_feature_cols = llm_feature_columns(profile=resolved_llm_feature_profile)
            if resolved_llm_missingness_contract == "missingness_token":
                llm_feature_cols = [*llm_feature_cols, "llm_missingness_token"]
            llm_semantic_group_scale_vector = _build_llm_semantic_group_scale_vector(
                llm_feature_cols,
                resolved_llm_semantic_group_scales,
            )
            llm_semantic_feature_scale_vector = _build_llm_semantic_feature_scale_vector(
                llm_feature_cols,
                resolved_llm_semantic_feature_scales,
            )
            llm_feature_builder = (
                lambda literature, feature_profile=resolved_llm_feature_profile: build_llm_feature_vector(
                    literature,
                    profile=feature_profile,
                )
            )

        parsed_rows: Optional[List[Dict[str, np.ndarray]]] = None
        coulomb_features = None
        geom_features = None
        max_cat = max_r1 = max_r2 = 0
        if use_coulomb or use_geometry:
            parsed_rows, coulomb_features, geom_features, max_cat, max_r1, max_r2 = _compute_borylation_3d_features(
                df,
                excel_path=excel_path,
                cat_xyz_col=cat_xyz_col,
                r1_xyz_col=r1_xyz_col,
                r2_xyz_col=r2_xyz_col,
                use_coulomb=use_coulomb,
                use_geometry=use_geometry,
                max_atoms_override=max_atoms_override,
            )
        self.max_cat_atoms = int(max_cat)
        self.max_r1_atoms = int(max_r1)
        self.max_r2_atoms = int(max_r2)

        feature_cols = list(numeric_cols)
        if qc_feature_cols:
            feature_cols += qc_feature_cols
        if rdkit_feature_cols:
            feature_cols += rdkit_feature_cols
        if morgan_feature_cols:
            feature_cols += morgan_feature_cols
        if ref_feature_cols:
            feature_cols += ref_feature_cols
        if use_coulomb:
            feature_cols += [f"coulomb_eig_{i}" for i in range((max_cat + max_r1 + max_r2) if max_cat else 0)]
        if use_geometry:
            feature_cols += [f"geom_feat_{i}" for i in range((max_cat + max_r1 + max_r2) * 4 if max_cat else 0)]
        self.feature_cols = feature_cols
        self.qc_feature_cols = qc_feature_cols
        self.qc_groups = qc_groups
        self.rdkit_feature_cols = rdkit_feature_cols
        self.morgan_feature_cols = morgan_feature_cols
        self.morgan_fingerprint_bits = int(morgan_fingerprint_bits)
        self.morgan_fingerprint_radius = int(morgan_fingerprint_radius)
        self.ref_feature_cols = ref_feature_cols
        self.numeric_cols = numeric_cols
        self.llm_feature_cols = llm_feature_cols
        self.llm_feature_profile = resolved_llm_feature_profile
        self.llm_missingness_contract = resolved_llm_missingness_contract
        self.llm_semantic_group_scales = dict(sorted(resolved_llm_semantic_group_scales.items()))
        self.llm_semantic_feature_scales = dict(sorted(resolved_llm_semantic_feature_scales.items()))
        self.llm_exact_row_scale = resolved_llm_exact_row_scale
        self.llm_propagated_row_scale = resolved_llm_propagated_row_scale
        self.llm_semantic_group_scale_vector = llm_semantic_group_scale_vector.copy()
        self.llm_semantic_feature_scale_vector = llm_semantic_feature_scale_vector.copy()
        self.native_pair_field_profile = str(native_pair_profile_spec["profile"])
        self.native_pair_field_profile_description = str(native_pair_profile_spec["description"])
        self.native_pair_summary_group_scales = dict(native_pair_profile_spec["summary_group_scales"])
        self.native_pair_token_group_scales = dict(native_pair_profile_spec["token_group_scales"])
        self.native_pair_summary_feature_names = list(native_pair_profile_spec["summary_feature_names"])
        self.native_pair_token_feature_names = list(native_pair_profile_spec["token_feature_names"])
        self.native_pair_summary_dim = _NATIVE_PAIR_SUMMARY_FEATURE_DIM
        self.native_pair_token_dim = _NATIVE_PAIR_TOKEN_FEATURE_DIM
        self.native_pair_max_tokens = _NATIVE_PAIR_MAX_TOKENS
        self.native_pair_max_experts = _NATIVE_PAIR_MAX_EXPERTS
        self.temperature_col = temperature_col
        self.qc_scale = float(qc_scale)
        self.qc_node_scale = float(qc_node_scale)
        self.qc_global_scale = float(qc_global_scale)
        self.qc_node_mode = (qc_node_mode or "none").lower()
        self.qc_feature_weights = qc_weights
        self.coordination_features = (coordination_features or "none").lower()
        self.coordination_node_scale = float(coordination_node_scale)
        self.coordination_feature_cols = (
            list(_COORDINATION_FEATURE_COLS) if self.coordination_features != "none" else []
        )
        self.build_atomic_interaction_pairs = bool(build_atomic_interaction_pairs)
        self.interaction_pair_mode = str(interaction_pair_mode or _INTERACTION_PAIR_MODE_CLASSIC).lower()
        self.interaction_pair_feature_dim = (
            _interaction_pair_feature_dim_for_mode(self.interaction_pair_mode) if self.build_atomic_interaction_pairs else 0
        )
        self.build_reaction_center_coupling = bool(build_reaction_center_coupling)
        self.reaction_center_feature_dim = (
            _REACTION_CENTER_PAIR_FEATURE_DIM if self.build_reaction_center_coupling else 0
        )
        qc_start = len(numeric_cols)
        qc_end = qc_start + len(qc_feature_cols)
        self._qc_feature_slice = slice(qc_start, qc_end)
        self.split_strategy = "random"
        self.split_group_source: Optional[str] = None

        rng = np.random.default_rng(seed)
        indices = np.arange(len(df))
        if max_samples is not None:
            rng.shuffle(indices)
            max_samples = int(max_samples)
            if max_samples <= 0:
                raise ValueError("max_samples must be positive")
            if max_samples < len(indices):
                indices = indices[:max_samples]
        requested_split_strategy = (split_strategy or "auto").lower()
        valid_split_strategies = {"auto", "reference", "structural_signature"}
        if requested_split_strategy not in valid_split_strategies:
            raise ValueError(
                f"Unknown split_strategy: {split_strategy}. "
                f"Expected one of {sorted(valid_split_strategies)}."
            )

        if requested_split_strategy == "structural_signature":
            signature_groups, signature_source_cols = _resolve_structural_signature_groups(
                df,
                signature_columns=signature_columns,
            )
            train_idx, val_idx, test_idx = _group_split_indices(
                indices,
                signature_groups[indices],
                val_fraction=val_fraction,
                test_fraction=test_fraction,
                seed=seed,
            )
            self.split_strategy = "group_structural_signature"
            self.split_group_source = ",".join(signature_source_cols)
        elif requested_split_strategy == "reference":
            reference_groups, reference_source = _resolve_reference_groups(df)
            if reference_groups is None:
                raise ValueError("split_strategy=reference requested but no reference groups could be resolved.")
            if reference_source == "row_index":
                raise ValueError(
                    "split_strategy=reference resolved to row_index only; use "
                    "split_strategy=structural_signature to avoid row-level leakage."
                )
            train_idx, val_idx, test_idx = _group_split_indices(
                indices,
                reference_groups[indices],
                val_fraction=val_fraction,
                test_fraction=test_fraction,
                seed=seed,
            )
            self.split_strategy = "group_reference"
            self.split_group_source = reference_source
        else:
            reference_groups, reference_source = _resolve_reference_groups(df)
            if reference_groups is not None:
                try:
                    train_idx, val_idx, test_idx = _group_split_indices(
                        indices,
                        reference_groups[indices],
                        val_fraction=val_fraction,
                        test_fraction=test_fraction,
                        seed=seed,
                    )
                    self.split_strategy = "group_reference"
                    self.split_group_source = reference_source
                except ValueError as exc:
                    warnings.warn(
                        f"Falling back to random split after reference-group split failure: {exc}",
                        RuntimeWarning,
                    )
                    shuffled = indices.copy()
                    rng.shuffle(shuffled)
                    train_idx, val_idx, test_idx = _split_indices(
                        shuffled,
                        val_fraction=val_fraction,
                        test_fraction=test_fraction,
                    )
            else:
                shuffled = indices.copy()
                rng.shuffle(shuffled)
                train_idx, val_idx, test_idx = _split_indices(
                    shuffled,
                    val_fraction=val_fraction,
                    test_fraction=test_fraction,
                )

        split_lower = str(split).lower()
        if split_lower == "train":
            use_idx = train_idx
        elif split_lower == "val":
            use_idx = val_idx
        elif split_lower == "test":
            use_idx = test_idx
        elif split_lower in {"inference", "predict", "all"}:
            use_idx = indices
        else:
            raise ValueError(f"Unknown split: {split}")
        self.train_indices = np.asarray(train_idx, dtype=np.int64)
        self.val_indices = np.asarray(val_idx, dtype=np.int64)
        self.test_indices = np.asarray(test_idx, dtype=np.int64)
        self.row_indices = np.asarray(use_idx, dtype=np.int64)
        self.target_col = target_col
        self.has_targets = bool(has_targets)

        train_df = df.iloc[train_idx]
        train_numeric = train_df[numeric_cols].values.astype(np.float32)
        feat_parts = [train_numeric]
        if qc_features is not None:
            feat_parts.append(qc_features[train_idx])
        if rdkit_features is not None:
            feat_parts.append(rdkit_features[train_idx])
        if morgan_features is not None:
            feat_parts.append(morgan_features[train_idx])
        if ref_features is not None:
            feat_parts.append(ref_features[train_idx])
        if coulomb_features is not None:
            feat_parts.append(coulomb_features[train_idx])
        if geom_features is not None:
            feat_parts.append(geom_features[train_idx])
        train_feats = np.concatenate(feat_parts, axis=1)

        if normalization_stats is not None:
            self.feat_mean, self.feat_std, self.y_mean, self.y_std = _normalization_stats_from_mapping(normalization_stats)
            if self.feat_mean.shape[0] != train_feats.shape[1]:
                raise ValueError(
                    f"Normalization stats feature dimension mismatch: expected {train_feats.shape[1]}, "
                    f"got {self.feat_mean.shape[0]}."
                )
            self.feat_std[self.feat_std < 1e-4] = 1.0
        else:
            self.feat_mean = train_feats.mean(axis=0, dtype=np.float64).astype(np.float32)
            self.feat_std = train_feats.std(axis=0, dtype=np.float64).astype(np.float32)
            self.feat_std[self.feat_std < 1e-4] = 1.0
            if has_targets:
                self.y_mean = np.float32(np.mean(train_df[target_col].to_numpy(dtype=np.float64)))
                self.y_std = np.float32(np.std(train_df[target_col].to_numpy(dtype=np.float64)))
                if self.y_std < 1e-8:
                    self.y_std = np.float32(1.0)
            else:
                self.y_mean = np.float32(0.0)
                self.y_std = np.float32(1.0)

        self.samples: List[Sample] = []
        literature_cache = _load_literature_cache(literature_cache_path) if include_literature else None
        mol_cache: Dict[str, object] = {}
        if temperature_col is not None:
            temperature_values = pd.to_numeric(df[temperature_col], errors="coerce")
            if temperature_values.notna().any():
                default_temperature = float(temperature_values.median())
            else:
                default_temperature = 25.0
        else:
            default_temperature = 25.0
        for row_idx, row in df.iloc[use_idx].iterrows():
            if parsed_rows is not None:
                parsed = parsed_rows[row_idx]
                cat_z = parsed["cat_z"].tolist()
                cat_pos = parsed["cat_pos"].tolist()
                cat_edge_index = torch.from_numpy(parsed["cat_edge_index"].copy())
                r1_z = parsed["r1_z"].tolist()
                r1_pos = parsed["r1_pos"].tolist()
                r1_edge_index = torch.from_numpy(parsed["r1_edge_index"].copy())
                r2_z = parsed["r2_z"].tolist()
                r2_pos = parsed["r2_pos"].tolist()
                r2_edge_index = torch.from_numpy(parsed["r2_edge_index"].copy())
            else:
                cat_z, cat_pos = _parse_xyz_block(row[cat_xyz_col])
                r1_z, r1_pos = _parse_xyz_block(row[r1_xyz_col])
                r2_z, r2_pos = _parse_xyz_block(row[r2_xyz_col])
                cat_edge_index = _build_knn_edges(cat_pos)
                r1_edge_index = _build_knn_edges(r1_pos)
                r2_edge_index = _build_knn_edges(r2_pos)

            feat_parts = [row[numeric_cols].values.astype(np.float32)]
            if qc_features is not None:
                feat_parts.append(qc_features[row_idx])
            if rdkit_features is not None:
                feat_parts.append(rdkit_features[row_idx])
            if morgan_features is not None:
                feat_parts.append(morgan_features[row_idx])
            if ref_features is not None:
                feat_parts.append(ref_features[row_idx])
            if coulomb_features is not None:
                feat_parts.append(coulomb_features[row_idx])
            if geom_features is not None:
                feat_parts.append(geom_features[row_idx])
            feat = np.concatenate(feat_parts, axis=0)
            feat = (feat - self.feat_mean) / self.feat_std
            if self._qc_feature_slice.start != self._qc_feature_slice.stop:
                qc_feat = feat[self._qc_feature_slice].copy()
                if self.qc_scale != 1.0:
                    qc_feat *= self.qc_scale
                    feat[self._qc_feature_slice] *= self.qc_scale
                if self.qc_feature_weights is not None:
                    qc_feat *= self.qc_feature_weights
                    feat[self._qc_feature_slice] *= self.qc_feature_weights
                if self.qc_node_scale != 1.0:
                    qc_feat *= self.qc_node_scale
                if self.qc_global_scale != 1.0:
                    feat[self._qc_feature_slice] *= self.qc_global_scale
            else:
                qc_feat = np.zeros((0,), dtype=np.float32)
            cat_node_qc = _build_node_qc(
                cat_z,
                cat_pos,
                qc_feat,
                self.qc_node_mode,
                coordination_features=self.coordination_features,
                coordination_node_scale=self.coordination_node_scale,
            )
            r1_node_qc = _build_node_qc(
                r1_z,
                r1_pos,
                qc_feat,
                self.qc_node_mode,
                coordination_features=self.coordination_features,
                coordination_node_scale=self.coordination_node_scale,
            )
            r2_node_qc = _build_node_qc(
                r2_z,
                r2_pos,
                qc_feat,
                self.qc_node_mode,
                coordination_features=self.coordination_features,
                coordination_node_scale=self.coordination_node_scale,
            )
            if has_targets:
                y = (np.float32(row[target_col]) - self.y_mean) / self.y_std
            else:
                y = np.float32(0.0)
            if temperature_col is not None:
                raw_temperature = pd.to_numeric(pd.Series([row.get(temperature_col)]), errors="coerce").iloc[0]
                if pd.isna(raw_temperature):
                    raw_temperature = default_temperature
            else:
                raw_temperature = default_temperature
            literature = None
            if include_literature:
                if literature_cache is not None:
                    literature = literature_cache.get(row_idx)
                else:
                    literature = _build_mock_literature(row_idx)
                if literature is not None:
                    cat_smiles = row.get(cat_smiles_col, "") if cat_smiles_col else ""
                    r1_smiles = row.get(r1_smiles_col, "") if r1_smiles_col else ""
                    r2_smiles = row.get(r2_smiles_col, "") if r2_smiles_col else ""
                    literature = _map_literature_entry(
                        literature,
                        cat_smiles if isinstance(cat_smiles, str) else "",
                        r1_smiles if isinstance(r1_smiles, str) else "",
                        r2_smiles if isinstance(r2_smiles, str) else "",
                        mol_cache,
                    )

            if llm_feature_builder is not None:
                llm_feat = llm_feature_builder(literature)
                llm_feat = _apply_llm_feature_overrides(
                    llm_feat,
                    literature=literature,
                    llm_feature_cols=llm_feature_cols,
                )
                if resolved_llm_missingness_contract == "missingness_token":
                    missingness_token = np.asarray(
                        [1.0 if literature is None else 0.0],
                        dtype=np.float32,
                    )
                    llm_feat = np.concatenate([llm_feat.astype(np.float32, copy=False), missingness_token], axis=0)
                if llm_semantic_group_scale_vector.size:
                    llm_feat = np.asarray(llm_feat, dtype=np.float32) * llm_semantic_group_scale_vector
                if llm_semantic_feature_scale_vector.size:
                    llm_feat = np.asarray(llm_feat, dtype=np.float32) * llm_semantic_feature_scale_vector
                row_provenance = _extract_llm_row_provenance(literature)
                if row_provenance == "exact" and resolved_llm_exact_row_scale != 1.0:
                    llm_feat = np.asarray(llm_feat, dtype=np.float32) * resolved_llm_exact_row_scale
                elif row_provenance == "propagated" and resolved_llm_propagated_row_scale != 1.0:
                    llm_feat = np.asarray(llm_feat, dtype=np.float32) * resolved_llm_propagated_row_scale
            else:
                llm_feat = np.zeros((len(llm_feature_cols),), dtype=np.float32)
            llm_confidence = _extract_llm_confidence(literature) if include_literature else 0.0
            native_pair_summary, native_pair_tokens, native_pair_token_mask = _build_native_pair_tensors(
                literature,
                field_profile=self.native_pair_field_profile,
            )
            (
                native_pair_expert_summaries,
                native_pair_expert_tokens,
                native_pair_expert_token_masks,
                native_pair_expert_priors,
            ) = _build_native_pair_expert_tensors(
                literature,
                field_profile=self.native_pair_field_profile,
            )

            combined_z = None
            combined_pos = None
            combined_node_type = None
            combined_edge_index = None
            combined_edge_attr = None
            interaction_cat_r1_index = None
            interaction_cat_r1_features = None
            interaction_cat_r2_index = None
            interaction_cat_r2_features = None
            if build_combined_graph:
                combined_z, combined_pos, combined_node_type, combined_edge_index, combined_edge_attr = (
                    _build_combined_graph(
                        cat_z,
                        cat_pos,
                        r1_z,
                        r1_pos,
                        r2_z,
                        r2_pos,
                        intra_k=combined_k,
                        inter_cutoff=combined_cutoff,
                    )
                )
            if self.build_atomic_interaction_pairs:
                interaction_cat_r1_index, interaction_cat_r1_features = _build_atomic_interaction_pairs(
                    cat_z,
                    cat_pos,
                    r1_z,
                    r1_pos,
                    partner_component="r1",
                    mode=self.interaction_pair_mode,
                )
                interaction_cat_r2_index, interaction_cat_r2_features = _build_atomic_interaction_pairs(
                    cat_z,
                    cat_pos,
                    r2_z,
                    r2_pos,
                    partner_component="r2",
                    mode=self.interaction_pair_mode,
                )
            reaction_center = None
            if self.build_reaction_center_coupling:
                reaction_center = _build_reaction_center_coupling(cat_z, cat_pos, r1_z, r1_pos, r2_z, r2_pos)
            cat_stats = _reaction_center_stats(cat_z, cat_pos, "cat")
            r1_stats = _reaction_center_stats(r1_z, r1_pos, "r1")
            r2_stats = _reaction_center_stats(r2_z, r2_pos, "r2")
            focus_kinetic_relay = _kinetic_relay_focus_flag(cat_stats, r1_stats, r2_stats, float(raw_temperature))
            electronic_frontier_score = _electronic_frontier_score(
                cat_stats,
                r1_stats,
                r2_stats,
                float(raw_temperature),
            )

            self.samples.append(
                Sample(
                    cat_z=torch.tensor(cat_z, dtype=torch.long),
                    cat_pos=torch.tensor(cat_pos, dtype=torch.float32),
                    cat_edge_index=cat_edge_index,
                    r1_z=torch.tensor(r1_z, dtype=torch.long),
                    r1_pos=torch.tensor(r1_pos, dtype=torch.float32),
                    r1_edge_index=r1_edge_index,
                    r2_z=torch.tensor(r2_z, dtype=torch.long),
                    r2_pos=torch.tensor(r2_pos, dtype=torch.float32),
                    r2_edge_index=r2_edge_index,
                    features=torch.tensor(feat, dtype=torch.float32),
                    target=torch.tensor([y], dtype=torch.float32),
                    temperature=torch.tensor([float(raw_temperature)], dtype=torch.float32),
                    qc_features=torch.tensor(qc_feat, dtype=torch.float32),
                    cat_node_qc=torch.tensor(cat_node_qc, dtype=torch.float32),
                    r1_node_qc=torch.tensor(r1_node_qc, dtype=torch.float32),
                    r2_node_qc=torch.tensor(r2_node_qc, dtype=torch.float32),
                    llm_features=torch.tensor(llm_feat, dtype=torch.float32),
                    llm_confidence=torch.tensor([llm_confidence], dtype=torch.float32),
                    exact_covered_indicator=torch.tensor(
                        [_extract_exact_covered_indicator(literature)],
                        dtype=torch.float32,
                    ),
                    literature=literature,
                    native_pair_summary=torch.tensor(native_pair_summary, dtype=torch.float32),
                    native_pair_tokens=torch.tensor(native_pair_tokens, dtype=torch.float32),
                    native_pair_token_mask=torch.tensor(native_pair_token_mask, dtype=torch.float32),
                    native_pair_expert_summaries=torch.tensor(
                        native_pair_expert_summaries,
                        dtype=torch.float32,
                    ),
                    native_pair_expert_tokens=torch.tensor(
                        native_pair_expert_tokens,
                        dtype=torch.float32,
                    ),
                    native_pair_expert_token_masks=torch.tensor(
                        native_pair_expert_token_masks,
                        dtype=torch.float32,
                    ),
                    native_pair_expert_priors=torch.tensor(
                        native_pair_expert_priors,
                        dtype=torch.float32,
                    ),
                    combined_z=combined_z,
                    combined_pos=combined_pos,
                    combined_node_type=combined_node_type,
                    combined_edge_index=combined_edge_index,
                    combined_edge_attr=combined_edge_attr,
                    interaction_cat_r1_index=interaction_cat_r1_index,
                    interaction_cat_r1_features=interaction_cat_r1_features,
                    interaction_cat_r2_index=interaction_cat_r2_index,
                    interaction_cat_r2_features=interaction_cat_r2_features,
                    reaction_center_cat_r1_index=None if reaction_center is None else reaction_center["cat_r1_index"],
                    reaction_center_cat_r1_features=None if reaction_center is None else reaction_center["cat_r1_features"],
                    reaction_center_cat_r2_index=None if reaction_center is None else reaction_center["cat_r2_index"],
                    reaction_center_cat_r2_features=None if reaction_center is None else reaction_center["cat_r2_features"],
                    reaction_center_targets=None if reaction_center is None else reaction_center["targets"],
                    focus_kinetic_relay=torch.tensor([focus_kinetic_relay], dtype=torch.float32),
                    electronic_frontier_score=torch.tensor([electronic_frontier_score], dtype=torch.float32),
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]

    @property
    def qc_feature_slice(self) -> slice:
        return self._qc_feature_slice
