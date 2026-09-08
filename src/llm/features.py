from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence
import math

import numpy as np


INTERACTION_TYPES = (
    "coordination",
    "h_bond",
    "covalent",
    "pi_stacking",
    "ionic",
    "van_der_waals",
)
STRENGTH_CATEGORIES = (
    "very_strong",
    "strong",
    "moderate",
    "weak",
    "negligible",
)
ELECTRONIC_EFFECTS = (
    "electron_donating",
    "electron_withdrawing",
    "neutral",
)
STERIC_EFFECTS = (
    "hindered",
    "favorable",
    "neutral",
)
COMPONENTS = ("cat", "r1", "r2")
SEMANTIC_META_FEATURES = (
    "steric_pressure",
    "electronic_flux",
    "temperature_alignment",
    "cat_focus",
    "r1_focus",
    "r2_focus",
    "hotspot_density",
)
CONFIDENCE_BREAKDOWN_FEATURES = (
    "steric",
    "electronic",
    "weak_interaction",
    "conformational",
    "temperature",
)
TEXT_SEMANTIC_BUCKETS = {
    "directing_preorganization": (
        "directed",
        "directing",
        "preorganization",
        "pre-transition-state",
        "pre transition state",
        "pre-ts",
        "chelat",
        "organized",
        "organizes",
        "organization",
        "amide-guided",
        "carbonyl-guided",
        "carbamate-guided",
        "n-directed",
        "o-directed",
    ),
    "steric_clash": (
        "steric",
        "crowded",
        "crowding",
        "clash",
        "congest",
        "hinder",
        "repulsion",
        "quadrant",
        "pocket",
        "side arm",
        "sidewall",
        "shielded",
        "open face",
    ),
    "boron_boryl": (
        "boryl",
        "b-o",
        "boron",
        "bpin",
        "pinacol",
        "boronate",
        "ir-b",
        "ir boryl",
    ),
    "oxidative_orbital": (
        "oxidative addition",
        "sigma*",
        "sigma-star",
        "dxy",
        "orbital",
        "overlap",
        "agostic",
    ),
    "transition_state": (
        "transition state",
        "stereodifferentiating",
        "enantiodetermining",
        "stereodetermining",
        "ts ",
        "ts1",
        "pre-ts",
    ),
    "conformation_geometry": (
        "conformation",
        "conformer",
        "flexible",
        "fold",
        "rotamer",
        "orientation",
        "trajectory",
        "geometry",
        "tilt",
        "approach",
    ),
    "outer_sphere_weak": (
        "through-space",
        "through space",
        "outer-sphere",
        "outer sphere",
        "van der waals",
        "weak interaction",
        "weak contact",
        "weak organizing",
        "modest outer-sphere",
    ),
    "octahedral_pocket": (
        "octahedral",
        "triarylmethane",
        "three-aryl",
        "three aryl",
        "pyridine framework",
        "pyridine-based",
        "pyridyl",
    ),
    "trans_cis_coordination": (
        "trans",
        "cis",
        "n-bound",
        "o-bound",
        "n-coordinated",
        "o-directed",
        "n-directed",
    ),
    "temperature_window": (
        "temperature",
        "thermal",
        "elevated temperature",
        "higher temperature",
        "80 c",
        "70 c",
        "60 c",
    ),
}
TEXT_META_FIELDS = (
    "major_path_hypothesis",
    "minor_path_hypothesis",
    "ee_determining_factor",
    "destabilizing_factor_for_minor_path",
    "dominant_weak_interaction",
    "catalyst_substrate_match_type",
    "temperature_sensitive_step",
    "evidence_anchor",
)
LLM_FEATURE_PROFILE_LEGACY = "legacy"
LLM_FEATURE_PROFILE_COMPONENT_ONLY = "component_only"
LLM_FEATURE_PROFILE_CORE_EMPHASIS = "core_emphasis"
LLM_FEATURE_PROFILE_HIGH_INFORMATION = "high_information"
LLM_FEATURE_PROFILE_CHEMISTRY_GROUNDED_COMPACT = "chemistry_grounded_compact"
LLM_FEATURE_PROFILE_PROPAGATED_COMPACT_V2 = "propagated_compact_v2"
LLM_FEATURE_PROFILE_PAIR_AWARE = "pair_aware"
_SUPPORTED_LLM_FEATURE_PROFILES = (
    LLM_FEATURE_PROFILE_LEGACY,
    LLM_FEATURE_PROFILE_COMPONENT_ONLY,
    LLM_FEATURE_PROFILE_CORE_EMPHASIS,
    LLM_FEATURE_PROFILE_HIGH_INFORMATION,
    LLM_FEATURE_PROFILE_CHEMISTRY_GROUNDED_COMPACT,
    LLM_FEATURE_PROFILE_PROPAGATED_COMPACT_V2,
    LLM_FEATURE_PROFILE_PAIR_AWARE,
)
LLM_SEMANTIC_BRANCH_ORDER = ("global", "cat", "r1", "r2", "pair")
LLM_PAIR_GROUP_ORDER = ("core_pair", "product_pair", "contrast")
LLM_SEMANTIC_GROUP_ORDER = (
    "global_scalar",
    "global_ratios",
    "semantic_meta",
    "confidence_breakdown",
    "global_text",
    "feature_group_prior",
    "reweighted_text",
    "cat_component",
    "r1_component",
    "r2_component",
    "pair_component",
    "other",
)
_REWEIGHTED_TEXT_COMPONENT_WEIGHTS = {
    "cat": 0.55,
    "r1": 0.35,
    "r2": 0.10,
}
_CORE_TEXT_BUCKET_NAMES = tuple(name for name in TEXT_SEMANTIC_BUCKETS if name != "boron_boryl")
_PRODUCT_TEXT_BUCKET_NAMES = ("boron_boryl",)
FEATURE_GROUP_PRIOR_COLUMNS = (
    "llm_feature_group_core_text_prior",
    "llm_feature_group_product_text_prior",
    "llm_feature_group_core_interaction_prior",
    "llm_feature_group_product_interaction_prior",
    "llm_feature_group_core_text_share",
    "llm_feature_group_core_interaction_share",
)
PAIRWISE_COMPATIBILITY_COLUMNS = (
    "llm_pair_cat_r1_interaction_balance",
    "llm_pair_cat_r1_confidence_balance",
    "llm_pair_cat_r1_focus_alignment",
    "llm_pair_cat_r1_type_overlap",
    "llm_pair_cat_r1_electronic_complement",
    "llm_pair_cat_r1_steric_compatibility",
    "llm_pair_cat_r1_text_synergy",
    "llm_pair_cat_r1_reaction_center_score",
    "llm_pair_cat_r2_interaction_balance",
    "llm_pair_cat_r2_confidence_balance",
    "llm_pair_cat_r2_focus_alignment",
    "llm_pair_cat_r2_type_overlap",
    "llm_pair_cat_r2_electronic_complement",
    "llm_pair_cat_r2_steric_compatibility",
    "llm_pair_cat_r2_text_synergy",
    "llm_pair_cat_r2_reaction_center_score",
    "llm_pair_core_vs_product_score_margin",
    "llm_pair_core_vs_product_focus_margin",
)
HIGH_INFORMATION_FEATURE_COLUMNS = (
    # Frozen from story 60-0 on the audited sparse `core_emphasis` surface:
    # keep columns that were non-constant across the full 800-row table and
    # reached |train corr| >= 0.05 on the locked train split, then drop four
    # exact missingness complements that duplicated retained presence flags.
    "llm_mean_strength",
    "llm_precision_level",
    "llm_overall_confidence",
    "llm_interaction_coordination_ratio",
    "llm_interaction_covalent_ratio",
    "llm_interaction_van_der_waals_ratio",
    "llm_strength_very_strong_ratio",
    "llm_strength_strong_ratio",
    "llm_strength_moderate_ratio",
    "llm_strength_weak_ratio",
    "llm_electronic_electron_withdrawing_ratio",
    "llm_electronic_neutral_ratio",
    "llm_steric_favorable_ratio",
    "llm_steric_neutral_ratio",
    "llm_component_cat_ratio",
    "llm_component_r1_ratio",
    "llm_component_r2_ratio",
    "llm_semantic_steric_pressure",
    "llm_semantic_temperature_alignment",
    "llm_semantic_cat_focus",
    "llm_semantic_r1_focus",
    "llm_semantic_r2_focus",
    "llm_breakdown_steric",
    "llm_breakdown_electronic",
    "llm_breakdown_weak_interaction",
    "llm_breakdown_conformational",
    "llm_breakdown_temperature",
    "llm_text_directing_preorganization",
    "llm_text_boron_boryl",
    "llm_text_oxidative_orbital",
    "llm_text_transition_state",
    "llm_text_octahedral_pocket",
    "llm_component_cat_total_interactions",
    "llm_component_cat_mean_confidence",
    "llm_component_cat_interaction_coordination_ratio",
    "llm_component_cat_interaction_covalent_ratio",
    "llm_component_cat_strength_very_strong_ratio",
    "llm_component_cat_strength_strong_ratio",
    "llm_component_cat_steric_favorable_ratio",
    "llm_component_cat_steric_neutral_ratio",
    "llm_component_cat_text_steric_clash",
    "llm_component_cat_text_boron_boryl",
    "llm_component_cat_text_oxidative_orbital",
    "llm_component_cat_text_octahedral_pocket",
    "llm_component_r1_has_interactions",
    "llm_component_r1_has_reasoning_text",
    "llm_component_r1_total_interactions",
    "llm_component_r1_interaction_van_der_waals_ratio",
    "llm_component_r1_strength_moderate_ratio",
    "llm_component_r1_strength_weak_ratio",
    "llm_component_r1_electronic_electron_withdrawing_ratio",
    "llm_component_r1_electronic_neutral_ratio",
    "llm_component_r1_steric_hindered_ratio",
    "llm_component_r1_text_oxidative_orbital",
    "llm_component_r1_text_transition_state",
    "llm_component_r1_text_conformation_geometry",
    "llm_component_r1_text_octahedral_pocket",
    "llm_component_r2_has_interactions",
    "llm_component_r2_has_reasoning_text",
    "llm_component_r2_total_interactions",
    "llm_component_r2_mean_strength",
    "llm_component_r2_mean_confidence",
    "llm_component_r2_interaction_van_der_waals_ratio",
    "llm_component_r2_strength_weak_ratio",
    "llm_component_r2_electronic_neutral_ratio",
    "llm_component_r2_steric_hindered_ratio",
    "llm_component_r2_text_steric_clash",
    "llm_component_r2_text_boron_boryl",
    "llm_component_r2_text_outer_sphere_weak",
    "llm_component_r2_text_trans_cis_coordination",
    "llm_reweighted_text_steric_clash",
    "llm_reweighted_text_boron_boryl",
    "llm_reweighted_text_oxidative_orbital",
    "llm_reweighted_text_transition_state",
    "llm_reweighted_text_conformation_geometry",
    "llm_reweighted_text_octahedral_pocket",
    "llm_feature_group_core_text_prior",
    "llm_feature_group_product_text_prior",
    "llm_feature_group_core_interaction_prior",
    "llm_feature_group_product_interaction_prior",
    "llm_feature_group_core_text_share",
    "llm_feature_group_core_interaction_share",
)
_HIGH_INFORMATION_INDEXES: Optional[Sequence[int]] = None
CHEMISTRY_GROUNDED_COMPACT_FEATURE_COLUMNS = (
    # Story 69-3: keep the audited authority core from `high_information`,
    # drop the dead `r2_component` and `reweighted_text` bulk, and retain only
    # the chemistry-grounded global text sentinels that stayed alive in the
    # story 69-2 group audit.
    "llm_mean_strength",
    "llm_precision_level",
    "llm_overall_confidence",
    "llm_interaction_coordination_ratio",
    "llm_interaction_covalent_ratio",
    "llm_interaction_van_der_waals_ratio",
    "llm_strength_very_strong_ratio",
    "llm_strength_strong_ratio",
    "llm_strength_moderate_ratio",
    "llm_strength_weak_ratio",
    "llm_electronic_electron_withdrawing_ratio",
    "llm_electronic_neutral_ratio",
    "llm_steric_favorable_ratio",
    "llm_steric_neutral_ratio",
    "llm_component_cat_ratio",
    "llm_component_r1_ratio",
    "llm_semantic_steric_pressure",
    "llm_semantic_temperature_alignment",
    "llm_semantic_cat_focus",
    "llm_semantic_r1_focus",
    "llm_semantic_r2_focus",
    "llm_breakdown_steric",
    "llm_breakdown_electronic",
    "llm_breakdown_weak_interaction",
    "llm_breakdown_conformational",
    "llm_breakdown_temperature",
    "llm_text_boron_boryl",
    "llm_text_transition_state",
    "llm_component_cat_total_interactions",
    "llm_component_cat_mean_confidence",
    "llm_component_cat_interaction_coordination_ratio",
    "llm_component_cat_interaction_covalent_ratio",
    "llm_component_cat_strength_very_strong_ratio",
    "llm_component_cat_strength_strong_ratio",
    "llm_component_cat_steric_favorable_ratio",
    "llm_component_cat_steric_neutral_ratio",
    "llm_component_cat_text_steric_clash",
    "llm_component_cat_text_boron_boryl",
    "llm_component_cat_text_oxidative_orbital",
    "llm_component_cat_text_octahedral_pocket",
    "llm_component_r1_has_interactions",
    "llm_component_r1_has_reasoning_text",
    "llm_component_r1_total_interactions",
    "llm_component_r1_interaction_van_der_waals_ratio",
    "llm_component_r1_strength_moderate_ratio",
    "llm_component_r1_strength_weak_ratio",
    "llm_component_r1_electronic_electron_withdrawing_ratio",
    "llm_component_r1_electronic_neutral_ratio",
    "llm_component_r1_steric_hindered_ratio",
    "llm_component_r1_text_oxidative_orbital",
    "llm_component_r1_text_transition_state",
    "llm_component_r1_text_conformation_geometry",
    "llm_component_r1_text_octahedral_pocket",
    "llm_feature_group_core_text_prior",
    "llm_feature_group_product_text_prior",
    "llm_feature_group_core_interaction_prior",
    "llm_feature_group_product_interaction_prior",
    "llm_feature_group_core_text_share",
    "llm_feature_group_core_interaction_share",
)
_CHEMISTRY_GROUNDED_COMPACT_INDEXES: Optional[Sequence[int]] = None
PROPAGATED_COMPACT_V2_FEATURE_COLUMNS = (
    # Story 67-12: propagated rows benefit most from donor-transfer signals that
    # are stable across same-reference assignments. Keep the catalyst/substrate
    # core plus authority text priors, and drop product-facing r2 features.
    "llm_mean_strength",
    "llm_precision_level",
    "llm_overall_confidence",
    "llm_interaction_coordination_ratio",
    "llm_interaction_covalent_ratio",
    "llm_interaction_van_der_waals_ratio",
    "llm_strength_very_strong_ratio",
    "llm_strength_strong_ratio",
    "llm_strength_moderate_ratio",
    "llm_strength_weak_ratio",
    "llm_electronic_electron_withdrawing_ratio",
    "llm_electronic_neutral_ratio",
    "llm_steric_favorable_ratio",
    "llm_steric_neutral_ratio",
    "llm_component_cat_ratio",
    "llm_component_r1_ratio",
    "llm_semantic_steric_pressure",
    "llm_semantic_temperature_alignment",
    "llm_semantic_cat_focus",
    "llm_semantic_r1_focus",
    "llm_breakdown_steric",
    "llm_breakdown_electronic",
    "llm_breakdown_weak_interaction",
    "llm_breakdown_conformational",
    "llm_breakdown_temperature",
    "llm_text_directing_preorganization",
    "llm_text_oxidative_orbital",
    "llm_text_transition_state",
    "llm_text_octahedral_pocket",
    "llm_component_cat_total_interactions",
    "llm_component_cat_mean_confidence",
    "llm_component_cat_interaction_coordination_ratio",
    "llm_component_cat_interaction_covalent_ratio",
    "llm_component_cat_strength_very_strong_ratio",
    "llm_component_cat_strength_strong_ratio",
    "llm_component_cat_steric_favorable_ratio",
    "llm_component_cat_steric_neutral_ratio",
    "llm_component_cat_text_oxidative_orbital",
    "llm_component_cat_text_octahedral_pocket",
    "llm_component_r1_has_interactions",
    "llm_component_r1_has_reasoning_text",
    "llm_component_r1_total_interactions",
    "llm_component_r1_interaction_van_der_waals_ratio",
    "llm_component_r1_strength_moderate_ratio",
    "llm_component_r1_strength_weak_ratio",
    "llm_component_r1_electronic_electron_withdrawing_ratio",
    "llm_component_r1_electronic_neutral_ratio",
    "llm_component_r1_steric_hindered_ratio",
    "llm_component_r1_text_oxidative_orbital",
    "llm_component_r1_text_transition_state",
    "llm_component_r1_text_conformation_geometry",
    "llm_component_r1_text_octahedral_pocket",
    "llm_reweighted_text_oxidative_orbital",
    "llm_reweighted_text_transition_state",
    "llm_reweighted_text_conformation_geometry",
    "llm_reweighted_text_octahedral_pocket",
    "llm_feature_group_core_text_prior",
    "llm_feature_group_core_interaction_prior",
    "llm_feature_group_core_text_share",
    "llm_feature_group_core_interaction_share",
)
_PROPAGATED_COMPACT_V2_INDEXES: Optional[Sequence[int]] = None


def _resolve_feature_profile(profile: str = LLM_FEATURE_PROFILE_LEGACY) -> str:
    resolved = str(profile or LLM_FEATURE_PROFILE_LEGACY).strip().lower()
    if resolved not in _SUPPORTED_LLM_FEATURE_PROFILES:
        raise ValueError(
            f"Unknown llm feature profile: {profile}. "
            f"Expected one of {list(_SUPPORTED_LLM_FEATURE_PROFILES)}."
        )
    return resolved


def _reweighted_text_bucket_columns() -> List[str]:
    return [f"llm_reweighted_text_{name}" for name in TEXT_SEMANTIC_BUCKETS]


def _pair_feature_columns() -> List[str]:
    return list(PAIRWISE_COMPATIBILITY_COLUMNS)


def _high_information_feature_columns() -> List[str]:
    return list(HIGH_INFORMATION_FEATURE_COLUMNS)


def _high_information_indexes() -> Sequence[int]:
    global _HIGH_INFORMATION_INDEXES
    if _HIGH_INFORMATION_INDEXES is None:
        core_columns = llm_feature_columns(profile=LLM_FEATURE_PROFILE_CORE_EMPHASIS)
        name_to_index = {name: idx for idx, name in enumerate(core_columns)}
        _HIGH_INFORMATION_INDEXES = tuple(name_to_index[name] for name in HIGH_INFORMATION_FEATURE_COLUMNS)
    return _HIGH_INFORMATION_INDEXES


def _propagated_compact_v2_feature_columns() -> List[str]:
    return list(PROPAGATED_COMPACT_V2_FEATURE_COLUMNS)


def _chemistry_grounded_compact_feature_columns() -> List[str]:
    return list(CHEMISTRY_GROUNDED_COMPACT_FEATURE_COLUMNS)


def _chemistry_grounded_compact_indexes() -> Sequence[int]:
    global _CHEMISTRY_GROUNDED_COMPACT_INDEXES
    if _CHEMISTRY_GROUNDED_COMPACT_INDEXES is None:
        core_columns = llm_feature_columns(profile=LLM_FEATURE_PROFILE_CORE_EMPHASIS)
        name_to_index = {name: idx for idx, name in enumerate(core_columns)}
        _CHEMISTRY_GROUNDED_COMPACT_INDEXES = tuple(
            name_to_index[name] for name in CHEMISTRY_GROUNDED_COMPACT_FEATURE_COLUMNS
        )
    return _CHEMISTRY_GROUNDED_COMPACT_INDEXES


def _propagated_compact_v2_indexes() -> Sequence[int]:
    global _PROPAGATED_COMPACT_V2_INDEXES
    if _PROPAGATED_COMPACT_V2_INDEXES is None:
        core_columns = llm_feature_columns(profile=LLM_FEATURE_PROFILE_CORE_EMPHASIS)
        name_to_index = {name: idx for idx, name in enumerate(core_columns)}
        _PROPAGATED_COMPACT_V2_INDEXES = tuple(
            name_to_index[name] for name in PROPAGATED_COMPACT_V2_FEATURE_COLUMNS
        )
    return _PROPAGATED_COMPACT_V2_INDEXES


def _safe_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(value) for value in values) / len(values))


def _component_feature_columns(component: str, *, include_presence_masks: bool = True) -> List[str]:
    cols: List[str] = []
    if include_presence_masks:
        cols.extend(
            [
                f"llm_component_{component}_has_interactions",
                f"llm_component_{component}_missing_interactions",
                f"llm_component_{component}_has_reasoning_text",
                f"llm_component_{component}_missing_reasoning_text",
            ]
        )
    cols.extend(
        [
            f"llm_component_{component}_total_interactions",
            f"llm_component_{component}_mean_strength",
            f"llm_component_{component}_mean_confidence",
            f"llm_component_{component}_mean_distance",
        ]
    )
    cols.extend([f"llm_component_{component}_interaction_{name}_ratio" for name in INTERACTION_TYPES])
    cols.extend([f"llm_component_{component}_strength_{name}_ratio" for name in STRENGTH_CATEGORIES])
    cols.extend([f"llm_component_{component}_electronic_{name}_ratio" for name in ELECTRONIC_EFFECTS])
    cols.extend([f"llm_component_{component}_steric_{name}_ratio" for name in STERIC_EFFECTS])
    cols.extend([f"llm_component_{component}_text_{name}" for name in TEXT_SEMANTIC_BUCKETS])
    return cols


def llm_semantic_branch_columns_from_feature_columns(feature_columns: Sequence[str]) -> Dict[str, List[str]]:
    branches: Dict[str, List[str]] = {name: [] for name in LLM_SEMANTIC_BRANCH_ORDER}
    for column in feature_columns:
        if column.startswith("llm_component_cat_"):
            branches["cat"].append(column)
        elif column.startswith("llm_component_r1_"):
            branches["r1"].append(column)
        elif column.startswith("llm_component_r2_"):
            branches["r2"].append(column)
        elif column.startswith("llm_pair_"):
            branches["pair"].append(column)
        else:
            branches["global"].append(column)
    return branches


def llm_semantic_branch_columns(profile: str = LLM_FEATURE_PROFILE_LEGACY) -> Dict[str, List[str]]:
    return llm_semantic_branch_columns_from_feature_columns(llm_feature_columns(profile=profile))


def llm_semantic_branch_index_map(feature_columns: Sequence[str]) -> Dict[str, List[int]]:
    branch_columns = llm_semantic_branch_columns_from_feature_columns(feature_columns)
    name_to_index = {name: idx for idx, name in enumerate(feature_columns)}
    return {
        branch: [int(name_to_index[column]) for column in columns if column in name_to_index]
        for branch, columns in branch_columns.items()
    }


def llm_pair_group_name(column: str) -> Optional[str]:
    if column.startswith("llm_pair_cat_r1_"):
        return "core_pair"
    if column.startswith("llm_pair_cat_r2_"):
        return "product_pair"
    if column.startswith("llm_pair_"):
        return "contrast"
    return None


def llm_pair_group_columns_from_feature_columns(feature_columns: Sequence[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {name: [] for name in LLM_PAIR_GROUP_ORDER}
    for column in feature_columns:
        group_name = llm_pair_group_name(column)
        if group_name is not None:
            groups[group_name].append(column)
    return groups


def llm_pair_group_columns(profile: str = LLM_FEATURE_PROFILE_LEGACY) -> Dict[str, List[str]]:
    return llm_pair_group_columns_from_feature_columns(llm_feature_columns(profile=profile))


def llm_pair_group_index_map(feature_columns: Sequence[str]) -> Dict[str, List[int]]:
    group_columns = llm_pair_group_columns_from_feature_columns(feature_columns)
    name_to_index = {name: idx for idx, name in enumerate(feature_columns)}
    return {
        group: [int(name_to_index[column]) for column in columns if column in name_to_index]
        for group, columns in group_columns.items()
    }


def llm_semantic_group_name(column: str) -> str:
    if column.startswith("llm_component_cat_"):
        return "cat_component"
    if column.startswith("llm_component_r1_"):
        return "r1_component"
    if column.startswith("llm_component_r2_"):
        return "r2_component"
    if column.startswith("llm_pair_"):
        return "pair_component"
    if column.startswith("llm_reweighted_text_"):
        return "reweighted_text"
    if column.startswith("llm_feature_group_"):
        return "feature_group_prior"
    if column.startswith("llm_text_"):
        return "global_text"
    if column.startswith("llm_semantic_"):
        return "semantic_meta"
    if column.startswith("llm_breakdown_"):
        return "confidence_breakdown"
    if (
        column.startswith("llm_interaction_")
        or column.startswith("llm_strength_")
        or column.startswith("llm_electronic_")
        or column.startswith("llm_steric_")
    ):
        return "global_ratios"
    if column in {
        "llm_total_interactions",
        "llm_mean_strength",
        "llm_mean_confidence",
        "llm_mean_distance",
        "llm_precision_level",
        "llm_overall_confidence",
    }:
        return "global_scalar"
    return "other"


def llm_semantic_group_columns_from_feature_columns(feature_columns: Sequence[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {name: [] for name in LLM_SEMANTIC_GROUP_ORDER}
    for column in feature_columns:
        groups[llm_semantic_group_name(column)].append(column)
    return groups


def llm_semantic_group_columns(profile: str = LLM_FEATURE_PROFILE_LEGACY) -> Dict[str, List[str]]:
    return llm_semantic_group_columns_from_feature_columns(llm_feature_columns(profile=profile))


def llm_semantic_group_index_map(feature_columns: Sequence[str]) -> Dict[str, List[int]]:
    group_columns = llm_semantic_group_columns_from_feature_columns(feature_columns)
    name_to_index = {name: idx for idx, name in enumerate(feature_columns)}
    return {
        group: [int(name_to_index[column]) for column in columns if column in name_to_index]
        for group, columns in group_columns.items()
    }


def llm_feature_columns(profile: str = LLM_FEATURE_PROFILE_LEGACY) -> List[str]:
    resolved_profile = _resolve_feature_profile(profile)
    include_presence_masks = resolved_profile not in {
        LLM_FEATURE_PROFILE_COMPONENT_ONLY,
        LLM_FEATURE_PROFILE_PAIR_AWARE,
    }
    cols = [
        "llm_total_interactions",
        "llm_mean_strength",
        "llm_mean_confidence",
        "llm_mean_distance",
        "llm_precision_level",
        "llm_overall_confidence",
    ]
    cols.extend([f"llm_interaction_{name}_ratio" for name in INTERACTION_TYPES])
    cols.extend([f"llm_strength_{name}_ratio" for name in STRENGTH_CATEGORIES])
    cols.extend([f"llm_electronic_{name}_ratio" for name in ELECTRONIC_EFFECTS])
    cols.extend([f"llm_steric_{name}_ratio" for name in STERIC_EFFECTS])
    cols.extend([f"llm_component_{name}_ratio" for name in COMPONENTS])
    cols.extend([f"llm_semantic_{name}" for name in SEMANTIC_META_FEATURES])
    cols.extend([f"llm_breakdown_{name}" for name in CONFIDENCE_BREAKDOWN_FEATURES])
    cols.extend([f"llm_text_{name}" for name in TEXT_SEMANTIC_BUCKETS])
    for component in COMPONENTS:
        cols.extend(_component_feature_columns(component, include_presence_masks=include_presence_masks))
    if resolved_profile == LLM_FEATURE_PROFILE_CORE_EMPHASIS:
        cols.extend(_reweighted_text_bucket_columns())
        cols.extend(FEATURE_GROUP_PRIOR_COLUMNS)
    if resolved_profile == LLM_FEATURE_PROFILE_HIGH_INFORMATION:
        return _high_information_feature_columns()
    if resolved_profile == LLM_FEATURE_PROFILE_CHEMISTRY_GROUNDED_COMPACT:
        return _chemistry_grounded_compact_feature_columns()
    if resolved_profile == LLM_FEATURE_PROFILE_PROPAGATED_COMPACT_V2:
        return _propagated_compact_v2_feature_columns()
    if resolved_profile == LLM_FEATURE_PROFILE_PAIR_AWARE:
        cols.extend(_pair_feature_columns())
    return cols


def _coerce_meta_feature(meta: Dict[str, object], key: str) -> float:
    value = meta.get(key, 0.0)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def _get_interaction_field(interaction: object, key: str) -> Optional[object]:
    if isinstance(interaction, dict):
        return interaction.get(key)
    return getattr(interaction, key, None)


def _collect_semantic_text(literature: Optional[dict]) -> str:
    if not literature:
        return ""
    segments: List[str] = []
    meta = literature.get("meta") or {}
    if isinstance(meta, dict):
        for key in TEXT_META_FIELDS:
            value = meta.get(key)
            if isinstance(value, str) and value.strip():
                segments.append(value.strip())
    for interaction in _iter_interactions(literature):
        reasoning = _get_interaction_field(interaction, "reasoning")
        if isinstance(reasoning, str) and reasoning.strip():
            segments.append(reasoning.strip())
    return " ".join(segments).lower()


def _collect_component_semantic_text(literature: Optional[dict], component: str) -> str:
    if not literature:
        return ""
    segments: List[str] = []
    for interaction in literature.get(component) or []:
        reasoning = _get_interaction_field(interaction, "reasoning")
        if isinstance(reasoning, str) and reasoning.strip():
            segments.append(reasoning.strip())
    return " ".join(segments).lower()


def _text_bucket_score(text: str, keywords: Sequence[str]) -> float:
    if not text:
        return 0.0
    hits = 0
    for keyword in keywords:
        if keyword and keyword.lower() in text:
            hits += 1
    if hits == 0:
        return 0.0
    return float(min(hits / 3.0, 1.0))


def _bucket_score(component_bucket_scores: Dict[str, Dict[str, float]], component: str, name: str) -> float:
    return float(component_bucket_scores.get(component, {}).get(name, 0.0))


def _component_distribution(bucket: Dict[str, object], key: str, names: Sequence[str]) -> Dict[str, float]:
    total = int(bucket.get("total") or 0)
    denom = float(total) if total else 1.0
    counts = bucket.get(key) or {}
    return {name: float(counts.get(name, 0)) / denom for name in names}


def _distribution_overlap(left: Dict[str, float], right: Dict[str, float], names: Sequence[str]) -> float:
    return float(np.clip(sum(min(float(left.get(name, 0.0)), float(right.get(name, 0.0))) for name in names), 0.0, 1.0))


def _pair_interaction_balance(left_total: int, right_total: int) -> float:
    left_presence = 1.0 - math.exp(-max(int(left_total), 0))
    right_presence = 1.0 - math.exp(-max(int(right_total), 0))
    return float(np.clip(math.sqrt(left_presence * right_presence), 0.0, 1.0))


def _pair_focus_alignment(meta: Dict[str, object], left: str, right: str) -> float:
    return float(
        np.clip(
            math.sqrt(_coerce_meta_feature(meta, f"{left}_focus") * _coerce_meta_feature(meta, f"{right}_focus")),
            0.0,
            1.0,
        )
    )


def _pair_electronic_complement(
    left_distribution: Dict[str, float],
    right_distribution: Dict[str, float],
) -> float:
    complementary = max(
        float(left_distribution.get("electron_donating", 0.0)) * float(right_distribution.get("electron_withdrawing", 0.0)),
        float(left_distribution.get("electron_withdrawing", 0.0)) * float(right_distribution.get("electron_donating", 0.0)),
    )
    assisted = 0.5 * max(
        float(left_distribution.get("electron_donating", 0.0)) * float(right_distribution.get("neutral", 0.0)),
        float(left_distribution.get("neutral", 0.0)) * float(right_distribution.get("electron_donating", 0.0)),
        float(left_distribution.get("electron_withdrawing", 0.0)) * float(right_distribution.get("neutral", 0.0)),
        float(left_distribution.get("neutral", 0.0)) * float(right_distribution.get("electron_withdrawing", 0.0)),
    )
    return float(np.clip(complementary + assisted, 0.0, 1.0))


def _pair_steric_compatibility(
    left_distribution: Dict[str, float],
    right_distribution: Dict[str, float],
) -> float:
    hindered_penalty = max(
        float(left_distribution.get("hindered", 0.0)),
        float(right_distribution.get("hindered", 0.0)),
    )
    favorable_support = math.sqrt(
        float(left_distribution.get("favorable", 0.0)) * float(right_distribution.get("favorable", 0.0))
    )
    return float(np.clip(0.5 * (1.0 - hindered_penalty) + 0.5 * favorable_support, 0.0, 1.0))


def _pair_text_synergy(
    component_bucket_scores: Dict[str, Dict[str, float]],
    pair_name: str,
) -> float:
    if pair_name == "cat_r1":
        synergy_terms = [
            math.sqrt(
                _bucket_score(component_bucket_scores, "cat", "directing_preorganization")
                * max(
                    _bucket_score(component_bucket_scores, "r1", "transition_state"),
                    _bucket_score(component_bucket_scores, "r1", "conformation_geometry"),
                )
            ),
            math.sqrt(
                _bucket_score(component_bucket_scores, "cat", "trans_cis_coordination")
                * max(
                    _bucket_score(component_bucket_scores, "r1", "directing_preorganization"),
                    _bucket_score(component_bucket_scores, "r1", "oxidative_orbital"),
                )
            ),
            math.sqrt(
                _bucket_score(component_bucket_scores, "cat", "oxidative_orbital")
                * _bucket_score(component_bucket_scores, "r1", "conformation_geometry")
            ),
        ]
    else:
        synergy_terms = [
            math.sqrt(
                _bucket_score(component_bucket_scores, "cat", "directing_preorganization")
                * _bucket_score(component_bucket_scores, "r2", "boron_boryl")
            ),
            math.sqrt(
                _bucket_score(component_bucket_scores, "cat", "transition_state")
                * max(
                    _bucket_score(component_bucket_scores, "r2", "outer_sphere_weak"),
                    _bucket_score(component_bucket_scores, "r2", "temperature_window"),
                )
            ),
            math.sqrt(
                _bucket_score(component_bucket_scores, "cat", "trans_cis_coordination")
                * _bucket_score(component_bucket_scores, "r2", "boron_boryl")
            ),
        ]
    return float(np.clip(_safe_mean(synergy_terms), 0.0, 1.0))


def _iter_interactions(literature: Optional[dict]) -> Iterable[Dict[str, object]]:
    if not literature:
        return []
    interactions: List[Dict[str, object]] = []
    for component in COMPONENTS:
        for inter in literature.get(component) or []:
            if isinstance(inter, dict):
                data = dict(inter)
            elif hasattr(inter, "model_dump"):
                data = inter.model_dump()
            elif hasattr(inter, "dict"):
                data = inter.dict()
            elif hasattr(inter, "__dict__"):
                data = dict(inter.__dict__)
            else:
                data = {"interaction": inter}
            data.setdefault("component", component)
            interactions.append(data)
    return interactions


def build_llm_feature_vector(
    literature: Optional[dict],
    profile: str = LLM_FEATURE_PROFILE_LEGACY,
) -> np.ndarray:
    resolved_profile = _resolve_feature_profile(profile)
    if resolved_profile == LLM_FEATURE_PROFILE_HIGH_INFORMATION:
        core_vector = build_llm_feature_vector(literature, profile=LLM_FEATURE_PROFILE_CORE_EMPHASIS)
        return core_vector[np.asarray(_high_information_indexes(), dtype=np.int64)]
    if resolved_profile == LLM_FEATURE_PROFILE_CHEMISTRY_GROUNDED_COMPACT:
        core_vector = build_llm_feature_vector(literature, profile=LLM_FEATURE_PROFILE_CORE_EMPHASIS)
        return core_vector[np.asarray(_chemistry_grounded_compact_indexes(), dtype=np.int64)]
    if resolved_profile == LLM_FEATURE_PROFILE_PROPAGATED_COMPACT_V2:
        core_vector = build_llm_feature_vector(literature, profile=LLM_FEATURE_PROFILE_CORE_EMPHASIS)
        return core_vector[np.asarray(_propagated_compact_v2_indexes(), dtype=np.int64)]
    include_presence_masks = resolved_profile not in {
        LLM_FEATURE_PROFILE_COMPONENT_ONLY,
        LLM_FEATURE_PROFILE_PAIR_AWARE,
    }
    total = 0
    strength_sum = 0.0
    conf_sum = 0.0
    dist_sum = 0.0
    dist_count = 0
    type_counts = {name: 0 for name in INTERACTION_TYPES}
    strength_counts = {name: 0 for name in STRENGTH_CATEGORIES}
    electronic_counts = {name: 0 for name in ELECTRONIC_EFFECTS}
    steric_counts = {name: 0 for name in STERIC_EFFECTS}
    component_counts = {name: 0 for name in COMPONENTS}
    component_stats = {
        name: {
            "total": 0,
            "strength_sum": 0.0,
            "conf_sum": 0.0,
            "dist_sum": 0.0,
            "dist_count": 0,
            "type_counts": {key: 0 for key in INTERACTION_TYPES},
            "strength_counts": {key: 0 for key in STRENGTH_CATEGORIES},
            "electronic_counts": {key: 0 for key in ELECTRONIC_EFFECTS},
            "steric_counts": {key: 0 for key in STERIC_EFFECTS},
        }
        for name in COMPONENTS
    }
    meta = literature.get("meta") or {} if literature else {}
    if not isinstance(meta, dict):
        meta = {}

    for inter in _iter_interactions(literature):
        total += 1
        inter_type = _get_interaction_field(inter, "interaction_type")
        if inter_type in type_counts:
            type_counts[inter_type] += 1
        strength_cat = _get_interaction_field(inter, "strength_category")
        if strength_cat in strength_counts:
            strength_counts[strength_cat] += 1
        electronic = _get_interaction_field(inter, "electronic_effect")
        if electronic in electronic_counts:
            electronic_counts[electronic] += 1
        steric = _get_interaction_field(inter, "steric_effect")
        if steric in steric_counts:
            steric_counts[steric] += 1
        component = _get_interaction_field(inter, "component")
        if component in component_counts:
            component_counts[component] += 1
        component_bucket = component_stats.get(component)
        if component_bucket is not None:
            component_bucket["total"] += 1
            if inter_type in component_bucket["type_counts"]:
                component_bucket["type_counts"][inter_type] += 1
            if strength_cat in component_bucket["strength_counts"]:
                component_bucket["strength_counts"][strength_cat] += 1
            if electronic in component_bucket["electronic_counts"]:
                component_bucket["electronic_counts"][electronic] += 1
            if steric in component_bucket["steric_counts"]:
                component_bucket["steric_counts"][steric] += 1

        strength_score = _get_interaction_field(inter, "strength_score")
        if strength_score is not None:
            try:
                strength_value = float(strength_score)
                strength_sum += strength_value
                if component_bucket is not None:
                    component_bucket["strength_sum"] += strength_value
            except (TypeError, ValueError):
                pass
        conf = _get_interaction_field(inter, "confidence")
        if conf is not None:
            try:
                conf_value = float(conf)
                conf_sum += conf_value
                if component_bucket is not None:
                    component_bucket["conf_sum"] += conf_value
            except (TypeError, ValueError):
                pass
        dist = _get_interaction_field(inter, "distance")
        if dist is not None:
            try:
                dist_value = float(dist)
                dist_sum += dist_value
                dist_count += 1
                if component_bucket is not None:
                    component_bucket["dist_sum"] += dist_value
                    component_bucket["dist_count"] += 1
            except (TypeError, ValueError):
                pass

    if total > 0:
        mean_strength = strength_sum / total
        mean_conf = conf_sum / total
    else:
        mean_strength = 0.0
        mean_conf = 0.0
    mean_dist = dist_sum / dist_count if dist_count else 0.0

    precision_level = 0.0
    overall_conf = 0.0
    if literature:
        for key in ("precision_level", "literature_precision_level"):
            if key in meta:
                precision_level = meta.get(key)
                break
        if precision_level == 0 and "precision_level" in literature:
            precision_level = literature.get("precision_level")
        if precision_level == 0 and "literature_precision_level" in literature:
            precision_level = literature.get("literature_precision_level")
        overall_conf = meta.get("overall_confidence", literature.get("overall_confidence", 0.0))
    try:
        precision_scaled = float(precision_level) / 5.0 if precision_level else 0.0
    except (TypeError, ValueError):
        precision_scaled = 0.0
    try:
        overall_conf = float(overall_conf) if overall_conf is not None else 0.0
    except (TypeError, ValueError):
        overall_conf = 0.0

    total_interactions = math.log1p(total)
    denom = float(total) if total else 1.0
    features = [
        total_interactions,
        mean_strength,
        mean_conf,
        mean_dist,
        precision_scaled,
        overall_conf,
    ]
    features.extend([type_counts[name] / denom for name in INTERACTION_TYPES])
    features.extend([strength_counts[name] / denom for name in STRENGTH_CATEGORIES])
    features.extend([electronic_counts[name] / denom for name in ELECTRONIC_EFFECTS])
    features.extend([steric_counts[name] / denom for name in STERIC_EFFECTS])
    features.extend([component_counts[name] / denom for name in COMPONENTS])
    features.extend([_coerce_meta_feature(meta if literature else {}, name) for name in SEMANTIC_META_FEATURES])
    breakdown = (meta if literature else {}).get("confidence_breakdown", {})
    if not isinstance(breakdown, dict):
        breakdown = {}
    features.extend([_coerce_meta_feature(breakdown, name) for name in CONFIDENCE_BREAKDOWN_FEATURES])
    semantic_text = _collect_semantic_text(literature)
    features.extend([_text_bucket_score(semantic_text, keywords) for keywords in TEXT_SEMANTIC_BUCKETS.values()])
    component_text_bucket_scores = {component: {} for component in COMPONENTS}
    component_text_presence = {component: 0.0 for component in COMPONENTS}
    component_interaction_presence = {component: 0.0 for component in COMPONENTS}
    component_mean_confidence = {component: 0.0 for component in COMPONENTS}
    for component in COMPONENTS:
        component_bucket = component_stats[component]
        component_total = int(component_bucket["total"])
        component_denom = float(component_total) if component_total else 1.0
        component_has_interactions_value = 1.0 if component_total else 0.0
        component_text = _collect_component_semantic_text(literature, component)
        component_has_reasoning_text_value = 1.0 if component_text else 0.0
        component_mean_strength = component_bucket["strength_sum"] / component_total if component_total else 0.0
        component_mean_confidence_value = component_bucket["conf_sum"] / component_total if component_total else 0.0
        component_mean_distance = (
            component_bucket["dist_sum"] / component_bucket["dist_count"]
            if component_bucket["dist_count"]
            else 0.0
        )
        component_text_presence[component] = component_has_reasoning_text_value
        component_interaction_presence[component] = component_has_interactions_value
        component_mean_confidence[component] = component_mean_confidence_value
        if include_presence_masks:
            features.extend(
                [
                    component_has_interactions_value,
                    1.0 - component_has_interactions_value,
                    component_has_reasoning_text_value,
                    1.0 - component_has_reasoning_text_value,
                ]
            )
        features.extend(
            [
                math.log1p(component_total),
                component_mean_strength,
                component_mean_confidence_value,
                component_mean_distance,
            ]
        )
        features.extend(
            [component_bucket["type_counts"][name] / component_denom for name in INTERACTION_TYPES]
        )
        features.extend(
            [component_bucket["strength_counts"][name] / component_denom for name in STRENGTH_CATEGORIES]
        )
        features.extend(
            [component_bucket["electronic_counts"][name] / component_denom for name in ELECTRONIC_EFFECTS]
        )
        features.extend(
            [component_bucket["steric_counts"][name] / component_denom for name in STERIC_EFFECTS]
        )
        component_bucket_values = [
            _text_bucket_score(component_text, keywords) for keywords in TEXT_SEMANTIC_BUCKETS.values()
        ]
        component_text_bucket_scores[component] = {
            name: value for name, value in zip(TEXT_SEMANTIC_BUCKETS, component_bucket_values)
        }
        features.extend(component_bucket_values)

    if resolved_profile == LLM_FEATURE_PROFILE_CORE_EMPHASIS:
        reweighted_bucket_values: List[float] = []
        r2_text_presence = float(component_text_presence["r2"])
        r2_interaction_presence = float(component_interaction_presence["r2"])
        for bucket_name in TEXT_SEMANTIC_BUCKETS:
            cat_score = float(component_text_bucket_scores["cat"].get(bucket_name, 0.0))
            r1_score = float(component_text_bucket_scores["r1"].get(bucket_name, 0.0))
            r2_score = float(component_text_bucket_scores["r2"].get(bucket_name, 0.0))
            weighted = (
                _REWEIGHTED_TEXT_COMPONENT_WEIGHTS["cat"] * cat_score
                + _REWEIGHTED_TEXT_COMPONENT_WEIGHTS["r1"] * r1_score
                + _REWEIGHTED_TEXT_COMPONENT_WEIGHTS["r2"] * r2_text_presence * r2_score
            )
            reweighted_bucket_values.append(float(np.clip(weighted, 0.0, 1.0)))
        features.extend(reweighted_bucket_values)

        cat_focus = _coerce_meta_feature(meta, "cat_focus")
        r1_focus = _coerce_meta_feature(meta, "r1_focus")
        r2_focus = _coerce_meta_feature(meta, "r2_focus")
        core_text_prior = float(
            np.clip(
                0.55 * _safe_mean([component_text_bucket_scores["cat"].get(name, 0.0) for name in _CORE_TEXT_BUCKET_NAMES])
                + 0.45 * _safe_mean([component_text_bucket_scores["r1"].get(name, 0.0) for name in _CORE_TEXT_BUCKET_NAMES]),
                0.0,
                1.0,
            )
        )
        product_text_prior = float(
            np.clip(
                0.30
                * r2_text_presence
                * _safe_mean([component_text_bucket_scores["r2"].get(name, 0.0) for name in _PRODUCT_TEXT_BUCKET_NAMES]),
                0.0,
                1.0,
            )
        )
        core_interaction_prior = float(
            np.clip(
                0.35 * cat_focus
                + 0.25 * r1_focus
                + 0.25 * float(component_interaction_presence["cat"])
                + 0.15 * float(component_interaction_presence["r1"]),
                0.0,
                1.0,
            )
        )
        product_interaction_prior = float(
            np.clip(
                0.35
                * (
                    0.55 * r2_interaction_presence
                    + 0.25 * r2_text_presence
                    + 0.20 * r2_focus * float(component_mean_confidence["r2"] > 0.0)
                ),
                0.0,
                1.0,
            )
        )
        core_text_share = float(
            np.clip(
                core_text_prior / max(core_text_prior + product_text_prior, 1e-6),
                0.0,
                1.0,
            )
        )
        core_interaction_share = float(
            np.clip(
                core_interaction_prior / max(core_interaction_prior + product_interaction_prior, 1e-6),
                0.0,
                1.0,
            )
        )
        features.extend(
            [
                core_text_prior,
                product_text_prior,
                core_interaction_prior,
                product_interaction_prior,
                core_text_share,
                core_interaction_share,
            ]
        )

    if resolved_profile == LLM_FEATURE_PROFILE_PAIR_AWARE:
        pair_feature_values: List[float] = []
        pair_scores: Dict[str, float] = {}
        pair_focus_scores: Dict[str, float] = {}
        for pair_name, left, right in (("cat_r1", "cat", "r1"), ("cat_r2", "cat", "r2")):
            left_bucket = component_stats[left]
            right_bucket = component_stats[right]
            left_type_distribution = _component_distribution(left_bucket, "type_counts", INTERACTION_TYPES)
            right_type_distribution = _component_distribution(right_bucket, "type_counts", INTERACTION_TYPES)
            left_electronic_distribution = _component_distribution(left_bucket, "electronic_counts", ELECTRONIC_EFFECTS)
            right_electronic_distribution = _component_distribution(right_bucket, "electronic_counts", ELECTRONIC_EFFECTS)
            left_steric_distribution = _component_distribution(left_bucket, "steric_counts", STERIC_EFFECTS)
            right_steric_distribution = _component_distribution(right_bucket, "steric_counts", STERIC_EFFECTS)

            interaction_balance = _pair_interaction_balance(int(left_bucket["total"]), int(right_bucket["total"]))
            confidence_balance = float(
                np.clip(
                    math.sqrt(component_mean_confidence[left] * component_mean_confidence[right]),
                    0.0,
                    1.0,
                )
            )
            focus_alignment = _pair_focus_alignment(meta, left, right)
            type_overlap = _distribution_overlap(left_type_distribution, right_type_distribution, INTERACTION_TYPES)
            electronic_complement = _pair_electronic_complement(
                left_electronic_distribution,
                right_electronic_distribution,
            )
            steric_compatibility = _pair_steric_compatibility(
                left_steric_distribution,
                right_steric_distribution,
            )
            text_synergy = _pair_text_synergy(component_text_bucket_scores, pair_name)
            reaction_center_score = float(
                np.clip(
                    0.18 * interaction_balance
                    + 0.14 * confidence_balance
                    + 0.16 * focus_alignment
                    + 0.14 * type_overlap
                    + 0.12 * electronic_complement
                    + 0.10 * steric_compatibility
                    + 0.12 * text_synergy
                    + 0.04 * _coerce_meta_feature(meta, "temperature_alignment"),
                    0.0,
                    1.0,
                )
            )
            pair_feature_values.extend(
                [
                    interaction_balance,
                    confidence_balance,
                    focus_alignment,
                    type_overlap,
                    electronic_complement,
                    steric_compatibility,
                    text_synergy,
                    reaction_center_score,
                ]
            )
            pair_scores[pair_name] = reaction_center_score
            pair_focus_scores[pair_name] = focus_alignment
        features.extend(pair_feature_values)
        features.extend(
            [
                float(np.clip(0.5 + 0.5 * (pair_scores["cat_r1"] - pair_scores["cat_r2"]), 0.0, 1.0)),
                float(np.clip(0.5 + 0.5 * (pair_focus_scores["cat_r1"] - pair_focus_scores["cat_r2"]), 0.0, 1.0)),
            ]
        )

    return np.asarray(features, dtype=np.float32)
