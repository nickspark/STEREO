from __future__ import annotations

from typing import List, Literal, Optional, Tuple
import re

try:
    from pydantic.v1 import BaseModel, Field, validator, root_validator
except ImportError:  # pragma: no cover - fallback for pydantic v1
    from pydantic import BaseModel, Field, validator, root_validator


_ATOM_ID_PATTERN = re.compile(r"^(?:(?P<symbol>[A-Za-z]{1,3})[:#\-])?(?P<map_num>\d+)$")


class AtomRole(BaseModel):
    """Role annotation for a single atom."""

    atom_id: str = Field(
        description="Atom map identifier like 'C:5' or '5'",
        examples=["C:5", "N:3", "Pd:1", "5"],
    )
    role: Literal[
        "not_participant",
        "participant",
        "active_site",
        "spectator",
    ] = Field(description="Role of the atom in the reaction")
    participation_detail: Optional[str] = Field(
        default=None,
        description="Detailed participation, e.g. 'as_carboxyl_carbon_with_Pd'",
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")

    @validator("atom_id")
    def _validate_atom_id(cls, value: str) -> str:
        if not _ATOM_ID_PATTERN.match(value.strip()):
            raise ValueError(f"Invalid atom_id format: {value}")
        return value


class Interaction(BaseModel):
    """Atom-level interaction extracted from literature."""

    atom_indices: Optional[List[int]] = Field(
        default=None,
        description="Optional graph indices corresponding to the interaction atoms",
        examples=[[0, 5]],
    )
    atoms: List[str] = Field(
        min_items=2,
        max_items=2,
        description="Two atom map IDs participating in the interaction",
        examples=[["C:5", "Pd:1"], ["N:3", "O:7"]],
    )
    component: Optional[Literal["cat", "r1", "r2", "p1"]] = Field(
        default=None,
        description="Reaction component this interaction belongs to",
    )
    interaction_type: Literal[
        "coordination",
        "h_bond",
        "covalent",
        "pi_stacking",
        "ionic",
        "van_der_waals",
    ] = Field(description="Interaction type")
    strength_category: Literal[
        "very_strong",
        "strong",
        "moderate",
        "weak",
        "negligible",
    ] = Field(description="Discrete strength category")
    strength_score: float = Field(ge=0.0, le=10.0, description="Continuous score 0-10")
    distance: Optional[float] = Field(default=None, description="Distance in Angstroms")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    reasoning: str = Field(description="Reasoning text")
    electronic_effect: Optional[
        Literal["electron_donating", "electron_withdrawing", "neutral"]
    ] = Field(default=None, description="Electronic effect")
    steric_effect: Optional[Literal["hindered", "favorable", "neutral"]] = Field(
        default=None, description="Steric effect"
    )

    @validator("atom_indices")
    def _validate_atom_indices(cls, value: Optional[List[int]]) -> Optional[List[int]]:
        if value is None:
            return value
        if len(value) != 2:
            raise ValueError("atom_indices must include exactly 2 indices")
        if value[0] == value[1]:
            raise ValueError("atom_indices must be distinct")
        if any(idx < 0 for idx in value):
            raise ValueError("atom_indices must be non-negative")
        return value

    @validator("atoms")
    def _validate_atoms(cls, value: List[str]) -> List[str]:
        if len(value) != 2:
            raise ValueError("Interaction must include exactly 2 atoms")
        if value[0] == value[1]:
            raise ValueError("Interaction atoms must be distinct")
        for atom_id in value:
            if not _ATOM_ID_PATTERN.match(atom_id.strip()):
                raise ValueError(f"Invalid atom_id format: {atom_id}")
        return value

    @validator("distance")
    def _validate_distance(cls, value: Optional[float]) -> Optional[float]:
        if value is None:
            return value
        if value <= 0:
            raise ValueError("distance must be positive")
        return value

    @root_validator
    def _validate_strength_category(cls, values: dict) -> dict:
        score = values.get("strength_score")
        category = values.get("strength_category")
        if score is None or category is None:
            return values
        expected = _score_to_category(score)
        if category != expected:
            raise ValueError(
                f"strength_category '{category}' inconsistent with score {score} (expected '{expected}')"
            )
        return values


class LiteratureExtraction(BaseModel):
    """Full extraction result from a literature snippet."""

    atom_roles: List[AtomRole] = Field(description="Roles for mentioned atoms")
    interactions: List[Interaction] = Field(description="Interaction list")
    overall_confidence: float = Field(ge=0.0, le=1.0, description="Overall confidence")
    literature_precision_level: int = Field(
        ge=1, le=5, description="Precision level 1-5"
    )
    reasoning_chain: List[str] = Field(description="Step-by-step reasoning")


def _score_to_category(score: float) -> str:
    if score >= 9:
        return "very_strong"
    if score >= 7:
        return "strong"
    if score >= 4:
        return "moderate"
    if score >= 2:
        return "weak"
    return "negligible"


def normalize_interaction_strength(score: float) -> Tuple[float, str]:
    """Return normalized score in [0,1] and its category."""
    if score < 0 or score > 10:
        raise ValueError("strength_score must be in [0, 10]")
    return score / 10.0, _score_to_category(score)
