from __future__ import annotations

import json
from typing import Dict, List, Optional

from .schema import _score_to_category


def _interaction(
    atom_a: str,
    atom_b: str,
    interaction_type: str,
    strength_score: float,
    confidence: float,
    reasoning: str,
    distance: Optional[float] = None,
) -> Dict[str, object]:
    return {
        "atoms": [atom_a, atom_b],
        "interaction_type": interaction_type,
        "strength_category": _score_to_category(strength_score),
        "strength_score": strength_score,
        "distance": distance,
        "confidence": confidence,
        "reasoning": reasoning,
        "electronic_effect": None,
        "steric_effect": None,
    }


def build_mock_extraction(seed: int) -> Dict[str, object]:
    atom_roles = [
        {"atom_id": "C:1", "role": "participant", "participation_detail": None, "confidence": 0.82},
        {"atom_id": "O:2", "role": "participant", "participation_detail": None, "confidence": 0.8},
        {"atom_id": "N:3", "role": "spectator", "participation_detail": None, "confidence": 0.6},
    ]
    interactions = [
        _interaction(
            "C:1",
            "O:2",
            "covalent",
            9.0,
            0.9,
            "Reported C=O bond.",
            distance=1.23,
        ),
        _interaction(
            "N:3",
            "C:1",
            "h_bond",
            5.0,
            0.7,
            f"Hydrogen bond inferred in example {seed}.",
            distance=2.8,
        ),
    ]
    return {
        "atom_roles": atom_roles,
        "interactions": interactions,
        "overall_confidence": 0.75,
        "literature_precision_level": 4,
        "reasoning_chain": [
            "Identify functional groups.",
            "Map atoms to interaction roles.",
            f"Aggregate evidence for sample {seed}.",
        ],
    }


MOCK_EXTRACTIONS: List[Dict[str, object]] = [build_mock_extraction(i) for i in range(10)]

MOCK_JSON_OUTPUTS: List[str] = [json.dumps(payload) for payload in MOCK_EXTRACTIONS]
