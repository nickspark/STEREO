from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence


_ATOM_ID_PATTERN = re.compile(r"^(?:(?P<symbol>[A-Za-z]{1,3})[:#\-])?(?P<map_num>\d+)$")
_REF_ID_PATTERN = re.compile(r"^ref-\d+")


def _interaction_to_dict(interaction: object) -> dict:
    if isinstance(interaction, dict):
        return dict(interaction)
    if hasattr(interaction, "model_dump"):
        return interaction.model_dump()
    if hasattr(interaction, "dict"):
        return interaction.dict()
    if hasattr(interaction, "__dict__"):
        return dict(interaction.__dict__)
    return {"interaction": interaction}


def _parse_atom_id(atom_id: object) -> Optional[int]:
    if isinstance(atom_id, int):
        return atom_id
    raw = str(atom_id).strip()
    match = _ATOM_ID_PATTERN.match(raw)
    if not match:
        return None
    return int(match.group("map_num"))


def map_interactions_to_indices(
    interactions: Optional[Sequence[object]],
    smiles: str,
    mol_cache: Optional[Dict[str, "MolecularRepresentation"]] = None,
) -> List[dict]:
    if not interactions:
        return []
    if not smiles:
        return [_interaction_to_dict(interaction) for interaction in interactions]
    if _REF_ID_PATTERN.match(smiles.strip().lower()):
        return [_interaction_to_dict(interaction) for interaction in interactions]

    mol_cache = mol_cache if mol_cache is not None else {}
    if smiles in mol_cache:
        mol_rep = mol_cache[smiles]
    else:
        from molecular_representation.core import MolecularRepresentation

        try:
            mol_rep = MolecularRepresentation.from_smiles(smiles)
        except Exception:
            # BH hub exports component ids (e.g. ref-1-1) instead of canonical SMILES.
            # Preserve raw interactions so downstream LLM feature extraction still works.
            return [_interaction_to_dict(interaction) for interaction in interactions]
        mol_cache[smiles] = mol_rep

    atom_map = mol_rep.atom_map_to_idx
    num_atoms = len(atom_map)

    mapped: List[dict] = []
    for interaction in interactions:
        data = _interaction_to_dict(interaction)
        atom_indices = data.get("atom_indices")
        if atom_indices is not None:
            if (
                isinstance(atom_indices, (list, tuple))
                and len(atom_indices) == 2
                and all(isinstance(idx, int) and 0 <= idx < num_atoms for idx in atom_indices)
            ):
                mapped.append(data)
            continue

        atoms = data.get("atoms") or []
        if len(atoms) < 2:
            continue
        map_nums = [_parse_atom_id(atom) for atom in atoms[:2]]
        if any(m is None for m in map_nums):
            continue
        if any(m not in atom_map for m in map_nums):
            continue
        data["atom_indices"] = [atom_map[m] for m in map_nums]
        mapped.append(data)
    return mapped
