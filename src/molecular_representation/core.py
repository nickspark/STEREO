from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from rdkit import Chem


def safe_mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return mol

    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:  # noqa: BLE001
        try:
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES)
        except Exception:  # noqa: BLE001
            mol.UpdatePropertyCache(strict=False)
    return mol


@dataclass
class MolecularRepresentation:
    original_smiles: str
    atom_mapped_smiles: str
    mol: Chem.Mol
    atom_map_to_idx: Dict[int, int]
    idx_to_atom_map: Dict[int, int]

    @classmethod
    def from_smiles(cls, smiles: str) -> "MolecularRepresentation":
        mol = safe_mol_from_smiles(smiles)
        if mol is None:
            raise ValueError(f"Failed to parse SMILES: {smiles}")

        atom_map_to_idx: Dict[int, int] = {}
        idx_to_atom_map: Dict[int, int] = {}
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            map_num = idx + 1
            atom.SetAtomMapNum(map_num)
            atom_map_to_idx[map_num] = idx
            idx_to_atom_map[idx] = map_num

        atom_mapped_smiles = Chem.MolToSmiles(mol)
        return cls(
            original_smiles=smiles,
            atom_mapped_smiles=atom_mapped_smiles,
            mol=mol,
            atom_map_to_idx=atom_map_to_idx,
            idx_to_atom_map=idx_to_atom_map,
        )
