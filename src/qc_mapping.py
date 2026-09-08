from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import numpy as np


_FALLBACK_METAL_Z = {
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
    57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
    72, 73, 74, 75, 76, 77, 78, 79, 80,
    89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103,
    104, 105, 106, 107, 108, 109, 110, 111, 112,
}


def _is_metal_z(z: int) -> bool:
    if z <= 0:
        return False
    try:
        from rdkit import Chem

        return bool(Chem.GetPeriodicTable().IsMetal(int(z)))
    except Exception:  # noqa: BLE001
        return int(z) in _FALLBACK_METAL_Z


def infer_metal_indices(zs: Sequence[int]) -> List[int]:
    return [idx for idx, z in enumerate(zs) if _is_metal_z(int(z))]


def _normalize_qc_values(qc_values: Optional[Iterable[float]]) -> np.ndarray:
    if qc_values is None:
        return np.zeros((0,), dtype=np.float32)
    arr = np.asarray(list(qc_values), dtype=np.float32).reshape(-1)
    return arr


def map_atomic_qc_to_nodes(
    zs: Sequence[int],
    qc_values: Optional[Iterable[float]],
    metal_indices: Optional[Sequence[int]] = None,
) -> np.ndarray:
    qc_vec = _normalize_qc_values(qc_values)
    num_nodes = len(zs)
    if qc_vec.size == 0 or num_nodes == 0:
        return np.zeros((num_nodes, 0), dtype=np.float32)
    if metal_indices is None:
        metal_indices = infer_metal_indices(zs)
    node_qc = np.zeros((num_nodes, qc_vec.size), dtype=np.float32)
    for idx in metal_indices:
        if 0 <= idx < num_nodes:
            node_qc[idx] = qc_vec
    return node_qc


def map_bond_qc_to_edges(
    edge_index: np.ndarray,
    num_nodes: int,
    qc_values: Optional[Iterable[float]],
    metal_indices: Optional[Sequence[int]] = None,
) -> np.ndarray:
    qc_vec = _normalize_qc_values(qc_values)
    if qc_vec.size == 0:
        return np.zeros((edge_index.shape[1] if edge_index is not None else 0, 0), dtype=np.float32)
    if edge_index is None or edge_index.size == 0:
        return np.zeros((0, qc_vec.size), dtype=np.float32)
    if metal_indices is None:
        metal_indices = []
    metal_set = set(int(i) for i in metal_indices if 0 <= int(i) < num_nodes)
    edge_qc = np.zeros((edge_index.shape[1], qc_vec.size), dtype=np.float32)
    if not metal_set:
        return edge_qc
    src = edge_index[0]
    dst = edge_index[1]
    for e_idx, (i, j) in enumerate(zip(src, dst)):
        if int(i) in metal_set or int(j) in metal_set:
            edge_qc[e_idx] = qc_vec
    return edge_qc
