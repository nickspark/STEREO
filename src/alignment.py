from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


_ATOM_ID_PATTERN = re.compile(r"^(?:(?P<symbol>[A-Za-z]{1,3})[:#\-])?(?P<map_num>\d+)$")


@dataclass(frozen=True)
class AlignmentRules:
    very_strong_multiplier: float = 5.0
    strong_multiplier: float = 3.0
    moderate_multiplier: float = 2.0
    weak_multiplier: float = 1.5
    negligible_multiplier: float = 1.0

    def weight_for_category(self, category: str) -> float:
        return {
            "very_strong": self.very_strong_multiplier,
            "strong": self.strong_multiplier,
            "moderate": self.moderate_multiplier,
            "weak": self.weak_multiplier,
            "negligible": self.negligible_multiplier,
        }.get(category, 1.0)


_INTERACTION_TYPES = [
    "coordination",
    "h_bond",
    "covalent",
    "pi_stacking",
    "ionic",
    "van_der_waals",
]

_STRENGTH_CATEGORIES = [
    "very_strong",
    "strong",
    "moderate",
    "weak",
    "negligible",
]


def _get_field(obj: object, name: str, default: object = None) -> object:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _parse_atom_id(atom_id: object) -> Optional[int]:
    if isinstance(atom_id, int):
        return atom_id
    raw = str(atom_id).strip()
    match = _ATOM_ID_PATTERN.match(raw)
    if not match:
        return None
    return int(match.group("map_num"))


def _parse_interaction_pair(interaction: object, num_nodes: int) -> Optional[Tuple[int, int]]:
    atom_indices = _get_field(interaction, "atom_indices")
    if atom_indices is not None:
        try:
            i, j = int(atom_indices[0]), int(atom_indices[1])
        except (TypeError, ValueError, IndexError):
            return None
    else:
        atoms = _get_field(interaction, "atoms")
        if not atoms or len(atoms) < 2:
            return None
        map_i = _parse_atom_id(atoms[0])
        map_j = _parse_atom_id(atoms[1])
        if map_i is None or map_j is None:
            return None
        i, j = map_i - 1, map_j - 1
    if i < 0 or j < 0 or i >= num_nodes or j >= num_nodes or i == j:
        return None
    return i, j


def _rule_weight(interaction: object, rules: AlignmentRules) -> float:
    category = _get_field(interaction, "strength_category")
    if category:
        base = rules.weight_for_category(str(category))
    else:
        score = _get_field(interaction, "strength_score")
        try:
            score_val = float(score)
        except (TypeError, ValueError):
            score_val = 0.0
        base = 1.0 + (score_val / 10.0) * 4.0
    confidence = _get_field(interaction, "confidence", 1.0)
    try:
        conf_val = float(confidence)
    except (TypeError, ValueError):
        conf_val = 1.0
    return base * conf_val


class InteractionFeatureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.type_to_idx = {name: idx for idx, name in enumerate(_INTERACTION_TYPES)}
        self.category_to_idx = {name: idx for idx, name in enumerate(_STRENGTH_CATEGORIES)}
        self.feature_dim = 2 + 2 + len(_INTERACTION_TYPES) + len(_STRENGTH_CATEGORIES)

    def forward(self, interactions: Sequence[object], device: torch.device) -> torch.Tensor:
        if not interactions:
            return torch.empty((0, self.feature_dim), device=device)
        feats = torch.zeros((len(interactions), self.feature_dim), device=device)
        for idx, inter in enumerate(interactions):
            score = _get_field(inter, "strength_score", 0.0)
            confidence = _get_field(inter, "confidence", 1.0)
            distance = _get_field(inter, "distance", 0.0)
            try:
                score_val = float(score) / 10.0
            except (TypeError, ValueError):
                score_val = 0.0
            try:
                conf_val = float(confidence)
            except (TypeError, ValueError):
                conf_val = 1.0
            try:
                dist_val = float(distance) / 5.0 if distance is not None else 0.0
            except (TypeError, ValueError):
                dist_val = 0.0
            has_dist = 1.0 if distance is not None else 0.0
            feats[idx, 0] = score_val
            feats[idx, 1] = conf_val
            feats[idx, 2] = dist_val
            feats[idx, 3] = has_dist
            interaction_type = _get_field(inter, "interaction_type")
            if interaction_type:
                t_idx = self.type_to_idx.get(str(interaction_type))
                if t_idx is not None:
                    feats[idx, 4 + t_idx] = 1.0
            strength_category = _get_field(inter, "strength_category")
            if strength_category:
                c_idx = self.category_to_idx.get(str(strength_category))
                if c_idx is not None:
                    feats[idx, 4 + len(_INTERACTION_TYPES) + c_idx] = 1.0
        return feats


def build_interaction_pair_features(
    interactions: Sequence[object],
    num_nodes: int,
    encoder: InteractionFeatureEncoder,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not interactions or num_nodes <= 0:
        return (
            torch.zeros((0, 2), dtype=torch.long, device=device),
            torch.zeros((0, encoder.feature_dim), device=device),
        )
    feats = encoder(interactions, device)
    pair_idx = []
    kept_feats = []
    for inter, feat in zip(interactions, feats):
        pair = _parse_interaction_pair(inter, num_nodes)
        if pair is None:
            continue
        pair_idx.append(pair)
        kept_feats.append(feat)
    if not kept_feats:
        return (
            torch.zeros((0, 2), dtype=torch.long, device=device),
            torch.zeros((0, encoder.feature_dim), device=device),
        )
    pair_index = torch.tensor(pair_idx, dtype=torch.long, device=device)
    pair_features = torch.stack(kept_feats, dim=0)
    return pair_index, pair_features


class AlignmentModule(nn.Module):
    def __init__(
        self,
        mode: str = "rule",
        hidden_dim: int = 64,
        dropout: float = 0.1,
        rules: Optional[AlignmentRules] = None,
    ):
        super().__init__()
        self.mode = mode
        self.rules = rules or AlignmentRules()
        self.encoder = InteractionFeatureEncoder()
        if mode == "learned":
            self.weight_predictor = nn.Sequential(
                nn.Linear(self.encoder.feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.weight_predictor = None

    def _learned_weights(self, interactions: Sequence[object], device: torch.device) -> torch.Tensor:
        feats = self.encoder(interactions, device)
        if feats.numel() == 0:
            return torch.empty((0,), device=device)
        raw = self.weight_predictor(feats).squeeze(-1)
        return 1.0 + 4.0 * torch.sigmoid(raw)

    def interaction_weights(self, interactions: Sequence[object], device: torch.device) -> List[float]:
        if not interactions:
            return []
        if self.mode == "learned" and self.weight_predictor is not None:
            weights = self._learned_weights(interactions, device)
            return weights.detach().cpu().tolist()
        return [_rule_weight(inter, self.rules) for inter in interactions]

    def build_edge_weights(
        self,
        interactions: Sequence[object],
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        device = edge_index.device
        edge_weights = torch.ones(edge_index.size(1), device=device)
        if not interactions:
            return edge_weights
        weights = self.interaction_weights(interactions, device)
        for inter, weight in zip(interactions, weights):
            pair = _parse_interaction_pair(inter, num_nodes)
            if pair is None:
                continue
            i, j = pair
            mask = ((edge_index[0] == i) & (edge_index[1] == j)) | (
                (edge_index[0] == j) & (edge_index[1] == i)
            )
            if mask.any():
                edge_weights[mask] *= float(weight)
        return edge_weights

    def build_interaction_edges(
        self,
        interactions: Sequence[object],
        num_nodes: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not interactions:
            return (
                torch.zeros((2, 0), dtype=torch.long, device=device),
                torch.zeros((0,), dtype=torch.float32, device=device),
            )
        weights = self.interaction_weights(interactions, device)
        edge_dict = {}
        for inter, weight in zip(interactions, weights):
            pair = _parse_interaction_pair(inter, num_nodes)
            if pair is None:
                continue
            i, j = pair
            key = (min(i, j), max(i, j))
            edge_dict[key] = max(edge_dict.get(key, 0.0), float(weight))
        if not edge_dict:
            return (
                torch.zeros((2, 0), dtype=torch.long, device=device),
                torch.zeros((0,), dtype=torch.float32, device=device),
            )
        src = []
        dst = []
        wts = []
        for (i, j), weight in edge_dict.items():
            src.extend([i, j])
            dst.extend([j, i])
            wts.extend([weight, weight])
        edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
        edge_weight = torch.tensor(wts, dtype=torch.float32, device=device)
        return edge_index, edge_weight


def build_adjacency_matrix(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    adj = torch.zeros((num_nodes, num_nodes), device=edge_weight.device)
    if edge_index.numel() == 0:
        return adj
    adj[edge_index[0], edge_index[1]] = edge_weight
    return adj


def build_hotspot_scores(
    interactions: Sequence[object],
    num_nodes: int,
    rules: Optional[AlignmentRules] = None,
    device: Optional[torch.device] = None,
    aggregation: str = "max",
) -> torch.Tensor:
    device = device if device is not None else torch.device("cpu")
    if num_nodes <= 0:
        return torch.zeros((0,), device=device)
    scores = torch.ones((num_nodes,), device=device)
    if not interactions:
        return scores
    rules = rules or AlignmentRules()
    weights = [_rule_weight(inter, rules) for inter in interactions]
    for inter, weight in zip(interactions, weights):
        pair = _parse_interaction_pair(inter, num_nodes)
        if pair is None:
            continue
        i, j = pair
        w = float(weight)
        if aggregation == "sum":
            boost = max(0.0, w - 1.0)
            scores[i] += boost
            scores[j] += boost
        elif aggregation == "max":
            if w > scores[i]:
                scores[i] = w
            if w > scores[j]:
                scores[j] = w
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
    return scores


def build_attention_bias_from_scores(
    query_scores: torch.Tensor,
    key_scores: torch.Tensor,
    scale: float = 1.0,
    mode: str = "sum",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if query_scores.numel() == 0 or key_scores.numel() == 0:
        bias = torch.zeros(
            (query_scores.numel(), key_scores.numel()),
            device=query_scores.device,
        )
        return bias, torch.zeros_like(query_scores), torch.zeros_like(key_scores)
    q_boost = torch.clamp(query_scores - 1.0, min=0.0)
    k_boost = torch.clamp(key_scores - 1.0, min=0.0)
    if mode == "key":
        bias = k_boost.unsqueeze(0).expand(query_scores.size(0), key_scores.size(0))
    elif mode == "sum":
        bias = q_boost.unsqueeze(1) + k_boost.unsqueeze(0)
    elif mode == "outer":
        bias = torch.outer(q_boost, k_boost)
    else:
        raise ValueError(f"Unknown bias mode: {mode}")
    return bias * float(scale), q_boost, k_boost


def summarize_attention_bias(
    bias: torch.Tensor,
    key_boost: torch.Tensor,
    eps: float = 1e-8,
) -> Dict[str, float]:
    if bias.numel() == 0 or key_boost.numel() == 0:
        return {"hotspot_ratio": 0.0, "hotspot_share": 0.0, "hotspot_count": 0}
    key_mask = key_boost > 0
    hotspot_count = int(key_mask.sum().item())
    if hotspot_count == 0:
        return {"hotspot_ratio": 0.0, "hotspot_share": 0.0, "hotspot_count": 0}
    attn = torch.softmax(bias, dim=-1)
    hotspot_share = float(attn[:, key_mask].sum(dim=-1).mean().item())
    cold_mask = ~key_mask
    if cold_mask.any():
        hotspot_mean = float(attn[:, key_mask].mean().item())
        cold_mean = float(attn[:, cold_mask].mean().item())
        ratio = hotspot_mean / (cold_mean + eps)
    else:
        ratio = 0.0
    return {
        "hotspot_ratio": float(ratio),
        "hotspot_share": hotspot_share,
        "hotspot_count": hotspot_count,
    }
