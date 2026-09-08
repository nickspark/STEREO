import math
from typing import Tuple, Optional, Sequence, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from alignment import (
    AlignmentModule,
    AlignmentRules,
    InteractionFeatureEncoder,
    build_hotspot_scores,
    build_interaction_pair_features,
)
from llm.features import LLM_PAIR_GROUP_ORDER, LLM_SEMANTIC_BRANCH_ORDER, LLM_SEMANTIC_GROUP_ORDER, llm_feature_columns


def _build_llm_feature_index() -> Dict[str, int]:
    return {name: idx for idx, name in enumerate(llm_feature_columns())}


_LLM_FEATURE_INDEX = _build_llm_feature_index()
_LOCAL_META_GATE_COLUMN_BY_NAME = {
    "overall_confidence": "llm_overall_confidence",
    "llm_overall_confidence": "llm_overall_confidence",
    "temperature_alignment": "llm_semantic_temperature_alignment",
    "llm_semantic_temperature_alignment": "llm_semantic_temperature_alignment",
    "hotspot_density": "llm_semantic_hotspot_density",
    "llm_semantic_hotspot_density": "llm_semantic_hotspot_density",
    "cat_focus": "llm_semantic_cat_focus",
    "llm_semantic_cat_focus": "llm_semantic_cat_focus",
    "r1_focus": "llm_semantic_r1_focus",
    "llm_semantic_r1_focus": "llm_semantic_r1_focus",
    "r2_focus": "llm_semantic_r2_focus",
    "llm_semantic_r2_focus": "llm_semantic_r2_focus",
}


def _llm_feature_column(features: torch.Tensor, name: str) -> torch.Tensor:
    if features is None or features.ndim != 2 or features.numel() == 0:
        if features is None:
            raise ValueError("features tensor is required")
        return features.new_zeros((features.size(0), 1))
    index = _LLM_FEATURE_INDEX.get(name)
    if index is None or index >= features.size(1):
        return features.new_zeros((features.size(0), 1))
    return features[:, index : index + 1]


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FeatureRanker(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class FeatureColumnRanker(nn.Module):
    def __init__(self, column_index: int):
        super().__init__()
        self.column_index = int(column_index)
        self.log_scale = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        col = features[:, self.column_index : self.column_index + 1]
        scale = torch.exp(self.log_scale)
        return col * scale + self.bias


class SchNetGatedColumnRanker(nn.Module):
    def __init__(
        self,
        column_index: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_rbf: int = 32,
        cutoff: float = 5.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.base = FeatureColumnRanker(column_index=column_index)
        self.cat_encoder = SchNetEncoder(hidden_dim, num_layers, num_rbf, cutoff, dropout=dropout)
        self.reactant_encoder = SchNetEncoder(hidden_dim, num_layers, num_rbf, cutoff, dropout=dropout)
        self.combine_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        cat_z: torch.Tensor,
        cat_pos: torch.Tensor,
        cat_batch: torch.Tensor,
        cat_edge_index: torch.Tensor,
        r1_z: torch.Tensor,
        r1_pos: torch.Tensor,
        r1_batch: torch.Tensor,
        r1_edge_index: torch.Tensor,
        r2_z: torch.Tensor,
        r2_pos: torch.Tensor,
        r2_batch: torch.Tensor,
        r2_edge_index: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        _, cat_emb = self.cat_encoder(cat_z, cat_pos, cat_batch, cat_edge_index)
        _, r1_emb = self.reactant_encoder(r1_z, r1_pos, r1_batch, r1_edge_index)
        _, r2_emb = self.reactant_encoder(r2_z, r2_pos, r2_batch, r2_edge_index)
        combined = self.combine_mlp(torch.cat([cat_emb, r1_emb, r2_emb], dim=-1))
        gate = torch.tanh(self.gate(combined))
        return self.base(features) + gate


class RBFExpansion(nn.Module):
    def __init__(self, num_rbf: int, cutoff: float):
        super().__init__()
        centers = torch.linspace(0.0, cutoff, num_rbf)
        self.register_buffer("centers", centers)
        if num_rbf > 1:
            delta = centers[1] - centers[0]
            gamma = 1.0 / (delta * delta + 1e-9)
        else:
            gamma = 1.0
        self.gamma = float(gamma)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        diff = dist.unsqueeze(-1) - self.centers
        return torch.exp(-self.gamma * diff * diff)


class GBFExpansion(nn.Module):
    def __init__(self, num_gbf: int, cutoff: float):
        super().__init__()
        centers = torch.linspace(0.0, cutoff, num_gbf)
        self.centers = nn.Parameter(centers)
        self.log_widths = nn.Parameter(torch.zeros(num_gbf))

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        widths = torch.exp(self.log_widths).clamp(min=1e-3)
        diff = (dist.unsqueeze(-1) - self.centers) / widths
        return torch.exp(-diff * diff)


class DistanceBias(nn.Module):
    def __init__(self, num_heads: int, num_rbf: int, num_gbf: int, cutoff: float):
        super().__init__()
        self.rbf = RBFExpansion(num_rbf, cutoff)
        self.gbf = GBFExpansion(num_gbf, cutoff)
        self.proj = nn.Linear(num_rbf + num_gbf, num_heads, bias=False)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([self.rbf(dist), self.gbf(dist)], dim=-1)
        bias = self.proj(feat)
        return bias.permute(2, 0, 1)


class DistanceBiasedCrossAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        num_rbf: int,
        num_gbf: int,
        cutoff: float,
        bias_scale: float = 1.0,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.bias = DistanceBias(num_heads, num_rbf, num_gbf, cutoff)
        self.bias_scale = bias_scale

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        distances: Optional[torch.Tensor] = None,
        external_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if query.numel() == 0:
            return query
        if key.numel() == 0:
            return query

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.view(-1, self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(-1, self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(-1, self.num_heads, self.head_dim).transpose(0, 1)

        logits = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if distances is not None and self.bias_scale != 0.0:
            logits = logits + self.bias_scale * self.bias(distances)
        if external_bias is not None:
            if external_bias.dim() == 2:
                logits = logits + external_bias.unsqueeze(0)
            elif external_bias.dim() == 3:
                logits = logits + external_bias
            else:
                raise ValueError("external_bias must be (Nq, Nk) or (H, Nq, Nk)")

        attn = torch.softmax(logits, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(0, 1).contiguous().view(query.size(0), self.embed_dim)
        return self.out_proj(out)


class SchNetInteractionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_rbf: int, cutoff: float):
        super().__init__()
        self.rbf = RBFExpansion(num_rbf, cutoff)
        self.filter_net = nn.Sequential(
            nn.Linear(num_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dense = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        dist: torch.Tensor,
        edge_gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return h
        src, dst = edge_index
        filt = self.filter_net(self.rbf(dist))
        msg = filt * h[src]
        if edge_gate is not None and edge_gate.numel() > 0:
            msg = msg * edge_gate
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, msg)
        return h + self.dense(agg)


class SchNetEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_rbf: int,
        cutoff: float,
        dropout: float = 0.1,
        qc_dim: int = 0,
        node_qc_dim: int = 0,
        qc_gate_dim: int = 0,
        node_qc_gate_dim: int = 0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(120, hidden_dim, padding_idx=0)
        self.qc_dim = int(qc_dim)
        self.qc_proj = None
        if self.qc_dim > 0:
            self.qc_proj = nn.Sequential(
                nn.Linear(self.qc_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        self.node_qc_dim = int(node_qc_dim)
        self.node_qc_proj = None
        if self.node_qc_dim > 0:
            self.node_qc_proj = nn.Sequential(
                nn.Linear(self.node_qc_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        self.node_qc_gate_dim = int(node_qc_gate_dim)
        self.node_qc_gate_proj = None
        if self.node_qc_gate_dim > 0:
            self.node_qc_gate_proj = nn.Sequential(
                nn.Linear(self.node_qc_gate_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
        self.qc_gate_dim = int(qc_gate_dim)
        self.qc_gate_proj = None
        if self.qc_gate_dim > 0:
            self.qc_gate_proj = nn.Sequential(
                nn.Linear(self.qc_gate_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
        self.interactions = nn.ModuleList(
            [SchNetInteractionBlock(hidden_dim, num_rbf, cutoff) for _ in range(num_layers)]
        )
        self.atomwise = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.pool_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def _build_edges(self, pos: torch.Tensor, batch: torch.Tensor, k: int = 8) -> torch.Tensor:
        device = pos.device
        edge_src = []
        edge_dst = []
        for b in batch.unique():
            idx = (batch == b).nonzero(as_tuple=False).view(-1)
            p = pos[idx]
            if p.size(0) <= 1:
                continue
            d = torch.cdist(p, p)
            knn = torch.topk(d, k=min(k + 1, d.size(1)), largest=False).indices[:, 1:]
            for i in range(p.size(0)):
                src_i = idx[i].item()
                for j in knn[i].tolist():
                    dst_i = idx[j].item()
                    edge_src.append(src_i)
                    edge_dst.append(dst_i)
        if not edge_src:
            return torch.zeros((2, 0), dtype=torch.long, device=device)
        return torch.tensor([edge_src, edge_dst], dtype=torch.long, device=device)

    def _pool(self, h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        pooled_sum = torch.zeros((num_graphs, h.size(-1)), device=h.device)
        pooled_sum.index_add_(0, batch, h)
        counts = torch.bincount(batch, minlength=num_graphs).clamp(min=1).unsqueeze(-1)
        pooled_mean = pooled_sum / counts
        pooled_max = torch.full((num_graphs, h.size(-1)), -float("inf"), device=h.device)
        for i in range(num_graphs):
            h_i = h[batch == i]
            pooled_max[i] = h_i.max(dim=0).values if h_i.numel() > 0 else 0.0
        pooled = torch.cat([pooled_mean, pooled_max], dim=-1)
        return self.pool_mlp(pooled)

    def forward(
        self,
        z: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        qc_features: Optional[torch.Tensor] = None,
        node_qc: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if edge_index is None:
            edge_index = self._build_edges(pos, batch)
        if edge_index.numel() == 0:
            dist = torch.zeros((0,), device=pos.device)
        else:
            src, dst = edge_index
            dist = torch.norm(pos[dst] - pos[src], dim=-1)

        h = self.embedding(z)
        if self.node_qc_proj is not None and node_qc is not None and node_qc.numel() > 0:
            if node_qc.size(0) != h.size(0):
                raise ValueError("node_qc must align with node count.")
            h = h + self.node_qc_proj(node_qc)
        if self.qc_proj is not None and qc_features is not None and qc_features.numel() > 0:
            qc_emb = self.qc_proj(qc_features)
            h = h + qc_emb[batch]
        edge_gate = None
        if (
            self.node_qc_gate_proj is not None
            and node_qc is not None
            and node_qc.numel() > 0
            and edge_index.numel() > 0
        ):
            node_gate = torch.sigmoid(self.node_qc_gate_proj(node_qc))
            edge_gate = node_gate[edge_index[0]]
        if (
            self.qc_gate_proj is not None
            and qc_features is not None
            and qc_features.numel() > 0
            and edge_index.numel() > 0
        ):
            gate = torch.sigmoid(self.qc_gate_proj(qc_features))
            qc_edge_gate = gate[batch[edge_index[0]]]
            edge_gate = qc_edge_gate if edge_gate is None else edge_gate * qc_edge_gate
        for block in self.interactions:
            h = block(h, edge_index, dist, edge_gate=edge_gate)
        h = self.atomwise(h)
        pooled = self._pool(h, batch)
        return h, pooled


class PaiNNLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_rbf: int, cutoff: float):
        super().__init__()
        self.rbf = RBFExpansion(num_rbf, cutoff)
        self.edge_mlp = nn.Sequential(
            nn.Linear(num_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.scalar_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.vector_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        s: torch.Tensor,
        v: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        edge_weight_mode: str = "both",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if edge_index.numel() == 0:
            return s, v
        src, dst = edge_index
        rij = pos[src] - pos[dst]
        dist = torch.norm(rij, dim=-1) + 1e-8
        direction = rij / dist.unsqueeze(-1)
        rbf = self.rbf(dist)
        edge_feat = self.edge_mlp(rbf)
        msg_in = torch.cat([s[src], s[dst], edge_feat], dim=-1)
        msg_scalar = self.scalar_mlp(msg_in)
        msg_vector = self.vector_mlp(msg_in).unsqueeze(-1) * direction.unsqueeze(1)
        if edge_weight is not None and edge_weight.numel() > 0:
            weight = edge_weight.view(-1, 1)
            if edge_weight_mode in {"both", "scalar"}:
                msg_scalar = msg_scalar * weight
            if edge_weight_mode in {"both", "vector"}:
                msg_vector = msg_vector * weight.unsqueeze(-1)
            if edge_weight_mode not in {"both", "scalar", "vector"}:
                raise ValueError(f"Unknown edge_weight_mode: {edge_weight_mode}")
        s_msg = torch.zeros_like(s)
        s_msg.index_add_(0, dst, msg_scalar)
        v_msg = torch.zeros_like(v)
        v_msg.index_add_(0, dst, msg_vector)
        s_update = self.update_mlp(torch.cat([s, s_msg], dim=-1))
        return s + s_update, v + v_msg


class PaiNNEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_rbf: int,
        cutoff: float,
        dropout: float = 0.1,
        node_qc_dim: int = 0,
        node_qc_gate_dim: int = 0,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.embedding = nn.Embedding(120, hidden_dim, padding_idx=0)
        self.node_qc_dim = int(node_qc_dim)
        self.node_qc_proj = None
        if self.node_qc_dim > 0:
            self.node_qc_proj = nn.Sequential(
                nn.Linear(self.node_qc_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        self.node_qc_gate_dim = int(node_qc_gate_dim)
        self.node_qc_gate_proj = None
        if self.node_qc_gate_dim > 0:
            self.node_qc_gate_proj = nn.Sequential(
                nn.Linear(self.node_qc_gate_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
        self.layers = nn.ModuleList(
            [PaiNNLayer(hidden_dim, num_rbf, cutoff) for _ in range(num_layers)]
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.pool_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def _init_state(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
        node_qc: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
        edge_weight_mode: str = "both",
    ) -> Dict[str, torch.Tensor]:
        s = self.embedding(z)
        if self.node_qc_proj is not None and node_qc is not None and node_qc.numel() > 0:
            if node_qc.size(0) != s.size(0):
                raise ValueError("node_qc must align with node count.")
            s = s + self.node_qc_proj(node_qc)
        v = torch.zeros((s.size(0), s.size(1), 3), device=s.device, dtype=s.dtype)
        node_edge_weight = None
        if (
            self.node_qc_gate_proj is not None
            and node_qc is not None
            and node_qc.numel() > 0
            and edge_index.numel() > 0
        ):
            src = edge_index[0]
            node_edge_weight = torch.sigmoid(self.node_qc_gate_proj(node_qc[src])).squeeze(-1)
        merged_edge_weight = edge_weight
        if node_edge_weight is not None:
            merged_edge_weight = node_edge_weight if merged_edge_weight is None else merged_edge_weight * node_edge_weight
        return {
            "s": s,
            "v": v,
            "edge_weight": merged_edge_weight,
            "edge_weight_mode": edge_weight_mode,
        }

    def _run_layer(
        self,
        state: Dict[str, torch.Tensor],
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        layer_idx: int,
    ) -> Dict[str, torch.Tensor]:
        s, v = self.layers[layer_idx](
            state["s"],
            state["v"],
            pos,
            edge_index,
            edge_weight=state.get("edge_weight"),
            edge_weight_mode=str(state.get("edge_weight_mode", "both")),
        )
        state["s"] = s
        state["v"] = v
        return state

    def _finalize_state(
        self,
        state: Dict[str, torch.Tensor],
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        v_norm = torch.norm(state["v"], dim=-1)
        node_feat = self.node_mlp(torch.cat([state["s"], v_norm], dim=-1))
        pooled = self._pool(node_feat, batch)
        return node_feat, pooled

    def _pool(self, h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        pooled_sum = torch.zeros((num_graphs, h.size(-1)), device=h.device)
        pooled_sum.index_add_(0, batch, h)
        counts = torch.bincount(batch, minlength=num_graphs).clamp(min=1).unsqueeze(-1)
        pooled_mean = pooled_sum / counts
        pooled_max = torch.full((num_graphs, h.size(-1)), -float("inf"), device=h.device)
        for i in range(num_graphs):
            h_i = h[batch == i]
            pooled_max[i] = h_i.max(dim=0).values if h_i.numel() > 0 else 0.0
        pooled = torch.cat([pooled_mean, pooled_max], dim=-1)
        return self.pool_mlp(pooled)

    def forward(
        self,
        z: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        edge_weight_mode: str = "both",
        node_qc: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        state = self._init_state(
            z,
            edge_index,
            node_qc=node_qc,
            edge_weight=edge_weight,
            edge_weight_mode=edge_weight_mode,
        )
        for layer_idx in range(self.num_layers):
            state = self._run_layer(state, pos, edge_index, layer_idx)
        return self._finalize_state(state, batch)


class PhysicsInjectedPaiNNEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_rbf: int,
        cutoff: float,
        qc_dim: int = 0,
        dropout: float = 0.1,
        node_qc_dim: int = 0,
        node_qc_gate_dim: int = 0,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.qc_dim = int(qc_dim)
        self.embedding = nn.Embedding(120, hidden_dim, padding_idx=0)
        self.node_qc_dim = int(node_qc_dim)
        self.node_qc_proj = None
        if self.node_qc_dim > 0:
            self.node_qc_proj = nn.Sequential(
                nn.Linear(self.node_qc_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        self.node_qc_gate_dim = int(node_qc_gate_dim)
        self.node_qc_gate_proj = None
        if self.node_qc_gate_dim > 0:
            self.node_qc_gate_proj = nn.Sequential(
                nn.Linear(self.node_qc_gate_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
        self.layers = nn.ModuleList(
            [PaiNNLayer(hidden_dim, num_rbf, cutoff) for _ in range(num_layers)]
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.pool_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.qc_context = None
        self.qc_scale = None
        self.qc_shift = None
        self.qc_edge_gate = None
        if self.qc_dim > 0:
            gate_hidden = max(16, hidden_dim // 2)
            self.qc_context = nn.Sequential(
                nn.Linear(self.qc_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
            )
            self.qc_scale = nn.Linear(hidden_dim, hidden_dim)
            self.qc_shift = nn.Linear(hidden_dim, hidden_dim)
            self.qc_edge_gate = nn.Sequential(
                nn.Linear(hidden_dim, gate_hidden),
                nn.SiLU(),
                nn.Linear(gate_hidden, 1),
            )

    def _init_state(
        self,
        z: torch.Tensor,
        batch: torch.Tensor,
        edge_index: torch.Tensor,
        qc_features: Optional[torch.Tensor] = None,
        node_qc: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        s = self.embedding(z)
        if self.node_qc_proj is not None and node_qc is not None and node_qc.numel() > 0:
            if node_qc.size(0) != s.size(0):
                raise ValueError("node_qc must align with node count.")
            s = s + self.node_qc_proj(node_qc)
        v = torch.zeros((s.size(0), s.size(1), 3), device=s.device, dtype=s.dtype)
        node_scale, node_shift, edge_weight = self._qc_modulation(batch, edge_index, qc_features, node_qc)
        return {
            "s": s,
            "v": v,
            "node_scale": node_scale,
            "node_shift": node_shift,
            "edge_weight": edge_weight,
        }

    def _run_layer(
        self,
        state: Dict[str, torch.Tensor],
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        layer_idx: int,
    ) -> Dict[str, torch.Tensor]:
        layer_input = state["s"]
        node_scale = state.get("node_scale")
        node_shift = state.get("node_shift")
        if node_scale is not None and node_shift is not None:
            layer_input = state["s"] * (1.0 + 0.25 * node_scale) + 0.1 * node_shift
        s_update, v = self.layers[layer_idx](
            layer_input,
            state["v"],
            pos,
            edge_index,
            edge_weight=state.get("edge_weight"),
            edge_weight_mode="both",
        )
        state["s"] = state["s"] + (s_update - layer_input)
        state["v"] = v
        return state

    def _finalize_state(
        self,
        state: Dict[str, torch.Tensor],
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        v_norm = torch.norm(state["v"], dim=-1)
        node_feat = self.node_mlp(torch.cat([state["s"], v_norm], dim=-1))
        pooled = self._pool(node_feat, batch)
        return node_feat, pooled

    def _pool(self, h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        pooled_sum = torch.zeros((num_graphs, h.size(-1)), device=h.device)
        pooled_sum.index_add_(0, batch, h)
        counts = torch.bincount(batch, minlength=num_graphs).clamp(min=1).unsqueeze(-1)
        pooled_mean = pooled_sum / counts
        pooled_max = torch.full((num_graphs, h.size(-1)), -float("inf"), device=h.device)
        for i in range(num_graphs):
            h_i = h[batch == i]
            pooled_max[i] = h_i.max(dim=0).values if h_i.numel() > 0 else 0.0
        pooled = torch.cat([pooled_mean, pooled_max], dim=-1)
        return self.pool_mlp(pooled)

    def _qc_modulation(
        self,
        batch: torch.Tensor,
        edge_index: torch.Tensor,
        qc_features: Optional[torch.Tensor],
        node_qc: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if (
            self.qc_context is None
            or self.qc_scale is None
            or self.qc_shift is None
            or self.qc_edge_gate is None
            or qc_features is None
            or qc_features.numel() == 0
        ):
            return None, None, None
        qc_context = self.qc_context(qc_features)
        node_context = qc_context[batch]
        node_scale = torch.sigmoid(self.qc_scale(node_context))
        node_shift = torch.tanh(self.qc_shift(node_context))
        edge_weight = None
        if edge_index.numel() > 0:
            src = edge_index[0]
            edge_logits = self.qc_edge_gate(node_context[src]).squeeze(-1)
            edge_weight = torch.sigmoid(edge_logits)
            if self.node_qc_gate_proj is not None and node_qc is not None and node_qc.numel() > 0:
                if node_qc.size(0) != batch.size(0):
                    raise ValueError("node_qc must align with node count.")
                node_gate = torch.sigmoid(self.node_qc_gate_proj(node_qc[src])).squeeze(-1)
                edge_weight = edge_weight * node_gate
        return node_scale, node_shift, edge_weight

    def forward(
        self,
        z: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        edge_index: torch.Tensor,
        qc_features: Optional[torch.Tensor] = None,
        node_qc: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        state = self._init_state(
            z,
            batch,
            edge_index,
            qc_features=qc_features,
            node_qc=node_qc,
        )
        for layer_idx in range(self.num_layers):
            state = self._run_layer(state, pos, edge_index, layer_idx)
        return self._finalize_state(state, batch)


class KineticInteractionLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        qc_dim: int,
        dropout: float = 0.1,
        reference_temperature: float = 298.15,
    ):
        super().__init__()
        self.qc_dim = int(qc_dim)
        self.reference_temperature = float(reference_temperature)
        self.qc_norm = nn.LayerNorm(self.qc_dim)
        self.qc_proj = nn.Sequential(
            nn.Linear(self.qc_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.coupling_proj = nn.Sequential(
            nn.Linear(self.qc_dim * 4, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.delta_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

    def _temperature_basis(self, temperature: torch.Tensor) -> torch.Tensor:
        temperature = temperature.view(-1, 1)
        temp_k = (temperature + 273.15).clamp(min=200.0)
        ref_temp = temp_k.new_tensor(self.reference_temperature)
        centered = (temp_k - ref_temp) / 50.0
        inverse_ratio = ref_temp / temp_k
        log_ratio = torch.log(temp_k / ref_temp)
        return torch.cat([centered, inverse_ratio, log_ratio, centered * inverse_ratio], dim=-1)

    def forward(
        self,
        struct_emb: torch.Tensor,
        qc_features: Optional[torch.Tensor],
        temperature: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if (
            qc_features is None
            or qc_features.numel() == 0
            or temperature is None
            or temperature.numel() == 0
        ):
            return struct_emb
        qc_features = self.qc_norm(qc_features)
        temperature_basis = self._temperature_basis(temperature)
        explicit_coupling = torch.einsum("bi,bj->bij", qc_features, temperature_basis).reshape(
            qc_features.size(0),
            -1,
        )
        qc_emb = self.qc_proj(qc_features)
        kinetic_emb = self.coupling_proj(explicit_coupling)
        delta = torch.tanh(self.delta_proj(torch.cat([qc_emb, kinetic_emb], dim=-1)))
        gate = self.gate(torch.cat([struct_emb, qc_emb, kinetic_emb], dim=-1))
        return struct_emb + gate * delta


class PhysicsInjectedPaiNNRanker(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        feat_dim: int = 0,
        qc_dim: int = 0,
        fusion_strategy: str = "gated",
        dropout: float = 0.1,
        num_rbf: int = 32,
        cutoff: float = 5.0,
        global_attention: bool = False,
        global_attention_heads: int = 4,
        global_attention_layers: int = 1,
        global_attention_dropout: float = 0.1,
    ):
        super().__init__()
        self.qc_dim = int(qc_dim)
        self.fusion_strategy = str(fusion_strategy).lower()
        if self.fusion_strategy not in {"concat", "add", "gated"}:
            raise ValueError(f"Unknown physics fusion strategy: {fusion_strategy}")
        self.kinetic_interaction_enabled = self.qc_dim > 0

        self.cat_encoder = PhysicsInjectedPaiNNEncoder(
            hidden_dim,
            num_layers,
            num_rbf,
            cutoff,
            qc_dim=self.qc_dim,
            dropout=dropout,
        )
        self.reactant_encoder = PhysicsInjectedPaiNNEncoder(
            hidden_dim,
            num_layers,
            num_rbf,
            cutoff,
            qc_dim=self.qc_dim,
            dropout=dropout,
        )
        self.global_attention = (
            GlobalTokenTransformer(
                hidden_dim=hidden_dim,
                num_heads=global_attention_heads,
                num_layers=global_attention_layers,
                dropout=global_attention_dropout,
            )
            if global_attention
            else None
        )
        self.combine_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.qc_proj = None
        self.concat_fusion = None
        self.add_proj = None
        self.gated_qc_proj = None
        self.gated_fusion = None
        if self.qc_dim > 0:
            self.qc_proj = nn.Sequential(
                nn.Linear(self.qc_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            if self.fusion_strategy == "concat":
                self.concat_fusion = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                )
            elif self.fusion_strategy == "add":
                self.add_proj = nn.Linear(hidden_dim, hidden_dim)
            elif self.fusion_strategy == "gated":
                self.gated_qc_proj = nn.Linear(hidden_dim, hidden_dim)
                self.gated_fusion = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Sigmoid(),
                )
        self.kinetic_interaction = (
            KineticInteractionLayer(hidden_dim=hidden_dim, qc_dim=self.qc_dim, dropout=dropout)
            if self.qc_dim > 0
            else None
        )

        self.feature_gate = None
        if feat_dim > 0:
            self.feature_gate = nn.Sequential(
                nn.Linear(hidden_dim, feat_dim, bias=False),
                nn.Tanh(),
            )
        head_in = hidden_dim + feat_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _fuse_qc(
        self,
        struct_emb: torch.Tensor,
        qc_features: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.qc_proj is None or qc_features is None or qc_features.numel() == 0:
            return struct_emb
        qc_emb = self.qc_proj(qc_features)
        if self.fusion_strategy == "concat":
            if self.concat_fusion is None:
                return struct_emb
            return self.concat_fusion(torch.cat([struct_emb, qc_emb], dim=-1))
        if self.fusion_strategy == "add":
            if self.add_proj is None:
                return struct_emb
            return struct_emb + self.add_proj(qc_emb)
        if self.gated_qc_proj is None or self.gated_fusion is None:
            return struct_emb
        qc_term = self.gated_qc_proj(qc_emb)
        gate = self.gated_fusion(torch.cat([struct_emb, qc_emb], dim=-1))
        return gate * struct_emb + (1.0 - gate) * qc_term

    def _predict_from_embedding(self, emb: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        if self.feature_gate is not None and features.numel() > 0:
            gate = self.feature_gate(emb)
            features = features * gate
        return self.head(torch.cat([emb, features], dim=-1))

    def _apply_kinetics(
        self,
        struct_emb: torch.Tensor,
        qc_features: Optional[torch.Tensor],
        temperature: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.kinetic_interaction is None:
            return struct_emb
        return self.kinetic_interaction(struct_emb, qc_features=qc_features, temperature=temperature)

    def forward(
        self,
        cat_z: torch.Tensor,
        cat_pos: torch.Tensor,
        cat_batch: torch.Tensor,
        cat_edge_index: torch.Tensor,
        r1_z: torch.Tensor,
        r1_pos: torch.Tensor,
        r1_batch: torch.Tensor,
        r1_edge_index: torch.Tensor,
        r2_z: torch.Tensor,
        r2_pos: torch.Tensor,
        r2_batch: torch.Tensor,
        r2_edge_index: torch.Tensor,
        features: torch.Tensor,
        temperature: Optional[torch.Tensor] = None,
        qc_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, cat_local = self.cat_encoder(cat_z, cat_pos, cat_batch, cat_edge_index, qc_features=qc_features)
        _, r1_local = self.reactant_encoder(r1_z, r1_pos, r1_batch, r1_edge_index, qc_features=qc_features)
        _, r2_local = self.reactant_encoder(r2_z, r2_pos, r2_batch, r2_edge_index, qc_features=qc_features)

        local_combined = self.combine_mlp(torch.cat([cat_local, r1_local, r2_local], dim=-1))
        local_fused = self._fuse_qc(local_combined, qc_features)
        local_fused = self._apply_kinetics(local_fused, qc_features=qc_features, temperature=temperature)
        local_pred = self._predict_from_embedding(local_fused, features)

        global_residual = torch.zeros_like(local_pred)
        if self.global_attention is not None:
            cat_global, r1_global, r2_global = self.global_attention(cat_local, r1_local, r2_local)
            global_combined = self.combine_mlp(torch.cat([cat_global, r1_global, r2_global], dim=-1))
            global_fused = self._fuse_qc(global_combined, qc_features)
            global_fused = self._apply_kinetics(global_fused, qc_features=qc_features, temperature=temperature)
            global_residual = self._predict_from_embedding(global_fused, features)

        return local_pred + global_residual


class GlobalTokenTransformer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
        num_tokens: int = 3,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads for global attention.")
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.token_position = nn.Parameter(torch.zeros(1, num_tokens, hidden_dim))
        nn.init.normal_(self.token_position, std=0.02)

    def forward(
        self,
        cat_emb: torch.Tensor,
        r1_emb: torch.Tensor,
        r2_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = torch.stack([cat_emb, r1_emb, r2_emb], dim=1)
        tokens = tokens + self.token_position[:, : tokens.size(1), :]
        tokens = self.encoder(tokens)
        return tokens[:, 0], tokens[:, 1], tokens[:, 2]


class CatalystSubstrateShellCrossAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads for catalyst-substrate cross attention.")
        self.hidden_dim = int(hidden_dim)
        self.relay_cutoff = 3.5
        self.steric_cutoff = 6.0
        self.cat_to_substrate = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.substrate_to_cat = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cat_norm = nn.LayerNorm(hidden_dim)
        self.substrate_norm = nn.LayerNorm(hidden_dim)
        self.cat_ffn_norm = nn.LayerNorm(hidden_dim)
        self.substrate_ffn_norm = nn.LayerNorm(hidden_dim)
        self.cat_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.substrate_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.cat_attention_scale = nn.Parameter(torch.tensor([2.0], dtype=torch.float32))
        self.substrate_attention_scale = nn.Parameter(torch.tensor([2.0], dtype=torch.float32))
        self.cat_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.substrate_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        if self.cat_gate[-1].bias is not None:
            nn.init.zeros_(self.cat_gate[-1].bias)
        if self.substrate_gate[-1].bias is not None:
            nn.init.zeros_(self.substrate_gate[-1].bias)
        self.last_layer_tags: List[int] = []

    def _graph_offsets(self, batch: torch.Tensor, num_graphs: int) -> List[int]:
        if batch.numel() == 0:
            return [0] * num_graphs
        counts = torch.bincount(batch, minlength=num_graphs).tolist()
        offsets = [0]
        running = 0
        for count in counts[:-1]:
            running += int(count)
            offsets.append(running)
        return offsets

    def _local_pair_tensors(
        self,
        pair_index: Optional[torch.Tensor],
        pair_features: Optional[torch.Tensor],
        pair_batch: Optional[torch.Tensor],
        graph_idx: int,
        src_offset: int,
        dst_offset: int,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if (
            pair_index is None
            or pair_features is None
            or pair_batch is None
            or pair_index.numel() == 0
            or pair_features.numel() == 0
            or pair_batch.numel() == 0
        ):
            return None, None
        graph_mask = pair_batch == graph_idx
        if not torch.any(graph_mask):
            return None, None
        local_index = pair_index[:, graph_mask].clone()
        local_index[0] = local_index[0] - int(src_offset)
        local_index[1] = local_index[1] - int(dst_offset)
        return local_index, pair_features[graph_mask]

    def _pair_bias(
        self,
        query_len: int,
        key_len: int,
        pair_index: Optional[torch.Tensor],
        pair_features: Optional[torch.Tensor],
        transpose: bool = False,
    ) -> Optional[torch.Tensor]:
        if (
            pair_index is None
            or pair_features is None
            or pair_index.numel() == 0
            or pair_features.numel() == 0
            or query_len == 0
            or key_len == 0
        ):
            return None
        bias = pair_features.new_full((query_len, key_len), -1e4)
        for edge_idx in range(pair_index.size(1)):
            src_idx = int(pair_index[0, edge_idx].item())
            dst_idx = int(pair_index[1, edge_idx].item())
            if pair_features.size(1) > 15:
                relay_bonus = (
                    0.8 * float(pair_features[edge_idx, 2].item())
                    + 1.2 * float(pair_features[edge_idx, 11].item())
                    + 0.7 * float(pair_features[edge_idx, 13].item())
                    + 0.6 * float(pair_features[edge_idx, 14].item())
                    + 1.0 * float(pair_features[edge_idx, 15].item())
                )
                steric_penalty = 1.0 * float(pair_features[edge_idx, 12].item())
                relay_bonus = 1.5 * relay_bonus - steric_penalty
            else:
                relay_bonus = 1.5 * float(pair_features[edge_idx, 2].item())
            steric_active = float(pair_features[edge_idx, 3].item())
            if steric_active <= 0.0:
                continue
            q_idx = dst_idx if transpose else src_idx
            k_idx = src_idx if transpose else dst_idx
            if q_idx < 0 or q_idx >= query_len or k_idx < 0 or k_idx >= key_len:
                continue
            bias[q_idx, k_idx] = max(float(bias[q_idx, k_idx].item()), relay_bonus)
        inactive_rows = torch.all(bias <= -1e3, dim=1)
        if torch.any(inactive_rows):
            bias[inactive_rows] = 0.0
        return bias

    def _cross_residual(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        attention: nn.MultiheadAttention,
        norm: nn.LayerNorm,
        ffn_norm: nn.LayerNorm,
        ffn: nn.Sequential,
        gate_mlp: nn.Sequential,
        scale: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if query.size(1) == 0 or key.size(1) == 0:
            return query
        attn_out, _ = attention(query, key, key, attn_mask=attn_mask, need_weights=False)
        branch_scale = F.softplus(scale)
        gate = torch.sigmoid(
            gate_mlp(torch.cat([query.mean(dim=1), key.mean(dim=1)], dim=-1))
        ).unsqueeze(1)
        fused = norm(query + branch_scale * gate * attn_out)
        return ffn_norm(fused + branch_scale * gate * ffn(fused))

    def forward(
        self,
        cat_nodes: torch.Tensor,
        cat_batch: torch.Tensor,
        r1_nodes: torch.Tensor,
        r1_batch: torch.Tensor,
        r2_nodes: torch.Tensor,
        r2_batch: torch.Tensor,
        interaction_cat_r1_index: Optional[torch.Tensor] = None,
        interaction_cat_r1_features: Optional[torch.Tensor] = None,
        interaction_cat_r1_batch: Optional[torch.Tensor] = None,
        interaction_cat_r2_index: Optional[torch.Tensor] = None,
        interaction_cat_r2_features: Optional[torch.Tensor] = None,
        interaction_cat_r2_batch: Optional[torch.Tensor] = None,
        layer_tag: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_sizes = [b for b in (cat_batch, r1_batch, r2_batch) if b.numel() > 0]
        if not batch_sizes:
            return cat_nodes, r1_nodes, r2_nodes
        num_graphs = max(int(b.max().item()) for b in batch_sizes) + 1
        if layer_tag is not None:
            self.last_layer_tags.append(int(layer_tag))
        cat_offsets = self._graph_offsets(cat_batch, num_graphs)
        r1_offsets = self._graph_offsets(r1_batch, num_graphs)
        r2_offsets = self._graph_offsets(r2_batch, num_graphs)
        cat_outputs: List[torch.Tensor] = []
        r1_outputs: List[torch.Tensor] = []
        r2_outputs: List[torch.Tensor] = []
        for graph_idx in range(num_graphs):
            cat_seq = cat_nodes[cat_batch == graph_idx]
            r1_seq = r1_nodes[r1_batch == graph_idx]
            r2_seq = r2_nodes[r2_batch == graph_idx]
            substrate_seq = torch.cat([r1_seq, r2_seq], dim=0)
            if cat_seq.numel() == 0 or substrate_seq.numel() == 0:
                cat_outputs.append(cat_seq)
                r1_outputs.append(r1_seq)
                r2_outputs.append(r2_seq)
                continue
            r1_pair_index, r1_pair_features = self._local_pair_tensors(
                interaction_cat_r1_index,
                interaction_cat_r1_features,
                interaction_cat_r1_batch,
                graph_idx,
                cat_offsets[graph_idx],
                r1_offsets[graph_idx],
            )
            r2_pair_index, r2_pair_features = self._local_pair_tensors(
                interaction_cat_r2_index,
                interaction_cat_r2_features,
                interaction_cat_r2_batch,
                graph_idx,
                cat_offsets[graph_idx],
                r2_offsets[graph_idx],
            )
            cat_to_r1_bias = self._pair_bias(cat_seq.size(0), r1_seq.size(0), r1_pair_index, r1_pair_features)
            cat_to_r2_bias = self._pair_bias(cat_seq.size(0), r2_seq.size(0), r2_pair_index, r2_pair_features)
            substrate_bias_parts: List[torch.Tensor] = []
            if cat_to_r1_bias is not None:
                substrate_bias_parts.append(cat_to_r1_bias)
            elif r1_seq.size(0) > 0:
                substrate_bias_parts.append(cat_seq.new_zeros((cat_seq.size(0), r1_seq.size(0))))
            if cat_to_r2_bias is not None:
                substrate_bias_parts.append(cat_to_r2_bias)
            elif r2_seq.size(0) > 0:
                substrate_bias_parts.append(cat_seq.new_zeros((cat_seq.size(0), r2_seq.size(0))))
            cat_attn_mask = torch.cat(substrate_bias_parts, dim=1) if substrate_bias_parts else None
            r1_to_cat_bias = self._pair_bias(r1_seq.size(0), cat_seq.size(0), r1_pair_index, r1_pair_features, transpose=True)
            r2_to_cat_bias = self._pair_bias(r2_seq.size(0), cat_seq.size(0), r2_pair_index, r2_pair_features, transpose=True)
            substrate_to_cat_mask = None
            if r1_to_cat_bias is not None or r2_to_cat_bias is not None:
                substrate_to_cat_parts: List[torch.Tensor] = []
                if r1_to_cat_bias is not None:
                    substrate_to_cat_parts.append(r1_to_cat_bias)
                elif r1_seq.size(0) > 0:
                    substrate_to_cat_parts.append(cat_seq.new_zeros((r1_seq.size(0), cat_seq.size(0))))
                if r2_to_cat_bias is not None:
                    substrate_to_cat_parts.append(r2_to_cat_bias)
                elif r2_seq.size(0) > 0:
                    substrate_to_cat_parts.append(cat_seq.new_zeros((r2_seq.size(0), cat_seq.size(0))))
                substrate_to_cat_mask = torch.cat(substrate_to_cat_parts, dim=0)
            cat_seq = cat_seq.unsqueeze(0)
            substrate_seq = substrate_seq.unsqueeze(0)
            cat_fused = self._cross_residual(
                cat_seq,
                substrate_seq,
                self.cat_to_substrate,
                self.cat_norm,
                self.cat_ffn_norm,
                self.cat_ffn,
                self.cat_gate,
                self.cat_attention_scale,
                attn_mask=cat_attn_mask,
            ).squeeze(0)
            substrate_fused = self._cross_residual(
                substrate_seq,
                cat_seq,
                self.substrate_to_cat,
                self.substrate_norm,
                self.substrate_ffn_norm,
                self.substrate_ffn,
                self.substrate_gate,
                self.substrate_attention_scale,
                attn_mask=substrate_to_cat_mask,
            ).squeeze(0)
            r1_count = r1_seq.size(0)
            cat_outputs.append(cat_fused)
            r1_outputs.append(substrate_fused[:r1_count])
            r2_outputs.append(substrate_fused[r1_count:])
        return torch.cat(cat_outputs, dim=0), torch.cat(r1_outputs, dim=0), torch.cat(r2_outputs, dim=0)


class ReactionCenterCoupling(nn.Module):
    class KineticRelayBlock(nn.Module):
        def __init__(
            self,
            hidden_dim: int,
            pair_feature_dim: int,
            coordination_dim: int,
            dropout: float = 0.1,
            reference_temperature: float = 298.15,
        ):
            super().__init__()
            self.hidden_dim = int(hidden_dim)
            self.pair_feature_dim = int(pair_feature_dim)
            self.coordination_dim = int(coordination_dim)
            self.reference_temperature = float(reference_temperature)
            self.pair_norm = nn.LayerNorm(self.pair_feature_dim)
            self.pair_proj = nn.Sequential(
                nn.Linear(self.pair_feature_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
            )
            self.temperature_proj = nn.Sequential(
                nn.Linear(4, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
            )
            self.coord_proj = nn.Sequential(
                nn.Linear(4, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
            )
            self.delta_proj = nn.Sequential(
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.gate_proj = nn.Sequential(
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            self.output_scale = nn.Parameter(torch.tensor([-0.5], dtype=torch.float32))
            self.last_stats: Optional[Dict[str, float]] = None

        def _temperature_basis(self, temperature: torch.Tensor) -> torch.Tensor:
            temperature = temperature.view(-1, 1)
            basis_dtype = temperature.dtype
            temp_k = (temperature.to(torch.float64) + 273.15).clamp(min=200.0)
            ref_temp = temp_k.new_tensor(self.reference_temperature)
            centered = (temp_k - ref_temp) / 50.0
            inverse_ratio = ref_temp / temp_k
            log_ratio = torch.log(temp_k / ref_temp)
            basis = torch.cat([centered, inverse_ratio, log_ratio, centered * inverse_ratio], dim=-1)
            return basis.to(dtype=basis_dtype)

        def _coordination_slice(
            self,
            node_qc: Optional[torch.Tensor],
            node_idx: torch.Tensor,
        ) -> torch.Tensor:
            if (
                node_qc is None
                or node_qc.numel() == 0
                or self.coordination_dim <= 0
                or node_qc.size(-1) < self.coordination_dim
            ):
                return torch.zeros((node_idx.size(0), 4), dtype=torch.float32, device=node_idx.device)
            coord = node_qc[:, -self.coordination_dim :][node_idx]
            potency = coord[:, 4:5]
            hetero_ratio = coord[:, 5:6]
            active = coord[:, 0:1]
            hetero_identity = coord[:, 1:4].sum(dim=-1, keepdim=True)
            return torch.cat([potency, hetero_ratio, active, hetero_identity], dim=-1)

        def forward(
            self,
            pair_hidden: torch.Tensor,
            pair_features: torch.Tensor,
            pair_batch: torch.Tensor,
            pair_index: torch.Tensor,
            temperature: Optional[torch.Tensor],
            cat_node_qc: Optional[torch.Tensor],
            partner_node_qc: Optional[torch.Tensor],
        ) -> torch.Tensor:
            if (
                pair_hidden.numel() == 0
                or pair_features.numel() == 0
                or pair_batch.numel() == 0
                or pair_index.numel() == 0
                or temperature is None
                or temperature.numel() == 0
            ):
                self.last_stats = None
                return pair_hidden
            cat_idx = pair_index[0]
            partner_idx = pair_index[1]
            pair_emb = self.pair_proj(self.pair_norm(pair_features))
            temp_emb = self.temperature_proj(self._temperature_basis(temperature[pair_batch]))
            cat_coord = self._coordination_slice(cat_node_qc, cat_idx)
            partner_coord = self._coordination_slice(partner_node_qc, partner_idx)
            coord_summary = torch.cat(
                [
                    0.5 * (cat_coord[:, 0:1] + partner_coord[:, 0:1]),
                    partner_coord[:, 0:1],
                    0.5 * (cat_coord[:, 1:2] + partner_coord[:, 1:2]),
                    torch.maximum(cat_coord[:, 2:3], partner_coord[:, 2:3]),
                ],
                dim=-1,
            )
            coord_emb = self.coord_proj(coord_summary)
            relay_features = torch.cat([pair_hidden, pair_emb, temp_emb, coord_emb], dim=-1)
            late_metal = pair_features[:, 15:16] if pair_features.size(-1) > 15 else pair_features.new_zeros((pair_features.size(0), 1))
            cat_flux = pair_features[:, 6:7] if pair_features.size(-1) > 6 else pair_features.new_zeros((pair_features.size(0), 1))
            partner_flux = pair_features[:, 7:8] if pair_features.size(-1) > 7 else pair_features.new_zeros((pair_features.size(0), 1))
            cat_steric = pair_features[:, 8:9] if pair_features.size(-1) > 8 else pair_features.new_zeros((pair_features.size(0), 1))
            partner_steric = pair_features[:, 9:10] if pair_features.size(-1) > 9 else pair_features.new_zeros((pair_features.size(0), 1))
            approach = pair_features[:, 14:15] if pair_features.size(-1) > 14 else pair_features.new_zeros((pair_features.size(0), 1))
            dist_norm = pair_features[:, 0:1]
            temp_centered = self._temperature_basis(temperature[pair_batch])[:, 0:1].abs()
            electronic_drive = torch.clamp(0.5 * (cat_flux + partner_flux) + 0.25 * approach + 0.35 * coord_summary[:, 0:1], 0.0, 2.0)
            steric_drag = torch.clamp(0.45 * (cat_steric + partner_steric) + 0.20 * dist_norm + 0.10 * temp_centered, 0.0, 2.0)
            relay_strength = torch.sigmoid(2.4 * (electronic_drive - steric_drag))
            gate = torch.sigmoid(self.gate_proj(relay_features))
            delta = torch.tanh(self.delta_proj(relay_features))
            scale = F.softplus(self.output_scale)
            effective_gate = scale * gate * relay_strength * late_metal
            self.last_stats = {
                "mean_gate": float(gate.mean().item()),
                "mean_relay_strength": float(relay_strength.mean().item()),
                "late_metal_pair_fraction": float(late_metal.mean().item()),
                "mean_coordination_potency": float(coord_summary[:, 0].mean().item()),
                "mean_temperature_c": float(temperature[pair_batch].mean().item()),
            }
            return pair_hidden + effective_gate * delta

    def __init__(
        self,
        hidden_dim: int,
        pair_feature_dim: int,
        dropout: float = 0.1,
        kinetic_relay: bool = False,
        coordination_dim: int = 0,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.pair_feature_dim = int(pair_feature_dim)
        self.kinetic_relay_enabled = bool(kinetic_relay)
        self.coordination_dim = int(coordination_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + pair_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.cat_update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.partner_update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cat_norm = nn.LayerNorm(hidden_dim)
        self.r1_norm = nn.LayerNorm(hidden_dim)
        self.r2_norm = nn.LayerNorm(hidden_dim)
        initial_update_logit = -2.0 if self.kinetic_relay_enabled else -3.0
        self.cat_update_logit = nn.Parameter(torch.tensor([initial_update_logit], dtype=torch.float32))
        self.partner_update_logit = nn.Parameter(torch.tensor([initial_update_logit], dtype=torch.float32))
        self.graph_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.graph_norm = nn.LayerNorm(hidden_dim)
        self.electronic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.steric_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.kinetic_relay = (
            self.KineticRelayBlock(
                hidden_dim=hidden_dim,
                pair_feature_dim=pair_feature_dim,
                coordination_dim=self.coordination_dim,
                dropout=dropout,
            )
            if self.kinetic_relay_enabled
            else None
        )
        self.last_kinetic_relay_stats: Optional[Dict[str, Dict[str, float]]] = None

    def _pair_pool_weights(self, pair_features: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if pair_features is None or pair_features.numel() == 0:
            return None
        if pair_features.size(-1) <= 15:
            return torch.ones((pair_features.size(0), 1), device=pair_features.device, dtype=pair_features.dtype)
        late_metal = pair_features[:, 15:16].clamp(0.0, 1.0)
        cat_flux = pair_features[:, 6:7].clamp(0.0, 1.0)
        partner_flux = pair_features[:, 7:8].clamp(0.0, 1.0)
        cat_steric = pair_features[:, 8:9].clamp(0.0, 1.0)
        partner_steric = pair_features[:, 9:10].clamp(0.0, 1.0)
        cat_density = pair_features[:, 12:13].clamp(0.0, 1.0)
        partner_density = pair_features[:, 13:14].clamp(0.0, 1.0)
        proximity = pair_features[:, 1:2].clamp(0.0, 1.0)
        approach = pair_features[:, 14:15].clamp(min=0.0, max=1.0)
        electronic_priority = (
            0.25 * (cat_flux + partner_flux)
            + 0.15 * (cat_density + partner_density)
            + 0.20 * proximity
            + 0.20 * approach
        )
        steric_drag = 0.10 * (cat_steric + partner_steric)
        relay_priority = torch.sigmoid(3.0 * (electronic_priority - steric_drag))
        return 1.0 + late_metal * relay_priority

    def _apply_pairs(
        self,
        cat_nodes: torch.Tensor,
        partner_nodes: torch.Tensor,
        pair_index: Optional[torch.Tensor],
        pair_features: Optional[torch.Tensor],
        pair_batch: Optional[torch.Tensor],
        cat_norm: nn.LayerNorm,
        partner_norm: nn.LayerNorm,
        temperature: Optional[torch.Tensor],
        cat_node_qc: Optional[torch.Tensor],
        partner_node_qc: Optional[torch.Tensor],
        tag: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if (
            pair_index is None
            or pair_features is None
            or pair_batch is None
            or pair_index.numel() == 0
            or pair_features.numel() == 0
        ):
            return cat_nodes, partner_nodes, None, None
        cat_idx = pair_index[0]
        partner_idx = pair_index[1]
        pair_hidden = self.edge_mlp(torch.cat([cat_nodes[cat_idx], partner_nodes[partner_idx], pair_features], dim=-1))
        if self.kinetic_relay is not None:
            pair_hidden = self.kinetic_relay(
                pair_hidden,
                pair_features,
                pair_batch,
                pair_index,
                temperature,
                cat_node_qc,
                partner_node_qc,
            )
            if self.kinetic_relay.last_stats is not None:
                if self.last_kinetic_relay_stats is None:
                    self.last_kinetic_relay_stats = {}
                self.last_kinetic_relay_stats[tag] = dict(self.kinetic_relay.last_stats)
        cat_delta = torch.zeros_like(cat_nodes)
        partner_delta = torch.zeros_like(partner_nodes)
        cat_delta.index_add_(0, cat_idx, self.cat_update(pair_hidden))
        partner_delta.index_add_(0, partner_idx, self.partner_update(pair_hidden))
        cat_scale = torch.sigmoid(self.cat_update_logit)
        partner_scale = torch.sigmoid(self.partner_update_logit)
        cat_nodes = cat_nodes + cat_scale * cat_norm(cat_delta)
        partner_nodes = partner_nodes + partner_scale * partner_norm(partner_delta)
        return cat_nodes, partner_nodes, pair_hidden, pair_batch

    def _pool_pairs(
        self,
        pair_hidden: torch.Tensor,
        pair_batch: torch.Tensor,
        num_graphs: int,
        pair_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if pair_hidden.numel() == 0:
            return pair_hidden.new_zeros((num_graphs, self.hidden_dim))
        pooled_sum = torch.zeros((num_graphs, pair_hidden.size(-1)), device=pair_hidden.device, dtype=pair_hidden.dtype)
        pooled_sum.index_add_(0, pair_batch, pair_hidden)
        counts = torch.bincount(pair_batch, minlength=num_graphs).clamp(min=1).unsqueeze(-1)
        pooled_mean = pooled_sum / counts
        pooled_weighted = pooled_mean
        pool_weights = self._pair_pool_weights(pair_features)
        if pool_weights is not None:
            weighted_hidden = pair_hidden * pool_weights
            weighted_sum = torch.zeros_like(pooled_sum)
            weighted_sum.index_add_(0, pair_batch, weighted_hidden)
            weight_norm = torch.zeros((num_graphs, 1), device=pair_hidden.device, dtype=pair_hidden.dtype)
            weight_norm.index_add_(0, pair_batch, pool_weights)
            pooled_weighted = weighted_sum / weight_norm.clamp(min=1.0)
        pooled_max = torch.full_like(pooled_mean, -float("inf"))
        for idx in range(num_graphs):
            values = pair_hidden[pair_batch == idx]
            pooled_max[idx] = values.max(dim=0).values if values.numel() > 0 else 0.0
        return self.graph_norm(self.graph_proj(torch.cat([pooled_mean, pooled_max, pooled_weighted], dim=-1)))

    def forward(
        self,
        cat_nodes: torch.Tensor,
        cat_batch: torch.Tensor,
        r1_nodes: torch.Tensor,
        r1_batch: torch.Tensor,
        r2_nodes: torch.Tensor,
        r2_batch: torch.Tensor,
        reaction_center_cat_r1_index: Optional[torch.Tensor] = None,
        reaction_center_cat_r1_features: Optional[torch.Tensor] = None,
        reaction_center_cat_r1_batch: Optional[torch.Tensor] = None,
        reaction_center_cat_r2_index: Optional[torch.Tensor] = None,
        reaction_center_cat_r2_features: Optional[torch.Tensor] = None,
        reaction_center_cat_r2_batch: Optional[torch.Tensor] = None,
        temperature: Optional[torch.Tensor] = None,
        cat_node_qc: Optional[torch.Tensor] = None,
        r1_node_qc: Optional[torch.Tensor] = None,
        r2_node_qc: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_sizes = [b for b in (cat_batch, r1_batch, r2_batch) if b.numel() > 0]
        num_graphs = max(int(b.max().item()) for b in batch_sizes) + 1 if batch_sizes else 1
        self.last_kinetic_relay_stats = None
        cat_nodes, r1_nodes, cat_r1_hidden, cat_r1_batch = self._apply_pairs(
            cat_nodes,
            r1_nodes,
            reaction_center_cat_r1_index,
            reaction_center_cat_r1_features,
            reaction_center_cat_r1_batch,
            self.cat_norm,
            self.r1_norm,
            temperature,
            cat_node_qc,
            r1_node_qc,
            "cat_r1",
        )
        cat_nodes, r2_nodes, cat_r2_hidden, cat_r2_batch = self._apply_pairs(
            cat_nodes,
            r2_nodes,
            reaction_center_cat_r2_index,
            reaction_center_cat_r2_features,
            reaction_center_cat_r2_batch,
            self.cat_norm,
            self.r2_norm,
            temperature,
            cat_node_qc,
            r2_node_qc,
            "cat_r2",
        )

        pair_hidden_parts = []
        pair_batch_parts = []
        if cat_r1_hidden is not None and cat_r1_batch is not None:
            pair_hidden_parts.append(cat_r1_hidden)
            pair_batch_parts.append(cat_r1_batch)
        if cat_r2_hidden is not None and cat_r2_batch is not None:
            pair_hidden_parts.append(cat_r2_hidden)
            pair_batch_parts.append(cat_r2_batch)
        if pair_hidden_parts:
            pair_hidden = torch.cat(pair_hidden_parts, dim=0)
            pair_batch = torch.cat(pair_batch_parts, dim=0)
            pair_feature_parts = []
            if reaction_center_cat_r1_features is not None and reaction_center_cat_r1_features.numel() > 0:
                pair_feature_parts.append(reaction_center_cat_r1_features)
            if reaction_center_cat_r2_features is not None and reaction_center_cat_r2_features.numel() > 0:
                pair_feature_parts.append(reaction_center_cat_r2_features)
            pair_features = torch.cat(pair_feature_parts, dim=0) if pair_feature_parts else None
            graph_context = self._pool_pairs(pair_hidden, pair_batch, num_graphs, pair_features=pair_features)
        else:
            graph_context = cat_nodes.new_zeros((num_graphs, self.hidden_dim))
        aux_pred = torch.cat(
            [
                torch.sigmoid(self.electronic_head(graph_context)),
                torch.sigmoid(self.steric_head(graph_context)),
            ],
            dim=-1,
        )
        return cat_nodes, r1_nodes, r2_nodes, graph_context, aux_pred


class AngularBasis(nn.Module):
    def __init__(self, num_spherical: int):
        super().__init__()
        self.num_spherical = int(num_spherical)

    def forward(self, cos_theta: torch.Tensor) -> torch.Tensor:
        cos_theta = cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        if self.num_spherical <= 1:
            return cos_theta.unsqueeze(-1)
        basis = [cos_theta]
        for power in range(2, self.num_spherical + 1):
            basis.append(cos_theta.pow(power))
        return torch.stack(basis, dim=-1)


class DimeNetLiteBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_spherical: int):
        super().__init__()
        self.angle_basis = AngularBasis(num_spherical)
        self.triplet_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + num_spherical, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        node_emb: torch.Tensor,
        edge_emb: torch.Tensor,
        edge_index: torch.Tensor,
        triplet_src: torch.Tensor,
        triplet_k: torch.Tensor,
        triplet_cos: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if edge_emb.numel() == 0:
            return node_emb, edge_emb
        if triplet_src.numel() > 0:
            angle_feat = self.angle_basis(triplet_cos)
            triplet_input = torch.cat(
                [edge_emb[triplet_src], edge_emb[triplet_k], angle_feat],
                dim=-1,
            )
            triplet_msg = self.triplet_mlp(triplet_input)
            edge_update = torch.zeros_like(edge_emb)
            edge_update.index_add_(0, triplet_src, triplet_msg)
            edge_emb = edge_emb + edge_update
        _, dst = edge_index
        node_msg = torch.zeros_like(node_emb)
        node_msg.index_add_(0, dst, edge_emb)
        node_emb = node_emb + self.node_mlp(torch.cat([node_emb, node_msg], dim=-1))
        return node_emb, edge_emb


class DimeNetLiteEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_rbf: int,
        num_spherical: int,
        cutoff: float,
        dropout: float = 0.1,
        max_triplet_neighbors: int = 4,
    ):
        super().__init__()
        self.embedding = nn.Embedding(120, hidden_dim, padding_idx=0)
        self.rbf = RBFExpansion(num_rbf, cutoff)
        self.max_triplet_neighbors = int(max_triplet_neighbors)
        self.edge_proj = nn.Sequential(
            nn.Linear(num_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList(
            [DimeNetLiteBlock(hidden_dim, num_spherical) for _ in range(num_layers)]
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.pool_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def _pool(self, h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        pooled_sum = torch.zeros((num_graphs, h.size(-1)), device=h.device)
        pooled_sum.index_add_(0, batch, h)
        counts = torch.bincount(batch, minlength=num_graphs).clamp(min=1).unsqueeze(-1)
        pooled_mean = pooled_sum / counts
        pooled_max = torch.full((num_graphs, h.size(-1)), -float("inf"), device=h.device)
        for i in range(num_graphs):
            h_i = h[batch == i]
            pooled_max[i] = h_i.max(dim=0).values if h_i.numel() > 0 else 0.0
        pooled = torch.cat([pooled_mean, pooled_max], dim=-1)
        return self.pool_mlp(pooled)

    def _build_triplets(
        self,
        edge_index: torch.Tensor,
        pos: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if edge_index.numel() == 0:
            device = pos.device
            return (
                torch.zeros((0,), dtype=torch.long, device=device),
                torch.zeros((0,), dtype=torch.long, device=device),
                torch.zeros((0,), dtype=torch.float32, device=device),
            )
        src, dst = edge_index
        edge_vec = pos[src] - pos[dst]
        edge_dist = torch.norm(edge_vec, dim=-1) + 1e-8
        num_nodes = int(pos.size(0))
        triplet_src: List[int] = []
        triplet_k: List[int] = []
        triplet_cos: List[torch.Tensor] = []
        for node in range(num_nodes):
            edges_in = (dst == node).nonzero(as_tuple=False).view(-1)
            if edges_in.numel() < 2:
                continue
            if edges_in.numel() > self.max_triplet_neighbors:
                dists = edge_dist[edges_in]
                _, idx = torch.topk(dists, k=self.max_triplet_neighbors, largest=False)
                edges_in = edges_in[idx]
            vecs = edge_vec[edges_in]
            norms = edge_dist[edges_in].unsqueeze(-1)
            normed = vecs / norms
            cos_mat = normed @ normed.T
            for i in range(edges_in.numel()):
                for k in range(edges_in.numel()):
                    if i == k:
                        continue
                    triplet_src.append(int(edges_in[i]))
                    triplet_k.append(int(edges_in[k]))
                    triplet_cos.append(cos_mat[i, k])
        if not triplet_src:
            device = pos.device
            return (
                torch.zeros((0,), dtype=torch.long, device=device),
                torch.zeros((0,), dtype=torch.long, device=device),
                torch.zeros((0,), dtype=torch.float32, device=device),
            )
        device = pos.device
        return (
            torch.tensor(triplet_src, dtype=torch.long, device=device),
            torch.tensor(triplet_k, dtype=torch.long, device=device),
            torch.stack(triplet_cos).to(device=device, dtype=pos.dtype),
        )

    def forward(
        self,
        z: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.embedding(z)
        if edge_index.numel() == 0:
            node_feat = self.node_mlp(h)
            pooled = self._pool(node_feat, batch)
            return node_feat, pooled
        src, dst = edge_index
        dist = torch.norm(pos[src] - pos[dst], dim=-1)
        edge_emb = self.edge_proj(self.rbf(dist))
        triplet_src, triplet_k, triplet_cos = self._build_triplets(edge_index, pos)
        for layer in self.layers:
            h, edge_emb = layer(h, edge_emb, edge_index, triplet_src, triplet_k, triplet_cos)
        node_feat = self.node_mlp(h)
        pooled = self._pool(node_feat, batch)
        return node_feat, pooled


class GemNetDTLiteBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_spherical: int):
        super().__init__()
        self.angle_basis = AngularBasis(num_spherical)
        self.triplet_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + num_spherical, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dir_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        node_emb: torch.Tensor,
        edge_emb: torch.Tensor,
        edge_index: torch.Tensor,
        edge_unit: torch.Tensor,
        triplet_src: torch.Tensor,
        triplet_k: torch.Tensor,
        triplet_cos: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if edge_emb.numel() == 0:
            return node_emb, edge_emb
        if triplet_src.numel() > 0:
            angle_feat = self.angle_basis(triplet_cos)
            triplet_input = torch.cat(
                [edge_emb[triplet_src], edge_emb[triplet_k], angle_feat],
                dim=-1,
            )
            triplet_msg = self.triplet_mlp(triplet_input)
            edge_update = torch.zeros_like(edge_emb)
            edge_update.index_add_(0, triplet_src, triplet_msg)
            edge_emb = edge_emb + edge_update
        src, dst = edge_index
        edge_msg = self.edge_mlp(torch.cat([edge_emb, node_emb[src]], dim=-1))
        node_msg = torch.zeros_like(node_emb)
        node_msg.index_add_(0, dst, edge_msg)
        edge_dir = self.dir_mlp(edge_emb).unsqueeze(-1) * edge_unit.unsqueeze(1)
        vec_msg = torch.zeros((node_emb.size(0), edge_dir.size(1), 3), device=edge_dir.device, dtype=edge_dir.dtype)
        vec_msg.index_add_(0, dst, edge_dir)
        vec_norm = torch.norm(vec_msg, dim=-1)
        node_emb = node_emb + self.node_mlp(torch.cat([node_emb, node_msg, vec_norm], dim=-1))
        return node_emb, edge_emb


class GemNetDTLiteEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_rbf: int,
        num_spherical: int,
        cutoff: float,
        dropout: float = 0.1,
        max_triplet_neighbors: int = 4,
    ):
        super().__init__()
        self.embedding = nn.Embedding(120, hidden_dim, padding_idx=0)
        self.rbf = RBFExpansion(num_rbf, cutoff)
        self.max_triplet_neighbors = int(max_triplet_neighbors)
        self.edge_proj = nn.Sequential(
            nn.Linear(num_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList(
            [GemNetDTLiteBlock(hidden_dim, num_spherical) for _ in range(num_layers)]
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.pool_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def _pool(self, h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        pooled_sum = torch.zeros((num_graphs, h.size(-1)), device=h.device)
        pooled_sum.index_add_(0, batch, h)
        counts = torch.bincount(batch, minlength=num_graphs).clamp(min=1).unsqueeze(-1)
        pooled_mean = pooled_sum / counts
        pooled_max = torch.full((num_graphs, h.size(-1)), -float("inf"), device=h.device)
        for i in range(num_graphs):
            h_i = h[batch == i]
            pooled_max[i] = h_i.max(dim=0).values if h_i.numel() > 0 else 0.0
        pooled = torch.cat([pooled_mean, pooled_max], dim=-1)
        return self.pool_mlp(pooled)

    def _build_triplets(
        self,
        edge_index: torch.Tensor,
        edge_vec: torch.Tensor,
        edge_dist: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if edge_index.numel() == 0:
            device = edge_vec.device
            return (
                torch.zeros((0,), dtype=torch.long, device=device),
                torch.zeros((0,), dtype=torch.long, device=device),
                torch.zeros((0,), dtype=torch.float32, device=device),
            )
        _, dst = edge_index
        num_nodes = int(dst.max().item()) + 1 if dst.numel() > 0 else 0
        triplet_src: List[int] = []
        triplet_k: List[int] = []
        triplet_cos: List[torch.Tensor] = []
        for node in range(num_nodes):
            edges_in = (dst == node).nonzero(as_tuple=False).view(-1)
            if edges_in.numel() < 2:
                continue
            if edges_in.numel() > self.max_triplet_neighbors:
                dists = edge_dist[edges_in]
                _, idx = torch.topk(dists, k=self.max_triplet_neighbors, largest=False)
                edges_in = edges_in[idx]
            vecs = edge_vec[edges_in]
            norms = edge_dist[edges_in].unsqueeze(-1)
            normed = vecs / norms
            cos_mat = normed @ normed.T
            for i in range(edges_in.numel()):
                for k in range(edges_in.numel()):
                    if i == k:
                        continue
                    triplet_src.append(int(edges_in[i]))
                    triplet_k.append(int(edges_in[k]))
                    triplet_cos.append(cos_mat[i, k])
        if not triplet_src:
            device = edge_vec.device
            return (
                torch.zeros((0,), dtype=torch.long, device=device),
                torch.zeros((0,), dtype=torch.long, device=device),
                torch.zeros((0,), dtype=torch.float32, device=device),
            )
        device = edge_vec.device
        return (
            torch.tensor(triplet_src, dtype=torch.long, device=device),
            torch.tensor(triplet_k, dtype=torch.long, device=device),
            torch.stack(triplet_cos).to(device=device, dtype=edge_vec.dtype),
        )

    def forward(
        self,
        z: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.embedding(z)
        if edge_index.numel() == 0:
            node_feat = self.node_mlp(h)
            pooled = self._pool(node_feat, batch)
            return node_feat, pooled
        src, dst = edge_index
        edge_vec = pos[src] - pos[dst]
        edge_dist = torch.norm(edge_vec, dim=-1) + 1e-8
        edge_unit = edge_vec / edge_dist.unsqueeze(-1)
        edge_emb = self.edge_proj(self.rbf(edge_dist))
        triplet_src, triplet_k, triplet_cos = self._build_triplets(edge_index, edge_vec, edge_dist)
        for layer in self.layers:
            h, edge_emb = layer(h, edge_emb, edge_index, edge_unit, triplet_src, triplet_k, triplet_cos)
        node_feat = self.node_mlp(h)
        pooled = self._pool(node_feat, batch)
        return node_feat, pooled


class EquivariantMessageLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_pos = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        h: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index
        rij = pos[dst] - pos[src]
        dist = torch.norm(rij, dim=-1, keepdim=True) + 1e-8

        m_ij = self.edge_mlp(torch.cat([h[src], h[dst], dist], dim=-1))
        if edge_weight is not None and edge_weight.numel() > 0:
            m_ij = m_ij * edge_weight.view(-1, 1)
        # aggregate messages to destination
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, m_ij)

        h = h + self.node_mlp(torch.cat([h, agg], dim=-1))

        # equivariant position update
        scalar = self.update_pos(m_ij)
        delta = scalar * (rij / dist)
        pos_update = torch.zeros_like(pos)
        pos_update.index_add_(0, dst, delta)
        pos = pos + pos_update

        return h, pos


class DualEquivariantGNN(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        heads: int = 4,
        feat_dim: int = 24,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(120, hidden_dim, padding_idx=0)
        self.layers = nn.ModuleList([EquivariantMessageLayer(hidden_dim) for _ in range(num_layers)])
        self.pool_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.combine_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _build_edges(self, pos: torch.Tensor, batch: torch.Tensor, k: int = 8) -> torch.Tensor:
        # naive kNN per-graph (CPU-friendly for small molecules)
        device = pos.device
        edge_src = []
        edge_dst = []
        for b in batch.unique():
            idx = (batch == b).nonzero(as_tuple=False).view(-1)
            p = pos[idx]
            if len(p) == 1:
                continue
            d = torch.cdist(p, p)
            knn = torch.topk(d, k=min(k + 1, d.size(1)), largest=False).indices[:, 1:]
            for i in range(p.size(0)):
                src_i = idx[i].item()
                for j in knn[i].tolist():
                    dst_i = idx[j].item()
                    edge_src.append(src_i)
                    edge_dst.append(dst_i)
        if not edge_src:
            return torch.zeros((2, 0), dtype=torch.long, device=device)
        return torch.tensor([edge_src, edge_dst], dtype=torch.long, device=device)

    def _encode_molecule(
        self,
        z: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.embedding(z)
        if edge_index is None:
            edge_index = self._build_edges(pos, batch)
        for layer in self.layers:
            h, pos = layer(h, pos, batch, edge_index, edge_weight=edge_weight)
        # mean + max pool
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        pooled_sum = torch.zeros((num_graphs, h.size(-1)), device=h.device)
        pooled_sum.index_add_(0, batch, h)
        counts = torch.bincount(batch, minlength=num_graphs).clamp(min=1).unsqueeze(-1)
        pooled_mean = pooled_sum / counts
        pooled_max = torch.full((num_graphs, h.size(-1)), -float("inf"), device=h.device)
        for i in range(num_graphs):
            h_i = h[batch == i]
            if h_i.numel() > 0:
                pooled_max[i] = h_i.max(dim=0).values
            else:
                pooled_max[i] = 0.0
        pooled = torch.cat([pooled_mean, pooled_max], dim=-1)
        return self.pool_mlp(pooled)

    def forward(
        self,
        cat_z: torch.Tensor,
        cat_pos: torch.Tensor,
        cat_batch: torch.Tensor,
        cat_edge_index: torch.Tensor,
        r1_z: torch.Tensor,
        r1_pos: torch.Tensor,
        r1_batch: torch.Tensor,
        r1_edge_index: torch.Tensor,
        r2_z: torch.Tensor,
        r2_pos: torch.Tensor,
        r2_batch: torch.Tensor,
        r2_edge_index: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        cat_emb = self._encode_molecule(cat_z, cat_pos, cat_batch, cat_edge_index)
        r1_emb = self._encode_molecule(r1_z, r1_pos, r1_batch, r1_edge_index)
        r2_emb = self._encode_molecule(r2_z, r2_pos, r2_batch, r2_edge_index)

        combined = self.combine_mlp(torch.cat([cat_emb, r1_emb, r2_emb], dim=-1))

        feat_emb = self.feat_proj(features)
        out = self.head(torch.cat([combined, feat_emb], dim=-1))
        return out


class DualSchNetCrossAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        heads: int = 4,
        feat_dim: int = 24,
        dropout: float = 0.1,
        num_rbf: int = 32,
        num_gbf: int = 32,
        cutoff: float = 5.0,
        cross_bias_scale: float = 0.0,
        qc_dim: int = 0,
    ):
        super().__init__()
        self.cat_encoder = SchNetEncoder(
            hidden_dim,
            num_layers,
            num_rbf,
            cutoff,
            dropout=dropout,
            qc_dim=qc_dim,
        )
        self.reactant_encoder = SchNetEncoder(
            hidden_dim,
            num_layers,
            num_rbf,
            cutoff,
            dropout=dropout,
            qc_dim=qc_dim,
        )
        self.cross_attention = DistanceBiasedCrossAttention(
            embed_dim=hidden_dim,
            num_heads=heads,
            dropout=dropout,
            num_rbf=num_rbf,
            num_gbf=num_gbf,
            cutoff=cutoff,
            bias_scale=cross_bias_scale,
        )
        # Enforce no cross-molecule distance bias to avoid shared-frame leakage.
        self.cross_attention.bias_scale = 0.0
        self.combine_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _cross_fuse(
        self,
        query_nodes: torch.Tensor,
        query_pos: torch.Tensor,
        query_batch: torch.Tensor,
        key_nodes: torch.Tensor,
        key_pos: torch.Tensor,
        key_batch: torch.Tensor,
        external_bias: Optional[Sequence[Optional[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        num_graphs = int(query_batch.max().item()) + 1 if query_batch.numel() > 0 else 1
        fused_nodes = []
        for g in range(num_graphs):
            q_mask = query_batch == g
            k_mask = key_batch == g
            q_nodes = query_nodes[q_mask]
            k_nodes = key_nodes[k_mask]
            if q_nodes.numel() == 0:
                continue
            if k_nodes.numel() == 0:
                fused = q_nodes
            else:
                # Never use inter-molecule distances in cross attention.
                dist = None
                ext = None
                if external_bias is not None and g < len(external_bias):
                    ext = external_bias[g]
                fused = q_nodes + self.cross_attention(q_nodes, k_nodes, k_nodes, distances=dist, external_bias=ext)
            fused_nodes.append(fused)
        if not fused_nodes:
            return query_nodes
        return torch.cat(fused_nodes, dim=0)

    def forward(
        self,
        cat_z: torch.Tensor,
        cat_pos: torch.Tensor,
        cat_batch: torch.Tensor,
        cat_edge_index: torch.Tensor,
        r1_z: torch.Tensor,
        r1_pos: torch.Tensor,
        r1_batch: torch.Tensor,
        r1_edge_index: torch.Tensor,
        r2_z: torch.Tensor,
        r2_pos: torch.Tensor,
        r2_batch: torch.Tensor,
        r2_edge_index: torch.Tensor,
        features: torch.Tensor,
        qc_features: Optional[torch.Tensor] = None,
        cross_bias: Optional[Dict[str, Sequence[Optional[torch.Tensor]]]] = None,
    ) -> torch.Tensor:
        cat_nodes, _ = self.cat_encoder(
            cat_z, cat_pos, cat_batch, cat_edge_index, qc_features=qc_features
        )
        r1_nodes, _ = self.reactant_encoder(
            r1_z, r1_pos, r1_batch, r1_edge_index, qc_features=qc_features
        )
        r2_nodes, _ = self.reactant_encoder(
            r2_z, r2_pos, r2_batch, r2_edge_index, qc_features=qc_features
        )

        bias_cat_r1 = cross_bias.get("cat_r1") if cross_bias else None
        bias_cat_r2 = cross_bias.get("cat_r2") if cross_bias else None
        bias_r1_cat = cross_bias.get("r1_cat") if cross_bias else None
        bias_r2_cat = cross_bias.get("r2_cat") if cross_bias else None

        cat_fused = self._cross_fuse(cat_nodes, cat_pos, cat_batch, r1_nodes, r1_pos, r1_batch, bias_cat_r1)
        cat_fused = self._cross_fuse(cat_fused, cat_pos, cat_batch, r2_nodes, r2_pos, r2_batch, bias_cat_r2)
        r1_fused = self._cross_fuse(r1_nodes, r1_pos, r1_batch, cat_nodes, cat_pos, cat_batch, bias_r1_cat)
        r2_fused = self._cross_fuse(r2_nodes, r2_pos, r2_batch, cat_nodes, cat_pos, cat_batch, bias_r2_cat)

        cat_emb = self.cat_encoder._pool(cat_fused, cat_batch)
        r1_emb = self.reactant_encoder._pool(r1_fused, r1_batch)
        r2_emb = self.reactant_encoder._pool(r2_fused, r2_batch)

        combined = self.combine_mlp(torch.cat([cat_emb, r1_emb, r2_emb], dim=-1))
        feat_emb = self.feat_proj(features)
        out = self.head(torch.cat([combined, feat_emb], dim=-1))
        return out


class GatedSchNetCrossAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        heads: int = 4,
        feat_dim: int = 24,
        dropout: float = 0.1,
        num_rbf: int = 32,
        num_gbf: int = 32,
        cutoff: float = 5.0,
        qc_dim: int = 0,
        gate_dropout: float = 0.1,
        qc_node_dim: Optional[int] = None,
        qc_gate_dim: Optional[int] = None,
    ):
        super().__init__()
        if qc_node_dim is None:
            qc_node_dim = qc_dim
        if qc_gate_dim is None:
            qc_gate_dim = qc_dim
        self.cat_encoder = SchNetEncoder(
            hidden_dim,
            num_layers,
            num_rbf,
            cutoff,
            dropout=dropout,
            qc_dim=qc_node_dim,
        )
        self.reactant_encoder = SchNetEncoder(
            hidden_dim,
            num_layers,
            num_rbf,
            cutoff,
            dropout=dropout,
            qc_dim=qc_node_dim,
        )
        self.cross_attention = DistanceBiasedCrossAttention(
            embed_dim=hidden_dim,
            num_heads=heads,
            dropout=dropout,
            num_rbf=num_rbf,
            num_gbf=num_gbf,
            cutoff=cutoff,
            bias_scale=0.0,
        )
        self.qc_proj = None
        gate_in_dim = hidden_dim * 2
        if qc_gate_dim and qc_gate_dim > 0:
            self.qc_proj = nn.Sequential(
                nn.Linear(qc_gate_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(gate_dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            gate_in_dim += hidden_dim
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(gate_dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.combine_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _compute_gate(
        self,
        query_emb: torch.Tensor,
        key_emb: torch.Tensor,
        qc_features: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.qc_proj is not None and qc_features is not None and qc_features.numel() > 0:
            qc_emb = self.qc_proj(qc_features)
            gate_input = torch.cat([query_emb, key_emb, qc_emb], dim=-1)
        else:
            gate_input = torch.cat([query_emb, key_emb], dim=-1)
        return torch.sigmoid(self.gate_mlp(gate_input))

    def _cross_fuse(
        self,
        query_nodes: torch.Tensor,
        query_batch: torch.Tensor,
        key_nodes: torch.Tensor,
        key_batch: torch.Tensor,
        gate_values: Optional[torch.Tensor],
        external_bias: Optional[Sequence[Optional[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        num_graphs = int(query_batch.max().item()) + 1 if query_batch.numel() > 0 else 1
        fused_nodes = []
        for g in range(num_graphs):
            q_mask = query_batch == g
            k_mask = key_batch == g
            q_nodes = query_nodes[q_mask]
            k_nodes = key_nodes[k_mask]
            if q_nodes.numel() == 0:
                continue
            if k_nodes.numel() == 0:
                fused = q_nodes
            else:
                ext = None
                if external_bias is not None and g < len(external_bias):
                    ext = external_bias[g]
                attn_out = self.cross_attention(q_nodes, k_nodes, k_nodes, distances=None, external_bias=ext)
                if gate_values is None or g >= gate_values.size(0):
                    gate = 1.0
                else:
                    gate = gate_values[g].view(1, 1)
                fused = q_nodes + gate * attn_out
            fused_nodes.append(fused)
        if not fused_nodes:
            return query_nodes
        return torch.cat(fused_nodes, dim=0)

    def forward(
        self,
        cat_z: torch.Tensor,
        cat_pos: torch.Tensor,
        cat_batch: torch.Tensor,
        cat_edge_index: torch.Tensor,
        r1_z: torch.Tensor,
        r1_pos: torch.Tensor,
        r1_batch: torch.Tensor,
        r1_edge_index: torch.Tensor,
        r2_z: torch.Tensor,
        r2_pos: torch.Tensor,
        r2_batch: torch.Tensor,
        r2_edge_index: torch.Tensor,
        features: torch.Tensor,
        qc_features: Optional[torch.Tensor] = None,
        cross_bias: Optional[Dict[str, Sequence[Optional[torch.Tensor]]]] = None,
    ) -> torch.Tensor:
        cat_nodes, cat_emb = self.cat_encoder(
            cat_z, cat_pos, cat_batch, cat_edge_index, qc_features=qc_features
        )
        r1_nodes, r1_emb = self.reactant_encoder(
            r1_z, r1_pos, r1_batch, r1_edge_index, qc_features=qc_features
        )
        r2_nodes, r2_emb = self.reactant_encoder(
            r2_z, r2_pos, r2_batch, r2_edge_index, qc_features=qc_features
        )

        gate_cat_r1 = self._compute_gate(cat_emb, r1_emb, qc_features)
        gate_cat_r2 = self._compute_gate(cat_emb, r2_emb, qc_features)
        gate_r1_cat = self._compute_gate(r1_emb, cat_emb, qc_features)
        gate_r2_cat = self._compute_gate(r2_emb, cat_emb, qc_features)

        bias_cat_r1 = cross_bias.get("cat_r1") if cross_bias else None
        bias_cat_r2 = cross_bias.get("cat_r2") if cross_bias else None
        bias_r1_cat = cross_bias.get("r1_cat") if cross_bias else None
        bias_r2_cat = cross_bias.get("r2_cat") if cross_bias else None

        cat_fused = self._cross_fuse(
            cat_nodes,
            cat_batch,
            r1_nodes,
            r1_batch,
            gate_cat_r1,
            bias_cat_r1,
        )
        cat_fused = self._cross_fuse(
            cat_fused,
            cat_batch,
            r2_nodes,
            r2_batch,
            gate_cat_r2,
            bias_cat_r2,
        )
        r1_fused = self._cross_fuse(
            r1_nodes,
            r1_batch,
            cat_nodes,
            cat_batch,
            gate_r1_cat,
            bias_r1_cat,
        )
        r2_fused = self._cross_fuse(
            r2_nodes,
            r2_batch,
            cat_nodes,
            cat_batch,
            gate_r2_cat,
            bias_r2_cat,
        )

        cat_emb = self.cat_encoder._pool(cat_fused, cat_batch)
        r1_emb = self.reactant_encoder._pool(r1_fused, r1_batch)
        r2_emb = self.reactant_encoder._pool(r2_fused, r2_batch)

        combined = self.combine_mlp(torch.cat([cat_emb, r1_emb, r2_emb], dim=-1))
        feat_emb = self.feat_proj(features)
        out = self.head(torch.cat([combined, feat_emb], dim=-1))
        return out


class SchNetBackboneRanker(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        feat_dim: int = 0,
        dropout: float = 0.1,
        num_rbf: int = 32,
        cutoff: float = 5.0,
        qc_dim: int = 0,
    ):
        super().__init__()
        self.cat_encoder = SchNetEncoder(
            hidden_dim,
            num_layers,
            num_rbf,
            cutoff,
            dropout=dropout,
            qc_dim=qc_dim,
        )
        self.reactant_encoder = SchNetEncoder(
            hidden_dim,
            num_layers,
            num_rbf,
            cutoff,
            dropout=dropout,
            qc_dim=qc_dim,
        )
        self.combine_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.feature_gate = None
        if feat_dim > 0:
            self.feature_gate = nn.Sequential(
                nn.Linear(hidden_dim, feat_dim, bias=False),
                nn.Tanh(),
            )
        head_in = hidden_dim + feat_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        cat_z: torch.Tensor,
        cat_pos: torch.Tensor,
        cat_batch: torch.Tensor,
        cat_edge_index: torch.Tensor,
        r1_z: torch.Tensor,
        r1_pos: torch.Tensor,
        r1_batch: torch.Tensor,
        r1_edge_index: torch.Tensor,
        r2_z: torch.Tensor,
        r2_pos: torch.Tensor,
        r2_batch: torch.Tensor,
        r2_edge_index: torch.Tensor,
        features: torch.Tensor,
        qc_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, cat_emb = self.cat_encoder(cat_z, cat_pos, cat_batch, cat_edge_index, qc_features=qc_features)
        _, r1_emb = self.reactant_encoder(r1_z, r1_pos, r1_batch, r1_edge_index, qc_features=qc_features)
        _, r2_emb = self.reactant_encoder(r2_z, r2_pos, r2_batch, r2_edge_index, qc_features=qc_features)

        combined = self.combine_mlp(torch.cat([cat_emb, r1_emb, r2_emb], dim=-1))
        if self.feature_gate is not None and features.numel() > 0:
            gate = self.feature_gate(combined)
            gated_features = features * gate
        else:
            gated_features = features
        head_input = torch.cat([combined, gated_features], dim=-1)
        return self.head(head_input)


class PaiNNBackboneRanker(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        feat_dim: int = 0,
        dropout: float = 0.1,
        num_rbf: int = 32,
        cutoff: float = 5.0,
        global_attention: bool = False,
        global_attention_heads: int = 4,
        global_attention_layers: int = 1,
        global_attention_dropout: float = 0.1,
    ):
        super().__init__()
        self.cat_encoder = PaiNNEncoder(hidden_dim, num_layers, num_rbf, cutoff, dropout=dropout)
        self.reactant_encoder = PaiNNEncoder(hidden_dim, num_layers, num_rbf, cutoff, dropout=dropout)
        self.global_attention = (
            GlobalTokenTransformer(
                hidden_dim=hidden_dim,
                num_heads=global_attention_heads,
                num_layers=global_attention_layers,
                dropout=global_attention_dropout,
            )
            if global_attention
            else None
        )
        self.combine_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.feature_gate = None
        if feat_dim > 0:
            self.feature_gate = nn.Sequential(
                nn.Linear(hidden_dim, feat_dim, bias=False),
                nn.Tanh(),
            )
        head_in = hidden_dim + feat_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        cat_z: torch.Tensor,
        cat_pos: torch.Tensor,
        cat_batch: torch.Tensor,
        cat_edge_index: torch.Tensor,
        r1_z: torch.Tensor,
        r1_pos: torch.Tensor,
        r1_batch: torch.Tensor,
        r1_edge_index: torch.Tensor,
        r2_z: torch.Tensor,
        r2_pos: torch.Tensor,
        r2_batch: torch.Tensor,
        r2_edge_index: torch.Tensor,
        features: torch.Tensor,
        qc_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, cat_local = self.cat_encoder(cat_z, cat_pos, cat_batch, cat_edge_index)
        _, r1_local = self.reactant_encoder(r1_z, r1_pos, r1_batch, r1_edge_index)
        _, r2_local = self.reactant_encoder(r2_z, r2_pos, r2_batch, r2_edge_index)

        local_combined = self.combine_mlp(torch.cat([cat_local, r1_local, r2_local], dim=-1))
        if self.feature_gate is not None and features.numel() > 0:
            local_gate = self.feature_gate(local_combined)
            local_features = features * local_gate
        else:
            local_features = features
        local_pred = self.head(torch.cat([local_combined, local_features], dim=-1))

        global_residual = torch.zeros_like(local_pred)
        if self.global_attention is not None:
            cat_global, r1_global, r2_global = self.global_attention(cat_local, r1_local, r2_local)
            global_combined = self.combine_mlp(torch.cat([cat_global, r1_global, r2_global], dim=-1))
            if self.feature_gate is not None and features.numel() > 0:
                global_gate = self.feature_gate(global_combined)
                global_features = features * global_gate
            else:
                global_features = features
            global_residual = self.head(torch.cat([global_combined, global_features], dim=-1))

        return local_pred + global_residual


class ReactionGraphTransformer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_tokens: int = 7,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads for reaction transformer.")
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.token_embed = nn.Parameter(torch.zeros(1, num_tokens, hidden_dim))
        nn.init.normal_(self.token_embed, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() != 3:
            raise ValueError("reaction transformer expects tokens with shape (batch, tokens, hidden).")
        if tokens.size(1) > self.token_embed.size(1):
            raise ValueError("number of tokens exceeds configured reaction transformer token embedding size.")
        tokens = tokens + self.token_embed[:, : tokens.size(1), :]
        return self.encoder(tokens)


class ReactionGraphTransformerRanker(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        feat_dim: int = 0,
        dropout: float = 0.1,
        num_rbf: int = 32,
        cutoff: float = 5.0,
        qc_dim: int = 0,
        reaction_heads: int = 4,
        reaction_layers: int = 2,
        reaction_dropout: float = 0.1,
    ):
        super().__init__()
        self.qc_dim = int(qc_dim)
        self.hidden_dim = int(hidden_dim)
        self.cat_encoder = PaiNNEncoder(hidden_dim, num_layers, num_rbf, cutoff, dropout=dropout)
        self.reactant_encoder = PaiNNEncoder(hidden_dim, num_layers, num_rbf, cutoff, dropout=dropout)
        self.pair_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.reaction_transformer = ReactionGraphTransformer(
            hidden_dim=hidden_dim,
            num_heads=reaction_heads,
            num_layers=reaction_layers,
            dropout=reaction_dropout,
            num_tokens=9,
        )
        self.site_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=reaction_heads,
            dropout=reaction_dropout,
            batch_first=True,
        )
        self.site_norm = nn.LayerNorm(hidden_dim)
        self.site_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.qc_proj = None
        self.qc_to_struct_attention = None
        self.struct_to_qc_attention = None
        self.struct_qc_gate = None
        self.qc_mix = None
        self.qc_norm = None
        if self.qc_dim > 0:
            self.qc_proj = nn.Sequential(
                nn.Linear(self.qc_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.qc_to_struct_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=reaction_heads,
                dropout=reaction_dropout,
                batch_first=True,
            )
            self.struct_to_qc_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=reaction_heads,
                dropout=reaction_dropout,
                batch_first=True,
            )
            self.struct_qc_gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid(),
            )
            self.qc_mix = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.qc_norm = nn.LayerNorm(hidden_dim)
        self.qc_null_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.qc_null_token, std=0.02)
        self.register_buffer(
            "site_metal_atomic_numbers",
            torch.tensor([21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 39, 40, 41, 42, 44, 45, 46, 47, 77, 78], dtype=torch.long),
            persistent=False,
        )
        self.combine_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.context_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.context_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.feature_gate = None
        if feat_dim > 0:
            self.feature_gate = nn.Sequential(
                nn.Linear(hidden_dim, feat_dim, bias=False),
                nn.Tanh(),
            )
        head_in = hidden_dim + feat_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _pair_token(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        return self.pair_mlp(torch.cat([lhs, rhs, lhs * rhs, torch.abs(lhs - rhs)], dim=-1))

    def _mean_or_zero(self, node_emb: torch.Tensor) -> torch.Tensor:
        if node_emb.numel() == 0:
            return torch.zeros((self.hidden_dim,), device=node_emb.device, dtype=node_emb.dtype)
        return node_emb.mean(dim=0)

    def _select_catalyst_site(
        self,
        node_emb: torch.Tensor,
        node_z: torch.Tensor,
    ) -> torch.Tensor:
        if node_emb.numel() == 0:
            return self._mean_or_zero(node_emb)
        metal_mask = (node_z.unsqueeze(-1) == self.site_metal_atomic_numbers.view(1, -1)).any(dim=1)
        if torch.any(metal_mask):
            return node_emb[metal_mask].mean(dim=0)
        return self._mean_or_zero(node_emb)

    def _select_cn_site(
        self,
        node_emb: torch.Tensor,
        node_z: torch.Tensor,
        node_pos: torch.Tensor,
    ) -> torch.Tensor:
        if node_emb.numel() == 0:
            return self._mean_or_zero(node_emb)
        n_idx = torch.nonzero(node_z == 7, as_tuple=False).view(-1)
        c_idx = torch.nonzero(node_z == 6, as_tuple=False).view(-1)
        selected = []
        if n_idx.numel() > 0:
            selected.append(node_emb[n_idx])
        if n_idx.numel() > 0 and c_idx.numel() > 0:
            n_pos = node_pos[n_idx]
            c_pos = node_pos[c_idx]
            dists = torch.cdist(n_pos, c_pos)
            nearest_c = c_idx[dists.argmin(dim=1)]
            nearest_c = torch.unique(nearest_c)
            selected.append(node_emb[nearest_c])
        elif c_idx.numel() > 0:
            selected.append(node_emb[c_idx[: min(2, int(c_idx.numel()))]])
        if selected:
            return torch.cat(selected, dim=0).mean(dim=0)
        return self._mean_or_zero(node_emb)

    def _build_site_tokens(
        self,
        cat_nodes: torch.Tensor,
        cat_z: torch.Tensor,
        cat_pos: torch.Tensor,
        cat_batch: torch.Tensor,
        r1_nodes: torch.Tensor,
        r1_z: torch.Tensor,
        r1_pos: torch.Tensor,
        r1_batch: torch.Tensor,
        r2_nodes: torch.Tensor,
        r2_z: torch.Tensor,
        r2_pos: torch.Tensor,
        r2_batch: torch.Tensor,
    ) -> torch.Tensor:
        num_graphs = int(cat_batch.max().item()) + 1 if cat_batch.numel() > 0 else 1
        site_tokens: List[torch.Tensor] = []
        for g in range(num_graphs):
            cat_mask = cat_batch == g
            r1_mask = r1_batch == g
            r2_mask = r2_batch == g
            cat_site = self._select_catalyst_site(cat_nodes[cat_mask], cat_z[cat_mask])
            r1_site = self._select_cn_site(r1_nodes[r1_mask], r1_z[r1_mask], r1_pos[r1_mask])
            r2_site = self._select_cn_site(r2_nodes[r2_mask], r2_z[r2_mask], r2_pos[r2_mask])
            site_tokens.append(torch.stack([cat_site, r1_site, r2_site], dim=0))
        return torch.stack(site_tokens, dim=0)

    def _apply_qc_cross_attention(
        self,
        struct_tokens: torch.Tensor,
        qc_features: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = struct_tokens.size(0)
        qc_token = self.qc_null_token.expand(batch_size, -1, -1)
        if (
            self.qc_proj is None
            or self.qc_to_struct_attention is None
            or self.struct_to_qc_attention is None
            or self.struct_qc_gate is None
            or self.qc_mix is None
            or self.qc_norm is None
            or qc_features is None
            or qc_features.numel() == 0
        ):
            return struct_tokens, qc_token
        qc_token = self.qc_proj(qc_features).unsqueeze(1)
        qc_context, _ = self.qc_to_struct_attention(qc_token, struct_tokens, struct_tokens)
        struct_context, _ = self.struct_to_qc_attention(struct_tokens, qc_token, qc_token)
        struct_gate = self.struct_qc_gate(torch.cat([struct_tokens, struct_context], dim=-1))
        struct_tokens = struct_tokens + struct_gate * struct_context
        qc_residual = self.qc_mix(torch.cat([qc_token, qc_context], dim=-1))
        qc_token = self.qc_norm(qc_token + qc_residual)
        return struct_tokens, qc_token

    def forward(
        self,
        cat_z: torch.Tensor,
        cat_pos: torch.Tensor,
        cat_batch: torch.Tensor,
        cat_edge_index: torch.Tensor,
        r1_z: torch.Tensor,
        r1_pos: torch.Tensor,
        r1_batch: torch.Tensor,
        r1_edge_index: torch.Tensor,
        r2_z: torch.Tensor,
        r2_pos: torch.Tensor,
        r2_batch: torch.Tensor,
        r2_edge_index: torch.Tensor,
        features: torch.Tensor,
        qc_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cat_nodes, cat_emb = self.cat_encoder(cat_z, cat_pos, cat_batch, cat_edge_index)
        r1_nodes, r1_emb = self.reactant_encoder(r1_z, r1_pos, r1_batch, r1_edge_index)
        r2_nodes, r2_emb = self.reactant_encoder(r2_z, r2_pos, r2_batch, r2_edge_index)

        cat_r1 = self._pair_token(cat_emb, r1_emb)
        cat_r2 = self._pair_token(cat_emb, r2_emb)
        r1_r2 = self._pair_token(r1_emb, r2_emb)

        cls_token = ((cat_emb + r1_emb + r2_emb) / 3.0).unsqueeze(1)
        struct_tokens = torch.stack([cat_emb, r1_emb, r2_emb, cat_r1, cat_r2, r1_r2], dim=1)
        struct_tokens, qc_token = self._apply_qc_cross_attention(struct_tokens, qc_features)

        site_tokens = self._build_site_tokens(
            cat_nodes,
            cat_z,
            cat_pos,
            cat_batch,
            r1_nodes,
            r1_z,
            r1_pos,
            r1_batch,
            r2_nodes,
            r2_z,
            r2_pos,
            r2_batch,
        )
        site_context, _ = self.site_attention(site_tokens, struct_tokens, struct_tokens)
        site_context = self.site_norm(site_tokens + site_context)
        site_token = site_context.mean(dim=1, keepdim=True)
        site_scale = torch.sigmoid(
            self.site_gate(torch.cat([site_token.squeeze(1), cls_token.squeeze(1)], dim=-1))
        ).unsqueeze(-1)
        site_token = site_token * site_scale

        fused_tokens = self.reaction_transformer(torch.cat([cls_token, struct_tokens, site_token, qc_token], dim=1))

        cls_out = fused_tokens[:, 0]
        node_out = fused_tokens[:, 1:4].mean(dim=1)
        pair_out = fused_tokens[:, 4:7].mean(dim=1)
        site_out = fused_tokens[:, 7]
        qc_out = fused_tokens[:, 8]
        reaction_core = self.combine_mlp(torch.cat([cls_out, node_out, pair_out], dim=-1))
        context_out = self.context_mlp(torch.cat([site_out, qc_out], dim=-1))
        context_gate = torch.sigmoid(self.context_gate(torch.cat([reaction_core, context_out], dim=-1)))
        reaction_emb = reaction_core + context_gate * context_out

        if self.feature_gate is not None and features.numel() > 0:
            gate = self.feature_gate(reaction_emb)
            features = features * gate
        return self.head(torch.cat([reaction_emb, features], dim=-1))


class GlobalKnowledgeNode(nn.Module):
    RULE_NAMES = (
        "ligand_temperature_sensitivity",
        "steric_temperature_window",
        "electronic_matching",
        "catalyst_shell_dominance",
        "arene_hotspot_bias",
        "boron_relay_dependence",
        "precision_weighted_consensus",
    )

    def __init__(self, llm_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.llm_dim = int(llm_dim)
        self.hidden_dim = int(hidden_dim)
        self.rule_dim = len(self.RULE_NAMES)
        self.rule_encoder = nn.Sequential(
            nn.Linear(self.rule_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.background_proj = nn.Sequential(
            nn.Linear(self.llm_dim + self.rule_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.rule_gate = nn.Sequential(
            nn.Linear(self.rule_dim, hidden_dim),
            nn.Sigmoid(),
        )

    @classmethod
    def derive_rule_scores(cls, llm_features: torch.Tensor) -> torch.Tensor:
        if llm_features is None or llm_features.ndim != 2 or llm_features.numel() == 0:
            raise ValueError("llm_features must be a 2D tensor with at least one feature column")

        steric_pressure = _llm_feature_column(llm_features, "llm_semantic_steric_pressure")
        electronic_flux = _llm_feature_column(llm_features, "llm_semantic_electronic_flux")
        temperature_alignment = _llm_feature_column(llm_features, "llm_semantic_temperature_alignment")
        cat_focus = _llm_feature_column(llm_features, "llm_semantic_cat_focus")
        r1_focus = _llm_feature_column(llm_features, "llm_semantic_r1_focus")
        r2_focus = _llm_feature_column(llm_features, "llm_semantic_r2_focus")
        hotspot_density = _llm_feature_column(llm_features, "llm_semantic_hotspot_density")
        breakdown_steric = _llm_feature_column(llm_features, "llm_breakdown_steric")
        breakdown_electronic = _llm_feature_column(llm_features, "llm_breakdown_electronic")
        breakdown_weak = _llm_feature_column(llm_features, "llm_breakdown_weak_interaction")
        breakdown_conform = _llm_feature_column(llm_features, "llm_breakdown_conformational")
        breakdown_temperature = _llm_feature_column(llm_features, "llm_breakdown_temperature")

        coordination_ratio = _llm_feature_column(llm_features, "llm_interaction_coordination_ratio")
        pi_ratio = _llm_feature_column(llm_features, "llm_interaction_pi_stacking_ratio")
        donating_ratio = _llm_feature_column(llm_features, "llm_electronic_electron_donating_ratio")
        hindered_ratio = _llm_feature_column(llm_features, "llm_steric_hindered_ratio")
        cat_ratio = _llm_feature_column(llm_features, "llm_component_cat_ratio")
        r2_ratio = _llm_feature_column(llm_features, "llm_component_r2_ratio")
        text_directing = _llm_feature_column(llm_features, "llm_text_directing_preorganization")
        text_steric = _llm_feature_column(llm_features, "llm_text_steric_clash")
        text_boron = _llm_feature_column(llm_features, "llm_text_boron_boryl")
        text_oxidative = _llm_feature_column(llm_features, "llm_text_oxidative_orbital")
        text_ts = _llm_feature_column(llm_features, "llm_text_transition_state")
        text_conformation = _llm_feature_column(llm_features, "llm_text_conformation_geometry")
        text_outer_sphere = _llm_feature_column(llm_features, "llm_text_outer_sphere_weak")
        text_octahedral = _llm_feature_column(llm_features, "llm_text_octahedral_pocket")
        text_trans_cis = _llm_feature_column(llm_features, "llm_text_trans_cis_coordination")
        text_temperature = _llm_feature_column(llm_features, "llm_text_temperature_window")

        overall_confidence = _llm_feature_column(llm_features, "llm_overall_confidence")
        mean_confidence = _llm_feature_column(llm_features, "llm_mean_confidence")
        precision_level = _llm_feature_column(llm_features, "llm_precision_level")
        mean_strength = _llm_feature_column(llm_features, "llm_mean_strength")
        mean_strength_scaled = torch.sigmoid((mean_strength - 5.0) / 1.5)

        rules = torch.cat(
            [
                0.26 * temperature_alignment + 0.20 * cat_focus + 0.15 * coordination_ratio + 0.14 * breakdown_temperature + 0.13 * text_temperature + 0.12 * text_ts,
                0.24 * steric_pressure + 0.18 * hindered_ratio + 0.16 * text_steric + 0.14 * breakdown_steric + 0.14 * text_conformation + 0.14 * text_directing,
                0.28 * electronic_flux + 0.18 * donating_ratio + 0.18 * breakdown_electronic + 0.18 * text_oxidative + 0.18 * pi_ratio,
                0.22 * cat_focus + 0.18 * coordination_ratio + 0.16 * cat_ratio + 0.16 * text_directing + 0.14 * text_octahedral + 0.14 * text_trans_cis,
                0.22 * r1_focus + 0.18 * pi_ratio + 0.16 * hotspot_density + 0.16 * text_steric + 0.14 * text_octahedral + 0.14 * text_outer_sphere,
                0.20 * r2_focus + 0.15 * r2_ratio + 0.17 * mean_strength_scaled + 0.18 * text_boron + 0.15 * breakdown_weak + 0.15 * text_outer_sphere,
                0.22 * overall_confidence + 0.16 * mean_confidence + 0.14 * precision_level + 0.14 * breakdown_conform + 0.14 * text_ts + 0.10 * text_directing + 0.10 * text_oxidative,
            ],
            dim=-1,
        )
        return rules.clamp(0.0, 1.0)

    def forward(self, llm_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        rule_scores = self.derive_rule_scores(llm_features)
        rule_context = self.rule_encoder(rule_scores)
        background_input = torch.cat([llm_features, rule_scores, rule_context], dim=-1)
        background = self.background_proj(background_input)
        gate = self.rule_gate(rule_scores)
        return self.output_norm(rule_context + gate * background), rule_scores


class NativePairSemanticEncoder(nn.Module):
    """Consume story-61 native pair objects directly via summary and joint-interaction tokens."""

    SUMMARY_TOKEN_BIAS = 0.35

    def __init__(
        self,
        hidden_dim: int,
        summary_dim: int,
        token_dim: int,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.summary_dim = int(summary_dim)
        self.token_dim = int(token_dim)
        if self.summary_dim <= 0:
            raise ValueError("summary_dim must be positive for native-pair encoding.")
        if self.token_dim <= 0:
            raise ValueError("token_dim must be positive for native-pair encoding.")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads}).")
        self.summary_projection = nn.Sequential(
            nn.Linear(self.summary_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.token_projection = nn.Sequential(
            nn.Linear(self.token_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.token_type_embedding = nn.Embedding(2, hidden_dim)
        self.missing_interaction_token = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.normal_(self.missing_interaction_token, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(1, int(num_layers)))
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.pool_gate = nn.Linear(hidden_dim, 1)
        self.last_token_mask: Optional[torch.Tensor] = None
        self.last_token_weights: Optional[torch.Tensor] = None
        self.last_pooled: Optional[torch.Tensor] = None

    def forward(
        self,
        pair_summary: torch.Tensor,
        pair_tokens: torch.Tensor,
        pair_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if pair_summary is None or pair_summary.ndim != 2 or pair_summary.numel() == 0:
            raise ValueError("pair_summary must be a non-empty 2D tensor for native-pair encoding.")
        if pair_tokens is None or pair_tokens.ndim != 3 or pair_tokens.numel() == 0:
            raise ValueError("pair_tokens must be a non-empty 3D tensor for native-pair encoding.")
        if pair_tokens.size(0) != pair_summary.size(0):
            raise ValueError("pair_summary and pair_tokens must share the same batch size.")
        if pair_summary.size(-1) != self.summary_dim:
            raise ValueError(
                f"pair_summary last dimension ({pair_summary.size(-1)}) does not match configured summary_dim ({self.summary_dim})."
            )
        if pair_tokens.size(-1) != self.token_dim:
            raise ValueError(
                f"pair_tokens last dimension ({pair_tokens.size(-1)}) does not match configured token_dim ({self.token_dim})."
            )

        batch_size = pair_summary.size(0)
        if pair_token_mask is None:
            token_mask = pair_tokens.abs().sum(dim=-1) > 1e-8
        else:
            token_mask = pair_token_mask > 0
        summary_token = self.summary_projection(pair_summary) + self.token_type_embedding.weight[0].unsqueeze(0)
        interaction_tokens = self.token_projection(pair_tokens) + self.token_type_embedding.weight[1].view(1, 1, -1)
        missing_token = self.missing_interaction_token.view(1, 1, -1).expand(batch_size, pair_tokens.size(1), -1)
        interaction_tokens = torch.where(token_mask.unsqueeze(-1), interaction_tokens, missing_token)
        token_tensor = torch.cat([summary_token.unsqueeze(1), interaction_tokens], dim=1)
        full_mask = torch.cat(
            [torch.ones((batch_size, 1), dtype=torch.bool, device=pair_summary.device), token_mask],
            dim=1,
        )
        encoded = self.encoder(self.token_norm(token_tensor), src_key_padding_mask=~full_mask)
        logits = self.pool_gate(encoded).squeeze(-1)
        logits[:, 0] = logits[:, 0] + self.SUMMARY_TOKEN_BIAS
        logits = logits.masked_fill(~full_mask, -1e9)
        weights = torch.softmax(logits, dim=-1)
        pooled = torch.sum(encoded * weights.unsqueeze(-1), dim=1)
        pooled = self.output_norm(pooled)
        self.last_token_mask = full_mask.detach()
        self.last_token_weights = weights.detach()
        self.last_pooled = pooled
        return pooled


class NativePairSemanticEncoderV2(nn.Module):
    """Summary-and-token primary native-pair encoder with explicit branch authority."""

    BRANCH_NAMES = ("summary", "token_attended", "token_max")
    BRANCH_PRIORS = {
        "summary": 0.65,
        "token_attended": 0.95,
        "token_max": 0.30,
    }
    SUMMARY_WEIGHT_FLOOR = 0.26
    SUMMARY_WEIGHT_RANGE = 0.10
    TOKEN_ATTENDED_BASE = 0.34
    TOKEN_ATTENDED_RANGE = 0.18
    TOKEN_ATTENDED_CENTER = 0.60
    TOKEN_ATTENDED_SHARPNESS = 9.0
    TOKEN_MAX_BASE = 0.12
    TOKEN_MAX_RANGE = 0.08

    def __init__(
        self,
        hidden_dim: int,
        summary_dim: int,
        token_dim: int,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.summary_dim = int(summary_dim)
        self.token_dim = int(token_dim)
        if self.summary_dim <= 0:
            raise ValueError("summary_dim must be positive for native_pair_v2 encoding.")
        if self.token_dim <= 0:
            raise ValueError("token_dim must be positive for native_pair_v2 encoding.")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads}).")
        self.summary_projection = nn.Sequential(
            nn.Linear(self.summary_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.token_projection = nn.Sequential(
            nn.Linear(self.token_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.branch_type_embedding = nn.Embedding(len(self.BRANCH_NAMES), hidden_dim)
        self.missing_token = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.normal_(self.missing_token, std=0.02)
        self.summary_to_token = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(1, int(num_layers)))
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.pool_gate = nn.Linear(hidden_dim, 1)
        branch_prior = [float(self.BRANCH_PRIORS[name]) for name in self.BRANCH_NAMES]
        self.register_buffer("branch_prior_bias", torch.tensor(branch_prior, dtype=torch.float32), persistent=False)
        self.last_token_mask: Optional[torch.Tensor] = None
        self.last_attention_weights: Optional[torch.Tensor] = None
        self.last_branch_weights: Optional[torch.Tensor] = None
        self.last_pair_evidence: Optional[torch.Tensor] = None
        self.last_pooled: Optional[torch.Tensor] = None

    def _pair_evidence(
        self,
        pair_summary: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        confidence = pair_summary[:, 0:1].clamp(0.0, 1.0) if pair_summary.size(-1) > 0 else pair_summary.new_zeros((pair_summary.size(0), 1))
        focus = pair_summary[:, 1:2].clamp(0.0, 1.0) if pair_summary.size(-1) > 1 else pair_summary.new_zeros((pair_summary.size(0), 1))
        reaction_center = pair_summary[:, 2:3].clamp(0.0, 1.0) if pair_summary.size(-1) > 2 else pair_summary.new_zeros((pair_summary.size(0), 1))
        steric = pair_summary[:, 3:4].clamp(0.0, 1.0) if pair_summary.size(-1) > 3 else pair_summary.new_zeros((pair_summary.size(0), 1))
        electronic = pair_summary[:, 4:5].clamp(0.0, 1.0) if pair_summary.size(-1) > 4 else pair_summary.new_zeros((pair_summary.size(0), 1))
        token_density = token_mask.float().mean(dim=-1, keepdim=True).to(dtype=pair_summary.dtype)
        return (
            0.24 * confidence
            + 0.22 * focus
            + 0.22 * reaction_center
            + 0.14 * steric
            + 0.10 * electronic
            + 0.08 * token_density
        )

    def _enforce_primary_branches(
        self,
        weights: torch.Tensor,
        pair_evidence: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        token_present = (token_mask.sum(dim=-1, keepdim=True) > 0).to(dtype=weights.dtype)
        priority = torch.sigmoid((pair_evidence - self.TOKEN_ATTENDED_CENTER) * self.TOKEN_ATTENDED_SHARPNESS)
        summary_floor = self.SUMMARY_WEIGHT_FLOOR + self.SUMMARY_WEIGHT_RANGE * pair_evidence.clamp(0.0, 1.0)
        attended_floor = token_present * (self.TOKEN_ATTENDED_BASE + self.TOKEN_ATTENDED_RANGE * priority)
        token_max_floor = token_present * (self.TOKEN_MAX_BASE + self.TOKEN_MAX_RANGE * priority)

        adjusted = weights.clone()
        floors = torch.cat([summary_floor, attended_floor, token_max_floor], dim=-1)
        adjusted = torch.maximum(adjusted, floors)
        adjusted = adjusted / adjusted.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return adjusted

    def forward(
        self,
        pair_summary: torch.Tensor,
        pair_tokens: torch.Tensor,
        pair_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if pair_summary is None or pair_summary.ndim != 2 or pair_summary.numel() == 0:
            raise ValueError("pair_summary must be a non-empty 2D tensor for native_pair_v2 encoding.")
        if pair_tokens is None or pair_tokens.ndim != 3 or pair_tokens.numel() == 0:
            raise ValueError("pair_tokens must be a non-empty 3D tensor for native_pair_v2 encoding.")
        if pair_tokens.size(0) != pair_summary.size(0):
            raise ValueError("pair_summary and pair_tokens must share the same batch size.")
        if pair_summary.size(-1) != self.summary_dim:
            raise ValueError(
                f"pair_summary last dimension ({pair_summary.size(-1)}) does not match configured summary_dim ({self.summary_dim})."
            )
        if pair_tokens.size(-1) != self.token_dim:
            raise ValueError(
                f"pair_tokens last dimension ({pair_tokens.size(-1)}) does not match configured token_dim ({self.token_dim})."
            )

        batch_size = pair_summary.size(0)
        token_mask = pair_tokens.abs().sum(dim=-1) > 1e-8 if pair_token_mask is None else pair_token_mask > 0
        token_mask = token_mask.clone()
        fallback_rows = token_mask.sum(dim=-1) == 0
        if fallback_rows.any():
            token_mask[fallback_rows, 0] = True

        summary_context = self.summary_projection(pair_summary)
        token_context = self.token_projection(pair_tokens)
        missing_token = self.missing_token.view(1, 1, -1).expand(batch_size, pair_tokens.size(1), -1)
        token_context = torch.where(token_mask.unsqueeze(-1), token_context, missing_token)

        attended_context, attention_weights = self.summary_to_token(
            summary_context.unsqueeze(1),
            token_context,
            token_context,
            key_padding_mask=~token_mask,
            need_weights=True,
        )
        attended_context = attended_context.squeeze(1)

        token_max = token_context.masked_fill(~token_mask.unsqueeze(-1), float("-inf")).max(dim=1).values
        token_max = torch.where(torch.isfinite(token_max), token_max, torch.zeros_like(token_max))

        type_ids = torch.arange(len(self.BRANCH_NAMES), device=pair_summary.device)
        type_embeddings = self.branch_type_embedding(type_ids)
        branch_tokens = torch.stack(
            [
                summary_context + type_embeddings[0].unsqueeze(0),
                attended_context + type_embeddings[1].unsqueeze(0),
                token_max + type_embeddings[2].unsqueeze(0),
            ],
            dim=1,
        )
        encoded = self.encoder(self.token_norm(branch_tokens))
        logits = self.pool_gate(encoded).squeeze(-1) + self.branch_prior_bias.to(device=encoded.device, dtype=encoded.dtype)
        weights = torch.softmax(logits, dim=-1)
        pair_evidence = self._pair_evidence(pair_summary, token_mask)
        adjusted_weights = self._enforce_primary_branches(weights, pair_evidence, token_mask)
        pooled = torch.sum(encoded * adjusted_weights.unsqueeze(-1), dim=1)
        pooled = self.output_norm(pooled)
        self.last_token_mask = token_mask.detach()
        self.last_attention_weights = attention_weights.detach()
        self.last_branch_weights = adjusted_weights.detach()
        self.last_pair_evidence = pair_evidence.detach()
        self.last_pooled = pooled
        return pooled


class SemanticExpertMixtureEncoder(nn.Module):
    """Learned router over per-row native-pair semantic experts."""

    def __init__(
        self,
        hidden_dim: int,
        summary_dim: int,
        token_dim: int,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.summary_dim = int(summary_dim)
        self.token_dim = int(token_dim)
        self.base_encoder = NativePairSemanticEncoderV2(
            hidden_dim=hidden_dim,
            summary_dim=summary_dim,
            token_dim=token_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 6, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.last_expert_weights: Optional[torch.Tensor] = None
        self.last_active_expert_mask: Optional[torch.Tensor] = None
        self.last_pooled_experts: Optional[torch.Tensor] = None
        self.last_pooled: Optional[torch.Tensor] = None

    def _expert_evidence(
        self,
        expert_summaries: torch.Tensor,
        expert_token_mask: torch.Tensor,
        expert_priors: torch.Tensor,
    ) -> torch.Tensor:
        confidence = expert_summaries[..., 0:1].clamp(0.0, 1.0)
        pair_focus = expert_summaries[..., 1:2].clamp(0.0, 1.0)
        reaction_center = expert_summaries[..., 2:3].clamp(0.0, 1.0)
        steric = expert_summaries[..., 3:4].clamp(0.0, 1.0)
        electronic = expert_summaries[..., 4:5].clamp(0.0, 1.0)
        token_density = expert_token_mask.float().mean(dim=-1, keepdim=True).to(dtype=expert_summaries.dtype)
        prior = expert_priors.unsqueeze(-1).clamp_min(0.0)
        return torch.cat(
            [confidence, pair_focus, reaction_center, steric, electronic, 0.5 * (token_density + prior)],
            dim=-1,
        )

    def forward(
        self,
        expert_summaries: torch.Tensor,
        expert_tokens: torch.Tensor,
        expert_token_masks: Optional[torch.Tensor] = None,
        expert_priors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if expert_summaries is None or expert_summaries.ndim != 3 or expert_summaries.numel() == 0:
            raise ValueError("expert_summaries must be a non-empty 3D tensor for semantic expert routing.")
        if expert_tokens is None or expert_tokens.ndim != 4 or expert_tokens.numel() == 0:
            raise ValueError("expert_tokens must be a non-empty 4D tensor for semantic expert routing.")
        if expert_summaries.size(0) != expert_tokens.size(0) or expert_summaries.size(1) != expert_tokens.size(1):
            raise ValueError("expert summaries and tokens must share batch and expert dimensions.")
        if expert_summaries.size(-1) != self.summary_dim:
            raise ValueError(
                f"expert_summaries last dimension ({expert_summaries.size(-1)}) does not match configured summary_dim ({self.summary_dim})."
            )
        if expert_tokens.size(-1) != self.token_dim:
            raise ValueError(
                f"expert_tokens last dimension ({expert_tokens.size(-1)}) does not match configured token_dim ({self.token_dim})."
            )

        batch_size, expert_count, token_count, _ = expert_tokens.shape
        if expert_token_masks is None:
            expert_token_masks = expert_tokens.abs().sum(dim=-1) > 1e-8
        else:
            expert_token_masks = expert_token_masks > 0
        if expert_priors is None:
            expert_priors = expert_summaries.new_zeros((batch_size, expert_count))
        else:
            expert_priors = expert_priors.to(dtype=expert_summaries.dtype, device=expert_summaries.device)

        active_mask = (
            expert_summaries.abs().sum(dim=-1) > 1e-8
        ) | (expert_tokens.abs().sum(dim=(-1, -2)) > 1e-8)
        fallback_rows = active_mask.sum(dim=-1) == 0
        if fallback_rows.any():
            active_mask = active_mask.clone()
            active_mask[fallback_rows, 0] = True
            expert_token_masks = expert_token_masks.clone()
            expert_token_masks[fallback_rows, 0, 0] = True

        flat_summaries = expert_summaries.reshape(batch_size * expert_count, self.summary_dim)
        flat_tokens = expert_tokens.reshape(batch_size * expert_count, token_count, self.token_dim)
        flat_token_masks = expert_token_masks.reshape(batch_size * expert_count, token_count)
        flat_embeddings = self.base_encoder(flat_summaries, flat_tokens, flat_token_masks)
        expert_embeddings = flat_embeddings.reshape(batch_size, expert_count, self.hidden_dim)
        expert_evidence = self._expert_evidence(expert_summaries, expert_token_masks, expert_priors)
        gate_logits = self.gate_mlp(torch.cat([expert_embeddings, expert_evidence], dim=-1)).squeeze(-1)
        prior_bias = torch.log(expert_priors.clamp_min(1e-6))
        gate_logits = gate_logits + prior_bias
        gate_logits = gate_logits.masked_fill(~active_mask, -1e9)
        expert_weights = torch.softmax(gate_logits, dim=-1)
        pooled = torch.sum(expert_embeddings * expert_weights.unsqueeze(-1), dim=1)
        pooled = self.output_norm(pooled)
        self.last_expert_weights = expert_weights.detach()
        self.last_active_expert_mask = active_mask.detach()
        self.last_pooled_experts = expert_embeddings.detach()
        self.last_pooled = pooled
        return pooled


class PairFirstLocalSemanticEncoder(nn.Module):
    """Native pair-first route with auxiliary component summaries kept secondary."""

    BRANCH_NAMES = ("pair", "global", "cat", "r1", "r2")
    BRANCH_PRIORS = {
        "pair": 1.10,
        "global": -0.20,
        "cat": 0.20,
        "r1": 0.25,
        "r2": -1.10,
    }
    AUXILIARY_CAPS = {
        "global": 0.16,
        "cat": 0.24,
        "r1": 0.24,
        "r2": 0.12,
    }
    PAIR_WEIGHT_BASE = 0.40
    PAIR_WEIGHT_RANGE = 0.20
    PAIR_EVIDENCE_CENTER = 0.62
    PAIR_EVIDENCE_SHARPNESS = 10.0

    def __init__(
        self,
        hidden_dim: int,
        summary_dim: int,
        token_dim: int,
        branch_index_map: Optional[Dict[str, Sequence[int]]] = None,
        branch_column_map: Optional[Dict[str, Sequence[str]]] = None,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.branch_names = tuple(self.BRANCH_NAMES)
        self.pair_encoder = NativePairSemanticEncoder(
            hidden_dim=hidden_dim,
            summary_dim=summary_dim,
            token_dim=token_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.branch_index_map = {
            name: tuple(int(index) for index in (branch_index_map or {}).get(name, ()))
            for name in LLM_SEMANTIC_BRANCH_ORDER
        }
        self.branch_column_map = {
            name: tuple(str(column) for column in (branch_column_map or {}).get(name, ()))
            for name in LLM_SEMANTIC_BRANCH_ORDER
        }
        self.branch_dims = {
            name: len(self.branch_index_map.get(name, ()))
            for name in LLM_SEMANTIC_BRANCH_ORDER
        }
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads}).")
        self.branch_projections = nn.ModuleDict()
        for name in LLM_SEMANTIC_BRANCH_ORDER:
            input_dim = self.branch_dims[name]
            if input_dim > 0:
                self.branch_projections[name] = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            else:
                self.branch_projections[name] = nn.Identity()
        self.branch_type_embedding = nn.Embedding(len(self.branch_names), hidden_dim)
        self.branch_missing_tokens = nn.Parameter(torch.zeros(len(self.branch_names), hidden_dim))
        nn.init.normal_(self.branch_missing_tokens, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(1, int(num_layers)))
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.pool_gate = nn.Linear(hidden_dim, 1)
        branch_prior = [float(self.BRANCH_PRIORS[name]) for name in self.branch_names]
        self.register_buffer("branch_prior_bias", torch.tensor(branch_prior, dtype=torch.float32), persistent=False)
        self.last_branch_weights: Optional[torch.Tensor] = None
        self.last_branch_mask: Optional[torch.Tensor] = None
        self.last_pair_evidence: Optional[torch.Tensor] = None
        self.last_pair_floor: Optional[torch.Tensor] = None
        self.last_pair_context: Optional[torch.Tensor] = None

    def _branch_presence_mask(self, llm_features: torch.Tensor) -> torch.Tensor:
        masks: List[torch.Tensor] = []
        batch_size = llm_features.size(0)
        device = llm_features.device
        dtype = llm_features.dtype
        eps = torch.tensor(1e-8, dtype=dtype, device=device)
        for name in self.branch_names[1:]:
            indexes = self.branch_index_map[name]
            columns = self.branch_column_map[name]
            if name == "global":
                masks.append(torch.ones((batch_size, 1), dtype=torch.bool, device=device))
                continue
            if not indexes:
                masks.append(torch.zeros((batch_size, 1), dtype=torch.bool, device=device))
                continue
            branch_values = llm_features[:, list(indexes)]
            branch_mask: Optional[torch.Tensor] = None
            for suffix in ("has_interactions", "has_reasoning_text"):
                for local_idx, column in enumerate(columns):
                    if column.endswith(suffix):
                        branch_mask = branch_values[:, local_idx : local_idx + 1] > 0.0
                        break
                if branch_mask is not None:
                    break
            if branch_mask is None:
                branch_mask = branch_values.abs().sum(dim=-1, keepdim=True) > eps
            masks.append(branch_mask)
        return torch.cat(masks, dim=-1)

    def _pair_priority(self, pair_summary: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        confidence = pair_summary[:, 0:1].clamp(0.0, 1.0) if pair_summary.size(-1) > 0 else pair_summary.new_zeros((pair_summary.size(0), 1))
        focus = pair_summary[:, 1:2].clamp(0.0, 1.0) if pair_summary.size(-1) > 1 else pair_summary.new_zeros((pair_summary.size(0), 1))
        reaction_center = pair_summary[:, 2:3].clamp(0.0, 1.0) if pair_summary.size(-1) > 2 else pair_summary.new_zeros((pair_summary.size(0), 1))
        joint_confidence = pair_summary[:, 10:11].clamp(0.0, 1.0) if pair_summary.size(-1) > 10 else pair_summary.new_zeros((pair_summary.size(0), 1))
        evidence = (
            0.35 * confidence
            + 0.30 * focus
            + 0.25 * reaction_center
            + 0.10 * joint_confidence
        )
        priority = torch.sigmoid((evidence - self.PAIR_EVIDENCE_CENTER) * self.PAIR_EVIDENCE_SHARPNESS)
        pair_floor = self.PAIR_WEIGHT_BASE + self.PAIR_WEIGHT_RANGE * priority
        return evidence, pair_floor

    def _enforce_pair_first_constraints(
        self,
        weights: torch.Tensor,
        branch_mask: torch.Tensor,
        pair_floor: torch.Tensor,
    ) -> torch.Tensor:
        pair_idx = self.branch_names.index("pair")
        pair_weight = weights[:, pair_idx : pair_idx + 1] * branch_mask[:, pair_idx : pair_idx + 1].to(dtype=weights.dtype)
        auxiliary_columns: Dict[str, torch.Tensor] = {}
        excess_total = torch.zeros_like(pair_weight)
        for name, cap in self.AUXILIARY_CAPS.items():
            branch_idx = self.branch_names.index(name)
            current = weights[:, branch_idx : branch_idx + 1] * branch_mask[:, branch_idx : branch_idx + 1].to(dtype=weights.dtype)
            branch_cap = branch_mask[:, branch_idx : branch_idx + 1].to(dtype=weights.dtype) * cap
            clamped = torch.minimum(current, branch_cap)
            auxiliary_columns[name] = clamped
            excess_total = excess_total + torch.clamp(current - clamped, min=0.0)
        pair_weight = pair_weight + excess_total

        deficit = torch.clamp(pair_floor - pair_weight, min=0.0)
        if torch.any(deficit > 0):
            donor_total = sum(auxiliary_columns.values()).clamp_min(1e-6)
            transfer = torch.minimum(deficit, donor_total)
            scale = torch.clamp(1.0 - (transfer / donor_total), min=0.0)
            for name in list(auxiliary_columns):
                auxiliary_columns[name] = auxiliary_columns[name] * scale
            pair_weight = pair_weight + transfer

        columns: List[torch.Tensor] = [pair_weight]
        for name in self.branch_names[1:]:
            columns.append(auxiliary_columns.get(name, torch.zeros_like(pair_weight)))
        adjusted = torch.cat(columns, dim=-1)
        adjusted = adjusted * branch_mask.to(dtype=adjusted.dtype)
        adjusted = adjusted / adjusted.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return adjusted

    def forward(
        self,
        llm_features: torch.Tensor,
        pair_summary: torch.Tensor,
        pair_tokens: torch.Tensor,
        pair_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if llm_features is None or llm_features.ndim != 2 or llm_features.numel() == 0:
            raise ValueError("llm_features must be a non-empty 2D tensor for pair-first local encoding.")
        if pair_summary is None or pair_tokens is None:
            raise ValueError("pair-first local encoding requires native pair summary and token tensors.")
        batch_size = llm_features.size(0)
        if pair_summary.size(0) != batch_size or pair_tokens.size(0) != batch_size:
            raise ValueError("pair-first local inputs must share the same batch size.")

        pair_token = self.pair_encoder(pair_summary, pair_tokens, pair_token_mask)
        auxiliary_mask = self._branch_presence_mask(llm_features)
        branch_mask = torch.cat(
            [torch.ones((batch_size, 1), dtype=torch.bool, device=llm_features.device), auxiliary_mask],
            dim=-1,
        )
        type_ids = torch.arange(len(self.branch_names), device=llm_features.device)
        type_embeddings = self.branch_type_embedding(type_ids)
        tokens: List[torch.Tensor] = [pair_token + type_embeddings[0].unsqueeze(0)]
        for branch_idx, name in enumerate(self.branch_names[1:], start=1):
            indexes = self.branch_index_map[name]
            if indexes:
                branch_values = llm_features[:, list(indexes)]
                projection = self.branch_projections[name]
                token = projection(branch_values)
            else:
                token = llm_features.new_zeros((batch_size, self.hidden_dim))
            token = token + type_embeddings[branch_idx].unsqueeze(0)
            missing_token = self.branch_missing_tokens[branch_idx].unsqueeze(0).expand(batch_size, -1)
            token = torch.where(branch_mask[:, branch_idx : branch_idx + 1], token, missing_token)
            tokens.append(token)
        token_tensor = self.token_norm(torch.stack(tokens, dim=1))
        encoded = self.encoder(token_tensor, src_key_padding_mask=~branch_mask)
        logits = self.pool_gate(encoded).squeeze(-1) + self.branch_prior_bias.to(device=encoded.device, dtype=encoded.dtype)
        logits = logits.masked_fill(~branch_mask, -1e9)
        weights = torch.softmax(logits, dim=-1)
        pair_evidence, pair_floor = self._pair_priority(pair_summary)
        adjusted_weights = self._enforce_pair_first_constraints(weights, branch_mask, pair_floor)
        pooled = torch.sum(encoded * adjusted_weights.unsqueeze(-1), dim=1)
        self.last_branch_weights = adjusted_weights.detach()
        self.last_branch_mask = branch_mask.detach()
        self.last_pair_evidence = pair_evidence.detach()
        self.last_pair_floor = pair_floor.detach()
        self.last_pair_context = pair_token
        return self.output_norm(pooled)


class PairConditionedPhysicalBridge(nn.Module):
    """Inject native pair semantics into the pooled physical route before QC and kinetics."""

    EVIDENCE_CENTER = 0.62
    EVIDENCE_SHARPNESS = 10.0
    INJECTION_LOCI = (
        "pre_qc_and_kinetic_fusion",
        "post_qc_pre_kinetic",
    )

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.1,
        injection_locus: str = "pre_qc_and_kinetic_fusion",
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.injection_locus = str(injection_locus or "pre_qc_and_kinetic_fusion").strip().lower()
        if self.injection_locus not in self.INJECTION_LOCI:
            raise ValueError(
                f"Unknown pair-conditioned bridge injection locus: {injection_locus}. "
                f"Expected one of {list(self.INJECTION_LOCI)}."
            )
        self.bridge_delta = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.bridge_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.pair_conditioned_bridge_scale = nn.Parameter(torch.tensor([-1.5], dtype=torch.float32))
        self.last_bridge_gate: Optional[torch.Tensor] = None
        self.last_pair_evidence: Optional[torch.Tensor] = None
        self.last_pair_context: Optional[torch.Tensor] = None

    @staticmethod
    def _pair_evidence(
        pair_summary: torch.Tensor,
        pair_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        confidence = pair_summary[:, 0:1].clamp(0.0, 1.0) if pair_summary.size(-1) > 0 else pair_summary.new_zeros((pair_summary.size(0), 1))
        focus = pair_summary[:, 1:2].clamp(0.0, 1.0) if pair_summary.size(-1) > 1 else pair_summary.new_zeros((pair_summary.size(0), 1))
        reaction_center = pair_summary[:, 2:3].clamp(0.0, 1.0) if pair_summary.size(-1) > 2 else pair_summary.new_zeros((pair_summary.size(0), 1))
        joint_confidence = pair_summary[:, 10:11].clamp(0.0, 1.0) if pair_summary.size(-1) > 10 else pair_summary.new_zeros((pair_summary.size(0), 1))
        token_density = pair_summary.new_zeros((pair_summary.size(0), 1))
        if pair_token_mask is not None and pair_token_mask.numel() > 0:
            token_density = pair_token_mask.float().mean(dim=-1, keepdim=True).to(dtype=pair_summary.dtype)
        return (
            0.30 * confidence
            + 0.25 * focus
            + 0.25 * reaction_center
            + 0.10 * joint_confidence
            + 0.10 * token_density
        )

    def forward(
        self,
        struct_emb: torch.Tensor,
        pair_context: torch.Tensor,
        pair_summary: torch.Tensor,
        pair_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if struct_emb.ndim != 2 or pair_context.ndim != 2 or pair_summary.ndim != 2:
            raise ValueError("pair-conditioned bridge expects 2D pooled tensors.")
        if struct_emb.size(0) != pair_context.size(0) or struct_emb.size(0) != pair_summary.size(0):
            raise ValueError("pair-conditioned bridge inputs must share the same batch size.")
        bridge_input = torch.cat([struct_emb, pair_context], dim=-1)
        pair_evidence = self._pair_evidence(pair_summary, pair_token_mask=pair_token_mask)
        gate_bias = (pair_evidence - self.EVIDENCE_CENTER) * self.EVIDENCE_SHARPNESS
        gate = torch.sigmoid(self.bridge_gate(bridge_input) + gate_bias)
        delta = torch.tanh(self.bridge_delta(bridge_input))
        bridge_scale = F.softplus(self.pair_conditioned_bridge_scale)
        fused = struct_emb + bridge_scale * gate * delta
        self.last_bridge_gate = gate.detach()
        self.last_pair_evidence = pair_evidence.detach()
        self.last_pair_context = pair_context.detach()
        return fused


class BranchStructuredSemanticEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        branch_index_map: Optional[Dict[str, Sequence[int]]],
        branch_column_map: Optional[Dict[str, Sequence[str]]] = None,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.branch_names = tuple(LLM_SEMANTIC_BRANCH_ORDER)
        self.branch_index_map = {
            name: tuple(int(index) for index in (branch_index_map or {}).get(name, ()))
            for name in self.branch_names
        }
        self.branch_column_map = {
            name: tuple(str(column) for column in (branch_column_map or {}).get(name, ()))
            for name in self.branch_names
        }
        self.branch_dims = {name: len(self.branch_index_map[name]) for name in self.branch_names}
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads}).")
        self.branch_projections = nn.ModuleDict()
        for name in self.branch_names:
            input_dim = self.branch_dims[name]
            if input_dim > 0:
                self.branch_projections[name] = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            else:
                self.branch_projections[name] = nn.Identity()
        self.branch_type_embedding = nn.Embedding(len(self.branch_names), hidden_dim)
        self.branch_missing_tokens = nn.Parameter(torch.zeros(len(self.branch_names), hidden_dim))
        nn.init.normal_(self.branch_missing_tokens, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(1, int(num_layers)))
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.pool_gate = nn.Linear(hidden_dim, 1)
        self.last_branch_weights: Optional[torch.Tensor] = None
        self.last_branch_mask: Optional[torch.Tensor] = None

    def _branch_presence_mask(self, llm_features: torch.Tensor) -> torch.Tensor:
        masks: List[torch.Tensor] = []
        batch_size = llm_features.size(0)
        device = llm_features.device
        dtype = llm_features.dtype
        for name in self.branch_names:
            indexes = self.branch_index_map[name]
            columns = self.branch_column_map[name]
            if name == "global":
                masks.append(torch.ones((batch_size, 1), dtype=torch.bool, device=device))
                continue
            if not indexes:
                masks.append(torch.zeros((batch_size, 1), dtype=torch.bool, device=device))
                continue
            branch_values = llm_features[:, list(indexes)]
            branch_mask: Optional[torch.Tensor] = None
            for suffix in ("has_interactions", "has_reasoning_text"):
                for local_idx, column in enumerate(columns):
                    if column.endswith(suffix):
                        branch_mask = branch_values[:, local_idx : local_idx + 1] > 0.0
                        break
                if branch_mask is not None:
                    break
            if branch_mask is None:
                branch_mask = branch_values.abs().sum(dim=-1, keepdim=True) > torch.tensor(1e-8, dtype=dtype, device=device)
            masks.append(branch_mask)
        return torch.cat(masks, dim=-1)

    def forward(self, llm_features: torch.Tensor) -> torch.Tensor:
        if llm_features is None or llm_features.ndim != 2 or llm_features.numel() == 0:
            raise ValueError("llm_features must be a non-empty 2D tensor for branch-structured encoding.")
        batch_size = llm_features.size(0)
        branch_mask = self._branch_presence_mask(llm_features)
        branch_mask[:, 0] = True
        type_ids = torch.arange(len(self.branch_names), device=llm_features.device)
        type_embeddings = self.branch_type_embedding(type_ids)
        tokens: List[torch.Tensor] = []
        for branch_idx, name in enumerate(self.branch_names):
            indexes = self.branch_index_map[name]
            if indexes:
                branch_values = llm_features[:, list(indexes)]
                projection = self.branch_projections[name]
                token = projection(branch_values)
            else:
                token = llm_features.new_zeros((batch_size, self.hidden_dim))
            token = token + type_embeddings[branch_idx].unsqueeze(0)
            missing_token = self.branch_missing_tokens[branch_idx].unsqueeze(0).expand(batch_size, -1)
            token = torch.where(branch_mask[:, branch_idx : branch_idx + 1], token, missing_token)
            tokens.append(token)
        token_tensor = self.token_norm(torch.stack(tokens, dim=1))
        encoded = self.encoder(token_tensor, src_key_padding_mask=~branch_mask)
        logits = self.pool_gate(encoded).squeeze(-1)
        logits = logits.masked_fill(~branch_mask, -1e9)
        weights = torch.softmax(logits, dim=-1)
        pooled = torch.sum(encoded * weights.unsqueeze(-1), dim=1)
        self.last_branch_weights = weights.detach()
        self.last_branch_mask = branch_mask.detach()
        return self.output_norm(pooled)


class ProductOptionalLocalSemanticEncoder(nn.Module):
    """Local semantic route with catalyst/reactant-first pooling and optional product authority."""

    CORE_BRANCH_PRIORS = {
        "global": 0.15,
        "cat": 0.95,
        "r1": 0.95,
        "r2": -1.10,
        "pair": 0.35,
    }
    PRODUCT_EVIDENCE_CENTER = 0.72
    PRODUCT_EVIDENCE_SHARPNESS = 10.0
    PRODUCT_WEIGHT_BASE = 0.08
    PRODUCT_WEIGHT_RANGE = 0.32

    def __init__(
        self,
        hidden_dim: int,
        branch_index_map: Optional[Dict[str, Sequence[int]]] = None,
        branch_column_map: Optional[Dict[str, Sequence[str]]] = None,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.branch_names = tuple(LLM_SEMANTIC_BRANCH_ORDER)
        self.branch_index_map = {
            name: tuple(int(index) for index in (branch_index_map or {}).get(name, ()))
            for name in self.branch_names
        }
        self.branch_column_map = {
            name: tuple(str(column) for column in (branch_column_map or {}).get(name, ()))
            for name in self.branch_names
        }
        self.branch_dims = {name: len(self.branch_index_map[name]) for name in self.branch_names}
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads}).")
        self.branch_projections = nn.ModuleDict()
        for name in self.branch_names:
            input_dim = self.branch_dims[name]
            if input_dim > 0:
                self.branch_projections[name] = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            else:
                self.branch_projections[name] = nn.Identity()
        self.branch_type_embedding = nn.Embedding(len(self.branch_names), hidden_dim)
        self.branch_missing_tokens = nn.Parameter(torch.zeros(len(self.branch_names), hidden_dim))
        nn.init.normal_(self.branch_missing_tokens, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(1, int(num_layers)))
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.pool_gate = nn.Linear(hidden_dim, 1)
        branch_prior = [float(self.CORE_BRANCH_PRIORS.get(name, 0.0)) for name in self.branch_names]
        self.register_buffer("branch_prior_bias", torch.tensor(branch_prior, dtype=torch.float32), persistent=False)
        self.last_branch_weights: Optional[torch.Tensor] = None
        self.last_branch_mask: Optional[torch.Tensor] = None
        self.last_product_evidence: Optional[torch.Tensor] = None
        self.last_product_cap: Optional[torch.Tensor] = None

    def _branch_presence_mask(self, llm_features: torch.Tensor) -> torch.Tensor:
        masks: List[torch.Tensor] = []
        batch_size = llm_features.size(0)
        device = llm_features.device
        dtype = llm_features.dtype
        eps = torch.tensor(1e-8, dtype=dtype, device=device)
        for name in self.branch_names:
            indexes = self.branch_index_map[name]
            columns = self.branch_column_map[name]
            if name == "global":
                masks.append(torch.ones((batch_size, 1), dtype=torch.bool, device=device))
                continue
            if not indexes:
                masks.append(torch.zeros((batch_size, 1), dtype=torch.bool, device=device))
                continue
            branch_values = llm_features[:, list(indexes)]
            branch_mask: Optional[torch.Tensor] = None
            for suffix in ("has_interactions", "has_reasoning_text"):
                for local_idx, column in enumerate(columns):
                    if column.endswith(suffix):
                        branch_mask = branch_values[:, local_idx : local_idx + 1] > 0.0
                        break
                if branch_mask is not None:
                    break
            if branch_mask is None:
                branch_mask = branch_values.abs().sum(dim=-1, keepdim=True) > eps
            masks.append(branch_mask)
        return torch.cat(masks, dim=-1)

    def _product_evidence(self, llm_features: torch.Tensor, branch_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = llm_features.size(0)
        device = llm_features.device
        dtype = llm_features.dtype
        r2_idx = self.branch_names.index("r2")
        if not self.branch_index_map["r2"]:
            zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
            return zeros, zeros

        indexes = list(self.branch_index_map["r2"])
        columns = list(self.branch_column_map["r2"])
        branch_values = llm_features[:, indexes]

        def _column_or_zero(suffix: str) -> torch.Tensor:
            for local_idx, column in enumerate(columns):
                if column.endswith(suffix):
                    return branch_values[:, local_idx : local_idx + 1].clamp(0.0, 1.0)
            return torch.zeros((batch_size, 1), dtype=dtype, device=device)

        interaction_presence = _column_or_zero("has_interactions")
        text_presence = _column_or_zero("has_reasoning_text")
        mean_confidence = _column_or_zero("mean_confidence")
        total_interactions = _column_or_zero("total_interactions")
        density = torch.sigmoid((total_interactions - math.log1p(1.5)) * 4.0)

        text_scores: List[torch.Tensor] = []
        for local_idx, column in enumerate(columns):
            if "_text_" in column:
                text_scores.append(branch_values[:, local_idx : local_idx + 1].clamp(0.0, 1.0))
        if text_scores:
            text_signal = text_scores[0]
            for term in text_scores[1:]:
                text_signal = torch.maximum(text_signal, term)
        else:
            text_signal = torch.zeros((batch_size, 1), dtype=dtype, device=device)

        presence = branch_mask[:, r2_idx : r2_idx + 1].to(dtype=dtype)
        evidence = (
            0.40 * torch.maximum(interaction_presence, text_presence)
            + 0.25 * mean_confidence
            + 0.20 * density
            + 0.15 * text_signal
        ) * presence
        strong_signal = torch.sigmoid(
            (evidence - self.PRODUCT_EVIDENCE_CENTER) * self.PRODUCT_EVIDENCE_SHARPNESS
        )
        product_cap = presence * (self.PRODUCT_WEIGHT_BASE + self.PRODUCT_WEIGHT_RANGE * strong_signal)
        return evidence, product_cap

    @staticmethod
    def _redistribute_from_product(weights: torch.Tensor, branch_mask: torch.Tensor, product_cap: torch.Tensor) -> torch.Tensor:
        r2_idx = int(LLM_SEMANTIC_BRANCH_ORDER.index("r2"))
        product_weight = weights[:, r2_idx : r2_idx + 1]
        clamped_product = torch.minimum(product_weight, product_cap)
        excess = torch.clamp(product_weight - clamped_product, min=0.0)
        redistribute_mask = branch_mask.clone()
        redistribute_mask[:, r2_idx] = False
        redistribute_weights = weights * redistribute_mask.to(dtype=weights.dtype)
        redistribute_denom = redistribute_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        redistributed_non_product = redistribute_weights + (redistribute_weights / redistribute_denom) * excess
        product_one_hot = torch.zeros_like(weights)
        product_one_hot[:, r2_idx : r2_idx + 1] = 1.0
        adjusted = redistributed_non_product + product_one_hot * clamped_product
        adjusted = adjusted * branch_mask.to(dtype=adjusted.dtype)
        adjusted = adjusted / adjusted.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return adjusted

    def forward(self, llm_features: torch.Tensor) -> torch.Tensor:
        if llm_features is None or llm_features.ndim != 2 or llm_features.numel() == 0:
            raise ValueError("llm_features must be a non-empty 2D tensor for product-optional local encoding.")
        batch_size = llm_features.size(0)
        branch_mask = self._branch_presence_mask(llm_features)
        branch_mask[:, 0] = True
        type_ids = torch.arange(len(self.branch_names), device=llm_features.device)
        type_embeddings = self.branch_type_embedding(type_ids)
        tokens: List[torch.Tensor] = []
        for branch_idx, name in enumerate(self.branch_names):
            indexes = self.branch_index_map[name]
            if indexes:
                branch_values = llm_features[:, list(indexes)]
                projection = self.branch_projections[name]
                token = projection(branch_values)
            else:
                token = llm_features.new_zeros((batch_size, self.hidden_dim))
            token = token + type_embeddings[branch_idx].unsqueeze(0)
            missing_token = self.branch_missing_tokens[branch_idx].unsqueeze(0).expand(batch_size, -1)
            token = torch.where(branch_mask[:, branch_idx : branch_idx + 1], token, missing_token)
            tokens.append(token)
        token_tensor = self.token_norm(torch.stack(tokens, dim=1))
        encoded = self.encoder(token_tensor, src_key_padding_mask=~branch_mask)
        logits = self.pool_gate(encoded).squeeze(-1) + self.branch_prior_bias.to(device=encoded.device, dtype=encoded.dtype)
        logits = logits.masked_fill(~branch_mask, -1e9)
        weights = torch.softmax(logits, dim=-1)
        product_evidence, product_cap = self._product_evidence(llm_features, branch_mask)
        adjusted_weights = self._redistribute_from_product(weights, branch_mask, product_cap)
        pooled = torch.sum(encoded * adjusted_weights.unsqueeze(-1), dim=1)
        self.last_branch_weights = adjusted_weights.detach()
        self.last_branch_mask = branch_mask.detach()
        self.last_product_evidence = product_evidence.detach()
        self.last_product_cap = product_cap.detach()
        return self.output_norm(pooled)


class ProvenanceDualBranchSemanticEncoder(nn.Module):
    """Encode shared semantics plus provenance-specific exact/propagated branches before pooling."""

    SHADOW_BRANCH_WEIGHT = 0.22
    BRANCH_NAMES = ("shared", "exact", "propagated")
    BRANCH_PRIORS = {
        "shared": 0.20,
        "exact": 0.65,
        "propagated": 0.25,
    }

    def __init__(
        self,
        hidden_dim: int,
        llm_dim: int,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.llm_dim = int(llm_dim)
        if self.llm_dim <= 0:
            raise ValueError("llm_dim must be positive for provenance_dual_branch encoding.")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads}).")
        self.shared_projection = nn.Sequential(
            nn.Linear(self.llm_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.exact_projection = nn.Sequential(
            nn.Linear(self.llm_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.propagated_projection = nn.Sequential(
            nn.Linear(self.llm_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.branch_type_embedding = nn.Embedding(len(self.BRANCH_NAMES), hidden_dim)
        self.branch_missing_tokens = nn.Parameter(torch.zeros(len(self.BRANCH_NAMES), hidden_dim))
        nn.init.normal_(self.branch_missing_tokens, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(1, int(num_layers)))
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.pool_gate = nn.Linear(hidden_dim, 1)
        branch_prior = [float(self.BRANCH_PRIORS[name]) for name in self.BRANCH_NAMES]
        self.register_buffer("branch_prior_bias", torch.tensor(branch_prior, dtype=torch.float32), persistent=False)
        self.last_branch_weights: Optional[torch.Tensor] = None
        self.last_provenance_strengths: Optional[torch.Tensor] = None
        self.last_exact_mask: Optional[torch.Tensor] = None

    def forward(
        self,
        llm_features: torch.Tensor,
        exact_covered_indicator: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if llm_features is None or llm_features.ndim != 2 or llm_features.numel() == 0:
            raise ValueError("llm_features must be a non-empty 2D tensor for provenance_dual_branch encoding.")
        if llm_features.size(-1) != self.llm_dim:
            raise ValueError(
                f"llm_features last dimension ({llm_features.size(-1)}) does not match configured llm_dim ({self.llm_dim})."
            )
        batch_size = llm_features.size(0)
        device = llm_features.device
        dtype = llm_features.dtype
        if exact_covered_indicator is None or exact_covered_indicator.numel() == 0:
            exact_mask = torch.zeros((batch_size, 1), dtype=dtype, device=device)
        else:
            exact_mask = (exact_covered_indicator.view(batch_size, 1) > 0.5).to(dtype=dtype)
        propagated_mask = 1.0 - exact_mask
        exact_strength = exact_mask + self.SHADOW_BRANCH_WEIGHT * propagated_mask
        propagated_strength = propagated_mask + self.SHADOW_BRANCH_WEIGHT * exact_mask

        type_ids = torch.arange(len(self.BRANCH_NAMES), device=device)
        type_embeddings = self.branch_type_embedding(type_ids)
        shared_token = self.shared_projection(llm_features) + type_embeddings[0].unsqueeze(0)
        exact_token = self.exact_projection(llm_features) * exact_strength + type_embeddings[1].unsqueeze(0)
        propagated_token = (
            self.propagated_projection(llm_features) * propagated_strength + type_embeddings[2].unsqueeze(0)
        )
        exact_missing = self.branch_missing_tokens[1].unsqueeze(0).expand(batch_size, -1)
        propagated_missing = self.branch_missing_tokens[2].unsqueeze(0).expand(batch_size, -1)
        exact_token = torch.where(exact_strength > 1e-8, exact_token, exact_missing)
        propagated_token = torch.where(propagated_strength > 1e-8, propagated_token, propagated_missing)

        token_tensor = self.token_norm(torch.stack([shared_token, exact_token, propagated_token], dim=1))
        encoded = self.encoder(token_tensor)

        prior_strengths = torch.cat(
            [
                torch.ones((batch_size, 1), dtype=dtype, device=device),
                exact_strength,
                propagated_strength,
            ],
            dim=-1,
        )
        logits = self.pool_gate(encoded).squeeze(-1)
        logits = logits + self.branch_prior_bias.to(device=device, dtype=dtype)
        logits = logits + torch.log(prior_strengths.clamp_min(1e-6))
        weights = torch.softmax(logits, dim=-1)
        pooled = torch.sum(encoded * weights.unsqueeze(-1), dim=1)

        self.last_branch_weights = weights.detach()
        self.last_provenance_strengths = prior_strengths.detach()
        self.last_exact_mask = exact_mask.detach()
        return self.output_norm(pooled)


class PairCompatibilitySemanticEncoder(nn.Module):
    """Pair-aware V2 head with explicit catalyst-reactant compatibility vs product contrast."""

    BRANCH_PRIORS = {
        "global": 0.05,
        "cat": 0.45,
        "r1": 0.65,
        "r2": -0.35,
        "pair": 0.95,
    }

    def __init__(
        self,
        hidden_dim: int,
        branch_index_map: Optional[Dict[str, Sequence[int]]] = None,
        branch_column_map: Optional[Dict[str, Sequence[str]]] = None,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.branch_names = tuple(LLM_SEMANTIC_BRANCH_ORDER)
        self.pair_group_names = tuple(LLM_PAIR_GROUP_ORDER)
        self.branch_index_map = {
            name: tuple(int(index) for index in (branch_index_map or {}).get(name, ()))
            for name in self.branch_names
        }
        self.branch_column_map = {
            name: tuple(str(column) for column in (branch_column_map or {}).get(name, ()))
            for name in self.branch_names
        }
        self.branch_dims = {name: len(self.branch_index_map[name]) for name in self.branch_names}
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads}).")
        self.branch_projections = nn.ModuleDict()
        for name in self.branch_names:
            input_dim = self.branch_dims[name]
            if input_dim > 0:
                self.branch_projections[name] = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            else:
                self.branch_projections[name] = nn.Identity()
        self.branch_type_embedding = nn.Embedding(len(self.branch_names), hidden_dim)
        self.branch_missing_tokens = nn.Parameter(torch.zeros(len(self.branch_names), hidden_dim))
        nn.init.normal_(self.branch_missing_tokens, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(1, int(num_layers)))
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.pool_gate = nn.Linear(hidden_dim, 1)
        branch_prior = [float(self.BRANCH_PRIORS.get(name, 0.0)) for name in self.branch_names]
        self.register_buffer("branch_prior_bias", torch.tensor(branch_prior, dtype=torch.float32), persistent=False)

        pair_columns = list(self.branch_column_map.get("pair", ()))
        self.pair_group_index_map: Dict[str, Tuple[int, ...]] = {}
        self.pair_group_column_map: Dict[str, Tuple[str, ...]] = {}
        for group_name in self.pair_group_names:
            local_indexes: List[int] = []
            local_columns: List[str] = []
            for local_idx, column in enumerate(pair_columns):
                if group_name == "core_pair" and column.startswith("llm_pair_cat_r1_"):
                    local_indexes.append(local_idx)
                    local_columns.append(column)
                elif group_name == "product_pair" and column.startswith("llm_pair_cat_r2_"):
                    local_indexes.append(local_idx)
                    local_columns.append(column)
                elif group_name == "contrast" and column.startswith("llm_pair_") and not (
                    column.startswith("llm_pair_cat_r1_") or column.startswith("llm_pair_cat_r2_")
                ):
                    local_indexes.append(local_idx)
                    local_columns.append(column)
            self.pair_group_index_map[group_name] = tuple(local_indexes)
            self.pair_group_column_map[group_name] = tuple(local_columns)
        self.pair_group_dims = {name: len(self.pair_group_index_map[name]) for name in self.pair_group_names}
        self.pair_group_projections = nn.ModuleDict()
        for name in self.pair_group_names:
            input_dim = self.pair_group_dims[name]
            if input_dim > 0:
                self.pair_group_projections[name] = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            else:
                self.pair_group_projections[name] = nn.Identity()
        self.core_pair_fuser = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.product_pair_fuser = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.last_branch_weights: Optional[torch.Tensor] = None
        self.last_branch_mask: Optional[torch.Tensor] = None
        self.last_core_pair_evidence: Optional[torch.Tensor] = None
        self.last_product_pair_evidence: Optional[torch.Tensor] = None
        self.last_pair_margin_signal: Optional[torch.Tensor] = None

    def _branch_presence_mask(self, llm_features: torch.Tensor) -> torch.Tensor:
        masks: List[torch.Tensor] = []
        batch_size = llm_features.size(0)
        device = llm_features.device
        dtype = llm_features.dtype
        eps = torch.tensor(1e-8, dtype=dtype, device=device)
        for name in self.branch_names:
            indexes = self.branch_index_map[name]
            columns = self.branch_column_map[name]
            if name == "global":
                masks.append(torch.ones((batch_size, 1), dtype=torch.bool, device=device))
                continue
            if not indexes:
                masks.append(torch.zeros((batch_size, 1), dtype=torch.bool, device=device))
                continue
            branch_values = llm_features[:, list(indexes)]
            branch_mask: Optional[torch.Tensor] = None
            for suffix in ("has_interactions", "has_reasoning_text"):
                for local_idx, column in enumerate(columns):
                    if column.endswith(suffix):
                        branch_mask = branch_values[:, local_idx : local_idx + 1] > 0.0
                        break
                if branch_mask is not None:
                    break
            if branch_mask is None:
                branch_mask = branch_values.abs().sum(dim=-1, keepdim=True) > eps
            masks.append(branch_mask)
        return torch.cat(masks, dim=-1)

    def _project_branch_tokens(
        self,
        llm_features: torch.Tensor,
        branch_mask: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        batch_size = llm_features.size(0)
        type_ids = torch.arange(len(self.branch_names), device=llm_features.device)
        type_embeddings = self.branch_type_embedding(type_ids)
        branch_tokens: Dict[str, torch.Tensor] = {}
        ordered_tokens: List[torch.Tensor] = []
        for branch_idx, name in enumerate(self.branch_names):
            indexes = self.branch_index_map[name]
            if indexes:
                branch_values = llm_features[:, list(indexes)]
                token = self.branch_projections[name](branch_values)
            else:
                token = llm_features.new_zeros((batch_size, self.hidden_dim))
            token = token + type_embeddings[branch_idx].unsqueeze(0)
            missing_token = self.branch_missing_tokens[branch_idx].unsqueeze(0).expand(batch_size, -1)
            token = torch.where(branch_mask[:, branch_idx : branch_idx + 1], token, missing_token)
            branch_tokens[name] = token
            ordered_tokens.append(token)
        return branch_tokens, self.token_norm(torch.stack(ordered_tokens, dim=1))

    def _pair_group_projection(self, llm_features: torch.Tensor, group_name: str) -> torch.Tensor:
        pair_indexes = self.branch_index_map.get("pair", ())
        local_indexes = self.pair_group_index_map.get(group_name, ())
        if not pair_indexes or not local_indexes:
            return llm_features.new_zeros((llm_features.size(0), self.hidden_dim))
        pair_values = llm_features[:, list(pair_indexes)]
        group_values = pair_values[:, list(local_indexes)]
        projection = self.pair_group_projections[group_name]
        return projection(group_values)

    def _pair_evidence(self, llm_features: torch.Tensor, group_name: str) -> torch.Tensor:
        batch_size = llm_features.size(0)
        device = llm_features.device
        dtype = llm_features.dtype
        pair_indexes = self.branch_index_map.get("pair", ())
        local_indexes = self.pair_group_index_map.get(group_name, ())
        columns = self.pair_group_column_map.get(group_name, ())
        if not pair_indexes or not local_indexes:
            return torch.zeros((batch_size, 1), dtype=dtype, device=device)
        pair_values = llm_features[:, list(pair_indexes)]
        selected = pair_values[:, list(local_indexes)]

        def _column_or_zero(suffix: str) -> torch.Tensor:
            for local_idx, column in enumerate(columns):
                if column.endswith(suffix):
                    return selected[:, local_idx : local_idx + 1].clamp(0.0, 1.0)
            return torch.zeros((batch_size, 1), dtype=dtype, device=device)

        return (
            0.28 * _column_or_zero("reaction_center_score")
            + 0.20 * _column_or_zero("text_synergy")
            + 0.14 * _column_or_zero("steric_compatibility")
            + 0.14 * _column_or_zero("electronic_complement")
            + 0.10 * _column_or_zero("interaction_balance")
            + 0.07 * _column_or_zero("confidence_balance")
            + 0.05 * _column_or_zero("focus_alignment")
            + 0.02 * _column_or_zero("type_overlap")
        )

    def _pair_margin_signal(self, llm_features: torch.Tensor) -> torch.Tensor:
        batch_size = llm_features.size(0)
        device = llm_features.device
        dtype = llm_features.dtype
        pair_indexes = self.branch_index_map.get("pair", ())
        local_indexes = self.pair_group_index_map.get("contrast", ())
        columns = self.pair_group_column_map.get("contrast", ())
        if not pair_indexes or not local_indexes:
            return torch.zeros((batch_size, 1), dtype=dtype, device=device)
        pair_values = llm_features[:, list(pair_indexes)]
        selected = pair_values[:, list(local_indexes)]

        def _column_or_center(suffix: str) -> torch.Tensor:
            for local_idx, column in enumerate(columns):
                if column.endswith(suffix):
                    return selected[:, local_idx : local_idx + 1].clamp(0.0, 1.0)
            return torch.full((batch_size, 1), 0.5, dtype=dtype, device=device)

        score_margin = _column_or_center("score_margin") - 0.5
        focus_margin = _column_or_center("focus_margin") - 0.5
        return 0.70 * score_margin + 0.30 * focus_margin

    def forward(self, llm_features: torch.Tensor) -> torch.Tensor:
        if llm_features is None or llm_features.ndim != 2 or llm_features.numel() == 0:
            raise ValueError("llm_features must be a non-empty 2D tensor for pair compatibility encoding.")
        branch_mask = self._branch_presence_mask(llm_features)
        branch_mask[:, 0] = True
        branch_tokens, token_tensor = self._project_branch_tokens(llm_features, branch_mask)

        core_pair_proj = self._pair_group_projection(llm_features, "core_pair")
        product_pair_proj = self._pair_group_projection(llm_features, "product_pair")
        contrast_proj = self._pair_group_projection(llm_features, "contrast")
        core_update = self.core_pair_fuser(
            torch.cat([branch_tokens["cat"], branch_tokens["r1"], core_pair_proj, contrast_proj], dim=-1)
        )
        product_update = self.product_pair_fuser(
            torch.cat([branch_tokens["cat"], branch_tokens["r2"], product_pair_proj, contrast_proj], dim=-1)
        )

        token_tensor = token_tensor.clone()
        cat_idx = self.branch_names.index("cat")
        r1_idx = self.branch_names.index("r1")
        r2_idx = self.branch_names.index("r2")
        pair_idx = self.branch_names.index("pair")
        token_tensor[:, cat_idx] = token_tensor[:, cat_idx] + 0.20 * core_update
        token_tensor[:, r1_idx] = token_tensor[:, r1_idx] + 0.35 * core_update
        token_tensor[:, r2_idx] = token_tensor[:, r2_idx] + 0.18 * product_update
        token_tensor[:, pair_idx] = token_tensor[:, pair_idx] + core_update - 0.35 * product_update

        encoded = self.encoder(token_tensor, src_key_padding_mask=~branch_mask)
        logits = self.pool_gate(encoded).squeeze(-1) + self.branch_prior_bias.to(device=encoded.device, dtype=encoded.dtype)
        core_evidence = self._pair_evidence(llm_features, "core_pair")
        product_evidence = self._pair_evidence(llm_features, "product_pair")
        margin_signal = self._pair_margin_signal(llm_features)
        evidence_bias = torch.zeros_like(logits)
        evidence_bias[:, cat_idx : cat_idx + 1] += 0.12 * core_evidence
        evidence_bias[:, r1_idx : r1_idx + 1] += 0.22 * core_evidence + 0.18 * margin_signal
        evidence_bias[:, r2_idx : r2_idx + 1] += 0.18 * product_evidence - 0.45 * margin_signal
        evidence_bias[:, pair_idx : pair_idx + 1] += 0.40 * core_evidence - 0.20 * product_evidence + 0.70 * margin_signal
        logits = logits + evidence_bias
        logits = logits.masked_fill(~branch_mask, -1e9)
        weights = torch.softmax(logits, dim=-1)
        pooled = torch.sum(encoded * weights.unsqueeze(-1), dim=1)
        self.last_branch_weights = weights.detach()
        self.last_branch_mask = branch_mask.detach()
        self.last_core_pair_evidence = core_evidence.detach()
        self.last_product_pair_evidence = product_evidence.detach()
        self.last_pair_margin_signal = margin_signal.detach()
        return self.output_norm(pooled)


class GroupedSemanticSelector(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        group_index_map: Optional[Dict[str, Sequence[int]]] = None,
        group_column_map: Optional[Dict[str, Sequence[str]]] = None,
        selection_top_k: int = 0,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.group_names = tuple(LLM_SEMANTIC_GROUP_ORDER)
        self.group_index_map = {
            name: tuple(int(index) for index in (group_index_map or {}).get(name, ()))
            for name in self.group_names
        }
        self.group_column_map = {
            name: tuple(str(column) for column in (group_column_map or {}).get(name, ()))
            for name in self.group_names
        }
        self.selection_top_k = max(0, int(selection_top_k))
        self.group_projections = nn.ModuleDict()
        for name, indexes in self.group_index_map.items():
            if indexes:
                self.group_projections[name] = nn.Sequential(
                    nn.Linear(len(indexes), hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                )
        self.group_type_embedding = nn.Embedding(len(self.group_names), hidden_dim)
        self.group_missing_tokens = nn.Parameter(torch.zeros(len(self.group_names), hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=max(1, int(num_heads)),
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(1, int(num_layers)))
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.selector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.last_group_weights: Optional[torch.Tensor] = None
        self.last_group_mask: Optional[torch.Tensor] = None
        self.last_group_selected_mask: Optional[torch.Tensor] = None

    def _group_presence_mask(self, llm_features: torch.Tensor) -> torch.Tensor:
        masks: List[torch.Tensor] = []
        batch_size = llm_features.size(0)
        device = llm_features.device
        dtype = llm_features.dtype
        eps = torch.tensor(1e-8, dtype=dtype, device=device)
        for name in self.group_names:
            indexes = self.group_index_map[name]
            if not indexes:
                masks.append(torch.zeros((batch_size, 1), dtype=torch.bool, device=device))
                continue
            group_values = llm_features[:, list(indexes)]
            group_mask: Optional[torch.Tensor] = None
            columns = self.group_column_map[name]
            for suffix in ("has_interactions", "has_reasoning_text"):
                for local_idx, column in enumerate(columns):
                    if column.endswith(suffix):
                        group_mask = group_values[:, local_idx : local_idx + 1] > 0.0
                        break
                if group_mask is not None:
                    break
            if group_mask is None:
                group_mask = group_values.abs().sum(dim=-1, keepdim=True) > eps
            masks.append(group_mask)
        return torch.cat(masks, dim=-1)

    def _topk_mask(self, logits: torch.Tensor, group_mask: torch.Tensor) -> torch.Tensor:
        if self.selection_top_k <= 0 or self.selection_top_k >= logits.size(-1):
            selected = group_mask.clone()
            fallback = selected.sum(dim=-1, keepdim=True) == 0
            if fallback.any():
                default_mask = torch.zeros_like(selected)
                default_mask[:, 0] = True
                selected = torch.where(fallback, default_mask, selected)
            return selected
        masked_logits = logits.masked_fill(~group_mask, -1e9)
        topk = min(self.selection_top_k, logits.size(-1))
        topk_idx = masked_logits.topk(topk, dim=-1).indices
        selected = torch.zeros_like(group_mask)
        selected.scatter_(1, topk_idx, True)
        selected = selected & group_mask
        fallback = selected.sum(dim=-1, keepdim=True) == 0
        if fallback.any():
            default_mask = torch.zeros_like(selected)
            default_mask[:, 0] = True
            selected = torch.where(fallback, default_mask, selected)
        return selected

    def forward(self, llm_features: torch.Tensor) -> torch.Tensor:
        if llm_features is None or llm_features.ndim != 2 or llm_features.numel() == 0:
            raise ValueError("llm_features must be a non-empty 2D tensor for grouped semantic selection.")
        batch_size = llm_features.size(0)
        group_mask = self._group_presence_mask(llm_features)
        type_ids = torch.arange(len(self.group_names), device=llm_features.device)
        type_embeddings = self.group_type_embedding(type_ids)
        tokens: List[torch.Tensor] = []
        for group_idx, name in enumerate(self.group_names):
            indexes = self.group_index_map[name]
            if indexes:
                group_values = llm_features[:, list(indexes)]
                token = self.group_projections[name](group_values)
            else:
                token = llm_features.new_zeros((batch_size, self.hidden_dim))
            token = token + type_embeddings[group_idx].unsqueeze(0)
            missing_token = self.group_missing_tokens[group_idx].unsqueeze(0).expand(batch_size, -1)
            token = torch.where(group_mask[:, group_idx : group_idx + 1], token, missing_token)
            tokens.append(token)
        token_tensor = self.token_norm(torch.stack(tokens, dim=1))
        encoded = self.encoder(token_tensor, src_key_padding_mask=~group_mask)
        logits = self.selector(encoded).squeeze(-1)
        selected_mask = self._topk_mask(logits, group_mask)
        logits = logits.masked_fill(~selected_mask, -1e9)
        weights = torch.softmax(logits, dim=-1)
        pooled = torch.sum(encoded * weights.unsqueeze(-1), dim=1)
        self.last_group_weights = weights.detach()
        self.last_group_mask = group_mask.detach()
        self.last_group_selected_mask = selected_mask.detach()
        return self.output_norm(pooled)


class PaiNNPrecisionHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        num_rbf: int = 32,
        cutoff: float = 5.0,
        feat_dim: int = 0,
        qc_dim: int = 0,
        physics_fusion: str = "gated",
        llm_dim: int = 0,
        native_pair_dim: int = 0,
        native_pair_token_dim: int = 0,
        gate_dropout: float = 0.1,
        min_llm_confidence: float = 0.7,
        gate_confidence_sharpness: float = 8.0,
        gate_density_index: int = 0,
        gate_density_center: float = 0.8,
        gate_density_sharpness: float = 3.0,
        gate_adaptive_strength: float = 1.2,
        llm_fusion_mode: str = "residual",
        llm_cross_attention_heads: int = 4,
        llm_conditioned_scale: float = 1.0,
        llm_semantic_encoder: str = "flat",
        llm_branch_index_map: Optional[Dict[str, Sequence[int]]] = None,
        llm_branch_column_map: Optional[Dict[str, Sequence[str]]] = None,
        llm_group_index_map: Optional[Dict[str, Sequence[int]]] = None,
        llm_group_column_map: Optional[Dict[str, Sequence[str]]] = None,
        llm_group_top_k: int = 0,
        llm_local_meta_gate: Optional[Sequence[str]] = None,
        llm_gate_product_focus_scale: float = 1.0,
        semantic_physical_bridge: bool = False,
        pair_conditioned_bridge: bool = False,
        pair_conditioned_bridge_locus: str = "pre_qc_and_kinetic_fusion",
        global_attention: bool = False,
        global_attention_heads: int = 4,
        global_attention_layers: int = 1,
        global_attention_dropout: float = 0.1,
        interaction_cross_attention: bool = True,
        node_qc_dim: int = 0,
        node_qc_gate_dim: int = 0,
        reaction_center_coupling: bool = False,
        reaction_center_pair_feature_dim: int = 0,
        kinetic_relay: bool = False,
    ):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.qc_dim = int(qc_dim)
        self.llm_dim = int(llm_dim)
        self.native_pair_dim = int(native_pair_dim)
        self.native_pair_token_dim = int(native_pair_token_dim)
        self.node_qc_dim = int(node_qc_dim)
        self.min_llm_confidence = float(min_llm_confidence)
        self.gate_confidence_sharpness = float(gate_confidence_sharpness)
        self.gate_density_index = int(gate_density_index)
        self.gate_density_center = float(gate_density_center)
        self.gate_density_sharpness = float(gate_density_sharpness)
        self.gate_adaptive_strength = float(gate_adaptive_strength)
        self.llm_semantic_encoder = str(llm_semantic_encoder or "flat").lower()
        if self.llm_semantic_encoder not in {"flat", "branch_structured", "grouped_selector", "product_optional_local", "provenance_dual_branch", "pair_compatibility", "native_pair", "native_pair_v2", "pair_first_local", "semantic_expert_moe"}:
            raise ValueError(f"Unknown llm_semantic_encoder: {llm_semantic_encoder}")
        self.llm_branch_index_map = {
            name: tuple(int(index) for index in (llm_branch_index_map or {}).get(name, ()))
            for name in LLM_SEMANTIC_BRANCH_ORDER
        }
        self.llm_branch_column_map = {
            name: tuple(str(column) for column in (llm_branch_column_map or {}).get(name, ()))
            for name in LLM_SEMANTIC_BRANCH_ORDER
        }
        self.llm_group_index_map = {
            name: tuple(int(index) for index in (llm_group_index_map or {}).get(name, ()))
            for name in LLM_SEMANTIC_GROUP_ORDER
        }
        self.llm_group_column_map = {
            name: tuple(str(column) for column in (llm_group_column_map or {}).get(name, ()))
            for name in LLM_SEMANTIC_GROUP_ORDER
        }
        self.llm_group_top_k = max(0, int(llm_group_top_k))
        self.local_meta_gate_features = self._normalize_local_meta_gate_features(llm_local_meta_gate)
        self.local_meta_gate_enabled = bool(self.local_meta_gate_features)
        if llm_gate_product_focus_scale < 0.0:
            raise ValueError("llm_gate_product_focus_scale must be non-negative")
        self.gate_product_focus_scale = float(llm_gate_product_focus_scale)
        self.llm_fusion_mode = str(llm_fusion_mode).lower()
        self.physics_fusion = str(physics_fusion).lower()
        self.reaction_center_coupling_enabled = bool(reaction_center_coupling and reaction_center_pair_feature_dim > 0)
        self.reaction_center_pair_feature_dim = int(reaction_center_pair_feature_dim)
        self.kinetic_relay_enabled = bool(kinetic_relay and self.reaction_center_coupling_enabled)
        self.kinetic_interaction_enabled = self.qc_dim > 0
        if self.llm_fusion_mode not in {"residual", "conditioned", "cross_attention", "moe"}:
            raise ValueError(f"Unknown llm_fusion_mode: {llm_fusion_mode}")
        if self.qc_dim > 0 and self.physics_fusion not in {"concat", "add", "gated"}:
            raise ValueError(f"Unknown physics fusion strategy: {physics_fusion}")
        self.llm_cross_attention_heads = int(llm_cross_attention_heads)
        if self.qc_dim > 0:
            self.cat_encoder = PhysicsInjectedPaiNNEncoder(
                hidden_dim,
                num_layers,
                num_rbf,
                cutoff,
                qc_dim=self.qc_dim,
                dropout=dropout,
                node_qc_dim=node_qc_dim,
                node_qc_gate_dim=node_qc_gate_dim,
            )
            self.reactant_encoder = PhysicsInjectedPaiNNEncoder(
                hidden_dim,
                num_layers,
                num_rbf,
                cutoff,
                qc_dim=self.qc_dim,
                dropout=dropout,
                node_qc_dim=node_qc_dim,
                node_qc_gate_dim=node_qc_gate_dim,
            )
        else:
            self.cat_encoder = PaiNNEncoder(
                hidden_dim,
                num_layers,
                num_rbf,
                cutoff,
                dropout=dropout,
                node_qc_dim=node_qc_dim,
                node_qc_gate_dim=node_qc_gate_dim,
            )
            self.reactant_encoder = PaiNNEncoder(
                hidden_dim,
                num_layers,
                num_rbf,
                cutoff,
                dropout=dropout,
                node_qc_dim=node_qc_dim,
                node_qc_gate_dim=node_qc_gate_dim,
            )
        self.global_attention = (
            GlobalTokenTransformer(
                hidden_dim=hidden_dim,
                num_heads=global_attention_heads,
                num_layers=global_attention_layers,
                dropout=global_attention_dropout,
            )
            if global_attention
            else None
        )
        interaction_heads = max(1, math.gcd(hidden_dim, max(1, int(global_attention_heads))))
        self.interaction_cross_attention = (
            CatalystSubstrateShellCrossAttention(
                hidden_dim=hidden_dim,
                num_heads=interaction_heads,
                dropout=dropout,
            )
            if interaction_cross_attention
            else None
        )
        self.interaction_cross_attention_layers = 2
        self.interaction_cross_attention_heads = interaction_heads
        self.interaction_cross_attention_enabled = bool(interaction_cross_attention)
        self.reaction_center_coupling = (
            ReactionCenterCoupling(
                hidden_dim=hidden_dim,
                pair_feature_dim=self.reaction_center_pair_feature_dim,
                dropout=dropout,
                kinetic_relay=self.kinetic_relay_enabled,
                coordination_dim=6 if node_qc_dim >= 6 else 0,
            )
            if self.reaction_center_coupling_enabled
            else None
        )
        self.reaction_center_gate = (
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            if self.reaction_center_coupling_enabled
            else None
        )
        self.reaction_center_gate_logit = (
            nn.Parameter(torch.tensor([-2.0 if self.kinetic_relay_enabled else -3.0], dtype=torch.float32))
            if self.reaction_center_coupling_enabled
            else None
        )
        self.reaction_center_head = (
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )
            if self.reaction_center_coupling_enabled
            else None
        )
        self.reaction_center_scale = (
            nn.Parameter(torch.tensor([-1.0 if self.kinetic_relay_enabled else -2.0], dtype=torch.float32))
            if self.reaction_center_coupling_enabled
            else None
        )
        self.last_aux_outputs: Optional[Dict[str, torch.Tensor]] = None
        self.combine_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.node_qc_summary_proj = None
        if self.node_qc_dim > 0:
            self.node_qc_summary_proj = nn.Sequential(
                nn.Linear(self.node_qc_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
        self.qc_proj = None
        self.concat_fusion = None
        self.add_proj = None
        self.gated_qc_proj = None
        self.gated_fusion = None
        if self.qc_dim > 0:
            self.qc_proj = nn.Sequential(
                nn.Linear(self.qc_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            if self.physics_fusion == "concat":
                self.concat_fusion = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                )
            elif self.physics_fusion == "add":
                self.add_proj = nn.Linear(hidden_dim, hidden_dim)
            elif self.physics_fusion == "gated":
                self.gated_qc_proj = nn.Linear(hidden_dim, hidden_dim)
                self.gated_fusion = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Sigmoid(),
                )
        self.kinetic_interaction = (
            KineticInteractionLayer(hidden_dim=hidden_dim, qc_dim=self.qc_dim, dropout=dropout)
            if self.qc_dim > 0
            else None
        )
        self.feature_gate = None
        if self.feat_dim > 0:
            self.feature_gate = nn.Sequential(
                nn.Linear(hidden_dim, self.feat_dim, bias=False),
                nn.Tanh(),
            )
        head_in = hidden_dim + self.feat_dim
        self.struct_head = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.llm_proj = None
        self.llm_branch_encoder = None
        self.llm_group_selector = None
        self.native_pair_encoder = None
        self.pair_first_local_encoder = None
        self.semantic_expert_encoder = None
        self.llm_head = None
        self.gate_mlp = None
        self.gate_scale = nn.Parameter(torch.zeros(1))
        self.global_background_node = None
        self.global_background_gate = None
        self.global_background_scale = None
        self.global_background_enabled = False
        self.global_rule_names = GlobalKnowledgeNode.RULE_NAMES
        self.multi_scale_moe_enabled = False
        self.ablation_disable_global_background = False
        self.ablation_disable_conditioned = False
        self.ablation_moe_route_override = None
        self.pair_conditioned_bridge_locus = str(pair_conditioned_bridge_locus or "pre_qc_and_kinetic_fusion").strip().lower()
        self.condition_mlp = None
        self.condition_head = None
        self.condition_gate = None
        self.condition_scale = None
        self.moe_physics_head = None
        self.moe_global_head = None
        self.moe_local_head = None
        self.moe_gate = None
        self.moe_scale = None
        self.last_moe_weights: Optional[torch.Tensor] = None
        self.last_moe_priors: Optional[torch.Tensor] = None
        self.llm_cross_attention = None
        self.llm_cross_norm = None
        self.semantic_rules = AlignmentRules()
        self.semantic_component_embedding = None
        self.semantic_interaction_encoder = None
        self.semantic_interaction_proj = None
        self.semantic_query = None
        self.semantic_hotspot_scale = None
        self.semantic_bridge_scale = None
        self.semantic_physical_bridge_scale = None
        self.semantic_attention_bridge_enabled = False
        self.semantic_physical_bridge_enabled = False
        self.pair_conditioned_bridge = None
        self.pair_conditioned_bridge_enabled = False
        if self.llm_dim > 0:
            self.multi_scale_moe_enabled = self.llm_fusion_mode == "moe"
            self.semantic_physical_bridge_enabled = bool(semantic_physical_bridge)
            self.global_background_node = GlobalKnowledgeNode(
                llm_dim=self.llm_dim,
                hidden_dim=hidden_dim,
                dropout=gate_dropout,
            )
            self.global_background_gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(gate_dropout),
                nn.Linear(hidden_dim, 1),
            )
            if hasattr(self.global_background_gate[-1], "bias") and self.global_background_gate[-1].bias is not None:
                nn.init.constant_(self.global_background_gate[-1].bias, 0.5)
            self.global_background_scale = nn.Parameter(torch.tensor([-0.5], dtype=torch.float32))
            self.global_background_enabled = True
            self.llm_proj = nn.Sequential(
                nn.Linear(self.llm_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            if self.llm_semantic_encoder == "branch_structured":
                self.llm_branch_encoder = BranchStructuredSemanticEncoder(
                    hidden_dim=hidden_dim,
                    branch_index_map=self.llm_branch_index_map,
                    branch_column_map=self.llm_branch_column_map,
                    num_heads=max(1, self.llm_cross_attention_heads),
                    num_layers=1,
                    dropout=dropout,
                )
            elif self.llm_semantic_encoder == "product_optional_local":
                self.llm_branch_encoder = ProductOptionalLocalSemanticEncoder(
                    hidden_dim=hidden_dim,
                    branch_index_map=self.llm_branch_index_map,
                    branch_column_map=self.llm_branch_column_map,
                    num_heads=max(1, self.llm_cross_attention_heads),
                    num_layers=1,
                    dropout=dropout,
                )
            elif self.llm_semantic_encoder == "provenance_dual_branch":
                self.llm_branch_encoder = ProvenanceDualBranchSemanticEncoder(
                    hidden_dim=hidden_dim,
                    llm_dim=self.llm_dim,
                    num_heads=max(1, self.llm_cross_attention_heads),
                    num_layers=1,
                    dropout=dropout,
                )
            elif self.llm_semantic_encoder == "pair_compatibility":
                self.llm_branch_encoder = PairCompatibilitySemanticEncoder(
                    hidden_dim=hidden_dim,
                    branch_index_map=self.llm_branch_index_map,
                    branch_column_map=self.llm_branch_column_map,
                    num_heads=max(1, self.llm_cross_attention_heads),
                    num_layers=1,
                    dropout=dropout,
                )
            elif self.llm_semantic_encoder == "native_pair":
                self.native_pair_encoder = NativePairSemanticEncoder(
                    hidden_dim=hidden_dim,
                    summary_dim=self.native_pair_dim,
                    token_dim=self.native_pair_token_dim,
                    num_heads=max(1, self.llm_cross_attention_heads),
                    num_layers=1,
                    dropout=dropout,
                )
            elif self.llm_semantic_encoder == "native_pair_v2":
                self.native_pair_encoder = NativePairSemanticEncoderV2(
                    hidden_dim=hidden_dim,
                    summary_dim=self.native_pair_dim,
                    token_dim=self.native_pair_token_dim,
                    num_heads=max(1, self.llm_cross_attention_heads),
                    num_layers=1,
                    dropout=dropout,
                )
            elif self.llm_semantic_encoder == "pair_first_local":
                self.pair_first_local_encoder = PairFirstLocalSemanticEncoder(
                    hidden_dim=hidden_dim,
                    summary_dim=self.native_pair_dim,
                    token_dim=self.native_pair_token_dim,
                    branch_index_map=self.llm_branch_index_map,
                    branch_column_map=self.llm_branch_column_map,
                    num_heads=max(1, self.llm_cross_attention_heads),
                    num_layers=1,
                    dropout=dropout,
                )
            elif self.llm_semantic_encoder == "semantic_expert_moe":
                self.semantic_expert_encoder = SemanticExpertMixtureEncoder(
                    hidden_dim=hidden_dim,
                    summary_dim=self.native_pair_dim,
                    token_dim=self.native_pair_token_dim,
                    num_heads=max(1, self.llm_cross_attention_heads),
                    num_layers=1,
                    dropout=dropout,
                )
            elif self.llm_semantic_encoder == "grouped_selector":
                self.llm_group_selector = GroupedSemanticSelector(
                    hidden_dim=hidden_dim,
                    group_index_map=self.llm_group_index_map,
                    group_column_map=self.llm_group_column_map,
                    selection_top_k=self.llm_group_top_k,
                    num_heads=max(1, self.llm_cross_attention_heads),
                    num_layers=1,
                    dropout=dropout,
                )
            self.llm_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )
            self.gate_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(gate_dropout),
                nn.Linear(hidden_dim, 1),
            )
            if hasattr(self.gate_mlp[-1], "bias") and self.gate_mlp[-1].bias is not None:
                nn.init.constant_(self.gate_mlp[-1].bias, -1.5)
            if self.llm_fusion_mode in {"conditioned", "cross_attention", "moe"}:
                self.condition_mlp = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(gate_dropout),
                    nn.Linear(hidden_dim, hidden_dim * 2),
                )
                self.condition_head = nn.Sequential(
                    nn.Linear(head_in, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, 1),
                )
                self.condition_gate = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(gate_dropout),
                    nn.Linear(hidden_dim, 1),
                )
                if hasattr(self.condition_gate[-1], "bias") and self.condition_gate[-1].bias is not None:
                    nn.init.constant_(self.condition_gate[-1].bias, -0.5)
                init_scale = math.log(math.exp(float(llm_conditioned_scale)) - 1.0) if llm_conditioned_scale > 0.0 else -6.0
                self.condition_scale = nn.Parameter(torch.tensor([init_scale], dtype=torch.float32))
            if self.multi_scale_moe_enabled:
                def _make_moe_head() -> nn.Sequential:
                    head = nn.Sequential(
                        nn.Linear(head_in, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim, hidden_dim // 2),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim // 2, 1),
                    )
                    final_layer = head[-1]
                    if hasattr(final_layer, "weight"):
                        nn.init.zeros_(final_layer.weight)
                    if hasattr(final_layer, "bias") and final_layer.bias is not None:
                        nn.init.zeros_(final_layer.bias)
                    return head

                self.moe_physics_head = _make_moe_head()
                self.moe_global_head = _make_moe_head()
                self.moe_local_head = _make_moe_head()
                self.moe_gate = nn.Sequential(
                    nn.Linear(hidden_dim * 4 + len(self.global_rule_names) + 3, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(gate_dropout),
                    nn.Linear(hidden_dim, 3),
                )
                final_gate = self.moe_gate[-1]
                if hasattr(final_gate, "weight"):
                    nn.init.zeros_(final_gate.weight)
                if hasattr(final_gate, "bias") and final_gate.bias is not None:
                    nn.init.zeros_(final_gate.bias)
                self.moe_scale = nn.Parameter(torch.tensor([init_scale], dtype=torch.float32))
            if self.llm_fusion_mode == "cross_attention" or self.semantic_physical_bridge_enabled:
                if self.llm_cross_attention_heads <= 0:
                    raise ValueError("llm_cross_attention_heads must be positive for semantic bridge fusion.")
                if hidden_dim % self.llm_cross_attention_heads != 0:
                    raise ValueError(
                        f"hidden_dim ({hidden_dim}) must be divisible by llm_cross_attention_heads ({self.llm_cross_attention_heads})."
                    )
                self.llm_cross_attention = nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=self.llm_cross_attention_heads,
                    dropout=gate_dropout,
                    batch_first=True,
                )
                self.llm_cross_norm = nn.LayerNorm(hidden_dim)
                self.semantic_component_embedding = nn.Embedding(3, hidden_dim)
                self.semantic_interaction_encoder = InteractionFeatureEncoder()
                self.semantic_interaction_proj = nn.Sequential(
                    nn.Linear(self.semantic_interaction_encoder.feature_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(gate_dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                )
                self.semantic_query = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(gate_dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                self.semantic_hotspot_scale = nn.Parameter(torch.ones(1, dtype=torch.float32))
            if self.llm_fusion_mode == "cross_attention":
                self.semantic_bridge_scale = nn.Parameter(torch.tensor([0.75], dtype=torch.float32))
                self.semantic_attention_bridge_enabled = True
            if self.semantic_physical_bridge_enabled:
                self.semantic_physical_bridge_scale = nn.Parameter(torch.tensor([0.75], dtype=torch.float32))
            if (
                pair_conditioned_bridge
                and self.llm_semantic_encoder in {"native_pair", "native_pair_v2", "pair_first_local", "semantic_expert_moe"}
                and self.native_pair_dim > 0
                and self.native_pair_token_dim > 0
            ):
                self.pair_conditioned_bridge = PairConditionedPhysicalBridge(
                    hidden_dim=hidden_dim,
                    dropout=gate_dropout,
                    injection_locus=self.pair_conditioned_bridge_locus,
                )
                self.pair_conditioned_bridge_enabled = True

    @staticmethod
    def _normalize_local_meta_gate_features(features: Optional[Sequence[str]]) -> Tuple[str, ...]:
        if not features:
            return ()
        resolved: List[str] = []
        for raw_name in features:
            key = str(raw_name).strip()
            if not key:
                continue
            column_name = _LOCAL_META_GATE_COLUMN_BY_NAME.get(key)
            if column_name is None:
                raise ValueError(
                    f"Unknown llm_local_meta_gate feature: {raw_name}. "
                    f"Expected one of {sorted(set(_LOCAL_META_GATE_COLUMN_BY_NAME) - set(_LLM_FEATURE_INDEX))}."
                )
            if column_name not in resolved:
                resolved.append(column_name)
        return tuple(resolved)

    def _encode_llm_features(
        self,
        llm_features: torch.Tensor,
        exact_covered_indicator: Optional[torch.Tensor] = None,
        native_pair_summary: Optional[torch.Tensor] = None,
        native_pair_tokens: Optional[torch.Tensor] = None,
        native_pair_token_mask: Optional[torch.Tensor] = None,
        native_pair_expert_summaries: Optional[torch.Tensor] = None,
        native_pair_expert_tokens: Optional[torch.Tensor] = None,
        native_pair_expert_token_masks: Optional[torch.Tensor] = None,
        native_pair_expert_priors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.llm_semantic_encoder in {"native_pair", "native_pair_v2"} and self.native_pair_encoder is not None:
            if native_pair_summary is None or native_pair_tokens is None:
                raise ValueError(f"{self.llm_semantic_encoder} encoder requires native_pair_summary and native_pair_tokens.")
            return self.native_pair_encoder(
                native_pair_summary,
                native_pair_tokens,
                native_pair_token_mask,
            )
        if self.llm_semantic_encoder == "pair_first_local" and self.pair_first_local_encoder is not None:
            if native_pair_summary is None or native_pair_tokens is None:
                raise ValueError("pair_first_local encoder requires native_pair_summary and native_pair_tokens.")
            return self.pair_first_local_encoder(
                llm_features,
                native_pair_summary,
                native_pair_tokens,
                native_pair_token_mask,
            )
        if self.llm_semantic_encoder == "semantic_expert_moe" and self.semantic_expert_encoder is not None:
            if native_pair_expert_summaries is None or native_pair_expert_tokens is None:
                raise ValueError(
                    "semantic_expert_moe encoder requires native_pair_expert_summaries and native_pair_expert_tokens."
                )
            return self.semantic_expert_encoder(
                native_pair_expert_summaries,
                native_pair_expert_tokens,
                native_pair_expert_token_masks,
                native_pair_expert_priors,
            )
        if self.llm_semantic_encoder == "provenance_dual_branch" and self.llm_branch_encoder is not None:
            return self.llm_branch_encoder(llm_features, exact_covered_indicator=exact_covered_indicator)
        if self.llm_semantic_encoder in {"branch_structured", "product_optional_local", "pair_compatibility"} and self.llm_branch_encoder is not None:
            return self.llm_branch_encoder(llm_features)
        if self.llm_semantic_encoder == "grouped_selector" and self.llm_group_selector is not None:
            return self.llm_group_selector(llm_features)
        if self.llm_proj is None:
            raise ValueError("llm_proj is not initialized")
        return self.llm_proj(llm_features)

    def encode_global_background(
        self,
        llm_features: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if (
            self.global_background_node is None
            or llm_features is None
            or llm_features.ndim != 2
            or llm_features.numel() == 0
        ):
            return None, None
        return self.global_background_node(llm_features)

    def _adaptive_gate_prior(
        self,
        llm_features: torch.Tensor,
        llm_confidence: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if llm_confidence is not None and llm_confidence.numel() > 0:
            confidence = llm_confidence.view(-1, 1).clamp(0.0, 1.0)
        else:
            confidence = torch.ones((llm_features.size(0), 1), dtype=llm_features.dtype, device=llm_features.device)
        confidence_weight = torch.sigmoid(
            (confidence - self.min_llm_confidence) * self.gate_confidence_sharpness
        )
        local_meta_weight = None
        if self.local_meta_gate_features:
            selected = {
                name: _llm_feature_column(llm_features, name).clamp(0.0, 1.0)
                for name in self.local_meta_gate_features
            }
            weighted_terms: List[Tuple[float, torch.Tensor]] = []
            confidence_like = [selected[name] for name in ("llm_overall_confidence", "llm_semantic_temperature_alignment") if name in selected]
            if confidence_like:
                weighted_terms.append((0.40, sum(confidence_like) / len(confidence_like)))
            if "llm_semantic_hotspot_density" in selected:
                weighted_terms.append((0.35, selected["llm_semantic_hotspot_density"]))
            focus_terms = [
                (
                    selected[name] * self.gate_product_focus_scale
                    if name == "llm_semantic_r2_focus"
                    else selected[name]
                )
                for name in ("llm_semantic_cat_focus", "llm_semantic_r1_focus", "llm_semantic_r2_focus")
                if name in selected
            ]
            if focus_terms:
                focus_signal = focus_terms[0]
                for term in focus_terms[1:]:
                    focus_signal = torch.maximum(focus_signal, term)
                weighted_terms.append((0.25, focus_signal))
            if weighted_terms:
                total_weight = sum(weight for weight, _ in weighted_terms)
                local_meta_score = sum(weight * term for weight, term in weighted_terms) / max(total_weight, 1e-6)
                local_meta_weight = torch.sigmoid(
                    (local_meta_score - self.gate_density_center) * self.gate_density_sharpness
                )
        if local_meta_weight is None:
            if 0 <= self.gate_density_index < llm_features.size(1):
                density = llm_features[:, self.gate_density_index : self.gate_density_index + 1]
                local_meta_weight = torch.sigmoid((density - self.gate_density_center) * self.gate_density_sharpness)
            else:
                local_meta_weight = torch.ones_like(confidence_weight)
        adaptive_prior = 0.5 * (confidence_weight + local_meta_weight)
        return confidence_weight, local_meta_weight, adaptive_prior

    def _semantic_guided_geometry_context(
        self,
        llm_emb: torch.Tensor,
        cat_nodes: torch.Tensor,
        cat_batch: torch.Tensor,
        r1_nodes: torch.Tensor,
        r1_batch: torch.Tensor,
        r2_nodes: torch.Tensor,
        r2_batch: torch.Tensor,
        cat_interactions: Optional[Sequence[Optional[Sequence[object]]]] = None,
        r1_interactions: Optional[Sequence[Optional[Sequence[object]]]] = None,
        r2_interactions: Optional[Sequence[Optional[Sequence[object]]]] = None,
    ) -> torch.Tensor:
        if (
            self.llm_cross_attention is None
            or self.llm_cross_norm is None
            or self.semantic_component_embedding is None
            or self.semantic_query is None
        ):
            return llm_emb
        num_graphs = llm_emb.size(0)
        if num_graphs == 0:
            return llm_emb
        component_specs = (
            (cat_nodes, cat_batch, cat_interactions, 0),
            (r1_nodes, r1_batch, r1_interactions, 1),
            (r2_nodes, r2_batch, r2_interactions, 2),
        )
        hotspot_scale = F.softplus(self.semantic_hotspot_scale) if self.semantic_hotspot_scale is not None else 0.0
        token_sequences: List[torch.Tensor] = []
        semantic_summaries: List[torch.Tensor] = []
        for graph_idx in range(num_graphs):
            graph_tokens: List[torch.Tensor] = []
            graph_semantics: List[torch.Tensor] = []
            for nodes, batch, interactions, component_idx in component_specs:
                node_mask = batch == graph_idx
                node_tokens = nodes[node_mask]
                if node_tokens.numel() == 0:
                    continue
                component_embed = self.semantic_component_embedding.weight[component_idx].unsqueeze(0)
                component_tokens = node_tokens + component_embed
                component_interactions = interactions[graph_idx] if interactions and graph_idx < len(interactions) else None
                hotspot_scores = build_hotspot_scores(
                    component_interactions or [],
                    component_tokens.size(0),
                    rules=self.semantic_rules,
                    device=component_tokens.device,
                    aggregation="max",
                )
                hotspot_boost = torch.clamp(hotspot_scores - 1.0, min=0.0).unsqueeze(-1)
                boosted_tokens = component_tokens * (1.0 + hotspot_scale * hotspot_boost)
                graph_tokens.append(boosted_tokens)
                hotspot_weights = hotspot_scores / hotspot_scores.sum().clamp(min=1e-6)
                hotspot_summary = (boosted_tokens * hotspot_weights.unsqueeze(-1)).sum(dim=0)
                interaction_summary = torch.zeros_like(hotspot_summary)
                if component_interactions and self.semantic_interaction_encoder is not None and self.semantic_interaction_proj is not None:
                    _, pair_features = build_interaction_pair_features(
                        component_interactions,
                        component_tokens.size(0),
                        self.semantic_interaction_encoder,
                        component_tokens.device,
                    )
                    if pair_features.numel() > 0:
                        interaction_summary = self.semantic_interaction_proj(pair_features).mean(dim=0)
                graph_semantics.append(hotspot_summary + interaction_summary + component_embed.squeeze(0))
            if graph_tokens:
                token_sequences.append(torch.cat(graph_tokens, dim=0))
                semantic_summaries.append(torch.stack(graph_semantics, dim=0).mean(dim=0))
            else:
                token_sequences.append(llm_emb.new_zeros((1, llm_emb.size(-1))))
                semantic_summaries.append(llm_emb.new_zeros((llm_emb.size(-1),)))
        max_tokens = max(tokens.size(0) for tokens in token_sequences)
        padded_tokens = llm_emb.new_zeros((num_graphs, max_tokens, llm_emb.size(-1)))
        key_padding_mask = torch.ones((num_graphs, max_tokens), dtype=torch.bool, device=llm_emb.device)
        for graph_idx, tokens in enumerate(token_sequences):
            padded_tokens[graph_idx, : tokens.size(0)] = tokens
            key_padding_mask[graph_idx, : tokens.size(0)] = False
        semantic_summary = torch.stack(semantic_summaries, dim=0)
        query = self.semantic_query(torch.cat([llm_emb, semantic_summary], dim=-1)).unsqueeze(1)
        attn_out, _ = self.llm_cross_attention(
            query,
            padded_tokens,
            padded_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return self.llm_cross_norm(query + attn_out + semantic_summary.unsqueeze(1)).squeeze(1)

    def _pair_bridge_context(
        self,
        native_pair_summary: Optional[torch.Tensor],
        native_pair_tokens: Optional[torch.Tensor],
        native_pair_token_mask: Optional[torch.Tensor],
        native_pair_expert_summaries: Optional[torch.Tensor] = None,
        native_pair_expert_tokens: Optional[torch.Tensor] = None,
        native_pair_expert_token_masks: Optional[torch.Tensor] = None,
        native_pair_expert_priors: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if (
            not self.pair_conditioned_bridge_enabled
            or (
                self.llm_semantic_encoder != "semantic_expert_moe"
                and (
                    native_pair_summary is None
                    or native_pair_tokens is None
                    or native_pair_summary.numel() == 0
                    or native_pair_tokens.numel() == 0
                )
            )
            or (
                self.llm_semantic_encoder == "semantic_expert_moe"
                and (
                    native_pair_expert_summaries is None
                    or native_pair_expert_tokens is None
                    or native_pair_expert_summaries.numel() == 0
                    or native_pair_expert_tokens.numel() == 0
                )
            )
        ):
            return None
        if self.llm_semantic_encoder == "pair_first_local" and self.pair_first_local_encoder is not None:
            cached = self.pair_first_local_encoder.last_pair_context
            if cached is not None and cached.size(0) == native_pair_summary.size(0):
                self.pair_first_local_encoder.last_pair_context = None
                return cached
            return self.pair_first_local_encoder.pair_encoder(
                native_pair_summary,
                native_pair_tokens,
                native_pair_token_mask,
            )
        if self.llm_semantic_encoder in {"native_pair", "native_pair_v2"} and self.native_pair_encoder is not None:
            cached = self.native_pair_encoder.last_pooled
            if cached is not None and cached.size(0) == native_pair_summary.size(0):
                self.native_pair_encoder.last_pooled = None
                return cached
            return self.native_pair_encoder(
                native_pair_summary,
                native_pair_tokens,
                native_pair_token_mask,
            )
        if self.llm_semantic_encoder == "semantic_expert_moe" and self.semantic_expert_encoder is not None:
            cached = self.semantic_expert_encoder.last_pooled
            if cached is not None and cached.size(0) == native_pair_expert_summaries.size(0):
                self.semantic_expert_encoder.last_pooled = None
                return cached
            return self.semantic_expert_encoder(
                native_pair_expert_summaries,
                native_pair_expert_tokens,
                native_pair_expert_token_masks,
                native_pair_expert_priors,
            )
        return None

    def _fuse_qc(
        self,
        struct_emb: torch.Tensor,
        qc_features: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.qc_proj is None or qc_features is None or qc_features.numel() == 0:
            return struct_emb
        qc_emb = self.qc_proj(qc_features)
        if self.physics_fusion == "concat":
            if self.concat_fusion is None:
                return struct_emb
            return self.concat_fusion(torch.cat([struct_emb, qc_emb], dim=-1))
        if self.physics_fusion == "add":
            if self.add_proj is None:
                return struct_emb
            return struct_emb + self.add_proj(qc_emb)
        if self.gated_qc_proj is None or self.gated_fusion is None:
            return struct_emb
        qc_term = self.gated_qc_proj(qc_emb)
        gate = self.gated_fusion(torch.cat([struct_emb, qc_emb], dim=-1))
        return gate * struct_emb + (1.0 - gate) * qc_term

    def _apply_kinetics(
        self,
        struct_emb: torch.Tensor,
        qc_features: Optional[torch.Tensor],
        temperature: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.kinetic_interaction is None:
            return struct_emb
        return self.kinetic_interaction(struct_emb, qc_features=qc_features, temperature=temperature)

    def _pool_node_qc(
        self,
        node_qc: Optional[torch.Tensor],
        batch: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if (
            self.node_qc_summary_proj is None
            or node_qc is None
            or node_qc.numel() == 0
            or batch.numel() == 0
        ):
            return None
        num_graphs = int(batch.max().item()) + 1
        pooled_sum = torch.zeros((num_graphs, node_qc.size(-1)), device=node_qc.device, dtype=node_qc.dtype)
        pooled_sum.index_add_(0, batch, node_qc)
        counts = torch.bincount(batch, minlength=num_graphs).clamp(min=1).unsqueeze(-1)
        pooled_mean = pooled_sum / counts
        pooled_max = torch.full_like(pooled_mean, -float("inf"))
        for idx in range(num_graphs):
            values = node_qc[batch == idx]
            pooled_max[idx] = values.max(dim=0).values if values.numel() > 0 else 0.0
        return self.node_qc_summary_proj(torch.cat([pooled_mean, pooled_max], dim=-1))

    def _apply_interaction_cross_attention(
        self,
        cat_nodes: torch.Tensor,
        cat_batch: torch.Tensor,
        r1_nodes: torch.Tensor,
        r1_batch: torch.Tensor,
        r2_nodes: torch.Tensor,
        r2_batch: torch.Tensor,
        interaction_cat_r1_index: Optional[torch.Tensor] = None,
        interaction_cat_r1_features: Optional[torch.Tensor] = None,
        interaction_cat_r1_batch: Optional[torch.Tensor] = None,
        interaction_cat_r2_index: Optional[torch.Tensor] = None,
        interaction_cat_r2_features: Optional[torch.Tensor] = None,
        interaction_cat_r2_batch: Optional[torch.Tensor] = None,
        layer_tag: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.interaction_cross_attention_enabled:
            return (
                cat_nodes,
                self.cat_encoder._pool(cat_nodes, cat_batch),
                r1_nodes,
                self.reactant_encoder._pool(r1_nodes, r1_batch),
                r2_nodes,
                self.reactant_encoder._pool(r2_nodes, r2_batch),
            )
        cat_nodes, r1_nodes, r2_nodes = self.interaction_cross_attention(
            cat_nodes,
            cat_batch,
            r1_nodes,
            r1_batch,
            r2_nodes,
            r2_batch,
            interaction_cat_r1_index=interaction_cat_r1_index,
            interaction_cat_r1_features=interaction_cat_r1_features,
            interaction_cat_r1_batch=interaction_cat_r1_batch,
            interaction_cat_r2_index=interaction_cat_r2_index,
            interaction_cat_r2_features=interaction_cat_r2_features,
            interaction_cat_r2_batch=interaction_cat_r2_batch,
            layer_tag=layer_tag,
        )
        return (
            cat_nodes,
            self.cat_encoder._pool(cat_nodes, cat_batch),
            r1_nodes,
            self.reactant_encoder._pool(r1_nodes, r1_batch),
            r2_nodes,
            self.reactant_encoder._pool(r2_nodes, r2_batch),
        )

    def _encode_with_atomic_interaction(
        self,
        cat_z: torch.Tensor,
        cat_pos: torch.Tensor,
        cat_batch: torch.Tensor,
        cat_edge_index: torch.Tensor,
        r1_z: torch.Tensor,
        r1_pos: torch.Tensor,
        r1_batch: torch.Tensor,
        r1_edge_index: torch.Tensor,
        r2_z: torch.Tensor,
        r2_pos: torch.Tensor,
        r2_batch: torch.Tensor,
        r2_edge_index: torch.Tensor,
        qc_features: Optional[torch.Tensor] = None,
        cat_node_qc: Optional[torch.Tensor] = None,
        r1_node_qc: Optional[torch.Tensor] = None,
        r2_node_qc: Optional[torch.Tensor] = None,
        interaction_cat_r1_index: Optional[torch.Tensor] = None,
        interaction_cat_r1_features: Optional[torch.Tensor] = None,
        interaction_cat_r1_batch: Optional[torch.Tensor] = None,
        interaction_cat_r2_index: Optional[torch.Tensor] = None,
        interaction_cat_r2_features: Optional[torch.Tensor] = None,
        interaction_cat_r2_batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.qc_dim > 0:
            cat_state = self.cat_encoder._init_state(
                cat_z,
                cat_batch,
                cat_edge_index,
                qc_features=qc_features,
                node_qc=cat_node_qc,
            )
            r1_state = self.reactant_encoder._init_state(
                r1_z,
                r1_batch,
                r1_edge_index,
                qc_features=qc_features,
                node_qc=r1_node_qc,
            )
            r2_state = self.reactant_encoder._init_state(
                r2_z,
                r2_batch,
                r2_edge_index,
                qc_features=qc_features,
                node_qc=r2_node_qc,
            )
        else:
            cat_state = self.cat_encoder._init_state(cat_z, cat_edge_index, node_qc=cat_node_qc)
            r1_state = self.reactant_encoder._init_state(r1_z, r1_edge_index, node_qc=r1_node_qc)
            r2_state = self.reactant_encoder._init_state(r2_z, r2_edge_index, node_qc=r2_node_qc)
        if self.interaction_cross_attention is not None:
            self.interaction_cross_attention.last_layer_tags = []
        interaction_start = max(0, int(self.cat_encoder.num_layers) - int(self.interaction_cross_attention_layers))
        for layer_idx in range(int(self.cat_encoder.num_layers)):
            cat_state = self.cat_encoder._run_layer(cat_state, cat_pos, cat_edge_index, layer_idx)
            r1_state = self.reactant_encoder._run_layer(r1_state, r1_pos, r1_edge_index, layer_idx)
            r2_state = self.reactant_encoder._run_layer(r2_state, r2_pos, r2_edge_index, layer_idx)
            if self.interaction_cross_attention is not None and layer_idx >= interaction_start:
                cat_state["s"], r1_state["s"], r2_state["s"] = self.interaction_cross_attention(
                    cat_state["s"],
                    cat_batch,
                    r1_state["s"],
                    r1_batch,
                    r2_state["s"],
                    r2_batch,
                    interaction_cat_r1_index=interaction_cat_r1_index,
                    interaction_cat_r1_features=interaction_cat_r1_features,
                    interaction_cat_r1_batch=interaction_cat_r1_batch,
                    interaction_cat_r2_index=interaction_cat_r2_index,
                    interaction_cat_r2_features=interaction_cat_r2_features,
                    interaction_cat_r2_batch=interaction_cat_r2_batch,
                    layer_tag=layer_idx,
                )
        cat_node_tokens, cat_local = self.cat_encoder._finalize_state(cat_state, cat_batch)
        r1_node_tokens, r1_local = self.reactant_encoder._finalize_state(r1_state, r1_batch)
        r2_node_tokens, r2_local = self.reactant_encoder._finalize_state(r2_state, r2_batch)
        return (
            cat_node_tokens,
            cat_local,
            r1_node_tokens,
            r1_local,
            r2_node_tokens,
            r2_local,
        )

    def forward(
        self,
        cat_z: torch.Tensor,
        cat_pos: torch.Tensor,
        cat_batch: torch.Tensor,
        cat_edge_index: torch.Tensor,
        r1_z: torch.Tensor,
        r1_pos: torch.Tensor,
        r1_batch: torch.Tensor,
        r1_edge_index: torch.Tensor,
        r2_z: torch.Tensor,
        r2_pos: torch.Tensor,
        r2_batch: torch.Tensor,
        r2_edge_index: torch.Tensor,
        features: torch.Tensor,
        temperature: Optional[torch.Tensor] = None,
        qc_features: Optional[torch.Tensor] = None,
        cat_node_qc: Optional[torch.Tensor] = None,
        r1_node_qc: Optional[torch.Tensor] = None,
        r2_node_qc: Optional[torch.Tensor] = None,
        llm_features: Optional[torch.Tensor] = None,
        llm_confidence: Optional[torch.Tensor] = None,
        exact_covered_indicator: Optional[torch.Tensor] = None,
        native_pair_summary: Optional[torch.Tensor] = None,
        native_pair_tokens: Optional[torch.Tensor] = None,
        native_pair_token_mask: Optional[torch.Tensor] = None,
        native_pair_expert_summaries: Optional[torch.Tensor] = None,
        native_pair_expert_tokens: Optional[torch.Tensor] = None,
        native_pair_expert_token_masks: Optional[torch.Tensor] = None,
        native_pair_expert_priors: Optional[torch.Tensor] = None,
        cat_interactions: Optional[Sequence[Optional[Sequence[object]]]] = None,
        r1_interactions: Optional[Sequence[Optional[Sequence[object]]]] = None,
        r2_interactions: Optional[Sequence[Optional[Sequence[object]]]] = None,
        interaction_cat_r1_index: Optional[torch.Tensor] = None,
        interaction_cat_r1_features: Optional[torch.Tensor] = None,
        interaction_cat_r1_batch: Optional[torch.Tensor] = None,
        interaction_cat_r2_index: Optional[torch.Tensor] = None,
        interaction_cat_r2_features: Optional[torch.Tensor] = None,
        interaction_cat_r2_batch: Optional[torch.Tensor] = None,
        reaction_center_cat_r1_index: Optional[torch.Tensor] = None,
        reaction_center_cat_r1_features: Optional[torch.Tensor] = None,
        reaction_center_cat_r1_batch: Optional[torch.Tensor] = None,
        reaction_center_cat_r2_index: Optional[torch.Tensor] = None,
        reaction_center_cat_r2_features: Optional[torch.Tensor] = None,
        reaction_center_cat_r2_batch: Optional[torch.Tensor] = None,
        llm_scale: float = 1.0,
    ) -> torch.Tensor:
        self.last_aux_outputs = None
        if self.native_pair_encoder is not None:
            self.native_pair_encoder.last_pooled = None
        if self.pair_first_local_encoder is not None:
            self.pair_first_local_encoder.last_pair_context = None
        if self.semantic_expert_encoder is not None:
            self.semantic_expert_encoder.last_pooled = None
        (
            cat_node_tokens,
            cat_local,
            r1_node_tokens,
            r1_local,
            r2_node_tokens,
            r2_local,
        ) = self._encode_with_atomic_interaction(
            cat_z,
            cat_pos,
            cat_batch,
            cat_edge_index,
            r1_z,
            r1_pos,
            r1_batch,
            r1_edge_index,
            r2_z,
            r2_pos,
            r2_batch,
            r2_edge_index,
            qc_features=qc_features,
            cat_node_qc=cat_node_qc,
            r1_node_qc=r1_node_qc,
            r2_node_qc=r2_node_qc,
            interaction_cat_r1_index=interaction_cat_r1_index,
            interaction_cat_r1_features=interaction_cat_r1_features,
            interaction_cat_r1_batch=interaction_cat_r1_batch,
            interaction_cat_r2_index=interaction_cat_r2_index,
            interaction_cat_r2_features=interaction_cat_r2_features,
            interaction_cat_r2_batch=interaction_cat_r2_batch,
        )
        reaction_center_context = None
        if self.reaction_center_coupling is not None:
            (
                cat_node_tokens,
                r1_node_tokens,
                r2_node_tokens,
                reaction_center_context,
                reaction_center_aux,
            ) = self.reaction_center_coupling(
                cat_node_tokens,
                cat_batch,
                r1_node_tokens,
                r1_batch,
                r2_node_tokens,
                r2_batch,
                reaction_center_cat_r1_index=reaction_center_cat_r1_index,
                reaction_center_cat_r1_features=reaction_center_cat_r1_features,
                reaction_center_cat_r1_batch=reaction_center_cat_r1_batch,
                reaction_center_cat_r2_index=reaction_center_cat_r2_index,
                reaction_center_cat_r2_features=reaction_center_cat_r2_features,
                reaction_center_cat_r2_batch=reaction_center_cat_r2_batch,
                temperature=temperature,
                cat_node_qc=cat_node_qc,
                r1_node_qc=r1_node_qc,
                r2_node_qc=r2_node_qc,
            )
            cat_local = self.cat_encoder._pool(cat_node_tokens, cat_batch)
            r1_local = self.reactant_encoder._pool(r1_node_tokens, r1_batch)
            r2_local = self.reactant_encoder._pool(r2_node_tokens, r2_batch)
            self.last_aux_outputs = {"reaction_center_aux": reaction_center_aux}
            if getattr(self.reaction_center_coupling, "last_kinetic_relay_stats", None) is not None:
                self.last_aux_outputs["kinetic_relay"] = self.reaction_center_coupling.last_kinetic_relay_stats
        cat_node_summary = self._pool_node_qc(cat_node_qc, cat_batch)
        r1_node_summary = self._pool_node_qc(r1_node_qc, r1_batch)
        r2_node_summary = self._pool_node_qc(r2_node_qc, r2_batch)
        if cat_node_summary is not None:
            cat_local = cat_local + cat_node_summary
        if r1_node_summary is not None:
            r1_local = r1_local + r1_node_summary
        if r2_node_summary is not None:
            r2_local = r2_local + r2_node_summary
        cat_tokens = cat_local
        r1_tokens = r1_local
        r2_tokens = r2_local
        llm_emb = None
        llm_available = (
            self.llm_dim > 0
            and (
                self.llm_proj is not None
                or self.llm_branch_encoder is not None
                or self.llm_group_selector is not None
                or self.native_pair_encoder is not None
                or self.pair_first_local_encoder is not None
                or self.semantic_expert_encoder is not None
            )
            and self.llm_head is not None
            and self.gate_mlp is not None
            and llm_features is not None
            and llm_features.numel() > 0
            and bool(torch.any(llm_features.abs() > 0).item())
            and (
                self.llm_semantic_encoder not in {"native_pair", "native_pair_v2", "pair_first_local", "semantic_expert_moe"}
                or (
                    (
                        self.llm_semantic_encoder in {"native_pair", "native_pair_v2", "pair_first_local"}
                        and native_pair_summary is not None
                        and native_pair_summary.numel() > 0
                        and native_pair_tokens is not None
                        and native_pair_tokens.numel() > 0
                    )
                    or (
                        self.llm_semantic_encoder == "semantic_expert_moe"
                        and native_pair_expert_summaries is not None
                        and native_pair_expert_summaries.numel() > 0
                        and native_pair_expert_tokens is not None
                        and native_pair_expert_tokens.numel() > 0
                    )
                )
            )
        )
        if llm_available:
            llm_emb = self._encode_llm_features(
                llm_features,
                exact_covered_indicator=exact_covered_indicator,
                native_pair_summary=native_pair_summary,
                native_pair_tokens=native_pair_tokens,
                native_pair_token_mask=native_pair_token_mask,
                native_pair_expert_summaries=native_pair_expert_summaries,
                native_pair_expert_tokens=native_pair_expert_tokens,
                native_pair_expert_token_masks=native_pair_expert_token_masks,
                native_pair_expert_priors=native_pair_expert_priors,
            )

        local_combined = self.combine_mlp(torch.cat([cat_local, r1_local, r2_local], dim=-1))
        if (
            reaction_center_context is not None
            and self.reaction_center_gate is not None
            and self.reaction_center_gate_logit is not None
        ):
            reaction_delta = self.reaction_center_gate(torch.cat([local_combined, reaction_center_context], dim=-1))
            local_combined = local_combined + torch.sigmoid(self.reaction_center_gate_logit) * reaction_delta
        if self.pair_conditioned_bridge is not None:
            pair_bridge_context = self._pair_bridge_context(
                native_pair_summary,
                native_pair_tokens,
                native_pair_token_mask,
                native_pair_expert_summaries=native_pair_expert_summaries,
                native_pair_expert_tokens=native_pair_expert_tokens,
                native_pair_expert_token_masks=native_pair_expert_token_masks,
                native_pair_expert_priors=native_pair_expert_priors,
            )
            if (
                pair_bridge_context is not None
                and self.pair_conditioned_bridge.injection_locus == "pre_qc_and_kinetic_fusion"
            ):
                local_combined = self.pair_conditioned_bridge(
                    local_combined,
                    pair_bridge_context,
                    native_pair_summary,
                    pair_token_mask=native_pair_token_mask,
                )
        semantic_physical_context = None
        if llm_emb is not None and self.semantic_physical_bridge_enabled:
            semantic_physical_context = self._semantic_guided_geometry_context(
                llm_emb,
                cat_nodes=cat_node_tokens,
                cat_batch=cat_batch,
                r1_nodes=r1_node_tokens,
                r1_batch=r1_batch,
                r2_nodes=r2_node_tokens,
                r2_batch=r2_batch,
                cat_interactions=cat_interactions,
                r1_interactions=r1_interactions,
                r2_interactions=r2_interactions,
            )
            physical_bridge_strength = (
                F.softplus(self.semantic_physical_bridge_scale)
                if self.semantic_physical_bridge_scale is not None
                else 1.0
            )
            local_combined = local_combined + physical_bridge_strength * semantic_physical_context
        local_fused = self._fuse_qc(local_combined, qc_features)
        if (
            self.pair_conditioned_bridge is not None
            and pair_bridge_context is not None
            and self.pair_conditioned_bridge.injection_locus == "post_qc_pre_kinetic"
        ):
            local_fused = self.pair_conditioned_bridge(
                local_fused,
                pair_bridge_context,
                native_pair_summary,
                pair_token_mask=native_pair_token_mask,
            )
        local_fused = self._apply_kinetics(local_fused, qc_features=qc_features, temperature=temperature)
        if self.feature_gate is not None and features.numel() > 0:
            local_gate = self.feature_gate(local_fused)
            local_features = features * local_gate
        else:
            local_features = features
        local_pred = self.struct_head(torch.cat([local_fused, local_features], dim=-1))
        if reaction_center_context is not None and self.reaction_center_head is not None and self.reaction_center_scale is not None:
            local_pred = local_pred + F.softplus(self.reaction_center_scale) * self.reaction_center_head(reaction_center_context)

        base_pred = local_pred
        gate_context = local_fused
        global_fused = local_fused
        global_features = local_features
        if self.global_attention is not None:
            cat_global, r1_global, r2_global = self.global_attention(cat_local, r1_local, r2_local)
            cat_tokens = cat_global
            r1_tokens = r1_global
            r2_tokens = r2_global
            global_combined = self.combine_mlp(torch.cat([cat_global, r1_global, r2_global], dim=-1))
            if semantic_physical_context is not None and self.semantic_physical_bridge_scale is not None:
                global_combined = global_combined + F.softplus(self.semantic_physical_bridge_scale) * semantic_physical_context
            global_fused = self._fuse_qc(global_combined, qc_features)
            global_fused = self._apply_kinetics(global_fused, qc_features=qc_features, temperature=temperature)
            if self.feature_gate is not None and features.numel() > 0:
                global_gate = self.feature_gate(global_fused)
                global_features = features * global_gate
            else:
                global_features = features
            base_pred = local_pred + self.struct_head(torch.cat([global_fused, global_features], dim=-1))
            gate_context = global_fused

        if llm_emb is None:
            self.last_aux_outputs = {
                **(self.last_aux_outputs or {}),
                "base_pred": base_pred,
            }
            return base_pred
        background_emb, rule_scores = self.encode_global_background(llm_features)
        background_scale = 1.0
        augmented_gate_context = gate_context
        if (
            not self.ablation_disable_global_background
            and background_emb is not None
            and self.global_background_gate is not None
            and self.global_background_scale is not None
        ):
            background_weight = torch.sigmoid(
                self.global_background_gate(torch.cat([gate_context, background_emb], dim=-1))
            )
            background_scale = F.softplus(self.global_background_scale)
            background_delta = background_scale * background_weight * background_emb
            llm_emb = llm_emb + background_delta
            augmented_gate_context = gate_context + background_delta
        residual = self.llm_head(llm_emb)
        gate_input = torch.cat([augmented_gate_context, llm_emb], dim=-1)
        gate_logits = self.gate_mlp(gate_input)
        confidence_weight, density_weight, adaptive_prior = self._adaptive_gate_prior(
            llm_features,
            llm_confidence,
        )
        gate = torch.sigmoid(gate_logits + self.gate_adaptive_strength * (adaptive_prior - 0.5))
        gate = gate * torch.sigmoid(self.gate_scale)
        if llm_scale != 1.0:
            residual = residual * llm_scale
        pred = base_pred + gate * residual
        conditioned_disabled = bool(self.ablation_disable_conditioned)
        if (
            self.llm_fusion_mode == "residual"
            or conditioned_disabled
            or self.condition_mlp is None
            or self.condition_head is None
            or self.condition_gate is None
            or self.condition_scale is None
        ):
            conditioned_context = augmented_gate_context
            conditioned_features = features
            hybrid_pred = pred
        else:
            conditioned_context = augmented_gate_context
            if (
                self.llm_fusion_mode == "cross_attention"
                and self.llm_cross_attention is not None
                and self.llm_cross_norm is not None
            ):
                semantic_context = self._semantic_guided_geometry_context(
                    llm_emb,
                    cat_nodes=cat_node_tokens,
                    cat_batch=cat_batch,
                    r1_nodes=r1_node_tokens,
                    r1_batch=r1_batch,
                    r2_nodes=r2_node_tokens,
                    r2_batch=r2_batch,
                    cat_interactions=cat_interactions,
                    r1_interactions=r1_interactions,
                    r2_interactions=r2_interactions,
                )
                bridge_strength = F.softplus(self.semantic_bridge_scale) if self.semantic_bridge_scale is not None else 1.0
                conditioned_context = conditioned_context + bridge_strength * semantic_context
            film = self.condition_mlp(torch.cat([conditioned_context, llm_emb], dim=-1))
            gamma, beta = film.chunk(2, dim=-1)
            conditioned_context = conditioned_context * (1.0 + torch.tanh(gamma)) + beta
            if self.feature_gate is not None and features.numel() > 0:
                conditioned_gate = self.feature_gate(conditioned_context)
                conditioned_features = features * conditioned_gate
            else:
                conditioned_features = features
            conditioned_pred = self.condition_head(torch.cat([conditioned_context, conditioned_features], dim=-1))
            conditioned_gate = torch.sigmoid(
                self.condition_gate(torch.cat([conditioned_context, llm_emb], dim=-1))
                + self.gate_adaptive_strength * (adaptive_prior - 0.5)
            )
            conditioned_strength = F.softplus(self.condition_scale)
            hybrid_pred = pred + conditioned_strength * conditioned_gate * conditioned_pred

        if (
            self.llm_fusion_mode == "residual"
            or (conditioned_disabled and not self.multi_scale_moe_enabled)
            or (self.condition_mlp is None and not self.multi_scale_moe_enabled)
        ):
            self.last_aux_outputs = {
                **(self.last_aux_outputs or {}),
                "base_pred": base_pred,
            }
            return pred
        if (
            not self.multi_scale_moe_enabled
            or self.moe_gate is None
            or self.moe_scale is None
            or self.moe_physics_head is None
            or self.moe_global_head is None
            or self.moe_local_head is None
        ):
            self.last_moe_weights = None
            self.last_moe_priors = None
            self.last_aux_outputs = {
                **(self.last_aux_outputs or {}),
                "base_pred": base_pred,
            }
            return hybrid_pred

        if rule_scores is None:
            rule_scores = torch.zeros(
                (llm_features.size(0), len(self.global_rule_names)),
                dtype=llm_features.dtype,
                device=llm_features.device,
            )
        global_rule_score = rule_scores.mean(dim=-1, keepdim=True)
        moe_priors = torch.cat(
            [
                1.0 - adaptive_prior,
                0.5 * (global_rule_score + density_weight),
                0.5 * (adaptive_prior + confidence_weight),
            ],
            dim=-1,
        )
        centered_priors = moe_priors - moe_priors.mean(dim=-1, keepdim=True)
        moe_gate_input = torch.cat(
            [
                local_fused,
                augmented_gate_context,
                conditioned_context,
                llm_emb,
                rule_scores,
                moe_priors,
            ],
            dim=-1,
        )
        moe_logits = self.moe_gate(moe_gate_input) + self.gate_adaptive_strength * centered_priors
        route_override = self.ablation_moe_route_override
        if route_override is None:
            moe_weights = torch.softmax(moe_logits, dim=-1)
        else:
            route_map = {"physics": 0, "global": 1, "local": 2}
            if route_override not in route_map:
                raise ValueError(f"Unknown ablation_moe_route_override: {route_override}")
            moe_weights = torch.zeros_like(moe_logits)
            moe_weights[:, route_map[route_override]] = 1.0

        physics_delta = self.moe_physics_head(torch.cat([local_fused, local_features], dim=-1))
        global_delta = self.moe_global_head(torch.cat([augmented_gate_context, global_features], dim=-1))
        local_delta = self.moe_local_head(torch.cat([conditioned_context, conditioned_features], dim=-1))
        moe_delta = torch.cat([physics_delta, global_delta, local_delta], dim=-1)

        self.last_moe_weights = moe_weights.detach()
        self.last_moe_priors = moe_priors.detach()
        self.last_aux_outputs = {
            **(self.last_aux_outputs or {}),
            "base_pred": base_pred,
        }
        return hybrid_pred + F.softplus(self.moe_scale) * (moe_weights * moe_delta).sum(dim=-1, keepdim=True)


class LLMAttentionBias(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        aggregation: str = "max",
    ):
        super().__init__()
        self.num_heads = int(num_heads)
        self.aggregation = aggregation
        self.encoder = InteractionFeatureEncoder()
        self.feature_dim = int(self.encoder.feature_dim)
        self.proj = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_heads),
        )

    def build_bias(
        self,
        interactions: Sequence[object],
        num_nodes: int,
        device: torch.device,
    ) -> torch.Tensor:
        if not interactions or num_nodes <= 0:
            return torch.zeros((self.num_heads, num_nodes, num_nodes), device=device)
        pair_index, pair_features = build_interaction_pair_features(
            interactions,
            num_nodes,
            self.encoder,
            device,
        )
        if pair_features.numel() == 0:
            return torch.zeros((self.num_heads, num_nodes, num_nodes), device=device)
        proj = self.proj(pair_features)
        idx_i = pair_index[:, 0]
        idx_j = pair_index[:, 1]
        flat_size = num_nodes * num_nodes
        bias_flat = torch.zeros((self.num_heads, flat_size), device=device, dtype=proj.dtype)
        lin = idx_i * num_nodes + idx_j
        lin_sym = idx_j * num_nodes + idx_i
        index = lin.unsqueeze(0).expand(self.num_heads, -1)
        index_sym = lin_sym.unsqueeze(0).expand(self.num_heads, -1)
        values = proj.transpose(0, 1)
        if self.aggregation == "sum":
            bias_flat = bias_flat.scatter_add(1, index, values)
            bias_flat = bias_flat.scatter_add(1, index_sym, values)
        elif self.aggregation == "max":
            bias_flat = bias_flat.scatter_reduce(1, index, values, reduce="amax", include_self=True)
            bias_flat = bias_flat.scatter_reduce(1, index_sym, values, reduce="amax", include_self=True)
        else:
            raise ValueError(f"Unknown bias aggregation: {self.aggregation}")
        return bias_flat.view(self.num_heads, num_nodes, num_nodes)


class PaiNNAttentionBiasRanker(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        feat_dim: int = 0,
        dropout: float = 0.1,
        num_rbf: int = 32,
        cutoff: float = 5.0,
        attn_heads: int = 4,
        bias_hidden_dim: int = 64,
        llm_bias_scale: float = 0.0,
        bias_aggregation: str = "max",
    ):
        super().__init__()
        self.llm_bias_scale = float(llm_bias_scale)
        self.attn_gate = nn.Parameter(torch.zeros(1))
        self.cat_encoder = PaiNNEncoder(hidden_dim, num_layers, num_rbf, cutoff, dropout=dropout)
        self.reactant_encoder = PaiNNEncoder(hidden_dim, num_layers, num_rbf, cutoff, dropout=dropout)
        self.attn_bias = LLMAttentionBias(
            num_heads=attn_heads,
            hidden_dim=bias_hidden_dim,
            dropout=dropout,
            aggregation=bias_aggregation,
        )
        self.attn = DistanceBiasedCrossAttention(
            embed_dim=hidden_dim,
            num_heads=attn_heads,
            dropout=dropout,
            num_rbf=1,
            num_gbf=1,
            cutoff=1.0,
            bias_scale=0.0,
        )
        self.combine_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.feature_gate = None
        if feat_dim > 0:
            self.feature_gate = nn.Sequential(
                nn.Linear(hidden_dim, feat_dim, bias=False),
                nn.Tanh(),
            )
        head_in = hidden_dim + feat_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _build_bias_list(
        self,
        interactions: Optional[Sequence[Optional[Sequence[object]]]],
        batch: torch.Tensor,
    ) -> Optional[List[Optional[torch.Tensor]]]:
        if self.llm_bias_scale == 0.0 or not interactions:
            return None
        if batch.numel() == 0:
            return None
        num_graphs = int(batch.max().item()) + 1
        bias_list: List[Optional[torch.Tensor]] = []
        for g in range(num_graphs):
            inter = interactions[g] if g < len(interactions) else None
            num_nodes = int((batch == g).sum().item())
            if not inter or num_nodes == 0:
                bias_list.append(None)
                continue
            bias = self.attn_bias.build_bias(inter, num_nodes, batch.device)
            if bias.numel() == 0:
                bias_list.append(None)
            else:
                bias_list.append(bias)
        return bias_list

    def _apply_attention_bias(
        self,
        nodes: torch.Tensor,
        batch: torch.Tensor,
        bias_list: Optional[Sequence[Optional[torch.Tensor]]],
        bias_scale: float,
    ) -> torch.Tensor:
        if bias_scale == 0.0 or not bias_list:
            return nodes
        if batch.numel() == 0:
            return nodes
        num_graphs = int(batch.max().item()) + 1
        fused_nodes = []
        for g in range(num_graphs):
            mask = batch == g
            nodes_g = nodes[mask]
            if nodes_g.numel() == 0:
                continue
            bias = bias_list[g] if g < len(bias_list) else None
            if bias is None or bias.numel() == 0:
                fused = nodes_g
            else:
                attn_out = self.attn(
                    nodes_g,
                    nodes_g,
                    nodes_g,
                    distances=None,
                    external_bias=bias * float(bias_scale),
                )
                fused = nodes_g + self.attn_gate * attn_out
            fused_nodes.append(fused)
        if not fused_nodes:
            return nodes
        return torch.cat(fused_nodes, dim=0)

    def forward(
        self,
        cat_z: torch.Tensor,
        cat_pos: torch.Tensor,
        cat_batch: torch.Tensor,
        cat_edge_index: torch.Tensor,
        r1_z: torch.Tensor,
        r1_pos: torch.Tensor,
        r1_batch: torch.Tensor,
        r1_edge_index: torch.Tensor,
        r2_z: torch.Tensor,
        r2_pos: torch.Tensor,
        r2_batch: torch.Tensor,
        r2_edge_index: torch.Tensor,
        features: torch.Tensor,
        cat_interactions: Optional[Sequence[Optional[Sequence[object]]]] = None,
        r1_interactions: Optional[Sequence[Optional[Sequence[object]]]] = None,
        r2_interactions: Optional[Sequence[Optional[Sequence[object]]]] = None,
        llm_bias_scale: Optional[float] = None,
    ) -> torch.Tensor:
        bias_scale = self.llm_bias_scale if llm_bias_scale is None else float(llm_bias_scale)

        cat_nodes, _ = self.cat_encoder(cat_z, cat_pos, cat_batch, cat_edge_index)
        r1_nodes, _ = self.reactant_encoder(r1_z, r1_pos, r1_batch, r1_edge_index)
        r2_nodes, _ = self.reactant_encoder(r2_z, r2_pos, r2_batch, r2_edge_index)

        cat_bias = self._build_bias_list(cat_interactions, cat_batch)
        r1_bias = self._build_bias_list(r1_interactions, r1_batch)
        r2_bias = self._build_bias_list(r2_interactions, r2_batch)

        cat_fused = self._apply_attention_bias(cat_nodes, cat_batch, cat_bias, bias_scale)
        r1_fused = self._apply_attention_bias(r1_nodes, r1_batch, r1_bias, bias_scale)
        r2_fused = self._apply_attention_bias(r2_nodes, r2_batch, r2_bias, bias_scale)

        cat_emb = self.cat_encoder._pool(cat_fused, cat_batch)
        r1_emb = self.reactant_encoder._pool(r1_fused, r1_batch)
        r2_emb = self.reactant_encoder._pool(r2_fused, r2_batch)

        combined = self.combine_mlp(torch.cat([cat_emb, r1_emb, r2_emb], dim=-1))
        if self.feature_gate is not None and features.numel() > 0:
            gate = self.feature_gate(combined)
            gated_features = features * gate
        else:
            gated_features = features
        head_input = torch.cat([combined, gated_features], dim=-1)
        return self.head(head_input)


class AlignmentGuidedPaiNNRanker(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        feat_dim: int = 0,
        dropout: float = 0.1,
        num_rbf: int = 32,
        cutoff: float = 5.0,
        alignment_mode: str = "rule",
        edge_weight_mode: str = "both",
    ):
        super().__init__()
        self.cat_encoder = PaiNNEncoder(hidden_dim, num_layers, num_rbf, cutoff, dropout=dropout)
        self.reactant_encoder = PaiNNEncoder(hidden_dim, num_layers, num_rbf, cutoff, dropout=dropout)
        self.combine_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.feature_gate = None
        if feat_dim > 0:
            self.feature_gate = nn.Sequential(
                nn.Linear(hidden_dim, feat_dim, bias=False),
                nn.Tanh(),
            )
        head_in = hidden_dim + feat_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        if edge_weight_mode not in {"both", "scalar", "vector"}:
            raise ValueError(f"Unknown edge_weight_mode: {edge_weight_mode}")
        self.alignment = AlignmentModule(mode=alignment_mode, hidden_dim=hidden_dim, dropout=dropout)
        self.edge_weight_mode = edge_weight_mode

    def forward(
        self,
        cat_z: torch.Tensor,
        cat_pos: torch.Tensor,
        cat_batch: torch.Tensor,
        cat_edge_index: torch.Tensor,
        r1_z: torch.Tensor,
        r1_pos: torch.Tensor,
        r1_batch: torch.Tensor,
        r1_edge_index: torch.Tensor,
        r2_z: torch.Tensor,
        r2_pos: torch.Tensor,
        r2_batch: torch.Tensor,
        r2_edge_index: torch.Tensor,
        features: torch.Tensor,
        cat_interactions: Optional[Sequence[Optional[Sequence[object]]]] = None,
        r1_interactions: Optional[Sequence[Optional[Sequence[object]]]] = None,
        r2_interactions: Optional[Sequence[Optional[Sequence[object]]]] = None,
    ) -> torch.Tensor:
        cat_edge_weight = _batch_edge_weights(self.alignment, cat_edge_index, cat_batch, cat_interactions)
        r1_edge_weight = _batch_edge_weights(self.alignment, r1_edge_index, r1_batch, r1_interactions)
        r2_edge_weight = _batch_edge_weights(self.alignment, r2_edge_index, r2_batch, r2_interactions)

        _, cat_emb = self.cat_encoder(
            cat_z,
            cat_pos,
            cat_batch,
            cat_edge_index,
            edge_weight=cat_edge_weight,
            edge_weight_mode=self.edge_weight_mode,
        )
        _, r1_emb = self.reactant_encoder(
            r1_z,
            r1_pos,
            r1_batch,
            r1_edge_index,
            edge_weight=r1_edge_weight,
            edge_weight_mode=self.edge_weight_mode,
        )
        _, r2_emb = self.reactant_encoder(
            r2_z,
            r2_pos,
            r2_batch,
            r2_edge_index,
            edge_weight=r2_edge_weight,
            edge_weight_mode=self.edge_weight_mode,
        )
        combined = self.combine_mlp(torch.cat([cat_emb, r1_emb, r2_emb], dim=-1))
        if self.feature_gate is not None and features.numel() > 0:
            gate = self.feature_gate(combined)
            gated_features = features * gate
        else:
            gated_features = features
        head_input = torch.cat([combined, gated_features], dim=-1)
        return self.head(head_input)


class DimeNetBackboneRanker(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        feat_dim: int = 0,
        dropout: float = 0.1,
        num_rbf: int = 32,
        num_spherical: int = 4,
        cutoff: float = 5.0,
    ):
        super().__init__()
        self.cat_encoder = DimeNetLiteEncoder(
            hidden_dim,
            num_layers,
            num_rbf,
            num_spherical,
            cutoff,
            dropout=dropout,
        )
        self.reactant_encoder = DimeNetLiteEncoder(
            hidden_dim,
            num_layers,
            num_rbf,
            num_spherical,
            cutoff,
            dropout=dropout,
        )
        self.combine_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.feature_gate = None
        if feat_dim > 0:
            self.feature_gate = nn.Sequential(
                nn.Linear(hidden_dim, feat_dim, bias=False),
                nn.Tanh(),
            )
        head_in = hidden_dim + feat_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        cat_z: torch.Tensor,
        cat_pos: torch.Tensor,
        cat_batch: torch.Tensor,
        cat_edge_index: torch.Tensor,
        r1_z: torch.Tensor,
        r1_pos: torch.Tensor,
        r1_batch: torch.Tensor,
        r1_edge_index: torch.Tensor,
        r2_z: torch.Tensor,
        r2_pos: torch.Tensor,
        r2_batch: torch.Tensor,
        r2_edge_index: torch.Tensor,
        features: torch.Tensor,
        qc_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, cat_emb = self.cat_encoder(cat_z, cat_pos, cat_batch, cat_edge_index)
        _, r1_emb = self.reactant_encoder(r1_z, r1_pos, r1_batch, r1_edge_index)
        _, r2_emb = self.reactant_encoder(r2_z, r2_pos, r2_batch, r2_edge_index)
        combined = self.combine_mlp(torch.cat([cat_emb, r1_emb, r2_emb], dim=-1))
        if self.feature_gate is not None and features.numel() > 0:
            gate = self.feature_gate(combined)
            gated_features = features * gate
        else:
            gated_features = features
        head_input = torch.cat([combined, gated_features], dim=-1)
        return self.head(head_input)


class GemNetDTBackboneRanker(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        feat_dim: int = 0,
        dropout: float = 0.1,
        num_rbf: int = 32,
        num_spherical: int = 4,
        cutoff: float = 5.0,
    ):
        super().__init__()
        self.cat_encoder = GemNetDTLiteEncoder(
            hidden_dim,
            num_layers,
            num_rbf,
            num_spherical,
            cutoff,
            dropout=dropout,
        )
        self.reactant_encoder = GemNetDTLiteEncoder(
            hidden_dim,
            num_layers,
            num_rbf,
            num_spherical,
            cutoff,
            dropout=dropout,
        )
        self.combine_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.feature_gate = None
        if feat_dim > 0:
            self.feature_gate = nn.Sequential(
                nn.Linear(hidden_dim, feat_dim, bias=False),
                nn.Tanh(),
            )
        head_in = hidden_dim + feat_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        cat_z: torch.Tensor,
        cat_pos: torch.Tensor,
        cat_batch: torch.Tensor,
        cat_edge_index: torch.Tensor,
        r1_z: torch.Tensor,
        r1_pos: torch.Tensor,
        r1_batch: torch.Tensor,
        r1_edge_index: torch.Tensor,
        r2_z: torch.Tensor,
        r2_pos: torch.Tensor,
        r2_batch: torch.Tensor,
        r2_edge_index: torch.Tensor,
        features: torch.Tensor,
        qc_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, cat_emb = self.cat_encoder(cat_z, cat_pos, cat_batch, cat_edge_index)
        _, r1_emb = self.reactant_encoder(r1_z, r1_pos, r1_batch, r1_edge_index)
        _, r2_emb = self.reactant_encoder(r2_z, r2_pos, r2_batch, r2_edge_index)
        combined = self.combine_mlp(torch.cat([cat_emb, r1_emb, r2_emb], dim=-1))
        if self.feature_gate is not None and features.numel() > 0:
            gate = self.feature_gate(combined)
            gated_features = features * gate
        else:
            gated_features = features
        head_input = torch.cat([combined, gated_features], dim=-1)
        return self.head(head_input)


class SchNetNodeQCRanker(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        feat_dim: int = 0,
        dropout: float = 0.1,
        num_rbf: int = 32,
        cutoff: float = 5.0,
        node_qc_dim: int = 0,
    ):
        super().__init__()
        self.cat_encoder = SchNetEncoder(
            hidden_dim,
            num_layers,
            num_rbf,
            cutoff,
            dropout=dropout,
            node_qc_dim=node_qc_dim,
        )
        self.reactant_encoder = SchNetEncoder(
            hidden_dim,
            num_layers,
            num_rbf,
            cutoff,
            dropout=dropout,
            node_qc_dim=node_qc_dim,
        )
        self.combine_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        head_in = hidden_dim + feat_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        cat_z: torch.Tensor,
        cat_pos: torch.Tensor,
        cat_batch: torch.Tensor,
        cat_edge_index: torch.Tensor,
        r1_z: torch.Tensor,
        r1_pos: torch.Tensor,
        r1_batch: torch.Tensor,
        r1_edge_index: torch.Tensor,
        r2_z: torch.Tensor,
        r2_pos: torch.Tensor,
        r2_batch: torch.Tensor,
        r2_edge_index: torch.Tensor,
        features: torch.Tensor,
        cat_node_qc: Optional[torch.Tensor] = None,
        r1_node_qc: Optional[torch.Tensor] = None,
        r2_node_qc: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, cat_emb = self.cat_encoder(
            cat_z, cat_pos, cat_batch, cat_edge_index, node_qc=cat_node_qc
        )
        _, r1_emb = self.reactant_encoder(
            r1_z, r1_pos, r1_batch, r1_edge_index, node_qc=r1_node_qc
        )
        _, r2_emb = self.reactant_encoder(
            r2_z, r2_pos, r2_batch, r2_edge_index, node_qc=r2_node_qc
        )

        combined = self.combine_mlp(torch.cat([cat_emb, r1_emb, r2_emb], dim=-1))
        head_input = torch.cat([combined, features], dim=-1)
        return self.head(head_input)


class SchNetQCGatedMessageRanker(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        feat_dim: int = 0,
        dropout: float = 0.1,
        num_rbf: int = 32,
        cutoff: float = 5.0,
        qc_gate_dim: int = 0,
    ):
        super().__init__()
        self.cat_encoder = SchNetEncoder(
            hidden_dim,
            num_layers,
            num_rbf,
            cutoff,
            dropout=dropout,
            qc_gate_dim=qc_gate_dim,
        )
        self.reactant_encoder = SchNetEncoder(
            hidden_dim,
            num_layers,
            num_rbf,
            cutoff,
            dropout=dropout,
            qc_gate_dim=qc_gate_dim,
        )
        self.combine_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        head_in = hidden_dim + feat_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        cat_z: torch.Tensor,
        cat_pos: torch.Tensor,
        cat_batch: torch.Tensor,
        cat_edge_index: torch.Tensor,
        r1_z: torch.Tensor,
        r1_pos: torch.Tensor,
        r1_batch: torch.Tensor,
        r1_edge_index: torch.Tensor,
        r2_z: torch.Tensor,
        r2_pos: torch.Tensor,
        r2_batch: torch.Tensor,
        r2_edge_index: torch.Tensor,
        features: torch.Tensor,
        qc_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, cat_emb = self.cat_encoder(
            cat_z, cat_pos, cat_batch, cat_edge_index, qc_features=qc_features
        )
        _, r1_emb = self.reactant_encoder(
            r1_z, r1_pos, r1_batch, r1_edge_index, qc_features=qc_features
        )
        _, r2_emb = self.reactant_encoder(
            r2_z, r2_pos, r2_batch, r2_edge_index, qc_features=qc_features
        )

        combined = self.combine_mlp(torch.cat([cat_emb, r1_emb, r2_emb], dim=-1))
        head_input = torch.cat([combined, features], dim=-1)
        return self.head(head_input)


class SchNetQCCrossAttentionRanker(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        heads: int = 4,
        feat_dim: int = 0,
        dropout: float = 0.1,
        num_rbf: int = 32,
        cutoff: float = 5.0,
        qc_dim: int = 0,
    ):
        super().__init__()
        self.cat_encoder = SchNetEncoder(
            hidden_dim,
            num_layers,
            num_rbf,
            cutoff,
            dropout=dropout,
        )
        self.reactant_encoder = SchNetEncoder(
            hidden_dim,
            num_layers,
            num_rbf,
            cutoff,
            dropout=dropout,
        )
        self.qc_proj = None
        if qc_dim > 0:
            self.qc_proj = nn.Sequential(
                nn.Linear(qc_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
        self.qc_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.combine_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        head_in = hidden_dim + feat_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _apply_qc_attention(
        self,
        nodes: torch.Tensor,
        batch: torch.Tensor,
        qc_features: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.qc_proj is None or qc_features is None or qc_features.numel() == 0:
            return nodes
        if batch.numel() == 0:
            return nodes
        num_graphs = int(batch.max().item()) + 1
        fused_nodes = []
        for g in range(num_graphs):
            mask = batch == g
            if not mask.any():
                continue
            nodes_g = nodes[mask]
            qc_token = self.qc_proj(qc_features[g : g + 1]).unsqueeze(0)
            attn_out, _ = self.qc_attn(nodes_g.unsqueeze(0), qc_token, qc_token)
            fused_nodes.append(nodes_g + attn_out.squeeze(0))
        if not fused_nodes:
            return nodes
        return torch.cat(fused_nodes, dim=0)

    def forward(
        self,
        cat_z: torch.Tensor,
        cat_pos: torch.Tensor,
        cat_batch: torch.Tensor,
        cat_edge_index: torch.Tensor,
        r1_z: torch.Tensor,
        r1_pos: torch.Tensor,
        r1_batch: torch.Tensor,
        r1_edge_index: torch.Tensor,
        r2_z: torch.Tensor,
        r2_pos: torch.Tensor,
        r2_batch: torch.Tensor,
        r2_edge_index: torch.Tensor,
        features: torch.Tensor,
        qc_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cat_nodes, _ = self.cat_encoder(cat_z, cat_pos, cat_batch, cat_edge_index)
        r1_nodes, _ = self.reactant_encoder(r1_z, r1_pos, r1_batch, r1_edge_index)
        r2_nodes, _ = self.reactant_encoder(r2_z, r2_pos, r2_batch, r2_edge_index)

        cat_fused = self._apply_qc_attention(cat_nodes, cat_batch, qc_features)
        r1_fused = self._apply_qc_attention(r1_nodes, r1_batch, qc_features)
        r2_fused = self._apply_qc_attention(r2_nodes, r2_batch, qc_features)

        cat_emb = self.cat_encoder._pool(cat_fused, cat_batch)
        r1_emb = self.reactant_encoder._pool(r1_fused, r1_batch)
        r2_emb = self.reactant_encoder._pool(r2_fused, r2_batch)

        combined = self.combine_mlp(torch.cat([cat_emb, r1_emb, r2_emb], dim=-1))
        head_input = torch.cat([combined, features], dim=-1)
        return self.head(head_input)


class SchNetQCHybridRanker(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        heads: int = 4,
        feat_dim: int = 0,
        dropout: float = 0.1,
        num_rbf: int = 32,
        cutoff: float = 5.0,
        qc_dim: int = 0,
        node_qc_dim: int = 0,
        qc_gate_dim: int = 0,
        node_qc_gate_dim: int = 0,
    ):
        super().__init__()
        self.qc_dim = int(qc_dim)
        self.supports_zero_struct = True
        self.cat_encoder = SchNetEncoder(
            hidden_dim,
            num_layers,
            num_rbf,
            cutoff,
            dropout=dropout,
            node_qc_dim=node_qc_dim,
            qc_gate_dim=qc_gate_dim,
            node_qc_gate_dim=node_qc_gate_dim,
        )
        self.reactant_encoder = SchNetEncoder(
            hidden_dim,
            num_layers,
            num_rbf,
            cutoff,
            dropout=dropout,
            node_qc_dim=node_qc_dim,
            qc_gate_dim=qc_gate_dim,
            node_qc_gate_dim=node_qc_gate_dim,
        )
        self.qc_proj = None
        if self.qc_dim > 0:
            self.qc_proj = nn.Sequential(
                nn.Linear(self.qc_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
        self.qc_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        gate_in = hidden_dim * 2 + (self.qc_dim if self.qc_dim > 0 else 0)
        self.mix_gate = nn.Sequential(
            nn.Linear(gate_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        if hasattr(self.mix_gate[-1], "bias") and self.mix_gate[-1].bias is not None:
            nn.init.constant_(self.mix_gate[-1].bias, -1.0)
        self.combine_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        head_in = hidden_dim + feat_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _apply_qc_attention(
        self,
        nodes: torch.Tensor,
        batch: torch.Tensor,
        qc_features: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.qc_proj is None or qc_features is None or qc_features.numel() == 0:
            return nodes
        if batch.numel() == 0:
            return nodes
        num_graphs = int(batch.max().item()) + 1
        fused_nodes = []
        for g in range(num_graphs):
            mask = batch == g
            if not mask.any():
                continue
            nodes_g = nodes[mask]
            qc_token = self.qc_proj(qc_features[g : g + 1]).unsqueeze(0)
            attn_out, _ = self.qc_attn(nodes_g.unsqueeze(0), qc_token, qc_token)
            attn_out = attn_out.squeeze(0)
            fused_nodes.append(nodes_g + attn_out)
        if not fused_nodes:
            return nodes
        return torch.cat(fused_nodes, dim=0)

    def _mix_embeddings(
        self,
        base_emb: torch.Tensor,
        attn_emb: torch.Tensor,
        qc_features: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.qc_dim > 0:
            if qc_features is None or qc_features.numel() == 0:
                qc_features = torch.zeros(
                    (base_emb.size(0), self.qc_dim),
                    device=base_emb.device,
                    dtype=base_emb.dtype,
                )
            gate_input = torch.cat([base_emb, attn_emb, qc_features], dim=-1)
        else:
            gate_input = torch.cat([base_emb, attn_emb], dim=-1)
        gate = torch.sigmoid(self.mix_gate(gate_input))
        return gate * attn_emb + (1.0 - gate) * base_emb

    def forward(
        self,
        cat_z: torch.Tensor,
        cat_pos: torch.Tensor,
        cat_batch: torch.Tensor,
        cat_edge_index: torch.Tensor,
        r1_z: torch.Tensor,
        r1_pos: torch.Tensor,
        r1_batch: torch.Tensor,
        r1_edge_index: torch.Tensor,
        r2_z: torch.Tensor,
        r2_pos: torch.Tensor,
        r2_batch: torch.Tensor,
        r2_edge_index: torch.Tensor,
        features: torch.Tensor,
        qc_features: Optional[torch.Tensor] = None,
        cat_node_qc: Optional[torch.Tensor] = None,
        r1_node_qc: Optional[torch.Tensor] = None,
        r2_node_qc: Optional[torch.Tensor] = None,
        zero_struct: bool = False,
    ) -> torch.Tensor:
        if zero_struct:
            qc_features = None
            cat_node_qc = None
            r1_node_qc = None
            r2_node_qc = None
            if features is not None:
                features = torch.zeros_like(features)
        cat_nodes, cat_base = self.cat_encoder(
            cat_z,
            cat_pos,
            cat_batch,
            cat_edge_index,
            qc_features=qc_features,
            node_qc=cat_node_qc,
        )
        r1_nodes, r1_base = self.reactant_encoder(
            r1_z,
            r1_pos,
            r1_batch,
            r1_edge_index,
            qc_features=qc_features,
            node_qc=r1_node_qc,
        )
        r2_nodes, r2_base = self.reactant_encoder(
            r2_z,
            r2_pos,
            r2_batch,
            r2_edge_index,
            qc_features=qc_features,
            node_qc=r2_node_qc,
        )

        cat_fused = self._apply_qc_attention(cat_nodes, cat_batch, qc_features)
        r1_fused = self._apply_qc_attention(r1_nodes, r1_batch, qc_features)
        r2_fused = self._apply_qc_attention(r2_nodes, r2_batch, qc_features)

        cat_attn = self.cat_encoder._pool(cat_fused, cat_batch)
        r1_attn = self.reactant_encoder._pool(r1_fused, r1_batch)
        r2_attn = self.reactant_encoder._pool(r2_fused, r2_batch)

        cat_emb = self._mix_embeddings(cat_base, cat_attn, qc_features)
        r1_emb = self._mix_embeddings(r1_base, r1_attn, qc_features)
        r2_emb = self._mix_embeddings(r2_base, r2_attn, qc_features)

        if zero_struct:
            combined = torch.zeros_like(cat_emb)
        else:
            combined = self.combine_mlp(torch.cat([cat_emb, r1_emb, r2_emb], dim=-1))
        head_input = torch.cat([combined, features], dim=-1)
        return self.head(head_input)


class SchNetBackboneGatedFusion(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        feat_dim: int = 0,
        dropout: float = 0.1,
        num_rbf: int = 32,
        cutoff: float = 5.0,
        qc_dim: int = 0,
        gate_dropout: float = 0.1,
    ):
        super().__init__()
        self.cat_encoder = SchNetEncoder(
            hidden_dim,
            num_layers,
            num_rbf,
            cutoff,
            dropout=dropout,
            qc_dim=qc_dim,
        )
        self.reactant_encoder = SchNetEncoder(
            hidden_dim,
            num_layers,
            num_rbf,
            cutoff,
            dropout=dropout,
            qc_dim=qc_dim,
        )
        self.combine_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.qc_gate = None
        if feat_dim > 0:
            self.qc_gate = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(gate_dropout),
                nn.Linear(hidden_dim, feat_dim),
                nn.Sigmoid(),
            )
        head_in = hidden_dim + feat_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        cat_z: torch.Tensor,
        cat_pos: torch.Tensor,
        cat_batch: torch.Tensor,
        cat_edge_index: torch.Tensor,
        r1_z: torch.Tensor,
        r1_pos: torch.Tensor,
        r1_batch: torch.Tensor,
        r1_edge_index: torch.Tensor,
        r2_z: torch.Tensor,
        r2_pos: torch.Tensor,
        r2_batch: torch.Tensor,
        r2_edge_index: torch.Tensor,
        features: torch.Tensor,
        qc_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, cat_emb = self.cat_encoder(cat_z, cat_pos, cat_batch, cat_edge_index, qc_features=qc_features)
        _, r1_emb = self.reactant_encoder(r1_z, r1_pos, r1_batch, r1_edge_index, qc_features=qc_features)
        _, r2_emb = self.reactant_encoder(r2_z, r2_pos, r2_batch, r2_edge_index, qc_features=qc_features)

        combined = self.combine_mlp(torch.cat([cat_emb, r1_emb, r2_emb], dim=-1))
        if self.qc_gate is not None and features.numel() > 0:
            gate = self.qc_gate(combined)
            gated_features = features * gate
        else:
            gated_features = features
        head_input = torch.cat([combined, gated_features], dim=-1)
        return self.head(head_input)


class SchNetBackboneMixtureRanker(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        feat_dim: int = 0,
        dropout: float = 0.1,
        num_rbf: int = 32,
        cutoff: float = 5.0,
        qc_dim: int = 0,
        gate_dropout: float = 0.1,
    ):
        super().__init__()
        self.cat_encoder = SchNetEncoder(
            hidden_dim,
            num_layers,
            num_rbf,
            cutoff,
            dropout=dropout,
            qc_dim=qc_dim,
        )
        self.reactant_encoder = SchNetEncoder(
            hidden_dim,
            num_layers,
            num_rbf,
            cutoff,
            dropout=dropout,
            qc_dim=qc_dim,
        )
        self.combine_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.struct_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.qc_head = None
        self.qc_gate_proj = None
        self.gate_mlp = None
        if feat_dim > 0:
            self.qc_head = nn.Sequential(
                nn.Linear(feat_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )
            self.qc_gate_proj = nn.Sequential(
                nn.Linear(feat_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(gate_dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.gate_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(gate_dropout),
                nn.Linear(hidden_dim // 2, 1),
            )
            if hasattr(self.gate_mlp[-1], "bias") and self.gate_mlp[-1].bias is not None:
                nn.init.constant_(self.gate_mlp[-1].bias, -1.0)

    def forward(
        self,
        cat_z: torch.Tensor,
        cat_pos: torch.Tensor,
        cat_batch: torch.Tensor,
        cat_edge_index: torch.Tensor,
        r1_z: torch.Tensor,
        r1_pos: torch.Tensor,
        r1_batch: torch.Tensor,
        r1_edge_index: torch.Tensor,
        r2_z: torch.Tensor,
        r2_pos: torch.Tensor,
        r2_batch: torch.Tensor,
        r2_edge_index: torch.Tensor,
        features: torch.Tensor,
        qc_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, cat_emb = self.cat_encoder(cat_z, cat_pos, cat_batch, cat_edge_index, qc_features=qc_features)
        _, r1_emb = self.reactant_encoder(r1_z, r1_pos, r1_batch, r1_edge_index, qc_features=qc_features)
        _, r2_emb = self.reactant_encoder(r2_z, r2_pos, r2_batch, r2_edge_index, qc_features=qc_features)

        combined = self.combine_mlp(torch.cat([cat_emb, r1_emb, r2_emb], dim=-1))
        struct_pred = self.struct_head(combined)
        if (
            self.qc_head is None
            or self.qc_gate_proj is None
            or self.gate_mlp is None
            or features.numel() == 0
        ):
            return struct_pred
        qc_pred = self.qc_head(features)
        qc_gate = self.qc_gate_proj(features)
        gate_input = torch.cat([combined, qc_gate], dim=-1)
        gate = torch.sigmoid(self.gate_mlp(gate_input))
        return gate * qc_pred + (1.0 - gate) * struct_pred


class SchNetBackboneResidualQCRanker(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        feat_dim: int = 0,
        dropout: float = 0.1,
        num_rbf: int = 32,
        cutoff: float = 5.0,
        qc_dim: int = 0,
        gate_dropout: float = 0.1,
    ):
        super().__init__()
        self.cat_encoder = SchNetEncoder(
            hidden_dim,
            num_layers,
            num_rbf,
            cutoff,
            dropout=dropout,
            qc_dim=qc_dim,
        )
        self.reactant_encoder = SchNetEncoder(
            hidden_dim,
            num_layers,
            num_rbf,
            cutoff,
            dropout=dropout,
            qc_dim=qc_dim,
        )
        self.combine_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.struct_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.qc_head = None
        self.qc_gate_proj = None
        self.gate_mlp = None
        if feat_dim > 0:
            self.qc_head = nn.Sequential(
                nn.Linear(feat_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )
            self.qc_gate_proj = nn.Sequential(
                nn.Linear(feat_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(gate_dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.gate_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(gate_dropout),
                nn.Linear(hidden_dim // 2, 1),
            )
            if hasattr(self.gate_mlp[-1], "bias") and self.gate_mlp[-1].bias is not None:
                nn.init.constant_(self.gate_mlp[-1].bias, -1.5)

    def forward(
        self,
        cat_z: torch.Tensor,
        cat_pos: torch.Tensor,
        cat_batch: torch.Tensor,
        cat_edge_index: torch.Tensor,
        r1_z: torch.Tensor,
        r1_pos: torch.Tensor,
        r1_batch: torch.Tensor,
        r1_edge_index: torch.Tensor,
        r2_z: torch.Tensor,
        r2_pos: torch.Tensor,
        r2_batch: torch.Tensor,
        r2_edge_index: torch.Tensor,
        features: torch.Tensor,
        qc_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, cat_emb = self.cat_encoder(cat_z, cat_pos, cat_batch, cat_edge_index, qc_features=qc_features)
        _, r1_emb = self.reactant_encoder(r1_z, r1_pos, r1_batch, r1_edge_index, qc_features=qc_features)
        _, r2_emb = self.reactant_encoder(r2_z, r2_pos, r2_batch, r2_edge_index, qc_features=qc_features)

        combined = self.combine_mlp(torch.cat([cat_emb, r1_emb, r2_emb], dim=-1))
        struct_pred = self.struct_head(combined)
        if (
            self.qc_head is None
            or self.qc_gate_proj is None
            or self.gate_mlp is None
            or features.numel() == 0
        ):
            return struct_pred
        qc_pred = self.qc_head(features)
        qc_gate = self.qc_gate_proj(features)
        gate_input = torch.cat([combined, qc_gate], dim=-1)
        gate = torch.sigmoid(self.gate_mlp(gate_input))
        return struct_pred + gate * qc_pred


class SharedSchNetBackboneRanker(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        num_rbf: int = 32,
        cutoff: float = 5.0,
    ):
        super().__init__()
        self.encoder = SchNetEncoder(
            hidden_dim,
            num_layers,
            num_rbf,
            cutoff,
            dropout=dropout,
            qc_dim=0,
        )
        self.combine_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        cat_z: torch.Tensor,
        cat_pos: torch.Tensor,
        cat_batch: torch.Tensor,
        cat_edge_index: torch.Tensor,
        r1_z: torch.Tensor,
        r1_pos: torch.Tensor,
        r1_batch: torch.Tensor,
        r1_edge_index: torch.Tensor,
        r2_z: torch.Tensor,
        r2_pos: torch.Tensor,
        r2_batch: torch.Tensor,
        r2_edge_index: torch.Tensor,
        features: torch.Tensor,
        qc_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if qc_features is not None and qc_features.numel() > 0:
            raise ValueError("SharedSchNetBackboneRanker is 3D-only; qc_features must be empty.")
        if features is not None and features.numel() > 0:
            raise ValueError("SharedSchNetBackboneRanker is 3D-only; features must be empty.")

        _, cat_emb = self.encoder(cat_z, cat_pos, cat_batch, cat_edge_index, qc_features=None)
        _, r1_emb = self.encoder(r1_z, r1_pos, r1_batch, r1_edge_index, qc_features=None)
        _, r2_emb = self.encoder(r2_z, r2_pos, r2_batch, r2_edge_index, qc_features=None)

        combined = self.combine_mlp(torch.cat([cat_emb, r1_emb, r2_emb], dim=-1))
        return self.head(combined)


def _batch_edge_weights(
    alignment: AlignmentModule,
    edge_index: torch.Tensor,
    batch: torch.Tensor,
    interactions: Optional[Sequence[Optional[Sequence[object]]]],
) -> torch.Tensor:
    edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
    if not interactions:
        return edge_weight
    num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
    for g in range(num_graphs):
        inter = interactions[g] if g < len(interactions) else None
        if not inter:
            continue
        node_idx = (batch == g).nonzero(as_tuple=False).view(-1)
        num_nodes = node_idx.numel()
        if num_nodes == 0:
            continue
        mapping = torch.full((batch.size(0),), -1, device=edge_index.device)
        mapping[node_idx] = torch.arange(num_nodes, device=edge_index.device)
        mask = batch[edge_index[0]] == g
        if mask.any():
            local_edge_index = mapping[edge_index[:, mask]]
            edge_weight[mask] = alignment.build_edge_weights(inter, local_edge_index, num_nodes)
    return edge_weight


def _batch_interaction_edges(
    alignment: AlignmentModule,
    batch: torch.Tensor,
    interactions: Optional[Sequence[Optional[Sequence[object]]]],
    fallback_edge_index: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = batch.device
    if batch.numel() == 0:
        return (
            torch.zeros((2, 0), dtype=torch.long, device=device),
            torch.zeros((0,), dtype=torch.float32, device=device),
        )
    edge_src = []
    edge_dst = []
    edge_wt = []
    num_graphs = int(batch.max().item()) + 1
    offset = 0
    for g in range(num_graphs):
        node_idx = (batch == g).nonzero(as_tuple=False).view(-1)
        num_nodes = node_idx.numel()
        inter = interactions[g] if interactions and g < len(interactions) else None
        if inter:
            local_edge_index, local_edge_weight = alignment.build_interaction_edges(inter, num_nodes, device)
            if local_edge_index.numel() > 0:
                edge_src.append(local_edge_index[0] + offset)
                edge_dst.append(local_edge_index[1] + offset)
                edge_wt.append(local_edge_weight)
            else:
                inter = None
        if not inter:
            mask = batch[fallback_edge_index[0]] == g
            if mask.any():
                edge_src.append(fallback_edge_index[0, mask])
                edge_dst.append(fallback_edge_index[1, mask])
                edge_wt.append(torch.ones(mask.sum(), device=device))
        offset += num_nodes
    if not edge_src:
        return (
            torch.zeros((2, 0), dtype=torch.long, device=device),
            torch.zeros((0,), dtype=torch.float32, device=device),
        )
    edge_index = torch.stack([torch.cat(edge_src), torch.cat(edge_dst)], dim=0)
    edge_weight = torch.cat(edge_wt, dim=0)
    return edge_index, edge_weight


class AlignmentGuidedGNN(DualEquivariantGNN):
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        heads: int = 4,
        feat_dim: int = 24,
        dropout: float = 0.1,
        alignment_mode: str = "rule",
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            feat_dim=feat_dim,
            dropout=dropout,
        )
        self.alignment = AlignmentModule(mode=alignment_mode, hidden_dim=hidden_dim, dropout=dropout)

    def forward(
        self,
        cat_z: torch.Tensor,
        cat_pos: torch.Tensor,
        cat_batch: torch.Tensor,
        cat_edge_index: torch.Tensor,
        r1_z: torch.Tensor,
        r1_pos: torch.Tensor,
        r1_batch: torch.Tensor,
        r1_edge_index: torch.Tensor,
        r2_z: torch.Tensor,
        r2_pos: torch.Tensor,
        r2_batch: torch.Tensor,
        r2_edge_index: torch.Tensor,
        features: torch.Tensor,
        cat_interactions: Optional[Sequence[Optional[Sequence[object]]]] = None,
        r1_interactions: Optional[Sequence[Optional[Sequence[object]]]] = None,
        r2_interactions: Optional[Sequence[Optional[Sequence[object]]]] = None,
    ) -> torch.Tensor:
        cat_edge_weight = _batch_edge_weights(self.alignment, cat_edge_index, cat_batch, cat_interactions)
        r1_edge_weight = _batch_edge_weights(self.alignment, r1_edge_index, r1_batch, r1_interactions)
        r2_edge_weight = _batch_edge_weights(self.alignment, r2_edge_index, r2_batch, r2_interactions)

        cat_emb = self._encode_molecule(cat_z, cat_pos, cat_batch, cat_edge_index, edge_weight=cat_edge_weight)
        r1_emb = self._encode_molecule(r1_z, r1_pos, r1_batch, r1_edge_index, edge_weight=r1_edge_weight)
        r2_emb = self._encode_molecule(r2_z, r2_pos, r2_batch, r2_edge_index, edge_weight=r2_edge_weight)

        combined = self.combine_mlp(torch.cat([cat_emb, r1_emb, r2_emb], dim=-1))
        feat_emb = self.feat_proj(features)
        out = self.head(torch.cat([combined, feat_emb], dim=-1))
        return out


class LiteratureGuidedDualPathGNN(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        heads: int = 4,
        feat_dim: int = 24,
        dropout: float = 0.1,
        alignment_mode: str = "rule",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(120, hidden_dim, padding_idx=0)
        self.path_a_layers = nn.ModuleList([EquivariantMessageLayer(hidden_dim) for _ in range(num_layers)])
        self.path_b_layers = nn.ModuleList([EquivariantMessageLayer(hidden_dim) for _ in range(num_layers)])
        self.pool_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.combine_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.alignment = AlignmentModule(mode=alignment_mode, hidden_dim=hidden_dim, dropout=dropout)

    def _pool(self, h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        pooled_sum = torch.zeros((num_graphs, h.size(-1)), device=h.device)
        pooled_sum.index_add_(0, batch, h)
        counts = torch.bincount(batch, minlength=num_graphs).clamp(min=1).unsqueeze(-1)
        pooled_mean = pooled_sum / counts
        pooled_max = torch.full((num_graphs, h.size(-1)), -float("inf"), device=h.device)
        for i in range(num_graphs):
            h_i = h[batch == i]
            pooled_max[i] = h_i.max(dim=0).values if h_i.numel() > 0 else 0.0
        pooled = torch.cat([pooled_mean, pooled_max], dim=-1)
        return self.pool_mlp(pooled)

    def _encode_path(
        self,
        z: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor],
        layers: nn.ModuleList,
    ) -> torch.Tensor:
        h = self.embedding(z)
        for layer in layers:
            h, pos = layer(h, pos, batch, edge_index, edge_weight=edge_weight)
        return h

    def _cross_attention_fuse(
        self,
        h_a: torch.Tensor,
        h_b: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        fused_nodes = []
        for g in range(num_graphs):
            nodes_a = h_a[batch == g]
            nodes_b = h_b[batch == g]
            if nodes_a.numel() == 0:
                continue
            if nodes_b.numel() == 0:
                fused = nodes_a
            else:
                attn_out, _ = self.cross_attention(
                    nodes_a.unsqueeze(0),
                    nodes_b.unsqueeze(0),
                    nodes_b.unsqueeze(0),
                )
                fused = nodes_a + attn_out.squeeze(0)
            fused_nodes.append(fused)
        if not fused_nodes:
            return h_a
        return torch.cat(fused_nodes, dim=0)

    def _encode_molecule(
        self,
        z: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        edge_index: torch.Tensor,
        interactions: Optional[Sequence[Optional[Sequence[object]]]] = None,
    ) -> torch.Tensor:
        edge_index_b, edge_weight_b = _batch_interaction_edges(
            self.alignment, batch, interactions, edge_index
        )
        edge_weight_a = None
        if interactions is not None:
            edge_weight_a = _batch_edge_weights(self.alignment, edge_index, batch, interactions)
        h_a = self._encode_path(z, pos, batch, edge_index, edge_weight_a, self.path_a_layers)
        h_b = self._encode_path(z, pos, batch, edge_index_b, edge_weight_b, self.path_b_layers)
        fused_nodes = self._cross_attention_fuse(h_a, h_b, batch)
        return self._pool(fused_nodes, batch)

    def forward(
        self,
        cat_z: torch.Tensor,
        cat_pos: torch.Tensor,
        cat_batch: torch.Tensor,
        cat_edge_index: torch.Tensor,
        r1_z: torch.Tensor,
        r1_pos: torch.Tensor,
        r1_batch: torch.Tensor,
        r1_edge_index: torch.Tensor,
        r2_z: torch.Tensor,
        r2_pos: torch.Tensor,
        r2_batch: torch.Tensor,
        r2_edge_index: torch.Tensor,
        features: torch.Tensor,
        cat_interactions: Optional[Sequence[Optional[Sequence[object]]]] = None,
        r1_interactions: Optional[Sequence[Optional[Sequence[object]]]] = None,
        r2_interactions: Optional[Sequence[Optional[Sequence[object]]]] = None,
    ) -> torch.Tensor:
        cat_emb = self._encode_molecule(cat_z, cat_pos, cat_batch, cat_edge_index, cat_interactions)
        r1_emb = self._encode_molecule(r1_z, r1_pos, r1_batch, r1_edge_index, r1_interactions)
        r2_emb = self._encode_molecule(r2_z, r2_pos, r2_batch, r2_edge_index, r2_interactions)

        combined = self.combine_mlp(torch.cat([cat_emb, r1_emb, r2_emb], dim=-1))
        feat_emb = self.feat_proj(features)
        out = self.head(torch.cat([combined, feat_emb], dim=-1))
        return out


def _edge_softmax(scores: torch.Tensor, dst_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    if scores.numel() == 0:
        return scores
    attn = torch.zeros_like(scores)
    for node in range(num_nodes):
        mask = dst_index == node
        if mask.any():
            attn[mask] = torch.softmax(scores[mask], dim=0)
    return attn


class GATInteractionBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        heads: int = 4,
        dropout: float = 0.1,
        edge_dim: int = 1,
    ):
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn = nn.Parameter(torch.empty(heads, 2 * self.head_dim + edge_dim))
        nn.init.xavier_uniform_(self.attn)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_dist: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return h
        src, dst = edge_index
        num_nodes = h.size(0)
        h_proj = self.proj(h).view(num_nodes, self.heads, self.head_dim)
        h_src = h_proj[src]
        h_dst = h_proj[dst]

        if edge_dist is None:
            edge_feat = torch.zeros((h_src.size(0), 1), device=h.device)
        else:
            edge_feat = edge_dist.view(-1, 1)
        edge_feat = edge_feat.unsqueeze(1).expand(-1, self.heads, -1)
        attn_input = torch.cat([h_src, h_dst, edge_feat], dim=-1)
        scores = self.leaky_relu((attn_input * self.attn).sum(dim=-1))
        attn = _edge_softmax(scores, dst, num_nodes)
        attn = self.dropout(attn)

        msg = attn.unsqueeze(-1) * h_src
        agg = torch.zeros((num_nodes, self.heads, self.head_dim), device=h.device)
        for head in range(self.heads):
            agg[:, head, :].index_add_(0, dst, msg[:, head, :])
        out = agg.reshape(num_nodes, self.hidden_dim)
        out = self.out_proj(out)
        return F.elu(out)


class GATEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(120, hidden_dim, padding_idx=0)
        self.layers = nn.ModuleList(
            [
                GATInteractionBlock(hidden_dim, heads=heads, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.pool_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def _pool(self, h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        pooled_sum = torch.zeros((num_graphs, h.size(-1)), device=h.device)
        pooled_sum.index_add_(0, batch, h)
        counts = torch.bincount(batch, minlength=num_graphs).clamp(min=1).unsqueeze(-1)
        pooled_mean = pooled_sum / counts
        pooled_max = torch.full((num_graphs, h.size(-1)), -float("inf"), device=h.device)
        for i in range(num_graphs):
            h_i = h[batch == i]
            pooled_max[i] = h_i.max(dim=0).values if h_i.numel() > 0 else 0.0
        pooled = torch.cat([pooled_mean, pooled_max], dim=-1)
        return self.pool_mlp(pooled)

    def forward(
        self,
        z: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            dist = torch.zeros((0,), device=pos.device)
        else:
            src, dst = edge_index
            dist = torch.norm(pos[dst] - pos[src], dim=-1)
        h = self.embedding(z)
        for layer in self.layers:
            h = h + layer(h, edge_index, dist)
        return self._pool(h, batch)


class DualGATRanker(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        heads: int = 4,
        feat_dim: int = 24,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cat_encoder = GATEncoder(hidden_dim, num_layers, heads, dropout=dropout)
        self.reactant_encoder = GATEncoder(hidden_dim, num_layers, heads, dropout=dropout)
        self.combine_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        cat_z: torch.Tensor,
        cat_pos: torch.Tensor,
        cat_batch: torch.Tensor,
        cat_edge_index: torch.Tensor,
        r1_z: torch.Tensor,
        r1_pos: torch.Tensor,
        r1_batch: torch.Tensor,
        r1_edge_index: torch.Tensor,
        r2_z: torch.Tensor,
        r2_pos: torch.Tensor,
        r2_batch: torch.Tensor,
        r2_edge_index: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        cat_emb = self.cat_encoder(cat_z, cat_pos, cat_batch, cat_edge_index)
        r1_emb = self.reactant_encoder(r1_z, r1_pos, r1_batch, r1_edge_index)
        r2_emb = self.reactant_encoder(r2_z, r2_pos, r2_batch, r2_edge_index)

        combined = self.combine_mlp(torch.cat([cat_emb, r1_emb, r2_emb], dim=-1))
        feat_emb = self.feat_proj(features)
        out = self.head(torch.cat([combined, feat_emb], dim=-1))
        return out


class _EdgeGINLayer(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, node_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim, node_dim),
        )
        self.bn = nn.BatchNorm1d(node_dim)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_feat: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            return h
        src, dst = edge_index
        m = self.mlp(torch.cat([h[src], edge_feat], dim=-1))
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, m)
        h = h + agg
        h = self.bn(h)
        return F.relu(h)


class SSGNNCombined(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        edge_hidden_dim: int = 128,
        num_layers: int = 2,
        feat_dim: int = 24,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.z_emb = nn.Embedding(120, 64, padding_idx=0)
        self.type_emb = nn.Embedding(3, 8)
        self.node_proj = nn.Sequential(
            nn.Linear(64 + 8 + 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.edge_type_emb = nn.Embedding(2, 4)
        self.pair_type_emb = nn.Embedding(6, 8)
        self.edge_proj = nn.Sequential(
            nn.Linear(4 + 8 + 1, edge_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(edge_hidden_dim, edge_hidden_dim),
        )

        self.layers = nn.ModuleList(
            [_EdgeGINLayer(hidden_dim, edge_hidden_dim, dropout=dropout) for _ in range(num_layers)]
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_hidden_dim, edge_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(edge_hidden_dim, edge_hidden_dim),
            nn.ReLU(),
        )
        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.head = nn.Sequential(
            nn.Linear(edge_hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _pool_edges(self, edge_repr: torch.Tensor, edge_batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
        pooled = torch.zeros((num_graphs, edge_repr.size(-1)), device=edge_repr.device)
        if edge_repr.numel() == 0:
            return pooled
        pooled.index_add_(0, edge_batch, edge_repr)
        return pooled

    def _pool_nodes(self, h: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
        pooled = torch.zeros((num_graphs, h.size(-1)), device=h.device)
        if h.numel() == 0:
            return pooled
        pooled.index_add_(0, batch, h)
        counts = torch.bincount(batch, minlength=num_graphs).clamp(min=1).unsqueeze(-1)
        return pooled / counts

    def forward(
        self,
        combined_z: torch.Tensor,
        combined_pos: torch.Tensor,
        combined_node_type: torch.Tensor,
        combined_batch: torch.Tensor,
        combined_edge_index: torch.Tensor,
        combined_edge_attr: torch.Tensor,
        combined_edge_batch: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        num_graphs = int(combined_batch.max().item()) + 1 if combined_batch.numel() > 0 else features.size(0)
        pos_scaled = combined_pos / 10.0
        h = torch.cat(
            [
                self.z_emb(combined_z),
                self.type_emb(combined_node_type),
                pos_scaled,
            ],
            dim=-1,
        )
        h = self.node_proj(h)

        if combined_edge_attr.numel() == 0:
            edge_feat = torch.zeros((0, self.edge_proj[-1].out_features), device=h.device)
        else:
            edge_type = combined_edge_attr[:, 0].long().clamp(min=0, max=1)
            pair_type = combined_edge_attr[:, 1].long().clamp(min=0, max=5)
            dist = combined_edge_attr[:, 2:3] / 10.0
            edge_raw = torch.cat(
                [
                    self.edge_type_emb(edge_type),
                    self.pair_type_emb(pair_type),
                    dist,
                ],
                dim=-1,
            )
            edge_feat = self.edge_proj(edge_raw)

        for layer in self.layers:
            h = layer(h, combined_edge_index, edge_feat)

        if combined_edge_index.numel() == 0:
            graph_repr = self._pool_nodes(h, combined_batch, num_graphs)
        else:
            src, dst = combined_edge_index
            edge_repr = self.edge_mlp(torch.cat([h[src], h[dst], edge_feat], dim=-1))
            graph_repr = self._pool_edges(edge_repr, combined_edge_batch, num_graphs)

        feat_emb = self.feat_proj(features)
        out = self.head(torch.cat([graph_repr, feat_emb], dim=-1))
        return out
