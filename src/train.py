import argparse
import json
import math
import os
import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional, Sequence, Iterable, List, Mapping

import numpy as np
import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import DataLoader, Sampler

from data import MLCBDataset, MLBorylationDataset, native_pair_field_profile_choices
from llm.features import (
    llm_feature_columns,
    llm_pair_group_columns_from_feature_columns,
    llm_semantic_branch_columns_from_feature_columns,
    llm_semantic_branch_index_map,
    llm_semantic_group_columns_from_feature_columns,
    llm_semantic_group_index_map,
)
from alignment import (
    AlignmentRules,
    build_attention_bias_from_scores,
    build_hotspot_scores,
    summarize_attention_bias,
)
from model import (
    DualEquivariantGNN,
    DualSchNetCrossAttention,
    GatedSchNetCrossAttention,
    SchNetBackboneRanker,
    PaiNNBackboneRanker,
    PhysicsInjectedPaiNNRanker,
    PaiNNPrecisionHead,
    PaiNNAttentionBiasRanker,
    DimeNetBackboneRanker,
    GemNetDTBackboneRanker,
    SchNetNodeQCRanker,
    SchNetQCCrossAttentionRanker,
    SchNetQCGatedMessageRanker,
    ReactionGraphTransformerRanker,
    SharedSchNetBackboneRanker,
    AlignmentGuidedGNN,
    AlignmentGuidedPaiNNRanker,
    LiteratureGuidedDualPathGNN,
    DualGATRanker,
    SSGNNCombined,
)


_LLM_FEATURE_INDEX = {name: idx for idx, name in enumerate(llm_feature_columns())}


def _llm_feature_column(features: Optional[torch.Tensor], name: str) -> Optional[torch.Tensor]:
    if features is None or not torch.is_tensor(features) or features.ndim != 2 or features.numel() == 0:
        return None
    index = _LLM_FEATURE_INDEX.get(name)
    if index is None or index >= features.size(1):
        return None
    return features[:, index : index + 1]


def collate_fn(
    batch,
    include_literature: bool = False,
    include_combined: bool = False,
    include_group: bool = False,
    include_rank_target: bool = False,
    llm_bias_scale: float = 0.0,
    llm_bias_mode: str = "sum",
    llm_bias_aggregation: str = "max",
    attn_audit: bool = False,
):
    # batch is list of Sample
    def pack(items, attr):
        zs = [getattr(it, f"{attr}_z") for it in items]
        poss = [getattr(it, f"{attr}_pos") for it in items]
        edges = [getattr(it, f"{attr}_edge_index") for it in items]
        for i in range(len(zs)):
            if zs[i].numel() == 0:
                # pad empty molecule with a dummy atom so batch sizes align
                zs[i] = torch.tensor([0], dtype=torch.long)
                poss[i] = torch.zeros((1, 3), dtype=torch.float32)
                edges[i] = torch.zeros((2, 0), dtype=torch.long)
        z = torch.cat(zs, dim=0)
        pos = torch.cat(poss, dim=0)
        batch_idx = []
        edge_src = []
        edge_dst = []
        offset = 0
        for i, zi in enumerate(zs):
            batch_idx.append(torch.full((zi.size(0),), i, dtype=torch.long))
            edge = edges[i]
            if edge.numel() > 0:
                edge_src.append(edge[0] + offset)
                edge_dst.append(edge[1] + offset)
            offset += zi.size(0)
        batch_idx = torch.cat(batch_idx, dim=0)
        if edge_src:
            edge_index = torch.stack([torch.cat(edge_src), torch.cat(edge_dst)], dim=0)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        node_qc_dim = 0
        for item in items:
            node_qc = getattr(item, f"{attr}_node_qc", None)
            if node_qc is not None and node_qc.numel() > 0:
                node_qc_dim = int(node_qc.size(-1))
                break
        node_qc_list = []
        for item, zi in zip(items, zs):
            node_qc = getattr(item, f"{attr}_node_qc", None)
            if node_qc is None or node_qc.numel() == 0:
                node_qc_list.append(torch.zeros((zi.size(0), node_qc_dim), dtype=torch.float32))
            else:
                node_qc_list.append(node_qc)
        if node_qc_dim:
            node_qc = torch.cat(node_qc_list, dim=0)
        else:
            node_qc = torch.zeros((z.size(0), 0), dtype=torch.float32)
        return z, pos, batch_idx, edge_index, node_qc

    cat_z, cat_pos, cat_batch, cat_edge_index, cat_node_qc = pack(batch, "cat")
    r1_z, r1_pos, r1_batch, r1_edge_index, r1_node_qc = pack(batch, "r1")
    r2_z, r2_pos, r2_batch, r2_edge_index, r2_node_qc = pack(batch, "r2")

    def component_sizes(items, attr):
        sizes = []
        for item in items:
            z = getattr(item, f"{attr}_z")
            sizes.append(int(z.size(0)) if z.numel() > 0 else 1)
        return sizes

    cat_sizes = component_sizes(batch, "cat")
    r1_sizes = component_sizes(batch, "r1")
    r2_sizes = component_sizes(batch, "r2")

    def pack_reaction_pairs(items, attr, src_sizes, dst_sizes):
        pair_indices = []
        pair_features = []
        pair_batch = []
        src_offset = 0
        dst_offset = 0
        feature_dim = 0
        for graph_idx, (item, src_size, dst_size) in enumerate(zip(items, src_sizes, dst_sizes)):
            local_index = getattr(item, f"{attr}_index", None)
            local_features = getattr(item, f"{attr}_features", None)
            if local_features is not None and local_features.numel() > 0:
                feature_dim = int(local_features.size(-1))
            if (
                local_index is not None
                and local_features is not None
                and local_index.numel() > 0
                and local_features.numel() > 0
            ):
                global_index = local_index.clone()
                global_index[0] = global_index[0] + src_offset
                global_index[1] = global_index[1] + dst_offset
                pair_indices.append(global_index)
                pair_features.append(local_features)
                pair_batch.append(torch.full((local_features.size(0),), graph_idx, dtype=torch.long))
            src_offset += int(src_size)
            dst_offset += int(dst_size)
        if pair_indices:
            packed_index = torch.cat(pair_indices, dim=1)
            packed_features = torch.cat(pair_features, dim=0)
            packed_batch = torch.cat(pair_batch, dim=0)
        else:
            packed_index = torch.zeros((2, 0), dtype=torch.long)
            packed_features = torch.zeros((0, feature_dim), dtype=torch.float32)
            packed_batch = torch.zeros((0,), dtype=torch.long)
        return packed_index, packed_features, packed_batch

    features = torch.stack([b.features for b in batch], dim=0)
    targets = torch.stack([b.target for b in batch], dim=0)
    qc_dim = 0
    for item in batch:
        if item.qc_features is not None:
            qc_dim = int(item.qc_features.numel())
            break
    qc_list = []
    for item in batch:
        if item.qc_features is None:
            qc_list.append(torch.zeros((qc_dim,), dtype=torch.float32))
        else:
            qc_list.append(item.qc_features)
    qc_features = torch.stack(qc_list, dim=0)

    llm_dim = 0
    for item in batch:
        llm_feat = getattr(item, "llm_features", None)
        if llm_feat is not None:
            llm_dim = int(llm_feat.numel())
            break
    if llm_dim:
        llm_list = []
        llm_conf_list = []
        exact_covered_list = []
        for item in batch:
            llm_feat = getattr(item, "llm_features", None)
            if llm_feat is None or llm_feat.numel() == 0:
                llm_list.append(torch.zeros((llm_dim,), dtype=torch.float32))
            else:
                llm_list.append(llm_feat)
            llm_conf = getattr(item, "llm_confidence", None)
            if llm_conf is None or (torch.is_tensor(llm_conf) and llm_conf.numel() == 0):
                llm_conf_list.append(torch.tensor([0.0], dtype=torch.float32))
            else:
                llm_conf_list.append(
                    llm_conf if torch.is_tensor(llm_conf) else torch.tensor([llm_conf], dtype=torch.float32)
                )
            exact_covered = getattr(item, "exact_covered_indicator", None)
            if exact_covered is None or (torch.is_tensor(exact_covered) and exact_covered.numel() == 0):
                exact_covered_list.append(torch.tensor([0.0], dtype=torch.float32))
            else:
                exact_covered_list.append(
                    exact_covered
                    if torch.is_tensor(exact_covered)
                    else torch.tensor([exact_covered], dtype=torch.float32)
                )
        llm_features = torch.stack(llm_list, dim=0)
        llm_confidence = torch.stack(llm_conf_list, dim=0)
        exact_covered_indicator = torch.stack(exact_covered_list, dim=0)
    else:
        llm_features = torch.zeros((len(batch), 0), dtype=torch.float32)
        llm_confidence = torch.zeros((len(batch), 1), dtype=torch.float32)
        exact_covered_indicator = torch.zeros((len(batch), 1), dtype=torch.float32)

    interaction_cat_r1_index, interaction_cat_r1_features, interaction_cat_r1_batch = pack_reaction_pairs(
        batch,
        "interaction_cat_r1",
        cat_sizes,
        r1_sizes,
    )
    interaction_cat_r2_index, interaction_cat_r2_features, interaction_cat_r2_batch = pack_reaction_pairs(
        batch,
        "interaction_cat_r2",
        cat_sizes,
        r2_sizes,
    )

    output = {
        "cat_z": cat_z,
        "cat_pos": cat_pos,
        "cat_batch": cat_batch,
        "cat_edge_index": cat_edge_index,
        "r1_z": r1_z,
        "r1_pos": r1_pos,
        "r1_batch": r1_batch,
        "r1_edge_index": r1_edge_index,
        "r2_z": r2_z,
        "r2_pos": r2_pos,
        "r2_batch": r2_batch,
        "r2_edge_index": r2_edge_index,
        "cat_node_qc": cat_node_qc,
        "r1_node_qc": r1_node_qc,
        "r2_node_qc": r2_node_qc,
        "features": features,
        "temperature": torch.stack(
            [
                b.temperature if getattr(b, "temperature", None) is not None else torch.tensor([0.0], dtype=torch.float32)
                for b in batch
            ],
            dim=0,
        ),
        "qc_features": qc_features,
        "llm_features": llm_features,
        "llm_confidence": llm_confidence,
        "exact_covered_indicator": exact_covered_indicator,
        "native_pair_summary": torch.stack(
            [
                getattr(b, "native_pair_summary", None)
                if getattr(b, "native_pair_summary", None) is not None
                else torch.zeros((0,), dtype=torch.float32)
                for b in batch
            ],
            dim=0,
        ),
        "native_pair_tokens": torch.stack(
            [
                getattr(b, "native_pair_tokens", None)
                if getattr(b, "native_pair_tokens", None) is not None
                else torch.zeros((0, 0), dtype=torch.float32)
                for b in batch
            ],
            dim=0,
        ),
        "native_pair_token_mask": torch.stack(
            [
                getattr(b, "native_pair_token_mask", None)
                if getattr(b, "native_pair_token_mask", None) is not None
                else torch.zeros((0,), dtype=torch.float32)
                for b in batch
            ],
            dim=0,
        ),
        "native_pair_expert_summaries": torch.stack(
            [
                getattr(b, "native_pair_expert_summaries", None)
                if getattr(b, "native_pair_expert_summaries", None) is not None
                else torch.zeros((0, 0), dtype=torch.float32)
                for b in batch
            ],
            dim=0,
        ),
        "native_pair_expert_tokens": torch.stack(
            [
                getattr(b, "native_pair_expert_tokens", None)
                if getattr(b, "native_pair_expert_tokens", None) is not None
                else torch.zeros((0, 0, 0), dtype=torch.float32)
                for b in batch
            ],
            dim=0,
        ),
        "native_pair_expert_token_masks": torch.stack(
            [
                getattr(b, "native_pair_expert_token_masks", None)
                if getattr(b, "native_pair_expert_token_masks", None) is not None
                else torch.zeros((0, 0), dtype=torch.float32)
                for b in batch
            ],
            dim=0,
        ),
        "native_pair_expert_priors": torch.stack(
            [
                getattr(b, "native_pair_expert_priors", None)
                if getattr(b, "native_pair_expert_priors", None) is not None
                else torch.zeros((0,), dtype=torch.float32)
                for b in batch
            ],
            dim=0,
        ),
        "interaction_cat_r1_index": interaction_cat_r1_index,
        "interaction_cat_r1_features": interaction_cat_r1_features,
        "interaction_cat_r1_batch": interaction_cat_r1_batch,
        "interaction_cat_r2_index": interaction_cat_r2_index,
        "interaction_cat_r2_features": interaction_cat_r2_features,
        "interaction_cat_r2_batch": interaction_cat_r2_batch,
        "targets": targets,
        "focus_kinetic_relay": torch.stack(
            [
                getattr(b, "focus_kinetic_relay", None)
                if getattr(b, "focus_kinetic_relay", None) is not None
                else torch.tensor([0.0], dtype=torch.float32)
                for b in batch
            ],
            dim=0,
        ),
        "electronic_frontier_score": torch.stack(
            [
                getattr(b, "electronic_frontier_score", None)
                if getattr(b, "electronic_frontier_score", None) is not None
                else torch.tensor([0.0], dtype=torch.float32)
                for b in batch
            ],
            dim=0,
        ),
    }
    reaction_target_dim = 0
    for item in batch:
        reaction_targets = getattr(item, "reaction_center_targets", None)
        if reaction_targets is not None and reaction_targets.numel() > 0:
            reaction_target_dim = int(reaction_targets.numel())
            break
    reaction_center_targets = []
    for item in batch:
        reaction_targets = getattr(item, "reaction_center_targets", None)
        if reaction_targets is None or reaction_targets.numel() == 0:
            reaction_center_targets.append(torch.zeros((reaction_target_dim,), dtype=torch.float32))
        else:
            reaction_center_targets.append(reaction_targets)
    if reaction_target_dim > 0:
        output["reaction_center_targets"] = torch.stack(reaction_center_targets, dim=0)
    else:
        output["reaction_center_targets"] = torch.zeros((len(batch), 0), dtype=torch.float32)
    cat_r1_index, cat_r1_features, cat_r1_pair_batch = pack_reaction_pairs(
        batch,
        "reaction_center_cat_r1",
        cat_sizes,
        r1_sizes,
    )
    cat_r2_index, cat_r2_features, cat_r2_pair_batch = pack_reaction_pairs(
        batch,
        "reaction_center_cat_r2",
        cat_sizes,
        r2_sizes,
    )
    output.update(
        {
            "reaction_center_cat_r1_index": cat_r1_index,
            "reaction_center_cat_r1_features": cat_r1_features,
            "reaction_center_cat_r1_batch": cat_r1_pair_batch,
            "reaction_center_cat_r2_index": cat_r2_index,
            "reaction_center_cat_r2_features": cat_r2_features,
            "reaction_center_cat_r2_batch": cat_r2_pair_batch,
        }
    )
    if include_group:
        group_ids = [b.group_id for b in batch]
        if any(gid is None for gid in group_ids):
            raise ValueError("Group ids missing in batch; ensure dataset configured with group_cols.")
        output["group_ids"] = torch.tensor(group_ids, dtype=torch.long)
    if include_rank_target:
        rank_targets = [b.rank_target for b in batch]
        if any(rt is None for rt in rank_targets):
            raise ValueError("Rank targets missing in batch; ensure dataset configured with rank_target_col.")
        output["rank_targets"] = torch.stack(rank_targets, dim=0)
    if include_combined:
        cz = []
        cp = []
        ctype = []
        cedge_src = []
        cedge_dst = []
        cedge_attr = []
        cbatch = []
        cedge_batch = []
        offset = 0
        for i, item in enumerate(batch):
            if item.combined_z is None:
                cz.append(torch.tensor([0], dtype=torch.long))
                cp.append(torch.zeros((1, 3), dtype=torch.float32))
                ctype.append(torch.tensor([0], dtype=torch.long))
                cedge = torch.zeros((2, 0), dtype=torch.long)
                cattr = torch.zeros((0, 3), dtype=torch.float32)
            else:
                cedge = item.combined_edge_index
                cattr = item.combined_edge_attr
                cz.append(item.combined_z)
                cp.append(item.combined_pos)
                ctype.append(item.combined_node_type)
            num_nodes = cz[-1].size(0)
            cbatch.append(torch.full((num_nodes,), i, dtype=torch.long))
            if cedge.numel() > 0:
                cedge_src.append(cedge[0] + offset)
                cedge_dst.append(cedge[1] + offset)
                cedge_attr.append(cattr)
                cedge_batch.append(torch.full((cedge.size(1),), i, dtype=torch.long))
            offset += num_nodes
        combined_z = torch.cat(cz, dim=0)
        combined_pos = torch.cat(cp, dim=0)
        combined_node_type = torch.cat(ctype, dim=0)
        combined_batch = torch.cat(cbatch, dim=0)
        if cedge_src:
            combined_edge_index = torch.stack([torch.cat(cedge_src), torch.cat(cedge_dst)], dim=0)
            combined_edge_attr = torch.cat(cedge_attr, dim=0)
            combined_edge_batch = torch.cat(cedge_batch, dim=0)
        else:
            combined_edge_index = torch.zeros((2, 0), dtype=torch.long)
            combined_edge_attr = torch.zeros((0, 3), dtype=torch.float32)
            combined_edge_batch = torch.zeros((0,), dtype=torch.long)
        output.update(
            {
                "combined_z": combined_z,
                "combined_pos": combined_pos,
                "combined_node_type": combined_node_type,
                "combined_batch": combined_batch,
                "combined_edge_index": combined_edge_index,
                "combined_edge_attr": combined_edge_attr,
                "combined_edge_batch": combined_edge_batch,
            }
        )
    if include_literature:
        output["cat_interactions"] = [
            (b.literature or {}).get("cat") for b in batch
        ]
        output["r1_interactions"] = [
            (b.literature or {}).get("r1") for b in batch
        ]
        output["r2_interactions"] = [
            (b.literature or {}).get("r2") for b in batch
        ]
        if llm_bias_scale != 0.0:
            rules = AlignmentRules()
            cat_sizes = component_sizes(batch, "cat")
            r1_sizes = component_sizes(batch, "r1")
            r2_sizes = component_sizes(batch, "r2")
            cross_bias = {"cat_r1": [], "cat_r2": [], "r1_cat": [], "r2_cat": []}
            cross_bias_stats = {key: [] for key in cross_bias}
            for idx, sample in enumerate(batch):
                literature = sample.literature or {}
                cat_inter = literature.get("cat") or []
                r1_inter = literature.get("r1") or []
                r2_inter = literature.get("r2") or []
                cat_scores = build_hotspot_scores(
                    cat_inter,
                    cat_sizes[idx],
                    rules=rules,
                    aggregation=llm_bias_aggregation,
                )
                r1_scores = build_hotspot_scores(
                    r1_inter,
                    r1_sizes[idx],
                    rules=rules,
                    aggregation=llm_bias_aggregation,
                )
                r2_scores = build_hotspot_scores(
                    r2_inter,
                    r2_sizes[idx],
                    rules=rules,
                    aggregation=llm_bias_aggregation,
                )

                bias, q_boost, k_boost = build_attention_bias_from_scores(
                    cat_scores,
                    r1_scores,
                    scale=llm_bias_scale,
                    mode=llm_bias_mode,
                )
                cross_bias["cat_r1"].append(bias)
                if attn_audit:
                    stats = summarize_attention_bias(bias, k_boost)
                    stats["query_hotspots"] = int((q_boost > 0).sum().item())
                    stats["key_hotspots"] = int((k_boost > 0).sum().item())
                    cross_bias_stats["cat_r1"].append(stats)

                bias, q_boost, k_boost = build_attention_bias_from_scores(
                    cat_scores,
                    r2_scores,
                    scale=llm_bias_scale,
                    mode=llm_bias_mode,
                )
                cross_bias["cat_r2"].append(bias)
                if attn_audit:
                    stats = summarize_attention_bias(bias, k_boost)
                    stats["query_hotspots"] = int((q_boost > 0).sum().item())
                    stats["key_hotspots"] = int((k_boost > 0).sum().item())
                    cross_bias_stats["cat_r2"].append(stats)

                bias, q_boost, k_boost = build_attention_bias_from_scores(
                    r1_scores,
                    cat_scores,
                    scale=llm_bias_scale,
                    mode=llm_bias_mode,
                )
                cross_bias["r1_cat"].append(bias)
                if attn_audit:
                    stats = summarize_attention_bias(bias, k_boost)
                    stats["query_hotspots"] = int((q_boost > 0).sum().item())
                    stats["key_hotspots"] = int((k_boost > 0).sum().item())
                    cross_bias_stats["r1_cat"].append(stats)

                bias, q_boost, k_boost = build_attention_bias_from_scores(
                    r2_scores,
                    cat_scores,
                    scale=llm_bias_scale,
                    mode=llm_bias_mode,
                )
                cross_bias["r2_cat"].append(bias)
                if attn_audit:
                    stats = summarize_attention_bias(bias, k_boost)
                    stats["query_hotspots"] = int((q_boost > 0).sum().item())
                    stats["key_hotspots"] = int((k_boost > 0).sum().item())
                    cross_bias_stats["r2_cat"].append(stats)

            output["cross_bias"] = cross_bias
            if attn_audit:
                output["cross_bias_stats"] = cross_bias_stats
    return output


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if y_true.size == 0:
        return {"mae": 0.0, "r2": 0.0, "pearson": 0.0}
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    if y_true.size < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        pearson = 0.0
    else:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
        if np.isnan(pearson):
            pearson = 0.0
    return {"mae": mae, "r2": r2, "pearson": pearson}


def _extract_llm_confidence(samples: Sequence[object]) -> np.ndarray:
    values = []
    for sample in samples:
        conf = getattr(sample, "llm_confidence", None)
        if conf is None:
            values.append(0.0)
        elif torch.is_tensor(conf):
            values.append(float(conf.view(-1)[0].item()) if conf.numel() > 0 else 0.0)
        else:
            try:
                values.append(float(conf))
            except (TypeError, ValueError):
                values.append(0.0)
    return np.asarray(values, dtype=np.float32)


def _subset_metrics(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    if mask.size == 0:
        return {"count": 0, "fraction": 0.0, "mae": 0.0, "r2": 0.0, "pearson": 0.0}
    count = int(mask.sum())
    fraction = float(mask.mean()) if mask.size else 0.0
    if count == 0:
        metrics = {"mae": 0.0, "r2": 0.0, "pearson": 0.0}
    else:
        metrics = compute_metrics(y_true[mask], y_pred[mask])
    return {"count": count, "fraction": fraction, **metrics}


def _get_interaction_field(interaction: object, key: str, default: object = None) -> object:
    if isinstance(interaction, dict):
        return interaction.get(key, default)
    return getattr(interaction, key, default)


def _init_phys_stats() -> Dict[str, object]:
    return {
        "total": 0,
        "mapped": 0,
        "within": 0,
        "distance_sum": 0.0,
        "distance_count": 0,
        "delta_sum": 0.0,
        "delta_count": 0,
        "min_distance": None,
        "max_distance": None,
    }


def _update_phys_stats(
    stats: Dict[str, object],
    pos: torch.Tensor,
    interactions: Optional[Sequence[object]],
    cutoff: float,
) -> None:
    if interactions is None:
        return
    if not torch.is_tensor(pos) or pos.numel() == 0:
        return
    for inter in interactions:
        stats["total"] = int(stats["total"]) + 1
        atom_indices = _get_interaction_field(inter, "atom_indices")
        if not atom_indices or len(atom_indices) != 2:
            continue
        try:
            i = int(atom_indices[0])
            j = int(atom_indices[1])
        except (TypeError, ValueError):
            continue
        if i == j or i < 0 or j < 0 or i >= pos.size(0) or j >= pos.size(0):
            continue
        stats["mapped"] = int(stats["mapped"]) + 1
        dist = float(torch.norm(pos[i] - pos[j]).item())
        stats["distance_sum"] = float(stats["distance_sum"]) + dist
        stats["distance_count"] = int(stats["distance_count"]) + 1
        if dist <= cutoff:
            stats["within"] = int(stats["within"]) + 1
        min_dist = stats["min_distance"]
        max_dist = stats["max_distance"]
        stats["min_distance"] = dist if min_dist is None else min(min_dist, dist)
        stats["max_distance"] = dist if max_dist is None else max(max_dist, dist)
        llm_dist = _get_interaction_field(inter, "distance")
        if llm_dist is not None:
            try:
                delta = abs(dist - float(llm_dist))
            except (TypeError, ValueError):
                delta = None
            if delta is not None:
                stats["delta_sum"] = float(stats["delta_sum"]) + float(delta)
                stats["delta_count"] = int(stats["delta_count"]) + 1


def _finalize_phys_stats(stats: Dict[str, object]) -> Dict[str, float]:
    total = int(stats["total"])
    mapped = int(stats["mapped"])
    within = int(stats["within"])
    distance_count = int(stats["distance_count"])
    delta_count = int(stats["delta_count"])
    mapped_ratio = mapped / total if total else 0.0
    within_ratio = within / mapped if mapped else 0.0
    mean_distance = float(stats["distance_sum"]) / distance_count if distance_count else 0.0
    mean_delta = float(stats["delta_sum"]) / delta_count if delta_count else 0.0
    min_distance = stats["min_distance"]
    max_distance = stats["max_distance"]
    return {
        "total_interactions": float(total),
        "mapped_interactions": float(mapped),
        "mapped_ratio": float(mapped_ratio),
        "within_cutoff": float(within),
        "within_cutoff_ratio": float(within_ratio),
        "mean_distance": float(mean_distance),
        "min_distance": float(min_distance) if min_distance is not None else 0.0,
        "max_distance": float(max_distance) if max_distance is not None else 0.0,
        "mean_abs_distance_error": float(mean_delta),
    }


def summarize_literature_physical_consistency(dataset, cutoff: float) -> Dict[str, Dict[str, float]]:
    stats = {
        "overall": _init_phys_stats(),
        "cat": _init_phys_stats(),
        "r1": _init_phys_stats(),
        "r2": _init_phys_stats(),
    }
    for sample in dataset.samples:
        literature = sample.literature or {}
        _update_phys_stats(stats["cat"], sample.cat_pos, literature.get("cat"), cutoff)
        _update_phys_stats(stats["r1"], sample.r1_pos, literature.get("r1"), cutoff)
        _update_phys_stats(stats["r2"], sample.r2_pos, literature.get("r2"), cutoff)
        _update_phys_stats(stats["overall"], sample.cat_pos, literature.get("cat"), cutoff)
        _update_phys_stats(stats["overall"], sample.r1_pos, literature.get("r1"), cutoff)
        _update_phys_stats(stats["overall"], sample.r2_pos, literature.get("r2"), cutoff)
    return {key: _finalize_phys_stats(value) for key, value in stats.items()}


def _load_summary(path: Optional[str]) -> Optional[dict]:
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        return None
    if isinstance(payload, list):
        return payload[-1] if payload else None
    if isinstance(payload, dict):
        return payload
    return None


def _attach_baseline_delta(
    summary: dict,
    baseline_path: Optional[str],
    current_r2: float,
    metric_label: str,
) -> None:
    baseline_summary = _load_summary(baseline_path)
    if not baseline_summary:
        return
    baseline_metrics = baseline_summary.get("test") or baseline_summary.get("val") or {}
    baseline_r2 = baseline_metrics.get("r2")
    if baseline_r2 is None:
        return
    delta = current_r2 - float(baseline_r2)
    summary["r2_delta"] = {
        "baseline_path": baseline_path,
        "baseline_r2": float(baseline_r2),
        "current_r2": float(current_r2),
        "delta": float(delta),
        "metric": metric_label,
    }


def _grad_norm(params: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    for param in params:
        if param.grad is None:
            continue
        total += float(param.grad.detach().pow(2).sum().item())
    return math.sqrt(total)


def build_grad_groups(model: nn.Module) -> Dict[str, Sequence[torch.nn.Parameter]]:
    groups: Dict[str, Sequence[torch.nn.Parameter]] = {}
    encoder_modules = []
    encoder_final_layer_params = []
    for name in ("cat_encoder", "reactant_encoder", "encoder", "layers"):
        module = getattr(model, name, None)
        if module is not None:
            encoder_modules.append(module)
            layers = getattr(module, "layers", None)
            if layers is not None and len(layers) > 0:
                start = max(0, len(layers) - 2)
                for layer in list(layers)[start:]:
                    encoder_final_layer_params.extend([p for p in layer.parameters() if p.requires_grad])
    if encoder_modules:
        params = [p for module in encoder_modules for p in module.parameters() if p.requires_grad]
        if params:
            groups["encoder"] = params
    if encoder_final_layer_params:
        groups["encoder_final_layers"] = encoder_final_layer_params
    cross_module = getattr(model, "cross_attention", None)
    if cross_module is not None:
        params = [p for p in cross_module.parameters() if p.requires_grad]
        if params:
            groups["cross_attention"] = params
    interaction_module = getattr(model, "interaction_cross_attention", None)
    if interaction_module is not None:
        params = [p for p in interaction_module.parameters() if p.requires_grad]
        if params:
            groups["interaction"] = params
    reaction_center_module = getattr(model, "reaction_center_coupling", None)
    if reaction_center_module is not None:
        params = [p for p in reaction_center_module.parameters() if p.requires_grad]
        if params:
            groups["reaction_center"] = params
    fusion_module = getattr(model, "combine_mlp", None)
    if fusion_module is not None:
        params = [p for p in fusion_module.parameters() if p.requires_grad]
        if params:
            groups["fusion"] = params
    feat_module = getattr(model, "feat_proj", None)
    if feat_module is not None:
        params = [p for p in feat_module.parameters() if p.requires_grad]
        if params:
            groups["feature_proj"] = params
    head_module = getattr(model, "head", None)
    if head_module is not None:
        params = [p for p in head_module.parameters() if p.requires_grad]
        if params:
            groups["head"] = params
    return groups


def log_grad_audit(log, grad_tracker: Dict[str, float]) -> None:
    group_entries = {k: v for k, v in grad_tracker.items() if k.startswith("grad_group_")}
    if group_entries:
        parts = []
        for key in sorted(group_entries):
            label = key.replace("grad_group_", "")
            parts.append(f"{label}={group_entries[key]:.4f}")
        log("Grad group norms: " + " | ".join(parts))
        enc = group_entries.get("grad_group_encoder")
        feat = group_entries.get("grad_group_feature_proj")
        if enc is not None and feat is not None:
            ratio = enc / (feat + 1e-8)
            log(f"Grad ratio encoder/feature_proj: {ratio:.3f}")
    if "feature_grad_qc_mean" in grad_tracker:
        qc_mean = grad_tracker.get("feature_grad_qc_mean", 0.0)
        non_qc_mean = grad_tracker.get("feature_grad_non_qc_mean", 0.0)
        ratio = grad_tracker.get("feature_grad_qc_ratio", 0.0)
        log(
            "Feature grad mean | qc={:.4e} | non_qc={:.4e} | qc/non_qc={:.3f}".format(
                qc_mean, non_qc_mean, ratio
            )
        )


def log_attention_audit(log, grad_tracker: Dict[str, float]) -> None:
    ratio_entries = {k: v for k, v in grad_tracker.items() if k.startswith("attn_bias_ratio_")}
    if not ratio_entries:
        return
    parts = []
    for key in sorted(ratio_entries):
        pair = key.replace("attn_bias_ratio_", "")
        ratio = ratio_entries[key]
        share = grad_tracker.get(f"attn_bias_share_{pair}", 0.0)
        hotspots = grad_tracker.get(f"attn_bias_hotspots_{pair}", 0.0)
        parts.append(f"{pair}:ratio={ratio:.2f} share={share:.2f} hotspots={hotspots:.1f}")
    log("LLM hotspot attention bias: " + " | ".join(parts))


class GroupBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        group_ids,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.group_to_indices: Dict[int, list[int]] = {}
        for idx, gid in enumerate(group_ids):
            if gid is None:
                continue
            self.group_to_indices.setdefault(int(gid), []).append(idx)

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        batches = []
        for indices in self.group_to_indices.values():
            indices = list(indices)
            if self.shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self) -> int:
        total = 0
        for indices in self.group_to_indices.values():
            count = len(indices) // self.batch_size
            if not self.drop_last and len(indices) % self.batch_size:
                count += 1
            total += count
        return total


def _ranknet_loss(preds: torch.Tensor, targets: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    preds = preds.view(-1)
    targets = targets.view(-1)
    if preds.numel() < 2:
        return preds.new_tensor(0.0)
    target_diff = targets[:, None] - targets[None, :]
    pos_mask = target_diff > 0
    if not torch.any(pos_mask):
        return preds.new_tensor(0.0)
    pred_diff = preds[:, None] - preds[None, :]
    if margin:
        pred_diff = pred_diff - margin
    losses = torch.nn.functional.softplus(-pred_diff)
    return losses[pos_mask].mean()


def _listnet_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    preds = preds.view(-1)
    targets = targets.view(-1)
    if preds.numel() == 0:
        return preds.new_tensor(0.0)
    true_prob = torch.softmax(targets, dim=0)
    pred_prob = torch.softmax(preds, dim=0)
    return -(true_prob * torch.log(pred_prob.clamp_min(1e-8))).sum()


def _pairwise_hinge_loss(preds: torch.Tensor, targets: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    preds = preds.view(-1)
    targets = targets.view(-1)
    if preds.numel() < 2:
        return preds.new_tensor(0.0)
    target_diff = targets[:, None] - targets[None, :]
    pos_mask = target_diff > 0
    if not torch.any(pos_mask):
        return preds.new_tensor(0.0)
    pred_diff = preds[:, None] - preds[None, :]
    losses = torch.relu(margin - pred_diff)
    return losses[pos_mask].mean()


def _groupwise_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    group_ids: Optional[torch.Tensor],
    loss_fn,
) -> torch.Tensor:
    if group_ids is None:
        return preds.new_tensor(0.0)
    group_ids = group_ids.view(-1)
    preds = preds.view(-1)
    targets = targets.view(-1)
    losses = []
    for gid in torch.unique(group_ids):
        mask = group_ids == gid
        if mask.sum() < 2:
            continue
        loss_val = loss_fn(preds[mask], targets[mask])
        losses.append(loss_val)
    if not losses:
        return preds.new_tensor(0.0)
    return torch.stack(losses).mean()


def _llm_strength_rank_regularizer(
    preds: torch.Tensor,
    llm_features: Optional[torch.Tensor],
    llm_confidence: Optional[torch.Tensor],
    min_confidence: float = 0.0,
) -> torch.Tensor:
    if llm_features is None or llm_features.numel() == 0:
        return preds.new_tensor(0.0)
    preds = preds.view(-1)
    if preds.numel() < 2 or llm_features.size(0) != preds.size(0):
        return preds.new_tensor(0.0)
    if llm_features.size(1) < 2:
        return preds.new_tensor(0.0)
    strength = llm_features[:, 1].view(-1)
    if llm_confidence is None or llm_confidence.numel() == 0:
        confidence = torch.ones_like(strength)
    else:
        confidence = llm_confidence.view(-1)
        if confidence.size(0) != preds.size(0):
            return preds.new_tensor(0.0)
    valid = confidence >= float(min_confidence)
    if valid.sum() < 2:
        return preds.new_tensor(0.0)
    preds = preds[valid]
    strength = strength[valid]
    confidence = confidence[valid]
    strength_diff = strength[:, None] - strength[None, :]
    pair_mask = strength_diff.abs() > 1e-6
    if not torch.any(pair_mask):
        return preds.new_tensor(0.0)
    pred_diff = preds.abs()[:, None] - preds.abs()[None, :]
    direction = torch.sign(strength_diff)
    conf_weight = confidence[:, None] * confidence[None, :]
    losses = torch.nn.functional.softplus(-direction * pred_diff) * conf_weight
    return losses[pair_mask].mean()


def _move_batch_to_device(batch: dict, device: torch.device) -> None:
    for key, value in batch.items():
        if torch.is_tensor(value):
            batch[key] = value.to(device)
    if "cross_bias" in batch:
        moved_bias = {}
        for key, entries in batch["cross_bias"].items():
            moved_entries = []
            for bias in entries:
                moved_entries.append(bias.to(device) if torch.is_tensor(bias) else bias)
            moved_bias[key] = moved_entries
        batch["cross_bias"] = moved_bias


def _update_batchnorm_stats(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    input_keys: Optional[Sequence[str]],
) -> None:
    batchnorm_modules = []
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean.zero_()
            module.running_var.fill_(1)
            module.num_batches_tracked.zero_()
            batchnorm_modules.append(module)
    if not batchnorm_modules:
        return
    model.train()
    with torch.no_grad():
        for batch in loader:
            _move_batch_to_device(batch, device)
            batch.pop("targets", None)
            batch.pop("rank_targets", None)
            batch.pop("group_ids", None)
            if input_keys is None:
                model_kwargs = batch
            else:
                model_kwargs = {key: batch[key] for key in input_keys}
            model(**model_kwargs)


def _build_mse_sample_weights(
    preds: torch.Tensor,
    targets: torch.Tensor,
    *,
    llm_confidence: Optional[torch.Tensor] = None,
    exact_covered_indicator: Optional[torch.Tensor] = None,
    focus_exact_covered_weight: float = 0.0,
    focus_confidence_threshold: float = 0.0,
    focus_confidence_weight: float = 0.0,
    llm_reliability_loss_weight: float = 0.0,
    llm_reliability_loss_center: float = 0.5,
    llm_reliability_loss_sharpness: float = 8.0,
    focus_yield_threshold: Optional[float] = None,
    focus_yield_weight: float = 0.0,
    r1_node_qc: Optional[torch.Tensor] = None,
    r1_batch: Optional[torch.Tensor] = None,
    coordination_feature_dim: int = 0,
    focus_coordination_threshold: Optional[float] = None,
    focus_coordination_weight: float = 0.0,
    electronic_frontier_score: Optional[torch.Tensor] = None,
    focus_electronic_frontier_weight: float = 0.0,
    focus_kinetic_relay: Optional[torch.Tensor] = None,
    focus_kinetic_relay_weight: float = 0.0,
) -> torch.Tensor:
    sample_weights = torch.ones_like(preds)
    if focus_exact_covered_weight > 0.0 and exact_covered_indicator is not None and exact_covered_indicator.numel() > 0:
        exact_mask = (exact_covered_indicator.view(-1, 1) > 0.5).to(preds.dtype)
        sample_weights = sample_weights * (1.0 + float(focus_exact_covered_weight) * exact_mask)
    if focus_confidence_weight > 0.0 and llm_confidence is not None and llm_confidence.numel() > 0:
        conf_mask = (llm_confidence.view(-1, 1) >= float(focus_confidence_threshold)).to(preds.dtype)
        sample_weights = sample_weights * (1.0 + float(focus_confidence_weight) * conf_mask)
    if llm_reliability_loss_weight > 0.0 and llm_confidence is not None and llm_confidence.numel() > 0:
        confidence = llm_confidence.view(-1, 1).to(device=preds.device, dtype=preds.dtype).clamp_(0.0, 1.0)
        sharpness = max(float(llm_reliability_loss_sharpness), 1e-6)
        reliability_gate = torch.sigmoid((confidence - float(llm_reliability_loss_center)) * sharpness)
        sample_weights = sample_weights * (1.0 + float(llm_reliability_loss_weight) * reliability_gate)
    if focus_yield_weight > 0.0 and focus_yield_threshold is not None:
        yield_mask = (targets >= float(focus_yield_threshold)).to(preds.dtype)
        sample_weights = sample_weights * (1.0 + float(focus_yield_weight) * yield_mask)
    if (
        focus_coordination_weight > 0.0
        and focus_coordination_threshold is not None
        and coordination_feature_dim > 0
        and r1_node_qc is not None
        and r1_batch is not None
        and r1_node_qc.numel() > 0
        and r1_batch.numel() > 0
    ):
        coord = r1_node_qc[:, -int(coordination_feature_dim) :]
        active = coord[:, 0]
        potency = coord[:, 4] if coord.size(1) > 4 else coord[:, 0]
        signal = active * potency
        num_graphs = int(r1_batch.max().item()) + 1
        signal_sum = torch.zeros((num_graphs,), device=signal.device, dtype=signal.dtype)
        signal_sum.index_add_(0, r1_batch, signal)
        counts = torch.bincount(r1_batch, minlength=num_graphs).clamp(min=1).to(signal.dtype)
        mean_signal = (signal_sum / counts).view(-1, 1)
        coord_mask = (mean_signal >= float(focus_coordination_threshold)).to(preds.dtype)
        sample_weights = sample_weights * (1.0 + float(focus_coordination_weight) * coord_mask)
    if (
        focus_electronic_frontier_weight > 0.0
        and electronic_frontier_score is not None
        and electronic_frontier_score.numel() > 0
    ):
        frontier_scale = electronic_frontier_score.view(-1, 1).to(device=preds.device, dtype=preds.dtype).clamp_(0.0, 1.0)
        sample_weights = sample_weights * (1.0 + float(focus_electronic_frontier_weight) * frontier_scale)
    if focus_kinetic_relay_weight > 0.0 and focus_kinetic_relay is not None and focus_kinetic_relay.numel() > 0:
        relay_mask = (focus_kinetic_relay.view(-1, 1) > 0.5).to(preds.dtype)
        sample_weights = sample_weights * (1.0 + float(focus_kinetic_relay_weight) * relay_mask)
    return sample_weights


def _apply_row_level_llm_dropout(
    model_kwargs: Dict[str, torch.Tensor],
    *,
    dropout_p: float = 0.0,
    confidence_scale: float = 0.0,
    exact_scale: float = 1.0,
    propagated_scale: float = 1.0,
) -> Dict[str, torch.Tensor]:
    if dropout_p <= 0.0:
        return model_kwargs
    llm_features = model_kwargs.get("llm_features")
    if llm_features is None or not torch.is_tensor(llm_features) or llm_features.numel() == 0:
        return model_kwargs

    effective_dropout_p = max(0.0, min(1.0, float(dropout_p)))
    confidence_scale = max(0.0, min(1.0, float(confidence_scale)))
    exact_scale = max(0.0, float(exact_scale))
    propagated_scale = max(0.0, float(propagated_scale))
    confidence = model_kwargs.get("llm_confidence")
    exact_covered_indicator = model_kwargs.get("exact_covered_indicator")
    feature_mask = (llm_features.abs().sum(dim=1, keepdim=True) > 0).to(dtype=llm_features.dtype)

    if confidence is not None and torch.is_tensor(confidence) and confidence.numel() > 0:
        confidence_values = confidence.view(-1, 1).to(device=llm_features.device, dtype=llm_features.dtype).clamp_(0.0, 1.0)
    else:
        confidence_values = torch.zeros((llm_features.size(0), 1), device=llm_features.device, dtype=llm_features.dtype)

    if (
        exact_covered_indicator is not None
        and torch.is_tensor(exact_covered_indicator)
        and exact_covered_indicator.numel() > 0
    ):
        exact_mask = (
            exact_covered_indicator.view(-1, 1)
            .to(device=llm_features.device, dtype=llm_features.dtype)
            .clamp_(0.0, 1.0)
        )
    else:
        exact_mask = torch.zeros((llm_features.size(0), 1), device=llm_features.device, dtype=llm_features.dtype)
    provenance_scale = exact_mask * exact_scale + (1.0 - exact_mask) * propagated_scale
    adjusted_dropout = effective_dropout_p * (1.0 - confidence_scale * confidence_values)
    adjusted_dropout = adjusted_dropout * provenance_scale
    adjusted_dropout = adjusted_dropout.clamp_(0.0, 1.0) * feature_mask
    keep_mask = (torch.rand_like(adjusted_dropout) >= adjusted_dropout).to(dtype=llm_features.dtype)

    dropped_kwargs = dict(model_kwargs)
    dropped_kwargs["llm_features"] = llm_features * keep_mask
    if confidence is not None and torch.is_tensor(confidence) and confidence.numel() > 0:
        dropped_kwargs["llm_confidence"] = confidence_values * keep_mask
    return dropped_kwargs


def run_epoch(
    model,
    loader,
    device,
    optimizer=None,
    input_keys=None,
    loss_mode: str = "mse",
    rank_weight: float = 0.2,
    rank_margin: float = 0.0,
    attn_audit: bool = False,
    grad_tracker: Optional[Dict[str, float]] = None,
    grad_groups: Optional[Dict[str, Sequence[torch.nn.Parameter]]] = None,
    feature_slice: Optional[slice] = None,
    llm_rule_weight: float = 0.0,
    llm_rule_min_confidence: float = 0.0,
    focus_exact_covered_weight: float = 0.0,
    focus_confidence_threshold: float = 0.0,
    focus_confidence_weight: float = 0.0,
    llm_row_dropout_p: float = 0.0,
    llm_row_dropout_confidence_scale: float = 0.0,
    llm_row_dropout_exact_scale: float = 1.0,
    llm_row_dropout_propagated_scale: float = 1.0,
    llm_reliability_loss_weight: float = 0.0,
    llm_reliability_loss_center: float = 0.5,
    llm_reliability_loss_sharpness: float = 8.0,
    focus_yield_threshold: Optional[float] = None,
    focus_yield_weight: float = 0.0,
    coordination_feature_dim: int = 0,
    focus_coordination_threshold: Optional[float] = None,
    focus_coordination_weight: float = 0.0,
    focus_electronic_frontier_weight: float = 0.0,
    focus_kinetic_relay_weight: float = 0.0,
    coord_jitter_std: float = 0.0,
    reaction_center_aux_weight: float = 0.0,
    exact_covered_consistency_weight: float = 0.0,
    llm_exact_row_scale: float = 1.0,
    llm_propagated_row_scale: float = 1.0,
    base_llm_exact_row_scale: float = 1.0,
    base_llm_propagated_row_scale: float = 1.0,
    llm_propagated_row_confidence_floor: float = 1.0,
    llm_propagated_row_confidence_power: float = 1.0,
    llm_propagated_row_temperature_floor: float = 1.0,
    llm_propagated_row_temperature_power: float = 1.0,
):
    training = optimizer is not None
    if training:
        model.train()
    else:
        model.eval()

    if len(loader.dataset) == 0:
        return 0.0, np.array([]), np.array([])

    y_true = []
    y_pred = []
    total_loss = 0.0
    criterion = nn.MSELoss()

    for batch in loader:
        _move_batch_to_device(batch, device)
        targets = batch.pop("targets")
        rank_targets = batch.pop("rank_targets", None)
        if rank_targets is None:
            rank_targets = targets
        group_ids = batch.pop("group_ids", None)

        feature_for_grad = None
        if (
            training
            and grad_tracker is not None
            and feature_slice is not None
            and not grad_tracker.get("feature_grad_done")
        ):
            features = batch.get("features")
            if features is not None and torch.is_tensor(features):
                features.requires_grad_(True)
                feature_for_grad = features

        if input_keys is None:
            model_kwargs = batch
        else:
            model_kwargs = {k: batch[k] for k in input_keys}
        if training and (coord_jitter_std > 0.0 or llm_row_dropout_p > 0.0):
            model_kwargs = dict(model_kwargs)
        if training and coord_jitter_std > 0.0:
            jitter_scale = float(coord_jitter_std)
            for key in ("cat_pos", "r1_pos", "r2_pos", "combined_pos"):
                value = model_kwargs.get(key)
                if value is None or not torch.is_tensor(value) or value.numel() == 0:
                    continue
                model_kwargs[key] = value + torch.randn_like(value) * jitter_scale
        if training and llm_row_dropout_p > 0.0:
            model_kwargs = _apply_row_level_llm_dropout(
                model_kwargs,
                dropout_p=llm_row_dropout_p,
                confidence_scale=llm_row_dropout_confidence_scale,
                exact_scale=llm_row_dropout_exact_scale,
                propagated_scale=llm_row_dropout_propagated_scale,
            )
        model_kwargs = _apply_llm_row_authority_schedule(
            model_kwargs,
            exact_scale=float(llm_exact_row_scale),
            propagated_scale=float(llm_propagated_row_scale),
            base_exact_scale=float(base_llm_exact_row_scale),
            base_propagated_scale=float(base_llm_propagated_row_scale),
            propagated_confidence_floor=float(llm_propagated_row_confidence_floor),
            propagated_confidence_power=float(llm_propagated_row_confidence_power),
            propagated_temperature_floor=float(llm_propagated_row_temperature_floor),
            propagated_temperature_power=float(llm_propagated_row_temperature_power),
        )
        preds = model(**model_kwargs)
        mse_loss = criterion(preds, targets)
        sample_weights = _build_mse_sample_weights(
            preds,
            targets,
            llm_confidence=batch.get("llm_confidence"),
            exact_covered_indicator=batch.get("exact_covered_indicator"),
            focus_exact_covered_weight=focus_exact_covered_weight,
            focus_confidence_threshold=focus_confidence_threshold,
            focus_confidence_weight=focus_confidence_weight,
            llm_reliability_loss_weight=llm_reliability_loss_weight,
            llm_reliability_loss_center=llm_reliability_loss_center,
            llm_reliability_loss_sharpness=llm_reliability_loss_sharpness,
            focus_yield_threshold=focus_yield_threshold,
            focus_yield_weight=focus_yield_weight,
            r1_node_qc=batch.get("r1_node_qc"),
            r1_batch=batch.get("r1_batch"),
            coordination_feature_dim=coordination_feature_dim,
            focus_coordination_threshold=focus_coordination_threshold,
            focus_coordination_weight=focus_coordination_weight,
            electronic_frontier_score=batch.get("electronic_frontier_score"),
            focus_electronic_frontier_weight=focus_electronic_frontier_weight,
            focus_kinetic_relay=batch.get("focus_kinetic_relay"),
            focus_kinetic_relay_weight=focus_kinetic_relay_weight,
        )
        if not torch.allclose(sample_weights, torch.ones_like(sample_weights)):
            mse_loss = (((preds - targets) ** 2) * sample_weights).mean()
        if loss_mode == "mse":
            loss = mse_loss
        elif loss_mode == "ranknet":
            loss = _ranknet_loss(preds, rank_targets, margin=rank_margin)
        elif loss_mode == "listnet":
            loss = _listnet_loss(preds, rank_targets)
        elif loss_mode == "group_ranknet":
            loss = _groupwise_loss(
                preds,
                rank_targets,
                group_ids,
                lambda p, t: _ranknet_loss(p, t, margin=rank_margin),
            )
        elif loss_mode == "pairwise_hinge":
            loss = _pairwise_hinge_loss(preds, rank_targets, margin=rank_margin)
        elif loss_mode == "group_pairwise_hinge":
            loss = _groupwise_loss(
                preds,
                rank_targets,
                group_ids,
                lambda p, t: _pairwise_hinge_loss(p, t, margin=rank_margin),
            )
        elif loss_mode == "group_listnet":
            loss = _groupwise_loss(preds, rank_targets, group_ids, _listnet_loss)
        elif loss_mode == "hybrid_ranknet":
            loss = mse_loss + rank_weight * _ranknet_loss(preds, rank_targets, margin=rank_margin)
        elif loss_mode == "hybrid_listnet":
            loss = mse_loss + rank_weight * _listnet_loss(preds, rank_targets)
        elif loss_mode == "hybrid_pairwise_hinge":
            loss = mse_loss + rank_weight * _pairwise_hinge_loss(
                preds,
                rank_targets,
                margin=rank_margin,
            )
        elif loss_mode == "hybrid_group_ranknet":
            loss = mse_loss + rank_weight * _groupwise_loss(
                preds,
                rank_targets,
                group_ids,
                lambda p, t: _ranknet_loss(p, t, margin=rank_margin),
            )
        elif loss_mode == "hybrid_group_listnet":
            loss = mse_loss + rank_weight * _groupwise_loss(preds, rank_targets, group_ids, _listnet_loss)
        elif loss_mode == "hybrid_group_pairwise_hinge":
            loss = mse_loss + rank_weight * _groupwise_loss(
                preds,
                rank_targets,
                group_ids,
                lambda p, t: _pairwise_hinge_loss(p, t, margin=rank_margin),
            )
        else:
            raise ValueError(f"Unknown loss_mode: {loss_mode}")
        aux_outputs = getattr(model, "last_aux_outputs", None)
        if (
            reaction_center_aux_weight > 0.0
            and aux_outputs
            and isinstance(aux_outputs, dict)
            and "reaction_center_aux" in aux_outputs
        ):
            reaction_center_targets = batch.get("reaction_center_targets")
            if reaction_center_targets is not None and reaction_center_targets.numel() > 0:
                reaction_center_aux = aux_outputs["reaction_center_aux"]
                rc_loss = torch.nn.functional.mse_loss(reaction_center_aux, reaction_center_targets)
                loss = loss + float(reaction_center_aux_weight) * rc_loss
                if grad_tracker is not None:
                    grad_tracker["reaction_center_aux_loss"] = float(rc_loss.detach().item())
        if (
            exact_covered_consistency_weight > 0.0
            and aux_outputs
            and isinstance(aux_outputs, dict)
            and "base_pred" in aux_outputs
        ):
            base_pred = aux_outputs["base_pred"]
            exact_covered_indicator = batch.get("exact_covered_indicator")
            if (
                exact_covered_indicator is not None
                and exact_covered_indicator.numel() > 0
                and base_pred is not None
                and torch.is_tensor(base_pred)
                and base_pred.numel() > 0
            ):
                exact_mask = exact_covered_indicator.view(-1, 1) > 0.5
                if bool(exact_mask.any()):
                    consistency_loss = torch.nn.functional.mse_loss(
                        preds[exact_mask],
                        base_pred.detach()[exact_mask],
                    )
                    loss = loss + float(exact_covered_consistency_weight) * consistency_loss
                    if grad_tracker is not None:
                        grad_tracker["exact_covered_consistency_loss"] = float(consistency_loss.detach().item())
        if llm_rule_weight > 0.0:
            llm_rule_loss = _llm_strength_rank_regularizer(
                preds,
                batch.get("llm_features"),
                batch.get("llm_confidence"),
                min_confidence=llm_rule_min_confidence,
            )
            loss = loss + float(llm_rule_weight) * llm_rule_loss

        if training:
            optimizer.zero_grad()
            loss.backward()
            if grad_tracker is not None and grad_groups is not None and not grad_tracker.get("group_grad_done"):
                for name, params in grad_groups.items():
                    grad_tracker[f"grad_group_{name}"] = _grad_norm(params)
                grad_tracker["group_grad_done"] = 1.0
            if (
                grad_tracker is not None
                and feature_for_grad is not None
                and feature_slice is not None
                and not grad_tracker.get("feature_grad_done")
            ):
                grad_input = feature_for_grad.grad
                if grad_input is None:
                    grad_input = torch.zeros_like(feature_for_grad)
                qc_start = feature_slice.start or 0
                qc_end = feature_slice.stop or 0
                qc_end = min(qc_end, grad_input.size(1))
                qc_start = min(qc_start, qc_end)
                qc_grad = grad_input[:, qc_start:qc_end] if qc_end > qc_start else None
                parts = []
                if qc_start > 0:
                    parts.append(grad_input[:, :qc_start])
                if qc_end < grad_input.size(1):
                    parts.append(grad_input[:, qc_end:])
                non_qc_grad = torch.cat(parts, dim=1) if parts else None
                qc_mean = float(qc_grad.abs().mean().item()) if qc_grad is not None else 0.0
                non_qc_mean = (
                    float(non_qc_grad.abs().mean().item()) if non_qc_grad is not None and non_qc_grad.numel() else 0.0
                )
                ratio = qc_mean / (non_qc_mean + 1e-8)
                grad_tracker["feature_grad_qc_mean"] = qc_mean
                grad_tracker["feature_grad_non_qc_mean"] = non_qc_mean
                grad_tracker["feature_grad_qc_ratio"] = ratio
                grad_tracker["feature_grad_done"] = 1.0
            if attn_audit and grad_tracker is not None and not grad_tracker.get("attn_bias_done"):
                bias_stats = batch.get("cross_bias_stats")
                if bias_stats:
                    for key, entries in bias_stats.items():
                        if not entries:
                            continue
                        ratios = [entry.get("hotspot_ratio", 0.0) for entry in entries]
                        shares = [entry.get("hotspot_share", 0.0) for entry in entries]
                        counts = [entry.get("key_hotspots", 0) for entry in entries]
                        grad_tracker[f"attn_bias_ratio_{key}"] = float(np.mean(ratios))
                        grad_tracker[f"attn_bias_share_{key}"] = float(np.mean(shares))
                        grad_tracker[f"attn_bias_hotspots_{key}"] = float(np.mean(counts))
                    grad_tracker["attn_bias_done"] = 1.0
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if grad_tracker is not None:
                grad_value = float(grad_norm) if torch.is_tensor(grad_norm) else float(grad_norm)
                grad_tracker["max_grad_norm"] = max(grad_tracker.get("max_grad_norm", 0.0), grad_value)
                if not math.isfinite(grad_value):
                    grad_tracker["non_finite"] = 1.0
            optimizer.step()

        total_loss += loss.item() * targets.size(0)
        y_true.append(targets.detach().cpu().numpy())
        y_pred.append(preds.detach().cpu().numpy())

    y_true = np.concatenate(y_true, axis=0).reshape(-1)
    y_pred = np.concatenate(y_pred, axis=0).reshape(-1)
    return total_loss / len(loader.dataset), y_true, y_pred


def compute_qc_descriptor_importance(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    input_keys: Optional[Sequence[str]],
    qc_feature_cols: Sequence[str],
) -> List[Dict[str, float]]:
    if not qc_feature_cols:
        return []

    model.eval()
    grad_abs_total = np.zeros((len(qc_feature_cols),), dtype=np.float64)
    grad_x_input_abs_total = np.zeros((len(qc_feature_cols),), dtype=np.float64)
    sample_count = 0

    for batch in loader:
        _move_batch_to_device(batch, device)
        batch.pop("targets", None)
        batch.pop("rank_targets", None)
        batch.pop("group_ids", None)

        qc_features = batch.get("qc_features")
        if qc_features is None or not torch.is_tensor(qc_features) or qc_features.numel() == 0:
            continue
        qc_features = qc_features.detach().clone().requires_grad_(True)

        if input_keys is None:
            model_kwargs = dict(batch)
        else:
            model_kwargs = {key: batch[key] for key in input_keys}
        model_kwargs["qc_features"] = qc_features

        preds = model(**model_kwargs)
        grads = torch.autograd.grad(preds.sum(), qc_features, allow_unused=True)[0]
        if grads is None:
            continue

        grad_abs_total += grads.detach().abs().sum(dim=0).cpu().numpy()
        grad_x_input_abs_total += (grads.detach() * qc_features.detach()).abs().sum(dim=0).cpu().numpy()
        sample_count += int(qc_features.size(0))

    if sample_count == 0:
        return [
            {
                "feature": feature,
                "grad_abs_mean": 0.0,
                "grad_abs_share": 0.0,
                "grad_x_input_abs_mean": 0.0,
                "grad_x_input_abs_share": 0.0,
            }
            for feature in qc_feature_cols
        ]

    grad_abs_mean = grad_abs_total / float(sample_count)
    grad_x_input_abs_mean = grad_x_input_abs_total / float(sample_count)
    grad_abs_denom = float(np.sum(grad_abs_mean))
    grad_x_input_denom = float(np.sum(grad_x_input_abs_mean))

    importance = []
    for idx, feature in enumerate(qc_feature_cols):
        grad_abs_val = float(grad_abs_mean[idx])
        grad_x_input_val = float(grad_x_input_abs_mean[idx])
        importance.append(
            {
                "feature": feature,
                "grad_abs_mean": grad_abs_val,
                "grad_abs_share": float(grad_abs_val / grad_abs_denom) if grad_abs_denom > 0 else 0.0,
                "grad_x_input_abs_mean": grad_x_input_val,
                "grad_x_input_abs_share": float(grad_x_input_val / grad_x_input_denom)
                if grad_x_input_denom > 0
                else 0.0,
            }
        )

    importance.sort(key=lambda entry: entry["grad_abs_mean"], reverse=True)
    return importance


def _make_logger(log_path: Optional[Path]):
    if log_path is None:
        def _log(msg: str):
            print(msg)
        return _log

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = log_path.open("w", encoding="utf-8")

    def _log(msg: str):
        print(msg)
        log_f.write(msg + "\n")
        log_f.flush()

    return _log


def _freeze_backbone(model: nn.Module) -> List[str]:
    frozen = []
    for name in ("cat_encoder", "reactant_encoder", "encoder"):
        module = getattr(model, name, None)
        if module is None:
            continue
        for param in module.parameters():
            param.requires_grad = False
        frozen.append(name)
    return frozen


def _freeze_physical_base(model: nn.Module) -> List[str]:
    frozen = []
    for name in (
        "cat_encoder",
        "reactant_encoder",
        "combine_mlp",
        "feature_gate",
        "struct_head",
        "global_attention",
        "reaction_center_coupling",
        "reaction_center_gate",
    ):
        module = getattr(model, name, None)
        if module is None:
            continue
        for param in module.parameters():
            param.requires_grad = False
        frozen.append(name)
    return frozen


def _apply_trainable_module_prefix_selector(model: nn.Module, prefixes: Sequence[str]) -> Dict[str, object]:
    requested = []
    for prefix in prefixes:
        normalized = str(prefix).strip()
        if normalized and normalized not in requested:
            requested.append(normalized)
    if not requested:
        return {
            "requested_prefixes": [],
            "matched_prefixes": [],
            "matched_parameter_names": [],
            "trainable_parameter_count": 0,
        }

    for param in model.parameters():
        param.requires_grad = False

    matched_prefixes = set()
    matched_parameter_names: List[str] = []
    trainable_parameter_count = 0
    for name, param in model.named_parameters():
        for prefix in requested:
            if name == prefix or name.startswith(prefix + "."):
                param.requires_grad = True
                matched_prefixes.add(prefix)
                matched_parameter_names.append(name)
                trainable_parameter_count += int(param.numel())
                break

    missing = [prefix for prefix in requested if prefix not in matched_prefixes]
    if missing:
        raise ValueError(
            "Unknown --trainable-module"
            "-prefix entries: "
            + ", ".join(missing)
        )

    return {
        "requested_prefixes": requested,
        "matched_prefixes": [prefix for prefix in requested if prefix in matched_prefixes],
        "matched_parameter_names": matched_parameter_names,
        "trainable_parameter_count": trainable_parameter_count,
    }


def _safe_hybrid_init(model: nn.Module, missing_keys: Sequence[str], skipped_mismatch: Sequence[Tuple[str, Tuple[int, ...], Tuple[int, ...]]]) -> List[str]:
    safe_actions: List[str] = []
    model_state = model.state_dict()
    target_keys = set(missing_keys)
    target_keys.update(name for name, _, _ in skipped_mismatch)
    for key in sorted(target_keys):
        tensor = model_state.get(key)
        if tensor is None:
            continue
        if key.endswith(
            (
                "gate_scale",
                "condition_scale",
                "moe_scale",
                "global_background_scale",
                "semantic_bridge_scale",
                "semantic_physical_bridge_scale",
                "pair_conditioned_bridge_scale",
                "reaction_center_scale",
            )
        ):
            tensor.fill_(-20.0)
            safe_actions.append(f"{key}=-20")
            continue
        lower_key = key.lower()
        if tensor.ndim == 1 and key.endswith("weight") and "norm" in lower_key:
            tensor.fill_(1.0)
            safe_actions.append(f"{key}=1")
            continue
        if key.endswith("bias") and "norm" in lower_key:
            tensor.zero_()
            safe_actions.append(f"{key}=0")
            continue
        if key.endswith((".weight", ".bias")) or key.endswith(("weight", "bias")):
            safe_actions.append(f"{key}=keep-default-init")
    model.load_state_dict(model_state, strict=False)
    return safe_actions


def _infer_node_aux_dim(dataset) -> int:
    for sample in getattr(dataset, "samples", []):
        for attr in ("cat_node_qc", "r1_node_qc", "r2_node_qc"):
            value = getattr(sample, attr, None)
            if value is not None and torch.is_tensor(value) and value.numel() > 0:
                return int(value.size(-1))
    return 0


def _coordination_focus_threshold(dataset, quantile: float) -> Optional[float]:
    coord_dim = len(getattr(dataset, "coordination_feature_cols", []) or [])
    if coord_dim <= 0:
        return None
    signals = []
    for sample in getattr(dataset, "samples", []):
        r1_node_qc = getattr(sample, "r1_node_qc", None)
        if r1_node_qc is None or r1_node_qc.numel() == 0:
            signals.append(0.0)
            continue
        coord = r1_node_qc[:, -coord_dim:]
        active = coord[:, 0]
        potency = coord[:, 4] if coord.size(1) > 4 else coord[:, 0]
        signals.append(float((active * potency).mean().item()))
    if not signals:
        return None
    return float(np.quantile(np.asarray(signals, dtype=np.float32), float(quantile)))


def _inverse_softplus_scale(value: float) -> float:
    bounded = max(float(value), 0.0)
    if bounded <= 1e-8:
        return -20.0
    return math.log(math.expm1(bounded))


def _interpolated_curriculum_scale(epoch: int, *, start: float, target: float, duration_epochs: int) -> float:
    if duration_epochs <= 1:
        return float(target)
    clamped_epoch = max(1, min(int(epoch), int(duration_epochs)))
    progress = float(clamped_epoch - 1) / float(duration_epochs - 1)
    return float(start) + (float(target) - float(start)) * progress


def _shaped_curriculum_progress(progress: float, *, shape: str) -> float:
    bounded_progress = min(max(float(progress), 0.0), 1.0)
    if shape == "linear":
        return bounded_progress
    if shape == "quadratic":
        return bounded_progress * bounded_progress
    raise ValueError(f"Unsupported conditioned-scale curriculum shape: {shape}")


def _curriculum_conditioned_scale(
    epoch: int,
    *,
    start: float,
    target: float,
    warmup_epochs: int,
    shape: str = "linear",
    stage1_target: Optional[float] = None,
    stage2_warmup_epochs: Optional[int] = None,
) -> float:
    def _shaped_scale(local_epoch: int, local_start: float, local_target: float, duration_epochs: int) -> float:
        if duration_epochs <= 1:
            return float(local_target)
        clamped_epoch = max(1, min(int(local_epoch), int(duration_epochs)))
        raw_progress = float(clamped_epoch - 1) / float(duration_epochs - 1)
        shaped_progress = _shaped_curriculum_progress(raw_progress, shape=shape)
        return float(local_start) + (float(local_target) - float(local_start)) * shaped_progress

    if stage1_target is None or stage2_warmup_epochs is None:
        return _shaped_scale(int(epoch), float(start), float(target), int(warmup_epochs))
    if int(epoch) <= int(warmup_epochs):
        return _shaped_scale(int(epoch), float(start), float(stage1_target), int(warmup_epochs))
    return _shaped_scale(
        int(epoch) - int(warmup_epochs),
        float(stage1_target),
        float(target),
        int(stage2_warmup_epochs),
    )


def _apply_llm_conditioned_scale_curriculum(
    model: nn.Module,
    *,
    epoch: int,
    enabled: bool,
    start: float,
    target: float,
    warmup_epochs: int,
    shape: str = "linear",
    stage1_target: Optional[float] = None,
    stage2_warmup_epochs: Optional[int] = None,
) -> Optional[float]:
    if not enabled:
        return None
    condition_scale = getattr(model, "condition_scale", None)
    if condition_scale is None or not torch.is_tensor(condition_scale):
        return None
    current_scale = _curriculum_conditioned_scale(
        epoch,
        start=float(start),
        target=float(target),
        warmup_epochs=int(warmup_epochs),
        shape=str(shape),
        stage1_target=None if stage1_target is None else float(stage1_target),
        stage2_warmup_epochs=None if stage2_warmup_epochs is None else int(stage2_warmup_epochs),
    )
    with torch.no_grad():
        condition_scale.fill_(_inverse_softplus_scale(current_scale))
    return current_scale


def _curriculum_row_scale(
    epoch: int,
    *,
    target: float,
    start: Optional[float] = None,
    warmup_epochs: int = 1,
    shape: str = "linear",
) -> float:
    if start is None:
        return float(target)
    return _curriculum_conditioned_scale(
        int(epoch),
        start=float(start),
        target=float(target),
        warmup_epochs=int(warmup_epochs),
        shape=str(shape),
    )


def _confidence_aware_propagated_scale(
    base_scale: float,
    *,
    confidence: float,
    floor: float = 1.0,
    power: float = 1.0,
) -> float:
    clipped_confidence = max(0.0, min(1.0, float(confidence)))
    clipped_floor = max(0.0, min(1.0, float(floor)))
    shaped_confidence = clipped_confidence ** max(float(power), 1e-12)
    return float(base_scale) * (clipped_floor + (1.0 - clipped_floor) * shaped_confidence)


def _temperature_aware_propagated_scale(
    base_scale: float,
    *,
    temperature_alignment: float,
    floor: float = 1.0,
    power: float = 1.0,
) -> float:
    clipped_alignment = max(0.0, min(1.0, float(temperature_alignment)))
    clipped_floor = max(0.0, min(1.0, float(floor)))
    shaped_alignment = clipped_alignment ** max(float(power), 1e-12)
    return float(base_scale) * (clipped_floor + (1.0 - clipped_floor) * shaped_alignment)


def _apply_llm_row_authority_schedule(
    model_kwargs: Dict[str, torch.Tensor],
    *,
    exact_scale: float,
    propagated_scale: float,
    base_exact_scale: float,
    base_propagated_scale: float,
    propagated_confidence_floor: float = 1.0,
    propagated_confidence_power: float = 1.0,
    propagated_temperature_floor: float = 1.0,
    propagated_temperature_power: float = 1.0,
) -> Dict[str, torch.Tensor]:
    llm_features = model_kwargs.get("llm_features")
    if llm_features is None or not torch.is_tensor(llm_features) or llm_features.numel() == 0:
        return model_kwargs
    exact_covered_indicator = model_kwargs.get("exact_covered_indicator")
    if (
        exact_covered_indicator is None
        or not torch.is_tensor(exact_covered_indicator)
        or exact_covered_indicator.numel() == 0
    ):
        return model_kwargs
    if abs(float(exact_scale) - float(base_exact_scale)) < 1e-12 and abs(
        float(propagated_scale) - float(base_propagated_scale)
    ) < 1e-12 and abs(float(propagated_confidence_floor) - 1.0) < 1e-12 and abs(
        float(propagated_temperature_floor) - 1.0
    ) < 1e-12:
        return model_kwargs

    exact_factor = 0.0 if abs(float(base_exact_scale)) < 1e-12 else float(exact_scale) / float(base_exact_scale)
    adjusted = llm_features.clone()
    exact_mask = (
        exact_covered_indicator.view(-1, 1).to(device=adjusted.device, dtype=adjusted.dtype) > 0.5
    )
    covered_mask = adjusted.abs().sum(dim=1, keepdim=True) > 0
    propagated_mask = covered_mask & (~exact_mask)
    if bool(exact_mask.any()):
        adjusted = torch.where(exact_mask, adjusted * exact_factor, adjusted)
    if bool(propagated_mask.any()):
        llm_confidence = model_kwargs.get("llm_confidence")
        if llm_confidence is not None and torch.is_tensor(llm_confidence) and llm_confidence.numel() > 0:
            confidence_values = llm_confidence.view(-1, 1).to(device=adjusted.device, dtype=adjusted.dtype)
        else:
            confidence_values = torch.zeros((adjusted.size(0), 1), device=adjusted.device, dtype=adjusted.dtype)
        propagated_floor_scale = _confidence_aware_propagated_scale(
            float(propagated_scale),
            confidence=0.0,
            floor=float(propagated_confidence_floor),
            power=float(propagated_confidence_power),
        )
        effective_propagated_scale = propagated_floor_scale + (
            float(propagated_scale) - propagated_floor_scale
        ) * confidence_values.clamp_(0.0, 1.0).pow(float(max(propagated_confidence_power, 1e-12)))
        if abs(float(base_propagated_scale)) < 1e-12:
            propagated_factor = torch.zeros_like(effective_propagated_scale)
        else:
            propagated_factor = effective_propagated_scale / float(base_propagated_scale)
        if abs(float(propagated_temperature_floor) - 1.0) > 1e-12 or abs(float(propagated_temperature_power) - 1.0) > 1e-12:
            temperature_alignment = _llm_feature_column(llm_features, "llm_semantic_temperature_alignment")
            if temperature_alignment is None:
                temperature_alignment = torch.zeros((adjusted.size(0), 1), device=adjusted.device, dtype=adjusted.dtype)
            else:
                temperature_alignment = temperature_alignment.to(device=adjusted.device, dtype=adjusted.dtype)
            propagated_temperature_floor_scale = _temperature_aware_propagated_scale(
                float(propagated_scale),
                temperature_alignment=0.0,
                floor=float(propagated_temperature_floor),
                power=float(propagated_temperature_power),
            )
            effective_temperature_scale = propagated_temperature_floor_scale + (
                float(propagated_scale) - propagated_temperature_floor_scale
            ) * temperature_alignment.clamp_(0.0, 1.0).pow(float(max(propagated_temperature_power, 1e-12)))
            if abs(float(propagated_scale)) < 1e-12:
                propagated_factor = torch.zeros_like(effective_temperature_scale)
            else:
                propagated_factor = propagated_factor * (effective_temperature_scale / float(propagated_scale))
        adjusted = torch.where(propagated_mask, adjusted * propagated_factor, adjusted)
    updated_kwargs = dict(model_kwargs)
    updated_kwargs["llm_features"] = adjusted
    return updated_kwargs


def _build_propagated_confidence_schedule_profile(
    samples: Sequence[object],
    *,
    authority_schedule: Sequence[Mapping[str, float]],
    floor: float,
    power: float,
) -> Dict[str, object]:
    propagated_confidences: List[float] = []
    for sample in samples:
        exact_covered = getattr(sample, "exact_covered_indicator", None)
        if torch.is_tensor(exact_covered) and exact_covered.numel() > 0 and float(exact_covered.view(-1)[0].item()) > 0.5:
            continue
        llm_features = getattr(sample, "llm_features", None)
        if not torch.is_tensor(llm_features) or llm_features.numel() == 0 or float(llm_features.abs().sum().item()) <= 0.0:
            continue
        llm_confidence = getattr(sample, "llm_confidence", None)
        confidence = 0.0
        if torch.is_tensor(llm_confidence) and llm_confidence.numel() > 0:
            confidence = float(llm_confidence.view(-1)[0].item())
        propagated_confidences.append(max(0.0, min(1.0, confidence)))
    if not propagated_confidences:
        return {
            "enabled": abs(float(floor) - 1.0) > 1e-12 or abs(float(power) - 1.0) > 1e-12,
            "propagated_row_count": 0,
            "confidence_floor": float(floor),
            "confidence_power": float(power),
            "confidence_quantiles": {},
            "effective_scale_schedule": [],
        }
    quantiles = {
        "q10": float(np.quantile(propagated_confidences, 0.10)),
        "q50": float(np.quantile(propagated_confidences, 0.50)),
        "q90": float(np.quantile(propagated_confidences, 0.90)),
    }
    effective_scale_schedule = []
    for item in authority_schedule:
        base_scale = float(item["llm_propagated_row_scale"])
        effective_scale_schedule.append(
            {
                "epoch": int(item["epoch"]),
                "base_propagated_scale": base_scale,
                "q10_effective_scale": _confidence_aware_propagated_scale(
                    base_scale,
                    confidence=quantiles["q10"],
                    floor=float(floor),
                    power=float(power),
                ),
                "q50_effective_scale": _confidence_aware_propagated_scale(
                    base_scale,
                    confidence=quantiles["q50"],
                    floor=float(floor),
                    power=float(power),
                ),
                "q90_effective_scale": _confidence_aware_propagated_scale(
                    base_scale,
                    confidence=quantiles["q90"],
                    floor=float(floor),
                    power=float(power),
                ),
            }
        )
    return {
        "enabled": abs(float(floor) - 1.0) > 1e-12 or abs(float(power) - 1.0) > 1e-12,
        "propagated_row_count": int(len(propagated_confidences)),
        "confidence_floor": float(floor),
        "confidence_power": float(power),
        "confidence_quantiles": quantiles,
        "effective_scale_schedule": effective_scale_schedule,
    }


def _build_propagated_temperature_schedule_profile(
    samples: Sequence[object],
    *,
    authority_schedule: Sequence[Mapping[str, float]],
    floor: float,
    power: float,
) -> Dict[str, object]:
    propagated_temperature_alignment: List[float] = []
    for sample in samples:
        exact_covered = getattr(sample, "exact_covered_indicator", None)
        if torch.is_tensor(exact_covered) and exact_covered.numel() > 0 and float(exact_covered.view(-1)[0].item()) > 0.5:
            continue
        llm_features = getattr(sample, "llm_features", None)
        if not torch.is_tensor(llm_features) or llm_features.numel() == 0 or float(llm_features.abs().sum().item()) <= 0.0:
            continue
        temperature_alignment = _llm_feature_column(llm_features.view(1, -1), "llm_semantic_temperature_alignment")
        alignment = 0.0
        if temperature_alignment is not None and temperature_alignment.numel() > 0:
            alignment = float(temperature_alignment.view(-1)[0].item())
        propagated_temperature_alignment.append(max(0.0, min(1.0, alignment)))
    if not propagated_temperature_alignment:
        return {
            "enabled": abs(float(floor) - 1.0) > 1e-12 or abs(float(power) - 1.0) > 1e-12,
            "propagated_row_count": 0,
            "temperature_floor": float(floor),
            "temperature_power": float(power),
            "temperature_alignment_quantiles": {},
            "effective_scale_schedule": [],
        }
    quantiles = {
        "q10": float(np.quantile(propagated_temperature_alignment, 0.10)),
        "q50": float(np.quantile(propagated_temperature_alignment, 0.50)),
        "q90": float(np.quantile(propagated_temperature_alignment, 0.90)),
    }
    effective_scale_schedule = []
    for item in authority_schedule:
        base_scale = float(item["llm_propagated_row_scale"])
        effective_scale_schedule.append(
            {
                "epoch": int(item["epoch"]),
                "base_propagated_scale": base_scale,
                "q10_effective_scale": _temperature_aware_propagated_scale(
                    base_scale,
                    temperature_alignment=quantiles["q10"],
                    floor=float(floor),
                    power=float(power),
                ),
                "q50_effective_scale": _temperature_aware_propagated_scale(
                    base_scale,
                    temperature_alignment=quantiles["q50"],
                    floor=float(floor),
                    power=float(power),
                ),
                "q90_effective_scale": _temperature_aware_propagated_scale(
                    base_scale,
                    temperature_alignment=quantiles["q90"],
                    floor=float(floor),
                    power=float(power),
                ),
            }
        )
    return {
        "enabled": abs(float(floor) - 1.0) > 1e-12 or abs(float(power) - 1.0) > 1e-12,
        "propagated_row_count": int(len(propagated_temperature_alignment)),
        "temperature_floor": float(floor),
        "temperature_power": float(power),
        "temperature_alignment_quantiles": quantiles,
        "effective_scale_schedule": effective_scale_schedule,
    }


def train_once(args, lr: float, hidden_dim: int, log_path: Optional[Path]):
    log = _make_logger(log_path)
    log(f"Run started: {datetime.now().isoformat()}")
    split_seed = args.seed if args.split_seed is None else int(args.split_seed)
    rank_group_cols = args.rank_group_cols if "group_" in args.loss else None
    rank_target_col = args.rank_target_col
    if args.dataset != "ml-cb":
        rank_group_cols = None
        rank_target_col = None
    if args.model == "schnet_backbone":
        if args.dataset != "ml-cb":
            raise ValueError("schnet_backbone is supported only for the ml-cb dataset.")
        if hidden_dim > 128 or args.layers > 3:
            raise ValueError("schnet_backbone requires hidden_dim <= 128 and layers <= 3.")
        if args.loss not in {"group_ranknet", "hybrid_group_ranknet"}:
            raise ValueError("schnet_backbone requires group_ranknet or hybrid_group_ranknet loss.")
        if rank_group_cols is None:
            raise ValueError("schnet_backbone requires --rank-group-cols for group-wise ranking.")
        if rank_target_col is not None:
            raise ValueError("schnet_backbone must rank by yield only; omit --rank-target-col.")
    if args.model == "shared_schnet":
        if args.dataset != "ml-cb":
            raise ValueError("shared_schnet is supported only for the ml-cb dataset.")
        if hidden_dim > 128 or args.layers > 3:
            raise ValueError("shared_schnet requires hidden_dim <= 128 and layers <= 3.")
        if args.loss not in {"group_ranknet", "hybrid_group_ranknet"}:
            raise ValueError("shared_schnet requires group_ranknet or hybrid_group_ranknet loss.")
        if rank_group_cols is None:
            raise ValueError("shared_schnet requires --rank-group-cols for group-wise ranking.")
        if rank_target_col is not None:
            raise ValueError("shared_schnet must rank by yield only; omit --rank-target-col.")
        if args.qc_fusion != "none":
            raise ValueError("shared_schnet is 3D-only; set --qc-fusion none.")
        if args.use_numeric:
            raise ValueError("shared_schnet is 3D-only; set --no-numeric to disable numeric features.")
        if args.use_coulomb:
            raise ValueError("shared_schnet is 3D-only; set --no-coulomb to disable coulomb features.")
        if args.use_geometry:
            raise ValueError("shared_schnet is 3D-only; set --no-geometry to disable geometry features.")
    if args.global_attention and args.model not in {"painn", "painn_precision", "painn_physics"}:
        raise ValueError(
            "--global-attention is currently supported for painn, painn_precision, and painn_physics models only."
        )
    if args.global_attention_heads <= 0:
        raise ValueError("--global-attention-heads must be positive.")
    if args.global_attention_layers <= 0:
        raise ValueError("--global-attention-layers must be positive.")
    if args.swa_start is not None and args.swa_start <= 0:
        raise ValueError("--swa-start must be positive when provided.")
    if args.swa_anneal_epochs <= 0:
        raise ValueError("--swa-anneal-epochs must be positive.")
    if args.focus_yield_weight < 0.0:
        raise ValueError("--focus-yield-weight must be non-negative.")
    if args.focus_exact_covered_weight < 0.0:
        raise ValueError("--focus-exact-covered-weight must be non-negative.")
    if args.llm_exact_row_scale < 0.0:
        raise ValueError("--llm-exact-row-scale must be non-negative.")
    if args.llm_exact_row_scale_curriculum_start is not None:
        if args.llm_exact_row_scale_curriculum_start < 0.0:
            raise ValueError("--llm-exact-row-scale-curriculum-start must be non-negative.")
        if args.llm_exact_row_scale_curriculum_warmup_epochs <= 0:
            raise ValueError("--llm-exact-row-scale-curriculum-warmup-epochs must be positive.")
        if args.llm_exact_row_scale_curriculum_start > args.llm_exact_row_scale:
            raise ValueError(
                "--llm-exact-row-scale-curriculum-start must be less than or equal to --llm-exact-row-scale."
            )
        if args.llm_exact_row_scale_curriculum_shape not in {"linear", "quadratic"}:
            raise ValueError("--llm-exact-row-scale-curriculum-shape must be one of: linear, quadratic.")
    if args.llm_propagated_row_scale < 0.0:
        raise ValueError("--llm-propagated-row-scale must be non-negative.")
    if args.llm_propagated_row_scale_curriculum_start is not None:
        if args.llm_propagated_row_scale_curriculum_start < 0.0:
            raise ValueError("--llm-propagated-row-scale-curriculum-start must be non-negative.")
        if args.llm_propagated_row_scale_curriculum_warmup_epochs <= 0:
            raise ValueError("--llm-propagated-row-scale-curriculum-warmup-epochs must be positive.")
        if args.llm_propagated_row_scale_curriculum_start > args.llm_propagated_row_scale:
            raise ValueError(
                "--llm-propagated-row-scale-curriculum-start must be less than or equal to "
                "--llm-propagated-row-scale."
            )
        if args.llm_propagated_row_scale_curriculum_shape not in {"linear", "quadratic"}:
            raise ValueError("--llm-propagated-row-scale-curriculum-shape must be one of: linear, quadratic.")
    if not 0.0 <= args.llm_propagated_row_confidence_floor <= 1.0:
        raise ValueError("--llm-propagated-row-confidence-floor must be between 0 and 1.")
    if args.llm_propagated_row_confidence_power <= 0.0:
        raise ValueError("--llm-propagated-row-confidence-power must be positive.")
    if not 0.0 <= args.llm_propagated_row_temperature_floor <= 1.0:
        raise ValueError("--llm-propagated-row-temperature-floor must be between 0 and 1.")
    if args.llm_propagated_row_temperature_power <= 0.0:
        raise ValueError("--llm-propagated-row-temperature-power must be positive.")
    if args.focus_yield_weight > 0.0 and args.focus_yield_threshold is None:
        raise ValueError("--focus-yield-threshold is required when --focus-yield-weight > 0.")
    if args.focus_coordination_weight < 0.0:
        raise ValueError("--focus-coordination-weight must be non-negative.")
    if args.focus_electronic_frontier_weight < 0.0:
        raise ValueError("--focus-electronic-frontier-weight must be non-negative.")
    if args.focus_kinetic_relay_weight < 0.0:
        raise ValueError("--focus-kinetic-relay-weight must be non-negative.")
    if not 0.0 <= args.focus_coordination_quantile <= 1.0:
        raise ValueError("--focus-coordination-quantile must be between 0 and 1.")
    if args.reaction_center_aux_weight < 0.0:
        raise ValueError("--reaction-center-aux-weight must be non-negative.")
    if args.exact_covered_consistency_weight < 0.0:
        raise ValueError("--exact-covered-consistency-weight must be non-negative.")
    if args.llm_conditioned_scale_curriculum_start is not None:
        if args.llm_conditioned_scale_curriculum_start < 0.0:
            raise ValueError("--llm-conditioned-scale-curriculum-start must be non-negative.")
        if args.llm_conditioned_scale_curriculum_warmup_epochs <= 0:
            raise ValueError("--llm-conditioned-scale-curriculum-warmup-epochs must be positive.")
        if args.llm_conditioned_scale_curriculum_start > args.llm_conditioned_scale:
            raise ValueError(
                "--llm-conditioned-scale-curriculum-start must be less than or equal to --llm-conditioned-scale."
            )
        if args.llm_conditioned_scale_curriculum_shape not in {"linear", "quadratic"}:
            raise ValueError("--llm-conditioned-scale-curriculum-shape must be one of: linear, quadratic.")
    llm_conditioned_scale_curriculum_two_stage_enabled = (
        args.llm_conditioned_scale_curriculum_stage1_target is not None
        or args.llm_conditioned_scale_curriculum_stage2_warmup_epochs is not None
    )
    if llm_conditioned_scale_curriculum_two_stage_enabled:
        if args.llm_conditioned_scale_curriculum_start is None:
            raise ValueError(
                "--llm-conditioned-scale-curriculum-stage1-target requires "
                "--llm-conditioned-scale-curriculum-start."
            )
        if args.llm_conditioned_scale_curriculum_stage1_target is None:
            raise ValueError(
                "--llm-conditioned-scale-curriculum-stage2-warmup-epochs requires "
                "--llm-conditioned-scale-curriculum-stage1-target."
            )
        if args.llm_conditioned_scale_curriculum_stage2_warmup_epochs is None:
            raise ValueError(
                "--llm-conditioned-scale-curriculum-stage1-target requires "
                "--llm-conditioned-scale-curriculum-stage2-warmup-epochs."
            )
        if args.llm_conditioned_scale_curriculum_stage1_target < args.llm_conditioned_scale_curriculum_start:
            raise ValueError(
                "--llm-conditioned-scale-curriculum-stage1-target must be greater than or equal to "
                "--llm-conditioned-scale-curriculum-start."
            )
        if args.llm_conditioned_scale_curriculum_stage1_target > args.llm_conditioned_scale:
            raise ValueError(
                "--llm-conditioned-scale-curriculum-stage1-target must be less than or equal to "
                "--llm-conditioned-scale."
            )
        if args.llm_conditioned_scale_curriculum_stage2_warmup_epochs <= 0:
            raise ValueError("--llm-conditioned-scale-curriculum-stage2-warmup-epochs must be positive.")
    if args.reaction_center_coupling and args.model != "painn_precision":
        raise ValueError("--reaction-center-coupling is currently supported only for painn_precision.")
    if args.reaction_center_aux_weight > 0.0 and not args.reaction_center_coupling:
        raise ValueError("--reaction-center-aux-weight requires --reaction-center-coupling.")
    if args.kinetic_relay and not args.reaction_center_coupling:
        raise ValueError("--kinetic-relay requires --reaction-center-coupling.")
    log(
        f"Config: lr={lr} hidden_dim={hidden_dim} epochs={args.epochs} "
        f"batch={args.batch_size} dataset={args.dataset} model={args.model} qc_fusion={args.qc_fusion} "
        f"split_strategy={args.split_strategy} signature_columns={args.signature_columns} "
        f"lr_scheduler={args.lr_scheduler} "
        f"lr_scheduler_patience={args.lr_scheduler_patience} "
        f"lr_scheduler_factor={args.lr_scheduler_factor} "
        f"lr_scheduler_min_lr={args.lr_scheduler_min_lr} "
        f"coord_jitter_std={args.coord_jitter_std} "
        f"qc_node_scale={args.qc_node_scale} qc_global_scale={args.qc_global_scale} "
        f"qc_node_mode={args.qc_node_mode} "
        f"coordination_features={args.coordination_features} "
        f"coordination_node_scale={args.coordination_node_scale} "
        f"reaction_center_coupling={args.reaction_center_coupling} "
        f"kinetic_relay={args.kinetic_relay} "
        f"reaction_center_aux_weight={args.reaction_center_aux_weight} "
        f"exact_covered_consistency_weight={args.exact_covered_consistency_weight} "
        f"physics_fusion={args.physics_fusion} "
        f"qc_weight_path={args.qc_weight_path} qc_weight_mode={args.qc_weight_mode} "
        f"qc_weight_scale={args.qc_weight_scale} "
        f"use_coulomb={args.use_coulomb} use_geometry={args.use_geometry} "
        f"use_numeric={args.use_numeric} "
        f"use_rdkit={args.use_rdkit} use_morgan_fingerprint={args.use_morgan_fingerprint} use_ref_data={args.use_ref_data} "
        f"literature_cache={args.literature_cache} "
        f"loss={args.loss} rank_weight={args.rank_weight} rank_margin={args.rank_margin} "
        f"rank_group_cols={rank_group_cols} rank_target_col={rank_target_col} "
        f"cross_bias_scale={args.cross_bias_scale} "
        f"llm_bias_scale={args.llm_bias_scale} llm_bias_mode={args.llm_bias_mode} "
        f"llm_bias_aggregation={args.llm_bias_aggregation} llm_edge_mode={args.llm_edge_mode} "
        f"llm_rule_weight={args.llm_rule_weight} llm_rule_min_confidence={args.llm_rule_min_confidence} "
        f"focus_exact_covered_weight={args.focus_exact_covered_weight} "
        f"focus_confidence_threshold={args.focus_confidence_threshold} "
        f"focus_confidence_weight={args.focus_confidence_weight} "
        f"llm_row_dropout_p={args.llm_row_dropout_p} "
        f"llm_row_dropout_confidence_scale={args.llm_row_dropout_confidence_scale} "
        f"llm_reliability_loss_weight={args.llm_reliability_loss_weight} "
        f"llm_reliability_loss_center={args.llm_reliability_loss_center} "
        f"llm_reliability_loss_sharpness={args.llm_reliability_loss_sharpness} "
        f"focus_yield_threshold={args.focus_yield_threshold} "
        f"focus_yield_weight={args.focus_yield_weight} "
        f"focus_coordination_quantile={args.focus_coordination_quantile} "
        f"focus_coordination_weight={args.focus_coordination_weight} "
        f"focus_electronic_frontier_weight={args.focus_electronic_frontier_weight} "
        f"focus_kinetic_relay_weight={args.focus_kinetic_relay_weight} "
        f"llm_gate_min_confidence={args.llm_gate_min_confidence} "
        f"llm_gate_confidence_sharpness={args.llm_gate_confidence_sharpness} "
        f"llm_gate_density_center={args.llm_gate_density_center} "
        f"llm_gate_density_sharpness={args.llm_gate_density_sharpness} "
        f"llm_gate_adaptive_strength={args.llm_gate_adaptive_strength} "
        f"llm_gate_product_focus_scale={args.llm_gate_product_focus_scale} "
        f"llm_feature_profile={args.llm_feature_profile} "
        f"llm_semantic_group_scale={args.llm_semantic_group_scale} "
        f"llm_semantic_feature_scale={args.llm_semantic_feature_scale} "
        f"llm_exact_row_scale={args.llm_exact_row_scale} "
        f"llm_propagated_row_scale={args.llm_propagated_row_scale} "
        f"llm_propagated_row_scale_curriculum_start={args.llm_propagated_row_scale_curriculum_start} "
        f"llm_propagated_row_scale_curriculum_warmup_epochs={args.llm_propagated_row_scale_curriculum_warmup_epochs} "
        f"llm_propagated_row_scale_curriculum_shape={args.llm_propagated_row_scale_curriculum_shape} "
        f"llm_propagated_row_confidence_floor={args.llm_propagated_row_confidence_floor} "
        f"llm_propagated_row_confidence_power={args.llm_propagated_row_confidence_power} "
        f"llm_fusion_mode={args.llm_fusion_mode} "
        f"llm_cross_attention_heads={args.llm_cross_attention_heads} "
        f"llm_conditioned_scale={args.llm_conditioned_scale} "
        f"llm_conditioned_scale_curriculum_start={args.llm_conditioned_scale_curriculum_start} "
        f"llm_conditioned_scale_curriculum_warmup_epochs={args.llm_conditioned_scale_curriculum_warmup_epochs} "
        f"llm_physical_bridge={args.llm_physical_bridge} "
        f"llm_pair_conditioned_bridge={args.llm_pair_conditioned_bridge} "
        f"llm_pair_conditioned_bridge_locus={args.llm_pair_conditioned_bridge_locus} "
        f"llm_disable_global_background={args.llm_disable_global_background} "
        f"llm_local_meta_gate={args.llm_local_meta_gate} "
        f"attn_audit={args.attn_audit} "
        f"llm_phys_audit={args.llm_phys_audit} llm_phys_cutoff={args.llm_phys_cutoff} "
        f"baseline_summary={args.baseline_summary} init_from={args.init_from} "
        f"freeze_backbone={args.freeze_backbone} freeze_physical_base={args.freeze_physical_base} "
        f"trainable_module_prefix={args.trainable_module_prefix} "
        f"global_attention={args.global_attention} "
        f"global_attention_heads={args.global_attention_heads} "
        f"global_attention_layers={args.global_attention_layers} "
        f"interaction_cross_attention={args.interaction_cross_attention} "
        f"interaction_pair_mode={args.interaction_pair_mode} "
        f"swa={args.swa} swa_start={args.swa_start} swa_lr={args.swa_lr} "
        f"split_seed={split_seed} "
        f"val_fraction={args.val_fraction} test_fraction={args.test_fraction} val_split={args.val_split}"
    )

    use_literature = not args.disable_literature_branch and (
        args.model in {
        "alignment",
        "dual_path",
        "painn_llm",
        "painn_precision",
        "painn_attn_bias",
    }
        or bool(args.literature_cache)
        or args.llm_rule_weight > 0.0
        or args.focus_confidence_weight > 0.0
        or args.focus_exact_covered_weight > 0.0
    )
    use_combined = args.model == "ssgnn"
    use_atomic_interaction_pairs = args.model == "painn_precision" and args.interaction_cross_attention
    if args.dataset == "ml-borylation":
        dataset_cls = MLBorylationDataset
        dataset_kwargs = {
            "include_literature": use_literature,
            "literature_cache_path": args.literature_cache,
            "llm_feature_profile": args.llm_feature_profile,
            "llm_missingness_contract": args.llm_missingness_contract,
            "llm_semantic_group_scales": args.llm_semantic_group_scale,
            "llm_semantic_feature_scales": args.llm_semantic_feature_scale,
            "llm_exact_row_scale": args.llm_exact_row_scale,
            "llm_propagated_row_scale": args.llm_propagated_row_scale,
            "native_pair_field_profile": args.native_pair_field_profile,
            "build_combined_graph": use_combined,
            "build_atomic_interaction_pairs": use_atomic_interaction_pairs,
            "interaction_pair_mode": args.interaction_pair_mode,
            "combined_k": args.combined_k,
            "combined_cutoff": args.combined_cutoff,
            "include_numeric": args.use_numeric,
            "qc_fusion": args.qc_fusion,
            "qc_scale": args.qc_scale,
            "qc_node_scale": args.qc_node_scale,
            "qc_global_scale": args.qc_global_scale,
            "qc_node_mode": args.qc_node_mode,
            "qc_weight_path": args.qc_weight_path,
            "qc_weight_mode": args.qc_weight_mode,
            "qc_weight_scale": args.qc_weight_scale,
            "qc_weight_min": args.qc_weight_min,
            "qc_weight_max": args.qc_weight_max,
            "coordination_features": args.coordination_features,
            "coordination_node_scale": args.coordination_node_scale,
            "build_reaction_center_coupling": args.reaction_center_coupling,
            "use_coulomb": args.use_coulomb,
            "use_geometry": args.use_geometry,
            "use_rdkit": args.use_rdkit,
            "use_morgan_fingerprint": args.use_morgan_fingerprint,
            "morgan_fingerprint_bits": args.morgan_fingerprint_bits,
            "morgan_fingerprint_radius": args.morgan_fingerprint_radius,
            "use_ref_data": args.use_ref_data,
            "val_fraction": args.val_fraction,
            "test_fraction": args.test_fraction,
            "split_strategy": args.split_strategy,
            "signature_columns": args.signature_columns,
        }
    else:
        dataset_cls = MLCBDataset
        dataset_kwargs = {
            "include_numeric": args.use_numeric,
            "build_combined_graph": use_combined,
            "build_atomic_interaction_pairs": use_atomic_interaction_pairs,
            "interaction_pair_mode": args.interaction_pair_mode,
            "combined_k": args.combined_k,
            "combined_cutoff": args.combined_cutoff,
            "qc_fusion": args.qc_fusion,
            "qc_scale": args.qc_scale,
            "qc_node_scale": args.qc_node_scale,
            "qc_global_scale": args.qc_global_scale,
            "qc_node_mode": args.qc_node_mode,
            "qc_weight_path": args.qc_weight_path,
            "qc_weight_mode": args.qc_weight_mode,
            "qc_weight_scale": args.qc_weight_scale,
            "qc_weight_min": args.qc_weight_min,
            "qc_weight_max": args.qc_weight_max,
            "coordination_features": args.coordination_features,
            "coordination_node_scale": args.coordination_node_scale,
            "build_reaction_center_coupling": args.reaction_center_coupling,
            "use_coulomb": args.use_coulomb,
            "use_geometry": args.use_geometry,
            "group_cols": rank_group_cols,
            "rank_target_col": rank_target_col,
            "val_fraction": args.val_fraction,
            "test_fraction": args.test_fraction,
        }

    train_ds = dataset_cls(args.data, split="train", seed=split_seed, max_samples=args.samples, **dataset_kwargs)
    if args.val_split == "train":
        val_ds = train_ds
    else:
        val_ds = dataset_cls(args.data, split="val", seed=split_seed, max_samples=args.samples, **dataset_kwargs)
    test_ds = dataset_cls(args.data, split="test", seed=split_seed, max_samples=args.samples, **dataset_kwargs)

    log(
        f"Dataset sizes: train={len(train_ds)} val={len(val_ds)} "
        f"test={len(test_ds)} total={len(train_ds)+len(val_ds)+len(test_ds)}"
    )
    split_strategy = getattr(train_ds, "split_strategy", "random")
    split_group_source = getattr(train_ds, "split_group_source", None)
    if split_strategy == "group_reference":
        log(f"Split strategy: GroupShuffleSplit by reference id ({split_group_source})")
    elif split_strategy == "group_structural_signature":
        log(f"Split strategy: GroupShuffleSplit by structural signature ({split_group_source})")
    else:
        log("Split strategy: random row shuffle")
    if getattr(train_ds, "qc_feature_cols", None):
        log(f"QC fusion features: {len(train_ds.qc_feature_cols)} -> {train_ds.qc_feature_cols}")
    if getattr(train_ds, "coordination_feature_cols", None):
        log(
            "Coordination node features: "
            f"{len(train_ds.coordination_feature_cols)} -> {train_ds.coordination_feature_cols}"
        )
    if getattr(train_ds, "qc_feature_weights", None) is not None:
        weights = np.asarray(train_ds.qc_feature_weights)
        if weights.size:
            log(
                "QC weight stats: "
                f"mode={args.qc_weight_mode} scale={args.qc_weight_scale} "
                f"min={weights.min():.3f} max={weights.max():.3f}"
            )
    if getattr(train_ds, "llm_feature_cols", None):
        if train_ds.llm_feature_cols:
            log(f"LLM feature cols: {len(train_ds.llm_feature_cols)} -> {train_ds.llm_feature_cols}")
    llm_missingness_contract = str(getattr(train_ds, "llm_missingness_contract", "profile_default") or "profile_default")
    if llm_missingness_contract != "profile_default":
        log(f"LLM missingness contract: {llm_missingness_contract}")
    native_pair_field_profile = str(getattr(train_ds, "native_pair_field_profile", "full") or "full")
    if native_pair_field_profile != "full":
        log(
            "Native pair field profile: "
            f"{native_pair_field_profile} "
            f"summary_scales={getattr(train_ds, 'native_pair_summary_group_scales', {})} "
            f"token_scales={getattr(train_ds, 'native_pair_token_group_scales', {})}"
        )

    llm_phys_summary = None
    if use_literature and args.llm_phys_audit:
        llm_phys_summary = summarize_literature_physical_consistency(
            train_ds,
            cutoff=args.llm_phys_cutoff,
        )
        overall = llm_phys_summary.get("overall", {})
        log(
            "LLM physical consistency (train): "
            f"mapped_ratio={overall.get('mapped_ratio', 0.0):.2f} "
            f"within_cutoff_ratio={overall.get('within_cutoff_ratio', 0.0):.2f} "
            f"mean_distance={overall.get('mean_distance', 0.0):.2f}Å "
            f"mean_abs_delta={overall.get('mean_abs_distance_error', 0.0):.2f}Å"
        )

    include_group = rank_group_cols is not None
    include_rank_target = rank_target_col is not None
    collate = lambda b: collate_fn(
        b,
        include_literature=use_literature,
        include_combined=use_combined,
        include_group=include_group,
        include_rank_target=include_rank_target,
        llm_bias_scale=args.llm_bias_scale,
        llm_bias_mode=args.llm_bias_mode,
        llm_bias_aggregation=args.llm_bias_aggregation,
        attn_audit=args.attn_audit,
    )
    if include_group:
        train_group_ids = [sample.group_id for sample in train_ds.samples]
        train_sampler = GroupBatchSampler(
            train_group_ids,
            batch_size=args.batch_size,
            shuffle=True,
            seed=args.seed,
        )
        train_loader = DataLoader(train_ds, batch_sampler=train_sampler, collate_fn=collate)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    qc_dim = len(getattr(train_ds, "qc_feature_cols", []) or [])
    node_aux_dim = _infer_node_aux_dim(train_ds)
    qc_node_dim = qc_dim if args.qc_node_scale != 0.0 and args.qc_node_mode != "none" else 0
    qc_gate_dim = qc_dim if args.qc_node_scale != 0.0 else 0
    if args.model == "baseline":
        model = DualSchNetCrossAttention(
            hidden_dim=hidden_dim,
            num_layers=args.layers,
            heads=args.heads,
            feat_dim=len(train_ds.feature_cols),
            dropout=args.dropout,
            cross_bias_scale=args.cross_bias_scale,
            qc_dim=qc_node_dim,
        )
    elif args.model == "gated_cross":
        model = GatedSchNetCrossAttention(
            hidden_dim=hidden_dim,
            num_layers=args.layers,
            heads=args.heads,
            feat_dim=len(train_ds.feature_cols),
            dropout=args.dropout,
            qc_dim=qc_dim,
            qc_node_dim=qc_node_dim,
            qc_gate_dim=qc_node_dim,
        )
    elif args.model == "schnet_node_qc":
        model = SchNetNodeQCRanker(
            hidden_dim=hidden_dim,
            num_layers=args.layers,
            feat_dim=len(train_ds.feature_cols),
            dropout=args.dropout,
            node_qc_dim=qc_node_dim,
        )
    elif args.model == "schnet_qc_gate":
        model = SchNetQCGatedMessageRanker(
            hidden_dim=hidden_dim,
            num_layers=args.layers,
            feat_dim=len(train_ds.feature_cols),
            dropout=args.dropout,
            qc_gate_dim=qc_gate_dim,
        )
    elif args.model == "schnet_qc_attn":
        model = SchNetQCCrossAttentionRanker(
            hidden_dim=hidden_dim,
            num_layers=args.layers,
            heads=args.heads,
            feat_dim=len(train_ds.feature_cols),
            dropout=args.dropout,
            qc_dim=qc_dim,
        )
    elif args.model == "schnet_backbone":
        model = SchNetBackboneRanker(
            hidden_dim=hidden_dim,
            num_layers=args.layers,
            feat_dim=len(train_ds.feature_cols),
            dropout=args.dropout,
            qc_dim=qc_node_dim,
        )
    elif args.model == "schnet_3d":
        model = SchNetBackboneRanker(
            hidden_dim=hidden_dim,
            num_layers=args.layers,
            feat_dim=len(train_ds.feature_cols),
            dropout=args.dropout,
            qc_dim=0,
        )
    elif args.model == "painn":
        model = PaiNNBackboneRanker(
            hidden_dim=hidden_dim,
            num_layers=args.layers,
            feat_dim=len(train_ds.feature_cols),
            dropout=args.dropout,
            global_attention=args.global_attention,
            global_attention_heads=args.global_attention_heads,
            global_attention_layers=args.global_attention_layers,
            global_attention_dropout=args.global_attention_dropout,
        )
    elif args.model == "reaction_graph_transformer":
        model = ReactionGraphTransformerRanker(
            hidden_dim=hidden_dim,
            num_layers=args.layers,
            feat_dim=len(train_ds.feature_cols),
            dropout=args.dropout,
            qc_dim=qc_dim,
            reaction_heads=args.reaction_transformer_heads,
            reaction_layers=args.reaction_transformer_layers,
            reaction_dropout=args.reaction_transformer_dropout,
        )
    elif args.model == "painn_physics":
        model = PhysicsInjectedPaiNNRanker(
            hidden_dim=hidden_dim,
            num_layers=args.layers,
            feat_dim=len(train_ds.feature_cols),
            qc_dim=qc_dim,
            fusion_strategy=args.physics_fusion,
            dropout=args.dropout,
            global_attention=args.global_attention,
            global_attention_heads=args.global_attention_heads,
            global_attention_layers=args.global_attention_layers,
            global_attention_dropout=args.global_attention_dropout,
        )
    elif args.model == "painn_precision":
        llm_feature_cols = list(getattr(train_ds, "llm_feature_cols", []) or [])
        model = PaiNNPrecisionHead(
            hidden_dim=hidden_dim,
            num_layers=args.layers,
            dropout=args.dropout,
            feat_dim=len(train_ds.feature_cols),
            qc_dim=qc_dim,
            physics_fusion=args.physics_fusion,
            llm_dim=len(llm_feature_cols),
            native_pair_dim=int(getattr(train_ds, "native_pair_summary_dim", 0)),
            native_pair_token_dim=int(getattr(train_ds, "native_pair_token_dim", 0)),
            gate_dropout=args.dropout,
            min_llm_confidence=args.llm_gate_min_confidence,
            gate_confidence_sharpness=args.llm_gate_confidence_sharpness,
            gate_density_center=args.llm_gate_density_center,
            gate_density_sharpness=args.llm_gate_density_sharpness,
            gate_adaptive_strength=args.llm_gate_adaptive_strength,
            llm_fusion_mode=args.llm_fusion_mode,
            llm_cross_attention_heads=args.llm_cross_attention_heads,
            llm_conditioned_scale=args.llm_conditioned_scale,
            llm_semantic_encoder=args.llm_semantic_encoder,
            llm_branch_index_map=llm_semantic_branch_index_map(llm_feature_cols),
            llm_branch_column_map=llm_semantic_branch_columns_from_feature_columns(llm_feature_cols),
            llm_group_index_map=llm_semantic_group_index_map(llm_feature_cols),
            llm_group_column_map=llm_semantic_group_columns_from_feature_columns(llm_feature_cols),
            llm_group_top_k=args.llm_semantic_group_top_k,
            llm_local_meta_gate=args.llm_local_meta_gate,
            llm_gate_product_focus_scale=args.llm_gate_product_focus_scale,
            semantic_physical_bridge=args.llm_physical_bridge,
            pair_conditioned_bridge=args.llm_pair_conditioned_bridge,
            pair_conditioned_bridge_locus=args.llm_pair_conditioned_bridge_locus,
            global_attention=args.global_attention,
            global_attention_heads=args.global_attention_heads,
            global_attention_layers=args.global_attention_layers,
            global_attention_dropout=args.global_attention_dropout,
            interaction_cross_attention=args.interaction_cross_attention,
            node_qc_dim=node_aux_dim,
            node_qc_gate_dim=node_aux_dim,
            reaction_center_coupling=args.reaction_center_coupling,
            reaction_center_pair_feature_dim=int(getattr(train_ds, "reaction_center_feature_dim", 0)),
            kinetic_relay=args.kinetic_relay,
        )
    elif args.model == "painn_attn_bias":
        model = PaiNNAttentionBiasRanker(
            hidden_dim=hidden_dim,
            num_layers=args.layers,
            feat_dim=len(train_ds.feature_cols),
            dropout=args.dropout,
            attn_heads=args.heads,
            bias_hidden_dim=hidden_dim,
            llm_bias_scale=args.llm_bias_scale,
            bias_aggregation=args.llm_bias_aggregation,
        )
    elif args.model == "painn_llm":
        model = AlignmentGuidedPaiNNRanker(
            hidden_dim=hidden_dim,
            num_layers=args.layers,
            feat_dim=len(train_ds.feature_cols),
            dropout=args.dropout,
            alignment_mode=args.alignment,
            edge_weight_mode=args.llm_edge_mode,
        )
    elif args.model == "dimenetpp":
        model = DimeNetBackboneRanker(
            hidden_dim=hidden_dim,
            num_layers=args.layers,
            feat_dim=len(train_ds.feature_cols),
            dropout=args.dropout,
        )
    elif args.model == "gemnet_dt":
        model = GemNetDTBackboneRanker(
            hidden_dim=hidden_dim,
            num_layers=args.layers,
            feat_dim=len(train_ds.feature_cols),
            dropout=args.dropout,
        )
    elif args.model == "shared_schnet":
        model = SharedSchNetBackboneRanker(
            hidden_dim=hidden_dim,
            num_layers=args.layers,
            dropout=args.dropout,
        )
    elif args.model == "equivariant":
        model = DualEquivariantGNN(
            hidden_dim=hidden_dim,
            num_layers=args.layers,
            heads=args.heads,
            feat_dim=len(train_ds.feature_cols),
            dropout=args.dropout,
        )
    elif args.model == "alignment":
        model = AlignmentGuidedGNN(
            hidden_dim=hidden_dim,
            num_layers=args.layers,
            heads=args.heads,
            feat_dim=len(train_ds.feature_cols),
            dropout=args.dropout,
            alignment_mode=args.alignment,
        )
    elif args.model == "dual_path":
        model = LiteratureGuidedDualPathGNN(
            hidden_dim=hidden_dim,
            num_layers=args.layers,
            heads=args.heads,
            feat_dim=len(train_ds.feature_cols),
            dropout=args.dropout,
            alignment_mode=args.alignment,
        )
    elif args.model == "gat":
        model = DualGATRanker(
            hidden_dim=hidden_dim,
            num_layers=args.layers,
            heads=args.heads,
            feat_dim=len(train_ds.feature_cols),
            dropout=args.dropout,
        )
    else:
        model = SSGNNCombined(
            hidden_dim=hidden_dim,
            edge_hidden_dim=hidden_dim,
            num_layers=args.layers,
            feat_dim=len(train_ds.feature_cols),
            dropout=args.dropout,
        )
    model = model.to(args.device)
    if hasattr(model, "ablation_disable_global_background"):
        model.ablation_disable_global_background = bool(args.llm_disable_global_background)

    if args.init_from:
        checkpoint = torch.load(args.init_from, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        if hasattr(model, "struct_head"):
            has_struct = any(key.startswith("struct_head.") for key in state_dict)
            if not has_struct:
                head_keys = {k: v for k, v in state_dict.items() if k.startswith("head.")}
                for key, value in head_keys.items():
                    mapped = "struct_head." + key[len("head.") :]
                    state_dict[mapped] = value
        model_state = model.state_dict()
        filtered_state = {}
        skipped_mismatch = []
        for key, value in state_dict.items():
            target = model_state.get(key)
            if target is None:
                filtered_state[key] = value
                continue
            if getattr(target, "shape", None) != getattr(value, "shape", None):
                skipped_mismatch.append((key, tuple(value.shape), tuple(target.shape)))
                continue
            filtered_state[key] = value
        missing = model.load_state_dict(filtered_state, strict=False)
        log(
            "Init-from loaded "
            f"missing_keys={len(missing.missing_keys)} "
            f"unexpected_keys={len(missing.unexpected_keys)}"
        )
        if skipped_mismatch:
            preview = ", ".join(
                f"{name}:{src}->{dst}" for name, src, dst in skipped_mismatch[:5]
            )
            if len(skipped_mismatch) > 5:
                preview += f", ... (+{len(skipped_mismatch) - 5} more)"
            log(f"Init-from skipped shape mismatches={len(skipped_mismatch)} [{preview}]")
        if args.safe_hybrid_init:
            safe_actions = _safe_hybrid_init(model, missing.missing_keys, skipped_mismatch)
            preview = ", ".join(safe_actions[:5]) if safe_actions else "none"
            if len(safe_actions) > 5:
                preview += f", ... (+{len(safe_actions) - 5} more)"
            log(f"Safe hybrid init applied to {len(safe_actions)} parameters [{preview}]")
    if args.freeze_backbone:
        frozen = _freeze_backbone(model)
        log(f"Frozen backbone modules: {', '.join(frozen) if frozen else 'none'}")
    if args.freeze_physical_base:
        frozen = _freeze_physical_base(model)
        log(f"Frozen physical-base modules: {', '.join(frozen) if frozen else 'none'}")
    selector_result = None
    if args.trainable_module_prefix:
        selector_result = _apply_trainable_module_prefix_selector(model, args.trainable_module_prefix)
        log(
            "Applied trainable-module"
            "-prefix selector: "
            f"requested={selector_result['requested_prefixes']} "
            f"matched={selector_result['matched_prefixes']} "
            f"trainable_parameters={selector_result['trainable_parameter_count']}"
        )

    optimizer = torch.optim.Adam(
        [param for param in model.parameters() if param.requires_grad],
        lr=lr,
        weight_decay=args.weight_decay,
    )
    scheduler = None
    if args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_scheduler_factor,
            patience=args.lr_scheduler_patience,
            min_lr=args.lr_scheduler_min_lr,
        )
    swa_model = None
    swa_scheduler = None
    swa_updates = 0
    swa_start_epoch = None
    swa_effective_lr = None
    if args.swa:
        swa_start_epoch = args.swa_start
        if swa_start_epoch is None:
            swa_start_epoch = max(1, int(math.floor(args.epochs * 0.75)))
        swa_start_epoch = max(1, min(int(swa_start_epoch), int(args.epochs)))
        swa_lr = float(args.swa_lr if args.swa_lr is not None else lr * 0.5)
        swa_effective_lr = swa_lr
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(
            optimizer,
            swa_lr=swa_lr,
            anneal_epochs=max(1, int(args.swa_anneal_epochs)),
        )
        log(
            f"SWA enabled: start_epoch={swa_start_epoch} "
            f"swa_lr={swa_lr:.6g} anneal_epochs={args.swa_anneal_epochs}"
        )

    best_val = math.inf
    best_state = None
    best_epoch = 0
    max_grad_norm = 0.0
    grad_non_finite = False
    grad_groups = build_grad_groups(model) if args.grad_audit else None
    feature_slice = train_ds.qc_feature_slice if args.grad_audit else None
    grad_tracker: Dict[str, float] = {}
    if args.model == "ssgnn":
        input_keys = (
            "combined_z",
            "combined_pos",
            "combined_node_type",
            "combined_batch",
            "combined_edge_index",
            "combined_edge_attr",
            "combined_edge_batch",
            "features",
        )
    else:
        input_keys = (
            "cat_z",
            "cat_pos",
            "cat_batch",
            "cat_edge_index",
            "r1_z",
            "r1_pos",
            "r1_batch",
            "r1_edge_index",
            "r2_z",
            "r2_pos",
            "r2_batch",
            "r2_edge_index",
            "features",
        )
        if args.model in {
            "baseline",
            "schnet_backbone",
            "gated_cross",
            "schnet_qc_gate",
            "schnet_qc_attn",
            "painn_physics",
            "reaction_graph_transformer",
        }:
            input_keys = input_keys + ("qc_features",)
        if args.model == "painn_physics":
            input_keys = input_keys + ("temperature",)
        if args.model == "painn_precision":
            input_keys = input_keys + ("temperature",)
            if qc_dim > 0:
                input_keys = input_keys + ("qc_features",)
            if node_aux_dim > 0:
                input_keys = input_keys + ("cat_node_qc", "r1_node_qc", "r2_node_qc")
            input_keys = input_keys + (
                "llm_features",
                "llm_confidence",
                "exact_covered_indicator",
                "native_pair_summary",
                "native_pair_tokens",
                "native_pair_token_mask",
                "native_pair_expert_summaries",
                "native_pair_expert_tokens",
                "native_pair_expert_token_masks",
                "native_pair_expert_priors",
            )
            if args.interaction_cross_attention:
                input_keys = input_keys + (
                    "interaction_cat_r1_index",
                    "interaction_cat_r1_features",
                    "interaction_cat_r1_batch",
                    "interaction_cat_r2_index",
                    "interaction_cat_r2_features",
                    "interaction_cat_r2_batch",
                )
            if use_literature:
                input_keys = input_keys + ("cat_interactions", "r1_interactions", "r2_interactions")
            if args.reaction_center_coupling:
                input_keys = input_keys + (
                    "reaction_center_cat_r1_index",
                    "reaction_center_cat_r1_features",
                    "reaction_center_cat_r1_batch",
                    "reaction_center_cat_r2_index",
                    "reaction_center_cat_r2_features",
                    "reaction_center_cat_r2_batch",
                )
        if args.model == "schnet_node_qc":
            input_keys = input_keys + ("cat_node_qc", "r1_node_qc", "r2_node_qc")
        if use_literature and args.model in {"alignment", "dual_path", "painn_llm", "painn_attn_bias"}:
            input_keys = input_keys + (
                "cat_interactions",
                "r1_interactions",
                "r2_interactions",
            )
        if (
            use_literature
            and args.llm_bias_scale != 0.0
            and args.model in {"baseline", "gated_cross"}
        ):
            input_keys = input_keys + ("cross_bias",)
    focus_yield_threshold = None
    if args.focus_yield_weight > 0.0 and args.focus_yield_threshold is not None:
        focus_yield_threshold = (
            (float(args.focus_yield_threshold) - float(train_ds.y_mean)) / float(train_ds.y_std)
        )
    focus_coordination_threshold = None
    if args.focus_coordination_weight > 0.0:
        focus_coordination_threshold = _coordination_focus_threshold(
            train_ds,
            quantile=args.focus_coordination_quantile,
        )
        log(
            "Coordination focus threshold: "
            f"{focus_coordination_threshold if focus_coordination_threshold is not None else 'disabled'} "
            f"(quantile={args.focus_coordination_quantile})"
        )
    focus_epoch_kwargs = {
        "llm_rule_weight": args.llm_rule_weight,
        "llm_rule_min_confidence": args.llm_rule_min_confidence,
        "focus_exact_covered_weight": args.focus_exact_covered_weight,
        "focus_confidence_threshold": args.focus_confidence_threshold,
        "focus_confidence_weight": args.focus_confidence_weight,
        "llm_row_dropout_p": args.llm_row_dropout_p,
        "llm_row_dropout_confidence_scale": args.llm_row_dropout_confidence_scale,
        "llm_row_dropout_exact_scale": args.llm_row_dropout_exact_scale,
        "llm_row_dropout_propagated_scale": args.llm_row_dropout_propagated_scale,
        "llm_reliability_loss_weight": args.llm_reliability_loss_weight,
        "llm_reliability_loss_center": args.llm_reliability_loss_center,
        "llm_reliability_loss_sharpness": args.llm_reliability_loss_sharpness,
        "focus_yield_threshold": focus_yield_threshold,
        "focus_yield_weight": args.focus_yield_weight,
        "coordination_feature_dim": len(getattr(train_ds, "coordination_feature_cols", []) or []),
        "focus_coordination_threshold": focus_coordination_threshold,
        "focus_coordination_weight": args.focus_coordination_weight,
        "focus_electronic_frontier_weight": args.focus_electronic_frontier_weight,
        "focus_kinetic_relay_weight": args.focus_kinetic_relay_weight,
        "reaction_center_aux_weight": args.reaction_center_aux_weight,
        "exact_covered_consistency_weight": args.exact_covered_consistency_weight,
    }
    llm_conditioned_scale_curriculum_enabled = (
        args.llm_conditioned_scale_curriculum_start is not None
        and hasattr(model, "condition_scale")
        and getattr(model, "condition_scale") is not None
    )
    llm_exact_row_scale_curriculum_enabled = args.llm_exact_row_scale_curriculum_start is not None
    llm_propagated_row_scale_curriculum_enabled = args.llm_propagated_row_scale_curriculum_start is not None
    llm_conditioned_scale_curriculum_schedule = []
    llm_row_authority_schedule = []
    if args.llm_conditioned_scale_curriculum_start is not None and not llm_conditioned_scale_curriculum_enabled:
        raise ValueError(
            "--llm-conditioned-scale-curriculum-start requires a model with a conditioned semantic scale parameter."
        )
    if llm_conditioned_scale_curriculum_enabled:
        if llm_conditioned_scale_curriculum_two_stage_enabled:
            log(
                "LLM conditioned-scale curriculum enabled: "
                f"stage1 {args.llm_conditioned_scale_curriculum_start:.4f} -> "
                f"{args.llm_conditioned_scale_curriculum_stage1_target:.4f} over "
                f"{args.llm_conditioned_scale_curriculum_warmup_epochs} epochs, "
                f"then stage2 -> {args.llm_conditioned_scale:.4f} over "
                f"{args.llm_conditioned_scale_curriculum_stage2_warmup_epochs} epochs "
                f"(shape={args.llm_conditioned_scale_curriculum_shape})"
            )
        else:
            log(
                "LLM conditioned-scale curriculum enabled: "
                f"start={args.llm_conditioned_scale_curriculum_start:.4f} "
                f"target={args.llm_conditioned_scale:.4f} "
                f"warmup_epochs={args.llm_conditioned_scale_curriculum_warmup_epochs} "
                f"shape={args.llm_conditioned_scale_curriculum_shape}"
            )
    if llm_exact_row_scale_curriculum_enabled:
        log(
            "LLM exact-row authority curriculum enabled: "
            f"exact_start={args.llm_exact_row_scale_curriculum_start:.4f} "
            f"exact_target={args.llm_exact_row_scale:.4f} "
            f"warmup_epochs={args.llm_exact_row_scale_curriculum_warmup_epochs} "
            f"shape={args.llm_exact_row_scale_curriculum_shape}"
        )
    if llm_propagated_row_scale_curriculum_enabled:
        log(
            "LLM propagated-row authority curriculum enabled: "
            f"exact={args.llm_exact_row_scale:.4f} "
            f"propagated_start={args.llm_propagated_row_scale_curriculum_start:.4f} "
            f"propagated_target={args.llm_propagated_row_scale:.4f} "
            f"warmup_epochs={args.llm_propagated_row_scale_curriculum_warmup_epochs} "
            f"shape={args.llm_propagated_row_scale_curriculum_shape}"
        )
    if abs(float(args.llm_propagated_row_confidence_floor) - 1.0) > 1e-12 or abs(
        float(args.llm_propagated_row_confidence_power) - 1.0
    ) > 1e-12:
        log(
            "LLM propagated-row confidence schedule enabled: "
            f"confidence_floor={args.llm_propagated_row_confidence_floor:.4f} "
            f"confidence_power={args.llm_propagated_row_confidence_power:.4f}"
        )
    if abs(float(args.llm_propagated_row_temperature_floor) - 1.0) > 1e-12 or abs(
        float(args.llm_propagated_row_temperature_power) - 1.0
    ) > 1e-12:
        log(
            "LLM propagated-row temperature schedule enabled: "
            f"temperature_floor={args.llm_propagated_row_temperature_floor:.4f} "
            f"temperature_power={args.llm_propagated_row_temperature_power:.4f}"
        )
    for epoch in range(1, args.epochs + 1):
        curriculum_scale = _apply_llm_conditioned_scale_curriculum(
            model,
            epoch=epoch,
            enabled=llm_conditioned_scale_curriculum_enabled,
            start=0.0 if args.llm_conditioned_scale_curriculum_start is None else args.llm_conditioned_scale_curriculum_start,
            target=args.llm_conditioned_scale,
            warmup_epochs=args.llm_conditioned_scale_curriculum_warmup_epochs,
            shape=args.llm_conditioned_scale_curriculum_shape,
            stage1_target=args.llm_conditioned_scale_curriculum_stage1_target,
            stage2_warmup_epochs=args.llm_conditioned_scale_curriculum_stage2_warmup_epochs,
        )
        current_exact_row_scale = _curriculum_row_scale(
            epoch,
            target=float(args.llm_exact_row_scale),
            start=args.llm_exact_row_scale_curriculum_start,
            warmup_epochs=int(args.llm_exact_row_scale_curriculum_warmup_epochs),
            shape=str(args.llm_exact_row_scale_curriculum_shape),
        )
        current_propagated_row_scale = _curriculum_row_scale(
            epoch,
            target=float(args.llm_propagated_row_scale),
            start=args.llm_propagated_row_scale_curriculum_start,
            warmup_epochs=int(args.llm_propagated_row_scale_curriculum_warmup_epochs),
            shape=str(args.llm_propagated_row_scale_curriculum_shape),
        )
        if curriculum_scale is not None:
            llm_conditioned_scale_curriculum_schedule.append(
                {
                    "epoch": int(epoch),
                    "llm_conditioned_scale": float(curriculum_scale),
                }
            )
        llm_row_authority_schedule.append(
            {
                "epoch": int(epoch),
                "llm_exact_row_scale": float(current_exact_row_scale),
                "llm_propagated_row_scale": float(current_propagated_row_scale),
            }
        )
        grad_tracker = {}
        train_loss, _, _ = run_epoch(
            model,
            train_loader,
            args.device,
            optimizer=optimizer,
            input_keys=input_keys,
            loss_mode=args.loss,
            rank_weight=args.rank_weight,
            rank_margin=args.rank_margin,
            attn_audit=args.attn_audit,
            grad_tracker=grad_tracker,
            grad_groups=grad_groups,
            feature_slice=feature_slice,
            coord_jitter_std=args.coord_jitter_std,
            llm_exact_row_scale=current_exact_row_scale,
            llm_propagated_row_scale=current_propagated_row_scale,
            base_llm_exact_row_scale=float(args.llm_exact_row_scale),
            base_llm_propagated_row_scale=float(args.llm_propagated_row_scale),
            llm_propagated_row_confidence_floor=float(args.llm_propagated_row_confidence_floor),
            llm_propagated_row_confidence_power=float(args.llm_propagated_row_confidence_power),
            llm_propagated_row_temperature_floor=float(args.llm_propagated_row_temperature_floor),
            llm_propagated_row_temperature_power=float(args.llm_propagated_row_temperature_power),
            **focus_epoch_kwargs,
        )
        val_loss, val_true, val_pred = run_epoch(
            model,
            val_loader,
            args.device,
            input_keys=input_keys,
            loss_mode=args.loss,
            rank_weight=args.rank_weight,
            rank_margin=args.rank_margin,
            llm_exact_row_scale=current_exact_row_scale,
            llm_propagated_row_scale=current_propagated_row_scale,
            base_llm_exact_row_scale=float(args.llm_exact_row_scale),
            base_llm_propagated_row_scale=float(args.llm_propagated_row_scale),
            llm_propagated_row_confidence_floor=float(args.llm_propagated_row_confidence_floor),
            llm_propagated_row_confidence_power=float(args.llm_propagated_row_confidence_power),
            llm_propagated_row_temperature_floor=float(args.llm_propagated_row_temperature_floor),
            llm_propagated_row_temperature_power=float(args.llm_propagated_row_temperature_power),
            **focus_epoch_kwargs,
        )
        val_metrics = compute_metrics(val_true, val_pred)
        current_lr = optimizer.param_groups[0]["lr"]
        log(
            f"Epoch {epoch:03d} | lr {current_lr:.6g} | train_loss {train_loss:.4f} | "
            f"val_loss {val_loss:.4f} | val_mae {val_metrics['mae']:.4f} | "
            f"val_r2 {val_metrics['r2']:.4f} | val_pearson {val_metrics['pearson']:.4f}"
            + (
                ""
                if curriculum_scale is None
                else f" | llm_conditioned_scale {float(curriculum_scale):.4f}"
            )
            + (
                ""
                if not llm_propagated_row_scale_curriculum_enabled
                else (
                    f" | llm_exact_row_scale {float(current_exact_row_scale):.4f}"
                    f" | llm_propagated_row_scale {float(current_propagated_row_scale):.4f}"
                )
            )
        )
        if swa_model is not None and swa_scheduler is not None and swa_start_epoch is not None and epoch >= swa_start_epoch:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            swa_updates += 1
            log(
                f"SWA update {swa_updates:03d} | "
                f"swa_lr {optimizer.param_groups[0]['lr']:.6g}"
            )
        elif scheduler is not None:
            prev_lr = float(current_lr)
            scheduler.step(val_loss)
            next_lr = float(optimizer.param_groups[0]["lr"])
            if next_lr < prev_lr:
                log(f"LR scheduler reduced lr: {prev_lr:.6g} -> {next_lr:.6g}")
        if grad_tracker:
            max_grad = grad_tracker.get("max_grad_norm", 0.0)
            non_finite = "yes" if grad_tracker.get("non_finite") else "no"
            log(f"Grad norm max: {max_grad:.4f} | non_finite={non_finite}")
            max_grad_norm = max(max_grad_norm, max_grad)
            if grad_tracker.get("non_finite"):
                grad_non_finite = True
            if args.grad_audit:
                log_grad_audit(log, grad_tracker)
            if args.attn_audit:
                log_attention_audit(log, grad_tracker)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_epoch = epoch

    swa_applied = False
    if swa_model is not None and swa_updates > 0:
        _update_batchnorm_stats(
            swa_model,
            train_loader,
            args.device,
            input_keys=input_keys,
        )
        model = swa_model.module.to(args.device)
        swa_applied = True
        log(f"Using SWA weights for final evaluation ({swa_updates} averaged checkpoints).")
    elif best_state is not None:
        model.load_state_dict(best_state)

    if args.model_save_path:
        save_path = Path(args.model_save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save(
            {
                "state_dict": state_dict,
                "model": args.model,
                "feature_cols": train_ds.feature_cols,
                "qc_feature_cols": getattr(train_ds, "qc_feature_cols", []),
                "config": vars(args),
            },
            save_path,
        )

    _, train_true, train_pred = run_epoch(
        model,
        train_loader,
        args.device,
        input_keys=input_keys,
        loss_mode=args.loss,
        rank_weight=args.rank_weight,
        rank_margin=args.rank_margin,
        llm_exact_row_scale=float(args.llm_exact_row_scale),
        llm_propagated_row_scale=float(args.llm_propagated_row_scale),
        base_llm_exact_row_scale=float(args.llm_exact_row_scale),
        base_llm_propagated_row_scale=float(args.llm_propagated_row_scale),
        llm_propagated_row_confidence_floor=float(args.llm_propagated_row_confidence_floor),
        llm_propagated_row_confidence_power=float(args.llm_propagated_row_confidence_power),
        llm_propagated_row_temperature_floor=float(args.llm_propagated_row_temperature_floor),
        llm_propagated_row_temperature_power=float(args.llm_propagated_row_temperature_power),
        **focus_epoch_kwargs,
    )
    _, val_true, val_pred = run_epoch(
        model,
        val_loader,
        args.device,
        input_keys=input_keys,
        loss_mode=args.loss,
        rank_weight=args.rank_weight,
        rank_margin=args.rank_margin,
        llm_exact_row_scale=float(args.llm_exact_row_scale),
        llm_propagated_row_scale=float(args.llm_propagated_row_scale),
        base_llm_exact_row_scale=float(args.llm_exact_row_scale),
        base_llm_propagated_row_scale=float(args.llm_propagated_row_scale),
        llm_propagated_row_confidence_floor=float(args.llm_propagated_row_confidence_floor),
        llm_propagated_row_confidence_power=float(args.llm_propagated_row_confidence_power),
        llm_propagated_row_temperature_floor=float(args.llm_propagated_row_temperature_floor),
        llm_propagated_row_temperature_power=float(args.llm_propagated_row_temperature_power),
        **focus_epoch_kwargs,
    )
    if len(test_ds) > 0:
        _, test_true, test_pred = run_epoch(
            model,
            test_loader,
            args.device,
            input_keys=input_keys,
            loss_mode=args.loss,
            rank_weight=args.rank_weight,
            rank_margin=args.rank_margin,
            llm_exact_row_scale=float(args.llm_exact_row_scale),
            llm_propagated_row_scale=float(args.llm_propagated_row_scale),
            base_llm_exact_row_scale=float(args.llm_exact_row_scale),
            base_llm_propagated_row_scale=float(args.llm_propagated_row_scale),
            llm_propagated_row_confidence_floor=float(args.llm_propagated_row_confidence_floor),
            llm_propagated_row_confidence_power=float(args.llm_propagated_row_confidence_power),
            llm_propagated_row_temperature_floor=float(args.llm_propagated_row_temperature_floor),
            llm_propagated_row_temperature_power=float(args.llm_propagated_row_temperature_power),
            **focus_epoch_kwargs,
        )
    else:
        test_true = np.array([])
        test_pred = np.array([])

    propagated_confidence_schedule_profile = _build_propagated_confidence_schedule_profile(
        train_ds.samples,
        authority_schedule=llm_row_authority_schedule,
        floor=float(args.llm_propagated_row_confidence_floor),
        power=float(args.llm_propagated_row_confidence_power),
    )
    propagated_temperature_schedule_profile = _build_propagated_temperature_schedule_profile(
        train_ds.samples,
        authority_schedule=llm_row_authority_schedule,
        floor=float(args.llm_propagated_row_temperature_floor),
        power=float(args.llm_propagated_row_temperature_power),
    )
    y_mean = train_ds.y_mean
    y_std = train_ds.y_std
    val_true_dn = val_true * y_std + y_mean
    val_pred_dn = val_pred * y_std + y_mean
    test_true_dn = test_true * y_std + y_mean if test_true.size else test_true
    test_pred_dn = test_pred * y_std + y_mean if test_pred.size else test_pred

    train_metrics_dn = compute_metrics(train_true * y_std + y_mean, train_pred * y_std + y_mean)
    val_metrics_dn = compute_metrics(val_true_dn, val_pred_dn)
    test_metrics_dn = compute_metrics(test_true_dn, test_pred_dn)
    global_true_parts = [train_true * y_std + y_mean, val_true_dn]
    global_pred_parts = [train_pred * y_std + y_mean, val_pred_dn]
    if test_true_dn.size:
        global_true_parts.append(test_true_dn)
        global_pred_parts.append(test_pred_dn)
    global_true_dn = np.concatenate(global_true_parts, axis=0) if global_true_parts else np.array([])
    global_pred_dn = np.concatenate(global_pred_parts, axis=0) if global_pred_parts else np.array([])
    global_metrics_dn = compute_metrics(global_true_dn, global_pred_dn)
    split_counts = {
        "train": int(train_true.shape[0]),
        "val": int(val_true.shape[0]),
        "test": int(test_true.shape[0]),
        "global": int(global_true_dn.shape[0]),
    }

    qc_descriptor_importance = None
    qc_descriptor_importance_split = None
    if args.model == "painn_physics" and getattr(train_ds, "qc_feature_cols", None):
        importance_loader = test_loader if len(test_ds) > 0 else val_loader
        qc_descriptor_importance_split = "test" if len(test_ds) > 0 else "val"
        qc_descriptor_importance = compute_qc_descriptor_importance(
            model=model,
            loader=importance_loader,
            device=args.device,
            input_keys=input_keys,
            qc_feature_cols=train_ds.qc_feature_cols,
        )
        top_entries = qc_descriptor_importance[:5]
        if top_entries:
            top_text = " | ".join(
                f"{entry['feature']}={entry['grad_abs_mean']:.4e}" for entry in top_entries
            )
            log(f"QC gated descriptor importance (top-5 grad_abs): {top_text}")

    high_confidence = None
    if use_literature:
        threshold = float(args.high_confidence_threshold)
        val_conf = _extract_llm_confidence(val_ds.samples)
        test_conf = _extract_llm_confidence(test_ds.samples) if len(test_ds) > 0 else np.zeros((0,))
        high_confidence = {
            "threshold": threshold,
            "val": _subset_metrics(val_true_dn, val_pred_dn, val_conf >= threshold),
            "test": _subset_metrics(test_true_dn, test_pred_dn, test_conf >= threshold),
        }

    log("\nFinal Metrics (de-normalized):")
    log(
        f"Train MAE: {train_metrics_dn['mae']:.4f} | Train R2: {train_metrics_dn['r2']:.4f} | "
        f"Train Pearson: {train_metrics_dn['pearson']:.4f}"
    )
    log(
        f"Valid MAE: {val_metrics_dn['mae']:.4f} | Valid R2: {val_metrics_dn['r2']:.4f} | "
        f"Valid Pearson: {val_metrics_dn['pearson']:.4f}"
    )
    if len(test_ds) > 0:
        log(
            f"Test  MAE: {test_metrics_dn['mae']:.4f} | Test  R2: {test_metrics_dn['r2']:.4f} | "
            f"Test  Pearson: {test_metrics_dn['pearson']:.4f}"
        )
    else:
        log("Test  MAE: n/a | Test  R2: n/a | Test  Pearson: n/a (no test split)")
    log(
        f"Global MAE: {global_metrics_dn['mae']:.4f} | Global R2: {global_metrics_dn['r2']:.4f} | "
        f"Global Pearson: {global_metrics_dn['pearson']:.4f}"
    )

    attn_bias_summary = None
    if args.attn_audit and grad_tracker:
        attn_bias_summary = {}
        for key, value in grad_tracker.items():
            if not key.startswith("attn_bias_ratio_"):
                continue
            pair = key.replace("attn_bias_ratio_", "")
            attn_bias_summary[pair] = {
                "ratio": float(value),
                "share": float(grad_tracker.get(f"attn_bias_share_{pair}", 0.0)),
                "hotspots": float(grad_tracker.get(f"attn_bias_hotspots_{pair}", 0.0)),
            }

    current_r2 = test_metrics_dn["r2"] if test_true.size else val_metrics_dn["r2"]
    current_metric_label = "test_r2" if test_true.size else "val_r2"

    summary = {
        "lr": lr,
        "hidden_dim": hidden_dim,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "layers": args.layers,
        "heads": args.heads,
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
        "lr_scheduler": args.lr_scheduler,
        "lr_scheduler_patience": args.lr_scheduler_patience,
        "lr_scheduler_factor": args.lr_scheduler_factor,
        "lr_scheduler_min_lr": args.lr_scheduler_min_lr,
        "coord_jitter_std": args.coord_jitter_std,
        "final_lr": optimizer.param_groups[0]["lr"],
        "seed": args.seed,
        "split_seed": split_seed,
        "samples": args.samples,
        "data_path": args.data,
        "dataset": args.dataset,
        "split_strategy_requested": args.split_strategy,
        "signature_columns_requested": args.signature_columns,
        "model": args.model,
        "alignment": args.alignment,
        "qc_fusion": args.qc_fusion,
        "qc_scale": args.qc_scale,
        "qc_node_scale": args.qc_node_scale,
        "qc_global_scale": args.qc_global_scale,
        "qc_node_mode": args.qc_node_mode,
        "coordination_features": args.coordination_features,
        "coordination_node_scale": args.coordination_node_scale,
        "coordination_feature_cols": getattr(train_ds, "coordination_feature_cols", []),
        "reaction_center_coupling": args.reaction_center_coupling,
        "kinetic_relay": args.kinetic_relay,
        "reaction_center_feature_dim": int(getattr(train_ds, "reaction_center_feature_dim", 0)),
        "reaction_center_aux_weight": args.reaction_center_aux_weight,
        "exact_covered_consistency_weight": args.exact_covered_consistency_weight,
        "node_aux_dim": node_aux_dim,
        "physics_fusion": args.physics_fusion,
        "qc_weight_path": args.qc_weight_path,
        "qc_weight_mode": args.qc_weight_mode,
        "qc_weight_scale": args.qc_weight_scale,
        "qc_weight_min": args.qc_weight_min,
        "qc_weight_max": args.qc_weight_max,
        "temperature_col": getattr(train_ds, "temperature_col", None),
        "kinetic_interaction_enabled": bool(getattr(model, "kinetic_interaction_enabled", False)),
        "llm_bias_scale": args.llm_bias_scale,
        "llm_bias_mode": args.llm_bias_mode,
        "llm_bias_aggregation": args.llm_bias_aggregation,
        "llm_edge_mode": args.llm_edge_mode,
        "llm_rule_weight": args.llm_rule_weight,
        "llm_rule_min_confidence": args.llm_rule_min_confidence,
        "focus_exact_covered_weight": args.focus_exact_covered_weight,
        "focus_confidence_threshold": args.focus_confidence_threshold,
        "focus_confidence_weight": args.focus_confidence_weight,
        "llm_row_dropout_p": args.llm_row_dropout_p,
        "llm_row_dropout_confidence_scale": args.llm_row_dropout_confidence_scale,
        "llm_row_dropout_exact_scale": args.llm_row_dropout_exact_scale,
        "llm_row_dropout_propagated_scale": args.llm_row_dropout_propagated_scale,
        "llm_reliability_loss_weight": args.llm_reliability_loss_weight,
        "llm_reliability_loss_center": args.llm_reliability_loss_center,
        "llm_reliability_loss_sharpness": args.llm_reliability_loss_sharpness,
        "focus_yield_threshold": args.focus_yield_threshold,
        "focus_yield_weight": args.focus_yield_weight,
        "focus_coordination_quantile": args.focus_coordination_quantile,
        "focus_coordination_weight": args.focus_coordination_weight,
        "focus_electronic_frontier_weight": args.focus_electronic_frontier_weight,
        "focus_kinetic_relay_weight": args.focus_kinetic_relay_weight,
        "focus_coordination_threshold": focus_coordination_threshold,
        "llm_gate_min_confidence": args.llm_gate_min_confidence,
        "llm_gate_confidence_sharpness": args.llm_gate_confidence_sharpness,
        "llm_gate_density_center": args.llm_gate_density_center,
        "llm_gate_density_sharpness": args.llm_gate_density_sharpness,
        "llm_gate_adaptive_strength": args.llm_gate_adaptive_strength,
        "llm_gate_product_focus_scale": args.llm_gate_product_focus_scale,
        "llm_feature_profile": getattr(train_ds, "llm_feature_profile", args.llm_feature_profile),
        "llm_missingness_contract": getattr(train_ds, "llm_missingness_contract", args.llm_missingness_contract),
        "llm_semantic_group_scales": dict(getattr(train_ds, "llm_semantic_group_scales", {}) or {}),
        "llm_semantic_feature_scales": dict(getattr(train_ds, "llm_semantic_feature_scales", {}) or {}),
        "llm_exact_row_scale": float(getattr(train_ds, "llm_exact_row_scale", args.llm_exact_row_scale)),
        "llm_exact_row_scale_curriculum_enabled": llm_exact_row_scale_curriculum_enabled,
        "llm_exact_row_scale_curriculum_start": (
            None if args.llm_exact_row_scale_curriculum_start is None else float(args.llm_exact_row_scale_curriculum_start)
        ),
        "llm_exact_row_scale_curriculum_warmup_epochs": int(args.llm_exact_row_scale_curriculum_warmup_epochs),
        "llm_exact_row_scale_curriculum_shape": str(args.llm_exact_row_scale_curriculum_shape),
        "llm_propagated_row_scale": float(
            getattr(train_ds, "llm_propagated_row_scale", args.llm_propagated_row_scale)
        ),
        "llm_propagated_row_scale_curriculum_enabled": llm_propagated_row_scale_curriculum_enabled,
        "llm_propagated_row_scale_curriculum_start": (
            None
            if args.llm_propagated_row_scale_curriculum_start is None
            else float(args.llm_propagated_row_scale_curriculum_start)
        ),
        "llm_propagated_row_scale_curriculum_warmup_epochs": int(
            args.llm_propagated_row_scale_curriculum_warmup_epochs
        ),
        "llm_propagated_row_scale_curriculum_shape": str(args.llm_propagated_row_scale_curriculum_shape),
        "llm_propagated_row_confidence_schedule_enabled": bool(propagated_confidence_schedule_profile["enabled"]),
        "llm_propagated_row_confidence_floor": float(args.llm_propagated_row_confidence_floor),
        "llm_propagated_row_confidence_power": float(args.llm_propagated_row_confidence_power),
        "llm_propagated_row_confidence_schedule_profile": propagated_confidence_schedule_profile,
        "llm_propagated_row_temperature_schedule_enabled": bool(propagated_temperature_schedule_profile["enabled"]),
        "llm_propagated_row_temperature_floor": float(args.llm_propagated_row_temperature_floor),
        "llm_propagated_row_temperature_power": float(args.llm_propagated_row_temperature_power),
        "llm_propagated_row_temperature_schedule_profile": propagated_temperature_schedule_profile,
        "llm_row_authority_schedule": llm_row_authority_schedule,
        "llm_fusion_mode": args.llm_fusion_mode,
        "llm_cross_attention_heads": args.llm_cross_attention_heads,
        "llm_conditioned_scale": args.llm_conditioned_scale,
        "llm_conditioned_scale_curriculum_enabled": llm_conditioned_scale_curriculum_enabled,
        "llm_conditioned_scale_curriculum_two_stage_enabled": llm_conditioned_scale_curriculum_two_stage_enabled,
        "llm_conditioned_scale_curriculum_start": (
            None
            if args.llm_conditioned_scale_curriculum_start is None
            else float(args.llm_conditioned_scale_curriculum_start)
        ),
        "llm_conditioned_scale_curriculum_warmup_epochs": int(args.llm_conditioned_scale_curriculum_warmup_epochs),
        "llm_conditioned_scale_curriculum_shape": str(args.llm_conditioned_scale_curriculum_shape),
        "llm_conditioned_scale_curriculum_stage1_target": (
            None
            if args.llm_conditioned_scale_curriculum_stage1_target is None
            else float(args.llm_conditioned_scale_curriculum_stage1_target)
        ),
        "llm_conditioned_scale_curriculum_stage2_warmup_epochs": (
            None
            if args.llm_conditioned_scale_curriculum_stage2_warmup_epochs is None
            else int(args.llm_conditioned_scale_curriculum_stage2_warmup_epochs)
        ),
        "llm_conditioned_scale_curriculum_schedule": llm_conditioned_scale_curriculum_schedule,
        "llm_physical_bridge": args.llm_physical_bridge,
        "llm_pair_conditioned_bridge": args.llm_pair_conditioned_bridge,
        "llm_pair_conditioned_bridge_locus": args.llm_pair_conditioned_bridge_locus,
        "llm_disable_global_background": args.llm_disable_global_background,
        "llm_local_meta_gate": list(args.llm_local_meta_gate or []),
        "llm_local_meta_gate_enabled": bool(getattr(model, "local_meta_gate_enabled", False)),
        "pair_conditioned_bridge_enabled": bool(getattr(model, "pair_conditioned_bridge_enabled", False)),
        "interaction_cross_attention_enabled": bool(getattr(model, "interaction_cross_attention_enabled", False)),
        "interaction_cross_attention_heads": int(getattr(model, "interaction_cross_attention_heads", 0)),
        "interaction_pair_mode": getattr(train_ds, "interaction_pair_mode", args.interaction_pair_mode),
        "interaction_pair_feature_dim": int(getattr(train_ds, "interaction_pair_feature_dim", 0)),
        "reaction_center_coupling_enabled": bool(getattr(model, "reaction_center_coupling_enabled", False)),
        "kinetic_relay_enabled": bool(getattr(model, "kinetic_relay_enabled", False)),
        "semantic_attention_bridge_enabled": bool(getattr(model, "semantic_attention_bridge_enabled", False)),
        "semantic_physical_bridge_enabled": bool(getattr(model, "semantic_physical_bridge_enabled", False)),
        "multi_scale_moe_enabled": bool(getattr(model, "multi_scale_moe_enabled", False)),
        "global_attention": args.global_attention,
        "global_attention_heads": args.global_attention_heads,
        "global_attention_layers": args.global_attention_layers,
        "global_attention_dropout": args.global_attention_dropout,
        "reaction_transformer_heads": args.reaction_transformer_heads,
        "reaction_transformer_layers": args.reaction_transformer_layers,
        "reaction_transformer_dropout": args.reaction_transformer_dropout,
        "swa": args.swa,
        "swa_start": swa_start_epoch,
        "swa_lr": swa_effective_lr,
        "swa_anneal_epochs": args.swa_anneal_epochs,
        "swa_updates": swa_updates,
        "swa_applied": swa_applied,
        "attn_audit": args.attn_audit,
        "use_coulomb": args.use_coulomb,
        "use_geometry": args.use_geometry,
        "use_numeric": args.use_numeric,
        "use_rdkit": args.use_rdkit,
        "use_morgan_fingerprint": args.use_morgan_fingerprint,
        "morgan_fingerprint_bits": args.morgan_fingerprint_bits,
        "morgan_fingerprint_radius": args.morgan_fingerprint_radius,
        "use_ref_data": args.use_ref_data,
        "disable_literature_branch": args.disable_literature_branch,
        "include_literature": use_literature,
        "literature_cache": args.literature_cache,
        "executed_command": " ".join(shlex.quote(arg) for arg in sys.argv),
        "llm_feature_dim": len(getattr(train_ds, "llm_feature_cols", []) or []),
        "llm_feature_cols": list(getattr(train_ds, "llm_feature_cols", []) or []),
        "native_pair_summary_dim": int(getattr(train_ds, "native_pair_summary_dim", 0)),
        "native_pair_token_dim": int(getattr(train_ds, "native_pair_token_dim", 0)),
        "native_pair_max_tokens": int(getattr(train_ds, "native_pair_max_tokens", 0)),
        "native_pair_max_experts": int(getattr(train_ds, "native_pair_max_experts", 0)),
        "native_pair_field_profile": native_pair_field_profile,
        "native_pair_field_profile_description": str(
            getattr(train_ds, "native_pair_field_profile_description", "")
        ),
        "native_pair_summary_group_scales": dict(
            getattr(train_ds, "native_pair_summary_group_scales", {}) or {}
        ),
        "native_pair_token_group_scales": dict(
            getattr(train_ds, "native_pair_token_group_scales", {}) or {}
        ),
        "native_pair_summary_feature_names": list(
            getattr(train_ds, "native_pair_summary_feature_names", []) or []
        ),
        "native_pair_token_feature_names": list(
            getattr(train_ds, "native_pair_token_feature_names", []) or []
        ),
        "llm_semantic_encoder": args.llm_semantic_encoder,
        "llm_semantic_group_top_k": args.llm_semantic_group_top_k,
        "llm_branch_columns": llm_semantic_branch_columns_from_feature_columns(
            list(getattr(train_ds, "llm_feature_cols", []) or [])
        ),
        "llm_branch_dims": {
            name: len(columns)
            for name, columns in llm_semantic_branch_columns_from_feature_columns(
                list(getattr(train_ds, "llm_feature_cols", []) or [])
            ).items()
        },
        "llm_pair_group_columns": llm_pair_group_columns_from_feature_columns(
            list(getattr(train_ds, "llm_feature_cols", []) or [])
        ),
        "llm_pair_group_dims": {
            name: len(columns)
            for name, columns in llm_pair_group_columns_from_feature_columns(
                list(getattr(train_ds, "llm_feature_cols", []) or [])
            ).items()
        },
        "llm_group_columns": llm_semantic_group_columns_from_feature_columns(
            list(getattr(train_ds, "llm_feature_cols", []) or [])
        ),
        "llm_group_dims": {
            name: len(columns)
            for name, columns in llm_semantic_group_columns_from_feature_columns(
                list(getattr(train_ds, "llm_feature_cols", []) or [])
            ).items()
        },
        "cross_bias_scale": args.cross_bias_scale,
        "model_save_path": args.model_save_path,
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        "val_split": args.val_split,
        "split_strategy": split_strategy,
        "split_group_source": split_group_source,
        "loss_mode": args.loss,
        "rank_weight": args.rank_weight,
        "rank_margin": args.rank_margin,
        "rank_group_cols": rank_group_cols,
        "rank_target_col": rank_target_col,
        "best_epoch": best_epoch,
        "grad_norm_max": max_grad_norm,
        "grad_non_finite": grad_non_finite,
        "attn_bias": attn_bias_summary,
        "llm_phys_audit": args.llm_phys_audit,
        "llm_phys_cutoff": args.llm_phys_cutoff,
        "baseline_summary": args.baseline_summary,
        "llm_physical_consistency": llm_phys_summary,
        "high_confidence": high_confidence,
        "init_from": args.init_from,
        "safe_hybrid_init": args.safe_hybrid_init,
        "freeze_backbone": args.freeze_backbone,
        "freeze_physical_base": args.freeze_physical_base,
        "trainable_module_prefix": list(args.trainable_module_prefix or []),
        "matched_trainable_module_prefix": []
        if selector_result is None
        else list(selector_result["matched_prefixes"]),
        "selected_trainable_parameter_names": []
        if selector_result is None
        else list(selector_result["matched_parameter_names"]),
        "selected_trainable_parameter_count": 0
        if selector_result is None
        else int(selector_result["trainable_parameter_count"]),
        "grad_audit": {
            k: v for k, v in grad_tracker.items() if k.startswith("grad_group_") or k.startswith("feature_grad_")
        }
        if args.grad_audit
        else None,
        "train": train_metrics_dn,
        "val": val_metrics_dn,
        "test": test_metrics_dn,
        "global": global_metrics_dn,
        "counts": split_counts,
        "qc_descriptor_importance_split": qc_descriptor_importance_split,
        "qc_descriptor_importance": qc_descriptor_importance,
    }
    _attach_baseline_delta(summary, args.baseline_summary, current_r2, current_metric_label)
    if "r2_delta" in summary:
        delta = summary["r2_delta"]
        log(
            "LLM R2 delta: "
            f"baseline={delta['baseline_r2']:.4f} current={delta['current_r2']:.4f} "
            f"delta={delta['delta']:.4f} ({delta['metric']})"
        )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/ml-cb-database.xlsx")
    parser.add_argument("--dataset", choices=["ml-cb", "ml-borylation"], default="ml-cb")
    parser.add_argument(
        "--model",
        choices=[
            "baseline",
            "gated_cross",
            "schnet_backbone",
            "schnet_3d",
            "painn",
            "reaction_graph_transformer",
            "painn_physics",
            "painn_precision",
            "painn_attn_bias",
            "painn_llm",
            "dimenetpp",
            "gemnet_dt",
            "schnet_node_qc",
            "schnet_qc_gate",
            "schnet_qc_attn",
            "shared_schnet",
            "equivariant",
            "alignment",
            "dual_path",
            "gat",
            "ssgnn",
        ],
        default="baseline",
    )
    parser.add_argument("--alignment", choices=["rule", "learned"], default="rule")
    parser.add_argument("--combined_k", type=int, default=8)
    parser.add_argument("--combined_cutoff", type=float, default=5.0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--lr-scheduler",
        choices=["none", "plateau"],
        default="none",
        help="Optional learning-rate scheduler.",
    )
    parser.add_argument(
        "--lr-scheduler-patience",
        type=int,
        default=5,
        help="Plateau scheduler patience in epochs.",
    )
    parser.add_argument(
        "--lr-scheduler-factor",
        type=float,
        default=0.5,
        help="Plateau scheduler decay factor.",
    )
    parser.add_argument(
        "--lr-scheduler-min-lr",
        type=float,
        default=1e-6,
        help="Minimum learning rate for plateau scheduling.",
    )
    parser.add_argument(
        "--coord-jitter-std",
        type=float,
        default=0.0,
        help="Gaussian coordinate jitter (Angstrom) applied during training only.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Optional random seed used only for train/val/test splitting (defaults to --seed).",
    )
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument(
        "--global-attention",
        action="store_true",
        help="Enable a transformer-style global attention block across catalyst/reactant graph embeddings.",
    )
    parser.add_argument(
        "--global-attention-heads",
        type=int,
        default=4,
        help="Attention heads for global attention transformer block.",
    )
    parser.add_argument(
        "--global-attention-layers",
        type=int,
        default=1,
        help="Transformer layers for global attention block.",
    )
    parser.add_argument(
        "--global-attention-dropout",
        type=float,
        default=0.1,
        help="Dropout used in global attention transformer block.",
    )
    parser.add_argument(
        "--no-interaction-cross-attention",
        dest="interaction_cross_attention",
        action="store_false",
        help="Disable the catalyst/substrate shell cross-attention block in PaiNN precision runs.",
    )
    parser.set_defaults(interaction_cross_attention=True)
    parser.add_argument(
        "--interaction-pair-mode",
        choices=["classic", "late_metal_dense"],
        default="classic",
        help="Atomic interaction pair feature recipe used by catalyst/substrate cross-attention.",
    )
    parser.add_argument(
        "--reaction-transformer-heads",
        type=int,
        default=4,
        help="Attention heads in the reaction-graph transformer pivot model.",
    )
    parser.add_argument(
        "--reaction-transformer-layers",
        type=int,
        default=2,
        help="Transformer depth in the reaction-graph transformer pivot model.",
    )
    parser.add_argument(
        "--reaction-transformer-dropout",
        type=float,
        default=0.1,
        help="Dropout used in the reaction-graph transformer pivot model.",
    )
    parser.add_argument(
        "--swa",
        action="store_true",
        help="Enable stochastic weight averaging during the final training phase.",
    )
    parser.add_argument(
        "--swa-start",
        type=int,
        default=None,
        help="Epoch index (1-based) to start SWA; defaults to 75% of training.",
    )
    parser.add_argument(
        "--swa-lr",
        type=float,
        default=None,
        help="Learning rate used by SWA; defaults to 0.5 * base lr.",
    )
    parser.add_argument(
        "--swa-anneal-epochs",
        type=int,
        default=5,
        help="Number of annealing epochs for SWA LR schedule.",
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--cross-bias-scale", type=float, default=0.0)
    parser.add_argument(
        "--llm-bias-scale",
        type=float,
        default=0.0,
        help="Scale for LLM-derived hotspot attention bias (0 disables).",
    )
    parser.add_argument(
        "--llm-bias-mode",
        choices=["sum", "key", "outer"],
        default="sum",
        help="How to combine query/key hotspot boosts into attention bias.",
    )
    parser.add_argument(
        "--llm-bias-aggregation",
        choices=["max", "sum"],
        default="max",
        help="Aggregate multiple interactions into node-level hotspot scores.",
    )
    parser.add_argument(
        "--llm-edge-mode",
        choices=["both", "scalar", "vector"],
        default="both",
        help="How to apply LLM-derived edge weights in PaiNN (scalar, vector, or both).",
    )
    parser.add_argument(
        "--llm-rule-weight",
        type=float,
        default=0.0,
        help="Soft-constraint weight for LLM strength ranking regularization.",
    )
    parser.add_argument(
        "--llm-rule-min-confidence",
        type=float,
        default=0.0,
        help="Minimum LLM confidence required to include samples in rule regularization.",
    )
    parser.add_argument(
        "--focus-exact-covered-weight",
        type=float,
        default=0.0,
        help="Additional multiplicative MSE weight applied only to exact-covered literature rows.",
    )
    parser.add_argument(
        "--focus-confidence-threshold",
        type=float,
        default=0.0,
        help="Confidence threshold for up-weighting MSE on high-confidence LLM samples.",
    )
    parser.add_argument(
        "--focus-confidence-weight",
        type=float,
        default=0.0,
        help="Additional MSE weight applied to samples above --focus-confidence-threshold.",
    )
    parser.add_argument(
        "--llm-row-dropout-p",
        type=float,
        default=0.0,
        help="Training-only probability of dropping the entire LLM semantic row before the forward pass.",
    )
    parser.add_argument(
        "--llm-row-dropout-confidence-scale",
        type=float,
        default=0.0,
        help="Confidence-based reduction factor for --llm-row-dropout-p so reliable rows are dropped less often.",
    )
    parser.add_argument(
        "--llm-row-dropout-exact-scale",
        type=float,
        default=1.0,
        help="Multiplier on semantic row dropout for exact-covered rows; values below 1.0 protect exact evidence.",
    )
    parser.add_argument(
        "--llm-row-dropout-propagated-scale",
        type=float,
        default=1.0,
        help="Multiplier on semantic row dropout for propagated-covered rows; values above 1.0 regularize transferred evidence harder.",
    )
    parser.add_argument(
        "--llm-reliability-loss-weight",
        type=float,
        default=0.0,
        help="Additional multiplicative MSE emphasis scaled by a smooth reliability gate over LLM confidence.",
    )
    parser.add_argument(
        "--llm-reliability-loss-center",
        type=float,
        default=0.5,
        help="Confidence center for the smooth reliability gate used by --llm-reliability-loss-weight.",
    )
    parser.add_argument(
        "--llm-reliability-loss-sharpness",
        type=float,
        default=8.0,
        help="Sharpness of the smooth reliability gate used by --llm-reliability-loss-weight.",
    )
    parser.add_argument(
        "--focus-yield-threshold",
        type=float,
        default=None,
        help="Raw target threshold for up-weighting high-yield samples in the MSE term.",
    )
    parser.add_argument(
        "--focus-yield-weight",
        type=float,
        default=0.0,
        help="Additional multiplicative MSE weight applied to samples at or above --focus-yield-threshold.",
    )
    parser.add_argument(
        "--focus-coordination-quantile",
        type=float,
        default=0.75,
        help="Training-set quantile used to define coordination-heavy r1 samples when --focus-coordination-weight > 0.",
    )
    parser.add_argument(
        "--focus-coordination-weight",
        type=float,
        default=0.0,
        help="Additional multiplicative MSE weight applied to coordination-heavy r1 samples derived from node features.",
    )
    parser.add_argument(
        "--focus-electronic-frontier-weight",
        type=float,
        default=0.0,
        help="Additional multiplicative MSE weight scaled smoothly by the Ru/Ir electronic-frontier score.",
    )
    parser.add_argument(
        "--focus-kinetic-relay-weight",
        type=float,
        default=0.0,
        help="Additional multiplicative MSE weight applied to Ru/Ir kinetic-relay focus samples.",
    )
    parser.add_argument(
        "--llm-gate-min-confidence",
        type=float,
        default=0.55,
        help="Center confidence for smooth adaptive gating in painn_precision.",
    )
    parser.add_argument(
        "--llm-gate-confidence-sharpness",
        type=float,
        default=8.0,
        help="Steepness of confidence-based adaptive gating in painn_precision.",
    )
    parser.add_argument(
        "--llm-gate-density-center",
        type=float,
        default=0.8,
        help="Center value for LLM density prior (feature index 0) in adaptive gating.",
    )
    parser.add_argument(
        "--llm-gate-density-sharpness",
        type=float,
        default=3.0,
        help="Steepness of LLM density prior in adaptive gating.",
    )
    parser.add_argument(
        "--llm-gate-adaptive-strength",
        type=float,
        default=1.2,
        help="Logit strength for combining confidence+density priors in adaptive gating.",
    )
    parser.add_argument(
        "--llm-fusion-mode",
        choices=["residual", "conditioned", "cross_attention", "moe"],
        default="residual",
        help="Fusion mode for painn_precision (residual gate, FiLM-conditioned, cross-attention conditioned, or multi-scale MoE).",
    )
    parser.add_argument(
        "--llm-cross-attention-heads",
        type=int,
        default=4,
        help="Cross-attention head count when --llm-fusion-mode=cross_attention.",
    )
    parser.add_argument(
        "--llm-conditioned-scale",
        type=float,
        default=1.0,
        help="Initial strength for the conditioned prediction branch in aggressive painn_precision fusion.",
    )
    parser.add_argument(
        "--llm-conditioned-scale-curriculum-start",
        type=float,
        default=None,
        help=(
            "Optional epoch-stage curriculum start for conditioned semantic authority. "
            "When set, the conditioned branch ramps linearly from this scale to --llm-conditioned-scale."
        ),
    )
    parser.add_argument(
        "--llm-conditioned-scale-curriculum-warmup-epochs",
        type=int,
        default=1,
        help="Number of epochs used to ramp conditioned semantic authority up to --llm-conditioned-scale.",
    )
    parser.add_argument(
        "--llm-conditioned-scale-curriculum-shape",
        choices=("linear", "quadratic"),
        default="linear",
        help=(
            "Shape used inside each conditioned semantic warmup segment. "
            "`linear` reproduces the existing schedule; `quadratic` keeps early growth more conservative."
        ),
    )
    parser.add_argument(
        "--llm-conditioned-scale-curriculum-stage1-target",
        type=float,
        default=None,
        help=(
            "Optional intermediate conditioned semantic scale reached at the end of stage 1 before a second in-run "
            "semantic release continues toward --llm-conditioned-scale."
        ),
    )
    parser.add_argument(
        "--llm-conditioned-scale-curriculum-stage2-warmup-epochs",
        type=int,
        default=None,
        help=(
            "Optional second-stage warmup duration used to ramp from the stage-1 conditioned semantic scale to "
            "--llm-conditioned-scale within the same fresh run."
        ),
    )
    parser.add_argument(
        "--llm-physical-bridge",
        action="store_true",
        help=(
            "Inject semantic context into the pooled physical path before QC fusion and kinetic interaction, "
            "instead of only after pooling in the conditioned branch."
        ),
    )
    parser.add_argument(
        "--llm-pair-conditioned-bridge",
        action="store_true",
        help=(
            "Inject native `cat_r1_pair` semantics directly into the pooled physical path before QC fusion "
            "and kinetic interaction. This route only applies to `native_pair`, `native_pair_v2`, "
            "`pair_first_local`, and `semantic_expert_moe` encoders."
        ),
    )
    parser.add_argument(
        "--llm-pair-conditioned-bridge-locus",
        choices=["pre_qc_and_kinetic_fusion", "post_qc_pre_kinetic"],
        default="pre_qc_and_kinetic_fusion",
        help=(
            "Injection locus for `--llm-pair-conditioned-bridge`: either before QC fusion and kinetics, "
            "or after QC fusion but before kinetic interaction."
        ),
    )
    parser.add_argument(
        "--llm-disable-global-background",
        action="store_true",
        help="Disable the global semantic background node so LLM routing stays local-only.",
    )
    parser.add_argument(
        "--llm-local-meta-gate",
        nargs="+",
        default=[],
        help=(
            "Optional audited semantic meta-features used to build the adaptive LLM gate prior "
            "(e.g. overall_confidence temperature_alignment hotspot_density cat_focus r1_focus r2_focus)."
        ),
    )
    parser.add_argument(
        "--llm-gate-product-focus-scale",
        type=float,
        default=1.0,
        help=(
            "Multiplier applied to product-side r2_focus when the adaptive local semantic gate is enabled. "
            "Use 1.0 to preserve the legacy gate, values below 1.0 to weaken sparse product authority, and 0.0 to remove it."
        ),
    )
    parser.add_argument(
        "--llm-feature-profile",
        choices=[
            "legacy",
            "component_only",
            "core_emphasis",
            "high_information",
            "chemistry_grounded_compact",
            "propagated_compact_v2",
            "pair_aware",
        ],
        default="legacy",
        help=(
            "Optional LLM encoder profile for borylation semantic features. "
            "`legacy` preserves the current missingness-aware story-58 surface, "
            "`component_only` preserves the original 58-E component-aware surface without presence masks, "
            "`core_emphasis` appends catalyst/reactant-weighted text buckets and feature-group priors, "
            "`high_information` freezes the audited story-60-1 sparse/useful subset of the core-emphasis surface, "
            "`chemistry_grounded_compact` keeps the story-69 authority core plus minimal chemistry/product sentinels, "
            "`propagated_compact_v2` prunes the compact surface toward catalyst/r1 donor-transfer signals for propagated rows, "
            "and `pair_aware` appends catalyst-reactant compatibility features on top of the component-only surface."
        ),
    )
    parser.add_argument(
        "--llm-missingness-contract",
        choices=["profile_default", "missingness_token"],
        default="profile_default",
        help=(
            "How uncovered literature rows should be represented on the LLM branch. "
            "`profile_default` preserves the feature-profile-native behavior, while "
            "`missingness_token` appends one explicit uncovered-row token that is 1.0 only when literature is absent."
        ),
    )
    parser.add_argument(
        "--llm-semantic-group-scale",
        nargs="+",
        default=[],
        help=(
            "Optional per-group multipliers applied to the built LLM feature vector before model ingestion. "
            "Pass one or more `group=scale` entries using story-69 semantic groups such as "
            "`global_text=0.55 reweighted_text=0.20 r2_component=0.35`."
        ),
    )
    parser.add_argument(
        "--llm-semantic-feature-scale",
        nargs="+",
        default=[],
        help=(
            "Optional per-feature multipliers applied to the built LLM feature vector after any group-level scaling "
            "and before row-level authority scaling. Pass one or more `feature=scale` entries such as "
            "`llm_text_boron_boryl=0.90 llm_semantic_r2_focus=0.90`."
        ),
    )
    parser.add_argument(
        "--llm-exact-row-scale",
        type=float,
        default=1.0,
        help=(
            "Optional multiplicative scale applied to the built LLM feature vector only for exact-covered rows "
            "after any semantic-group rescaling."
        ),
    )
    parser.add_argument(
        "--llm-exact-row-scale-curriculum-start",
        type=float,
        default=None,
        help=(
            "Optional epoch-stage curriculum start for exact-row semantic authority. "
            "Exact rows ramp from this scale to --llm-exact-row-scale."
        ),
    )
    parser.add_argument(
        "--llm-exact-row-scale-curriculum-warmup-epochs",
        type=int,
        default=1,
        help="Number of epochs used to ramp exact-row semantic authority to --llm-exact-row-scale.",
    )
    parser.add_argument(
        "--llm-exact-row-scale-curriculum-shape",
        choices=("linear", "quadratic"),
        default="linear",
        help=(
            "Shape used inside the exact-row semantic authority warmup. "
            "`quadratic` keeps early exact authority lower before releasing later."
        ),
    )
    parser.add_argument(
        "--llm-propagated-row-scale",
        type=float,
        default=1.0,
        help=(
            "Optional multiplicative scale applied to the built LLM feature vector only for propagated-covered rows "
            "after any semantic-group rescaling."
        ),
    )
    parser.add_argument(
        "--llm-propagated-row-scale-curriculum-start",
        type=float,
        default=None,
        help=(
            "Optional epoch-stage curriculum start for propagated-row semantic authority. "
            "Exact rows stay at --llm-exact-row-scale while propagated rows ramp from this scale "
            "to --llm-propagated-row-scale."
        ),
    )
    parser.add_argument(
        "--llm-propagated-row-scale-curriculum-warmup-epochs",
        type=int,
        default=1,
        help="Number of epochs used to ramp propagated-row semantic authority to --llm-propagated-row-scale.",
    )
    parser.add_argument(
        "--llm-propagated-row-scale-curriculum-shape",
        choices=("linear", "quadratic"),
        default="linear",
        help=(
            "Shape used inside the propagated-row semantic authority warmup. "
            "`quadratic` keeps early propagated authority lower before releasing later."
        ),
    )
    parser.add_argument(
        "--llm-propagated-row-confidence-floor",
        type=float,
        default=1.0,
        help=(
            "Optional confidence-aware floor applied only to propagated rows. "
            "Use 1.0 to disable confidence shaping, or a lower value to keep weak propagated rows on a reduced authority tier."
        ),
    )
    parser.add_argument(
        "--llm-propagated-row-confidence-power",
        type=float,
        default=1.0,
        help=(
            "Exponent used by the propagated-row confidence schedule. "
            "Values above 1.0 emphasize the gap between weaker and stronger propagated rows."
        ),
    )
    parser.add_argument(
        "--llm-propagated-row-temperature-floor",
        type=float,
        default=1.0,
        help=(
            "Optional temperature-aware floor applied only to propagated rows. "
            "Use 1.0 to disable temperature shaping, or a lower value to keep thermally mismatched transfers on a reduced authority tier."
        ),
    )
    parser.add_argument(
        "--llm-propagated-row-temperature-power",
        type=float,
        default=1.0,
        help=(
            "Exponent used by the propagated-row temperature schedule. "
            "Values above 1.0 emphasize the gap between weaker and stronger temperature alignment."
        ),
    )
    parser.add_argument(
        "--native-pair-field-profile",
        choices=list(native_pair_field_profile_choices()),
        default="full",
        help=(
            "Optional native-pair summary/token field profile. "
            "`full` preserves all native-pair channels, while other profiles can remove or downweight "
            "specific native-pair field groups for ablation and denoising studies."
        ),
    )
    parser.add_argument(
        "--llm-semantic-encoder",
        choices=["flat", "branch_structured", "grouped_selector", "product_optional_local", "provenance_dual_branch", "pair_compatibility", "native_pair", "native_pair_v2", "pair_first_local", "semantic_expert_moe"],
        default="flat",
        help=(
            "Semantic object encoder for painn_precision. "
            "`flat` preserves the original single-vector projection, and "
            "`branch_structured` splits the LLM surface into global/cat/r1/r2/pair branches before encoding. "
            "`grouped_selector` learns feature-family tokens and keeps only the top-k semantic groups before pooling. "
            "`product_optional_local` keeps catalyst/reactant branches primary and caps product-side authority unless product evidence is strong. "
            "`provenance_dual_branch` keeps a shared semantic token but routes exact-covered and propagated-covered rows through "
            "separate provenance-specific branches before conditioned pooling. "
            "`pair_compatibility` reuses the pair-aware branches but adds an explicit catalyst-reactant compatibility head with "
            "core-vs-product pair contrast. "
            "`native_pair` reads the story-61 `cat_r1_pair` object directly via pair-summary and joint-interaction tokens. "
            "`native_pair_v2` upgrades that direct path by making pair summary, attended token context, and token-max context "
            "separate primary branches before pooling. "
            "`pair_first_local` makes the native `cat_r1_pair` branch primary while keeping separate component summaries auxiliary. "
            "`semantic_expert_moe` routes across multiple per-row native-pair schema experts packed into one semantic asset."
        ),
    )
    parser.add_argument(
        "--llm-semantic-group-top-k",
        type=int,
        default=0,
        help=(
            "Top-k group count for `--llm-semantic-encoder grouped_selector`. "
            "Use 0 to keep every semantic group, or a positive value to sparsify selection."
        ),
    )
    parser.add_argument(
        "--high-confidence-threshold",
        type=float,
        default=0.7,
        help="Minimum LLM confidence for high-confidence subset metrics.",
    )
    parser.add_argument(
        "--loss",
        default="mse",
        choices=[
            "mse",
            "ranknet",
            "listnet",
            "pairwise_hinge",
            "group_ranknet",
            "group_listnet",
            "group_pairwise_hinge",
            "hybrid_ranknet",
            "hybrid_listnet",
            "hybrid_pairwise_hinge",
            "hybrid_group_ranknet",
            "hybrid_group_listnet",
            "hybrid_group_pairwise_hinge",
        ],
        help="Training loss mode (ranking losses operate over each batch or group).",
    )
    parser.add_argument("--rank-weight", type=float, default=0.2)
    parser.add_argument("--rank-margin", type=float, default=0.0)
    parser.add_argument(
        "--rank-group-cols",
        nargs="+",
        default=["r1", "r2", "wavelength"],
        help="Optional group columns for group-wise ranking losses (ml-cb only).",
    )
    parser.add_argument(
        "--rank-target-col",
        default=None,
        help="Optional numeric column to drive ranking losses (e.g., Vertical_IP_metal2).",
    )
    parser.add_argument(
        "--qc-fusion",
        dest="qc_fusion",
        default="none",
        choices=[
            "none",
            "pooled",
            "interaction",
            "pooled_interaction",
            "raw",
            "raw_interaction",
            "energetics",
            "core22",
        ],
        help="Augment QC descriptors with pooled stats and/or interaction terms.",
    )
    parser.add_argument(
        "--physics-fusion",
        choices=["concat", "add", "gated"],
        default="gated",
        help="Fusion strategy for painn_physics when blending QC descriptors with geometric embeddings.",
    )
    parser.add_argument("--qc-scale", type=float, default=1.0, help="Scale QC features after normalization.")
    parser.add_argument(
        "--qc-node-scale",
        type=float,
        default=1.0,
        help="Scale QC features used for node init/gating (0 disables node-side QC).",
    )
    parser.add_argument(
        "--qc-global-scale",
        type=float,
        default=1.0,
        help="Scale QC features in the global feature vector (0 disables global QC).",
    )
    parser.add_argument(
        "--qc-node-mode",
        choices=["none", "metal", "all"],
        default="none",
        help="Map QC descriptors to nodes (none/metal/all) for point-level injection.",
    )
    parser.add_argument(
        "--coordination-features",
        choices=["none", "potency"],
        default="none",
        help="Inject heuristic N/O/S coordination-potency node features into PaiNN-compatible models.",
    )
    parser.add_argument(
        "--coordination-node-scale",
        type=float,
        default=1.0,
        help="Scale applied to heuristic coordination node features.",
    )
    parser.add_argument(
        "--reaction-center-coupling",
        action="store_true",
        help="Enable pre-pooling reaction-center coupling pairs for painn_precision runs.",
    )
    parser.add_argument(
        "--kinetic-relay",
        action="store_true",
        help="Enable the kinetic-aware relay gate inside reaction-center coupling.",
    )
    parser.add_argument(
        "--reaction-center-aux-weight",
        type=float,
        default=0.0,
        help="Auxiliary MSE weight for reaction-center electronic-alignment and steric-blockade supervision.",
    )
    parser.add_argument(
        "--exact-covered-consistency-weight",
        type=float,
        default=0.0,
        help="Auxiliary MSE weight that keeps final predictions close to the physical base prediction on exact-covered rows.",
    )
    parser.add_argument(
        "--qc-weight-path",
        default=None,
        help="Optional QC sensitivity JSON for feature weighting.",
    )
    parser.add_argument(
        "--qc-weight-mode",
        choices=["none", "positive", "signed"],
        default="positive",
        help="Mode for QC feature weighting when --qc-weight-path is set.",
    )
    parser.add_argument("--qc-weight-scale", type=float, default=0.5)
    parser.add_argument("--qc-weight-min", type=float, default=0.25)
    parser.add_argument("--qc-weight-max", type=float, default=2.0)
    parser.add_argument(
        "--grad-audit",
        action="store_true",
        help="Log gradient norms for structural vs feature parameters and QC input gradients.",
    )
    parser.add_argument(
        "--attn-audit",
        action="store_true",
        help="Log hotspot attention bias ratios from literature guidance.",
    )
    parser.add_argument(
        "--llm-phys-audit",
        action="store_true",
        help="Audit physical consistency of LLM interaction cues against 3D geometry.",
    )
    parser.add_argument(
        "--llm-phys-cutoff",
        type=float,
        default=5.0,
        help="Distance cutoff (Angstroms) for LLM interaction physical consistency.",
    )
    parser.add_argument("--use-numeric", dest="use_numeric", action="store_true", default=True)
    parser.add_argument("--no-numeric", dest="use_numeric", action="store_false")
    parser.add_argument("--use-coulomb", dest="use_coulomb", action="store_true", default=True)
    parser.add_argument("--no-coulomb", dest="use_coulomb", action="store_false")
    parser.add_argument("--use-geometry", dest="use_geometry", action="store_true", default=True)
    parser.add_argument("--no-geometry", dest="use_geometry", action="store_false")
    parser.add_argument("--use-rdkit", dest="use_rdkit", action="store_true", default=False)
    parser.add_argument("--no-rdkit", dest="use_rdkit", action="store_false")
    parser.add_argument(
        "--use-morgan-fingerprint",
        dest="use_morgan_fingerprint",
        action="store_true",
        default=False,
    )
    parser.add_argument("--no-morgan-fingerprint", dest="use_morgan_fingerprint", action="store_false")
    parser.add_argument("--morgan-fingerprint-bits", type=int, default=1024)
    parser.add_argument("--morgan-fingerprint-radius", type=int, default=2)
    parser.add_argument("--use-ref-data", dest="use_ref_data", action="store_true", default=False)
    parser.add_argument("--no-ref-data", dest="use_ref_data", action="store_false")
    parser.add_argument(
        "--literature-cache",
        dest="literature_cache",
        default=os.getenv("LLM_LITERATURE_CACHE"),
        help="Path to JSON cache with LLM literature interactions.",
    )
    parser.add_argument(
        "--disable-literature-branch",
        action="store_true",
        help="Force the literature/LLM branch off even for models such as painn_precision that normally enable it.",
    )
    parser.add_argument(
        "--baseline-summary",
        default=None,
        help="Optional baseline summary JSON to log R2 deltas for LLM fusion runs.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument(
        "--split-strategy",
        choices=["auto", "reference", "structural_signature"],
        default="auto",
        help="How to split ml-borylation data into train/val/test partitions.",
    )
    parser.add_argument(
        "--signature-columns",
        nargs="+",
        default=None,
        help="Optional structural-signature columns when --split-strategy=structural_signature.",
    )
    parser.add_argument("--val-split", choices=["val", "train"], default="val")
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--log_dir", default="progress")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--summary_path", default=None)
    parser.add_argument(
        "--model-save-path",
        default=None,
        help="Optional path to save the best model weights (torch checkpoint).",
    )
    parser.add_argument(
        "--init-from",
        default=None,
        help="Optional checkpoint to initialize model weights before training.",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze encoder backbones (cat/reactant) during training.",
    )
    parser.add_argument(
        "--freeze-physical-base",
        action="store_true",
        help="Freeze painn_precision physical predictor blocks to train only LLM residual modules.",
    )
    parser.add_argument(
        "--safe-hybrid-init",
        action="store_true",
        help="Zero newly introduced hybrid branches after warm-starting so transfer begins from the base prediction before semantic heads learn.",
    )
    parser.add_argument(
        "--trainable-module"
        "-prefix",
        nargs="+",
        default=[],
        help="Optional allowlist of module-name prefixes to keep trainable after initialization/freezing.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    log_dir = Path(args.log_dir)
    if args.sweep:
        summaries = []
        for lr in (1e-3, 1e-4):
            for hidden_dim in (128, 256):
                tag = f"lr{lr:g}_h{hidden_dim}"
                if args.samples is not None:
                    tag += f"_s{args.samples}"
                log_path = log_dir / f"phase1_e3nn_{tag}.log"
                summary = train_once(args, lr, hidden_dim, log_path)
                summaries.append(summary)
        summary_path = log_dir / "phase1_e3nn_sweep_summary.json"
        summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    else:
        if args.dataset == "ml-cb" and args.model == "baseline":
            prefix = "phase1_schnet"
        else:
            prefix = f"phase3_{args.dataset}_{args.model}"
        tag = f"lr{args.lr:g}_h{args.hidden_dim}"
        if args.samples is not None:
            tag += f"_s{args.samples}"
        log_path = log_dir / f"{prefix}_{tag}.log"
        summary = train_once(args, args.lr, args.hidden_dim, log_path)
        if args.summary_path:
            Path(args.summary_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
