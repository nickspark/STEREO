"""Run a saved model on a released extrapolation table.

The training table is needed only to recover feature normalization and element
vocabulary.  No source path is assumed and no input file is modified.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
try:
    from .data import MLBorylationDataset
    from .train import collate_fn, run_epoch
    from .model import PaiNNPrecisionHead
    from .llm.features import (llm_semantic_branch_columns_from_feature_columns,
        llm_semantic_branch_index_map, llm_semantic_group_columns_from_feature_columns,
        llm_semantic_group_index_map)
except ImportError:
    from data import MLBorylationDataset
    from train import collate_fn, run_epoch
    from model import PaiNNPrecisionHead
    from llm.features import (llm_semantic_branch_columns_from_feature_columns,
        llm_semantic_branch_index_map, llm_semantic_group_columns_from_feature_columns,
        llm_semantic_group_index_map)

def _model(summary, ds):
    cols=list(getattr(ds,"llm_feature_cols",[]) or [])
    return PaiNNPrecisionHead(hidden_dim=int(summary.get("hidden_dim") or 160), num_layers=int(summary.get("layers") or 8),
      dropout=float(summary.get("dropout") or 0.0), feat_dim=len(ds.feature_cols), qc_dim=len(getattr(ds,"qc_feature_cols",[]) or []),
      physics_fusion=str(summary.get("physics_fusion") or "concat"), llm_dim=len(cols),
      native_pair_dim=int(getattr(ds,"native_pair_summary_dim",0)), native_pair_token_dim=int(getattr(ds,"native_pair_token_dim",0)),
      gate_dropout=float(summary.get("dropout") or 0.0), min_llm_confidence=float(summary.get("llm_gate_min_confidence") or 0.55),
      gate_confidence_sharpness=float(summary.get("llm_gate_confidence_sharpness") or 8.0), gate_density_center=float(summary.get("llm_gate_density_center") or 0.8),
      gate_density_sharpness=float(summary.get("llm_gate_density_sharpness") or 3.0), gate_adaptive_strength=float(summary.get("llm_gate_adaptive_strength") or 1.2),
      llm_fusion_mode=str(summary.get("llm_fusion_mode") or "conditioned"), llm_cross_attention_heads=int(summary.get("llm_cross_attention_heads") or 4),
      llm_conditioned_scale=float(summary.get("llm_conditioned_scale") or 1.0), llm_semantic_encoder=str(summary.get("llm_semantic_encoder") or "flat"),
      llm_branch_index_map=llm_semantic_branch_index_map(cols), llm_branch_column_map=llm_semantic_branch_columns_from_feature_columns(cols),
      llm_group_index_map=llm_semantic_group_index_map(cols), llm_group_column_map=llm_semantic_group_columns_from_feature_columns(cols),
      llm_group_top_k=int(summary.get("llm_semantic_group_top_k") or 0), llm_local_meta_gate=list(summary.get("llm_local_meta_gate") or []),
      llm_gate_product_focus_scale=float(summary.get("llm_gate_product_focus_scale") or 0.0),
      semantic_physical_bridge=bool(summary.get("llm_physical_bridge", False)),
      pair_conditioned_bridge=bool(summary.get("llm_pair_conditioned_bridge", False)),
      pair_conditioned_bridge_locus=str(summary.get("llm_pair_conditioned_bridge_locus") or "pre_qc_and_kinetic_fusion"),
      global_attention=bool(summary.get("global_attention", False)),
      global_attention_heads=int(summary.get("global_attention_heads") or 4),
      global_attention_layers=int(summary.get("global_attention_layers") or 1),
      global_attention_dropout=float(summary.get("global_attention_dropout") or 0.1),
      interaction_cross_attention=bool(summary.get("interaction_cross_attention_enabled", False)),
      node_qc_dim=0, node_qc_gate_dim=0,
      reaction_center_coupling=bool(summary.get("reaction_center_coupling_enabled", False)),
      reaction_center_pair_feature_dim=int(getattr(ds,"reaction_center_feature_dim",0)),
      kinetic_relay=bool(summary.get("kinetic_relay_enabled", False)))

def _dataset_kwargs(summary):
    split = str(summary.get("split_strategy") or "structural_signature")
    if split == "group_structural_signature": split = "structural_signature"
    return {"seed": int(summary.get("split_seed") or summary.get("seed") or 42),
      "include_numeric": bool(summary.get("use_numeric", True)), "use_coulomb": bool(summary.get("use_coulomb", True)),
      "use_geometry": bool(summary.get("use_geometry", True)), "use_rdkit": bool(summary.get("use_rdkit", False)),
      "use_ref_data": bool(summary.get("use_ref_data", False)), "qc_fusion": str(summary.get("qc_fusion") or "none"),
      "qc_scale": float(summary.get("qc_scale") or 1.0), "qc_node_scale": float(summary.get("qc_node_scale") or 1.0),
      "qc_global_scale": float(summary.get("qc_global_scale") or 1.0), "qc_node_mode": str(summary.get("qc_node_mode") or "none"),
      "qc_weight_path": summary.get("qc_weight_path"), "qc_weight_mode": str(summary.get("qc_weight_mode") or "positive"),
      "qc_weight_scale": float(summary.get("qc_weight_scale") or 0.5), "qc_weight_min": float(summary.get("qc_weight_min") or 0.25),
      "qc_weight_max": float(summary.get("qc_weight_max") or 2.0), "llm_feature_profile": str(summary.get("llm_feature_profile") or "legacy"),
      "native_pair_field_profile": str(summary.get("native_pair_field_profile") or "full"),
      "coordination_features": str(summary.get("coordination_features") or "none"), "coordination_node_scale": float(summary.get("coordination_node_scale") or 1.0),
      "build_combined_graph": False, "build_atomic_interaction_pairs": bool(summary.get("interaction_cross_attention_enabled", False)),
      "interaction_pair_mode": str(summary.get("interaction_pair_mode") or "classic"),
      "build_reaction_center_coupling": bool(summary.get("reaction_center_coupling_enabled", False)),
      "combined_k": int(summary.get("combined_k") or 8), "combined_cutoff": float(summary.get("combined_cutoff") or 5.0),
      "val_fraction": float(summary.get("val_fraction") or 0.1), "test_fraction": float(summary.get("test_fraction") or 0.1), "split_strategy": split}

def main():
    p=argparse.ArgumentParser(description="Predict on a BVS extrapolation CSV.")
    p.add_argument("--summary", required=True, type=Path); p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--training-data", required=True, type=Path); p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path); p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--literature-cache", type=Path, default=Path("assets/literature/stereo_lit_cache.json"))
    p.add_argument("--extrapolation-cache", type=Path, default=Path("assets/literature/empty_extrapolation_cache.json"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); a=p.parse_args()
    summary=json.loads(a.summary.read_text())
    include_literature=bool(summary.get("literature_cache"))
    cache_path=str(a.literature_cache) if include_literature else None
    extrapolation_cache_path=str(a.extrapolation_cache) if include_literature else None
    options=_dataset_kwargs(summary)
    train=MLBorylationDataset(str(a.training_data), split="test", include_literature=include_literature, literature_cache_path=cache_path, **options)
    stats={"feat_mean":train.feat_mean,"feat_std":train.feat_std,"y_mean":float(train.y_mean),"y_std":float(train.y_std)}
    # The checkpoint input width is set by the training table's largest structures.
    # Reusing those widths pads smaller extrapolation molecules and truncates larger ones.
    train_atom_widths=(train.max_cat_atoms, train.max_r1_atoms, train.max_r2_atoms)
    infer=MLBorylationDataset(str(a.input), split="inference", include_literature=include_literature, literature_cache_path=extrapolation_cache_path, allow_missing_target=True, normalization_stats=stats, max_atoms_override=train_atom_widths, **options)
    loader=DataLoader(infer,batch_size=a.batch_size,shuffle=False,collate_fn=lambda b: collate_fn(b,include_literature=include_literature))
    model=_model(summary,infer)
    if hasattr(model, "ablation_disable_global_background"):
        model.ablation_disable_global_background=bool(summary.get("llm_disable_global_background",False))
    model=model.to(a.device); ck=torch.load(a.checkpoint,map_location=a.device); model.load_state_dict(ck.get("state_dict",ck),strict=True)
    keys=("cat_z","cat_pos","cat_batch","cat_edge_index","r1_z","r1_pos","r1_batch","r1_edge_index","r2_z","r2_pos","r2_batch","r2_edge_index","features","temperature","qc_features")
    if include_literature: keys += ("llm_features","llm_confidence","native_pair_summary","native_pair_tokens","native_pair_token_mask","native_pair_expert_summaries","native_pair_expert_tokens","native_pair_expert_token_masks","native_pair_expert_priors")
    _,_,pred=run_epoch(model,loader,a.device,input_keys=keys,loss_mode="mse")
    frame=pd.read_csv(a.input); result=frame.copy(); result["predicted_ee"]=torch.as_tensor(pred).numpy() * float(infer.y_std) + float(infer.y_mean); a.output.parent.mkdir(parents=True,exist_ok=True); result.to_csv(a.output,index=False)
    print(json.dumps({"rows":len(result),"output":str(a.output),"device":str(a.device)},indent=2))
if __name__ == "__main__": main()
