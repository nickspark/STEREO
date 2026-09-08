from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from src.train import train_once


class TrainConfig(SimpleNamespace):
    def __getattr__(self, name: str):
        return None


DEFAULT_LITERATURE_CACHE = Path("assets/literature/stereo_lit_cache.json")
DEFAULT_OUTPUT_DIR = Path("outputs")


def build_args(cli_args: argparse.Namespace) -> SimpleNamespace:
    use_literature = cli_args.variant == "stereo-lit"
    defaults = {
        "data": cli_args.data,
        "dataset": "ml-borylation",
        "model": "painn_precision",
        "alignment": "rule",
        "combined_k": 8,
        "combined_cutoff": 5.0,
        "batch_size": cli_args.batch_size,
        "epochs": cli_args.epochs,
        "lr": 0.00012,
        "lr_scheduler": "plateau",
        "lr_scheduler_patience": 5,
        "lr_scheduler_factor": 0.7,
        "lr_scheduler_min_lr": 1e-5,
        "seed": cli_args.seed,
        "split_seed": cli_args.split_seed,
        "hidden_dim": 160,
        "layers": 8,
        "heads": 4,
        "gate_hidden_dim": 64,
        "num_rbf": 32,
        "cutoff": 5.0,
        "dropout": 0.05,
        "weight_decay": 3e-5,
        "cross_bias_scale": 0.0,
        "llm_bias_scale": 0.0,
        "llm_bias_mode": "sum",
        "llm_bias_aggregation": "max",
        "llm_edge_mode": "both",
        "llm_rule_weight": 0.0,
        "llm_rule_min_confidence": 0.0,
        "high_confidence_threshold": 0.8,
        "llm_reliability_loss_weight": 0.0,
        "llm_reliability_loss_center": 0.5,
        "llm_reliability_loss_sharpness": 8.0,
        "focus_yield_threshold": None,
        "focus_yield_weight": 0.0,
        "focus_coordination_quantile": 0.75,
        "focus_coordination_weight": 0.0,
        "focus_electronic_frontier_weight": 0.0,
        "focus_kinetic_relay_weight": 0.0,
        "llm_gate_min_confidence": 0.78,
        "llm_gate_confidence_sharpness": 12.0,
        "llm_gate_density_center": 0.55,
        "llm_gate_density_sharpness": 6.0,
        "llm_gate_adaptive_strength": 1.2,
        "llm_gate_product_focus_scale": 0.0,
        "llm_feature_profile": "high_information",
        "llm_missingness_contract": "profile_default",
        "llm_semantic_group_scale": [],
        "llm_semantic_feature_scale": [],
        "llm_exact_row_scale": 1.0,
        "llm_exact_row_scale_curriculum_start": 0.95,
        "llm_exact_row_scale_curriculum_warmup_epochs": 6,
        "llm_exact_row_scale_curriculum_shape": "quadratic",
        "llm_propagated_row_scale": 1.0,
        "llm_propagated_row_scale_curriculum_start": 0.45,
        "llm_propagated_row_scale_curriculum_warmup_epochs": 10,
        "llm_propagated_row_scale_curriculum_shape": "quadratic",
        "llm_propagated_row_confidence_floor": 1.0,
        "llm_propagated_row_confidence_power": 1.0,
        "llm_propagated_row_temperature_floor": 1.0,
        "llm_propagated_row_temperature_power": 1.0,
        "llm_fusion_mode": "conditioned",
        "llm_cross_attention_heads": 4,
        "llm_conditioned_scale": 0.19,
        "llm_conditioned_scale_curriculum_start": 0.01,
        "llm_conditioned_scale_curriculum_warmup_epochs": 8,
        "llm_conditioned_scale_curriculum_shape": "linear",
        "llm_conditioned_scale_curriculum_stage1_target": None,
        "llm_conditioned_scale_curriculum_stage2_warmup_epochs": None,
        "llm_physical_bridge": False,
        "llm_pair_conditioned_bridge": False,
        "llm_pair_conditioned_bridge_locus": "pre_qc_and_kinetic_fusion",
        "llm_disable_global_background": True,
        "disable_literature_branch": not use_literature,
        "llm_local_meta_gate": [
            "overall_confidence",
            "temperature_alignment",
            "hotspot_density",
            "cat_focus",
            "r1_focus",
            "r2_focus",
        ],
        "attn_audit": False,
        "grad_audit": False,
        "llm_phys_audit": False,
        "llm_phys_cutoff": 5.0,
        "baseline_summary": None,
        "init_from": None,
        "freeze_backbone": False,
        "freeze_physical_base": False,
        "safe_hybrid_init": False,
        "trainable_module_prefix": [],
        "global_attention": False,
        "global_attention_heads": 4,
        "global_attention_layers": 1,
        "global_attention_dropout": 0.1,
        "interaction_cross_attention": False,
        "swa": False,
        "swa_start": None,
        "swa_lr": None,
        "swa_anneal_epochs": 10,
        "split_strategy": "structural_signature",
        "signature_columns": None,
        "val_split": "val",
        "samples": cli_args.samples,
        "log_dir": cli_args.log_dir,
        "sweep": False,
        "summary_path": cli_args.summary_path,
        "model_save_path": cli_args.model_save_path,
        "device": cli_args.device,
        "use_numeric": False,
        "use_coulomb": True,
        "use_geometry": True,
        "use_rdkit": False,
        "use_morgan_fingerprint": False,
        "morgan_fingerprint_bits": 1024,
        "morgan_fingerprint_radius": 2,
        "use_ref_data": False,
        "val_fraction": 0.1,
        "test_fraction": 0.1,
        "qc_fusion": "core22",
        "qc_scale": 1.0,
        "qc_node_scale": 1.0,
        "qc_global_scale": 1.0,
        "qc_node_mode": "none",
        "qc_weight_path": None,
        "qc_weight_mode": "positive",
        "qc_weight_scale": 0.5,
        "qc_weight_min": 0.25,
        "qc_weight_max": 2.0,
        "coord_jitter_std": 0.0,
        "coordination_features": "none",
        "coordination_node_scale": 1.0,
        "reaction_center_coupling": False,
        "kinetic_relay": False,
        "reaction_center_aux_weight": 0.0,
        "focus_exact_covered_weight": 0.0,
        "exact_covered_consistency_weight": 0.0,
        "focus_confidence_threshold": 0.0,
        "focus_confidence_weight": 0.0,
        "llm_row_dropout_p": 0.0,
        "llm_row_dropout_confidence_scale": 0.0,
        "llm_row_dropout_exact_scale": 1.0,
        "llm_row_dropout_propagated_scale": 1.0,
        "physics_fusion": "concat",
        "rank_weight": 0.2,
        "rank_margin": 0.0,
        "rank_group_cols": None,
        "rank_target_col": None,
        "loss": "mse",
        "literature_cache": str(DEFAULT_LITERATURE_CACHE) if use_literature else None,
        "native_pair_field_profile": "full",
        "llm_semantic_encoder": "flat",
        "llm_group_top_k": 0,
        "llm_semantic_group_top_k": 0,
        "build_combined_graph": False,
        "build_atomic_interaction_pairs": False,
        "interaction_pair_mode": "classic",
        "no_interaction_cross_attention": True,
    }
    if not use_literature:
        defaults["weight_decay"] = 3e-5
        defaults["llm_fusion_mode"] = "residual"
        defaults["llm_feature_profile"] = "legacy"
        defaults["split_strategy"] = "structural_signature"
        defaults["llm_conditioned_scale"] = 1.0
        defaults["llm_conditioned_scale_curriculum_start"] = None
        defaults["llm_conditioned_scale_curriculum_warmup_epochs"] = 1
        defaults["llm_exact_row_scale_curriculum_start"] = None
        defaults["llm_exact_row_scale_curriculum_warmup_epochs"] = 1
        defaults["llm_propagated_row_scale_curriculum_start"] = None
        defaults["llm_propagated_row_scale_curriculum_warmup_epochs"] = 1
        defaults["llm_disable_global_background"] = False
        defaults["llm_local_meta_gate"] = []
        defaults["llm_gate_min_confidence"] = 0.55
        defaults["llm_gate_density_center"] = 0.8
        defaults["llm_gate_density_sharpness"] = 3.0
        defaults["llm_gate_product_focus_scale"] = 1.0
    return TrainConfig(**defaults)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train STEREO or STEREO-Lit on the public borylation dataset.")
    parser.add_argument("--variant", choices=["stereo", "stereo-lit"], default="stereo-lit")
    parser.add_argument("--data", required=True, help="Path to the released labelled training CSV or XLSX.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--summary-path", default=None)
    parser.add_argument("--model-save-path", default=None)
    return parser.parse_args()


def main() -> None:
    cli_args = parse_args()
    variant_slug = cli_args.variant.replace("-", "_")
    output_dir = Path(cli_args.output_dir)
    if cli_args.log_dir is None:
        cli_args.log_dir = str(output_dir / f"{variant_slug}_run")
    if cli_args.summary_path is None:
        cli_args.summary_path = str(output_dir / f"{variant_slug}_summary.json")
    if cli_args.model_save_path is None:
        cli_args.model_save_path = str(output_dir / f"{variant_slug}_model.pt")

    args = build_args(cli_args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_name = f"{variant_slug}_ml-borylation_painn_precision_lr{args.lr:g}_h{args.hidden_dim}"
    if args.samples is not None:
        log_name += f"_s{args.samples}"
    log_path = log_dir / f"{log_name}.log"

    summary = train_once(args, args.lr, args.hidden_dim, log_path)

    if args.summary_path:
        summary_path = Path(args.summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
