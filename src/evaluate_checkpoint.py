"""Evaluate a released checkpoint on the labelled structural test split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from .data import MLBorylationDataset
    from .predict_extrapolation import _dataset_kwargs, _model
    from .train import collate_fn, compute_metrics, run_epoch
except ImportError:
    from data import MLBorylationDataset
    from predict_extrapolation import _dataset_kwargs, _model
    from train import collate_fn, compute_metrics, run_epoch


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a STEREO checkpoint on its held-out test split.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument(
        "--literature-cache",
        type=Path,
        default=Path("assets/literature/stereo_lit_cache.json"),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    summary = json.loads(args.config.read_text(encoding="utf-8"))
    include_literature = bool(summary.get("literature_cache")) and not bool(
        summary.get("disable_literature_branch", False)
    )
    dataset = MLBorylationDataset(
        str(args.data),
        split="test",
        include_literature=include_literature,
        literature_cache_path=str(args.literature_cache) if include_literature else None,
        **_dataset_kwargs(summary),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, include_literature=include_literature),
    )

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    checkpoint_features = checkpoint.get("feature_cols", []) if isinstance(checkpoint, dict) else []
    if checkpoint_features and len(checkpoint_features) != len(dataset.feature_cols):
        raise ValueError(
            "Checkpoint feature dimension does not match the released data: "
            f"{len(checkpoint_features)} != {len(dataset.feature_cols)}"
        )

    model = _model(summary, dataset)
    if hasattr(model, "ablation_disable_global_background"):
        model.ablation_disable_global_background = bool(summary.get("llm_disable_global_background", False))
    model = model.to(args.device)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict, strict=True)

    input_keys = (
        "cat_z", "cat_pos", "cat_batch", "cat_edge_index",
        "r1_z", "r1_pos", "r1_batch", "r1_edge_index",
        "r2_z", "r2_pos", "r2_batch", "r2_edge_index",
        "features", "temperature", "qc_features",
    )
    if include_literature:
        input_keys += (
            "llm_features", "llm_confidence", "native_pair_summary",
            "native_pair_tokens", "native_pair_token_mask",
            "native_pair_expert_summaries", "native_pair_expert_tokens",
            "native_pair_expert_token_masks", "native_pair_expert_priors",
        )

    _, y_true_norm, y_pred_norm = run_epoch(
        model,
        loader,
        args.device,
        input_keys=input_keys,
        loss_mode="mse",
        llm_exact_row_scale=float(summary.get("llm_exact_row_scale") or 1.0),
        llm_propagated_row_scale=float(summary.get("llm_propagated_row_scale") or 1.0),
    )
    y_true = np.asarray(y_true_norm) * float(dataset.y_std) + float(dataset.y_mean)
    y_pred = np.asarray(y_pred_norm) * float(dataset.y_std) + float(dataset.y_mean)
    metrics = compute_metrics(y_true, y_pred)
    expected_r2 = (summary.get("test") or {}).get("r2")
    result = {
        "rows": len(dataset),
        "split_strategy": dataset.split_strategy,
        "mae": metrics["mae"],
        "r2": metrics["r2"],
        "pearson": metrics["pearson"],
        "reported_checkpoint_r2": expected_r2,
        "r2_delta": None if expected_r2 is None else metrics["r2"] - float(expected_r2),
        "device": str(args.device),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
