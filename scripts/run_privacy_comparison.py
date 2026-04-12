#!/usr/bin/env python
"""Run privacy analysis across all model variants.

Compares Membership Inference Attack (MIA) results for:
1. BM - Base Model (no adapter)
2. UM - Universal Model (C1+C2)
3. BM + (C3) - Fine-tuned on C3
4. BM + (C3 w UM KD) - KD from UM on C3
5. BM + AVG(C1,C2,C3) - Federated average of all clients
"""

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from tqdm import tqdm
from src.model import load_base_model, load_adapter
from src.data import load_sciq, preprocess_dataset, create_dataloader
from src.evaluator import get_loss_distribution
from src.attacks import membership_inference_attack


def set_seed(seed: int = 42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return json.load(f)


def save_results(results: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {path}")


def get_mia_dataloaders(tokenizer, config, num_samples=1000):
    """Load SciQ data for MIA: train (members) vs validation (non-members)."""
    train_data = load_sciq("train", num_samples)
    val_data = load_sciq("validation", num_samples)

    max_length = config["training"].get("max_seq_length", 512)

    train_dataset = preprocess_dataset(train_data, tokenizer, "sciq", max_length)
    val_dataset = preprocess_dataset(val_data, tokenizer, "sciq", max_length)

    train_loader = create_dataloader(train_dataset, batch_size=8, shuffle=False)
    val_loader = create_dataloader(val_dataset, batch_size=8, shuffle=False)

    return train_loader, val_loader


def run_mia_on_model(model, tokenizer, config, model_name: str) -> dict:
    """Run MIA on a single model variant."""
    print(f"\n--- Running MIA on {model_name} ---")

    num_samples = config["privacy"].get("num_shadow_samples", 1000)
    train_loader, val_loader = get_mia_dataloaders(tokenizer, config, num_samples)

    model.eval()

    # Get loss distributions
    print("Computing member losses (training data)...")
    member_losses = get_loss_distribution(model, train_loader)

    print("Computing non-member losses (validation data)...")
    non_member_losses = get_loss_distribution(model, val_loader)

    # Run attack
    results = membership_inference_attack(member_losses, non_member_losses)
    results["model"] = model_name

    print(f"MIA Results for {model_name}:")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  AUC: {results['auc']:.4f}")
    print(f"  Baseline: {results['baseline_accuracy']:.4f}")

    return results


def mia_base_model(config: dict) -> dict:
    """MIA on Base Model (no adapter)."""
    model, tokenizer = load_base_model(
        model_name=config["model"]["name"],
        dtype=config["model"].get("dtype", "bfloat16"),
        gradient_checkpointing=False
    )

    results = run_mia_on_model(model, tokenizer, config, "BM")

    del model
    torch.cuda.empty_cache()
    return results


def mia_universal_model(config: dict, um_adapter_path: str) -> dict:
    """MIA on Universal Model (C1+C2)."""
    model, tokenizer = load_base_model(
        model_name=config["model"]["name"],
        dtype=config["model"].get("dtype", "bfloat16"),
        gradient_checkpointing=False
    )
    model = load_adapter(model, um_adapter_path)

    results = run_mia_on_model(model, tokenizer, config, "UM")

    del model
    torch.cuda.empty_cache()
    return results


def mia_finetuned_model(config: dict, adapter_path: str, model_name: str) -> dict:
    """MIA on a fine-tuned model variant."""
    model, tokenizer = load_base_model(
        model_name=config["model"]["name"],
        dtype=config["model"].get("dtype", "bfloat16"),
        gradient_checkpointing=False
    )
    model = load_adapter(model, adapter_path)

    results = run_mia_on_model(model, tokenizer, config, model_name)

    del model
    torch.cuda.empty_cache()
    return results


def print_comparison_table(results: dict):
    """Print MIA comparison table."""
    print("\n" + "=" * 80)
    print("PRIVACY LEAKAGE COMPARISON (MIA on C3 = SciQ)")
    print("=" * 80)
    print(f"{'Model':<25} {'Accuracy':>10} {'AUC':>10} {'Precision':>10} {'Recall':>10}")
    print("-" * 80)

    # Sort by AUC (higher = more leakage)
    sorted_results = sorted(results.items(), key=lambda x: x[1].get('auc', 0))

    for name, r in sorted_results:
        print(f"{name:<25} "
              f"{r.get('accuracy', 0):>10.4f} "
              f"{r.get('auc', 0):>10.4f} "
              f"{r.get('precision', 0):>10.4f} "
              f"{r.get('recall', 0):>10.4f}")

    print("-" * 80)
    print("Note: AUC closer to 0.5 = less leakage (better privacy)")
    print("      AUC closer to 1.0 = more leakage (worse privacy)")


def main():
    parser = argparse.ArgumentParser(description="Run privacy comparison")
    parser.add_argument("--config", type=str,
                        default="configs/fedlora_squad_triviaqa.json")
    parser.add_argument("--output-dir", type=str,
                        default="outputs/privacy_comparison")
    parser.add_argument("--um-adapter", type=str, required=True,
                        help="Path to Universal Model adapter (C1+C2)")
    parser.add_argument("--c3-adapter", type=str,
                        help="Path to BM+C3 adapter (direct fine-tune)")
    parser.add_argument("--kd-adapter", type=str,
                        help="Path to BM+C3 KD adapter")
    parser.add_argument("--avg-adapter", type=str,
                        help="Path to AVG(C1,C2,C3) adapter")
    parser.add_argument("--models", type=str, default="all",
                        help="Comma-separated list: bm,um,c3,kd,avg or 'all'")
    args = parser.parse_args()

    set_seed(42)

    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models_to_run = args.models.lower().split(",") if args.models != "all" else [
        "bm", "um", "c3", "kd", "avg"
    ]

    all_results = {}

    # 1. Base Model
    if "bm" in models_to_run:
        all_results["BM"] = mia_base_model(config)

    # 2. Universal Model
    if "um" in models_to_run:
        all_results["UM"] = mia_universal_model(config, args.um_adapter)

    # 3. BM + C3 (direct fine-tune)
    if "c3" in models_to_run and args.c3_adapter:
        all_results["BM + (C3)"] = mia_finetuned_model(
            config, args.c3_adapter, "BM + (C3)"
        )

    # 4. BM + C3 w KD
    if "kd" in models_to_run and args.kd_adapter:
        all_results["BM + (C3 w KD)"] = mia_finetuned_model(
            config, args.kd_adapter, "BM + (C3 w KD)"
        )

    # 5. AVG(C1, C2, C3)
    if "avg" in models_to_run and args.avg_adapter:
        all_results["BM + AVG(C1,C2,C3)"] = mia_finetuned_model(
            config, args.avg_adapter, "BM + AVG(C1,C2,C3)"
        )

    if all_results:
        print_comparison_table(all_results)
        save_results(all_results, output_dir / "mia_comparison.json")


if __name__ == "__main__":
    main()
