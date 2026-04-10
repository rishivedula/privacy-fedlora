#!/usr/bin/env python
"""Run C3 multi-domain experiments with ensemble distillation.

For each C3 domain (50 train / 150 held-out), runs 4 conditions:
  BM          - Base model, no fine-tuning
  UM          - Universal teacher (ensemble of C1+C2 adapters) at inference
  BM + (C3)  - Base model fine-tuned on 50 C3 examples
  BM + KD     - Fine-tuned on C3 with dual-teacher KD from ensemble

The universal teacher is built via EnsembleTeacher (logit averaging), NOT
FedAvg weight averaging — FedAvg collapses to zero on small models.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.model import load_base_model, setup_lora, load_adapter, save_adapter
from src.data import (
    load_sciq, format_sciq_example,
    load_pubmedqa, format_pubmedqa_example,
    load_medqa, format_medqa_example,
    load_finqa, format_finqa_example,
    load_code_qa, format_code_qa_example,
    load_arc, format_arc_example,
    load_openbookqa, format_openbookqa_example,
    load_commonsense_qa, format_commonsense_qa_example,
    preprocess_dataset, create_dataloader,
)
from src.evaluator import evaluate_qa
from src.trainer import train_lora
from src.kd_trainer import train_with_dual_teacher_kd
from src.aggregator import EnsembleTeacher, fedavg_lora, load_adapter_weights, save_aggregated_adapter


# Maps dataset name -> (train_loader, eval_split, format_fn, dataset_type)
_DOMAIN_REGISTRY = {
    "sciq":           (load_sciq,           "validation", format_sciq_example,           "sciq"),
    "pubmedqa":       (load_pubmedqa,       "test",       format_pubmedqa_example,       "pubmedqa"),
    "medqa":          (load_medqa,          "test",       format_medqa_example,          "medqa"),
    "finqa":          (load_finqa,          "test",       format_finqa_example,          "finqa"),
    "code_qa":        (load_code_qa,        "test",       format_code_qa_example,        "code_qa"),
    "arc":            (load_arc,            "validation", format_arc_example,            "arc"),
    "openbookqa":     (load_openbookqa,     "validation", format_openbookqa_example,     "openbookqa"),
    "commonsense_qa": (load_commonsense_qa, "validation", format_commonsense_qa_example, "commonsense_qa"),
}


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


def get_domain_data(domain_cfg: dict, tokenizer, config: dict):
    """Load train/eval data for a C3 domain."""
    dataset_name = domain_cfg["dataset"]
    train_n = domain_cfg["train_samples"]
    eval_n = domain_cfg["eval_samples"]
    max_length = config["training"].get("max_seq_length", 512)

    if dataset_name not in _DOMAIN_REGISTRY:
        raise ValueError(f"Unknown C3 domain: {dataset_name}")

    load_fn, eval_split, format_fn, dtype_key = _DOMAIN_REGISTRY[dataset_name]

    train_raw = load_fn("train", train_n)
    eval_raw = load_fn(eval_split, eval_n)

    train_dataset = preprocess_dataset(train_raw, tokenizer, dtype_key, max_length)
    eval_examples = [format_fn(ex) for ex in eval_raw]

    return train_dataset, eval_examples


def build_ensemble_teacher(config: dict, c1_adapter: str, c2_adapter: str) -> EnsembleTeacher:
    """Build universal teacher by ensembling C1 and C2 adapters."""
    model_name = config["model"]["name"]
    dtype = config["model"].get("dtype", "bfloat16")
    return EnsembleTeacher(model_name, [c1_adapter, c2_adapter], dtype=dtype)


# ---------------------------------------------------------------------------
# Per-domain experiment functions
# ---------------------------------------------------------------------------

def run_bm(domain_cfg: dict, config: dict, tokenizer, output_dir: Path) -> dict:
    """Condition 1: Base model, no fine-tuning."""
    model, _ = load_base_model(
        model_name=config["model"]["name"],
        dtype=config["model"].get("dtype", "bfloat16"),
        gradient_checkpointing=False
    )
    _, eval_examples = get_domain_data(domain_cfg, tokenizer, config)
    results = evaluate_qa(model, tokenizer, eval_examples, max_samples=domain_cfg["eval_samples"])
    del model
    torch.cuda.empty_cache()
    return results


def run_um(domain_cfg: dict, config: dict, tokenizer, ensemble: EnsembleTeacher,
           output_dir: Path) -> dict:
    """Condition 2: Universal teacher (ensemble) at inference — no C3 fine-tuning."""
    _, eval_examples = get_domain_data(domain_cfg, tokenizer, config)
    results = evaluate_qa(ensemble, tokenizer, eval_examples, max_samples=domain_cfg["eval_samples"])
    return results


def run_bm_c3(domain_cfg: dict, config: dict, tokenizer, output_dir: Path):
    """Condition 3: Base model fine-tuned on 50 C3 examples."""
    model, _ = load_base_model(
        model_name=config["model"]["name"],
        dtype=config["model"].get("dtype", "bfloat16"),
        gradient_checkpointing=True
    )
    model = setup_lora(model, config.get("lora", {}))

    train_dataset, eval_examples = get_domain_data(domain_cfg, tokenizer, config)
    batch_size = config["training"].get("batch_size", 4)
    train_loader = create_dataloader(train_dataset, batch_size=batch_size)

    train_with = train_lora(model, train_loader, config)

    adapter_path = output_dir / f"{domain_cfg['name']}_bm_c3_adapter"
    save_adapter(model, str(adapter_path))

    model.eval()
    results = evaluate_qa(model, tokenizer, eval_examples, max_samples=domain_cfg["eval_samples"])
    results["train_loss"] = train_with["train_loss"]

    del model
    torch.cuda.empty_cache()
    return results, str(adapter_path)


def run_bm_kd(domain_cfg: dict, config: dict, tokenizer, ensemble: EnsembleTeacher,
              output_dir: Path) -> dict:
    """Condition 4: Fine-tune on C3 with dual-teacher KD from ensemble."""
    base_teacher, _ = load_base_model(
        model_name=config["model"]["name"],
        dtype=config["model"].get("dtype", "bfloat16"),
        gradient_checkpointing=False
    )
    student, _ = load_base_model(
        model_name=config["model"]["name"],
        dtype=config["model"].get("dtype", "bfloat16"),
        gradient_checkpointing=True
    )
    student = setup_lora(student, config.get("lora", {}))

    train_dataset, eval_examples = get_domain_data(domain_cfg, tokenizer, config)
    batch_size = config["training"].get("batch_size", 4)
    train_loader = create_dataloader(train_dataset, batch_size=batch_size)

    # EnsembleTeacher duck-types as universal_teacher_model in train_with_dual_teacher_kd
    train_metrics = train_with_dual_teacher_kd(
        student, base_teacher, ensemble, train_loader, config
    )

    adapter_path = output_dir / f"{domain_cfg['name']}_kd_adapter"
    save_adapter(student, str(adapter_path))

    student.eval()
    results = evaluate_qa(student, tokenizer, eval_examples, max_samples=domain_cfg["eval_samples"])
    results["train_loss"] = train_metrics["train_loss"]

    del student, base_teacher
    torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def print_results_table(all_results: dict):
    """Print per-domain results across all conditions."""
    conditions = ["BM", "UM", "BM + (C3)", "BM + KD"]
    col_w = 12

    print("\n" + "=" * 90)
    print("RESULTS TABLE  (F1)  —  C3 domains × conditions")
    print("=" * 90)

    header = f"{'Domain':<18}" + "".join(f"{c:>{col_w}}" for c in conditions)
    print(header)
    print("-" * 90)

    for domain, cond_results in all_results.items():
        row = f"{domain:<18}"
        for c in conditions:
            f1 = cond_results.get(c, {}).get("f1", float("nan"))
            row += f"{f1:>{col_w}.4f}"
        print(row)

    print("=" * 90)

    # Delta vs BM
    print("\nDELTA vs BM  (UM - BM, KD - BM):")
    print("-" * 60)
    for domain, cond_results in all_results.items():
        bm_f1 = cond_results.get("BM", {}).get("f1", float("nan"))
        um_f1 = cond_results.get("UM", {}).get("f1", float("nan"))
        kd_f1 = cond_results.get("BM + KD", {}).get("f1", float("nan"))
        delta_um = um_f1 - bm_f1
        delta_kd = kd_f1 - bm_f1
        print(f"  {domain:<18}  UM delta: {delta_um:+.4f}    KD delta: {delta_kd:+.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run multi-domain C3 ensemble distillation experiments")
    parser.add_argument("--config", type=str, default="configs/c3_experiment.json")
    parser.add_argument("--output-dir", type=str, default="outputs/c3_experiments")
    parser.add_argument("--c1-adapter", type=str, required=True, help="Path to C1 (squad_v2) adapter")
    parser.add_argument("--c2-adapter", type=str, required=True, help="Path to C2 (triviaqa) adapter")
    parser.add_argument("--domain", type=str, default="all",
                        help="Run a single domain by name, or 'all'")
    parser.add_argument("--condition", type=str, default="all",
                        choices=["all", "bm", "um", "bm_c3", "bm_kd"])
    args = parser.parse_args()

    set_seed(42)
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer once
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build ensemble teacher once (shared across all domains)
    print("\nBuilding EnsembleTeacher from C1 + C2 adapters...")
    ensemble = build_ensemble_teacher(config, args.c1_adapter, args.c2_adapter)

    domains = config["c3_domains"]
    if args.domain != "all":
        domains = [d for d in domains if d["name"] == args.domain]
        if not domains:
            raise ValueError(f"Domain '{args.domain}' not found in config")

    all_results = {}

    for domain_cfg in domains:
        name = domain_cfg["name"]
        print(f"\n{'='*60}")
        print(f"DOMAIN: {name.upper()}")
        print(f"{'='*60}")

        domain_out = output_dir / name
        domain_out.mkdir(parents=True, exist_ok=True)
        domain_results = {}

        if args.condition in ("all", "bm"):
            print(f"\n[{name}] Condition: BM")
            r = run_bm(domain_cfg, config, tokenizer, domain_out)
            domain_results["BM"] = r
            save_results(r, domain_out / "bm.json")
            print(f"  F1={r.get('f1', 0):.4f}  EM={r.get('exact_match', 0):.4f}")

        if args.condition in ("all", "um"):
            print(f"\n[{name}] Condition: UM (ensemble)")
            r = run_um(domain_cfg, config, tokenizer, ensemble, domain_out)
            domain_results["UM"] = r
            save_results(r, domain_out / "um.json")
            print(f"  F1={r.get('f1', 0):.4f}  EM={r.get('exact_match', 0):.4f}")

        if args.condition in ("all", "bm_c3"):
            print(f"\n[{name}] Condition: BM + (C3)")
            r, _ = run_bm_c3(domain_cfg, config, tokenizer, domain_out)
            domain_results["BM + (C3)"] = r
            save_results(r, domain_out / "bm_c3.json")
            print(f"  F1={r.get('f1', 0):.4f}  EM={r.get('exact_match', 0):.4f}")

        if args.condition in ("all", "bm_kd"):
            print(f"\n[{name}] Condition: BM + KD (ensemble teacher)")
            r = run_bm_kd(domain_cfg, config, tokenizer, ensemble, domain_out)
            domain_results["BM + KD"] = r
            save_results(r, domain_out / "bm_kd.json")
            print(f"  F1={r.get('f1', 0):.4f}  EM={r.get('exact_match', 0):.4f}")

        all_results[name] = domain_results
        save_results(domain_results, domain_out / "all_conditions.json")

    if all_results:
        print_results_table(all_results)
        save_results(all_results, output_dir / "aggregate_results.json")


if __name__ == "__main__":
    main()
