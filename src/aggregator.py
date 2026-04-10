"""Federated aggregation utilities for LoRA adapters."""

import torch
from typing import Dict, List, Optional
from pathlib import Path
from peft import PeftModel


class EnsembleTeacher:
    """Universal teacher via ensemble inference over client adapters.

    Replaces FedAvg weight averaging (which collapses on small models) with
    logit averaging: each adapter runs on its own base model instance, then
    logits are averaged at inference time.
    """

    def __init__(
        self,
        base_model_name: str,
        adapter_paths: List[str],
        dtype: str = "bfloat16"
    ):
        from src.model import load_base_model, load_adapter

        self.models = []
        for i, path in enumerate(adapter_paths):
            print(f"EnsembleTeacher: loading member {i + 1}/{len(adapter_paths)} from {path}")
            model, _ = load_base_model(base_model_name, dtype=dtype, gradient_checkpointing=False)
            model = load_adapter(model, path)
            model.eval()
            self.models.append(model)

        print(f"EnsembleTeacher: {len(self.models)} members ready")

    def eval(self):
        for model in self.models:
            model.eval()
        return self

    def __call__(self, **batch):
        """Forward pass: average logits across all ensemble members."""
        all_logits = []
        primary_device = self.models[0].device
        for model in self.models:
            model_batch = {k: v.to(model.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**model_batch)
                all_logits.append(outputs.logits.to(primary_device))

        avg_logits = torch.stack(all_logits).mean(dim=0)

        class _Output:
            def __init__(self, logits):
                self.logits = logits

        return _Output(avg_logits)

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 50,
        **kwargs
    ) -> torch.Tensor:
        """Greedy ensemble generation: average logits at each decoding step."""
        device = self.models[0].device
        generated = input_ids.to(device)
        attn = attention_mask.to(device) if attention_mask is not None else None
        eos_id = getattr(self.models[0].config, "eos_token_id", None)

        for _ in range(max_new_tokens):
            all_next = []
            for model in self.models:
                m_gen = generated.to(model.device)
                m_attn = attn.to(model.device) if attn is not None else None
                with torch.no_grad():
                    out = model(input_ids=m_gen, attention_mask=m_attn)
                    all_next.append(out.logits[:, -1, :].to(device))

            next_token = torch.stack(all_next).mean(dim=0).argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)

            if attn is not None:
                attn = torch.cat([
                    attn,
                    torch.ones((attn.shape[0], 1), device=device, dtype=attn.dtype)
                ], dim=-1)

            if eos_id is not None and (next_token == eos_id).all():
                break

        return generated


def fedavg_lora(
    adapter_state_dicts: List[Dict[str, torch.Tensor]],
    weights: List[float] = None
) -> Dict[str, torch.Tensor]:
    """Aggregate LoRA adapters using FedAvg.

    Args:
        adapter_state_dicts: List of adapter state dicts from clients
        weights: Optional weights for each client (defaults to equal)

    Returns:
        Aggregated adapter state dict
    """
    if not adapter_state_dicts:
        raise ValueError("No adapters to aggregate")

    num_clients = len(adapter_state_dicts)

    if weights is None:
        weights = [1.0 / num_clients] * num_clients
    else:
        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]

    # Initialize aggregated dict with zeros
    aggregated = {}
    first_dict = adapter_state_dicts[0]

    for key in first_dict:
        aggregated[key] = torch.zeros_like(first_dict[key])

    # Weighted sum
    for state_dict, weight in zip(adapter_state_dicts, weights):
        for key in state_dict:
            aggregated[key] += weight * state_dict[key]

    return aggregated


def load_adapter_weights(adapter_path: str) -> Dict[str, torch.Tensor]:
    """Load adapter weights from saved checkpoint.

    Args:
        adapter_path: Path to adapter directory

    Returns:
        State dict with LoRA weights
    """
    path = Path(adapter_path)
    adapter_file = path / "adapter_model.bin"

    if adapter_file.exists():
        return torch.load(adapter_file, map_location="cpu")

    # Try safetensors format
    safetensor_file = path / "adapter_model.safetensors"
    if safetensor_file.exists():
        from safetensors.torch import load_file
        return load_file(safetensor_file)

    raise FileNotFoundError(f"No adapter found at {adapter_path}")


def save_aggregated_adapter(
    aggregated_weights: Dict[str, torch.Tensor],
    output_path: str,
    base_adapter_path: str
) -> None:
    """Save aggregated adapter weights.

    Copies config from base adapter and saves new weights.

    Args:
        aggregated_weights: Aggregated LoRA weights
        output_path: Where to save
        base_adapter_path: Path to copy config from
    """
    import shutil
    from pathlib import Path

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    base_path = Path(base_adapter_path)

    # Copy config files
    for config_file in ["adapter_config.json", "README.md"]:
        src = base_path / config_file
        if src.exists():
            shutil.copy(src, output_path / config_file)

    # Save weights
    torch.save(aggregated_weights, output_path / "adapter_model.bin")


def aggregate_from_paths(
    adapter_paths: List[str],
    output_path: str,
    weights: List[float] = None
) -> Dict[str, torch.Tensor]:
    """Load adapters from paths, aggregate, and save.

    Args:
        adapter_paths: List of paths to adapter directories
        output_path: Where to save aggregated adapter
        weights: Optional client weights

    Returns:
        Aggregated weights dict
    """
    # Load all adapters
    adapter_dicts = []
    for path in adapter_paths:
        weights_dict = load_adapter_weights(path)
        adapter_dicts.append(weights_dict)
        print(f"Loaded adapter from {path}")

    # Aggregate
    aggregated = fedavg_lora(adapter_dicts, weights)
    print(f"Aggregated {len(adapter_dicts)} adapters")

    # Save
    save_aggregated_adapter(aggregated, output_path, adapter_paths[0])
    print(f"Saved aggregated adapter to {output_path}")

    return aggregated
