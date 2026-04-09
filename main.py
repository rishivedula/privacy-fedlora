"""Main entry point for all experiments."""

import argparse
import json
import random
from pathlib import Path

import numpy as np


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_config(config_path: str) -> dict:
    """Load and validate experiment configuration."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(path) as f:
        config = json.load(f)

    # Validate required fields
    required = ["experiment_name", "seed"]
    missing = [k for k in required if k not in config]
    if missing:
        raise ValueError(f"Config missing required fields: {missing}")

    return config


def main(config: dict) -> None:
    """Run experiment with given configuration."""
    set_seed(config["seed"])
    print(f"Running experiment: {config['experiment_name']}")
    # TODO: Implement experiment logic


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run federated learning experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    main(config)
