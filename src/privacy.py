"""Privacy mechanisms for differential privacy."""

from typing import Tuple


def clip_gradients(gradients: dict, max_norm: float) -> dict:
    """Clip gradients to max L2 norm."""
    raise NotImplementedError


def add_noise(gradients: dict, noise_scale: float) -> dict:
    """Add Gaussian noise for differential privacy."""
    raise NotImplementedError


def compute_privacy_spent(
    num_steps: int,
    noise_multiplier: float,
    sample_rate: float,
    delta: float
) -> float:
    """Compute epsilon given privacy parameters."""
    raise NotImplementedError
