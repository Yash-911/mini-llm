"""Shared helpers used across the project."""
import os
import random

# Let unsupported MPS ops fall back to CPU instead of crashing.
# Must be set before torch initialises MPS, so we set it at import time.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch


def get_device() -> str:
    """Pick the best available device: cuda (Kaggle) > mps (Mac) > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(device: str) -> None:
    """Wait for queued GPU work to finish. GPU ops are async, so call this
    before reading a timer, or you only measure how long the launch took."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def set_seed(seed: int = 1337) -> None:
    """Seed every RNG we use so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds cuda/mps
