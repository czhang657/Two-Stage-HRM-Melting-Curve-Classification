"""Utility functions for reproducibility and common operations."""

import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed (int): Random seed value. Default is 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seed set to {seed} for reproducibility")
