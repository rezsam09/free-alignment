"""seeds.py -- one place where every seed is set. Values come from config['seeds'] (frozen)."""
from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, torch_deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if torch_deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def rng(cfg: dict, name: str) -> np.random.Generator:
    """Named, independent generator, e.g. rng(cfg, 'bootstrap') or rng(cfg, 'permutation')."""
    return np.random.default_rng(int(cfg["seeds"][name]))
