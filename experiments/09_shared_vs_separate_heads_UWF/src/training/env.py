"""Runtime environment: device selection and full reproducibility seeding.

Embedded as the notebook's setup cell so `device`, `seed_everything`, and the lesion vocabulary are
available to every later cell in one flat namespace.
"""
import random

import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def seed_everything(seed):
    """Fix every RNG for deterministic, reproducible runs (cudnn deterministic, benchmark off)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_rng_state():
    """Snapshot RNG states so a resumed run continues the exact same stochastic stream."""
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def set_rng_state(state):
    if not state:
        return
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state.get('cuda') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


# Fixed lesion vocabulary (repo-wide constant).
LESION_NAMES = ['MA', 'HE', 'IH', 'VB/IRMA', 'NV', 'VH', 'RD']
RARE_LESION_IDX = [4, 5, 6]
COMMON_LESION_IDX = [0, 1, 2, 3]
GRADE_NAMES = [f'Grade{g}' for g in range(5)]
