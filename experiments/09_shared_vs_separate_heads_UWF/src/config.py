"""Config loader: merges the six YAML files under configs/ into ONE flat CFG dict.

The YAML files are the single source of truth for every hyperparameter. Both the local package and
the notebook generator (create_nb.py) read the CFG through here, so there is no duplicated config.

Usage:
    from config import load_config
    cfg = load_config(head_mode='separate', seed=42)     # optional overrides
"""
import os

try:
    import yaml
except ImportError:                                            # pragma: no cover
    yaml = None

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs')
_FILES = ['data.yaml', 'model.yaml', 'loss.yaml', 'train.yaml', 'eval.yaml', 'kaggle.yaml']


def _load_yaml(path):
    with open(path, 'r', encoding='utf-8') as fh:
        if yaml is not None:
            return yaml.safe_load(fh) or {}
        return _tiny_yaml(fh.read())                            # zero-dependency fallback


def _tiny_yaml(text):
    """Minimal YAML reader for the flat 'key: value' / 'key: [a, b]' files here (no nesting).
    Only used if PyYAML is unavailable; the real files are plain enough for this to be exact."""
    import ast
    out = {}
    for raw in text.splitlines():
        line = raw.split('#', 1)[0].rstrip()
        if not line or ':' not in line:
            continue
        key, val = line.split(':', 1)
        key, val = key.strip(), val.strip()
        if val == '':
            continue
        low = val.lower()
        if low in ('true', 'false'):
            out[key] = (low == 'true')
        elif val.startswith('['):
            out[key] = ast.literal_eval(val)
        else:
            try:
                out[key] = ast.literal_eval(val)
            except (ValueError, SyntaxError):
                out[key] = val
    return out


def load_config(config_dir=CONFIG_DIR, **overrides):
    """Merge all config YAMLs into one flat dict, then apply keyword overrides."""
    cfg = {}
    for name in _FILES:
        path = os.path.join(config_dir, name)
        if os.path.exists(path):
            section = _load_yaml(path)
            dup = set(section) & set(cfg)
            if dup:
                raise ValueError(f"duplicate config keys across YAMLs: {sorted(dup)}")
            cfg.update(section)
    cfg.update(overrides)
    return cfg


# Shared, fixed lesion vocabulary (repo-wide constant; kept here so every module agrees).
LESION_NAMES = ['MA', 'HE', 'IH', 'VB/IRMA', 'NV', 'VH', 'RD']
RARE_LESION_IDX = [4, 5, 6]         # NV, VH, RD -> PDR-defining, sight-threatening
COMMON_LESION_IDX = [0, 1, 2, 3]    # MA, HE, IH, VB/IRMA
GRADE_NAMES = [f'Grade{g}' for g in range(5)]

# Head-parameter budget (lesion head only), verified by exact arithmetic; asserted at model-build
# time for the fairness table. The crucial pair is shared_mlp (66,567) vs separate (68,103): within
# 2.3%, so comparing B against A' isolates HEAD SEPARATION rather than added capacity.
#   shared     : 512*7+7                                             = 3,591
#   shared_mlp : (512*128+128) + (128*7+7)                           = 66,567
#   separate   : (512*4+4) + (512*128+128) + (128*3+3)               = 68,103
#   adapter    : (512*128+128) + (128*4+4) + (128*64+64) + (64*3+3)  = 74,631
EXPECTED_HEAD_PARAMS = {'shared': 3591, 'shared_mlp': 66567, 'separate': 68103, 'adapter': 74631}
