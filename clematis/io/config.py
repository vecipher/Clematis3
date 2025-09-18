from __future__ import annotations
from dataclasses import asdict
from typing import Any, Dict
try:
    import yaml
except ImportError:
    yaml = None
from ..engine.types import Config

def load_config(path: str | None = None) -> Config:
    # If PyYAML unavailable or file missing, return defaults.
    if not path or (yaml is None):
        return Config()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = yaml.safe_load(f) or {}
        cfg = Config()
        # Shallow update known sections
        for k in ["k_surface","surface_method","t1","t2","t3","t4","budgets","flags"]:
            if k in data:
                setattr(cfg, k, data[k])
        return cfg
    except FileNotFoundError:
        return Config()
