from __future__ import annotations

from pathlib import Path
import yaml

from px4_nmpc_core.paths import config_candidates


def _candidates():
    return config_candidates('ipopt_cpp.yaml', env_var='PX4_NMPC_IPOPT_CPP_CONFIG')


def load_ipopt_cpp_config(path=None):
    candidates = [Path(path)] if path else list(_candidates())
    for p in candidates:
        if p and p.exists():
            data = yaml.safe_load(p.read_text()) or {}
            cfg = data.get('ipopt_cpp', data)
            if not isinstance(cfg, dict):
                raise ValueError(f'invalid ipopt_cpp configuration in {p}')
            out = dict(cfg)
            out['_path'] = str(p)
            return out
    raise FileNotFoundError('ipopt_cpp.yaml not found; set PX4_NMPC_IPOPT_CPP_CONFIG or install px4_nmpc_config')
