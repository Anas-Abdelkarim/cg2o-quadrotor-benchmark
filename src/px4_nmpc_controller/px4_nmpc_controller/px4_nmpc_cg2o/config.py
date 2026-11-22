from __future__ import annotations

from pathlib import Path

import yaml

from px4_nmpc_core.paths import config_candidates


def _config_candidates():
    return config_candidates('cg2o.yaml', env_var='PX4_NMPC_CG2O_CONFIG')


def load_cg2o_config(path=None):
    candidates = [Path(path)] if path else list(_config_candidates())
    for p in candidates:
        if p and p.exists():
            data = yaml.safe_load(p.read_text()) or {}
            cfg = data.get('cg2o', data)
            if not isinstance(cfg, dict):
                raise ValueError(f'invalid cg2o configuration in {p}')
            out = dict(cfg)
            out['_path'] = str(p)
            return out
    raise FileNotFoundError(
        'cg2o.yaml was not found. Expected px4_nmpc_config/config/cg2o.yaml '
        'or set PX4_NMPC_CG2O_CONFIG.'
    )
