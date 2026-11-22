from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shutil
import sys
import subprocess
from importlib import metadata
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from px4_nmpc_core.scenario import generate_scenario, scenario_parameters
from px4_nmpc_core.paths import repo_root


def config_dir(root: Path | None = None) -> Path:
    return (root or repo_root()) / 'src' / 'px4_nmpc_config' / 'config'




def resolve_experiment_bundle(path: str | Path):
    """Resolve one experiment-bundle manifest to absolute config paths.

    Bundles keep scenario geometry, controller overrides and comparison settings
    together without creating a duplicate controller package. Paths inside the
    manifest are relative to the bundle file, so the whole experiment directory
    can be moved/copied as one unit.
    """
    manifest = Path(path).expanduser().resolve()
    raw = load_yaml(manifest) or {}
    data = raw.get('experiment_bundle', raw.get('bundle'))
    if not isinstance(data, dict):
        raise ValueError(f'{manifest} must contain an experiment_bundle mapping')
    files = data.get('files', {})
    if not isinstance(files, dict):
        raise ValueError('experiment_bundle.files must be a mapping')
    base = manifest.parent

    def _resolve(key, required=True):
        value = files.get(key)
        if value in (None, ''):
            if required:
                raise ValueError(f'experiment bundle is missing files.{key}')
            return None
        q = Path(value).expanduser()
        if not q.is_absolute():
            q = base / q
        q = q.resolve()
        if not q.exists():
            raise FileNotFoundError(f'experiment bundle files.{key} does not exist: {q}')
        return q

    out = {
        'manifest': manifest,
        'name': str(data.get('name', manifest.stem)),
        'formulation': str(data.get('formulation', 'trajectory')),
        'nmpc': _resolve('nmpc'),
        'comparison': _resolve('comparison'),
        'experiment': _resolve('experiment'),
        'scenario': _resolve('scenario'),
        'generator': _resolve('generator', required=False),
        'raw': data,
    }
    return out


def parse_ints(spec: str):
    if ':' in spec and ',' not in spec:
        a, b = (int(x) for x in spec.split(':', 1))
        return list(range(a, b))
    return [int(x.strip()) for x in spec.split(',') if x.strip()]


def load_yaml(path: str | Path):
    return yaml.safe_load(Path(path).read_text())


def geometry_hash(scn) -> str:
    data = scn.to_dict() if hasattr(scn, 'to_dict') else dict(scn)
    payload = {
        'start': np.asarray(data['start_ned'], float).round(12).tolist(),
        'goal': np.asarray(data['goal_ned'], float).round(12).tolist(),
        'centers': np.asarray(data['obstacle_centers_ned'], float).round(12).tolist(),
        'radii': np.asarray(data['obstacle_radii'], float).round(12).tolist(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def comparison_settings(path: str | Path):
    return load_yaml(path)['comparison']


def experiment_for_count(experiment_path: str | Path, comparison: dict, count: int):
    exp = dict(load_yaml(experiment_path)['experiment'])
    dist = comparison.get('obstacle_distribution', {})
    exp.update({
        'obstacle_count': int(count),
        'obstacle_radius_mean_m': float(dist.get('radius_mean_m', 0.50)),
        'obstacle_radius_std_m': float(dist.get('radius_std_m', 0.10)),
        'obstacle_radius_min_m': float(dist.get('radius_min_m', 0.30)),
        'obstacle_radius_max_m': float(dist.get('radius_max_m', 1.00)),
        'obstacle_min_gap_m': float(dist.get('min_gap_m', 0.30)),
    })
    return exp


def load_base_params(nmpc_path: str | Path):
    p = dict(load_yaml(nmpc_path)['px4_nmpc_controller']['ros__parameters'])
    p['control_interface'] = 'trajectory'
    return p


def params_for_scenario(base_params, exp, scn, backend):
    """Build backend parameters from one already-generated scenario.

    Comparison code deliberately generates each `(seed, obstacle_count)` geometry
    once and passes this same scenario object to every backend.
    """
    params = dict(base_params)
    params.update(scenario_parameters(scn, float(exp.get('wall_margin_m', 0.45))))
    params['solver_backend'] = str(backend)
    # never let a benchmark silently measure a different solver than the one requested
    params['solver_allow_fallback'] = False
    return params


def paired_case(base_params, exp, seed, backend):
    """Compatibility helper for callers that need one generated case."""
    scn = generate_scenario(exp, int(seed))
    return scn, params_for_scenario(base_params, exp, scn, backend)


def write_scenario(scn, exp, seed, count, path: Path):
    """Write a controller/Gazebo-compatible schema-5 scenario JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = scn.to_dict() if hasattr(scn, 'to_dict') else dict(scn)
    data['seed'] = int(seed)
    data['experiment_config'] = dict(exp)
    data['comparison_metadata'] = {
        'obstacle_count': int(count),
        'geometry_hash': geometry_hash(scn),
        'generator': {
            k: exp[k] for k in (
                'obstacle_radius_mean_m', 'obstacle_radius_std_m',
                'obstacle_radius_min_m', 'obstacle_radius_max_m',
                'obstacle_min_gap_m')
        },
    }
    path.write_text(json.dumps(data, indent=2))
    return data


def write_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)



def unique_run_dir(root: Path, label: str, explicit: str | Path | None = None, group: str = 'comparison') -> Path:
    """Create a unique timestamped results directory unless an explicit path is supplied.

    Default names start with ``YYMMDD_HHMMSS`` (for example
    ``260531_142245_level1_single_solve``), so result folders sort
    chronologically and never overwrite an earlier experiment.
    """
    if explicit:
        out = Path(explicit).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        return out
    base = Path(root) / 'results' / group
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%y%m%d_%H%M%S')
    candidate = base / f'{stamp}_{label}'
    idx = 1
    while candidate.exists():
        candidate = base / f'{stamp}_{label}_{idx:02d}'
        idx += 1
    candidate.mkdir(parents=True)
    return candidate


def _git_revision(path: Path):
    """Record an exact dependency checkout when present; tolerate release ZIPs."""
    if not (path / '.git').exists():
        return None
    try:
        result = subprocess.run(['git', '-C', str(path), 'rev-parse', 'HEAD'],
                                check=True, capture_output=True, text=True, timeout=3)
        return result.stdout.strip() or None
    except (subprocess.SubprocessError, OSError):
        return None


def _dependency_versions():
    versions = {}
    for package in ('casadi', 'numpy', 'PyYAML', 'matplotlib', 'pybind11'):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def snapshot_run_context(out: Path, config_paths, metadata=None):
    """Capture inputs and software revisions without colliding config filenames.

    Most baseline runs still use config_used/nmpc.yaml etc. If two inputs
    share a basename, suffix the second rather than silently overwriting it.
    This function only records metadata; it never modifies the optimization.
    """
    out = Path(out)
    cfg_dir = out / 'config_used'
    cfg_dir.mkdir(parents=True, exist_ok=True)
    copied, hashes = {}, {}
    for path in config_paths:
        if not path:
            continue
        p = Path(path).expanduser()
        if not p.is_file():
            continue
        name = p.name
        index = 2
        while name in copied:
            name = f'{p.stem}_{index}{p.suffix}'
            index += 1
        dst = cfg_dir / name
        shutil.copy2(p, dst)
        copied[name] = str(dst.relative_to(out))
        hashes[name] = hashlib.sha256(dst.read_bytes()).hexdigest()
    root = repo_root()
    external_names = ('g2o', 'cg2o', 'PX4-Autopilot', 'Micro-XRCE-DDS-Agent')
    meta = {
        'created_local': datetime.now().astimezone().isoformat(timespec='seconds'),
        'argv': sys.argv,
        'cwd': os.getcwd(),
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'configs': copied,
        'config_sha256': hashes,
        'software': {
            'project_version': (root / 'VERSION').read_text().strip(),
            'project_revision': _git_revision(root),
            'dependency_versions': _dependency_versions(),
            'external_revisions': {key: _git_revision(root / 'externals' / key)
                                   for key in external_names},
        },
    }
    if metadata:
        meta.update(metadata)
    (out / 'run_context.yaml').write_text(yaml.safe_dump(meta, sort_keys=False))
    return meta
