"""Locate project files without importing ROS or depending on the working directory.

Configuration lookup keeps the existing priority: an explicit environment
override, a ROS package share (when available), then the source checkout.
"""
from __future__ import annotations

import os
from pathlib import Path
from collections.abc import Iterator


def repo_root(start: str | Path | None = None) -> Path:
    """Find the source checkout containing ``start`` (this module by default)."""
    here = Path(start if start is not None else __file__).expanduser().resolve()
    for parent in (here, *here.parents):
        if ((parent / 'VERSION').is_file()
                and (parent / 'src/px4_nmpc_config/config/nmpc.yaml').is_file()):
            return parent
    raise FileNotFoundError('repository root not found; run from a source checkout')


def config_candidates(
    filename: str, *, env_var: str | None = None, start: str | Path | None = None,
) -> Iterator[Path]:
    """Yield configuration locations in override / installed / source order."""
    override = os.environ.get(env_var) if env_var else None
    if override:
        yield Path(override).expanduser()
    try:
        from ament_index_python.packages import get_package_share_directory
        share = get_package_share_directory('px4_nmpc_config')
    except (ImportError, LookupError):
        pass
    else:
        yield Path(share) / 'config' / filename
    try:
        root = repo_root(start)
    except FileNotFoundError:
        return
    yield root / 'src/px4_nmpc_config/config' / filename


def config_file(
    filename: str, *, env_var: str | None = None, start: str | Path | None = None,
) -> Path:
    """Return the first existing configuration file, or an actionable error."""
    candidates = list(config_candidates(filename, env_var=env_var, start=start))
    for path in candidates:
        if path.is_file():
            return path
    hint = f'; set {env_var} to override' if env_var else ''
    raise FileNotFoundError(f'{filename} not found; searched {candidates}{hint}')
