"""Source and installed configuration discovery must survive the layout change."""
from importlib import import_module
from pathlib import Path
import sys
import types

import pytest

from px4_nmpc_core.paths import config_file, repo_root


def _installed_config(monkeypatch, directory):
    package = types.ModuleType('ament_index_python')
    packages = types.ModuleType('ament_index_python.packages')
    packages.get_package_share_directory = lambda name: str(directory)
    package.packages = packages
    monkeypatch.setitem(sys.modules, 'ament_index_python', package)
    monkeypatch.setitem(sys.modules, 'ament_index_python.packages', packages)


def test_source_defaults_work_from_an_unrelated_directory(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, 'ament_index_python.packages', None)
    monkeypatch.chdir(tmp_path)
    path = config_file('nmpc.yaml')
    assert path == repo_root() / 'src/px4_nmpc_config/config/nmpc.yaml'
    assert path.is_file()


def test_installed_share_precedes_source(monkeypatch, tmp_path):
    share = tmp_path / 'share/px4_nmpc_config'
    (share / 'config').mkdir(parents=True)
    installed = share / 'config/nmpc.yaml'
    installed.write_text('installed: true\n')
    _installed_config(monkeypatch, share)
    assert config_file('nmpc.yaml') == installed


def test_override_precedes_installed_share(monkeypatch, tmp_path):
    override = tmp_path / 'override.yaml'
    override.write_text('override: true\n')
    _installed_config(monkeypatch, tmp_path / 'share')
    monkeypatch.setenv('PX4_NMPC_TEST_CONFIG', str(override))
    assert config_file('nmpc.yaml', env_var='PX4_NMPC_TEST_CONFIG') == override


def test_missing_candidates_fall_back_to_source(monkeypatch, tmp_path):
    _installed_config(monkeypatch, tmp_path / 'missing_share')
    monkeypatch.setenv('PX4_NMPC_TEST_CONFIG', str(tmp_path / 'missing.yaml'))
    path = config_file('nmpc.yaml', env_var='PX4_NMPC_TEST_CONFIG')
    assert path == repo_root() / 'src/px4_nmpc_config/config/nmpc.yaml'


def test_missing_file_reports_searched_paths(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, 'ament_index_python.packages', None)
    with pytest.raises(FileNotFoundError, match='missing.yaml not found; searched'):
        config_file('missing.yaml', start=tmp_path)
    with pytest.raises(FileNotFoundError, match='repository root not found'):
        repo_root(tmp_path.anchor)


@pytest.mark.parametrize('backend,env_var,loader', [
    ('cg2o', 'PX4_NMPC_CG2O_CONFIG', 'load_cg2o_config'),
    ('ipopt_cpp', 'PX4_NMPC_IPOPT_CPP_CONFIG', 'load_ipopt_cpp_config'),
])
def test_backend_loaders_keep_environment_overrides(monkeypatch, tmp_path, backend, env_var, loader):
    override = tmp_path / f'{backend}.yaml'
    override.write_text(f'{backend}:\n  cleanup_test: 42\n')
    monkeypatch.setenv(env_var, str(override))
    module = import_module(f'px4_nmpc_controller.px4_nmpc_{backend}.config')
    config = getattr(module, loader)()
    assert config['cleanup_test'] == 42
    assert Path(config['_path']) == override
