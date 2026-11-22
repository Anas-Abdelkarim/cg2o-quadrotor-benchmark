"""A failing build component must stop the build script with a non-zero exit code before later steps run."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _script(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('#!/usr/bin/env bash\nset -euo pipefail\n' + text + '\n')
    path.chmod(0o755)


@pytest.mark.parametrize('dispatcher,first,second,args', [
    ('build_backends.sh', 'build_backends/cg2o.sh', 'build_backends/ipopt_cpp.sh', []),
    ('build_externals.sh', 'build_externals/g2o.sh', 'build_externals/cg2o.sh', ['--only=g2o,cg2o']),
    ('build_externals.sh', 'build_externals/ipopt.sh', 'build_externals/g2o.sh', ['--only=ipopt,g2o']),
    ('rebuild_cg2o.sh', 'build_externals/cg2o.sh', 'build_backends/cg2o.sh', []),
])
def test_dispatcher_stops_when_a_component_fails(tmp_path, dispatcher, first, second, args):
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    shutil.copy2(ROOT / 'scripts' / dispatcher, scripts / dispatcher)
    sentinel = tmp_path / 'should_not_build'
    _script(scripts / first, 'exit 17')
    _script(scripts / second, 'touch "$BUILD_SENTINEL"')
    result = subprocess.run(['bash', str(scripts / dispatcher), *args],
                            env=dict(os.environ, BUILD_SENTINEL=str(sentinel)),
                            capture_output=True, text=True)
    assert result.returncode != 0, result.stdout + result.stderr
    assert not sentinel.exists()


def test_build_benchmark_stops_when_the_backends_fail(tmp_path):
    root = tmp_path
    (root / 'scripts').mkdir()
    shutil.copy2(ROOT / 'scripts/build_benchmark.sh', root / 'scripts/build_benchmark.sh')
    _script(root / 'scripts/build_backends.sh', 'exit 17')
    for lib in ('g2o', 'cg2o', 'ipopt'):
        (root / 'install/externals' / lib).mkdir(parents=True)
    venv = root / 'venv'
    (venv / 'bin').mkdir(parents=True)
    (venv / 'pyvenv.cfg').write_text('include-system-site-packages = true\n')
    for name in ('python', 'python3'):
        _script(venv / 'bin' / name, 'exit 0')
    sentinel = tmp_path / 'cmake_ran'
    fake_bin = tmp_path / 'fake_bin'
    _script(fake_bin / 'cmake', 'touch "$BUILD_SENTINEL"')
    result = subprocess.run(['bash', str(root / 'scripts/build_benchmark.sh'), '--no-ros'],
                            env=dict(os.environ, BUILD_SENTINEL=str(sentinel), PYTHON=str(venv / 'bin/python'),
                                     PATH=str(fake_bin) + os.pathsep + os.environ['PATH']),
                            capture_output=True, text=True)
    assert result.returncode == 17, result.stdout + result.stderr
    assert not sentinel.exists()
