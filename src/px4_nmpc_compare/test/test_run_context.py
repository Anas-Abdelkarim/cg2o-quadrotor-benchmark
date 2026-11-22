"""Benchmark input provenance must not silently overwrite same-name files."""
from pathlib import Path
import hashlib
import yaml
from px4_nmpc_compare.common import repo_root, snapshot_run_context


def test_run_context_keeps_duplicate_config_basenames(tmp_path):
    first = tmp_path / 'a'; second = tmp_path / 'b'; first.mkdir(); second.mkdir()
    (first / 'nmpc.yaml').write_text('backend: first\n')
    (second / 'nmpc.yaml').write_text('backend: second\n')
    out = tmp_path / 'result'
    result = snapshot_run_context(out, [first / 'nmpc.yaml', second / 'nmpc.yaml'])
    assert (out / 'config_used/nmpc.yaml').read_text() == 'backend: first\n'
    assert (out / 'config_used/nmpc_2.yaml').read_text() == 'backend: second\n'
    assert list(result['configs']) == ['nmpc.yaml', 'nmpc_2.yaml']
    assert result['config_sha256']['nmpc.yaml'] == hashlib.sha256(b'backend: first\n').hexdigest()
    assert result['software']['project_version'] == (repo_root() / 'VERSION').read_text().strip()
    assert yaml.safe_load((out / 'run_context.yaml').read_text())['software']['project_version'] == (repo_root() / 'VERSION').read_text().strip()
