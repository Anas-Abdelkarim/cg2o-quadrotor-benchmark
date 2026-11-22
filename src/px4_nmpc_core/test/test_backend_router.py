"""Solver-backend routing.  These tests need no compiled extension: they check
how ``make_ocp_solver`` behaves when a native backend can or cannot be used."""
import pytest

from px4_nmpc_core.config import build_configs
from px4_nmpc_core.ocp import make_ocp_solver


def _cfg(**params):
    base = {'control_interface': 'trajectory', 'horizon_steps': 10}
    base.update(params)
    return build_configs(base)[0]


def test_default_backend_is_ipopt_casadi():
    solver = make_ocp_solver(_cfg())
    assert solver.backend == 'ipopt_casadi'
    assert not solver.backend_error


def test_legacy_ipopt_name_still_works():
    solver = make_ocp_solver(_cfg(solver_backend='ipopt'))
    assert solver.backend in ('ipopt', 'ipopt_casadi')
    assert not solver.backend_error


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match='unsupported solver backend'):
        make_ocp_solver(_cfg(solver_backend='does_not_exist'))


@pytest.mark.parametrize('backend', ['cg2o', 'ipopt_cpp'])
def test_native_backend_with_non_trajectory_interface_falls_back(backend):
    solver = make_ocp_solver(_cfg(solver_backend=backend, control_interface='attitude', horizon_steps=8))
    assert solver.backend == 'ipopt_casadi'
    assert backend in solver.backend_error
    assert solver.interface == 'attitude'


@pytest.mark.parametrize('backend', ['cg2o', 'ipopt_cpp'])
def test_strict_mode_raises_instead_of_falling_back(backend):
    cfg = _cfg(solver_backend=backend, control_interface='attitude', horizon_steps=8,
               solver_allow_fallback=False)
    with pytest.raises(ValueError, match='trajectory'):
        make_ocp_solver(cfg)


@pytest.mark.parametrize('module,backend', [
    ('px4_nmpc_controller.px4_nmpc_ipopt_cpp.solver', 'ipopt_cpp'),
    ('px4_nmpc_controller.px4_nmpc_cg2o.solver', 'cg2o'),
])
def test_missing_extension_falls_back_with_reason(monkeypatch, module, backend):
    def missing():
        raise ImportError('extension not built (test)')
    monkeypatch.setattr(f'{module}._load_extension', missing)
    solver = make_ocp_solver(_cfg(solver_backend=backend))
    assert solver.backend == 'ipopt_casadi'
    assert 'extension not built' in solver.backend_error
    # the fallback solver is fully usable
    assert solver.nz > 0 and solver.ng > 0

    with pytest.raises(ImportError):
        make_ocp_solver(_cfg(solver_backend=backend, solver_allow_fallback=False))
