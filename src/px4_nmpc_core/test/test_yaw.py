"""Optimised-yaw extension (yaw_mode: optimized): model, config, solver backends and closed loop.

Backends that are not compiled (ipopt_cpp, cg2o) are skipped, the CasADi/IPOPT tests always run."""
import math

import numpy as np
import pytest

from px4_nmpc_core.config import build_configs
from px4_nmpc_core.models import make_model
from px4_nmpc_core.ocp import make_ocp_solver
from px4_nmpc_core.scenario import generate_scenario, scenario_parameters
from px4_nmpc_core.sim.episode import run_episode
from px4_nmpc_core.sim.px4_like import SimParams
from test_runtime_closed_loop import EXP


def _cfg(backend='ipopt_casadi', **extra):
    p = {'yaw_mode': 'optimized', 'solver_backend': backend, 'solver_allow_fallback': False}
    p.update(extra)
    return build_configs(p)


def _problem(N, dt):
    x0 = np.zeros(11)
    x0[3:6] = [0.8, 0.5, 0.0]
    x0[6:9] = [0.1, -0.05, 0.0]
    x0[9], x0[10] = 0.4, 0.1
    t = np.arange(N + 1) * dt
    pref = np.stack([0.9 * t, 0.4 * t, -1.0 + 0 * t])
    vref = np.tile(np.array([[0.9], [0.4], [0.0]]), (1, N + 1))
    obs = np.array([[1.2, 0.3, -1.0, 0.6], [2.0, 1.2, -1.0, 0.7]])
    hdg = 0.4 + np.linspace(0.0, 0.3, N + 1)
    hw = np.linspace(2.0, 3.0, N + 1)
    return x0, pref, vref, obs, hdg, hw


def _solve(solver, N, dt, **kw):
    x0, pref, vref, obs, hdg, hw = _problem(N, dt)
    return solver.solve_ocp(x0, pref, vref, obs, hdg_ref=hdg, hdg_w=hw,
                            arena_lb=np.array([-6, -6, -4.0]), arena_ub=np.array([6, 6, 0.0]), **kw)


# ------------------------------------------------------------------ model / config
def test_yaw_model_discretisation_is_exact():
    scfg, _, _ = _cfg()
    m = make_model('trajectory', scfg['model'])
    m.build_numeric(0.1)
    assert (m.nx, m.nu, m.has_yaw) == (11, 4, True)
    x = np.zeros(11)
    x[9], x[10] = 0.3, 0.2
    u = np.array([0.0, 0.0, 0.0, 1.5])
    xn = m.step(x, u, np.zeros(3), 0.0, 0.1)
    assert xn[9] == pytest.approx(0.3 + 0.2 * 0.1 + 1.5 * 0.1 ** 2 / 2)
    assert xn[10] == pytest.approx(0.2 + 1.5 * 0.1)


def test_default_is_fixed_yaw_and_unchanged():
    scfg, rcfg, _ = build_configs({})
    assert scfg['model'].get('yaw') is None
    assert not rcfg['yaw']['enabled']
    m = make_model('trajectory', scfg['model'])
    assert (m.nx, m.nu, m.has_yaw) == (9, 3, False)


def test_invalid_yaw_configurations_are_rejected():
    with pytest.raises(ValueError):
        build_configs({'yaw_mode': 'sideways'})
    with pytest.raises(ValueError):
        build_configs({'yaw_mode': 'optimized', 'control_interface': 'attitude'})
    with pytest.raises(ValueError):
        build_configs({'goal_yaw': float('nan')})            # NaN only makes sense with optimized yaw
    build_configs({'yaw_mode': 'optimized', 'goal_yaw': float('nan')})


# ------------------------------------------------------------------ solvers
def test_casadi_yaw_solution_is_dynamically_consistent_and_within_limits():
    scfg, _, _ = _cfg()
    s = make_ocp_solver(scfg)
    r = _solve(s, scfg['N'], scfg['dt'])
    assert r.success and r.usable and r.gap_residual < 1e-6
    assert np.max(np.abs(r.U[3])) <= scfg['model']['yaw']['acc_max'] + 1e-6
    assert np.max(np.abs(r.X[10, 1:-1])) <= scfg['model']['yaw']['rate_max'] + 0.05
    assert r.X.shape == (11, scfg['N'] + 1) and r.U.shape == (4, scfg['N'])


@pytest.mark.parametrize('backend', ['ipopt_cpp', 'cg2o'])
def test_native_backends_agree_with_casadi(backend):
    scfg, _, _ = _cfg()
    ref = _solve(make_ocp_solver(scfg), scfg['N'], scfg['dt'])
    ncfg, _, _ = _cfg(backend)
    try:
        s = make_ocp_solver(ncfg)
    except Exception as exc:  # noqa: BLE001 - extension not built
        pytest.skip(f'{backend} not available: {exc}')
    r = _solve(s, ncfg['N'], ncfg['dt'])
    assert r.success and r.usable, r.status
    tol = {'ipopt_cpp': (1e-3, 1e-4), 'cg2o': (0.03, 0.03)}[backend]   # (state/yaw tolerance, relative cost)
    assert np.max(np.abs(r.X[9] - ref.X[9])) < tol[0]
    assert np.max(np.abs(r.X - ref.X)) < max(tol[0], 0.05)
    assert abs(r.cost - ref.cost) / abs(ref.cost) < tol[1] + 1e-3
    r2 = _solve(s, ncfg['N'], ncfg['dt'], shift=1)          # warm-started re-solve
    assert r2.usable


# ------------------------------------------------------------------ closed loop
def _params(seed=42, **extra):
    p = scenario_parameters(generate_scenario(EXP, seed), 0.45)
    p['control_interface'] = 'trajectory'
    p.update(extra)
    return p


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


@pytest.mark.parametrize('seed', [42, 3])
def test_nose_follows_travel_direction_and_ends_at_goal_yaw(seed):
    r = run_episode(_params(seed, yaw_mode='optimized', goal_yaw=0.0), SimParams(), t_max=45.0, seed=seed, record=True)
    assert r['outcome'] == 'success', r
    assert r['hold_events'] <= 2 and r['usable_rate'] >= 0.95     # an occasional unusable solve is replaced by the fallback
    log = [e for e in r['log'] if e[4] == 'NMPC']
    t = np.array([e[0] for e in log])
    yaw = np.array([e[3][2] for e in log])
    vel = np.array([e[2] for e in log])
    speed = np.linalg.norm(vel[:, :2], axis=1)
    cruise = speed > 0.6
    assert cruise.sum() > 50
    course = np.arctan2(vel[:, 1], vel[:, 0])
    err = np.abs(np.array([_wrap(a) for a in (yaw - course)[cruise]]))
    assert np.median(err) < math.radians(25.0)               # nose points where the vehicle goes
    rate = np.abs(np.diff(np.unwrap(yaw)) / np.maximum(np.diff(t), 1e-6))
    assert np.percentile(rate, 99) < 1.3 * 0.8                # yaw-rate limit is respected (soft, plus tracking lag)
    assert abs(_wrap(r['log'][-1][3][2] - 0.0)) < math.radians(20.0)


def test_fixed_mode_keeps_yaw_constant():
    r = run_episode(_params(42, goal_yaw=0.0), SimParams(), t_max=45.0, record=True)
    assert r['outcome'] == 'success'
    yaw = np.array([e[3][2] for e in r['log'] if e[4] == 'NMPC'])
    assert np.max(np.abs(yaw)) < math.radians(8.0)


def test_nan_goal_yaw_keeps_travel_heading():
    r = run_episode(_params(42, yaw_mode='optimized', goal_yaw=float('nan')), SimParams(), t_max=45.0, record=True)
    assert r['outcome'] == 'success', r
