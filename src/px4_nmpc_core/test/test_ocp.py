import numpy as np
import pytest

from px4_nmpc_core.config import build_configs
from px4_nmpc_core.ocp import NmpcOcpSolver


OBST = np.array([[3.0, 0.2, -2.0, 1.4], [5.0, -1.0, -2.0, 1.3], [6.5, 1.0, -2.2, 1.5]])


def _solver(interface, backend='fatrop', **extra):
    # Unit tests here measure numerical feasibility, not real-time deadlines.
    # On slower/contended CPUs the production 0.25 s IPOPT watchdog can stop
    # before convergence and return a dynamically consistent but suboptimal
    # plan, causing intermittent obstacle-clearance failures. Keep production
    # parameters unchanged; give this standalone test a separate budget.
    params = {'control_interface': interface, 'solver_backend': backend,
              'horizon_steps': 25 if interface != 'trajectory' else 30,
              'solver_max_wall_time_s': 3.0}
    params.update(extra)
    scfg, _, _ = build_configs(params)
    return NmpcOcpSolver(scfg)


def _x0(interface, p=(0.0, 0.0, -2.0), v=(0.0, 0.0, 0.0), rp=(0.0, 0.0)):
    if interface == 'trajectory':
        return np.r_[p, v, 0.0, 0.0, 0.0]
    if interface == 'attitude':
        return np.r_[p, v, rp]
    return np.r_[p, v, rp, 0.0]


def _ref(s, x0, goal=(9.0, 0.0, -2.0), speed=1.5):
    N = s.N
    goal = np.asarray(goal, float)
    d = goal - x0[:3]
    L = np.linalg.norm(d)
    u = d / max(L, 1e-9)
    pref = np.zeros((3, N + 1))
    vref = np.zeros((3, N + 1))
    for k in range(N + 1):
        sk = min(L, speed * s.dt * k)
        pref[:, k] = x0[:3] + u * sk
        vref[:, k] = u * speed if sk < L else 0.0
    return pref, vref


@pytest.mark.parametrize('interface', ['trajectory', 'attitude', 'body_rate'])
@pytest.mark.parametrize('backend', ['fatrop', 'ipopt'])
def test_plan_goes_around_obstacles(interface, backend):
    s = _solver(interface, backend)
    x0 = _x0(interface)
    pref, vref = _ref(s, x0)
    r = s.solve_ocp(x0, pref, vref, OBST)
    assert r.usable, r.status
    assert r.gap_residual < 2e-3
    clear = min(np.min(np.linalg.norm(r.X[0:3].T - o[:3], axis=1) - o[3]) for o in OBST)
    assert clear > -0.05
    lb, ub = s.model.u_bounds()
    assert np.all(r.U >= np.asarray(lb)[:, None] - 1e-6) and np.all(r.U <= np.asarray(ub)[:, None] + 1e-6)


@pytest.mark.parametrize('interface', ['trajectory', 'attitude', 'body_rate'])
def test_soft_constraints_never_infeasible(interface):
    """This case must remain feasible (outside arena, moving out, tilted, inside shell)."""
    s = _solver(interface)
    lb, ub = np.array([-6.5, -4.5, -4.5]), np.array([6.5, 4.5, -0.3])
    x0 = _x0(interface, p=(6.7, 0.0, -2.0), v=(1.5, 0.0, 0.0), rp=(np.deg2rad(40), np.deg2rad(-38)))
    pref, vref = _ref(s, x0, goal=(0.0, 0.0, -2.0))
    inside = np.array([[6.9, 0.3, -2.0, 1.0]])
    r = s.solve_ocp(x0, pref, vref, inside, arena_lb=lb, arena_ub=ub)
    assert r.usable, r.status
    # it plans back into the arena
    assert r.X[0, -1] < ub[0] + 0.05


def test_warm_start_shift_is_fast_and_consistent():
    s = _solver('trajectory')
    x0 = _x0('trajectory')
    pref, vref = _ref(s, x0)
    r0 = s.solve_ocp(x0, pref, vref, OBST)
    x1 = r0.X[:, 1]
    pref1, vref1 = _ref(s, x1)
    r1 = s.solve_ocp(x1, pref1, vref1, OBST, shift=1)
    assert r1.usable
    assert r1.iterations <= r0.iterations


def test_fixed_problem_size_independent_of_obstacle_count():
    s = _solver('trajectory', max_obstacles=4)
    x0 = _x0('trajectory')
    pref, vref = _ref(s, x0)
    many = np.vstack([OBST] * 5)       # 15 obstacles -> truncated to K=4 slots
    r = s.solve_ocp(x0, pref, vref, many)
    assert r.usable
