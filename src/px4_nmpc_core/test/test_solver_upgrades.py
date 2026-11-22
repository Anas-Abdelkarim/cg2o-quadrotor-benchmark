"""Regression tests of the v3.5 solver upgrades.

* stable obstacle slots (warm-started multipliers must belong to the same obstacle),
* the common scorer (one objective for every backend) and the cg2o cost at the RETURNED point,
* cg2o ISPD: linear-solver selection, warm-start strategies (primal shift, final / middle iterate), shift by n stages,
* ipopt_cpp: primal-dual warm start and IPOPT option pass-through.

Native-backend tests are skipped when the extension is not built.
"""
import numpy as np
import pytest

from px4_nmpc_core.config import build_configs
from px4_nmpc_core.guidance import StableObstacleSlots, is_padding_row, select_obstacles
from px4_nmpc_core.ocp import make_ocp_solver
from px4_nmpc_core.scoring import CommonScorer

OBST = np.array([[3.8, 0.0, -3.0, 1.2], [6.4, 1.2, -3.0, 1.1]])
LB, UB = np.array([-7., -5., -5.]), np.array([7., 5., 0.])


def _cfg(backend, **extra):
    params = {'solver_backend': backend, 'control_interface': 'trajectory', 'horizon_steps': 25,
              'solver_allow_fallback': False, 'solver_max_wall_time_s': 5.0}
    params.update(extra)
    return build_configs(params)[0]


def _ref(N, dt, x0, goal=(5.0, 0.0, -2.0), speed=1.0):
    goal = np.asarray(goal, float)
    d = goal - x0[:3]
    L = np.linalg.norm(d)
    u = d / max(L, 1e-9)
    pref, vref = np.zeros((3, N + 1)), np.zeros((3, N + 1))
    for k in range(N + 1):
        sk = min(L, speed * dt * k)
        pref[:, k] = x0[:3] + u * sk
        vref[:, k] = u * speed if sk < L else 0.0
    return pref, vref


def _native(backend, **extra):
    try:
        return make_ocp_solver(_cfg(backend, **extra))
    except (ImportError, FileNotFoundError) as exc:
        pytest.skip(f'{backend} extension not available: {exc}')


# ------------------------------------------------------------------ stable slots
def test_stable_slots_keep_obstacle_identity():
    eff = np.array([[1.0, 0, 0, 0.5], [2.0, 0, 0, 0.5], [3.0, 0, 0, 0.5], [9.0, 0, 0, 0.5]])
    slots = StableObstacleSlots(3)
    ref = np.zeros((1, 3))
    a = slots.assign(eff, np.array([0.0, 0, 0]), ref)
    first = {tuple(r) for r in a}
    assert first == {tuple(e) for e in eff[:3]}
    # the vehicle moves on: obstacle 0 drops out, obstacle 3 enters; 1 and 2 keep their slots
    b = slots.assign(eff, np.array([7.0, 0, 0]), np.array([[8.0, 0, 0]]))
    for j in range(3):
        if tuple(a[j]) in {tuple(eff[1]), tuple(eff[2])}:
            assert np.allclose(a[j], b[j])
    assert any(np.allclose(r, eff[3]) for r in b)
    # same selection as the classic (sorted) helper
    assert {tuple(r) for r in b} == {tuple(r) for r in select_obstacles(eff, [7.0, 0, 0], [[8.0, 0, 0]], 3)}


def test_stable_slots_pad_unused_slots():
    slots = StableObstacleSlots(4)
    out = slots.assign(OBST, np.zeros(3), np.zeros((1, 3)))
    assert out.shape == (4, 4)
    assert sum(is_padding_row(r, np.zeros(3)) for r in out) == 2


# ------------------------------------------------------------------ scorer
def test_common_scorer_equals_casadi_objective():
    cfg = _cfg('ipopt_casadi')
    s = make_ocp_solver(cfg)
    x0 = np.zeros(9)
    pref, vref = _ref(s.N, s.dt, x0)
    r = s.solve_ocp(x0, pref, vref, OBST, arena_lb=LB, arena_ub=UB, shift=None)
    assert r.usable
    sc = CommonScorer(cfg).score(r.X, r.U, r.S, x0, pref, vref, OBST, arena_lb=LB, arena_ub=UB)
    assert abs(sc['cost'] - r.cost) <= 1e-8 * max(1.0, abs(r.cost))
    assert sc['eq_residual'] < 1e-6 and sc['ineq_violation'] < 1e-4


def test_cg2o_reports_the_cost_of_the_returned_point():
    """lastCost() re-evaluates every cost edge (chi2() alone is one Newton step behind)."""
    s = _native('cg2o')
    x0 = np.zeros(9)
    pref, vref = _ref(s.N, s.dt, x0)
    r = s.solve_ocp(x0, pref, vref, OBST, arena_lb=LB, arena_ub=UB, shift=None)
    assert r.usable
    sc = CommonScorer(s.cfg).score(r.X, r.U, r.S, x0, pref, vref, OBST, arena_lb=LB, arena_ub=UB)
    assert abs(sc['cost'] - r.cost) <= 1e-9 * max(1.0, abs(r.cost))


# ------------------------------------------------------------------ cg2o
def test_cg2o_linear_solvers_give_the_same_step():
    x0 = np.zeros(9)
    out = {}
    for ls in ('umfpack', 'qdldlt'):
        s = _native('cg2o', cg2o_overrides={'linear_solver': {'type': ls}})
        pref, vref = _ref(s.N, s.dt, x0)
        out[ls] = s.solve_ocp(x0, pref, vref, OBST, arena_lb=LB, arena_ub=UB, shift=None)
    assert out['umfpack'].iterations == out['qdldlt'].iterations
    assert np.isclose(out['umfpack'].cost, out['qdldlt'].cost, rtol=1e-8)


def test_cg2o_ispd_is_used_and_unknown_optimizer_is_rejected():
    s = _native('cg2o')
    assert s.cg2o_optimizer == 'ispd' and s._cpp.optimizer == 'ispd'
    with pytest.raises(ValueError, match='optimizer'):
        make_ocp_solver(_cfg('cg2o', cg2o_overrides={'optimizer': 'newton'}))


@pytest.mark.parametrize('strategy', [0, 1, 2])
@pytest.mark.parametrize('shift', [1, 2])
def test_cg2o_warm_start_strategies(strategy, shift):
    s = _native('cg2o', cg2o_overrides={'linear_solver': {'type': 'qdldlt'}, 'warm_start': {'strategy': strategy}})
    x0 = np.zeros(9)
    pref, vref = _ref(s.N, s.dt, x0)
    cold = s.solve_ocp(x0, pref, vref, OBST, arena_lb=LB, arena_ub=UB, shift=None)
    assert cold.usable and cold.success
    # next OCP: the vehicle followed the plan for `shift` stages
    x1 = cold.X[:, shift]
    pref1, vref1 = _ref(s.N, s.dt, x1)
    warm = s.solve_ocp(x1, pref1, vref1, OBST, arena_lb=LB, arena_ub=UB, shift=shift)
    assert warm.usable and warm.success, warm.status
    assert np.allclose(warm.X[:, 0], x1)                     # fixed x0 survives cg2o's own restore + shift
    if strategy == 1:
        assert warm.extra['warm_used'] == 1
        assert warm.iterations <= cold.iterations


# ------------------------------------------------------------------ ipopt_cpp
def test_ipopt_cpp_primal_dual_warm_start():
    s = _native('ipopt_cpp', ipopt_cpp_overrides={'warm_start': 'primal_dual'})
    x0 = np.zeros(9)
    pref, vref = _ref(s.N, s.dt, x0)
    cold = s.solve_ocp(x0, pref, vref, OBST, arena_lb=LB, arena_ub=UB, shift=None)
    assert cold.usable and not cold.extra['warm_duals']
    x1 = cold.X[:, 1]
    pref1, vref1 = _ref(s.N, s.dt, x1)
    warm = s.solve_ocp(x1, pref1, vref1, OBST, arena_lb=LB, arena_ub=UB, shift=1)
    assert warm.usable and warm.extra['warm_duals'] and warm.extra['warm_primal']


def test_ipopt_cpp_option_pass_through():
    s = _native('ipopt_cpp', ipopt_cpp_overrides={'options': {'min_refinement_steps': 0, 'mehrotra_algorithm': False}})
    x0 = np.zeros(9)
    pref, vref = _ref(s.N, s.dt, x0)
    assert s.solve_ocp(x0, pref, vref, OBST, arena_lb=LB, arena_ub=UB, shift=None).usable
    with pytest.raises(Exception, match='invalid IPOPT'):
        make_ocp_solver(_cfg('ipopt_cpp', ipopt_cpp_overrides={'options': {'no_such_option': 1}}))
