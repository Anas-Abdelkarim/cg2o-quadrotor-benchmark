"""Tests for the cg2o (constrained g2o factor-graph) trajectory-interface
backend. See INSTALL.md for the full build/validation writeup.

These tests are deliberately honest about a real, discovered limitation:
this backend's plain-Gauss-Newton-based AL solver is only *reliably*
convergent from the exact all-zero state. As soon as a state carries both
nonzero velocity and nonzero acceleration at once (exactly what a real
closed-loop step produces) it can diverge outright -- not just converge
slowly -- regardless of warm-start strategy (a brand-new solver instance
solving the very same state cold shows the same divergence). We were not
able to root-cause this within cg2o's own linear-algebra internals in the
time available; see the cg2o installation/build notes in INSTALL.md.

So test_solve_from_rest_converges is a real regression guard for the one
case we know is solid; test_closed_loop_never_returns_garbage is a safety
guard (never crash, never report usable=True on a dynamically-inconsistent
plan) rather than a claim that closed-loop tracking actually works today.
"""
import numpy as np
import pytest

from px4_nmpc_core.config import build_configs

cg2o_backend = pytest.importorskip(
    'px4_nmpc_core.cg2o_backend',
    reason='px4_nmpc_cg2o extension not built; see INSTALL.md',
)

try:
    cg2o_backend._load_extension()
except ImportError as exc:
    pytest.skip(f'compiled cpp_px4_nmpc_cg2o extension not found: {exc}', allow_module_level=True)

from px4_nmpc_core.cg2o_backend import Cg2oOcpSolver  # noqa: E402
from px4_nmpc_core.ocp import make_ocp_solver  # noqa: E402

OBST = np.array([[3.8, 0.0, -3.0, 1.2], [6.4, 1.2, -3.0, 1.1]])


def _solver_cfg(**extra):
    params = {'solver_backend': 'cg2o', 'control_interface': 'trajectory', 'horizon_steps': 25}
    params.update(extra)
    scfg, rcfg, _ = build_configs(params)
    return scfg, rcfg


def _straight_ref(N, dt, x0, goal=(5.0, 0.0, -2.0), speed=1.0):
    goal = np.asarray(goal, float)
    d = goal - x0[:3]
    L = np.linalg.norm(d)
    u = d / max(L, 1e-9)
    pref = np.zeros((3, N + 1))
    vref = np.zeros((3, N + 1))
    for k in range(N + 1):
        sk = min(L, speed * dt * k)
        pref[:, k] = x0[:3] + u * sk
        vref[:, k] = u * speed if sk < L else 0.0
    return pref, vref


def test_unsupported_interface_falls_back_to_ipopt():
    """make_ocp_solver must catch Cg2oOcpSolver's interface check and fall
    back to ipopt with a backend_error, exactly like a missing fatrop
    plugin -- never raise out of the factory."""
    scfg, _ = _solver_cfg(control_interface='attitude', horizon_steps=15)
    solver = make_ocp_solver(scfg)
    assert solver.backend in ('ipopt', 'ipopt_casadi')
    assert 'cg2o' in solver.backend_error


def test_direct_construction_rejects_non_trajectory_interface():
    scfg, _ = _solver_cfg(control_interface='body_rate', horizon_steps=15)
    with pytest.raises(ValueError, match='trajectory'):
        Cg2oOcpSolver(scfg)


def test_make_ocp_solver_selects_cg2o_for_trajectory():
    scfg, _ = _solver_cfg()
    solver = make_ocp_solver(scfg)
    assert isinstance(solver, Cg2oOcpSolver)
    assert solver.backend == 'cg2o'
    assert not solver.backend_error
    assert solver.nz > 0 and solver.ng > 0  # controller_node.py logs these unconditionally


def test_solve_from_rest_converges():
    """The one case validated to be reliably solid (see module docstring):
    starting from the exact zero state, a cold solve should converge to a
    dynamically-consistent, low-cost plan in well under the iteration cap."""
    scfg, _ = _solver_cfg()
    s = Cg2oOcpSolver(scfg)
    x0 = np.zeros(9)
    pref, vref = _straight_ref(s.N, s.dt, x0)
    r = s.solve_ocp(x0, pref, vref, OBST, arena_lb=np.array([-7., -5., -5.]),
                    arena_ub=np.array([7., 5., 0.]), shift=None)
    assert r.usable, r.status
    assert r.success
    assert r.gap_residual < 2e-3
    assert r.iterations < scfg['max_iter']
    # Vehicle should have made real progress toward the goal, not stalled.
    assert r.X[0, -1] > 1.0


def test_reset_warm_start_is_reproducible():
    scfg, _ = _solver_cfg()
    s = Cg2oOcpSolver(scfg)
    x0 = np.zeros(9)
    pref, vref = _straight_ref(s.N, s.dt, x0)
    lb, ub = np.array([-7., -5., -5.]), np.array([7., 5., 0.])
    r1 = s.solve_ocp(x0, pref, vref, OBST, arena_lb=lb, arena_ub=ub, shift=None)
    s.reset_warm_start()
    r2 = s.solve_ocp(x0, pref, vref, OBST, arena_lb=lb, arena_ub=ub, shift=None)
    assert r1.usable and r2.usable
    assert r1.iterations == r2.iterations
    assert np.isclose(r1.cost, r2.cost)


def test_closed_loop_never_returns_garbage():
    """Safety-net regression guard, not a tracking-quality claim (see module
    docstring): drive a short closed loop from rest and require that every
    solve either (a) is flagged usable=False -- never a silently-accepted
    divergent plan -- or (b) is genuinely dynamically consistent. The
    backend must never raise or return NaN/inf out of solve_ocp."""
    scfg, _ = _solver_cfg()
    s = Cg2oOcpSolver(scfg)
    pref, vref = _straight_ref(s.N, s.dt, np.zeros(9))
    lb, ub = np.array([-7., -5., -5.]), np.array([7., 5., 0.])
    x0 = np.zeros(9)
    shift = None
    saw_usable = False
    for _ in range(6):
        r = s.solve_ocp(x0, pref, vref, OBST, arena_lb=lb, arena_ub=ub, shift=shift)
        shift = 1
        assert r.status  # never blows up uncaught
        if r.X is not None:
            assert np.all(np.isfinite(r.X)) and np.all(np.isfinite(r.U))
        if r.usable:
            saw_usable = True
            assert r.gap_residual < float(scfg['gap_tolerance'])
            x0 = s.model.step(x0, r.U[:, 0], np.zeros(3), 0.0, s.dt)
        # else: a real controller would hold its last accepted setpoint and
        # retry next cycle -- x0 (from real odometry) barely moves either way.
    assert saw_usable  # at least the very first (from-rest) solve must work
