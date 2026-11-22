"""Checks of the challenging nonlinear MPC (benchmark/model)."""
import math

import numpy as np
import pytest

C = pytest.importorskip('px4_nmpc_compare.challenging')
try:
    EXT = C.extension()
except ImportError:  # pragma: no cover - native module not built
    pytest.skip('build/challenging_mpc not built', allow_module_level=True)


def params(**overrides):
    return C.make_params(C.load_course(), overrides)


def problem(p, pos=(-8.5, 0.1, -3.3), vel=(2., 0., 0.), rpy=(.05, -.1, .02)):
    x0 = C.measured_state(p, np.array(pos), np.array(vel), np.array(rpy))
    return C.make_problem(p, x0, [p.hover(), 0., 0., 0., 2.])


def test_rotation_and_thrust_direction_match_ned_frd():
    p = params()
    R = EXT.rotation(.1, -.2, .3)
    assert np.allclose(R @ R.T, np.eye(3))
    x = np.r_[0, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.]
    # Level hover thrust balances gravity; pitching nose down accelerates North.
    assert np.allclose(EXT.rhs(x, np.r_[p.hover(), 0, 0, 0, 0], p)[3:6], 0, atol=1e-12)
    x[7] = -.2
    assert EXT.rhs(x, np.r_[p.hover(), 0, 0, 0, 0], p)[3] > 0


def test_tilt_cone_row_is_exact():
    p = params(tilt_max_deg=40.)
    b = problem(p)
    z = b.cold_start()
    X = b.states(z)
    X[6, 1], X[7, 1] = math.radians(30.), math.radians(30.)   # tilt acos(cos30 cos30) = 41.4 deg
    U = b.inputs(z)
    r = b.evaluate(b.pack(X, U, b.slacks(z)))
    assert r['final_max_ineq_violation'] > 0


def test_derivatives_match_finite_differences():
    p = params()
    b = problem(p)
    z = b.cold_start()
    for point in (z, z + .05 * np.random.default_rng(3).standard_normal(z.size)):
        a = b.audit(point)
        assert a['max_jacobian_fd_error'] < 1e-5
        assert a['max_relative_hessian_fd_error'] < 1e-6


def test_problem_hash_covers_data_and_cold_start_is_deterministic():
    p = params()
    b1, b2 = problem(p), problem(p)
    assert C.problem_sha256(b1) == C.problem_sha256(b2)
    assert C.array_sha256(b1.cold_start()) == C.array_sha256(b2.cold_start())
    b3 = problem(p, pos=(-8.5, 0.1001, -3.3))
    assert C.problem_sha256(b3) != C.problem_sha256(b1)
    p2 = params(margin=.2)
    assert C.problem_sha256(problem(p2)) != C.problem_sha256(b1)


def test_cold_start_is_infeasible_on_the_guide_line():
    p = params()
    b = problem(p)
    r = b.evaluate(b.cold_start())
    assert r['obstacle_rows_inside_reserve'] > 0
    assert r['max_reserve_penetration_m'] > .5


@pytest.mark.parametrize('pos,vel,rpy', [((-14, 0, -2.5), (0, 0, 0), (0, 0, 0)),
                                         ((-8.5, .1, -3.3), (2, 0, 0), (.05, -.1, .02))])
def test_cg2o_and_ipopt_reach_the_same_solution(pos, vel, rpy):
    p = params()
    b = problem(p, pos, vel, rpy)
    z = b.cold_start()
    cg = EXT.CGSession()
    cg.configure(C.cg2o_settings())
    r1 = cg.solve(b, z, 250, 1e-4)
    ip = EXT.IPOPTSession()
    ip.set_options(dict(hessian_approximation='exact'))
    r2 = ip.solve(b, z, 250, 1e-4)
    for r in (r1, r2):
        assert r['converged']
        assert r['final_max_eq_residual'] < 1e-3 and r['final_max_ineq_violation'] < 1e-3
        # the obstacle rows are softened by the stage slack (L2 penalty)
        assert r['min_body_clearance_m'] > p.margin - r['max_slack'] * 2 - 1e-3
    # The NLP is nonconvex: the two methods may end in different local optima
    # (e.g. another side of an obstacle), so their costs are compared in the
    # study, not asserted equal here.


def test_warm_start_shift_runs():
    p = params()
    b = problem(p)
    cg = EXT.CGSession()
    settings = C.cg2o_settings()
    # Detector off: the shifted primal-dual iterate itself must be used. With
    # the repository default (reset_mode 2) any changed row resets all duals,
    # which cg2o reports as a cold start.
    settings['warm_start']['reset_tolerance'] = 0.0
    cg.configure(settings)
    r = cg.solve(b, b.cold_start(), 250, 1e-4)
    U = b.inputs(r['z'])
    xs = EXT.step(b.x0, U[:, 0], p, p.dt)
    b2 = C.make_problem(p, C.measured_state(p, xs[:3], xs[3:6], xs[6:9], xs[9:12]), U[:, 0])
    r2 = cg.solve(b2, b2.cold_start(), 250, 1e-4, warm=True, shift=1)
    # The warm point is the shifted cg2o iterate (method 1 = PreviousFinal).
    # Whether it saves iterations is a study result (it depends on the
    # change-detector settings), not an invariant.
    assert r2['converged'] and r2['warm_start_used'] == 1


def test_rate_states_follow_setpoint_with_identified_lag():
    p = params()
    x = np.zeros(13)
    x[2] = -2.
    u = np.r_[p.hover(), 1., 0., 0., 0.]
    tau = float(np.asarray(p.tau_rate)[0])
    y = EXT.step(x, u, p, tau)
    assert y[9] == pytest.approx(1. - math.exp(-1.), rel=2e-2)   # one time constant
    assert y[6] > 0.   # roll angle integrates the actual rate
