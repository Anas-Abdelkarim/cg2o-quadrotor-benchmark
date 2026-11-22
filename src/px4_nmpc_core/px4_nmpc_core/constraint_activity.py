"""Backend-independent inequality-activity diagnostics for NMPC solutions.

The diagnostics are evaluated *after* a solve from the returned primal
trajectory.  They therefore use the same definitions for CasADi/IPOPT,
native C++ IPOPT and cg2o and do not depend on solver-specific multipliers.

For a soft inequality the reported residual is the amount of room left after
its shared slack has been applied.  A residual close to zero means the
constraint is active; a negative residual means the returned point violates
the softened inequality beyond the configured numerical tolerance.

Only stages 1..N-1 are counted for obstacle/arena/dynamic soft constraints,
matching the OCP implementations.  Stage 0 is fixed and its slacks are pinned
to zero; the terminal state uses an exterior penalty instead of those stage
inequalities.  Hard input bounds (jerk and, for optimized yaw, yaw
acceleration) are counted at all N input stages.
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np


OBSTACLE_EPS = 0.05  # must match ipopt/solver.py and cg2o TrajParams


def _summary(values, tol: float) -> Dict[str, float | int]:
    a = np.asarray(values, dtype=float).reshape(-1)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {
            'count': 0,
            'active': 0,
            'fraction': float('nan'),
            'violated': 0,
            'min_residual': float('nan'),
        }
    # Residual convention: feasible means residual >= 0.  Count the
    # numerical boundary band as active; larger negative values are violations
    # and are intentionally not double-counted as active.
    active = int(np.count_nonzero((a >= -tol) & (a <= tol)))
    violated = int(np.count_nonzero(a < -tol))
    return {
        'count': int(a.size),
        'active': active,
        'fraction': active / float(a.size),
        'violated': violated,
        'min_residual': float(np.min(a)),
    }


def _input_bound_residuals(U: np.ndarray, model) -> np.ndarray:
    """Return lower/upper room for every hard input box inequality."""
    lb, ub = model.u_bounds()
    lb = np.asarray(lb, dtype=float).reshape(-1, 1)
    ub = np.asarray(ub, dtype=float).reshape(-1, 1)
    if U.shape[0] != lb.shape[0]:
        return np.empty(0)
    return np.r_[U - lb, ub - U].reshape(-1)


def evaluate_constraint_activity(result, request, runtime, tolerance: float = 1.0e-3,
                                 input_tolerance: float | None = None) -> Dict[str, float | int | bool]:
    """Evaluate active inequalities for one returned trajectory.

    Parameters
    ----------
    result:
        Solver ``SolveResult`` containing ``X``, ``U`` and ``S``.
    request:
        The exact ``SolveRequest`` used for that solve.  In particular its
        obstacle array contains the *effective* radii selected for the OCP.
    runtime:
        ``NmpcRuntime`` that created the request.  It supplies arena bounds and
        the model limits.
    tolerance:
        Activity tolerance for soft inequalities, in the units of each
        normalized residual.  ``1e-3`` is the default experiment diagnostic;
        the raw margins are also reported so results do not depend solely on
        this threshold.
    input_tolerance:
        Activity tolerance for hard input bounds.  Defaults to ``tolerance``.

    Returns
    -------
    Flat dictionary suitable for direct insertion into ``runs.csv``.
    """
    out: Dict[str, float | int | bool] = {'constraint_activity_available': False}
    if result is None or getattr(result, 'X', None) is None or getattr(result, 'U', None) is None:
        return out

    X = np.asarray(result.X, dtype=float)
    U = np.asarray(result.U, dtype=float)
    S = np.asarray(getattr(result, 'S', np.empty((0, 0))), dtype=float)
    if X.ndim != 2 or U.ndim != 2 or X.shape[1] != U.shape[1] + 1:
        return out
    if S.shape != (3, U.shape[1]):
        return out

    model = runtime.model
    if getattr(model, 'interface', None) != 'trajectory':
        # The present research comparison uses the trajectory formulation.
        # Do not silently invent activity definitions for attitude/body-rate.
        return out

    tol = max(0.0, float(tolerance))
    itol = tol if input_tolerance is None else max(0.0, float(input_tolerance))
    N = U.shape[1]
    stage_ids = range(1, N)  # exact OCP stage set for soft inequalities

    obstacle_residuals = []
    obstacle_raw_margins = []
    arena_residuals = []
    velocity_residuals = []
    tilt_residuals = []
    vertical_acc_residuals = []
    yaw_rate_residuals = []

    obs = np.asarray(getattr(request, 'obstacles', np.empty((0, 4))), dtype=float).reshape(-1, 4)
    x0_req = getattr(request, 'x0', None)
    if x0_req is not None and len(obs):
        # unused (padded) obstacle slots are not constraints of interest; same set as before stable slots
        from .guidance import is_padding_row
        obs = np.asarray([o for o in obs if not is_padding_row(o, np.asarray(x0_req, float)[0:3])]).reshape(-1, 4)
    arena_lb = getattr(runtime, 'arena_lb', None)
    arena_ub = getattr(runtime, 'arena_ub', None)
    arena_lb = None if arena_lb is None else np.asarray(arena_lb, dtype=float).reshape(3)
    arena_ub = None if arena_ub is None else np.asarray(arena_ub, dtype=float).reshape(3)

    vmax = float(model.vmax)
    g = float(model.g)
    tilt_max = float(model.tilt_max)
    tan2 = math.tan(tilt_max) ** 2
    acc_down = float(model.acc_down_max)
    acc_up = float(model.acc_up_max)
    has_yaw = bool(getattr(model, 'has_yaw', False))
    yaw_rate_max = float(getattr(model, 'yaw_rate_max', np.nan)) if has_yaw else float('nan')

    for k in stage_ids:
        x = X[:, k]
        s_obs, s_arena, s_dyn = (float(v) for v in S[:, k])
        p, v, a = x[0:3], x[3:6], x[6:9]

        for row in obs:
            c, radius = row[0:3], float(row[3])
            h = math.sqrt(float(np.dot(p - c, p - c)) + OBSTACLE_EPS ** 2) - radius
            obstacle_raw_margins.append(h)
            obstacle_residuals.append(h + s_obs)

        if arena_lb is not None and arena_ub is not None:
            # Same h+s >= 0 convention as the CasADi OCP.
            arena_residuals.extend((p - arena_lb + s_arena).tolist())
            arena_residuals.extend((arena_ub - p + s_arena).tolist())

        e_vel = (float(np.dot(v, v)) - vmax * vmax) / (2.0 * vmax)
        e_tilt = (float(np.dot(a[0:2], a[0:2])) - tan2 * (g - float(a[2])) ** 2) / (2.0 * g)
        e_down = float(a[2]) - acc_down
        e_up = -float(a[2]) - acc_up
        velocity_residuals.append(s_dyn - e_vel)
        tilt_residuals.append(s_dyn - e_tilt)
        vertical_acc_residuals.extend((s_dyn - e_down, s_dyn - e_up))

        if has_yaw and X.shape[0] >= 11 and math.isfinite(yaw_rate_max) and yaw_rate_max > 0.0:
            omega = float(x[10])
            e_yaw_rate = (omega * omega - yaw_rate_max * yaw_rate_max) / (2.0 * yaw_rate_max)
            yaw_rate_residuals.append(s_dyn - e_yaw_rate)

    input_residuals = _input_bound_residuals(U, model)

    cats = {
        'obstacle': _summary(obstacle_residuals, tol),
        'arena': _summary(arena_residuals, tol),
        'velocity': _summary(velocity_residuals, tol),
        'tilt': _summary(tilt_residuals, tol),
        'vertical_acc': _summary(vertical_acc_residuals, tol),
        'yaw_rate': _summary(yaw_rate_residuals, tol),
        'input_bound': _summary(input_residuals, itol),
    }

    total_count = sum(int(v['count']) for v in cats.values())
    total_active = sum(int(v['active']) for v in cats.values())
    total_violated = sum(int(v['violated']) for v in cats.values())

    out = {
        'constraint_activity_available': True,
        'active_constraint_tolerance': tol,
        'active_input_tolerance': itol,
        'active_constraints_total': total_active,
        'constraint_count_total': total_count,
        'active_constraint_fraction': total_active / float(total_count) if total_count else float('nan'),
        'violated_constraints_total': total_violated,
        'active_obstacle_constraints': cats['obstacle']['active'],
        'obstacle_constraint_count': cats['obstacle']['count'],
        'active_obstacle_fraction': cats['obstacle']['fraction'],
        'violated_obstacle_constraints': cats['obstacle']['violated'],
        'active_arena_constraints': cats['arena']['active'],
        'active_velocity_constraints': cats['velocity']['active'],
        'active_tilt_constraints': cats['tilt']['active'],
        'active_vertical_acc_constraints': cats['vertical_acc']['active'],
        'active_yaw_rate_constraints': cats['yaw_rate']['active'],
        'active_input_bound_constraints': cats['input_bound']['active'],
        'min_obstacle_margin_m': (float(np.min(obstacle_raw_margins))
                                  if obstacle_raw_margins else float('nan')),
        'min_obstacle_soft_residual_m': cats['obstacle']['min_residual'],
        'min_arena_soft_residual_m': cats['arena']['min_residual'],
        'min_velocity_soft_residual': cats['velocity']['min_residual'],
        'min_tilt_soft_residual': cats['tilt']['min_residual'],
        'min_vertical_acc_soft_residual': cats['vertical_acc']['min_residual'],
        'min_yaw_rate_soft_residual': cats['yaw_rate']['min_residual'],
        'min_input_bound_residual': cats['input_bound']['min_residual'],
    }
    return out
