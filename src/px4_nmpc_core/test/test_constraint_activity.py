from types import SimpleNamespace

import numpy as np

from px4_nmpc_core.constraint_activity import evaluate_constraint_activity
from px4_nmpc_core.models import PointMassJerkModel, PointMassJerkYawModel


def _model(yaw=False):
    cfg = {
        'g': 9.81,
        'vmax': 2.0,
        'tilt_max': np.deg2rad(20.0),
        'drag': [0.0, 0.0, 0.0],
        'T_min': 2.0,
        'T_max': 17.0,
        'jerk_max': [6.0, 6.0, 4.0],
        'acc_up_max': 3.0,
        'acc_down_max': 2.5,
    }
    if yaw:
        cfg['yaw'] = {'enabled': True, 'rate_max': 0.5, 'acc_max': 1.0,
                      'w_lateral': 2.0, 'w_rate': 0.05, 'w_acc': 0.05}
        return PointMassJerkYawModel(cfg)
    return PointMassJerkModel(cfg)


def test_obstacle_and_velocity_activity_detected():
    model = _model(False)
    N = 3
    X = np.zeros((9, N + 1))
    U = np.zeros((3, N))
    S = np.zeros((3, N))
    # Stages 1 and 2 lie approximately on a radius-1 effective shell.
    X[0, 1:3] = 1.0
    # Stage 1 is also exactly on the speed limit.
    X[3, 1] = 2.0
    result = SimpleNamespace(X=X, U=U, S=S)
    request = SimpleNamespace(obstacles=np.array([[0.0, 0.0, 0.0, 1.0]]))
    runtime = SimpleNamespace(model=model, arena_lb=None, arena_ub=None)

    a = evaluate_constraint_activity(result, request, runtime, tolerance=2.0e-3)
    assert a['constraint_activity_available']
    assert a['active_obstacle_constraints'] == 2
    assert a['active_velocity_constraints'] >= 1
    assert a['violated_constraints_total'] == 0


def test_optimized_yaw_rate_and_yaw_acc_bound_activity_detected():
    model = _model(True)
    N = 3
    X = np.zeros((11, N + 1))
    U = np.zeros((4, N))
    S = np.zeros((3, N))
    X[10, 1] = 0.5       # yaw-rate soft boundary
    U[3, 0] = 1.0        # yaw-acc hard upper bound
    result = SimpleNamespace(X=X, U=U, S=S)
    request = SimpleNamespace(obstacles=np.empty((0, 4)))
    runtime = SimpleNamespace(model=model, arena_lb=None, arena_ub=None)

    a = evaluate_constraint_activity(result, request, runtime, tolerance=1.0e-6)
    assert a['active_yaw_rate_constraints'] >= 1
    assert a['active_input_bound_constraints'] >= 1
