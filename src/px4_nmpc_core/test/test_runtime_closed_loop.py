import pytest

from px4_nmpc_core.scenario import generate_scenario, scenario_parameters
from px4_nmpc_core.sim.episode import run_episode
from px4_nmpc_core.sim.px4_like import SimParams

EXP = {
    'seed': 42, 'arena_north_m': 14.0, 'arena_east_m': 10.0, 'arena_height_m': 5.0,
    'wall_margin_m': 0.45, 'quadrant_margin_m': 1.2, 'start_quadrant': 'lower_right',
    'goal_quadrant': 'upper_left', 'start_altitude_m': 0.15, 'goal_altitude_min_m': 2.2,
    'goal_altitude_max_m': 4.0, 'obstacle_count': 10, 'obstacle_radius_mean_m': 0.65,
    'obstacle_radius_std_m': 0.15, 'obstacle_radius_min_m': 0.35, 'obstacle_radius_max_m': 1.0,
    'obstacle_min_gap_m': 0.35, 'obstacle_boundary_clearance_m': 0.25, 'start_goal_clearance_m': 1.4,
    'feasibility_resolution_m': 0.35, 'feasibility_uav_clearance_m': 0.75,
}


def _params(seed, interface):
    p = scenario_parameters(generate_scenario(EXP, seed), 0.45)
    p['control_interface'] = interface
    if interface != 'trajectory':
        p['horizon_steps'] = 25
    return p


@pytest.mark.parametrize('seed,interface', [(42, 'trajectory'), (3, 'trajectory'), (42, 'attitude'), (5, 'body_rate')])
def test_randomized_scene_reaches_goal_without_collision(seed, interface):
    r = run_episode(_params(seed, interface), SimParams(wind_acc=(0.2, -0.2, 0.0)), t_max=45.0, seed=seed)
    assert r['outcome'] == 'success', r
    assert r['min_physical_clearance'] > 0.0
    assert r['hold_events'] == 0


def test_slow_solver_latency_is_tolerated():
    """Emulate a machine 5x slower than this one (solver latency compensation + plan playback)."""
    r = run_episode(_params(42, 'trajectory'), SimParams(), latency_scale=5.0, t_max=45.0)
    assert r['outcome'] == 'success', r


def test_wrong_hover_throttle_is_absorbed_by_observer():
    p = _params(42, 'attitude')
    p['hover_throttle'] = 0.62          # real vehicle hovers at 0.729
    r = run_episode(p, SimParams(), t_max=45.0)
    assert r['outcome'] == 'success', r
    assert r['disturbance_estimate'][2] > 0.5    # real thrust lower than modelled -> downward bias learned


def test_goal_can_be_changed_after_goal_without_rearming():
    import numpy as np
    from px4_nmpc_core.config import build_configs
    from px4_nmpc_core.ocp import NmpcOcpSolver
    from px4_nmpc_core.runtime import NmpcRuntime

    p = _params(42, 'trajectory')
    solver_cfg, runtime_cfg, _ = build_configs(p)
    rt = NmpcRuntime(runtime_cfg, NmpcOcpSolver(solver_cfg))
    old_goal = runtime_cfg['goal'].copy()
    rt.on_odometry(10.0, old_goal, np.zeros(3), np.zeros(3))
    rt.phase = 'GOAL'
    rt.armed = True
    rt.offboard = True
    rt.leg_index = 1
    rt.t_nav_start = 2.0
    rt.t_goal = 9.0

    # Find a valid free point inside the arena, different from the old goal.
    candidate = None
    for n in np.linspace(rt.arena_lb[0] + 0.4, rt.arena_ub[0] - 0.4, 5):
        for e in np.linspace(rt.arena_lb[1] + 0.4, rt.arena_ub[1] - 0.4, 5):
            q = np.array([n, e, min(rt.arena_ub[2] - 0.2, -2.0)])
            ok, _ = rt._validate_goal(q)
            if ok and np.linalg.norm(q - old_goal) > 1.5:
                candidate = q
                break
        if candidate is not None:
            break
    assert candidate is not None
    ok, msg = rt.set_goal(candidate, 10.1)
    assert ok, msg
    assert rt.phase == 'NMPC'
    assert rt.leg_index == 2
    assert rt.t_goal is None
    assert np.allclose(rt.goal, candidate)
    assert np.allclose(rt.mission_start, old_goal)
    req = rt.make_request(10.2)
    assert req is not None
    assert np.allclose(rt.guidance.goal, candidate)


def test_environment_can_be_staged_at_goal_without_commanding_new_goal():
    """A scene reload must keep PX4 holding the completed leg until commit."""
    import numpy as np
    from px4_nmpc_core.config import build_configs
    from px4_nmpc_core.ocp import NmpcOcpSolver
    from px4_nmpc_core.runtime import NmpcRuntime

    p = _params(42, 'trajectory')
    solver_cfg, runtime_cfg, _ = build_configs(p)
    rt = NmpcRuntime(runtime_cfg, NmpcOcpSolver(solver_cfg))
    old_goal = runtime_cfg['goal'].copy()
    rt.on_odometry(10.0, old_goal, np.zeros(3), np.zeros(3))
    rt.phase = 'GOAL'
    rt.armed = rt.offboard = True
    rt.leg_index = 1
    rt.goal_hold_point = old_goal.copy()
    rt.t_goal = 9.0

    candidate = None
    for n in np.linspace(rt.arena_lb[0] + 0.4, rt.arena_ub[0] - 0.4, 5):
        for e in np.linspace(rt.arena_lb[1] + 0.4, rt.arena_ub[1] - 0.4, 5):
            q = np.array([n, e, min(rt.arena_ub[2] - 0.2, -2.0)])
            ok, _ = rt._validate_goal(q)
            if ok and np.linalg.norm(q - old_goal) > 1.5:
                candidate = q
                break
        if candidate is not None:
            break
    assert candidate is not None

    ok, msg = rt.set_environment(rt.obstacles_phys, rt.arena_lb, rt.arena_ub,
                                 goal=candidate, t=10.1, resume=False)
    assert ok, msg
    assert rt.phase == 'GOAL'
    assert np.allclose(rt.goal, candidate)
    assert np.allclose(rt.goal_hold_point, old_goal)
    assert np.allclose(rt._setpoint(10.2).position, old_goal)

    ok, msg = rt.set_goal(candidate, 10.3, resume=True)
    assert ok, msg
    assert rt.phase == 'NMPC'
    assert rt.leg_index == 2
    assert rt.goal_hold_point is None


def test_lifecycle_rejects_restart_and_retarget_during_landing():
    """Interactive commands must not expose a still-landing vehicle as a new mission."""
    import numpy as np
    from px4_nmpc_core.config import build_configs
    from px4_nmpc_core.ocp import NmpcOcpSolver
    from px4_nmpc_core.runtime import NmpcRuntime

    p = _params(42, 'trajectory')
    solver_cfg, runtime_cfg, _ = build_configs(p)
    rt = NmpcRuntime(runtime_cfg, NmpcOcpSolver(solver_cfg))
    rt.on_odometry(1.0, np.zeros(3), np.zeros(3), np.zeros(3))
    rt.phase = 'NMPC'
    ok, msg = rt.start(1.1)
    assert not ok and 'cannot start' in msg

    rt.armed = True
    rt.offboard = True
    rt.begin_landing(2.0)
    assert rt.phase == 'LANDING'
    ok, msg = rt.set_goal(rt.goal, 2.1)
    assert not ok and 'landing' in msg

    # PX4 disarm is the event that finally exposes the controller as IDLE.
    rt.set_vehicle_status(3.0, armed=False, offboard=False)
    assert rt.phase == 'IDLE'


def test_start_allowed_again_after_abort():
    import numpy as np
    from px4_nmpc_core.config import build_configs
    from px4_nmpc_core.ocp import NmpcOcpSolver
    from px4_nmpc_core.runtime import NmpcRuntime

    p = _params(42, 'trajectory')
    solver_cfg, runtime_cfg, _ = build_configs(p)
    rt = NmpcRuntime(runtime_cfg, NmpcOcpSolver(solver_cfg))
    rt.on_odometry(1.0, np.zeros(3), np.zeros(3), np.zeros(3))
    rt.phase = 'ABORTED'
    ok, msg = rt.start(1.1)
    assert ok, msg
    assert rt.phase == 'ARMING'


def test_brief_offboard_status_transient_does_not_abort_but_sustained_loss_does():
    """PX4 status jitter must not make a mission start and immediately stop."""
    import numpy as np
    from px4_nmpc_core.config import build_configs
    from px4_nmpc_core.ocp import NmpcOcpSolver
    from px4_nmpc_core.runtime import NmpcRuntime

    p = _params(42, 'trajectory')
    p['offboard_loss_grace_s'] = 1.0
    p['disarm_loss_grace_s'] = 0.4
    solver_cfg, runtime_cfg, _ = build_configs(p)
    rt = NmpcRuntime(runtime_cfg, NmpcOcpSolver(solver_cfg))
    rt.on_odometry(0.0, np.zeros(3), np.zeros(3), np.zeros(3))
    rt.phase = 'NMPC'
    rt.armed = True
    rt.offboard = True

    rt.set_vehicle_status(2.0, armed=True, offboard=False)
    assert rt.phase == 'NMPC'
    rt.set_vehicle_status(2.6, armed=True, offboard=False)
    assert rt.phase == 'NMPC'
    rt.set_vehicle_status(2.7, armed=True, offboard=True)
    assert rt.phase == 'NMPC'

    rt.set_vehicle_status(4.0, armed=True, offboard=False)
    rt.set_vehicle_status(5.1, armed=True, offboard=False)
    assert rt.phase == 'ABORTED'
    assert 'OFFBOARD' in rt.abort_reason
