import numpy as np

from px4_nmpc_core.guidance import GuidanceReference, effective_obstacles, select_obstacles, safe_takeoff_height
from px4_nmpc_core.planner import GridAStar


def test_plan_robust_from_blocked_start_and_shortcut_is_clear():
    obs = [(np.array([2.0, 0.0, 0.0]), 0.6), (np.array([4.0, 1.0, 0.0]), 0.5)]
    p = GridAStar(0.25, [-2, -3], [7, 3], obs, 0.7)
    assert p.plan([1.0, 0.05], [6.0, 0.0]) is None           # blocked-start semantics
    path = p.plan_robust([1.0, 0.05], [6.0, 0.0])
    assert path is not None
    assert np.allclose(path[0], [1.0, 0.05]) and np.allclose(path[-1], [6.0, 0.0])
    for a, b in zip(path[1:-1], path[2:]):
        assert p.segment_free(a, b)


def _guidance(**kw):
    cfg = dict(goal=[8.0, 0.0, -2.0], N=20, dt=0.1, desired_speed=1.5, reference_accel_max=1.0,
               reference_decel_max=0.8, planner_enabled=True, planner_resolution=0.25,
               planner_xy_min=[-2, -4], planner_xy_max=[10, 4], planner_inflation=0.75,
               obstacles=[(np.array([4.0, 0.0, -2.0]), 0.7)])
    cfg.update(kw)
    return GuidanceReference(cfg)


def test_reference_velocity_is_bounded_with_lateral_offset():
    g = _guidance()
    p0 = np.array([0.0, 0.8, -2.0])   # 0.8 m off the guide
    g.update(0.0, np.array([0.0, 0.0, -2.0]))
    pref, vref, rem = g.horizon(p0, np.zeros(3))
    assert np.max(np.linalg.norm(vref, axis=0)) <= 1.5 + 1e-6    # reference must remain speed-limited
    assert rem > 7.0


def test_reference_reaches_goal_and_stops():
    g = _guidance()
    p0 = np.array([7.9, 0.0, -2.0])
    g.update(0.0, p0)
    pref, vref, rem = g.horizon(p0, np.zeros(3))
    assert np.allclose(pref[:, -1], [8.0, 0.0, -2.0], atol=1e-6)
    assert np.allclose(vref[:, -1], 0.0)


def test_select_obstacles_keeps_nearest():
    eff = effective_obstacles([(np.array([10.0 * i, 0, 0]), 0.5) for i in range(10)], 0.35, 0.35)
    sel = select_obstacles(eff, np.zeros(3), np.zeros((1, 3)), 3)
    assert sel.shape == (3, 4)
    assert set(np.round(sel[:, 0]).astype(int)) == {0, 10, 20}


def test_safe_takeoff_height_stops_below_overhead_obstacle():
    eff = effective_obstacles([(np.array([0.0, 0.0, -1.9]), 0.4)], 0.35, 0.35)
    h = safe_takeoff_height(np.zeros(3), 1.5, eff)
    assert 0.0 < h < 0.8


def test_fixed_line_reference_does_not_move_with_vehicle():
    g = _guidance(planner_enabled=False, mode='fixed_line', obstacles=[])
    start = np.array([0.0, 0.0, -2.0])
    g.update(0.0, start)
    assert np.allclose(g.path3[0], start)

    # Move the vehicle away from the centerline.  The guide must remain the
    # original start-goal line rather than following the vehicle laterally.
    off_line = np.array([1.0, 1.2, -1.4])
    g.update(1.0, off_line)
    assert np.allclose(g.path3[0], start)
    assert np.allclose(g.path3[-1], [8.0, 0.0, -2.0])

    pref, _, _ = g.horizon(off_line, np.zeros(3))
    # The first reference sample is the projection on the fixed line.  It is
    # therefore on E=0, D=-2 rather than at the deviated vehicle position.
    assert abs(pref[1, 0]) < 1e-9
    assert abs(pref[2, 0] + 2.0) < 1e-9
    assert not np.allclose(pref[:, 0], off_line)


def test_direct_mode_keeps_legacy_planner_disabled_behavior():
    g = _guidance(planner_enabled=False, mode='direct', obstacles=[])
    p0 = np.array([0.0, 0.7, -2.0])
    g.update(0.0, p0)
    assert np.allclose(g.path3[0], p0)
