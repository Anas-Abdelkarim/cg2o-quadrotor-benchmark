import numpy as np

from px4_nmpc_core.scenario import generate_scenario


def cfg():
    return {
        'seed': 7,
        'arena_north_m': 14.0,
        'arena_east_m': 10.0,
        'arena_height_m': 5.0,
        'wall_margin_m': 0.45,
        'quadrant_margin_m': 1.2,
        'start_quadrant': 'lower_right',
        'goal_quadrant': 'upper_left',
        'start_altitude_m': 0.15,
        'goal_altitude_min_m': 2.2,
        'goal_altitude_max_m': 4.0,
        'goal_yaw': 0.0,
        'obstacle_count': 8,
        'obstacle_radius_mean_m': 0.6,
        'obstacle_radius_std_m': 0.1,
        'obstacle_radius_min_m': 0.35,
        'obstacle_radius_max_m': 0.9,
        'obstacle_min_gap_m': 0.3,
        'obstacle_boundary_clearance_m': 0.2,
        'start_goal_clearance_m': 1.2,
        'feasibility_resolution_m': 0.35,
        'feasibility_uav_clearance_m': 0.7,
        'max_layout_attempts': 50,
        'max_candidate_attempts': 1000,
    }


def test_deterministic_seed_and_quadrants():
    a = generate_scenario(cfg(), 11)
    b = generate_scenario(cfg(), 11)
    assert a.to_dict() == b.to_dict()
    # North-up/East-right top view: start lower-right, goal upper-left.
    assert a.start_ned[0] < 0.0 and a.start_ned[1] > 0.0
    assert a.goal_ned[0] > 0.0 and a.goal_ned[1] < 0.0


def test_obstacles_separated_and_inside_arena():
    c = cfg()
    s = generate_scenario(c, 4)
    assert len(s.obstacle_radii) == c['obstacle_count']
    for i, (center, r) in enumerate(zip(s.obstacle_centers_ned, s.obstacle_radii)):
        center = np.asarray(center)
        assert abs(center[0]) + r < c['arena_north_m'] / 2.0
        assert abs(center[1]) + r < c['arena_east_m'] / 2.0
        alt = -center[2]
        assert r < alt < c['arena_height_m'] - r
        assert np.linalg.norm(center - np.asarray(s.start_ned)) > r
        assert np.linalg.norm(center - np.asarray(s.goal_ned)) > r
        for j in range(i):
            other = np.asarray(s.obstacle_centers_ned[j])
            assert np.linalg.norm(center - other) >= r + s.obstacle_radii[j] + c['obstacle_min_gap_m'] - 1e-9


def test_invalid_altitude_range_is_rejected():
    c = cfg()
    c['goal_altitude_max_m'] = c['arena_height_m']
    try:
        generate_scenario(c, 1)
    except ValueError as exc:
        assert 'goal altitude range' in str(exc)
    else:
        raise AssertionError('invalid goal altitude range was accepted')


# --- take-off corridor and seed reproducibility ---

def _point_segment_distance(point, a, b):
    point = np.asarray(point, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ab = b - a
    t = np.clip(np.dot(point - a, ab) / np.dot(ab, ab), 0.0, 1.0)
    return np.linalg.norm(point - (a + t * ab))


def test_vertical_takeoff_corridor_is_clear():
    c = dict(cfg(), takeoff_altitude_m=1.50, takeoff_corridor_clearance_m=0.75)
    for seed in range(40):
        s = generate_scenario(c, seed)
        start = np.asarray(s.start_ned, dtype=float)
        takeoff = start.copy()
        takeoff[2] = -c['takeoff_altitude_m']
        for center, radius in zip(s.obstacle_centers_ned, s.obstacle_radii):
            assert _point_segment_distance(center, start, takeoff) >= radius + 0.75 - 1e-9
        d = s.to_dict()
        assert d['schema_version'] == 5 and np.allclose(d['takeoff_ned'], takeoff)


def test_takeoff_is_handed_to_the_controller():
    from px4_nmpc_core.scenario import controller_geometry, scenario_parameters
    c = dict(cfg(), takeoff_altitude_m=1.50)
    s = generate_scenario(c, 42)
    assert np.allclose(controller_geometry(s)['takeoff_local_ned'], [0.0, 0.0, -(1.50 - 0.15)])
    assert abs(scenario_parameters(s)['takeoff_height_m'] - 1.35) < 1e-12


def test_recorded_seeds_remain_deterministic():
    """A recorded seed must remain deterministic across runs."""
    expected = {5: (-58.486334384487456, 9.654071874230056),
                39: (-39.42555858593847, 10.165312024996084),
                88: (-32.18330643921824, 10.466977625278023)}
    import pathlib
    import yaml
    exp_yaml = pathlib.Path(__file__).resolve().parents[2] / 'px4_nmpc_config/config/experiment.yaml'
    exp = yaml.safe_load(exp_yaml.read_text())['experiment']
    exp.pop('takeoff_altitude_m', None)          # test scenario without take-off corridor
    exp.pop('takeoff_corridor_clearance_m', None)
    for seed, (sum_c, sum_r) in expected.items():
        s = generate_scenario(exp, seed)
        assert abs(float(np.sum(s.obstacle_centers_ned)) - sum_c) < 1e-6
        assert abs(float(np.sum(s.obstacle_radii)) - sum_r) < 1e-6


def test_scenario_keeps_persistent_controller_origin_for_live_leg():
    from px4_nmpc_core.scenario import controller_geometry, controller_origin_ned_of
    c = cfg()
    c.pop('takeoff_altitude_m', None)
    c['start_ned_override'] = [1.5, -0.8, -2.4]
    c['goal_ned_override'] = [-2.0, 1.2, -3.1]
    c['controller_origin_ned'] = [-3.0, 2.0, -0.15]
    s = generate_scenario(c, 123)
    g = controller_geometry(s)
    origin = np.asarray(c['controller_origin_ned'], float)
    assert np.allclose(controller_origin_ned_of(s), origin)
    assert np.allclose(g['start_local_ned'], np.asarray(c['start_ned_override']) - origin)
    assert np.allclose(g['goal_local_ned'], np.asarray(c['goal_ned_override']) - origin)
