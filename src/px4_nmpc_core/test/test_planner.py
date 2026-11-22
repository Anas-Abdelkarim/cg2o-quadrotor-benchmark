import numpy as np
from px4_nmpc_core.planner import GridAStar, sample_polyline


def test_astar_avoids_obstacle():
    planner = GridAStar(0.5, [-2, -2], [5, 2], [([2, 0, 0], 0.6)], 0.2)
    path = planner.plan([0, 0], [4, 0])
    assert path is not None and len(path) > 2
    for point in path:
        assert np.linalg.norm(np.asarray(point) - np.array([2.0, 0.0])) > 0.8 - 1e-9


def test_astar_rejects_blocked_goal():
    planner = GridAStar(0.5, [-2, -2], [5, 2], [([4, 0, 0], 0.6)], 0.2)
    assert planner.plan([0, 0], [4, 0]) is None


def test_sampler_progresses_forward_not_back_to_old_start():
    path = [np.array([0., 0., -3.]), np.array([2., 0., -3.]), np.array([4., 0., -3.])]
    samples = sample_polyline(path, np.array([3., 0., -3.]), speed=1.0, dt=0.5, count=3)
    assert samples[0, 0] >= 2.9
    assert np.all(np.diff(samples[:, 0]) >= -1e-9)


def test_speed_profile_accelerates_without_instant_speed_step():
    from px4_nmpc_core.planner import speed_profile_distances
    d = speed_profile_distances(0.0, 10.0, 1.5, 0.8, 1.0, 0.1, 6)
    increments = np.diff(d) / 0.1
    assert np.isclose(increments[0], 0.08, atol=1e-9)
    assert np.all(np.diff(increments[:4]) <= 0.08 + 1e-9)
    assert np.all(increments <= 1.5 + 1e-9)


def test_sample_polyline_distances_uses_explicit_progress():
    from px4_nmpc_core.planner import sample_polyline_distances
    path = [np.array([0., 0., 0.]), np.array([2., 0., 0.]), np.array([2., 2., 0.])]
    pts = sample_polyline_distances(path, np.zeros(3), [0.0, 1.0, 2.5, 4.0])
    assert np.allclose(pts[0], [0., 0., 0.])
    assert np.allclose(pts[1], [1., 0., 0.])
    assert np.allclose(pts[2], [2., 0.5, 0.])
    assert np.allclose(pts[3], [2., 2., 0.])
