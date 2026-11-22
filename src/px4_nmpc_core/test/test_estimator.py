import numpy as np

from px4_nmpc_core.estimator import DisturbanceObserver, ThrustMap, thrust_axis_ned


def test_thrust_map_round_trip_and_hover():
    for model in ('linear', 'quadratic_esc'):
        m = ThrustMap(hover_throttle=0.729, model=model)
        assert abs(m.to_throttle(9.81) - 0.729) < 1e-9
        for T in (5.0, 9.81, 12.0):
            assert abs(m.to_accel(m.to_throttle(T)) - T) < 1e-6


def test_gz_x500_hover_throttle_model_is_physically_distinct_from_half_throttle():
    """For the configured gz_x500 thrust map, half throttle is well below hover thrust."""
    real = ThrustMap(hover_throttle=0.729, model='quadratic_esc')
    produced = real.to_accel(0.5) / 9.81
    assert 0.5 < produced < 0.6


def test_observer_converges_to_constant_disturbance():
    obs = DisturbanceObserver(omega=2.0, zeta=0.9)
    d_true = np.array([0.4, -0.3, 1.2])
    v = np.zeros(3)
    rpy = np.zeros(3)
    T = 9.81
    h = 0.01
    for k in range(1000):
        acc = np.array([0, 0, 9.81]) - T * thrust_axis_ned(rpy) + d_true
        v = v + h * acc
        obs.update(k * h, v, rpy, T, active=True)
    assert np.allclose(obs.d_hat, d_true, atol=0.05)
