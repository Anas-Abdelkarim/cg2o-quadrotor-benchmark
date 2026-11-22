import numpy as np
from px4_nmpc_core.math_utils import quat_wxyz_to_euler, ned_to_enu, wrap_pi


def test_identity_quaternion():
    assert np.allclose(quat_wxyz_to_euler([1, 0, 0, 0]), [0, 0, 0])


def test_ned_to_enu():
    assert np.allclose(ned_to_enu([1, 2, -3]), [2, 1, 3])


def test_wrap_pi():
    assert np.isclose(wrap_pi(3*np.pi), -np.pi)
