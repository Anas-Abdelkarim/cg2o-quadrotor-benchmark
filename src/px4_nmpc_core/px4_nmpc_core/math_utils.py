import math
import numpy as np


def quat_wxyz_to_euler(q):
    """Hamilton quaternion q=(w,x,y,z), body-to-world, to roll/pitch/yaw."""
    w, x, y, z = [float(v) for v in q]
    sinr_cosp = 2.0 * (w*x + y*z)
    cosr_cosp = 1.0 - 2.0 * (x*x + y*y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w*y - z*x)
    pitch = math.copysign(math.pi/2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    siny_cosp = 2.0 * (w*z + x*y)
    cosy_cosp = 1.0 - 2.0 * (y*y + z*z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=float)


def wrap_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def ned_to_enu(p_ned):
    n, e, d = p_ned
    return np.array([e, n, -d], dtype=float)


def enu_to_ned(p_enu):
    """ENU [E,N,U] -> NED [N,E,D].  This transform is its own inverse."""
    e, n, u = p_enu
    return np.array([n, e, -u], dtype=float)


def euler_to_quat_wxyz(roll, pitch, yaw):
    """ZYX Euler angles -> Hamilton quaternion (w, x, y, z), body-to-world."""
    cr, sr = math.cos(0.5 * roll), math.sin(0.5 * roll)
    cp, sp = math.cos(0.5 * pitch), math.sin(0.5 * pitch)
    cy, sy = math.cos(0.5 * yaw), math.sin(0.5 * yaw)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dtype=float)


def quat_wxyz_to_rotation(q):
    w, x, y, z = [float(v) for v in q]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
