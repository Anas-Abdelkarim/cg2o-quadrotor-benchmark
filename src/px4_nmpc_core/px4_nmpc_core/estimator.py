"""Offset-free MPC helpers: disturbance observer and PX4 thrust mapping.

``ThrustMap`` converts between specific thrust and PX4 normalized throttle for
attitude/body-rate interfaces. ``DisturbanceObserver`` estimates slowly varying
model mismatch as an acceleration bias that is fed back into the prediction
model.  The default trajectory interface leaves thrust/attitude regulation to
PX4 and does not require this observer for normal operation.
"""
from __future__ import annotations

import math

import numpy as np


def thrust_axis_ned(rpy):
    phi, theta, psi = [float(v) for v in rpy]
    cph, sph = math.cos(phi), math.sin(phi)
    cth, sth = math.cos(theta), math.sin(theta)
    cps, sps = math.cos(psi), math.sin(psi)
    return np.array([cps * sth * cph + sps * sph,
                     sps * sth * cph - cps * sph,
                     cth * cph])


class ThrustMap:
    """Collective specific thrust [m/s^2] <-> PX4 normalised throttle [0, 1]."""

    def __init__(self, g=9.81, hover_throttle=0.729, model='quadratic_esc',
                 omega_min=150.0, omega_max=1000.0, throttle_min=0.05, throttle_max=1.0):
        self.g = float(g)
        self.hover = float(hover_throttle)
        self.model = str(model)
        self.w0 = float(omega_min)
        self.w1 = float(omega_max)
        self.umin = float(throttle_min)
        self.umax = float(throttle_max)
        if not 0.05 < self.hover < 0.95:
            raise ValueError('hover_throttle must be within (0.05, 0.95)')
        if self.model not in ('linear', 'quadratic_esc'):
            raise ValueError("thrust model must be 'linear' or 'quadratic_esc'")

    def to_throttle(self, thrust_accel):
        T = max(0.0, float(thrust_accel))
        if self.model == 'linear':
            u = T / self.g * self.hover
        else:
            dw = self.w1 - self.w0
            w_h = self.w0 + dw * self.hover
            w = w_h * math.sqrt(T / self.g)
            u = (w - self.w0) / dw
        return float(np.clip(u, self.umin, self.umax))

    def to_accel(self, throttle):
        u = float(np.clip(throttle, 0.0, 1.0))
        if self.model == 'linear':
            return u / self.hover * self.g
        dw = self.w1 - self.w0
        w_h = self.w0 + dw * self.hover
        w = self.w0 + dw * u
        return self.g * (w / w_h) ** 2


class DisturbanceObserver:
    """Luenberger observer for an additive NED acceleration disturbance.

    Model: dv/dt = g e3 - T * R e3 - drag * v + d,   dd/dt = 0.
    Innovation e = v_meas - v_hat drives both v_hat and d_hat (2nd-order
    error dynamics with natural frequency ``omega`` and damping ``zeta``).
    """

    def __init__(self, g=9.81, drag=(0.0, 0.0, 0.0), omega=1.5, zeta=0.9,
                 d_max=(2.5, 2.5, 4.0)):
        self.g = float(g)
        self.drag = np.asarray(drag, dtype=float).reshape(3)
        self.l1 = 2.0 * float(zeta) * float(omega)
        self.l2 = float(omega) ** 2
        self.d_max = np.asarray(d_max, dtype=float).reshape(3)
        self.reset()

    def reset(self):
        self.v_hat = None
        self.d_hat = np.zeros(3)
        self.t_last = None

    def update(self, t, v_meas, rpy, thrust_accel, active=True):
        v_meas = np.asarray(v_meas, dtype=float).reshape(3)
        if self.v_hat is None or self.t_last is None or not active:
            self.v_hat = v_meas.copy()
            self.t_last = t
            return self.d_hat
        h = float(np.clip(t - self.t_last, 0.0, 0.05))
        self.t_last = t
        if h <= 0.0:
            return self.d_hat
        acc = np.array([0.0, 0.0, self.g]) - float(thrust_accel) * thrust_axis_ned(rpy) - self.drag * v_meas
        e = v_meas - self.v_hat
        self.v_hat = self.v_hat + h * (acc + self.d_hat + self.l1 * e)
        self.d_hat = np.clip(self.d_hat + h * self.l2 * e, -self.d_max, self.d_max)
        return self.d_hat
