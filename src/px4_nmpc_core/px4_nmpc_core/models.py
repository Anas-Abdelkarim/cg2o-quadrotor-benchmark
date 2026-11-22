"""Prediction models for the three PX4 offboard control interfaces.

All models share the same leading state layout ``x[0:3] = p_NED`` and
``x[3:6] = v_NED`` so that the OCP builder can attach obstacle, arena, speed
and reference-tracking terms without knowing the model.

Interfaces
----------
``trajectory``  (default, most robust)
    Point-mass/triple-integrator planner.  State ``[p, v, a]``, input jerk.
    The NMPC plans a smooth collision-free trajectory; PX4's own position
    controller tracks ``position/velocity/acceleration`` from
    ``TrajectorySetpoint`` at its full rate.  PX4 therefore owns thrust
    mapping, hover-thrust estimation, tilt limiting and attitude stabilisation.

``trajectory`` with ``yaw_mode: optimized``
    The same planner plus the heading: state ``[p, v, a, psi, omega]``, input
    ``[jerk, alpha]`` (yaw acceleration).  The heading is optimised together
    with the path (see :class:`PointMassJerkYawModel`) and sent to PX4 as
    ``TrajectorySetpoint.yaw`` + ``yawspeed``.

``attitude``
    Translational dynamics + first-order closed-loop roll/pitch response of
    PX4's attitude controller (Kamel et al., 2017).  State ``[p, v, roll,
    pitch]``, input ``[roll_cmd, pitch_cmd, thrust_accel]``.  Yaw is held at
    the reference yaw by PX4.  Sent as ``VehicleAttitudeSetpoint``.

``body_rate``
    Nine-state model ``[p, v, roll, pitch, yaw]`` with input
    ``[thrust_accel, p, q, r]``.  Sent as ``VehicleRatesSetpoint``.  This is the
    most demanding interface: it needs solve times well below ``dt``.

The attitude/body-rate models include an additive acceleration disturbance
``d`` (offset-free MPC, Pannocchia & Rawlings 2003) that is estimated online by
:class:`px4_nmpc_core.estimator.DisturbanceObserver`, plus linear drag.
"""
from __future__ import annotations

import math

import casadi as ca
import numpy as np


INTERFACES = ('trajectory', 'attitude', 'body_rate')


def _rk4(f, x, u, h, substeps=1):
    hs = h / float(substeps)
    for _ in range(substeps):
        k1 = f(x, u)
        k2 = f(x + 0.5 * hs * k1, u)
        k3 = f(x + 0.5 * hs * k2, u)
        k4 = f(x + hs * k3, u)
        x = x + hs * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return x


def _thrust_axis_ned(phi, theta, psi, lib=ca):
    """Third column of R = Rz(psi) Ry(theta) Rx(phi) (body FRD -> NED)."""
    cph, sph = lib.cos(phi), lib.sin(phi)
    cth, sth = lib.cos(theta), lib.sin(theta)
    cps, sps = lib.cos(psi), lib.sin(psi)
    return (cps * sth * cph + sps * sph,
            sps * sth * cph - cps * sph,
            cth * cph)


class ModelBase:
    """Common interface used by :class:`px4_nmpc_core.ocp.NmpcOcpSolver`."""

    name = 'base'
    interface = 'none'
    nx = 0
    nu = 0
    state_names = ()
    input_names = ()

    def __init__(self, cfg):
        self.g = float(cfg.get('g', 9.81))
        self.vmax = float(cfg['vmax'])
        self.tilt_max = float(cfg['tilt_max'])
        self.drag = np.asarray(cfg.get('drag', [0.0, 0.0, 0.0]), dtype=float).reshape(3)
        self.T_min = float(cfg.get('T_min', 2.0))
        self.T_max = float(cfg.get('T_max', 17.0))

    # --- symbolic pieces -------------------------------------------------
    def discrete(self, x, u, d, yaw_ref, dt):
        raise NotImplementedError

    def u_bounds(self):
        raise NotImplementedError

    def u_ref(self, d):
        """Nominal (hover) input, symbolic in the disturbance estimate."""
        raise NotImplementedError

    def soft_path_constraints(self, x, u):
        """Expressions e_i that must satisfy ``e_i <= s_dyn`` (normalised)."""
        raise NotImplementedError

    def extra_stage_cost(self, x, u, d, yaw_ref, w):
        return 0.0

    has_yaw = False

    def heading_cost(self, x, hdg_ref, hdg_w):
        """State cost of the optimised heading (only models with ``has_yaw``).

        ``hdg_ref`` / ``hdg_w``: reference heading [rad] and its weight at this stage.
        """
        return 0.0

    def input_weight(self, w):
        raise NotImplementedError

    # --- numeric helpers -------------------------------------------------
    def build_numeric(self, dt):
        x = ca.SX.sym('x', self.nx)
        u = ca.SX.sym('u', self.nu)
        d = ca.SX.sym('d', 3)
        yaw = ca.SX.sym('yaw')
        h = ca.SX.sym('h')
        self.F_num = ca.Function('F_num', [x, u, d, yaw, h], [self.discrete(x, u, d, yaw, h)])
        self.dt = float(dt)

    def step(self, x, u, d, yaw_ref, h):
        return np.asarray(self.F_num(x, u, d, yaw_ref, h), dtype=float).reshape(-1)

    def state_from_measurement(self, pos, vel, rpy, accel=None):
        raise NotImplementedError

    def hover_input(self, d):
        return np.asarray(ca.DM(self.u_ref(ca.DM(np.asarray(d, dtype=float)))), dtype=float).reshape(-1)

    def guess_input(self, x, p_ref, v_ref, d, yaw_ref):
        """Simple stabilising feedback used to roll out a dynamically
        consistent cold-start guess (zero shooting gaps from the start)."""
        return self.hover_input(d)

    def _desired_acc(self, x, p_ref, v_ref, kp=1.2, kd=2.0, amax=4.0):
        a = kp * (np.asarray(p_ref) - x[0:3]) + kd * (np.asarray(v_ref) - x[3:6])
        n = float(np.linalg.norm(a))
        return a * (amax / n) if n > amax else a

    def _thrust_attitude(self, a_des, d, yaw):
        """Collective thrust and roll/pitch producing a_des (NED) at yaw."""
        f = np.asarray(a_des, float) - np.array([0.0, 0.0, self.g]) - np.asarray(d, float) + self.drag * 0.0
        T = float(np.linalg.norm(f))
        if T < 1e-6:
            return self.g, 0.0, 0.0
        z = -f / T                        # body z axis (down) in NED
        c, s = math.cos(yaw), math.sin(yaw)
        xp = c * z[0] + s * z[1]
        yp = -s * z[0] + c * z[1]
        phi = math.asin(float(np.clip(-yp, -1.0, 1.0)))
        theta = math.atan2(xp, z[2])
        m = self.tilt_max
        return float(np.clip(T, self.T_min, self.T_max)), float(np.clip(phi, -m, m)), float(np.clip(theta, -m, m))


class PointMassJerkModel(ModelBase):
    name = 'point_mass_jerk'
    interface = 'trajectory'
    nx = 9
    nu = 3
    state_names = ('pN', 'pE', 'pD', 'vN', 'vE', 'vD', 'aN', 'aE', 'aD')
    input_names = ('jN', 'jE', 'jD')

    def __init__(self, cfg):
        super().__init__(cfg)
        self.jerk_max = np.asarray(cfg.get('jerk_max', [12.0, 12.0, 8.0]), dtype=float).reshape(3)
        self.acc_up_max = float(cfg.get('acc_up_max', 4.0))       # upward (negative D)
        self.acc_down_max = float(cfg.get('acc_down_max', 3.0))   # downward (positive D)

    def discrete(self, x, u, d, yaw_ref, dt):
        # Exact discretisation of the triple integrator (no approximation).
        p, v, a = x[0:3], x[3:6], x[6:9]
        return ca.vertcat(
            p + v * dt + a * dt ** 2 / 2.0 + u * dt ** 3 / 6.0,
            v + a * dt + u * dt ** 2 / 2.0,
            a + u * dt,
        )

    def u_bounds(self):
        return (-self.jerk_max).tolist(), self.jerk_max.tolist()

    def u_ref(self, d):
        return ca.DM.zeros(3) if isinstance(d, ca.DM) else ca.SX.zeros(3)

    def soft_path_constraints(self, x, u):
        v, a = x[3:6], x[6:9]
        g, tan2 = self.g, math.tan(self.tilt_max) ** 2
        return [
            (ca.sumsqr(v) - self.vmax ** 2) / (2.0 * self.vmax),
            # Thrust direction tilt cone: |a_xy| <= tan(tilt) * (g - a_D).
            (ca.sumsqr(a[0:2]) - tan2 * (g - a[2]) ** 2) / (2.0 * g),
            a[2] - self.acc_down_max,
            -a[2] - self.acc_up_max,
        ]

    def extra_stage_cost(self, x, u, d, yaw_ref, w):
        a = x[6:9]
        return float(w.get('acc', 0.3)) * ca.sumsqr(a)

    def input_weight(self, w):
        return np.full(3, float(w.get('jerk', 0.02)))

    def guess_input(self, x, p_ref, v_ref, d, yaw_ref):
        a_des = self._desired_acc(x, p_ref, v_ref)
        j = (a_des - x[6:9]) / 0.3
        return np.clip(j, -self.jerk_max, self.jerk_max)

    def state_from_measurement(self, pos, vel, rpy, accel=None):
        a = np.zeros(3) if accel is None else np.asarray(accel, dtype=float)
        return np.r_[np.asarray(pos, float), np.asarray(vel, float), a]

    def clip_state(self, x):
        """Keep a (non-measured) acceleration seed inside the admissible set."""
        x = np.asarray(x, dtype=float).copy()
        amax_xy = math.tan(self.tilt_max) * self.g * 0.9
        axy = x[6:8]
        n = float(np.linalg.norm(axy))
        if n > amax_xy:
            x[6:8] = axy * (amax_xy / n)
        x[8] = float(np.clip(x[8], -0.9 * self.acc_up_max, 0.9 * self.acc_down_max))
        return x


class PointMassJerkYawModel(PointMassJerkModel):
    """Triple-integrator planner plus an optimised heading.

    State ``[p(3), v(3), a(3), psi, omega]``, input ``[jerk(3), alpha]``:
    ``psi`` is the (unwrapped) yaw angle, ``omega`` the yaw rate and ``alpha`` the yaw
    acceleration, a double integrator that is discretised exactly like the translation::

        psi+   = psi + omega*dt + alpha*dt^2/2          omega+ = omega + alpha*dt

    PX4's position controller tracks ``yaw`` and uses ``yawspeed`` as a feed-forward, so
    this is the natural command to send.  Using the yaw *acceleration* as the input makes
    the planned heading smooth (C^1) and lets the rate and acceleration limits be
    expressed directly.  An unwrapped angle is used on purpose: the references handed to
    the solver are unwrapped relative to the current yaw, so no angle wrapping (and no
    singularity at +-pi) ever appears inside the NLP.

    Heading cost per stage (``hw_k`` / ``psi_ref_k`` are solver parameters, see
    :func:`px4_nmpc_core.runtime.NmpcRuntime._heading_reference`)::

        hw_k (psi_k - psi_ref_k)^2                       follow the travel / goal heading
      + w_lat (v_N sin psi_k - v_E cos psi_k)^2          no sideways flight (couples psi and v)
      + w_rate omega_k^2 + w_acc alpha_k^2               smoothness

    The second term is the squared sideways velocity in the body frame: it makes the
    optimiser turn the nose along the *planned* (obstacle-avoiding) velocity instead of
    only along the reference, and it vanishes by itself when the vehicle hovers.
    ``omega`` is limited softly through the shared dynamic-limit slack, ``alpha`` by a box.
    """
    name = 'point_mass_jerk_yaw'
    has_yaw = True
    nx = 11
    nu = 4
    state_names = PointMassJerkModel.state_names + ('psi', 'omega')
    input_names = PointMassJerkModel.input_names + ('alpha',)

    def __init__(self, cfg):
        super().__init__(cfg)
        y = dict(cfg.get('yaw', {}))
        self.yaw_rate_max = float(y.get('rate_max', 0.8))
        self.yaw_acc_max = float(y.get('acc_max', 2.0))
        self.w_lat = float(y.get('w_lateral', 1.0))
        self.w_rate = float(y.get('w_rate', 0.05))
        self.w_alpha = float(y.get('w_acc', 0.05))
        if self.yaw_rate_max <= 0.0 or self.yaw_acc_max <= 0.0:
            raise ValueError('yaw rate and acceleration limits must be positive')

    def discrete(self, x, u, d, yaw_ref, dt):
        base = super().discrete(x[0:9], u[0:3], d, yaw_ref, dt)
        psi, om, al = x[9], x[10], u[3]
        return ca.vertcat(base, psi + om * dt + al * dt ** 2 / 2.0, om + al * dt)

    def u_bounds(self):
        lb, ub = super().u_bounds()
        return list(lb) + [-self.yaw_acc_max], list(ub) + [self.yaw_acc_max]

    def u_ref(self, d):
        return ca.DM.zeros(4) if isinstance(d, ca.DM) else ca.SX.zeros(4)

    def soft_path_constraints(self, x, u):
        out = super().soft_path_constraints(x, u)
        out.append((x[10] ** 2 - self.yaw_rate_max ** 2) / (2.0 * self.yaw_rate_max))
        return out

    def extra_stage_cost(self, x, u, d, yaw_ref, w):
        return super().extra_stage_cost(x, u, d, yaw_ref, w)

    def heading_cost(self, x, hdg_ref, hdg_w):
        psi, om = x[9], x[10]
        lat = x[3] * ca.sin(psi) - x[4] * ca.cos(psi)
        return hdg_w * (psi - hdg_ref) ** 2 + self.w_lat * lat ** 2 + self.w_rate * om ** 2

    def input_weight(self, w):
        return np.r_[super().input_weight(w), self.w_alpha]

    def guess_input(self, x, p_ref, v_ref, d, yaw_ref):
        # `yaw_ref` is the reference heading of the target stage: critically damped PD law.
        j = super().guess_input(x[0:9], p_ref, v_ref, d, yaw_ref)
        al = 6.0 * (float(yaw_ref) - x[9]) - 3.5 * x[10]
        return np.r_[j, np.clip(al, -self.yaw_acc_max, self.yaw_acc_max)]

    def state_from_measurement(self, pos, vel, rpy, accel=None, yaw_rate=0.0):
        base = super().state_from_measurement(pos, vel, rpy, accel)
        return np.r_[base, float(np.asarray(rpy, float).reshape(3)[2]), float(yaw_rate)]

    def clip_state(self, x):
        x = np.asarray(x, dtype=float).copy()
        x[0:9] = super().clip_state(x[0:9])
        x[10] = float(np.clip(x[10], -0.9 * self.yaw_rate_max, 0.9 * self.yaw_rate_max))
        return x


class AttitudeModel(ModelBase):
    name = 'attitude_first_order'
    interface = 'attitude'
    nx = 8
    nu = 3
    state_names = ('pN', 'pE', 'pD', 'vN', 'vE', 'vD', 'roll', 'pitch')
    input_names = ('roll_cmd', 'pitch_cmd', 'thrust_accel')

    def __init__(self, cfg):
        super().__init__(cfg)
        self.tau_roll = float(cfg.get('tau_roll', 0.18))
        self.tau_pitch = float(cfg.get('tau_pitch', 0.18))
        self.k_roll = float(cfg.get('k_roll', 1.0))
        self.k_pitch = float(cfg.get('k_pitch', 1.0))
        self.tilt_cmd_max = float(cfg.get('tilt_cmd_max', self.tilt_max))
        self.rk4_substeps = int(cfg.get('rk4_substeps', 2))

    def _f(self, d, yaw):
        def f(x, u):
            v, phi, theta = x[3:6], x[6], x[7]
            phi_c, theta_c, T = u[0], u[1], u[2]
            ax, ay, az = _thrust_axis_ned(phi, theta, yaw)
            acc = ca.vertcat(-T * ax, -T * ay, self.g - T * az) - ca.DM(self.drag) * v + d
            return ca.vertcat(v, acc,
                              (self.k_roll * phi_c - phi) / self.tau_roll,
                              (self.k_pitch * theta_c - theta) / self.tau_pitch)
        return f

    def discrete(self, x, u, d, yaw_ref, dt):
        return _rk4(self._f(d, yaw_ref), x, u, dt, self.rk4_substeps)

    def u_bounds(self):
        m = self.tilt_cmd_max
        return [-m, -m, self.T_min], [m, m, self.T_max]

    def u_ref(self, d):
        return ca.vertcat(0.0, 0.0, self.g + d[2])

    def soft_path_constraints(self, x, u):
        phi, theta = x[6], x[7]
        m = self.tilt_max
        return [
            (ca.sumsqr(x[3:6]) - self.vmax ** 2) / (2.0 * self.vmax),
            phi - m, -phi - m, theta - m, -theta - m,
        ]

    def extra_stage_cost(self, x, u, d, yaw_ref, w):
        return float(w.get('tilt', 1.0)) * (x[6] ** 2 + x[7] ** 2)

    def input_weight(self, w):
        return np.asarray(w.get('attitude_input', [2.0, 2.0, 0.05]), dtype=float)

    def guess_input(self, x, p_ref, v_ref, d, yaw_ref):
        T, phi, theta = self._thrust_attitude(self._desired_acc(x, p_ref, v_ref), d, yaw_ref)
        m = self.tilt_cmd_max
        return np.array([np.clip(phi, -m, m), np.clip(theta, -m, m), T])

    def state_from_measurement(self, pos, vel, rpy, accel=None):
        return np.r_[np.asarray(pos, float), np.asarray(vel, float), float(rpy[0]), float(rpy[1])]


class BodyRateModel(ModelBase):
    name = 'body_rate_euler'
    interface = 'body_rate'
    nx = 9
    nu = 4
    state_names = ('pN', 'pE', 'pD', 'vN', 'vE', 'vD', 'roll', 'pitch', 'yaw')
    input_names = ('thrust_accel', 'p', 'q', 'r')

    def __init__(self, cfg):
        super().__init__(cfg)
        self.rate_max = np.asarray(cfg.get('rate_max', [2.5, 2.5, 1.5]), dtype=float).reshape(3)
        self.rk4_substeps = int(cfg.get('rk4_substeps', 2))

    def _f(self, d):
        def f(x, u):
            v = x[3:6]
            phi, theta, psi = x[6], x[7], x[8]
            T, p, q, r = u[0], u[1], u[2], u[3]
            ax, ay, az = _thrust_axis_ned(phi, theta, psi)
            acc = ca.vertcat(-T * ax, -T * ay, self.g - T * az) - ca.DM(self.drag) * v + d
            sph, cph = ca.sin(phi), ca.cos(phi)
            tth, cth = ca.tan(theta), ca.cos(theta)
            return ca.vertcat(v, acc,
                              p + sph * tth * q + cph * tth * r,
                              cph * q - sph * r,
                              (sph * q + cph * r) / cth)
        return f

    def discrete(self, x, u, d, yaw_ref, dt):
        return _rk4(self._f(d), x, u, dt, self.rk4_substeps)

    def u_bounds(self):
        r = self.rate_max
        return [self.T_min, -r[0], -r[1], -r[2]], [self.T_max, r[0], r[1], r[2]]

    def u_ref(self, d):
        return ca.vertcat(self.g + d[2], 0.0, 0.0, 0.0)

    def soft_path_constraints(self, x, u):
        phi, theta = x[6], x[7]
        m = self.tilt_max
        return [
            (ca.sumsqr(x[3:6]) - self.vmax ** 2) / (2.0 * self.vmax),
            phi - m, -phi - m, theta - m, -theta - m,
        ]

    def extra_stage_cost(self, x, u, d, yaw_ref, w):
        dyaw = x[8] - yaw_ref
        yaw_err = ca.atan2(ca.sin(dyaw), ca.cos(dyaw))
        return (float(w.get('tilt', 1.0)) * (x[6] ** 2 + x[7] ** 2)
                + float(w.get('yaw', 2.0)) * yaw_err ** 2)

    def input_weight(self, w):
        return np.asarray(w.get('rate_input', [0.05, 0.3, 0.3, 0.3]), dtype=float)

    def guess_input(self, x, p_ref, v_ref, d, yaw_ref):
        T, phi_d, theta_d = self._thrust_attitude(self._desired_acc(x, p_ref, v_ref), d, x[8])
        dyaw = math.atan2(math.sin(yaw_ref - x[8]), math.cos(yaw_ref - x[8]))
        rates = np.array([4.0 * (phi_d - x[6]), 4.0 * (theta_d - x[7]), 1.5 * dyaw])
        return np.r_[T, np.clip(rates, -self.rate_max, self.rate_max)]

    def state_from_measurement(self, pos, vel, rpy, accel=None):
        return np.r_[np.asarray(pos, float), np.asarray(vel, float), np.asarray(rpy, float)]


def make_model(interface, cfg):
    if interface == 'trajectory':
        if dict(cfg.get('yaw', {})).get('enabled', False):
            return PointMassJerkYawModel(cfg)
        return PointMassJerkModel(cfg)
    if interface == 'attitude':
        return AttitudeModel(cfg)
    if interface == 'body_rate':
        return BodyRateModel(cfg)
    raise ValueError(f'unknown control interface {interface!r}; expected one of {INTERFACES}')
