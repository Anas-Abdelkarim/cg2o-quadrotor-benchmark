"""A small PX4-like multicopter simulator for offline closed-loop testing.

It is *not* Gazebo, but it reproduces the parts of the PX4 + gz_x500 stack
that decide whether an offboard NMPC works:

* PX4 cascade: position P -> velocity PID (+ feed-forward) -> thrust vector
  -> attitude P -> body-rate loop (first order) -> motors (first order);
  default gains from PX4 ``mc_pos_control`` / ``mc_att_control``;
* the three offboard setpoint types (trajectory/position, attitude, rates);
* the gz_x500 ESC: rotor speed linear in throttle 150..1000 rad/s, thrust
  ~ speed^2, real hover throttle configurable (``MPC_THR_HOVER`` = 0.6);
* PX4's hover-thrust estimator (first-order convergence) in position mode;
* rotor drag, constant wind, odometry noise/latency, ground contact;
* offboard heartbeat / arming / failsafe (position hold) behaviour.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..math_utils import quat_wxyz_to_euler, quat_wxyz_to_rotation


G = 9.81


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _quat_from_bodyz_yaw(body_z, yaw):
    """PX4 ControlMath::bodyzToAttitude."""
    body_z = body_z / max(np.linalg.norm(body_z), 1e-9)
    y_c = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    body_x = np.cross(y_c, body_z)
    if body_z[2] < 0.0:
        body_x = -body_x
    if abs(body_z[2]) < 1e-6:
        body_x = np.array([0.0, 0.0, 1.0])
    body_x /= max(np.linalg.norm(body_x), 1e-9)
    body_y = np.cross(body_z, body_x)
    R = np.column_stack([body_x, body_y, body_z])
    return _rot_to_quat(R)


def _rot_to_quat(R):
    tr = np.trace(R)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.asarray(q)
    return q / np.linalg.norm(q)


@dataclass
class SimParams:
    # gz_x500: m = 2.064 kg, k_f = 8.54858e-6, ESC 150..1000 rad/s
    #   -> hover rotor speed 769.5 rad/s -> hover throttle (769.5-150)/850 = 0.729
    hover_throttle_true: float = 0.729
    hover_throttle_param: float = 0.60       # PX4 MPC_THR_HOVER initial value (estimator converges)
    esc_omega_min: float = 150.0
    esc_omega_max: float = 1000.0
    tau_motor: float = 0.02
    tau_rate: float = 0.035
    rate_max: tuple = (3.8, 3.8, 3.5)
    att_p: tuple = (6.5, 6.5, 2.8)
    pos_p: tuple = (0.95, 0.95, 1.0)
    vel_p: tuple = (1.8, 1.8, 4.0)
    vel_i: tuple = (0.4, 0.4, 2.0)
    vel_d: tuple = (0.2, 0.2, 0.0)
    tilt_max_deg: float = 45.0
    thr_min: float = 0.12
    vel_max_xy: float = 12.0
    vel_max_up: float = 3.0
    vel_max_down: float = 1.5
    hte_tau: float = 3.0                     # hover-thrust estimator convergence
    drag: tuple = (0.10, 0.10, 0.05)
    wind_acc: tuple = (0.0, 0.0, 0.0)
    pos_noise: float = 0.01
    vel_noise: float = 0.02
    att_noise_deg: float = 0.2
    odom_delay_s: float = 0.01
    ground_down: float = 0.0                 # local D at which the airframe rests on the floor
    offboard_loss_s: float = 1.0
    phys_radius: float = 0.30                # collision radius of the airframe


class Px4LikeQuad:
    def __init__(self, params: SimParams | None = None, rng=None, start=(0.0, 0.0, 0.0), yaw=0.0):
        self.p = params or SimParams()
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.pos = np.asarray(start, float).copy()
        self.vel = np.zeros(3)
        c, s = math.cos(yaw / 2), math.sin(yaw / 2)
        self.q = np.array([c, 0.0, 0.0, s])
        self.w = np.zeros(3)
        self.T = 0.0                     # specific thrust [m/s^2]
        self.vel_int = np.zeros(3)
        self.vel_prev = np.zeros(3)
        self.hover_est = self.p.hover_throttle_param
        self.on_ground = True
        self.armed = False
        self.offboard = False
        self.failsafe = False
        self.sp = None
        self.sp_t = -1e9
        self.heartbeat_since = None
        self.pending_cmd_t = None
        self.hold_sp = None
        self.t = 0.0
        self.odom_buf = []
        w_h = self.p.esc_omega_min + (self.p.esc_omega_max - self.p.esc_omega_min) * self.p.hover_throttle_true
        self._w_hover = w_h

    # --- commands --------------------------------------------------------
    def request_offboard_and_arm(self):
        self.pending_cmd_t = self.t

    def disarm(self):
        self.armed = False
        self.offboard = False

    def set_setpoint(self, sp):
        if sp is None:
            return
        if self.heartbeat_since is None or self.t - self.sp_t > 0.5:
            self.heartbeat_since = self.t
        self.sp = sp
        self.sp_t = self.t

    # --- PX4 controllers -------------------------------------------------
    def _thrust_from_throttle(self, u):
        u = float(np.clip(u, 0.0, 1.0))
        w = self.p.esc_omega_min + (self.p.esc_omega_max - self.p.esc_omega_min) * u
        return G * (w / self._w_hover) ** 2 if u > 0.0 else 0.0

    def _position_control(self, sp, dt, yaw_sp):
        pos_sp = np.asarray(sp.position if sp.position is not None else np.full(3, np.nan), float)
        vel_ff = np.asarray(sp.velocity if sp.velocity is not None else np.full(3, np.nan), float)
        acc_ff = np.asarray(sp.acceleration if sp.acceleration is not None else np.full(3, np.nan), float)
        vel_sp = np.where(np.isnan(vel_ff), 0.0, vel_ff)
        pos_term = np.where(np.isnan(pos_sp), 0.0, (pos_sp - self.pos) * np.asarray(self.p.pos_p))
        vel_sp = vel_sp + pos_term
        xy = vel_sp[:2]
        n = np.linalg.norm(xy)
        if n > self.p.vel_max_xy:
            vel_sp[:2] = xy * self.p.vel_max_xy / n
        vel_sp[2] = float(np.clip(vel_sp[2], -self.p.vel_max_up, self.p.vel_max_down))
        vel_err = vel_sp - self.vel
        vel_dot = (self.vel - self.vel_prev) / max(dt, 1e-4)
        self.vel_prev = self.vel.copy()
        acc = vel_err * np.asarray(self.p.vel_p) + self.vel_int - vel_dot * np.asarray(self.p.vel_d)
        acc = acc + np.where(np.isnan(acc_ff), 0.0, acc_ff)
        # thrust vector (PX4 _accelerationControl)
        body_z = np.array([-acc[0], -acc[1], G])
        body_z /= np.linalg.norm(body_z)
        tilt_max = math.radians(self.p.tilt_max_deg)
        cos_t = body_z[2]
        if cos_t < math.cos(tilt_max):
            xy = body_z[:2]
            xy_n = np.linalg.norm(xy)
            body_z = np.r_[xy / max(xy_n, 1e-9) * math.sin(tilt_max), math.cos(tilt_max)]
        h = self.hover_est
        collective = acc[2] * (h / G) - h
        collective /= body_z[2]
        collective = min(collective, -self.p.thr_min)
        thr = max(-1.0, collective)
        if not self.on_ground or thr < -h * 0.5:
            # anti-windup: integrate only when not saturated
            self.vel_int += vel_err * np.asarray(self.p.vel_i) * dt
            self.vel_int = np.clip(self.vel_int, -3.0, 3.0)
        q_d = _quat_from_bodyz_yaw(body_z, yaw_sp)
        return q_d, -thr

    def _attitude_control(self, q_d):
        qe = _qmul(_qconj(self.q), q_d)
        if qe[0] < 0:
            qe = -qe
        return np.clip(2.0 * qe[1:] * np.asarray(self.p.att_p), -np.asarray(self.p.rate_max), np.asarray(self.p.rate_max))

    # --- simulation ------------------------------------------------------
    def step(self, dt):
        p = self.p
        self.t += dt
        # arming / offboard acceptance (PX4 needs a live setpoint stream)
        if self.pending_cmd_t is not None and self.t - self.pending_cmd_t > 0.3:
            if self.heartbeat_since is not None and self.t - self.heartbeat_since >= 1.0 and self.t - self.sp_t < 0.5:
                self.armed = True
                self.offboard = True
                self.pending_cmd_t = None
        lost = (self.t - self.sp_t) > p.offboard_loss_s
        if self.offboard and lost:
            self.failsafe = True
            self.offboard = False
            self.hold_sp = self.pos.copy()

        yaw = quat_wxyz_to_euler(self.q)[2]
        throttle = 0.0
        rate_sp = np.zeros(3)
        if self.armed:
            sp = self.sp
            if self.failsafe or sp is None:
                from types import SimpleNamespace
                sp = SimpleNamespace(kind='position', position=self.hold_sp if self.hold_sp is not None else self.pos,
                                     velocity=None, acceleration=None, yaw=yaw)
            kind = sp.kind
            if kind in ('position', 'trajectory'):
                yaw_sp = sp.yaw if (sp.yaw is not None and np.isfinite(sp.yaw)) else yaw
                q_d, throttle = self._position_control(sp, dt, yaw_sp)
                rate_sp = self._attitude_control(q_d)
                ys = getattr(sp, 'yawspeed', float('nan'))
                if ys is not None and np.isfinite(ys):
                    # PX4 adds the yaw-rate feed-forward of the setpoint to the attitude controller's rate demand
                    rate_sp = rate_sp + quat_wxyz_to_rotation(self.q).T @ np.array([0.0, 0.0, float(ys)])
                if not self.on_ground:
                    self.hover_est += (p.hover_throttle_true - self.hover_est) * dt / p.hte_tau
            elif kind == 'attitude':
                rate_sp = self._attitude_control(np.asarray(sp.q_d, float))
                throttle = float(sp.thrust_normalized)
                self.vel_int[:] = 0.0
            elif kind == 'body_rate':
                rate_sp = np.clip(np.asarray(sp.rates, float), -np.asarray(p.rate_max), np.asarray(p.rate_max))
                throttle = float(sp.thrust_normalized)
                self.vel_int[:] = 0.0
        T_cmd = self._thrust_from_throttle(throttle) if self.armed else 0.0
        self.T += (T_cmd - self.T) * min(1.0, dt / p.tau_motor)
        self.w += (rate_sp - self.w) * min(1.0, dt / p.tau_rate)

        R = quat_wxyz_to_rotation(self.q)
        acc = np.array([0.0, 0.0, G]) - self.T * R[:, 2] - np.asarray(p.drag) * self.vel + np.asarray(p.wind_acc)
        if self.on_ground:
            if acc[2] < 0.0:
                self.on_ground = False
            else:
                acc[:] = 0.0
                self.vel[:] = 0.0
                self.w[:] = 0.0
        self.vel = self.vel + acc * dt
        self.pos = self.pos + self.vel * dt
        if self.pos[2] >= p.ground_down and not self.on_ground:
            self.pos[2] = p.ground_down
            if self.vel[2] > 0.0:
                self.vel[:] = 0.0
            if self.T < G * 0.9:
                self.on_ground = True
                yaw_now = quat_wxyz_to_euler(self.q)[2]
                self.q = np.array([math.cos(yaw_now / 2), 0.0, 0.0, math.sin(yaw_now / 2)])
        dq = _qmul(self.q, np.r_[0.0, self.w])
        self.q = self.q + 0.5 * dq * dt
        self.q /= np.linalg.norm(self.q)

        # delayed, noisy odometry
        rpy = quat_wxyz_to_euler(self.q)
        meas = (self.t,
                self.pos + self.rng.normal(0, p.pos_noise, 3),
                self.vel + self.rng.normal(0, p.vel_noise, 3),
                rpy + self.rng.normal(0, math.radians(p.att_noise_deg), 3),
                self.w.copy())          # body rates (FRD), like VehicleOdometry.angular_velocity
        self.odom_buf.append(meas)

    def odometry(self):
        """Most recent odometry sample older than the configured delay."""
        cutoff = self.t - self.p.odom_delay_s
        out = None
        while self.odom_buf and self.odom_buf[0][0] <= cutoff + 1e-12:
            out = self.odom_buf.pop(0)
        if out is not None:
            self._last_odom = out
        return getattr(self, '_last_odom', None)

    def rpy(self):
        return quat_wxyz_to_euler(self.q)
