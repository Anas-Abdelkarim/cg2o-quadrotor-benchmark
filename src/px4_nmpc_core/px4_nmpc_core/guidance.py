"""Global A* guidance, horizon reference generation, and obstacle selection.

The nonlinear MPC is a local optimizer.  A coarse collision-free A* route
provides the global homotopy choice, while the NMPC refines it subject to
vehicle dynamics and constraints.  The guide is replanned from the current
position, shortcut for smoothness, speed-limited, and blended toward the goal
altitude over the available path length.
"""
from __future__ import annotations


import numpy as np

from .planner import (
    GridAStar,
    polyline_remaining_length,
    polyline_tangent,
    sample_polyline_distances,
    speed_profile_distances,
)


def effective_obstacles(obstacles, uav_radius, safety_margin):
    """[(center, r_phys)] -> array (M, 4) [N, E, D, r_eff]."""
    out = np.zeros((len(obstacles), 4))
    for i, (c, r) in enumerate(obstacles):
        out[i, 0:3] = np.asarray(c, dtype=float).reshape(3)
        out[i, 3] = float(r) + float(uav_radius) + float(safety_margin)
    return out


def select_obstacles(eff_obstacles, position, ref_points, K, horizon_reach=None):
    """Choose the K obstacles closest to the vehicle and its reference.

    Distances are surface distances (``|p-c| - r_eff``).  The fixed-size
    selection keeps the NLP dimension constant (see ``ocp.py``).
    """
    eff = np.asarray(eff_obstacles, dtype=float).reshape(-1, 4)
    if len(eff) == 0:
        return eff
    pts = np.vstack([np.asarray(position, float).reshape(1, 3), np.asarray(ref_points, float).reshape(-1, 3)])
    d = np.linalg.norm(pts[:, None, :] - eff[None, :, 0:3], axis=2) - eff[None, :, 3]
    score = d.min(axis=0)
    order = np.argsort(score)
    if horizon_reach is not None:
        order = [i for i in order if score[i] <= horizon_reach] or list(order[:1])
    return eff[np.asarray(order[:K], dtype=int)]


PAD_OFFSET_M = 1.0e3   # padded (unused) obstacle slot: centre this far from the vehicle, radius PAD_RADIUS_M
PAD_RADIUS_M = 0.1      # (same padding as the three solver backends)


def is_padding_row(row, position):
    """True for an unused obstacle slot written by :class:`StableObstacleSlots` / the solvers."""
    row = np.asarray(row, float).reshape(4)
    return bool(np.linalg.norm(row[0:3] - np.asarray(position, float).reshape(3)) > 0.5 * PAD_OFFSET_M)


class StableObstacleSlots:
    """Fixed-size obstacle slots that keep each obstacle in the same OCP slot.

    ``select_obstacles`` returns the K nearest obstacles sorted by distance, so the
    obstacle in slot ``j`` can change from one NMPC solve to the next.  Primal warm
    starts do not care, but warm-started *multipliers* (cg2o ISPD strategies 1/2,
    IPOPT primal-dual warm start) belong to a constraint row: they must refer to the
    same obstacle in consecutive solves.  This assigner selects the same K obstacles
    as ``select_obstacles`` and

    * keeps an obstacle in its slot while it stays selected,
    * puts newly selected obstacles into freed slots (nearest first),
    * pads unused slots with a far-away dummy (as the solvers do).
    """

    def __init__(self, K):
        self.K = int(K)
        self.slot_ids = [None] * self.K

    def reset(self):
        self.slot_ids = [None] * self.K

    def assign(self, eff_obstacles, position, ref_points):
        eff = np.asarray(eff_obstacles, dtype=float).reshape(-1, 4)
        position = np.asarray(position, float).reshape(3)
        out = np.zeros((self.K, 4))
        out[:, 0:3] = position + PAD_OFFSET_M
        out[:, 3] = PAD_RADIUS_M
        if len(eff) == 0:
            self.reset()
            return out
        pts = np.vstack([position.reshape(1, 3), np.asarray(ref_points, float).reshape(-1, 3)])
        d = np.linalg.norm(pts[:, None, :] - eff[None, :, 0:3], axis=2) - eff[None, :, 3]
        order = [int(i) for i in np.argsort(d.min(axis=0))[:self.K]]   # same choice as select_obstacles
        chosen = set(order)
        self.slot_ids = [i if (i is not None and i in chosen) else None for i in self.slot_ids]
        kept = {i for i in self.slot_ids if i is not None}
        free = [j for j, i in enumerate(self.slot_ids) if i is None]
        for i in order:
            if i not in kept:
                self.slot_ids[free.pop(0)] = i
        for j, i in enumerate(self.slot_ids):
            if i is not None:
                out[j] = eff[i]
        return out


def min_clearance(eff_obstacles, p):
    eff = np.asarray(eff_obstacles, dtype=float).reshape(-1, 4)
    if len(eff) == 0:
        return float('inf')
    return float(np.min(np.linalg.norm(eff[:, 0:3] - np.asarray(p, float).reshape(1, 3), axis=1) - eff[:, 3]))


class GuidanceReference:
    """Builds the position/velocity reference over the NMPC horizon."""

    def __init__(self, cfg):
        self.goal = np.asarray(cfg['goal'], dtype=float).reshape(3)
        self.N = int(cfg['N'])
        self.dt = float(cfg['dt'])
        self.desired_speed = float(cfg['desired_speed'])
        self.accel = float(cfg['reference_accel_max'])
        self.decel = float(cfg['reference_decel_max'])
        requested_mode = str(cfg.get('mode', 'auto')).strip().lower()
        planner_enabled = bool(cfg.get('planner_enabled', True))
        if requested_mode == 'auto':
            requested_mode = 'planner' if planner_enabled else 'direct'
        if requested_mode not in ('planner', 'direct', 'fixed_line'):
            raise ValueError('guidance mode must be auto, planner, direct or fixed_line')
        self.mode = requested_mode
        # Keep planner_enabled as a public compatibility attribute used by older
        # diagnostics/configuration code.  The explicit mode is authoritative.
        self.planner_enabled = self.mode == 'planner'
        self.replan_period = float(cfg.get('planner_replan_period_s', 1.0))
        self.climb_fraction = float(cfg.get('altitude_blend_fraction', 0.5))
        self.climb_min_dist = float(cfg.get('altitude_blend_min_m', 2.0))
        self.path_xy = None
        self.path3 = None
        anchor = cfg.get('fixed_line_start_ned', None)
        self._configured_fixed_line_start = (None if anchor is None else
                                             np.asarray(anchor, dtype=float).reshape(3).copy())
        self.fixed_line_start = (None if self._configured_fixed_line_start is None else
                                 self._configured_fixed_line_start.copy())
        self.last_plan_t = -float('inf')
        self.plan_failures = 0
        self.planner = None
        if self.planner_enabled:
            self.planner = GridAStar(
                float(cfg['planner_resolution']),
                cfg['planner_xy_min'], cfg['planner_xy_max'],
                [(np.asarray(c, float), float(r)) for c, r in cfg['obstacles']],
                float(cfg['planner_inflation']),
                corner_cutting=False,
            )

    def reset(self):
        self.path_xy = None
        self.path3 = None
        self.fixed_line_start = (None if self._configured_fixed_line_start is None else
                                 self._configured_fixed_line_start.copy())
        self.last_plan_t = -float('inf')
        self.plan_failures = 0

    def _lift(self, xy, p_now):
        """Attach altitude: converge from current D to goal D along arc length."""
        xy = np.asarray(xy, dtype=float)
        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        total = float(cum[-1])
        blend = max(min(total, self.climb_min_dist), self.climb_fraction * total, 1e-6)
        alpha = np.clip(cum / blend, 0.0, 1.0)
        # smoothstep keeps the vertical reference velocity continuous
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        down = p_now[2] + alpha * (self.goal[2] - p_now[2])
        pts = np.column_stack([xy, down])
        pts[0] = p_now
        pts[-1] = self.goal
        return [q.copy() for q in pts]

    def update(self, t, p_now, force=False):
        p_now = np.asarray(p_now, dtype=float).reshape(3)
        if self.mode == 'fixed_line':
            # Deliberately keep the first navigation line fixed even after the
            # vehicle leaves it.  The NMPC therefore pays cross-track cost for
            # obstacle avoidance and is pulled back toward the same centerline.
            # This is used by the infeasible-slalom solver stress experiment.
            if self.fixed_line_start is None:
                self.fixed_line_start = p_now.copy()
            self.path3 = [self.fixed_line_start.copy(), self.goal.copy()]
            self.path_xy = [self.fixed_line_start[:2].copy(), self.goal[:2].copy()]
            return True
        if self.mode == 'direct':
            self.path3 = [p_now.copy(), self.goal.copy()]
            self.path_xy = [p_now[:2].copy(), self.goal[:2].copy()]
            return True
        due = force or self.path_xy is None or (t - self.last_plan_t) >= self.replan_period
        if not due:
            return self.path3 is not None
        xy = self.planner.plan_robust(p_now[:2], self.goal[:2])
        self.last_plan_t = t
        if xy is None:
            self.plan_failures += 1
            if self.path3 is None:
                # No route at all: keep the NMPC pointed at the goal; the hard
                # obstacle constraints still guarantee avoidance.
                self.path3 = [p_now.copy(), self.goal.copy()]
            return False
        self.path_xy = xy
        self.path3 = self._lift(xy, p_now)
        return True

    def horizon(self, p0, v0):
        """Reference positions/velocities for stages 0..N from state (p0, v0)."""
        p0 = np.asarray(p0, dtype=float).reshape(3)
        v0 = np.asarray(v0, dtype=float).reshape(3)
        path = self.path3 if self.path3 is not None else [p0, self.goal]
        tangent = polyline_tangent(path, p0)
        initial_speed = max(0.0, float(np.dot(v0, tangent)))
        remaining = polyline_remaining_length(path, p0)
        dist = speed_profile_distances(initial_speed, remaining, self.desired_speed,
                                       self.accel, self.decel, self.dt, self.N + 1)
        # Samples start at the projection of p0 on the guide.  Stage 0 is not
        # overwritten with p0: a lateral offset would then be
        # turned into a huge finite-difference reference velocity at k=0.
        pos = sample_polyline_distances(path, p0, dist)
        vel = np.zeros_like(pos)
        vel[:-1] = (pos[1:] - pos[:-1]) / self.dt
        vel[-1] = vel[-2] if remaining > dist[-1] + 1e-6 else 0.0
        return pos.T.copy(), vel.T.copy(), remaining

    def distance_to_goal(self, p):
        return float(np.linalg.norm(np.asarray(p, float).reshape(3) - self.goal))


def safe_takeoff_height(start, desired_height, eff_obstacles, arena_top_d=None, extra=0.15, step=0.05):
    """Largest climb <= desired_height whose vertical segment keeps clearance."""
    start = np.asarray(start, dtype=float).reshape(3)
    h = 0.0
    best = 0.0
    while h <= desired_height + 1e-9:
        p = start + np.array([0.0, 0.0, -h])
        if arena_top_d is not None and p[2] < arena_top_d:
            break
        if min_clearance(eff_obstacles, p) < extra:
            break
        best = h
        h += step
    return best
