import heapq
import math
import numpy as np


class GridAStar:
    """8-connected A* on a 2-D occupancy grid of vertically extruded spheres.

    The occupancy grid is computed once with NumPy (obstacles are static), so a
    replan costs only the graph search. ``plan`` uses conservative blocked-start semantics
    (``None`` when start or goal is blocked); ``plan_robust`` snaps a blocked
    start/goal to the nearest free cell, which is needed for online
    replanning because the NMPC may legitimately fly inside the planner's
    (more conservative) inflation.
    """

    _NBR = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))

    def __init__(self, resolution, xy_min, xy_max, obstacles, inflation, corner_cutting=True):
        # corner_cutting=True allows diagonal motion between diagonally adjacent
        # free cells even when one of the orthogonal side cells is blocked.  The
        # online guidance disables corner cutting and then applies line-of-sight
        # shortcutting for a conservative, smoother route.
        self.corner_cutting = bool(corner_cutting)
        self.res = float(resolution)
        self.xy_min = np.asarray(xy_min, dtype=float).reshape(-1)
        self.xy_max = np.asarray(xy_max, dtype=float).reshape(-1)
        self.obstacles = [(np.asarray(c, dtype=float).reshape(-1), float(r)) for c, r in obstacles]
        self.inflation = float(inflation)
        if self.res <= 0.0:
            raise ValueError('resolution must be > 0')
        if self.xy_min.shape != (2,) or self.xy_max.shape != (2,) or np.any(self.xy_max <= self.xy_min):
            raise ValueError('xy_min/xy_max must be valid 2-D bounds')
        self.shape = tuple((np.floor((self.xy_max - self.xy_min) / self.res + 1e-9).astype(int) + 1).tolist())
        ii, jj = np.meshgrid(np.arange(self.shape[0]), np.arange(self.shape[1]), indexing='ij')
        px = self.xy_min[0] + self.res * ii
        py = self.xy_min[1] + self.res * jj
        free = np.ones(self.shape, dtype=bool)
        for c, radius in self.obstacles:
            free &= np.hypot(px - c[0], py - c[1]) > radius + self.inflation
        self.free_grid = free

    def _to_idx(self, p):
        return tuple(np.rint((np.asarray(p, dtype=float)[:2] - self.xy_min) / self.res).astype(int).tolist())

    def _to_xy(self, idx):
        return self.xy_min + self.res * np.asarray(idx, dtype=float)

    def _inside(self, idx):
        return 0 <= idx[0] < self.shape[0] and 0 <= idx[1] < self.shape[1]

    def _free(self, idx):
        return self._inside(idx) and bool(self.free_grid[idx[0], idx[1]])

    def point_free(self, xy, extra=0.0):
        """Exact (non-gridded) clearance test of a 2-D point."""
        p = np.asarray(xy, dtype=float)[:2]
        if np.any(p < self.xy_min - 1e-9) or np.any(p > self.xy_max + 1e-9):
            return False
        return all(np.hypot(*(p - c[:2])) > r + self.inflation + extra for c, r in self.obstacles)

    def segment_free(self, a, b, extra=0.0):
        a = np.asarray(a, dtype=float)[:2]
        b = np.asarray(b, dtype=float)[:2]
        n = max(2, int(np.ceil(np.linalg.norm(b - a) / (0.25 * self.res))) + 1)
        return all(self.point_free(a + t * (b - a), extra) for t in np.linspace(0.0, 1.0, n))

    def nearest_free(self, xy, max_radius_m=3.0):
        idx = self._to_idx(xy)
        if self._free(idx):
            return idx
        best, best_d = None, float('inf')
        rmax = int(np.ceil(max_radius_m / self.res))
        for di in range(-rmax, rmax + 1):
            for dj in range(-rmax, rmax + 1):
                cand = (idx[0] + di, idx[1] + dj)
                if self._free(cand):
                    dd = di * di + dj * dj
                    if dd < best_d:
                        best, best_d = cand, dd
        return best

    def _search(self, s, g, goal_xy):
        queue = [(0.0, s)]
        parent = {s: None}
        cost = {s: 0.0}
        closed = set()
        goal_xy = np.asarray(goal_xy, dtype=float)[:2]
        while queue:
            _, cur = heapq.heappop(queue)
            if cur in closed:
                continue
            closed.add(cur)
            if cur == g:
                path = []
                while cur is not None:
                    path.append(self._to_xy(cur))
                    cur = parent[cur]
                return path[::-1]
            for dx, dy in self._NBR:
                nxt = (cur[0] + dx, cur[1] + dy)
                if nxt in closed or not self._free(nxt):
                    continue
                if dx and dy and not self.corner_cutting and \
                        not (self._free((cur[0] + dx, cur[1])) and self._free((cur[0], cur[1] + dy))):
                    continue  # no diagonal corner cutting
                step = self.res * (math.sqrt(2.0) if dx and dy else 1.0)
                new_cost = cost[cur] + step
                if nxt not in cost or new_cost < cost[nxt]:
                    cost[nxt] = new_cost
                    parent[nxt] = cur
                    heuristic = float(np.linalg.norm(self._to_xy(nxt) - goal_xy))
                    heapq.heappush(queue, (new_cost + heuristic, nxt))
        return None

    def plan(self, start_xy, goal_xy):
        s, g = self._to_idx(start_xy), self._to_idx(goal_xy)
        if not self._free(s) or not self._free(g):
            return None
        return self._search(s, g, goal_xy)

    def plan_robust(self, start_xy, goal_xy, smooth=True):
        """Plan from arbitrary start/goal (snapped to free cells) and shortcut.

        Returns a list of 2-D points starting at ``start_xy`` and ending at
        ``goal_xy``, or ``None`` if the free space is disconnected.  A start
        (goal) inside the inflated region is connected only to its nearest
        free cell, never shortcut across an obstacle.
        """
        start_xy = np.asarray(start_xy, dtype=float)[:2]
        goal_xy = np.asarray(goal_xy, dtype=float)[:2]
        s = self.nearest_free(start_xy)
        g = self.nearest_free(goal_xy)
        if s is None or g is None:
            return None
        cells = self._search(s, g, goal_xy)
        if cells is None:
            return None
        pts = [np.asarray(p, dtype=float) for p in cells]
        if self.point_free(start_xy):
            pts[0] = start_xy
        else:
            pts.insert(0, start_xy)
        if self.point_free(goal_xy):
            pts[-1] = goal_xy
        else:
            pts.append(goal_xy)
        if len(pts) == 1:
            pts.append(goal_xy)
        return self.shortcut(pts) if smooth else pts

    def shortcut(self, path):
        """Greedy line-of-sight pruning (string pulling) of a grid path."""
        pts = [np.asarray(p, dtype=float)[:2] for p in path]
        if len(pts) <= 2:
            return pts
        free = [self.point_free(p) for p in pts]
        out = [pts[0]]
        i = 0
        last = len(pts) - 1
        while i < last:
            nxt = i + 1
            if free[i]:
                for j in range(last, i + 1, -1):
                    if free[j] and self.segment_free(pts[i], pts[j]):
                        nxt = j
                        break
            out.append(pts[nxt])
            i = nxt
        return out


def _clean_polyline(polyline):
    pts = [np.asarray(p, dtype=float) for p in polyline]
    if not pts:
        return []
    clean = [pts[0]]
    for p in pts[1:]:
        if p.shape != clean[0].shape:
            raise ValueError('all polyline points must have the same dimension')
        if np.linalg.norm(p - clean[-1]) > 1e-9:
            clean.append(p)
    return clean


def _closest_projection_on_polyline(polyline, point):
    """Return (segment_index, interpolation_fraction) closest to point."""
    pts = np.asarray(polyline, dtype=float)
    p = np.asarray(point, dtype=float)
    if len(pts) < 2:
        return 0, 0.0
    best_d2 = float('inf')
    best = (0, 0.0)
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        ab = b - a
        denom = float(np.dot(ab, ab))
        t = 0.0 if denom < 1e-12 else float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
        proj = a + t * ab
        d2 = float(np.dot(p - proj, p - proj))
        if d2 < best_d2:
            best_d2 = d2
            best = (i, t)
    return best


def _remaining_polyline(polyline, point):
    clean = _clean_polyline(polyline)
    if not clean:
        return []
    if len(clean) == 1:
        return clean
    i0, t0 = _closest_projection_on_polyline(clean, point)
    projected = clean[i0] + t0 * (clean[i0 + 1] - clean[i0])
    remaining = [projected] + clean[i0 + 1:]
    if len(remaining) >= 2 and np.linalg.norm(remaining[0] - remaining[1]) < 1e-9:
        remaining = remaining[1:]
    return remaining


def polyline_remaining_length(path_xyz, start_xyz):
    """Arc length remaining after projecting start_xyz onto a polyline."""
    remaining = _remaining_polyline(path_xyz, start_xyz)
    if len(remaining) < 2:
        return 0.0
    return float(sum(np.linalg.norm(remaining[i + 1] - remaining[i])
                     for i in range(len(remaining) - 1)))


def polyline_tangent(path_xyz, start_xyz):
    """Unit forward tangent of the remaining polyline, or zeros at the end."""
    remaining = _remaining_polyline(path_xyz, start_xyz)
    if len(remaining) < 2:
        dim = len(np.asarray(start_xyz).reshape(-1))
        return np.zeros(dim, dtype=float)
    d = remaining[1] - remaining[0]
    norm = float(np.linalg.norm(d))
    return d / norm if norm > 1e-12 else np.zeros_like(d)


def speed_profile_distances(initial_speed, path_length, desired_speed, accel_max, decel_max, dt, count):
    """Generate distance-along-path samples with acceleration and braking limits.

    The profile accelerates toward desired_speed but also caps speed by the
    distance required to stop at the end of the path. distances[0] is always 0.
    """
    if count <= 0:
        return np.empty(0, dtype=float)
    values = [initial_speed, path_length, desired_speed, accel_max, decel_max, dt]
    if not np.all(np.isfinite(values)):
        raise ValueError('speed profile inputs must be finite')
    if path_length < 0.0 or desired_speed < 0.0 or accel_max <= 0.0 or decel_max <= 0.0 or dt <= 0.0:
        raise ValueError('invalid speed-profile parameters')

    s = 0.0
    v = float(np.clip(initial_speed, 0.0, desired_speed))
    out = np.zeros(count, dtype=float)
    for k in range(1, count):
        remaining = max(0.0, path_length - s)
        stopping_speed = math.sqrt(max(0.0, 2.0 * decel_max * remaining))
        target = min(desired_speed, stopping_speed)
        if target >= v:
            v = min(target, v + accel_max * dt)
        else:
            v = max(target, v - decel_max * dt)
        s = min(path_length, s + v * dt)
        out[k] = s
    return out


def sample_polyline_distances(path_xyz, start_xyz, distances):
    """Sample a polyline at explicit forward arc-length distances."""
    distances = np.asarray(distances, dtype=float).reshape(-1)
    if np.any(~np.isfinite(distances)) or np.any(distances < 0.0):
        raise ValueError('distances must be finite and nonnegative')
    if len(distances) > 1 and np.any(np.diff(distances) < -1e-12):
        raise ValueError('distances must be nondecreasing')

    remaining = _remaining_polyline(path_xyz, start_xyz)
    if not remaining:
        return np.tile(np.asarray(start_xyz, dtype=float), (len(distances), 1))
    if len(remaining) == 1:
        return np.tile(remaining[0], (len(distances), 1))

    seglens = np.asarray([
        np.linalg.norm(remaining[i + 1] - remaining[i])
        for i in range(len(remaining) - 1)
    ], dtype=float)
    cum = np.concatenate([[0.0], np.cumsum(seglens)])
    out = []
    for s in distances:
        s = min(float(s), float(cum[-1]))
        idx = min(max(np.searchsorted(cum, s, side='right') - 1, 0), len(remaining) - 2)
        frac = 0.0 if seglens[idx] < 1e-12 else (s - cum[idx]) / seglens[idx]
        out.append((1.0 - frac) * remaining[idx] + frac * remaining[idx + 1])
    return np.asarray(out)


def sample_polyline(path_xyz, start_xyz, speed, dt, count):
    """Sample forward from the closest location on a fixed polyline at constant speed."""
    if count <= 0:
        return np.empty((0, 3), dtype=float)
    if speed < 0.0 or dt <= 0.0:
        raise ValueError('speed must be >= 0 and dt must be > 0')
    distances = np.arange(count, dtype=float) * float(speed) * float(dt)
    return sample_polyline_distances(path_xyz, start_xyz, distances)
