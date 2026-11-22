"""Deterministic random scenario generation for PX4 NMPC experiments.

Two frames are used deliberately:

* Arena NED: room floor center is [N,E,D] = [0,0,0]. Random start,
  goal, and obstacles are generated in this frame.
* PX4-local NED: the vehicle start is treated as the local origin. Controller
  goals, obstacles, and arena bounds are translated from Arena NED into this
  frame before the ROS controller is launched.

Both frames have the same NED axis directions; only the origin changes.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np

from .planner import GridAStar


SCENARIO_SCHEMA_VERSION = 5
# The scenario schema stores a persistent PX4/EKF origin separately from
# the current mission-leg start. This is required for live re-targeting and
# environment changes without restarting SITL or moving the local frame.
SUPPORTED_SCENARIO_SCHEMAS = (SCENARIO_SCHEMA_VERSION,)


@dataclass
class Scenario:
    seed: int
    # Geometry below is stored in the arena-fixed NED frame.
    start_ned: List[float]
    goal_ned: List[float]
    goal_yaw: float
    obstacle_centers_ned: List[List[float]]
    obstacle_radii: List[float]
    arena_north_m: float
    arena_east_m: float
    arena_height_m: float
    # Altitude above the arena floor reached by the vertical take-off before
    # navigation starts. The vertical segment start -> take-off point is kept
    # free of obstacles by the generator.
    takeoff_altitude_m: Optional[float] = None
    # Arena-frame point that corresponds to PX4-local [0,0,0].  For a fresh
    # trial this equals start_ned.  For live re-planning it stays at the
    # original PX4/EKF origin while start_ned becomes the current vehicle pose.
    controller_origin_ned: Optional[List[float]] = None

    def to_dict(self) -> Dict:
        data = asdict(self)
        origin = self.start_ned if self.controller_origin_ned is None else self.controller_origin_ned
        data['controller_origin_ned'] = [float(v) for v in origin]
        data['schema_version'] = SCENARIO_SCHEMA_VERSION
        data['coordinate_frame'] = 'arena_ned_floor_center_origin'
        data['controller_frame'] = 'px4_local_ned_persistent_origin'
        if self.takeoff_altitude_m is not None:
            data['takeoff_ned'] = [float(self.start_ned[0]), float(self.start_ned[1]),
                                   -float(self.takeoff_altitude_m)]
        return data


def takeoff_altitude_of(scenario) -> Optional[float]:
    """Take-off altitude above the floor when present in the scenario."""
    s = scenario.to_dict() if isinstance(scenario, Scenario) else scenario
    value = s.get('takeoff_altitude_m')
    if value is None:
        value = s.get('experiment_config', {}).get('takeoff_altitude_m')
    return None if value is None else float(value)


def arena_to_local_ned(point_ned, start_ned):
    """Translate an arena-frame NED point into PX4-local NED."""
    p = np.asarray(point_ned, dtype=float).reshape(3)
    s = np.asarray(start_ned, dtype=float).reshape(3)
    return p - s


def controller_origin_ned_of(scenario: Scenario | Dict):
    """Arena-frame point represented by PX4-local NED [0,0,0]."""
    s = scenario.to_dict() if isinstance(scenario, Scenario) else scenario
    if 'controller_origin_ned' not in s:
        raise ValueError('scenario is missing controller_origin_ned; regenerate it with this project')
    return np.asarray(s['controller_origin_ned'], dtype=float).reshape(3)


def arena_origin_local_ned(scenario: Scenario | Dict):
    """Arena floor-center origin [0,0,0] expressed in PX4-local NED."""
    return -controller_origin_ned_of(scenario)


def arena_to_gazebo_enu(point_ned, start_ned):
    """Map an arena-frame NED point to the translated Gazebo ENU world.

    PX4 is spawned horizontally at Gazebo x=y=0. The room is translated so
    the randomized arena start lies below that world location. Gazebo z remains
    absolute height above the arena floor, hence z=-D rather than -(D-D_start).
    """
    p = np.asarray(point_ned, dtype=float).reshape(3)
    s = np.asarray(start_ned, dtype=float).reshape(3)
    return np.array([p[1] - s[1], p[0] - s[0], -p[2]], dtype=float)


def controller_geometry(scenario: Scenario | Dict):
    """Return scenario geometry in the persistent PX4-local NED frame.

    In schema 5 the mission-leg start may differ from the PX4/EKF origin.
    This is what allows a vehicle holding an old goal to receive a new
    environment and continue to a new target without restarting SITL.
    """
    s = scenario.to_dict() if isinstance(scenario, Scenario) else scenario
    start = np.asarray(s['start_ned'], dtype=float).reshape(3)
    origin = controller_origin_ned_of(s)
    out = {
        'start_local_ned': arena_to_local_ned(start, origin),
        'goal_local_ned': arena_to_local_ned(s['goal_ned'], origin),
        'obstacle_centers_local_ned': [
            arena_to_local_ned(c, origin) for c in s['obstacle_centers_ned']
        ],
        'arena_origin_local_ned': -origin,
        'controller_origin_ned': origin.copy(),
    }
    takeoff_alt = takeoff_altitude_of(s)
    if takeoff_alt is not None:
        out['takeoff_local_ned'] = arena_to_local_ned([start[0], start[1], -takeoff_alt], origin)
    return out


def _sample_quadrant_xy(rng, north_m, east_m, margin, quadrant):
    n_half = north_m / 2.0
    e_half = east_m / 2.0
    # Keep endpoints away from both room walls and the central dividing lines.
    if quadrant == 'lower_right':
        n = rng.uniform(-n_half + margin, -margin)
        e = rng.uniform(margin, e_half - margin)
    elif quadrant == 'upper_left':
        n = rng.uniform(margin, n_half - margin)
        e = rng.uniform(-e_half + margin, -margin)
    elif quadrant == 'lower_left':
        n = rng.uniform(-n_half + margin, -margin)
        e = rng.uniform(-e_half + margin, -margin)
    elif quadrant == 'upper_right':
        n = rng.uniform(margin, n_half - margin)
        e = rng.uniform(margin, e_half - margin)
    else:
        raise ValueError(f'unsupported quadrant: {quadrant}')
    return np.array([n, e], dtype=float)


def _clip_normal(rng, mean, std, low, high, count):
    if std <= 0.0:
        return np.full(count, np.clip(mean, low, high), dtype=float)
    return np.clip(rng.normal(mean, std, count), low, high)


def _point_segment_distance(point, a, b):
    """Euclidean distance from a 3-D point to a finite segment."""
    p = np.asarray(point, dtype=float).reshape(3)
    a = np.asarray(a, dtype=float).reshape(3)
    b = np.asarray(b, dtype=float).reshape(3)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        return float(np.linalg.norm(p - a))
    t = float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _path_exists(start_xy, goal_xy, obstacles, north_m, east_m, resolution, clearance, wall_margin):
    # Conservative 2-D check: every 3-D sphere is treated as a vertical column
    # inflated by the configured UAV clearance.
    inflated = [(np.asarray(c[:2]), float(r) + clearance) for c, r in obstacles]
    planner = GridAStar(
        resolution=float(resolution),
        xy_min=[-north_m / 2.0 + wall_margin, -east_m / 2.0 + wall_margin],
        xy_max=[north_m / 2.0 - wall_margin, east_m / 2.0 - wall_margin],
        obstacles=inflated,
        inflation=0.0,
    )
    return planner.plan(start_xy, goal_xy) is not None


def generate_scenario(cfg: Dict, seed: int | None = None) -> Scenario:
    """Generate a collision-aware, reproducible arena scenario."""
    seed = int(cfg.get('seed', 1) if seed is None else seed)
    rng = np.random.default_rng(seed)

    north_m = float(cfg['arena_north_m'])
    east_m = float(cfg['arena_east_m'])
    height_m = float(cfg['arena_height_m'])
    wall_margin = float(cfg.get('wall_margin_m', 0.45))
    quadrant_margin = float(cfg.get('quadrant_margin_m', 1.0))
    obstacle_count = int(cfg['obstacle_count'])
    radius_mean = float(cfg['obstacle_radius_mean_m'])
    radius_std = float(cfg['obstacle_radius_std_m'])
    radius_min = float(cfg.get('obstacle_radius_min_m', 0.3))
    radius_max = float(cfg.get('obstacle_radius_max_m', 1.1))
    min_gap = float(cfg.get('obstacle_min_gap_m', 0.35))
    endpoint_clearance = float(cfg.get('start_goal_clearance_m', 1.4))
    boundary_clearance = float(cfg.get('obstacle_boundary_clearance_m', 0.25))
    planner_resolution = float(cfg.get('feasibility_resolution_m', 0.35))
    uav_clearance = float(cfg.get('feasibility_uav_clearance_m', 0.75))
    min_start_goal = float(cfg.get('min_start_goal_horizontal_distance_m', 0.0))
    max_endpoint_attempts = int(cfg.get('max_endpoint_attempts', 200))
    max_layout_attempts = int(cfg.get('max_layout_attempts', 100))
    max_candidate_attempts = int(cfg.get('max_candidate_attempts', 2000))

    if not np.all(np.isfinite([north_m, east_m, height_m])) or min(north_m, east_m, height_m) <= 0.0:
        raise ValueError('arena dimensions must be finite and positive')
    # In each signed half-axis, the interval is [margin, half_extent-margin].
    # It is nonempty only when 2*margin < half_extent -> 4*margin < dimension.
    if quadrant_margin <= 0.0 or 4.0 * quadrant_margin >= min(north_m, east_m):
        raise ValueError(
            'quadrant_margin_m must be > 0 and smaller than one quarter of each horizontal arena dimension')
    if obstacle_count < 0:
        raise ValueError('obstacle_count must be >= 0')
    if not (0.0 < radius_min <= radius_max):
        raise ValueError('invalid obstacle radius bounds')
    if radius_mean <= 0.0 or radius_std < 0.0:
        raise ValueError('obstacle radius mean/std must satisfy mean>0, std>=0')
    if min(wall_margin, min_gap, endpoint_clearance, boundary_clearance, uav_clearance, min_start_goal) < 0.0:
        raise ValueError('margins, clearances, and minimum separation must be nonnegative')
    if planner_resolution <= 0.0:
        raise ValueError('feasibility_resolution_m must be positive')
    if min(max_endpoint_attempts, max_layout_attempts, max_candidate_attempts) < 1:
        raise ValueError('generation attempt counts must be positive integers')

    start_alt = float(cfg.get('start_altitude_m', 0.15))
    # Keep the complete vertical take-off corridor clear of obstacles.
    takeoff_alt = cfg.get('takeoff_altitude_m')
    takeoff_alt = None if takeoff_alt is None else float(takeoff_alt)
    takeoff_clearance = float(cfg.get('takeoff_corridor_clearance_m', uav_clearance))
    goal_alt_min = float(cfg.get('goal_altitude_min_m', 2.0))
    goal_alt_max = float(cfg.get('goal_altitude_max_m', min(4.0, height_m - 0.6)))
    if not (0.0 <= start_alt < height_m):
        raise ValueError('start_altitude_m must lie inside the arena height')
    if takeoff_alt is not None and not (start_alt < takeoff_alt < height_m - boundary_clearance):
        raise ValueError(
            'takeoff_altitude_m must be above start_altitude_m and below the ceiling clearance')
    if takeoff_clearance < 0.0:
        raise ValueError('takeoff_corridor_clearance_m must be nonnegative')
    if not (0.0 < goal_alt_min <= goal_alt_max < height_m):
        raise ValueError('goal altitude range must satisfy 0 < min <= max < arena_height_m')

    horizontal_pad = wall_margin + boundary_clearance + radius_max
    if 2.0 * horizontal_pad >= min(north_m, east_m):
        raise ValueError('arena too small for obstacle_radius_max_m and configured wall/boundary margins')
    if 2.0 * (radius_max + boundary_clearance) >= height_m:
        raise ValueError('arena height too small for obstacle_radius_max_m and boundary clearance')

    # Optional fixed endpoints are used by the live mission tool.  They are
    # expressed in the arena-fixed NED frame, exactly like Scenario.start_ned
    # and goal_ned.  Normal randomized experiments leave both unset.
    fixed_start = cfg.get('start_ned_override')
    fixed_goal = cfg.get('goal_ned_override')
    if fixed_start is not None:
        start_ned = np.asarray(fixed_start, dtype=float).reshape(3)
        start_xy = start_ned[:2].copy()
        start_alt = -float(start_ned[2])
        if not (0.0 <= start_alt < height_m):
            raise ValueError('start_ned_override altitude must lie inside the arena')
        if abs(start_xy[0]) > north_m / 2.0 - wall_margin or abs(start_xy[1]) > east_m / 2.0 - wall_margin:
            raise ValueError('start_ned_override lies outside the arena wall margin')
    else:
        start_xy = _sample_quadrant_xy(
            rng, north_m, east_m, quadrant_margin, cfg.get('start_quadrant', 'lower_right'))
        start_ned = np.array([start_xy[0], start_xy[1], -start_alt], dtype=float)

    if fixed_goal is not None:
        goal_ned = np.asarray(fixed_goal, dtype=float).reshape(3)
        goal_xy = goal_ned[:2].copy()
        goal_alt = -float(goal_ned[2])
        if not (0.0 < goal_alt < height_m):
            raise ValueError('goal_ned_override altitude must lie inside the arena')
        if abs(goal_xy[0]) > north_m / 2.0 - wall_margin or abs(goal_xy[1]) > east_m / 2.0 - wall_margin:
            raise ValueError('goal_ned_override lies outside the arena wall margin')
    else:
        goal_xy = None
        for _ in range(max_endpoint_attempts):
            candidate_goal = _sample_quadrant_xy(
                rng, north_m, east_m, quadrant_margin, cfg.get('goal_quadrant', 'upper_left'))
            if np.linalg.norm(candidate_goal - start_xy) >= min_start_goal:
                goal_xy = candidate_goal
                break
        if goal_xy is None:
            raise RuntimeError(
                'Could not sample a goal with the requested separation. '
                'Reduce min_start_goal_horizontal_distance_m or adjust the goal quadrant/margins.')
        goal_alt = rng.uniform(goal_alt_min, goal_alt_max)
        goal_ned = np.array([goal_xy[0], goal_xy[1], -goal_alt], dtype=float)

    if np.linalg.norm(goal_xy - start_xy) < min_start_goal:
        raise ValueError('fixed start/goal violate min_start_goal_horizontal_distance_m')

    takeoff_ned = None if takeoff_alt is None else np.array([start_xy[0], start_xy[1], -takeoff_alt], dtype=float)

    radii = _clip_normal(rng, radius_mean, radius_std, radius_min, radius_max, obstacle_count)
    n_cells = max(1, int(math.ceil(math.sqrt(max(1, obstacle_count)))))
    e_cells = max(1, int(math.ceil(max(1, obstacle_count) / n_cells)))

    for _layout in range(max_layout_attempts):
        centers: List[np.ndarray] = []
        cells = [(i, j) for i in range(n_cells) for j in range(e_cells)]
        rng.shuffle(cells)

        for obs_idx in range(obstacle_count):
            r = float(radii[obs_idx])
            placed = False
            for local_try in range(max_candidate_attempts):
                ci, cj = cells[(obs_idx + local_try) % len(cells)]
                n_lo = -north_m / 2.0 + wall_margin + r + boundary_clearance
                n_hi = north_m / 2.0 - wall_margin - r - boundary_clearance
                e_lo = -east_m / 2.0 + wall_margin + r + boundary_clearance
                e_hi = east_m / 2.0 - wall_margin - r - boundary_clearance
                if n_lo >= n_hi or e_lo >= e_hi:
                    raise ValueError('arena too small for configured obstacle sizes/margins')

                cn0 = -north_m / 2.0 + ci * north_m / n_cells
                cn1 = -north_m / 2.0 + (ci + 1) * north_m / n_cells
                ce0 = -east_m / 2.0 + cj * east_m / e_cells
                ce1 = -east_m / 2.0 + (cj + 1) * east_m / e_cells
                lo_n, hi_n = max(n_lo, cn0), min(n_hi, cn1)
                lo_e, hi_e = max(e_lo, ce0), min(e_hi, ce1)
                if lo_n >= hi_n or lo_e >= hi_e:
                    continue

                n = rng.uniform(lo_n, hi_n)
                e = rng.uniform(lo_e, hi_e)
                alt_lo = r + boundary_clearance
                alt_hi = height_m - r - boundary_clearance
                if alt_lo >= alt_hi:
                    continue
                alt = rng.uniform(alt_lo, alt_hi)
                c = np.array([n, e, -alt], dtype=float)

                if np.linalg.norm(c - start_ned) < r + endpoint_clearance:
                    continue
                # The navigation experiment starts only after a vertical take-off.
                # Keep that entire ascent corridor clear, not merely the start point.
                if takeoff_ned is not None and \
                        _point_segment_distance(c, start_ned, takeoff_ned) < r + takeoff_clearance:
                    continue
                if np.linalg.norm(c - goal_ned) < r + endpoint_clearance:
                    continue
                if any(
                    np.linalg.norm(c - other) < r + float(rother) + min_gap
                    for other, rother in zip(centers, radii[:len(centers)])
                ):
                    continue
                centers.append(c)
                placed = True
                break
            if not placed:
                break

        if len(centers) != obstacle_count:
            continue

        obstacles = [(c, float(r)) for c, r in zip(centers, radii)]
        if _path_exists(
            start_xy, goal_xy, obstacles, north_m, east_m,
            planner_resolution, uav_clearance, wall_margin,
        ):
            return Scenario(
                seed=seed,
                start_ned=start_ned.tolist(),
                goal_ned=goal_ned.tolist(),
                goal_yaw=float(cfg.get('goal_yaw', 0.0)),
                obstacle_centers_ned=[c.tolist() for c in centers],
                obstacle_radii=[float(r) for r in radii],
                arena_north_m=north_m,
                arena_east_m=east_m,
                arena_height_m=height_m,
                takeoff_altitude_m=takeoff_alt,
                controller_origin_ned=(None if cfg.get('controller_origin_ned') is None
                                       else [float(v) for v in cfg['controller_origin_ned']]),
            )

    raise RuntimeError(
        'Could not generate a feasible scenario. Reduce obstacle_count/radii or enlarge the arena.')


def scenario_parameters(scenario: Scenario | Dict, wall_margin_m: float = 0.45) -> Dict:
    """Flat controller parameters (PX4-local NED) for a generated scenario.

    This is exactly what ``nmpc_experiment new`` writes into scenario.yaml;
    the offline simulator and the MAVLink runner use the same mapping.
    """
    s = scenario.to_dict() if isinstance(scenario, Scenario) else scenario
    geometry = controller_geometry(s)
    origin = geometry['arena_origin_local_ned']
    n_m, e_m, h_m = float(s['arena_north_m']), float(s['arena_east_m']), float(s['arena_height_m'])
    params = {
        'goal_ned': [float(v) for v in geometry['goal_local_ned']],
        'goal_yaw': float(s['goal_yaw']),
        'obstacle_centers_flat': [float(v) for c in geometry['obstacle_centers_local_ned'] for v in c],
        'obstacle_radii': [float(v) for v in s['obstacle_radii']],
        'arena_north_m': n_m,
        'arena_east_m': e_m,
        'arena_height_m': h_m,
        'arena_origin_local_ned': [float(v) for v in origin],
        'planner_xy_min': [float(origin[0] - n_m / 2.0 + wall_margin_m), float(origin[1] - e_m / 2.0 + wall_margin_m)],
        'planner_xy_max': [float(origin[0] + n_m / 2.0 - wall_margin_m), float(origin[1] + e_m / 2.0 - wall_margin_m)],
    }
    if 'takeoff_local_ned' in geometry:
        # the runtime climbs takeoff_height_m above the start point
        params['takeoff_height_m'] = float(-geometry['takeoff_local_ned'][2])
    # Optional generic anchor for experiments that deliberately keep one fixed
    # reference line after the vehicle deviates.  Stored in arena NED in the
    # scenario, converted here to the controller's persistent local frame.
    if s.get('reference_start_ned') is not None:
        ref_start = arena_to_local_ned(s['reference_start_ned'], geometry['controller_origin_ned'])
        params['fixed_line_start_ned'] = [float(v) for v in ref_start]
    return params


def validate_scenario_dict(data: Dict) -> Dict:
    """Check a scenario mapping (scenario.json) and return it unchanged.

    Raises ``ValueError`` with an actionable message when the file was written
    by an incompatible version or is incomplete.
    """
    if not isinstance(data, dict):
        raise ValueError('scenario must be a JSON object')
    required = {
        'schema_version', 'coordinate_frame', 'controller_frame', 'seed',
        'start_ned', 'goal_ned', 'goal_yaw', 'obstacle_centers_ned', 'obstacle_radii',
        'arena_north_m', 'arena_east_m', 'arena_height_m', 'controller_origin_ned',
    }
    missing = sorted(required - data.keys())
    if missing:
        raise ValueError(f'scenario is missing fields {missing}; regenerate it with '
                         '`ros2 run px4_nmpc_tools nmpc_experiment new --seed <seed>`')
    if int(data['schema_version']) not in SUPPORTED_SCENARIO_SCHEMAS:
        raise ValueError(f'unsupported scenario schema {data["schema_version"]} '
                         f'(expected {SCENARIO_SCHEMA_VERSION}); regenerate the scenario')
    if data['coordinate_frame'] != 'arena_ned_floor_center_origin':
        raise ValueError('unsupported scenario coordinate_frame; regenerate the scenario')
    if data['controller_frame'] != 'px4_local_ned_persistent_origin':
        raise ValueError('unsupported scenario controller_frame; regenerate the scenario')
    for key in ('start_ned', 'goal_ned', 'controller_origin_ned'):
        if len(data[key]) != 3:
            raise ValueError(f'scenario {key} must contain 3 values')
    if len(data['obstacle_centers_ned']) != len(data['obstacle_radii']):
        raise ValueError('scenario obstacle centers/radii sizes disagree')
    if any(len(c) != 3 for c in data['obstacle_centers_ned']):
        raise ValueError('every scenario obstacle center must contain 3 values')
    if any(float(r) <= 0.0 for r in data['obstacle_radii']):
        raise ValueError('scenario obstacle radii must be positive')
    return data


def load_scenario_json(path) -> Dict:
    """Read and validate a scenario.json file."""
    import json
    import pathlib
    p = pathlib.Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f'scenario file not found: {p}')
    return validate_scenario_dict(json.loads(p.read_text()))


def scenario_wall_margin(scenario: Dict, default: float = 0.45) -> float:
    return float((scenario.get('experiment_config') or {}).get('wall_margin_m', default))
