from __future__ import annotations

import argparse
import json
import math
import pathlib
import shutil
import subprocess
from typing import Dict

import yaml

from px4_nmpc_core.paths import config_file
from px4_nmpc_core.scenario import (
    arena_to_gazebo_enu,
    controller_geometry,
    controller_origin_ned_of,
    generate_scenario,
    load_scenario_json,
    scenario_parameters,
    takeoff_altitude_of,
)


DEFAULT_WORLD = 'default'


def _config_dir():
    return config_file('experiment.yaml').parent


def _default_experiment_cfg():
    return _config_dir() / 'experiment.yaml'


def _default_runtime_dir():
    return pathlib.Path.home() / '.ros' / 'px4_nmpc'


def _load_experiment(path=None) -> Dict:
    p = pathlib.Path(path).expanduser() if path else _default_experiment_cfg()
    if not p.exists():
        raise FileNotFoundError(f'Experiment configuration not found: {p}')
    data = yaml.safe_load(p.read_text())
    if not isinstance(data, dict) or not isinstance(data.get('experiment'), dict):
        raise ValueError(f'{p} must contain a top-level "experiment" mapping')
    return data['experiment']


def _scenario_paths(runtime_dir=None):
    d = pathlib.Path(runtime_dir).expanduser() if runtime_dir else _default_runtime_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d / 'scenario.yaml', d / 'scenario.json'


def _write_scenario(scenario, yaml_path, json_path, experiment_cfg):
    geometry = controller_geometry(scenario)
    # This YAML is deliberately controller-local. It overrides only geometry
    # that changes per random scenario; solver/controller tuning stays in
    # nmpc.yaml.  scenario_parameters() is shared with the offline simulator,
    # the MAVLink runner and the tests, so all of them see identical geometry.
    wall_clearance = float(experiment_cfg.get('wall_margin_m', 0.45))
    ros_params = scenario_parameters(scenario, wall_clearance)
    params = {'px4_nmpc_controller': {'ros__parameters': ros_params}}
    yaml_path.write_text(yaml.safe_dump(params, sort_keys=False))

    # JSON is the complete reproducibility record for the random geometry.
    metadata = scenario.to_dict() if hasattr(scenario, 'to_dict') else dict(scenario)
    cfg_snapshot = dict(experiment_cfg)
    cfg_snapshot['seed'] = int(metadata['seed'])
    metadata['experiment_config'] = cfg_snapshot
    metadata['controller_start_local_ned'] = [float(v) for v in geometry['start_local_ned']]
    metadata['controller_origin_ned'] = [float(v) for v in geometry['controller_origin_ned']]
    metadata['controller_goal_local_ned'] = [float(v) for v in geometry['goal_local_ned']]
    metadata['controller_obstacle_centers_local_ned'] = [
        [float(v) for v in c] for c in geometry['obstacle_centers_local_ned']
    ]
    metadata['arena_origin_local_ned'] = [float(v) for v in geometry['arena_origin_local_ned']]
    if 'takeoff_local_ned' in geometry:
        metadata['takeoff_local_ned'] = [float(v) for v in geometry['takeoff_local_ned']]
    metadata['planner_xy_min_local_ned'] = ros_params['planner_xy_min']
    metadata['planner_xy_max_local_ned'] = ros_params['planner_xy_max']
    json_path.write_text(json.dumps(metadata, indent=2) + '\n')


def _load_runtime_scenario(runtime_dir=None):
    _, json_path = _scenario_paths(runtime_dir)
    if not json_path.exists():
        raise FileNotFoundError(
            f'No scenario exists at {json_path}. Run: '
            'ros2 run px4_nmpc_tools nmpc_experiment new --seed <seed>')
    data = load_scenario_json(json_path)
    if 'experiment_config' not in data:
        raise ValueError('Scenario file has no experiment_config; regenerate it with '
                         'ros2 run px4_nmpc_tools nmpc_experiment new --seed <seed>')
    return data


def _require_gz():
    if shutil.which('gz') is None:
        raise RuntimeError('Gazebo command "gz" was not found in PATH')


def _gz_service(service, reqtype, request, world=DEFAULT_WORLD):
    _require_gz()
    cmd = [
        'gz', 'service', '-s', f'/world/{world}/{service}',
        '--reqtype', reqtype, '--reptype', 'gz.msgs.Boolean',
        '--timeout', '3000', '--req', request,
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    output = (proc.stdout + '\n' + proc.stderr).strip()
    if proc.returncode != 0 or 'data: false' in output.lower():
        detail = output if output else f'command exited with code {proc.returncode}'
        raise RuntimeError(f'Gazebo service /world/{world}/{service} failed: {detail}')
    return output


def _create_model(sdf, world=DEFAULT_WORLD):
    req = 'sdf: "' + sdf.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '') + '"'
    _gz_service('create', 'gz.msgs.EntityFactory', req, world)


def _remove_model(name, world=DEFAULT_WORLD):
    req = f'name: "{name}" type: MODEL'
    _gz_service('remove', 'gz.msgs.Entity', req, world)


def _box_visual(name, x, y, z, sx, sy, sz, rgba, collision=True, transparency=None):
    r, g, b, a = rgba
    trans = '' if transparency is None else f'<transparency>{transparency}</transparency>'
    collision_xml = ''
    if collision:
        collision_xml = (
            f'<collision name="{name}_c"><geometry><box><size>{sx} {sy} {sz}</size>'
            '</box></geometry></collision>'
        )
    return f'''
      <link name="{name}">
        <pose>{x} {y} {z} 0 0 0</pose>
        {collision_xml}
        <visual name="{name}_v">
          <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
          <material><ambient>{r} {g} {b} {a}</ambient><diffuse>{r} {g} {b} {a}</diffuse></material>
          {trans}
        </visual>
      </link>'''


def _arena_sdf(appearance_cfg, scenario):
    # Geometry dimensions are saved in the scenario, so replay is not changed
    # by later edits to experiment.yaml. Appearance/collision options are also
    # read from the saved experiment_config snapshot.
    north_m = float(scenario['arena_north_m'])
    east_m = float(scenario['arena_east_m'])
    height_m = float(scenario['arena_height_m'])
    thickness = float(appearance_cfg.get('wall_thickness_m', 0.18))
    gap = float(appearance_cfg.get('floor_tile_gap_m', 0.04))
    origin = controller_origin_ned_of(scenario)

    # Gazebo world remains anchored to the original PX4/EKF origin. For a
    # fresh trial origin==start; for a live next-leg scenario start may be the
    # vehicle's current arena position while the room stays fixed.
    room_x = -float(origin[1])
    room_y = -float(origin[0])
    parts = []

    tile_e = east_m / 2.0 - gap
    tile_n = north_m / 2.0 - gap
    if tile_e <= 0.0 or tile_n <= 0.0:
        raise ValueError('floor_tile_gap_m is too large for the saved arena dimensions')

    colors = [
        (0.26, 0.29, 0.33, 1.0), (0.30, 0.33, 0.37, 1.0),
        (0.28, 0.31, 0.35, 1.0), (0.32, 0.35, 0.39, 1.0),
    ]
    centers = [
        (-east_m / 4, -north_m / 4), (east_m / 4, -north_m / 4),
        (-east_m / 4, north_m / 4), (east_m / 4, north_m / 4),
    ]
    for i, ((x, y), color) in enumerate(zip(centers, colors)):
        parts.append(_box_visual(
            f'floor_{i}', x, y, 0.006, tile_e, tile_n, 0.012,
            color, collision=False))

    line = (0.72, 0.75, 0.80, 1.0)
    parts.append(_box_visual('line_n', 0, 0, 0.014, 0.035, north_m, 0.012, line, collision=False))
    parts.append(_box_visual('line_e', 0, 0, 0.014, east_m, 0.035, 0.012, line, collision=False))

    wall = (0.12, 0.16, 0.22, 1.0)
    parts.append(_box_visual(
        'wall_east', east_m / 2 + thickness / 2, 0, height_m / 2,
        thickness, north_m + 2 * thickness, height_m, wall))
    parts.append(_box_visual(
        'wall_west', -east_m / 2 - thickness / 2, 0, height_m / 2,
        thickness, north_m + 2 * thickness, height_m, wall))
    parts.append(_box_visual(
        'wall_north', 0, north_m / 2 + thickness / 2, height_m / 2,
        east_m, thickness, height_m, wall))
    parts.append(_box_visual(
        'wall_south', 0, -north_m / 2 - thickness / 2, height_m / 2,
        east_m, thickness, height_m, wall))

    if bool(appearance_cfg.get('ceiling_collision', True)):
        parts.append(_box_visual(
            'ceiling', 0, 0, height_m + thickness / 2,
            east_m, north_m, thickness,
            (0.55, 0.60, 0.68, 0.25), collision=True,
            transparency=float(appearance_cfg.get('ceiling_transparency', 0.82))))

    return (
        '<sdf version="1.9"><model name="nmpc_arena"><static>true</static>'
        f'<pose>{room_x} {room_y} 0 0 0 0</pose>'
        + ''.join(parts) + '</model></sdf>'
    )


def _sphere_sdf(name, enu, radius, rgba, collision=True, transparency=None):
    x, y, z = [float(v) for v in enu]
    radius = float(radius)
    r, g, b, a = rgba
    collision_xml = '' if not collision else (
        f'<collision name="c"><geometry><sphere><radius>{radius}</radius></sphere></geometry></collision>'
    )
    trans = '' if transparency is None else f'<transparency>{float(transparency)}</transparency>'
    return (
        f'<sdf version="1.9"><model name="{name}"><static>true</static>'
        f'<pose>{x} {y} {z} 0 0 0</pose><link name="link">{collision_xml}'
        f'<visual name="v"><geometry><sphere><radius>{radius}</radius></sphere></geometry>'
        f'<material><ambient>{r} {g} {b} {a}</ambient><diffuse>{r} {g} {b} {a}</diffuse>'
        f'<emissive>{0.15*r} {0.15*g} {0.15*b} {a}</emissive></material>{trans}</visual>'
        '</link></model></sdf>'
    )


def _cylinder_sdf(name, enu, radius, length, rgba):
    x, y, z = [float(v) for v in enu]
    r, g, b, a = rgba
    return (
        f'<sdf version="1.9"><model name="{name}"><static>true</static>'
        f'<pose>{x} {y} {z} 0 0 0</pose><link name="link"><visual name="v">'
        f'<geometry><cylinder><radius>{radius}</radius><length>{length}</length></cylinder></geometry>'
        f'<material><ambient>{r} {g} {b} {a}</ambient><diffuse>{r} {g} {b} {a}</diffuse>'
        f'<emissive>{0.2*r} {0.2*g} {0.2*b} {a}</emissive></material></visual></link></model></sdf>'
    )


def _rod_sdf(name, enu, radius, length, rgba, along_x=False):
    """Static cylinder with collision: vertical, or horizontal along Gazebo x (East)."""
    x, y, z = [float(v) for v in enu]
    r, g, b, a = rgba
    pitch = 1.5707963267948966 if along_x else 0.0
    geometry = f'<geometry><cylinder><radius>{radius}</radius><length>{length}</length></cylinder></geometry>'
    return (
        f'<sdf version="1.9"><model name="{name}"><static>true</static>'
        f'<pose>{x} {y} {z} 0 {pitch} 0</pose><link name="link">'
        f'<collision name="c">{geometry}</collision><visual name="v">{geometry}'
        f'<material><ambient>{r} {g} {b} {a}</ambient><diffuse>{r} {g} {b} {a}</diffuse>'
        f'<emissive>{0.15*r} {0.15*g} {0.15*b} {a}</emissive></material></visual></link></model></sdf>'
    )


def _typed_obstacle_models(scenario, origin):
    """Posts (type 1) and bars (type 2) of the challenging course; spheres use obstacle_centers_ned."""
    height = float(scenario.get('arena_height_m', 6.0))
    width = float(scenario.get('arena_east_m', 10.0))
    models = []
    for i, (kind, a, b, c, radius) in enumerate(scenario.get('typed_obstacles', [])):
        if int(kind) == 1:
            enu = arena_to_gazebo_enu([a, b, -0.5 * height], origin)
            models.append((f'nmpc_post_{i}', _rod_sdf(f'nmpc_post_{i}', enu, radius, height,
                                                       (0.85, 0.55, 0.10, 1.0))))
        elif int(kind) == 2:
            enu = arena_to_gazebo_enu([a, 0.0, c], origin)
            enu[0] = -float(origin[1])          # centred across the hall (arena E = 0)
            models.append((f'nmpc_bar_{i}', _rod_sdf(f'nmpc_bar_{i}', enu, radius, width,
                                                      (0.55, 0.20, 0.75, 1.0), along_x=True)))
    return models


def _list_models():
    _require_gz()
    proc = subprocess.run(['gz', 'model', '--list'], check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        detail = (proc.stdout + '\n' + proc.stderr).strip()
        raise RuntimeError(f'Could not query Gazebo models: {detail}')
    names = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith('- '):
            names.append(line[2:].strip())
    return names


def clear_scene(world=DEFAULT_WORLD):
    removed = []
    for name in _list_models():
        if name.startswith('nmpc_'):
            _remove_model(name, world)
            removed.append(name)
    return removed


def apply_scene(scenario, world=DEFAULT_WORLD):
    appearance_cfg = scenario['experiment_config']
    clear_scene(world)
    _create_model(_arena_sdf(appearance_cfg, scenario), world)

    origin = controller_origin_ned_of(scenario)
    for i, (center, radius) in enumerate(zip(
            scenario['obstacle_centers_ned'], scenario['obstacle_radii'])):
        _create_model(_sphere_sdf(
            f'nmpc_obs_{i}', arena_to_gazebo_enu(center, origin), float(radius),
            (0.82, 0.18, 0.07, 1.0), collision=True), world)

    typed = _typed_obstacle_models(scenario, origin)
    for _, sdf in typed:
        _create_model(sdf, world)

    # Mark the start of the *current mission leg*.  On the first leg this is
    # under the PX4 spawn; on a live next leg it is the previous goal/current pose.
    start_world = arena_to_gazebo_enu(scenario['start_ned'], origin)
    _create_model(_cylinder_sdf(
        'nmpc_start_marker', [start_world[0], start_world[1], 0.025], 0.45, 0.03,
        (0.08, 0.35, 0.95, 1.0)), world)
    _create_model(_sphere_sdf(
        'nmpc_goal_marker', arena_to_gazebo_enu(scenario['goal_ned'], origin), 0.28,
        (0.10, 0.90, 0.25, 0.85), collision=False, transparency=0.15), world)

    expected = {'nmpc_arena', 'nmpc_start_marker', 'nmpc_goal_marker'} | {
        f'nmpc_obs_{i}' for i in range(len(scenario['obstacle_radii']))
    } | {name for name, _ in typed}
    present = set(_list_models())
    missing = sorted(expected - present)
    if missing:
        raise RuntimeError(f'Gazebo scene creation returned but models are missing: {missing}')


def _print_summary(s):
    geometry = controller_geometry(s)
    print(f"schema: {s.get('schema_version', 'n/a')}")
    print(f"seed: {s['seed']}")
    print(f"start arena NED: {s['start_ned']}")
    print(f"controller origin arena NED: {controller_origin_ned_of(s).tolist()}")
    print(f"start PX4-local NED: {geometry['start_local_ned'].tolist()}")
    print(f"goal  arena NED: {s['goal_ned']}")
    print(f"goal  PX4-local NED: {geometry['goal_local_ned'].tolist()}")
    print(f"arena origin PX4-local NED: {geometry['arena_origin_local_ned'].tolist()}")
    takeoff_alt = takeoff_altitude_of(s)
    if takeoff_alt is not None:
        print(f"take-off altitude: {takeoff_alt:.2f} m above floor "
              f"(local D={geometry['takeoff_local_ned'][2]:.3f}; corridor kept clear)")
    else:
        print('take-off altitude: not specified; nmpc.yaml takeoff_height_m is used')
    print(f"obstacles: {len(s['obstacle_radii'])}")
    if s['obstacle_radii']:
        mean = sum(s['obstacle_radii']) / len(s['obstacle_radii'])
        var = sum((r - mean) ** 2 for r in s['obstacle_radii']) / len(s['obstacle_radii'])
        print(f"sampled radius mean/std: {mean:.3f} / {math.sqrt(var):.3f} m")


def main():
    parser = argparse.ArgumentParser(description='Generate and control randomized PX4 NMPC experiments.')
    parser.add_argument('--config', default=None, help='experiment.yaml used by the NEW command')
    parser.add_argument('--runtime-dir', default=None, help='scenario directory (default ~/.ros/px4_nmpc)')
    sub = parser.add_subparsers(dest='command', required=True)

    p_new = sub.add_parser('new', help='Generate a new reproducible scenario')
    p_new.add_argument('--seed', type=int, default=None)
    p_new.add_argument('--start-ned', nargs=3, type=float, metavar=('N','E','D'),
                       help='optional fixed arena-NED start instead of random sampling')
    p_new.add_argument('--goal-ned', nargs=3, type=float, metavar=('N','E','D'),
                       help='optional fixed arena-NED goal instead of random sampling')
    p_load = sub.add_parser('load', help='Use a fixed scenario JSON (e.g. the infeasible-slalom bundle) as the runtime scenario')
    p_load.add_argument('scenario', help='schema-5 scenario JSON (e.g. config/experiments/infeasible_slalom/scenario_10.json)')
    p_load.add_argument('--experiment', default=None,
                        help='experiment.yaml of the bundle (arena size, wall margin, take-off altitude)')
    sub.add_parser('show', help='Show the current runtime scenario')
    sub.add_parser('apply', help='Spawn/verify arena, obstacles, start and goal markers in Gazebo default world')
    sub.add_parser('clear', help='Remove nmpc_* experiment entities from Gazebo default world')
    sub.add_parser('px4-command', help='Print the PX4 startup command for the current scenario')
    args = parser.parse_args()

    yaml_path, json_path = _scenario_paths(args.runtime_dir)

    if args.command == 'new':
        cfg = _load_experiment(args.config)
        if args.start_ned is not None:
            cfg['start_ned_override'] = list(args.start_ned)
        if args.goal_ned is not None:
            cfg['goal_ned_override'] = list(args.goal_ned)
        scenario = generate_scenario(cfg, args.seed)
        _write_scenario(scenario, yaml_path, json_path, cfg)
        print(f'Wrote: {yaml_path}')
        print(f'Wrote: {json_path}')
        _print_summary(scenario.to_dict())
        print('\nNext:')
        print('  ros2 run px4_nmpc_tools nmpc_experiment px4-command   # start PX4/Gazebo with the printed command')
        print('  ros2 run px4_nmpc_tools nmpc_doctor                   # PX4 <-> ROS 2 link check')
        print('  ros2 run px4_nmpc_tools nmpc_experiment apply         # walls/obstacles/markers into Gazebo')
        print('  ros2 launch px4_nmpc_bringup nmpc.launch.py           # reads this scenario.json')
        print('  ros2 run px4_nmpc_tools nmpc_mission start')
        return

    if args.command == 'load':
        from px4_nmpc_core.scenario import Scenario, load_scenario_json
        data = dict(load_scenario_json(args.scenario))
        cfg = dict(_load_experiment(args.experiment)) if args.experiment else {}
        cfg.update(dict(data.get('experiment_config') or {}))
        fields = set(Scenario.__dataclass_fields__)
        Scenario(**{k: v for k, v in data.items() if k in fields})          # validates the geometry
        # the complete JSON, not the dataclass: experiment fields such as reference_start_ned (the fixed
        # reference line of the slalom) are not Scenario fields and must reach scenario.yaml
        _write_scenario(data, yaml_path, json_path, cfg)
        print(f'Wrote: {yaml_path}')
        print(f'Wrote: {json_path}')
        _print_summary(data)
        return

    if args.command == 'clear':
        removed = clear_scene(DEFAULT_WORLD)
        print(f'Removed {len(removed)} experiment model(s) from Gazebo.')
        return

    scenario = _load_runtime_scenario(args.runtime_dir)
    if args.command == 'show':
        _print_summary(scenario)
    elif args.command == 'apply':
        apply_scene(scenario, DEFAULT_WORLD)
        print('Arena, obstacles, and start/goal markers applied and verified.')
    elif args.command == 'px4-command':
        start_alt = -float(controller_origin_ned_of(scenario)[2])
        print('cd ~/PX4-Autopilot && \\')
        print(f'PX4_GZ_MODEL_POSE="0,0,{start_alt:.3f},0,0,0" make px4_sitl gz_x500')


if __name__ == '__main__':
    main()
