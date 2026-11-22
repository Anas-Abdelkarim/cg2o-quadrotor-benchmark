"""Controller + visualizer + monitor + RViz for the PX4 NMPC experiment.

    ros2 launch px4_nmpc_bringup nmpc.launch.py                    # everything
    ros2 launch px4_nmpc_bringup nmpc.launch.py launch_rviz:=false
    ros2 launch px4_nmpc_bringup nmpc.launch.py control_interface:=attitude
    ros2 launch px4_nmpc_bringup nmpc.launch.py yaw_mode:=optimized   # NMPC plans the heading too

Geometry: the controller and the visualizer both read ``scenario_json``
(default ``~/.ros/px4_nmpc/scenario.json``, written by ``nmpc_experiment new``
and used by ``nmpc_experiment apply`` for Gazebo).  If it does not exist they
both fall back to the packaged example scenario.  ``scenario_json:=none`` flies
the geometry in nmpc.yaml (+ optional ``scenario_file``) instead.

Only the controller matters for flight: the visualizer, the monitor and RViz
may fail without affecting it.
"""
import os
import shutil

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

try:
    from px4_nmpc_core.px4_interface import sanitize_gui_env
except ImportError:  # keep the launch usable even if the core package is broken
    def sanitize_gui_env(env=None):
        return dict(os.environ if env is None else env), []


def _truthy(context, name):
    return LaunchConfiguration(name).perform(context).strip().lower() in ('1', 'true', 'yes', 'on')


def _controller(context, base_config, fallback_json):
    params = [base_config]
    extra = LaunchConfiguration('extra_params_file').perform(context).strip()
    if extra and extra.lower() != 'none':
        params.append(os.path.expanduser(extra))     # e.g. an experiment bundle's nmpc.yaml (slalom overrides)
    scenario_file = LaunchConfiguration('scenario_file').perform(context).strip()
    if scenario_file and scenario_file.lower() != 'none':
        params.append(os.path.expanduser(scenario_file))
    params.append({
        'scenario_json': os.path.expanduser(LaunchConfiguration('scenario_json').perform(context).strip()),
        'fallback_scenario_json': fallback_json,
    })
    overrides = {}
    for key in ('control_interface', 'solver_backend', 'yaw_mode'):
        value = LaunchConfiguration(key).perform(context).strip()
        if value:
            overrides[key] = value
    if LaunchConfiguration('control_interface').perform(context).strip() in ('attitude', 'body_rate'):
        # the attitude / body-rate models are more expensive per stage
        overrides.setdefault('horizon_steps', int(LaunchConfiguration('horizon_steps_attitude').perform(context)))
    if overrides:
        params.append(overrides)
    return [Node(
        package='px4_nmpc_controller',
        executable='nmpc_controller',
        name='px4_nmpc_controller',
        output='screen',
        emulate_tty=True,
        parameters=params,
    )]


def _visualizer(context, base_config, fallback_json):
    if not _truthy(context, 'launch_visualizer'):
        return []
    scenario_file = LaunchConfiguration('scenario_file').perform(context).strip()
    return [Node(
        package='px4_nmpc_tools',
        executable='nmpc_visualizer',
        name='nmpc_visualizer',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'scenario_json': os.path.expanduser(LaunchConfiguration('scenario_json').perform(context).strip()),
            'nmpc_config_yaml': base_config,
            'scenario_yaml': os.path.expanduser(scenario_file) if scenario_file.lower() != 'none' else '',
            'fallback_scenario_json': fallback_json,
        }],
    )]


def _rviz(context):
    if not _truthy(context, 'launch_rviz'):
        return []
    exe = shutil.which('rviz2')
    if exe is None:
        return [LogInfo(msg='[nmpc.launch] rviz2 not found; controller/visualizer keep running. '
                            'Install it with: sudo apt install ros-humble-rviz2')]
    actions = []
    env, removed = sanitize_gui_env()
    if removed:
        # A terminal opened inside a snap application (VS Code/PyCharm snap)
        # exports GTK_PATH, GIO_MODULE_DIR, ... into /snap/.  rviz2 then loads
        # /snap/core20/.../libpthread.so.0 and exits with
        # "undefined symbol: __libc_pthread_init, version GLIBC_PRIVATE".
        actions.append(LogInfo(msg='[nmpc.launch] Starting RViz without snap-injected variables: '
                                   + ', '.join(removed)))
    gui_available = bool(env.get('DISPLAY') or env.get('WAYLAND_DISPLAY'))
    if not gui_available:
        actions.append(LogInfo(msg='[nmpc.launch] DISPLAY/WAYLAND_DISPLAY not set: RViz needs a graphical session. '
                                   'Auto-restart is disabled to avoid a crash loop.'))
    respawn = _truthy(context, 'rviz_respawn') and gui_available
    if respawn:
        actions.append(LogInfo(msg='[nmpc.launch] RViz auto-restart is enabled: close the RViz window to restart it.'))
    actions.append(Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', os.path.expanduser(LaunchConfiguration('rviz_config').perform(context))],
        env=env,
        respawn=respawn,
        respawn_delay=1.0,
    ))
    return actions


def generate_launch_description():
    config_dir = os.path.join(get_package_share_directory('px4_nmpc_config'), 'config')
    bringup_dir = get_package_share_directory('px4_nmpc_bringup')
    base_config = os.path.join(config_dir, 'nmpc.yaml')
    fallback_json = os.path.join(config_dir, 'scenario_example.json')
    runtime_json = os.path.join(os.path.expanduser('~'), '.ros', 'px4_nmpc', 'scenario.json')

    args = [
        DeclareLaunchArgument('scenario_json', default_value=runtime_json,
                              description='Scenario used by controller AND visualizer (none = use ROS params)'),
        DeclareLaunchArgument('extra_params_file', default_value='none',
                              description='additional controller YAML applied after nmpc.yaml (experiment overrides)'),
        DeclareLaunchArgument('scenario_file', default_value='none',
                              description='Optional extra ROS parameter YAML for the controller'),
        DeclareLaunchArgument('control_interface', default_value='',
                              description='trajectory | attitude | body_rate (empty = value in nmpc.yaml)'),
        DeclareLaunchArgument('solver_backend', default_value='',
                              description='ipopt_casadi | ipopt_cpp | cg2o (legacy: ipopt | fatrop; empty = nmpc.yaml)'),
        DeclareLaunchArgument('yaw_mode', default_value='',
                              description='fixed | optimized (NMPC plans the yaw; trajectory interface; empty = nmpc.yaml)'),
        DeclareLaunchArgument('horizon_steps_attitude', default_value='25',
                              description='horizon used when control_interface is attitude/body_rate'),
        DeclareLaunchArgument('launch_visualizer', default_value='true'),
        DeclareLaunchArgument('launch_monitor', default_value='true'),
        DeclareLaunchArgument('launch_rviz', default_value='true'),
        DeclareLaunchArgument('rviz_respawn', default_value='true',
                              description='Restart RViz automatically if its process exits'),
        DeclareLaunchArgument('rviz_config', default_value=os.path.join(bringup_dir, 'config', 'nmpc.rviz')),
    ]

    monitor = Node(
        package='px4_nmpc_tools',
        executable='nmpc_monitor',
        name='nmpc_monitor',
        output='screen',
        emulate_tty=True,
        condition=IfCondition(LaunchConfiguration('launch_monitor')),
    )
    return LaunchDescription(args + [
        OpaqueFunction(function=_controller, args=[base_config, fallback_json]),
        OpaqueFunction(function=_visualizer, args=[base_config, fallback_json]),
        monitor,
        OpaqueFunction(function=_rviz),
    ])
