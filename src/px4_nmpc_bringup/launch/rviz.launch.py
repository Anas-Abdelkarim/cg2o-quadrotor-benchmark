"""Standalone RViz launcher for the PX4 NMPC visualization.

Use this when the main NMPC launch was started with ``launch_rviz:=false`` or
when you want RViz in its own terminal.  It uses the same snap-environment
sanitization as ``nmpc.launch.py`` and auto-respawns by default.
"""
import os
import shutil

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

try:
    from px4_nmpc_core.px4_interface import sanitize_gui_env
except ImportError:
    def sanitize_gui_env(env=None):
        return dict(os.environ if env is None else env), []


def _truthy(context, name):
    return LaunchConfiguration(name).perform(context).strip().lower() in ('1', 'true', 'yes', 'on')


def _rviz(context):
    exe = shutil.which('rviz2')
    if exe is None:
        return [LogInfo(msg='[rviz.launch] rviz2 not found. Install it with: sudo apt install ros-humble-rviz2')]

    env, removed = sanitize_gui_env()
    actions = []
    if removed:
        actions.append(LogInfo(msg='[rviz.launch] Starting RViz without snap-injected variables: ' + ', '.join(removed)))
    gui_available = bool(env.get('DISPLAY') or env.get('WAYLAND_DISPLAY'))
    if not gui_available:
        actions.append(LogInfo(msg='[rviz.launch] DISPLAY/WAYLAND_DISPLAY not set: RViz needs a graphical session. '
                                   'Auto-restart is disabled to avoid a crash loop.'))

    respawn = _truthy(context, 'respawn') and gui_available
    if respawn:
        actions.append(LogInfo(msg='[rviz.launch] Auto-restart enabled: close the RViz window to restart it.'))
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
    bringup_dir = get_package_share_directory('px4_nmpc_bringup')
    return LaunchDescription([
        DeclareLaunchArgument(
            'rviz_config',
            default_value=os.path.join(bringup_dir, 'config', 'nmpc.rviz'),
            description='RViz config file',
        ),
        DeclareLaunchArgument(
            'respawn',
            default_value='true',
            description='Restart RViz automatically if its process exits',
        ),
        OpaqueFunction(function=_rviz),
    ])
