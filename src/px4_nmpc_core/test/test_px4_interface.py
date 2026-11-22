"""PX4 interface helpers: topic naming, diagnostic levels, GUI environment."""
import pathlib
import re

import pytest

from px4_nmpc_core.px4_interface import (
    active_flag_names,
    constant_names,
    diagnostic_level,
    px4_topic_name,
    sanitize_gui_env,
    snap_contamination,
    topic_version_mode,
)

MSG_DIR = pathlib.Path(__file__).resolve().parents[2] / 'px4_msgs' / 'msg'


def _version(name):
    m = re.search(r'MESSAGE_VERSION\s*=\s*(\d+)', (MSG_DIR / f'{name}.msg').read_text())
    return int(m.group(1)) if m else 0


def test_topic_names_match_px4_v117_rule():
    # PX4 v1.17 utilities.hpp: generate_topic_name appends _v<N> for MESSAGE_VERSION != 0
    assert _version('VehicleStatus') == 1
    assert _version('VehicleAttitudeSetpoint') == 1
    assert px4_topic_name('/fmu', 'out', 'vehicle_status', _version('VehicleStatus')) == '/fmu/out/vehicle_status_v1'
    assert px4_topic_name('/fmu', 'in', 'vehicle_attitude_setpoint', 1) == '/fmu/in/vehicle_attitude_setpoint_v1'
    for base, name in (('vehicle_odometry', 'VehicleOdometry'), ('trajectory_setpoint', 'TrajectorySetpoint'),
                       ('vehicle_command', 'VehicleCommand'), ('vehicle_rates_setpoint', 'VehicleRatesSetpoint'),
                       ('offboard_control_mode', 'OffboardControlMode')):
        assert px4_topic_name('/fmu', 'in', base, _version(name)) == f'/fmu/in/{base}'
    assert px4_topic_name('fmu/', 'out', 'vehicle_status', 1, 'never') == '/fmu/out/vehicle_status'
    assert px4_topic_name('/px4_1/fmu', 'out', 'vehicle_status', 1) == '/px4_1/fmu/out/vehicle_status_v1'


@pytest.mark.parametrize('value,mode', [('auto', 'auto'), ('true', 'auto'), (True, 'auto'),
                                        ('false', 'never'), (False, 'never'), ('never', 'never')])
def test_topic_version_mode(value, mode):
    assert topic_version_mode(value) == mode


def test_topic_version_mode_rejects_garbage():
    with pytest.raises(ValueError):
        topic_version_mode('sometimes')


@pytest.mark.parametrize('raw,expected', [(b'\x00', 0), (b'\x01', 1), (b'\x02', 2), (bytearray(b'\x03'), 3),
                                          (2, 2), ('1', 1), ("b'\\x02'", 2), (b'', 0), (None, 0)])
def test_diagnostic_level(raw, expected):
    assert diagnostic_level(raw) == expected


def test_int_of_rclpy_byte_level_is_the_v201_monitor_crash():
    with pytest.raises(ValueError):
        int(b'\x02')


def test_constant_and_flag_names():
    class Msg:
        NAVIGATION_STATE_OFFBOARD = 14
        NAVIGATION_STATE_AUTO_RTL = 5
        OTHER = 3

        def __init__(self):
            self.timestamp = 1
            self.mode_req_offboard_signal = 7
            self.offboard_control_signal_lost = True
            self.gcs_connection_lost = False
            self.battery_warning = 2
    assert constant_names(Msg, 'NAVIGATION_STATE_') == {14: 'OFFBOARD', 5: 'AUTO_RTL'}
    assert active_flag_names(Msg()) == ['offboard_control_signal_lost', 'battery_warning=2']


def test_sanitize_gui_env_removes_snap_injection():
    env = {
        'PATH': '/snap/bin:/usr/bin',
        'DISPLAY': ':0',
        'GTK_PATH': '/snap/code/190/usr/lib/x86_64-linux-gnu/gtk-3.0',
        'GIO_MODULE_DIR': '/home/u/snap/code/common/.cache/gio-modules',
        'LOCPATH': '/snap/code/190/usr/lib/locale',
        'LD_LIBRARY_PATH': '/opt/ros/humble/lib:/snap/core20/current/lib/x86_64-linux-gnu',
        'XDG_DATA_DIRS': '/home/u/snap/code/190/.local/share:/snap/code/190/usr/share',
        'XDG_DATA_DIRS_VSCODE_SNAP_ORIG': '/usr/share/ubuntu:/usr/local/share:/usr/share',
        'SNAP': '/snap/code/190',
        'SNAP_NAME': 'code',
        'ROS_DISTRO': 'humble',
    }
    clean, removed = sanitize_gui_env(env)
    assert clean['PATH'] == env['PATH']
    assert clean['LD_LIBRARY_PATH'] == '/opt/ros/humble/lib'
    assert clean['XDG_DATA_DIRS'] == '/usr/share/ubuntu:/usr/local/share:/usr/share'
    for key in ('GTK_PATH', 'GIO_MODULE_DIR', 'LOCPATH', 'SNAP'):
        assert key not in clean and key in removed
    assert clean['DISPLAY'] == ':0' and clean['ROS_DISTRO'] == 'humble'
    assert not any('/snap/' in v for k, v in clean.items() if k != 'PATH')


def test_sanitize_gui_env_leaves_normal_desktop_unchanged():
    env = {'PATH': '/usr/bin', 'DISPLAY': ':0', 'XDG_DATA_DIRS': '/usr/local/share:/usr/share',
           'LD_LIBRARY_PATH': '/opt/ros/humble/lib'}
    clean, removed = sanitize_gui_env(env)
    assert clean == env and removed == []
    assert snap_contamination(env) == []
