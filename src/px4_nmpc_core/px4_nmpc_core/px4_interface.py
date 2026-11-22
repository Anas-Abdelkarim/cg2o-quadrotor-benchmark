"""ROS-free helpers for talking to PX4 over uXRCE-DDS.

Everything here is plain Python so it can be unit-tested without ROS 2 and
shared by the controller, the visualizer, the doctor and the launch file.

Topic names (PX4 >= v1.16)
--------------------------
PX4's uXRCE-DDS client appends ``_v<MESSAGE_VERSION>`` to the DDS topic name of
every *versioned* message whose version is greater than zero
(``src/modules/uxrce_dds_client/utilities.hpp: generate_topic_name`` in
PX4 v1.17.0).  With the px4_msgs bundled here (identical to PX4 v1.17.0):

=====================================  ===========================================
message (MESSAGE_VERSION)              DDS topic used by PX4 v1.17
=====================================  ===========================================
VehicleStatus (1)                      ``/fmu/out/vehicle_status_v1``
VehicleAttitudeSetpoint (1)            ``/fmu/in/vehicle_attitude_setpoint_v1``
VehicleOdometry (0)                    ``/fmu/out/vehicle_odometry``
TrajectorySetpoint (0)                 ``/fmu/in/trajectory_setpoint``
VehicleRatesSetpoint (0)               ``/fmu/in/vehicle_rates_setpoint``
VehicleCommand (0)                     ``/fmu/in/vehicle_command``
OffboardControlMode (unversioned)      ``/fmu/in/offboard_control_mode``
=====================================  ===========================================

Subscribing to ``/fmu/out/vehicle_status`` (without ``_v1``) therefore receives
nothing from PX4 v1.16/v1.17.
"""
from __future__ import annotations

import os

# 'auto' reproduces PX4's own rule and is the only correct choice for PX4
# v1.16+ with matching px4_msgs.  'never' is for PX4 <= v1.15 (no versioning).
TOPIC_VERSION_MODES = ('auto', 'never')
_TRUE = ('true', '1', 'yes', 'on', 'auto', 'versioned')
_FALSE = ('false', '0', 'no', 'off', 'never', 'unversioned', 'legacy')


def topic_version_mode(value) -> str:
    """Normalise the ``px4_versioned_topics`` parameter to 'auto' or 'never'."""
    text = str(value).strip().lower()
    if text in _TRUE:
        return 'auto'
    if text in _FALSE:
        return 'never'
    raise ValueError(f"px4_versioned_topics must be 'auto' or 'never' (got {value!r})")


def message_version(msg_cls) -> int:
    """``MESSAGE_VERSION`` constant of a px4_msgs class (0 if it has none)."""
    try:
        return int(getattr(msg_cls, 'MESSAGE_VERSION', 0) or 0)
    except (TypeError, ValueError):
        return 0


def px4_topic_name(prefix: str, direction: str, base: str, version: int = 0, mode: str = 'auto') -> str:
    """Exact DDS topic name PX4 uses, e.g. ``/fmu/out/vehicle_status_v1``."""
    if direction not in ('in', 'out'):
        raise ValueError("direction must be 'in' or 'out'")
    name = f'/{str(prefix).strip("/")}/{direction}/{base}'
    if topic_version_mode(mode) == 'auto' and int(version) > 0:
        name += f'_v{int(version)}'
    return name


def px4_topic(prefix: str, direction: str, base: str, msg_cls, mode: str = 'auto') -> str:
    return px4_topic_name(prefix, direction, base, message_version(msg_cls), mode)


def constant_names(msg_cls, prefix: str) -> dict:
    """{value: NAME} for integer constants such as NAVIGATION_STATE_* ."""
    out = {}
    for key in dir(msg_cls):
        if key.startswith(prefix):
            value = getattr(msg_cls, key)
            if isinstance(value, int) and not isinstance(value, bool):
                out.setdefault(value, key[len(prefix):])
    return out


def active_flag_names(msg, skip_prefixes=('mode_req_',), skip=('timestamp',),
                      level_fields=('battery_warning',)) -> list:
    """Names of all boolean fields that are True (e.g. of a FailsafeFlags message).

    Integer "level" fields such as ``battery_warning`` are reported as
    ``name=value`` when non-zero.
    """
    names = []
    fields = getattr(msg, '_fields_and_field_types', None)
    keys = list(fields.keys()) if isinstance(fields, dict) else [k for k in vars(msg) if not k.startswith('_')]
    for key in keys:
        key = key.lstrip('_')
        if key in skip or key.startswith(skip_prefixes):
            continue
        value = getattr(msg, key, None)
        if isinstance(value, bool):
            if value:
                names.append(key)
        elif key in level_fields and isinstance(value, int) and value:
            names.append(f'{key}={value}')
    return names


# ---------------------------------------------------------------- diagnostics
def diagnostic_level(level) -> int:
    """``diagnostic_msgs/DiagnosticStatus.level`` as an int.

    In rclpy the ``byte`` field is a length-1 ``bytes`` object (b'\\x02'), so
    ``int(level)`` raises ``ValueError``.  Accept bytes, ints and digit strings.
    """
    if isinstance(level, (bytes, bytearray)):
        return int(level[0]) if len(level) else 0
    if isinstance(level, str):
        text = level.strip()
        if text.startswith(("b'", 'b"')):          # str(b'\x02') from older tools
            try:
                import ast
                return diagnostic_level(ast.literal_eval(text))
            except (ValueError, SyntaxError):
                return 0
        try:
            return int(float(text))
        except ValueError:
            return 0
    try:
        return int(level)
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------- GUI environment
# Snap-packaged applications (VS Code, PyCharm, ... installed as snaps) export
# variables such as GTK_PATH, GIO_MODULE_DIR, LOCPATH or XDG_DATA_DIRS that point
# into /snap/.  A terminal opened inside such an application passes them on to
# rviz2 / gz sim, the dynamic loader then picks up the snap's
# /snap/core20/current/lib/x86_64-linux-gnu/libpthread.so.0 (glibc 2.31) and
# the process dies immediately with
#   symbol lookup error: ... undefined symbol: __libc_pthread_init, version GLIBC_PRIVATE
_ORIG_SUFFIX = '_VSCODE_SNAP_ORIG'   # VS Code saves the pre-snap value here


def sanitize_gui_env(env=None):
    """Return ``(clean_env, removed_names)`` with every /snap/ injection removed.

    * scalar variables whose value points into /snap/ are dropped,
    * ':'-separated lists keep only their non-snap entries,
    * ``<NAME>_VSCODE_SNAP_ORIG`` values are restored when present,
    * ``PATH`` is never modified.

    A normal (non-snap) desktop environment is returned unchanged.
    """
    src = dict(os.environ if env is None else env)
    out, removed = {}, []
    for key, value in src.items():
        if key.endswith(_ORIG_SUFFIX):
            continue
        if key == 'PATH' or '/snap/' not in str(value):
            out[key] = value
            continue
        removed.append(key)
        if ':' in value and not key.startswith('SNAP'):
            kept = [p for p in value.split(':') if p and '/snap/' not in p]
            if kept:
                out[key] = ':'.join(kept)
    for key, value in src.items():
        if key.endswith(_ORIG_SUFFIX):
            base = key[:-len(_ORIG_SUFFIX)]
            if base in removed and value and '/snap/' not in value:
                out[base] = value
    if 'XDG_DATA_DIRS' in src and not out.get('XDG_DATA_DIRS'):
        out['XDG_DATA_DIRS'] = '/usr/local/share:/usr/share'
    return out, sorted(set(removed))


def snap_contamination(env=None) -> list:
    """Names of variables that would break rviz2/Gazebo GUIs (empty = clean)."""
    return sanitize_gui_env(env)[1]
