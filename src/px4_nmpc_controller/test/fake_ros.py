"""Minimal stand-ins for rclpy / ROS message packages.

They let ``test_node_closed_loop.py`` import and drive the *real*
``controller_node.py`` without a ROS 2 installation: timers are called by the
test's simulated clock, publishers record messages, and ``px4_msgs`` classes
are generated from the bundled ``.msg`` files (field names, array sizes and
constants therefore match the real package).
"""
from __future__ import annotations

import pathlib
import re
import sys
import types


class _Logger:
    def __init__(self, sink):
        self.sink = sink

    def _log(self, level, msg, **_):
        self.sink.append((level, str(msg)))

    def info(self, msg, **kw):
        self._log('info', msg, **kw)

    def warning(self, msg, **kw):
        self._log('warn', msg, **kw)

    warn = warning

    def error(self, msg, **kw):
        self._log('error', msg, **kw)


class _Time:
    def __init__(self, ns):
        self.nanoseconds = int(ns)

    def to_msg(self):
        return types.SimpleNamespace(sec=self.nanoseconds // 10**9, nanosec=self.nanoseconds % 10**9)


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return _Time(self.t * 1e9)


CLOCK = FakeClock()


class _Param:
    def __init__(self, value):
        self.value = value


class Node:
    OVERRIDES = {}

    def __init__(self, name):
        self._name = name
        self._params = {}
        self.logs = []
        self.publishers = {}
        self.subscriptions = {}
        self.services = {}
        self.timers = []

    def declare_parameter(self, name, default=None, descriptor=None):
        self._params[name] = self.OVERRIDES.get(name, default)
        return _Param(self._params[name])

    def get_parameter(self, name):
        return _Param(self._params[name])

    def get_logger(self):
        return _Logger(self.logs)

    def get_clock(self):
        return CLOCK

    def create_publisher(self, msg_type, topic, qos):
        pub = types.SimpleNamespace(topic=topic, msgs=[], msg_type=msg_type)
        pub.publish = lambda m, _p=pub: _p.msgs.append(m)
        self.publishers[topic] = pub
        return pub

    def create_subscription(self, msg_type, topic, cb, qos):
        self.subscriptions[topic] = cb
        return cb

    def create_service(self, srv_type, name, cb):
        self.services[name] = cb
        return cb

    def create_timer(self, period, cb):
        timer = types.SimpleNamespace(period=float(period), cb=cb, next=0.0)
        self.timers.append(timer)
        return timer

    def get_topic_names_and_types(self):
        names = sorted(set(self.publishers) | set(self.subscriptions))
        return [(n, ['unknown']) for n in names]

    def count_publishers(self, topic):
        return 1 if topic in self.publishers else 0

    def count_subscribers(self, topic):
        return 1 if topic in self.subscriptions else 0

    def destroy_node(self):
        pass


def _make_msg_class(name, fields, consts):
    def __init__(self):
        for fname, default in fields:
            setattr(self, fname, list(default) if isinstance(default, list) else default)
    cls = type(name, (), {'__init__': __init__})
    for k, v in consts.items():
        setattr(cls, k, v)
    return cls


def _parse_msg(path):
    fields, consts = [], {}
    for raw in path.read_text().splitlines():
        line = raw.split('#', 1)[0].strip()
        if not line:
            continue
        m = re.match(r'^(\w+)(\[(\d*)\])?\s+(\w+)\s*=\s*(.+)$', line)
        if m:
            typ, val = m.group(1), m.group(5).strip()
            consts[m.group(4)] = float(val) if typ.startswith('float') else (val.lower() == 'true' if typ == 'bool' else int(val))
            continue
        m = re.match(r'^(\w+)(\[(\d*)\])?\s+(\w+)$', line)
        if m:
            typ, arr, n, fname = m.group(1), m.group(2), m.group(3), m.group(4)
            base = False if typ == 'bool' else (0.0 if typ.startswith('float') else 0)
            fields.append((fname, [base] * int(n) if arr and n else ([] if arr else base)))
    return fields, consts


def install(px4_msgs_dir):
    """Register fake modules in sys.modules."""
    rclpy = types.ModuleType('rclpy')
    rclpy.init = lambda args=None: None
    rclpy.ok = lambda: True
    rclpy.shutdown = lambda: None
    rclpy.spin = lambda node: None
    node_mod = types.ModuleType('rclpy.node')
    node_mod.Node = Node
    qos = types.ModuleType('rclpy.qos')

    class _Enum:
        BEST_EFFORT = RELIABLE = TRANSIENT_LOCAL = VOLATILE = KEEP_LAST = 0
    qos.QoSProfile = lambda **kw: kw
    qos.ReliabilityPolicy = qos.DurabilityPolicy = qos.HistoryPolicy = _Enum
    executors = types.ModuleType('rclpy.executors')

    class ExternalShutdownException(Exception):
        pass
    executors.ExternalShutdownException = ExternalShutdownException
    rclpy.node = node_mod
    rclpy.qos = qos
    rclpy.executors = executors
    sys.modules.update({'rclpy': rclpy, 'rclpy.node': node_mod, 'rclpy.qos': qos, 'rclpy.executors': executors})

    def simple_pkg(pkg, sub, names):
        top = sys.modules.get(pkg) or types.ModuleType(pkg)
        m = types.ModuleType(f'{pkg}.{sub}')
        for n in names:
            setattr(m, n, type(n, (), {'__init__': lambda self: None}))
        setattr(top, sub, m)
        sys.modules[pkg] = top
        sys.modules[f'{pkg}.{sub}'] = m
        return m

    rcl = simple_pkg('rcl_interfaces', 'msg', [])
    rcl.ParameterDescriptor = lambda **kw: kw
    srv = simple_pkg('std_srvs', 'srv', [])
    srv.Trigger = type('Trigger', (), {})
    diag = simple_pkg('diagnostic_msgs', 'msg', [])

    class KeyValue:
        def __init__(self):
            self.key = ''
            self.value = ''

    class DiagnosticStatus:
        # rclpy maps the IDL ``byte`` type to a length-1 ``bytes`` object
        OK, WARN, ERROR, STALE = b'\x00', b'\x01', b'\x02', b'\x03'

        def __init__(self):
            self.level = b'\x00'
            self.name = self.message = self.hardware_id = ''
            self.values = []

    class DiagnosticArray:
        def __init__(self):
            self.header = types.SimpleNamespace(stamp=None, frame_id='')
            self.status = []
    diag.KeyValue, diag.DiagnosticStatus, diag.DiagnosticArray = KeyValue, DiagnosticStatus, DiagnosticArray
    geo = simple_pkg('geometry_msgs', 'msg', [])

    class PoseStamped:
        def __init__(self):
            self.header = None
            self.pose = types.SimpleNamespace(position=types.SimpleNamespace(x=0.0, y=0.0, z=0.0),
                                              orientation=types.SimpleNamespace(w=1.0, x=0.0, y=0.0, z=0.0))
    geo.PoseStamped = PoseStamped

    class Point:
        def __init__(self):
            self.x = self.y = self.z = 0.0
    geo.Point = Point

    viz = simple_pkg('visualization_msgs', 'msg', [])

    class Marker:
        ARROW, CUBE, SPHERE, CYLINDER, LINE_STRIP, LINE_LIST = 0, 1, 2, 3, 4, 5
        CUBE_LIST, SPHERE_LIST, POINTS, TEXT_VIEW_FACING = 6, 7, 8, 9
        ADD, MODIFY, DELETE, DELETEALL = 0, 0, 2, 3

        def __init__(self):
            self.header = types.SimpleNamespace(stamp=None, frame_id='')
            self.ns = ''
            self.id = 0
            self.type = 0
            self.action = 0
            self.pose = types.SimpleNamespace(position=types.SimpleNamespace(x=0.0, y=0.0, z=0.0),
                                              orientation=types.SimpleNamespace(w=1.0, x=0.0, y=0.0, z=0.0))
            self.scale = types.SimpleNamespace(x=0.0, y=0.0, z=0.0)
            self.color = types.SimpleNamespace(r=0.0, g=0.0, b=0.0, a=0.0)
            self.points = []
            self.text = ''

    class MarkerArray:
        def __init__(self):
            self.markers = []
    viz.Marker, viz.MarkerArray = Marker, MarkerArray
    nav = simple_pkg('nav_msgs', 'msg', [])

    class Path:
        def __init__(self):
            self.header = types.SimpleNamespace(stamp=None, frame_id='')
            self.poses = []
    nav.Path = Path

    px4 = simple_pkg('px4_msgs', 'msg', [])
    for f in sorted(pathlib.Path(px4_msgs_dir).glob('*.msg')):
        fields, consts = _parse_msg(f)
        setattr(px4, f.stem, _make_msg_class(f.stem, fields, consts))
    return CLOCK
