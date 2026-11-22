# px4_nmpc_bringup

ROS launch descriptions and RViz configuration. Launch names remain unchanged:

```bash
ros2 launch px4_nmpc_bringup nmpc.launch.py
ros2 launch px4_nmpc_bringup rviz.launch.py
```

`launch/` owns process startup and launch arguments; `config/nmpc.rviz` owns the
RViz display setup. Controller and experiment parameters belong in
`px4_nmpc_config`, rather than in copied launch files.

See
[installation](../../INSTALL.md) for ROS/PX4 setup.
