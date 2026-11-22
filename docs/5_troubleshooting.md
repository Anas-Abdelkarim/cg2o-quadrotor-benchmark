# 5. Troubleshooting

Build and installation problems: see the table at the end of [INSTALL.md](../INSTALL.md#troubleshooting). This page
covers the PX4 software-in-the-loop flights (`make tuning-flights`, `make flights`, `make video`,
`run_experiment.py full`). Every flight writes its console output to `<flight>.log` next to the flight folder, and
the PX4 log to `<flight>/flight.ulg`.

## RViz dies at start: `symbol lookup error ... /snap/core20/.../libpthread.so.0`

The terminal was opened inside a snap-packaged application (for example VS Code installed as a snap). Such
terminals export `GTK_PATH`, `GIO_MODULE_DIR`, `LOCPATH`, ... pointing into `/snap/`; Qt then loads a GTK module
built for an older glibc, and RViz exits with code 127. The launcher starts RViz with these variables removed. Other
GUIs started from the same terminal (`gz sim`, a manual `rviz2`) need a normal terminal or

```bash
unset GTK_PATH GTK_EXE_PREFIX GTK_IM_MODULE_FILE GIO_MODULE_DIR GSETTINGS_SCHEMA_DIR LOCPATH
```

## The vehicle never arms, or PX4 reports `Failsafe activated`

* PX4 v1.17 publishes versioned topics (`/fmu/out/vehicle_status_v1`). The controller follows PX4's naming rule
  automatically; check with `ros2 topic list | grep vehicle_status`. If you override the controller parameters in
  your own YAML, do not set `px4_versioned_topics: false`.
* The controller logs the reason from `/fmu/out/failsafe_flags`. `offboard_control_signal_lost` means the setpoint
  stream stopped: the controller crashed or the machine stalled for more than PX4's `COM_OF_LOSS_T` (1 s), for
  example because other heavy work ran on the CPUs of the MPC node (`CONTROLLER_CPUS`).
* `tone_alarm: battery warning` right after `Failsafe activated` is PX4's generic failsafe tune, not a low battery.
* PX4 rejects arming right after start-up while its estimator converges. The controller keeps requesting OFFBOARD
  and arming for up to `arming_timeout_s` (15 s) and then aborts the flight; the pre-arm reason is in the PX4 part
  of `<flight>.log`.

## No `/fmu/...` topics, or the controller waits for odometry

PX4 and the Micro XRCE-DDS agent must both run, in the same `ROS_DOMAIN_ID` as the controller. The launcher starts
the agent (`MicroXRCEAgent`, on the `PATH` after `source scripts/setup_env.sh`); if it is missing, run
`./scripts/build_externals.sh --only=px4,xrce`. Gazebo needs PX4's own setup once
(`bash externals/PX4-Autopilot/Tools/setup/ubuntu.sh`, run by the build script; log out and in afterwards).

## The flight reaches the goal but the times are much larger than in the paper

Solve times depend on the CPU and on what else runs on it. The paper ran the MPC node on CPUs 2 and 3, isolated from
the scheduler (`isolcpus=0-3` on the kernel command line), with nothing else on them. Choose free CPUs with
`CONTROLLER_CPUS` (flights) and `CPUS` (replay, tuning), and run one timed step at a time. A machine that suspends
during a replay leaves one inflated solve time in that flight: replay that flight again.

## The video shows desktop notifications or other windows

`make video` records the RViz and Gazebo windows from the screen. Turn notifications off for the recording, for
example on GNOME: `gsettings set org.gnome.desktop.notifications show-banners false` (and `true` afterwards).

## cg2o crashes with a segmentation fault in the first solve

g2o, cg2o and the extensions were compiled with different `-march` settings, so Eigen's memory alignment differs.
Rebuild all of them with the same setting: `MARCH_NATIVE=ON ./scripts/build_externals.sh --solver-only --clean`,
then `./scripts/build_benchmark.sh`.
