# px4_nmpc_tools

Mission commands, scenario management, visualization, monitoring, diagnosis,
offline simulation, benchmarking, and video capture.

Python entry points are declared in `setup.py`. The offline simulator and
benchmark commands work from the source tree without ROS; live mission,
visualization, and PX4 commands need their ROS/system dependencies.

Run `make test` from the repository root; the tools tests install fake ROS
modules and run in their own pytest process. `sitl_video` runs the PX4 SITL
flights (see [docs/3_reproduce.md](../../docs/3_reproduce.md)).
