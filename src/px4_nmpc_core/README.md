# px4_nmpc_core

ROS-independent dynamics, planning, guidance, mission runtime, numerical scoring,
scenario geometry, and the PX4-like simulator.

Python modules live in `px4_nmpc_core/`; regression tests live in `test/`.
`paths.py` provides shared source/configuration discovery. `ocp.py` lazily loads
solver implementations from `px4_nmpc_controller` when constructing a solver.

From the repository root, run `make test`.
