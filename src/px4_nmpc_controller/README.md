# px4_nmpc_controller

Connects the shared NMPC runtime to ROS 2 and PX4. The ROS entry point remains
`ros2 run px4_nmpc_controller nmpc_controller`.

| Directory/module | Responsibility |
| --- | --- |
| `controller_node.py` | ROS callbacks, timers, diagnostics, and PX4 setpoints |
| `px4_nmpc_ipopt/` | CasADi solver implementation |
| `px4_nmpc_ipopt_casadi/` | Public CasADi backend wrapper |
| `px4_nmpc_ipopt_cpp/` | Native IPOPT wrapper and C++ formulation |
| `px4_nmpc_cg2o/` | cg2o wrapper and C++ graph formulation |

Paths in this table are relative to the inner `px4_nmpc_controller/` Python
package. Native sources stay under each backend's `cpp/` directory.

Run `make test` from the repository root for the
fake-ROS integration suite. Build native extensions with
`./scripts/build_backends.sh`; verify them with
`python3 scripts/smoke_test_backends.py`.
The benchmark of the paper uses `challenging_solver.py`; see [docs/1_problem.md](../../docs/1_problem.md).
