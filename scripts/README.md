# Scripts

Run from the repository root. Builds go through these scripts; experiments through `experiments/run_experiment.py`.

| Task | Command |
| --- | --- |
| Third-party solvers: IPOPT, HSL, g2o, cg2o (pinned versions) | `HSL_SOURCE=<coinhsl> MARCH_NATIVE=ON ./scripts/build_externals.sh --solver-only` |
| PX4 SITL and the XRCE-DDS agent | `./scripts/build_externals.sh --only=px4,xrce` |
| One component | `./scripts/build_externals/{ipopt,hsl,g2o,cg2o,px4,xrce}.sh [--clean]` |
| This repository: backends, benchmark extension, ROS 2 workspace | `./scripts/build_benchmark.sh [--no-ros]` |
| Native solver backends only | `./scripts/build_backends.sh --clean` |
| ROS 2 packages only | `./scripts/build_workspace.sh` |
| Python source paths without ROS | `source scripts/source_env.sh` |
| Full environment (ROS 2, PX4, built packages) | `source scripts/setup_env.sh` |
| One real solve with each backend | `python3 scripts/smoke_test_backends.py` |
| Clean colcon output, keep external and native builds | `./scripts/clean_colcon_artifacts.sh` |
| Rebuild against a local cg2o checkout | `CG2O_DIR=/absolute/path ./scripts/rebuild_cg2o.sh` |
| Check the build environment | `bash scripts/check_build_env.sh` |

`build_cg2o_backend.sh`, `build_ipopt_cpp_backend.sh` and `build_native_backends.sh` are wrappers
around `build_backends.sh`. Every build script stops with a non-zero exit code when a step fails.
