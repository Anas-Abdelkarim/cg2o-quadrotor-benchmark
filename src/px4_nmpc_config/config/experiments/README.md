# Experiment bundles

Experiments are intentionally separated from controller formulations and solver backends.

```text
formulation = mathematical OCP (currently `trajectory`; future examples may add `attitude`, `rigid_body`, ...)
backend     = numerical implementation (`ipopt_casadi`, `ipopt_cpp`, `cg2o`)
experiment  = scenario/reference/initialization used to exercise a formulation
```

A scenario change does **not** justify copying `px4_nmpc_controller`.  Add an experiment
bundle under this directory.  Add backend formulation code only when the OCP itself
changes.

Current bundles:

- `challenging_course/`: the benchmark of the paper: course, cg2o overrides and the
  four tuning specifications.
- `infeasible_slalom/`: base flight configuration (arena, take-off, RViz) that the
  benchmark's PX4 launcher builds on; ten-obstacle, high-speed, deliberately infeasible straight-line
  initialization; Level 2 uses shifted NMPC warm starts after the first solve.

Every bundle has one `bundle.yaml` entry point.  Paths inside it are relative to the bundle
folder, making paper experiments reproducible and easy to archive with their exact configs.
