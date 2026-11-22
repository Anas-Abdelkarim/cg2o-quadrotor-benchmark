# Configuration layout

The legacy flat files in this directory remain supported so existing launch and
benchmark commands do not break.

New solver experiments should be self-contained under:

```text
config/experiments/<experiment_name>/
```

Each experiment should contain a `bundle.yaml` that records the mathematical
`formulation` and relative paths to its NMPC overrides, scenario, generator and
comparison settings. This separates three concepts:

```text
formulation = the mathematical OCP (trajectory, future attitude, ...)
backend     = the numerical implementation (ipopt_casadi, ipopt_cpp, cg2o)
experiment  = geometry/reference/initialization used to test that formulation
```

Do not duplicate controller code merely to add another scenario. A new backend
implementation is needed only when the mathematical formulation changes.

See `experiments/infeasible_slalom/README.md` for the first bundled experiment.
