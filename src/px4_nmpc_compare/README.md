# px4_nmpc_compare

Problem data and solver comparison of the thrust-and-rate quadrotor MPC.

| Module | Role |
| --- | --- |
| `challenging.py` | course and vehicle parameters, problem data of one MPC step, SHA-256 hash of every problem, cold-start rollout, loading of the native extension `_challenging` |
| `challenging_replay.py` | paired replay of one recorded flight: rebuilds every problem, checks its hash against the flight and solves it with cg2o and with IPOPT, one after the other, each with its own warm-start history |
| `challenging_tune.py` | objective and quality gates of the equal-budget Bayesian tuning (`python -m tuning run ...`) |
| `common.py` | run context (commit, threads, CPU affinity) written with every result |
| `study_figures.py` | obstacle geometry and clearance helpers used by the flight figure of `px4_nmpc_tools.sitl_video` |

```bash
taskset -c 0-3 python -m px4_nmpc_compare.challenging_replay <flight> <flight>/replay \
    --ipopt-params <selected_parameters.json>
```

See [docs/2_protocol.md](../../docs/2_protocol.md) for what is compared and how it is timed.
