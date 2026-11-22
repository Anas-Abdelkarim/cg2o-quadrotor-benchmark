# Solver tuning

A Python package (`tuning`) for tuning expensive simulations, optimizers and external programs under hard quality
constraints. In this repository it runs the four equal-budget tuning studies of the quadrotor benchmark (cg2o and
IPOPT, warm and cold); their specifications are in
`src/px4_nmpc_config/config/experiments/challenging_course/tuning/` and the experiment adapter in
`src/px4_nmpc_compare/px4_nmpc_compare/challenging_tune.py` (see the [project quickstart](../../QUICKSTART.md)).

The search engine has no dependency on IPOPT, cg2o, NumPy or plotting libraries. Install it with
`python3 -m pip install -e tools/solver_tuning[bo]` (the project's `requirements.txt` does this).

## Start in five minutes

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -e '.[bo,plots,dev]'
PYTHONPATH=examples python -m tuning run examples/toy.yaml --out results/toy --trials 30
python -m tuning status results/toy
python -m tuning plot results/toy
```

The toy runs an analytic experiment with constraints and an additional validation case; it needs no solver.
For Bayesian optimization set `method: tpe` before starting a new study (Optuna 4.x). Search never promises a
global optimum.

## Features

- Python API and CLI; Python-function and external-command adapters.
- Random search, coordinate search, conditional multivariate TPE and differential evolution.
- Mixed integer/categorical/logarithmic domains, optional settings, conditions and cross-parameter relations.
- Identical quality gates for the baseline and the candidates, checked per run and repeat.
- Validation covers the union of the screening and the extra validation cases; only configurations that passed
  screening are validated.
- An invalid baseline can normalize measurements but cannot be selected; without a validated feasible
  configuration, no best configuration is exported.
- Exact inputs, commands, resolved settings and raw measurements are kept.
- Transactional SQLite history, process locking and compatible resume.
- `jobs` is the maximum number of simultaneous experiment calls; cases and repeats inside one evaluation run in
  sequence.
- Failed trials are infeasible observations with infinite objective for TPE, not a finite penalty.

## Resume and budgets

```bash
PYTHONPATH=examples python -m tuning run examples/toy.yaml --out results/toy --trials 100 --resume
```

A new run prefixes the folder name with its start time (`results/toy` → `results/20261027_153000_toy`);
`--resume` and `--finish` take the stamped folder as it is. `--trials` is the **total** search budget, including
earlier attempts; baseline and finalist validation are additional. Cached proposals and combinations rejected
before evaluation still consume an attempt. Interrupted requests are marked `abandoned` on resume.

Resume checks a manifest with package source hashes, Python and dependency versions, experiment fingerprint,
baseline, search domains, method, constraints, objective, cases, repeats and seed; if any differs, start a new
study folder. `--max-seconds` limits the scheduling of new search batches. `--screen-only` writes a report but never
exports a validated configuration; `--finish` validates without further search. Ctrl-C (or SIGTERM) stops new
work, waits for running calls and keeps all completed records.

## Outputs

| File | Purpose |
| --- | --- |
| `manifest.json` | study identity and resume guard |
| `trials.sqlite3` | live request/result journal |
| `trials.csv` | live table, one row per evaluation (command in line 1); rewritten at exit |
| `log.jsonl` | full-detail export when the study exits |
| `artifacts/trial-.../` | per-run input, output, metrics, commands and logs |
| `report.json`, `report.txt` | final status and selected validation measurement |
| `best_config.json`, `selected_parameters.json` | validated parameters, only when a winner exists |
| `search_progress.png` | optional screening plot |

`status` reads the live database, so it works while a study runs.

## Write an experiment

1. **Python function:** `evaluate(parameters, context) -> dict[str, float]`; `context` supplies case, fidelity,
   seed, baseline flag and a unique artifact directory. See `examples/toy_problem.py` and `examples/python_api.py`.
2. **External program:** read `{request}` JSON and write `{result}` JSON containing `{"metrics": {...}}`; no shell
   interpolation is used. See `examples/external_program.py` and `examples/command.yaml`.
3. **Custom experiment:** subclass `Experiment`, implement `run()` and `fingerprint()`, and pass it to `Study`.

The objective minimizes or maximizes a metric, raw or relative to the baseline. Constraints are hard bounds applied
to each replicate before aggregation. Cases have equal weight; repetitions are averaged within a case. Custom search
strategies: [extension guide](docs/EXTENDING.md); structure: [architecture](docs/ARCHITECTURE.md).

## Tests

```bash
python -m pytest -q tools/solver_tuning/tests
```

## Limitations

Repeated runs measure variability; they do not establish generalization to independent scenarios. Keep accuracy
requirements fixed when claiming solver efficiency. Primal feasibility and objective agreement do not establish KKT
stationarity or global optimality.
