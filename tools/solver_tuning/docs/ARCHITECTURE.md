# Architecture and invariants

| Layer | Responsibilities | Knows about a specific experiment? |
| --- | --- | --- |
| `params.py` | Domains, conditions, relations, sampling | No |
| `model.py` | Metrics, contexts, quality bounds, scalar objective | No |
| `experiment.py` | Experiment interface | No |
| `strategies/` | Candidate generation and search feedback | No |
| `evaluator.py` | Run execution, cache, raw-run gates and measurement | No |
| `study.py` | Baselines, batches, validation and final selection | No |
| `storage.py` | Transactional journal and controller lock | No |
| `adapters/` | Python functions and subprocesses | No |
| `tables.py` | Live `trials.csv` and the `status` text | No |
| `reporting.py` | Screening plot, parameter-effect plot, selected-vs-baseline plot | No |
| `cli.py` | Declarative YAML loading and commands | Routes to optional adapters |

## Evaluation lifecycle

1. Preserve and measure the exact baseline; its parameters may be outside the search
   bounds. Baseline quality is checked exactly like candidate quality.
2. Generate a proposal. Normalize only candidate values and validate domain rules.
3. Reserve a durable request before executing any expensive work.
4. Execute cases and independent trajectory repetitions under one worker budget.
5. Reject errors, nonfinite metrics and any quality-bound violation in any repeat.
6. Compute diagnostic scalar score; invalid evaluations remain unselectable even
   if their score is finite. The baseline can supply a positive normalization
   denominator even when it fails a quality gate.
7. Commit the result. Feed accepted/infeasible feedback to the search method.
8. Deduplicate the best feasible screening configurations and validate the top K
   at the validation fidelity, including every screening and extra validation case.
9. Select only among feasible validation measurements, including the baseline.
10. Export or explicitly report no valid configuration.

All quality checks operate before averaging repeated measurements. The objective
averages metric values within a case, then averages case scores. Relative scores
use a baseline measured at the same fidelity, cases and repetition count.

## Reproducibility and failure handling

The SQLite journal owns truth; CSV and JSONL are regenerable views. Requests are
written before execution. Result records do not share mutable objects with cache
entries. Cached results receive a new evaluation identity and reference the
original trial. Failed executions are not cached. A valid deterministic
measurement that fails a scientific quality gate may be reused.

Each case/repeat uses a stable seed shared across configurations (common random
numbers); validation uses a different seed namespace. Adapters must honor the
seed for stochastic simulations. Core
caching is inappropriate for timing/noisy independent resampling: disable it.

Worker exceptions propagate to the controller unless they are experiment failures,
which are recorded as rejected results. Completed work is not silently discarded.
One local OS lock protects each study controller. Separate studies can run at the
same time; they should not share a timing machine during measurements.

The package supports local single-controller studies, not distributed scheduling.
Python function adapters run in threads and cannot be forcibly timed out; use
external commands when process isolation or timeouts are required.

## Search methods

- Random: independent samples with seeds derived from proposal index.
- Coordinate: greedy neighbor enumeration from the best feasible screening point;
  one proposal at a time, then random restart when neighbors are exhausted.
- TPE: Optuna 4.x conditional multivariate TPE; random startup, infeasibility
  constraints, infinite rejected objective. Search state is reconstructed from
  completed trials on resume. Per-proposal seeds avoid serialized Python objects.
- DE: DE/rand/1/bin in normalized coordinates; logarithmic domains are encoded in
  log coordinates, integers are decoded by rounding. Requires numeric,
  unconditional, nonoptional parameters. Rejected individuals do not displace
  feasible individuals. With several workers, selection happens in synchronous
  batches rather than the strictly sequential textbook variant.

The original baseline is compared by the study, not inserted into TPE's parameter
model when it lies outside the declared domains. Serial deterministic searches
are the strongest reproducibility mode; parallel ordering and noisy simulators
may change suggestions. No strategy proves a global optimum.
